"""Convert Overture GeoParquet road segments to an OSM PBF routable graph.

This is the Overture -> routable-graph ingestion tax shared by any map-matching
baseline (Valhalla Meili, GraphHopper). Unlike ``convert/osm.py`` (which emits
XML for Hootenanny with negative synthetic ids and ``matcher_*`` provenance
tags), this module writes a compact ``.osm.pbf`` with **positive** node/way ids
suitable for a routing-graph build, and carries the Overture GERS id as the OSM
``way_id`` so a map-matcher's returned ``way_id`` maps straight back to a GERS id
with no join.

Topology: Overture segments meet exactly at shared *connector* coordinates, so
we build the routable graph by **coordinate de-duplication** — vertices with the
same (rounded) lon/lat collapse to one OSM node, which is exactly what makes two
touching ways routable through a shared node. This is simpler and more robust
than parsing the ``connectors`` column and produces the same connectivity.

Every way gets a routable ``highway=*`` tag (mapped from the Overture ``class``,
defaulting to ``residential`` so nothing is dropped from the graph). We do NOT
emit ``oneway`` tags: map-matching a local segment is direction-agnostic (a
sidewalk/road and its reverse are the same physical feature), so the graph is
left bidirectional and directionality is handled at match time by costing.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, MultiLineString

from cbench.convert.osm import CLASS_TO_HIGHWAY

# Coordinate rounding for node de-duplication. 7 decimal degrees ~= 1.1 cm at the
# equator — finer than any real Overture connector coincidence tolerance, so
# genuinely-shared endpoints collapse while distinct vertices stay distinct.
COORD_DECIMALS = 7

# Fallback highway tag for segments with a missing/unknown class. ``residential``
# is routable by every Valhalla costing model, so no segment is dropped from the
# graph for lack of a class.
DEFAULT_HIGHWAY = "residential"


def convert_overture_to_pbf(
    reference_path: Path,
    pbf_path: Path,
    id_map_path: Path,
    id_column: str = "id",
    class_column: str = "class",
) -> dict:
    """Convert an Overture segments GeoParquet to an OSM PBF routable graph.

    Args:
        reference_path: Path to the Overture segments GeoParquet.
        pbf_path: Output ``.osm.pbf`` path.
        id_map_path: Output JSON sidecar mapping ``str(way_id) -> gers_id``.
        id_column: Column holding the GERS id.
        class_column: Column holding the Overture road class.

    Returns:
        Metadata dict: ``{n_nodes, n_ways, n_input, highway_counts}``.
    """
    import osmium
    from osmium.osm.mutable import Node, Way

    logger.info(f"Loading reference {reference_path}")
    gdf = gpd.read_parquet(reference_path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    logger.info(f"Loaded {len(gdf)} reference segments")

    if id_column not in gdf.columns:
        raise ValueError(f"Reference missing id column '{id_column}'")

    node_coords: dict[tuple[float, float], int] = {}
    node_list: list[tuple[int, float, float]] = []  # (node_id, lon, lat)
    ways: list[tuple[int, list[int], str]] = []  # (way_id, node_ids, highway)
    id_map: dict[str, str] = {}
    highway_counts: dict[str, int] = {}

    next_node_id = 1
    next_way_id = 1

    classes = gdf[class_column] if class_column in gdf.columns else None
    ids = gdf[id_column].astype(str)

    for pos, (geom, gers_id) in enumerate(zip(gdf.geometry.values, ids.values)):
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            parts = [geom]
        elif isinstance(geom, MultiLineString):
            parts = list(geom.geoms)
        else:
            continue

        road_class = None
        if classes is not None:
            rc = classes.iloc[pos]
            if rc is not None and str(rc) != "nan":
                road_class = str(rc).lower()
        highway = (
            CLASS_TO_HIGHWAY.get(road_class, DEFAULT_HIGHWAY) if road_class else DEFAULT_HIGHWAY
        )

        for line in parts:
            if not isinstance(line, LineString):
                continue
            node_ids: list[int] = []
            for x, y, *_ in line.coords:
                key = (round(x, COORD_DECIMALS), round(y, COORD_DECIMALS))
                nid = node_coords.get(key)
                if nid is None:
                    nid = next_node_id
                    next_node_id += 1
                    node_coords[key] = nid
                    node_list.append((nid, key[0], key[1]))
                # Collapse consecutive duplicate vertices (would create zero-length edges).
                if node_ids and node_ids[-1] == nid:
                    continue
                node_ids.append(nid)

            if len(node_ids) < 2:
                continue

            way_id = next_way_id
            next_way_id += 1
            ways.append((way_id, node_ids, highway))
            # Carry the GERS id as the way_id -> gers_id mapping. A MultiLineString
            # produces several ways all mapping to the same GERS id, which is fine:
            # the matcher aggregates by gers_id.
            id_map[str(way_id)] = gers_id
            highway_counts[highway] = highway_counts.get(highway, 0) + 1

    logger.info(
        f"Built graph: {len(node_list)} nodes, {len(ways)} ways (from {len(gdf)} input segments)"
    )

    pbf_path.parent.mkdir(parents=True, exist_ok=True)
    if pbf_path.exists():
        pbf_path.unlink()
    writer = osmium.SimpleWriter(str(pbf_path))
    try:
        # OSM PBF convention: nodes (ascending id) before ways (ascending id).
        for nid, lon, lat in node_list:
            writer.add_node(Node(id=nid, location=(lon, lat)))
        for way_id, node_ids, highway in ways:
            writer.add_way(Way(id=way_id, nodes=node_ids, tags={"highway": highway}))
    finally:
        writer.close()

    id_map_path.parent.mkdir(parents=True, exist_ok=True)
    id_map_path.write_text(json.dumps(id_map))
    logger.success(f"Wrote PBF to {pbf_path} and id-map ({len(id_map)} ways) to {id_map_path}")

    return {
        "n_nodes": len(node_list),
        "n_ways": len(ways),
        "n_input": int(len(gdf)),
        "highway_counts": highway_counts,
    }
