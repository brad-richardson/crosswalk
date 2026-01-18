#!/usr/bin/env python
"""Fetch OSM road data using Overpass API.

Alternative to osmium-tool based fetching for environments where
osmium-tool is not available. Works well for small bounding boxes.

Usage:
    python scripts/fetch_osm_overpass.py --bbox 75.78,26.90,75.82,26.94 -o data/india_test/
"""

from pathlib import Path

import geopandas as gpd
import requests
from loguru import logger
from shapely.geometry import LineString, Point

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_osm_via_overpass(
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    timeout: int = 120,
) -> tuple[Path, Path]:
    """Fetch OSM road data using Overpass API.

    Args:
        bbox: Bounding box (xmin, ymin, xmax, ymax) in WGS84
        output_dir: Directory for output files
        timeout: API timeout in seconds

    Returns:
        Tuple of (segments_path, connectors_path)
    """
    xmin, ymin, xmax, ymax = bbox
    # Overpass uses south,west,north,east format
    overpass_bbox = f"{ymin},{xmin},{ymax},{xmax}"

    logger.info(f"Fetching OSM data via Overpass API for bbox: {bbox}")

    # Query for highway ways with geometry
    query = f"""
[out:json][timeout:{timeout}];
way["highway"]({overpass_bbox});
out geom;
"""

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=timeout + 10,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Overpass API timeout - try a smaller bbox")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Overpass API error: {e}")
        raise

    data = response.json()
    elements = data.get("elements", [])
    ways = [e for e in elements if e["type"] == "way"]

    logger.info(f"Fetched {len(ways)} highway ways from OSM")

    # Convert to GeoDataFrame
    records = []
    connectors = []
    connector_ids = set()

    for way in ways:
        way_id = f"w{way['id']}"
        tags = way.get("tags", {})
        geometry = way.get("geometry", [])

        if len(geometry) < 2:
            continue

        # Build LineString
        coords = [(p["lon"], p["lat"]) for p in geometry]
        line = LineString(coords)

        # Extract relevant tags
        name = tags.get("name")
        highway = tags.get("highway", "unclassified")

        # Build record matching Overture schema
        record = {
            "id": way_id,
            "geometry": line,
            "name": name,
            "class": highway,
            "subtype": "road",
            "sources": [{"dataset": "OpenStreetMap", "record_id": way_id}],
            "names": {"primary": name} if name else None,
            "road_flags": _build_road_flags(tags, highway),
            "level_rules": _build_level_rules(tags),
            "source_tags": tags,
        }
        records.append(record)

        # Extract connectors (endpoints and intersections)
        for _i, point in enumerate(geometry):
            node_id = f"n{hash((point['lon'], point['lat'])) % 10000000000}"
            if node_id not in connector_ids:
                connector_ids.add(node_id)
                connectors.append(
                    {
                        "id": node_id,
                        "geometry": Point(point["lon"], point["lat"]),
                        "sources": [{"dataset": "OpenStreetMap", "record_id": node_id}],
                    }
                )

    # Create GeoDataFrames
    segments_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    connectors_gdf = gpd.GeoDataFrame(connectors, crs="EPSG:4326")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_path = output_dir / "osm_segments.parquet"
    connectors_path = output_dir / "osm_connectors.parquet"

    segments_gdf.to_parquet(segments_path)
    connectors_gdf.to_parquet(connectors_path)

    logger.info(f"Saved {len(segments_gdf)} OSM segments to {segments_path}")
    logger.info(f"Saved {len(connectors_gdf)} OSM connectors to {connectors_path}")

    return segments_path, connectors_path


def _build_road_flags(tags: dict, road_class: str) -> list:
    """Build Overture road_flags array from OSM tags."""
    flags = []

    # Bridge
    bridge = tags.get("bridge", "")
    valid_bridge_values = {
        "yes",
        "viaduct",
        "boardwalk",
        "cantilever",
        "covered",
        "low_water_crossing",
        "movable",
        "trestle",
        "aqueduct",
    }
    if bridge in valid_bridge_values:
        flags.append("is_bridge")

    # Tunnel
    tunnel = tags.get("tunnel", "")
    if tunnel in ("yes", "building_passage"):
        flags.append("is_tunnel")

    # Link
    if road_class and str(road_class).endswith("_link"):
        flags.append("is_link")

    if flags:
        return [{"values": flags}]
    return []


def _build_level_rules(tags: dict) -> list:
    """Build Overture level_rules array from OSM tags."""
    layer = tags.get("layer")
    if layer is None:
        return []

    try:
        level = int(layer)
        if level == 0:
            return []
        return [{"value": level}]
    except (ValueError, TypeError):
        return []


if __name__ == "__main__":
    import typer

    def main(
        bbox: str = typer.Option(..., "--bbox", "-b", help="Bounding box: xmin,ymin,xmax,ymax"),
        output: str = typer.Option("data/osm/", "-o", "--output", help="Output directory"),
    ):
        """Fetch OSM data via Overpass API."""
        parts = [float(x) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("Bbox must have 4 values: xmin,ymin,xmax,ymax")

        fetch_osm_via_overpass(
            bbox=tuple(parts),
            output_dir=Path(output),
        )

    typer.run(main)
