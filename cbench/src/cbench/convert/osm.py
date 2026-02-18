"""Convert GeoParquet road data to OSM XML format.

The conversion:
- Creates OSM <node> elements for vertices
- Creates OSM <way> elements for each LineString
- Maps 'class' column to highway=* tags
- Maps 'names' column to name=* tags
- Preserves original IDs in tags

Topology preservation:
- When connectors are provided, segments sharing the same connector_id
  will share OSM nodes, preserving network topology
- Without connectors, each vertex gets a unique node (no topology inference)
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, MultiLineString, Point

# Mapping from class values to OSM highway tags
CLASS_TO_HIGHWAY = {
    # Major roads
    "motorway": "motorway",
    "trunk": "trunk",
    "primary": "primary",
    "secondary": "secondary",
    "tertiary": "tertiary",
    # Link roads
    "motorway_link": "motorway_link",
    "trunk_link": "trunk_link",
    "primary_link": "primary_link",
    "secondary_link": "secondary_link",
    "tertiary_link": "tertiary_link",
    # Minor roads
    "residential": "residential",
    "living_street": "living_street",
    "unclassified": "unclassified",
    "service": "service",
    # Paths
    "pedestrian": "pedestrian",
    "footway": "footway",
    "sidewalk": "footway",
    "steps": "steps",
    "path": "path",
    "track": "track",
    "cycleway": "cycleway",
    "bridleway": "bridleway",
    # Unknown
    "road": "road",
    "unknown": "road",
    # Lifecycle states
    "construction": "construction",
    "proposed": "proposed",
    "abandoned": "abandoned",
}


def extract_name(names_value) -> str | None:
    """Extract primary name from names column.

    Handles various formats:
    - String: "Main Street"
    - List: [{"value": "Main Street", "language": "en"}] (Overture format)
    - Dict: {"primary": "Main Street", "common": None, ...}
    """
    if names_value is None:
        return None

    if isinstance(names_value, str):
        return names_value if names_value.strip() else None

    if isinstance(names_value, list) and len(names_value) > 0:
        first = names_value[0]
        if isinstance(first, dict):
            return first.get("value") or first.get("name")
        if isinstance(first, str):
            return first
        return None

    if isinstance(names_value, dict):
        for key in ["primary", "common", "name", "value"]:
            if key in names_value and names_value[key]:
                val = names_value[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    return extract_name(val)
        for v in names_value.values():
            if isinstance(v, str) and v:
                return v

    return None


class OSMConverter:
    """Converts GeoDataFrame to OSM XML format."""

    def __init__(self, connectors_gdf: gpd.GeoDataFrame | None = None):
        self.node_id_counter = -1
        self.way_id_counter = -1
        self.connector_node_map: dict[str, int] = {}
        self.connector_geom_map: dict[str, Point] = {}

        if connectors_gdf is not None:
            self._load_connectors(connectors_gdf)

        self.nodes: dict[int, tuple[float, float]] = {}

    def _load_connectors(self, connectors_gdf: gpd.GeoDataFrame) -> None:
        """Load connector geometries for topology preservation."""
        if connectors_gdf.crs and connectors_gdf.crs.to_epsg() != 4326:
            connectors_gdf = connectors_gdf.to_crs(epsg=4326)

        for _, row in connectors_gdf.iterrows():
            conn_id = str(row["id"])
            geom = row.geometry
            if geom is not None and not geom.is_empty:
                self.connector_geom_map[conn_id] = geom

        logger.info(f"Loaded {len(self.connector_geom_map)} connector geometries")

    def _hash_to_negative_int(self, s: str) -> int:
        """Convert string to a stable negative integer ID."""
        h = hashlib.sha256(s.encode()).digest()[:4]
        val = int.from_bytes(h, byteorder="big") % (2**31 - 1) + 1
        return -val

    def _get_connector_node_id(self, connector_id: str) -> int:
        """Get or create node ID for a connector (shared across segments)."""
        if connector_id not in self.connector_node_map:
            node_id = self._hash_to_negative_int(f"connector:{connector_id}")
            self.connector_node_map[connector_id] = node_id

            if connector_id in self.connector_geom_map:
                pt = self.connector_geom_map[connector_id]
                self.nodes[node_id] = (pt.x, pt.y)

        return self.connector_node_map[connector_id]

    def _create_node(self, lon: float, lat: float) -> int:
        """Create a new unique node (no deduplication).

        TODO: This destroys topology for non-connector vertices. Two adjacent
        segments sharing an endpoint get different node IDs, so Hootenanny
        treats them as disconnected. Fix by adding a coordinate-based dedup
        index (e.g., round to 7dp and reuse existing node at same coords).
        See also: connector nodes use raw precision while these are rounded,
        creating an inconsistency.
        """
        node_id = self.node_id_counter
        self.node_id_counter -= 1
        self.nodes[node_id] = (round(lon, 7), round(lat, 7))
        return node_id

    def _get_way_id(self) -> int:
        way_id = self.way_id_counter
        self.way_id_counter -= 1
        return way_id

    def convert(
        self,
        gdf: gpd.GeoDataFrame,
        id_column: str = "id",
        class_column: str = "class",
        name_column: str = "names",
        connectors_column: str = "connectors",
        source_tag: str | None = None,
    ) -> ET.Element:
        """Convert GeoDataFrame to OSM XML Element.

        Args:
            gdf: Input GeoDataFrame with LineString geometries.
            id_column: Column containing segment IDs.
            class_column: Column containing road class.
            name_column: Column containing road names.
            connectors_column: Column containing connector references.
            source_tag: If provided, creates 'matcher_{source_tag}_{sanitized_id}' tags.
        """
        self._source_tag = source_tag

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            logger.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs(epsg=4326)

        has_connectors = connectors_column in gdf.columns and len(self.connector_geom_map) > 0
        if has_connectors:
            logger.info("Using connector-based topology preservation")
        else:
            logger.info("No connectors provided, each vertex gets unique node")

        osm = ET.Element("osm", version="0.6", generator="cbench-convert")
        ways_data = []

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if isinstance(geom, LineString):
                lines = [geom]
            elif isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            else:
                logger.warning(f"Skipping non-line geometry: {geom.geom_type}")
                continue

            connector_at_map = {}
            if has_connectors:
                connectors_val = row.get(connectors_column)
                if connectors_val is not None:
                    for conn in connectors_val:
                        if isinstance(conn, dict):
                            at_pos = conn.get("at")
                            conn_id = conn.get("connector_id")
                            if at_pos is not None and conn_id:
                                connector_at_map[float(at_pos)] = str(conn_id)

            for line in lines:
                if not isinstance(line, LineString):
                    continue

                coords = list(line.coords)
                num_coords = len(coords)
                node_ids = []

                for i, coord in enumerate(coords):
                    lon, lat = coord[0], coord[1]
                    at_pos = i / (num_coords - 1) if num_coords > 1 else 0.0

                    connector_id = None
                    for conn_at, conn_id in connector_at_map.items():
                        if abs(conn_at - at_pos) < 0.001:
                            connector_id = conn_id
                            break

                    if connector_id and connector_id in self.connector_geom_map:
                        node_id = self._get_connector_node_id(connector_id)
                    else:
                        node_id = self._create_node(lon, lat)

                    node_ids.append(node_id)

                unique_node_ids = set(node_ids)
                if len(unique_node_ids) < 2:
                    original_id = (
                        str(row.get(id_column, idx)) if id_column in gdf.columns else str(idx)
                    )
                    logger.warning(
                        f"Skipping way {original_id}: only {len(unique_node_ids)} "
                        f"distinct node(s) after rounding"
                    )
                    continue

                way_data = {
                    "node_ids": node_ids,
                    "original_id": str(row.get(id_column, idx))
                    if id_column in gdf.columns
                    else str(idx),
                    "highway": None,
                    "name": None,
                }

                if class_column in gdf.columns:
                    road_class = row.get(class_column)
                    if road_class:
                        way_data["highway"] = CLASS_TO_HIGHWAY.get(
                            str(road_class).lower(), "unclassified"
                        )

                if name_column in gdf.columns:
                    way_data["name"] = extract_name(row.get(name_column))

                ways_data.append(way_data)

        shared_nodes = len(self.connector_node_map)
        unique_nodes = len(self.nodes) - shared_nodes
        logger.info(
            f"Created {len(self.nodes)} nodes ({shared_nodes} shared via connectors, "
            f"{unique_nodes} unique), {len(ways_data)} ways"
        )

        for node_id, (lon, lat) in sorted(self.nodes.items(), reverse=True):
            ET.SubElement(osm, "node", id=str(node_id), lat=str(lat), lon=str(lon), version="1")

        for way_data in ways_data:
            way_id = self._get_way_id()
            way_elem = ET.SubElement(osm, "way", id=str(way_id), version="1")

            for node_id in way_data["node_ids"]:
                ET.SubElement(way_elem, "nd", ref=str(node_id))

            if way_data["highway"]:
                ET.SubElement(way_elem, "tag", k="highway", v=way_data["highway"])
            if way_data["name"]:
                ET.SubElement(way_elem, "tag", k="name", v=way_data["name"])

            if self._source_tag:
                sanitized = way_data["original_id"].replace("-", "_").replace(":", "_")
                tag_key = f"matcher_{self._source_tag}_{sanitized}"
                ET.SubElement(way_elem, "tag", k=tag_key, v=way_data["original_id"])
            else:
                ET.SubElement(way_elem, "tag", k="matcher:id", v=way_data["original_id"])

        return osm

    def convert_to_string(self, gdf: gpd.GeoDataFrame, pretty: bool = True, **kwargs) -> str:
        """Convert GeoDataFrame to OSM XML string."""
        osm_elem = self.convert(gdf, **kwargs)
        if pretty:
            xml_str = ET.tostring(osm_elem, encoding="unicode")
            dom = minidom.parseString(xml_str)
            return dom.toprettyxml(indent="  ")
        else:
            return ET.tostring(osm_elem, encoding="unicode")


def convert_parquet_to_osm(
    input_path: Path,
    output_path: Path,
    connectors_path: Path | None = None,
    id_column: str = "id",
    class_column: str = "class",
    name_column: str = "names",
    source_tag: str | None = None,
) -> None:
    """Convert a GeoParquet file to OSM XML.

    Args:
        input_path: Path to input GeoParquet.
        output_path: Path to output OSM XML.
        connectors_path: Optional path to connectors GeoParquet (for topology).
        id_column: Column containing segment IDs.
        class_column: Column containing road class.
        name_column: Column containing road names.
        source_tag: If provided, creates 'matcher_{source_tag}_{sanitized_id}' tags.
    """
    logger.info(f"Loading {input_path}")
    gdf = gpd.read_parquet(input_path)
    logger.info(f"Loaded {len(gdf)} features")

    connectors_gdf = None
    if connectors_path and connectors_path.exists():
        logger.info(f"Loading connectors from {connectors_path}")
        connectors_gdf = gpd.read_parquet(connectors_path)
        logger.info(f"Loaded {len(connectors_gdf)} connectors")

    converter = OSMConverter(connectors_gdf=connectors_gdf)
    xml_str = converter.convert_to_string(
        gdf,
        id_column=id_column,
        class_column=class_column,
        name_column=name_column,
        source_tag=source_tag,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml_str)
    logger.success(f"Wrote OSM XML to {output_path}")
