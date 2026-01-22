#!/usr/bin/env python3
"""Convert GeoParquet road data to OSM XML format.

This script converts matcher's standard GeoParquet format to OSM XML,
enabling comparison with tools like Hootenanny that require OSM input.

Usage:
    python scripts/convert_to_osm.py data/raw/boston_streets.parquet -o boston_streets.osm

    # With Overture data (uses connectors for topology)
    python scripts/convert_to_osm.py data/raw/overture_segments.parquet \
        --connectors data/raw/overture_connectors.parquet -o overture.osm

The conversion:
- Creates OSM <node> elements for vertices
- Creates OSM <way> elements for each LineString
- Maps 'class' column to highway=* tags
- Maps 'names' column to name=* tags
- Preserves original IDs in tags

Topology preservation:
- When --connectors is provided, segments sharing the same connector_id
  will share OSM nodes, preserving network topology
- Without connectors, each vertex gets a unique node (no topology inference)
"""

import argparse
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, MultiLineString, Point

# Mapping from our class values to OSM highway tags
# Aligned with HIGHWAY_VALUES in src/matcher/fetch/osm_pbf.py
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
    "sidewalk": "footway",  # Map sidewalk to footway
    "steps": "steps",
    "path": "path",
    "track": "track",
    "cycleway": "cycleway",
    "bridleway": "bridleway",
    # Unknown (OSM "road" maps to Overture "unknown")
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

    Aligned with _extract_name_string in src/matcher/features/semantic.py.
    """
    if names_value is None:
        return None

    if isinstance(names_value, str):
        return names_value if names_value.strip() else None

    # Handle Overture list format: [{"value": "...", "language": "en"}, ...]
    if isinstance(names_value, list) and len(names_value) > 0:
        first = names_value[0]
        if isinstance(first, dict):
            return first.get("value") or first.get("name")
        if isinstance(first, str):
            return first
        return None

    # Handle dict format (OSM/Overture): {"primary": "...", "common": "...", ...}
    if isinstance(names_value, dict):
        # Try common keys in order of preference (aligned with semantic.py)
        for key in ["primary", "common", "name", "value"]:
            if key in names_value and names_value[key]:
                val = names_value[key]
                # Handle nested extraction
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    return extract_name(val)
        # Last resort - return first non-None string value
        for v in names_value.values():
            if isinstance(v, str) and v:
                return v

    return None


class OSMConverter:
    """Converts GeoDataFrame to OSM XML format."""

    def __init__(self, connectors_gdf: gpd.GeoDataFrame | None = None):
        """Initialize converter.

        Args:
            connectors_gdf: Optional GeoDataFrame of connector points (id, geometry).
                           When provided, segments sharing connector_ids will share nodes.
        """
        self.node_id_counter = -1  # OSM uses negative IDs for new data
        self.way_id_counter = -1

        # connector_id -> OSM node_id (for topology preservation)
        self.connector_node_map: dict[str, int] = {}

        # connector_id -> Point geometry (for looking up coordinates)
        self.connector_geom_map: dict[str, Point] = {}

        if connectors_gdf is not None:
            self._load_connectors(connectors_gdf)

        # (lon, lat) -> node_id for nodes (only used for output)
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
        # Use first 8 bytes of SHA256 hash, converted to negative int
        h = hashlib.sha256(s.encode()).digest()[:8]
        return -abs(int.from_bytes(h, byteorder="big"))

    def _get_connector_node_id(self, connector_id: str) -> int:
        """Get or create node ID for a connector (shared across segments)."""
        if connector_id not in self.connector_node_map:
            # Use hash-based ID for deterministic output
            node_id = self._hash_to_negative_int(f"connector:{connector_id}")
            self.connector_node_map[connector_id] = node_id

            # Get coordinates from connector geometry
            if connector_id in self.connector_geom_map:
                pt = self.connector_geom_map[connector_id]
                self.nodes[node_id] = (pt.x, pt.y)

        return self.connector_node_map[connector_id]

    def _create_node(self, lon: float, lat: float) -> int:
        """Create a new unique node (no deduplication)."""
        node_id = self.node_id_counter
        self.node_id_counter -= 1
        self.nodes[node_id] = (round(lon, 7), round(lat, 7))
        return node_id

    def _get_way_id(self) -> int:
        """Get next way ID."""
        way_id = self.way_id_counter
        self.way_id_counter -= 1
        return way_id

    def _interpolate_at(self, line: LineString, at: float) -> tuple[float, float]:
        """Get coordinate at fractional position along line."""
        pt = line.interpolate(at, normalized=True)
        return (pt.x, pt.y)

    def convert(
        self,
        gdf: gpd.GeoDataFrame,
        id_column: str = "id",
        class_column: str = "class",
        name_column: str = "names",
        connectors_column: str = "connectors",
    ) -> ET.Element:
        """Convert GeoDataFrame to OSM XML Element.

        Args:
            gdf: Input GeoDataFrame with LineString geometries
            id_column: Column containing segment IDs
            class_column: Column containing road class
            name_column: Column containing road names
            connectors_column: Column containing connector references (Overture format)

        Returns:
            ElementTree Element representing OSM XML
        """
        # Ensure WGS84
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            logger.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs(epsg=4326)

        has_connectors = connectors_column in gdf.columns and len(self.connector_geom_map) > 0
        if has_connectors:
            logger.info("Using connector-based topology preservation")
        else:
            logger.info("No connectors provided, each vertex gets unique node")

        # Create root element
        osm = ET.Element("osm", version="0.6", generator="matcher-convert")

        # First pass: collect all nodes from all geometries
        ways_data = []

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Handle both LineString and MultiLineString
            if isinstance(geom, LineString):
                lines = [geom]
            elif isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            else:
                logger.warning(f"Skipping non-line geometry: {geom.geom_type}")
                continue

            # Parse connector positions for this segment
            connector_at_map = {}  # at_position -> connector_id
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

                # Collect node IDs for this way
                node_ids = []
                for i, coord in enumerate(coords):
                    lon, lat = coord[0], coord[1]

                    # Calculate fractional position along line
                    if num_coords > 1:
                        at_pos = i / (num_coords - 1)
                    else:
                        at_pos = 0.0

                    # Check if this position has a connector (with small tolerance)
                    connector_id = None
                    for conn_at, conn_id in connector_at_map.items():
                        if abs(conn_at - at_pos) < 0.001:  # ~0.1% tolerance
                            connector_id = conn_id
                            break

                    if connector_id and connector_id in self.connector_geom_map:
                        # Use shared connector node
                        node_id = self._get_connector_node_id(connector_id)
                    else:
                        # Create unique node (no deduplication)
                        node_id = self._create_node(lon, lat)

                    node_ids.append(node_id)

                # Validate: OSM ways require at least 2 distinct nodes
                # Coordinate rounding could create degenerate single-node ways
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

                # Collect way attributes
                way_data = {
                    "node_ids": node_ids,
                    "original_id": str(row.get(id_column, idx))
                    if id_column in gdf.columns
                    else str(idx),
                    "highway": None,
                    "name": None,
                }

                # Map class to highway tag
                if class_column in gdf.columns:
                    road_class = row.get(class_column)
                    if road_class:
                        way_data["highway"] = CLASS_TO_HIGHWAY.get(
                            str(road_class).lower(), "unclassified"
                        )

                # Extract name
                if name_column in gdf.columns:
                    way_data["name"] = extract_name(row.get(name_column))

                ways_data.append(way_data)

        shared_nodes = len(self.connector_node_map)
        unique_nodes = len(self.nodes) - shared_nodes
        logger.info(
            f"Created {len(self.nodes)} nodes ({shared_nodes} shared via connectors, "
            f"{unique_nodes} unique), {len(ways_data)} ways"
        )

        # Add all nodes to XML (sorted by ID descending for consistency)
        for node_id, (lon, lat) in sorted(self.nodes.items(), reverse=True):
            ET.SubElement(
                osm,
                "node",
                id=str(node_id),
                lat=str(lat),
                lon=str(lon),
                version="1",
            )

        # Add all ways to XML
        for way_data in ways_data:
            way_id = self._get_way_id()
            way_elem = ET.SubElement(
                osm,
                "way",
                id=str(way_id),
                version="1",
            )

            # Add node references
            for node_id in way_data["node_ids"]:
                ET.SubElement(way_elem, "nd", ref=str(node_id))

            # Add tags
            if way_data["highway"]:
                ET.SubElement(way_elem, "tag", k="highway", v=way_data["highway"])

            if way_data["name"]:
                ET.SubElement(way_elem, "tag", k="name", v=way_data["name"])

            # Preserve original ID
            ET.SubElement(way_elem, "tag", k="matcher:id", v=way_data["original_id"])

        return osm

    def convert_to_string(
        self,
        gdf: gpd.GeoDataFrame,
        pretty: bool = True,
        **kwargs,
    ) -> str:
        """Convert GeoDataFrame to OSM XML string.

        Args:
            gdf: Input GeoDataFrame
            pretty: Whether to format with indentation
            **kwargs: Passed to convert()

        Returns:
            OSM XML as string
        """
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
) -> None:
    """Convert a GeoParquet file to OSM XML.

    Args:
        input_path: Path to input GeoParquet
        output_path: Path to output OSM XML
        connectors_path: Optional path to connectors GeoParquet (for topology)
        id_column: Column containing segment IDs
        class_column: Column containing road class
        name_column: Column containing road names
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
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml_str)
    logger.success(f"Wrote OSM XML to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert GeoParquet road data to OSM XML format")
    parser.add_argument(
        "input",
        type=Path,
        help="Input GeoParquet file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output OSM XML file (default: input.osm)",
    )
    parser.add_argument(
        "-c",
        "--connectors",
        type=Path,
        default=None,
        help="Connectors GeoParquet file (for topology preservation)",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Column containing segment IDs (default: id)",
    )
    parser.add_argument(
        "--class-column",
        type=str,
        default="class",
        help="Column containing road class (default: class)",
    )
    parser.add_argument(
        "--name-column",
        type=str,
        default="names",
        help="Column containing road names (default: names)",
    )
    args = parser.parse_args()

    # Default output path
    output_path = args.output or args.input.with_suffix(".osm")

    convert_parquet_to_osm(
        args.input,
        output_path,
        connectors_path=args.connectors,
        id_column=args.id_column,
        class_column=args.class_column,
        name_column=args.name_column,
    )


if __name__ == "__main__":
    main()
