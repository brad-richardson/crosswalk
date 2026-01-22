#!/usr/bin/env python3
"""Convert GeoParquet road data to OSM XML format.

This script converts matcher's standard GeoParquet format to OSM XML,
enabling comparison with tools like Hootenanny that require OSM input.

Usage:
    python scripts/convert_to_osm.py data/raw/boston_streets.parquet -o boston_streets.osm
    python scripts/convert_to_osm.py data/raw/overture_segments.parquet -o overture.osm

The conversion:
- Creates OSM <node> elements for all unique vertices
- Creates OSM <way> elements for each LineString
- Maps 'class' column to highway=* tags
- Maps 'names' column to name=* tags
- Preserves original IDs in tags
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, MultiLineString

# Mapping from our class values to OSM highway tags
CLASS_TO_HIGHWAY = {
    "motorway": "motorway",
    "motorway_link": "motorway_link",
    "trunk": "trunk",
    "trunk_link": "trunk_link",
    "primary": "primary",
    "primary_link": "primary_link",
    "secondary": "secondary",
    "secondary_link": "secondary_link",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary_link",
    "residential": "residential",
    "living_street": "living_street",
    "service": "service",
    "unclassified": "unclassified",
    "footway": "footway",
    "sidewalk": "footway",
    "path": "path",
    "pedestrian": "pedestrian",
    "cycleway": "cycleway",
    "track": "track",
    "steps": "steps",
}


def extract_name(names_value) -> str | None:
    """Extract primary name from names column.

    Handles various formats:
    - String: "Main Street"
    - List: [{"value": "Main Street", "language": "en"}]
    - Dict: {"primary": "Main Street"}
    """
    if names_value is None:
        return None

    if isinstance(names_value, str):
        return names_value if names_value.strip() else None

    if isinstance(names_value, list) and len(names_value) > 0:
        first = names_value[0]
        if isinstance(first, dict):
            return first.get("value") or first.get("name")
        return str(first)

    if isinstance(names_value, dict):
        return names_value.get("primary") or names_value.get("value")

    return None


class OSMConverter:
    """Converts GeoDataFrame to OSM XML format."""

    def __init__(self):
        self.node_id_counter = -1  # OSM uses negative IDs for new data
        self.way_id_counter = -1
        self.node_cache: dict[tuple[float, float], int] = {}

    def _get_node_id(self, lon: float, lat: float) -> int:
        """Get or create node ID for a coordinate pair."""
        # Round to 7 decimal places (OSM standard, ~1cm precision)
        key = (round(lon, 7), round(lat, 7))

        if key not in self.node_cache:
            self.node_cache[key] = self.node_id_counter
            self.node_id_counter -= 1

        return self.node_cache[key]

    def _get_way_id(self) -> int:
        """Get next way ID."""
        way_id = self.way_id_counter
        self.way_id_counter -= 1
        return way_id

    def convert(
        self,
        gdf: gpd.GeoDataFrame,
        id_column: str = "id",
        class_column: str = "class",
        name_column: str = "names",
    ) -> ET.Element:
        """Convert GeoDataFrame to OSM XML Element.

        Args:
            gdf: Input GeoDataFrame with LineString geometries
            id_column: Column containing segment IDs
            class_column: Column containing road class
            name_column: Column containing road names

        Returns:
            ElementTree Element representing OSM XML
        """
        # Ensure WGS84
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            logger.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs(epsg=4326)

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

            for line in lines:
                if not isinstance(line, LineString):
                    continue

                # Collect node IDs for this way
                node_ids = []
                for coord in line.coords:
                    lon, lat = coord[0], coord[1]
                    node_id = self._get_node_id(lon, lat)
                    node_ids.append(node_id)

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

        logger.info(f"Collected {len(self.node_cache)} unique nodes, {len(ways_data)} ways")

        # Add all nodes to XML
        for (lon, lat), node_id in sorted(
            self.node_cache.items(), key=lambda x: x[1], reverse=True
        ):
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

    def convert_file(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs,
    ) -> None:
        """Convert a GeoParquet file to OSM XML file.

        Args:
            input_path: Path to input GeoParquet
            output_path: Path to output OSM XML
            **kwargs: Passed to convert()
        """
        logger.info(f"Loading {input_path}")
        gdf = gpd.read_parquet(input_path)
        logger.info(f"Loaded {len(gdf)} features")

        xml_str = self.convert_to_string(gdf, **kwargs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(xml_str)
        logger.success(f"Wrote OSM XML to {output_path}")


def convert_parquet_to_osm(
    input_path: Path,
    output_path: Path,
    id_column: str = "id",
    class_column: str = "class",
    name_column: str = "names",
) -> None:
    """Convert a GeoParquet file to OSM XML.

    Args:
        input_path: Path to input GeoParquet
        output_path: Path to output OSM XML
        id_column: Column containing segment IDs
        class_column: Column containing road class
        name_column: Column containing road names
    """
    converter = OSMConverter()
    converter.convert_file(
        input_path,
        output_path,
        id_column=id_column,
        class_column=class_column,
        name_column=name_column,
    )


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
        id_column=args.id_column,
        class_column=args.class_column,
        name_column=args.name_column,
    )


if __name__ == "__main__":
    main()
