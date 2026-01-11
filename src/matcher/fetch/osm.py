"""Fetch OSM road data using pyrosm for PBF parsing.

Two modes:
1. From local PBF file with bbox filter
2. Download regional extract then filter
"""

from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
from loguru import logger


def fetch_osm_roads(
    bbox: Tuple[float, float, float, float],
    pbf_path: Optional[Path] = None,
    region: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Extract road network from OSM PBF file.

    Args:
        bbox: (minx, miny, maxx, maxy) bounding box in WGS84
        pbf_path: Path to local PBF file
        region: Pyrosm region name for auto-download (e.g., "oregon" for testing)
        output_path: Optional path to save as GeoParquet

    Returns:
        GeoDataFrame with road geometries and attributes
    """
    from pyrosm import OSM, get_data

    if pbf_path is None and region:
        logger.info(f"Downloading OSM extract for region: {region}")
        pbf_path = Path(get_data(region))

    if pbf_path is None:
        raise ValueError("Either pbf_path or region must be provided")

    logger.info(f"Loading OSM data from {pbf_path} with bbox {bbox}")
    osm = OSM(str(pbf_path), bounding_box=bbox)

    # Get driving network (roads suitable for vehicles)
    logger.info("Extracting driving network...")
    roads = osm.get_network(network_type="driving")

    if roads is None or len(roads) == 0:
        logger.warning("No roads found in the specified area")
        return gpd.GeoDataFrame()

    logger.info(f"Found {len(roads)} road segments")

    # Normalize to common schema
    roads = _normalize_osm_schema(roads)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        roads.to_parquet(output_path)
        logger.info(f"Saved OSM roads to {output_path}")

    return roads


def _normalize_osm_schema(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize OSM attributes to a common schema.

    Maps OSM tags to standardized column names and extracts
    bridge/tunnel/layer information.
    """
    # Create normalized columns
    normalized = roads.copy()

    # Rename standard columns
    column_mapping = {
        "name": "name",
        "highway": "road_class",
        "maxspeed": "speed_limit",
        "oneway": "oneway",
        "lanes": "lanes",
        "surface": "surface",
    }

    for osm_col, new_col in column_mapping.items():
        if osm_col in normalized.columns and osm_col != new_col:
            normalized[new_col] = normalized[osm_col]

    # Extract bridge/tunnel/layer attributes
    if "bridge" in normalized.columns:
        normalized["is_bridge"] = normalized["bridge"].apply(
            lambda x: x is not None and x not in ["no", "0", False]
        )
    else:
        normalized["is_bridge"] = False

    if "tunnel" in normalized.columns:
        normalized["is_tunnel"] = normalized["tunnel"].apply(
            lambda x: x is not None and x not in ["no", "0", False]
        )
    else:
        normalized["is_tunnel"] = False

    if "layer" in normalized.columns:
        normalized["layer"] = normalized["layer"].apply(_parse_layer)
    else:
        normalized["layer"] = 0

    # Map highway classes to standardized road classes
    if "road_class" in normalized.columns:
        normalized["road_class_normalized"] = normalized["road_class"].apply(
            _normalize_road_class
        )

    return normalized


def _parse_layer(layer_value) -> int:
    """Parse OSM layer tag to integer."""
    if layer_value is None:
        return 0
    try:
        return int(layer_value)
    except (ValueError, TypeError):
        return 0


def _normalize_road_class(highway: Optional[str]) -> str:
    """Normalize OSM highway tag to standard road class."""
    if highway is None:
        return "unclassified"

    highway = highway.lower()

    # Map to Overture-like classes
    class_mapping = {
        "motorway": "motorway",
        "motorway_link": "motorway",
        "trunk": "trunk",
        "trunk_link": "trunk",
        "primary": "primary",
        "primary_link": "primary",
        "secondary": "secondary",
        "secondary_link": "secondary",
        "tertiary": "tertiary",
        "tertiary_link": "tertiary",
        "residential": "residential",
        "living_street": "residential",
        "service": "service",
        "unclassified": "unclassified",
        "track": "track",
        "path": "path",
    }

    return class_mapping.get(highway, "unclassified")


def load_osm_roads(path: Path) -> gpd.GeoDataFrame:
    """Load OSM roads from a GeoParquet file.

    Args:
        path: Path to GeoParquet file

    Returns:
        GeoDataFrame with OSM roads
    """
    logger.info(f"Loading OSM roads from {path}")
    gdf = gpd.read_parquet(path)
    logger.info(f"Loaded {len(gdf)} OSM road segments")
    return gdf
