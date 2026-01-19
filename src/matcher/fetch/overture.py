"""Fetch Overture Transportation data for a bounding box.

Uses the overturemaps-py library to query Overture GeoParquet files
and automatically handles release version detection.
"""

from pathlib import Path

import geopandas as gpd
from loguru import logger
from overturemaps.core import geodataframe, get_latest_release
from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Bounding box in EPSG:4326 (WGS84), matching Overture schema."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def to_wkt(self) -> str:
        """Convert to WKT polygon."""
        return (
            f"POLYGON(({self.xmin} {self.ymin}, {self.xmax} {self.ymin}, "
            f"{self.xmax} {self.ymax}, {self.xmin} {self.ymax}, {self.xmin} {self.ymin}))"
        )

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert to tuple (xmin, ymin, xmax, ymax) for overturemaps."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)


def fetch_overture_segments(
    bbox: BoundingBox,
    output_path: Path,
    release: str | None = None,
) -> Path:
    """Download Overture road segments for a bounding box.

    Uses the overturemaps-py library which automatically detects
    the latest release if not specified.

    Args:
        bbox: Bounding box in WGS84 coordinates
        output_path: Path for output GeoParquet file
        release: Overture release version (default: latest)

    Returns:
        Path to the output file
    """
    logger.info(f"Fetching Overture segments for bbox: {bbox}")

    if release is None:
        release = get_latest_release()
        logger.info(f"Using latest Overture release: {release}")

    # Fetch segments using overturemaps library
    # The library handles S3 access and bbox filtering efficiently
    gdf = geodataframe("segment", bbox=bbox.to_tuple(), release=release)

    # Filter to road subtype only
    if "subtype" in gdf.columns:
        gdf = gdf[gdf["subtype"] == "road"]

    logger.info(f"Fetched {len(gdf)} road segments")

    # Ensure CRS is set (Overture data is always WGS84)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)

    logger.info(f"Saved Overture segments to {output_path}")
    return output_path


def fetch_overture_connectors(
    bbox: BoundingBox,
    output_path: Path,
    release: str | None = None,
) -> Path:
    """Download Overture connectors (intersections) for a bounding box.

    Uses the overturemaps-py library which automatically detects
    the latest release if not specified.

    Args:
        bbox: Bounding box in WGS84 coordinates
        output_path: Path for output GeoParquet file
        release: Overture release version (default: latest)

    Returns:
        Path to the output file
    """
    logger.info(f"Fetching Overture connectors for bbox: {bbox}")

    if release is None:
        release = get_latest_release()
        logger.info(f"Using latest Overture release: {release}")

    # Fetch connectors using overturemaps library
    gdf = geodataframe("connector", bbox=bbox.to_tuple(), release=release)

    logger.info(f"Fetched {len(gdf)} connectors")

    # Ensure CRS is set (Overture data is always WGS84)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)

    logger.info(f"Saved Overture connectors to {output_path}")
    return output_path


def load_overture_segments(path: Path) -> gpd.GeoDataFrame:
    """Load Overture segments from a GeoParquet file.

    Extracts flat fields (is_bridge, is_tunnel, level, name) from
    Overture schema structs for downstream processing.

    Args:
        path: Path to GeoParquet file

    Returns:
        GeoDataFrame with Overture segments and extracted flat fields
    """
    logger.info(f"Loading Overture segments from {path}")
    gdf = gpd.read_parquet(path)

    # Ensure CRS is set (Overture data is always WGS84)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Extract name from names struct if not already flat
    if "name" not in gdf.columns and "names" in gdf.columns:
        gdf["name"] = gdf["names"].apply(
            lambda x: x.get("primary") if isinstance(x, dict) else None
        )

    # Extract bridge/tunnel flags from road struct
    # Overture stores road_flags as array of {values: [flag_names]}
    if "road" in gdf.columns:
        gdf["is_bridge"] = gdf["road"].apply(
            lambda x: _has_road_flag(x, "is_bridge") if x else False
        )
        gdf["is_tunnel"] = gdf["road"].apply(
            lambda x: _has_road_flag(x, "is_tunnel") if x else False
        )
    else:
        gdf["is_bridge"] = False
        gdf["is_tunnel"] = False

    # Extract level (may already be flat, or in level_rules)
    if "level" not in gdf.columns:
        if "level_rules" in gdf.columns:
            gdf["level"] = gdf["level_rules"].apply(_get_level_from_rules)
        else:
            gdf["level"] = 0

    # Also populate 'layer' for compatibility with existing code
    gdf["layer"] = gdf["level"]

    # Normalize road class if present
    if "class" in gdf.columns and "road_class" not in gdf.columns:
        gdf["road_class"] = gdf["class"]

    logger.info(f"Loaded {len(gdf)} Overture segments")
    return gdf


def _has_road_flag(road_struct, flag_name: str) -> bool:
    """Check if road struct contains a specific flag.

    Handles both old format (road.flags.is_bridge) and new format
    (road_flags array of {values: [flag_names]}).

    Args:
        road_struct: Road struct from Overture data
        flag_name: Flag name to check (e.g., "is_bridge")

    Returns:
        True if flag is present
    """
    if not road_struct or not isinstance(road_struct, dict):
        return False

    # Check new format: road_flags array
    road_flags = road_struct.get("road_flags") or road_struct.get("flags")
    if isinstance(road_flags, list):
        for rule in road_flags:
            if isinstance(rule, dict):
                values = rule.get("values") or []
                if flag_name in values:
                    return True
    # Check old format: flags.is_bridge (boolean)
    elif isinstance(road_flags, dict):
        return road_flags.get(flag_name, False)

    return False


def _get_level_from_rules(level_rules) -> int:
    """Extract level value from level_rules array.

    Args:
        level_rules: Array of level rule structs

    Returns:
        Level value (0 for ground level)
    """
    if not level_rules or len(level_rules) == 0:
        return 0
    first_rule = level_rules[0]
    if isinstance(first_rule, dict):
        return first_rule.get("value", 0)
    return 0
