"""Fetch Overture Transportation data for a bounding box.

Uses the overturemaps-py library to query Overture GeoParquet files
and automatically handles release version detection.
"""

from pathlib import Path

import geopandas as gpd
from loguru import logger
from overturemaps.core import geodataframe, get_latest_release
from pydantic import BaseModel

from ..config import DATA_VERSION, SCHEMA_VERSION, TRANSFORM_VERSION
from ..utils import filter_to_linestrings
from .metadata import FetchMetadata, save_metadata

# Overture segment classes to exclude (non-road transport)
# These are valid Overture transportation classes but not roads
EXCLUDED_CLASSES = {
    "railway",  # Rail lines
    "ferry",  # Ferry routes
    "aerialway",  # Ski lifts, cable cars, etc.
}

# Default buffer distance (meters) for fetching Overture data
# This ensures we get complete network topology at edges by including:
# - Roads running parallel just outside the boundary
# - Road connections/intersections just outside the target area
# - Complete network for integration purposes
# Note: Partial overlaps ARE included (features intersecting bbox), but
# the buffer ensures we capture nearby parallel roads and complete connectivity.
DEFAULT_OVERTURE_BUFFER_M = 1000.0


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

    def expand(self, buffer_m: float) -> "BoundingBox":
        """Expand the bounding box by a buffer distance in meters.

        Converts meters to approximate degrees at the center latitude.
        This is useful for fetching extra data around the edges to avoid
        fringe effects during integration.

        Args:
            buffer_m: Buffer distance in meters

        Returns:
            New BoundingBox expanded by the buffer distance
        """
        import math

        # Approximate center latitude
        center_lat = (self.ymin + self.ymax) / 2

        # Convert meters to degrees (approximate)
        # 1 degree latitude = ~111,320 meters
        # 1 degree longitude = ~111,320 * cos(latitude) meters
        lat_buffer = buffer_m / 111320.0
        lon_buffer = buffer_m / (111320.0 * math.cos(math.radians(center_lat)))

        return BoundingBox(
            xmin=self.xmin - lon_buffer,
            ymin=self.ymin - lat_buffer,
            xmax=self.xmax + lon_buffer,
            ymax=self.ymax + lat_buffer,
        )


def fetch_overture_segments(
    bbox: BoundingBox,
    output_path: Path,
    release: str | None = None,
    original_bbox: BoundingBox | None = None,
    buffer_m: float | None = None,
) -> Path:
    """Download Overture road segments for a bounding box.

    Uses the overturemaps-py library which automatically detects
    the latest release if not specified.

    Filters out non-road transport types (railways, ferries, aerialways).

    Args:
        bbox: Bounding box in WGS84 coordinates (may be buffered)
        output_path: Path for output GeoParquet file
        release: Overture release version (default: latest)
        original_bbox: Original unbuffered bbox (for metadata tracking)
        buffer_m: Buffer distance in meters that was applied to bbox

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

    initial_count = len(gdf)

    # Filter to road subtype only
    if "subtype" in gdf.columns:
        gdf = gdf[gdf["subtype"] == "road"]
        logger.debug(f"Filtered to road subtype: {initial_count} -> {len(gdf)} segments")

    # Filter out excluded classes (railways, ferries, etc.)
    if "class" in gdf.columns:
        pre_filter_count = len(gdf)
        gdf = gdf[~gdf["class"].isin(EXCLUDED_CLASSES)]
        excluded_count = pre_filter_count - len(gdf)
        if excluded_count > 0:
            logger.info(
                f"Filtered out {excluded_count} non-road segments (railways, ferries, etc.)"
            )

    logger.info(f"Fetched {len(gdf)} road segments")

    # Ensure CRS is set (Overture data is always WGS84)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Filter to LineString geometries only (drop MultiLineStrings)
    gdf = filter_to_linestrings(gdf, source_name="overture_segments")

    # Drop existing bbox column if present (newer Overture releases include it)
    # to avoid conflict with write_covering_bbox
    if "bbox" in gdf.columns:
        gdf = gdf.drop(columns=["bbox"])

    # Save to parquet with bbox metadata for DuckDB spatial predicate pushdown
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, write_covering_bbox=True)

    # Save fetch metadata
    metadata = FetchMetadata(
        source="overture",
        release=release,
        bbox=original_bbox.to_tuple() if original_bbox else bbox.to_tuple(),
        bbox_buffered=bbox.to_tuple() if buffer_m else None,
        bbox_buffer_m=buffer_m,
        feature_count=len(gdf),
        geometry_types=list(gdf.geometry.geom_type.unique()) if len(gdf) > 0 else [],
        filters={"subtype": "road", "excluded_classes": list(EXCLUDED_CLASSES)},
        # Version tracking
        transform_version=TRANSFORM_VERSION,
        schema_version=SCHEMA_VERSION,
        data_version=DATA_VERSION,
        # ID column tracking (Overture uses 'id' as the ID column)
        id_column="id",
    )
    meta_path = save_metadata(output_path, metadata)
    logger.debug(f"Saved fetch metadata to {meta_path}")

    logger.info(f"Saved Overture segments to {output_path}")
    return output_path


def fetch_overture_connectors(
    bbox: BoundingBox,
    output_path: Path,
    release: str | None = None,
    original_bbox: BoundingBox | None = None,
    buffer_m: float | None = None,
) -> Path:
    """Download Overture connectors (intersections) for a bounding box.

    Uses the overturemaps-py library which automatically detects
    the latest release if not specified.

    Args:
        bbox: Bounding box in WGS84 coordinates (may be buffered)
        output_path: Path for output GeoParquet file
        release: Overture release version (default: latest)
        original_bbox: Original unbuffered bbox (for metadata tracking)
        buffer_m: Buffer distance in meters that was applied to bbox

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

    # Drop existing bbox column if present (newer Overture releases include it)
    # to avoid conflict with write_covering_bbox
    if "bbox" in gdf.columns:
        gdf = gdf.drop(columns=["bbox"])

    # Save to parquet with bbox metadata for DuckDB spatial predicate pushdown
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, write_covering_bbox=True)

    # Save fetch metadata
    metadata = FetchMetadata(
        source="overture",
        release=release,
        bbox=original_bbox.to_tuple() if original_bbox else bbox.to_tuple(),
        bbox_buffered=bbox.to_tuple() if buffer_m else None,
        bbox_buffer_m=buffer_m,
        feature_count=len(gdf),
        geometry_types=list(gdf.geometry.geom_type.unique()) if len(gdf) > 0 else [],
        # Version tracking
        transform_version=TRANSFORM_VERSION,
        schema_version=SCHEMA_VERSION,
        data_version=DATA_VERSION,
        # ID column tracking (Overture uses 'id' as the ID column)
        id_column="id",
    )
    meta_path = save_metadata(output_path, metadata)
    logger.debug(f"Saved fetch metadata to {meta_path}")

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

    # Filter to LineString geometries only (drop MultiLineStrings)
    gdf = filter_to_linestrings(gdf, source_name=str(path.name))

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
