"""Fetch Overture Transportation data for a bounding box.

Uses the overturemaps-py library to query Overture GeoParquet files
and automatically handles release version detection.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
from loguru import logger
from overturemaps.core import geodataframe, get_latest_release
from pydantic import BaseModel

from ..config import DATA_VERSION, SCHEMA_VERSION, TRANSFORM_VERSION
from ..utils import filter_to_linestrings
from ..utils.linear_ref import (
    LinearReferencedAttribute,
    create_trivial_lr,
    normalize_ranges,
)
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


def get_buffered_bbox(
    original_bbox: BoundingBox,
    buffer_m: float | None,
    default_buffer_m: float,
) -> tuple[BoundingBox, float | None]:
    """Apply buffer to bbox, using default if not specified.

    Args:
        original_bbox: The original bounding box
        buffer_m: User-specified buffer in meters, or None to use default
        default_buffer_m: Default buffer to use when buffer_m is None

    Returns:
        Tuple of (buffered_bbox, effective_buffer_m)
        - If buffer is 0, returns (original_bbox, None)
        - Otherwise returns (expanded_bbox, buffer_used)
    """
    # Use default if not specified
    effective_buffer = buffer_m if buffer_m is not None else default_buffer_m

    # Apply buffer if positive
    if effective_buffer > 0:
        return original_bbox.expand(effective_buffer), effective_buffer

    # Buffer explicitly disabled (0) or negative
    return original_bbox, None


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

    # Extract linear-referenced attributes (names_lr, subclass_lr, etc.)
    gdf = extract_lr_attributes(gdf)

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

    # Extract linear-referenced attributes
    gdf = extract_lr_attributes(gdf)

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


# -----------------------------------------------------------------------------
# Linear-Referenced Attribute Parsing
# -----------------------------------------------------------------------------

# Priority order for name variants (lower = higher priority)
NAME_VARIANT_PRIORITY = {
    "common": 0,
    "official": 1,
    "alternate": 2,
    "alt": 2,
    "short": 3,
    "colloquial": 4,
    "historical": 5,
}


def _get_variant_priority(variant: str | None) -> int:
    """Get priority value for a name variant.

    Args:
        variant: Name variant string (e.g., "common", "alt")

    Returns:
        Priority value (lower = higher priority), default 10 for unknown
    """
    if variant is None:
        return 0  # No variant = most preferred (common)
    return NAME_VARIANT_PRIORITY.get(variant.lower(), 10)


def _get_language_priority(language: str | None) -> int:
    """Get priority value for language.

    Priority order:
    1. Bare names (no language specified) - highest priority
    2. English names - preferred for English-speaking reviewers
    3. Other language-specific names

    Args:
        language: Language code or None for bare names

    Returns:
        Priority value (lower = higher priority)
    """
    if language is None:
        return 0  # Bare names highest priority
    if language.lower().startswith("en"):  # "en", "en-US", "en-GB", etc.
        return 1  # English preferred over other languages
    return 2  # Other languages


def _extract_range_from_rule(rule: dict) -> tuple[float, float] | None:
    """Extract geometric range from a rule's scope.

    Overture rules can have geometric scopes like:
    - {"between": [0.2, 0.6]}
    - No between = applies to entire segment

    Args:
        rule: Rule dict that may contain scope/between

    Returns:
        Tuple of (start, end) or None if no geometric scope
    """
    # Check for "between" at the top level (common format)
    between = rule.get("between")
    if between is not None:
        # Convert numpy array to list if needed
        if hasattr(between, "tolist"):
            between = between.tolist()
        if isinstance(between, (list, tuple)) and len(between) == 2:
            return (float(between[0]), float(between[1]))

    # Check for scope.between (alternative format)
    scope = rule.get("scope")
    if scope and isinstance(scope, dict):
        between = scope.get("between")
        if between is not None:
            # Convert numpy array to list if needed
            if hasattr(between, "tolist"):
                between = between.tolist()
            if isinstance(between, (list, tuple)) and len(between) == 2:
                return (float(between[0]), float(between[1]))

    return None


def parse_names_lr(names_dict: dict | None) -> LinearReferencedAttribute:
    """Parse Overture names struct into linear-referenced attribute.

    Overture names structure:
    - primary: The default/main name (string)
    - rules: Array of name rules with potential geometric scopes

    Each rule may have:
    - value: The name string
    - variant: Type (common, alt, short, etc.)
    - language: Language code or None for bare names
    - between: [start, end] geometric scope (0-1 fractions)

    Priority for overlapping ranges:
    1. Variant priority (common > alt > short)
    2. Language priority (bare > language-specific)
    3. Order in array (first wins for ties)

    Args:
        names_dict: Overture names struct or None

    Returns:
        LinearReferencedAttribute with normalized name ranges
    """
    # Handle None/missing
    if not names_dict or not isinstance(names_dict, dict):
        return create_trivial_lr(None)

    # Get default value (primary name)
    primary = names_dict.get("primary")
    default_value = primary if isinstance(primary, str) else None

    # Get rules array (may be list or numpy array from parquet)
    rules = names_dict.get("rules")
    if rules is None:
        return create_trivial_lr(default_value)
    # Convert numpy array to list if needed
    if hasattr(rules, "tolist"):
        rules = rules.tolist()
    if not isinstance(rules, list) or len(rules) == 0:
        return create_trivial_lr(default_value)

    # Build list of (start, end, value, priority) tuples
    rule_tuples: list[tuple[float, float, str, int]] = []

    for rule in rules:
        if not isinstance(rule, dict):
            continue

        # Extract value
        value = rule.get("value")
        if not isinstance(value, str):
            continue

        # Extract range (default to full segment)
        range_tuple = _extract_range_from_rule(rule)
        if range_tuple:
            start, end = range_tuple
        else:
            start, end = 0.0, 1.0

        # Calculate priority (lower = higher priority)
        variant = rule.get("variant")
        language = rule.get("language")
        variant_priority = _get_variant_priority(variant)
        language_priority = _get_language_priority(language)
        # Combine priorities: variant is more important, then language
        priority = variant_priority * 10 + language_priority

        rule_tuples.append((start, end, value, priority))

    if not rule_tuples:
        return create_trivial_lr(default_value)

    return normalize_ranges(rule_tuples, default_value)


def _parse_simple_rules_lr(
    rules: list | None,
    default_value: Any,
    value_extractor: Callable[[dict], Any | None],
) -> LinearReferencedAttribute:
    """Generic parser for simple LR rules (subclass, level, road_flags).

    This handles the common pattern of:
    1. None/empty check with numpy array conversion
    2. Looping through rules to extract values
    3. Building rule_tuples with index-based priority
    4. Normalizing ranges

    Args:
        rules: Array of rule dicts (may be numpy array)
        default_value: Default value for gaps and empty input
        value_extractor: Function that extracts value from a rule dict,
                        returns None to skip the rule

    Returns:
        LinearReferencedAttribute with normalized ranges
    """
    if rules is None:
        return create_trivial_lr(default_value)

    # Convert numpy array to list if needed
    if hasattr(rules, "tolist"):
        rules = rules.tolist()

    if not isinstance(rules, list) or len(rules) == 0:
        return create_trivial_lr(default_value)

    # Build list of (start, end, value, priority) tuples
    rule_tuples: list[tuple[float, float, Any, int]] = []

    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue

        value = value_extractor(rule)
        if value is None:
            continue

        range_tuple = _extract_range_from_rule(rule)
        if range_tuple:
            start, end = range_tuple
        else:
            start, end = 0.0, 1.0

        # Use index as priority (earlier rules win)
        rule_tuples.append((start, end, value, i))

    if not rule_tuples:
        return create_trivial_lr(default_value)

    return normalize_ranges(rule_tuples, default_value)


def parse_subclass_rules_lr(
    subclass_rules: list | None, default_subclass: str | None = None
) -> LinearReferencedAttribute:
    """Parse Overture subclass_rules into linear-referenced attribute.

    Each rule has:
    - value: The subclass string
    - between: [start, end] geometric scope (optional)

    Args:
        subclass_rules: Array of subclass rule structs
        default_subclass: Default subclass from flat column

    Returns:
        LinearReferencedAttribute with normalized subclass ranges
    """

    def extract_subclass(rule: dict) -> str | None:
        value = rule.get("value")
        return value if isinstance(value, str) else None

    return _parse_simple_rules_lr(subclass_rules, default_subclass, extract_subclass)


def parse_level_rules_lr(level_rules: list | None) -> LinearReferencedAttribute:
    """Parse Overture level_rules into linear-referenced attribute.

    Each rule has:
    - value: The level integer (0 = ground, positive = elevated, negative = underground)
    - between: [start, end] geometric scope (optional)

    Args:
        level_rules: Array of level rule structs

    Returns:
        LinearReferencedAttribute with normalized level ranges
    """

    def extract_level(rule: dict) -> int | None:
        value = rule.get("value")
        if isinstance(value, int):
            return value
        # Try to convert
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return _parse_simple_rules_lr(level_rules, 0, extract_level)  # 0 = ground level


def parse_road_flags_lr(road_flags: list | None) -> LinearReferencedAttribute:
    """Parse Overture road_flags into linear-referenced attribute.

    Each rule has:
    - values: List of flag strings (e.g., ["is_bridge", "is_link"])
    - between: [start, end] geometric scope (optional)

    The value stored is a frozenset of flags for hashability.

    Args:
        road_flags: Array of road flag rule structs

    Returns:
        LinearReferencedAttribute with normalized flag ranges
    """

    def extract_flags(rule: dict) -> frozenset[str] | None:
        values = rule.get("values")
        if values is None:
            return None
        # Convert numpy array to list if needed
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, list):
            return None
        # Convert to frozenset of strings
        return frozenset(str(v) for v in values if v)

    return _parse_simple_rules_lr(road_flags, frozenset(), extract_flags)


def extract_lr_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract linear-referenced attributes from Overture columns.

    Adds *_lr columns for each supported attribute type:
    - names_lr: From names struct
    - subclass_lr: From subclass_rules array
    - level_lr: From level_rules array
    - road_flags_lr: From road.road_flags array

    The LR columns store JSON-serializable dict lists for parquet compatibility.

    Args:
        gdf: GeoDataFrame with Overture columns

    Returns:
        GeoDataFrame with added *_lr columns
    """
    # Parse names linear reference
    if "names" in gdf.columns:
        gdf["names_lr"] = gdf["names"].apply(lambda x: parse_names_lr(x).to_dict_list())
    else:
        gdf["names_lr"] = gdf.get("name", None).apply(lambda x: create_trivial_lr(x).to_dict_list())

    # Parse subclass linear reference
    if "subclass_rules" in gdf.columns:
        default_subclass = gdf.get("subclass", None)
        gdf["subclass_lr"] = gdf.apply(
            lambda row: parse_subclass_rules_lr(
                row.get("subclass_rules"),
                row.get("subclass") if default_subclass is not None else None,
            ).to_dict_list(),
            axis=1,
        )
    elif "subclass" in gdf.columns:
        gdf["subclass_lr"] = gdf["subclass"].apply(lambda x: create_trivial_lr(x).to_dict_list())
    else:
        gdf["subclass_lr"] = [[{"start": 0.0, "end": 1.0, "value": None}] for _ in range(len(gdf))]

    # Parse level linear reference
    if "level_rules" in gdf.columns:
        gdf["level_lr"] = gdf["level_rules"].apply(lambda x: parse_level_rules_lr(x).to_dict_list())
    elif "level" in gdf.columns:
        gdf["level_lr"] = gdf["level"].apply(
            lambda x: create_trivial_lr(x if isinstance(x, int) else 0).to_dict_list()
        )
    else:
        gdf["level_lr"] = [[{"start": 0.0, "end": 1.0, "value": 0}] for _ in range(len(gdf))]

    # Parse road_flags linear reference
    if "road" in gdf.columns:
        gdf["road_flags_lr"] = gdf["road"].apply(
            lambda x: parse_road_flags_lr(
                x.get("road_flags") if isinstance(x, dict) else None
            ).to_dict_list()
        )
    else:
        gdf["road_flags_lr"] = [[{"start": 0.0, "end": 1.0, "value": []}] for _ in range(len(gdf))]

    # Convert frozensets to lists for JSON serialization in road_flags_lr
    def convert_frozensets(lr_data: list) -> list:
        """Convert frozenset values to lists for JSON compatibility."""
        result = []
        for item in lr_data:
            value = item.get("value")
            if isinstance(value, frozenset):
                item = {**item, "value": sorted(value)}
            result.append(item)
        return result

    gdf["road_flags_lr"] = gdf["road_flags_lr"].apply(convert_frozensets)

    return gdf
