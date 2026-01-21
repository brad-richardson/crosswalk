"""Fetch OSM road data from Geofabrik PBF extracts.

This module downloads regional PBF extracts from Geofabrik, clips to the
requested bounding box using osmium CLI, and parses roads using pyosmium.
"""

from pathlib import Path

import geopandas as gpd
from loguru import logger

from ..config import settings
from .metadata import FetchMetadata, save_metadata
from .osm_download import download_and_extract
from .osm_pbf import parse_pbf
from .overture import BoundingBox

# Default buffer distance (meters) for fetching OSM data
# This ensures we get complete network topology at edges by including:
# - Roads running parallel just outside the boundary
# - Road connections/intersections just outside the target area
# - Complete network for integration purposes
# Note: OSM ways with any node inside bbox are included (partial overlaps),
# but the buffer ensures we capture nearby parallel roads and complete connectivity.
DEFAULT_OSM_BUFFER_M = 1000.0


def fetch_osm_data(
    bbox: BoundingBox,
    output_dir: Path,
    cache_dir: Path | None = None,
    force_download: bool = False,
    keep_pbf: bool = False,
    original_bbox: BoundingBox | None = None,
    buffer_m: float | None = None,
    name: str = "osm",
) -> tuple[Path, Path]:
    """Download and parse OSM road data (ways/nodes) for a bounding box.

    This is the main entry point for OSM data fetching. It:
    1. Finds the smallest Geofabrik region containing the bbox
    2. Downloads the regional PBF (cached for 24 hours)
    3. Extracts the bbox area using osmium CLI (or pyosmium fallback)
    4. Parses roads (ways) and intersections (nodes) using pyosmium
    5. Saves as GeoParquet in Overture-compatible schema

    Args:
        bbox: Bounding box in WGS84 coordinates (may be buffered)
        output_dir: Directory for output GeoParquet files
        cache_dir: Directory for caching PBF files (default from settings)
        force_download: Force re-download even if cached
        keep_pbf: Keep the extracted bbox PBF file
        original_bbox: Original unbuffered bbox (for metadata tracking)
        buffer_m: Buffer distance in meters that was applied to bbox
        name: Dataset name for output files (e.g., "osm" -> "osm_segments.parquet")

    Returns:
        Tuple of (segments_path, connectors_path)
    """
    cache_dir = cache_dir or settings.pbf_cache_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching OSM data for bbox: {bbox}")

    # Download and extract PBF
    result_path = download_and_extract(
        bbox=bbox,
        output_dir=output_dir,
        cache_dir=cache_dir,
        force=force_download,
    )

    # Check if pyosmium extraction was used (indicated by marker file)
    pyosmium_marker = output_dir / ".pyosmium_extracted"
    if pyosmium_marker.exists():
        # Data was extracted directly by pyosmium - load and transform
        logger.info("Using pyosmium-extracted data")
        pyosmium_marker.unlink()

        roads_path = output_dir / "osm_roads_raw.parquet"
        connectors_raw_path = output_dir / "osm_connectors_raw.parquet"

        roads_gdf = gpd.read_parquet(roads_path)
        connectors_gdf = gpd.read_parquet(connectors_raw_path)

        # Cleanup raw files
        roads_path.unlink()
        connectors_raw_path.unlink()
    else:
        # Standard path - parse from PBF
        pbf_path = result_path
        roads_gdf, connectors_gdf = parse_pbf(pbf_path)

        # Cleanup extracted PBF unless keeping
        if not keep_pbf and pbf_path.exists():
            pbf_path.unlink()
            logger.debug(f"Removed temporary PBF: {pbf_path}")

    # Transform roads to match expected schema (for load_osm_roads compatibility)
    roads_gdf = _transform_to_overture_schema(roads_gdf)

    # Transform connectors to match expected schema
    connectors_gdf = _transform_connectors_schema(connectors_gdf)

    # Save to parquet with bbox metadata for DuckDB spatial predicate pushdown
    segments_path = output_dir / f"{name}_segments.parquet"
    connectors_path = output_dir / f"{name}_connectors.parquet"

    roads_gdf.to_parquet(segments_path, write_covering_bbox=True)
    connectors_gdf.to_parquet(connectors_path, write_covering_bbox=True)

    # Save fetch metadata for segments (roads/ways)
    segments_metadata = FetchMetadata(
        source="osm",
        source_url="https://download.geofabrik.de/",
        bbox=original_bbox.to_tuple() if original_bbox else bbox.to_tuple(),
        bbox_buffered=bbox.to_tuple() if buffer_m else None,
        bbox_buffer_m=buffer_m,
        feature_count=len(roads_gdf),
        geometry_types=list(roads_gdf.geometry.geom_type.unique()) if len(roads_gdf) > 0 else [],
        notes=f"OSM ways for dataset '{name}' fetched from Geofabrik regional PBF extract",
    )
    save_metadata(segments_path, segments_metadata)

    # Save fetch metadata for connectors (intersections/nodes)
    connectors_metadata = FetchMetadata(
        source="osm",
        source_url="https://download.geofabrik.de/",
        bbox=original_bbox.to_tuple() if original_bbox else bbox.to_tuple(),
        bbox_buffered=bbox.to_tuple() if buffer_m else None,
        bbox_buffer_m=buffer_m,
        feature_count=len(connectors_gdf),
        geometry_types=list(connectors_gdf.geometry.geom_type.unique())
        if len(connectors_gdf) > 0
        else [],
        notes=f"OSM nodes for dataset '{name}' fetched from Geofabrik regional PBF extract",
    )
    save_metadata(connectors_path, connectors_metadata)

    logger.info(f"Saved {len(roads_gdf)} {name} segments to {segments_path}")
    logger.info(f"Saved {len(connectors_gdf)} {name} connectors to {connectors_path}")

    return segments_path, connectors_path


def fetch_osm_segments(
    bbox: BoundingBox,
    output_path: Path,
    cache_dir: Path | None = None,
    force_download: bool = False,
    keep_pbf: bool = False,
    original_bbox: BoundingBox | None = None,
    buffer_m: float | None = None,
    name: str = "osm",
) -> Path:
    """Download and parse OSM road segments (ways) for a bounding box.

    Convenience wrapper around fetch_osm_data that returns only segments.

    Args:
        bbox: Bounding box in WGS84 coordinates (may be buffered)
        output_path: Path for output GeoParquet file
        cache_dir: Directory for caching PBF files (default from settings)
        force_download: Force re-download even if cached
        keep_pbf: Keep the extracted bbox PBF file
        original_bbox: Original unbuffered bbox (for metadata tracking)
        buffer_m: Buffer distance in meters that was applied to bbox
        name: Dataset name for output files

    Returns:
        Path to the output GeoParquet file
    """
    segments_path, _ = fetch_osm_data(
        bbox=bbox,
        output_dir=output_path.parent,
        cache_dir=cache_dir,
        force_download=force_download,
        keep_pbf=keep_pbf,
        original_bbox=original_bbox,
        buffer_m=buffer_m,
        name=name,
    )

    # Move to requested path if different
    if segments_path != output_path:
        segments_path.rename(output_path)
        return output_path

    return segments_path


def _transform_to_overture_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Transform PBF-parsed data to match Overture schema.

    Builds the full Overture-compatible schema at fetch time:
    - id, geometry, names, class, subtype, sources
    - road_flags, level_rules (transformed from OSM tags)
    - source_tags (preserved for debugging/advanced use)
    """
    if len(gdf) == 0:
        return gdf

    result = gdf.copy()

    # Extract class from tags (highway value)
    result["class"] = result["tags"].apply(
        lambda t: t.get("highway", "unclassified") if isinstance(t, dict) else "unclassified"
    )

    # Build names struct to match Overture format
    result["names"] = result["name"].apply(lambda n: {"primary": n} if n else None)

    # Subtype is always 'road' for highway features
    result["subtype"] = "road"

    # Sources array with record_id
    result["sources"] = result["id"].apply(
        lambda osm_id: [{"dataset": "OpenStreetMap", "record_id": osm_id}]
    )

    # Build road_flags from tags (matching Overture schema)
    result["road_flags"] = result.apply(
        lambda row: _build_road_flags(row["tags"], row["class"]),
        axis=1,
    )

    # Build level_rules from tags
    result["level_rules"] = result["tags"].apply(_build_level_rules)

    # Rename tags to source_tags for clarity
    result["source_tags"] = result["tags"]

    # Drop internal columns
    columns_to_drop = ["tags", "name", "node_ids"]
    result = result.drop(columns=[c for c in columns_to_drop if c in result.columns])

    return result


def _transform_connectors_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Transform connector data to match Overture-like schema."""
    if len(gdf) == 0:
        return gdf

    result = gdf.copy()

    # Sources array with record_id (OSM node ID)
    result["sources"] = result["id"].apply(
        lambda osm_id: [{"dataset": "OpenStreetMap", "record_id": osm_id}]
    )

    return result


def load_osm_roads(path: Path) -> gpd.GeoDataFrame:
    """Load OSM roads from GeoParquet and add convenience fields.

    The data already has Overture schema (road_flags, level_rules, names, etc.)
    from fetch time. This function adds flat convenience fields for downstream
    processing.

    Args:
        path: Path to GeoParquet file

    Returns:
        GeoDataFrame with OSM roads and flat convenience fields
    """
    logger.info(f"Loading OSM roads from {path}")
    gdf = gpd.read_parquet(path)

    # Extract flat name from names struct
    if "names" in gdf.columns:
        gdf["name"] = gdf["names"].apply(
            lambda x: x.get("primary") if isinstance(x, dict) else None
        )
    else:
        gdf["name"] = None

    # Extract flat fields from road_flags (already in Overture schema)
    if "road_flags" in gdf.columns:
        gdf["is_bridge"] = gdf["road_flags"].apply(lambda f: _has_flag(f, "is_bridge"))
        gdf["is_tunnel"] = gdf["road_flags"].apply(lambda f: _has_flag(f, "is_tunnel"))
    else:
        gdf["is_bridge"] = False
        gdf["is_tunnel"] = False

    # Extract level from level_rules
    if "level_rules" in gdf.columns:
        gdf["level"] = gdf["level_rules"].apply(_get_level_from_rules)
    else:
        gdf["level"] = 0

    # Also populate 'layer' for compatibility with existing code
    gdf["layer"] = gdf["level"]

    # Normalize road class
    if "class" in gdf.columns:
        gdf["road_class"] = gdf["class"]
        gdf["road_class_normalized"] = gdf["class"].apply(_normalize_road_class)
    else:
        gdf["road_class"] = "unclassified"
        gdf["road_class_normalized"] = "unclassified"

    logger.info(f"Loaded {len(gdf)} OSM road segments")
    return gdf


def _build_road_flags(source_tags, road_class) -> list:
    """Build Overture road_flags array from OSM source_tags.

    Matches combobulator logic from tf-data-platform/overture_transportation.
    """
    if not source_tags:
        source_tags = {}

    flags = []

    # Bridge: explicit whitelist of valid bridge values from OSM wiki
    bridge = source_tags.get("bridge", "")
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

    # Tunnel: value == 'yes' or 'building_passage'
    tunnel = source_tags.get("tunnel", "")
    if tunnel in ("yes", "building_passage"):
        flags.append("is_tunnel")

    # Covered: value == 'yes'
    if source_tags.get("covered") == "yes":
        flags.append("is_covered")

    # Abandoned/disused: value == 'yes'
    if source_tags.get("abandoned") == "yes" or source_tags.get("disused") == "yes":
        flags.append("is_abandoned")

    # Indoor: value != 'no' (and not empty)
    indoor = source_tags.get("indoor", "")
    if indoor and indoor != "no":
        flags.append("is_indoor")

    # Construction: only when explicitly 'yes'
    if source_tags.get("construction") == "yes":
        flags.append("is_under_construction")

    # Link: class ends with '_link'
    if road_class and str(road_class).endswith("_link"):
        flags.append("is_link")

    if flags:
        return [{"values": flags}]
    return []


def _build_level_rules(source_tags) -> list:
    """Build Overture level_rules array from OSM source_tags."""
    if not source_tags:
        return []

    layer = source_tags.get("layer")
    if layer is None:
        return []

    try:
        level = int(layer)
        if level == 0:
            return []  # Ground level is omitted
        return [{"value": level}]
    except (ValueError, TypeError):
        return []


def _has_flag(road_flags, flag_name: str) -> bool:
    """Check if road_flags array contains a specific flag."""
    if road_flags is None or (hasattr(road_flags, "__len__") and len(road_flags) == 0):
        return False
    for rule in road_flags:
        if isinstance(rule, dict):
            values = rule.get("values", [])
            # Handle numpy arrays
            if hasattr(values, "__iter__") and flag_name in values:
                return True
    return False


def _get_level_from_rules(level_rules) -> int:
    """Extract level value from level_rules array."""
    if level_rules is None or (hasattr(level_rules, "__len__") and len(level_rules) == 0):
        return 0
    first_rule = level_rules[0]
    if isinstance(first_rule, dict):
        return first_rule.get("value", 0)
    return 0


def _normalize_road_class(road_class: str | None) -> str:
    """Normalize road class to standard values.

    Args:
        road_class: Overture road class value

    Returns:
        Normalized road class string
    """
    if road_class is None:
        return "unclassified"

    road_class = road_class.lower()

    # Map to standard classes (Overture classes are already normalized,
    # but handle link variants)
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
        "footway": "path",
        "cycleway": "path",
        "pedestrian": "path",
        "steps": "path",
        "bridleway": "path",
    }

    return class_mapping.get(road_class, "unclassified")
