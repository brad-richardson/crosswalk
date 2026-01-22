"""Candidate generation via spatial indexing.

Uses STRtree for efficient spatial queries to find potential matches
without O(N*M) comparisons.

Also provides DuckDB-based candidate generation for improved performance
by filtering datasets before loading into memory.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from loguru import logger
from pyproj import CRS
from shapely import LineString
from shapely.strtree import STRtree

from ..config import settings


def _create_local_projection_crs(gdf: gpd.GeoDataFrame) -> CRS | None:
    """Create local Azimuthal Equidistant CRS centered on data centroid.

    This projection has no zone boundaries (unlike UTM) and provides
    accurate distance measurements near the center point.

    Args:
        gdf: GeoDataFrame to compute centroid from

    Returns:
        CRS for local projection, or None if data is already projected
    """
    if gdf.crs is None:
        return None

    # Check if CRS is geographic (lat/lon)
    try:
        crs = CRS.from_user_input(gdf.crs)
        if not crs.is_geographic:
            return None  # Already projected
    except Exception:
        return None

    # Get centroid of bounds
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    # Check if coordinates look like geographic
    if not (-180 <= center_lon <= 180 and -90 <= center_lat <= 90):
        return None

    # Create local azimuthal equidistant CRS
    proj_string = f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m"
    return CRS.from_proj4(proj_string)


def _compute_headings_vectorized(geometries: gpd.GeoSeries) -> np.ndarray:
    """Compute headings for all geometries using vectorized operations.

    Args:
        geometries: GeoSeries of LineString geometries

    Returns:
        Array of headings in degrees (0-360)
    """
    coords = shapely.get_coordinates(geometries.values)
    n_coords = shapely.get_num_coordinates(geometries.values)

    # Get indices for first and last points of each geometry
    end_indices = np.cumsum(n_coords) - 1
    start_indices = np.concatenate([[0], end_indices[:-1] + 1])

    start_coords = coords[start_indices]
    end_coords = coords[end_indices]

    dx = end_coords[:, 0] - start_coords[:, 0]
    dy = end_coords[:, 1] - start_coords[:, 1]

    headings = np.degrees(np.arctan2(dy, dx))
    return (headings + 360) % 360


def _angle_diff_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized angle difference computation.

    Args:
        a: First array of angles in degrees
        b: Second array of angles in degrees

    Returns:
        Array of minimum angle differences (0-90 for bidirectional roads)
    """
    diff = np.abs(a - b)
    diff = np.where(diff > 180, 360 - diff, diff)
    opposite_diff = np.abs(180 - diff)
    return np.minimum(diff, opposite_diff)


@dataclass
class CandidatePair:
    """A candidate match between reference and target edges."""

    ref_id: int
    ref_idx: int  # Index in reference GeoDataFrame
    target_id: int
    target_idx: int  # Index in target GeoDataFrame
    distance_estimate: float
    heading_diff: float
    length_ratio: float


def generate_candidates(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance_m: float = None,
    max_heading_diff: float = None,  # Kept for API compatibility, not used for filtering
    max_length_ratio: float = None,  # Kept for API compatibility, not used for filtering
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
) -> list[CandidatePair]:
    """Generate candidate pairs using vectorized spatial join.

    Uses gpd.sjoin for efficient spatial joining. Heading and length ratio
    are computed for use as ML features but NOT used as blocking filters,
    since different segmentation schemes make them unreliable for filtering.

    Args:
        reference: Reference edges (Overture) GeoDataFrame
        target: Target edges (local data) GeoDataFrame
        buffer_distance_m: Search radius in meters
        max_heading_diff: Deprecated - not used for filtering (ML model handles scoring)
        max_length_ratio: Deprecated - not used for filtering (ML model handles scoring)
        ref_id_column: Column name for reference IDs
        target_id_column: Column name for target IDs

    Returns:
        List of CandidatePair objects
    """
    buffer_distance_m = buffer_distance_m or settings.buffer_distance_m
    # Note: heading/length params kept for API compatibility but not used

    logger.info(f"Generating candidates: {len(reference)} reference x {len(target)} target")
    logger.info(f"  buffer_distance_m: {buffer_distance_m}m")
    logger.info("  Note: heading/length filters disabled - ML model handles scoring")

    # Check if data is in geographic CRS and needs projection for accurate buffering
    local_crs = _create_local_projection_crs(target)
    if local_crs is not None:
        logger.info("  Projecting to local AEQD CRS for accurate spatial operations")
        target_proj = target.to_crs(local_crs)
        reference_proj = reference.to_crs(local_crs)
    else:
        target_proj = target
        reference_proj = reference

    # Prepare target with buffer geometry and pre-computed attributes
    target_prep = target_proj.copy()
    target_prep["_target_idx"] = range(len(target))
    target_prep["_target_heading"] = _compute_headings_vectorized(target_proj.geometry)
    target_prep["_target_length"] = target_proj.geometry.length
    target_prep["_target_id"] = (
        target[target_id_column] if target_id_column in target.columns else range(len(target))
    )
    # Store original geometry (in projected CRS) before buffering
    target_prep["_target_geom"] = target_prep.geometry

    # Buffer target geometries for spatial join (now in meters for projected CRS)
    target_prep = target_prep.set_geometry(target_prep.geometry.buffer(buffer_distance_m))

    # Prepare reference with pre-computed attributes
    reference_prep = reference_proj.copy()
    reference_prep["_ref_idx"] = range(len(reference))
    reference_prep["_ref_heading"] = _compute_headings_vectorized(reference_proj.geometry)
    reference_prep["_ref_length"] = reference_proj.geometry.length
    reference_prep["_ref_id"] = (
        reference[ref_id_column] if ref_id_column in reference.columns else range(len(reference))
    )
    reference_prep["_ref_geom"] = reference_prep.geometry

    # Perform spatial join (vectorized!) - in projected CRS for accurate distance
    # Keep only needed columns from reference to avoid column name conflicts
    ref_cols = ["geometry", "_ref_idx", "_ref_heading", "_ref_length", "_ref_id", "_ref_geom"]
    joined = gpd.sjoin(
        target_prep,
        reference_prep[ref_cols],
        how="inner",
        predicate="intersects",
    )

    logger.info(f"  Spatial join found {len(joined)} candidate pairs")

    if len(joined) == 0:
        return []

    # Compute heading and length ratio for use as CandidatePair attributes
    # (used by ML model as features, not as blocking filters)
    heading_diff = _angle_diff_vectorized(
        joined["_target_heading"].values,
        joined["_ref_heading"].values,
    )

    min_len = np.minimum(joined["_target_length"].values, joined["_ref_length"].values)
    max_len = np.maximum(joined["_target_length"].values, joined["_ref_length"].values)
    length_ratio = max_len / np.maximum(min_len, 0.1)

    # No blocking filters - spatial proximity is sufficient for candidate generation
    # The ML model will use heading_diff and length_ratio as scoring features
    joined_filtered = joined
    heading_diff_filtered = heading_diff
    length_ratio_filtered = length_ratio

    logger.info(f"  After spatial join: {len(joined_filtered)} candidates")

    if len(joined_filtered) == 0:
        return []

    # Compute centroid distances (vectorized)
    target_centroids = joined_filtered["_target_geom"].centroid
    ref_centroids = joined_filtered["_ref_geom"].centroid
    distances = target_centroids.distance(ref_centroids).values

    # Pre-extract arrays for fast list comprehension construction
    # This is ~6x faster than using iterrows()
    ref_ids = joined_filtered["_ref_id"].values
    ref_idxs = joined_filtered["_ref_idx"].values.astype(int)
    target_ids = joined_filtered["_target_id"].values
    target_idxs = joined_filtered["_target_idx"].values.astype(int)
    inv_length_ratios = 1.0 / length_ratio_filtered  # Normalize to 0-1

    # Build CandidatePair objects using list comprehension (vectorized extraction)
    candidates = [
        CandidatePair(
            ref_id=ref_ids[i],
            ref_idx=ref_idxs[i],
            target_id=target_ids[i],
            target_idx=target_idxs[i],
            distance_estimate=distances[i],
            heading_diff=heading_diff_filtered[i],
            length_ratio=inv_length_ratios[i],
        )
        for i in range(len(joined_filtered))
    ]

    logger.info(f"  Generated {len(candidates)} candidates")
    return candidates


def generate_candidates_iter(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance_m: float = None,
    max_heading_diff: float = None,  # Kept for API compatibility, not used
    max_length_ratio: float = None,  # Kept for API compatibility, not used
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
) -> Iterator[CandidatePair]:
    """Iterator version of generate_candidates for memory efficiency.

    Yields candidate pairs one at a time instead of building full list.
    Like generate_candidates, only uses spatial proximity for filtering.
    """
    buffer_distance_m = buffer_distance_m or settings.buffer_distance_m

    # Build spatial index on reference
    ref_tree = STRtree(reference.geometry.values)

    # Pre-compute headings for feature computation (not filtering)
    ref_headings = reference.geometry.apply(_compute_overall_heading)

    for target_idx in range(len(target)):
        target_row = target.iloc[target_idx]
        target_geom = target_row.geometry
        target_heading = _compute_overall_heading(target_geom)
        target_length = target_geom.length

        if target_id_column in target_row.index:
            target_id = target_row[target_id_column]
        else:
            target_id = target_idx

        # Buffer query - only spatial filtering
        buffered = target_geom.buffer(buffer_distance_m)
        candidate_indices = ref_tree.query(buffered)

        for ref_idx in candidate_indices:
            ref_row = reference.iloc[ref_idx]
            ref_heading = ref_headings.iloc[ref_idx]
            ref_length = ref_row.geometry.length

            ref_id = ref_row[ref_id_column] if ref_id_column in ref_row.index else ref_idx

            # Compute heading and length for ML features (not filtering)
            heading_diff = _angle_diff(target_heading, ref_heading)
            length_ratio = max(target_length, ref_length) / max(min(target_length, ref_length), 0.1)

            distance_estimate = target_geom.centroid.distance(ref_row.geometry.centroid)

            yield CandidatePair(
                ref_id=ref_id,
                ref_idx=ref_idx,
                target_id=target_id,
                target_idx=target_idx,
                distance_estimate=distance_estimate,
                heading_diff=heading_diff,
                length_ratio=1.0 / length_ratio,
            )


def _compute_overall_heading(geom: LineString) -> float:
    """Compute heading from first to last point in degrees (0-360)."""
    coords = np.array(geom.coords)
    if len(coords) < 2:
        return 0.0

    dx = coords[-1, 0] - coords[0, 0]
    dy = coords[-1, 1] - coords[0, 1]
    heading = np.degrees(np.arctan2(dy, dx))

    return (heading + 360) % 360


def _angle_diff(a: float, b: float) -> float:
    """Compute minimum angle difference in degrees (0-180).

    Handles the bidirectional nature of roads.
    """
    diff = abs(a - b)
    if diff > 180:
        diff = 360 - diff

    # Consider opposite direction
    opposite_diff = abs(180 - diff)

    return min(diff, opposite_diff)


def filter_candidates_by_name(
    candidates: list[CandidatePair],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_name_column: str = "name",
    target_name_column: str = "name",
    min_similarity: float = 0.5,
) -> list[CandidatePair]:
    """Filter candidates by name similarity.

    Useful as an additional filter when names are reliable.

    Args:
        candidates: List of candidate pairs
        reference: Reference GeoDataFrame
        target: Target GeoDataFrame
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        min_similarity: Minimum name similarity to keep candidate

    Returns:
        Filtered list of candidates
    """
    from ..features.semantic import compute_name_similarity

    filtered = []

    for cand in candidates:
        ref_name = reference.iloc[cand.ref_idx].get(ref_name_column)
        target_name = target.iloc[cand.target_idx].get(target_name_column)

        # If both have names, check similarity
        if ref_name and target_name:
            sim = compute_name_similarity(ref_name, target_name)
            if sim["token_sort_ratio"] >= min_similarity:
                filtered.append(cand)
        else:
            # Keep candidates where one or both names are missing
            filtered.append(cand)

    logger.info(f"Name filter: {len(candidates)} -> {len(filtered)} candidates")
    return filtered


def estimate_candidate_count(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance_m: float = None,
) -> int:
    """Estimate the number of candidates without generating them.

    Useful for progress bars and memory planning.
    """
    buffer_distance_m = buffer_distance_m or settings.buffer_distance_m

    # Sample-based estimation
    sample_size = min(100, len(target))
    sample_indices = np.random.choice(len(target), sample_size, replace=False)

    ref_tree = STRtree(reference.geometry.values)
    total_candidates = 0

    for idx in sample_indices:
        geom = target.iloc[idx].geometry
        buffered = geom.buffer(buffer_distance_m)
        candidates = ref_tree.query(buffered)
        total_candidates += len(candidates)

    # Extrapolate to full dataset
    estimated = int(total_candidates / sample_size * len(target))
    return estimated


def generate_candidates_duckdb(
    reference_path: Path,
    target_path: Path,
    buffer_distance_m: float = None,
    ref_id_column: str = "id",
    target_id_column: str = "id",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[CandidatePair]]:
    """Generate candidates using DuckDB spatial join for improved performance.

    This function performs a two-phase candidate generation:
    1. Use DuckDB to find candidate pairs via spatial join (IDs only)
    2. Load only segments that appear in candidates into memory
    3. Generate CandidatePair objects with heading/distance using existing logic

    This approach significantly reduces memory usage and loading time by only
    loading segments that have potential matches.

    Args:
        reference_path: Path to reference GeoParquet file
        target_path: Path to target GeoParquet file
        buffer_distance_m: Search radius in meters
        ref_id_column: Column name for reference IDs
        target_id_column: Column name for target IDs

    Returns:
        Tuple of:
        - reference GeoDataFrame (filtered to only candidates)
        - target GeoDataFrame (filtered to only candidates)
        - list of CandidatePair objects
    """
    from ..data.duckdb_loader import load_filtered_by_ids, spatial_join_parquet

    buffer_distance_m = buffer_distance_m or settings.buffer_distance_m

    logger.info(f"Generating candidates with DuckDB: {reference_path} x {target_path}")
    logger.info(f"  buffer_distance_m: {buffer_distance_m}m")

    # Step 1: Use DuckDB spatial join to find candidate pairs
    pairs_df, ref_ids, target_ids = spatial_join_parquet(
        reference_path=reference_path,
        target_path=target_path,
        buffer_distance_m=buffer_distance_m,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )

    if len(pairs_df) == 0:
        logger.warning("DuckDB spatial join found no candidates")
        return (
            gpd.GeoDataFrame(crs="EPSG:4326"),
            gpd.GeoDataFrame(crs="EPSG:4326"),
            [],
        )

    logger.info(f"  Found {len(ref_ids)} reference segments with candidates")
    logger.info(f"  Found {len(target_ids)} target segments with candidates")

    # Step 2: Load only segments that appear in candidates
    reference = load_filtered_by_ids(reference_path, ref_id_column, ref_ids)
    target = load_filtered_by_ids(target_path, target_id_column, target_ids)

    logger.info(f"  Loaded {len(reference)} reference segments")
    logger.info(f"  Loaded {len(target)} target segments")

    if len(reference) == 0 or len(target) == 0:
        logger.warning("No segments loaded after filtering - check ID columns")
        return reference, target, []

    # Step 3: Generate CandidatePair objects using existing logic
    # This computes heading diff, length ratio, and distance for ML features
    candidates = generate_candidates(
        reference=reference,
        target=target,
        buffer_distance_m=buffer_distance_m,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )

    return reference, target, candidates
