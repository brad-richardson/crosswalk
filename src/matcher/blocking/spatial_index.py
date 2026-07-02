"""Candidate generation via spatial indexing.

Uses STRtree for efficient spatial queries to find potential matches
without O(N*M) comparisons.

Also provides DuckDB-based candidate generation for improved performance
by filtering datasets before loading into memory.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
from loguru import logger
from pyproj import CRS
from shapely import LineString
from shapely.strtree import STRtree

from ..config import NAMES_COLUMN, settings


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

    ref_id: Any  # ID from reference dataset (str for GERS UUIDs, int for OSM IDs)
    ref_idx: int  # Index in reference GeoDataFrame
    target_id: Any  # ID from target dataset (str or int depending on source)
    target_idx: int  # Index in target GeoDataFrame
    distance_estimate: float
    heading_diff: float


@dataclass
class CandidateBatch:
    """Memory-efficient batch storage for candidate pairs using numpy arrays.

    Instead of 8.4M CandidatePair Python objects (~9.5GB), stores data as
    numpy arrays (~67MB for indices + IDs as object array).

    Provides iteration interface compatible with list[CandidatePair] for
    backward compatibility.
    """

    ref_ids: np.ndarray  # object array of IDs
    ref_idxs: np.ndarray  # int32 array of DataFrame indices
    target_ids: np.ndarray  # object array of IDs
    target_idxs: np.ndarray  # int32 array of DataFrame indices
    distances: np.ndarray  # float32 array
    heading_diffs: np.ndarray  # float32 array

    def __len__(self) -> int:
        return len(self.ref_idxs)

    def __iter__(self):
        """Iterate yielding CandidatePair objects (for backward compatibility)."""
        for i in range(len(self)):
            yield CandidatePair(
                ref_id=self.ref_ids[i],
                ref_idx=int(self.ref_idxs[i]),
                target_id=self.target_ids[i],
                target_idx=int(self.target_idxs[i]),
                distance_estimate=float(self.distances[i]),
                heading_diff=float(self.heading_diffs[i]),
            )

    def __getitem__(self, idx: int) -> CandidatePair:
        """Get single CandidatePair by index."""
        return CandidatePair(
            ref_id=self.ref_ids[idx],
            ref_idx=self.ref_idxs[idx].item(),
            target_id=self.target_ids[idx],
            target_idx=self.target_idxs[idx].item(),
            distance_estimate=self.distances[idx].item(),
            heading_diff=self.heading_diffs[idx].item(),
        )

    def get_unique_indices(self) -> tuple[set[int], set[int]]:
        """Get unique reference and target indices efficiently."""
        return set(self.ref_idxs.tolist()), set(self.target_idxs.tolist())

    def get_index_pairs(self) -> list[tuple[int, int]]:
        """Get list of (ref_idx, target_idx) tuples for work distribution."""
        return list(zip(self.ref_idxs.tolist(), self.target_idxs.tolist()))

    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        # Numeric arrays
        arr_bytes = (
            self.ref_idxs.nbytes
            + self.target_idxs.nbytes
            + self.distances.nbytes
            + self.heading_diffs.nbytes
        )
        # Object arrays (IDs) - include nbytes plus estimated string storage
        # nbytes for object arrays only counts pointer storage (~8 bytes each)
        # Actual string content adds ~50 bytes per ID on average
        id_arr_bytes = self.ref_ids.nbytes + self.target_ids.nbytes
        id_content_bytes = len(self) * 50 * 2  # Estimated string content
        return (arr_bytes + id_arr_bytes + id_content_bytes) / (1024 * 1024)


def generate_candidates(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance_m: float = None,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
) -> CandidateBatch:
    """Generate candidate pairs using memory-efficient STRtree spatial query.

    Uses STRtree.query for efficient spatial matching without copying DataFrames.
    Heading and length ratio are computed only for matched pairs.

    Args:
        reference: Reference edges (Overture) GeoDataFrame
        target: Target edges (local data) GeoDataFrame
        buffer_distance_m: Search radius in meters (None = settings default)
        ref_id_column: Column name for reference IDs
        target_id_column: Column name for target IDs

    Returns:
        CandidateBatch with candidate pairs stored as numpy arrays

    Raises:
        ValueError: If CRS is None (buffer calculations require known CRS)
    """
    if buffer_distance_m is None:
        buffer_distance_m = settings.buffer_distance_m

    # Require CRS to be set - buffer calculations need known units
    if reference.crs is None:
        raise ValueError(
            "reference has no CRS set. Cannot compute accurate buffer distances. "
            "Call gdf.set_crs('EPSG:4326') if data is in WGS84 coordinates."
        )
    if target.crs is None:
        raise ValueError(
            "target has no CRS set. Cannot compute accurate buffer distances. "
            "Call gdf.set_crs('EPSG:4326') if data is in WGS84 coordinates."
        )

    logger.info(f"Generating candidates: {len(reference)} reference x {len(target)} target")
    logger.info(f"  buffer_distance_m: {buffer_distance_m}m")
    logger.info("  Note: Using memory-efficient STRtree query")

    # Check if data is in geographic CRS - need to handle buffer distance conversion
    is_geographic = False
    try:
        crs = CRS.from_user_input(target.crs)
        is_geographic = crs.is_geographic
    except (ValueError, TypeError):
        # CRS parsing failed - assume projected CRS (safer for buffer distance)
        # This can happen with malformed CRS strings or unsupported formats
        pass

    # For geographic CRS, convert buffer distance to approximate degrees
    # At equator: 1 degree ≈ 111km, so 50m ≈ 0.00045 degrees
    # This is approximate but sufficient for candidate generation (blocking)
    if is_geographic:
        # Get center latitude for better approximation
        bounds = target.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        # meters per degree longitude varies with latitude
        meters_per_degree = 111320 * np.cos(np.radians(center_lat))
        buffer_degrees = buffer_distance_m / meters_per_degree
        logger.info(
            f"  Geographic CRS: buffer={buffer_degrees:.6f}° (~{buffer_distance_m}m at lat={center_lat:.1f}°)"
        )
    else:
        buffer_degrees = buffer_distance_m  # Already in projected units (meters)

    # Build STRtree on reference geometries (no copying!)
    ref_geoms = reference.geometry.values
    ref_tree = STRtree(ref_geoms)

    # Buffer target geometries for query
    target_geoms = target.geometry.values
    target_buffered = shapely.buffer(target_geoms, buffer_degrees)

    # Query all matches at once - returns (target_idx, ref_idx) pairs
    # This is much more memory-efficient than sjoin
    target_idxs_arr, ref_idxs_arr = ref_tree.query(target_buffered, predicate="intersects")

    n_pairs = len(ref_idxs_arr)
    logger.info(f"  STRtree query found {n_pairs:,} candidate pairs")

    if n_pairs == 0:
        return CandidateBatch(
            ref_ids=np.array([], dtype=object),
            ref_idxs=np.array([], dtype=np.int32),
            target_ids=np.array([], dtype=object),
            target_idxs=np.array([], dtype=np.int32),
            distances=np.array([], dtype=np.float32),
            heading_diffs=np.array([], dtype=np.float32),
        )

    # Convert to int32 for memory efficiency
    ref_idxs = ref_idxs_arr.astype(np.int32)
    target_idxs = target_idxs_arr.astype(np.int32)

    # Extract IDs using numpy advanced indexing (no DataFrame operations)
    if ref_id_column in reference.columns:
        ref_id_arr = reference[ref_id_column].values
        ref_ids = ref_id_arr[ref_idxs]
    else:
        ref_ids = ref_idxs.astype(object)

    if target_id_column in target.columns:
        target_id_arr = target[target_id_column].values
        target_ids = target_id_arr[target_idxs]
    else:
        target_ids = target_idxs.astype(object)

    # Compute heading differences for matched pairs only
    # Pre-compute all headings first (vectorized)
    ref_headings = _compute_headings_vectorized(reference.geometry)
    target_headings = _compute_headings_vectorized(target.geometry)

    # Index into headings for matched pairs
    matched_ref_headings = ref_headings[ref_idxs]
    matched_target_headings = target_headings[target_idxs]
    heading_diffs = _angle_diff_vectorized(matched_target_headings, matched_ref_headings).astype(
        np.float32
    )

    # Compute centroid distances for matched pairs
    ref_centroids = shapely.centroid(ref_geoms[ref_idxs])
    target_centroids = shapely.centroid(target_geoms[target_idxs])
    distances = shapely.distance(ref_centroids, target_centroids).astype(np.float32)

    candidates = CandidateBatch(
        ref_ids=ref_ids,
        ref_idxs=ref_idxs,
        target_ids=target_ids,
        target_idxs=target_idxs,
        distances=distances,
        heading_diffs=heading_diffs,
    )

    logger.info(
        f"  Generated {len(candidates):,} candidates (~{candidates.memory_usage_mb():.0f} MB)"
    )
    return candidates


def generate_candidates_iter(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance_m: float = None,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
) -> Iterator[CandidatePair]:
    """Iterator version of generate_candidates for memory efficiency.

    Yields candidate pairs one at a time instead of building full list.
    Only uses spatial proximity for filtering.
    """
    if buffer_distance_m is None:
        buffer_distance_m = settings.buffer_distance_m

    # Build spatial index on reference
    ref_tree = STRtree(reference.geometry.values)

    # Pre-compute headings for feature computation (not filtering)
    ref_headings = reference.geometry.apply(_compute_overall_heading)

    for target_idx in range(len(target)):
        target_row = target.iloc[target_idx]
        target_geom = target_row.geometry
        target_heading = _compute_overall_heading(target_geom)

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
            ref_id = ref_row[ref_id_column] if ref_id_column in ref_row.index else ref_idx

            # Compute heading for ML features (not filtering)
            heading_diff = _angle_diff(target_heading, ref_heading)
            distance_estimate = target_geom.centroid.distance(ref_row.geometry.centroid)

            yield CandidatePair(
                ref_id=ref_id,
                ref_idx=ref_idx,
                target_id=target_id,
                target_idx=target_idx,
                distance_estimate=distance_estimate,
                heading_diff=heading_diff,
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
    ref_name_column: str = NAMES_COLUMN,
    target_name_column: str = NAMES_COLUMN,
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
        ref_name_raw = reference.iloc[cand.ref_idx].get(ref_name_column)
        target_name_raw = target.iloc[cand.target_idx].get(target_name_column)

        # Extract primary string from names struct if needed
        ref_name = ref_name_raw.get("primary") if isinstance(ref_name_raw, dict) else ref_name_raw
        target_name = (
            target_name_raw.get("primary") if isinstance(target_name_raw, dict) else target_name_raw
        )

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
