"""Relational feature extraction for spatial context.

This module computes features based on spatial relationships between segments,
particularly useful for matching parallel infrastructure (sidewalks, bike lanes)
to their anchor roads, and for inferring connectivity from endpoint proximity.

These features work without requiring explicit topology in the target data,
making them suitable for raw "spaghetti" line datasets.

CRS Requirements:
-----------------
**IMPORTANT**: All geometries passed to functions in this module MUST be in a
projected CRS (e.g., UTM) where units are meters. Distance calculations use
Euclidean geometry and will produce incorrect results if geometries are in a
geographic CRS (lat/lon degrees).

Projection should happen early in the pipeline (see runner.py). Since bare
Shapely geometries don't carry CRS information, validation must occur at the
caller level (e.g., in ml.py or rules.py).

Key Features:
------------
1. **Perpendicular offset features**: For matching parallel infrastructure
   - perpendicular_offset: Mean perpendicular distance to anchor line
   - offset_consistency: Variance in offset (low = consistent parallel)
   - parallel_alignment: How parallel the segments are (0-1)

2. **Endpoint connectivity**: Inferred from proximity
   - endpoint_proximity: Distance to nearest endpoint of other segments
   - shared_endpoint_count: Segments with endpoints within tolerance

3. **Context propagation**: Agreement with neighboring matches
   - neighbor_agreement: Score based on nearby match confidence
"""

import re
import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import shapely
from loguru import logger
from shapely import LineString, STRtree, line_interpolate_point
from shapely import distance as shapely_distance

from ..config import (
    DEFAULT_EXPECTED_HALF_WIDTH_M,
    DEFAULT_SNAP_TOLERANCE_M,
    EXPECTED_HALF_WIDTH_BY_CLASS_M,
    PARALLEL_SIBLING_MAX_OFFSET_M,
    PARALLEL_SIBLING_MIN_OFFSET_M,
    PARALLEL_SIBLING_UNNAMED_MIN_LENGTH_RATIO,
    PARALLEL_SIBLING_UNNAMED_MIN_PARALLEL_FRACTION,
)
from ._exact_stats import percentile_sorted
from ._jit_helpers import (
    compute_endpoint_proximity_numba,
    compute_local_parallel_alignment_numba,
    compute_parallel_alignment_numba,
)


def _offset_stats(
    offsets: np.ndarray,
    return_percentile: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute (mean, iqr, p95, pN) offset statistics for one pair.

    Replaces three ``np.percentile`` calls (~50 µs of pure-Python machinery
    each) with one ``np.sort`` plus exact scalar interpolation
    (``percentile_sorted``, bitwise-equal to ``np.percentile``'s linear
    method). ``np.mean`` is kept as-is: its pairwise summation is not
    trivially reproducible, and it is already cheap.

    pN is ``inf`` when ``return_percentile`` is None.
    """
    mean_offset = float(np.mean(offsets))
    sorted_offsets = np.sort(offsets)
    if np.isnan(sorted_offsets[-1]):
        # NaNs present (sort places them last): preserve np.percentile's
        # NaN propagation exactly rather than interpolating around them.
        p25, p75 = np.percentile(offsets, [25, 75])
        offset_iqr = float(p75 - p25)
        offset_p95 = float(np.percentile(offsets, 95))
        offset_pn = (
            float(np.percentile(offsets, return_percentile))
            if return_percentile is not None
            else float("inf")
        )
        return mean_offset, offset_iqr, offset_p95, offset_pn

    p25 = percentile_sorted(sorted_offsets, 25.0)
    p75 = percentile_sorted(sorted_offsets, 75.0)
    offset_iqr = float(p75 - p25)
    offset_p95 = percentile_sorted(sorted_offsets, 95.0)
    offset_pn = (
        percentile_sorted(sorted_offsets, float(return_percentile))
        if return_percentile is not None
        else float("inf")
    )
    return mean_offset, offset_iqr, offset_p95, offset_pn


class ParallelSiblingResult(NamedTuple):
    """Result of parallel sibling detection.

    Attributes:
        has_sibling: True if a parallel sibling was found
        sibling_distance: Lateral offset to sibling in meters (inf if no sibling)
        parallel_fraction: Fraction of segment with nearby parallel sibling (0.0 if no sibling)
    """

    has_sibling: bool
    sibling_distance: float
    parallel_fraction: float


@dataclass
class SiblingSearchContext:
    """Context for per-pair parallel sibling detection.

    Holds the spatial index and segment metadata needed to search for
    parallel siblings within a dataset. This is built once per dataset
    and reused for all pairs involving that dataset.

    The search is performed per-pair on the aligned portion (not the full
    geometry), allowing accurate sibling detection for partial alignments.
    """

    spatial_index: STRtree
    """Spatial index over all segment geometries (projected to meters)."""

    segment_data: list[tuple[str, str | None, str | None]]
    """List of (id, name, class) tuples, parallel to spatial_index geometries."""

    segment_coords: list[np.ndarray] | None = None
    """Per-segment coordinate arrays (same values as np.array(geom.coords)),
    parallel to spatial_index geometries. Precomputed once per dataset so the
    per-pair sibling search does not re-extract coordinates for every
    spatial-query candidate. None when unavailable (fallback: extract per call)."""

    segment_valid: np.ndarray | None = None
    """Boolean array: geometry is present (not None) and not empty."""

    segment_lengths: np.ndarray | None = None
    """Float array of geometry lengths (0.0 for missing geometries)."""

    segment_tiers: np.ndarray | None = None
    """Traffic tier per segment encoded as small ints (see _TIER_CODES);
    used by the crossing-angle fast path in compute.py."""

    segment_headings: np.ndarray | None = None
    """Gross heading (degrees, 0-360) per segment, computed with the exact
    formula compute_crossing_angle_features uses (arctan2 over first/last
    points). NaN for missing/empty geometries."""


# Integer codes for traffic tiers (segment_tiers array). Code 0 = unknown
# (get_traffic_tier returned None), matching the "skip neighbor" behavior.
_TIER_CODES: dict[str | None, int] = {
    None: 0,
    "vehicle": 1,
    "bicycle": 2,
    "pedestrian": 3,
    "neutral": 4,
}


def _precompute_context_arrays(
    geometries: list[LineString],
    classes: list[str | None],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-segment arrays for the sibling/crossing fast paths.

    All values are bitwise-identical to what the per-pair code previously
    computed on the fly:
    - coords via shapely.get_coordinates (== np.array(geom.coords))
    - lengths via shapely.length (== geom.length)
    - headings via get_point/get_x/get_y + arctan2, the exact formula in
      compute_crossing_angle_features
    """
    import shapely as shapely_mod

    from .semantic import get_traffic_tier

    geom_arr = np.empty(len(geometries), dtype=object)
    geom_arr[:] = geometries

    valid = ~(shapely_mod.is_missing(geom_arr) | shapely_mod.is_empty(geom_arr))
    lengths = np.where(valid, shapely_mod.length(geom_arr), 0.0)

    # Per-segment coordinate arrays. get_coordinates over the full array with
    # return_index=False + split by counts gives contiguous row-slice views,
    # value-identical to np.array(geom.coords) per geometry. include_z matches
    # the dimensionality np.array(geom.coords) would produce; mixed 2D/3D
    # datasets fall back to per-geometry extraction to preserve per-geom shape.
    # Only Point/LineString/LinearRing expose .coords; np.array(geom.coords) on
    # multi-part geometries and polygons raises NotImplementedError. Store None
    # for those so the per-pair fallback re-runs np.array(geom.coords) and
    # raises the IDENTICAL exception the pre-optimization code raised (the
    # affected pair gets error features either way — behavior preserved).
    type_ids = shapely_mod.get_type_id(geom_arr)
    has_coords_seq = (type_ids >= 0) & (type_ids <= 2)

    has_z = shapely_mod.has_z(geom_arr)
    has_z = np.where(valid, has_z, False)
    if has_z.any() and not has_z[valid].all():
        coords_list: list[np.ndarray | None] = [
            np.array(g.coords) if (v and hc) else None
            for g, v, hc in zip(geometries, valid, has_coords_seq)
        ]
    else:
        include_z = bool(has_z.any())
        all_coords = shapely_mod.get_coordinates(geom_arr, include_z=include_z)
        counts = shapely_mod.get_num_coordinates(geom_arr)
        offsets = np.zeros(len(geometries) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        coords_list = [
            all_coords[offsets[i] : offsets[i + 1]] if has_coords_seq[i] else None
            for i in range(len(geometries))
        ]

    # Gross headings (exact formula from compute_crossing_angle_features)
    headings = np.full(len(geometries), np.nan)
    if valid.any():
        vgeoms = geom_arr[valid]
        starts = shapely_mod.get_point(vgeoms, 0)
        ends = shapely_mod.get_point(vgeoms, -1)
        dx = shapely_mod.get_x(ends) - shapely_mod.get_x(starts)
        dy = shapely_mod.get_y(ends) - shapely_mod.get_y(starts)
        headings[valid] = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    tiers = np.fromiter(
        (_TIER_CODES[get_traffic_tier(cls)] for cls in classes),
        dtype=np.int8,
        count=len(classes),
    )

    return coords_list, valid, lengths, tiers, headings


def build_sibling_search_context(
    geometries: list[LineString],
    segment_ids: list[str],
    names: list,
    classes: list[str | None],
) -> SiblingSearchContext:
    """Build a SiblingSearchContext for parallel sibling detection.

    Args:
        geometries: List of full segment geometries (projected to meters).
            Must be full/original geometries, not aligned portions, since
            the spatial index needs complete segments for sibling search.
        segment_ids: List of segment IDs
        names: List of segment names (may contain None, strings, dicts, or lists)
        classes: List of road classes (may contain None)

    Returns:
        SiblingSearchContext for use with find_parallel_sibling
    """
    n_segments = len(geometries)
    logger.info(f"Building sibling search context for {n_segments} segments...")
    t0 = time.perf_counter()

    spatial_index = STRtree(geometries)
    t_tree = time.perf_counter() - t0
    logger.debug(f"[TIMING] STRtree construction: {t_tree:.2f}s ({n_segments} geometries)")

    # Normalize names to plain strings — extract primary from any dicts
    t1 = time.perf_counter()
    normalized_names = [n.get("primary") if isinstance(n, dict) else n for n in names]
    t_names = time.perf_counter() - t1
    logger.debug(f"[TIMING] Name normalization: {t_names:.2f}s ({n_segments} names)")

    # Precompute per-segment arrays (coords, lengths, tiers, headings) once so
    # the per-pair sibling search and crossing-angle features avoid per-call
    # Shapely property access and coordinate extraction.
    t2 = time.perf_counter()
    coords_list, valid, lengths, tiers, headings = _precompute_context_arrays(geometries, classes)
    t_pre = time.perf_counter() - t2
    logger.debug(f"[TIMING] Context array precompute: {t_pre:.2f}s ({n_segments} segments)")

    segment_data = list(zip(segment_ids, normalized_names, classes))
    return SiblingSearchContext(
        spatial_index=spatial_index,
        segment_data=segment_data,
        segment_coords=coords_list,
        segment_valid=valid,
        segment_lengths=lengths,
        segment_tiers=tiers,
        segment_headings=headings,
    )


class RelationalFeatures(NamedTuple):
    """Relational features for a candidate pair.

    These features capture spatial relationships that help match parallel
    infrastructure (sidewalks, bike lanes) to their anchor roads.
    """

    perpendicular_offset: float
    """Mean perpendicular distance from target to anchor line (meters).
    Low values indicate the segment runs parallel and close to the anchor."""

    offset_iqr: float
    """Interquartile range (p75 - p25) of perpendicular distances (meters).
    Robust to outliers. Low values indicate consistent parallel offset."""

    offset_p95: float
    """95th percentile of perpendicular distances (meters).
    Captures worst-case offset while ignoring extreme outliers."""

    parallel_alignment: float
    """How parallel the segment is to anchor (0-1).
    Based on heading difference: 1.0 = parallel, 0.0 = perpendicular."""


def compute_perpendicular_offset(
    target_geom: LineString,
    anchor_geom: LineString,
    sample_interval: float = 5.0,
    return_percentile: float | None = None,
) -> tuple[float, float, float] | tuple[float, float, float, float]:
    """Compute perpendicular offset from target to anchor line.

    Samples points along the target line and measures perpendicular
    distance to the anchor. Returns mean offset, IQR (interquartile range),
    and 95th percentile for robust outlier handling.

    Uses Shapely's vectorized functions for efficient batch computation.

    IMPORTANT: Geometries MUST be in a projected CRS (meters).
    Results will be incorrect if geometries are in geographic CRS (degrees).

    Args:
        target_geom: Target geometry (sidewalk, bike lane) in projected CRS
        anchor_geom: Anchor geometry (road centerline) in projected CRS
        sample_interval: Distance between sample points (meters)
        return_percentile: If provided, also return this percentile (e.g., 10 for p10)
            which is useful for diverging carriageways where lower percentiles
            capture the offset of the parallel portion rather than the diverging ends.

    Returns:
        If return_percentile is None:
            Tuple of (mean_offset, offset_iqr, offset_p95)
        If return_percentile is provided:
            Tuple of (mean_offset, offset_iqr, offset_p95, offset_pN)

        - mean_offset: Mean perpendicular distance (meters)
        - offset_iqr: Interquartile range (p75 - p25), robust to outliers
        - offset_p95: 95th percentile of offsets
        - offset_pN: Nth percentile of offsets (only if return_percentile provided)

    Example:
        A sidewalk 3m from a road with consistent offset:
        >>> offset, iqr, p95 = compute_perpendicular_offset(sidewalk, road)
        >>> print(f"Offset: {offset:.1f}m, IQR: {iqr:.2f}m, P95: {p95:.2f}m")
        Offset: 3.0m, IQR: 0.20m, P95: 3.5m
    """
    if target_geom.is_empty or anchor_geom.is_empty:
        if return_percentile is not None:
            return float("inf"), float("inf"), float("inf"), float("inf")
        return float("inf"), float("inf"), float("inf")

    # Sample points along target using vectorized interpolation
    n_samples = max(3, int(target_geom.length / sample_interval))
    distances_along = np.linspace(0, target_geom.length, n_samples)

    # Vectorized point creation using absolute distances (matches original behavior)
    points = line_interpolate_point(target_geom, distances_along, normalized=False)

    # Vectorized distance computation - all points to anchor line
    offsets = shapely_distance(points, anchor_geom)

    mean_offset, offset_iqr, offset_p95, offset_pN = _offset_stats(offsets, return_percentile)

    if return_percentile is not None:
        return mean_offset, offset_iqr, offset_p95, offset_pN

    return mean_offset, offset_iqr, offset_p95


def compute_perpendicular_offset_batch(
    target_geoms: np.ndarray,
    anchor_geoms: np.ndarray,
    sample_interval: float = 5.0,
    return_percentile: float | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Batch perpendicular offset for multiple pairs.

    Concatenates sample points across all pairs into single vectorized
    line_interpolate_point and distance calls, reducing Python dispatch
    overhead from O(N) to O(1).

    Args:
        target_geoms: Array of target geometries (shape N).
        anchor_geoms: Array of anchor geometries (shape N).
        sample_interval: Distance between sample points (meters).
        return_percentile: If provided, also return this percentile (e.g., 25
            for p25) as a 4th array. Mirrors compute_perpendicular_offset's
            single-pair option, which sibling detection relies on to capture
            the parallel portion's offset for diverging carriageways.

    Returns:
        If return_percentile is None:
            Tuple of (mean_offsets, iqr_offsets, p95_offsets), each shape (N,).
        If return_percentile is provided:
            Tuple of (mean_offsets, iqr_offsets, p95_offsets, pN_offsets).
        Invalid pairs (empty/None geometries) get inf values.
    """
    N = len(target_geoms)
    mean_offsets = np.full(N, float("inf"))
    iqr_offsets = np.full(N, float("inf"))
    p95_offsets = np.full(N, float("inf"))
    pN_offsets = np.full(N, float("inf"))

    if N == 0:
        if return_percentile is not None:
            return mean_offsets, iqr_offsets, p95_offsets, pN_offsets
        return mean_offsets, iqr_offsets, p95_offsets

    # Determine valid pairs first. Vectorized: shapely.is_missing handles None
    # entries and shapely.is_empty returns False for None, so the combined mask
    # is identical to the previous per-element Python loop
    # (t is not None and a is not None and not is_empty(t) and not is_empty(a)).
    valid_mask = ~(
        shapely.is_missing(target_geoms)
        | shapely.is_missing(anchor_geoms)
        | shapely.is_empty(target_geoms)
        | shapely.is_empty(anchor_geoms)
    )

    if not valid_mask.any():
        if return_percentile is not None:
            return mean_offsets, iqr_offsets, p95_offsets, pN_offsets
        return mean_offsets, iqr_offsets, p95_offsets

    valid_indices = np.where(valid_mask)[0]
    valid_lengths = shapely.length(target_geoms[valid_indices])
    n_samples_per = np.maximum(3, (valid_lengths / sample_interval).astype(int))

    # Boundaries for splitting results back per pair
    boundaries = np.empty(len(valid_indices) + 1, dtype=int)
    boundaries[0] = 0
    np.cumsum(n_samples_per, out=boundaries[1:])
    total_points = int(boundaries[-1])

    # Build repeated geometry arrays using np.repeat (avoids Python loop)
    valid_targets = target_geoms[valid_indices]
    valid_anchors = anchor_geoms[valid_indices]
    repeated_targets = np.repeat(valid_targets, n_samples_per)
    repeated_anchors = np.repeat(valid_anchors, n_samples_per)

    # Build distance array — each pair needs linspace(0, length, n_samples).
    # Construct normalized fractions [0..1] per pair, then scale by length.
    all_distances = np.empty(total_points)
    for j in range(len(valid_indices)):
        start = boundaries[j]
        end = boundaries[j + 1]
        ns = n_samples_per[j]
        all_distances[start:end] = np.linspace(0, valid_lengths[j], ns)

    # Two vectorized Shapely calls for all points across all pairs
    all_points = line_interpolate_point(repeated_targets, all_distances, normalized=False)
    all_dists = shapely_distance(all_points, repeated_anchors)

    # Split by pair and compute per-pair statistics
    for j, vi in enumerate(valid_indices):
        start = boundaries[j]
        end = boundaries[j + 1]
        offsets = all_dists[start:end]

        mean_offsets[vi], iqr_offsets[vi], p95_offsets[vi], pn = _offset_stats(
            offsets, return_percentile
        )
        if return_percentile is not None:
            pN_offsets[vi] = pn

    if return_percentile is not None:
        return mean_offsets, iqr_offsets, p95_offsets, pN_offsets
    return mean_offsets, iqr_offsets, p95_offsets


def compute_parallel_alignment(
    line_a: LineString,
    line_b: LineString,
    *,
    coords_a: np.ndarray | None = None,
    coords_b: np.ndarray | None = None,
    return_fraction: bool = False,
    use_local_alignment: bool = False,
) -> float | tuple[float, float]:
    """Compute how parallel two lines are (0-1).

    By default, uses overall heading (first to last point) for backward compatibility.
    When use_local_alignment=True, uses segment-wise local alignment that handles
    curved segments and partial parallelism better.

    Args:
        line_a: First line geometry
        line_b: Second line geometry
        coords_a: Pre-extracted coordinates for line_a (optional)
        coords_b: Pre-extracted coordinates for line_b (optional)
        return_fraction: If True, also return the parallel fraction (requires use_local_alignment)
        use_local_alignment: If True, use local alignment sampling instead of overall heading

    Returns:
        If return_fraction=False: Alignment score (0-1) where 1 = parallel
        If return_fraction=True: Tuple of (mean_alignment, parallel_fraction)
    """
    if line_a.is_empty or line_b.is_empty:
        if return_fraction:
            return 0.0, 0.0
        return 0.0

    if coords_a is None:
        coords_a = np.array(line_a.coords)
    if coords_b is None:
        coords_b = np.array(line_b.coords)

    if use_local_alignment or return_fraction:
        # Defaults passed explicitly: omitted-default numba calls can miss the
        # fast C dispatch path and pay ~80 µs of _compile_for_args per call
        # (numba 0.63). Values match the function's declared defaults.
        mean_alignment, parallel_fraction = compute_local_parallel_alignment_numba(
            coords_a, coords_b, 16, 0.7
        )
        if return_fraction:
            return mean_alignment, parallel_fraction
        return mean_alignment

    return compute_parallel_alignment_numba(coords_a, coords_b)


def _compute_heading(start: np.ndarray, end: np.ndarray) -> float:
    """Compute heading in degrees from start to end point (0-360)."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    heading = np.degrees(np.arctan2(dy, dx))
    return (heading + 360) % 360


def compute_relational_features(
    target_geom: LineString,
    anchor_geom: LineString,
    sample_interval: float = 5.0,
) -> RelationalFeatures:
    """Compute all relational features for a target/anchor pair.

    IMPORTANT: Geometries MUST be in a projected CRS (meters).
    Results will be incorrect if geometries are in geographic CRS (degrees).

    Args:
        target_geom: Target geometry (sidewalk, bike lane) in projected CRS
        anchor_geom: Anchor geometry (road centerline) in projected CRS
        sample_interval: Distance between sample points (meters)

    Returns:
        RelationalFeatures named tuple with distances in meters
    """
    # Perpendicular offset (now returns mean, iqr, p95)
    offset, offset_iqr, offset_p95 = compute_perpendicular_offset(
        target_geom, anchor_geom, sample_interval
    )

    # Parallel alignment
    alignment = compute_parallel_alignment(target_geom, anchor_geom)

    return RelationalFeatures(
        perpendicular_offset=offset,
        offset_iqr=offset_iqr,
        offset_p95=offset_p95,
        parallel_alignment=alignment,
    )


def compute_endpoint_proximity(
    target_geom: LineString,
    endpoint_coords: np.ndarray,
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    *,
    target_coords: np.ndarray | None = None,
) -> tuple[float, float, int]:
    """Compute endpoint proximity features.

    Finds distance from target's endpoints to the nearest endpoints
    of other segments (provided as an array of coordinates).

    Args:
        target_geom: Target geometry
        endpoint_coords: Array of shape (N, 2) with other endpoint coordinates
        tolerance_m: Distance threshold for counting "shared" endpoints (meters)
        target_coords: Pre-extracted coordinates for target (optional)

    Returns:
        Tuple of (start_proximity, end_proximity, shared_count) where:
        - start_proximity: Distance to nearest endpoint from start
        - end_proximity: Distance to nearest endpoint from end
        - shared_count: Number of endpoints within tolerance
    """
    if target_geom.is_empty or len(endpoint_coords) == 0:
        return float("inf"), float("inf"), 0

    if target_coords is None:
        target_coords = np.array(target_geom.coords)

    start = target_coords[0]
    end = target_coords[-1]

    result = compute_endpoint_proximity_numba(start, end, endpoint_coords, tolerance_m)
    return float(result[0]), float(result[1]), int(result[2])


def compute_neighbor_agreement(
    candidate_confidence: float,
    neighbor_confidences: list[float],
    neighbor_match_same_ref: list[bool],
    decay_factor: float = 0.5,
) -> float:
    """Compute agreement score with neighboring matched segments.

    If neighbors are confidently matched to the same reference segment (or
    connected reference segments), that provides evidence for this match.
    If neighbors are matched to different references, that's evidence against.

    Args:
        candidate_confidence: Initial confidence score for this candidate
        neighbor_confidences: Confidence scores of neighboring candidates
        neighbor_match_same_ref: Whether each neighbor matches the same ref family
        decay_factor: Weight decay for neighbor influence (0-1)

    Returns:
        Agreement-adjusted confidence score (0-1)

    Example:
        If neighbors are confidently matched to the same reference:
        >>> score = compute_neighbor_agreement(0.6, [0.9, 0.85], [True, True])
        >>> print(f"Adjusted score: {score:.2f}")
        Adjusted score: 0.72  # Boosted due to neighborhood agreement
    """
    if not neighbor_confidences:
        return candidate_confidence

    neighbor_confidences = np.array(neighbor_confidences)
    neighbor_match_same_ref = np.array(neighbor_match_same_ref)

    # Weight neighbors by their confidence
    weights = neighbor_confidences * decay_factor

    # Agreement: neighbors matching same ref boost, different ref penalize
    agreement_signals = np.where(neighbor_match_same_ref, 1.0, -1.0)
    weighted_agreement = np.sum(weights * agreement_signals) / (np.sum(weights) + 1e-10)

    # Adjust candidate confidence based on agreement
    # Positive agreement boosts, negative agreement penalizes
    adjustment = weighted_agreement * 0.2  # Max +/- 20% adjustment
    adjusted_confidence = np.clip(candidate_confidence + adjustment, 0.0, 1.0)

    return float(adjusted_confidence)


# ============================================================================
# Parallel Sibling Detection for Split Carriageway Recognition
# ============================================================================


# Regex pattern to extract route numbers from road names
# Matches patterns like: I-90, US-101, US 1, Route 66, State Highway 1, SR-12, A1, M25, Interstate 90
_ROUTE_NUMBER_PATTERN = re.compile(
    r"""
    (?:^|[^\w])       # Start of string or non-word character
    (?:
        (?:Interstate|I|US|SR|M|A)  # Common highway prefixes
        [-\s]?        # Optional separator
        (\d+)         # Route number (captured - group 1)
    |
        (?:Route|Highway|Hwy|State\s+(?:Route|Highway))  # Route/Highway keyword
        \s*
        (\d+)         # Route number (captured - group 2)
    )
    (?:[^\d]|$)       # Non-digit or end of string
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_route_number(name: str | None) -> str | None:
    """Extract numeric route number from a road name.

    Examples:
        - "I-90" -> "90"
        - "US Highway 101" -> "101"
        - "Route 66" -> "66"
        - "Main Street" -> None

    Args:
        name: Road name string

    Returns:
        Route number as string, or None if no route number found
    """
    if not name:
        return None

    match = _ROUTE_NUMBER_PATTERN.search(name)
    if match:
        # Return whichever group matched
        return match.group(1) or match.group(2)
    return None


def names_compatible(name_a: str | None, name_b: str | None) -> bool | None:
    """Check if names are compatible for sibling matching.

    Philosophy: Require BOTH segments to have names for name-based matching.
    If either is unnamed, return None to indicate "no opinion" - caller must
    rely on class matching instead.

    Args:
        name_a: First road name (may be None)
        name_b: Second road name (may be None)

    Returns:
        True if names match, False if names conflict, None if inconclusive
        (one or both names missing)
    """
    # If either is unnamed, we can't use names for matching
    # Return None to signal "no opinion" - must rely on class
    if not name_a or not isinstance(name_a, str) or not name_b or not isinstance(name_b, str):
        return None

    # Both have names - require match
    name_a_norm = name_a.lower().strip()
    name_b_norm = name_b.lower().strip()
    if name_a_norm == name_b_norm:
        return True

    # Numeric route match (I-90 == Interstate 90 == I 90)
    route_a = extract_route_number(name_a)
    route_b = extract_route_number(name_b)
    return bool(route_a and route_b and route_a == route_b)


# Road class hierarchy for sibling detection
# Lower numbers = higher priority roads
_CLASS_HIERARCHY: dict[str, int] = {
    "motorway": 0,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "unclassified": 5,
    "residential": 6,
    "service": 7,
    "living_street": 8,
    "pedestrian": 9,
    "track": 10,
    "path": 11,
    "cycleway": 12,
}


def classes_compatible(class_a: str | None, class_b: str | None, max_tier_diff: int = 1) -> bool:
    """Check if road classes are compatible for sibling matching.

    Allows matching within a specified tier difference in the road hierarchy.

    Args:
        class_a: First road class
        class_b: Second road class
        max_tier_diff: Maximum allowed tier difference (default 1)

    Returns:
        True if classes are within max_tier_diff tiers of each other
    """
    # If either is None/unknown, be permissive
    if not class_a or not isinstance(class_a, str) or not class_b or not isinstance(class_b, str):
        return True

    tier_a = _CLASS_HIERARCHY.get(class_a.lower(), 5)  # Default to unclassified
    tier_b = _CLASS_HIERARCHY.get(class_b.lower(), 5)

    return abs(tier_a - tier_b) <= max_tier_diff


def get_expected_half_width(road_class: str | None) -> float:
    """Get expected half-width for a road class.

    Args:
        road_class: Road class (e.g., "motorway", "residential")

    Returns:
        Expected half-width in meters
    """
    if not road_class or not isinstance(road_class, str):
        return DEFAULT_EXPECTED_HALF_WIDTH_M

    return EXPECTED_HALF_WIDTH_BY_CLASS_M.get(road_class.lower(), DEFAULT_EXPECTED_HALF_WIDTH_M)


def _normalize_road_class(road_class: str | None) -> str | None:
    """Normalize a road class value for exact-match comparison.

    Treats missing data as missing: pandas/NumPy NaN (a float, and truthy!),
    None, non-strings, and empty/whitespace strings all normalize to None.
    This matters because `str(np.nan) == "nan"` would otherwise make two
    MISSING classes look like an exact class match.

    Args:
        road_class: Raw class value (may be str, None, NaN, or other types)

    Returns:
        Lowercased stripped class string, or None if missing/invalid.
    """
    if not isinstance(road_class, str):
        return None
    normalized = road_class.strip().lower()
    return normalized if normalized else None


def find_parallel_sibling(
    segment: LineString,
    segment_id: str,
    segment_name: str | None,
    segment_class: str | None,
    spatial_index: STRtree,
    segment_data: list[tuple[str, str | None, str | None]],
    min_offset: float = PARALLEL_SIBLING_MIN_OFFSET_M,
    max_offset: float = PARALLEL_SIBLING_MAX_OFFSET_M,
    min_alignment: float = 0.7,  # Lowered from 0.9 for local alignment
    min_parallel_fraction: float = 0.3,  # At least 30% of segment must be parallel
    unnamed_min_parallel_fraction: float = PARALLEL_SIBLING_UNNAMED_MIN_PARALLEL_FRACTION,
    unnamed_min_length_ratio: float = PARALLEL_SIBLING_UNNAMED_MIN_LENGTH_RATIO,
    context: "SiblingSearchContext | None" = None,
) -> ParallelSiblingResult:
    """Find parallel sibling segment (other half of split carriageway).

    A "sibling" means the *same road* split into two carriageways, not any
    parallel neighbor. Detecting a sibling therefore requires positive evidence
    that the twin is the same road:

    - If the two names are compatible, accept (same road can be classified
      differently across datasets, so class tolerance is fine here).
    - If the two names conflict, reject.
    - If name evidence is absent (either segment unnamed — very common for
      Overture service roads, links, and connectors), fall back to *positive
      geometric same-road evidence* rather than class tolerance alone:
        1. Exact road-class match (both present and identical).
        2. High parallel fraction (a real twin runs parallel along most of the
           shared stretch, not just briefly).
        3. Comparable extent (the twin spans roughly the same stretch, so the
           two lengths must be similar).
      Requiring all three prevents the detector from firing on any parallel
      same-class neighbor in the 5-30m offset band (e.g. adjacent grid streets,
      ramps, service roads), which made the feature anti-signal.

    Uses local alignment sampling to handle:
    - Curved segments (local heading differs from overall heading)
    - Partial parallelism (only a portion of segment is parallel)
    - Split carriageways that converge/diverge at endpoints

    Args:
        segment: Geometry of the segment to check
        segment_id: ID of the segment
        segment_name: Name of the segment (may be None)
        segment_class: Road class of the segment (may be None)
        spatial_index: STRtree built from all segment geometries
        segment_data: List of (id, name, class) tuples parallel to spatial_index geometries
        min_offset: Minimum lateral offset for sibling detection (meters)
        max_offset: Maximum lateral offset for sibling detection (meters)
        min_alignment: Minimum mean parallel alignment score (0-1), default 0.7
        min_parallel_fraction: Minimum fraction of segment that must be parallel (0-1), default 0.3
        unnamed_min_parallel_fraction: Minimum parallel fraction required on the
            unnamed path (positive same-road evidence), default 0.6
        unnamed_min_length_ratio: Minimum min(len)/max(len) required on the
            unnamed path (comparable extent), default 0.5
        context: Optional SiblingSearchContext with precomputed per-segment
            arrays (coords, validity, lengths). When provided, the per-candidate
            coordinate extraction and Shapely property access are replaced with
            array lookups — value-identical, just faster.

    Returns:
        ParallelSiblingResult with has_sibling, sibling_distance, and parallel_fraction.
    """
    if segment.is_empty:
        return ParallelSiblingResult(False, float("inf"), 0.0)

    # Query spatial index with buffer
    buffer_geom = segment.buffer(max_offset)
    candidate_indices = spatial_index.query(buffer_geom)

    # Get segment coords/length once for efficiency
    segment_coords = np.array(segment.coords)
    segment_length = segment.length

    # ---- Phase 1: cheap alignment + name pre-filter ----
    # Collect survivors that pass the parallel-alignment gate and are not name
    # conflicts. The (more expensive, now batched) perpendicular-offset check
    # and the same-road gates run only on survivors.
    survivor_geoms: list[LineString] = []
    # Each entry: (candidate_class, parallel_fraction, name_match)
    survivor_info: list[tuple[str | None, float, bool | None]] = []

    # Precomputed per-segment arrays (value-identical fast lookups)
    ctx_coords = context.segment_coords if context is not None else None
    ctx_valid = context.segment_valid if context is not None else None

    for candidate_idx in candidate_indices:
        # O(1) lookup using index directly - segment_data is parallel to spatial_index
        candidate_id, candidate_name, candidate_class = segment_data[candidate_idx]

        if candidate_id == segment_id:
            continue

        # Get the candidate geometry from the tree
        candidate_geom = spatial_index.geometries[candidate_idx]

        if ctx_valid is not None:
            if not ctx_valid[candidate_idx]:
                continue
        elif candidate_geom is None or candidate_geom.is_empty:
            continue

        # Name compatibility (cheap, no geometry). Reject conflicts immediately.
        name_match = names_compatible(segment_name, candidate_name)
        if name_match is False:
            continue

        # Check parallel alignment using LOCAL alignment (handles curves and partial parallelism)
        candidate_coords = ctx_coords[candidate_idx] if ctx_coords is not None else None
        if candidate_coords is None:
            # Not precomputed (no context, or geometry without a coordinate
            # sequence). np.array(geom.coords) raises NotImplementedError for
            # multi-part geometries — intentionally identical to the
            # pre-optimization behavior (the pair gets error features).
            candidate_coords = np.array(candidate_geom.coords)
        # Direct numba call (== compute_parallel_alignment with
        # use_local_alignment=True, return_fraction=True): both geometries are
        # known non-empty here, so the wrapper's is_empty checks (2 GEOS calls
        # per candidate) are redundant. Defaults passed explicitly for numba
        # fast dispatch.
        alignment, parallel_fraction = compute_local_parallel_alignment_numba(
            segment_coords, candidate_coords, 16, 0.7
        )

        # Require both mean alignment AND minimum parallel fraction
        if alignment < min_alignment or parallel_fraction < min_parallel_fraction:
            continue

        survivor_geoms.append(candidate_geom)
        survivor_info.append((candidate_class, parallel_fraction, name_match))

    if not survivor_geoms:
        return ParallelSiblingResult(False, float("inf"), 0.0)

    # ---- Phase 2: batched perpendicular offset for survivors ----
    # Use p25 for diverging cases - captures the parallel portion's offset
    # (mean would be inflated by diverging sections). Batching concatenates all
    # survivors into single vectorized Shapely calls instead of per-candidate.
    target_arr = np.empty(len(survivor_geoms), dtype=object)
    target_arr[:] = survivor_geoms
    anchor_arr = np.empty(len(survivor_geoms), dtype=object)
    anchor_arr[:] = [segment] * len(survivor_geoms)
    _, _, _, offsets_p25 = compute_perpendicular_offset_batch(
        target_arr, anchor_arr, return_percentile=25
    )

    # ---- Phase 3: offset band + same-road evidence gates ----
    best_parallel_fraction = 0.0
    best_offset = float("inf")
    found_sibling = False
    seg_class_norm = _normalize_road_class(segment_class)

    for i, (candidate_class, parallel_fraction, name_match) in enumerate(survivor_info):
        offset_p25 = float(offsets_p25[i])
        if not (min_offset <= offset_p25 <= max_offset):
            continue

        if name_match is True:
            # Names positively match -> same road; accept regardless of class.
            pass
        else:
            # Name evidence absent (None) -> require positive geometric
            # same-road evidence instead of class tolerance alone.
            # 1. Exact class match (both present and identical). NaN/None/empty
            #    classes are MISSING data, not evidence — reject. (NaN is a
            #    truthy float and str(nan)=="nan", so naive comparison would
            #    treat two missing classes as an exact match.)
            cand_class_norm = _normalize_road_class(candidate_class)
            if seg_class_norm is None or cand_class_norm is None:
                continue
            if seg_class_norm != cand_class_norm:
                continue
            # 2. High parallel fraction (parallel along most of the stretch).
            if parallel_fraction < unnamed_min_parallel_fraction:
                continue
            # 3. Comparable extent (twin spans roughly the same stretch).
            candidate_length = survivor_geoms[i].length
            longer = max(segment_length, candidate_length)
            if longer <= 0.0:
                continue
            length_ratio = min(segment_length, candidate_length) / longer
            if length_ratio < unnamed_min_length_ratio:
                continue

        # Valid sibling - track the best by parallel_fraction.
        if parallel_fraction > best_parallel_fraction:
            best_parallel_fraction = parallel_fraction
            best_offset = offset_p25
            found_sibling = True

    return ParallelSiblingResult(found_sibling, best_offset, best_parallel_fraction)


def precompute_parallel_siblings(
    geometries: list[LineString],
    segment_ids: list[str],
    names: list[str | None],
    classes: list[str | None],
    ids_to_compute: set[str] | None = None,
    spatial_index: STRtree | None = None,
) -> dict[str, ParallelSiblingResult]:
    """Pre-compute parallel sibling info for segments in a dataset.

    This is called once per dataset (ref and target) during Pass 1 of feature
    computation. Results are cached and reused for all candidate pairs.

    The spatial index includes ALL segments (for finding potential siblings),
    but sibling detection is only performed for segments in ids_to_compute.

    Args:
        geometries: List of segment geometries (projected to meters)
        segment_ids: List of segment IDs
        names: List of segment names (may contain None)
        classes: List of road classes (may contain None)
        ids_to_compute: Optional set of segment IDs to compute sibling info for.
            If None, computes for all segments. Use this to filter to only
            labeled segments for efficiency during backfill.
        spatial_index: Optional pre-built STRtree over geometries. If provided,
            skips building a new one (saves O(N log N) construction time).

    Returns:
        Dict mapping segment_id -> ParallelSiblingResult
    """
    # Use provided spatial index or build a new one
    if spatial_index is None:
        spatial_index = STRtree(geometries)

    # Build parallel list of (id, name, class) - matches spatial_index order
    segment_data: list[tuple[str, str | None, str | None]] = list(zip(segment_ids, names, classes))

    # Build lookup for O(1) access by ID
    id_to_idx = {seg_id: i for i, seg_id in enumerate(segment_ids)}

    # Compute sibling info - only for requested segments
    result: dict[str, ParallelSiblingResult] = {}

    # If filtered, only iterate through requested IDs (O(k) instead of O(N))
    ids_to_process = ids_to_compute if ids_to_compute is not None else segment_ids
    for seg_id in ids_to_process:
        idx = id_to_idx.get(seg_id)
        if idx is None:
            continue  # ID not in dataset

        geom = geometries[idx]
        name = segment_data[idx][1]
        cls = segment_data[idx][2]

        result[seg_id] = find_parallel_sibling(
            segment=geom,
            segment_id=seg_id,
            segment_name=name,
            segment_class=cls,
            spatial_index=spatial_index,
            segment_data=segment_data,
        )

    return result
