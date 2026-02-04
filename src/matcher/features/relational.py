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
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import shapely
from shapely import LineString, STRtree, line_interpolate_point
from shapely import distance as shapely_distance

from ..config import (
    DEFAULT_EXPECTED_HALF_WIDTH_M,
    EXPECTED_HALF_WIDTH_BY_CLASS_M,
    PARALLEL_SIBLING_MAX_OFFSET_M,
    PARALLEL_SIBLING_MIN_OFFSET_M,
)
from ._jit_helpers import (
    compute_endpoint_proximity_numba,
    compute_local_parallel_alignment_numba,
    compute_parallel_alignment_numba,
)


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

    The search is performed per-pair on the aligned subline (not the full
    geometry), allowing accurate sibling detection for partial alignments.
    """

    spatial_index: STRtree
    """Spatial index over all segment geometries (projected to meters)."""

    segment_data: list[tuple[str, str | None, str | None]]
    """List of (id, name, class) tuples, parallel to spatial_index geometries."""


def build_sibling_search_context(
    geometries: list[LineString],
    segment_ids: list[str],
    names: list,
    classes: list[str | None],
) -> SiblingSearchContext:
    """Build a SiblingSearchContext for parallel sibling detection.

    Args:
        geometries: List of segment geometries (projected to meters)
        segment_ids: List of segment IDs
        names: List of segment names (may contain None, strings, dicts, or lists)
        classes: List of road classes (may contain None)

    Returns:
        SiblingSearchContext for use with find_parallel_sibling
    """
    from .semantic import _extract_name_string

    spatial_index = STRtree(geometries)
    # Normalize names to plain strings using existing extraction logic
    normalized_names = [_extract_name_string(n) for n in names]
    segment_data = list(zip(segment_ids, normalized_names, classes))
    return SiblingSearchContext(spatial_index=spatial_index, segment_data=segment_data)


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

    mean_offset = float(np.mean(offsets))

    # IQR (interquartile range) - robust to outliers
    p25, p75 = np.percentile(offsets, [25, 75])
    offset_iqr = float(p75 - p25)

    # P95 - captures worst-case while ignoring extreme outliers
    offset_p95 = float(np.percentile(offsets, 95))

    if return_percentile is not None:
        offset_pN = float(np.percentile(offsets, return_percentile))
        return mean_offset, offset_iqr, offset_p95, offset_pN

    return mean_offset, offset_iqr, offset_p95


def compute_perpendicular_offset_batch(
    target_geoms: np.ndarray,
    anchor_geoms: np.ndarray,
    sample_interval: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch perpendicular offset for multiple pairs.

    Concatenates sample points across all pairs into single vectorized
    line_interpolate_point and distance calls, reducing Python dispatch
    overhead from O(N) to O(1).

    Args:
        target_geoms: Array of target geometries (shape N).
        anchor_geoms: Array of anchor geometries (shape N).
        sample_interval: Distance between sample points (meters).

    Returns:
        Tuple of (mean_offsets, iqr_offsets, p95_offsets), each shape (N,).
        Invalid pairs (empty/None geometries) get inf values.
    """
    N = len(target_geoms)
    mean_offsets = np.full(N, float("inf"))
    iqr_offsets = np.full(N, float("inf"))
    p95_offsets = np.full(N, float("inf"))

    if N == 0:
        return mean_offsets, iqr_offsets, p95_offsets

    # Determine valid pairs first (before calling Shapely ufuncs that can't handle None)
    valid_mask = np.array(
        [
            t is not None and a is not None and not shapely.is_empty(t) and not shapely.is_empty(a)
            for t, a in zip(target_geoms, anchor_geoms)
        ]
    )

    if not valid_mask.any():
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

        mean_offsets[vi] = float(np.mean(offsets))
        p25, p75 = np.percentile(offsets, [25, 75])
        iqr_offsets[vi] = float(p75 - p25)
        p95_offsets[vi] = float(np.percentile(offsets, 95))

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
        mean_alignment, parallel_fraction = compute_local_parallel_alignment_numba(
            coords_a, coords_b
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
    tolerance_m: float = 5.0,
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


def names_compatible(name_a: str | None, name_b: str | None) -> bool:
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
    if not name_a or not name_b:
        return None  # type: ignore[return-value]

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
    if not class_a or not class_b:
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
    if not road_class:
        return DEFAULT_EXPECTED_HALF_WIDTH_M

    return EXPECTED_HALF_WIDTH_BY_CLASS_M.get(road_class.lower(), DEFAULT_EXPECTED_HALF_WIDTH_M)


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
) -> ParallelSiblingResult:
    """Find parallel sibling segment (other half of split highway).

    Detects when a segment has a nearby parallel "twin" with same name/class,
    indicating it's part of a split carriageway representation.

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

    Returns:
        ParallelSiblingResult with has_sibling, sibling_distance, and parallel_fraction.
    """
    if segment.is_empty:
        return ParallelSiblingResult(False, float("inf"), 0.0)

    # Query spatial index with buffer
    buffer_geom = segment.buffer(max_offset)
    candidate_indices = spatial_index.query(buffer_geom)

    # Get segment coords once for efficiency
    segment_coords = np.array(segment.coords)

    # Track best parallel_fraction across all candidates
    best_parallel_fraction = 0.0
    best_offset = float("inf")
    found_sibling = False

    for candidate_idx in candidate_indices:
        # O(1) lookup using index directly - segment_data is parallel to spatial_index
        candidate_id, candidate_name, candidate_class = segment_data[candidate_idx]

        if candidate_id == segment_id:
            continue

        # Get the candidate geometry from the tree
        candidate_geom = spatial_index.geometries[candidate_idx]

        if candidate_geom is None or candidate_geom.is_empty:
            continue

        # 1. Check parallel alignment using LOCAL alignment (handles curves and partial parallelism)
        candidate_coords = np.array(candidate_geom.coords)
        alignment, parallel_fraction = compute_parallel_alignment(
            segment,
            candidate_geom,
            coords_a=segment_coords,
            coords_b=candidate_coords,
            return_fraction=True,
            use_local_alignment=True,
        )

        # Require both mean alignment AND minimum parallel fraction
        if alignment < min_alignment or parallel_fraction < min_parallel_fraction:
            continue

        # 2. Check lateral offset (must be in dual-carriageway range)
        # Use p10/p25 for diverging cases - captures the parallel portion's offset
        offset_result = compute_perpendicular_offset(candidate_geom, segment, return_percentile=25)
        # Use p25 as the sibling distance - represents the parallel portion
        # Mean would be inflated by diverging sections
        _, _, _, offset_p25 = offset_result
        if not (min_offset <= offset_p25 <= max_offset):
            continue

        # 3. Check name/class compatibility (same road)
        # Need at least one of: matching names OR compatible classes
        # If names positively match, that overrides class differences
        name_match = names_compatible(segment_name, candidate_name)
        class_match = classes_compatible(segment_class, candidate_class)

        # If names explicitly conflict, skip
        if name_match is False:
            continue

        # If names positively match, accept regardless of class difference
        # (same road can be classified differently in different datasets)
        if name_match is True:
            # Track best parallel_fraction but keep searching for better
            if parallel_fraction > best_parallel_fraction:
                best_parallel_fraction = parallel_fraction
                best_offset = offset_p25
                found_sibling = True
            continue  # Keep looking for better match

        # Names are inconclusive (None) - fall back to class check
        # If classes explicitly conflict, skip
        if not class_match:
            continue

        # Need at least one positive signal (compatible classes when names inconclusive)
        # If names are inconclusive (None) AND classes are missing, skip
        if name_match is None and (not segment_class or not candidate_class):
            continue

        # Found a valid sibling - track if it's the best
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
