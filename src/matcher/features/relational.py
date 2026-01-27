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

2. **Side of street**: Left/right relative to road direction of travel
   - Uses cross product of direction vectors

3. **Endpoint connectivity**: Inferred from proximity
   - endpoint_proximity: Distance to nearest endpoint of other segments
   - shared_endpoint_count: Segments with endpoints within tolerance

4. **Context propagation**: Agreement with neighboring matches
   - neighbor_agreement: Score based on nearby match confidence
"""

from typing import NamedTuple

import numpy as np
import shapely
from shapely import LineString, Point, line_interpolate_point
from shapely import distance as shapely_distance

from ._jit_helpers import (
    compute_endpoint_proximity_numba,
    compute_parallel_alignment_numba,
    side_of_street_vote_numba,
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

    side_of_street: str
    """Which side of the anchor road: 'left', 'right', or 'unknown'.
    Determined by cross product of direction vectors."""

    side_confidence: float
    """Confidence in side determination (0-1).
    Low when segment crosses back and forth across anchor."""


def compute_perpendicular_offset(
    target_geom: LineString,
    anchor_geom: LineString,
    sample_interval: float = 5.0,
) -> tuple[float, float, float]:
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

    Returns:
        Tuple of (mean_offset, offset_iqr, offset_p95)
        - mean_offset: Mean perpendicular distance (meters)
        - offset_iqr: Interquartile range (p75 - p25), robust to outliers
        - offset_p95: 95th percentile of offsets

    Example:
        A sidewalk 3m from a road with consistent offset:
        >>> offset, iqr, p95 = compute_perpendicular_offset(sidewalk, road)
        >>> print(f"Offset: {offset:.1f}m, IQR: {iqr:.2f}m, P95: {p95:.2f}m")
        Offset: 3.0m, IQR: 0.20m, P95: 3.5m
    """
    if target_geom.is_empty or anchor_geom.is_empty:
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


def compute_side_of_street(
    target_geom: LineString,
    anchor_geom: LineString,
    sample_interval: float = 10.0,
) -> tuple[str, float]:
    """Determine which side of the anchor road the target is on.

    Uses the cross product of the anchor's direction vector and the
    vector from anchor to target. The sign determines left/right based
    on the anchor's direction of travel.

    Args:
        target_geom: Target geometry (sidewalk, bike lane)
        anchor_geom: Anchor geometry (road centerline)
        sample_interval: Distance between sample points for voting

    Returns:
        Tuple of (side, confidence) where:
        - side: 'left', 'right', or 'unknown'
        - confidence: 0-1 based on consistency of side determination

    Example:
        >>> side, conf = compute_side_of_street(sidewalk, road)
        >>> print(f"Side: {side}, Confidence: {conf:.2f}")
        Side: left, Confidence: 0.95
    """
    if target_geom.is_empty or anchor_geom.is_empty:
        return "unknown", 0.0

    # Get anchor direction vector (overall direction)
    anchor_coords = np.array(anchor_geom.coords)
    anchor_dir = anchor_coords[-1] - anchor_coords[0]
    anchor_dir_norm = anchor_dir / (np.linalg.norm(anchor_dir) + 1e-10)

    # Sample points along target
    n_samples = max(3, int(target_geom.length / sample_interval))
    distances_along = np.linspace(0, target_geom.length, n_samples)

    # Pre-sample target points (Shapely, outside JIT boundary)
    target_points = np.array([target_geom.interpolate(d).coords[0] for d in distances_along])

    # Pre-compute anchor projections (Shapely calls, outside JIT boundary)
    anchor_points = np.empty((n_samples, 2), dtype=np.float64)
    for i, tp in enumerate(target_points):
        nearest_dist = anchor_geom.project(Point(tp))
        anchor_points[i] = anchor_geom.interpolate(nearest_dist).coords[0][:2]

    # JIT-compiled voting (pure NumPy, fast)
    left_count, right_count, _ = side_of_street_vote_numba(
        target_points[:, :2], anchor_points, anchor_dir_norm
    )

    total_decisive = left_count + right_count

    if total_decisive == 0:
        return "unknown", 0.0

    # Determine side by majority vote
    if left_count > right_count:
        side = "left"
        confidence = left_count / total_decisive
    elif right_count > left_count:
        side = "right"
        confidence = right_count / total_decisive
    else:
        side = "unknown"
        confidence = 0.5

    return side, float(confidence)


def compute_parallel_alignment(
    line_a: LineString,
    line_b: LineString,
    *,
    coords_a: np.ndarray | None = None,
    coords_b: np.ndarray | None = None,
) -> float:
    """Compute how parallel two lines are (0-1).

    Based on the heading difference between the overall directions.
    Returns 1.0 for perfectly parallel lines (0 or 180 degree difference),
    0.0 for perpendicular lines.

    Args:
        line_a: First line geometry
        line_b: Second line geometry
        coords_a: Pre-extracted coordinates for line_a (optional)
        coords_b: Pre-extracted coordinates for line_b (optional)

    Returns:
        Alignment score (0-1) where 1 = parallel
    """
    if line_a.is_empty or line_b.is_empty:
        return 0.0

    if coords_a is None:
        coords_a = np.array(line_a.coords)
    if coords_b is None:
        coords_b = np.array(line_b.coords)

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

    # Side of street
    side, side_conf = compute_side_of_street(target_geom, anchor_geom, sample_interval * 2)

    # Parallel alignment
    alignment = compute_parallel_alignment(target_geom, anchor_geom)

    return RelationalFeatures(
        perpendicular_offset=offset,
        offset_iqr=offset_iqr,
        offset_p95=offset_p95,
        parallel_alignment=alignment,
        side_of_street=side,
        side_confidence=side_conf,
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
