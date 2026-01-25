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
from shapely import LineString, Point, line_interpolate_point
from shapely import distance as shapely_distance

from ._jit_helpers import compute_endpoint_proximity_numba, compute_parallel_alignment_numba


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

    # For each sample point, determine which side it's on
    sides = []
    for d in distances_along:
        target_point = np.array(target_geom.interpolate(d).coords[0])

        # Find nearest point on anchor
        nearest_dist = anchor_geom.project(Point(target_point))
        anchor_point = np.array(anchor_geom.interpolate(nearest_dist).coords[0])

        # Vector from anchor to target
        to_target = target_point - anchor_point

        # Cross product z-component determines side
        # Positive = left, Negative = right (using right-hand rule)
        cross_z = anchor_dir_norm[0] * to_target[1] - anchor_dir_norm[1] * to_target[0]

        if abs(cross_z) < 0.1:  # Too close to call
            sides.append(0)
        elif cross_z > 0:
            sides.append(1)  # Left
        else:
            sides.append(-1)  # Right

    sides = np.array(sides)

    # Count votes
    left_count = np.sum(sides > 0)
    right_count = np.sum(sides < 0)
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
