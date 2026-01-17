"""Relational feature extraction for spatial context.

This module computes features based on spatial relationships between segments,
particularly useful for matching parallel infrastructure (sidewalks, bike lanes)
to their anchor roads, and for inferring connectivity from endpoint proximity.

These features work without requiring explicit topology in the target data,
making them suitable for raw "spaghetti" line datasets.

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
from shapely import LineString, MultiLineString, Point
from shapely.ops import linemerge


class RelationalFeatures(NamedTuple):
    """Relational features for a candidate pair.

    These features capture spatial relationships that help match parallel
    infrastructure (sidewalks, bike lanes) to their anchor roads.
    """

    perpendicular_offset: float
    """Mean perpendicular distance from target to anchor line (meters).
    Low values indicate the segment runs parallel and close to the anchor."""

    offset_consistency: float
    """Standard deviation of perpendicular distances (meters).
    Low values indicate consistent parallel offset along the entire length."""

    parallel_alignment: float
    """How parallel the segment is to anchor (0-1).
    Based on heading difference: 1.0 = parallel, 0.0 = perpendicular."""

    side_of_street: str
    """Which side of the anchor road: 'left', 'right', or 'unknown'.
    Determined by cross product of direction vectors."""

    side_confidence: float
    """Confidence in side determination (0-1).
    Low when segment crosses back and forth across anchor."""


def _to_linestring(geom: LineString | MultiLineString) -> LineString:
    """Convert geometry to LineString."""
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        if geom.is_empty or len(geom.geoms) == 0:
            return LineString()
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
        return max(geom.geoms, key=lambda g: g.length)
    raise TypeError(f"Expected LineString or MultiLineString, got {type(geom)}")


def compute_perpendicular_offset(
    target_geom: LineString | MultiLineString,
    anchor_geom: LineString | MultiLineString,
    sample_interval: float = 5.0,
) -> tuple[float, float]:
    """Compute perpendicular offset from target to anchor line.

    Samples points along the target line and measures perpendicular
    distance to the anchor. Returns both mean offset and consistency
    (standard deviation).

    Args:
        target_geom: Target geometry (sidewalk, bike lane)
        anchor_geom: Anchor geometry (road centerline)
        sample_interval: Distance between sample points (meters)

    Returns:
        Tuple of (mean_offset, offset_std)

    Example:
        A sidewalk 3m from a road with consistent offset:
        >>> offset, std = compute_perpendicular_offset(sidewalk, road)
        >>> print(f"Offset: {offset:.1f}m, Std: {std:.2f}m")
        Offset: 3.0m, Std: 0.15m
    """
    target = _to_linestring(target_geom)
    anchor = _to_linestring(anchor_geom)

    if target.is_empty or anchor.is_empty:
        return float("inf"), float("inf")

    # Sample points along target
    n_samples = max(3, int(target.length / sample_interval))
    distances_along = np.linspace(0, target.length, n_samples)

    # Compute perpendicular distance at each sample point
    offsets = []
    for d in distances_along:
        point = target.interpolate(d)
        offset = anchor.distance(point)
        offsets.append(offset)

    offsets = np.array(offsets)
    mean_offset = float(np.mean(offsets))
    offset_std = float(np.std(offsets))

    return mean_offset, offset_std


def compute_side_of_street(
    target_geom: LineString | MultiLineString,
    anchor_geom: LineString | MultiLineString,
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
    target = _to_linestring(target_geom)
    anchor = _to_linestring(anchor_geom)

    if target.is_empty or anchor.is_empty:
        return "unknown", 0.0

    # Get anchor direction vector (overall direction)
    anchor_coords = np.array(anchor.coords)
    anchor_dir = anchor_coords[-1] - anchor_coords[0]
    anchor_dir_norm = anchor_dir / (np.linalg.norm(anchor_dir) + 1e-10)

    # Sample points along target
    n_samples = max(3, int(target.length / sample_interval))
    distances_along = np.linspace(0, target.length, n_samples)

    # For each sample point, determine which side it's on
    sides = []
    for d in distances_along:
        target_point = np.array(target.interpolate(d).coords[0])

        # Find nearest point on anchor
        nearest_dist = anchor.project(Point(target_point))
        anchor_point = np.array(anchor.interpolate(nearest_dist).coords[0])

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
    line_a: LineString | MultiLineString,
    line_b: LineString | MultiLineString,
) -> float:
    """Compute how parallel two lines are (0-1).

    Based on the heading difference between the overall directions.
    Returns 1.0 for perfectly parallel lines (0 or 180 degree difference),
    0.0 for perpendicular lines.

    Args:
        line_a: First line geometry
        line_b: Second line geometry

    Returns:
        Alignment score (0-1) where 1 = parallel
    """
    a = _to_linestring(line_a)
    b = _to_linestring(line_b)

    if a.is_empty or b.is_empty:
        return 0.0

    # Compute overall headings
    coords_a = np.array(a.coords)
    coords_b = np.array(b.coords)

    heading_a = _compute_heading(coords_a[0], coords_a[-1])
    heading_b = _compute_heading(coords_b[0], coords_b[-1])

    # Compute minimum angle difference (accounting for bidirectional)
    diff = abs(heading_a - heading_b)
    if diff > 180:
        diff = 360 - diff

    # Consider opposite direction as parallel too
    opposite_diff = abs(180 - diff)
    min_diff = min(diff, opposite_diff)

    # Convert to 0-1 score (0 degrees = 1.0, 90 degrees = 0.0)
    alignment = max(0.0, 1.0 - min_diff / 90.0)

    return alignment


def _compute_heading(start: np.ndarray, end: np.ndarray) -> float:
    """Compute heading in degrees from start to end point (0-360)."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    heading = np.degrees(np.arctan2(dy, dx))
    return (heading + 360) % 360


def compute_relational_features(
    target_geom: LineString | MultiLineString,
    anchor_geom: LineString | MultiLineString,
    sample_interval: float = 5.0,
) -> RelationalFeatures:
    """Compute all relational features for a target/anchor pair.

    Args:
        target_geom: Target geometry (sidewalk, bike lane)
        anchor_geom: Anchor geometry (road centerline)
        sample_interval: Distance between sample points

    Returns:
        RelationalFeatures named tuple
    """
    # Perpendicular offset
    offset, offset_std = compute_perpendicular_offset(target_geom, anchor_geom, sample_interval)

    # Side of street
    side, side_conf = compute_side_of_street(target_geom, anchor_geom, sample_interval * 2)

    # Parallel alignment
    alignment = compute_parallel_alignment(target_geom, anchor_geom)

    return RelationalFeatures(
        perpendicular_offset=offset,
        offset_consistency=offset_std,
        parallel_alignment=alignment,
        side_of_street=side,
        side_confidence=side_conf,
    )


def compute_endpoint_proximity(
    target_geom: LineString | MultiLineString,
    endpoint_coords: np.ndarray,
    tolerance: float = 5.0,
) -> tuple[float, float, int]:
    """Compute endpoint proximity features.

    Finds distance from target's endpoints to the nearest endpoints
    of other segments (provided as an array of coordinates).

    Args:
        target_geom: Target geometry
        endpoint_coords: Array of shape (N, 2) with other endpoint coordinates
        tolerance: Distance threshold for counting "shared" endpoints (meters)

    Returns:
        Tuple of (start_proximity, end_proximity, shared_count) where:
        - start_proximity: Distance to nearest endpoint from start
        - end_proximity: Distance to nearest endpoint from end
        - shared_count: Number of endpoints within tolerance
    """
    target = _to_linestring(target_geom)

    if target.is_empty or len(endpoint_coords) == 0:
        return float("inf"), float("inf"), 0

    target_coords = np.array(target.coords)
    start = target_coords[0]
    end = target_coords[-1]

    # Compute distances from start/end to all other endpoints
    start_dists = np.linalg.norm(endpoint_coords - start, axis=1)
    end_dists = np.linalg.norm(endpoint_coords - end, axis=1)

    # Find minimum distances
    start_proximity = float(np.min(start_dists)) if len(start_dists) > 0 else float("inf")
    end_proximity = float(np.min(end_dists)) if len(end_dists) > 0 else float("inf")

    # Count endpoints within tolerance of either end
    within_start = np.sum(start_dists <= tolerance)
    within_end = np.sum(end_dists <= tolerance)
    shared_count = int(within_start + within_end)

    return start_proximity, end_proximity, shared_count


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
