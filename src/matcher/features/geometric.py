"""Geometric feature extraction for candidate edge pairs.

This module computes geometric similarity features between road segments for
matching/conflation. The metrics are designed to handle common challenges in
road network data:

1. **Segmentation differences**: Different datasets split roads at different
   points (intersections, administrative boundaries). Two segments may represent
   the same road but only partially overlap.

2. **Digitization direction**: Roads can be digitized in either direction.
   Metrics should be direction-agnostic.

3. **Positional accuracy**: Datasets have varying accuracy. Small offsets
   shouldn't prevent matching.

Metric Selection Rationale:
--------------------------
- **hausdorff_distance**: Classic max-deviation metric. Sensitive to segmentation
  differences (one bad endpoint tanks the score). Useful for detecting outliers.

- **mean_hausdorff_distance**: Mean of min-distances instead of max. Robust to
  segmentation - a partial overlap still scores well if the overlapping portions
  align. Preferred for datasets with different segmentation schemes.

- **buffer_iou**: Intersection-over-Union of buffered geometries. Robust to small
  positional offsets and segmentation. Good general-purpose similarity metric.

- **overlap_ratio**: What fraction of line A falls within line B's buffer? Answers
  "how much of this segment has a corresponding segment in the other dataset?"

- **projection_distance**: Average perpendicular distance between curves. Computed
  bidirectionally (A→B and B→A) for symmetry. More stable than Hausdorff for
  typical road matching scenarios.

- **heading_delta**: Overall direction difference. Handles bidirectional roads
  (0° and 180° both score well). Helps distinguish parallel roads from the same road.

- **length_ratio**: Ratio of lengths (shorter/longer). Helps identify segmentation
  mismatches vs. true non-matches.

- **centroid_distance**: Simple proximity check. Fast to compute, useful for
  initial filtering.
"""

from typing import NamedTuple, Union

import numpy as np
from shapely import LineString, MultiLineString, Point, hausdorff_distance
from shapely.ops import linemerge


def _to_linestring(geom: Union[LineString, MultiLineString]) -> LineString:
    """Convert geometry to LineString.

    For MultiLineString, tries to merge first, then falls back to longest component.
    """
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        # Handle empty MultiLineString
        if geom.is_empty or len(geom.geoms) == 0:
            return LineString()
        # Try to merge connected components
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
        # Fall back to longest component
        return max(geom.geoms, key=lambda g: g.length)
    raise TypeError(f"Expected LineString or MultiLineString, got {type(geom)}")


class GeometricFeatures(NamedTuple):
    """Geometric features for a candidate pair.

    All distance metrics are in the same units as the input geometries
    (should be projected CRS, typically meters).

    For matching scores, distances are normalized to 0-1 where higher = better:
        score = max(0, 1 - distance / threshold)
    """

    hausdorff_distance: float
    """Maximum deviation between curves (meters).
    Sensitive to segmentation - one far endpoint ruins the score.
    Use mean_hausdorff_distance for robustness to partial overlaps."""

    mean_hausdorff_distance: float
    """Mean of minimum distances from each vertex to the other curve (meters).
    Robust to segmentation differences - partial overlaps score well if
    the overlapping portions align. Preferred over hausdorff_distance for
    datasets with different segmentation schemes."""

    buffer_iou: float
    """Intersection over Union of buffered geometries (0-1).
    Robust to small positional offsets. Good general-purpose metric."""

    heading_delta: float
    """Overall direction difference in degrees (0-180).
    Accounts for bidirectional roads (0° and 180° both indicate alignment).
    Helps distinguish parallel roads from the same road."""

    length_ratio: float
    """Ratio of lengths: min(len_a, len_b) / max(len_a, len_b) (0-1).
    Value of 1 means same length. Low values suggest segmentation mismatch
    or different roads entirely."""

    projection_distance: float
    """Bidirectional average perpendicular distance (meters).
    Averages distances from A's vertices to B and B's vertices to A.
    More stable than Hausdorff for typical road matching."""

    centroid_distance: float
    """Distance between segment centroids (meters).
    Simple proximity check, useful for initial filtering."""

    overlap_ratio: float
    """Fraction of line_a that falls within line_b's buffer (0-1).
    Answers: 'how much of this segment has a corresponding segment?'
    Useful for detecting segmentation mismatches."""


def compute_geometric_features(
    line_a: Union[LineString, MultiLineString],
    line_b: Union[LineString, MultiLineString],
    buffer_radius: float = 10.0,
) -> GeometricFeatures:
    """Compute geometric similarity features between two LineStrings.

    Args:
        line_a: First geometry (LineString or MultiLineString, projected CRS)
        line_b: Second geometry (LineString or MultiLineString, projected CRS)
        buffer_radius: Buffer radius for IoU calculation (meters)

    Returns:
        GeometricFeatures tuple
    """
    # Convert MultiLineString to LineString if needed
    line_a = _to_linestring(line_a)
    line_b = _to_linestring(line_b)

    coords_a = np.array(line_a.coords)
    coords_b = np.array(line_b.coords)

    # Hausdorff distance (max deviation) - using Shapely's implementation
    # Hausdorff is symmetric, so digitization direction doesn't matter
    hausdorff = hausdorff_distance(line_a, line_b)

    # Mean Hausdorff (robust to segmentation - uses mean instead of max)
    mean_hausdorff = _mean_hausdorff_distance(line_a, line_b)

    # Buffer IoU
    buffer_iou = _buffer_iou(line_a, line_b, buffer_radius)

    # Heading delta (overall direction)
    heading_a = _compute_heading(coords_a[0], coords_a[-1])
    heading_b = _compute_heading(coords_b[0], coords_b[-1])
    heading_delta = _angle_diff(heading_a, heading_b)

    # Length ratio
    len_a, len_b = line_a.length, line_b.length
    length_ratio = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 0.0

    # Average projection distance
    projection_distance = _avg_projection_distance(line_a, line_b)

    # Centroid distance
    centroid_distance = line_a.centroid.distance(line_b.centroid)

    # Overlap ratio
    overlap_ratio = _overlap_ratio(line_a, line_b, buffer_radius)

    return GeometricFeatures(
        hausdorff_distance=hausdorff,
        mean_hausdorff_distance=mean_hausdorff,
        buffer_iou=buffer_iou,
        heading_delta=heading_delta,
        length_ratio=length_ratio,
        projection_distance=projection_distance,
        centroid_distance=centroid_distance,
        overlap_ratio=overlap_ratio,
    )


def _buffer_iou(line_a: LineString, line_b: LineString, radius: float) -> float:
    """Compute Intersection over Union of buffered geometries."""
    buf_a = line_a.buffer(radius)
    buf_b = line_b.buffer(radius)

    intersection_area = buf_a.intersection(buf_b).area
    union_area = buf_a.union(buf_b).area

    return intersection_area / union_area if union_area > 0 else 0.0


def _compute_heading(start: np.ndarray, end: np.ndarray) -> float:
    """Compute heading in degrees from start to end point (0-360)."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    heading = np.degrees(np.arctan2(dy, dx))
    return (heading + 360) % 360


def _angle_diff(a: float, b: float) -> float:
    """Compute minimum angle difference in degrees (0-180).

    Handles the fact that roads can be traversed in either direction.
    """
    diff = abs(a - b)
    if diff > 180:
        diff = 360 - diff

    # Consider opposite direction (road could be traversed either way)
    opposite_diff = abs(180 - diff)

    return min(diff, opposite_diff)


def _mean_hausdorff_distance(line_a: LineString, line_b: LineString) -> float:
    """Compute mean Hausdorff distance (mean of min distances).

    Standard Hausdorff uses max(min_distances), which is sensitive to
    segmentation - if one endpoint is far from the other curve, the whole
    score is ruined. This "Modified Hausdorff" uses mean instead of max.

    Example:
        Line A: 100m long, overlaps with B for 70m
        Line B: 70m long, fully within A's extent

        Standard Hausdorff: ~30m (the gap at A's endpoints)
        Mean Hausdorff: ~15m (averages the well-aligned middle with the gaps)

    This is widely used in road conflation literature because real-world
    datasets often have different segmentation schemes.

    References:
        Dubuisson & Jain (1994) "A Modified Hausdorff Distance for Object Matching"
    """
    # Min distances from each point in A to line B
    dists_a_to_b = [line_b.distance(Point(coord)) for coord in line_a.coords]
    # Min distances from each point in B to line A
    dists_b_to_a = [line_a.distance(Point(coord)) for coord in line_b.coords]

    all_min_dists = dists_a_to_b + dists_b_to_a

    if not all_min_dists:
        return float("inf")

    return np.mean(all_min_dists)


def _avg_projection_distance(line_a: LineString, line_b: LineString) -> float:
    """Compute bidirectional average perpendicular distance.

    For each vertex in A, finds distance to nearest point on B, and vice versa.
    Returns the mean of all these distances.

    Note: This is mathematically equivalent to mean_hausdorff_distance.
    Kept as separate function for semantic clarity - "projection distance"
    emphasizes alignment quality, while "mean Hausdorff" emphasizes the
    relationship to the classic Hausdorff metric.
    """
    # Delegate to mean_hausdorff_distance to avoid code duplication
    return _mean_hausdorff_distance(line_a, line_b)


def _overlap_ratio(line_a: LineString, line_b: LineString, buffer_radius: float) -> float:
    """Compute the ratio of line_a that overlaps with line_b's buffer."""
    buf_b = line_b.buffer(buffer_radius)
    overlap = line_a.intersection(buf_b)

    if overlap.is_empty:
        return 0.0

    overlap_length = overlap.length if hasattr(overlap, "length") else 0.0
    return overlap_length / line_a.length if line_a.length > 0 else 0.0


def compute_segment_heading(line: LineString) -> float:
    """Compute the overall heading of a line segment."""
    coords = np.array(line.coords)
    return _compute_heading(coords[0], coords[-1])


def compute_heading_consistency(line: LineString, sample_interval: float = 10.0) -> float:
    """Compute how consistent the heading is along the line.

    Returns a value 0-1 where 1 means perfectly straight.
    """
    if line.length < sample_interval * 2:
        return 1.0

    # Sample points along the line
    n_samples = max(3, int(line.length / sample_interval))
    distances = np.linspace(0, line.length, n_samples)
    points = [np.array(line.interpolate(d).coords[0]) for d in distances]

    # Compute headings between consecutive points
    headings = []
    for i in range(len(points) - 1):
        h = _compute_heading(points[i], points[i + 1])
        headings.append(h)

    if len(headings) < 2:
        return 1.0

    # Compute variance of headings (accounting for circular nature)
    heading_diffs = []
    for i in range(len(headings) - 1):
        diff = _angle_diff(headings[i], headings[i + 1])
        heading_diffs.append(diff)

    avg_diff = np.mean(heading_diffs)

    # Normalize to 0-1 (0 degrees diff = 1.0, 90 degrees diff = 0.0)
    return max(0.0, 1.0 - avg_diff / 90.0)
