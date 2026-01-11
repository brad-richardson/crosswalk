"""Geometric feature extraction for candidate edge pairs."""

from typing import NamedTuple

import numpy as np
from shapely import LineString, Point
from shapely.ops import nearest_points
from scipy.spatial.distance import directed_hausdorff


class GeometricFeatures(NamedTuple):
    """Geometric features for a candidate pair."""

    hausdorff_distance: float  # Maximum deviation between curves
    frechet_distance: float  # Shape similarity (discrete approximation)
    buffer_iou: float  # Intersection over Union of buffered geometries
    heading_delta: float  # Overall direction difference (degrees, 0-180)
    length_ratio: float  # Ratio of lengths (0-1, 1 = same length)
    projection_distance: float  # Average perpendicular distance
    centroid_distance: float  # Distance between centroids
    overlap_ratio: float  # Ratio of overlapping length


def compute_geometric_features(
    line_a: LineString,
    line_b: LineString,
    buffer_radius: float = 10.0,
) -> GeometricFeatures:
    """Compute geometric similarity features between two LineStrings.

    Args:
        line_a: First LineString (should be in projected CRS, meters)
        line_b: Second LineString (should be in projected CRS, meters)
        buffer_radius: Buffer radius for IoU calculation (meters)

    Returns:
        GeometricFeatures tuple
    """
    coords_a = np.array(line_a.coords)
    coords_b = np.array(line_b.coords)

    # Hausdorff distance (max deviation)
    hausdorff = _hausdorff_distance(coords_a, coords_b)

    # Discrete Frechet distance (approximate)
    frechet = _discrete_frechet(coords_a, coords_b)

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
        frechet_distance=frechet,
        buffer_iou=buffer_iou,
        heading_delta=heading_delta,
        length_ratio=length_ratio,
        projection_distance=projection_distance,
        centroid_distance=centroid_distance,
        overlap_ratio=overlap_ratio,
    )


def _hausdorff_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute Hausdorff distance between two point sequences."""
    h1 = directed_hausdorff(P, Q)[0]
    h2 = directed_hausdorff(Q, P)[0]
    return max(h1, h2)


def _discrete_frechet(P: np.ndarray, Q: np.ndarray, max_points: int = 50) -> float:
    """Compute discrete Frechet distance using dynamic programming.

    This is an O(nm) approximation of the continuous Frechet distance.
    For performance, lines with more than max_points are resampled.

    Args:
        P: First point sequence (n x 2)
        Q: Second point sequence (m x 2)
        max_points: Maximum points before resampling (for O(n*m) performance)

    Returns:
        Discrete Frechet distance
    """
    n, m = len(P), len(Q)

    if n == 0 or m == 0:
        return float("inf")

    # Resample if too many points (O(n*m) complexity can be expensive)
    if n > max_points:
        indices = np.linspace(0, n - 1, max_points, dtype=int)
        P = P[indices]
        n = len(P)

    if m > max_points:
        indices = np.linspace(0, m - 1, max_points, dtype=int)
        Q = Q[indices]
        m = len(Q)

    # Use iterative DP instead of recursive to avoid stack overflow
    ca = np.zeros((n, m))

    # Precompute distance matrix
    for i in range(n):
        for j in range(m):
            ca[i, j] = np.linalg.norm(P[i] - Q[j])

    # DP table
    dp = np.zeros((n, m))
    dp[0, 0] = ca[0, 0]

    # Fill first column
    for i in range(1, n):
        dp[i, 0] = max(dp[i - 1, 0], ca[i, 0])

    # Fill first row
    for j in range(1, m):
        dp[0, j] = max(dp[0, j - 1], ca[0, j])

    # Fill rest of table
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(
                min(dp[i - 1, j], dp[i - 1, j - 1], dp[i, j - 1]),
                ca[i, j],
            )

    return dp[n - 1, m - 1]


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


def _avg_projection_distance(line_a: LineString, line_b: LineString) -> float:
    """Compute average perpendicular distance from line_a vertices to line_b."""
    distances = []

    for coord in line_a.coords:
        point = Point(coord)
        dist = line_b.distance(point)
        distances.append(dist)

    if not distances:
        return float("inf")

    return np.mean(distances)


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
