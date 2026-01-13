"""Geometric feature extraction for candidate edge pairs."""

from typing import NamedTuple, Union

import numpy as np
from shapely import LineString, MultiLineString, Point, frechet_distance, hausdorff_distance
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
    hausdorff = hausdorff_distance(line_a, line_b)

    # Frechet distance (shape similarity) - using Shapely's implementation
    frechet = frechet_distance(line_a, line_b)

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
