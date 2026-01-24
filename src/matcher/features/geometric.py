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

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from shapely import LineString, hausdorff_distance, points
from shapely import distance as shapely_distance

if TYPE_CHECKING:
    from shapely import Polygon


# Buffer cache size - limits memory usage while providing speedup for repeated geometries
# With ~500K segments and 2 radii, full caching would use ~8GB. Limit to ~800MB.
_BUFFER_CACHE_SIZE = 50_000


@lru_cache(maxsize=_BUFFER_CACHE_SIZE)
def _cached_buffer(geom_wkb: bytes, radius: float):
    """Compute buffer with LRU caching based on geometry WKB and radius.

    Uses WKB (Well-Known Binary) representation as the cache key since
    LineString objects are not hashable. This provides automatic cache
    invalidation when geometries change.

    Args:
        geom_wkb: WKB representation of the geometry
        radius: Buffer radius in meters

    Returns:
        Buffered polygon geometry
    """
    from shapely import wkb

    return wkb.loads(geom_wkb).buffer(radius)


def get_cached_buffer(geom: LineString, radius: float):
    """Get a buffered geometry, using cache when possible.

    This function provides a caching layer for buffer operations.
    For repeated geometries (common in ML scoring where the same
    segment appears in multiple candidate pairs), this avoids
    redundant buffer computations.

    Args:
        geom: LineString geometry to buffer
        radius: Buffer radius in meters

    Returns:
        Buffered polygon geometry
    """
    try:
        # Use WKB as cache key (hashable representation of geometry)
        return _cached_buffer(geom.wkb, radius)
    except (AttributeError, TypeError):
        # Fall back to direct computation if WKB serialization fails
        # (e.g., geometry has no wkb attribute or WKB is not hashable)
        return geom.buffer(radius)


def clear_buffer_cache():
    """Clear the buffer cache to free memory.

    Call this after completing a batch of feature computations
    if memory is a concern, or for testing purposes.
    """
    _cached_buffer.cache_clear()


def get_buffer_cache_info():
    """Get buffer cache statistics.

    Returns:
        CacheInfo with hits, misses, maxsize, and currsize
    """
    return _cached_buffer.cache_info()


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

    hausdorff_p95_distance: float
    """95th percentile of min-distances (meters).
    More robust than max (hausdorff_distance) as it ignores the top 5% of
    outliers (spurs, vertex errors), but more conservative than mean."""

    buffer_iou_5m: float
    """Intersection over Union of 5m buffered geometries (0-1).
    Captures tight alignment - exact centerline matches."""

    buffer_iou_15m: float
    """Intersection over Union of 15m buffered geometries (0-1).
    Captures offset alignment - sidewalks, bike lanes parallel to roads."""

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

    collinear_gap_ratio: float
    """Penalizes collinear segments with poor along-track overlap (0-1).
    1.0 = not collinear OR collinear with good overlap (no penalty)
    0.0-1.0 = collinear with poor overlap (tip-to-tip penalty)
    Addresses false matches between consecutive road segments that are
    end-to-end but have perfect name similarity and heading alignment."""


def compute_geometric_features(
    line_a: LineString,
    line_b: LineString,
) -> GeometricFeatures:
    """Compute geometric similarity features between two LineStrings.

    Args:
        line_a: First geometry (LineString, projected CRS)
        line_b: Second geometry (LineString, projected CRS)

    Returns:
        GeometricFeatures tuple
    """

    coords_a = np.array(line_a.coords)
    coords_b = np.array(line_b.coords)

    # Hausdorff distance (max deviation) - using Shapely's implementation
    # Hausdorff is symmetric, so digitization direction doesn't matter
    hausdorff = hausdorff_distance(line_a, line_b)

    # Mean and P95 Hausdorff (robust to segmentation)
    mean_hausdorff, p95_hausdorff = _compute_hausdorff_stats(line_a, line_b)

    # Multi-scale Buffer IoU:
    # - 5m: Captures tight alignment (exact centerline matches)
    # - 15m: Captures offset alignment (sidewalks, bike lanes parallel to roads)
    # Use cached buffers when possible for repeated geometries
    buf_a_5m = get_cached_buffer(line_a, 5.0)
    buf_b_5m = get_cached_buffer(line_b, 5.0)
    buf_a_15m = get_cached_buffer(line_a, 15.0)
    buf_b_15m = get_cached_buffer(line_b, 15.0)

    buffer_iou_5m = _buffer_iou_from_buffers(buf_a_5m, buf_b_5m)
    buffer_iou_15m = _buffer_iou_from_buffers(buf_a_15m, buf_b_15m)

    # Heading delta (overall direction)
    heading_a = _compute_heading(coords_a[0], coords_a[-1])
    heading_b = _compute_heading(coords_b[0], coords_b[-1])
    heading_delta = _angle_diff(heading_a, heading_b)

    # Length ratio
    len_a, len_b = line_a.length, line_b.length
    length_ratio = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 0.0

    # Average projection distance (mathematically equivalent to mean_hausdorff)
    # Reuse already-computed value to avoid redundant computation
    projection_distance = mean_hausdorff

    # Centroid distance
    centroid_distance = line_a.centroid.distance(line_b.centroid)

    # Overlap ratio - NOTE: Changed from 10m to 15m buffer for performance.
    # This is a semantic change but overlap_ratio is typically ~1.0 anyway
    # due to blocking bias (candidates already spatially filtered).
    # Using 15m buffer avoids creating an additional buffer geometry.
    overlap_ratio = _overlap_ratio(line_a, buf_b_15m)

    # Collinear gap ratio (penalty for tip-to-tip segments)
    collinear_gap_ratio = compute_collinear_gap_ratio(line_a, line_b)

    return GeometricFeatures(
        hausdorff_distance=hausdorff,
        mean_hausdorff_distance=mean_hausdorff,
        hausdorff_p95_distance=p95_hausdorff,
        buffer_iou_5m=buffer_iou_5m,
        buffer_iou_15m=buffer_iou_15m,
        heading_delta=heading_delta,
        length_ratio=length_ratio,
        projection_distance=projection_distance,
        centroid_distance=centroid_distance,
        overlap_ratio=overlap_ratio,
        collinear_gap_ratio=collinear_gap_ratio,
    )


def _buffer_iou(line_a: LineString, line_b: LineString, radius: float) -> float:
    """Compute Intersection over Union of buffered geometries."""
    buf_a = line_a.buffer(radius)
    buf_b = line_b.buffer(radius)
    return _buffer_iou_from_buffers(buf_a, buf_b)


def _buffer_iou_from_buffers(buf_a: "Polygon", buf_b: "Polygon") -> float:
    """Compute Intersection over Union from pre-computed buffers.

    Uses the identity: Union = A + B - Intersection to avoid computing
    the union geometry explicitly, which is more expensive than area lookups.

    Args:
        buf_a: Pre-computed buffer polygon for line A
        buf_b: Pre-computed buffer polygon for line B

    Returns:
        IoU score between 0 and 1
    """
    intersection_area = buf_a.intersection(buf_b).area
    # Optimization: union_area = A + B - intersection (avoids union geometry op)
    union_area = buf_a.area + buf_b.area - intersection_area

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


def _compute_hausdorff_stats(line_a: LineString, line_b: LineString) -> tuple[float, float]:
    """Compute mean and P95 Hausdorff distances (from min distances).

    Standard Hausdorff uses max(min_distances), which is sensitive to
    segmentation/outliers.

    Uses Shapely's vectorized distance function for efficient computation
    of distances from all vertices to the opposite line.

    Returns:
        Tuple of (mean_distance, p95_distance)
    """
    coords_a = np.array(line_a.coords)
    coords_b = np.array(line_b.coords)

    if len(coords_a) == 0 or len(coords_b) == 0:
        return float("inf"), float("inf")

    # Create point arrays from coordinates (vectorized)
    points_a = points(coords_a)
    points_b = points(coords_b)

    # Note: prepare() is not used here as it primarily optimizes predicate
    # operations (contains, intersects) in Shapely 2.0, not distance calculations.
    # The vectorized shapely.distance() is already optimized for batch operations.

    # Vectorized distance computation: all points in A to line B
    dists_a_to_b = shapely_distance(points_a, line_b)
    # Vectorized distance computation: all points in B to line A
    dists_b_to_a = shapely_distance(points_b, line_a)

    # Combine all distances
    all_min_dists = np.concatenate([dists_a_to_b, dists_b_to_a])

    mean_dist = float(np.mean(all_min_dists))
    p95_dist = float(np.percentile(all_min_dists, 95))

    return mean_dist, p95_dist


def _avg_projection_distance(line_a: LineString, line_b: LineString) -> float:
    """Compute bidirectional average perpendicular distance.

    For each vertex in A, finds distance to nearest point on B, and vice versa.
    Returns the mean of all these distances.

    Note: This is mathematically equivalent to mean_hausdorff_distance.
    Kept as separate function for semantic clarity - "projection distance"
    emphasizes alignment quality, while "mean Hausdorff" emphasizes the
    relationship to the classic Hausdorff metric.
    """
    # Delegate to _compute_hausdorff_stats to avoid code duplication
    mean_dist, _ = _compute_hausdorff_stats(line_a, line_b)
    return mean_dist


def _overlap_ratio(line_a: LineString, buf_b: "Polygon") -> float:
    """Compute the ratio of line_a that overlaps with line_b's buffer.

    Args:
        line_a: Line to measure overlap for
        buf_b: Pre-computed buffer polygon of line_b

    Returns:
        Fraction of line_a length that falls within buf_b (0 to 1)
    """
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


def compute_sinuosity(line: LineString) -> float:
    """Compute sinuosity of a line (ratio of path length to straight-line distance).

    Sinuosity = line_length / straight_distance
    - 1.0 = perfectly straight
    - >1.0 = increasingly curvy
    - Loop (start == end): returns 100.0 to indicate loop

    Args:
        line: LineString geometry

    Returns:
        Sinuosity ratio (>= 1.0, or 100.0 for loops)
    """
    if line is None or line.is_empty:
        return 1.0

    line_length = line.length
    if line_length <= 0:
        return 1.0

    coords = np.array(line.coords)
    if len(coords) < 2:
        return 1.0

    # Compute straight-line (Euclidean) distance between endpoints
    start = coords[0]
    end = coords[-1]
    straight_distance = np.sqrt(np.sum((end - start) ** 2))

    # Handle loop case (start == end)
    if straight_distance < 1e-9:
        return 100.0

    return line_length / straight_distance


def compute_vertex_density(line: LineString) -> float:
    """Compute vertex density of a line (vertices per meter).

    Higher density often indicates more detailed/higher-quality data.
    Lower density indicates simpler geometry.

    Args:
        line: LineString geometry

    Returns:
        Vertices per meter (>= 0.0)
    """
    if line is None or line.is_empty:
        return 0.0

    line_length = line.length
    if line_length <= 0:
        return 0.0

    n_vertices = len(line.coords)
    return n_vertices / line_length


def compute_length_bin(length_m: float) -> int:
    """Compute length bin for a segment.

    Bins:
    - 0 = short (<10m)
    - 1 = medium (10-100m)
    - 2 = long (100-500m)
    - 3 = highway (>500m)

    Args:
        length_m: Segment length in meters

    Returns:
        Length bin (0-3)
    """
    if length_m < 10:
        return 0
    elif length_m < 100:
        return 1
    elif length_m < 500:
        return 2
    else:
        return 3


def compute_shape_complexity(line: LineString, angle_threshold: float = 10.0) -> int:
    """Count significant direction changes (turns) in a line.

    A "significant turn" is where the heading changes by more than
    angle_threshold degrees between consecutive segments.

    Args:
        line: LineString geometry
        angle_threshold: Minimum angle change to count as a turn (degrees)

    Returns:
        Number of significant turns (>= 0)
    """
    if line is None or line.is_empty:
        return 0

    coords = np.array(line.coords)
    if len(coords) < 3:
        return 0

    turn_count = 0

    # Compute headings between consecutive points
    for i in range(len(coords) - 2):
        # Heading from point i to i+1
        heading1 = _compute_heading(coords[i], coords[i + 1])
        # Heading from point i+1 to i+2
        heading2 = _compute_heading(coords[i + 1], coords[i + 2])

        # Compute angle difference
        angle_diff = abs(heading1 - heading2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        if angle_diff > angle_threshold:
            turn_count += 1

    return turn_count


def compute_collinear_gap_ratio(
    line_a: LineString,
    line_b: LineString,
    heading_threshold: float = 15.0,
    min_overlap_fraction: float = 0.1,
) -> float:
    """Detect collinear segments that barely touch (tip-to-tip penalty).

    This feature addresses the problem where consecutive road segments
    (same street, same direction, but end-to-end) score artificially high
    because name similarity and heading alignment are perfect.

    Algorithm:
    1. Check if segments are collinear (heading_delta < threshold)
    2. If not collinear → return 1.0 (no penalty, let other features decide)
    3. Project both segments onto their common direction axis
    4. Compute 1D overlap ratio along that axis
    5. If good overlap (≥ min_overlap_fraction) → return 1.0 (no penalty)
    6. If poor overlap or gap → return low value (0.0-1.0)

    Args:
        line_a: First geometry (LineString, projected CRS)
        line_b: Second geometry (LineString, projected CRS)
        heading_threshold: Max heading difference to consider collinear (degrees)
        min_overlap_fraction: Minimum overlap to not penalize (fraction 0-1)

    Returns:
        1.0 = not collinear OR collinear with good overlap (no penalty)
        0.0-1.0 = collinear with poor overlap (penalty scaled by overlap)

    Example:
        Segment A: (0,0) → (100,0)
        Segment B: (100,0) → (200,0)  # tip-to-tip
        → Returns ~0.0 (strong penalty)

        Segment A: (0,0) → (100,0)
        Segment B: (25,0) → (75,0)  # contained within
        → Returns 1.0 (good overlap, no penalty)
    """
    # Handle degenerate cases
    if line_a.is_empty or line_b.is_empty:
        return 1.0
    if line_a.length <= 0 or line_b.length <= 0:
        return 1.0

    coords_a = np.array(line_a.coords)
    coords_b = np.array(line_b.coords)

    # Step 1: Check collinearity via heading
    heading_a = _compute_heading(coords_a[0], coords_a[-1])
    heading_b = _compute_heading(coords_b[0], coords_b[-1])
    heading_diff = _angle_diff(heading_a, heading_b)

    if heading_diff > heading_threshold:
        # Not collinear - no penalty
        return 1.0

    # Step 2: Project onto common direction
    # Use the average of both headings as the reference direction
    # Handle bidirectional case: if headings differ by ~180°, align them first
    if abs(heading_a - heading_b) > 90 and abs(heading_a - heading_b) < 270:
        # They're roughly opposite, flip one
        heading_b_aligned = (heading_b + 180) % 360
    else:
        heading_b_aligned = heading_b

    avg_heading = (heading_a + heading_b_aligned) / 2
    ref_angle = np.radians(avg_heading)
    ref_dir = np.array([np.cos(ref_angle), np.sin(ref_angle)])

    # Project all endpoints onto the reference direction
    a_start_proj = np.dot(coords_a[0], ref_dir)
    a_end_proj = np.dot(coords_a[-1], ref_dir)
    b_start_proj = np.dot(coords_b[0], ref_dir)
    b_end_proj = np.dot(coords_b[-1], ref_dir)

    # Step 3: Compute 1D overlap
    a_min, a_max = min(a_start_proj, a_end_proj), max(a_start_proj, a_end_proj)
    b_min, b_max = min(b_start_proj, b_end_proj), max(b_start_proj, b_end_proj)

    overlap_start = max(a_min, b_min)
    overlap_end = min(a_max, b_max)
    overlap_length = max(0, overlap_end - overlap_start)

    # Use the smaller segment's extent as the denominator
    # This matches the labeling guideline: "10% of shorter segment"
    smaller_extent = min(a_max - a_min, b_max - b_min)
    if smaller_extent <= 0:
        # Degenerate case (point-like segment)
        return 1.0

    along_track_overlap = overlap_length / smaller_extent

    # Step 4: Return score
    if along_track_overlap >= min_overlap_fraction:
        # Good overlap - no penalty
        return 1.0

    # Poor overlap - scale penalty based on how bad
    # 0% overlap → 0.0, min_overlap_fraction% → 1.0
    return along_track_overlap / min_overlap_fraction
