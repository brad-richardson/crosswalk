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

CRS Requirements:
-----------------
**IMPORTANT**: All geometries passed to functions in this module MUST be in a
projected CRS (e.g., UTM) where units are meters. Distance calculations assume
Euclidean geometry and will produce incorrect results if geometries are in a
geographic CRS (lat/lon degrees).

Projection should happen early in the pipeline (see runner.py). Since bare
Shapely geometries don't carry CRS information, validation must occur at the
caller level (e.g., in ml.py or rules.py).

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

"""

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import shapely as shapely_mod
from shapely import LineString, get_coordinates, line_interpolate_point, points
from shapely import distance as shapely_distance

from ._jit_helpers import (
    collinear_gap_ratio_numba,
    compute_angle_histogram_numba,
    compute_crossing_angle_stats_numba,
    compute_heading_consistency_numba,
    compute_heading_numba,
    compute_shape_complexity_numba,
    histogram_intersection_numba,
)

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


def compute_buffer_iou(line_a, line_b, radius: float) -> float:
    """Compute Intersection over Union of buffered geometries, using cache.

    This is the shared implementation used by both feature computation and
    integration filters/combiner. Uses get_cached_buffer() for efficiency
    when the same geometry appears in multiple candidate pairs.

    Args:
        line_a: First geometry (LineString)
        line_b: Second geometry (LineString)
        radius: Buffer radius in meters

    Returns:
        IoU value (0.0 to 1.0)
    """
    buf_a = get_cached_buffer(line_a, radius)
    buf_b = get_cached_buffer(line_b, radius)
    intersection_area = buf_a.intersection(buf_b).area
    union_area = buf_a.union(buf_b).area
    return intersection_area / union_area if union_area > 0 else 0.0


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

    IMPORTANT: Both geometries MUST be in a projected CRS (meters).
    If geometries are in a geographic CRS (degrees), all distance-based
    features will be incorrect. Ensure projection happens before calling.

    This is a thin wrapper around compute_geometric_features_batch() for
    single-pair callers. Non-batchable features (hausdorff stats, collinear
    gap) are computed per-pair after the batch call.

    Note: Callers typically pass aligned portions (via create_subline)
    or full geometries when coverage >= 99.5%. See compute.py for the
    selection logic (ref_geom_aligned / target_geom_aligned).

    Args:
        line_a: First geometry (LineString in projected CRS with meter units)
        line_b: Second geometry (LineString in projected CRS with meter units)

    Returns:
        GeometricFeatures tuple with distances in meters
    """
    # Batch path (size-1 arrays)
    arr_a = np.array([line_a], dtype=object)
    arr_b = np.array([line_b], dtype=object)

    batch = compute_geometric_features_batch(arr_a, arr_b)

    # Non-batchable per-pair ops
    coords_a = np.array(line_a.coords)
    coords_b = np.array(line_b.coords)
    mean_h, p95_h = _compute_hausdorff_stats(line_a, line_b, coords_a=coords_a, coords_b=coords_b)
    collinear = compute_collinear_gap_ratio(line_a, line_b, coords_a=coords_a, coords_b=coords_b)

    return GeometricFeatures(
        hausdorff_distance=float(batch.hausdorff_distances[0]),
        mean_hausdorff_distance=mean_h,
        hausdorff_p95_distance=p95_h,
        buffer_iou_5m=float(batch.buffer_iou_5m[0]),
        buffer_iou_15m=float(batch.buffer_iou_15m[0]),
        heading_delta=float(batch.heading_deltas[0]),
        overlap_ratio=float(batch.overlap_ratios[0]),
        collinear_gap_ratio=collinear,
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


def compute_buffer_iou_batch(
    buf_a_array: np.ndarray,
    buf_b_array: np.ndarray,
) -> np.ndarray:
    """Compute buffer IoU for arrays of buffer polygons using vectorized Shapely.

    Uses shapely.intersection and shapely.area on arrays to avoid per-element
    Python overhead. Typically 1.5-3x faster than looping _buffer_iou_from_buffers.

    Args:
        buf_a_array: Array of buffer polygons (dtype=object)
        buf_b_array: Array of buffer polygons (dtype=object)

    Returns:
        Array of IoU values (0-1)
    """
    import shapely as shapely_mod

    intersections = shapely_mod.intersection(buf_a_array, buf_b_array)
    int_areas = shapely_mod.area(intersections)
    a_areas = shapely_mod.area(buf_a_array)
    b_areas = shapely_mod.area(buf_b_array)
    union_areas = a_areas + b_areas - int_areas
    return np.where(union_areas > 0, int_areas / union_areas, 0.0)


class BatchGeometricResult(NamedTuple):
    """Results from batch geometric computation.

    All arrays are shape (N,) where N is the number of pairs.
    """

    hausdorff_distances: np.ndarray  # (N,) float64
    buffer_iou_15m: np.ndarray  # (N,) float64
    buffer_iou_5m: np.ndarray  # (N,) float64
    heading_deltas: np.ndarray  # (N,) float64
    overlap_ratios: np.ndarray  # (N,) float64
    lengths_a: np.ndarray  # (N,) float64 - needed downstream
    lengths_b: np.ndarray  # (N,) float64 - needed downstream


def compute_geometric_features_batch(
    lines_a: np.ndarray,
    lines_b: np.ndarray,
) -> BatchGeometricResult:
    """Compute batchable geometric features using vectorized Shapely 2.0 operations.

    This function performs all geometry operations that can be vectorized across
    arrays of LineStrings, avoiding per-pair Python dispatch overhead.

    Note: Callers typically pass aligned portions (via create_subline)
    or full geometries when coverage >= 99.5%. See compute.py for the
    selection logic (ref_geom_aligned / target_geom_aligned).

    Args:
        lines_a: Array of LineString geometries (N,), dtype=object
        lines_b: Array of LineString geometries (N,), dtype=object

    Returns:
        BatchGeometricResult with all batchable geometric features.
    """
    N = len(lines_a)

    # 1. Hausdorff distance - single vectorized call
    hausdorff_dists = shapely_mod.hausdorff_distance(lines_a, lines_b)

    # 2. Lengths - vectorized (needed for overlap_ratio denominator)
    lengths_a = shapely_mod.length(lines_a)
    lengths_b = shapely_mod.length(lines_b)

    # 3. Build 15m buffer arrays — vectorized
    bufs_a_15m = shapely_mod.buffer(lines_a, 15.0, quad_segs=16)
    bufs_b_15m = shapely_mod.buffer(lines_b, 15.0, quad_segs=16)

    # 4. Buffer IoU 15m - vectorized
    iou_15m = compute_buffer_iou_batch(bufs_a_15m, bufs_b_15m)

    # 5. Buffer IoU 5m with short-circuit: only process pairs where iou_15m > 0.3
    iou_5m = np.zeros(N, dtype=np.float64)
    qualifying_mask = iou_15m > 0.3
    if qualifying_mask.any():
        bufs_a_5m_q = shapely_mod.buffer(lines_a[qualifying_mask], 5.0, quad_segs=16)
        bufs_b_5m_q = shapely_mod.buffer(lines_b[qualifying_mask], 5.0, quad_segs=16)
        iou_5m[qualifying_mask] = compute_buffer_iou_batch(bufs_a_5m_q, bufs_b_5m_q)

    # 6. Overlap ratio: length(intersection(lines_a, bufs_b_15m)) / lengths_a
    intersections = shapely_mod.intersection(lines_a, bufs_b_15m)
    intersection_lengths = shapely_mod.length(intersections)
    overlap_ratios = np.where(lengths_a > 0, intersection_lengths / lengths_a, 0.0)

    # 7. Heading delta - vectorized via get_point + get_x/get_y + numpy arctan2
    heading_deltas = _compute_heading_deltas_batch(lines_a, lines_b)

    return BatchGeometricResult(
        hausdorff_distances=hausdorff_dists,
        buffer_iou_15m=iou_15m,
        buffer_iou_5m=iou_5m,
        heading_deltas=heading_deltas,
        overlap_ratios=overlap_ratios,
        lengths_a=lengths_a,
        lengths_b=lengths_b,
    )


def _compute_heading_deltas_batch(
    lines_a: np.ndarray,
    lines_b: np.ndarray,
) -> np.ndarray:
    """Compute heading deltas for arrays of LineStrings using vectorized Shapely.

    Uses shapely.get_point to extract first/last points, then numpy for
    arctan2 and angle difference computation.

    Args:
        lines_a: Array of LineString geometries (N,)
        lines_b: Array of LineString geometries (N,)

    Returns:
        Array of heading deltas in degrees (0-90), accounting for bidirectional roads.
    """
    # Get first and last points of each line
    start_a = shapely_mod.get_point(lines_a, 0)
    end_a = shapely_mod.get_point(lines_a, -1)
    start_b = shapely_mod.get_point(lines_b, 0)
    end_b = shapely_mod.get_point(lines_b, -1)

    # Extract x, y coordinates
    sx_a = shapely_mod.get_x(start_a)
    sy_a = shapely_mod.get_y(start_a)
    ex_a = shapely_mod.get_x(end_a)
    ey_a = shapely_mod.get_y(end_a)

    sx_b = shapely_mod.get_x(start_b)
    sy_b = shapely_mod.get_y(start_b)
    ex_b = shapely_mod.get_x(end_b)
    ey_b = shapely_mod.get_y(end_b)

    # Compute headings (0-360 degrees)
    heading_a = np.degrees(np.arctan2(ey_a - sy_a, ex_a - sx_a))
    heading_a = (heading_a + 360) % 360
    heading_b = np.degrees(np.arctan2(ey_b - sy_b, ex_b - sx_b))
    heading_b = (heading_b + 360) % 360

    # Angle difference (0-180)
    diff = np.abs(heading_a - heading_b)
    diff = np.where(diff > 180, 360 - diff, diff)

    # Account for bidirectional roads (0 and 180 both indicate alignment)
    opposite_diff = np.abs(180 - diff)
    return np.minimum(diff, opposite_diff)


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


def _compute_hausdorff_stats(
    line_a: LineString,
    line_b: LineString,
    *,
    coords_a: np.ndarray | None = None,
    coords_b: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute mean and P95 Hausdorff distances (from min distances).

    Standard Hausdorff uses max(min_distances), which is sensitive to
    segmentation/outliers.

    Uses Shapely's vectorized distance function for efficient computation
    of distances from all vertices to the opposite line.

    Args:
        line_a: First geometry (LineString)
        line_b: Second geometry (LineString)
        coords_a: Pre-extracted coordinates for line_a (optional, avoids redundant extraction)
        coords_b: Pre-extracted coordinates for line_b (optional, avoids redundant extraction)

    Returns:
        Tuple of (mean_distance, p95_distance)
    """
    if coords_a is None:
        coords_a = np.array(line_a.coords)
    if coords_b is None:
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


def compute_heading_consistency(
    line: LineString,
    sample_interval: float = 10.0,
    *,
    sampled_points: np.ndarray | None = None,
) -> float:
    """Compute how consistent the heading is along the line.

    Returns a value 0-1 where 1 means perfectly straight.

    For short segments (< 2 * sample_interval), uses the line's own vertices
    instead of interpolated samples. This avoids masking all short segments
    as "perfectly straight" (1.0) when they may actually have turns.

    Args:
        line: LineString geometry
        sample_interval: Distance between sample points (meters)
        sampled_points: Pre-extracted sampled points (optional, avoids redundant extraction)

    Returns:
        Consistency score (0-1) where 1 = perfectly straight
    """
    if sampled_points is not None:
        return compute_heading_consistency_numba(sampled_points)

    if line.length < sample_interval * 2:
        # Short segment: use actual vertices instead of skipping entirely.
        # Need >= 3 vertices to compute any heading change.
        coords = get_coordinates(line)
        if len(coords) < 3:
            return 1.0  # Genuinely straight (only 2 points = a line)
        return compute_heading_consistency_numba(coords)

    # Normal path: sample points along the line using vectorized Shapely interpolation
    n_samples = max(3, int(line.length / sample_interval))
    distances = np.linspace(0, line.length, n_samples)
    pts = line_interpolate_point(line, distances)
    sampled_points = get_coordinates(pts)

    return compute_heading_consistency_numba(sampled_points)


def compute_sinuosity(
    line: LineString,
    *,
    coords: np.ndarray | None = None,
) -> float:
    """Compute sinuosity of a line (ratio of path length to straight-line distance).

    Sinuosity = line_length / straight_distance
    - 1.0 = perfectly straight
    - >1.0 = increasingly curvy
    - Capped at 10.0 to prevent outliers from distorting XGBoost splits

    Args:
        line: LineString geometry
        coords: Pre-extracted coordinates (optional, avoids redundant extraction)

    Returns:
        Sinuosity ratio (1.0 to 10.0)
    """
    MAX_SINUOSITY = 10.0

    if line is None or line.is_empty:
        return 1.0

    line_length = line.length
    if line_length <= 0:
        return 1.0

    if coords is None:
        coords = np.array(line.coords)
    if len(coords) < 2:
        return 1.0

    # Compute straight-line (Euclidean) distance between endpoints
    start = coords[0]
    end = coords[-1]
    straight_distance = np.sqrt(np.sum((end - start) ** 2))

    # Handle loop case (start == end) — cap instead of returning extreme outlier.
    # Previously returned 100.0, which distorted sinuosity_delta for any pair
    # involving a loop/cul-de-sac.
    if straight_distance < 1e-9:
        return MAX_SINUOSITY

    return min(line_length / straight_distance, MAX_SINUOSITY)


def compute_vertex_density(
    line: LineString,
    *,
    coords: np.ndarray | None = None,
) -> float:
    """Compute vertex density of a line (vertices per meter).

    Higher density often indicates more detailed/higher-quality data.
    Lower density indicates simpler geometry.

    Args:
        line: LineString geometry
        coords: Pre-extracted coordinates (optional, avoids redundant extraction)

    Returns:
        Vertices per meter (>= 0.0)
    """
    if line is None or line.is_empty:
        return 0.0

    line_length = line.length
    if line_length <= 0:
        return 0.0

    # Use pre-extracted coords if provided, otherwise extract
    if coords is not None:
        n_vertices = len(coords)
    else:
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


def compute_shape_complexity(
    line: LineString,
    angle_threshold: float = 10.0,
    *,
    coords: np.ndarray | None = None,
) -> int:
    """Count significant direction changes (turns) in a line.

    A "significant turn" is where the heading changes by more than
    angle_threshold degrees between consecutive segments.

    Args:
        line: LineString geometry
        angle_threshold: Minimum angle change to count as a turn (degrees)
        coords: Pre-extracted coordinates (optional, avoids redundant extraction)

    Returns:
        Number of significant turns (>= 0)
    """
    if line is None or line.is_empty:
        return 0

    if coords is None:
        coords = np.array(line.coords)

    if len(coords) < 3:
        return 0

    return compute_shape_complexity_numba(coords, angle_threshold)


def compute_physical_overlap_m(
    ref_geom: LineString,
    target_geom: LineString,
    buffer_m: float = 5.0,
) -> float:
    """Compute actual geometric intersection length (no alignment translation).

    This measures the physical overlap between two segments without any
    sliding/translation. For segments that barely touch at tips, this
    will be ~0m even if alignment reports high coverage.

    Args:
        ref_geom: Reference geometry (projected CRS, meters)
        target_geom: Target geometry (projected CRS, meters)
        buffer_m: Buffer distance to handle GPS tolerance

    Returns:
        Length of intersection in meters
    """
    from shapely.errors import GEOSException

    try:
        intersection = ref_geom.intersection(target_geom.buffer(buffer_m))
        if intersection.is_empty:
            return 0.0
        if hasattr(intersection, "length"):
            return float(intersection.length)
        return 0.0  # Point intersection
    except GEOSException:
        # Topology errors from invalid geometries - treat as no overlap
        return 0.0


def compute_collinear_gap_ratio(
    line_a: LineString,
    line_b: LineString,
    heading_threshold: float = 15.0,
    *,
    coords_a: np.ndarray | None = None,
    coords_b: np.ndarray | None = None,
) -> float:
    """Compute along-track overlap fraction for collinear segments.

    This feature addresses the problem where consecutive road segments
    (same street, same direction, but end-to-end) score artificially high
    because name similarity and heading alignment are perfect.

    Algorithm:
    1. Check if segments are collinear (heading_delta < threshold)
    2. If not collinear → return 1.0 (let other features decide)
    3. Project both segments onto their common direction axis
    4. Return the raw overlap fraction (0-1) along that axis

    Args:
        line_a: First geometry (LineString, projected CRS)
        line_b: Second geometry (LineString, projected CRS)
        heading_threshold: Max heading difference to consider collinear (degrees)
        coords_a: Pre-extracted coordinates for line_a (optional, avoids redundant extraction)
        coords_b: Pre-extracted coordinates for line_b (optional, avoids redundant extraction)

    Returns:
        1.0 = not collinear or degenerate (no penalty)
        0.0-1.0 = collinear overlap fraction (0.0 = no overlap, 1.0 = full overlap)

    Example:
        Segment A: (0,0) → (100,0)
        Segment B: (100,0) → (200,0)  # tip-to-tip
        → Returns ~0.0 (no overlap)

        Segment A: (0,0) → (100,0)
        Segment B: (25,0) → (75,0)  # contained within
        → Returns 1.0 (full overlap)
    """
    # Handle degenerate cases
    if line_a.is_empty or line_b.is_empty:
        return 1.0
    if line_a.length <= 0 or line_b.length <= 0:
        return 1.0

    # Use pre-extracted coords if provided, otherwise extract
    if coords_a is None:
        coords_a = np.array(line_a.coords)
    if coords_b is None:
        coords_b = np.array(line_b.coords)

    return collinear_gap_ratio_numba(coords_a, coords_b, heading_threshold)


def compute_angle_histogram_similarity(
    line_a: LineString,
    line_b: LineString,
    *,
    coords_a: np.ndarray | None = None,
    coords_b: np.ndarray | None = None,
    n_bins: int = 8,
) -> float:
    """Compare angle histograms of two lines using histogram intersection.

    Creates a shape fingerprint for each line by binning turn angles at vertices
    into a histogram, then compares the distributions. This captures the overall
    shape signature - distinguishing curves from zigzags from straight segments.

    Based on Hootenanny's "Angle histogram from RoadMatcher" feature, which is
    a core RF classifier feature for road matching.

    Args:
        line_a: First geometry (LineString, projected CRS)
        line_b: Second geometry (LineString, projected CRS)
        coords_a: Pre-extracted coordinates for line_a (optional)
        coords_b: Pre-extracted coordinates for line_b (optional)
        n_bins: Number of histogram bins (default 8 = 22.5° per bin)

    Returns:
        Similarity score (0-1) where 1 = identical angle distributions.
        For 2-point (straight) lines: both straight → 1.0,
        one straight → compared against a straight-line histogram.
    """
    if line_a.is_empty or line_b.is_empty:
        return 1.0

    if coords_a is None:
        coords_a = np.array(line_a.coords)
    if coords_b is None:
        coords_b = np.array(line_b.coords)

    a_has_turns = len(coords_a) >= 3
    b_has_turns = len(coords_b) >= 3

    if not a_has_turns and not b_has_turns:
        return 1.0  # Both straight - identical shape profiles

    # Straight-line histogram: all weight at bin 0 (zero-degree turns)
    straight_hist = np.zeros(n_bins, dtype=np.float64)
    straight_hist[0] = 1.0

    if a_has_turns and b_has_turns:
        hist_a = compute_angle_histogram_numba(coords_a, n_bins)
        hist_b = compute_angle_histogram_numba(coords_b, n_bins)
    elif a_has_turns:
        hist_a = compute_angle_histogram_numba(coords_a, n_bins)
        hist_b = straight_hist
    else:
        hist_a = straight_hist
        hist_b = compute_angle_histogram_numba(coords_b, n_bins)

    return histogram_intersection_numba(hist_a, hist_b)


def compute_edge_distance_rmse(
    line_a: LineString,
    line_b: LineString,
    sample_interval: float = 5.0,
) -> float:
    """Compute bidirectional RMSE of distances at fixed sample intervals.

    Samples points at regular intervals along both lines, measures distances
    to the opposite line, and aggregates with RMSE. This is Hootenanny's
    primary classifier feature for road matching.

    Unlike mean_hausdorff_distance which samples at vertices:
    - This samples at fixed intervals for consistency across different
      vertex densities (different digitization granularities)
    - Uses RMSE aggregation which penalizes large deviations more than mean

    Reuses patterns from:
    - relational.py:176-183 (vectorized line_interpolate_point sampling)
    - geometric.py:509-517 (vectorized shapely_distance)

    Args:
        line_a: First geometry (LineString, projected CRS with meter units)
        line_b: Second geometry (LineString, projected CRS with meter units)
        sample_interval: Distance between sample points in meters (default 5.0)

    Returns:
        RMSE of all bidirectional distances in meters.
        Returns MAX_DISTANCE_METERS for empty/invalid geometries.
    """
    from ..config import MAX_DISTANCE_METERS

    if line_a.is_empty or line_b.is_empty:
        return MAX_DISTANCE_METERS

    len_a, len_b = line_a.length, line_b.length
    if len_a <= 0 or len_b <= 0:
        return MAX_DISTANCE_METERS

    # Sample points along each line (reuses relational.py pattern)
    n_samples_a = max(3, int(len_a / sample_interval))
    n_samples_b = max(3, int(len_b / sample_interval))

    distances_a = np.linspace(0, len_a, n_samples_a)
    distances_b = np.linspace(0, len_b, n_samples_b)

    # Vectorized sampling (reuses shapely pattern)
    points_a = line_interpolate_point(line_a, distances_a)
    points_b = line_interpolate_point(line_b, distances_b)

    # Vectorized distance (reuses _compute_hausdorff_stats pattern)
    dists_a_to_b = shapely_distance(points_a, line_b)
    dists_b_to_a = shapely_distance(points_b, line_a)

    # RMSE aggregation (different from mean in hausdorff)
    all_dists = np.concatenate([dists_a_to_b, dists_b_to_a])
    return float(np.sqrt(np.mean(all_dists**2)))


def _sample_local_headings(
    line: LineString,
    sample_interval: float = 10.0,
    min_samples: int = 3,
) -> np.ndarray:
    """Sample local tangent headings at regular intervals along a line.

    Uses vectorized Shapely line_interpolate_point for efficient sampling,
    then vectorized numpy arctan2 for heading computation.

    Args:
        line: LineString geometry (projected CRS, meters)
        sample_interval: Distance between sample points in meters
        min_samples: Minimum number of sample points

    Returns:
        Array of headings in degrees (0-360), shape (N,).
        Empty array if line is degenerate.
    """
    length = line.length
    if length <= 0:
        return np.array([], dtype=np.float64)

    n_samples = max(min_samples, int(length / sample_interval) + 1)
    # Use n_samples + 1 to get tangent vectors between consecutive points
    n_interp = max(n_samples + 1, 3)
    distances = np.linspace(0, length, n_interp)
    # Vectorized Shapely interpolation — single C call for all points
    pts = line_interpolate_point(line, distances)
    coords = get_coordinates(pts)

    # Vectorized heading computation via numpy
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])
    headings = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    return headings


def compute_crossing_angle_features(
    candidate: LineString,
    nearby_geometries: list[LineString],
    nearby_classes: list[str | None],
    candidate_class: str | None,
    sample_interval: float = 10.0,
    transverse_threshold: float = 60.0,
) -> dict[str, float]:
    """Compute crossing-angle features for a candidate segment vs nearby corridor.

    Measures how transverse the candidate is relative to nearby segments of a
    different traffic tier. Designed to detect ACROSS-role segments (crosswalks,
    bike crossings) which cross a linear facility at a high angle.

    Algorithm:
    1. Filter nearby segments to those in a different traffic tier
    2. For each different-tier neighbor, compute its gross heading (vectorized)
    3. Sample local headings along the candidate using vectorized Shapely
    4. Delegate to numba JIT for the O(samples x neighbors) angle computation
    5. Report statistics: min, mean, std, transverse_neighbor_fraction

    Performance:
    - Shapely vectorized for point sampling (line_interpolate_point on arrays)
    - Numpy vectorized for heading computation (arctan2 on arrays)
    - Numba JIT for the O(N*M) angle matrix (compute_crossing_angle_stats_numba)
    - Neighbor heading extraction uses vectorized Shapely get_point/get_x/get_y

    Args:
        candidate: The segment being evaluated (projected CRS, meters)
        nearby_geometries: Geometries of nearby segments (from spatial query)
        nearby_classes: Road classes of nearby segments (parallel to geometries)
        candidate_class: Road class of the candidate segment
        sample_interval: Distance between heading sample points (meters)
        transverse_threshold: Angle threshold for "transverse" classification (degrees)

    Returns:
        Dict with keys:
        - crossing_angle_min: Minimum angle to nearest different-tier segment (0-90)
        - crossing_angle_mean: Mean of per-sample minimum angles (0-90)
        - crossing_angle_std: Std dev of per-sample minimum angles (0-45ish)
        - transverse_neighbor_fraction: Fraction of different-tier neighbors
          whose gross heading is >transverse_threshold from candidate (0-1)
    """
    from .semantic import get_traffic_tier

    neutral = {
        "crossing_angle_min": float("nan"),
        "crossing_angle_mean": float("nan"),
        "crossing_angle_std": float("nan"),
        "transverse_neighbor_fraction": float("nan"),
    }

    if candidate.is_empty or candidate.length <= 0:
        return neutral

    candidate_tier = get_traffic_tier(candidate_class)

    # Filter to different-tier neighbors and extract headings
    # Use vectorized Shapely for endpoint extraction where possible
    diff_tier_geoms = []
    for geom, cls in zip(nearby_geometries, nearby_classes):
        if geom is None or geom.is_empty or geom.length <= 0:
            continue
        neighbor_tier = get_traffic_tier(cls)
        if neighbor_tier is None or candidate_tier is None:
            continue
        if neighbor_tier == candidate_tier:
            continue
        if neighbor_tier == "neutral" or candidate_tier == "neutral":
            continue
        diff_tier_geoms.append(geom)

    if not diff_tier_geoms:
        return neutral

    # Vectorized heading extraction for filtered neighbors
    geom_array = np.array(diff_tier_geoms, dtype=object)
    starts = shapely_mod.get_point(geom_array, 0)
    ends = shapely_mod.get_point(geom_array, -1)
    dx = shapely_mod.get_x(ends) - shapely_mod.get_x(starts)
    dy = shapely_mod.get_y(ends) - shapely_mod.get_y(starts)
    neighbor_headings = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    # Sample local headings along the candidate (vectorized Shapely + numpy)
    candidate_headings = _sample_local_headings(candidate, sample_interval)
    if len(candidate_headings) == 0:
        return neutral

    # Candidate gross heading for transverse fraction
    candidate_coords = np.array(candidate.coords)
    candidate_gross_heading = float(
        compute_heading_numba(
            candidate_coords[-1, 0] - candidate_coords[0, 0],
            candidate_coords[-1, 1] - candidate_coords[0, 1],
        )
    )

    # Delegate to numba JIT for the O(N*M) angle computation
    crossing_min, crossing_mean, crossing_std, transverse_frac = compute_crossing_angle_stats_numba(
        candidate_headings,
        neighbor_headings,
        transverse_threshold,
        candidate_gross_heading,
    )

    return {
        "crossing_angle_min": float(crossing_min),
        "crossing_angle_mean": float(crossing_mean),
        "crossing_angle_std": float(crossing_std),
        "transverse_neighbor_fraction": float(transverse_frac),
    }
