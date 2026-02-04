"""Numba JIT-compiled helper functions for feature computation.

These functions are extracted from pure-Python implementations to enable
Numba's nopython mode compilation for significant performance improvements.

When coords are pre-extracted (avoiding repeated np.array(line.coords) calls),
JIT provides consistent performance for batch processing scenarios.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def compute_heading_numba(dx: float, dy: float) -> float:
    """Compute heading in degrees (0-360) from delta x/y.

    Args:
        dx: Delta x (end_x - start_x)
        dy: Delta y (end_y - start_y)

    Returns:
        Heading in degrees from 0 to 360.
    """
    heading = np.degrees(np.arctan2(dy, dx))
    return (heading + 360.0) % 360.0


@njit(cache=True)
def angle_diff_numba(a: float, b: float) -> float:
    """Compute minimum angle difference in degrees (0-90).

    Handles the fact that roads can be traversed in either direction,
    so 0° and 180° are considered equivalent directions.

    Args:
        a: First heading in degrees (0-360)
        b: Second heading in degrees (0-360)

    Returns:
        Minimum angle difference from 0 to 90 degrees.
    """
    diff = abs(a - b)
    if diff > 180.0:
        diff = 360.0 - diff

    # Consider opposite direction (road could be traversed either way)
    opposite_diff = abs(180.0 - diff)

    return min(diff, opposite_diff)


@njit(cache=True)
def collinear_gap_ratio_numba(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    heading_threshold: float,
    min_overlap_fraction: float,
) -> float:
    """Compute collinear gap ratio using JIT-compiled code.

    This is the complete JIT implementation that handles both the heading
    check and overlap computation. Expects pre-extracted coordinates.

    Args:
        coords_a: Nx2 array of coordinates for line A
        coords_b: Mx2 array of coordinates for line B
        heading_threshold: Max heading difference to consider collinear (degrees)
        min_overlap_fraction: Minimum overlap to not penalize (fraction 0-1)

    Returns:
        Gap ratio from 0.0 (collinear with poor overlap) to 1.0 (no penalty).
    """
    # Compute headings
    dx_a = coords_a[-1, 0] - coords_a[0, 0]
    dy_a = coords_a[-1, 1] - coords_a[0, 1]
    heading_a = compute_heading_numba(dx_a, dy_a)

    dx_b = coords_b[-1, 0] - coords_b[0, 0]
    dy_b = coords_b[-1, 1] - coords_b[0, 1]
    heading_b = compute_heading_numba(dx_b, dy_b)

    # Check collinearity
    heading_diff = angle_diff_numba(heading_a, heading_b)
    if heading_diff > heading_threshold:
        return 1.0

    # Handle bidirectional case: if headings differ by ~180°, align them first
    heading_diff_raw = abs(heading_a - heading_b)
    if heading_diff_raw > 90.0 and heading_diff_raw < 270.0:
        heading_b_aligned = (heading_b + 180.0) % 360.0
    else:
        heading_b_aligned = heading_b

    avg_heading = (heading_a + heading_b_aligned) / 2.0
    ref_angle = np.radians(avg_heading)
    ref_dir_x = np.cos(ref_angle)
    ref_dir_y = np.sin(ref_angle)

    # Project all endpoints onto the reference direction
    a_start_proj = coords_a[0, 0] * ref_dir_x + coords_a[0, 1] * ref_dir_y
    a_end_proj = coords_a[-1, 0] * ref_dir_x + coords_a[-1, 1] * ref_dir_y
    b_start_proj = coords_b[0, 0] * ref_dir_x + coords_b[0, 1] * ref_dir_y
    b_end_proj = coords_b[-1, 0] * ref_dir_x + coords_b[-1, 1] * ref_dir_y

    # Compute 1D overlap
    a_min = min(a_start_proj, a_end_proj)
    a_max = max(a_start_proj, a_end_proj)
    b_min = min(b_start_proj, b_end_proj)
    b_max = max(b_start_proj, b_end_proj)

    overlap_start = max(a_min, b_min)
    overlap_end = min(a_max, b_max)
    overlap_length = max(0.0, overlap_end - overlap_start)

    # Use the smaller segment's extent as the denominator
    smaller_extent = min(a_max - a_min, b_max - b_min)
    if smaller_extent <= 0.0:
        return 1.0

    along_track_overlap = overlap_length / smaller_extent

    if along_track_overlap >= min_overlap_fraction:
        return 1.0

    return along_track_overlap / min_overlap_fraction


@njit(cache=True)
def compute_shape_complexity_numba(
    coords: np.ndarray,
    angle_threshold: float,
) -> int:
    """Count significant direction changes (turns) in a line.

    Args:
        coords: Nx2 array of coordinates
        angle_threshold: Minimum angle change to count as a turn (degrees)

    Returns:
        Number of significant turns
    """
    n_points = coords.shape[0]
    if n_points < 3:
        return 0

    turn_count = 0
    for i in range(n_points - 2):
        # Heading from point i to i+1
        dx1 = coords[i + 1, 0] - coords[i, 0]
        dy1 = coords[i + 1, 1] - coords[i, 1]
        heading1 = compute_heading_numba(dx1, dy1)

        # Heading from point i+1 to i+2
        dx2 = coords[i + 2, 0] - coords[i + 1, 0]
        dy2 = coords[i + 2, 1] - coords[i + 1, 1]
        heading2 = compute_heading_numba(dx2, dy2)

        # Compute angle difference (not bidirectional - actual turn)
        angle_diff = abs(heading1 - heading2)
        if angle_diff > 180.0:
            angle_diff = 360.0 - angle_diff

        if angle_diff > angle_threshold:
            turn_count += 1

    return turn_count


@njit(cache=True)
def compute_parallel_alignment_numba(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> float:
    """Compute how parallel two lines are (0-1).

    Args:
        coords_a: Nx2 array for line A
        coords_b: Mx2 array for line B

    Returns:
        Alignment score (0-1) where 1 = parallel
    """
    # Compute overall headings
    dx_a = coords_a[-1, 0] - coords_a[0, 0]
    dy_a = coords_a[-1, 1] - coords_a[0, 1]
    heading_a = compute_heading_numba(dx_a, dy_a)

    dx_b = coords_b[-1, 0] - coords_b[0, 0]
    dy_b = coords_b[-1, 1] - coords_b[0, 1]
    heading_b = compute_heading_numba(dx_b, dy_b)

    # Use angle_diff_numba which handles bidirectional
    min_diff = angle_diff_numba(heading_a, heading_b)

    # Convert to 0-1 score (0 degrees = 1.0, 90 degrees = 0.0)
    return max(0.0, 1.0 - min_diff / 90.0)


@njit(cache=True)
def compute_endpoint_proximity_numba(
    start: np.ndarray,
    end: np.ndarray,
    endpoint_coords: np.ndarray,
    tolerance: float,
) -> tuple:
    """Compute endpoint proximity features.

    Args:
        start: 2D point (start of target)
        end: 2D point (end of target)
        endpoint_coords: Nx2 array of other endpoint coordinates
        tolerance: Distance threshold for counting "shared" endpoints

    Returns:
        Tuple of (start_proximity, end_proximity, shared_count)
    """
    n = endpoint_coords.shape[0]
    if n == 0:
        return (np.inf, np.inf, 0)

    start_min = np.inf
    end_min = np.inf
    within_start = 0
    within_end = 0

    for i in range(n):
        # Distance from start to this endpoint
        dx_s = endpoint_coords[i, 0] - start[0]
        dy_s = endpoint_coords[i, 1] - start[1]
        dist_s = np.sqrt(dx_s * dx_s + dy_s * dy_s)

        if dist_s < start_min:
            start_min = dist_s
        if dist_s <= tolerance:
            within_start += 1

        # Distance from end to this endpoint
        dx_e = endpoint_coords[i, 0] - end[0]
        dy_e = endpoint_coords[i, 1] - end[1]
        dist_e = np.sqrt(dx_e * dx_e + dy_e * dy_e)

        if dist_e < end_min:
            end_min = dist_e
        if dist_e <= tolerance:
            within_end += 1

    return (start_min, end_min, within_start + within_end)


@njit(cache=True)
def query_nearby_endpoints_numba(
    endpoint_coords: np.ndarray,
    candidate_indices: np.ndarray,
    point_coords: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """JIT-compiled endpoint distance filtering.

    Filters candidate endpoints by Euclidean distance from query point.
    Much faster than Python loop with np.linalg.norm calls.

    Args:
        endpoint_coords: (N, 2) array of all endpoint coordinates
        candidate_indices: Array of candidate indices to check (from spatial index)
        point_coords: (2,) query point coordinates
        radius: Maximum distance threshold

    Returns:
        Tuple of (result_indices, result_distances) for endpoints within radius
    """
    n_candidates = len(candidate_indices)
    result_indices = np.empty(n_candidates, dtype=np.int64)
    result_dists = np.empty(n_candidates, dtype=np.float64)
    count = 0

    for i in range(n_candidates):
        ep_idx = candidate_indices[i]
        dx = endpoint_coords[ep_idx, 0] - point_coords[0]
        dy = endpoint_coords[ep_idx, 1] - point_coords[1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist <= radius:
            result_indices[count] = ep_idx
            result_dists[count] = dist
            count += 1

    return result_indices[:count], result_dists[:count]


@njit(cache=True)
def compute_heading_consistency_numba(
    points: np.ndarray,
) -> float:
    """Compute heading consistency from sampled points.

    Args:
        points: Nx2 array of sampled points along the line

    Returns:
        Consistency score (0-1) where 1 = perfectly straight
    """
    n_points = points.shape[0]
    if n_points < 3:
        return 1.0

    # Compute headings between consecutive points
    n_headings = n_points - 1
    headings = np.empty(n_headings)

    for i in range(n_headings):
        dx = points[i + 1, 0] - points[i, 0]
        dy = points[i + 1, 1] - points[i, 1]
        headings[i] = compute_heading_numba(dx, dy)

    if n_headings < 2:
        return 1.0

    # Compute heading differences
    total_diff = 0.0
    for i in range(n_headings - 1):
        diff = angle_diff_numba(headings[i], headings[i + 1])
        total_diff += diff

    avg_diff = total_diff / (n_headings - 1)

    # Normalize to 0-1 (0 degrees diff = 1.0, 90 degrees diff = 0.0)
    return max(0.0, 1.0 - avg_diff / 90.0)


@njit(cache=True)
def compute_angle_histogram_numba(coords: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Compute normalized histogram of turn angles at vertices.

    Creates a shape fingerprint by binning turn angles into a histogram.
    This captures the distribution of direction changes along a line,
    distinguishing curves from zigzags and straight segments.

    Similar pattern to: compute_shape_complexity_numba (lines 130-167)

    Args:
        coords: Nx2 array of coordinates
        n_bins: Number of histogram bins (default 8 = 22.5° per bin over 0-180°)

    Returns:
        Normalized histogram array of shape (n_bins,), sums to 1.0
        Returns zeros if fewer than 3 points.
    """
    n_points = coords.shape[0]
    if n_points < 3:
        return np.zeros(n_bins, dtype=np.float64)

    histogram = np.zeros(n_bins, dtype=np.float64)
    bin_width = 180.0 / n_bins  # Turn angles range 0-180

    for i in range(n_points - 2):
        # Heading from point i to i+1 (reuses compute_heading_numba pattern)
        dx1 = coords[i + 1, 0] - coords[i, 0]
        dy1 = coords[i + 1, 1] - coords[i, 1]
        heading1 = compute_heading_numba(dx1, dy1)

        # Heading from point i+1 to i+2
        dx2 = coords[i + 2, 0] - coords[i + 1, 0]
        dy2 = coords[i + 2, 1] - coords[i + 1, 1]
        heading2 = compute_heading_numba(dx2, dy2)

        # Turn angle (0-180) - actual direction change, not bidirectional
        turn = abs(heading2 - heading1)
        if turn > 180.0:
            turn = 360.0 - turn

        bin_idx = min(int(turn / bin_width), n_bins - 1)
        histogram[bin_idx] += 1.0

    # Normalize to sum to 1.0
    total = histogram.sum()
    if total > 0:
        histogram /= total
    return histogram


@njit(cache=True)
def histogram_intersection_numba(h1: np.ndarray, h2: np.ndarray) -> float:
    """Compute histogram intersection similarity (0-1).

    Histogram intersection is the sum of element-wise minimums.
    For normalized histograms (sum=1), the result is in [0, 1]
    where 1 means identical distributions.

    Args:
        h1: First normalized histogram
        h2: Second normalized histogram

    Returns:
        Intersection similarity score (0-1)
    """
    return np.minimum(h1, h2).sum()


@njit(cache=True)
def compute_local_parallel_alignment_numba(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    n_samples: int = 16,
    parallel_threshold: float = 0.7,
) -> tuple[float, float]:
    """Compute local alignment by sampling both lines and comparing headings.

    Instead of using overall heading (first to last point), this function:
    1. Samples both lines at n_samples intervals
    2. Computes local heading at each sample (direction to next sample)
    3. Returns both mean alignment and fraction of samples that are parallel

    This handles curved segments and segments that are only partially parallel
    (e.g., split carriageways that diverge/converge at endpoints).

    Args:
        coords_a: Nx2 array of coordinates for line A
        coords_b: Mx2 array of coordinates for line B
        n_samples: Number of sample points (default 16)
        parallel_threshold: Alignment threshold to count as "parallel" (default 0.7)

    Returns:
        Tuple of (mean_alignment, parallel_fraction) where:
        - mean_alignment: Average alignment across all samples (0-1)
        - parallel_fraction: Fraction of samples with alignment > threshold (0-1)
    """
    # Handle degenerate cases
    if coords_a.shape[0] < 2 or coords_b.shape[0] < 2:
        return 0.0, 0.0

    # Get total length of each line for sampling
    len_a = 0.0
    for i in range(coords_a.shape[0] - 1):
        dx = coords_a[i + 1, 0] - coords_a[i, 0]
        dy = coords_a[i + 1, 1] - coords_a[i, 1]
        len_a += np.sqrt(dx * dx + dy * dy)

    len_b = 0.0
    for i in range(coords_b.shape[0] - 1):
        dx = coords_b[i + 1, 0] - coords_b[i, 0]
        dy = coords_b[i + 1, 1] - coords_b[i, 1]
        len_b += np.sqrt(dx * dx + dy * dy)

    if len_a < 1e-10 or len_b < 1e-10:
        return 0.0, 0.0

    # Sample points along each line at regular intervals
    # We need n_samples-1 segments to compare headings
    sample_distances_a = np.linspace(0, len_a, n_samples)
    sample_distances_b = np.linspace(0, len_b, n_samples)

    # Interpolate points along line A
    points_a = np.empty((n_samples, 2))
    cumulative_dist = 0.0
    seg_idx = 0
    for i in range(n_samples):
        target_dist = sample_distances_a[i]
        # Walk along segments to find the right position
        while seg_idx < coords_a.shape[0] - 1:
            dx = coords_a[seg_idx + 1, 0] - coords_a[seg_idx, 0]
            dy = coords_a[seg_idx + 1, 1] - coords_a[seg_idx, 1]
            seg_len = np.sqrt(dx * dx + dy * dy)
            if cumulative_dist + seg_len >= target_dist or seg_idx == coords_a.shape[0] - 2:
                # Interpolate within this segment
                if seg_len > 1e-10:
                    t = (target_dist - cumulative_dist) / seg_len
                    t = max(0.0, min(1.0, t))
                else:
                    t = 0.0
                points_a[i, 0] = coords_a[seg_idx, 0] + t * dx
                points_a[i, 1] = coords_a[seg_idx, 1] + t * dy
                break
            cumulative_dist += seg_len
            seg_idx += 1

    # Interpolate points along line B
    points_b = np.empty((n_samples, 2))
    cumulative_dist = 0.0
    seg_idx = 0
    for i in range(n_samples):
        target_dist = sample_distances_b[i]
        while seg_idx < coords_b.shape[0] - 1:
            dx = coords_b[seg_idx + 1, 0] - coords_b[seg_idx, 0]
            dy = coords_b[seg_idx + 1, 1] - coords_b[seg_idx, 1]
            seg_len = np.sqrt(dx * dx + dy * dy)
            if cumulative_dist + seg_len >= target_dist or seg_idx == coords_b.shape[0] - 2:
                if seg_len > 1e-10:
                    t = (target_dist - cumulative_dist) / seg_len
                    t = max(0.0, min(1.0, t))
                else:
                    t = 0.0
                points_b[i, 0] = coords_b[seg_idx, 0] + t * dx
                points_b[i, 1] = coords_b[seg_idx, 1] + t * dy
                break
            cumulative_dist += seg_len
            seg_idx += 1

    # Compute local headings between consecutive samples and compare
    total_alignment = 0.0
    parallel_count = 0
    n_comparisons = n_samples - 1

    for i in range(n_comparisons):
        # Local heading for line A
        dx_a = points_a[i + 1, 0] - points_a[i, 0]
        dy_a = points_a[i + 1, 1] - points_a[i, 1]
        if abs(dx_a) < 1e-10 and abs(dy_a) < 1e-10:
            continue
        heading_a = compute_heading_numba(dx_a, dy_a)

        # Local heading for line B
        dx_b = points_b[i + 1, 0] - points_b[i, 0]
        dy_b = points_b[i + 1, 1] - points_b[i, 1]
        if abs(dx_b) < 1e-10 and abs(dy_b) < 1e-10:
            continue
        heading_b = compute_heading_numba(dx_b, dy_b)

        # Compute alignment (handles bidirectional)
        min_diff = angle_diff_numba(heading_a, heading_b)
        alignment = max(0.0, 1.0 - min_diff / 90.0)

        total_alignment += alignment
        if alignment >= parallel_threshold:
            parallel_count += 1

    if n_comparisons == 0:
        return 0.0, 0.0

    mean_alignment = total_alignment / n_comparisons
    parallel_fraction = float(parallel_count) / n_comparisons

    return mean_alignment, parallel_fraction
