"""Numba JIT-compiled helper functions for feature computation.

These functions are extracted from pure-Python implementations to enable
Numba's nopython mode compilation for significant performance improvements.

When coords are pre-extracted (avoiding repeated np.array(line.coords) calls),
JIT provides consistent performance for batch processing scenarios.
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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
