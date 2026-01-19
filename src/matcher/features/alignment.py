"""
Linestring alignment for finding where two road geometries overlap.
Ported from overture_udfs with Numba JIT compilation for performance.

Key functions:
- linestring_alignment: Find where two lines overlap, returns fractional positions
- create_subline: Extract a portion of a linestring given start/end fractions
- walk_distance: Integrated Euclidean distance between two aligned lines
- walk_parallelness: How parallel two aligned lines are (squared dot product)
- compute_alignment_batch: Parallel batch processing for multiple pairs
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numba import jit
from shapely.geometry import LineString
from shapely.ops import substring


@dataclass
class AlignmentResult:
    """Result of aligning two linestrings."""

    overture_start_frac: float  # Where alignment starts on reference (0-1)
    overture_end_frac: float  # Where alignment ends on reference (0-1)
    dataset_start_frac: float  # Where alignment starts on target (0-1)
    dataset_end_frac: float  # Where alignment ends on target (0-1)

    @property
    def overture_coverage(self) -> float:
        """Fraction of reference line that is covered by alignment."""
        return self.overture_end_frac - self.overture_start_frac

    @property
    def dataset_coverage(self) -> float:
        """Fraction of target line that is covered by alignment."""
        return self.dataset_end_frac - self.dataset_start_frac


@jit(nopython=True, cache=True)
def _interpolate_along_line(
    coords: np.ndarray, distances: np.ndarray, t: float
) -> tuple[float, float]:
    """
    Interpolate a point at distance t along the line defined by coords.
    coords: Nx2 array of (x, y) coordinates
    distances: cumulative distances along the line (length N)
    t: distance along line to interpolate
    Returns: (x, y) tuple
    """
    if t <= 0:
        return coords[0, 0], coords[0, 1]
    if t >= distances[-1]:
        return coords[-1, 0], coords[-1, 1]

    # Binary search for the segment containing t
    idx = np.searchsorted(distances, t)
    if idx == 0:
        idx = 1

    # Linear interpolation within segment
    d0 = distances[idx - 1]
    d1 = distances[idx]
    frac = (t - d0) / (d1 - d0) if d1 > d0 else 0.0

    x = coords[idx - 1, 0] + frac * (coords[idx, 0] - coords[idx - 1, 0])
    y = coords[idx - 1, 1] + frac * (coords[idx, 1] - coords[idx - 1, 1])
    return x, y


@jit(nopython=True, cache=True)
def _get_score_numba(
    overture_coords: np.ndarray,
    overture_distances: np.ndarray,
    overture_length: float,
    dataset_coords: np.ndarray,
    dataset_distances: np.ndarray,
    dataset_length: float,
    dx: float,
    buffer_distance: float,
    num_samples: int = 16,
) -> float:
    """
    Numba-optimized scoring function for linestring alignment.
    """
    # Determine the overlapping region
    comparison_start = max(0.0, dx)
    comparison_end = min(overture_length, dataset_length + dx)
    comparison_length = comparison_end - comparison_start

    # Require minimum 1% overlap to avoid floating point precision issues
    min_overlap = 0.01 * min(overture_length, dataset_length)
    if comparison_length <= min_overlap:
        return 0.0

    sqsum = 0.0
    pa_x, pa_y = 0.0, 0.0
    pb_x, pb_y = 0.0, 0.0
    has_prev = False

    # Ensure at least 2 samples to avoid division by zero
    if num_samples < 2:
        num_samples = 2

    # Sample points along the overlapping portion
    for i in range(num_samples):
        t = comparison_start + (comparison_end - comparison_start) * i / (num_samples - 1)

        # Interpolate points
        a_x, a_y = _interpolate_along_line(overture_coords, overture_distances, t)
        b_x, b_y = _interpolate_along_line(dataset_coords, dataset_distances, t - dx)

        # Facing-the-same-way-ness check (dot product)
        dot2 = 1.0
        if has_prev:
            # Vector from previous to current for both lines
            va_x = a_x - pa_x
            va_y = a_y - pa_y
            vb_x = b_x - pb_x
            vb_y = b_y - pb_y

            va_norm = np.sqrt(va_x * va_x + va_y * va_y)
            vb_norm = np.sqrt(vb_x * vb_x + vb_y * vb_y)

            if va_norm > 1e-9 and vb_norm > 1e-9:
                # Normalize
                va_x /= va_norm
                va_y /= va_norm
                vb_x /= vb_norm
                vb_y /= vb_norm
                # Dot product
                dot = va_x * vb_x + va_y * vb_y
                dot2 = max(0.0, dot)

        # Distance between interpolated points
        point_distance = np.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)
        sqsum += dot2 / (buffer_distance + point_distance)

        pa_x, pa_y = a_x, a_y
        pb_x, pb_y = b_x, b_y
        has_prev = True

    return sqsum


@jit(nopython=True, cache=True)
def _find_best_alignment_numba(
    overture_coords: np.ndarray,
    overture_distances: np.ndarray,
    overture_length: float,
    dataset_coords: np.ndarray,
    dataset_distances: np.ndarray,
    dataset_length: float,
    grid_samples: int,
    refinement_steps: int,
) -> tuple[float, float]:
    """
    Numba-optimized grid search + ternary refinement for best alignment.
    Returns: (best_offset, best_score)
    """
    if overture_length == 0 or dataset_length == 0:
        return 0.0, 0.0

    # Ensure at least 2 grid samples to avoid division by zero
    if grid_samples < 2:
        grid_samples = 2

    # Grid search range
    lower = -dataset_length
    upper = overture_length
    buffer_distance = 0.5 * min(overture_length, dataset_length) / grid_samples

    best_score = 0.0
    best_offset = 0.0

    # Grid search
    for i in range(grid_samples):
        x = lower + (upper - lower) * i / (grid_samples - 1)
        score = _get_score_numba(
            overture_coords,
            overture_distances,
            overture_length,
            dataset_coords,
            dataset_distances,
            dataset_length,
            x,
            buffer_distance,
        )
        if score > best_score:
            best_score = score
            best_offset = x

    # Ternary search refinement
    dx = (upper - lower) / grid_samples / 2.0
    for _ in range(refinement_steps):
        if dx < 1e-6:
            break

        offset_left = best_offset - dx
        offset_right = best_offset + dx

        score_left = _get_score_numba(
            overture_coords,
            overture_distances,
            overture_length,
            dataset_coords,
            dataset_distances,
            dataset_length,
            offset_left,
            buffer_distance,
        )
        score_right = _get_score_numba(
            overture_coords,
            overture_distances,
            overture_length,
            dataset_coords,
            dataset_distances,
            dataset_length,
            offset_right,
            buffer_distance,
        )

        if score_left > best_score and score_left > score_right:
            best_offset = offset_left
            best_score = score_left
        elif score_right > best_score:
            best_offset = offset_right
            best_score = score_right

        dx /= 3.0

    return best_offset, best_score


def _prepare_line_data(line: LineString) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extract coordinates and compute cumulative distances for a LineString.
    Returns: (coords array, distances array, total_length)
    """
    coords = np.array(line.coords)
    # Compute segment distances
    diffs = np.diff(coords, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    # Cumulative distances (starting from 0)
    distances = np.zeros(len(coords))
    distances[1:] = np.cumsum(segment_lengths)
    total_length = distances[-1]
    return coords, distances, total_length


def linestring_alignment(
    reference: LineString,
    target: LineString,
    grid_samples: int = 16,
    refinement_steps: int = 8,
) -> AlignmentResult:
    """
    Calculates the best alignment between two LineStrings.

    Compares reference with target, and reference with a reversed target,
    to find the best possible match.

    NOTE: The ternary search refinement assumes the score function is unimodal
    (has a single peak). Complex geometries like switchbacks might have multiple
    local optima. The initial grid search (16 samples) mitigates this by finding
    the right region first, but edge cases may be slightly suboptimal.
    Consider multi-start optimization if needed for complex road geometries.

    Args:
        reference: Reference LineString (e.g., Overture segment)
        target: Target LineString (e.g., local road segment)
        grid_samples: Number of samples for initial grid search
        refinement_steps: Number of ternary search refinement steps

    Returns:
        AlignmentResult with fractional start/end positions on each line
    """
    # Prepare line data once
    ref_coords, ref_distances, ref_length = _prepare_line_data(reference)
    target_coords, target_distances, target_length = _prepare_line_data(target)

    if ref_length == 0 or target_length == 0:
        return AlignmentResult(0.0, 1.0, 0.0, 1.0)

    # Compare normally (forward)
    forward_offset, forward_score = _find_best_alignment_numba(
        ref_coords,
        ref_distances,
        ref_length,
        target_coords,
        target_distances,
        target_length,
        grid_samples,
        refinement_steps,
    )

    # Compare with the second linestring reversed
    target_coords_rev = target_coords[::-1].copy()
    target_distances_rev = target_length - target_distances[::-1]

    backward_offset, backward_score = _find_best_alignment_numba(
        ref_coords,
        ref_distances,
        ref_length,
        target_coords_rev,
        target_distances_rev,
        target_length,
        grid_samples,
        refinement_steps,
    )

    def unit_clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    # Choose the best alignment (forward or backward)
    is_forward = forward_score >= backward_score
    offset = forward_offset if is_forward else backward_offset

    # Calculate the fractional start/end of the alignment on reference
    ref_start_frac = float(max(offset, 0) / ref_length)
    ref_end_frac = float(min(offset + target_length, ref_length) / ref_length)

    # Calculate the fractional start/end of the alignment on target
    target_start_frac = float(max(-offset, 0) / target_length)
    target_end_frac = float(min(-offset + ref_length, target_length) / target_length)

    if is_forward:
        return AlignmentResult(
            overture_start_frac=unit_clamp(ref_start_frac),
            overture_end_frac=unit_clamp(ref_end_frac),
            dataset_start_frac=unit_clamp(target_start_frac),
            dataset_end_frac=unit_clamp(target_end_frac),
        )
    else:
        # If the backward alignment was better, the target fractions must be flipped
        return AlignmentResult(
            overture_start_frac=unit_clamp(ref_start_frac),
            overture_end_frac=unit_clamp(ref_end_frac),
            dataset_start_frac=unit_clamp(1.0 - target_end_frac),
            dataset_end_frac=unit_clamp(1.0 - target_start_frac),
        )


def create_subline(line: LineString, start_frac: float, end_frac: float) -> LineString | None:
    """
    Extracts a sub-linestring from a LineString given start and end fractions.

    Args:
        line: The input Shapely LineString.
        start_frac: The normalized start distance (0.0 to 1.0).
        end_frac: The normalized end distance (0.0 to 1.0).

    Returns:
        A new LineString representing the segment, or None if the input is invalid.
    """
    if line is None or line.is_empty or line.length == 0:
        return None

    # Clamp fractions to valid range
    start_frac = max(0.0, min(1.0, start_frac))
    end_frac = max(0.0, min(1.0, end_frac))

    # Ensure start fraction is less than end fraction
    if start_frac > end_frac:
        start_frac, end_frac = end_frac, start_frac

    # Calculate absolute distances along the line
    start_dist = line.length * start_frac
    end_dist = line.length * end_frac

    return substring(line, start_dist, end_dist)


@jit(nopython=True, cache=True)
def _walk_distance_numba(
    L1_coords: np.ndarray,
    L1_distances: np.ndarray,
    L1_length: float,
    L2_coords: np.ndarray,
    L2_distances: np.ndarray,
    L2_length: float,
    samples: int = 16,
) -> float:
    """Numba-optimized walk distance calculation."""
    # Ensure at least 2 samples to avoid division by zero
    if samples < 2:
        samples = 2
    cum_distance = 0.0
    for i in range(samples):
        x = i / (samples - 1)
        posA_x, posA_y = _interpolate_along_line(L1_coords, L1_distances, x * L1_length)
        posB_x, posB_y = _interpolate_along_line(L2_coords, L2_distances, x * L2_length)
        cum_distance += np.sqrt((posA_x - posB_x) ** 2 + (posA_y - posB_y) ** 2)
    return cum_distance / samples


def walk_distance(L1: LineString, L2: LineString, samples: int = 16) -> float:
    """
    Computes integrated Euclidean distance between two LineStrings by 'walking'
    along each line and accumulating the distances between them.

    Lower values indicate better geometric alignment.
    """
    L1_coords, L1_distances, L1_length = _prepare_line_data(L1)
    L2_coords, L2_distances, L2_length = _prepare_line_data(L2)

    return float(
        _walk_distance_numba(
            L1_coords,
            L1_distances,
            L1_length,
            L2_coords,
            L2_distances,
            L2_length,
            samples,
        )
    )


@jit(nopython=True, cache=True)
def _walk_parallelness_numba(
    L1_coords: np.ndarray,
    L1_distances: np.ndarray,
    L1_length: float,
    L2_coords: np.ndarray,
    L2_distances: np.ndarray,
    L2_length: float,
    samples: int = 16,
) -> float:
    """Numba-optimized walk parallelness calculation."""
    # Ensure at least 2 samples to avoid division by zero
    if samples < 2:
        samples = 2
    # Pre-compute all positions
    posA_arr = np.zeros((samples, 2))
    posB_arr = np.zeros((samples, 2))

    for i in range(samples):
        x = i / (samples - 1)
        posA_arr[i, 0], posA_arr[i, 1] = _interpolate_along_line(
            L1_coords, L1_distances, x * L1_length
        )
        posB_arr[i, 0], posB_arr[i, 1] = _interpolate_along_line(
            L2_coords, L2_distances, x * L2_length
        )

    cum_parallelness = 0.0
    for i in range(1, samples):
        va_x = posA_arr[i, 0] - posA_arr[i - 1, 0]
        va_y = posA_arr[i, 1] - posA_arr[i - 1, 1]
        vb_x = posB_arr[i, 0] - posB_arr[i - 1, 0]
        vb_y = posB_arr[i, 1] - posB_arr[i - 1, 1]

        va_norm = np.sqrt(va_x * va_x + va_y * va_y)
        vb_norm = np.sqrt(vb_x * vb_x + vb_y * vb_y)

        if va_norm == 0 or vb_norm == 0:
            dot2 = 0.0
        else:
            va_x /= va_norm
            va_y /= va_norm
            vb_x /= vb_norm
            vb_y /= vb_norm
            dot = va_x * vb_x + va_y * vb_y
            # Square in case lines are oriented in opposite directions
            dot2 = dot * dot
        cum_parallelness += dot2

    return cum_parallelness / (samples - 1)


def walk_parallelness(L1: LineString, L2: LineString, samples: int = 16) -> float:
    """
    Computes integrated squared dot product of directions between two LineStrings
    by 'walking' along each line.

    Values close to 1.0 indicate the lines are parallel.
    Values close to 0.0 indicate the lines are perpendicular.
    """
    L1_coords, L1_distances, L1_length = _prepare_line_data(L1)
    L2_coords, L2_distances, L2_length = _prepare_line_data(L2)

    return float(
        _walk_parallelness_numba(
            L1_coords,
            L1_distances,
            L1_length,
            L2_coords,
            L2_distances,
            L2_length,
            samples,
        )
    )


# Module-level globals for multiprocessing worker data
_alignment_worker_data = None


def _init_alignment_worker(data):
    """Initialize worker process with shared geometry data."""
    global _alignment_worker_data
    _alignment_worker_data = data


def _compute_single_alignment(args):
    """Compute alignment for a single pair (worker function).

    Args:
        args: Tuple of (ref_idx, target_idx)

    Returns:
        AlignmentResult or None if computation fails
    """
    ref_idx, target_idx = args

    try:
        ref_geom = _alignment_worker_data["ref_geoms"][ref_idx]
        target_geom = _alignment_worker_data["target_geoms"][target_idx]

        if ref_geom is None or target_geom is None:
            return None
        if ref_geom.is_empty or target_geom.is_empty:
            return None

        return linestring_alignment(ref_geom, target_geom)
    except Exception:
        return None


def compute_alignment_batch(
    candidates: list,
    ref_geoms: np.ndarray,
    target_geoms: np.ndarray,
    n_jobs: int = -1,
) -> dict[tuple[int, int], AlignmentResult]:
    """Compute alignments for multiple candidate pairs in parallel.

    This function is designed to be called after blocking but before feature
    computation. It processes alignment for all candidate pairs using parallel
    workers, then returns results as a dictionary keyed by (ref_idx, target_idx).

    Args:
        candidates: List of CandidatePair objects with ref_idx and target_idx
        ref_geoms: NumPy array of reference geometries
        target_geoms: NumPy array of target geometries
        n_jobs: Number of parallel jobs (-1 for all cores minus 2)

    Returns:
        Dict mapping (ref_idx, target_idx) -> AlignmentResult
        Pairs that fail alignment computation are omitted from the result.
    """
    if not candidates:
        return {}

    # Determine number of workers
    if n_jobs == -1:
        n_workers = max(1, mp.cpu_count() - 2)
    else:
        n_workers = max(1, n_jobs)

    n_candidates = len(candidates)
    logger.info(f"Computing alignments for {n_candidates} candidates using {n_workers} workers...")

    # Prepare worker data
    worker_data = {
        "ref_geoms": ref_geoms,
        "target_geoms": target_geoms,
    }

    # Prepare work items as simple tuples
    work_items = [(cand.ref_idx, cand.target_idx) for cand in candidates]

    # Process with ProcessPoolExecutor
    chunk_size = max(1000, n_candidates // (n_workers * 4))
    results_list = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_alignment_worker,
        initargs=(worker_data,),
    ) as executor:
        # Process in chunks for progress reporting
        for i in range(0, len(work_items), chunk_size * n_workers):
            batch = work_items[i : i + chunk_size * n_workers]
            batch_results = list(
                executor.map(_compute_single_alignment, batch, chunksize=chunk_size)
            )
            results_list.extend(batch_results)
            logger.debug(
                f"Alignment progress: {min(i + len(batch), len(work_items))}/{len(work_items)}"
            )

    # Build result dictionary
    alignments = {}
    successful = 0
    for i, result in enumerate(results_list):
        if result is not None:
            ref_idx, target_idx = work_items[i]
            alignments[(ref_idx, target_idx)] = result
            successful += 1

    logger.info(f"Computed {successful}/{n_candidates} alignments successfully")
    return alignments


def compute_coverage_features(alignment: AlignmentResult | None) -> dict[str, float]:
    """Compute coverage features from an alignment result.

    These features are used by the ML model to learn appropriate overlap
    thresholds rather than applying hard filters.

    Args:
        alignment: AlignmentResult from linestring_alignment, or None

    Returns:
        Dict with coverage features:
        - ref_coverage: Fraction of reference covered (0-1)
        - target_coverage: Fraction of target covered (0-1)
        - min_coverage: Minimum of the two coverages
        - coverage_ratio: Symmetry of coverage (min/max)
    """
    if alignment is None:
        # No alignment available - return neutral values
        return {
            "ref_coverage": 0.0,
            "target_coverage": 0.0,
            "min_coverage": 0.0,
            "coverage_ratio": 0.0,
        }

    ref_cov = alignment.overture_coverage
    target_cov = alignment.dataset_coverage

    min_cov = min(ref_cov, target_cov)
    max_cov = max(ref_cov, target_cov)
    coverage_ratio = min_cov / max_cov if max_cov > 0 else 0.0

    return {
        "ref_coverage": ref_cov,
        "target_coverage": target_cov,
        "min_coverage": min_cov,
        "coverage_ratio": coverage_ratio,
    }
