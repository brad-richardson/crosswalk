"""Parity tests for the alignment seed-projection performance refactor.

The refactor (linestring_alignment split into a thin wrapper over
``_linestring_alignment_prepared`` with per-segment caching in the batch workers)
keeps the seeds bit-identical to the original GEOS computation while removing
per-pair overhead:

* Coordinate extraction (``_prepare_line_data``) is cached per unique segment in
  the batch workers.
* The three per-pair ``reference.project(Point(...))`` calls and one
  ``target.interpolate(0.5)`` are replaced with a cached-per-target seed-point
  array (``_target_seed_points``, midpoint via ``shapely.line_interpolate_point``)
  plus a single batched ``shapely.line_locate_point`` per pair. These vectorized
  Shapely calls hit the identical GEOS kernel as ``.interpolate``/``.project`` and
  return bit-identical values.

This module pins the invariant that the change is numerically inert: it inlines the
*original* GEOS-seeded implementation (``_old_linestring_alignment``) — reusing the
current, unchanged numba/geometry helpers so only the seed computation path differs
— and asserts that the new ``linestring_alignment`` produces identical results to
1e-12 over a battery of hand-picked and randomized geometries. Because the seeds are
bit-identical, the two paths agree exactly even on adversarial self-intersecting
polylines (where the score function's ternary refinement is sensitive to sub-1e-12
seed perturbations).
"""

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from crosswalk.config import (
    DIVERGENCE_DISTANCE_MULTIPLIER,
    DIVERGENCE_MIN_DISTANCE_M,
    DIVERGENCE_PARALLELNESS_THRESHOLD,
)
from crosswalk.features.alignment import (
    _ENDPOINT_SEED_THRESHOLD,
    _MIN_GRID_SAMPLES,
    _SEED_BUFFER_FRACTION,
    AlignmentResult,
    _detect_divergence_endpoints,
    _find_best_alignment_numba,
    _get_score_numba,
    _prepare_line_data,
    _tighten_junction_overlap,
    linestring_alignment,
)


def _old_linestring_alignment(
    reference: LineString,
    target: LineString,
    grid_samples: int = 24,
    refinement_steps: int = 16,
    detect_divergence: bool = True,
) -> AlignmentResult:
    """Verbatim copy of the pre-refactor implementation (GEOS-based seeding).

    Downstream helpers are imported from the live module (they were not touched
    by the refactor), so any difference between this and the new implementation
    is attributable solely to the seed-projection change under test.
    """
    ref_coords, ref_distances, ref_length = _prepare_line_data(reference)
    target_coords, target_distances, target_length = _prepare_line_data(target)

    if ref_length == 0 or target_length == 0:
        return AlignmentResult(0.0, 1.0, 0.0, 1.0)

    # --- Original GEOS-based seeding ---
    target_mid = target.interpolate(0.5, normalized=True)
    proj_start = reference.project(Point(target.coords[0]))
    proj_mid = reference.project(target_mid)
    proj_end = reference.project(Point(target.coords[-1]))

    buffer_distance_for_seed = (
        _SEED_BUFFER_FRACTION
        * min(ref_length, target_length)
        / max(grid_samples, _MIN_GRID_SAMPLES)
    )

    midpoint_seed = proj_mid - target_length / 2.0

    def _best_offset_with_endpoint_seeds(tgt_coords, tgt_distances, endpoint_seed_candidates):
        mp_score = _get_score_numba(
            ref_coords,
            ref_distances,
            ref_length,
            tgt_coords,
            tgt_distances,
            target_length,
            midpoint_seed,
            buffer_distance_for_seed,
        )

        best_ep_seed = None
        best_ep_score = mp_score
        for s in endpoint_seed_candidates:
            score = _get_score_numba(
                ref_coords,
                ref_distances,
                ref_length,
                tgt_coords,
                tgt_distances,
                target_length,
                s,
                buffer_distance_for_seed,
            )
            if score > mp_score * _ENDPOINT_SEED_THRESHOLD and score > best_ep_score:
                best_ep_seed = s
                best_ep_score = score

        best_offset, best_score = _find_best_alignment_numba(
            ref_coords,
            ref_distances,
            ref_length,
            tgt_coords,
            tgt_distances,
            target_length,
            grid_samples,
            refinement_steps,
            midpoint_seed,
        )

        if best_ep_seed is not None:
            ep_offset, ep_score = _find_best_alignment_numba(
                ref_coords,
                ref_distances,
                ref_length,
                tgt_coords,
                tgt_distances,
                target_length,
                grid_samples,
                refinement_steps,
                best_ep_seed,
            )
            if ep_score > best_score:
                best_offset, best_score = ep_offset, ep_score

        return best_offset, best_score

    forward_offset, forward_score = _best_offset_with_endpoint_seeds(
        target_coords,
        target_distances,
        [proj_start, proj_end - target_length],
    )

    target_coords_rev = target_coords[::-1].copy()
    target_distances_rev = target_length - target_distances[::-1]

    backward_offset, backward_score = _best_offset_with_endpoint_seeds(
        target_coords_rev,
        target_distances_rev,
        [proj_end, proj_start - target_length],
    )

    def unit_clamp(x):
        return max(0.0, min(1.0, x))

    is_forward = forward_score >= backward_score
    offset = forward_offset if is_forward else backward_offset

    used_target_coords = target_coords if is_forward else target_coords_rev
    used_target_distances = target_distances if is_forward else target_distances_rev

    clamped_grid_samples = max(grid_samples, 2)
    buffer_distance = 0.5 * min(ref_length, target_length) / clamped_grid_samples

    ref_start_frac = float(max(offset, 0) / ref_length)
    ref_end_frac = float(min(offset + target_length, ref_length) / ref_length)

    target_start_frac = float(max(-offset, 0) / target_length)
    target_end_frac = float(min(-offset + ref_length, target_length) / target_length)

    ref_start_frac, ref_end_frac, target_start_frac, target_end_frac = _tighten_junction_overlap(
        ref_coords,
        ref_distances,
        ref_length,
        used_target_coords,
        used_target_distances,
        target_length,
        ref_start_frac,
        ref_end_frac,
        target_start_frac,
        target_end_frac,
    )

    if detect_divergence and (ref_end_frac - ref_start_frac) > 0.1:
        new_ref_start, new_ref_end = _detect_divergence_endpoints(
            ref_coords,
            ref_distances,
            ref_length,
            used_target_coords,
            used_target_distances,
            target_length,
            offset,
            buffer_distance,
            num_samples=32,
            distance_multiplier=DIVERGENCE_DISTANCE_MULTIPLIER,
            min_distance_threshold=DIVERGENCE_MIN_DISTANCE_M,
            parallelness_threshold=DIVERGENCE_PARALLELNESS_THRESHOLD,
        )

        if new_ref_start > ref_start_frac or new_ref_end < ref_end_frac:
            original_ref_overlap = ref_end_frac - ref_start_frac
            original_target_overlap = target_end_frac - target_start_frac

            if original_ref_overlap > 0:
                start_truncation = (new_ref_start - ref_start_frac) / original_ref_overlap
                end_truncation = (ref_end_frac - new_ref_end) / original_ref_overlap

                target_start_frac = target_start_frac + start_truncation * original_target_overlap
                target_end_frac = target_end_frac - end_truncation * original_target_overlap

                ref_start_frac = new_ref_start
                ref_end_frac = new_ref_end

    if is_forward:
        return AlignmentResult(
            overture_start_frac=unit_clamp(ref_start_frac),
            overture_end_frac=unit_clamp(ref_end_frac),
            dataset_start_frac=unit_clamp(target_start_frac),
            dataset_end_frac=unit_clamp(target_end_frac),
        )
    else:
        return AlignmentResult(
            overture_start_frac=unit_clamp(ref_start_frac),
            overture_end_frac=unit_clamp(ref_end_frac),
            dataset_start_frac=unit_clamp(1.0 - target_end_frac),
            dataset_end_frac=unit_clamp(1.0 - target_start_frac),
            is_reversed=True,
        )


def _assert_identical(
    old: AlignmentResult, new: AlignmentResult, tol: float = 1e-12, msg: str = ""
):
    assert new.overture_start_frac == pytest.approx(old.overture_start_frac, abs=tol), msg
    assert new.overture_end_frac == pytest.approx(old.overture_end_frac, abs=tol), msg
    assert new.dataset_start_frac == pytest.approx(old.dataset_start_frac, abs=tol), msg
    assert new.dataset_end_frac == pytest.approx(old.dataset_end_frac, abs=tol), msg
    # The forward/backward decision (is_forward) must also be preserved — it
    # drives the is_reversed flag consumed by directional topology features.
    assert new.is_reversed == old.is_reversed, msg


# Hand-picked diverse scenarios mirroring the existing alignment fixtures:
# identical, reversed, partial overlaps, T-junction, asymmetric lengths, zigzag,
# divergence, winding loop, and the projected-UTM Weston Road junction case.
_DIVERSE_CASES = [
    ("identical", [(0, 0), (100, 0)], [(0, 0), (100, 0)]),
    ("reversed", [(0, 0), (100, 0)], [(100, 0), (0, 0)]),
    ("first_half", [(0, 0), (100, 0)], [(0, 0), (50, 0)]),
    ("second_half", [(0, 0), (100, 0)], [(50, 0), (100, 0)]),
    ("target_extends_beyond", [(25, 0), (75, 0)], [(0, 0), (100, 0)]),
    ("parallel_offset", [(0, 0), (100, 0)], [(0, 10), (100, 10)]),
    ("zigzag_ref", [(0, 0), (25, 10), (50, 0), (75, 10), (100, 0)], [(20, 5), (80, 5)]),
    ("t_junction", [(0, 0), (100, 0)], [(50, 0), (50, 100)]),
    ("diverging_end", [(0, 0), (80, 0), (100, 0)], [(0, 0), (80, 0), (100, 50)]),
    ("diverging_start", [(0, 0), (20, 0), (100, 0)], [(0, 50), (20, 0), (100, 0)]),
    ("symmetric_divergence", [(0, 0), (50, 0), (100, 0)], [(0, 30), (50, 0), (100, 30)]),
    (
        "winding_loop",
        [(0, 0), (25, 0), (25, 200), (75, 200), (75, 0), (50, 0)],
        [(0, 0), (50, 0)],
    ),
    ("asymmetric_lengths", [(0, 0), (1000, 0)], [(490, 3), (510, 3)]),
    (
        "weston_road_junction",
        [
            (617466.03, 4844691.99),
            (617463.95, 4844703.05),
            (617459.56, 4844726.42),
            (617450.84, 4844772.80),
            (617446.95, 4844793.49),
            (617445.04, 4844804.00),
            (617438.18, 4844844.67),
            (617434.03, 4844865.00),
            (617426.45, 4844908.20),
            (617424.03, 4844922.00),
            (617420.80, 4844940.86),
            (617419.39, 4844948.23),
        ],
        [(617419.81, 4844947.81), (617371.86, 4845233.37)],
    ),
]


class TestSeedRefactorParity:
    """The numeric-seed refactor must be numerically inert vs the GEOS-seeded original."""

    @pytest.mark.parametrize("case", _DIVERSE_CASES, ids=[c[0] for c in _DIVERSE_CASES])
    def test_diverse_scenarios_identical(self, case):
        name, ref_coords, target_coords = case
        ref = LineString(ref_coords)
        target = LineString(target_coords)
        old = _old_linestring_alignment(ref, target)
        new = linestring_alignment(ref, target)
        _assert_identical(old, new, msg=f"scenario={name}")

    def test_randomized_pairs_identical(self):
        """Fuzz many random polyline pairs; new path must match old to 1e-12."""
        rng = np.random.default_rng(20260704)
        n_pairs = 3000
        max_delta = 0.0
        for _ in range(n_pairs):
            # Mix of scales: local-metric (~hundreds of m) and projected-UTM offsets.
            base = rng.uniform(-500, 500, size=2)
            n_ref = int(rng.integers(2, 7))
            n_tgt = int(rng.integers(2, 7))
            ref_pts = base + rng.uniform(-300, 300, size=(n_ref, 2))
            tgt_pts = base + rng.uniform(-300, 300, size=(n_tgt, 2))
            ref = LineString(ref_pts)
            target = LineString(tgt_pts)
            if ref.length == 0 or target.length == 0:
                continue
            old = _old_linestring_alignment(ref, target)
            new = linestring_alignment(ref, target)
            assert new.is_reversed == old.is_reversed
            max_delta = max(
                max_delta,
                abs(new.overture_start_frac - old.overture_start_frac),
                abs(new.overture_end_frac - old.overture_end_frac),
                abs(new.dataset_start_frac - old.dataset_start_frac),
                abs(new.dataset_end_frac - old.dataset_end_frac),
            )
        assert max_delta < 1e-12, f"max fraction deviation {max_delta} exceeds 1e-12"

    def test_near_collinear_junction_pairs_identical(self):
        """Barely-overlapping collinear pairs exercise the endpoint-seed branch."""
        rng = np.random.default_rng(11)
        max_delta = 0.0
        for _ in range(1500):
            length_a = rng.uniform(20, 400)
            length_b = rng.uniform(20, 400)
            gap = rng.uniform(-5, 5)  # small overlap/gap at the shared tip
            jitter = rng.uniform(-2, 2, size=4)
            ref = LineString([(0.0 + jitter[0], jitter[1]), (length_a, 0.0)])
            target = LineString(
                [(length_a + gap, jitter[2]), (length_a + gap + length_b, jitter[3])]
            )
            if ref.length == 0 or target.length == 0:
                continue
            old = _old_linestring_alignment(ref, target)
            new = linestring_alignment(ref, target)
            assert new.is_reversed == old.is_reversed
            max_delta = max(
                max_delta,
                abs(new.overture_start_frac - old.overture_start_frac),
                abs(new.overture_end_frac - old.overture_end_frac),
                abs(new.dataset_start_frac - old.dataset_start_frac),
                abs(new.dataset_end_frac - old.dataset_end_frac),
            )
        assert max_delta < 1e-12, f"max fraction deviation {max_delta} exceeds 1e-12"
