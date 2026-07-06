"""Tests for intersection overlap features: post_node_continuation_m and endpoint_heading_divergence."""

import math

import numpy as np
import pytest
from shapely.geometry import LineString

from crosswalk.features.alignment import AlignmentResult
from crosswalk.features.compute import _compute_intersection_overlap_features


def _make_result(ref, target, alignment):
    return _compute_intersection_overlap_features(ref, target, alignment)


# ── Integration tests: geometry scenarios → feature expectations ──────────


@pytest.mark.parametrize(
    "name, ref, target, alignment, cont_range, div_range",
    [
        pytest.param(
            "t_junction_end",
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 0), (10, 0), (10, 50)]),
            AlignmentResult(0.0, 0.10, 0.0, 10.0 / 60.0),
            (0, 15),  # low continuation (target turns away)
            (30, 90),  # high divergence (perpendicular)
            id="t_junction_end",
        ),
        pytest.param(
            "perpendicular_crossing",
            LineString([(0, 0), (200, 0)]),
            LineString([(100, -50), (100, 0), (100, 50)]),
            AlignmentResult(0.50, 0.501, 0.49, 0.51),
            (0, 5),  # near-zero continuation
            (30, 90),  # high divergence
            id="perpendicular_crossing",
        ),
        pytest.param(
            "collinear_partial_overlap",
            LineString([(0, 0), (100, 0)]),
            LineString([(25, 0), (75, 0), (125, 0)]),
            AlignmentResult(0.25, 0.75, 0.0, 0.50),
            (30, None),  # high continuation (same direction)
            (0, 15),  # low divergence
            id="collinear_partial_overlap",
        ),
        pytest.param(
            "parallel_offset",
            LineString([(0, 0), (200, 0)]),
            LineString([(25, 2), (175, 2)]),
            AlignmentResult(0.125, 0.875, 0.0, 1.0),
            None,  # full target coverage → NaN
            (0, 10),  # low divergence (parallel)
            id="parallel_offset",
        ),
        pytest.param(
            "reversed_digitization",
            LineString([(0, 0), (100, 0)]),
            LineString([(150, 0), (100, 0), (50, 0)]),
            AlignmentResult(0.50, 1.0, 0.333, 1.0),
            (25, None),  # start remainder goes along ref → high continuation
            (0, 15),  # bidirectional heading → low divergence
            id="reversed_digitization",
        ),
    ],
)
def test_intersection_overlap_scenarios(name, ref, target, alignment, cont_range, div_range):
    """Parameterized test covering key geometry scenarios."""
    result = _make_result(ref, target, alignment)

    cont = result["post_node_continuation_m"]
    div = result["endpoint_heading_divergence"]

    if cont_range is None:
        assert math.isnan(cont), f"{name}: expected NaN continuation, got {cont}"
    else:
        lo, hi = cont_range
        assert not math.isnan(cont), f"{name}: unexpected NaN continuation"
        assert cont >= lo, f"{name}: continuation {cont} < {lo}"
        if hi is not None:
            assert cont <= hi, f"{name}: continuation {cont} > {hi}"

    lo, hi = div_range
    assert not math.isnan(div), f"{name}: unexpected NaN divergence"
    assert lo <= div <= hi, f"{name}: divergence {div} not in [{lo}, {hi}]"


# ── Edge cases ────────────────────────────────────────────────────────────


def test_no_alignment_returns_defaults():
    """None alignment → NaN for both features."""
    ref = LineString([(0, 0), (100, 0)])
    target = LineString([(200, 200), (300, 300)])
    result = _make_result(ref, target, None)

    assert math.isnan(result["post_node_continuation_m"])
    assert math.isnan(result["endpoint_heading_divergence"])


def test_full_target_coverage_nan_continuation():
    """Target fully within alignment → NaN continuation."""
    result = _make_result(
        LineString([(0, 0), (100, 0)]),
        LineString([(20, 1), (80, 1)]),
        AlignmentResult(0.20, 0.80, 0.0, 1.0),
    )
    assert math.isnan(result["post_node_continuation_m"])
    assert not math.isnan(result["endpoint_heading_divergence"])


def test_short_ref_no_crash():
    """5m ref segment → valid output without crash."""
    result = _make_result(
        LineString([(0, 0), (5, 0)]),
        LineString([(0, 0), (5, 0), (10, 5)]),
        AlignmentResult(0.0, 1.0, 0.0, 0.5),
    )
    assert isinstance(result["endpoint_heading_divergence"], float)
    assert not math.isnan(result["endpoint_heading_divergence"])


def test_tiny_remainder_is_nan():
    """Target remainder < 0.5m → NaN continuation."""
    result = _make_result(
        LineString([(0, 0), (100, 0)]),
        LineString([(0, 0), (50, 0), (50.3, 0)]),
        AlignmentResult(0.0, 0.50, 0.0, 50.0 / 50.3),
    )
    assert math.isnan(result["post_node_continuation_m"])


# ── JIT helper unit tests ────────────────────────────────────────────────


class TestContinuationNumba:
    """Direct tests for compute_continuation_along_heading_numba."""

    @pytest.mark.parametrize(
        "coords, dx, dy, expected",
        [
            ([(0, 0), (50, 0), (100, 0)], 1.0, 0.0, 100.0),  # straight along heading
            ([(0, 0), (0, 50)], 1.0, 0.0, 0.0),  # perpendicular
            ([(0, 0), (50, 50)], 1.0, 0.0, 50.0),  # 45° diagonal
            ([(0, 0)], 1.0, 0.0, 0.0),  # single point
        ],
        ids=["straight", "perpendicular", "diagonal_45", "single_point"],
    )
    def test_continuation(self, coords, dx, dy, expected):
        from crosswalk.features._jit_helpers import compute_continuation_along_heading_numba

        result = compute_continuation_along_heading_numba(np.array(coords, dtype=float), dx, dy)
        assert result == pytest.approx(expected, abs=2.0)


class TestHeadingAtFractionNumba:
    """Direct tests for compute_heading_at_fraction_numba."""

    def test_horizontal_line_start(self):
        from crosswalk.features._jit_helpers import compute_heading_at_fraction_numba

        coords = np.array([(0.0, 0.0), (100.0, 0.0)])
        heading = compute_heading_at_fraction_numba(coords, np.array([100.0]), 100.0, 0.0)
        assert heading == pytest.approx(0.0, abs=1.0)

    def test_midpoint_of_turn(self):
        from crosswalk.features._jit_helpers import compute_heading_at_fraction_numba

        coords = np.array([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)])
        heading = compute_heading_at_fraction_numba(coords, np.array([50.0, 50.0]), 100.0, 0.5)
        assert isinstance(heading, float)
