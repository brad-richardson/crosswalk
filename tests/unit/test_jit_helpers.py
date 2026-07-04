"""Tests for Numba JIT-compiled helper functions.

These tests verify correctness of the JIT functions used for
collinear gap ratio computation with pre-extracted coordinates.
"""

import numpy as np
import pytest
from shapely import LineString

from matcher.features._jit_helpers import (
    angle_diff_numba,
    collinear_gap_ratio_numba,
    compute_crossing_angle_stats_numba,
    compute_endpoint_proximity_numba,
    compute_heading_consistency_numba,
    compute_heading_numba,
    compute_parallel_alignment_numba,
    compute_shape_complexity_numba,
    query_nearby_endpoints_numba,
)
from matcher.features.geometric import (
    _compute_hausdorff_stats,
    compute_collinear_gap_ratio,
    compute_shape_complexity,
    compute_sinuosity,
    compute_vertex_density,
)
from matcher.features.relational import compute_endpoint_proximity, compute_parallel_alignment


class TestComputeHeadingNumba:
    """Tests for compute_heading_numba function."""

    def test_east(self):
        """East direction should be 0 degrees."""
        assert compute_heading_numba(1.0, 0.0) == pytest.approx(0.0)

    def test_north(self):
        """North direction should be 90 degrees."""
        assert compute_heading_numba(0.0, 1.0) == pytest.approx(90.0)

    def test_west(self):
        """West direction should be 180 degrees."""
        assert compute_heading_numba(-1.0, 0.0) == pytest.approx(180.0)

    def test_south(self):
        """South direction should be 270 degrees."""
        assert compute_heading_numba(0.0, -1.0) == pytest.approx(270.0)

    def test_northeast(self):
        """Northeast direction should be 45 degrees."""
        assert compute_heading_numba(1.0, 1.0) == pytest.approx(45.0)

    def test_northwest(self):
        """Northwest direction should be 135 degrees."""
        assert compute_heading_numba(-1.0, 1.0) == pytest.approx(135.0)


class TestAngleDiffNumba:
    """Tests for angle_diff_numba function."""

    def test_same_direction(self):
        """Same direction should have 0 difference."""
        assert angle_diff_numba(45.0, 45.0) == pytest.approx(0.0)

    def test_opposite_direction(self):
        """Opposite direction should have 0 difference (roads are bidirectional)."""
        assert angle_diff_numba(0.0, 180.0) == pytest.approx(0.0)
        assert angle_diff_numba(90.0, 270.0) == pytest.approx(0.0)

    def test_perpendicular(self):
        """Perpendicular directions should have 90 degree difference."""
        assert angle_diff_numba(0.0, 90.0) == pytest.approx(90.0)
        assert angle_diff_numba(45.0, 135.0) == pytest.approx(90.0)

    def test_small_difference(self):
        """Small angle differences should be preserved."""
        assert angle_diff_numba(10.0, 15.0) == pytest.approx(5.0)
        assert angle_diff_numba(350.0, 5.0) == pytest.approx(15.0)

    def test_near_opposite(self):
        """Near-opposite directions should have small difference."""
        assert angle_diff_numba(0.0, 175.0) == pytest.approx(5.0)
        assert angle_diff_numba(10.0, 185.0) == pytest.approx(5.0)

    def test_wrap_around(self):
        """Should handle wrap-around at 360 degrees."""
        assert angle_diff_numba(5.0, 355.0) == pytest.approx(10.0)


class TestCollinearGapRatioNumba:
    """Tests for collinear_gap_ratio_numba JIT function."""

    def test_tip_to_tip_no_overlap(self):
        """Tip-to-tip segments should have zero overlap fraction."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[100.0, 0.0], [200.0, 0.0]])

        result = collinear_gap_ratio_numba(coords_a, coords_b, heading_threshold=15.0)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_full_overlap(self):
        """Fully overlapping segments should return 1.0."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[25.0, 0.0], [75.0, 0.0]])

        result = collinear_gap_ratio_numba(coords_a, coords_b, heading_threshold=15.0)
        assert result == pytest.approx(1.0)

    def test_partial_overlap_returns_fraction(self):
        """Partial overlap should return the raw overlap fraction."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[80.0, 0.0], [180.0, 0.0]])

        result = collinear_gap_ratio_numba(coords_a, coords_b, heading_threshold=15.0)
        # 20m overlap on 100m segment = 0.2
        assert result == pytest.approx(0.2, abs=0.05)

    def test_opposite_heading_collinear(self):
        """Segments with opposite headings should still compute overlap correctly."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[50.0, 0.0], [25.0, 0.0]])  # Reversed direction

        result = collinear_gap_ratio_numba(coords_a, coords_b, heading_threshold=15.0)
        # Should detect overlap despite opposite heading
        assert result == pytest.approx(1.0)

    def test_non_collinear_returns_one(self):
        """Non-collinear segments should return 1.0 (no penalty)."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])  # Horizontal
        coords_b = np.array([[50.0, 0.0], [50.0, 100.0]])  # Vertical

        result = collinear_gap_ratio_numba(coords_a, coords_b, heading_threshold=15.0)
        assert result == pytest.approx(1.0)


class TestCollinearGapRatioConsistency:
    """Test that wrapper function correctly uses JIT implementation."""

    @pytest.fixture
    def synthetic_lines(self):
        """Generate synthetic lines for testing."""
        np.random.seed(42)
        lines = []
        for _ in range(50):
            start = np.random.rand(2) * 1000
            angle = np.random.rand() * 2 * np.pi
            length = 50 + np.random.rand() * 100
            n_points = np.random.randint(2, 6)
            coords = [start]
            current = start.copy()
            segment_length = length / n_points
            for _ in range(n_points):
                angle += (np.random.rand() - 0.5) * 0.3
                current = current + segment_length * np.array([np.cos(angle), np.sin(angle)])
                coords.append(current.copy())
            lines.append(LineString(coords))
        return lines

    def test_valid_range(self, synthetic_lines):
        """Results should always be in valid range [0.0, 1.0]."""
        for i in range(len(synthetic_lines)):
            line_a = synthetic_lines[i]
            line_b = synthetic_lines[(i + 7) % len(synthetic_lines)]

            result = compute_collinear_gap_ratio(line_a, line_b)
            assert 0.0 <= result <= 1.0, f"Result {result} out of range"

    def test_collinear_parallel_lines(self):
        """Parallel collinear lines with overlap should return 1.0."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(20, 0), (80, 0)])

        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result == pytest.approx(1.0)

    def test_tip_to_tip_lines(self):
        """Tip-to-tip collinear lines should return low score."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(100, 0), (200, 0)])

        result = compute_collinear_gap_ratio(line_a, line_b)
        # Should penalize tip-to-tip
        assert result < 0.1

    def test_non_collinear_lines(self):
        """Non-collinear lines should return 1.0 (no penalty)."""
        line_a = LineString([(0, 0), (100, 0)])  # Horizontal
        line_b = LineString([(50, 0), (50, 100)])  # Vertical

        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result == pytest.approx(1.0)

    def test_empty_line(self):
        """Empty lines should return 1.0."""
        line_a = LineString()
        line_b = LineString([(0, 0), (100, 0)])

        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result == pytest.approx(1.0)

    def test_with_pre_extracted_coords(self):
        """Should work with pre-extracted coordinates."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(20, 0), (80, 0)])
        coords_a = np.array(line_a.coords)
        coords_b = np.array(line_b.coords)

        result = compute_collinear_gap_ratio(line_a, line_b, coords_a=coords_a, coords_b=coords_b)
        assert result == pytest.approx(1.0)

    def test_pre_extracted_matches_auto_extracted(self):
        """Pre-extracted coords should give same result as auto-extraction."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(100, 0), (200, 0)])

        result_auto = compute_collinear_gap_ratio(line_a, line_b)

        coords_a = np.array(line_a.coords)
        coords_b = np.array(line_b.coords)
        result_pre = compute_collinear_gap_ratio(
            line_a, line_b, coords_a=coords_a, coords_b=coords_b
        )

        assert result_auto == pytest.approx(result_pre)


class TestComputeShapeComplexityNumba:
    """Tests for compute_shape_complexity_numba function."""

    def test_straight_line(self):
        """Straight line should have 0 turns."""
        coords = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
        assert compute_shape_complexity_numba(coords, 10.0) == 0

    def test_single_90_degree_turn(self):
        """90 degree turn should count as 1 turn."""
        coords = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]])
        assert compute_shape_complexity_numba(coords, 10.0) == 1

    def test_multiple_turns(self):
        """Multiple turns should be counted."""
        coords = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [100.0, 50.0]])
        assert compute_shape_complexity_numba(coords, 10.0) == 2

    def test_small_angle_below_threshold(self):
        """Small angle changes below threshold should not count."""
        coords = np.array([[0.0, 0.0], [50.0, 1.0], [100.0, 0.0]])
        # Angle is ~2 degrees, below 10 degree threshold
        assert compute_shape_complexity_numba(coords, 10.0) == 0

    def test_too_few_points(self):
        """Fewer than 3 points should return 0."""
        coords = np.array([[0.0, 0.0], [100.0, 0.0]])
        assert compute_shape_complexity_numba(coords, 10.0) == 0

    def test_duplicate_vertex_does_not_fabricate_turn(self):
        """A duplicated vertex on a straight diagonal must not create turns.

        Regression: arctan2(0, 0) = 0° gave the zero-length segment a phantom
        eastward heading, counting 2 spurious turns on a straight diagonal
        (masked on axis-aligned lines where 0° coincides with the true
        heading).
        """
        coords = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [10.0, 10.0]])
        assert compute_shape_complexity_numba(coords, 10.0) == 0

    def test_duplicate_vertex_preserves_real_turn(self):
        """A genuine turn at a duplicated vertex must still be counted."""
        coords = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 0.0], [50.0, 50.0]])
        assert compute_shape_complexity_numba(coords, 10.0) == 1

    def test_duplicate_vertex_heading_consistency(self):
        """Duplicated vertex must not degrade heading consistency."""
        points = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [10.0, 10.0]])
        assert compute_heading_consistency_numba(points) == pytest.approx(1.0)

    def test_wrapper_matches_numba(self):
        """Wrapper function should return same result as JIT function."""
        line = LineString([(0, 0), (50, 0), (50, 50), (100, 50)])
        coords = np.array(line.coords)

        result_wrapper = compute_shape_complexity(line)
        result_numba = compute_shape_complexity_numba(coords, 10.0)

        assert result_wrapper == result_numba

    def test_wrapper_with_pre_extracted_coords(self):
        """Wrapper should work with pre-extracted coordinates."""
        line = LineString([(0, 0), (50, 0), (50, 50)])
        coords = np.array(line.coords)

        result = compute_shape_complexity(line, coords=coords)
        assert result == 1


class TestComputeParallelAlignmentNumba:
    """Tests for compute_parallel_alignment_numba function."""

    def test_parallel_lines(self):
        """Parallel lines should return 1.0."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[0.0, 10.0], [100.0, 10.0]])
        assert compute_parallel_alignment_numba(coords_a, coords_b) == pytest.approx(1.0)

    def test_opposite_direction_parallel(self):
        """Opposite direction but parallel should return 1.0."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[100.0, 10.0], [0.0, 10.0]])
        assert compute_parallel_alignment_numba(coords_a, coords_b) == pytest.approx(1.0)

    def test_perpendicular_lines(self):
        """Perpendicular lines should return 0.0."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[50.0, 0.0], [50.0, 100.0]])
        assert compute_parallel_alignment_numba(coords_a, coords_b) == pytest.approx(0.0)

    def test_45_degree_angle(self):
        """45 degree angle should return 0.5."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[0.0, 0.0], [100.0, 100.0]])
        assert compute_parallel_alignment_numba(coords_a, coords_b) == pytest.approx(0.5)

    def test_wrapper_matches_numba(self):
        """Wrapper function should return same result as JIT function."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(0, 10), (100, 10)])
        coords_a = np.array(line_a.coords)
        coords_b = np.array(line_b.coords)

        result_wrapper = compute_parallel_alignment(line_a, line_b)
        result_numba = compute_parallel_alignment_numba(coords_a, coords_b)

        assert result_wrapper == pytest.approx(result_numba)

    def test_wrapper_with_pre_extracted_coords(self):
        """Wrapper should work with pre-extracted coordinates."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(0, 10), (100, 10)])
        coords_a = np.array(line_a.coords)
        coords_b = np.array(line_b.coords)

        result = compute_parallel_alignment(line_a, line_b, coords_a=coords_a, coords_b=coords_b)
        assert result == pytest.approx(1.0)


class TestComputeEndpointProximityNumba:
    """Tests for compute_endpoint_proximity_numba function."""

    def test_single_nearby_endpoint(self):
        """Should find distance to single nearby endpoint."""
        start = np.array([0.0, 0.0])
        end = np.array([100.0, 0.0])
        endpoints = np.array([[5.0, 0.0]])  # 5m from start

        s_prox, e_prox, shared = compute_endpoint_proximity_numba(start, end, endpoints, 10.0)
        assert s_prox == pytest.approx(5.0)
        assert e_prox == pytest.approx(95.0)
        assert shared == 1  # Only start is within tolerance

    def test_multiple_endpoints(self):
        """Should find minimum distance among multiple endpoints."""
        start = np.array([0.0, 0.0])
        end = np.array([100.0, 0.0])
        endpoints = np.array(
            [
                [10.0, 0.0],  # 10m from start
                [3.0, 0.0],  # 3m from start (closest)
                [97.0, 0.0],  # 3m from end
            ]
        )

        s_prox, e_prox, shared = compute_endpoint_proximity_numba(start, end, endpoints, 5.0)
        assert s_prox == pytest.approx(3.0)
        assert e_prox == pytest.approx(3.0)
        assert shared == 2  # start and end each have 1 within tolerance

    def test_empty_endpoints(self):
        """Empty endpoint array should return inf."""
        start = np.array([0.0, 0.0])
        end = np.array([100.0, 0.0])
        endpoints = np.empty((0, 2))

        s_prox, e_prox, shared = compute_endpoint_proximity_numba(start, end, endpoints, 5.0)
        assert s_prox == np.inf
        assert e_prox == np.inf
        assert shared == 0

    def test_wrapper_matches_numba(self):
        """Wrapper function should return same result as JIT function."""
        target = LineString([(0, 0), (100, 0)])
        endpoints = np.array([[5.0, 0.0], [95.0, 0.0]])

        result_wrapper = compute_endpoint_proximity(target, endpoints, 10.0)
        target_coords = np.array(target.coords)
        result_numba = compute_endpoint_proximity_numba(
            target_coords[0], target_coords[-1], endpoints, 10.0
        )

        assert result_wrapper[0] == pytest.approx(result_numba[0])
        assert result_wrapper[1] == pytest.approx(result_numba[1])
        assert result_wrapper[2] == result_numba[2]

    def test_wrapper_with_pre_extracted_coords(self):
        """Wrapper should work with pre-extracted coordinates."""
        target = LineString([(0, 0), (100, 0)])
        endpoints = np.array([[5.0, 0.0]])
        target_coords = np.array(target.coords)

        result = compute_endpoint_proximity(target, endpoints, 10.0, target_coords=target_coords)
        assert result[0] == pytest.approx(5.0)


class TestComputeHeadingConsistencyNumba:
    """Tests for compute_heading_consistency_numba function."""

    def test_straight_line(self):
        """Straight line should return 1.0."""
        points = np.array(
            [
                [0.0, 0.0],
                [25.0, 0.0],
                [50.0, 0.0],
                [75.0, 0.0],
                [100.0, 0.0],
            ]
        )
        assert compute_heading_consistency_numba(points) == pytest.approx(1.0)

    def test_90_degree_turn(self):
        """90 degree turn should return ~0.0."""
        points = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]])
        # One heading diff of 90 degrees -> avg = 90 -> 1 - 90/90 = 0
        assert compute_heading_consistency_numba(points) == pytest.approx(0.0)

    def test_gentle_curve(self):
        """Gentle curve should return high consistency."""
        # Small deviations should result in high consistency
        points = np.array(
            [
                [0.0, 0.0],
                [25.0, 0.5],
                [50.0, 1.5],
                [75.0, 3.0],
                [100.0, 5.0],
            ]
        )
        result = compute_heading_consistency_numba(points)
        assert result > 0.9  # Should be close to 1.0

    def test_too_few_points(self):
        """Fewer than 3 points should return 1.0."""
        points = np.array([[0.0, 0.0], [100.0, 0.0]])
        assert compute_heading_consistency_numba(points) == pytest.approx(1.0)

    def test_single_point(self):
        """Single point should return 1.0."""
        points = np.array([[0.0, 0.0]])
        assert compute_heading_consistency_numba(points) == pytest.approx(1.0)


class TestHausdorffStatsOptionalCoords:
    """Tests for _compute_hausdorff_stats with optional coords parameter."""

    def test_with_pre_extracted_coords(self):
        """Should work with pre-extracted coordinates."""
        line_a = LineString([(0, 0), (50, 0), (100, 0)])
        line_b = LineString([(0, 5), (50, 5), (100, 5)])
        coords_a = np.array(line_a.coords)
        coords_b = np.array(line_b.coords)

        mean_dist, p95_dist = _compute_hausdorff_stats(
            line_a, line_b, coords_a=coords_a, coords_b=coords_b
        )
        assert mean_dist == pytest.approx(5.0)
        assert p95_dist == pytest.approx(5.0)

    def test_pre_extracted_matches_auto_extracted(self):
        """Pre-extracted coords should give same result as auto-extraction."""
        line_a = LineString([(0, 0), (50, 10), (100, 0)])
        line_b = LineString([(0, 20), (50, 30), (100, 20)])

        # Auto-extraction (no coords passed)
        mean_auto, p95_auto = _compute_hausdorff_stats(line_a, line_b)

        # Pre-extracted
        coords_a = np.array(line_a.coords)
        coords_b = np.array(line_b.coords)
        mean_pre, p95_pre = _compute_hausdorff_stats(
            line_a, line_b, coords_a=coords_a, coords_b=coords_b
        )

        assert mean_auto == pytest.approx(mean_pre)
        assert p95_auto == pytest.approx(p95_pre)

    def test_partial_coords_provided(self):
        """Should handle when only one set of coords is provided."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(0, 10), (100, 10)])
        coords_a = np.array(line_a.coords)

        # Only coords_a provided
        mean_dist, p95_dist = _compute_hausdorff_stats(line_a, line_b, coords_a=coords_a)
        assert mean_dist == pytest.approx(10.0)
        assert p95_dist == pytest.approx(10.0)


class TestSinuosityOptionalCoords:
    """Tests for compute_sinuosity with optional coords parameter."""

    def test_straight_line(self):
        """Straight line should have sinuosity of 1.0."""
        line = LineString([(0, 0), (100, 0)])
        assert compute_sinuosity(line) == pytest.approx(1.0)

    def test_with_pre_extracted_coords(self):
        """Should work with pre-extracted coordinates."""
        line = LineString([(0, 0), (100, 0)])
        coords = np.array(line.coords)

        result = compute_sinuosity(line, coords=coords)
        assert result == pytest.approx(1.0)

    def test_pre_extracted_matches_auto_extracted(self):
        """Pre-extracted coords should give same result as auto-extraction."""
        # Create a curvy line
        line = LineString([(0, 0), (25, 10), (50, 0), (75, 10), (100, 0)])

        result_auto = compute_sinuosity(line)

        coords = np.array(line.coords)
        result_pre = compute_sinuosity(line, coords=coords)

        assert result_auto == pytest.approx(result_pre)

    def test_curvy_line(self):
        """Curvy line should have sinuosity > 1.0."""
        # Create a line that curves up and back down
        line = LineString([(0, 0), (50, 50), (100, 0)])
        coords = np.array(line.coords)

        result = compute_sinuosity(line, coords=coords)
        assert result > 1.0  # Path is longer than straight-line distance

    def test_loop_returns_nan(self):
        """Loop (start == end) should return NaN (sinuosity undefined)."""
        line = LineString([(0, 0), (50, 50), (100, 0), (50, -50), (0, 0)])
        coords = np.array(line.coords)

        result = compute_sinuosity(line, coords=coords)
        assert np.isnan(result)


class TestQueryNearbyEndpointsNumba:
    """Tests for query_nearby_endpoints_numba function."""

    def test_single_endpoint_within_radius(self):
        """Should find single endpoint within radius."""
        endpoint_coords = np.array([[10.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
        candidate_indices = np.array([0, 1, 2], dtype=np.int64)
        point_coords = np.array([0.0, 0.0])
        radius = 15.0

        result_indices, result_dists = query_nearby_endpoints_numba(
            endpoint_coords, candidate_indices, point_coords, radius
        )

        assert len(result_indices) == 1
        assert result_indices[0] == 0
        assert result_dists[0] == pytest.approx(10.0)

    def test_multiple_endpoints_within_radius(self):
        """Should find multiple endpoints within radius."""
        endpoint_coords = np.array([[5.0, 0.0], [10.0, 0.0], [100.0, 0.0]])
        candidate_indices = np.array([0, 1, 2], dtype=np.int64)
        point_coords = np.array([0.0, 0.0])
        radius = 15.0

        result_indices, result_dists = query_nearby_endpoints_numba(
            endpoint_coords, candidate_indices, point_coords, radius
        )

        assert len(result_indices) == 2
        assert set(result_indices) == {0, 1}
        assert result_dists[0] == pytest.approx(5.0)
        assert result_dists[1] == pytest.approx(10.0)

    def test_no_endpoints_within_radius(self):
        """Should return empty arrays when no endpoints within radius."""
        endpoint_coords = np.array([[100.0, 0.0], [200.0, 0.0]])
        candidate_indices = np.array([0, 1], dtype=np.int64)
        point_coords = np.array([0.0, 0.0])
        radius = 50.0

        result_indices, result_dists = query_nearby_endpoints_numba(
            endpoint_coords, candidate_indices, point_coords, radius
        )

        assert len(result_indices) == 0
        assert len(result_dists) == 0

    def test_empty_candidates(self):
        """Should handle empty candidate array."""
        endpoint_coords = np.array([[10.0, 0.0]])
        candidate_indices = np.array([], dtype=np.int64)
        point_coords = np.array([0.0, 0.0])
        radius = 50.0

        result_indices, result_dists = query_nearby_endpoints_numba(
            endpoint_coords, candidate_indices, point_coords, radius
        )

        assert len(result_indices) == 0
        assert len(result_dists) == 0

    def test_diagonal_distance(self):
        """Should compute correct diagonal (Euclidean) distance."""
        endpoint_coords = np.array([[3.0, 4.0]])  # Distance 5 from origin
        candidate_indices = np.array([0], dtype=np.int64)
        point_coords = np.array([0.0, 0.0])
        radius = 5.0

        result_indices, result_dists = query_nearby_endpoints_numba(
            endpoint_coords, candidate_indices, point_coords, radius
        )

        assert len(result_indices) == 1
        assert result_dists[0] == pytest.approx(5.0)

    def test_subset_of_candidates(self):
        """Should only check specified candidate indices."""
        endpoint_coords = np.array([[5.0, 0.0], [10.0, 0.0], [1.0, 0.0]])
        # Only check indices 0 and 1, not 2 (which is closest)
        candidate_indices = np.array([0, 1], dtype=np.int64)
        point_coords = np.array([0.0, 0.0])
        radius = 15.0

        result_indices, result_dists = query_nearby_endpoints_numba(
            endpoint_coords, candidate_indices, point_coords, radius
        )

        assert len(result_indices) == 2
        # Index 2 (distance 1m) should NOT be in results
        assert 2 not in result_indices


class TestComputeVertexDensityOptionalCoords:
    """Tests for compute_vertex_density with optional coords parameter."""

    def test_basic_density(self):
        """Should compute correct vertex density."""
        # Line with 3 vertices over 100m = 0.03 vertices/meter
        line = LineString([(0, 0), (50, 0), (100, 0)])
        result = compute_vertex_density(line)
        assert result == pytest.approx(3 / 100)

    def test_with_pre_extracted_coords(self):
        """Should work with pre-extracted coordinates."""
        line = LineString([(0, 0), (50, 0), (100, 0)])
        coords = np.array(line.coords)

        result = compute_vertex_density(line, coords=coords)
        assert result == pytest.approx(3 / 100)

    def test_pre_extracted_matches_auto_extracted(self):
        """Pre-extracted coords should give same result as auto-extraction."""
        line = LineString([(0, 0), (25, 10), (50, 0), (75, 10), (100, 0)])

        result_auto = compute_vertex_density(line)

        coords = np.array(line.coords)
        result_pre = compute_vertex_density(line, coords=coords)

        assert result_auto == pytest.approx(result_pre)

    def test_empty_line(self):
        """Empty line should return 0."""
        line = LineString()
        result = compute_vertex_density(line)
        assert result == 0.0

    def test_high_density_line(self):
        """Line with many vertices should have higher density."""
        # 11 vertices over 100m = 0.11 vertices/meter
        coords = [(i * 10, 0) for i in range(11)]
        line = LineString(coords)
        result = compute_vertex_density(line)
        assert result == pytest.approx(11 / 100)


class TestComputeAngleHistogramNumba:
    """Tests for compute_angle_histogram_numba JIT function."""

    def test_straight_line_all_in_first_bin(self):
        """Straight line should have all turns in first bin (0° turns)."""
        from matcher.features._jit_helpers import compute_angle_histogram_numba

        # Perfectly straight line - all turn angles should be ~0
        coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
        histogram = compute_angle_histogram_numba(coords)

        # All turns should be in first bin (0-22.5°)
        assert histogram[0] == pytest.approx(1.0)
        assert histogram.sum() == pytest.approx(1.0)

    def test_right_angle_turns(self):
        """Line with 90° turns should have turns in the 90° bin."""
        from matcher.features._jit_helpers import compute_angle_histogram_numba

        # Zigzag with 90° turns
        coords = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [20.0, 10.0]])
        histogram = compute_angle_histogram_numba(coords)

        # 90° falls in bin 4 (90/22.5 = 4)
        # Two 90° turns -> bin 4 should have all the mass
        assert histogram[4] == pytest.approx(1.0)
        assert histogram.sum() == pytest.approx(1.0)

    def test_mixed_angles(self):
        """Line with mixed turn angles should distribute across bins."""
        from matcher.features._jit_helpers import compute_angle_histogram_numba

        # One small turn (~15°) and one larger turn (~60°)
        # Point 1 -> 2: heading ~0°
        # Point 2 -> 3: heading ~15° (small turn)
        # Point 3 -> 4: heading ~75° (60° turn from 15°)
        coords = np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [20.0, 2.68],  # ~15° turn
                [25.0, 12.0],  # ~60° turn
            ]
        )
        histogram = compute_angle_histogram_numba(coords)

        # Should have distribution across multiple bins
        assert histogram.sum() == pytest.approx(1.0)
        # Should not be all in one bin
        assert np.count_nonzero(histogram) >= 2

    def test_too_few_points_returns_zeros(self):
        """Lines with fewer than 3 points should return zeros."""
        from matcher.features._jit_helpers import compute_angle_histogram_numba

        # 2 points - no turns possible
        coords = np.array([[0.0, 0.0], [10.0, 0.0]])
        histogram = compute_angle_histogram_numba(coords)

        assert histogram.sum() == pytest.approx(0.0)
        assert len(histogram) == 8  # Default 8 bins

    def test_histogram_is_normalized(self):
        """Histogram should sum to 1.0 for valid lines."""
        from matcher.features._jit_helpers import compute_angle_histogram_numba

        # Complex line with multiple turns
        coords = np.array(
            [[0.0, 0.0], [10.0, 5.0], [20.0, 0.0], [30.0, 10.0], [40.0, 5.0], [50.0, 15.0]]
        )
        histogram = compute_angle_histogram_numba(coords)

        assert histogram.sum() == pytest.approx(1.0)


class TestHistogramIntersectionNumba:
    """Tests for histogram_intersection_numba JIT function."""

    def test_identical_histograms(self):
        """Identical histograms should have intersection of 1.0."""
        from matcher.features._jit_helpers import histogram_intersection_numba

        h1 = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        h2 = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = histogram_intersection_numba(h1, h2)
        assert result == pytest.approx(1.0)

    def test_disjoint_histograms(self):
        """Completely disjoint histograms should have intersection of 0.0."""
        from matcher.features._jit_helpers import histogram_intersection_numba

        h1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        result = histogram_intersection_numba(h1, h2)
        assert result == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partially overlapping histograms should have intermediate intersection."""
        from matcher.features._jit_helpers import histogram_intersection_numba

        h1 = np.array([0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        h2 = np.array([0.4, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])

        # min(0.6, 0.4) + min(0.4, 0.4) + min(0, 0.2) = 0.4 + 0.4 + 0 = 0.8
        result = histogram_intersection_numba(h1, h2)
        assert result == pytest.approx(0.8)

    def test_uniform_histograms(self):
        """Uniform histograms should have intersection of 1.0."""
        from matcher.features._jit_helpers import histogram_intersection_numba

        h1 = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
        h2 = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])

        result = histogram_intersection_numba(h1, h2)
        assert result == pytest.approx(1.0)


class TestComputeCrossingAngleStatsNumba:
    """Tests for compute_crossing_angle_stats_numba JIT function."""

    def test_perpendicular_single_sample_single_neighbor(self):
        """Single perpendicular pair should return ~90°."""
        # Candidate heading: 90° (north)
        candidate_headings = np.array([90.0])
        # Neighbor heading: 0° (east) -> bidirectional delta = 90°
        neighbor_headings = np.array([0.0])

        min_a, mean_a, std_a, trans_frac = compute_crossing_angle_stats_numba(
            candidate_headings, neighbor_headings, 60.0, 90.0
        )

        assert min_a == pytest.approx(90.0)
        assert mean_a == pytest.approx(90.0)
        assert std_a == pytest.approx(0.0)
        assert trans_frac == pytest.approx(1.0)

    def test_parallel_headings(self):
        """Parallel headings should return ~0°."""
        candidate_headings = np.array([0.0, 0.0, 0.0])
        neighbor_headings = np.array([0.0])

        min_a, mean_a, std_a, trans_frac = compute_crossing_angle_stats_numba(
            candidate_headings, neighbor_headings, 60.0, 0.0
        )

        assert min_a == pytest.approx(0.0)
        assert mean_a == pytest.approx(0.0)
        assert trans_frac == pytest.approx(0.0)

    def test_opposite_direction_is_parallel(self):
        """Headings 0° and 180° should be treated as parallel (bidirectional)."""
        candidate_headings = np.array([0.0])
        neighbor_headings = np.array([180.0])

        min_a, mean_a, std_a, trans_frac = compute_crossing_angle_stats_numba(
            candidate_headings, neighbor_headings, 60.0, 0.0
        )

        assert min_a == pytest.approx(0.0)
        assert mean_a == pytest.approx(0.0)

    def test_varying_sample_headings(self):
        """Headings that vary along segment should produce non-zero std."""
        # Simulates a ramp that starts parallel (0°) and veers to 45°
        candidate_headings = np.array([0.0, 5.0, 10.0, 20.0, 35.0, 45.0])
        # Corridor running east-west
        neighbor_headings = np.array([0.0])

        min_a, mean_a, std_a, trans_frac = compute_crossing_angle_stats_numba(
            candidate_headings, neighbor_headings, 60.0, 0.0
        )

        # Min should be 0° (first sample is parallel)
        assert min_a == pytest.approx(0.0)
        # Mean should be between 0 and 45
        assert 10.0 < mean_a < 35.0
        # Std should be elevated
        assert std_a > 5.0

    def test_multiple_neighbors_takes_min(self):
        """Per-sample angle should be the min across all neighbors."""
        # Candidate heading: 45° (northeast)
        candidate_headings = np.array([45.0])
        # Neighbor 1: 0° (east) -> 45° delta
        # Neighbor 2: 90° (north) -> 45° delta
        # Neighbor 3: 30° -> 15° delta (closest)
        neighbor_headings = np.array([0.0, 90.0, 30.0])

        min_a, mean_a, std_a, trans_frac = compute_crossing_angle_stats_numba(
            candidate_headings, neighbor_headings, 60.0, 45.0
        )

        # Should pick the closest neighbor (30° -> 15° delta)
        assert min_a == pytest.approx(15.0)

    def test_transverse_fraction_threshold(self):
        """Transverse fraction should respect threshold."""
        candidate_headings = np.array([0.0])
        # Two neighbors: one perpendicular (90°), one at 50° from candidate
        neighbor_headings = np.array([90.0, 50.0])

        # With threshold 60°: only the 90° neighbor is transverse
        _, _, _, trans_frac = compute_crossing_angle_stats_numba(
            candidate_headings, neighbor_headings, 60.0, 0.0
        )
        assert trans_frac == pytest.approx(0.5)  # 1 of 2

        # With threshold 40°: both are transverse
        _, _, _, trans_frac = compute_crossing_angle_stats_numba(
            candidate_headings, neighbor_headings, 40.0, 0.0
        )
        assert trans_frac == pytest.approx(1.0)  # 2 of 2
