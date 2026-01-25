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
    compute_endpoint_proximity_numba,
    compute_heading_consistency_numba,
    compute_heading_numba,
    compute_parallel_alignment_numba,
    compute_shape_complexity_numba,
)
from matcher.features.geometric import compute_collinear_gap_ratio, compute_shape_complexity
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
        """Tip-to-tip segments should have low overlap score."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[100.0, 0.0], [200.0, 0.0]])

        result = collinear_gap_ratio_numba(
            coords_a, coords_b, heading_threshold=15.0, min_overlap_fraction=0.1
        )
        # Tip-to-tip with 0 overlap should return 0.0
        assert result == pytest.approx(0.0, abs=0.01)

    def test_full_overlap(self):
        """Fully overlapping segments should return 1.0."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[25.0, 0.0], [75.0, 0.0]])

        result = collinear_gap_ratio_numba(
            coords_a, coords_b, heading_threshold=15.0, min_overlap_fraction=0.1
        )
        assert result == pytest.approx(1.0)

    def test_partial_overlap_above_threshold(self):
        """Partial overlap above threshold should return 1.0."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[80.0, 0.0], [180.0, 0.0]])

        result = collinear_gap_ratio_numba(
            coords_a, coords_b, heading_threshold=15.0, min_overlap_fraction=0.1
        )
        # 20m overlap on 100m segment = 20% overlap > 10% threshold
        assert result == pytest.approx(1.0)

    def test_opposite_heading_collinear(self):
        """Segments with opposite headings should still compute overlap correctly."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])
        coords_b = np.array([[50.0, 0.0], [25.0, 0.0]])  # Reversed direction

        result = collinear_gap_ratio_numba(
            coords_a, coords_b, heading_threshold=15.0, min_overlap_fraction=0.1
        )
        # Should detect overlap despite opposite heading
        assert result == pytest.approx(1.0)

    def test_non_collinear_returns_one(self):
        """Non-collinear segments should return 1.0 (no penalty)."""
        coords_a = np.array([[0.0, 0.0], [100.0, 0.0]])  # Horizontal
        coords_b = np.array([[50.0, 0.0], [50.0, 100.0]])  # Vertical

        result = collinear_gap_ratio_numba(
            coords_a, coords_b, heading_threshold=15.0, min_overlap_fraction=0.1
        )
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
