"""Tests for batch geometric feature computation.

Validates numerical equivalence between the vectorized batch path
(compute_geometric_features_batch) and the single-pair wrapper
(compute_geometric_features).
"""

import numpy as np
import pytest
from shapely.geometry import LineString

from matcher.features.geometric import (
    BatchGeometricResult,
    compute_geometric_features,
    compute_geometric_features_batch,
)


def _make_line(coords):
    """Helper to create a LineString from coordinate tuples."""
    return LineString(coords)


# Test geometries in projected CRS (meters)
IDENTICAL_LINE = _make_line([(0, 0), (100, 0)])
PARALLEL_LINE = _make_line([(0, 5), (100, 5)])
PERPENDICULAR_LINE = _make_line([(50, -50), (50, 50)])
OFFSET_LINE = _make_line([(10, 3), (90, 3)])
SHORT_LINE = _make_line([(0, 0), (10, 0)])
LONG_LINE = _make_line([(0, 0), (500, 0)])
FAR_LINE = _make_line([(1000, 1000), (1100, 1000)])
DIAGONAL_LINE = _make_line([(0, 0), (100, 100)])
CURVED_LINE = _make_line([(0, 0), (30, 10), (70, -10), (100, 0)])

# Pairs for testing
TEST_PAIRS = [
    ("identical", IDENTICAL_LINE, IDENTICAL_LINE),
    ("parallel_close", IDENTICAL_LINE, PARALLEL_LINE),
    ("perpendicular", IDENTICAL_LINE, PERPENDICULAR_LINE),
    ("offset_shorter", IDENTICAL_LINE, OFFSET_LINE),
    ("short_vs_long", SHORT_LINE, LONG_LINE),
    ("far_apart", IDENTICAL_LINE, FAR_LINE),
    ("diagonal_vs_horizontal", IDENTICAL_LINE, DIAGONAL_LINE),
    ("curved_vs_straight", IDENTICAL_LINE, CURVED_LINE),
]


class TestBatchGeometricEquivalence:
    """Verify batch path produces identical results to single-pair path."""

    @pytest.fixture
    def batch_results(self):
        """Compute batch results for all test pairs at once."""
        lines_a = np.array([p[1] for p in TEST_PAIRS], dtype=object)
        lines_b = np.array([p[2] for p in TEST_PAIRS], dtype=object)
        return compute_geometric_features_batch(lines_a, lines_b)

    @pytest.fixture
    def single_results(self):
        """Compute single-pair results for all test pairs."""
        results = []
        for _, line_a, line_b in TEST_PAIRS:
            results.append(compute_geometric_features(line_a, line_b))
        return results

    def test_hausdorff_distance(self, batch_results, single_results):
        for i, (name, _, _) in enumerate(TEST_PAIRS):
            assert batch_results.hausdorff_distances[i] == pytest.approx(
                single_results[i].hausdorff_distance, abs=1e-6
            ), f"Hausdorff mismatch for {name}"

    def test_buffer_iou_15m(self, batch_results, single_results):
        for i, (name, _, _) in enumerate(TEST_PAIRS):
            assert batch_results.buffer_iou_15m[i] == pytest.approx(
                single_results[i].buffer_iou_15m, abs=1e-6
            ), f"Buffer IoU 15m mismatch for {name}"

    def test_buffer_iou_5m(self, batch_results, single_results):
        for i, (name, _, _) in enumerate(TEST_PAIRS):
            assert batch_results.buffer_iou_5m[i] == pytest.approx(
                single_results[i].buffer_iou_5m, abs=1e-6
            ), f"Buffer IoU 5m mismatch for {name}"

    def test_heading_delta(self, batch_results, single_results):
        for i, (name, _, _) in enumerate(TEST_PAIRS):
            assert batch_results.heading_deltas[i] == pytest.approx(
                single_results[i].heading_delta, abs=1e-6
            ), f"Heading delta mismatch for {name}"

    def test_length_ratio(self, batch_results, single_results):
        for i, (name, _, _) in enumerate(TEST_PAIRS):
            assert batch_results.length_ratios[i] == pytest.approx(
                single_results[i].length_ratio, abs=1e-6
            ), f"Length ratio mismatch for {name}"

    def test_centroid_distance(self, batch_results, single_results):
        for i, (name, _, _) in enumerate(TEST_PAIRS):
            assert batch_results.centroid_distances[i] == pytest.approx(
                single_results[i].centroid_distance, abs=1e-6
            ), f"Centroid distance mismatch for {name}"

    def test_overlap_ratio(self, batch_results, single_results):
        for i, (name, _, _) in enumerate(TEST_PAIRS):
            assert batch_results.overlap_ratios[i] == pytest.approx(
                single_results[i].overlap_ratio, abs=1e-6
            ), f"Overlap ratio mismatch for {name}"


class TestBatchGeometricEdgeCases:
    """Test edge cases for the batch geometric computation."""

    def test_single_pair(self):
        """Batch should work with a single pair."""
        arr_a = np.array([IDENTICAL_LINE], dtype=object)
        arr_b = np.array([PARALLEL_LINE], dtype=object)
        result = compute_geometric_features_batch(arr_a, arr_b)
        assert isinstance(result, BatchGeometricResult)
        assert len(result.hausdorff_distances) == 1

    def test_identical_lines_perfect_scores(self):
        """Identical lines should have zero distance and perfect IoU."""
        arr = np.array([IDENTICAL_LINE], dtype=object)
        result = compute_geometric_features_batch(arr, arr)
        assert result.hausdorff_distances[0] == pytest.approx(0.0, abs=1e-6)
        assert result.buffer_iou_15m[0] == pytest.approx(1.0, abs=1e-6)
        assert result.buffer_iou_5m[0] == pytest.approx(1.0, abs=1e-6)
        assert result.heading_deltas[0] == pytest.approx(0.0, abs=1e-6)
        assert result.length_ratios[0] == pytest.approx(1.0, abs=1e-6)
        assert result.centroid_distances[0] == pytest.approx(0.0, abs=1e-6)
        assert result.overlap_ratios[0] == pytest.approx(1.0, abs=1e-6)

    def test_far_apart_lines_zero_iou(self):
        """Lines far apart should have zero IoU."""
        arr_a = np.array([IDENTICAL_LINE], dtype=object)
        arr_b = np.array([FAR_LINE], dtype=object)
        result = compute_geometric_features_batch(arr_a, arr_b)
        assert result.buffer_iou_15m[0] == pytest.approx(0.0, abs=1e-6)
        assert result.buffer_iou_5m[0] == pytest.approx(0.0, abs=1e-6)

    def test_perpendicular_lines_90_degree_heading(self):
        """Perpendicular lines should have ~90 degree heading delta reduced to ~0 for bidirectional."""
        # Both 0 and 90 degrees should map to small values due to bidirectional handling
        arr_a = np.array([IDENTICAL_LINE], dtype=object)
        arr_b = np.array([PERPENDICULAR_LINE], dtype=object)
        result = compute_geometric_features_batch(arr_a, arr_b)
        # Perpendicular: heading_a=0, heading_b=90. diff=90, opposite_diff=90.
        # min(90, 90) = 90. But with bidirectional: min(90, |180-90|) = min(90, 90) = 90
        assert result.heading_deltas[0] == pytest.approx(90.0, abs=1.0)

    def test_5m_short_circuit(self):
        """Pairs with low 15m IoU should have zero 5m IoU (short-circuit)."""
        arr_a = np.array([FAR_LINE], dtype=object)
        arr_b = np.array([IDENTICAL_LINE], dtype=object)
        result = compute_geometric_features_batch(arr_a, arr_b)
        # 15m IoU is 0 for far-apart lines, so 5m should be short-circuited to 0
        assert result.buffer_iou_15m[0] == pytest.approx(0.0, abs=1e-6)
        assert result.buffer_iou_5m[0] == pytest.approx(0.0, abs=1e-6)

    def test_result_shapes(self):
        """All result arrays should have the correct shape."""
        N = 5
        arr_a = np.array([IDENTICAL_LINE] * N, dtype=object)
        arr_b = np.array([PARALLEL_LINE] * N, dtype=object)
        result = compute_geometric_features_batch(arr_a, arr_b)

        for field_name in BatchGeometricResult._fields:
            arr = getattr(result, field_name)
            assert arr.shape == (N,), f"{field_name} has wrong shape: {arr.shape}"


class TestPhysicalOverlapM:
    """Tests for compute_physical_overlap_m() function.

    This function measures actual geometric intersection length without
    alignment translation. It's used as an early filter to remove
    tip-contact candidates.
    """

    def test_identical_lines_full_overlap(self):
        """Identical lines should have full overlap."""
        from matcher.features.geometric import compute_physical_overlap_m

        line = _make_line([(0, 0), (100, 0)])
        overlap = compute_physical_overlap_m(line, line)
        assert overlap == pytest.approx(100.0, abs=1.0)

    def test_parallel_lines_within_buffer(self):
        """Parallel lines within buffer distance should have high overlap."""
        from matcher.features.geometric import compute_physical_overlap_m

        line1 = _make_line([(0, 0), (100, 0)])
        line2 = _make_line([(0, 3), (100, 3)])  # 3m offset, within 5m buffer
        overlap = compute_physical_overlap_m(line1, line2, buffer_m=5.0)
        assert overlap > 90  # Should capture most of the line

    def test_parallel_lines_outside_buffer(self):
        """Parallel lines outside buffer distance should have zero overlap."""
        from matcher.features.geometric import compute_physical_overlap_m

        line1 = _make_line([(0, 0), (100, 0)])
        line2 = _make_line([(0, 20), (100, 20)])  # 20m offset, outside 5m buffer
        overlap = compute_physical_overlap_m(line1, line2, buffer_m=5.0)
        assert overlap == pytest.approx(0.0, abs=0.1)

    def test_tip_to_tip_collinear_minimal_overlap(self):
        """Collinear segments touching at tips should have minimal overlap."""
        from matcher.features.geometric import compute_physical_overlap_m

        line1 = _make_line([(0, 0), (100, 0)])
        line2 = _make_line([(100, 0), (200, 0)])  # Continuation, tip-to-tip
        overlap = compute_physical_overlap_m(line1, line2, buffer_m=5.0)
        # Only the buffer around the endpoint provides overlap
        assert overlap < 10  # Should be ~5m (buffer radius at tip)

    def test_partial_overlap(self):
        """Partially overlapping collinear lines should have proportional overlap."""
        from matcher.features.geometric import compute_physical_overlap_m

        line1 = _make_line([(0, 0), (100, 0)])
        line2 = _make_line([(50, 0), (150, 0)])  # 50m overlap
        overlap = compute_physical_overlap_m(line1, line2, buffer_m=5.0)
        assert overlap > 45  # Should capture most of the 50m overlap

    def test_crossing_intersection(self):
        """Crossing lines should have minimal overlap at the intersection point."""
        from matcher.features.geometric import compute_physical_overlap_m

        line1 = _make_line([(0, 0), (100, 0)])
        line2 = _make_line([(50, -50), (50, 50)])  # Perpendicular crossing at midpoint
        overlap = compute_physical_overlap_m(line1, line2, buffer_m=5.0)
        # Only ~10m of line1 falls within the 5m buffer of line2 at crossing
        assert 5 < overlap < 20

    def test_disjoint_lines_zero_overlap(self):
        """Completely disjoint lines should have zero overlap."""
        from matcher.features.geometric import compute_physical_overlap_m

        line1 = _make_line([(0, 0), (100, 0)])
        line2 = _make_line([(200, 0), (300, 0)])  # Far away
        overlap = compute_physical_overlap_m(line1, line2, buffer_m=5.0)
        assert overlap == pytest.approx(0.0, abs=0.1)

    def test_small_gap_between_collinear(self):
        """Collinear lines with a small gap should have low overlap."""
        from matcher.features.geometric import compute_physical_overlap_m

        line1 = _make_line([(0, 0), (100, 0)])
        line2 = _make_line([(110, 0), (200, 0)])  # 10m gap
        overlap = compute_physical_overlap_m(line1, line2, buffer_m=5.0)
        # Gap is larger than buffer, so no overlap
        assert overlap == pytest.approx(0.0, abs=0.1)
