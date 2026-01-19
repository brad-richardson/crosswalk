"""Tests for linestring alignment functions."""

import numpy as np
import pytest
from shapely.geometry import LineString

from matcher.features.alignment import (
    AlignmentResult,
    _interpolate_along_line,
    _prepare_line_data,
    compute_coverage_features,
    create_subline,
    linestring_alignment,
    walk_distance,
    walk_parallelness,
)


class TestInterpolateAlongLine:
    """Tests for _interpolate_along_line boundary conditions."""

    @pytest.fixture
    def simple_line_data(self):
        """Simple 3-point line: (0,0) -> (10,0) -> (10,10) with total length 20."""
        line = LineString([(0, 0), (10, 0), (10, 10)])
        coords, distances, total_length = _prepare_line_data(line)
        return coords, distances, total_length

    @pytest.mark.parametrize(
        "t,expected_x,expected_y,description",
        [
            (-5.0, 0.0, 0.0, "negative t returns first point"),
            (0.0, 0.0, 0.0, "t=0 returns first point"),
            (5.0, 5.0, 0.0, "mid-first-segment interpolation"),
            (10.0, 10.0, 0.0, "at first vertex"),
            (15.0, 10.0, 5.0, "mid-second-segment interpolation"),
            (20.0, 10.0, 10.0, "t=max returns last point"),
            (25.0, 10.0, 10.0, "t > max returns last point"),
        ],
        ids=[
            "negative_t",
            "t_zero",
            "mid_first_segment",
            "at_vertex",
            "mid_second_segment",
            "t_max",
            "t_beyond_max",
        ],
    )
    def test_interpolation_boundary_conditions(
        self, simple_line_data, t, expected_x, expected_y, description
    ):
        """Verify interpolation boundary conditions."""
        coords, distances, _ = simple_line_data
        x, y = _interpolate_along_line(coords, distances, t)
        assert x == pytest.approx(expected_x, abs=0.001), description
        assert y == pytest.approx(expected_y, abs=0.001), description


class TestGetScoreNumba:
    """Tests for _get_score_numba scoring behavior.

    We test indirectly through linestring_alignment since the internal
    scoring function is tightly coupled to the alignment algorithm.
    """

    @pytest.mark.parametrize(
        "ref_coords,target_coords,expected_behavior",
        [
            # No overlap: target is far from reference
            (
                [(0, 0), (100, 0)],
                [(500, 500), (600, 500)],
                "low",
            ),
            # Perfect overlap: identical lines
            (
                [(0, 0), (100, 0)],
                [(0, 0), (100, 0)],
                "high",
            ),
            # Parallel offset: same direction, 10m lateral offset
            (
                [(0, 0), (100, 0)],
                [(0, 10), (100, 10)],
                "medium_high",
            ),
            # Opposite direction: same geometry, reversed
            (
                [(0, 0), (100, 0)],
                [(100, 0), (0, 0)],
                "high",
            ),
        ],
        ids=[
            "no_overlap",
            "perfect_overlap",
            "parallel_offset",
            "opposite_direction",
        ],
    )
    def test_scoring_behavior(self, ref_coords, target_coords, expected_behavior):
        """Test that alignment scoring matches expected behavior categories."""
        ref = LineString(ref_coords)
        target = LineString(target_coords)
        result = linestring_alignment(ref, target)

        if expected_behavior == "low":
            # No overlap should result in minimal coverage
            assert result.overture_coverage < 0.3 or result.dataset_coverage < 0.3
        elif expected_behavior == "high":
            # Perfect match should have high coverage
            assert result.overture_coverage > 0.9
            assert result.dataset_coverage > 0.9
        elif expected_behavior == "medium_high":
            # Parallel offset should still have good coverage
            assert result.overture_coverage > 0.8
            assert result.dataset_coverage > 0.8


class TestLinestringAlignment:
    """Tests for linestring_alignment high-level alignment."""

    @pytest.mark.parametrize(
        "scenario,ref_coords,target_coords,checks",
        [
            # Identical lines
            (
                "identical_lines",
                [(0, 0), (100, 0)],
                [(0, 0), (100, 0)],
                {"ref_coverage_min": 0.99, "target_coverage_min": 0.99},
            ),
            # Reversed target (should detect and handle)
            (
                "reversed_target",
                [(0, 0), (100, 0)],
                [(100, 0), (0, 0)],
                {"ref_coverage_min": 0.99, "target_coverage_min": 0.99},
            ),
            # Target covers first half of reference
            (
                "target_covers_first_half",
                [(0, 0), (100, 0)],
                [(0, 0), (50, 0)],
                {"ref_coverage_range": (0.4, 0.6), "target_coverage_min": 0.9},
            ),
            # Target covers second half of reference
            (
                "target_covers_second_half",
                [(0, 0), (100, 0)],
                [(50, 0), (100, 0)],
                {"ref_coverage_range": (0.4, 0.6), "target_coverage_min": 0.9},
            ),
            # Target extends beyond reference on both sides
            (
                "target_extends_beyond",
                [(25, 0), (75, 0)],
                [(0, 0), (100, 0)],
                {"ref_coverage_min": 0.9, "target_coverage_range": (0.4, 0.6)},
            ),
            # Parallel offset lines (10m apart)
            (
                "parallel_offset_lines",
                [(0, 0), (100, 0)],
                [(0, 10), (100, 10)],
                {"ref_coverage_min": 0.9, "target_coverage_min": 0.9},
            ),
        ],
        ids=[
            "identical_lines",
            "reversed_target",
            "target_covers_first_half",
            "target_covers_second_half",
            "target_extends_beyond",
            "parallel_offset_lines",
        ],
    )
    def test_alignment_scenarios(self, scenario, ref_coords, target_coords, checks):
        """Test various alignment scenarios."""
        ref = LineString(ref_coords)
        target = LineString(target_coords)
        result = linestring_alignment(ref, target)

        if "ref_coverage_min" in checks:
            assert result.overture_coverage >= checks["ref_coverage_min"], (
                f"{scenario}: ref_coverage {result.overture_coverage} < {checks['ref_coverage_min']}"
            )

        if "target_coverage_min" in checks:
            assert result.dataset_coverage >= checks["target_coverage_min"], (
                f"{scenario}: target_coverage {result.dataset_coverage} < {checks['target_coverage_min']}"
            )

        if "ref_coverage_range" in checks:
            low, high = checks["ref_coverage_range"]
            assert low <= result.overture_coverage <= high, (
                f"{scenario}: ref_coverage {result.overture_coverage} not in [{low}, {high}]"
            )

        if "target_coverage_range" in checks:
            low, high = checks["target_coverage_range"]
            assert low <= result.dataset_coverage <= high, (
                f"{scenario}: target_coverage {result.dataset_coverage} not in [{low}, {high}]"
            )

    def test_zero_length_reference(self):
        """Zero-length reference should return default alignment."""
        ref = LineString([(50, 50), (50, 50)])
        target = LineString([(0, 0), (100, 0)])
        result = linestring_alignment(ref, target)

        assert result.overture_start_frac == 0.0
        assert result.overture_end_frac == 1.0

    def test_zero_length_target(self):
        """Zero-length target should return default alignment."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(50, 50), (50, 50)])
        result = linestring_alignment(ref, target)

        assert result.dataset_start_frac == 0.0
        assert result.dataset_end_frac == 1.0

    def test_multi_segment_alignment(self):
        """Test alignment with multi-segment lines."""
        # Zigzag reference
        ref = LineString([(0, 0), (25, 10), (50, 0), (75, 10), (100, 0)])
        # Straight target that covers middle portion
        target = LineString([(20, 5), (80, 5)])

        result = linestring_alignment(ref, target)

        # Should find some alignment
        assert result.overture_coverage > 0.3
        assert result.dataset_coverage > 0.3


class TestCreateSubline:
    """Tests for create_subline extraction."""

    @pytest.fixture
    def reference_line(self):
        """100-meter line along x-axis."""
        return LineString([(0, 0), (100, 0)])

    @pytest.mark.parametrize(
        "start_frac,end_frac,expected_length",
        [
            (0.0, 1.0, 100.0),  # Full line
            (0.0, 0.5, 50.0),  # First half
            (0.5, 1.0, 50.0),  # Second half
            (0.25, 0.75, 50.0),  # Middle half
            (0.0, 0.1, 10.0),  # First 10%
            (0.9, 1.0, 10.0),  # Last 10%
        ],
        ids=[
            "full_line",
            "first_half",
            "second_half",
            "middle_half",
            "first_10pct",
            "last_10pct",
        ],
    )
    def test_subline_extraction(self, reference_line, start_frac, end_frac, expected_length):
        """Test subline extraction with various fractions."""
        result = create_subline(reference_line, start_frac, end_frac)
        assert result is not None
        assert result.length == pytest.approx(expected_length, abs=0.01)

    def test_swapped_fractions_auto_corrected(self, reference_line):
        """Start > end fractions should be auto-corrected."""
        result = create_subline(reference_line, 0.75, 0.25)
        assert result is not None
        assert result.length == pytest.approx(50.0, abs=0.01)

    @pytest.mark.parametrize(
        "start_frac,end_frac,expected_clamped_length",
        [
            (-0.5, 0.5, 50.0),  # Negative start clamped to 0
            (0.5, 1.5, 50.0),  # End > 1 clamped to 1
            (-1.0, 2.0, 100.0),  # Both out of range clamped
        ],
        ids=["negative_start", "end_beyond_one", "both_out_of_range"],
    )
    def test_fraction_clamping(self, reference_line, start_frac, end_frac, expected_clamped_length):
        """Out-of-range fractions should be clamped to [0, 1]."""
        result = create_subline(reference_line, start_frac, end_frac)
        assert result is not None
        assert result.length == pytest.approx(expected_clamped_length, abs=0.01)

    def test_none_input(self):
        """None input should return None."""
        result = create_subline(None, 0.0, 1.0)
        assert result is None

    def test_empty_geometry(self):
        """Empty geometry should return None."""
        empty = LineString()
        result = create_subline(empty, 0.0, 1.0)
        assert result is None

    def test_zero_length_line(self):
        """Zero-length line (point) should return None."""
        point_line = LineString([(50, 50), (50, 50)])
        result = create_subline(point_line, 0.0, 1.0)
        assert result is None


class TestWalkDistance:
    """Tests for walk_distance integrated Euclidean distance."""

    def test_identical_lines_zero_distance(self):
        """Identical lines should have zero walk distance."""
        line = LineString([(0, 0), (100, 0)])
        dist = walk_distance(line, line)
        assert dist == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize(
        "offset_y,expected_distance",
        [
            (10, 10.0),  # 10m parallel offset
            (20, 20.0),  # 20m parallel offset
            (5, 5.0),  # 5m parallel offset
        ],
        ids=["10m_offset", "20m_offset", "5m_offset"],
    )
    def test_parallel_offset_distance(self, offset_y, expected_distance):
        """Parallel lines should have walk distance equal to offset."""
        L1 = LineString([(0, 0), (100, 0)])
        L2 = LineString([(0, offset_y), (100, offset_y)])
        dist = walk_distance(L1, L2)
        assert dist == pytest.approx(expected_distance, abs=0.5)

    def test_different_length_lines(self):
        """Walk distance with different length lines."""
        L1 = LineString([(0, 0), (100, 0)])  # 100m
        L2 = LineString([(0, 0), (200, 0)])  # 200m

        dist = walk_distance(L1, L2)
        # Proportional sampling means points diverge
        assert dist > 0  # They diverge as sampling proceeds

    def test_samples_parameter(self):
        """More samples should give similar result for simple geometry."""
        L1 = LineString([(0, 0), (100, 0)])
        L2 = LineString([(0, 10), (100, 10)])

        dist_8 = walk_distance(L1, L2, samples=8)
        dist_16 = walk_distance(L1, L2, samples=16)
        dist_32 = walk_distance(L1, L2, samples=32)

        # All should be approximately 10m for parallel offset
        assert dist_8 == pytest.approx(10.0, abs=0.5)
        assert dist_16 == pytest.approx(10.0, abs=0.5)
        assert dist_32 == pytest.approx(10.0, abs=0.5)


class TestWalkParallelness:
    """Tests for walk_parallelness squared dot product."""

    def test_identical_lines_max_parallelness(self):
        """Identical lines should have parallelness of 1.0."""
        line = LineString([(0, 0), (100, 0)])
        par = walk_parallelness(line, line)
        assert par == pytest.approx(1.0, abs=0.01)

    def test_parallel_lines_max_parallelness(self):
        """Parallel offset lines should have parallelness of 1.0."""
        L1 = LineString([(0, 0), (100, 0)])
        L2 = LineString([(0, 10), (100, 10)])
        par = walk_parallelness(L1, L2)
        assert par == pytest.approx(1.0, abs=0.01)

    def test_opposite_direction_still_parallel(self):
        """Lines going opposite directions are still parallel (squared dot)."""
        L1 = LineString([(0, 0), (100, 0)])
        L2 = LineString([(100, 0), (0, 0)])  # Reversed
        par = walk_parallelness(L1, L2)
        # Squared dot product makes opposite directions still = 1.0
        assert par == pytest.approx(1.0, abs=0.01)

    def test_perpendicular_lines_zero_parallelness(self):
        """Perpendicular lines should have parallelness near 0.0."""
        L1 = LineString([(0, 0), (100, 0)])
        L2 = LineString([(0, 0), (0, 100)])  # 90 degrees
        par = walk_parallelness(L1, L2)
        assert par == pytest.approx(0.0, abs=0.05)

    @pytest.mark.parametrize(
        "angle_degrees,expected_parallelness",
        [
            (0, 1.0),  # Parallel
            (30, 0.75),  # cos^2(30) = 0.75
            (45, 0.5),  # cos^2(45) = 0.5
            (60, 0.25),  # cos^2(60) = 0.25
            (90, 0.0),  # Perpendicular
        ],
        ids=["0_deg", "30_deg", "45_deg", "60_deg", "90_deg"],
    )
    def test_angle_vs_parallelness(self, angle_degrees, expected_parallelness):
        """Test parallelness at various angles."""
        L1 = LineString([(0, 0), (100, 0)])

        # Compute endpoint for angled line
        angle_rad = np.radians(angle_degrees)
        end_x = 100 * np.cos(angle_rad)
        end_y = 100 * np.sin(angle_rad)
        L2 = LineString([(0, 0), (end_x, end_y)])

        par = walk_parallelness(L1, L2)
        assert par == pytest.approx(expected_parallelness, abs=0.05)


class TestAlignmentResult:
    """Tests for AlignmentResult dataclass properties."""

    def test_overture_coverage_computation(self):
        """Test overture_coverage property."""
        result = AlignmentResult(
            overture_start_frac=0.25,
            overture_end_frac=0.75,
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )
        assert result.overture_coverage == pytest.approx(0.5)

    def test_dataset_coverage_computation(self):
        """Test dataset_coverage property."""
        result = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=0.1,
            dataset_end_frac=0.9,
        )
        assert result.dataset_coverage == pytest.approx(0.8)

    def test_full_coverage(self):
        """Full alignment should have coverage of 1.0."""
        result = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )
        assert result.overture_coverage == pytest.approx(1.0)
        assert result.dataset_coverage == pytest.approx(1.0)


class TestPrepareLineData:
    """Tests for _prepare_line_data helper."""

    def test_simple_line(self):
        """Test data preparation for simple line."""
        line = LineString([(0, 0), (100, 0)])
        coords, distances, total_length = _prepare_line_data(line)

        assert coords.shape == (2, 2)
        assert len(distances) == 2
        assert distances[0] == 0.0
        assert distances[1] == pytest.approx(100.0)
        assert total_length == pytest.approx(100.0)

    def test_multi_segment_line(self):
        """Test data preparation for multi-segment line."""
        # L-shaped line: 100m right, then 50m up
        line = LineString([(0, 0), (100, 0), (100, 50)])
        coords, distances, total_length = _prepare_line_data(line)

        assert coords.shape == (3, 2)
        assert len(distances) == 3
        assert distances[0] == 0.0
        assert distances[1] == pytest.approx(100.0)
        assert distances[2] == pytest.approx(150.0)
        assert total_length == pytest.approx(150.0)

    def test_diagonal_line(self):
        """Test data preparation for diagonal line."""
        line = LineString([(0, 0), (30, 40)])  # 3-4-5 triangle scaled by 10
        coords, distances, total_length = _prepare_line_data(line)

        assert total_length == pytest.approx(50.0)  # sqrt(30^2 + 40^2) = 50


class TestComputeCoverageFeatures:
    """Tests for compute_coverage_features function."""

    def test_full_coverage(self):
        """Full alignment should have coverage of 1.0."""
        alignment = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )
        features = compute_coverage_features(alignment)

        assert features["ref_coverage"] == pytest.approx(1.0)
        assert features["target_coverage"] == pytest.approx(1.0)
        assert features["min_coverage"] == pytest.approx(1.0)
        assert features["coverage_ratio"] == pytest.approx(1.0)

    def test_partial_ref_coverage(self):
        """Partial reference coverage should be computed correctly."""
        alignment = AlignmentResult(
            overture_start_frac=0.25,
            overture_end_frac=0.75,
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )
        features = compute_coverage_features(alignment)

        assert features["ref_coverage"] == pytest.approx(0.5)
        assert features["target_coverage"] == pytest.approx(1.0)
        assert features["min_coverage"] == pytest.approx(0.5)
        assert features["coverage_ratio"] == pytest.approx(0.5)

    def test_partial_target_coverage(self):
        """Partial target coverage should be computed correctly."""
        alignment = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=0.1,
            dataset_end_frac=0.9,
        )
        features = compute_coverage_features(alignment)

        assert features["ref_coverage"] == pytest.approx(1.0)
        assert features["target_coverage"] == pytest.approx(0.8)
        assert features["min_coverage"] == pytest.approx(0.8)
        assert features["coverage_ratio"] == pytest.approx(0.8)

    def test_both_partial_coverage(self):
        """Both partial coverages should compute min correctly."""
        alignment = AlignmentResult(
            overture_start_frac=0.25,
            overture_end_frac=0.75,  # 0.5 coverage
            dataset_start_frac=0.1,
            dataset_end_frac=0.7,  # 0.6 coverage
        )
        features = compute_coverage_features(alignment)

        assert features["ref_coverage"] == pytest.approx(0.5)
        assert features["target_coverage"] == pytest.approx(0.6)
        assert features["min_coverage"] == pytest.approx(0.5)
        assert features["coverage_ratio"] == pytest.approx(0.5 / 0.6, rel=0.01)

    def test_none_alignment(self):
        """None alignment should return zeros."""
        features = compute_coverage_features(None)

        assert features["ref_coverage"] == 0.0
        assert features["target_coverage"] == 0.0
        assert features["min_coverage"] == 0.0
        assert features["coverage_ratio"] == 0.0

    def test_zero_coverage(self):
        """Zero coverage alignment should handle division by zero."""
        alignment = AlignmentResult(
            overture_start_frac=0.5,
            overture_end_frac=0.5,  # 0 coverage
            dataset_start_frac=0.5,
            dataset_end_frac=0.5,  # 0 coverage
        )
        features = compute_coverage_features(alignment)

        assert features["ref_coverage"] == 0.0
        assert features["target_coverage"] == 0.0
        assert features["min_coverage"] == 0.0
        assert features["coverage_ratio"] == 0.0


class TestAlignedFeatureComputation:
    """Tests for feature computation on aligned sublines."""

    def test_partial_overlap_alignment_improves_hausdorff(self):
        """Aligned features should have better hausdorff for partial overlaps.

        When two segments only partially overlap, computing Hausdorff on aligned
        sublines should give a smaller distance than on full geometries.
        """
        from matcher.features.geometric import compute_geometric_features

        # Reference: 100m line
        ref = LineString([(0, 0), (100, 0)])
        # Target: 50m line at same position as second half of reference
        target = LineString([(50, 2), (100, 2)])  # 2m lateral offset

        # Full geometry features
        full_features = compute_geometric_features(ref, target)

        # Compute alignment
        alignment = linestring_alignment(ref, target)

        # Extract sublines
        ref_subline = create_subline(
            ref, alignment.overture_start_frac, alignment.overture_end_frac
        )
        target_subline = create_subline(
            target, alignment.dataset_start_frac, alignment.dataset_end_frac
        )

        # Aligned subline features
        aligned_features = compute_geometric_features(ref_subline, target_subline)

        # Hausdorff on aligned sublines should be smaller (or equal)
        # because we're comparing comparable portions
        assert aligned_features.hausdorff_distance <= full_features.hausdorff_distance + 1

    def test_tip_to_tip_segments_low_coverage(self):
        """Tip-to-tip (consecutive) segments should have low coverage."""
        # Two consecutive segments on the same street
        segment_a = LineString([(0, 0), (100, 0)])
        segment_b = LineString([(100, 0), (200, 0)])

        alignment = linestring_alignment(segment_a, segment_b)
        coverage = compute_coverage_features(alignment)

        # Consecutive segments have minimal overlap
        assert coverage["min_coverage"] < 0.3

    def test_contained_segment_high_coverage(self):
        """Segment fully contained in another should have high coverage for shorter."""
        # Reference is longer
        ref = LineString([(0, 0), (100, 0)])
        # Target is contained within reference
        target = LineString([(25, 0), (75, 0)])

        alignment = linestring_alignment(ref, target)
        coverage = compute_coverage_features(alignment)

        # Target should have high coverage (fully aligned)
        assert coverage["target_coverage"] > 0.9
        # Reference has partial coverage (only where target exists)
        assert 0.4 < coverage["ref_coverage"] < 0.6
