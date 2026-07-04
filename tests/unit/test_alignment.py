"""Tests for linestring alignment functions."""

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from matcher.features.alignment import (
    AlignmentResult,
    _compute_centroid,
    _create_local_equidistant_crs,
    _interpolate_along_line,
    _is_geographic,
    _nearest_frac_on_line,
    _prepare_line_data,
    compute_coverage_features,
    create_subline,
    geodetic_length,
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


class TestAlignmentOrientation:
    """Tests for the is_reversed flag and orientation-aware target end helpers.

    Regression coverage for the topology-orientation bug: when the best alignment
    is backward (target digitized opposite to ref), the target fractions are
    flipped into the target's own coordinate order, so dataset_start_frac points at
    the target end PHYSICALLY OPPOSITE the reference's from end.
    """

    def test_forward_alignment_not_reversed(self):
        """A forward-aligned control keeps is_reversed False and unswapped helpers."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (100, 0)])  # same digitization direction
        result = linestring_alignment(ref, target)

        assert result.is_reversed is False
        # Helpers pass through the raw fracs when forward.
        assert result.target_from_frac == result.dataset_start_frac
        assert result.target_to_frac == result.dataset_end_frac

        # target_from_frac must sit at the target end nearest ref's from end (0, 0).
        from_pt = target.interpolate(result.target_from_frac, normalized=True)
        assert from_pt.distance(Point(0, 0)) < 1.0

    def test_reversed_alignment_flags_and_swaps_physical_ends(self):
        """Synthetic reversed repro: ref (0,0)->(100,0), target (100,0)->(0,0).

        The raw dataset_start_frac points at the target's coord[0] = (100, 0), the
        OPPOSITE end from ref's from end (0, 0). target_from_frac must instead
        resolve to the target end physically nearest (0, 0).
        """
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(100, 0), (0, 0)])
        result = linestring_alignment(ref, target)

        assert result.is_reversed is True

        # The raw start frac points at the physically OPPOSITE end (the bug).
        raw_start_pt = target.interpolate(result.dataset_start_frac, normalized=True)
        assert raw_start_pt.distance(Point(100, 0)) < 1.0

        # The orientation-aware helper resolves to the physically correct end:
        # nearest to ref's from end (0, 0), and swaps from/to relative to raw.
        assert result.target_from_frac == result.dataset_end_frac
        assert result.target_to_frac == result.dataset_start_frac
        from_pt = target.interpolate(result.target_from_frac, normalized=True)
        assert from_pt.distance(Point(0, 0)) < 1.0
        to_pt = target.interpolate(result.target_to_frac, normalized=True)
        assert to_pt.distance(Point(100, 0)) < 1.0

    def test_default_is_reversed_false(self):
        """The dataclass defaults is_reversed to False for old pickles/callers."""
        result = AlignmentResult(0.0, 1.0, 0.0, 1.0)
        assert result.is_reversed is False
        assert result.target_from_frac == 0.0
        assert result.target_to_frac == 1.0


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

    def test_equal_fractions_returns_none(self, reference_line):
        """Equal start and end fractions should return None (would produce a Point)."""
        result = create_subline(reference_line, 0.5, 0.5)
        assert result is None

    def test_result_is_always_linestring_or_none(self, reference_line):
        """Return type is always LineString or None, never Point or other geometry."""
        fractions = [
            (0.0, 1.0),
            (0.0, 0.5),
            (0.3, 0.7),
            (0.5, 0.5),  # degenerate
            (0.0, 0.0),  # degenerate
            (1.0, 1.0),  # degenerate
        ]
        for start, end in fractions:
            result = create_subline(reference_line, start, end)
            assert result is None or isinstance(result, LineString), (
                f"create_subline({start}, {end}) returned {type(result)}"
            )


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

    def test_none_alignment_returns_zeros(self):
        """None alignment should return zeros by default."""
        features = compute_coverage_features(None)

        assert features["ref_coverage"] == 0.0
        assert features["target_coverage"] == 0.0
        assert features["min_coverage"] == 0.0
        assert features["coverage_ratio"] == 0.0

    def test_none_alignment_explicit_failure(self):
        """None alignment with return_none_on_failure=True should return Nones."""
        features = compute_coverage_features(None, return_none_on_failure=True)

        assert features["ref_coverage"] is None
        assert features["target_coverage"] is None
        assert features["min_coverage"] is None
        assert features["coverage_ratio"] is None

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


class TestGeodeticLength:
    """Tests for geodetic_length function using WGS84 ellipsoid."""

    def test_none_input(self):
        """None input should return 0.0."""
        assert geodetic_length(None) == 0.0

    def test_empty_geometry(self):
        """Empty geometry should return 0.0."""
        empty = LineString()
        assert geodetic_length(empty) == 0.0

    def test_point_geometry(self):
        """Zero-length line (point) should return 0.0."""
        point_line = LineString([(0, 0), (0, 0)])
        assert geodetic_length(point_line) == pytest.approx(0.0, abs=0.01)

    def test_known_distance_equator(self):
        """Test a known distance along the equator.

        1 degree of longitude at the equator is approximately 111,320 meters.
        """
        # Line along equator: 1 degree of longitude
        line = LineString([(0, 0), (1, 0)])
        length = geodetic_length(line)
        # Expected: ~111,320 meters (varies slightly due to ellipsoid)
        assert 111000 < length < 112000

    def test_known_distance_latitude(self):
        """Test a known distance along a meridian.

        1 degree of latitude is approximately 111,000-111,300 meters.
        """
        # Line along meridian: 1 degree of latitude
        line = LineString([(0, 0), (0, 1)])
        length = geodetic_length(line)
        # Expected: ~111,000 meters
        assert 110000 < length < 112000

    def test_multi_segment_line(self):
        """Test multi-segment line computes total length."""
        # L-shaped line: 0.01 degrees east, then 0.01 degrees north
        # At equator, this is approximately 1.1km each segment
        line = LineString([(0, 0), (0.01, 0), (0.01, 0.01)])
        length = geodetic_length(line)
        # Should be approximately 2.2km total
        assert 2000 < length < 2400

    def test_boston_street_segment(self):
        """Test realistic street segment in Boston area.

        A typical city block is 100-200 meters.
        """
        # Small street segment in Boston (approximate coordinates)
        # ~0.001 degrees longitude at 42N is about 80-90 meters
        line = LineString([(-71.05, 42.35), (-71.049, 42.35)])
        length = geodetic_length(line)
        # Expected: ~80-90 meters
        assert 70 < length < 100

    def test_always_positive(self):
        """Length should always be positive regardless of direction."""
        line1 = LineString([(0, 0), (1, 0)])
        line2 = LineString([(1, 0), (0, 0)])  # Reversed

        assert geodetic_length(line1) > 0
        assert geodetic_length(line2) > 0
        assert geodetic_length(line1) == pytest.approx(geodetic_length(line2), rel=0.001)


class TestLocalEquidistantProjection:
    """Tests for local azimuthal equidistant projection functions."""

    def test_compute_centroid_empty_array(self):
        """Empty array should return None."""
        result = _compute_centroid(np.array([], dtype=object))
        assert result is None

    def test_compute_centroid_all_none(self):
        """Array of None values should return None."""
        geoms = np.array([None, None, None], dtype=object)
        result = _compute_centroid(geoms)
        assert result is None

    def test_compute_centroid_single_geometry(self):
        """Single geometry should return its centroid."""
        line = LineString([(0, 0), (10, 0)])
        geoms = np.array([line], dtype=object)
        result = _compute_centroid(geoms)

        assert result is not None
        lon, lat = result
        assert lon == pytest.approx(5.0)
        assert lat == pytest.approx(0.0)

    def test_compute_centroid_multiple_geometries(self):
        """Multiple geometries should return average centroid."""
        line1 = LineString([(0, 0), (10, 0)])  # Centroid: (5, 0)
        line2 = LineString([(0, 10), (10, 10)])  # Centroid: (5, 10)
        geoms = np.array([line1, line2], dtype=object)
        result = _compute_centroid(geoms)

        assert result is not None
        lon, lat = result
        assert lon == pytest.approx(5.0)
        assert lat == pytest.approx(5.0)

    def test_is_geographic_true_for_wgs84(self):
        """WGS84 coordinates should be detected as geographic."""
        # Boston area coordinates
        line = LineString([(-71.05, 42.35), (-71.04, 42.35)])
        geoms = np.array([line], dtype=object)
        assert _is_geographic(geoms) is True

    def test_is_geographic_false_for_projected(self):
        """Projected coordinates (large values) should not be detected as geographic."""
        # UTM coordinates (meters)
        line = LineString([(500000, 4700000), (500100, 4700000)])
        geoms = np.array([line], dtype=object)
        assert _is_geographic(geoms) is False

    def test_is_geographic_false_for_empty(self):
        """Empty array should return False."""
        geoms = np.array([], dtype=object)
        assert _is_geographic(geoms) is False

    def test_create_local_equidistant_crs(self):
        """Local AEQD CRS should be created with correct parameters."""
        crs = _create_local_equidistant_crs(-71.05, 42.35)

        assert crs is not None
        assert crs.is_projected
        # Check that the projection is azimuthal equidistant
        assert "aeqd" in crs.to_proj4().lower()
        # Check center coordinates are in the proj string
        assert "42.35" in crs.to_proj4()
        assert "-71.05" in crs.to_proj4()

    def test_projection_preserves_distance_at_center(self):
        """Distances near the center should be accurate."""
        from pyproj import Transformer
        from shapely.ops import transform

        center_lon, center_lat = -71.05, 42.35
        crs = _create_local_equidistant_crs(center_lon, center_lat)

        # Create transformer
        from pyproj import CRS

        transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)

        # Project a small line near the center
        line_wgs84 = LineString([(center_lon, center_lat), (center_lon + 0.001, center_lat)])
        line_projected = transform(transformer.transform, line_wgs84)

        # Compare with geodetic length
        geodetic = geodetic_length(line_wgs84)
        euclidean = line_projected.length

        # Should be within 1% for small distances near center
        assert euclidean == pytest.approx(geodetic, rel=0.01)


class TestDivergenceDetection:
    """Tests for alignment truncation at divergence points."""

    def test_diverging_end_truncates_alignment(self):
        """Road that curves away at end should have truncated coverage.

        The reference is a straight horizontal line.
        The target follows the reference for most of its length, then
        curves sharply upward at the end (50m divergence over 20m).
        The alignment should truncate at the divergence point.
        """
        ref = LineString([(0, 0), (80, 0), (100, 0)])
        target = LineString([(0, 0), (80, 0), (100, 50)])  # Sharp curve at end
        result = linestring_alignment(ref, target)

        # Should NOT report full coverage - divergence at end should truncate
        assert result.overture_coverage < 0.95, (
            f"Expected truncated coverage, got overture_coverage={result.overture_coverage}"
        )
        assert result.dataset_coverage < 0.95, (
            f"Expected truncated coverage, got dataset_coverage={result.dataset_coverage}"
        )

    def test_diverging_start_truncates_alignment(self):
        """Road that diverges at start should truncate there.

        The reference is a straight horizontal line.
        The target starts 50m above the reference start, then joins
        the reference line at 20m and follows it to the end.
        """
        ref = LineString([(0, 0), (20, 0), (100, 0)])
        target = LineString([(0, 50), (20, 0), (100, 0)])  # Divergent start
        result = linestring_alignment(ref, target)

        # Start should be truncated past the divergent portion
        assert result.overture_start_frac > 0.05, (
            f"Expected truncated start, got overture_start_frac={result.overture_start_frac}"
        )

    def test_parallel_offset_no_truncation(self):
        """Consistent parallel offset should NOT truncate.

        Two lines with a constant 10m lateral offset should maintain
        full alignment - the offset is consistent, not diverging.
        """
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 10), (100, 10)])  # 10m offset throughout
        result = linestring_alignment(ref, target)

        assert result.overture_coverage > 0.95, (
            f"Parallel offset should not truncate, got overture_coverage={result.overture_coverage}"
        )
        assert result.dataset_coverage > 0.95, (
            f"Parallel offset should not truncate, got dataset_coverage={result.dataset_coverage}"
        )

    def test_slight_curve_tolerated(self):
        """Slight curvature should be tolerated, not truncated.

        A small bulge (2m over 100m) should not trigger divergence detection.
        """
        ref = LineString([(0, 0), (50, 0), (100, 0)])
        target = LineString([(0, 0), (50, 2), (100, 0)])  # Slight 2m bulge
        result = linestring_alignment(ref, target)

        assert result.overture_coverage > 0.9, (
            f"Slight curve should not truncate, got overture_coverage={result.overture_coverage}"
        )
        assert result.dataset_coverage > 0.9, (
            f"Slight curve should not truncate, got dataset_coverage={result.dataset_coverage}"
        )

    def test_symmetric_divergence_both_ends(self):
        """Lines that diverge at both ends should truncate both.

        A Y-shaped divergence at both the start and end.
        """
        ref = LineString([(0, 0), (50, 0), (100, 0)])
        target = LineString([(0, 30), (50, 0), (100, 30)])  # Diverges at both ends
        result = linestring_alignment(ref, target)

        # Coverage should be reduced significantly (middle portion only)
        assert result.overture_coverage < 0.8, (
            f"Expected both ends truncated, got overture_coverage={result.overture_coverage}"
        )

    def test_gradual_divergence_not_truncated(self):
        """Gradual divergence (small angle) should not truncate.

        A line that gradually diverges at 5 degrees should be tolerated
        since the directions are still mostly parallel.
        """
        import math

        # 5 degree angle - over 100m, this is about 8.7m divergence
        angle_rad = math.radians(5)
        end_y = 100 * math.tan(angle_rad)  # ~8.7m

        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (100, end_y)])
        result = linestring_alignment(ref, target)

        # Gradual divergence should still have high coverage
        # The parallelness threshold is 0.5 (45 degrees), 5 degrees is well within
        assert result.overture_coverage > 0.8, (
            f"Gradual divergence should not truncate heavily, got {result.overture_coverage}"
        )

    def test_winding_road_out_of_bounds_detected(self):
        """Reference that loops far away and back should be detected as diverged.

        When the ref loops away and t - offset falls outside [0, target_length],
        _interpolate_along_line used to silently clamp to the target endpoint,
        making the out-of-bounds samples appear aligned. After the fix, these
        samples are marked as diverged, producing reduced coverage.
        """
        # Target: short straight line
        target = LineString([(0, 0), (50, 0)])
        # Reference: starts at target, loops far away (200m up), then comes back
        ref = LineString([(0, 0), (25, 0), (25, 200), (75, 200), (75, 0), (50, 0)])
        result = linestring_alignment(ref, target)

        # The looping portion should be detected as divergent, reducing coverage.
        # Without the fix, the clamped interpolation makes the loop appear aligned.
        assert result.overture_coverage < 0.8, (
            f"Winding road loop should truncate coverage, got {result.overture_coverage}"
        )


class TestJunctionSegmentAlignment:
    """Tests for segments that meet at a junction but don't truly overlap."""

    def test_sequential_segments_sharing_junction_have_minimal_overlap(self):
        """Two sequential segments on the same road sharing a junction point
        should have near-zero aligned overlap.

        Real-world case: Weston Road in Toronto — the reference segment ends
        at a junction and the target segment begins there. The aligned portion
        should be at most ~2m (the junction proximity), not ~7m.

        Coordinates are in projected UTM (EPSG:32617), units in meters.
        """
        # Reference: Weston Road segment ending at junction (260m, going north)
        ref = LineString(
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
            ]
        )
        # Target: Weston Road segment starting at junction (290m, going north)
        target = LineString(
            [
                (617419.81, 4844947.81),
                (617371.86, 4845233.37),
            ]
        )

        result = linestring_alignment(ref, target)

        # The aligned portion on the reference should be tiny — just the
        # junction proximity (~0.5m), not 6-7m.
        ref_aligned_length = (result.overture_end_frac - result.overture_start_frac) * ref.length
        target_aligned_length = (
            result.dataset_end_frac - result.dataset_start_frac
        ) * target.length

        assert ref_aligned_length < 2.0, (
            f"Sequential junction segments should have < 2m ref overlap, "
            f"got {ref_aligned_length:.1f}m "
            f"(fracs {result.overture_start_frac:.4f}-{result.overture_end_frac:.4f})"
        )
        assert target_aligned_length < 2.0, (
            f"Sequential junction segments should have < 2m target overlap, "
            f"got {target_aligned_length:.1f}m "
            f"(fracs {result.dataset_start_frac:.4f}-{result.dataset_end_frac:.4f})"
        )


class TestNearestFracOnLine:
    """Tests for _nearest_frac_on_line point-to-polyline projection."""

    def _make_line_data(self, coords_list):
        """Helper to build coords/distances/length arrays from coordinate list."""
        coords = np.array(coords_list, dtype=float)
        distances = np.zeros(len(coords))
        distances[1:] = np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1)))
        return coords, distances, distances[-1]

    def test_point_on_line_start(self):
        """Point at the start of the line returns frac 0."""
        coords, dists, length = self._make_line_data([(0, 0), (100, 0)])
        assert _nearest_frac_on_line(0, 0, coords, dists, length) == pytest.approx(0.0)

    def test_point_on_line_end(self):
        """Point at the end of the line returns frac 1."""
        coords, dists, length = self._make_line_data([(0, 0), (100, 0)])
        assert _nearest_frac_on_line(100, 0, coords, dists, length) == pytest.approx(1.0)

    def test_point_on_line_midpoint(self):
        """Point at the midpoint of a straight line returns frac 0.5."""
        coords, dists, length = self._make_line_data([(0, 0), (100, 0)])
        assert _nearest_frac_on_line(50, 0, coords, dists, length) == pytest.approx(0.5)

    def test_point_offset_perpendicular(self):
        """Point offset perpendicularly projects to the nearest point on the line."""
        coords, dists, length = self._make_line_data([(0, 0), (100, 0)])
        # 10m above the line at x=30
        assert _nearest_frac_on_line(30, 10, coords, dists, length) == pytest.approx(0.3)

    def test_point_beyond_line_end_clamps(self):
        """Point past the end of the line clamps to frac 1.0."""
        coords, dists, length = self._make_line_data([(0, 0), (100, 0)])
        assert _nearest_frac_on_line(150, 0, coords, dists, length) == pytest.approx(1.0)

    def test_point_before_line_start_clamps(self):
        """Point before the start of the line clamps to frac 0.0."""
        coords, dists, length = self._make_line_data([(0, 0), (100, 0)])
        assert _nearest_frac_on_line(-50, 0, coords, dists, length) == pytest.approx(0.0)

    def test_multi_segment_line(self):
        """Point projects correctly onto a multi-segment polyline."""
        # L-shaped line: (0,0)→(100,0)→(100,100), total length 200
        coords, dists, length = self._make_line_data([(0, 0), (100, 0), (100, 100)])
        # Point offset from second segment at (110, 50) → projects to (100, 50)
        # which is at distance 100 + 50 = 150 along the line → frac 0.75
        assert _nearest_frac_on_line(110, 50, coords, dists, length) == pytest.approx(0.75)

    def test_zero_length_segment_handled(self):
        """Degenerate zero-length segment doesn't cause division by zero."""
        coords, dists, length = self._make_line_data([(0, 0), (0, 0), (100, 0)])
        assert _nearest_frac_on_line(50, 0, coords, dists, length) == pytest.approx(0.5)
