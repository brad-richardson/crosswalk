"""Regression tests for alignment - ensures good matches stay good after changes.

These tests verify that the alignment algorithm maintains expected behavior
across various scenarios. They use synthetic geometries to ensure tests are
reproducible regardless of data file availability.
"""

from shapely.geometry import LineString

from matcher.features.alignment import linestring_alignment


class TestSyntheticAlignmentRegression:
    """Synthetic tests to ensure basic alignment behavior doesn't regress."""

    def test_identical_lines_full_coverage(self):
        """Identical lines should have 100% coverage."""
        line = LineString([(0, 0), (50, 0), (100, 0)])
        result = linestring_alignment(line, line)

        assert result.overture_coverage >= 0.99
        assert result.dataset_coverage >= 0.99

    def test_parallel_offset_full_coverage(self):
        """Parallel offset lines should have full coverage (no divergence)."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 5), (100, 5)])  # 5m offset

        result = linestring_alignment(ref, target)

        assert result.overture_coverage >= 0.95
        assert result.dataset_coverage >= 0.95

    def test_partial_overlap_correct_coverage(self):
        """Partial overlap should report correct coverage fractions."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(25, 0), (75, 0)])  # 50% overlap

        result = linestring_alignment(ref, target)

        # Reference should show ~50% coverage (the overlapping portion)
        assert 0.4 <= result.overture_coverage <= 0.6
        # Target should show ~100% coverage (it's fully aligned)
        assert result.dataset_coverage >= 0.9

    def test_reversed_line_full_coverage(self):
        """Reversed lines should still have full coverage."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(100, 0), (0, 0)])  # Reversed

        result = linestring_alignment(ref, target)

        assert result.overture_coverage >= 0.95
        assert result.dataset_coverage >= 0.95

    def test_slight_lateral_variation_not_truncated(self):
        """Lines with slight lateral variations shouldn't be truncated."""
        ref = LineString([(0, 0), (25, 0), (50, 0), (75, 0), (100, 0)])
        # Target has small lateral deviations (GPS jitter simulation)
        target = LineString([(0, 1), (25, -1), (50, 2), (75, -1), (100, 1)])

        result = linestring_alignment(ref, target)

        # Should maintain high coverage despite small deviations
        assert result.overture_coverage >= 0.9
        assert result.dataset_coverage >= 0.9

    def test_curvy_road_maintains_coverage(self):
        """Curvy roads that follow the same path should maintain coverage."""
        import math

        # Create a curvy road (sinusoidal)
        ref_coords = [(x, 5 * math.sin(x / 10)) for x in range(0, 101, 5)]
        target_coords = [(x, 5 * math.sin(x / 10) + 2) for x in range(0, 101, 5)]

        ref = LineString(ref_coords)
        target = LineString(target_coords)

        result = linestring_alignment(ref, target)

        # Curvy but parallel should have high coverage
        assert result.overture_coverage >= 0.9
        assert result.dataset_coverage >= 0.9

    def test_different_lengths_partial_coverage(self):
        """Lines of different lengths should show proportional coverage."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (200, 0)])  # Target is 2x longer

        result = linestring_alignment(ref, target)

        # Reference should be fully covered
        assert result.overture_coverage >= 0.95
        # Target should be ~50% covered
        assert 0.4 <= result.dataset_coverage <= 0.6

    def test_multi_segment_alignment(self):
        """Complex multi-segment lines should still align properly."""
        ref = LineString([(0, 0), (30, 10), (60, 0), (90, 10), (120, 0)])
        target = LineString([(0, 2), (30, 12), (60, 2), (90, 12), (120, 2)])

        result = linestring_alignment(ref, target)

        # Should maintain high coverage for parallel zigzag pattern
        assert result.overture_coverage >= 0.9
        assert result.dataset_coverage >= 0.9


class TestDivergenceDetectionRegression:
    """Regression tests specifically for divergence detection behavior."""

    def test_sharp_end_divergence_detected(self):
        """Sharp divergence at the end should be detected and truncated."""
        ref = LineString([(0, 0), (80, 0), (100, 0)])
        target = LineString([(0, 0), (80, 0), (100, 50)])  # 50m divergence at end

        result = linestring_alignment(ref, target)

        # Should detect divergence and truncate
        assert result.overture_coverage < 0.95
        assert result.dataset_coverage < 0.95

    def test_sharp_start_divergence_detected(self):
        """Sharp divergence at the start should be detected and truncated."""
        ref = LineString([(0, 0), (20, 0), (100, 0)])
        target = LineString([(0, 50), (20, 0), (100, 0)])  # 50m divergence at start

        result = linestring_alignment(ref, target)

        # Should detect divergence at start
        assert result.overture_start_frac > 0.05

    def test_both_ends_divergence_detected(self):
        """Divergence at both ends should truncate both."""
        ref = LineString([(0, 0), (50, 0), (100, 0)])
        target = LineString([(0, 30), (50, 0), (100, 30)])

        result = linestring_alignment(ref, target)

        # Should truncate both ends
        assert result.overture_coverage < 0.8

    def test_gradual_curve_tolerated(self):
        """Gradual curves (small angle) should not be truncated."""
        import math

        # 10 degree angle over 100m = ~17m divergence
        angle_rad = math.radians(10)
        end_y = 100 * math.tan(angle_rad)

        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (100, end_y)])

        result = linestring_alignment(ref, target)

        # Gradual curve should still have decent coverage
        assert result.overture_coverage >= 0.7

    def test_15m_constant_offset_no_truncation(self):
        """15m constant offset should not trigger divergence truncation."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 15), (100, 15)])

        result = linestring_alignment(ref, target)

        # Constant offset is not divergence
        assert result.overture_coverage >= 0.95
        assert result.dataset_coverage >= 0.95

    def test_divergence_detection_can_be_disabled(self):
        """Divergence detection should be disableable."""
        ref = LineString([(0, 0), (80, 0), (100, 0)])
        target = LineString([(0, 0), (80, 0), (100, 50)])

        # With divergence detection
        with_detection = linestring_alignment(ref, target, detect_divergence=True)

        # Without divergence detection
        without_detection = linestring_alignment(ref, target, detect_divergence=False)

        # Without detection should show higher coverage
        assert without_detection.overture_coverage >= with_detection.overture_coverage
