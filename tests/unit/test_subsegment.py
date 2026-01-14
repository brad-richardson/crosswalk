"""Tests for sub-segment extraction and linear referencing utilities."""

import pytest
from shapely.geometry import LineString, MultiLineString, Point

from matcher.labeling.subsegment import (
    compute_subsegment_length,
    distance_to_pct,
    estimate_overlap_range,
    extract_subsegment,
    get_point_at_pct,
    get_point_pct,
    is_subsegment_selection,
    pct_to_distance,
)


class TestExtractSubsegment:
    """Tests for extract_subsegment function."""

    def test_full_segment_returns_original(self):
        """0-100% should return original line."""
        line = LineString([(0, 0), (100, 0)])
        result = extract_subsegment(line, 0.0, 1.0)
        assert result.length == pytest.approx(line.length)

    def test_first_half(self):
        """0-50% should return first half."""
        line = LineString([(0, 0), (100, 0)])
        result = extract_subsegment(line, 0.0, 0.5)
        assert result.length == pytest.approx(50.0)
        # Check it starts at the beginning
        assert result.coords[0] == pytest.approx((0, 0), abs=0.01)

    def test_second_half(self):
        """50-100% should return second half."""
        line = LineString([(0, 0), (100, 0)])
        result = extract_subsegment(line, 0.5, 1.0)
        assert result.length == pytest.approx(50.0)
        # Check it ends at the end
        assert result.coords[-1] == pytest.approx((100, 0), abs=0.01)

    def test_middle_portion(self):
        """25-75% should return middle half."""
        line = LineString([(0, 0), (100, 0)])
        result = extract_subsegment(line, 0.25, 0.75)
        assert result.length == pytest.approx(50.0)

    def test_invalid_range_returns_original(self):
        """If start >= end, return original line."""
        line = LineString([(0, 0), (100, 0)])
        result = extract_subsegment(line, 0.7, 0.3)
        assert result.length == pytest.approx(line.length)

    def test_clamps_to_valid_range(self):
        """Values outside 0-1 should be clamped."""
        line = LineString([(0, 0), (100, 0)])
        result = extract_subsegment(line, -0.5, 1.5)
        assert result.length == pytest.approx(line.length)

    def test_multilinestring_raises_typeerror(self):
        """MultiLineString should raise TypeError."""
        multi = MultiLineString([[(0, 0), (50, 0)], [(60, 0), (100, 0)]])
        with pytest.raises(TypeError, match="must be LineString"):
            extract_subsegment(multi, 0.0, 0.5)

    def test_empty_line_raises_valueerror(self):
        """Empty LineString should raise ValueError."""
        empty = LineString()
        with pytest.raises(ValueError, match="must not be empty"):
            extract_subsegment(empty, 0.0, 0.5)


class TestEstimateOverlapRange:
    """Tests for estimate_overlap_range function."""

    def test_identical_lines(self):
        """Identical lines should have full overlap."""
        line = LineString([(0, 0), (100, 0)])
        result = estimate_overlap_range(line, line)
        assert result["ref_start_pct"] == pytest.approx(0.0, abs=0.01)
        assert result["ref_end_pct"] == pytest.approx(1.0, abs=0.01)
        assert result["target_start_pct"] == pytest.approx(0.0, abs=0.01)
        assert result["target_end_pct"] == pytest.approx(1.0, abs=0.01)

    def test_target_is_subset_of_ref(self):
        """Target that covers middle of ref."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(20, 1), (80, 1)])  # Parallel, offset, shorter
        result = estimate_overlap_range(ref, target)
        # Target endpoints project to 20% and 80% on ref
        assert result["ref_start_pct"] == pytest.approx(0.2, abs=0.01)
        assert result["ref_end_pct"] == pytest.approx(0.8, abs=0.01)
        # Ref endpoints project outside target, so clamped to 0-1
        assert result["target_start_pct"] == pytest.approx(0.0, abs=0.01)
        assert result["target_end_pct"] == pytest.approx(1.0, abs=0.01)

    def test_ref_is_subset_of_target(self):
        """Ref that covers middle of target."""
        ref = LineString([(30, 0), (70, 0)])
        target = LineString([(0, 1), (100, 1)])
        result = estimate_overlap_range(ref, target)
        assert result["ref_start_pct"] == pytest.approx(0.0, abs=0.01)
        assert result["ref_end_pct"] == pytest.approx(1.0, abs=0.01)
        assert result["target_start_pct"] == pytest.approx(0.3, abs=0.01)
        assert result["target_end_pct"] == pytest.approx(0.7, abs=0.01)

    def test_multilinestring_raises_typeerror(self):
        """MultiLineString should raise TypeError."""
        line = LineString([(0, 0), (100, 0)])
        multi = MultiLineString([[(0, 0), (50, 0)], [(60, 0), (100, 0)]])
        with pytest.raises(TypeError, match="must be LineString"):
            estimate_overlap_range(multi, line)
        with pytest.raises(TypeError, match="must be LineString"):
            estimate_overlap_range(line, multi)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_pct_to_distance(self):
        """Convert percentage to distance."""
        line = LineString([(0, 0), (100, 0)])
        assert pct_to_distance(line, 0.5) == pytest.approx(50.0)
        assert pct_to_distance(line, 0.0) == pytest.approx(0.0)
        assert pct_to_distance(line, 1.0) == pytest.approx(100.0)

    def test_distance_to_pct(self):
        """Convert distance to percentage."""
        line = LineString([(0, 0), (100, 0)])
        assert distance_to_pct(line, 50.0) == pytest.approx(0.5)
        assert distance_to_pct(line, 0.0) == pytest.approx(0.0)
        assert distance_to_pct(line, 100.0) == pytest.approx(1.0)

    def test_get_point_pct(self):
        """Get percentage where point projects onto line."""
        line = LineString([(0, 0), (100, 0)])
        point = Point(50, 10)  # Above the midpoint
        assert get_point_pct(line, point) == pytest.approx(0.5)

    def test_get_point_at_pct(self):
        """Get point at percentage along line."""
        line = LineString([(0, 0), (100, 0)])
        point = get_point_at_pct(line, 0.5)
        assert point.x == pytest.approx(50.0)
        assert point.y == pytest.approx(0.0)

    def test_compute_subsegment_length(self):
        """Compute subsegment length."""
        line = LineString([(0, 0), (100, 0)])
        assert compute_subsegment_length(line, 0.0, 1.0) == pytest.approx(100.0)
        assert compute_subsegment_length(line, 0.0, 0.5) == pytest.approx(50.0)
        assert compute_subsegment_length(line, 0.25, 0.75) == pytest.approx(50.0)

    def test_is_subsegment_selection(self):
        """Check if selection is a subsegment."""
        # Full segment is not a subsegment
        assert is_subsegment_selection(0.0, 1.0, 0.0, 1.0) is False
        # Any deviation is a subsegment
        assert is_subsegment_selection(0.0, 0.5, 0.0, 1.0) is True
        assert is_subsegment_selection(0.0, 1.0, 0.5, 1.0) is True
        assert is_subsegment_selection(0.2, 0.8, 0.3, 0.9) is True
