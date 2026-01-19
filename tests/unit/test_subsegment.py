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

    @pytest.fixture
    def line(self):
        """Standard 100-unit line for testing."""
        return LineString([(0, 0), (100, 0)])

    @pytest.mark.parametrize(
        "pct,expected_distance",
        [(0.0, 0.0), (0.5, 50.0), (1.0, 100.0)],
        ids=["start", "middle", "end"],
    )
    def test_pct_to_distance(self, line, pct, expected_distance):
        """Convert percentage to distance along line."""
        assert pct_to_distance(line, pct) == pytest.approx(expected_distance)

    @pytest.mark.parametrize(
        "distance,expected_pct",
        [(0.0, 0.0), (50.0, 0.5), (100.0, 1.0)],
        ids=["start", "middle", "end"],
    )
    def test_distance_to_pct(self, line, distance, expected_pct):
        """Convert distance to percentage along line."""
        assert distance_to_pct(line, distance) == pytest.approx(expected_pct)

    def test_get_point_pct(self, line):
        """Get percentage where point projects onto line."""
        point = Point(50, 10)  # Above the midpoint
        assert get_point_pct(line, point) == pytest.approx(0.5)

    def test_get_point_at_pct(self, line):
        """Get point at percentage along line."""
        point = get_point_at_pct(line, 0.5)
        assert point.x == pytest.approx(50.0)
        assert point.y == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "start_pct,end_pct,expected_length",
        [(0.0, 1.0, 100.0), (0.0, 0.5, 50.0), (0.25, 0.75, 50.0)],
        ids=["full", "first_half", "middle_half"],
    )
    def test_compute_subsegment_length(self, line, start_pct, end_pct, expected_length):
        """Compute subsegment length between percentages."""
        assert compute_subsegment_length(line, start_pct, end_pct) == pytest.approx(expected_length)

    @pytest.mark.parametrize(
        "ref_start,ref_end,target_start,target_end,expected",
        [
            (0.0, 1.0, 0.0, 1.0, False),  # Full segment is not a subsegment
            (0.0, 0.5, 0.0, 1.0, True),  # Ref partial
            (0.0, 1.0, 0.5, 1.0, True),  # Target partial
            (0.2, 0.8, 0.3, 0.9, True),  # Both partial
        ],
        ids=["full_segment", "ref_partial", "target_partial", "both_partial"],
    )
    def test_is_subsegment_selection(self, ref_start, ref_end, target_start, target_end, expected):
        """Check if selection represents a subsegment."""
        result = is_subsegment_selection(ref_start, ref_end, target_start, target_end)
        assert result is expected
