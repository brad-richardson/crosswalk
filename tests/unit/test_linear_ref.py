"""Unit tests for linear-referenced attribute handling."""

import pytest

from crosswalk.utils.linear_ref import (
    AttributeRange,
    LinearReferencedAttribute,
    coverage_for_value,
    create_trivial_lr,
    extract_aligned_attributes,
    extract_lr_name,
    extract_lr_value,
    extract_majority,
    normalize_ranges,
)


class TestAttributeRange:
    """Tests for AttributeRange dataclass."""

    def test_valid_range(self):
        """Test creating a valid range."""
        r = AttributeRange(start=0.2, end=0.6, value="test")
        assert r.start == 0.2
        assert r.end == 0.6
        assert r.value == "test"

    def test_invalid_start(self):
        """Test that start must be in [0, 1]."""
        with pytest.raises(ValueError, match="start must be"):
            AttributeRange(start=-0.1, end=0.5, value="test")
        with pytest.raises(ValueError, match="start must be"):
            AttributeRange(start=1.1, end=1.0, value="test")

    def test_invalid_end(self):
        """Test that end must be in [0, 1]."""
        with pytest.raises(ValueError, match="end must be"):
            AttributeRange(start=0.0, end=1.1, value="test")
        with pytest.raises(ValueError, match="end must be"):
            AttributeRange(start=0.0, end=-0.1, value="test")

    def test_start_after_end(self):
        """Test that start cannot be greater than end."""
        with pytest.raises(ValueError, match="start .* must be <= end"):
            AttributeRange(start=0.8, end=0.2, value="test")

    def test_length(self):
        """Test length property."""
        r = AttributeRange(start=0.2, end=0.6, value="test")
        assert r.length == pytest.approx(0.4)

    def test_overlaps(self):
        """Test overlap detection."""
        r1 = AttributeRange(start=0.2, end=0.6, value="A")
        r2 = AttributeRange(start=0.4, end=0.8, value="B")
        r3 = AttributeRange(start=0.7, end=1.0, value="C")

        assert r1.overlaps(r2)
        assert r2.overlaps(r1)
        assert not r1.overlaps(r3)
        assert r2.overlaps(r3)

    def test_intersection(self):
        """Test intersection length calculation."""
        r = AttributeRange(start=0.2, end=0.6, value="test")

        # Fully contained
        assert r.intersection(0.3, 0.5) == pytest.approx(0.2)
        # Partial overlap left
        assert r.intersection(0.0, 0.4) == pytest.approx(0.2)
        # Partial overlap right
        assert r.intersection(0.4, 1.0) == pytest.approx(0.2)
        # No overlap
        assert r.intersection(0.7, 1.0) == pytest.approx(0.0)
        # Range contains query
        assert r.intersection(0.0, 1.0) == pytest.approx(0.4)


class TestLinearReferencedAttribute:
    """Tests for LinearReferencedAttribute dataclass."""

    def test_is_uniform(self):
        """Test uniform detection."""
        lr_uniform = LinearReferencedAttribute(
            ranges=[AttributeRange(0.0, 1.0, "single")],
            default_value="single",
        )
        lr_multi = LinearReferencedAttribute(
            ranges=[
                AttributeRange(0.0, 0.5, "A"),
                AttributeRange(0.5, 1.0, "B"),
            ],
            default_value="A",
        )
        assert lr_uniform.is_uniform()
        assert not lr_multi.is_uniform()

    def test_get_value_at(self):
        """Test value lookup at position."""
        lr = LinearReferencedAttribute(
            ranges=[
                AttributeRange(0.0, 0.3, "first"),
                AttributeRange(0.3, 0.7, "middle"),
                AttributeRange(0.7, 1.0, "last"),
            ],
            default_value="default",
        )
        assert lr.get_value_at(0.0) == "first"
        assert lr.get_value_at(0.15) == "first"
        assert lr.get_value_at(0.3) == "middle"
        assert lr.get_value_at(0.5) == "middle"
        assert lr.get_value_at(0.7) == "last"
        assert lr.get_value_at(0.9) == "last"
        assert lr.get_value_at(1.0) == "last"

    def test_to_dict_list(self):
        """Test serialization to dict list."""
        lr = LinearReferencedAttribute(
            ranges=[
                AttributeRange(0.0, 0.5, "A"),
                AttributeRange(0.5, 1.0, "B"),
            ],
            default_value="A",
        )
        result = lr.to_dict_list()
        assert result == [
            {"between": [0.0, 0.5], "value": "A"},
            {"between": [0.5, 1.0], "value": "B"},
        ]

    def test_from_dict_list(self):
        """Test deserialization from dict list."""
        data = [
            {"between": [0.0, 0.5], "value": "A"},
            {"between": [0.5, 1.0], "value": "B"},
        ]
        lr = LinearReferencedAttribute.from_dict_list(data, default_value="default")
        assert len(lr.ranges) == 2
        assert lr.ranges[0].value == "A"
        assert lr.ranges[1].value == "B"
        assert lr.default_value == "default"


class TestNormalizeRanges:
    """Tests for normalize_ranges function."""

    def test_empty_rules(self):
        """Test with no rules - should return default value for full range."""
        result = normalize_ranges([], default_value="default")
        assert len(result.ranges) == 1
        assert result.ranges[0].start == 0.0
        assert result.ranges[0].end == 1.0
        assert result.ranges[0].value == "default"

    def test_single_rule_full_coverage(self):
        """Test single rule covering entire segment."""
        rules = [(0.0, 1.0, "Oak St", 0)]
        result = normalize_ranges(rules, default_value="Unnamed")
        assert len(result.ranges) == 1
        assert result.ranges[0].value == "Oak St"

    def test_single_rule_partial(self):
        """Test single rule with gaps filled by default."""
        rules = [(0.3, 0.7, "Oak St", 0)]
        result = normalize_ranges(rules, default_value="Unnamed")
        assert len(result.ranges) == 3
        # Gap before
        assert result.ranges[0].start == 0.0
        assert result.ranges[0].end == 0.3
        assert result.ranges[0].value == "Unnamed"
        # Rule
        assert result.ranges[1].start == 0.3
        assert result.ranges[1].end == 0.7
        assert result.ranges[1].value == "Oak St"
        # Gap after
        assert result.ranges[2].start == 0.7
        assert result.ranges[2].end == 1.0
        assert result.ranges[2].value == "Unnamed"

    def test_overlapping_rules_priority(self):
        """Test overlapping rules resolved by priority."""
        # Rule A (priority 0) and Rule B (priority 1) overlap at [0.4, 0.6]
        rules = [
            (0.2, 0.6, "A", 0),  # Higher priority
            (0.4, 0.8, "B", 1),  # Lower priority
        ]
        result = normalize_ranges(rules, default_value="X")

        # Expected: [0.0-0.2]=X, [0.2-0.6]=A, [0.6-0.8]=B, [0.8-1.0]=X
        assert len(result.ranges) == 4

        # Check values
        values = [(r.start, r.end, r.value) for r in result.ranges]
        assert values[0] == pytest.approx((0.0, 0.2, "X"), rel=0.01)
        assert values[1] == pytest.approx((0.2, 0.6, "A"), rel=0.01)
        assert values[2] == pytest.approx((0.6, 0.8, "B"), rel=0.01)
        assert values[3] == pytest.approx((0.8, 1.0, "X"), rel=0.01)

    def test_adjacent_merge(self):
        """Test that adjacent ranges with same value are merged."""
        # Two non-overlapping rules with same value
        rules = [
            (0.0, 0.5, "Same", 0),
            (0.5, 1.0, "Same", 0),
        ]
        result = normalize_ranges(rules, default_value="default")
        # Should be merged into one range
        assert len(result.ranges) == 1
        assert result.ranges[0].start == 0.0
        assert result.ranges[0].end == 1.0
        assert result.ranges[0].value == "Same"

    def test_gaps_filled_with_default(self):
        """Test that gaps between rules are filled with default value."""
        rules = [
            (0.0, 0.2, "Start", 0),
            (0.8, 1.0, "End", 0),
        ]
        result = normalize_ranges(rules, default_value="Middle")
        assert len(result.ranges) == 3
        assert result.ranges[0].value == "Start"
        assert result.ranges[1].value == "Middle"
        assert result.ranges[1].start == 0.2
        assert result.ranges[1].end == 0.8
        assert result.ranges[2].value == "End"


class TestExtractMajority:
    """Tests for extract_majority function."""

    def test_single_range(self):
        """Test with single range covering full segment."""
        lr = LinearReferencedAttribute(
            ranges=[AttributeRange(0.0, 1.0, "Only Value")],
            default_value="Only Value",
        )
        assert extract_majority(lr, 0.0, 1.0) == "Only Value"
        assert extract_majority(lr, 0.2, 0.8) == "Only Value"

    def test_spanning_multiple_ranges(self):
        """Test query spanning multiple ranges - returns majority."""
        lr = LinearReferencedAttribute(
            ranges=[
                AttributeRange(0.0, 0.3, "A"),  # 0.3 of segment
                AttributeRange(0.3, 0.8, "B"),  # 0.5 of segment
                AttributeRange(0.8, 1.0, "C"),  # 0.2 of segment
            ],
            default_value="A",
        )
        # Query [0.2, 0.9] covers:
        # - A: 0.1 (from 0.2 to 0.3)
        # - B: 0.5 (from 0.3 to 0.8)
        # - C: 0.1 (from 0.8 to 0.9)
        # B wins with longest coverage
        assert extract_majority(lr, 0.2, 0.9) == "B"

    def test_tie_first_wins(self):
        """Test that tie goes to first encountered value."""
        lr = LinearReferencedAttribute(
            ranges=[
                AttributeRange(0.0, 0.5, "First"),
                AttributeRange(0.5, 1.0, "Second"),
            ],
            default_value="First",
        )
        # Query [0.25, 0.75] covers equal amounts of both
        # First should win (encountered first)
        assert extract_majority(lr, 0.25, 0.75) == "First"

    def test_empty_query_returns_default(self):
        """Test that empty query (start >= end) returns default."""
        lr = LinearReferencedAttribute(
            ranges=[AttributeRange(0.0, 1.0, "Value")],
            default_value="default",
        )
        assert extract_majority(lr, 0.5, 0.5) == "default"
        assert extract_majority(lr, 0.8, 0.2) == "default"

    def test_clamps_to_valid_range(self):
        """Test that out-of-bounds queries are clamped."""
        lr = LinearReferencedAttribute(
            ranges=[AttributeRange(0.0, 1.0, "Value")],
            default_value="default",
        )
        # These should work without raising
        assert extract_majority(lr, -0.5, 0.5) == "Value"
        assert extract_majority(lr, 0.5, 1.5) == "Value"


class TestCoverageForValue:
    """Tests for coverage_for_value function."""

    def test_full_coverage(self):
        """Test coverage when value covers entire query range."""
        lr = LinearReferencedAttribute(
            ranges=[AttributeRange(0.0, 1.0, "All")],
            default_value="All",
        )
        assert coverage_for_value(lr, 0.0, 1.0, "All") == pytest.approx(1.0)
        assert coverage_for_value(lr, 0.2, 0.8, "All") == pytest.approx(0.6)

    def test_partial_coverage(self):
        """Test partial coverage."""
        lr = LinearReferencedAttribute(
            ranges=[
                AttributeRange(0.0, 0.5, "A"),
                AttributeRange(0.5, 1.0, "B"),
            ],
            default_value="A",
        )
        # Query [0.0, 1.0] for value A -> 0.5 coverage
        assert coverage_for_value(lr, 0.0, 1.0, "A") == pytest.approx(0.5)

    def test_no_coverage(self):
        """Test when value has no coverage in query range."""
        lr = LinearReferencedAttribute(
            ranges=[AttributeRange(0.0, 1.0, "Only")],
            default_value="Only",
        )
        assert coverage_for_value(lr, 0.0, 1.0, "Other") == pytest.approx(0.0)


class TestExtractAlignedAttributes:
    """Tests for extract_aligned_attributes function."""

    def test_extracts_all_attributes(self):
        """Test that all attributes are extracted."""
        lr_name = LinearReferencedAttribute(
            ranges=[AttributeRange(0.0, 1.0, "Oak St")],
            default_value="Oak St",
        )
        lr_subclass = LinearReferencedAttribute(
            ranges=[AttributeRange(0.0, 1.0, "residential")],
            default_value="residential",
        )

        lr_data = {"name": lr_name, "subclass": lr_subclass}
        result = extract_aligned_attributes(lr_data, 0.2, 0.8)

        assert result["name"] == "Oak St"
        assert result["subclass"] == "residential"

    def test_alignment_affects_result(self):
        """Test that alignment fractions affect which value is extracted."""
        lr = LinearReferencedAttribute(
            ranges=[
                AttributeRange(0.0, 0.4, "First Street"),
                AttributeRange(0.4, 1.0, "Second Street"),
            ],
            default_value="First Street",
        )

        lr_data = {"name": lr}

        # Alignment covers mostly first part
        result1 = extract_aligned_attributes(lr_data, 0.0, 0.3)
        assert result1["name"] == "First Street"

        # Alignment covers mostly second part
        result2 = extract_aligned_attributes(lr_data, 0.5, 1.0)
        assert result2["name"] == "Second Street"


class TestCreateTrivialLr:
    """Tests for create_trivial_lr function."""

    def test_creates_single_range(self):
        """Test that trivial LR has single range covering full segment."""
        lr = create_trivial_lr("Oak St")
        assert len(lr.ranges) == 1
        assert lr.ranges[0].start == 0.0
        assert lr.ranges[0].end == 1.0
        assert lr.ranges[0].value == "Oak St"
        assert lr.default_value == "Oak St"

    def test_handles_none(self):
        """Test that None values work."""
        lr = create_trivial_lr(None)
        assert lr.ranges[0].value is None
        assert lr.default_value is None

    def test_serializes_correctly(self):
        """Test that trivial LR serializes to expected format matching Overture schema."""
        lr = create_trivial_lr("test")
        result = lr.to_dict_list()
        assert result == [{"between": [0.0, 1.0], "value": "test"}]


# Reusable LR data for extract_lr_value/extract_lr_name tests
SPLIT_LR = [
    {"between": [0.0, 0.5], "value": "First Avenue"},
    {"between": [0.5, 1.0], "value": "Second Street"},
]


class TestExtractLrValue:
    """Tests for extract_lr_value() shared helper."""

    def test_none_data_returns_none(self):
        assert extract_lr_value(None, 0.0, 1.0) is None

    def test_valid_uniform_data(self):
        lr_data = [{"between": [0.0, 1.0], "value": 42}]
        assert extract_lr_value(lr_data, 0.0, 1.0) == 42

    def test_partial_alignment_majority(self):
        lr_data = [
            {"between": [0.0, 0.4], "value": "slow"},
            {"between": [0.4, 1.0], "value": "fast"},
        ]
        # Aligned to 0.3-0.9: slow covers 0.1, fast covers 0.5 → fast wins
        assert extract_lr_value(lr_data, 0.3, 0.9) == "fast"

    def test_custom_key(self):
        lr_data = [{"between": [0.0, 1.0], "value": "some_value"}]
        assert extract_lr_value(lr_data, 0.0, 1.0, key="custom") == "some_value"

    def test_malformed_data_returns_none(self):
        assert extract_lr_value([{"invalid": "data"}], 0.0, 1.0) is None

    def test_empty_list_returns_none(self):
        assert extract_lr_value([], 0.0, 1.0) is None


class TestExtractLrName:
    """Tests for extract_lr_name() shared helper."""

    def test_none_data_returns_none(self):
        assert extract_lr_name(None, 0.0, 1.0) is None

    def test_uniform_name(self):
        lr_data = [{"between": [0.0, 1.0], "value": "Main Street"}]
        assert extract_lr_name(lr_data, 0.0, 1.0) == "Main Street"

    def test_split_name_first_half(self):
        assert extract_lr_name(SPLIT_LR, 0.0, 0.4) == "First Avenue"

    def test_split_name_second_half(self):
        assert extract_lr_name(SPLIT_LR, 0.6, 1.0) == "Second Street"

    def test_split_name_majority_wins(self):
        # 0.3-0.8: First Avenue covers 0.2, Second Street covers 0.3 → Second wins
        assert extract_lr_name(SPLIT_LR, 0.3, 0.8) == "Second Street"

    def test_none_value_returns_none(self):
        lr_data = [{"between": [0.0, 1.0], "value": None}]
        assert extract_lr_name(lr_data, 0.0, 1.0) is None
