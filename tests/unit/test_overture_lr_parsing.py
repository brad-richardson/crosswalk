"""Unit tests for Overture linear-referenced attribute parsing."""

from matcher.fetch.overture import (
    _extract_range_from_rule,
    _get_language_priority,
    _get_variant_priority,
    parse_level_rules_lr,
    parse_names_lr,
    parse_road_flags_lr,
    parse_subclass_rules_lr,
)


class TestVariantPriority:
    """Tests for name variant priority."""

    def test_common_highest_priority(self):
        """Test that 'common' has highest priority (0)."""
        assert _get_variant_priority("common") == 0
        assert _get_variant_priority("Common") == 0

    def test_priority_order(self):
        """Test variant priority ordering."""
        assert _get_variant_priority("common") < _get_variant_priority("alternate")
        assert _get_variant_priority("alternate") < _get_variant_priority("short")
        assert _get_variant_priority("official") < _get_variant_priority("alt")

    def test_unknown_variant(self):
        """Test that unknown variants get low priority."""
        unknown = _get_variant_priority("unknown_variant")
        assert unknown > _get_variant_priority("common")
        assert unknown == 10  # Default for unknown

    def test_none_variant(self):
        """Test that None variant gets highest priority."""
        assert _get_variant_priority(None) == 0


class TestLanguagePriority:
    """Tests for language priority."""

    def test_bare_preferred(self):
        """Test that bare (no language) names are preferred."""
        assert _get_language_priority(None) == 0
        assert _get_language_priority("en") == 1

    def test_all_languages_equal(self):
        """Test that all specific languages have equal priority."""
        assert _get_language_priority("en") == _get_language_priority("es")
        assert _get_language_priority("fr") == _get_language_priority("de")


class TestExtractRangeFromRule:
    """Tests for range extraction from rules."""

    def test_top_level_between(self):
        """Test extraction from top-level 'between' key."""
        rule = {"between": [0.2, 0.6], "value": "test"}
        result = _extract_range_from_rule(rule)
        assert result == (0.2, 0.6)

    def test_scope_between(self):
        """Test extraction from scope.between."""
        rule = {"scope": {"between": [0.3, 0.7]}, "value": "test"}
        result = _extract_range_from_rule(rule)
        assert result == (0.3, 0.7)

    def test_no_range(self):
        """Test that rules without range return None."""
        rule = {"value": "test"}
        result = _extract_range_from_rule(rule)
        assert result is None

    def test_invalid_between(self):
        """Test that invalid between values return None."""
        # Single value
        assert _extract_range_from_rule({"between": [0.5]}) is None
        # Not a list
        assert _extract_range_from_rule({"between": "invalid"}) is None
        # Nested but invalid
        assert _extract_range_from_rule({"scope": {"between": None}}) is None


class TestParseNamesLr:
    """Tests for parse_names_lr function."""

    def test_none_input(self):
        """Test with None input."""
        result = parse_names_lr(None)
        assert len(result.ranges) == 1
        assert result.ranges[0].value is None

    def test_primary_only(self):
        """Test with only primary name, no rules."""
        names = {"primary": "Oak Street"}
        result = parse_names_lr(names)
        assert len(result.ranges) == 1
        assert result.ranges[0].value == "Oak Street"

    def test_with_rules_full_coverage(self):
        """Test with rules that cover full segment."""
        names = {
            "primary": "Default Name",
            "rules": [{"value": "Oak Street", "variant": "common"}],
        }
        result = parse_names_lr(names)
        # Rule covers full segment since no 'between' specified
        assert len(result.ranges) == 1
        assert result.ranges[0].value == "Oak Street"

    def test_with_rules_partial_coverage(self):
        """Test with rules that cover partial segment."""
        names = {
            "primary": "Main Street",
            "rules": [
                {"value": "Oak Avenue", "between": [0.0, 0.5], "variant": "common"},
            ],
        }
        result = parse_names_lr(names)
        # Should have two ranges: Oak Avenue [0-0.5], Main Street [0.5-1]
        assert len(result.ranges) == 2
        assert result.ranges[0].value == "Oak Avenue"
        assert result.ranges[1].value == "Main Street"

    def test_variant_priority(self):
        """Test that common variant beats alt variant."""
        names = {
            "primary": "Default",
            "rules": [
                {"value": "Alt Name", "variant": "alt", "between": [0.0, 1.0]},
                {"value": "Common Name", "variant": "common", "between": [0.0, 1.0]},
            ],
        }
        result = parse_names_lr(names)
        # Common should win
        assert len(result.ranges) == 1
        assert result.ranges[0].value == "Common Name"

    def test_language_priority(self):
        """Test that bare names beat language-specific names."""
        names = {
            "primary": "Default",
            "rules": [
                {
                    "value": "English Name",
                    "variant": "common",
                    "language": "en",
                    "between": [0.0, 1.0],
                },
                {
                    "value": "Bare Name",
                    "variant": "common",
                    "language": None,
                    "between": [0.0, 1.0],
                },
            ],
        }
        result = parse_names_lr(names)
        # Bare should win over language-specific
        assert len(result.ranges) == 1
        assert result.ranges[0].value == "Bare Name"

    def test_empty_rules(self):
        """Test with empty rules array."""
        names = {"primary": "Main St", "rules": []}
        result = parse_names_lr(names)
        assert len(result.ranges) == 1
        assert result.ranges[0].value == "Main St"


class TestParseSubclassRulesLr:
    """Tests for parse_subclass_rules_lr function."""

    def test_none_input(self):
        """Test with None input."""
        result = parse_subclass_rules_lr(None)
        assert len(result.ranges) == 1
        assert result.ranges[0].value is None

    def test_with_default_subclass(self):
        """Test with default subclass."""
        result = parse_subclass_rules_lr(None, default_subclass="residential")
        assert result.ranges[0].value == "residential"

    def test_single_rule(self):
        """Test with single rule."""
        rules = [{"value": "living_street"}]
        result = parse_subclass_rules_lr(rules)
        assert len(result.ranges) == 1
        assert result.ranges[0].value == "living_street"

    def test_multiple_rules_with_ranges(self):
        """Test with multiple rules having different ranges."""
        rules = [
            {"value": "residential", "between": [0.0, 0.5]},
            {"value": "service", "between": [0.5, 1.0]},
        ]
        result = parse_subclass_rules_lr(rules, default_subclass="unknown")
        assert len(result.ranges) == 2
        assert result.ranges[0].value == "residential"
        assert result.ranges[1].value == "service"


class TestParseLevelRulesLr:
    """Tests for parse_level_rules_lr function."""

    def test_none_input(self):
        """Test with None input - should default to 0."""
        result = parse_level_rules_lr(None)
        assert len(result.ranges) == 1
        assert result.ranges[0].value == 0

    def test_elevated_section(self):
        """Test with elevated section (positive level)."""
        rules = [
            {"value": 1, "between": [0.3, 0.7]},
        ]
        result = parse_level_rules_lr(rules)
        # Should have: ground [0-0.3], elevated [0.3-0.7], ground [0.7-1.0]
        assert len(result.ranges) == 3
        assert result.ranges[0].value == 0
        assert result.ranges[1].value == 1
        assert result.ranges[2].value == 0

    def test_underground_section(self):
        """Test with underground section (negative level)."""
        rules = [
            {"value": -1, "between": [0.0, 1.0]},
        ]
        result = parse_level_rules_lr(rules)
        assert len(result.ranges) == 1
        assert result.ranges[0].value == -1


class TestParseRoadFlagsLr:
    """Tests for parse_road_flags_lr function."""

    def test_none_input(self):
        """Test with None input - should default to empty frozenset."""
        result = parse_road_flags_lr(None)
        assert len(result.ranges) == 1
        assert result.ranges[0].value == frozenset()

    def test_bridge_flag(self):
        """Test with bridge flag."""
        rules = [
            {"values": ["is_bridge"], "between": [0.3, 0.7]},
        ]
        result = parse_road_flags_lr(rules)
        # Should have: empty [0-0.3], bridge [0.3-0.7], empty [0.7-1.0]
        assert len(result.ranges) == 3
        assert result.ranges[0].value == frozenset()
        assert result.ranges[1].value == frozenset({"is_bridge"})
        assert result.ranges[2].value == frozenset()

    def test_multiple_flags(self):
        """Test with multiple flags in one rule."""
        rules = [
            {"values": ["is_bridge", "is_link"]},
        ]
        result = parse_road_flags_lr(rules)
        assert len(result.ranges) == 1
        assert "is_bridge" in result.ranges[0].value
        assert "is_link" in result.ranges[0].value

    def test_tunnel_flag(self):
        """Test with tunnel flag."""
        rules = [
            {"values": ["is_tunnel"]},
        ]
        result = parse_road_flags_lr(rules)
        assert result.ranges[0].value == frozenset({"is_tunnel"})
