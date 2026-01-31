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


class TestParseNamesLrRealWorldPatterns:
    """Tests based on real Overture data patterns observed in production."""

    def test_alternate_full_segment_with_common_ranges(self):
        """Test: alternate covers full segment, common variants have specific ranges.

        Real example from br_sao_paulo_roads (GERS: 53baf988-3a99-4a35-9088-b5cb63b57876):
        - primary: 'Radial Leste'
        - rules[0]: variant=alternate, between=None (full segment)
        - rules[1]: variant=common, between=[0.0, 0.154]
        - rules[2]: variant=common, between=[0.154, 1.0]

        Expected: common variants should win over alternate.
        """
        names = {
            "primary": "Radial Leste",
            "rules": [
                {"value": "Radial Leste", "variant": "alternate", "between": None},
                {"value": "Avenida Alcântara Machado", "variant": "common", "between": [0.0, 0.154]},
                {"value": "Viaduto Alcântara Machado", "variant": "common", "between": [0.154, 1.0]},
            ],
        }
        result = parse_names_lr(names)

        # Common variants should win, alternate should be ignored
        assert len(result.ranges) == 2
        assert result.ranges[0].value == "Avenida Alcântara Machado"
        assert result.ranges[0].start == 0.0
        assert abs(result.ranges[0].end - 0.154) < 0.001
        assert result.ranges[1].value == "Viaduto Alcântara Machado"
        assert abs(result.ranges[1].start - 0.154) < 0.001
        assert result.ranges[1].end == 1.0

    def test_gap_filled_with_primary(self):
        """Test: rules don't cover full segment, gap filled with primary.

        Real example from br_sao_paulo_roads (GERS: 1da06405-d1aa-4f26-8fdf-6f5ad7bcddb3):
        - primary: 'Ciclovia Avenida Educador Paulo Freire'
        - rules[0]: between=[0.0, 0.493]
        - rules[1]: between=[0.493, 0.948]
        - Gap: [0.948, 1.0] should be filled with primary

        Expected: 3 ranges with gap filled by primary.
        """
        names = {
            "primary": "Ciclovia Avenida Educador Paulo Freire",
            "rules": [
                {"value": "Ciclovia Avenida Educador Paulo Freire", "variant": "common", "between": [0.0, 0.493]},
                {"value": "Ciclovia Ponte Aricanduva", "variant": "common", "between": [0.493, 0.948]},
            ],
        }
        result = parse_names_lr(names)

        assert len(result.ranges) == 3
        assert result.ranges[0].value == "Ciclovia Avenida Educador Paulo Freire"
        assert result.ranges[1].value == "Ciclovia Ponte Aricanduva"
        assert result.ranges[2].value == "Ciclovia Avenida Educador Paulo Freire"  # Gap filled
        assert abs(result.ranges[2].start - 0.948) < 0.001
        assert result.ranges[2].end == 1.0

    def test_three_name_segments(self):
        """Test: segment with three different names along its length.

        Real example from us_fort_collins_streets (GERS: 67c4dea6-0017-49ae-b87d-269c7177d665):
        - Cross Creek Drive [0.0, 0.379]
        - Owl Creek Drive [0.379, 0.644]
        - Yellow Creek Drive [0.644, 1.0]
        """
        names = {
            "primary": "Cross Creek Drive",
            "rules": [
                {"value": "Cross Creek Drive", "variant": "common", "between": [0.0, 0.379]},
                {"value": "Owl Creek Drive", "variant": "common", "between": [0.379, 0.644]},
                {"value": "Yellow Creek Drive", "variant": "common", "between": [0.644, 1.0]},
            ],
        }
        result = parse_names_lr(names)

        assert len(result.ranges) == 3
        assert result.ranges[0].value == "Cross Creek Drive"
        assert result.ranges[1].value == "Owl Creek Drive"
        assert result.ranges[2].value == "Yellow Creek Drive"

    def test_duplicate_rules_different_variants(self):
        """Test: same value with both common and alternate variants.

        Real example from co_bogota_roads (GERS: 448332d5-91fc-4bbb-823d-cf4e44639fa0):
        - rules has 'Calle 54D Sur' with variant=common AND variant=alternate
          for the same range [0.763, 1.0]

        Expected: common wins, no duplicate ranges.
        """
        names = {
            "primary": "Carrera 5 Este",
            "rules": [
                {"value": "Carrera 5 Este", "variant": "common", "between": [0.0, 0.763]},
                {"value": "Calle 54D Sur", "variant": "common", "between": [0.763, 1.0]},
                {"value": "Calle 54D Sur", "variant": "alternate", "between": [0.763, 1.0]},
            ],
        }
        result = parse_names_lr(names)

        # Should have 2 ranges, not 3 (common wins, alternate ignored)
        assert len(result.ranges) == 2
        assert result.ranges[0].value == "Carrera 5 Este"
        assert result.ranges[1].value == "Calle 54D Sur"

    def test_bogota_calle_carrera_pattern(self):
        """Test: typical Bogota pattern with Calle/Carrera name changes.

        Real example from co_bogota_roads (GERS: 7c144478-3966-4c87-9b7d-31095d73ba16):
        - Diagonal 37C Sur [0.0, 0.352]
        - Calle 37B Sur [0.352, 1.0]
        """
        names = {
            "primary": "Calle 37B Sur",
            "rules": [
                {"value": "Diagonal 37C Sur", "variant": "common", "between": [0.0, 0.352]},
                {"value": "Calle 37B Sur", "variant": "common", "between": [0.352, 1.0]},
            ],
        }
        result = parse_names_lr(names)

        assert len(result.ranges) == 2
        assert result.ranges[0].value == "Diagonal 37C Sur"
        assert result.ranges[1].value == "Calle 37B Sur"
        # Verify the boundary
        assert abs(result.ranges[0].end - 0.352) < 0.001
        assert abs(result.ranges[1].start - 0.352) < 0.001


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
