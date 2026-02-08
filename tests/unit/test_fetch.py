"""Tests for fetch modules (overture, arcgis).

These modules handle data fetching and normalization, which is critical
for the pipeline. Tests cover:
- One-way direction parsing and normalization
- Speed limit parsing and unit conversion
- Linear-referenced attribute creation
"""

import numpy as np
import pandas as pd
import pytest

from matcher.fetch.arcgis import _is_truthy
from matcher.fetch.normalize import map_column, normalize_oneway_value, normalize_speed_to_kph
from matcher.fetch.overture import (
    parse_names_lr,
    parse_oneway_lr,
    parse_speed_limit_lr,
)


class TestMapColumn:
    """Tests for map_column in normalize.py."""

    def test_string_keys_match(self):
        """String keys match string values."""
        series = pd.Series(["a", "b", "c"])
        mapping = {"a": "alpha", "b": "beta"}
        result = map_column(series, mapping, fallback="unknown")
        assert list(result) == ["alpha", "beta", "unknown"]

    def test_int_keys_match_string_values(self):
        """Integer mapping keys match string series values."""
        series = pd.Series(["1", "2", "3"])
        mapping = {1: "one", 2: "two"}
        result = map_column(series, mapping, fallback="unknown")
        assert list(result) == ["one", "two", "unknown"]

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        series = pd.Series(["Residential", "COMMERCIAL", "other"])
        mapping = {"residential": "res", "commercial": "com"}
        result = map_column(series, mapping, fallback="unknown")
        assert list(result) == ["res", "com", "unknown"]

    def test_float_upcast(self):
        """Float values like 1.0 match integer keys like 1."""
        series = pd.Series([1.0, 2.0, 3.0])
        mapping = {1: "one", 2: "two"}
        result = map_column(series, mapping, fallback="unknown")
        assert list(result) == ["one", "two", "unknown"]

    def test_nan_values_use_fallback(self):
        """NaN values in series get the fallback."""
        series = pd.Series(["a", None, np.nan])
        mapping = {"a": "alpha"}
        result = map_column(series, mapping, fallback="unknown")
        assert list(result) == ["alpha", "unknown", "unknown"]

    def test_no_fallback_keeps_nan(self):
        """Without fallback, unmatched rows stay NaN."""
        series = pd.Series(["a", "b"])
        mapping = {"a": "alpha"}
        result = map_column(series, mapping)
        assert result[0] == "alpha"
        assert pd.isna(result[1])


class TestNormalizeOnewayValue:
    """Tests for normalize_oneway_value in normalize.py."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            # Forward indicators
            ("yes", "forward"),
            ("Yes", "forward"),
            ("YES", "forward"),
            ("y", "forward"),
            ("Y", "forward"),
            ("1", "forward"),
            (1, "forward"),
            ("FT", "forward"),
            ("ft", "forward"),
            ("F", "forward"),
            ("forward", "forward"),
            ("Forward", "forward"),
            ("one-way", "forward"),
            ("oneway", "forward"),
            ("from-to", "forward"),
            # Backward indicators
            ("-1", "backward"),
            ("TF", "backward"),
            ("tf", "backward"),
            ("T", "backward"),
            ("backward", "backward"),
            ("Backward", "backward"),
            ("reverse", "backward"),
            ("to-from", "backward"),
            # Bidirectional indicators
            ("no", "both"),
            ("No", "both"),
            ("NO", "both"),
            ("n", "both"),
            ("N", "both"),
            ("0", "both"),
            (0, "both"),
            ("B", "both"),
            ("b", "both"),
            ("both", "both"),
            ("Both", "both"),
            ("two-way", "both"),
            ("twoway", "both"),
            # None/null values
            (None, None),
            ("", None),
            ("null", None),
            ("none", None),
            ("nan", None),
            (np.nan, None),
            (pd.NA, None),
            # Unknown values return None
            ("unknown", None),
            ("xyz", None),
            ("2", None),
        ],
    )
    def testnormalize_oneway_value(self, value, expected):
        """Test various one-way value normalizations."""
        assert normalize_oneway_value(value) == expected


class TestNormalizeSpeedToKph:
    """Tests for normalize_speed_to_kph in normalize.py."""

    @pytest.mark.parametrize(
        "value,unit,expected",
        [
            # KPH values (no conversion)
            (50, "kph", 50),
            (100, "kph", 100),
            (30.5, "kph", 30),  # Truncates to int
            ("60", "kph", 60),  # String conversion
            # MPH values (conversion)
            (30, "mph", 48),  # 30 * 1.60934 = 48.28
            (65, "mph", 104),  # 65 * 1.60934 = 104.6
            (55, "mi/h", 88),  # Alternative unit string
            ("45", "mph", 72),  # String + conversion
            # Invalid values
            (None, "kph", None),
            (np.nan, "mph", None),
            (pd.NA, "kph", None),
            (0, "kph", None),  # Zero speed invalid
            (-10, "kph", None),  # Negative speed invalid
            ("invalid", "kph", None),  # Non-numeric string
        ],
    )
    def testnormalize_speed_to_kph(self, value, unit, expected):
        """Test speed normalization with various inputs."""
        assert normalize_speed_to_kph(value, unit) == expected


class TestIsTruthy:
    """Tests for _is_truthy in arcgis.py."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            # Truthy strings
            ("yes", True),
            ("Yes", True),
            ("YES", True),
            ("y", True),
            ("Y", True),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("t", True),
            ("T", True),
            # Truthy numbers
            (1, True),
            (True, True),
            # Falsy strings
            ("no", False),
            ("No", False),
            ("NO", False),
            ("n", False),
            ("N", False),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("f", False),
            ("F", False),
            # Falsy values
            (0, False),
            (False, False),
            (None, False),
            ("", False),
        ],
    )
    def test_is_truthy(self, value, expected):
        """Test boolean value conversion."""
        assert _is_truthy(value) == expected


class TestParseOnewayLr:
    """Tests for parse_oneway_lr in overture.py."""

    def test_none_input(self):
        """None input returns trivial LR with None value."""
        result = parse_oneway_lr(None)
        assert len(result.ranges) == 1
        assert result.ranges[0].value is None

    def test_empty_list(self):
        """Empty list returns trivial LR with None value."""
        result = parse_oneway_lr([])
        assert len(result.ranges) == 1
        assert result.ranges[0].value is None

    def test_numpy_array_conversion(self):
        """Numpy arrays are converted to lists."""
        arr = np.array([])
        result = parse_oneway_lr(arr)
        assert len(result.ranges) == 1
        assert result.ranges[0].value is None

    def test_no_heading_restrictions(self):
        """No heading restrictions means bidirectional."""
        access = [
            {"access_type": "allowed", "when": {"mode": "car"}},
        ]
        result = parse_oneway_lr(access)
        assert result.ranges[0].value == "both"

    def test_backward_denied_means_forward(self):
        """Backward denied = forward one-way."""
        access = [
            {"access_type": "denied", "when": {"heading": "backward"}},
        ]
        result = parse_oneway_lr(access)
        assert result.ranges[0].value == "forward"

    def test_forward_denied_means_backward(self):
        """Forward denied = backward one-way."""
        access = [
            {"access_type": "denied", "when": {"heading": "forward"}},
        ]
        result = parse_oneway_lr(access)
        assert result.ranges[0].value == "backward"

    def test_both_denied_means_none(self):
        """Both directions denied = no through traffic."""
        access = [
            {"access_type": "denied", "when": {"heading": "forward"}},
            {"access_type": "denied", "when": {"heading": "backward"}},
        ]
        result = parse_oneway_lr(access)
        assert result.ranges[0].value == "none"

    def test_ignores_non_dict_rules(self):
        """Non-dict rules are ignored."""
        access = [
            "not a dict",
            {"access_type": "denied", "when": {"heading": "backward"}},
        ]
        result = parse_oneway_lr(access)
        assert result.ranges[0].value == "forward"

    def test_handles_missing_when(self):
        """Missing 'when' field is handled gracefully."""
        access = [
            {"access_type": "denied"},  # No 'when'
        ]
        result = parse_oneway_lr(access)
        assert result.ranges[0].value == "both"  # No heading restriction found


class TestParseSpeedLimitLr:
    """Tests for parse_speed_limit_lr in overture.py."""

    def test_none_input(self):
        """None input returns trivial LR with None value."""
        result = parse_speed_limit_lr(None)
        assert len(result.ranges) == 1
        assert result.ranges[0].value is None

    def test_empty_list(self):
        """Empty list returns trivial LR with None value."""
        result = parse_speed_limit_lr([])
        assert len(result.ranges) == 1
        assert result.ranges[0].value is None

    def test_simple_speed_limit_kph(self):
        """Simple speed limit in km/h."""
        limits = [
            {"max_speed": {"value": 50, "unit": "km/h"}},
        ]
        result = parse_speed_limit_lr(limits)
        assert len(result.ranges) == 1
        assert result.ranges[0].value == 50

    def test_speed_limit_mph_conversion(self):
        """Speed limit in mph is converted to kph."""
        limits = [
            {"max_speed": {"value": 30, "unit": "mph"}},
        ]
        result = parse_speed_limit_lr(limits)
        # 30 * 1.60934 = 48.28 -> 48
        assert result.ranges[0].value == 48

    def test_speed_limit_with_range(self):
        """Speed limit with 'between' range."""
        limits = [
            {"max_speed": {"value": 50, "unit": "km/h"}, "between": [0.0, 0.5]},
            {"max_speed": {"value": 30, "unit": "km/h"}, "between": [0.5, 1.0]},
        ]
        result = parse_speed_limit_lr(limits)
        assert len(result.ranges) == 2
        # First rule
        assert result.ranges[0].start == 0.0
        assert result.ranges[0].end == 0.5
        assert result.ranges[0].value == 50
        # Second rule
        assert result.ranges[1].start == 0.5
        assert result.ranges[1].end == 1.0
        assert result.ranges[1].value == 30

    def test_invalid_speed_values_skipped(self):
        """Invalid speed values are skipped."""
        limits = [
            {"max_speed": {"value": "invalid", "unit": "km/h"}},
            {"max_speed": {"value": 50, "unit": "km/h"}},
        ]
        result = parse_speed_limit_lr(limits)
        assert result.ranges[0].value == 50

    def test_missing_max_speed_skipped(self):
        """Rules without max_speed are skipped."""
        limits = [
            {"not_max_speed": 50},
            {"max_speed": {"value": 60, "unit": "km/h"}},
        ]
        result = parse_speed_limit_lr(limits)
        assert result.ranges[0].value == 60

    def test_numpy_array_conversion(self):
        """Numpy arrays are converted to lists."""
        arr = np.array([])
        result = parse_speed_limit_lr(arr)
        assert len(result.ranges) == 1
        assert result.ranges[0].value is None


class TestParseNamesLr:
    """Tests for parse_names_lr in overture.py.

    Overture names structure:
    - primary: The default/main name (string)
    - rules: Array of name rules with value, variant, language, between
    """

    def test_none_input(self):
        """None input returns trivial LR with None value."""
        result = parse_names_lr(None)
        assert len(result.ranges) == 1
        assert result.ranges[0].value is None

    def test_simple_primary_name(self):
        """Simple primary name extraction (no rules)."""
        names = {"primary": "Main Street"}
        result = parse_names_lr(names)
        assert result.ranges[0].value == "Main Street"

    def test_rules_with_value(self):
        """Rules array with value extracts the name."""
        names = {"rules": [{"value": "Broadway"}]}
        result = parse_names_lr(names)
        assert result.ranges[0].value == "Broadway"

    def test_rules_with_between(self):
        """Rules with 'between' create LR segments."""
        names = {
            "rules": [
                {"value": "North St", "between": [0.0, 0.5]},
                {"value": "South St", "between": [0.5, 1.0]},
            ]
        }
        result = parse_names_lr(names)
        assert len(result.ranges) == 2
        assert result.ranges[0].value == "North St"
        assert result.ranges[1].value == "South St"

    def test_primary_as_default(self):
        """Primary name used as default for gaps in rules."""
        names = {
            "primary": "Main Street",
            "rules": [{"value": "Broadway", "between": [0.0, 0.5]}],
        }
        result = parse_names_lr(names)
        # First half is Broadway, second half is Main Street (default)
        assert len(result.ranges) == 2
        assert result.ranges[0].value == "Broadway"
        assert result.ranges[1].value == "Main Street"

    def test_language_priority(self):
        """Bare names (no language) have priority over language-specific."""
        names = {
            "rules": [
                {"value": "strada", "language": "it"},
                {"value": "street"},  # No language = bare name
            ]
        }
        result = parse_names_lr(names)
        # Bare name should be selected (higher priority)
        assert result.ranges[0].value == "street"

    def test_variant_priority(self):
        """Common variant has priority over alternate."""
        names = {
            "rules": [
                {"value": "Alt Name", "variant": "alternate"},
                {"value": "Common Name", "variant": "common"},
            ]
        }
        result = parse_names_lr(names)
        # Common variant should be selected (higher priority)
        assert result.ranges[0].value == "Common Name"

    def test_empty_rules(self):
        """Empty rules array falls back to primary."""
        names = {"primary": "Main St", "rules": []}
        result = parse_names_lr(names)
        assert result.ranges[0].value == "Main St"

    def test_non_string_value_skipped(self):
        """Rules with non-string values are skipped."""
        names = {
            "primary": "Default",
            "rules": [{"value": 123}],  # Not a string
        }
        result = parse_names_lr(names)
        assert result.ranges[0].value == "Default"
