"""Tests for feature extraction."""

import math

import numpy as np
import pytest
from shapely import LineString

from matcher.features.geometric import (
    _buffer_iou_from_buffers,
    compute_buffer_iou_batch,
    compute_collinear_gap_ratio,
    compute_geometric_features,
    compute_segment_heading,
)
from matcher.features.semantic import (
    _has_non_latin_alpha,
    _names_are_cross_script,
    compute_class_similarity,
    compute_name_similarity,
    get_class_info,
    get_traffic_tier,
    names_likely_same_road,
    resolve_best_name_variant,
)


class TestGeometricFeatures:
    """Tests for geometric feature extraction."""

    def test_identical_lines(self):
        """Identical lines should have perfect geometric scores."""
        line = LineString([(0, 0), (100, 0)])

        features = compute_geometric_features(line, line)

        assert features.hausdorff_distance == pytest.approx(0.0)
        assert features.mean_hausdorff_distance == pytest.approx(0.0)
        assert features.hausdorff_p95_distance == pytest.approx(0.0)
        assert features.buffer_iou_5m == pytest.approx(1.0, abs=0.01)
        assert features.buffer_iou_15m == pytest.approx(1.0, abs=0.01)
        assert features.heading_delta == pytest.approx(0.0)
        assert features.length_ratio == pytest.approx(1.0)

    def test_parallel_lines(self):
        """Parallel lines should have 0 heading delta."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(0, 10), (100, 10)])

        features = compute_geometric_features(line_a, line_b)

        assert features.heading_delta == pytest.approx(0.0)
        assert features.hausdorff_distance == pytest.approx(10.0)
        assert features.length_ratio == pytest.approx(1.0)

    def test_perpendicular_lines(self):
        """Perpendicular lines should have 90 degree heading delta."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(50, -50), (50, 50)])

        features = compute_geometric_features(line_a, line_b)

        assert features.heading_delta == pytest.approx(90.0, abs=1.0)

    def test_opposite_direction_lines(self):
        """Opposite direction lines should have 0 heading delta (roads are bidirectional)."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(100, 0), (0, 0)])  # Same line, opposite direction

        features = compute_geometric_features(line_a, line_b)

        # Should be 0 because roads can be traversed in either direction
        assert features.heading_delta == pytest.approx(0.0, abs=1.0)

    def test_different_length_lines(self):
        """Lines of different lengths should have correct length ratio."""
        line_a = LineString([(0, 0), (100, 0)])  # Length 100
        line_b = LineString([(0, 0), (50, 0)])  # Length 50

        features = compute_geometric_features(line_a, line_b)

        assert features.length_ratio == pytest.approx(0.5)

    def test_buffer_iou_no_overlap(self):
        """Non-overlapping lines should have low buffer IoU."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(0, 100), (100, 100)])  # 100m apart

        features = compute_geometric_features(line_a, line_b)

        # With 5m/15m buffers, 100m apart lines should not overlap
        assert features.buffer_iou_5m < 0.1
        assert features.buffer_iou_15m < 0.1

    def test_buffer_iou_batch_matches_per_pair(self):
        """Batch buffer IoU should produce identical results to per-pair computation."""
        lines_a = [
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 0), (50, 0)]),
        ]
        lines_b = [
            LineString([(0, 5), (100, 5)]),  # Close parallel
            LineString([(0, 100), (100, 100)]),  # Far apart
            LineString([(25, 3), (75, 3)]),  # Partial overlap
        ]
        radius = 15.0

        # Per-pair results
        per_pair = []
        bufs_a = []
        bufs_b = []
        for la, lb in zip(lines_a, lines_b):
            ba = la.buffer(radius)
            bb = lb.buffer(radius)
            bufs_a.append(ba)
            bufs_b.append(bb)
            per_pair.append(_buffer_iou_from_buffers(ba, bb))

        # Batch results
        batch = compute_buffer_iou_batch(
            np.array(bufs_a, dtype=object),
            np.array(bufs_b, dtype=object),
        )

        for i in range(len(lines_a)):
            assert batch[i] == pytest.approx(per_pair[i], abs=1e-10)


class TestSemanticFeatures:
    """Tests for semantic feature extraction."""

    def test_name_similarity_exact(self):
        """Exact name match should return 1.0."""
        result = compute_name_similarity("Main Street", "Main Street")

        assert result["levenshtein_ratio"] == pytest.approx(1.0)
        assert result["token_sort_ratio"] == pytest.approx(1.0)

    def test_name_similarity_abbreviation(self):
        """Common abbreviations should score high after normalization."""
        result = compute_name_similarity("Main St", "Main Street")

        # After normalization, should be identical
        assert result["token_sort_ratio"] > 0.9

    def test_name_similarity_direction_prefix(self):
        """Direction prefixes should be normalized."""
        result = compute_name_similarity("N Main St", "North Main Street")

        assert result["token_sort_ratio"] > 0.9

    def test_name_similarity_none(self):
        """Missing names should return NaN scores and flag as missing."""
        result = compute_name_similarity(None, "Main Street")

        # NaN signals missing data — XGBoost handles natively
        assert math.isnan(result["levenshtein_ratio"])
        assert math.isnan(result["token_sort_ratio"])
        assert result["names_missing"] is True

    def test_name_similarity_both_none(self):
        """Both names missing should return NaN scores."""
        result = compute_name_similarity(None, None)

        assert math.isnan(result["levenshtein_ratio"])
        assert result["names_missing"] is True

    @pytest.mark.parametrize(
        "class_a,class_b,expected_min,expected_max",
        [
            ("primary", "primary", 1.0, 1.0),  # same class
            ("primary", "secondary", 0.7, 1.0),  # adjacent classes
            ("motorway", "residential", 0.0, 0.5),  # distant classes
        ],
        ids=["same_class", "adjacent_classes", "distant_classes"],
    )
    def test_class_similarity(self, class_a, class_b, expected_min, expected_max):
        """Class similarity should vary based on road class distance."""
        result = compute_class_similarity(class_a, class_b)
        assert expected_min <= result <= expected_max

    @pytest.mark.parametrize(
        "class_a,class_b,subclass_a,subclass_b,expected",
        [
            ("footway", "footway", "sidewalk", "sidewalk", 1.0),  # same class+subclass
            ("footway", "footway", "sidewalk", "crosswalk", 0.85),  # same class, diff subclass
            ("footway", "footway", "sidewalk", None, 0.9),  # same class, one subclass missing
        ],
        ids=["same_subclass", "different_subclass", "one_subclass_missing"],
    )
    def test_class_similarity_with_subclass(
        self, class_a, class_b, subclass_a, subclass_b, expected
    ):
        """Class+subclass similarity should account for subclass differences."""
        result = compute_class_similarity(class_a, class_b, subclass_a, subclass_b)
        assert result == pytest.approx(expected)

    def test_names_likely_same_road(self):
        """Test quick name matching heuristic."""
        assert names_likely_same_road("Main Street", "Main St")
        assert names_likely_same_road("Interstate 5", "I-5")
        assert not names_likely_same_road("Main Street", "Oak Avenue")


class TestNonLatinNameHandling:
    """Tests for non-Latin script handling (CJK, Arabic, Cyrillic, etc.)."""

    # -- Script detection --

    @pytest.mark.parametrize(
        "text",
        [
            "皇后大道中",  # Chinese
            "北京路",
            "東京駅",  # Japanese Kanji
            "とうきょう",  # Hiragana
            "トウキョウ",  # Katakana
            "서울로",  # Korean Hangul
            "شارع الملك",  # Arabic
            "Невский проспект",  # Cyrillic
            "राजपथ",  # Devanagari
            "ถนนสุขุมวิท",  # Thai
        ],
        ids=[
            "chinese",
            "chinese_road",
            "japanese_kanji",
            "hiragana",
            "katakana",
            "korean",
            "arabic",
            "cyrillic",
            "devanagari",
            "thai",
        ],
    )
    def test_non_latin_detected(self, text):
        """Non-Latin scripts should be detected."""
        assert _has_non_latin_alpha(text)

    def test_latin_not_detected_as_non_latin(self):
        """Pure Latin text should not be detected as non-Latin."""
        assert not _has_non_latin_alpha("Main Street")
        assert not _has_non_latin_alpha("Queen's Road Central")
        assert not _has_non_latin_alpha("123 Route 66")  # digits/numbers only

    def test_mixed_script_detected(self):
        """Mixed non-Latin + Latin text should be detected as non-Latin."""
        assert _has_non_latin_alpha("北京 Beijing Road")

    def test_cross_script_detection(self):
        """Cross-script pairs should be detected."""
        assert _names_are_cross_script("皇后大道中", "Queen's Road Central")
        assert _names_are_cross_script("東京駅", "Tokyo Station")
        assert _names_are_cross_script("شارع الملك", "King Street")  # Arabic vs Latin
        assert _names_are_cross_script("Невский", "Nevsky")  # Cyrillic vs Latin
        assert not _names_are_cross_script("Main Street", "Oak Avenue")  # Both Latin
        assert not _names_are_cross_script("北京路", "上海路")  # Both CJK

    # -- Phonetic feature handling --

    def test_non_latin_phonetic_features_return_nan(self):
        """Soundex and Metaphone should return NaN for non-Latin names."""
        result = compute_name_similarity("皇后大道中", "北京路")
        assert math.isnan(result["soundex_match"])
        assert math.isnan(result["metaphone_similarity"])

    def test_non_latin_phonetic_nan_cross_script(self):
        """Phonetics should return NaN when one name is non-Latin and other is Latin."""
        result = compute_name_similarity("皇后大道中", "Queen's Road Central")
        assert math.isnan(result["soundex_match"])
        assert math.isnan(result["metaphone_similarity"])

    def test_arabic_phonetic_nan(self):
        """Arabic names should also produce NaN phonetics."""
        result = compute_name_similarity("شارع الملك", "King Street")
        assert math.isnan(result["soundex_match"])
        assert math.isnan(result["metaphone_similarity"])

    def test_cyrillic_phonetic_nan(self):
        """Cyrillic names should also produce NaN phonetics."""
        result = compute_name_similarity("Невский проспект", "Nevsky Prospect")
        assert math.isnan(result["soundex_match"])
        assert math.isnan(result["metaphone_similarity"])

    def test_latin_phonetics_still_work(self):
        """Latin-only names should still get proper phonetic scores."""
        result = compute_name_similarity("Main Street", "Maine Street")
        assert not math.isnan(result["soundex_match"])
        assert not math.isnan(result["metaphone_similarity"])

    def test_non_latin_same_name_high_levenshtein(self):
        """Identical non-Latin names should get high Levenshtein scores."""
        result = compute_name_similarity("皇后大道中", "皇后大道中")
        assert result["levenshtein_ratio"] == pytest.approx(1.0)
        assert result["jaro_winkler"] == pytest.approx(1.0)

    def test_unicode_normalization_fullwidth(self):
        """Full-width Latin chars should match half-width after NFKC normalization."""
        # Ｍａｉｎ (full-width) vs Main (half-width)
        result = compute_name_similarity("Ｍａｉｎ Ｓｔｒｅｅｔ", "Main Street")
        assert result["levenshtein_ratio"] > 0.9

    def test_resolve_best_variant_selects_english(self):
        """When Overture has Chinese primary + English alt, should pick English."""
        ref_names_raw = {
            "primary": "皇后大道中",
            "rules": [
                {"value": "皇后大道中", "language": "zh", "variant": "common"},
                {"value": "Queen's Road Central", "language": "en", "variant": "common"},
            ],
        }
        best = resolve_best_name_variant(ref_names_raw, "皇后大道中", "Queen's Road Central")
        assert best == "Queen's Road Central"

    def test_resolve_best_variant_keeps_original_when_no_alt(self):
        """When only one variant exists, should keep the original name."""
        ref_names_raw = {"primary": "皇后大道中", "rules": []}
        best = resolve_best_name_variant(ref_names_raw, "皇后大道中", "Queen's Road Central")
        assert best == "皇后大道中"

    def test_resolve_best_variant_none_raw(self):
        """When ref_names_raw is None, should return ref_name unchanged."""
        best = resolve_best_name_variant(None, "Main Street", "Main St")
        assert best == "Main Street"

    def test_resolve_best_variant_picks_closer_match(self):
        """When variants have different similarity, should pick the best one."""
        ref_names_raw = {
            "primary": "Rue de Rivoli",
            "rules": [
                {"value": "Rue de Rivoli", "language": "fr", "variant": "common"},
                {"value": "Rivoli Street", "language": "en", "variant": "common"},
            ],
        }
        best = resolve_best_name_variant(ref_names_raw, "Rue de Rivoli", "Rivoli Street")
        # Should prefer the English variant since it matches the target better
        assert best == "Rivoli Street"

    def test_resolve_best_variant_none_target(self):
        """When target_name is None, should return ref_name unchanged."""
        ref_names_raw = {
            "primary": "Main Street",
            "rules": [{"value": "Main St", "language": "en"}],
        }
        best = resolve_best_name_variant(ref_names_raw, "Main Street", None)
        assert best == "Main Street"

    def test_resolve_best_variant_none_ref_name(self):
        """When ref_name is None but variants exist, should still pick best."""
        ref_names_raw = {
            "primary": "皇后大道中",
            "rules": [
                {"value": "Queen's Road Central", "language": "en", "variant": "common"},
            ],
        }
        best = resolve_best_name_variant(ref_names_raw, None, "Queen's Road Central")
        # ref_name is None, but target_name exists so we still resolve variants
        assert best == "Queen's Road Central"

    def test_resolve_best_variant_duplicate_values_deduplicated(self):
        """Duplicate name values (case-insensitive) should be deduplicated."""
        ref_names_raw = {
            "primary": "Main Street",
            "rules": [
                {"value": "main street", "language": "en"},
                {"value": "MAIN STREET", "language": "en", "variant": "official"},
                {"value": "Oak Avenue", "language": "en", "variant": "alt"},
            ],
        }
        best = resolve_best_name_variant(ref_names_raw, "Main Street", "Oak Avenue")
        assert best == "Oak Avenue"

    def test_resolve_best_variant_empty_rules_list(self):
        """Empty rules list should fall back to ref_name (only 1 variant)."""
        ref_names_raw = {"primary": "Main Street", "rules": []}
        best = resolve_best_name_variant(ref_names_raw, "Main Street", "Oak Avenue")
        assert best == "Main Street"

    def test_resolve_best_variant_no_rules_key(self):
        """Missing rules key should fall back to ref_name (only 1 variant)."""
        ref_names_raw = {"primary": "Main Street"}
        best = resolve_best_name_variant(ref_names_raw, "Main Street", "Oak Avenue")
        assert best == "Main Street"

    def test_resolve_best_variant_malformed_rules(self):
        """Rules with missing/invalid values should be skipped gracefully."""
        ref_names_raw = {
            "primary": "北京路",
            "rules": [
                {"language": "zh"},  # Missing 'value' key
                {"value": None, "language": "en"},  # None value
                {"value": 123, "language": "en"},  # Non-string value
                {"value": "", "language": "en"},  # Empty string
                {"value": "Beijing Road", "language": "en"},  # Valid
            ],
        }
        best = resolve_best_name_variant(ref_names_raw, "北京路", "Beijing Road")
        assert best == "Beijing Road"

    def test_resolve_best_variant_non_dict_rules(self):
        """Non-list rules should fall back to ref_name."""
        ref_names_raw = {"primary": "Main Street", "rules": "invalid"}
        best = resolve_best_name_variant(ref_names_raw, "Main Street", "Oak Avenue")
        assert best == "Main Street"

    def test_resolve_best_variant_target_is_dict(self):
        """Target name as dict format should be handled."""
        ref_names_raw = {
            "primary": "北京路",
            "rules": [
                {"value": "Beijing Road", "language": "en"},
            ],
        }
        best = resolve_best_name_variant(ref_names_raw, "北京路", {"primary": "Beijing Road"})
        assert best == "Beijing Road"

    def test_resolve_best_variant_three_languages(self):
        """Should pick the best match from three language variants."""
        ref_names_raw = {
            "primary": "東京駅",
            "rules": [
                {"value": "東京駅", "language": "ja", "variant": "common"},
                {"value": "Tokyo Station", "language": "en", "variant": "common"},
                {"value": "도쿄역", "language": "ko", "variant": "common"},
            ],
        }
        # Target is Korean
        best = resolve_best_name_variant(ref_names_raw, "東京駅", "도쿄역")
        assert best == "도쿄역"
        # Target is English
        best = resolve_best_name_variant(ref_names_raw, "東京駅", "Tokyo Station")
        assert best == "Tokyo Station"


class TestExtractAllNameVariants:
    """Exhaustive tests for _extract_all_name_variants covering the full Overture Names schema.

    Overture Names schema has three sources of name strings:
    - primary: Single default name string
    - common: Dict of language code -> name (multilingual common names)
    - rules: Array of NameRule dicts with value, variant, language, between, side

    See: https://docs.overturemaps.org/schema/reference/transportation/segment/
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from matcher.features.semantic import _extract_all_name_variants

        self.extract = _extract_all_name_variants

    # -- Invalid / empty inputs --

    def test_none_returns_empty(self):
        assert self.extract(None) == []

    def test_empty_dict_returns_empty(self):
        assert self.extract({}) == []

    def test_non_dict_returns_empty(self):
        assert self.extract("not a dict") == []
        assert self.extract(42) == []
        assert self.extract([]) == []

    # -- primary only --

    def test_primary_only(self):
        assert self.extract({"primary": "Main Street"}) == ["Main Street"]

    def test_primary_none(self):
        assert self.extract({"primary": None}) == []

    def test_primary_empty_string(self):
        assert self.extract({"primary": ""}) == []

    def test_primary_non_string(self):
        assert self.extract({"primary": 123}) == []

    # -- common dict (Overture multilingual common names) --

    def test_common_dict_multilingual(self):
        """common dict should extract names from all languages."""
        names = {
            "primary": "Le Léman",
            "common": {
                "de": "Genfersee",
                "en": "Lake Geneva",
                "fr": "Le Léman",  # duplicate of primary (case-insensitive)
            },
        }
        variants = self.extract(names)
        assert "Le Léman" in variants
        assert "Genfersee" in variants
        assert "Lake Geneva" in variants
        assert len(variants) == 3  # Le Léman deduplicated

    def test_common_dict_only_no_primary(self):
        """common dict without primary should still extract names."""
        names = {"common": {"en": "Lake Geneva", "fr": "Le Léman"}}
        variants = self.extract(names)
        assert len(variants) == 2
        assert "Lake Geneva" in variants
        assert "Le Léman" in variants

    def test_common_dict_with_none_values(self):
        """None values in common dict should be skipped."""
        names = {"primary": "Main", "common": {"en": None, "fr": "Principale"}}
        variants = self.extract(names)
        assert variants == ["Main", "Principale"]

    def test_common_dict_non_dict_ignored(self):
        """Non-dict common should be ignored gracefully."""
        names = {"primary": "Main", "common": "not a dict"}
        assert self.extract(names) == ["Main"]

    def test_common_dict_empty(self):
        names = {"primary": "Main", "common": {}}
        assert self.extract(names) == ["Main"]

    # -- rules array --

    def test_rules_with_multiple_variants(self):
        """Rules with official, alternate, and short names."""
        names = {
            "primary": "City of New York",
            "rules": [
                {"value": "New York", "variant": "official", "language": "en"},
                {"value": "New York City", "variant": "alternate", "language": "en"},
                {"value": "The Big Apple", "variant": "alternate", "language": "en"},
                {"value": "NYC", "variant": "alternate"},
            ],
        }
        variants = self.extract(names)
        assert len(variants) == 5
        assert "City of New York" in variants
        assert "New York" in variants
        assert "New York City" in variants
        assert "The Big Apple" in variants
        assert "NYC" in variants

    def test_rules_with_between_ranges(self):
        """Rules with between ranges should still have their values extracted.

        Note: _extract_all_name_variants intentionally ignores the between range
        for cross-language variant resolution. The LR name resolution in
        parse_names_lr handles range-specific selection for the primary comparison.
        """
        names = {
            "primary": "Fir St",
            "rules": [
                {"value": "2 Ave", "variant": "common", "between": [0, 0.3], "side": "left"},
                {"value": "Fir St", "variant": "common", "between": [0.3, 1], "side": "left"},
                {"value": "Fir St", "variant": "common", "side": "right"},
            ],
        }
        variants = self.extract(names)
        # "Fir St" appears 3 times but should be deduplicated
        assert "Fir St" in variants
        assert "2 Ave" in variants
        assert len(variants) == 2

    def test_rules_with_side_scoping(self):
        """Rules with side (left/right) scoping."""
        names = {
            "primary": "Main St",
            "rules": [
                {"value": "Elm Dr", "variant": "common", "side": "left"},
                {"value": "Main St", "variant": "common", "side": "right"},
            ],
        }
        variants = self.extract(names)
        assert len(variants) == 2
        assert "Main St" in variants
        assert "Elm Dr" in variants

    def test_rules_multilingual_with_between(self):
        """Multilingual rules with LR scoping — real-world HK scenario."""
        names = {
            "primary": "皇后大道中",
            "rules": [
                {
                    "value": "皇后大道中",
                    "variant": "common",
                    "language": "zh",
                    "between": [0.0, 1.0],
                },
                {
                    "value": "Queen's Road Central",
                    "variant": "common",
                    "language": "en",
                    "between": [0.0, 1.0],
                },
                {
                    "value": "皇后大道西",
                    "variant": "common",
                    "language": "zh",
                    "between": [0.0, 0.4],
                },
                {
                    "value": "Queen's Road West",
                    "variant": "common",
                    "language": "en",
                    "between": [0.0, 0.4],
                },
            ],
        }
        variants = self.extract(names)
        assert "皇后大道中" in variants
        assert "Queen's Road Central" in variants
        assert "皇后大道西" in variants
        assert "Queen's Road West" in variants
        assert len(variants) == 4

    # -- Deduplication --

    def test_dedup_case_insensitive(self):
        """Duplicate names differing only in case should be deduplicated."""
        names = {
            "primary": "MAIN STREET",
            "common": {"en": "Main Street"},
            "rules": [{"value": "main street", "variant": "common"}],
        }
        variants = self.extract(names)
        assert len(variants) == 1
        # First occurrence wins
        assert variants[0] == "MAIN STREET"

    def test_dedup_across_sources(self):
        """Same name appearing in primary, common, and rules should appear once."""
        names = {
            "primary": "Oak Ave",
            "common": {"en": "Oak Ave"},
            "rules": [{"value": "Oak Ave", "variant": "official"}],
        }
        assert self.extract(names) == ["Oak Ave"]

    # -- Malformed rules --

    def test_rules_missing_value_key(self):
        names = {
            "primary": "Main",
            "rules": [{"language": "en"}, {"value": "Alt", "language": "en"}],
        }
        assert self.extract(names) == ["Main", "Alt"]

    def test_rules_none_value(self):
        names = {"primary": "Main", "rules": [{"value": None}]}
        assert self.extract(names) == ["Main"]

    def test_rules_non_string_value(self):
        names = {"primary": "Main", "rules": [{"value": 123}]}
        assert self.extract(names) == ["Main"]

    def test_rules_empty_string_value(self):
        names = {"primary": "Main", "rules": [{"value": ""}]}
        assert self.extract(names) == ["Main"]

    def test_rules_non_dict_entries(self):
        names = {"primary": "Main", "rules": ["not a dict", 42, None]}
        assert self.extract(names) == ["Main"]

    def test_rules_non_list(self):
        names = {"primary": "Main", "rules": "not a list"}
        assert self.extract(names) == ["Main"]

    def test_rules_none(self):
        names = {"primary": "Main", "rules": None}
        assert self.extract(names) == ["Main"]

    # -- Full Overture schema: primary + common + rules --

    def test_all_three_sources(self):
        """Extract from primary, common dict, and rules simultaneously."""
        names = {
            "primary": "皇后大道中",
            "common": {
                "en": "Queen's Road Central",
                "zh-Hant": "皇后大道中",  # duplicate of primary
            },
            "rules": [
                {"value": "QRC", "variant": "short", "language": "en"},
                {"value": "皇后大道中", "variant": "common", "language": "zh"},  # dup
            ],
        }
        variants = self.extract(names)
        assert "皇后大道中" in variants
        assert "Queen's Road Central" in variants
        assert "QRC" in variants
        assert len(variants) == 3

    # -- resolve_best_name_variant with common dict --

    def test_resolve_uses_common_dict(self):
        """resolve_best_name_variant should find names from common dict."""
        names = {
            "primary": "Le Léman",
            "common": {"en": "Lake Geneva", "de": "Genfersee"},
        }
        best = resolve_best_name_variant(names, "Le Léman", "Lake Geneva")
        assert best == "Lake Geneva"

    def test_resolve_uses_common_dict_cyrillic(self):
        """Cyrillic primary with Latin common should resolve to Latin."""
        names = {
            "primary": "Невский проспект",
            "common": {"en": "Nevsky Prospect"},
        }
        best = resolve_best_name_variant(names, "Невский проспект", "Nevsky Prospect")
        assert best == "Nevsky Prospect"


class TestClassInfo:
    """Tests for get_class_info diagnostic function."""

    @pytest.mark.parametrize(
        "input_class,expected_normalized,expected_known,expected_rank",
        [
            ("motorway", "motorway", True, 1),
            ("RESIDENTIAL", "residential", True, 6),  # case-insensitive
            ("some_unknown_class", "some_unknown_class", False, 6),  # unknown -> default rank
            (None, None, False, None),  # None input
            ("motorway_link", "motorway_link", True, 1),  # link roads
            ("footway", "footway", True, 10),  # pedestrian
        ],
        ids=[
            "known_class",
            "case_insensitive",
            "unknown_class",
            "none_input",
            "link_road",
            "pedestrian",
        ],
    )
    def test_class_info_lookup(
        self, input_class, expected_normalized, expected_known, expected_rank
    ):
        """get_class_info should return correct info for various inputs."""
        result = get_class_info(input_class)
        assert result["normalized"] == expected_normalized
        assert result["known"] is expected_known
        assert result["rank"] == expected_rank


class TestPhoneticFeatures:
    """Tests for phonetic name matching features."""

    def test_soundex_match_same_sound(self):
        """Phonetically similar names should match via Soundex."""
        # "Main" and "Mane" have the same Soundex code (M500)
        result = compute_name_similarity("Main Street", "Mane Street")
        assert result["soundex_match"] == 1.0

    def test_soundex_no_match_different_sound(self):
        """Phonetically different names should not match via Soundex."""
        result = compute_name_similarity("Main Street", "Oak Street")
        assert result["soundex_match"] == 0.0

    def test_metaphone_typo_tolerance(self):
        """Metaphone should tolerate common typos."""
        result = compute_name_similarity("Main Street", "Main Stret")
        assert result["metaphone_similarity"] > 0.8

    def test_metaphone_similar_names(self):
        """Metaphone should give high similarity for similar-sounding names."""
        result = compute_name_similarity("Washington Avenue", "Washingten Avenue")
        assert result["metaphone_similarity"] > 0.9

    def test_phonetic_missing_one_name(self):
        """Missing one name should return NaN phonetic scores."""
        result = compute_name_similarity(None, "Main Street")
        assert math.isnan(result["soundex_match"])
        assert math.isnan(result["metaphone_similarity"])

    def test_phonetic_missing_both_names(self):
        """Missing both names should return NaN phonetic scores."""
        result = compute_name_similarity(None, None)
        assert math.isnan(result["soundex_match"])
        assert math.isnan(result["metaphone_similarity"])

    def test_phonetic_with_abbreviations(self):
        """Phonetic matching should work with abbreviations after normalization."""
        result = compute_name_similarity("N Main St", "North Main Street")
        # After normalization, both become "north main street"
        assert result["soundex_match"] == 1.0
        assert result["metaphone_similarity"] > 0.9


class TestComputeSegmentHeading:
    """Tests for segment heading calculation."""

    @pytest.mark.parametrize(
        "end_point,expected_heading",
        [
            ((100, 0), 0.0),  # East
            ((0, 100), 90.0),  # North
            ((100, 100), 45.0),  # Northeast
            ((-100, 0), 180.0),  # West
            ((0, -100), 270.0),  # South
        ],
        ids=["east", "north", "northeast", "west", "south"],
    )
    def test_heading_by_direction(self, end_point, expected_heading):
        """Segment heading should match expected angle (0-360) for various directions."""
        line = LineString([(0, 0), end_point])
        heading = compute_segment_heading(line)
        assert heading == pytest.approx(expected_heading, abs=1.0)


class TestCollinearGapRatio:
    """Tests for collinear gap penalty feature.

    This feature detects "tip-to-tip" collinear segments that should not match
    because they represent consecutive road segments, not the same segment.
    """

    @pytest.mark.parametrize(
        "coords_a,coords_b",
        [
            # Identical lines (perfect overlap → 1.0)
            ([(0, 0), (100, 0)], [(0, 0), (100, 0)]),
            # Perpendicular (not collinear, heading > 15° → 1.0)
            ([(0, 0), (100, 0)], [(50, -50), (50, 50)]),
            # Parallel offset (good along-track overlap despite lateral offset)
            ([(0, 0), (100, 0)], [(0, 10), (100, 10)]),
            # Empty line (degenerate case)
            ([(0, 0), (100, 0)], []),
        ],
        ids=[
            "identical_lines",
            "perpendicular",
            "parallel_offset",
            "empty_line",
        ],
    )
    def test_no_penalty_cases(self, coords_a, coords_b):
        """Cases that should have no penalty (ratio = 1.0)."""
        line_a = LineString(coords_a)
        line_b = LineString(coords_b) if coords_b else LineString()
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result == pytest.approx(1.0)

    def test_partial_overlap_returns_fraction(self):
        """50% overlap should return ~0.5 (raw overlap fraction)."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(50, 0), (150, 0)])
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result == pytest.approx(0.5, abs=0.05)

    @pytest.mark.parametrize(
        "coords_a,coords_b",
        [
            # Tip-to-tip horizontal
            ([(0, 0), (100, 0)], [(100, 0), (200, 0)]),
            # Opposite direction tip-to-tip
            ([(0, 0), (100, 0)], [(200, 0), (100, 0)]),
            # Diagonal tip-to-tip
            ([(0, 0), (100, 100)], [(100, 100), (200, 200)]),
        ],
        ids=[
            "tip_to_tip_horizontal",
            "tip_to_tip_opposite_direction",
            "tip_to_tip_diagonal",
        ],
    )
    def test_strong_penalty_cases(self, coords_a, coords_b):
        """Tip-to-tip collinear segments should receive strong penalty (ratio < 0.1)."""
        line_a = LineString(coords_a)
        line_b = LineString(coords_b)
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result < 0.1

    def test_gap_between_collinear_zero_penalty(self):
        """Collinear segments with a gap should have zero overlap ratio."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(110, 0), (200, 0)])  # 10m gap
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result == pytest.approx(0.0)

    def test_contained_segment_no_penalty(self):
        """Segment fully contained within another should have no penalty."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(25, 0), (75, 0)])  # 50m fully contained
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result > 0.9

    def test_small_overlap_returns_raw_fraction(self):
        """Small overlap should return raw fraction (5m / 100m = 0.05)."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(95, 0), (195, 0)])  # 5m overlap out of 100m
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result == pytest.approx(0.05, abs=0.02)

    def test_included_in_geometric_features(self):
        """collinear_gap_ratio should be included in compute_geometric_features output."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(100, 0), (200, 0)])
        features = compute_geometric_features(line_a, line_b)
        assert hasattr(features, "collinear_gap_ratio")
        assert features.collinear_gap_ratio < 0.1


class TestTrafficTierClassSimilarity:
    """Tests for traffic tier-based class similarity scoring.

    Traffic tiers separate road types by traffic type:
    - vehicle: motorway, trunk, primary, secondary, tertiary, residential, etc.
    - bicycle: cycleway
    - pedestrian: footway, sidewalk, path, pedestrian, steps
    - neutral: bridleway (uncommon, treated neutrally)

    Cross-tier penalties:
    - vehicle↔pedestrian: 0.1 (strong - incompatible traffic types)
    - vehicle↔bicycle: 0.7 (mild - bikes often share roads)
    - bicycle↔pedestrian: 0.5 (moderate - shared paths exist)
    """

    @pytest.mark.parametrize(
        "road_class,expected_tier",
        [
            # Vehicle tier
            ("motorway", "vehicle"),
            ("motorway_link", "vehicle"),
            ("trunk", "vehicle"),
            ("primary", "vehicle"),
            ("secondary", "vehicle"),
            ("tertiary", "vehicle"),
            ("residential", "vehicle"),
            ("living_street", "vehicle"),
            ("service", "vehicle"),
            ("unclassified", "vehicle"),
            ("track", "vehicle"),
            # Bicycle tier
            ("cycleway", "bicycle"),
            # Pedestrian tier
            ("footway", "pedestrian"),
            ("sidewalk", "pedestrian"),
            ("path", "pedestrian"),
            ("pedestrian", "pedestrian"),
            ("steps", "pedestrian"),
            # Neutral tier
            ("bridleway", "neutral"),
            # Unknown class -> None
            ("unknown_class", None),
            (None, None),
        ],
        ids=[
            "motorway",
            "motorway_link",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "residential",
            "living_street",
            "service",
            "unclassified",
            "track",
            "cycleway",
            "footway",
            "sidewalk",
            "path",
            "pedestrian",
            "steps",
            "bridleway",
            "unknown_class",
            "none_input",
        ],
    )
    def test_get_traffic_tier(self, road_class, expected_tier):
        """get_traffic_tier should return correct tier for each road class."""
        result = get_traffic_tier(road_class)
        assert result == expected_tier

    @pytest.mark.parametrize(
        "class_a,class_b,expected",
        [
            # Cross-tier: vehicle vs pedestrian (strong penalty)
            ("residential", "footway", 0.1),
            ("primary", "sidewalk", 0.1),
            ("service", "path", 0.1),
            ("motorway", "pedestrian", 0.1),
            ("tertiary", "steps", 0.1),
            # Cross-tier: vehicle vs bicycle (mild penalty - bikes share roads)
            ("residential", "cycleway", 0.7),
            ("primary", "cycleway", 0.7),
            ("motorway", "cycleway", 0.7),
            # Cross-tier: pedestrian vs bicycle (moderate)
            ("footway", "cycleway", 0.5),
            ("sidewalk", "cycleway", 0.5),
            ("path", "cycleway", 0.5),
            # Same tier: pedestrian (all pedestrian classes have same rank 10)
            ("footway", "sidewalk", 1.0),
            ("footway", "path", 1.0),
            ("sidewalk", "pedestrian", 1.0),
            # Same tier: vehicle (existing rank logic)
            ("residential", "service", 0.8),
            ("primary", "secondary", 0.8),
            ("motorway", "trunk", 0.8),
            # Neutral tier: bridleway -> NaN (unknown signal)
            ("bridleway", "residential", float("nan")),
            ("bridleway", "footway", float("nan")),
            ("bridleway", "cycleway", float("nan")),
            # Unknown -> NaN (unknown signal)
            ("residential", "unknown", float("nan")),
            (None, "footway", float("nan")),
            ("", "residential", float("nan")),
        ],
        ids=[
            "vehicle_pedestrian_residential_footway",
            "vehicle_pedestrian_primary_sidewalk",
            "vehicle_pedestrian_service_path",
            "vehicle_pedestrian_motorway_pedestrian",
            "vehicle_pedestrian_tertiary_steps",
            "vehicle_bicycle_residential",
            "vehicle_bicycle_primary",
            "vehicle_bicycle_motorway",
            "pedestrian_bicycle_footway",
            "pedestrian_bicycle_sidewalk",
            "pedestrian_bicycle_path",
            "pedestrian_same_footway_sidewalk",
            "pedestrian_same_footway_path",
            "pedestrian_same_sidewalk_pedestrian",
            "vehicle_same_residential_service",
            "vehicle_same_primary_secondary",
            "vehicle_same_motorway_trunk",
            "neutral_bridleway_vehicle",
            "neutral_bridleway_pedestrian",
            "neutral_bridleway_bicycle",
            "unknown_class",
            "none_class",
            "empty_class",
        ],
    )
    def test_traffic_tier_class_similarity(self, class_a, class_b, expected):
        """Class similarity should use traffic tier penalties for cross-tier comparisons."""
        result = compute_class_similarity(class_a, class_b)
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(result), f"Expected NaN for ({class_a}, {class_b}), got {result}"
        else:
            assert result == pytest.approx(expected, abs=0.05)


class TestComputePairFeaturesWithAlignment:
    """Tests for compute_pair_features with alignment parameter."""

    def test_compute_pair_features_includes_coverage_features(self):
        """compute_pair_features should include coverage features when alignment provided."""
        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (100, 0)])
        alignment = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name="Main Street",
            target_name="Main Street",
            ref_class="residential",
            target_class="residential",
            alignment=alignment,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # Should include coverage features
        assert "ref_coverage" in features
        assert "target_coverage" in features
        assert "min_coverage" in features
        assert "coverage_ratio" in features

        # Full alignment should have full coverage
        assert features["ref_coverage"] == pytest.approx(1.0)
        assert features["target_coverage"] == pytest.approx(1.0)

    def test_compute_pair_features_without_alignment(self):
        """compute_pair_features should work without alignment (backward compatible)."""
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (100, 0)])

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name="Main Street",
            target_name="Main Street",
            ref_class="residential",
            target_class="residential",
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # Should still include coverage features (zeros without alignment)
        assert "ref_coverage" in features
        assert features["ref_coverage"] == 0.0

    def test_compute_pair_features_uses_sublines_with_alignment(self):
        """With alignment, similarity features should be computed on sublines."""
        from matcher.features.alignment import linestring_alignment
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        # Reference is longer than target, target matches second half
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(50, 2), (100, 2)])  # Small offset, second half

        alignment = linestring_alignment(ref, target)

        features_aligned = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            alignment=alignment,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        features_unaligned = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            alignment=None,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # Aligned features should have better (lower) hausdorff because
        # we compare the matching portions only
        # (The full geometry hausdorff includes the non-overlapping 50m)
        assert (
            features_aligned["hausdorff_distance_m"] <= features_unaligned["hausdorff_distance_m"]
        )

    def test_all_feature_columns_present(self):
        """compute_pair_features should return all expected feature columns."""
        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import ALL_FEATURE_COLUMNS, compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (100, 0)])
        alignment = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name="Main Street",
            target_name="Main Street",
            ref_class="residential",
            target_class="residential",
            alignment=alignment,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # All feature columns should be present
        for col in ALL_FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"

    def test_lateral_offset_uses_aligned_sublines(self):
        """Lateral offset should be computed on aligned sublines, not full geometries.

        Regression test: A target that extends beyond the reference should not
        inflate the lateral offset. Only the overlapping portion should be measured.
        """
        from matcher.features.alignment import linestring_alignment
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        # Reference: 100m segment
        ref = LineString([(0, 0), (100, 0)])
        # Target: 300m segment, first 100m overlaps at 3m offset, then extends 200m
        target = LineString([(0, 3), (100, 3), (300, 3)])

        alignment = linestring_alignment(ref, target)

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            alignment=alignment,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # Lateral offset should be ~3m (the offset in the overlapping region)
        # NOT ~100m (which would happen if measuring full 300m target to 100m ref)
        assert features["lateral_offset_m"] < 10.0, (
            f"Lateral offset {features['lateral_offset_m']:.1f}m is too high. "
            "Should be ~3m for aligned sublines, not inflated by non-overlapping portion."
        )


class TestLengthRatioUsesFullGeometry:
    """Regression tests: length_ratio must use original geometries, not sublines.

    Subline alignment clips both geometries to the matching portion, which makes
    their lengths nearly identical (ratio ~1.0). The length_ratio feature should
    reflect the actual segmentation difference between the full original geometries.
    """

    def test_length_ratio_reflects_original_lengths_with_alignment(self):
        """When target is 3x longer than ref, length_ratio should be ~0.33."""
        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        ref = LineString([(0, 0), (100, 0)])  # 100m
        target = LineString([(0, 0), (300, 0)])  # 300m

        # Alignment says ref covers the first 1/3 of target
        alignment = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=0.0,
            dataset_end_frac=0.333,
        )

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            alignment=alignment,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # length_ratio = min/max = 100/300 ≈ 0.333
        assert features["length_ratio"] == pytest.approx(1 / 3, abs=0.01), (
            f"length_ratio={features['length_ratio']:.3f}, expected ~0.333. "
            "Should use original geometry lengths, not subline lengths."
        )

    def test_length_ratio_without_alignment(self):
        """Without alignment, length_ratio should still reflect original lengths."""
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        ref = LineString([(0, 0), (50, 0)])  # 50m
        target = LineString([(0, 0), (200, 0)])  # 200m

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # length_ratio = 50/200 = 0.25
        assert features["length_ratio"] == pytest.approx(0.25, abs=0.01)


class TestEndpointProximityInfCapping:
    """Regression tests: endpoint proximity must never contain Inf values.

    The JIT helper returns np.inf when no endpoints are found. compute.py must
    cap these to MAX_DISTANCE_METERS so the ML model sees finite values.
    """

    def test_inf_endpoint_proximity_capped(self):
        """Inf values from endpoint computation should be capped to MAX_DISTANCE_METERS."""
        from matcher.config import MAX_DISTANCE_METERS
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_TOPOLOGY_FEATURES

        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (100, 0)])

        # Pass endpoint features with Inf (simulating the JIT helper bug)
        endpoint_features = {
            "min_endpoint_proximity_m": float("inf"),
            "max_endpoint_proximity_m": float("inf"),
            "shared_endpoint_count": 0,
        }

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            endpoint_features=endpoint_features,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        assert features["min_endpoint_proximity_m"] == MAX_DISTANCE_METERS
        assert features["max_endpoint_proximity_m"] == MAX_DISTANCE_METERS

    def test_finite_endpoint_proximity_unchanged(self):
        """Finite endpoint values below MAX_DISTANCE_METERS should pass through unchanged."""
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_TOPOLOGY_FEATURES

        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (100, 0)])

        endpoint_features = {
            "min_endpoint_proximity_m": 5.0,
            "max_endpoint_proximity_m": 42.0,
            "shared_endpoint_count": 1,
        }

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            endpoint_features=endpoint_features,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        assert features["min_endpoint_proximity_m"] == 5.0
        assert features["max_endpoint_proximity_m"] == 42.0
        assert features["shared_endpoint_count"] == 1


class TestAngleHistogramSimilarity:
    """Tests for compute_angle_histogram_similarity function."""

    def test_identical_lines(self):
        """Identical lines should have similarity of 1.0."""
        from matcher.features.geometric import compute_angle_histogram_similarity

        line = LineString([(0, 0), (10, 5), (20, 0), (30, 5)])
        result = compute_angle_histogram_similarity(line, line)
        assert result == pytest.approx(1.0)

    def test_straight_lines_similar(self):
        """Two straight lines should have high similarity."""
        from matcher.features.geometric import compute_angle_histogram_similarity

        straight1 = LineString([(0, 0), (10, 0), (20, 0), (30, 0)])
        straight2 = LineString([(0, 0), (15, 0), (25, 0), (50, 0)])

        result = compute_angle_histogram_similarity(straight1, straight2)
        assert result == pytest.approx(1.0)

    def test_straight_vs_curved_different(self):
        """Straight line vs curved line should have lower similarity."""
        from matcher.features.geometric import compute_angle_histogram_similarity

        straight = LineString([(0, 0), (10, 0), (20, 0), (30, 0)])
        curved = LineString([(0, 0), (10, 5), (20, 0), (30, 5)])

        result = compute_angle_histogram_similarity(straight, curved)
        # Curved has turns, straight doesn't - should be different
        assert result < 1.0

    def test_similar_curves(self):
        """Two curves with similar shape should have high similarity."""
        from matcher.features.geometric import compute_angle_histogram_similarity

        curve1 = LineString([(0, 0), (10, 5), (20, 0), (30, 5), (40, 0)])
        # Same pattern, just translated
        curve2 = LineString([(5, 0), (15, 5), (25, 0), (35, 5), (45, 0)])

        result = compute_angle_histogram_similarity(curve1, curve2)
        assert result >= 0.9  # Should be very similar

    def test_both_short_lines_return_one(self):
        """Both lines with < 3 points (both straight) should return 1.0."""
        from matcher.features.geometric import compute_angle_histogram_similarity

        short1 = LineString([(0, 0), (10, 0)])
        short2 = LineString([(0, 0), (20, 10)])

        result = compute_angle_histogram_similarity(short1, short2)
        assert result == pytest.approx(1.0)

    def test_one_short_line_compared_to_straight_hist(self):
        """One short line vs multi-point line uses straight histogram for short."""
        from matcher.features.geometric import compute_angle_histogram_similarity

        short = LineString([(0, 0), (10, 0)])
        # Straight multi-point → same shape → high similarity
        straight = LineString([(0, 0), (10, 0), (20, 0), (30, 0)])

        result = compute_angle_histogram_similarity(short, straight)
        assert result > 0.9  # Both effectively straight

    def test_empty_line_returns_one(self):
        """Empty lines should return 1.0."""
        from matcher.features.geometric import compute_angle_histogram_similarity

        empty = LineString()
        normal = LineString([(0, 0), (10, 0), (20, 0)])

        result = compute_angle_histogram_similarity(empty, normal)
        assert result == pytest.approx(1.0)

    def test_with_pre_extracted_coords(self):
        """Should work with pre-extracted coordinates."""
        import numpy as np

        from matcher.features.geometric import compute_angle_histogram_similarity

        line_a = LineString([(0, 0), (10, 5), (20, 0), (30, 5)])
        line_b = LineString([(0, 0), (10, 5), (20, 0), (30, 5)])
        coords_a = np.array(line_a.coords)
        coords_b = np.array(line_b.coords)

        result = compute_angle_histogram_similarity(
            line_a, line_b, coords_a=coords_a, coords_b=coords_b
        )
        assert result == pytest.approx(1.0)


class TestEdgeDistanceRmse:
    """Tests for compute_edge_distance_rmse function."""

    def test_identical_lines(self):
        """Identical lines should have RMSE of 0."""
        from matcher.features.geometric import compute_edge_distance_rmse

        line = LineString([(0, 0), (100, 0)])
        result = compute_edge_distance_rmse(line, line)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_parallel_lines_offset(self):
        """Parallel lines with constant offset should have RMSE equal to offset."""
        from matcher.features.geometric import compute_edge_distance_rmse

        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(0, 5), (100, 5)])  # 5m offset

        result = compute_edge_distance_rmse(line_a, line_b)
        assert result == pytest.approx(5.0, abs=0.1)

    def test_diverging_lines(self):
        """Diverging lines should have higher RMSE than parallel lines."""
        from matcher.features.geometric import compute_edge_distance_rmse

        line_a = LineString([(0, 0), (100, 0)])
        # Starts at same point, ends 20m away
        line_b = LineString([(0, 0), (100, 20)])

        rmse_diverging = compute_edge_distance_rmse(line_a, line_b)

        # Compare with parallel 5m offset
        line_c = LineString([(0, 5), (100, 5)])
        rmse_parallel = compute_edge_distance_rmse(line_a, line_c)

        # Diverging should be worse than constant 5m offset
        assert rmse_diverging > rmse_parallel

    def test_empty_line_returns_max_distance(self):
        """Empty lines should return MAX_DISTANCE_METERS."""
        from matcher.config import MAX_DISTANCE_METERS
        from matcher.features.geometric import compute_edge_distance_rmse

        empty = LineString()
        normal = LineString([(0, 0), (100, 0)])

        result = compute_edge_distance_rmse(empty, normal)
        assert result == MAX_DISTANCE_METERS

    def test_different_lengths(self):
        """Should handle lines of different lengths."""
        from matcher.features.geometric import compute_edge_distance_rmse

        line_a = LineString([(0, 0), (100, 0)])  # 100m
        line_b = LineString([(0, 3), (50, 3)])  # 50m, 3m offset

        result = compute_edge_distance_rmse(line_a, line_b)
        # RMSE should reflect the offset and the non-overlapping portions
        assert result > 3.0  # Greater than just the offset

    def test_consistent_with_different_vertex_densities(self):
        """RMSE should be similar regardless of vertex density.

        This is a key advantage over mean_hausdorff_distance which samples at vertices.
        """
        from matcher.features.geometric import compute_edge_distance_rmse

        # Low density line (2 vertices)
        line_a_low = LineString([(0, 0), (100, 0)])
        line_b_low = LineString([(0, 5), (100, 5)])

        # High density line (11 vertices along same path)
        line_a_high = LineString([(i * 10, 0) for i in range(11)])
        line_b_high = LineString([(i * 10, 5) for i in range(11)])

        rmse_low = compute_edge_distance_rmse(line_a_low, line_b_low)
        rmse_high = compute_edge_distance_rmse(line_a_high, line_b_high)

        # Both should be ~5m regardless of vertex density
        assert rmse_low == pytest.approx(5.0, abs=0.1)
        assert rmse_high == pytest.approx(5.0, abs=0.1)
        assert rmse_low == pytest.approx(rmse_high, abs=0.1)

    def test_custom_sample_interval(self):
        """Should work with custom sample interval."""
        from matcher.features.geometric import compute_edge_distance_rmse

        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(0, 5), (100, 5)])

        # Default 5m interval
        rmse_default = compute_edge_distance_rmse(line_a, line_b)
        # Finer 2m interval
        rmse_fine = compute_edge_distance_rmse(line_a, line_b, sample_interval=2.0)

        # Both should give ~5m (the actual offset)
        assert rmse_default == pytest.approx(5.0, abs=0.1)
        assert rmse_fine == pytest.approx(5.0, abs=0.1)


class TestRoutePrefixMatch:
    """Tests for route prefix matching feature."""

    def test_same_interstate_routes(self):
        """Same interstate type should return 1.0."""
        from matcher.features.semantic import compute_route_prefix_match

        result = compute_route_prefix_match("I-5", "Interstate 5")
        assert result == pytest.approx(1.0)

    def test_same_us_routes(self):
        """Same US route type should return 1.0."""
        from matcher.features.semantic import compute_route_prefix_match

        result = compute_route_prefix_match("US-101", "U.S. Route 101")
        assert result == pytest.approx(1.0)

    def test_different_route_types(self):
        """Different route types should return 0.0."""
        from matcher.features.semantic import compute_route_prefix_match

        result = compute_route_prefix_match("I-5", "US-5")
        assert result == pytest.approx(0.0)

    def test_interstate_vs_state_route(self):
        """Interstate vs state route should return 0.0."""
        from matcher.features.semantic import compute_route_prefix_match

        result = compute_route_prefix_match("I-90", "SR-90")
        assert result == pytest.approx(0.0)

    def test_non_routes(self):
        """Non-routes should return NaN (missing signal)."""
        from matcher.features.semantic import compute_route_prefix_match

        result = compute_route_prefix_match("Main Street", "Oak Avenue")
        assert math.isnan(result)

    def test_one_route_one_non_route(self):
        """One route, one non-route should return NaN (missing signal)."""
        from matcher.features.semantic import compute_route_prefix_match

        result = compute_route_prefix_match("I-5", "Main Street")
        assert math.isnan(result)

    def test_canonicalize_route_name(self):
        """Test route name canonicalization."""
        from matcher.features.semantic import canonicalize_route_name

        assert canonicalize_route_name("I-5") == ("interstate", 5)
        assert canonicalize_route_name("Interstate 90") == ("interstate", 90)
        assert canonicalize_route_name("US-101") == ("us_route", 101)
        assert canonicalize_route_name("SR-99") == ("state_route", 99)
        assert canonicalize_route_name("County Road 15") == ("county_road", 15)
        assert canonicalize_route_name("Highway 1") == ("highway", 1)
        assert canonicalize_route_name("Main Street") == (None, None)
        assert canonicalize_route_name(None) == (None, None)


class TestClusteringCoefficientFeatures:
    """Tests for clustering coefficient feature extraction."""

    def test_clustering_coef_with_full_features(self):
        """Should extract clustering coefficient from full feature vectors."""
        import numpy as np

        from matcher.features.spatial_context import compute_clustering_coefficient_features

        # Create mock feature vectors with clustering at index 3
        ref_features = {
            0: np.array([2.0, 0.0, 0.0, 0.5, 0.0, 0.0]),  # clustering = 0.5
            1: np.array([3.0, 0.0, 0.0, 0.3, 0.0, 0.0]),  # clustering = 0.3
        }
        target_features = {
            10: np.array([2.0, 0.0, 0.0, 0.4, 0.0, 0.0]),  # clustering = 0.4
            11: np.array([3.0, 0.0, 0.0, 0.2, 0.0, 0.0]),  # clustering = 0.2
        }

        ref_seg_to_connectors = {"seg_ref": [(0.0, 0), (1.0, 1)]}
        target_seg_to_connectors = {"seg_target": [(0.0, 10), (1.0, 11)]}

        result = compute_clustering_coefficient_features(
            "seg_ref",
            "seg_target",
            ref_features,
            target_features,
            ref_seg_to_connectors,
            target_seg_to_connectors,
        )

        # Ref clustering: (0.5 + 0.3) / 2 = 0.4
        # Target clustering: (0.4 + 0.2) / 2 = 0.3
        assert result["clustering_coef_ref"] == pytest.approx(0.4)
        assert result["clustering_coef_target"] == pytest.approx(0.3)
        assert result["clustering_coef_delta"] == pytest.approx(0.1)

    def test_clustering_coef_with_degrees_only(self):
        """Should return defaults when only degree values are available."""
        from matcher.features.spatial_context import compute_clustering_coefficient_features

        # Degrees-only mode (int values)
        ref_features = {0: 2, 1: 3}
        target_features = {10: 2, 11: 3}

        ref_seg_to_connectors = {"seg_ref": [(0.0, 0), (1.0, 1)]}
        target_seg_to_connectors = {"seg_target": [(0.0, 10), (1.0, 11)]}

        result = compute_clustering_coefficient_features(
            "seg_ref",
            "seg_target",
            ref_features,
            target_features,
            ref_seg_to_connectors,
            target_seg_to_connectors,
        )

        # Should return NaN defaults (no clustering data available)
        assert math.isnan(result["clustering_coef_ref"])
        assert math.isnan(result["clustering_coef_target"])
        assert math.isnan(result["clustering_coef_delta"])


class TestCrossingAngleFeatures:
    """Tests for crossing angle feature extraction.

    These features detect ACROSS-role segments (crosswalks, bike crossings)
    by measuring the angle between a candidate segment and nearby segments
    of a different traffic tier.
    """

    def test_crosswalk_perpendicular_to_vehicle_road(self):
        """A short N-S footway crossing an E-W vehicle road should have high angles."""
        from matcher.features.geometric import compute_crossing_angle_features

        # Crosswalk: short north-south segment
        crosswalk = LineString([(50, 0), (50, 15)])
        # Nearby east-west vehicle road segments
        nearby_geoms = [
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 15), (100, 15)]),
        ]
        nearby_classes = ["residential", "residential"]

        result = compute_crossing_angle_features(
            candidate=crosswalk,
            nearby_geometries=nearby_geoms,
            nearby_classes=nearby_classes,
            candidate_class="footway",
        )

        # Perpendicular: angles should be near 90°
        assert result["crossing_angle_min"] > 75.0
        assert result["crossing_angle_mean"] > 75.0
        # Low std since consistently perpendicular
        assert result["crossing_angle_std"] < 10.0
        # All neighbors are transverse
        assert result["transverse_neighbor_fraction"] == 1.0

    def test_sidewalk_parallel_to_vehicle_road(self):
        """A sidewalk running parallel to a road should have low angles."""
        from matcher.features.geometric import compute_crossing_angle_features

        # Sidewalk: runs east-west, parallel to the road
        sidewalk = LineString([(0, 5), (100, 5)])
        # Nearby east-west vehicle road
        nearby_geoms = [LineString([(0, 0), (100, 0)])]
        nearby_classes = ["residential"]

        result = compute_crossing_angle_features(
            candidate=sidewalk,
            nearby_geometries=nearby_geoms,
            nearby_classes=nearby_classes,
            candidate_class="footway",
        )

        # Parallel: angles should be near 0°
        assert result["crossing_angle_min"] < 15.0
        assert result["crossing_angle_mean"] < 15.0
        # No neighbors are transverse
        assert result["transverse_neighbor_fraction"] == 0.0

    def test_same_tier_neighbors_are_ignored(self):
        """Neighbors of the same traffic tier should not influence crossing angle."""
        from matcher.features.geometric import compute_crossing_angle_features

        # Candidate is a vehicle road
        candidate = LineString([(0, 0), (100, 0)])
        # Nearby segments are also vehicle roads (same tier)
        nearby_geoms = [
            LineString([(50, -50), (50, 50)]),  # Perpendicular vehicle road
        ]
        nearby_classes = ["residential"]

        result = compute_crossing_angle_features(
            candidate=candidate,
            nearby_geometries=nearby_geoms,
            nearby_classes=nearby_classes,
            candidate_class="secondary",
        )

        # Should return NaN since all neighbors are same tier
        assert math.isnan(result["crossing_angle_min"])
        assert math.isnan(result["crossing_angle_mean"])
        assert math.isnan(result["crossing_angle_std"])
        assert math.isnan(result["transverse_neighbor_fraction"])

    def test_slip_road_mostly_parallel_then_veers(self):
        """A long ramp parallel to motorway then veering off should have LOW min angle.

        This is the critical edge case: the ramp is mostly parallel to the
        motorway (vehicle tier), so the minimum angle to corridor should be
        near 0° even though the end veers off at ~45°.
        """
        from matcher.features.geometric import compute_crossing_angle_features

        # Slip road: runs parallel for ~200m, then veers off at 45° for ~50m
        ramp = LineString(
            [
                (0, 10),
                (50, 10),
                (100, 10),
                (150, 10),
                (200, 10),
                (230, 30),
                (250, 50),  # Veer off at ~45°
            ]
        )
        # Nearby motorway corridor running east-west
        nearby_geoms = [
            LineString([(0, 0), (300, 0)]),
        ]
        nearby_classes = ["motorway"]

        result = compute_crossing_angle_features(
            candidate=ramp,
            nearby_geometries=nearby_geoms,
            nearby_classes=nearby_classes,
            candidate_class="footway",  # Different tier to trigger computation
        )

        # Min should be very low (parallel portion)
        assert result["crossing_angle_min"] < 15.0
        # Mean should be moderate (mix of parallel and angled)
        assert result["crossing_angle_mean"] < 45.0
        # Std should be elevated (heading varies along segment)
        assert result["crossing_angle_std"] > 5.0

    def test_diagonal_crosswalk(self):
        """A diagonal crosswalk at ~45° to the road should have moderate angles."""
        from matcher.features.geometric import compute_crossing_angle_features

        # Diagonal crosswalk crossing an E-W road
        crosswalk = LineString([(40, 0), (60, 20)])
        nearby_geoms = [LineString([(0, 10), (100, 10)])]
        nearby_classes = ["residential"]

        result = compute_crossing_angle_features(
            candidate=crosswalk,
            nearby_geometries=nearby_geoms,
            nearby_classes=nearby_classes,
            candidate_class="footway",
        )

        # Should be ~45° (diagonal crossing)
        assert 30.0 < result["crossing_angle_min"] < 60.0
        assert 30.0 < result["crossing_angle_mean"] < 60.0

    def test_empty_candidate_returns_neutral(self):
        """Empty or degenerate candidate should return neutral values."""
        from matcher.features.geometric import compute_crossing_angle_features

        empty = LineString()
        result = compute_crossing_angle_features(
            candidate=empty,
            nearby_geometries=[LineString([(0, 0), (100, 0)])],
            nearby_classes=["residential"],
            candidate_class="footway",
        )

        assert math.isnan(result["crossing_angle_min"])
        assert math.isnan(result["crossing_angle_mean"])
        assert math.isnan(result["crossing_angle_std"])
        assert math.isnan(result["transverse_neighbor_fraction"])

    def test_no_different_tier_neighbors_returns_neutral(self):
        """When all neighbors are same tier, should return NaN."""
        from matcher.features.geometric import compute_crossing_angle_features

        candidate = LineString([(0, 0), (50, 50)])
        nearby_geoms = [LineString([(0, 10), (100, 10)])]
        # Same tier as candidate
        nearby_classes = ["path"]

        result = compute_crossing_angle_features(
            candidate=candidate,
            nearby_geometries=nearby_geoms,
            nearby_classes=nearby_classes,
            candidate_class="footway",
        )

        for key in result:
            assert math.isnan(result[key]), f"Expected NaN for {key}, got {result[key]}"

    def test_no_neighbors_returns_neutral(self):
        """When no nearby geometries exist, should return NaN."""
        from matcher.features.geometric import compute_crossing_angle_features

        candidate = LineString([(0, 0), (50, 50)])

        result = compute_crossing_angle_features(
            candidate=candidate,
            nearby_geometries=[],
            nearby_classes=[],
            candidate_class="footway",
        )

        for key in result:
            assert math.isnan(result[key]), f"Expected NaN for {key}, got {result[key]}"

    def test_unknown_class_returns_neutral(self):
        """When candidate or neighbor class is unknown, return neutral."""
        from matcher.features.geometric import compute_crossing_angle_features

        candidate = LineString([(0, 0), (50, 50)])
        nearby_geoms = [LineString([(0, 10), (100, 10)])]

        # Unknown candidate class
        result = compute_crossing_angle_features(
            candidate=candidate,
            nearby_geometries=nearby_geoms,
            nearby_classes=["residential"],
            candidate_class=None,
        )
        assert math.isnan(result["crossing_angle_min"])

        # Unknown neighbor class
        result = compute_crossing_angle_features(
            candidate=candidate,
            nearby_geometries=nearby_geoms,
            nearby_classes=[None],
            candidate_class="footway",
        )
        assert math.isnan(result["crossing_angle_min"])

    def test_multiple_corridors_different_angles(self):
        """With corridors at different angles, min should reflect closest."""
        from matcher.features.geometric import compute_crossing_angle_features

        # North-south candidate
        candidate = LineString([(50, 0), (50, 30)])
        # One corridor E-W (perpendicular) and one corridor NE-SW (45°)
        nearby_geoms = [
            LineString([(0, 15), (100, 15)]),  # E-W (90° to candidate)
            LineString([(0, 0), (100, 100)]),  # NE-SW (45° to candidate)
        ]
        nearby_classes = ["residential", "primary"]

        result = compute_crossing_angle_features(
            candidate=candidate,
            nearby_geometries=nearby_geoms,
            nearby_classes=nearby_classes,
            candidate_class="footway",
        )

        # Min should reflect the 45° corridor (closer angle)
        assert result["crossing_angle_min"] < 55.0
        # Mean should be between 45° and 90°
        assert 35.0 < result["crossing_angle_mean"] < 80.0

    def test_cycleway_crossing_vehicle_road(self):
        """Cycleway crossing a vehicle road should be detected."""
        from matcher.features.geometric import compute_crossing_angle_features

        # Bike crossing: north-south
        bike_crossing = LineString([(50, 0), (50, 12)])
        nearby_geoms = [LineString([(0, 6), (100, 6)])]
        nearby_classes = ["tertiary"]

        result = compute_crossing_angle_features(
            candidate=bike_crossing,
            nearby_geometries=nearby_geoms,
            nearby_classes=nearby_classes,
            candidate_class="cycleway",
        )

        # Perpendicular crossing
        assert result["crossing_angle_min"] > 75.0
        assert result["transverse_neighbor_fraction"] == 1.0

    def test_very_short_segment(self):
        """Very short segments (< sample_interval) should still work."""
        from matcher.features.geometric import compute_crossing_angle_features

        # 3m crosswalk (shorter than default 10m sample interval)
        short_crosswalk = LineString([(50, 0), (50, 3)])
        nearby_geoms = [LineString([(0, 1.5), (100, 1.5)])]
        nearby_classes = ["residential"]

        result = compute_crossing_angle_features(
            candidate=short_crosswalk,
            nearby_geometries=nearby_geoms,
            nearby_classes=nearby_classes,
            candidate_class="footway",
        )

        # Should still detect the perpendicular relationship
        assert result["crossing_angle_min"] > 70.0
        assert result["crossing_angle_mean"] > 70.0

    def test_degenerate_zero_length_neighbor_ignored(self):
        """Zero-length neighbor geometries should be filtered out."""
        from matcher.features.geometric import compute_crossing_angle_features

        candidate = LineString([(50, 0), (50, 20)])
        nearby_geoms = [
            LineString([(10, 10), (10, 10)]),  # Zero length (degenerate)
            LineString([(0, 10), (100, 10)]),  # Valid E-W road
        ]
        nearby_classes = ["residential", "residential"]

        result = compute_crossing_angle_features(
            candidate=candidate,
            nearby_geometries=nearby_geoms,
            nearby_classes=nearby_classes,
            candidate_class="footway",
        )

        # Should use only the valid neighbor
        assert result["crossing_angle_min"] > 75.0


class TestAlignedLengthM:
    """Tests for aligned_length_m feature.

    aligned_length_m = ref_geom.length * (end_frac - start_frac).
    Distinguishes short intersection-only overlaps from substantial partial matches.
    """

    @pytest.mark.parametrize(
        "ref_length,start_frac,end_frac,expected",
        [
            (100, 0.0, 1.0, 100.0),  # Full alignment
            (200, 0.25, 0.75, 100.0),  # 50% partial
            (12, 0.1, 0.9, 9.6),  # Short intersection overlap
            (1000, 0.0, 0.05, 50.0),  # Tiny overlap on long segment
        ],
        ids=["full", "partial_50pct", "short_intersection", "tiny_on_long"],
    )
    def test_aligned_length_computation(self, ref_length, start_frac, end_frac, expected):
        """aligned_length_m = ref_length * (end_frac - start_frac)."""
        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        ref = LineString([(0, 0), (ref_length, 0)])
        target = LineString([(0, 2), (ref_length, 2)])

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            alignment=AlignmentResult(
                overture_start_frac=start_frac,
                overture_end_frac=end_frac,
                dataset_start_frac=0.0,
                dataset_end_frac=1.0,
            ),
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        assert features["aligned_length_m"] == pytest.approx(expected, abs=0.5)

    def test_no_alignment_returns_zero(self):
        """Without alignment, aligned_length_m = 0.0 (consistent with coverage features)."""
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        ref = LineString([(0, 0), (150, 0)])
        target = LineString([(0, 5), (150, 5)])

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            alignment=None,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        assert features["aligned_length_m"] == 0.0

    def test_uses_ref_geom_length_not_subline(self):
        """Uses original ref geometry length, not subline (exact, no extraction error)."""
        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        # Curved ref — subline extraction may lose/gain tiny amounts
        ref = LineString([(0, 0), (50, 10), (100, 0)])
        target = LineString([(25, 12), (75, 12)])

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            alignment=AlignmentResult(
                overture_start_frac=0.25,
                overture_end_frac=0.75,
                dataset_start_frac=0.0,
                dataset_end_frac=1.0,
            ),
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        assert features["aligned_length_m"] == pytest.approx(ref.length * 0.5, abs=0.01)

    def test_error_case_returns_nan(self):
        """Error features should have aligned_length_m = NaN."""
        from matcher.features.compute import _get_error_features

        error_feats = _get_error_features(error=ValueError("test"))
        assert math.isnan(error_feats["aligned_length_m"])
