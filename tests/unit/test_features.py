"""Tests for feature extraction."""

import pytest
from shapely import LineString

from matcher.features.geometric import (
    compute_geometric_features,
    compute_segment_heading,
)
from matcher.features.semantic import (
    compute_class_similarity,
    compute_name_similarity,
    names_likely_same_road,
)


class TestGeometricFeatures:
    """Tests for geometric feature extraction."""

    def test_identical_lines(self):
        """Identical lines should have perfect geometric scores."""
        line = LineString([(0, 0), (100, 0)])

        features = compute_geometric_features(line, line)

        assert features.hausdorff_distance == pytest.approx(0.0)
        assert features.mean_hausdorff_distance == pytest.approx(0.0)
        assert features.buffer_iou == pytest.approx(1.0, abs=0.01)
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

        features = compute_geometric_features(line_a, line_b, buffer_radius=10.0)

        # With 10m buffer, 100m apart lines should not overlap
        assert features.buffer_iou < 0.1


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
        """Missing names should return neutral score (0.5) and flag as missing."""
        result = compute_name_similarity(None, "Main Street")

        # Neutral scores avoid penalizing valid geometric matches
        assert result["levenshtein_ratio"] == 0.5
        assert result["token_sort_ratio"] == 0.5
        assert result["names_missing"] is True

    def test_name_similarity_both_none(self):
        """Both names missing should return neutral score (0.5)."""
        result = compute_name_similarity(None, None)

        assert result["levenshtein_ratio"] == 0.5
        assert result["names_missing"] is True

    def test_class_similarity_same(self):
        """Same road class should return 1.0."""
        result = compute_class_similarity("primary", "primary")

        assert result == pytest.approx(1.0)

    def test_class_similarity_adjacent(self):
        """Adjacent road classes should have high similarity."""
        result = compute_class_similarity("primary", "secondary")

        assert result > 0.7

    def test_class_similarity_distant(self):
        """Distant road classes should have lower similarity."""
        result = compute_class_similarity("motorway", "residential")

        assert result < 0.5

    def test_class_similarity_same_class_same_subclass(self):
        """Same class and subclass should return 1.0."""
        result = compute_class_similarity("footway", "footway", "sidewalk", "sidewalk")

        assert result == pytest.approx(1.0)

    def test_class_similarity_same_class_different_subclass(self):
        """Same class but different subclass should have slightly lower similarity."""
        result = compute_class_similarity("footway", "footway", "sidewalk", "crosswalk")

        assert result == pytest.approx(0.85)

    def test_class_similarity_same_class_one_subclass_missing(self):
        """Same class with one subclass missing should have slight penalty."""
        result = compute_class_similarity("footway", "footway", "sidewalk", None)

        assert result == pytest.approx(0.9)

    def test_names_likely_same_road(self):
        """Test quick name matching heuristic."""
        assert names_likely_same_road("Main Street", "Main St")
        assert names_likely_same_road("Interstate 5", "I-5")
        assert not names_likely_same_road("Main Street", "Oak Avenue")


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
        """Missing one name should return neutral phonetic scores."""
        result = compute_name_similarity(None, "Main Street")
        assert result["soundex_match"] == 0.5
        assert result["metaphone_similarity"] == 0.5

    def test_phonetic_missing_both_names(self):
        """Missing both names should return neutral phonetic scores."""
        result = compute_name_similarity(None, None)
        assert result["soundex_match"] == 0.5
        assert result["metaphone_similarity"] == 0.5

    def test_phonetic_with_abbreviations(self):
        """Phonetic matching should work with abbreviations after normalization."""
        result = compute_name_similarity("N Main St", "North Main Street")
        # After normalization, both become "north main street"
        assert result["soundex_match"] == 1.0
        assert result["metaphone_similarity"] > 0.9


class TestComputeSegmentHeading:
    """Tests for segment heading calculation."""

    def test_east_heading(self):
        """East-pointing line should have ~90 degree heading."""
        line = LineString([(0, 0), (100, 0)])
        heading = compute_segment_heading(line)

        assert heading == pytest.approx(0.0, abs=1.0)

    def test_north_heading(self):
        """North-pointing line should have ~0 degree heading."""
        line = LineString([(0, 0), (0, 100)])
        heading = compute_segment_heading(line)

        assert heading == pytest.approx(90.0, abs=1.0)

    def test_northeast_heading(self):
        """Northeast-pointing line should have ~45 degree heading."""
        line = LineString([(0, 0), (100, 100)])
        heading = compute_segment_heading(line)

        assert heading == pytest.approx(45.0, abs=1.0)
