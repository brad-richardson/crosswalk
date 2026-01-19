"""Tests for feature extraction."""

import pytest
from shapely import LineString

from matcher.features.geometric import (
    compute_collinear_gap_ratio,
    compute_geometric_features,
    compute_segment_heading,
)
from matcher.features.semantic import (
    compute_class_similarity,
    compute_name_similarity,
    get_class_info,
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

    def test_identical_lines_no_penalty(self):
        """Identical lines should have no penalty (perfect overlap)."""
        line = LineString([(0, 0), (100, 0)])
        result = compute_collinear_gap_ratio(line, line)
        assert result == pytest.approx(1.0)

    def test_tip_to_tip_collinear_penalty(self):
        """Tip-to-tip collinear segments should receive strong penalty."""
        # Two consecutive segments on the same line
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(100, 0), (200, 0)])
        result = compute_collinear_gap_ratio(line_a, line_b)
        # Should be 0.0 since they just touch (0% overlap)
        assert result < 0.1

    def test_gap_between_collinear_penalty(self):
        """Collinear segments with a gap should receive strong penalty."""
        # Two segments on the same line with a gap
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(110, 0), (200, 0)])  # 10m gap
        result = compute_collinear_gap_ratio(line_a, line_b)
        # Should be 0.0 since there's no overlap
        assert result == pytest.approx(0.0)

    def test_overlapping_collinear_no_penalty(self):
        """Collinear segments with good overlap should have no penalty."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(25, 0), (75, 0)])  # 50m fully contained within 100m
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result > 0.9

    def test_partial_overlap_collinear(self):
        """Partial overlap above threshold should have no penalty."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(50, 0), (150, 0)])  # 50% overlap
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result == pytest.approx(1.0)

    def test_perpendicular_no_penalty(self):
        """Perpendicular segments should have no penalty (not collinear)."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(50, -50), (50, 50)])
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result == pytest.approx(1.0)

    def test_parallel_offset_no_penalty(self):
        """Parallel but offset segments should have no penalty (handled by other features)."""
        # Parallel roads 10m apart - different roads, not tip-to-tip
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(0, 10), (100, 10)])
        result = compute_collinear_gap_ratio(line_a, line_b)
        # heading delta is 0, so it's collinear check-wise, but they have 100% along-track overlap
        assert result == pytest.approx(1.0)

    def test_opposite_direction_collinear(self):
        """Opposite direction collinear segments should still be detected."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(200, 0), (100, 0)])  # Opposite direction, tip-to-tip
        result = compute_collinear_gap_ratio(line_a, line_b)
        # Should detect the tip-to-tip scenario
        assert result < 0.1

    def test_small_overlap_scaled_penalty(self):
        """Small overlap should produce scaled penalty (not full penalty)."""
        # 5% overlap at threshold of 10%
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(95, 0), (195, 0)])  # 5m overlap out of 100m smaller segment
        result = compute_collinear_gap_ratio(line_a, line_b)
        # 5% overlap / 10% threshold = 0.5
        assert 0.4 <= result <= 0.6

    def test_diagonal_collinear_tip_to_tip(self):
        """Tip-to-tip detection should work for diagonal lines."""
        line_a = LineString([(0, 0), (100, 100)])
        line_b = LineString([(100, 100), (200, 200)])
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result < 0.1

    def test_included_in_geometric_features(self):
        """collinear_gap_ratio should be included in compute_geometric_features output."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(100, 0), (200, 0)])
        features = compute_geometric_features(line_a, line_b)
        assert hasattr(features, "collinear_gap_ratio")
        assert features.collinear_gap_ratio < 0.1  # tip-to-tip should be penalized

    def test_empty_line_no_penalty(self):
        """Empty lines should have no penalty (degenerate case)."""
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString()  # Empty
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert result == pytest.approx(1.0)
