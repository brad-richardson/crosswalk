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
    get_traffic_tier,
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

    @pytest.mark.parametrize(
        "coords_a,coords_b",
        [
            # Identical lines (perfect overlap)
            ([(0, 0), (100, 0)], [(0, 0), (100, 0)]),
            # Partial overlap (50%) - above 10% threshold
            ([(0, 0), (100, 0)], [(50, 0), (150, 0)]),
            # Perpendicular (not collinear, heading > 15°)
            ([(0, 0), (100, 0)], [(50, -50), (50, 50)]),
            # Parallel offset (good along-track overlap despite lateral offset)
            ([(0, 0), (100, 0)], [(0, 10), (100, 10)]),
            # Empty line (degenerate case)
            ([(0, 0), (100, 0)], []),
        ],
        ids=[
            "identical_lines",
            "partial_overlap_50pct",
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

    def test_small_overlap_scaled_penalty(self):
        """Small overlap should produce scaled penalty proportional to overlap."""
        # 5% overlap at threshold of 10% -> ratio = 0.5
        line_a = LineString([(0, 0), (100, 0)])
        line_b = LineString([(95, 0), (195, 0)])  # 5m overlap out of 100m
        result = compute_collinear_gap_ratio(line_a, line_b)
        assert 0.4 <= result <= 0.6

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
            # Neutral tier: bridleway -> 0.5
            ("bridleway", "residential", 0.5),
            ("bridleway", "footway", 0.5),
            ("bridleway", "cycleway", 0.5),
            # Unknown -> neutral 0.5
            ("residential", "unknown", 0.5),
            (None, "footway", 0.5),
            ("", "residential", 0.5),
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
        assert result == pytest.approx(expected, abs=0.05)


class TestComputePairFeaturesWithAlignment:
    """Tests for compute_pair_features with alignment parameter."""

    def test_compute_pair_features_includes_coverage_features(self):
        """compute_pair_features should include coverage features when alignment provided."""
        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import compute_pair_features

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

        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (100, 0)])

        features = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name="Main Street",
            target_name="Main Street",
            ref_class="residential",
            target_class="residential",
        )

        # Should still include coverage features (zeros without alignment)
        assert "ref_coverage" in features
        assert features["ref_coverage"] == 0.0

    def test_compute_pair_features_uses_sublines_with_alignment(self):
        """With alignment, similarity features should be computed on sublines."""
        from matcher.features.alignment import linestring_alignment
        from matcher.features.compute import compute_pair_features

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
        )

        features_unaligned = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            alignment=None,
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
        )

        # All feature columns should be present
        for col in ALL_FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"
