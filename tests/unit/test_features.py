"""Tests for feature extraction."""

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
        from tests.conftest import MOCK_ENDPOINT_FEATURES

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
        from tests.conftest import MOCK_ENDPOINT_FEATURES

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
        )

        # Should still include coverage features (zeros without alignment)
        assert "ref_coverage" in features
        assert features["ref_coverage"] == 0.0

    def test_compute_pair_features_uses_sublines_with_alignment(self):
        """With alignment, similarity features should be computed on sublines."""
        from matcher.features.alignment import linestring_alignment
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES

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
        from tests.conftest import MOCK_ENDPOINT_FEATURES

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
        from tests.conftest import MOCK_ENDPOINT_FEATURES

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
        )

        # Lateral offset should be ~3m (the offset in the overlapping region)
        # NOT ~100m (which would happen if measuring full 300m target to 100m ref)
        assert features["lateral_offset_m"] < 10.0, (
            f"Lateral offset {features['lateral_offset_m']:.1f}m is too high. "
            "Should be ~3m for aligned sublines, not inflated by non-overlapping portion."
        )


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

    def test_short_lines_return_one(self):
        """Lines with < 3 points should return 1.0 (no turns to compare)."""
        from matcher.features.geometric import compute_angle_histogram_similarity

        short1 = LineString([(0, 0), (10, 0)])
        short2 = LineString([(0, 0), (20, 10)])

        result = compute_angle_histogram_similarity(short1, short2)
        assert result == pytest.approx(1.0)

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
        """Non-routes should return neutral 0.5."""
        from matcher.features.semantic import compute_route_prefix_match

        result = compute_route_prefix_match("Main Street", "Oak Avenue")
        assert result == pytest.approx(0.5)

    def test_one_route_one_non_route(self):
        """One route, one non-route should return neutral 0.5."""
        from matcher.features.semantic import compute_route_prefix_match

        result = compute_route_prefix_match("I-5", "Main Street")
        assert result == pytest.approx(0.5)

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

        # Should return defaults
        assert result["clustering_coef_ref"] == 0.0
        assert result["clustering_coef_target"] == 0.0
        assert result["clustering_coef_delta"] == 0.0
