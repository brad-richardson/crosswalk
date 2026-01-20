"""Tests for exclude_semantic flag in ML training."""

import pytest

from matcher.config import FEATURE_COLUMNS, SEMANTIC_FEATURES


class TestExcludeSemanticFlag:
    """Tests for semantic feature exclusion."""

    def test_semantic_features_defined_in_config(self):
        """SEMANTIC_FEATURES should be defined in config."""
        assert len(SEMANTIC_FEATURES) == 6
        assert "name_levenshtein" in SEMANTIC_FEATURES
        assert "name_jaro_winkler" in SEMANTIC_FEATURES
        assert "name_token_sort" in SEMANTIC_FEATURES
        assert "name_soundex" in SEMANTIC_FEATURES
        assert "name_metaphone" in SEMANTIC_FEATURES
        assert "class_similarity" in SEMANTIC_FEATURES

    def test_all_semantic_features_are_in_feature_columns(self):
        """All SEMANTIC_FEATURES should be subset of FEATURE_COLUMNS."""
        for feature in SEMANTIC_FEATURES:
            assert feature in FEATURE_COLUMNS, f"{feature} not in FEATURE_COLUMNS"

    def test_exclude_semantic_filters_correctly(self):
        """Excluding semantic features should leave only non-semantic features."""
        filtered = [f for f in FEATURE_COLUMNS if f not in SEMANTIC_FEATURES]

        # Should have fewer features
        assert len(filtered) == len(FEATURE_COLUMNS) - len(SEMANTIC_FEATURES)

        # None of the semantic features should be in filtered list
        for semantic_feature in SEMANTIC_FEATURES:
            assert semantic_feature not in filtered

        # Geometric features should still be present
        assert "hausdorff_distance" in filtered
        assert "buffer_iou" in filtered
        assert "length_ratio" in filtered

        # Topology features should still be present
        assert "from_degree_ref" in filtered
        assert "degree_match_score" in filtered

        # Note: Graphlet features are not in FEATURE_COLUMNS
        # (computed in backfill only, not in real-time scoring)

    @pytest.mark.parametrize(
        "feature_category,expected_features",
        [
            (
                "geometric",
                [
                    "hausdorff_distance",
                    "mean_hausdorff_distance",
                    "buffer_iou",
                    "overlap_ratio",
                    "heading_delta",
                    "length_ratio",
                    "projection_distance",
                    "centroid_distance",
                    "collinear_gap_ratio",
                ],
            ),
            (
                "semantic_name",
                [
                    "name_levenshtein",
                    "name_jaro_winkler",
                    "name_token_sort",
                    "name_soundex",
                    "name_metaphone",
                ],
            ),
            (
                "semantic_class",
                ["class_similarity"],
            ),
            (
                "endpoint",
                [
                    "start_endpoint_proximity",
                    "end_endpoint_proximity",
                    "shared_endpoint_count",
                ],
            ),
            (
                "lateral",
                ["lateral_offset", "lateral_offset_consistency"],
            ),
            (
                "topology",
                [
                    "from_degree_ref",
                    "to_degree_ref",
                    "from_degree_target",
                    "to_degree_target",
                    "degree_match_score",
                    "degree_signature_similarity",
                    "is_dead_end_ref",
                    "is_dead_end_target",
                    "dead_end_match",
                    "is_intersection_ref",
                    "is_intersection_target",
                    "intersection_match",
                ],
            ),
            (
                "coverage",
                ["ref_coverage", "target_coverage", "min_coverage", "coverage_ratio"],
            ),
            # Note: Graphlet features are computed in backfill only, not in real-time scoring
        ],
    )
    def test_feature_category_presence(self, feature_category, expected_features):
        """Test that expected features are present in FEATURE_COLUMNS."""
        for feature in expected_features:
            assert feature in FEATURE_COLUMNS, f"{feature} not in FEATURE_COLUMNS"

    def test_geometry_only_model_features(self):
        """Geometry-only model should include all non-semantic features."""
        geom_only_features = [f for f in FEATURE_COLUMNS if f not in SEMANTIC_FEATURES]

        # Count by category
        geometric_count = sum(
            1
            for f in geom_only_features
            if f
            in [
                "hausdorff_distance",
                "mean_hausdorff_distance",
                "buffer_iou",
                "overlap_ratio",
                "heading_delta",
                "length_ratio",
                "projection_distance",
                "centroid_distance",
                "collinear_gap_ratio",
            ]
        )
        assert geometric_count == 9, "Should have 9 geometric features"

        endpoint_count = sum(
            1
            for f in geom_only_features
            if f in ["start_endpoint_proximity", "end_endpoint_proximity", "shared_endpoint_count"]
        )
        assert endpoint_count == 3, "Should have 3 endpoint features"

        lateral_count = sum(
            1 for f in geom_only_features if f in ["lateral_offset", "lateral_offset_consistency"]
        )
        assert lateral_count == 2, "Should have 2 lateral features"

        topology_features = [
            "from_degree_ref",
            "to_degree_ref",
            "from_degree_target",
            "to_degree_target",
            "degree_match_score",
            "degree_signature_similarity",
            "is_dead_end_ref",
            "is_dead_end_target",
            "dead_end_match",
            "is_intersection_ref",
            "is_intersection_target",
            "intersection_match",
        ]
        topology_count = sum(1 for f in geom_only_features if f in topology_features)
        assert topology_count == 12, "Should have 12 topology features"

        coverage_count = sum(1 for f in geom_only_features if "coverage" in f)
        assert coverage_count == 4, "Should have 4 coverage features"

        # Note: Graphlet features (graphlet_similarity, endpoint_degree_similarity)
        # are computed in backfill only, not in real-time scoring pipeline

        # Total geometry-only features: 9 + 3 + 2 + 12 + 4 = 30
        assert len(geom_only_features) == 30, (
            f"Expected 30 geometry-only features, got {len(geom_only_features)}"
        )
