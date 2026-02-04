"""Tests for exclude_semantic flag in ML training."""

import pytest

from matcher.config import FEATURE_COLUMNS, SEMANTIC_FEATURES


class TestExcludeSemanticFlag:
    """Tests for semantic feature exclusion."""

    def test_semantic_features_defined_in_config(self):
        """SEMANTIC_FEATURES should be defined in config."""
        # 11 semantic features (includes route_prefix_match)
        assert len(SEMANTIC_FEATURES) == 11
        assert "name_levenshtein" in SEMANTIC_FEATURES
        assert "name_jaro_winkler" in SEMANTIC_FEATURES
        assert "name_token_sort" in SEMANTIC_FEATURES
        assert "name_soundex" in SEMANTIC_FEATURES
        assert "name_metaphone" in SEMANTIC_FEATURES
        assert "has_name_ref" in SEMANTIC_FEATURES
        assert "has_name_target" in SEMANTIC_FEATURES
        assert "name_is_generic" in SEMANTIC_FEATURES
        assert "class_similarity" in SEMANTIC_FEATURES
        assert "name_numeric_match" in SEMANTIC_FEATURES
        assert "route_prefix_match" in SEMANTIC_FEATURES

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
        assert "hausdorff_distance_m" in filtered
        assert "buffer_iou_5m" in filtered
        assert "buffer_iou_15m" in filtered
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
                    "hausdorff_distance_m",
                    "mean_hausdorff_distance_m",
                    "hausdorff_p95_m",
                    "buffer_iou_5m",
                    "buffer_iou_15m",
                    "heading_delta",
                    "length_ratio",
                    "centroid_distance_m",
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
                    "has_name_ref",
                    "has_name_target",
                    "name_is_generic",
                ],
            ),
            ("semantic_class", ["class_similarity"]),
            (
                "endpoint",
                [
                    "min_endpoint_proximity_m",
                    "max_endpoint_proximity_m",
                    "shared_endpoint_count",
                ],
            ),
            (
                "lateral",
                ["lateral_offset_m", "lateral_offset_iqr_m", "lateral_offset_p95_m"],
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

        # Count by category (11 geometric features)
        geometric_count = sum(
            1
            for f in geom_only_features
            if f
            in [
                "hausdorff_distance_m",
                "mean_hausdorff_distance_m",
                "hausdorff_p95_m",
                "buffer_iou_5m",
                "buffer_iou_15m",
                "heading_delta",
                "length_ratio",
                "centroid_distance_m",
                "collinear_gap_ratio",
                "angle_histogram_similarity",
                "edge_distance_rmse_m",
            ]
        )
        assert geometric_count == 11, "Should have 11 geometric features"

        endpoint_count = sum(
            1
            for f in geom_only_features
            if f
            in ["min_endpoint_proximity_m", "max_endpoint_proximity_m", "shared_endpoint_count"]
        )
        assert endpoint_count == 3, "Should have 3 endpoint features"

        lateral_count = sum(
            1
            for f in geom_only_features
            if f in ["lateral_offset_m", "lateral_offset_iqr_m", "lateral_offset_p95_m"]
        )
        assert lateral_count == 3, "Should have 3 lateral features"

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

        # Graphlet features are now included in real-time scoring pipeline
        graphlet_features = ["graphlet_similarity", "endpoint_degree_similarity"]
        graphlet_count = sum(1 for f in geom_only_features if f in graphlet_features)
        assert graphlet_count == 2, "Should have 2 graphlet features"

        # Clustering coefficient features
        clustering_features = [
            "clustering_coef_ref",
            "clustering_coef_target",
            "clustering_coef_delta",
        ]
        clustering_count = sum(1 for f in geom_only_features if f in clustering_features)
        assert clustering_count == 3, "Should have 3 clustering coefficient features"

        # New geometric features: sinuosity (3), heading_consistency (3),
        # vertex_density (3), length_bin (3), min_length_m (1), shape_complexity (3)
        sinuosity_count = sum(1 for f in geom_only_features if f.startswith("sinuosity"))
        assert sinuosity_count == 3, "Should have 3 sinuosity features"

        heading_consistency_count = sum(
            1 for f in geom_only_features if f.startswith("heading_consistency")
        )
        assert heading_consistency_count == 3, "Should have 3 heading consistency features"

        vertex_density_count = sum(1 for f in geom_only_features if f.startswith("vertex_density"))
        assert vertex_density_count == 3, "Should have 3 vertex density features"

        # Length features: min_length_m only (length_bin_* removed as redundant)
        length_feature_count = sum(1 for f in geom_only_features if f == "min_length_m")
        assert length_feature_count == 1, "Should have 1 length feature"

        shape_complexity_count = sum(
            1 for f in geom_only_features if f.startswith("shape_complexity")
        )
        assert shape_complexity_count == 3, "Should have 3 shape complexity features"

        # Total geometry-only features:
        # 11 (geometric) + 3 (endpoint) + 3 (lateral) + 12 (topology) + 4 (coverage) +
        # 2 (graphlet) + 3 (clustering) + 3 (sinuosity) + 3 (heading_consistency) +
        # 3 (vertex_density) + 1 (length) + 3 (shape_complexity) + 5 (parallel_sibling) = 56
        assert len(geom_only_features) == 56, (
            f"Expected 56 geometry-only features, got {len(geom_only_features)}"
        )
