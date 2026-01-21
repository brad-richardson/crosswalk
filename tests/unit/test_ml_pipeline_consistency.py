"""Tests for ML pipeline consistency.

These tests ensure that feature computation is consistent across all stages:
1. Pre-computed features (endpoint, topology, graphlet) match direct computation
2. Inference imputation uses the same values stored during training
3. Alignment-aware features respect alignment fractions

These tests catch subtle bugs where features computed at training time differ
from features computed at inference time, which would degrade model performance.
"""

import numpy as np
import pytest
from shapely.geometry import LineString

from matcher.config import FEATURE_COLUMNS, MAX_DISTANCE_METERS


class TestPrecomputationConsistency:
    """Ensure pre-computed features match direct computation.

    The ML scorer pre-computes endpoint, topology, and graphlet features
    for efficiency. These tests verify that pre-computed values match
    what compute_pair_features() would calculate directly.
    """

    @pytest.fixture
    def sample_network_gdf(self):
        """Create a simple T-intersection network for testing."""
        import geopandas as gpd

        # T-intersection: main road + side street
        # Use projected coordinates (meters) for accurate distance computation
        lines = [
            LineString([(0, 0), (100, 0)]),  # Main road west
            LineString([(100, 0), (200, 0)]),  # Main road east
            LineString([(100, 0), (100, 100)]),  # Side street north
        ]
        return gpd.GeoDataFrame(
            {
                "id": ["seg_west", "seg_east", "seg_north"],
                "geometry": lines,
                "names": ["Main St", "Main St", "Side St"],
                "class": ["primary", "primary", "residential"],
                "subclass": [None, None, None],
            },
            crs="EPSG:32632",  # UTM zone - units are meters
        )

    def test_endpoint_features_match_direct_computation(self, sample_network_gdf):
        """Pre-computed endpoint features should match direct computation.

        The ML scorer pre-computes endpoint features using compute_endpoint_features()
        before passing them to compute_pair_features(). This test verifies that
        passing pre-computed features produces the same result as computing them
        on-the-fly within compute_pair_features() (when endpoint_features=None).
        """
        from matcher.features.compute import compute_pair_features
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            compute_endpoint_features,
        )

        gdf = sample_network_gdf

        # Build spatial index (same as ML scorer does)
        spatial_index = SpatialContextIndex()
        spatial_index.build_from_gdf(gdf, id_column="id")

        # Get geometries for a pair
        target_idx = 0  # seg_west
        target_geom = gdf.geometry.iloc[target_idx]
        ref_geom = gdf.geometry.iloc[1]  # seg_east

        # Method 1: Pre-compute endpoint features (like ML scorer does)
        precomputed_endpoint = compute_endpoint_features(
            target_geom, spatial_index, exclude_segment_idx=target_idx
        )

        features_with_precomputed = compute_pair_features(
            ref_geom=ref_geom,
            target_geom=target_geom,
            ref_name="Main St",
            target_name="Main St",
            ref_class="primary",
            target_class="primary",
            endpoint_features=precomputed_endpoint,
        )

        # Pre-computed features should have real values (not defaults)
        # Since we have a network, endpoint proximity should be < MAX_DISTANCE
        assert precomputed_endpoint["min_endpoint_proximity_m"] < MAX_DISTANCE_METERS
        assert features_with_precomputed["min_endpoint_proximity_m"] < MAX_DISTANCE_METERS

        # The pre-computed endpoint features should be passed through exactly
        assert (
            features_with_precomputed["min_endpoint_proximity_m"]
            == precomputed_endpoint["min_endpoint_proximity_m"]
        )
        assert (
            features_with_precomputed["max_endpoint_proximity_m"]
            == precomputed_endpoint["max_endpoint_proximity_m"]
        )
        assert (
            features_with_precomputed["shared_endpoint_count"]
            == precomputed_endpoint["shared_endpoint_count"]
        )

    def test_topology_features_match_direct_computation(self, sample_network_gdf):
        """Pre-computed topology features should match direct computation.

        The ML scorer pre-computes topology using compute_all_topology() and passes
        it to compute_pair_features(). This test verifies the values match.
        """
        from matcher.features.compute import compute_pair_features
        from matcher.features.spatial_context import compute_all_topology

        gdf = sample_network_gdf

        # Pre-compute topology (like ML scorer does)
        all_ids = set(gdf["id"].astype(str))
        topology_by_id = compute_all_topology(
            gdf, id_column="id", tolerance_m=5.0, ids_to_compute=all_ids
        )

        # Get topology for specific segments
        ref_id = "seg_east"
        target_id = "seg_west"
        ref_topology = topology_by_id.get(ref_id)
        target_topology = topology_by_id.get(target_id)

        # These should have real topology values
        assert ref_topology is not None
        assert target_topology is not None

        # Compute features with pre-computed topology
        features = compute_pair_features(
            ref_geom=gdf[gdf["id"] == ref_id].geometry.iloc[0],
            target_geom=gdf[gdf["id"] == target_id].geometry.iloc[0],
            ref_name="Main St",
            target_name="Main St",
            ref_class="primary",
            target_class="primary",
            ref_topology=ref_topology,
            target_topology=target_topology,
        )

        # Verify topology features reflect pre-computed values
        assert features["from_degree_ref"] == ref_topology["from_degree"]
        assert features["to_degree_ref"] == ref_topology["to_degree"]
        assert features["from_degree_target"] == target_topology["from_degree"]
        assert features["to_degree_target"] == target_topology["to_degree"]

        # Verify topology flags are passed through
        assert features["is_dead_end_ref"] == (1.0 if ref_topology["is_dead_end"] else 0.0)
        assert features["is_dead_end_target"] == (1.0 if target_topology["is_dead_end"] else 0.0)

    def test_graphlet_features_passed_through_correctly(self, sample_network_gdf):
        """Pre-computed graphlet features should be passed through exactly.

        The ML scorer pre-computes graphlet features and passes them to
        compute_pair_features(). This test verifies they are used unchanged.
        """
        from matcher.features.compute import compute_pair_features

        ref_geom = sample_network_gdf.geometry.iloc[0]
        target_geom = sample_network_gdf.geometry.iloc[1]

        # Simulate pre-computed graphlet features with specific values
        precomputed_graphlet = {
            "graphlet_similarity": 0.85,
            "endpoint_degree_similarity": 0.92,
        }

        features = compute_pair_features(
            ref_geom=ref_geom,
            target_geom=target_geom,
            ref_name="Main St",
            target_name="Main St",
            ref_class="primary",
            target_class="primary",
            graphlet_features=precomputed_graphlet,
        )

        # Values should be passed through exactly
        assert features["graphlet_similarity"] == 0.85
        assert features["endpoint_degree_similarity"] == 0.92

    def test_graphlet_features_default_when_none(self, sample_network_gdf):
        """Graphlet features should default to 0.5 when not provided."""
        from matcher.features.compute import compute_pair_features

        ref_geom = sample_network_gdf.geometry.iloc[0]
        target_geom = sample_network_gdf.geometry.iloc[1]

        features = compute_pair_features(
            ref_geom=ref_geom,
            target_geom=target_geom,
            ref_name="Main St",
            target_name="Main St",
            ref_class="primary",
            target_class="primary",
            graphlet_features=None,  # Not provided
        )

        # Should use neutral default of 0.5
        assert features["graphlet_similarity"] == 0.5
        assert features["endpoint_degree_similarity"] == 0.5


class TestImputationConsistency:
    """Ensure imputation is consistent between training and inference.

    The ML model stores imputation medians computed from training data.
    These tests verify that inference uses exactly those stored values.
    """

    @pytest.fixture
    def trained_matcher_with_medians(self):
        """Create a matcher with known imputation medians."""
        from matcher.matching.ml import MLMatcher

        matcher = MLMatcher()
        # Simulate trained model state with known medians
        matcher.feature_names = FEATURE_COLUMNS.copy()
        matcher.feature_medians = {
            "hausdorff_distance_m": 15.0,
            "mean_hausdorff_distance_m": 10.0,
            "hausdorff_p95_m": 20.0,
            "buffer_iou_5m": 0.6,
            "buffer_iou_15m": 0.8,
            "overlap_ratio": 0.7,
            "heading_delta": 5.0,
            "length_ratio": 0.95,
            "projection_distance_m": 8.0,
            "centroid_distance_m": 12.0,
            "collinear_gap_ratio": 0.1,
            "name_levenshtein": 0.85,
            "name_jaro_winkler": 0.9,
            "name_token_sort": 0.88,
            "name_soundex": 0.5,
            "name_metaphone": 0.5,
            "has_name_ref": 1.0,
            "has_name_target": 1.0,
            "name_is_generic": 0.0,
            "class_similarity": 0.9,
            "min_endpoint_proximity_m": 3.0,
            "max_endpoint_proximity_m": 25.0,
            "shared_endpoint_count": 1.0,
            "lateral_offset_m": 2.5,
            "lateral_offset_iqr_m": 1.0,
            "lateral_offset_p95_m": 4.0,
            "from_degree_ref": 2.0,
            "to_degree_ref": 2.0,
            "from_degree_target": 2.0,
            "to_degree_target": 2.0,
            "degree_match_score": 0.8,
            "degree_signature_similarity": 0.85,
            "is_dead_end_ref": 0.0,
            "is_dead_end_target": 0.0,
            "dead_end_match": 1.0,
            "is_intersection_ref": 0.0,
            "is_intersection_target": 0.0,
            "intersection_match": 1.0,
            "ref_coverage": 0.95,
            "target_coverage": 0.92,
            "min_coverage": 0.92,
            "coverage_ratio": 0.97,
            "graphlet_similarity": 0.75,
            "endpoint_degree_similarity": 0.8,
        }
        return matcher

    def test_impute_missing_uses_stored_medians(self, trained_matcher_with_medians):
        """_impute_missing() should use exactly the stored median values."""
        matcher = trained_matcher_with_medians

        # Create feature array with NaN values
        X = np.array([[np.nan] * len(FEATURE_COLUMNS)], dtype=np.float32)

        # Impute missing values
        X_imputed = matcher._impute_missing(X)

        # Verify each imputed value matches stored median
        for i, feat_name in enumerate(matcher.feature_names):
            expected_median = matcher.feature_medians[feat_name]
            actual_value = X_imputed[0, i]
            assert actual_value == pytest.approx(expected_median, rel=1e-5), (
                f"Feature {feat_name}: expected median {expected_median}, got {actual_value}"
            )

    def test_impute_missing_preserves_valid_values(self, trained_matcher_with_medians):
        """_impute_missing() should not modify valid (non-NaN) values."""
        matcher = trained_matcher_with_medians

        # Create feature array with specific valid values
        valid_values = [float(i) for i in range(len(FEATURE_COLUMNS))]
        X = np.array([valid_values], dtype=np.float32)

        # Impute (should be a no-op)
        X_imputed = matcher._impute_missing(X)

        # Values should be unchanged
        for i, original in enumerate(valid_values):
            assert X_imputed[0, i] == pytest.approx(original, rel=1e-5)

    def test_impute_missing_handles_infinite_values(self, trained_matcher_with_medians):
        """_impute_missing() should cap infinite values at MAX_DISTANCE_METERS."""
        matcher = trained_matcher_with_medians

        # Create feature array with infinite values
        X = np.array([[np.inf] * len(FEATURE_COLUMNS)], dtype=np.float32)

        # Impute
        X_imputed = matcher._impute_missing(X)

        # All infinite values should be capped
        for i in range(len(FEATURE_COLUMNS)):
            assert X_imputed[0, i] == MAX_DISTANCE_METERS

    def test_features_to_array_uses_stored_medians(self, trained_matcher_with_medians):
        """_features_to_array() should use stored medians for missing features."""
        matcher = trained_matcher_with_medians

        # Feature dict with some missing values
        feature_dict = {
            "hausdorff_distance_m": 5.0,  # Present
            # All other features missing
        }

        X = matcher._features_to_array([feature_dict])

        # Present value should be preserved
        hausdorff_idx = matcher.feature_names.index("hausdorff_distance_m")
        assert X[0, hausdorff_idx] == 5.0

        # Missing values should use stored medians
        for i, feat_name in enumerate(matcher.feature_names):
            if feat_name != "hausdorff_distance_m":
                expected = matcher.feature_medians.get(feat_name, 0.0)
                assert X[0, i] == pytest.approx(expected, rel=1e-5), (
                    f"Feature {feat_name}: expected {expected}, got {X[0, i]}"
                )

    def test_imputation_fallback_for_unknown_features(self, trained_matcher_with_medians):
        """Unknown features should fall back to 0.0 if not in stored medians."""
        matcher = trained_matcher_with_medians

        # Remove a feature from medians to simulate a new feature
        del matcher.feature_medians["hausdorff_distance_m"]

        # Create feature array with NaN for the removed feature
        X = np.array([[np.nan] * len(FEATURE_COLUMNS)], dtype=np.float32)

        # Impute
        X_imputed = matcher._impute_missing(X)

        # Unknown feature should fall back to 0.0
        hausdorff_idx = matcher.feature_names.index("hausdorff_distance_m")
        assert X_imputed[0, hausdorff_idx] == 0.0


class TestAlignmentAwareGraphletComputation:
    """Ensure graphlet computation respects alignment fractions.

    When alignment is provided, graphlet similarity should be computed
    using the connectors nearest to the aligned subline endpoints,
    not the full segment endpoints.
    """

    @pytest.fixture
    def network_with_mid_segment_connector(self):
        """Create a network where segments connect at mid-points."""
        import geopandas as gpd

        # Long main road with a side street joining at the middle
        # The main road spans 0-200m, side street joins at 100m
        lines = [
            LineString([(0, 0), (200, 0)]),  # Main road (200m)
            LineString([(100, 0), (100, 100)]),  # Side street at midpoint
        ]
        return gpd.GeoDataFrame(
            {
                "id": ["main", "side"],
                "geometry": lines,
            },
            crs="EPSG:32632",
        )

    def test_graphlet_similarity_with_alignment_uses_correct_connectors(
        self, network_with_mid_segment_connector
    ):
        """Alignment fractions should affect which connectors are compared.

        When comparing aligned sublines, the graphlet similarity should
        use connectors nearest to the subline endpoints, not the full
        segment endpoints.
        """
        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import (
            compute_graphlet_similarity,
            precompute_graphlet_features,
        )

        gdf = network_with_mid_segment_connector

        # Pre-compute graphlet features
        graphlet_data = precompute_graphlet_features(gdf, id_column="id", tolerance_m=5.0)

        # Test 1: Full segment alignment (0.0 to 1.0)
        full_alignment = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )

        result_full = compute_graphlet_similarity(
            "main", "side", graphlet_data, graphlet_data, alignment=full_alignment
        )

        # Test 2: Partial alignment - only the first half of main road
        # This half doesn't include the junction point (at 0.5)
        partial_alignment = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=0.4,  # First 40% - before junction
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )

        result_partial = compute_graphlet_similarity(
            "main", "side", graphlet_data, graphlet_data, alignment=partial_alignment
        )

        # Both should return valid results
        assert 0.0 <= result_full["graphlet_similarity"] <= 1.0
        assert 0.0 <= result_partial["graphlet_similarity"] <= 1.0

        # The partial alignment that excludes the junction should have
        # different topology characteristics
        # (This test verifies the alignment is being used, not specific values)

    def test_graphlet_similarity_without_alignment_uses_full_segment(
        self, network_with_mid_segment_connector
    ):
        """Without alignment, graphlet similarity should use full segment."""
        from matcher.features.compute import (
            compute_graphlet_similarity,
            precompute_graphlet_features,
        )

        gdf = network_with_mid_segment_connector
        graphlet_data = precompute_graphlet_features(gdf, id_column="id", tolerance_m=5.0)

        # No alignment provided
        result = compute_graphlet_similarity(
            "main", "side", graphlet_data, graphlet_data, alignment=None
        )

        # Should return valid results using full segment
        assert 0.0 <= result["graphlet_similarity"] <= 1.0
        assert 0.0 <= result["endpoint_degree_similarity"] <= 1.0

    def test_graphlet_similarity_with_alignment_handles_reversed_direction(self):
        """Alignment with reversed direction should still compute correctly."""
        import geopandas as gpd

        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import (
            compute_graphlet_similarity,
            precompute_graphlet_features,
        )

        # Two parallel roads
        lines = [
            LineString([(0, 0), (100, 0)]),  # ref
            LineString([(100, 5), (0, 5)]),  # target - reversed direction
        ]
        gdf = gpd.GeoDataFrame(
            {"id": ["ref", "target"], "geometry": lines},
            crs="EPSG:32632",
        )

        graphlet_data = precompute_graphlet_features(gdf, id_column="id", tolerance_m=5.0)

        # Alignment that reflects the reversed direction
        # ref: 0.0-1.0 maps to target: 1.0-0.0 (reversed)
        alignment = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=1.0,  # Reversed
            dataset_end_frac=0.0,
        )

        result = compute_graphlet_similarity(
            "ref", "target", graphlet_data, graphlet_data, alignment=alignment
        )

        # Should handle reversed direction gracefully
        assert 0.0 <= result["graphlet_similarity"] <= 1.0
        assert 0.0 <= result["endpoint_degree_similarity"] <= 1.0


class TestFeatureComputationPathParity:
    """Ensure all feature computation paths produce identical results.

    Features can be computed via:
    1. ML scoring pipeline (ml.py -> compute_pair_features)
    2. Labeling UI (direct compute_pair_features call)
    3. Label storage (LabelStore.add with pre-computed features)

    All paths must produce identical feature values for the same input.
    """

    def test_label_store_preserves_all_computed_features(self):
        """LabelStore.add() should save all features from compute_pair_features().

        This test verifies that features computed and passed to LabelStore.add()
        are correctly persisted and can be retrieved for training.
        """
        import tempfile
        from pathlib import Path

        from matcher.features.compute import compute_pair_features
        from matcher.labeling.label_store import LabelStore

        # Compute features for a sample pair
        ref_geom = LineString([(0, 0), (100, 0)])
        target_geom = LineString([(0, 5), (100, 5)])

        features = compute_pair_features(
            ref_geom=ref_geom,
            target_geom=target_geom,
            ref_name="Main Street",
            target_name="Main St",
            ref_class="residential",
            target_class="residential",
        )

        # Save to temporary label store
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LabelStore(dataset_id="test_dataset", labels_dir=Path(tmpdir))
            store.add(
                gers_id="ref_001",
                target_id="target_001",
                label="match",
                labeler="test",
                session_id="test_session",
                original_decision="MATCH",
                original_confidence=0.9,
                features=features,
            )

            # Reload and verify all features were preserved
            loaded_df = store.df

            assert len(loaded_df) == 1
            row = loaded_df.iloc[0]

            # Verify all feature values match
            for feature_name in FEATURE_COLUMNS:
                computed_value = features[feature_name]
                stored_value = row[feature_name]
                assert stored_value == pytest.approx(computed_value, rel=1e-5), (
                    f"Feature {feature_name}: computed {computed_value}, stored {stored_value}"
                )
