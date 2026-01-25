"""Tests for ML pipeline consistency.

These tests ensure feature computation is consistent across training and inference:
1. Pre-computed features (endpoint, topology) are passed through correctly
2. Inference imputation uses stored training medians
3. Alignment-aware graphlet computation respects alignment fractions

These catch subtle training/inference skew that would degrade model performance.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from matcher.config import FEATURE_COLUMNS, MAX_DISTANCE_METERS


class TestPrecomputedFeaturePassthrough:
    """Verify pre-computed features are passed through to compute_pair_features().

    The ML scorer pre-computes endpoint, topology, and graphlet features for
    efficiency. These tests verify the values are passed through unchanged.
    """

    @pytest.fixture
    def t_network(self):
        """T-intersection network with known topology."""
        return gpd.GeoDataFrame(
            {
                "id": ["main_w", "main_e", "side"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (200, 0)]),
                    LineString([(100, 0), (100, 100)]),
                ],
            },
            crs="EPSG:32632",
        )

    def test_endpoint_features_passed_through(self, t_network):
        """Pre-computed endpoint features should be used unchanged."""
        from matcher.features.compute import compute_pair_features
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            compute_endpoint_features,
        )

        spatial_index = SpatialContextIndex()
        spatial_index.build_from_gdf(t_network, id_column="id")

        target_geom = t_network.geometry.iloc[0]
        precomputed = compute_endpoint_features(target_geom, spatial_index, exclude_segment_idx=0)

        features = compute_pair_features(
            ref_geom=t_network.geometry.iloc[1],
            target_geom=target_geom,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            endpoint_features=precomputed,
        )

        # Pre-computed values should pass through exactly
        assert features["min_endpoint_proximity_m"] == precomputed["min_endpoint_proximity_m"]
        assert features["max_endpoint_proximity_m"] == precomputed["max_endpoint_proximity_m"]
        assert features["shared_endpoint_count"] == precomputed["shared_endpoint_count"]
        # Should have real values (network has nearby endpoints)
        assert precomputed["min_endpoint_proximity_m"] < MAX_DISTANCE_METERS

    def test_topology_features_passed_through(self, t_network):
        """Pre-computed topology features should be used unchanged."""
        from matcher.features.compute import compute_pair_features
        from matcher.features.spatial_context import compute_all_topology
        from tests.conftest import MOCK_ENDPOINT_FEATURES

        topology = compute_all_topology(
            t_network, id_column="id", tolerance_m=5.0, ids_to_compute={"main_w", "main_e"}
        )

        features = compute_pair_features(
            ref_geom=t_network.geometry.iloc[1],
            target_geom=t_network.geometry.iloc[0],
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            ref_topology=topology["main_e"],
            target_topology=topology["main_w"],
            endpoint_features=MOCK_ENDPOINT_FEATURES,
        )

        # Topology values should pass through
        assert features["from_degree_ref"] == topology["main_e"]["from_degree"]
        assert features["to_degree_ref"] == topology["main_e"]["to_degree"]
        assert features["is_dead_end_ref"] == (1.0 if topology["main_e"]["is_dead_end"] else 0.0)

    @pytest.mark.parametrize(
        "graphlet_input,expected_sim,expected_deg",
        [
            ({"graphlet_similarity": 0.85, "endpoint_degree_similarity": 0.92}, 0.85, 0.92),
            ({"graphlet_similarity": 0.0, "endpoint_degree_similarity": 1.0}, 0.0, 1.0),
        ],
        ids=["typical_values", "edge_values"],
    )
    def test_graphlet_features_passed_through(
        self, t_network, graphlet_input, expected_sim, expected_deg
    ):
        """Pre-computed graphlet features should be used unchanged."""
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES

        features = compute_pair_features(
            ref_geom=t_network.geometry.iloc[0],
            target_geom=t_network.geometry.iloc[1],
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            graphlet_features=graphlet_input,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
        )

        assert features["graphlet_similarity"] == expected_sim
        assert features["endpoint_degree_similarity"] == expected_deg


class TestImputationConsistency:
    """Verify imputation uses stored training medians consistently."""

    @pytest.fixture
    def matcher_with_medians(self):
        """Matcher with programmatically generated median values."""
        from matcher.matching.ml import MLMatcher

        matcher = MLMatcher()
        matcher.feature_names = FEATURE_COLUMNS.copy()
        # Generate predictable medians: feature index * 0.1
        matcher.feature_medians = {feat: i * 0.1 for i, feat in enumerate(FEATURE_COLUMNS)}
        return matcher

    def test_nan_values_replaced_with_medians(self, matcher_with_medians):
        """NaN values should be replaced with stored medians."""
        X = np.array([[np.nan] * len(FEATURE_COLUMNS)], dtype=np.float32)
        X_imputed = matcher_with_medians._impute_missing(X)

        for i, feat in enumerate(FEATURE_COLUMNS):
            expected = matcher_with_medians.feature_medians[feat]
            assert X_imputed[0, i] == pytest.approx(expected, rel=1e-5)

    def test_valid_values_preserved(self, matcher_with_medians):
        """Valid values should not be modified by imputation."""
        original = np.arange(len(FEATURE_COLUMNS), dtype=np.float32).reshape(1, -1)
        X_imputed = matcher_with_medians._impute_missing(original.copy())

        np.testing.assert_array_almost_equal(X_imputed, original)

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            (np.inf, MAX_DISTANCE_METERS),
            (-np.inf, MAX_DISTANCE_METERS),
        ],
        ids=["positive_inf", "negative_inf"],
    )
    def test_infinite_values_capped(self, matcher_with_medians, input_value, expected):
        """Infinite values should be capped at MAX_DISTANCE_METERS."""
        X = np.full((1, len(FEATURE_COLUMNS)), input_value, dtype=np.float32)
        X_imputed = matcher_with_medians._impute_missing(X)

        assert X_imputed[0, 0] == expected

    def test_missing_median_falls_back_to_zero(self, matcher_with_medians):
        """Features not in stored medians should fall back to 0.0."""
        del matcher_with_medians.feature_medians["hausdorff_distance_m"]
        X = np.array([[np.nan] * len(FEATURE_COLUMNS)], dtype=np.float32)

        X_imputed = matcher_with_medians._impute_missing(X)

        idx = FEATURE_COLUMNS.index("hausdorff_distance_m")
        assert X_imputed[0, idx] == 0.0

    def test_features_to_array_applies_medians(self, matcher_with_medians):
        """_features_to_array() should apply medians for missing dict keys."""
        feature_dict = {"hausdorff_distance_m": 99.0}  # Only one feature present
        X = matcher_with_medians._features_to_array([feature_dict])

        # Present value preserved
        idx = FEATURE_COLUMNS.index("hausdorff_distance_m")
        assert X[0, idx] == 99.0

        # Missing values get medians
        other_idx = FEATURE_COLUMNS.index("buffer_iou_5m")
        expected = matcher_with_medians.feature_medians["buffer_iou_5m"]
        assert X[0, other_idx] == pytest.approx(expected, rel=1e-5)


class TestAlignmentAwareGraphletComputation:
    """Verify graphlet computation respects alignment fractions."""

    @pytest.fixture
    def mid_junction_network(self):
        """Network where side street joins main road at midpoint."""
        return gpd.GeoDataFrame(
            {
                "id": ["main", "side"],
                "geometry": [
                    LineString([(0, 0), (200, 0)]),  # 200m main road
                    LineString([(100, 0), (100, 100)]),  # Side at midpoint
                ],
            },
            crs="EPSG:32632",
        )

    @pytest.mark.parametrize(
        "alignment_fracs,description",
        [
            ((0.0, 1.0, 0.0, 1.0), "full_segment"),
            ((0.0, 0.4, 0.0, 1.0), "partial_before_junction"),
            ((0.6, 1.0, 0.0, 1.0), "partial_after_junction"),
        ],
        ids=["full", "partial_before", "partial_after"],
    )
    def test_alignment_produces_valid_similarity(
        self, mid_junction_network, alignment_fracs, description
    ):
        """Graphlet similarity should be valid for various alignment fractions."""
        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import (
            compute_graphlet_similarity,
            precompute_graphlet_features,
        )

        graphlet_data = precompute_graphlet_features(
            mid_junction_network, id_column="id", tolerance_m=5.0
        )

        alignment = AlignmentResult(
            overture_start_frac=alignment_fracs[0],
            overture_end_frac=alignment_fracs[1],
            dataset_start_frac=alignment_fracs[2],
            dataset_end_frac=alignment_fracs[3],
        )

        result = compute_graphlet_similarity(
            "main", "side", graphlet_data, graphlet_data, alignment=alignment
        )

        assert 0.0 <= result["graphlet_similarity"] <= 1.0
        assert 0.0 <= result["endpoint_degree_similarity"] <= 1.0

    def test_reversed_direction_handled(self):
        """Reversed alignment direction should compute correctly."""
        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import (
            compute_graphlet_similarity,
            precompute_graphlet_features,
        )

        gdf = gpd.GeoDataFrame(
            {
                "id": ["ref", "target"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 5), (0, 5)]),  # Reversed direction
                ],
            },
            crs="EPSG:32632",
        )

        graphlet_data = precompute_graphlet_features(gdf, id_column="id", tolerance_m=5.0)
        alignment = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=1.0,  # Reversed
            dataset_end_frac=0.0,
        )

        result = compute_graphlet_similarity(
            "ref", "target", graphlet_data, graphlet_data, alignment=alignment
        )

        assert 0.0 <= result["graphlet_similarity"] <= 1.0


class TestLabelStoreParity:
    """Verify features survive the label storage round-trip."""

    def test_all_features_preserved_after_storage(self):
        """All computed features should be retrievable from stored labels."""
        from matcher.features.compute import compute_pair_features
        from matcher.labeling.label_store import LabelStore

        features = compute_pair_features(
            ref_geom=LineString([(0, 0), (100, 0)]),
            target_geom=LineString([(0, 5), (100, 5)]),
            ref_name="Main Street",
            target_name="Main St",
            ref_class="residential",
            target_class="residential",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LabelStore(dataset_id="test", labels_dir=Path(tmpdir))
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

            row = store.df.iloc[0]
            for feat in FEATURE_COLUMNS:
                assert row[feat] == pytest.approx(features[feat], rel=1e-5), f"Mismatch: {feat}"
