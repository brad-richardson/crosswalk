"""Tests for ML pipeline consistency.

These tests ensure feature computation is consistent across training and inference:
1. Pre-computed features (endpoint, topology) are passed through correctly
2. Missing values: NaN is preserved for XGBoost's native handling, inf is capped
3. Alignment-aware graphlet computation respects alignment fractions

These catch subtle training/inference skew that would degrade model performance.
"""

import math
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
        from tests.conftest import MOCK_TOPOLOGY_FEATURES

        spatial_index = SpatialContextIndex()
        spatial_index.build_from_gdf(t_network, id_column="id")

        target_geom = t_network.geometry.iloc[0]
        precomputed = compute_endpoint_features(target_geom, spatial_index, exclude_segment_idx=0)

        features = compute_pair_features(
            ref_geom_full=t_network.geometry.iloc[1],
            target_geom_full=target_geom,
            ref_class=None,
            target_class=None,
            endpoint_features=precomputed,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # Pre-computed values should pass through (capped at MAX_DISTANCE_METERS)
        assert features["min_endpoint_proximity_m"] == precomputed["min_endpoint_proximity_m"]
        assert features["shared_endpoint_count"] == precomputed["shared_endpoint_count"]
        # Should have real values (network has nearby endpoints)
        assert precomputed["min_endpoint_proximity_m"] < MAX_DISTANCE_METERS
        # max_endpoint_proximity_m may be Inf from JIT helper when endpoints
        # are not found — compute.py caps it to MAX_DISTANCE_METERS
        assert features["max_endpoint_proximity_m"] <= MAX_DISTANCE_METERS

    def test_topology_features_passed_through(self, t_network):
        """Pre-computed topology features should be used unchanged."""
        from matcher.features.compute import compute_pair_features
        from matcher.features.spatial_context import compute_all_topology
        from tests.conftest import MOCK_ENDPOINT_FEATURES

        topology = compute_all_topology(
            t_network, id_column="id", tolerance_m=5.0, ids_to_compute={"main_w", "main_e"}
        )

        features = compute_pair_features(
            ref_geom_full=t_network.geometry.iloc[1],
            target_geom_full=t_network.geometry.iloc[0],
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
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        features = compute_pair_features(
            ref_geom_full=t_network.geometry.iloc[0],
            target_geom_full=t_network.geometry.iloc[1],
            ref_class=None,
            target_class=None,
            graphlet_features=graphlet_input,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        assert features["graphlet_similarity"] == expected_sim
        assert features["endpoint_degree_similarity"] == expected_deg


class TestMissingValueHandling:
    """Verify NaN is preserved for XGBoost and inf is capped."""

    @pytest.fixture
    def matcher_instance(self):
        """Matcher instance for testing."""
        from matcher.matching.ml import MLMatcher

        matcher = MLMatcher()
        matcher.feature_names = FEATURE_COLUMNS.copy()
        return matcher

    def test_nan_values_preserved(self, matcher_instance):
        """NaN values should be preserved for XGBoost's native handling."""
        X = np.array([[np.nan] * len(FEATURE_COLUMNS)], dtype=np.float32)
        X_out = matcher_instance._cap_infinities(X)

        for i in range(len(FEATURE_COLUMNS)):
            assert np.isnan(X_out[0, i]), f"NaN should be preserved for {FEATURE_COLUMNS[i]}"

    def test_valid_values_preserved(self, matcher_instance):
        """Valid values should not be modified."""
        original = np.arange(len(FEATURE_COLUMNS), dtype=np.float32).reshape(1, -1)
        X_out = matcher_instance._cap_infinities(original.copy())

        np.testing.assert_array_almost_equal(X_out, original)

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            (np.inf, MAX_DISTANCE_METERS),
            (-np.inf, MAX_DISTANCE_METERS),
        ],
        ids=["positive_inf", "negative_inf"],
    )
    def test_infinite_values_capped(self, matcher_instance, input_value, expected):
        """Infinite values should be capped at MAX_DISTANCE_METERS."""
        X = np.full((1, len(FEATURE_COLUMNS)), input_value, dtype=np.float32)
        X_out = matcher_instance._cap_infinities(X)

        assert X_out[0, 0] == expected

    def test_features_to_array_preserves_nan(self, matcher_instance):
        """_features_to_array() should preserve NaN for missing dict keys."""
        feature_dict = {"hausdorff_distance_m": 99.0}  # Only one feature present
        X = matcher_instance._features_to_array([feature_dict])

        # Present value preserved
        idx = FEATURE_COLUMNS.index("hausdorff_distance_m")
        assert X[0, idx] == 99.0

        # Missing values are NaN (XGBoost handles natively)
        other_idx = FEATURE_COLUMNS.index("buffer_iou_5m")
        assert np.isnan(X[0, other_idx])


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
        from matcher.labeling.feature_store import FeatureStore
        from matcher.labeling.label_store import LabelStore
        from tests.conftest import MOCK_TOPOLOGY_FEATURES

        features = compute_pair_features(
            ref_geom_full=LineString([(0, 0), (100, 0)]),
            target_geom_full=LineString([(0, 5), (100, 5)]),
            ref_class="residential",
            target_class="residential",
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
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

            # Features are now stored in FeatureStore (normalized format)
            feature_store = FeatureStore(dataset_id="test", features_dir=Path(tmpdir) / "features")
            stored_features = feature_store.get("ref_001", "target_001")
            assert stored_features is not None, "Features not found in FeatureStore"

            for feat in FEATURE_COLUMNS:
                expected = features[feat]
                actual = stored_features[feat]
                # NaN == NaN should pass (both sides consistently NaN)
                if isinstance(expected, float) and math.isnan(expected):
                    assert isinstance(actual, float) and math.isnan(actual), (
                        f"Mismatch: {feat} — expected NaN, got {actual}"
                    )
                else:
                    assert actual == pytest.approx(expected, rel=1e-5), f"Mismatch: {feat}"
