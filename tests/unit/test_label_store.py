"""Tests for label store and feature parity."""

from shapely.geometry import LineString

from matcher.features.compute import ALL_FEATURE_COLUMNS, compute_pair_features
from matcher.labeling.label_store import LabelStore
from matcher.matching.ml import FEATURE_COLUMNS
from tests.conftest import MOCK_TOPOLOGY_FEATURES


class TestFeatureParity:
    """Ensure computed features match what gets saved to labels.

    This is a critical invariant: any feature computed during matching
    must also be saved to labels, otherwise ML training can't use it.
    """

    def test_compute_pair_features_returns_all_declared_features(self):
        """compute_pair_features() should return all features in ALL_FEATURE_COLUMNS.

        This catches bugs where a feature is declared but not actually computed.
        """
        from shapely import LineString

        # Create simple test geometries
        ref_geom = LineString([(0, 0), (100, 0)])
        target_geom = LineString([(0, 5), (100, 5)])

        features = compute_pair_features(
            ref_geom=ref_geom,
            target_geom=target_geom,
            ref_class="residential",
            target_class="residential",
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        declared_features = set(ALL_FEATURE_COLUMNS)
        # Filter out internal metadata fields (prefixed with _)
        computed_features = {k for k in features if not k.startswith("_")}

        missing_from_output = declared_features - computed_features
        assert not missing_from_output, (
            f"Features declared in ALL_FEATURE_COLUMNS but not returned by "
            f"compute_pair_features: {sorted(missing_from_output)}"
        )

        extra_in_output = computed_features - declared_features
        assert not extra_in_output, (
            f"Features returned by compute_pair_features but not declared in "
            f"ALL_FEATURE_COLUMNS: {sorted(extra_in_output)}\n"
            f"Add these to ALL_FEATURE_COLUMNS in compute.py."
        )

    def test_ml_feature_columns_match_computed_features(self):
        """ML FEATURE_COLUMNS must match ALL_FEATURE_COLUMNS.

        This ensures the ML model uses the same features that are computed
        and saved to labels. A mismatch would cause training failures or
        incorrect predictions.
        """
        ml_features = set(FEATURE_COLUMNS)
        computed_features = set(ALL_FEATURE_COLUMNS)

        missing_from_ml = computed_features - ml_features
        assert not missing_from_ml, (
            f"Features computed but not used by ML model: {sorted(missing_from_ml)}\n"
            f"Add these to FEATURE_COLUMNS in ml.py."
        )

        extra_in_ml = ml_features - computed_features
        assert not extra_in_ml, (
            f"ML model uses features that are not computed: {sorted(extra_in_ml)}\n"
            f"Either add these to ALL_FEATURE_COLUMNS in compute.py or remove from ml.py."
        )


class TestGeometryPersistence:
    """Test that LabelStore.add() with geometry params creates companion file."""

    def test_add_with_geometry_creates_companion_file(self, tmp_path):
        """Adding a label with geometry params persists to normalized stores."""

        from matcher.labeling.data_store import DataStore
        from matcher.labeling.feature_store import FeatureStore

        labels_dir = tmp_path / "labels"
        dataset_id = "test_dataset_geo_persist"

        store = LabelStore(dataset_id, labels_dir=labels_dir)

        ref_geom = LineString([(0.0, 0.0), (1.0, 1.0)])
        target_geom = LineString([(0.0, 0.1), (1.0, 1.1)])

        store.add(
            gers_id="ref-001",
            target_id="target-001",
            label="match",
            labeler="tester",
            session_id="sess-001",
            original_decision="review",
            original_confidence=0.75,
            features={col: 0.5 for col in ALL_FEATURE_COLUMNS},
            ref_geometry=ref_geom,
            target_geometry=target_geom,
            ref_class_raw="residential",
            target_class_raw="residential",
            ref_subclass="urban",
            target_subclass="urban",
            ref_names={"primary": "Main St"},
            target_names={"primary": "Main Street"},
        )

        # Label should be saved to labels/human/
        human_csv_path = labels_dir / "human" / f"dataset={dataset_id}" / "data.csv"
        assert human_csv_path.exists(), f"Human labels not created at {human_csv_path}"
        assert len(store.df) == 1
        assert store.df.iloc[0]["gers_id"] == "ref-001"

        # Data should be saved to labels/data/
        data_path = labels_dir / "data" / f"dataset={dataset_id}" / "data.parquet"
        assert data_path.exists(), f"Data file not created at {data_path}"

        # Features should be saved to labels/features/
        features_path = labels_dir / "features" / f"dataset={dataset_id}" / "data.parquet"
        assert features_path.exists(), f"Features file not created at {features_path}"

        # Verify geometry was persisted correctly
        data_store = DataStore(dataset_id, data_dir=labels_dir / "data")
        result = data_store.get_pair("ref-001", "target-001")
        assert result is not None
        assert result["ref_names"]["primary"] == "Main St"
        assert isinstance(result["ref_geometry"], LineString)

        # Verify features were persisted correctly
        feature_store = FeatureStore(dataset_id, features_dir=labels_dir / "features")
        features = feature_store.get("ref-001", "target-001")
        assert features is not None
        assert features["buffer_iou_5m"] == 0.5

    def test_add_without_geometry_no_error(self, tmp_path):
        """Adding a label without geometry params does not raise an error."""
        labels_dir = tmp_path / "labels"

        store = LabelStore("test_dataset", labels_dir=labels_dir)

        store.add(
            gers_id="ref-001",
            target_id="target-001",
            label="match",
            labeler="tester",
            session_id="sess-001",
            original_decision="review",
            original_confidence=0.75,
            features={col: 0.5 for col in ALL_FEATURE_COLUMNS},
        )

        # No error should be raised, label should be saved
        assert len(store.df) == 1


class TestLabelUpdateDelete:
    """Test update_label and delete_label methods."""

    def test_update_label(self, tmp_path):
        """update_label should modify label value while preserving other metadata."""
        labels_dir = tmp_path / "labels"
        dataset_id = "test_update"

        store = LabelStore(dataset_id, labels_dir=labels_dir)

        # Add initial label
        store.add(
            gers_id="ref-001",
            target_id="target-001",
            label="match",
            labeler="user1",
            session_id="sess-001",
            original_decision="review",
            original_confidence=0.75,
            features={col: 0.5 for col in ALL_FEATURE_COLUMNS},
        )

        assert store.df.iloc[0]["label"] == "match"
        assert store.df.iloc[0]["labeler"] == "user1"

        # Update label
        result = store.update_label("ref-001", "target-001", "no_match", "user2")
        assert result is True

        # Reload store to verify persistence
        store2 = LabelStore(dataset_id, labels_dir=labels_dir)
        assert store2.df.iloc[0]["label"] == "no_match"
        assert store2.df.iloc[0]["labeler"] == "user2"
        # Original metadata should be preserved
        assert store2.df.iloc[0]["original_decision"] == "review"
        assert store2.df.iloc[0]["original_confidence"] == 0.75

    def test_update_label_not_found(self, tmp_path):
        """update_label should return False when pair not found."""
        labels_dir = tmp_path / "labels"
        store = LabelStore("test_update_missing", labels_dir=labels_dir)

        result = store.update_label("nonexistent", "pair", "match", "user")
        assert result is False

    def test_delete_label(self, tmp_path):
        """delete_label should remove label and associated data."""
        from matcher.labeling.data_store import DataStore
        from matcher.labeling.feature_store import FeatureStore

        labels_dir = tmp_path / "labels"
        dataset_id = "test_delete"

        store = LabelStore(dataset_id, labels_dir=labels_dir)

        ref_geom = LineString([(0.0, 0.0), (1.0, 1.0)])
        target_geom = LineString([(0.0, 0.1), (1.0, 1.1)])

        # Add label with geometry
        store.add(
            gers_id="ref-001",
            target_id="target-001",
            label="match",
            labeler="tester",
            session_id="sess-001",
            original_decision="review",
            original_confidence=0.75,
            features={col: 0.5 for col in ALL_FEATURE_COLUMNS},
            ref_geometry=ref_geom,
            target_geometry=target_geom,
            ref_names={"primary": "Main St"},
            target_names={"primary": "Main Street"},
        )

        # Verify all stores have data
        assert len(store.df) == 1
        fs = FeatureStore(dataset_id, features_dir=labels_dir / "features")
        assert fs.has_pair("ref-001", "target-001")
        ds = DataStore(dataset_id, data_dir=labels_dir / "data")
        assert ds.has_pair("ref-001", "target-001")

        # Delete label
        result = store.delete_label("ref-001", "target-001")
        assert result is True

        # Verify all stores are empty
        store2 = LabelStore(dataset_id, labels_dir=labels_dir)
        assert len(store2.df) == 0
        fs2 = FeatureStore(dataset_id, features_dir=labels_dir / "features")
        assert not fs2.has_pair("ref-001", "target-001")
        ds2 = DataStore(dataset_id, data_dir=labels_dir / "data")
        assert not ds2.has_pair("ref-001", "target-001")

    def test_delete_label_not_found(self, tmp_path):
        """delete_label should return False when pair not found."""
        labels_dir = tmp_path / "labels"
        store = LabelStore("test_delete_missing", labels_dir=labels_dir)

        result = store.delete_label("nonexistent", "pair")
        assert result is False
