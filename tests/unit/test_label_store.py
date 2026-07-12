"""Tests for label store and feature parity."""

import pandas as pd
import pytest
from shapely.geometry import LineString

from crosswalk.config import FEATURE_COLUMNS
from crosswalk.features.compute import compute_pair_features
from crosswalk.labeling.feature_store import FeatureStore
from crosswalk.labeling.label_store import LabelStore
from tests.conftest import MOCK_TOPOLOGY_FEATURES


class TestFeatureParity:
    """Ensure computed features match what gets saved to labels.

    This is a critical invariant: any feature computed during matching
    must also be saved to labels, otherwise ML training can't use it.
    """

    def test_compute_pair_features_returns_all_declared_features(self):
        """compute_pair_features() should return all features in FEATURE_COLUMNS.

        This catches bugs where a feature is declared but not actually computed.
        """
        from shapely import LineString

        # Create simple test geometries
        ref_geom = LineString([(0, 0), (100, 0)])
        target_geom = LineString([(0, 5), (100, 5)])

        features = compute_pair_features(
            ref_geom_full=ref_geom,
            target_geom_full=target_geom,
            ref_class="residential",
            target_class="residential",
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        declared_features = set(FEATURE_COLUMNS)
        # Filter out internal metadata fields (prefixed with _)
        computed_features = {k for k in features if not k.startswith("_")}

        missing_from_output = declared_features - computed_features
        assert not missing_from_output, (
            f"Features declared in FEATURE_COLUMNS but not returned by "
            f"compute_pair_features: {sorted(missing_from_output)}"
        )

        extra_in_output = computed_features - declared_features
        assert not extra_in_output, (
            f"Features returned by compute_pair_features but not declared in "
            f"FEATURE_COLUMNS: {sorted(extra_in_output)}\n"
            f"Add these to FEATURE_COLUMNS in config.py."
        )


class TestGeometryPersistence:
    """Test that LabelStore.add() with geometry params creates companion file."""

    def test_add_with_geometry_creates_companion_file(self, tmp_path):
        """Adding a label with geometry params persists to normalized stores."""

        from crosswalk.labeling.data_store import DataStore
        from crosswalk.labeling.feature_store import FeatureStore

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
            features={col: 0.5 for col in FEATURE_COLUMNS},
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
            features={col: 0.5 for col in FEATURE_COLUMNS},
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
            features={col: 0.5 for col in FEATURE_COLUMNS},
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
        from crosswalk.labeling.data_store import DataStore
        from crosswalk.labeling.feature_store import FeatureStore

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
            features={col: 0.5 for col in FEATURE_COLUMNS},
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


class TestBulkLabelLoading:
    @staticmethod
    def _write_human_partition(human_dir, dataset_id, rows):
        partition = human_dir / f"dataset={dataset_id}"
        partition.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(partition / "data.csv", index=False)

    def test_human_labels_return_canonical_row_order(self, tmp_path):
        """Partition creation and CSV row order do not affect bulk row order."""
        human_dir = tmp_path / "human"
        self._write_human_partition(
            human_dir,
            "dataset_b",
            [
                {"gers_id": "ref-002", "target_id": "target-001", "label": "match"},
                {"gers_id": "ref-001", "target_id": "target-003", "label": "match"},
            ],
        )
        self._write_human_partition(
            human_dir,
            "dataset_a",
            [
                {"gers_id": "ref-003", "target_id": "target-002", "label": "match"},
                {"gers_id": "ref-003", "target_id": "target-001", "label": "match"},
            ],
        )

        rows = LabelStore.load_human_labels(human_dir)[
            ["dataset", "gers_id", "target_id"]
        ].itertuples(index=False, name=None)

        assert list(rows) == [
            ("dataset_a", "ref-003", "target-001"),
            ("dataset_a", "ref-003", "target-002"),
            ("dataset_b", "ref-001", "target-003"),
            ("dataset_b", "ref-002", "target-001"),
        ]

    def test_load_all_returns_canonical_training_order(self, tmp_path):
        """The joined label/feature training table has a canonical row order."""
        labels_dir = tmp_path / "labels"
        features = {column: 0.5 for column in FEATURE_COLUMNS}

        for dataset_id, pairs in [
            ("dataset_b", [("ref-002", "target-001"), ("ref-001", "target-003")]),
            ("dataset_a", [("ref-003", "target-002"), ("ref-003", "target-001")]),
        ]:
            store = LabelStore(dataset_id, labels_dir=labels_dir)
            for gers_id, target_id in pairs:
                store.add(
                    gers_id=gers_id,
                    target_id=target_id,
                    label="match",
                    labeler="tester",
                    session_id="session",
                    original_decision="review",
                    original_confidence=0.5,
                    features=features,
                )

        rows = LabelStore.load_all(labels_dir, skip_errors=False)[
            ["dataset", "gers_id", "target_id"]
        ].itertuples(index=False, name=None)

        assert list(rows) == [
            ("dataset_a", "ref-003", "target-001"),
            ("dataset_a", "ref-003", "target-002"),
            ("dataset_b", "ref-001", "target-003"),
            ("dataset_b", "ref-002", "target-001"),
        ]

    def test_strict_human_loading_rejects_missing_partition_file(self, tmp_path):
        """Strict bulk loading fails on a declared partition without data.csv."""
        human_dir = tmp_path / "human"
        (human_dir / "dataset=missing").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="no data.csv"):
            LabelStore.load_human_labels(human_dir, skip_errors=False)

    def test_load_all_strict_requires_feature_partition(self, tmp_path):
        """Strict training loading cannot drop a label dataset lacking features."""
        labels_dir = tmp_path / "labels"
        self._write_human_partition(
            labels_dir / "human",
            "dataset_a",
            [{"gers_id": "ref-001", "target_id": "target-001", "label": "match"}],
        )

        with pytest.raises(FileNotFoundError, match="Feature partition"):
            LabelStore.load_all(labels_dir, skip_errors=False)

    def test_load_all_strict_rejects_missing_feature_join_key(self, tmp_path):
        """Strict training loading cannot silently inner-join away a label row."""
        labels_dir = tmp_path / "labels"
        self._write_human_partition(
            labels_dir / "human",
            "dataset_a",
            [{"gers_id": "ref-001", "target_id": "target-001", "label": "match"}],
        )
        feature_store = FeatureStore("dataset_a", features_dir=labels_dir / "features")
        feature_store.add(
            "different-ref",
            "different-target",
            {column: 0.5 for column in FEATURE_COLUMNS},
        )
        feature_store.save()

        with pytest.raises(ValueError, match="missing feature join keys"):
            LabelStore.load_all(labels_dir, skip_errors=False)

    def test_load_all_strict_rejects_duplicate_feature_join_key(self, tmp_path):
        """Strict training loading cannot multiply labels via duplicate features."""
        labels_dir = tmp_path / "labels"
        self._write_human_partition(
            labels_dir / "human",
            "dataset_a",
            [{"gers_id": "ref-001", "target_id": "target-001", "label": "match"}],
        )
        feature_store = FeatureStore("dataset_a", features_dir=labels_dir / "features")
        feature_store.add(
            "ref-001",
            "target-001",
            {column: 0.5 for column in FEATURE_COLUMNS},
        )
        feature_store.save()
        parquet_path = labels_dir / "features" / "dataset=dataset_a" / "data.parquet"
        features = pd.read_parquet(parquet_path)
        pd.concat([features, features], ignore_index=True).to_parquet(parquet_path, index=False)

        with pytest.raises(ValueError, match="duplicate training join keys"):
            LabelStore.load_all(labels_dir, skip_errors=False)
