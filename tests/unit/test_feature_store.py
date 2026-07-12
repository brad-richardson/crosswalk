"""Tests for FeatureStore - Parquet storage for computed features."""

import numpy as np
import pandas as pd
import pytest

from crosswalk.config import FEATURE_COLUMNS, FEATURE_VERSION
from crosswalk.labeling.feature_store import FeatureStore


@pytest.fixture
def tmp_features_dir(tmp_path):
    """Temporary directory for feature store files."""
    return tmp_path / "features"


@pytest.fixture
def sample_features():
    """Sample feature values for testing."""
    return {col: float(i) / 10.0 for i, col in enumerate(FEATURE_COLUMNS)}


class TestFeatureStoreAddAndRetrieve:
    def test_add_and_retrieve_pair(self, tmp_features_dir, sample_features):
        """Round-trip through add + get returns correct data."""
        store = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        store.add(
            gers_id="ref-001",
            target_id="target-001",
            features=sample_features,
        )

        result = store.get("ref-001", "target-001")
        assert result is not None
        assert result["gers_id"] == "ref-001"
        assert result["target_id"] == "target-001"
        assert result["feature_version"] == FEATURE_VERSION

        # Check feature values
        for col in FEATURE_COLUMNS:
            assert col in result
            assert abs(result[col] - sample_features[col]) < 1e-6

    def test_missing_pair_returns_none(self, tmp_features_dir):
        """get returns None for non-existent pair."""
        store = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        result = store.get("nonexistent", "also-nonexistent")
        assert result is None

    def test_has_pair(self, tmp_features_dir, sample_features):
        """has_pair correctly detects presence/absence."""
        store = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        assert not store.has_pair("ref-001", "target-001")

        store.add("ref-001", "target-001", sample_features)
        assert store.has_pair("ref-001", "target-001")
        assert not store.has_pair("ref-002", "target-001")

    def test_duplicate_pair_keeps_latest(self, tmp_features_dir, sample_features):
        """Adding a pair twice keeps only the latest entry."""
        store = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        store.add("ref-001", "target-001", sample_features)

        # Add again with different values
        updated_features = {col: 0.999 for col in FEATURE_COLUMNS}
        store.add("ref-001", "target-001", updated_features)

        result = store.get("ref-001", "target-001")
        assert abs(result[FEATURE_COLUMNS[0]] - 0.999) < 1e-6

        # Should only have one entry
        mask = (store.df["gers_id"] == "ref-001") & (store.df["target_id"] == "target-001")
        assert mask.sum() == 1

    def test_custom_feature_version(self, tmp_features_dir, sample_features):
        """Can specify custom feature_version."""
        store = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        store.add("ref-001", "target-001", sample_features, feature_version="custom-v1")

        result = store.get("ref-001", "target-001")
        assert result["feature_version"] == "custom-v1"

    def test_missing_feature_columns_filled_with_nan(self, tmp_features_dir):
        """Missing feature columns are filled with NaN."""
        store = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        partial_features = {FEATURE_COLUMNS[0]: 0.5}  # Only one feature
        store.add("ref-001", "target-001", partial_features)

        result = store.get("ref-001", "target-001")
        assert abs(result[FEATURE_COLUMNS[0]] - 0.5) < 1e-6
        # Other columns should be NaN
        for col in FEATURE_COLUMNS[1:]:
            assert np.isnan(result[col])


class TestFeatureStorePersistence:
    def test_save_and_reload(self, tmp_features_dir, sample_features):
        """Persist to disk and reload preserves data."""
        store1 = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        store1.add("ref-001", "target-001", sample_features)
        store1.save()

        # Reload from disk
        store2 = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        result = store2.get("ref-001", "target-001")
        assert result is not None
        assert result["gers_id"] == "ref-001"
        for col in FEATURE_COLUMNS:
            assert abs(result[col] - sample_features[col]) < 1e-6

    def test_atomic_save_with_backup(self, tmp_features_dir, sample_features):
        """Save creates backup of existing file."""
        store = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        store.add("ref-001", "target-001", sample_features)
        store.save()

        # Modify and save again
        store.add("ref-002", "target-002", sample_features)
        store.save()

        # Check backup exists
        backup_path = store.parquet_path.with_suffix(".parquet.bak")
        assert backup_path.exists()

        # Backup should have only 1 row (original save)
        backup_df = pd.read_parquet(backup_path)
        assert len(backup_df) == 1

        # Primary should have 2 rows
        primary_df = pd.read_parquet(store.parquet_path)
        assert len(primary_df) == 2

    def test_save_does_not_leave_tmp_file(self, tmp_features_dir, sample_features):
        """Temp file is cleaned up after save."""
        store = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        store.add("ref-001", "target-001", sample_features)
        store.save()

        tmp_path = store.parquet_path.with_suffix(".parquet.tmp")
        assert not tmp_path.exists()

    def test_empty_store_on_missing_file(self, tmp_features_dir):
        """Loading from non-existent file returns empty DataFrame."""
        store = FeatureStore("nonexistent_dataset", features_dir=tmp_features_dir)
        assert len(store.df) == 0


class TestFeatureStoreLoadAll:
    def test_load_all_multiple_datasets(self, tmp_features_dir, sample_features):
        """load_all combines features from multiple dataset partitions."""
        # Create features for two datasets
        store1 = FeatureStore("dataset_a", features_dir=tmp_features_dir)
        store1.add("ref-001", "target-001", sample_features)
        store1.save()

        store2 = FeatureStore("dataset_b", features_dir=tmp_features_dir)
        store2.add("ref-002", "target-002", sample_features)
        store2.save()

        # Load all
        all_features = FeatureStore.load_all(tmp_features_dir)
        assert len(all_features) == 2
        assert set(all_features["dataset"].unique()) == {"dataset_a", "dataset_b"}

    def test_load_all_empty_dir(self, tmp_features_dir):
        """load_all returns empty DataFrame for non-existent directory."""
        all_features = FeatureStore.load_all(tmp_features_dir)
        assert len(all_features) == 0
        assert "dataset" in all_features.columns

    def test_load_all_has_correct_schema(self, tmp_features_dir, sample_features):
        """load_all returns DataFrame with all expected columns."""
        store = FeatureStore("test_dataset", features_dir=tmp_features_dir)
        store.add("ref-001", "target-001", sample_features)
        store.save()

        all_features = FeatureStore.load_all(tmp_features_dir)

        # Check key columns
        assert "gers_id" in all_features.columns
        assert "target_id" in all_features.columns
        assert "feature_version" in all_features.columns
        assert "dataset" in all_features.columns

        # Check all feature columns
        for col in FEATURE_COLUMNS:
            assert col in all_features.columns

    def test_load_all_returns_canonical_row_order(self, tmp_features_dir, sample_features):
        """Partition creation and in-file row order do not affect bulk row order."""
        store_b = FeatureStore("dataset_b", features_dir=tmp_features_dir)
        store_b.add("ref-002", "target-002", sample_features)
        store_b.add("ref-001", "target-003", sample_features)
        store_b.save()

        store_a = FeatureStore("dataset_a", features_dir=tmp_features_dir)
        store_a.add("ref-003", "target-002", sample_features)
        store_a.add("ref-003", "target-001", sample_features)
        store_a.save()

        rows = FeatureStore.load_all(tmp_features_dir)[
            ["dataset", "gers_id", "target_id"]
        ].itertuples(index=False, name=None)

        assert list(rows) == [
            ("dataset_a", "ref-003", "target-001"),
            ("dataset_a", "ref-003", "target-002"),
            ("dataset_b", "ref-001", "target-003"),
            ("dataset_b", "ref-002", "target-002"),
        ]

    def test_load_all_strict_rejects_missing_partition_file(self, tmp_features_dir):
        """Strict bulk loading fails instead of silently skipping an empty partition."""
        partition = tmp_features_dir / "dataset=missing"
        partition.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="no data.parquet"):
            FeatureStore.load_all(tmp_features_dir, skip_errors=False)

    def test_strict_required_datasets_ignore_unrelated_orphan(
        self, tmp_features_dir, sample_features
    ):
        """An archival directory outside the requested training set does not block loading."""
        store = FeatureStore("required", features_dir=tmp_features_dir)
        store.add("ref-001", "target-001", sample_features)
        store.save()
        (tmp_features_dir / "dataset=orphan").mkdir()

        loaded = FeatureStore.load_all(
            tmp_features_dir,
            skip_errors=False,
            required_datasets={"required"},
        )

        assert list(loaded["dataset"]) == ["required"]

    def test_load_all_strict_propagates_corrupt_partition(self, tmp_features_dir):
        """Strict bulk loading propagates parquet read failures."""
        partition = tmp_features_dir / "dataset=corrupt"
        partition.mkdir(parents=True)
        (partition / "data.parquet").write_bytes(b"not a parquet file")

        # The default remains tolerant for inspection/reporting callers.
        assert FeatureStore.load_all(tmp_features_dir).empty
        with pytest.raises(ValueError):
            FeatureStore.load_all(tmp_features_dir, skip_errors=False)
