"""Tests for DataStore - GeoParquet storage for pair geometries and attributes."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from matcher.labeling.data_store import DataStore


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory for data store files."""
    return tmp_path / "data"


@pytest.fixture
def sample_pair_data():
    """Sample pair data for testing."""
    return {
        "gers_id": "ref-001",
        "target_id": "target-001",
        "ref_geometry": LineString([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]),
        "target_geometry": LineString([(0.1, 0.1), (1.1, 1.1), (2.1, 0.1)]),
        "ref_name": "Main Street",
        "target_name": "Main St",
        "ref_class": "residential",
        "target_class": "residential",
        "ref_subclass": "urban",
        "target_subclass": "urban",
    }


class TestDataStoreAddAndRetrieve:
    def test_add_and_retrieve_pair(self, tmp_data_dir, sample_pair_data):
        """Round-trip through add + get_pair returns correct data."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        store.add(**sample_pair_data)

        result = store.get_pair("ref-001", "target-001")
        assert result is not None
        assert result["gers_id"] == "ref-001"
        assert result["target_id"] == "target-001"
        assert result["ref_name"] == "Main Street"
        assert result["target_name"] == "Main St"
        assert result["ref_class"] == "residential"
        assert result["target_class"] == "residential"
        assert result["ref_subclass"] == "urban"
        assert result["target_subclass"] == "urban"
        assert isinstance(result["ref_geometry"], LineString)
        assert isinstance(result["target_geometry"], LineString)

    def test_missing_pair_returns_none(self, tmp_data_dir):
        """get_pair returns None for non-existent pair."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        result = store.get_pair("nonexistent", "also-nonexistent")
        assert result is None

    def test_has_pair(self, tmp_data_dir, sample_pair_data):
        """has_pair correctly detects presence/absence."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        assert not store.has_pair("ref-001", "target-001")

        store.add(**sample_pair_data)
        assert store.has_pair("ref-001", "target-001")
        assert not store.has_pair("ref-002", "target-001")

    def test_duplicate_pair_keeps_latest(self, tmp_data_dir, sample_pair_data):
        """Adding a pair twice keeps only the latest entry."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        store.add(**sample_pair_data)

        # Add again with different name
        updated = {**sample_pair_data, "ref_name": "Updated Street"}
        store.add(**updated)

        result = store.get_pair("ref-001", "target-001")
        assert result["ref_name"] == "Updated Street"

        # Should only have one entry
        mask = (store.gdf["gers_id"] == "ref-001") & (store.gdf["target_id"] == "target-001")
        assert mask.sum() == 1

    def test_none_attributes_handled(self, tmp_data_dir):
        """Adding with None attributes works correctly."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        store.add(
            gers_id="ref-001",
            target_id="target-001",
            ref_geometry=LineString([(0, 0), (1, 1)]),
            target_geometry=LineString([(0, 0.1), (1, 1.1)]),
        )

        result = store.get_pair("ref-001", "target-001")
        assert result is not None
        assert result["ref_name"] is None
        assert result["target_name"] is None
        assert result["ref_class"] is None

    def test_linear_referenced_names(self, tmp_data_dir, sample_pair_data):
        """Linear-referenced name columns are stored as JSON strings."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        lr_data = '[{"start": 0.0, "end": 0.5, "value": "Main St"}, {"start": 0.5, "end": 1.0, "value": "Oak Ave"}]'
        store.add(
            **sample_pair_data,
            ref_names_lr=lr_data,
            target_names_lr=lr_data,
        )

        result = store.get_pair("ref-001", "target-001")
        assert result["ref_names_lr"] == lr_data
        assert result["target_names_lr"] == lr_data


class TestDataStorePersistence:
    def test_save_and_reload(self, tmp_data_dir, sample_pair_data):
        """Persist to disk and reload preserves data."""
        store1 = DataStore("test_dataset", data_dir=tmp_data_dir)
        store1.add(**sample_pair_data)
        store1.save()

        # Reload from disk
        store2 = DataStore("test_dataset", data_dir=tmp_data_dir)
        result = store2.get_pair("ref-001", "target-001")
        assert result is not None
        assert result["ref_name"] == "Main Street"
        assert isinstance(result["ref_geometry"], LineString)
        assert isinstance(result["target_geometry"], LineString)

    def test_atomic_save_with_backup(self, tmp_data_dir, sample_pair_data):
        """Save creates backup of existing file."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        store.add(**sample_pair_data)
        store.save()

        # Modify and save again
        store.add(
            gers_id="ref-002",
            target_id="target-002",
            ref_geometry=LineString([(0, 0), (1, 1)]),
            target_geometry=LineString([(0, 0.1), (1, 1.1)]),
        )
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

    def test_save_does_not_leave_tmp_file(self, tmp_data_dir, sample_pair_data):
        """Temp file is cleaned up after save."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        store.add(**sample_pair_data)
        store.save()

        tmp_path = store.parquet_path.with_suffix(".parquet.tmp")
        assert not tmp_path.exists()

    def test_empty_store_on_missing_file(self, tmp_data_dir):
        """Loading from non-existent file returns empty GeoDataFrame."""
        store = DataStore("nonexistent_dataset", data_dir=tmp_data_dir)
        assert len(store.gdf) == 0

    def test_geometry_fidelity(self, tmp_data_dir):
        """Geometry coordinates preserved through save/load cycle."""
        original_ref = LineString([(0.123456789, 1.987654321), (2.111111111, 3.222222222)])
        original_target = LineString([(0.123456789, 1.987654321), (4.333333333, 5.444444444)])

        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        store.add(
            gers_id="ref-001",
            target_id="target-001",
            ref_geometry=original_ref,
            target_geometry=original_target,
        )
        store.save()

        # Reload and check geometry fidelity
        store2 = DataStore("test_dataset", data_dir=tmp_data_dir)
        result = store2.get_pair("ref-001", "target-001")

        assert result is not None
        ref_coords = list(result["ref_geometry"].coords)
        target_coords = list(result["target_geometry"].coords)

        orig_ref_coords = list(original_ref.coords)
        orig_target_coords = list(original_target.coords)

        # GeoParquet preserves full precision
        for (rx, ry), (ox, oy) in zip(ref_coords, orig_ref_coords):
            assert abs(rx - ox) < 1e-9
            assert abs(ry - oy) < 1e-9

        for (tx, ty), (otx, oty) in zip(target_coords, orig_target_coords):
            assert abs(tx - otx) < 1e-9
            assert abs(ty - oty) < 1e-9


class TestDataStoreLoadAll:
    def test_load_all_multiple_datasets(self, tmp_data_dir, sample_pair_data):
        """load_all combines data from multiple dataset partitions."""
        # Create data for two datasets
        store1 = DataStore("dataset_a", data_dir=tmp_data_dir)
        store1.add(**sample_pair_data)
        store1.save()

        store2 = DataStore("dataset_b", data_dir=tmp_data_dir)
        modified = {**sample_pair_data, "gers_id": "ref-002", "target_id": "target-002"}
        store2.add(**modified)
        store2.save()

        # Load all
        all_data = DataStore.load_all(tmp_data_dir)
        assert len(all_data) == 2
        assert set(all_data["dataset"].unique()) == {"dataset_a", "dataset_b"}

    def test_load_all_empty_dir(self, tmp_data_dir):
        """load_all returns empty GeoDataFrame for non-existent directory."""
        all_data = DataStore.load_all(tmp_data_dir)
        assert len(all_data) == 0
        assert "dataset" in all_data.columns

    def test_load_all_returns_geodataframe(self, tmp_data_dir, sample_pair_data):
        """load_all returns a GeoDataFrame with geometry columns."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        store.add(**sample_pair_data)
        store.save()

        all_data = DataStore.load_all(tmp_data_dir)
        assert isinstance(all_data, gpd.GeoDataFrame)
        assert "ref_geometry" in all_data.columns
        assert "target_geometry" in all_data.columns

    def test_load_all_has_correct_schema(self, tmp_data_dir, sample_pair_data):
        """load_all returns GeoDataFrame with all expected columns."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        store.add(**sample_pair_data)
        store.save()

        all_data = DataStore.load_all(tmp_data_dir)

        # Check key columns
        assert "gers_id" in all_data.columns
        assert "target_id" in all_data.columns
        assert "dataset" in all_data.columns

        # Check attribute columns
        assert "ref_name" in all_data.columns
        assert "target_name" in all_data.columns
        assert "ref_class" in all_data.columns
        assert "target_class" in all_data.columns

        # Check geometry columns
        assert "ref_geometry" in all_data.columns
        assert "target_geometry" in all_data.columns
