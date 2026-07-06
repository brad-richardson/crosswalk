"""Tests for DataStore - GeoParquet storage for pair geometries and attributes."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from crosswalk.features.semantic import display_name
from crosswalk.labeling.data_store import DataStore


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
        "ref_names": {"primary": "Main Street"},
        "target_names": {"primary": "Main St"},
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
        assert result["ref_names"]["primary"] == "Main Street"
        assert result["target_names"]["primary"] == "Main St"
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
        updated = {**sample_pair_data, "ref_names": {"primary": "Updated Street"}}
        store.add(**updated)

        result = store.get_pair("ref-001", "target-001")
        assert result["ref_names"]["primary"] == "Updated Street"

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
        assert result["ref_names"] is None
        assert result["target_names"] is None
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

    def test_topo_sampled_round_trip(self, tmp_data_dir, sample_pair_data):
        """target_topo_sampled survives add/save/load/get_pair cycle."""
        from crosswalk.labeling.data_store import reconstruct_topo_connectors_from_sampled

        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        sampled = [(0.0, 3), (0.25, 1), (0.5, 4), (1.0, 2)]
        store.add(**sample_pair_data, target_topo_sampled=sampled)
        store.save()

        # Reload from disk
        store2 = DataStore("test_dataset", data_dir=tmp_data_dir)
        result = store2.get_pair("ref-001", "target-001")
        assert result is not None
        assert result["target_topo_sampled"] == sampled

        # Reconstruction produces valid connectors
        connectors, node_features = reconstruct_topo_connectors_from_sampled(
            result["target_topo_sampled"]
        )
        assert len(connectors) == 4
        assert all(isinstance(nid, int) for _, nid in connectors)
        fracs = [f for f, _ in connectors]
        assert fracs == [0.0, 0.25, 0.5, 1.0]
        # Each connector's node_id should map to correct degree
        for (_frac, node_id), (_orig_frac, orig_degree) in zip(connectors, sampled):
            assert node_features[node_id] == orig_degree

    def test_topo_sampled_none_by_default(self, tmp_data_dir, sample_pair_data):
        """target_topo_sampled is None when not provided."""
        store = DataStore("test_dataset", data_dir=tmp_data_dir)
        store.add(**sample_pair_data)
        store.save()

        store2 = DataStore("test_dataset", data_dir=tmp_data_dir)
        result = store2.get_pair("ref-001", "target-001")
        assert result["target_topo_sampled"] is None

    def test_reconstructed_node_ids_globally_unique(self, tmp_data_dir):
        """Node IDs from separate reconstruct calls don't collide."""
        from crosswalk.labeling.data_store import reconstruct_topo_connectors_from_sampled

        sampled_a = [(0.0, 3), (1.0, 2)]
        sampled_b = [(0.0, 1), (1.0, 4)]

        _, feats_a = reconstruct_topo_connectors_from_sampled(sampled_a)
        _, feats_b = reconstruct_topo_connectors_from_sampled(sampled_b)

        # Node IDs should not overlap
        assert set(feats_a.keys()).isdisjoint(set(feats_b.keys()))

        # Merged dict should preserve both sets of degrees
        merged = {**feats_a, **feats_b}
        assert len(merged) == 4


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
        assert result["ref_names"]["primary"] == "Main Street"
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
        assert "ref_names" in all_data.columns
        assert "target_names" in all_data.columns
        assert "ref_class" in all_data.columns
        assert "target_class" in all_data.columns

        # Check geometry columns
        assert "ref_geometry" in all_data.columns
        assert "target_geometry" in all_data.columns


class TestDisplayName:
    """Tests for display_name() helper."""

    def test_english_common_preferred(self):
        """English common name is preferred over primary."""
        names = {
            "primary": "海榮路 Hoi Wing Road",
            "common": {"en": "Hoi Wing Road", "zh": "海榮路"},
        }
        assert display_name(names) == "Hoi Wing Road"

    def test_primary_fallback(self):
        """Falls back to primary when no common English name."""
        names = {"primary": "Main Street"}
        assert display_name(names) == "Main Street"

    def test_none_input(self):
        """Returns None for None input."""
        assert display_name(None) is None

    def test_non_dict_input(self):
        """Returns string for non-dict input."""
        assert display_name("Some Street") == "Some Street"

    def test_empty_dict(self):
        """Returns None for empty dict (no primary)."""
        assert display_name({}) is None

    def test_common_without_english(self):
        """Falls back to primary when common has no English."""
        names = {"primary": "海榮路", "common": {"zh": "海榮路"}}
        assert display_name(names) == "海榮路"

    def test_falsy_non_dict(self):
        """Returns None for falsy non-dict input."""
        assert display_name("") is None
        assert display_name(0) is None
