"""Tests for GeometryStore - companion CSV for persisting geometries and attributes."""

import pandas as pd
import pytest
from shapely.geometry import LineString

from crosswalk.labeling.geometry_store import GeometryStore


@pytest.fixture
def tmp_geo_dir(tmp_path):
    """Temporary directory for geometry store files."""
    return tmp_path / "label_geometries"


@pytest.fixture
def sample_geometries():
    """Sample geometries and attributes for testing."""
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


class TestGeometryStoreAddAndRetrieve:
    def test_add_and_retrieve_pair(self, tmp_geo_dir, sample_geometries):
        """Round-trip through add + get_pair returns correct data."""
        store = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
        store.add(**sample_geometries)

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

    def test_missing_pair_returns_none(self, tmp_geo_dir):
        """get_pair returns None for non-existent pair."""
        store = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
        result = store.get_pair("nonexistent", "also-nonexistent")
        assert result is None

    def test_has_pair(self, tmp_geo_dir, sample_geometries):
        """has_pair correctly detects presence/absence."""
        store = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
        assert not store.has_pair("ref-001", "target-001")

        store.add(**sample_geometries)
        assert store.has_pair("ref-001", "target-001")
        assert not store.has_pair("ref-002", "target-001")

    def test_duplicate_pair_keeps_latest(self, tmp_geo_dir, sample_geometries):
        """Adding a pair twice keeps only the latest entry."""
        store = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
        store.add(**sample_geometries)

        # Add again with different name
        updated = {**sample_geometries, "ref_name": "Updated Street"}
        store.add(**updated)

        result = store.get_pair("ref-001", "target-001")
        assert result["ref_name"] == "Updated Street"

        # Should only have one entry
        mask = (store.df["gers_id"] == "ref-001") & (store.df["target_id"] == "target-001")
        assert mask.sum() == 1

    def test_none_attributes_handled(self, tmp_geo_dir):
        """Adding with None attributes works correctly."""
        store = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
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


class TestGeometryStorePersistence:
    def test_save_and_reload(self, tmp_geo_dir, sample_geometries):
        """Persist to disk and reload preserves data."""
        store1 = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
        store1.add(**sample_geometries)
        store1.save()

        # Reload from disk
        store2 = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
        result = store2.get_pair("ref-001", "target-001")
        assert result is not None
        assert result["ref_name"] == "Main Street"
        assert isinstance(result["ref_geometry"], LineString)
        assert isinstance(result["target_geometry"], LineString)

    def test_atomic_save_with_backup(self, tmp_geo_dir, sample_geometries):
        """Save creates backup of existing file."""
        store = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
        store.add(**sample_geometries)
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
        backup_path = store.csv_path.with_suffix(".csv.bak")
        assert backup_path.exists()

        # Backup should have only 1 row (original save)
        backup_df = pd.read_csv(backup_path)
        assert len(backup_df) == 1

        # Primary should have 2 rows
        primary_df = pd.read_csv(store.csv_path)
        assert len(primary_df) == 2

    def test_save_does_not_leave_tmp_file(self, tmp_geo_dir, sample_geometries):
        """Temp file is cleaned up after save."""
        store = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
        store.add(**sample_geometries)
        store.save()

        tmp_path = store.csv_path.with_suffix(".csv.tmp")
        assert not tmp_path.exists()

    def test_empty_store_on_missing_file(self, tmp_geo_dir):
        """Loading from non-existent file returns empty DataFrame."""
        store = GeometryStore("nonexistent_dataset", geometries_dir=tmp_geo_dir)
        assert len(store.df) == 0

    def test_wkt_roundtrip(self, tmp_geo_dir):
        """Geometry fidelity through WKT serialization."""
        original_ref = LineString([(0.123456789, 1.987654321), (2.111111111, 3.222222222)])
        original_target = LineString([(0.123456789, 1.987654321), (4.333333333, 5.444444444)])

        store = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
        store.add(
            gers_id="ref-001",
            target_id="target-001",
            ref_geometry=original_ref,
            target_geometry=original_target,
        )
        store.save()

        # Reload and check geometry fidelity
        store2 = GeometryStore("test_dataset", geometries_dir=tmp_geo_dir)
        result = store2.get_pair("ref-001", "target-001")

        assert result is not None
        ref_coords = list(result["ref_geometry"].coords)
        target_coords = list(result["target_geometry"].coords)

        orig_ref_coords = list(original_ref.coords)
        orig_target_coords = list(original_target.coords)

        # WKT has limited precision, but should be very close
        for (rx, ry), (ox, oy) in zip(ref_coords, orig_ref_coords):
            assert abs(rx - ox) < 1e-6
            assert abs(ry - oy) < 1e-6

        for (tx, ty), (otx, oty) in zip(target_coords, orig_target_coords):
            assert abs(tx - otx) < 1e-6
            assert abs(ty - oty) < 1e-6
