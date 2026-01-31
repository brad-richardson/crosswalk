"""Tests for quality fingerprint module."""

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from matcher.quality import (
    QualityFingerprint,
    compute_quality_metrics,
    save_quality_report,
)
from matcher.quality.report import compare_fingerprints, load_quality_report


class TestQualityFingerprint:
    """Tests for QualityFingerprint dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        fp = QualityFingerprint(
            dataset_name="test",
            total_segments=100,
            total_length_m=5000.5,
            name_coverage_ratio=0.85,
        )

        d = fp.to_dict()

        assert d["dataset_name"] == "test"
        assert d["total_segments"] == 100
        assert d["total_length_m"] == 5000.5
        assert d["name_coverage_ratio"] == 0.85

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "dataset_name": "test",
            "timestamp": "2025-01-01T12:00:00+00:00",
            "total_segments": 50,
            "total_length_m": 2500.0,
            "island_count": 2,
        }

        fp = QualityFingerprint.from_dict(data)

        assert fp.dataset_name == "test"
        assert fp.total_segments == 50
        assert fp.total_length_m == 2500.0
        assert fp.island_count == 2

    def test_roundtrip(self):
        """Test dict -> fingerprint -> dict roundtrip."""
        original = QualityFingerprint(
            dataset_name="roundtrip_test",
            total_segments=200,
            total_length_m=10000.0,
            vertex_density_mean=0.05,
            vertex_density_std=0.02,
            name_coverage_ratio=0.75,
            class_distribution={"residential": 150, "tertiary": 50},
        )

        d = original.to_dict()
        restored = QualityFingerprint.from_dict(d)

        assert restored.dataset_name == original.dataset_name
        assert restored.total_segments == original.total_segments
        assert restored.name_coverage_ratio == original.name_coverage_ratio


class TestComputeQualityMetrics:
    """Tests for compute_quality_metrics function."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        edges = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        fp = compute_quality_metrics(edges, "empty_test")

        assert fp.dataset_name == "empty_test"
        assert fp.total_segments == 0
        assert fp.total_length_m == 0.0

    def test_basic_metrics(self):
        """Test basic metric computation."""
        edges = gpd.GeoDataFrame(
            {
                "id": [1, 2, 3],
                "names": ["Main St", "Oak Ave", None],  # Standardized column name
            },
            geometry=[
                LineString([(0, 0), (0.001, 0)]),
                LineString([(0.001, 0), (0.001, 0.001)]),
                LineString([(0.002, 0), (0.003, 0)]),
            ],
            crs="EPSG:4326",
        )

        fp = compute_quality_metrics(edges, "basic_test")

        assert fp.dataset_name == "basic_test"
        assert fp.total_segments == 3
        assert fp.total_length_m > 0
        # 2 out of 3 edges have names
        assert 0.6 < fp.name_coverage_ratio < 0.7

    def test_invalid_geometry_count(self):
        """Test counting invalid geometries."""
        edges = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[
                LineString([(0, 0), (1, 0)]),
                None,  # Invalid
                LineString([(2, 0), (3, 0)]),
            ],
            crs="EPSG:4326",
        )

        fp = compute_quality_metrics(edges, "invalid_test")

        assert fp.invalid_geometry_count == 1


class TestQualityReport:
    """Tests for quality report I/O."""

    def test_save_and_load(self):
        """Test saving and loading a quality report."""
        fp = QualityFingerprint(
            dataset_name="io_test",
            total_segments=100,
            name_coverage_ratio=0.9,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "quality.json"
            save_quality_report(fp, path)

            loaded = load_quality_report(path)

            assert loaded.dataset_name == "io_test"
            assert loaded.total_segments == 100
            assert loaded.name_coverage_ratio == 0.9

    def test_compare_fingerprints(self):
        """Test comparing two fingerprints."""
        fp1 = QualityFingerprint(
            dataset_name="before",
            total_segments=100,
            total_length_m=5000.0,
            name_coverage_ratio=0.8,
            island_count=5,
        )

        fp2 = QualityFingerprint(
            dataset_name="after",
            total_segments=120,
            total_length_m=6000.0,
            name_coverage_ratio=0.85,
            island_count=3,
        )

        comparison = compare_fingerprints(fp1, fp2)

        assert comparison["segment_count_delta"] == 20
        assert comparison["length_delta_m"] == 1000.0
        assert comparison["name_coverage_delta"] == pytest.approx(0.05)
        assert comparison["island_count_delta"] == -2
