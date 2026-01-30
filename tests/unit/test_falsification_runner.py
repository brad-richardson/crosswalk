"""Tests for falsification runner."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from matcher.falsification import FalsificationReport
from matcher.falsification.runner import (
    _build_report,
    _get_bridge_ref_column,
    _get_bridge_target_column,
    _get_id_column,
)


class TestIdColumnDetection:
    """Tests for ID column detection helpers."""

    def test_get_id_column_finds_id(self):
        """Test finding 'id' column."""
        gdf = gpd.GeoDataFrame({"id": [1, 2], "name": ["a", "b"]})
        assert _get_id_column(gdf, "test") == "id"

    def test_get_id_column_finds_ID(self):
        """Test finding 'ID' column."""
        gdf = gpd.GeoDataFrame({"ID": [1, 2], "name": ["a", "b"]})
        assert _get_id_column(gdf, "test") == "ID"

    def test_get_id_column_raises_if_not_found(self):
        """Test error when no ID column found."""
        gdf = gpd.GeoDataFrame({"name": ["a", "b"]})
        with pytest.raises(ValueError, match="Could not determine ID column"):
            _get_id_column(gdf, "test")

    def test_get_bridge_ref_column(self):
        """Test finding ref_id in bridge file."""
        gdf = gpd.GeoDataFrame({"ref_id": [1], "target_id": [2]})
        assert _get_bridge_ref_column(gdf) == "ref_id"

    def test_get_bridge_target_column(self):
        """Test finding target_id in bridge file."""
        gdf = gpd.GeoDataFrame({"ref_id": [1], "target_id": [2]})
        assert _get_bridge_target_column(gdf) == "target_id"


class TestFalsificationReport:
    """Tests for FalsificationReport."""

    def test_fail_rate(self):
        """Test fail rate calculation."""
        report = FalsificationReport(
            total_matches=100,
            passed=90,
            failed=5,
            warned=3,
            skipped=2,
        )
        assert report.fail_rate == 0.05

    def test_warn_rate(self):
        """Test warn rate calculation."""
        report = FalsificationReport(
            total_matches=100,
            passed=90,
            failed=5,
            warned=3,
            skipped=2,
        )
        assert report.warn_rate == 0.03

    def test_rates_with_zero_matches(self):
        """Test rates when no matches."""
        report = FalsificationReport(
            total_matches=0,
            passed=0,
            failed=0,
            warned=0,
            skipped=0,
        )
        assert report.fail_rate == 0.0
        assert report.warn_rate == 0.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = FalsificationReport(
            total_matches=100,
            passed=90,
            failed=5,
            warned=3,
            skipped=2,
            test_results={"water_body": {"pass": 90, "fail": 5, "warn": 3, "skip": 2}},
        )
        d = report.to_dict()

        assert d["total_matches"] == 100
        assert d["passed"] == 90
        assert d["failed"] == 5
        assert d["fail_rate"] == 0.05
        assert "water_body" in d["test_results"]
