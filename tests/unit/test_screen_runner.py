"""Tests for screen runner."""

import geopandas as gpd
import pytest

from matcher.screen import ScreenReport
from matcher.screen.runner import (
    _get_bridge_ref_column,
    _get_bridge_target_column,
    _get_id_column,
)


class TestIdColumnDetection:
    """Tests for ID column detection helpers."""

    def test_get_id_column_finds_id(self):
        gdf = gpd.GeoDataFrame({"id": [1, 2], "name": ["a", "b"]})
        assert _get_id_column(gdf, "test") == "id"

    def test_get_id_column_finds_ID(self):
        gdf = gpd.GeoDataFrame({"ID": [1, 2], "name": ["a", "b"]})
        assert _get_id_column(gdf, "test") == "ID"

    def test_get_id_column_raises_if_not_found(self):
        gdf = gpd.GeoDataFrame({"name": ["a", "b"]})
        with pytest.raises(ValueError, match="Could not determine ID column"):
            _get_id_column(gdf, "test")

    def test_get_bridge_ref_column(self):
        gdf = gpd.GeoDataFrame({"ref_id": [1], "target_id": [2]})
        assert _get_bridge_ref_column(gdf) == "ref_id"

    def test_get_bridge_target_column(self):
        gdf = gpd.GeoDataFrame({"ref_id": [1], "target_id": [2]})
        assert _get_bridge_target_column(gdf) == "target_id"


class TestScreenReport:
    """Tests for ScreenReport."""

    def test_fail_rate(self):
        report = ScreenReport(
            total_matches=100,
            passed=90,
            failed=5,
            warned=3,
            skipped=2,
        )
        assert report.fail_rate == 0.05

    def test_warn_rate(self):
        report = ScreenReport(
            total_matches=100,
            passed=90,
            failed=5,
            warned=3,
            skipped=2,
        )
        assert report.warn_rate == 0.03

    def test_rates_with_zero_matches(self):
        report = ScreenReport(
            total_matches=0,
            passed=0,
            failed=0,
            warned=0,
            skipped=0,
        )
        assert report.fail_rate == 0.0
        assert report.warn_rate == 0.0

    def test_to_dict(self):
        report = ScreenReport(
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
