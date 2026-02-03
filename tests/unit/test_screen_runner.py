"""Tests for screen runner."""

import geopandas as gpd

from matcher.screen import ScreenReport
from matcher.screen.runner import _get_bridge_target_column
from matcher.utils.dataframe import find_id_column


class TestIdColumnDetection:
    """Tests for ID column detection helpers."""

    def test_find_id_column_finds_id(self):
        gdf = gpd.GeoDataFrame({"id": [1, 2], "name": ["a", "b"]})
        assert find_id_column(gdf) == "id"

    def test_find_id_column_finds_ID(self):
        gdf = gpd.GeoDataFrame({"ID": [1, 2], "name": ["a", "b"]})
        assert find_id_column(gdf) == "ID"

    def test_find_id_column_raises_when_no_id_column(self):
        import pytest

        gdf = gpd.GeoDataFrame({"name": ["a", "b"]})
        # Raises by default when no ID column found
        with pytest.raises(ValueError, match="Could not find ID column"):
            find_id_column(gdf)

    def test_find_id_column_returns_none_when_no_raise(self):
        gdf = gpd.GeoDataFrame({"name": ["a", "b"]})
        # Returns None when raise_on_missing=False
        assert find_id_column(gdf, raise_on_missing=False) is None

    def test_find_id_column_falls_back_when_requested(self):
        gdf = gpd.GeoDataFrame({"name": ["a", "b"]})
        # Falls back to first column with warning when explicitly requested
        assert find_id_column(gdf, fallback=True) == "name"

    def test_get_bridge_target_column(self):
        gdf = gpd.GeoDataFrame({"ref_id": [1], "target_id": [2]})
        assert _get_bridge_target_column(gdf) == "target_id"

    def test_get_bridge_target_column_local_id(self):
        gdf = gpd.GeoDataFrame({"ref_id": [1], "local_id": [2]})
        assert _get_bridge_target_column(gdf) == "local_id"


class TestScreenReport:
    """Tests for ScreenReport."""

    def test_fail_rate(self):
        report = ScreenReport(
            total_candidates=100,
            passed=90,
            failed=5,
            warned=3,
            skipped=2,
        )
        assert report.fail_rate == 0.05

    def test_warn_rate(self):
        report = ScreenReport(
            total_candidates=100,
            passed=90,
            failed=5,
            warned=3,
            skipped=2,
        )
        assert report.warn_rate == 0.03

    def test_pass_rate(self):
        report = ScreenReport(
            total_candidates=100,
            passed=90,
            failed=5,
            warned=3,
            skipped=2,
        )
        assert report.pass_rate == 0.9

    def test_rates_with_zero_candidates(self):
        report = ScreenReport(
            total_candidates=0,
            passed=0,
            failed=0,
            warned=0,
            skipped=0,
        )
        assert report.fail_rate == 0.0
        assert report.warn_rate == 0.0
        assert report.pass_rate == 0.0

    def test_to_dict(self):
        report = ScreenReport(
            total_candidates=100,
            passed=90,
            failed=5,
            warned=3,
            skipped=2,
            test_results={"water_body": {"pass": 90, "fail": 5, "warn": 3, "skip": 2}},
        )
        d = report.to_dict()

        assert d["total_candidates"] == 100
        assert d["passed"] == 90
        assert d["failed"] == 5
        assert d["pass_rate"] == 0.9
        assert d["fail_rate"] == 0.05
        assert "water_body" in d["test_results"]
