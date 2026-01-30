"""Tests for falsification test implementations."""

from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from matcher.falsification import FalsificationOutcome, MatchContext
from matcher.falsification.tests.building_test import BuildingTest
from matcher.falsification.tests.water_body_test import WaterBodyTest


class TestWaterBodyTest:
    """Tests for water body falsification test."""

    def test_road_not_in_water_passes(self):
        """Test that a road not intersecting water passes."""
        test = WaterBodyTest()

        # Create water body away from road
        water = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
        test.water_gdf = gpd.GeoDataFrame(geometry=[water], crs="EPSG:4326")
        test.water_union = water
        test._metric_crs = "EPSG:32618"

        # Road that doesn't intersect water
        ctx = MatchContext(
            match_id="1",
            ref_id="ref_1",
            target_id="target_1",
            ref_geom=LineString([(0, 0), (1, 1)]),
            target_geom=LineString([(0, 0), (1, 1)]),
            confidence=0.95,
        )

        result = test.test_match(ctx)
        assert result.outcome == FalsificationOutcome.PASS

    def test_road_entirely_in_water_fails(self):
        """Test that a road entirely in water fails."""
        test = WaterBodyTest()

        # Large water body containing the road
        water = Polygon([(-1, -1), (-1, 2), (2, 2), (2, -1)])
        test.water_gdf = gpd.GeoDataFrame(geometry=[water], crs="EPSG:4326")
        test.water_union = water
        test._metric_crs = "EPSG:32618"

        # Road entirely within water
        ctx = MatchContext(
            match_id="1",
            ref_id="ref_1",
            target_id="target_1",
            ref_geom=LineString([(0, 0), (1, 1)]),
            target_geom=LineString([(0, 0), (1, 1)]),
            confidence=0.95,
        )

        result = test.test_match(ctx)
        assert result.outcome == FalsificationOutcome.FAIL
        assert "water" in result.reason.lower()

    def test_short_bridge_passes(self):
        """Test that a short bridge crossing passes (within thresholds)."""
        test = WaterBodyTest()

        # Water body that the road crosses briefly (5% of road length)
        # Road is 0.01 degrees long (~1km), water is 0.0005 degrees wide (~50m)
        water = Polygon([(0.00475, -0.001), (0.00475, 0.001), (0.00525, 0.001), (0.00525, -0.001)])
        test.water_gdf = gpd.GeoDataFrame(geometry=[water], crs="EPSG:4326")
        test.water_union = water
        test._metric_crs = "EPSG:32618"

        # Road crossing water briefly (< 10% of length)
        ctx = MatchContext(
            match_id="1",
            ref_id="ref_1",
            target_id="target_1",
            ref_geom=LineString([(0, 0), (0.01, 0)]),
            target_geom=LineString([(0, 0), (0.01, 0)]),
            confidence=0.95,
        )

        result = test.test_match(ctx)
        # Should pass because intersection is < 10% of road length
        assert result.outcome in [FalsificationOutcome.PASS, FalsificationOutcome.WARN]

    def test_no_water_skips(self):
        """Test that no water data results in SKIP."""
        test = WaterBodyTest()
        test.water_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        test.water_union = None

        ctx = MatchContext(
            match_id="1",
            ref_id="ref_1",
            target_id="target_1",
            ref_geom=LineString([(0, 0), (1, 1)]),
            target_geom=LineString([(0, 0), (1, 1)]),
            confidence=0.95,
        )

        result = test.test_match(ctx)
        assert result.outcome == FalsificationOutcome.SKIP


class TestBuildingTest:
    """Tests for building footprint falsification test."""

    def test_road_not_through_building_passes(self):
        """Test that a road not intersecting buildings passes."""
        test = BuildingTest()

        # Building away from road
        building = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
        test.building_gdf = gpd.GeoDataFrame(geometry=[building], crs="EPSG:4326")
        test.building_union = building
        test._metric_crs = "EPSG:32618"

        ctx = MatchContext(
            match_id="1",
            ref_id="ref_1",
            target_id="target_1",
            ref_geom=LineString([(0, 0), (1, 1)]),
            target_geom=LineString([(0, 0), (1, 1)]),
            confidence=0.95,
        )

        result = test.test_match(ctx)
        assert result.outcome == FalsificationOutcome.PASS

    def test_road_through_building_fails(self):
        """Test that a road going through a large building fails."""
        test = BuildingTest()

        # Large building that the road passes through entirely
        building = Polygon([(-1, -1), (-1, 2), (2, 2), (2, -1)])
        test.building_gdf = gpd.GeoDataFrame(geometry=[building], crs="EPSG:4326")
        test.building_union = building
        test._metric_crs = "EPSG:32618"

        ctx = MatchContext(
            match_id="1",
            ref_id="ref_1",
            target_id="target_1",
            ref_geom=LineString([(0, 0), (1, 1)]),
            target_geom=LineString([(0, 0), (1, 1)]),
            confidence=0.95,
        )

        result = test.test_match(ctx)
        assert result.outcome == FalsificationOutcome.FAIL
        assert "building" in result.reason.lower()

    def test_no_buildings_skips(self):
        """Test that no building data results in SKIP."""
        test = BuildingTest()
        test.building_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        test.building_union = None

        ctx = MatchContext(
            match_id="1",
            ref_id="ref_1",
            target_id="target_1",
            ref_geom=LineString([(0, 0), (1, 1)]),
            target_geom=LineString([(0, 0), (1, 1)]),
            confidence=0.95,
        )

        result = test.test_match(ctx)
        assert result.outcome == FalsificationOutcome.SKIP

    def test_building_test_registered(self):
        """Test that building test is registered."""
        from matcher.falsification import get_registered_tests

        tests = get_registered_tests()
        assert "building" in tests

    def test_water_test_registered(self):
        """Test that water body test is registered."""
        from matcher.falsification import get_registered_tests

        tests = get_registered_tests()
        assert "water_body" in tests
