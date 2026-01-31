"""Tests for screen test implementations."""

import geopandas as gpd
from shapely.geometry import LineString, Polygon

from matcher.screen import MatchContext, ScreenOutcome, get_registered_tests
from matcher.screen.tests.building_test import BuildingTest
from matcher.screen.tests.water_body_test import WaterBodyTest


class TestWaterBodyTest:
    def test_road_not_in_water_passes(self):
        test = WaterBodyTest()
        water = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
        test.water_gdf = gpd.GeoDataFrame(geometry=[water], crs="EPSG:4326")
        test.water_union = water
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
        assert result.outcome == ScreenOutcome.PASS

    def test_road_entirely_in_water_fails(self):
        test = WaterBodyTest()
        water = Polygon([(-1, -1), (-1, 2), (2, 2), (2, -1)])
        test.water_gdf = gpd.GeoDataFrame(geometry=[water], crs="EPSG:4326")
        test.water_union = water
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
        assert result.outcome == ScreenOutcome.FAIL
        assert "water" in result.reason.lower()

    def test_short_bridge_passes(self):
        test = WaterBodyTest()
        # Water body that the road crosses briefly
        water = Polygon([(0.00475, -0.001), (0.00475, 0.001), (0.00525, 0.001), (0.00525, -0.001)])
        test.water_gdf = gpd.GeoDataFrame(geometry=[water], crs="EPSG:4326")
        test.water_union = water
        test._metric_crs = "EPSG:32618"

        ctx = MatchContext(
            match_id="1",
            ref_id="ref_1",
            target_id="target_1",
            ref_geom=LineString([(0, 0), (0.01, 0)]),
            target_geom=LineString([(0, 0), (0.01, 0)]),
            confidence=0.95,
        )

        result = test.test_match(ctx)
        assert result.outcome in [ScreenOutcome.PASS, ScreenOutcome.WARN]

    def test_no_water_skips(self):
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
        assert result.outcome == ScreenOutcome.SKIP


class TestBuildingTest:
    def test_road_not_through_building_passes(self):
        test = BuildingTest()
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
        assert result.outcome == ScreenOutcome.PASS

    def test_road_through_building_fails(self):
        test = BuildingTest()
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
        assert result.outcome == ScreenOutcome.FAIL
        assert "building" in result.reason.lower()

    def test_no_buildings_skips(self):
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
        assert result.outcome == ScreenOutcome.SKIP


class TestRegistration:
    def test_water_test_registered(self):
        tests = get_registered_tests()
        assert "water_body" in tests

    def test_building_test_registered(self):
        tests = get_registered_tests()
        assert "building" in tests
