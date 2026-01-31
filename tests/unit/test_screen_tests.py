"""Tests for screen test implementations."""

import geopandas as gpd
from shapely.geometry import LineString, Polygon

from matcher.screen import MatchContext, ScreenOutcome, get_registered_tests
from matcher.screen.tests.building_test import BuildingTest
from matcher.screen.tests.landcover_test import LandcoverTest
from matcher.screen.tests.travel_mode import get_travel_mode
from matcher.screen.tests.water_body_test import WaterBodyTest


class TestTravelMode:
    def test_vehicle_classes(self):
        assert get_travel_mode("motorway") == "vehicle"
        assert get_travel_mode("residential") == "vehicle"
        assert get_travel_mode("service") == "vehicle"

    def test_bicycle_classes(self):
        assert get_travel_mode("cycleway") == "bicycle"

    def test_pedestrian_classes(self):
        assert get_travel_mode("footway") == "pedestrian"
        assert get_travel_mode("path") == "pedestrian"
        assert get_travel_mode("steps") == "pedestrian"

    def test_none_defaults_to_vehicle(self):
        assert get_travel_mode(None) == "vehicle"

    def test_case_insensitive(self):
        assert get_travel_mode("MOTORWAY") == "vehicle"
        assert get_travel_mode("Footway") == "pedestrian"


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
            road_class="residential",
        )

        result = test.test_match(ctx)
        assert result.outcome == ScreenOutcome.PASS

    def test_road_in_water_fails(self):
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
            road_class="residential",
        )

        result = test.test_match(ctx)
        assert result.outcome == ScreenOutcome.FAIL
        assert "water" in result.reason.lower()

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

    def test_buffer_varies_by_road_class(self):
        test = WaterBodyTest()
        # Check that buffers are set correctly
        assert test.buffers["vehicle"] > test.buffers["pedestrian"]


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
            road_class="residential",
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
            road_class="residential",
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

    def test_buffer_varies_by_road_class(self):
        test = BuildingTest()
        assert test.buffers["vehicle"] > test.buffers["pedestrian"]


class TestLandcoverTest:
    def test_road_not_in_landcover_passes(self):
        test = LandcoverTest()
        wetland = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
        test.landcover_gdf = gpd.GeoDataFrame(geometry=[wetland], crs="EPSG:4326")
        test.landcover_union = wetland
        test._metric_crs = "EPSG:32618"

        ctx = MatchContext(
            match_id="1",
            ref_id="ref_1",
            target_id="target_1",
            ref_geom=LineString([(0, 0), (1, 1)]),
            target_geom=LineString([(0, 0), (1, 1)]),
            confidence=0.95,
            road_class="residential",
        )

        result = test.test_match(ctx)
        assert result.outcome == ScreenOutcome.PASS

    def test_road_through_wetland_fails(self):
        test = LandcoverTest()
        wetland = Polygon([(-1, -1), (-1, 2), (2, 2), (2, -1)])
        test.landcover_gdf = gpd.GeoDataFrame(geometry=[wetland], crs="EPSG:4326")
        test.landcover_union = wetland
        test._metric_crs = "EPSG:32618"

        ctx = MatchContext(
            match_id="1",
            ref_id="ref_1",
            target_id="target_1",
            ref_geom=LineString([(0, 0), (1, 1)]),
            target_geom=LineString([(0, 0), (1, 1)]),
            confidence=0.95,
            road_class="residential",
        )

        result = test.test_match(ctx)
        assert result.outcome == ScreenOutcome.FAIL
        assert "landcover" in result.reason.lower()

    def test_no_landcover_skips(self):
        test = LandcoverTest()
        test.landcover_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        test.landcover_union = None

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

    def test_buffer_varies_by_road_class(self):
        test = LandcoverTest()
        assert test.buffers["vehicle"] > test.buffers["pedestrian"]


class TestRegistration:
    def test_water_test_registered(self):
        tests = get_registered_tests()
        assert "water_body" in tests

    def test_building_test_registered(self):
        tests = get_registered_tests()
        assert "building" in tests

    def test_landcover_test_registered(self):
        tests = get_registered_tests()
        assert "landcover" in tests
