"""Tests for screen test implementations."""

import geopandas as gpd
from shapely.geometry import LineString, Polygon

from crosswalk.screen import CandidateContext, ScreenOutcome
from crosswalk.screen.tests.building_test import BuildingTest
from crosswalk.screen.tests.landcover_test import LandcoverTest
from crosswalk.screen.tests.travel_mode import get_travel_mode
from crosswalk.screen.tests.water_body_test import WaterBodyTest


def _buffer_polygon(poly: Polygon, buffer_m: float, crs: str = "EPSG:32618") -> Polygon:
    """Helper to buffer a polygon in metric CRS and return in EPSG:4326."""
    series = gpd.GeoSeries([poly], crs="EPSG:4326")
    metric = series.to_crs(crs)
    buffered = metric.buffer(buffer_m)
    return buffered.to_crs("EPSG:4326").iloc[0]


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
        test.polygon_gdf = gpd.GeoDataFrame(geometry=[water], crs="EPSG:4326")
        test.polygon_union = water
        # Pre-compute buffered geometries
        for mode, buffer_m in test.buffers.items():
            test._buffered[mode] = _buffer_polygon(water, buffer_m)

        ctx = CandidateContext(
            target_id="target_1",
            target_geom=LineString([(0, 0), (1, 1)]),
            road_class="residential",
        )

        result = test.test_candidate(ctx)
        assert result.outcome == ScreenOutcome.PASS

    def test_road_in_water_fails(self):
        test = WaterBodyTest()
        water = Polygon([(-1, -1), (-1, 2), (2, 2), (2, -1)])
        test.polygon_gdf = gpd.GeoDataFrame(geometry=[water], crs="EPSG:4326")
        test.polygon_union = water
        for mode, buffer_m in test.buffers.items():
            test._buffered[mode] = _buffer_polygon(water, buffer_m)

        ctx = CandidateContext(
            target_id="target_1",
            target_geom=LineString([(0, 0), (1, 1)]),
            road_class="residential",
        )

        result = test.test_candidate(ctx)
        assert result.outcome == ScreenOutcome.FAIL
        assert "water" in result.reason.lower()

    def test_no_water_skips(self):
        test = WaterBodyTest()
        test.polygon_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        test.polygon_union = None

        ctx = CandidateContext(
            target_id="target_1",
            target_geom=LineString([(0, 0), (1, 1)]),
        )

        result = test.test_candidate(ctx)
        assert result.outcome == ScreenOutcome.SKIP

    def test_buffer_varies_by_road_class(self):
        test = WaterBodyTest()
        assert test.buffers["vehicle"] > test.buffers["pedestrian"]


class TestBuildingTest:
    def test_road_not_through_building_passes(self):
        test = BuildingTest()
        building = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
        test.polygon_gdf = gpd.GeoDataFrame(geometry=[building], crs="EPSG:4326")
        test.polygon_union = building
        for mode, buffer_m in test.buffers.items():
            test._buffered[mode] = _buffer_polygon(building, buffer_m)

        ctx = CandidateContext(
            target_id="target_1",
            target_geom=LineString([(0, 0), (1, 1)]),
            road_class="residential",
        )

        result = test.test_candidate(ctx)
        assert result.outcome == ScreenOutcome.PASS

    def test_road_through_building_fails(self):
        test = BuildingTest()
        building = Polygon([(-1, -1), (-1, 2), (2, 2), (2, -1)])
        test.polygon_gdf = gpd.GeoDataFrame(geometry=[building], crs="EPSG:4326")
        test.polygon_union = building
        for mode, buffer_m in test.buffers.items():
            test._buffered[mode] = _buffer_polygon(building, buffer_m)

        ctx = CandidateContext(
            target_id="target_1",
            target_geom=LineString([(0, 0), (1, 1)]),
            road_class="residential",
        )

        result = test.test_candidate(ctx)
        assert result.outcome == ScreenOutcome.FAIL
        assert "building" in result.reason.lower()

    def test_no_buildings_skips(self):
        test = BuildingTest()
        test.polygon_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        test.polygon_union = None

        ctx = CandidateContext(
            target_id="target_1",
            target_geom=LineString([(0, 0), (1, 1)]),
        )

        result = test.test_candidate(ctx)
        assert result.outcome == ScreenOutcome.SKIP

    def test_buffer_varies_by_road_class(self):
        test = BuildingTest()
        assert test.buffers["vehicle"] > test.buffers["pedestrian"]


class TestLandcoverTest:
    def test_road_not_in_landcover_passes(self):
        test = LandcoverTest()
        wetland = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
        test.polygon_gdf = gpd.GeoDataFrame(geometry=[wetland], crs="EPSG:4326")
        test.polygon_union = wetland
        for mode, buffer_m in test.buffers.items():
            test._buffered[mode] = _buffer_polygon(wetland, buffer_m)

        ctx = CandidateContext(
            target_id="target_1",
            target_geom=LineString([(0, 0), (1, 1)]),
            road_class="residential",
        )

        result = test.test_candidate(ctx)
        assert result.outcome == ScreenOutcome.PASS

    def test_road_through_wetland_fails(self):
        test = LandcoverTest()
        wetland = Polygon([(-1, -1), (-1, 2), (2, 2), (2, -1)])
        test.polygon_gdf = gpd.GeoDataFrame(geometry=[wetland], crs="EPSG:4326")
        test.polygon_union = wetland
        for mode, buffer_m in test.buffers.items():
            test._buffered[mode] = _buffer_polygon(wetland, buffer_m)

        ctx = CandidateContext(
            target_id="target_1",
            target_geom=LineString([(0, 0), (1, 1)]),
            road_class="residential",
        )

        result = test.test_candidate(ctx)
        assert result.outcome == ScreenOutcome.FAIL
        assert "landcover" in result.reason.lower()

    def test_no_landcover_skips(self):
        test = LandcoverTest()
        test.polygon_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        test.polygon_union = None

        ctx = CandidateContext(
            target_id="target_1",
            target_geom=LineString([(0, 0), (1, 1)]),
        )

        result = test.test_candidate(ctx)
        assert result.outcome == ScreenOutcome.SKIP

    def test_buffer_varies_by_road_class(self):
        test = LandcoverTest()
        assert test.buffers["vehicle"] > test.buffers["pedestrian"]


class TestRegistration:
    """Test that screen test classes have correct name attributes for registration."""

    def test_water_test_has_name(self):
        assert WaterBodyTest.name == "water_body"

    def test_building_test_has_name(self):
        assert BuildingTest.name == "building"

    def test_landcover_test_has_name(self):
        assert LandcoverTest.name == "landcover"
