"""Tests for screen context fetchers."""

from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

from crosswalk.screen.context import (
    fetch_overture_buildings,
    fetch_overture_landcover,
    fetch_overture_water,
    get_building_union,
    get_landcover_union,
    get_water_union,
)


class TestFetchOvertureWater:
    @patch("crosswalk.screen.context.overture_polygons.geodataframe")
    @patch("crosswalk.screen.context.overture_polygons.get_latest_release")
    def test_fetch_returns_polygons(self, mock_release, mock_geodataframe):
        mock_release.return_value = "2024-01-01"
        water1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        water2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        mock_geodataframe.return_value = gpd.GeoDataFrame(
            {"id": [1, 2]}, geometry=[water1, water2], crs="EPSG:4326"
        )

        result = fetch_overture_water((-1, -1, 4, 4))

        assert len(result) == 2
        assert all(result.geometry.geom_type.isin(["Polygon", "MultiPolygon"]))

    @patch("crosswalk.screen.context.overture_polygons.geodataframe")
    @patch("crosswalk.screen.context.overture_polygons.get_latest_release")
    def test_fetch_filters_small_water_bodies(self, mock_release, mock_geodataframe):
        mock_release.return_value = "2024-01-01"
        large = Polygon([(0, 0), (0, 0.01), (0.01, 0.01), (0.01, 0)])
        small = Polygon([(1, 1), (1, 1.000001), (1.000001, 1.000001), (1.000001, 1)])
        mock_geodataframe.return_value = gpd.GeoDataFrame(
            {"id": [1, 2]}, geometry=[large, small], crs="EPSG:4326"
        )

        result = fetch_overture_water((-1, -1, 4, 4), min_area_m2=100.0)
        assert len(result) == 1

    @patch("crosswalk.screen.context.overture_polygons.geodataframe")
    @patch("crosswalk.screen.context.overture_polygons.get_latest_release")
    def test_fetch_empty_returns_empty_gdf(self, mock_release, mock_geodataframe):
        mock_release.return_value = "2024-01-01"
        mock_geodataframe.return_value = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        result = fetch_overture_water((-1, -1, 4, 4))
        assert len(result) == 0
        assert result.crs == "EPSG:4326"


class TestGetWaterUnion:
    def test_union_multiple_polygons(self):
        water1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        water2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        gdf = gpd.GeoDataFrame(geometry=[water1, water2], crs="EPSG:4326")

        union = get_water_union(gdf)
        assert union is not None
        assert isinstance(union, (Polygon, MultiPolygon))

    def test_union_empty_returns_none(self):
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        assert get_water_union(gdf) is None


class TestFetchOvertureBuildings:
    @patch("crosswalk.screen.context.overture_polygons.geodataframe")
    @patch("crosswalk.screen.context.overture_polygons.get_latest_release")
    def test_fetch_returns_polygons(self, mock_release, mock_geodataframe):
        mock_release.return_value = "2024-01-01"
        b1 = Polygon([(0, 0), (0, 0.001), (0.001, 0.001), (0.001, 0)])
        b2 = Polygon([(0.002, 0.002), (0.002, 0.003), (0.003, 0.003), (0.003, 0.002)])
        mock_geodataframe.return_value = gpd.GeoDataFrame(
            {"id": [1, 2]}, geometry=[b1, b2], crs="EPSG:4326"
        )

        result = fetch_overture_buildings((-1, -1, 4, 4))
        assert len(result) == 2

    @patch("crosswalk.screen.context.overture_polygons.geodataframe")
    @patch("crosswalk.screen.context.overture_polygons.get_latest_release")
    def test_fetch_filters_small_buildings(self, mock_release, mock_geodataframe):
        mock_release.return_value = "2024-01-01"
        large = Polygon([(0, 0), (0, 0.001), (0.001, 0.001), (0.001, 0)])
        small = Polygon([(1, 1), (1, 1.000001), (1.000001, 1.000001), (1.000001, 1)])
        mock_geodataframe.return_value = gpd.GeoDataFrame(
            {"id": [1, 2]}, geometry=[large, small], crs="EPSG:4326"
        )

        result = fetch_overture_buildings((-1, -1, 4, 4), min_area_m2=20.0)
        assert len(result) == 1

    @patch("crosswalk.screen.context.overture_polygons.geodataframe")
    @patch("crosswalk.screen.context.overture_polygons.get_latest_release")
    def test_fetch_empty_returns_empty_gdf(self, mock_release, mock_geodataframe):
        mock_release.return_value = "2024-01-01"
        mock_geodataframe.return_value = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        result = fetch_overture_buildings((-1, -1, 4, 4))
        assert len(result) == 0


class TestGetBuildingUnion:
    def test_union_multiple_buildings(self):
        b1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        b2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        gdf = gpd.GeoDataFrame(geometry=[b1, b2], crs="EPSG:4326")

        union = get_building_union(gdf)
        assert union is not None
        assert isinstance(union, (Polygon, MultiPolygon))

    def test_union_empty_returns_none(self):
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        assert get_building_union(gdf) is None


class TestFetchOvertureLandcover:
    @patch("crosswalk.screen.context.overture_polygons.geodataframe")
    @patch("crosswalk.screen.context.overture_polygons.get_latest_release")
    def test_fetch_returns_polygons(self, mock_release, mock_geodataframe):
        mock_release.return_value = "2024-01-01"
        wetland = Polygon([(0, 0), (0, 0.01), (0.01, 0.01), (0.01, 0)])
        pitch = Polygon([(0.02, 0.02), (0.02, 0.03), (0.03, 0.03), (0.03, 0.02)])
        mock_geodataframe.return_value = gpd.GeoDataFrame(
            {"id": [1, 2], "subtype": ["wetland", "pitch"]},
            geometry=[wetland, pitch],
            crs="EPSG:4326",
        )

        result = fetch_overture_landcover((-1, -1, 4, 4))
        assert len(result) == 2

    @patch("crosswalk.screen.context.overture_polygons.geodataframe")
    @patch("crosswalk.screen.context.overture_polygons.get_latest_release")
    def test_fetch_filters_by_subtype(self, mock_release, mock_geodataframe):
        mock_release.return_value = "2024-01-01"
        wetland = Polygon([(0, 0), (0, 0.01), (0.01, 0.01), (0.01, 0)])
        park = Polygon([(0.02, 0.02), (0.02, 0.03), (0.03, 0.03), (0.03, 0.02)])
        mock_geodataframe.return_value = gpd.GeoDataFrame(
            {"id": [1, 2], "subtype": ["wetland", "park"]},
            geometry=[wetland, park],
            crs="EPSG:4326",
        )

        result = fetch_overture_landcover((-1, -1, 4, 4))
        # Only wetland should be included (park is not in RESTRICTED_SUBTYPES)
        assert len(result) == 1

    @patch("crosswalk.screen.context.overture_polygons.geodataframe")
    @patch("crosswalk.screen.context.overture_polygons.get_latest_release")
    def test_fetch_empty_returns_empty_gdf(self, mock_release, mock_geodataframe):
        mock_release.return_value = "2024-01-01"
        mock_geodataframe.return_value = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        result = fetch_overture_landcover((-1, -1, 4, 4))
        assert len(result) == 0


class TestGetLandcoverUnion:
    def test_union_multiple_polygons(self):
        p1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        p2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        gdf = gpd.GeoDataFrame(geometry=[p1, p2], crs="EPSG:4326")

        union = get_landcover_union(gdf)
        assert union is not None
        assert isinstance(union, (Polygon, MultiPolygon))

    def test_union_empty_returns_none(self):
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        assert get_landcover_union(gdf) is None
