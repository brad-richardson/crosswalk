"""Tests for falsification context fetchers."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from matcher.falsification.context.overture_buildings import (
    fetch_overture_buildings,
    get_building_union,
)
from matcher.falsification.context.overture_water import (
    fetch_overture_water,
    get_water_union,
)


class TestFetchOvertureWater:
    """Tests for water body fetching."""

    @patch("matcher.falsification.context.overture_water.geodataframe")
    @patch("matcher.falsification.context.overture_water.get_latest_release")
    def test_fetch_returns_polygons(self, mock_release, mock_geodataframe):
        """Test that fetch returns polygon geometries."""
        mock_release.return_value = "2024-01-01"

        # Create mock water body data
        water1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        water2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        mock_gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[water1, water2],
            crs="EPSG:4326",
        )
        mock_geodataframe.return_value = mock_gdf

        result = fetch_overture_water((-1, -1, 4, 4))

        assert len(result) == 2
        assert all(result.geometry.geom_type.isin(["Polygon", "MultiPolygon"]))

    @patch("matcher.falsification.context.overture_water.geodataframe")
    @patch("matcher.falsification.context.overture_water.get_latest_release")
    def test_fetch_filters_small_water_bodies(self, mock_release, mock_geodataframe):
        """Test that small water bodies are filtered out."""
        mock_release.return_value = "2024-01-01"

        # Create one large and one small water body (in degrees, ~100m2 is tiny)
        large_water = Polygon([(0, 0), (0, 0.01), (0.01, 0.01), (0.01, 0)])  # ~1km2
        small_water = Polygon([(1, 1), (1, 1.000001), (1.000001, 1.000001), (1.000001, 1)])
        mock_gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[large_water, small_water],
            crs="EPSG:4326",
        )
        mock_geodataframe.return_value = mock_gdf

        result = fetch_overture_water((-1, -1, 4, 4), min_area_m2=100.0)

        # Small one should be filtered
        assert len(result) == 1

    @patch("matcher.falsification.context.overture_water.geodataframe")
    @patch("matcher.falsification.context.overture_water.get_latest_release")
    def test_fetch_empty_returns_empty_gdf(self, mock_release, mock_geodataframe):
        """Test that empty results return empty GeoDataFrame."""
        mock_release.return_value = "2024-01-01"
        mock_geodataframe.return_value = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        result = fetch_overture_water((-1, -1, 4, 4))

        assert len(result) == 0
        assert result.crs == "EPSG:4326"


class TestGetWaterUnion:
    """Tests for water union helper."""

    def test_union_multiple_polygons(self):
        """Test unioning multiple water polygons."""
        water1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        water2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        gdf = gpd.GeoDataFrame(geometry=[water1, water2], crs="EPSG:4326")

        union = get_water_union(gdf)

        assert union is not None
        assert isinstance(union, (Polygon, MultiPolygon))

    def test_union_empty_returns_none(self):
        """Test that empty GDF returns None."""
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        assert get_water_union(gdf) is None


class TestFetchOvertureBuildings:
    """Tests for building footprint fetching."""

    @patch("matcher.falsification.context.overture_buildings.geodataframe")
    @patch("matcher.falsification.context.overture_buildings.get_latest_release")
    def test_fetch_returns_polygons(self, mock_release, mock_geodataframe):
        """Test that fetch returns polygon geometries."""
        mock_release.return_value = "2024-01-01"

        building1 = Polygon([(0, 0), (0, 0.001), (0.001, 0.001), (0.001, 0)])
        building2 = Polygon([(0.002, 0.002), (0.002, 0.003), (0.003, 0.003), (0.003, 0.002)])
        mock_gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[building1, building2],
            crs="EPSG:4326",
        )
        mock_geodataframe.return_value = mock_gdf

        result = fetch_overture_buildings((-1, -1, 4, 4))

        assert len(result) == 2
        assert all(result.geometry.geom_type.isin(["Polygon", "MultiPolygon"]))

    @patch("matcher.falsification.context.overture_buildings.geodataframe")
    @patch("matcher.falsification.context.overture_buildings.get_latest_release")
    def test_fetch_empty_returns_empty_gdf(self, mock_release, mock_geodataframe):
        """Test that empty results return empty GeoDataFrame."""
        mock_release.return_value = "2024-01-01"
        mock_geodataframe.return_value = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        result = fetch_overture_buildings((-1, -1, 4, 4))

        assert len(result) == 0


class TestGetBuildingUnion:
    """Tests for building union helper."""

    def test_union_multiple_buildings(self):
        """Test unioning multiple building polygons."""
        b1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        b2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        gdf = gpd.GeoDataFrame(geometry=[b1, b2], crs="EPSG:4326")

        union = get_building_union(gdf)

        assert union is not None
        assert isinstance(union, (Polygon, MultiPolygon))

    def test_union_empty_returns_none(self):
        """Test that empty GDF returns None."""
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        assert get_building_union(gdf) is None
