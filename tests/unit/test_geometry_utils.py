"""Tests for geometry utility functions."""

import geopandas as gpd
from shapely import LineString, MultiLineString, Point, Polygon

from matcher.utils.geometry import filter_to_linestrings


class TestFilterToLinestrings:
    """Tests for filter_to_linestrings function."""

    def test_preserves_linestrings(self):
        """LineString geometries should be preserved."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
                LineString([(2, 2), (3, 3)]),
            ],
        )
        result = filter_to_linestrings(gdf, "test")
        assert len(result) == 3
        assert all(g.geom_type == "LineString" for g in result.geometry)

    def test_filters_multilinestrings(self):
        """MultiLineString geometries should be filtered."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[
                LineString([(0, 0), (1, 1)]),
                MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
            ],
        )
        result = filter_to_linestrings(gdf, "test_source")
        assert len(result) == 1
        assert result.iloc[0]["id"] == 1

    def test_filters_other_geometry_types(self):
        """Non-LineString geometries (Point, Polygon) should be filtered."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[
                LineString([(0, 0), (1, 1)]),
                Point(0, 0),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            ],
        )
        result = filter_to_linestrings(gdf, "test")
        assert len(result) == 1
        assert result.iloc[0]["id"] == 1

    def test_handles_empty_geodataframe(self):
        """Empty GeoDataFrame should be returned unchanged."""
        gdf = gpd.GeoDataFrame({"id": []}, geometry=[])
        result = filter_to_linestrings(gdf, "test")
        assert len(result) == 0

    def test_handles_null_geometries(self):
        """Null geometries should be filtered."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[
                LineString([(0, 0), (1, 1)]),
                None,
                LineString([(2, 2), (3, 3)]),
            ],
        )
        result = filter_to_linestrings(gdf, "test")
        assert len(result) == 2
        assert list(result["id"]) == [1, 3]

    def test_handles_mixed_geometry_types(self):
        """Mixed geometry types should be filtered appropriately."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3, 4, 5]},
            geometry=[
                LineString([(0, 0), (1, 1)]),  # Keep
                MultiLineString([[(0, 0), (1, 1)]]),  # Filter (MultiLineString)
                Point(0, 0),  # Filter (Point)
                None,  # Filter (null)
                LineString([(2, 2), (3, 3)]),  # Keep
            ],
        )
        result = filter_to_linestrings(gdf, "test")
        assert len(result) == 2
        assert list(result["id"]) == [1, 5]

    def test_returns_empty_when_all_filtered(self):
        """Empty result should be returned when all geometries are filtered."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[
                Point(0, 0),
                MultiLineString([[(0, 0), (1, 1)]]),
            ],
        )
        result = filter_to_linestrings(gdf, "test")
        assert len(result) == 0

    def test_preserves_columns(self):
        """All columns should be preserved in filtered output."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2], "name": ["a", "b"], "value": [10, 20]},
            geometry=[
                LineString([(0, 0), (1, 1)]),
                MultiLineString([[(0, 0), (1, 1)]]),
            ],
        )
        result = filter_to_linestrings(gdf, "test")
        assert len(result) == 1
        assert list(result.columns) == ["id", "name", "value", "geometry"]
        assert result.iloc[0]["name"] == "a"
        assert result.iloc[0]["value"] == 10

    def test_preserves_crs(self):
        """CRS should be preserved in filtered output."""
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:4326",
        )
        result = filter_to_linestrings(gdf, "test")
        assert result.crs == gdf.crs

    def test_preserves_index(self):
        """Original DataFrame index should be preserved."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[
                LineString([(0, 0), (1, 1)]),
                MultiLineString([[(0, 0), (1, 1)]]),
                LineString([(2, 2), (3, 3)]),
            ],
            index=["a", "b", "c"],
        )
        result = filter_to_linestrings(gdf, "test")
        assert len(result) == 2
        assert list(result.index) == ["a", "c"]
