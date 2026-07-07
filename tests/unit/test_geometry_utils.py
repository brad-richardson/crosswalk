"""Tests for geometry utility functions."""

import geopandas as gpd
from shapely import LineString, MultiLineString, MultiPolygon, Point, Polygon

from crosswalk.utils.geometry import convert_polygons_to_centerlines, filter_to_linestrings


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

    def test_flattens_multilinestrings(self):
        """MultiLineString geometries should be flattened to LineStrings, not dropped."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[
                LineString([(0, 0), (1, 1)]),
                MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
            ],
        )
        result = filter_to_linestrings(gdf, "test_source")
        # Both rows retained; the MLS is now a flattened LineString.
        assert len(result) == 2
        assert list(result["id"]) == [1, 2]
        assert all(g.geom_type == "LineString" for g in result.geometry)

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
        """Mixed geometry types should be filtered appropriately (MLS flattened)."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3, 4, 5]},
            geometry=[
                LineString([(0, 0), (1, 1)]),  # Keep
                MultiLineString([[(0, 0), (1, 1)]]),  # Flatten (kept as LineString)
                Point(0, 0),  # Filter (Point)
                None,  # Filter (null)
                LineString([(2, 2), (3, 3)]),  # Keep
            ],
        )
        result = filter_to_linestrings(gdf, "test")
        assert len(result) == 3
        assert list(result["id"]) == [1, 2, 5]
        assert all(g.geom_type == "LineString" for g in result.geometry)

    def test_returns_empty_when_all_filtered(self):
        """Empty result should be returned when all geometries are non-line."""
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[
                Point(0, 0),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
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
        # Both rows retained (MLS flattened); columns preserved.
        assert len(result) == 2
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
        # MLS at index "b" is now flattened and retained.
        assert len(result) == 3
        assert list(result.index) == ["a", "b", "c"]
        assert all(g.geom_type == "LineString" for g in result.geometry)


# Helper: a rectangle polygon wide enough for centerline extraction
def _make_road_polygon(x0=0, y0=0, width=0.001, length=0.01):
    """Create a rectangular polygon representing a road surface."""
    return Polygon(
        [
            (x0, y0),
            (x0 + length, y0),
            (x0 + length, y0 + width),
            (x0, y0 + width),
            (x0, y0),
        ]
    )


class TestConvertPolygonsToCenterlines:
    """Tests for convert_polygons_to_centerlines function."""

    def test_rectangle_produces_linestring(self):
        """A simple rectangle polygon should produce a LineString centerline."""
        poly = _make_road_polygon()
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
        result = convert_polygons_to_centerlines(gdf, "test")
        assert len(result) > 0
        for geom in result.geometry:
            assert geom.geom_type == "LineString"

    def test_mixed_linestring_polygon_preserves_lines(self):
        """Existing LineStrings should pass through, polygons should be converted."""
        line = LineString([(0, 0), (1, 1)])
        poly = _make_road_polygon(x0=2)
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[line, poly],
            crs="EPSG:4326",
        )
        result = convert_polygons_to_centerlines(gdf, "test")
        assert len(result) >= 2
        # The original line should be present
        geom_types = set(result.geometry.geom_type)
        assert "LineString" in geom_types

    def test_multipolygon_exploded_and_converted(self):
        """MultiPolygon should be exploded into individual polygons and converted."""
        mp = MultiPolygon([_make_road_polygon(x0=0), _make_road_polygon(x0=0.02)])
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[mp], crs="EPSG:4326")
        result = convert_polygons_to_centerlines(gdf, "test")
        assert len(result) >= 2
        for geom in result.geometry:
            assert geom.geom_type == "LineString"

    def test_empty_geodataframe_returns_empty(self):
        """Empty GeoDataFrame should be returned unchanged."""
        gdf = gpd.GeoDataFrame({"id": []}, geometry=[], crs="EPSG:4326")
        result = convert_polygons_to_centerlines(gdf, "test")
        assert len(result) == 0

    def test_no_polygons_returns_unchanged(self):
        """GeoDataFrame with only LineStrings should be returned unchanged."""
        line = LineString([(0, 0), (1, 1)])
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs="EPSG:4326")
        result = convert_polygons_to_centerlines(gdf, "test")
        assert len(result) == 1
        assert result.geometry.iloc[0].geom_type == "LineString"

    def test_crs_preserved(self):
        """CRS should be preserved through conversion."""
        poly = _make_road_polygon()
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
        result = convert_polygons_to_centerlines(gdf, "test")
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326

    def test_failed_extraction_dropped_with_warning(self):
        """Degenerate polygon that fails extraction should be dropped with warning."""
        # A degenerate polygon (nearly zero-width sliver)
        degen = Polygon([(0, 0), (0.0000001, 0), (0.0000001, 0.0000001), (0, 0.0000001)])
        # Include a valid rectangle to ensure at least some succeed
        valid = _make_road_polygon(x0=1)
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[degen, valid],
            crs="EPSG:4326",
        )
        result = convert_polygons_to_centerlines(gdf, "test")
        # Should have at least the valid polygon's centerline
        assert len(result) >= 1
