"""Tests for the LineString-only boundary in crosswalk.utils.geometry."""

import io

import geopandas as gpd
import pytest
import shapely
from loguru import logger
from shapely.geometry import (
    LineString,
    MultiLineString,
    Point,
    Polygon,
)

from crosswalk.utils.geometry import filter_to_linestrings, flatten_to_linestring


@pytest.fixture
def log_capture():
    """Capture loguru output for assertion."""
    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}", level="INFO")
    yield sink
    logger.remove(handler_id)


class TestFlattenToLineString:
    def test_merges_contiguous_parts(self):
        # Two parts that share an endpoint merge into a single LineString.
        mls = MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])
        result = flatten_to_linestring(mls)
        assert result.geom_type == "LineString"
        # Merged line spans the full extent.
        assert result.length == 2.0
        assert set(result.coords) >= {(0.0, 0.0), (2.0, 0.0)}

    def test_takes_longest_disjoint_part(self):
        # Disjoint parts can't merge; the longest part is returned.
        short = [(0, 0), (1, 0)]  # length 1
        long = [(10, 0), (10, 5)]  # length 5
        mls = MultiLineString([short, long])
        result = flatten_to_linestring(mls)
        assert result.geom_type == "LineString"
        assert result.length == 5.0
        assert set(result.coords) == {(10.0, 0.0), (10.0, 5.0)}

    def test_passes_linestring_through_unchanged(self):
        ls = LineString([(0, 0), (1, 1)])
        result = flatten_to_linestring(ls)
        assert result is ls

    def test_passes_none_through_unchanged(self):
        assert flatten_to_linestring(None) is None

    def test_passes_empty_through_unchanged(self):
        empty = shapely.from_wkt("MULTILINESTRING EMPTY")
        result = flatten_to_linestring(empty)
        assert result.is_empty

    def test_passes_point_through_unchanged(self):
        pt = Point(0, 0)
        assert flatten_to_linestring(pt) is pt


class TestFilterToLineStrings:
    def test_retains_multipart_mls_as_flattened_linestring(self):
        # A multi-part (disjoint) MLS row is now retained, flattened to a LineString.
        rows = [
            LineString([(0, 0), (1, 0)]),
            MultiLineString([[(0, 0), (1, 0)], [(10, 0), (10, 5)]]),
        ]
        gdf = gpd.GeoDataFrame(geometry=rows, crs="EPSG:4326")
        result = filter_to_linestrings(gdf, source_name="test")
        # Row count preserved (nothing dropped).
        assert len(result) == 2
        # Both rows are now LineStrings.
        assert set(result.geometry.geom_type) == {"LineString"}

    def test_retains_contiguous_mls_merged(self):
        rows = [MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])]
        gdf = gpd.GeoDataFrame(geometry=rows, crs="EPSG:4326")
        result = filter_to_linestrings(gdf, source_name="test")
        assert len(result) == 1
        assert result.geometry.iloc[0].geom_type == "LineString"
        assert result.geometry.iloc[0].length == 2.0

    def test_drops_points_and_polygons(self):
        rows = [
            LineString([(0, 0), (1, 0)]),
            Point(5, 5),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
        ]
        gdf = gpd.GeoDataFrame(geometry=rows, crs="EPSG:4326")
        result = filter_to_linestrings(gdf, source_name="test")
        assert len(result) == 1
        assert result.geometry.iloc[0].geom_type == "LineString"

    def test_drops_null_and_empty(self):
        rows = [
            LineString([(0, 0), (1, 0)]),
            None,
            shapely.from_wkt("LINESTRING EMPTY"),
        ]
        gdf = gpd.GeoDataFrame(geometry=rows, crs="EPSG:4326")
        result = filter_to_linestrings(gdf, source_name="test")
        assert len(result) == 1
        assert result.geometry.iloc[0].geom_type == "LineString"

    def test_empty_gdf_returns_empty(self):
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        result = filter_to_linestrings(gdf, source_name="test")
        assert result.empty

    def test_preserves_attributes_on_flattened_rows(self):
        rows = [
            MultiLineString([[(0, 0), (1, 0)], [(10, 0), (10, 5)]]),
            LineString([(0, 0), (1, 1)]),
        ]
        gdf = gpd.GeoDataFrame({"name": ["a", "b"]}, geometry=rows, crs="EPSG:4326")
        result = filter_to_linestrings(gdf, source_name="test")
        assert len(result) == 2
        assert list(result["name"]) == ["a", "b"]

    def test_logged_counts_are_correct_for_mixed_input(self, log_capture):
        """Regression test: an empty LineString has geom_type == "LineString",
        so it satisfies BOTH line_mask (counted as a "line") and null_mask
        (counted as null/empty). The old ``other_count = original_count -
        line_mask.sum() - null_count`` formula subtracted it twice, silently
        undercounting (or zeroing out) the logged "other" (Points/Polygons)
        warning even though those geometries were still correctly dropped from
        the returned GeoDataFrame.

        Mix: 1 valid LineString, 1 empty LineString, 1 disjoint MultiLineString
        (flattens to a LineString via the longest-part fallback), 1 empty
        MultiLineString, 1 Polygon.
        """
        rows = [
            LineString([(0, 0), (1, 0)]),  # valid line -> kept
            shapely.from_wkt("LINESTRING EMPTY"),  # empty line -> null/empty
            MultiLineString([[(0, 0), (1, 0)], [(10, 0), (10, 5)]]),  # disjoint -> flattened
            shapely.from_wkt("MULTILINESTRING EMPTY"),  # empty MLS -> null/empty
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),  # non-line -> "other"
        ]
        gdf = gpd.GeoDataFrame(geometry=rows, crs="EPSG:4326")
        result = filter_to_linestrings(gdf, source_name="test")

        # Behavior: valid line + flattened MLS survive; the rest are dropped.
        assert len(result) == 2
        assert set(result.geometry.geom_type) == {"LineString"}

        log_text = log_capture.getvalue()
        assert "Flattened 1 MultiLineString geometries to LineStrings in test" in log_text
        assert "Filtered 2 null/empty geometries from test (2/5 features)" in log_text
        # The Polygon must be reported as a dropped "other" geometry (count 1),
        # not silently swallowed by the empty-LineString double-subtraction bug.
        assert "Filtered 1 non-LineString geometries from test (1/5 features)" in log_text
