"""Tests for relational feature extraction."""

import pytest
import numpy as np
import geopandas as gpd
from shapely import LineString

from matcher.features.relational import (
    compute_perpendicular_offset,
    compute_side_of_street,
    compute_parallel_alignment,
    compute_endpoint_proximity,
)
from matcher.features.spatial_context import (
    AnchorRoadMatcher,
    SpatialContextIndex,
    compute_endpoint_features,
)


class TestRelationalFeatures:
    """Tests for core relational feature functions."""

    @pytest.mark.parametrize(
        "line_b,expected_alignment",
        [
            (LineString([(0, 10), (100, 10)]), 1.0),  # Parallel
            (LineString([(100, 10), (0, 10)]), 1.0),  # Parallel, reversed
            (LineString([(50, 0), (50, 100)]), 0.0),  # Perpendicular
            (LineString([(0, 0), (100, 100)]), 0.5),  # 45 degrees
        ],
    )
    def test_parallel_alignment(self, line_b, expected_alignment):
        line_a = LineString([(0, 0), (100, 0)])
        assert compute_parallel_alignment(line_a, line_b) == pytest.approx(
            expected_alignment, abs=0.1
        )

    @pytest.mark.parametrize(
        "sidewalk,expected_side",
        [
            (LineString([(0, 3), (100, 3)]), "left"),  # North of east-going road
            (LineString([(0, -3), (100, -3)]), "right"),  # South of east-going road
        ],
    )
    def test_side_of_street(self, sidewalk, expected_side):
        road = LineString([(0, 0), (100, 0)])
        side, confidence = compute_side_of_street(sidewalk, road)
        assert side == expected_side
        assert confidence > 0.8

    def test_perpendicular_offset(self):
        road = LineString([(0, 0), (100, 0)])
        sidewalk = LineString([(0, 3), (100, 3)])
        offset, std = compute_perpendicular_offset(sidewalk, road)
        assert offset == pytest.approx(3.0, abs=0.1)
        assert std < 0.5  # Consistent parallel offset

    def test_endpoint_proximity(self):
        target = LineString([(0, 0), (100, 0)])
        other_endpoints = np.array([[0, 1], [200, 0]])  # One near start, one far
        start_prox, end_prox, shared = compute_endpoint_proximity(
            target, other_endpoints, tolerance=5.0
        )
        assert start_prox == pytest.approx(1.0, abs=0.1)
        assert end_prox > 50
        assert shared >= 1


class TestSpatialContextIndex:
    """Tests for spatial context indexing - the primary utility for endpoint features."""

    @pytest.fixture
    def segments(self):
        return gpd.GeoDataFrame(
            {
                "id": ["s1", "s2", "s3"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (200, 0)]),  # Connected to s1 at (100,0)
                    LineString([(0, 50), (100, 50)]),  # Not connected
                ],
            },
            crs="EPSG:32618",
        )

    def test_infer_connectivity(self, segments):
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(segments, id_column="id")
        connected = ctx.infer_connectivity(0, tolerance=5.0)
        assert 1 in connected  # s2 shares endpoint
        assert 2 not in connected  # s3 is far

    def test_endpoint_features(self, segments):
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(segments, id_column="id")
        features = compute_endpoint_features(
            segments.iloc[0].geometry, ctx, exclude_segment_idx=0, tolerance=5.0
        )
        assert features["end_endpoint_proximity"] < 5.0
        assert features["shared_endpoint_count"] >= 1


class TestAnchorRoadMatcher:
    """Tests for anchor road matching."""

    def test_find_anchor_road(self):
        roads = gpd.GeoDataFrame(
            {
                "id": ["road1", "road2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 50), (100, 50)]),
                ],
            },
            crs="EPSG:32618",
        )
        matcher = AnchorRoadMatcher(roads, max_offset=10.0, id_column="id")

        match = matcher.find_anchor_road(LineString([(0, 3), (100, 3)]))
        assert match is not None
        assert match.anchor_id == "road1"
        assert match.perpendicular_offset == pytest.approx(3.0, abs=0.5)

    def test_no_anchor_when_too_far(self):
        roads = gpd.GeoDataFrame(
            {"id": ["road1"], "geometry": [LineString([(0, 0), (100, 0)])]},
            crs="EPSG:32618",
        )
        matcher = AnchorRoadMatcher(roads, max_offset=5.0, id_column="id")
        assert matcher.find_anchor_road(LineString([(0, 20), (100, 20)])) is None
