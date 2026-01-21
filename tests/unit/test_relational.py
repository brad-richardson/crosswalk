"""Tests for relational feature extraction."""

import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString

from matcher.features.relational import (
    compute_endpoint_proximity,
    compute_parallel_alignment,
    compute_perpendicular_offset,
    compute_side_of_street,
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
            target, other_endpoints, tolerance_m=5.0
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
        connected = ctx.infer_connectivity(0, tolerance_m=5.0)
        assert 1 in connected  # s2 shares endpoint
        assert 2 not in connected  # s3 is far

    def test_endpoint_features(self, segments):
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(segments, id_column="id")
        features = compute_endpoint_features(
            segments.iloc[0].geometry, ctx, exclude_segment_idx=0, tolerance_m=5.0
        )
        assert features["end_endpoint_proximity_m"] < 5.0
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


class TestInferEndpointDegree:
    """Tests for topology inference from endpoint proximity."""

    @pytest.fixture
    def intersection_network(self):
        """Create a simple network with a 4-way intersection at (100, 100)."""
        return gpd.GeoDataFrame(
            {
                "id": ["n", "s", "e", "w"],
                "geometry": [
                    LineString([(100, 100), (100, 200)]),  # North
                    LineString([(100, 0), (100, 100)]),  # South
                    LineString([(100, 100), (200, 100)]),  # East
                    LineString([(0, 100), (100, 100)]),  # West
                ],
            },
            crs="EPSG:32618",
        )

    @pytest.fixture
    def dead_end_network(self):
        """Create a network with dead ends connecting at endpoints."""
        return gpd.GeoDataFrame(
            {
                "id": ["main", "dead1", "dead2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),  # Main road
                    LineString([(100, 0), (100, 50)]),  # Dead end from main's end
                    LineString([(200, 0), (250, 0)]),  # Isolated (both ends dead)
                ],
            },
            crs="EPSG:32618",
        )

    def test_infer_degree_at_intersection(self, intersection_network):
        """Segments meeting at intersection should have high degree at that endpoint."""
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            infer_endpoint_degree,
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(intersection_network, id_column="id")

        # North segment: start at intersection (100,100), end at (100,200)
        north_geom = intersection_network.iloc[0].geometry
        start_deg, end_deg = infer_endpoint_degree(north_geom, ctx, tolerance_m=5.0)

        # Start is at 4-way intersection, end is dead end
        assert start_deg == 4
        assert end_deg == 1

    def test_infer_degree_dead_end(self, dead_end_network):
        """Dead end segments should have degree 1 at dead end."""
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            infer_endpoint_degree,
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(dead_end_network, id_column="id")

        # Dead end segment
        dead_geom = dead_end_network.iloc[1].geometry
        start_deg, end_deg = infer_endpoint_degree(dead_geom, ctx, tolerance_m=5.0)

        # Start connects to main road (exactly 2: dead end + main road), end is dead end
        assert start_deg == 2  # Connects to main road endpoint
        assert end_deg == 1  # Dead end

    def test_isolated_segment_degree_one(self, dead_end_network):
        """Isolated segment should have degree 1 at both ends."""
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            infer_endpoint_degree,
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(dead_end_network, id_column="id")

        # Isolated segment (id="dead2")
        isolated_geom = dead_end_network.iloc[2].geometry
        start_deg, end_deg = infer_endpoint_degree(isolated_geom, ctx, tolerance_m=5.0)

        # Both ends are isolated
        assert start_deg == 1
        assert end_deg == 1


class TestTopologyFeatures:
    """Tests for topology feature computation."""

    @pytest.fixture
    def intersection_network(self):
        """4-way intersection at (100, 100)."""
        return gpd.GeoDataFrame(
            {
                "id": ["n", "s", "e", "w"],
                "geometry": [
                    LineString([(100, 100), (100, 200)]),
                    LineString([(100, 0), (100, 100)]),
                    LineString([(100, 100), (200, 100)]),
                    LineString([(0, 100), (100, 100)]),
                ],
            },
            crs="EPSG:32618",
        )

    def test_compute_topology_features(self, intersection_network):
        """Verify topology features are computed correctly."""
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            compute_topology_features,
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(intersection_network, id_column="id")

        # North segment: intersection at start, dead end at end
        north_geom = intersection_network.iloc[0].geometry
        features = compute_topology_features(north_geom, ctx, tolerance_m=5.0)

        assert features["from_degree"] == 4
        assert features["to_degree"] == 1
        assert features["is_dead_end"] is True  # Has degree 1 endpoint
        assert features["is_intersection"] is True  # Has degree > 2 endpoint
        assert features["degree_signature"] == (1, 4)

    def test_degree_match_score_identical(self):
        """Identical degrees should have perfect match score."""
        from matcher.features.spatial_context import compute_degree_match_score

        score = compute_degree_match_score(4, 1, 4, 1)
        assert score == pytest.approx(1.0)

    def test_degree_match_score_swapped(self):
        """Swapped endpoints should still have perfect match score."""
        from matcher.features.spatial_context import compute_degree_match_score

        score = compute_degree_match_score(4, 1, 1, 4)  # Endpoints reversed
        assert score == pytest.approx(1.0)

    def test_degree_match_score_different(self):
        """Different degrees should have lower match score."""
        from matcher.features.spatial_context import compute_degree_match_score

        score = compute_degree_match_score(4, 1, 2, 2)
        assert score < 1.0
        assert score > 0.0

    def test_degree_signature_similarity_identical(self):
        """Identical signatures should have similarity 1.0."""
        from matcher.features.spatial_context import compute_degree_signature_similarity

        sim = compute_degree_signature_similarity((1, 4), (1, 4))
        assert sim == pytest.approx(1.0)

    def test_degree_signature_similarity_different(self):
        """Different signatures should have lower similarity."""
        from matcher.features.spatial_context import compute_degree_signature_similarity

        sim = compute_degree_signature_similarity((1, 4), (2, 2))
        assert sim < 1.0

    def test_degree_signature_similarity_empty(self):
        """Empty signatures should return 0."""
        from matcher.features.spatial_context import compute_degree_signature_similarity

        assert compute_degree_signature_similarity((), (1, 2)) == 0.0
        assert compute_degree_signature_similarity((1, 2), ()) == 0.0
