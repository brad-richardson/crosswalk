"""Tests for spatial context indexing and topology computation."""

import geopandas as gpd
from shapely import LineString

from matcher.features.spatial_context import (
    SpatialContextIndex,
    UnionFind,
    compute_all_topology,
    compute_topology_features,
)


class TestUnionFind:
    """Tests for the Union-Find data structure."""

    def test_init(self):
        """Union-Find should initialize with n elements."""
        uf = UnionFind(5)
        assert len(uf.parent) == 5
        assert len(uf.rank) == 5

    def test_find_self(self):
        """Each element is initially its own root."""
        uf = UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_basic(self):
        """Union should merge two sets."""
        uf = UnionFind(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_union_chain(self):
        """Union should work transitively."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_union_disjoint_sets(self):
        """Separate unions create disjoint sets."""
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)

        # 0, 1, 2 should be in one set
        assert uf.find(0) == uf.find(1) == uf.find(2)

        # 3, 4 should be in another set
        assert uf.find(3) == uf.find(4)

        # 5 should be alone
        assert uf.find(5) != uf.find(0)
        assert uf.find(5) != uf.find(3)


class TestComputeAllTopology:
    """Tests for the compute_all_topology function."""

    def test_empty_dataframe(self):
        """Empty GeoDataFrame should return empty dict."""
        gdf = gpd.GeoDataFrame({"id": [], "geometry": []}, crs="EPSG:32610")
        result = compute_all_topology(gdf)
        assert result == {}

    def test_single_segment(self):
        """Single segment has degree 1 at both endpoints (dead end)."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1"],
                "geometry": [LineString([(0, 0), (100, 0)])],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology(gdf)

        assert "seg1" in result
        assert result["seg1"]["from_degree"] == 1
        assert result["seg1"]["to_degree"] == 1
        assert result["seg1"]["is_dead_end"] is True
        assert result["seg1"]["is_intersection"] is False

    def test_two_connected_segments(self):
        """Two segments sharing an endpoint should have degree 2 at that point."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),  # Ends at (100, 0)
                    LineString([(100, 0), (100, 100)]),  # Starts at (100, 0)
                ],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology(gdf, tolerance=5.0)

        # seg1's end and seg2's start are connected
        assert result["seg1"]["to_degree"] == 2
        assert result["seg2"]["from_degree"] == 2

        # Other endpoints are dead ends
        assert result["seg1"]["from_degree"] == 1
        assert result["seg2"]["to_degree"] == 1

    def test_t_junction_with_shared_endpoint(self):
        """T-junction with shared endpoint should have correct degrees."""
        # Main road split into two segments with a shared endpoint
        gdf = gpd.GeoDataFrame(
            {
                "id": ["main_left", "main_right", "side"],
                "geometry": [
                    LineString([(0, 0), (50, 0)]),  # Left segment
                    LineString([(50, 0), (100, 0)]),  # Right segment
                    LineString([(50, 50), (50, 0)]),  # Side street
                ],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology(gdf, tolerance=5.0)

        # Junction at (50, 0) should have degree 3
        assert result["main_left"]["to_degree"] == 3
        assert result["main_right"]["from_degree"] == 3
        assert result["side"]["to_degree"] == 3

        # Outer endpoints are dead ends
        assert result["main_left"]["from_degree"] == 1
        assert result["main_right"]["to_degree"] == 1
        assert result["side"]["from_degree"] == 1

        # The junction point is an intersection
        assert result["main_left"]["is_intersection"] is True
        assert result["main_right"]["is_intersection"] is True
        assert result["side"]["is_intersection"] is True

    def test_cross_intersection(self):
        """4-way intersection should have degree 4."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["north", "south", "east", "west"],
                "geometry": [
                    LineString([(50, 50), (50, 100)]),  # North
                    LineString([(50, 50), (50, 0)]),  # South
                    LineString([(50, 50), (100, 50)]),  # East
                    LineString([(50, 50), (0, 50)]),  # West
                ],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology(gdf, tolerance=5.0)

        # All segments share the center point (50, 50) - degree 4
        for seg_id in ["north", "south", "east", "west"]:
            assert result[seg_id]["from_degree"] == 4
            assert result[seg_id]["to_degree"] == 1
            assert result[seg_id]["is_intersection"] is True

    def test_ids_to_compute_filter(self):
        """Should only return topology for specified IDs."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2", "seg3"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (100, 100)]),
                    LineString([(200, 200), (300, 200)]),
                ],
            },
            crs="EPSG:32610",
        )

        # Only request topology for seg1 and seg3
        result = compute_all_topology(gdf, ids_to_compute={"seg1", "seg3"})

        assert "seg1" in result
        assert "seg3" in result
        assert "seg2" not in result

    def test_tolerance_affects_connectivity(self):
        """Tolerance should control which endpoints are considered connected."""
        # Two segments with endpoints 3 meters apart
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 3), (100, 100)]),  # 3m gap
                ],
            },
            crs="EPSG:32610",
        )

        # With 5m tolerance, they should be connected
        result_5m = compute_all_topology(gdf, tolerance=5.0)
        assert result_5m["seg1"]["to_degree"] == 2
        assert result_5m["seg2"]["from_degree"] == 2

        # With 1m tolerance, they should not be connected
        result_1m = compute_all_topology(gdf, tolerance=1.0)
        assert result_1m["seg1"]["to_degree"] == 1
        assert result_1m["seg2"]["from_degree"] == 1

    def test_degree_signature(self):
        """Degree signature should be sorted tuple of degrees."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (100, 100)]),
                ],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology(gdf, tolerance=5.0)

        # seg1: from=1, to=2 -> signature should be (1, 2)
        assert result["seg1"]["degree_signature"] == (1, 2)

    def test_multilinestring_geometry(self):
        """Should handle MultiLineString geometries."""
        from shapely import MultiLineString

        gdf = gpd.GeoDataFrame(
            {
                "id": ["multi1", "seg2"],
                "geometry": [
                    MultiLineString([[(0, 0), (50, 0)], [(50, 0), (100, 0)]]),
                    LineString([(100, 0), (100, 100)]),
                ],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology(gdf, tolerance=5.0)

        # multi1's last endpoint (100, 0) should connect to seg2's start
        assert result["multi1"]["to_degree"] == 2
        assert result["seg2"]["from_degree"] == 2


class TestSpatialContextIndexClustering:
    """Tests for the _cluster_endpoints method with Union-Find."""

    def test_build_from_gdf_simple(self):
        """Basic index building should work."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (100, 100)]),
                ],
            },
            crs="EPSG:32610",
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance=5.0)

        # Should have 4 endpoints (2 per segment)
        assert len(ctx.endpoint_coords) == 4

        # Check that endpoint_to_segment mapping works
        assert len(ctx.endpoint_to_segment) == 4

    def test_infer_connectivity(self):
        """Infer connectivity should find connected segments."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2", "seg3"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (100, 100)]),
                    LineString([(200, 200), (300, 200)]),  # Disconnected
                ],
            },
            crs="EPSG:32610",
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance=5.0)

        # seg1 (index 0) should be connected to seg2 (index 1)
        connected = ctx.infer_connectivity(0, tolerance=5.0)
        assert 1 in connected
        assert 2 not in connected

        # seg3 (index 2) should not be connected to anything
        connected = ctx.infer_connectivity(2, tolerance=5.0)
        assert len(connected) == 0


class TestTopologyFeaturesConsistency:
    """Tests to verify compute_all_topology matches compute_topology_features."""

    def test_results_match_for_simple_case(self):
        """Batch and per-segment approaches should produce same results."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2", "seg3"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (100, 100)]),
                    LineString([(100, 100), (0, 100)]),
                ],
            },
            crs="EPSG:32610",
        )

        # Batch computation
        batch_result = compute_all_topology(gdf, id_column="id", tolerance=5.0)

        # Per-segment computation using SpatialContextIndex
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance=5.0)

        for _idx, row in gdf.iterrows():
            seg_id = row["id"]
            per_segment = compute_topology_features(row.geometry, ctx, tolerance=5.0)

            # Compare results
            assert batch_result[seg_id]["from_degree"] == per_segment["from_degree"]
            assert batch_result[seg_id]["to_degree"] == per_segment["to_degree"]
            assert batch_result[seg_id]["is_dead_end"] == per_segment["is_dead_end"]
            assert batch_result[seg_id]["is_intersection"] == per_segment["is_intersection"]
