"""Tests for spatial context indexing and topology computation."""

import geopandas as gpd
import pytest
from shapely import LineString

from matcher.features.spatial_context import (
    SpatialContextIndex,
    UnionFind,
    _cluster_endpoints_fast,
    compute_all_topology,
    compute_degree_match_score,
    compute_degree_signature_similarity,
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
        result = compute_all_topology(gdf, tolerance_m=5.0)

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
        result = compute_all_topology(gdf, tolerance_m=5.0)

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
        result = compute_all_topology(gdf, tolerance_m=5.0)

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
        result_5m = compute_all_topology(gdf, tolerance_m=5.0)
        assert result_5m["seg1"]["to_degree"] == 2
        assert result_5m["seg2"]["from_degree"] == 2

        # With 1m tolerance, they should not be connected
        result_1m = compute_all_topology(gdf, tolerance_m=1.0)
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
        result = compute_all_topology(gdf, tolerance_m=5.0)

        # seg1: from=1, to=2 -> signature should be (1, 2)
        assert result["seg1"]["degree_signature"] == (1, 2)

    def test_multilinestring_geometry(self):
        """MultiLineStrings should be filtered out of topology computation.

        The topology computation requires simple LineStrings to extract
        start/end points. MultiLineStrings are filtered out with a warning.
        """
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
        result = compute_all_topology(gdf, tolerance_m=5.0)

        # MultiLineString should be filtered out, only LineString remains
        assert "multi1" not in result, "MultiLineString should be filtered out"
        assert "seg2" in result, "LineString should be kept"
        # seg2 is isolated (no connections after multi1 filtered)
        assert result["seg2"]["from_degree"] == 1
        assert result["seg2"]["to_degree"] == 1


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
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance_m=5.0)

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
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance_m=5.0)

        # seg1 (index 0) should be connected to seg2 (index 1)
        connected = ctx.infer_connectivity(0, tolerance_m=5.0)
        assert 1 in connected
        assert 2 not in connected

        # seg3 (index 2) should not be connected to anything
        connected = ctx.infer_connectivity(2, tolerance_m=5.0)
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
        batch_result = compute_all_topology(gdf, id_column="id", tolerance_m=5.0)

        # Per-segment computation using SpatialContextIndex
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance_m=5.0)

        for _idx, row in gdf.iterrows():
            seg_id = row["id"]
            per_segment = compute_topology_features(row.geometry, ctx, tolerance_m=5.0)

            # Compare results
            assert batch_result[seg_id]["from_degree"] == per_segment["from_degree"]
            assert batch_result[seg_id]["to_degree"] == per_segment["to_degree"]
            assert batch_result[seg_id]["is_dead_end"] == per_segment["is_dead_end"]
            assert batch_result[seg_id]["is_intersection"] == per_segment["is_intersection"]


class TestComputeDegreeMatchScore:
    """Parameterized tests for compute_degree_match_score function."""

    @pytest.mark.parametrize(
        "ref_from,ref_to,target_from,target_to,expected",
        [
            # All zeros -> max similarity
            (0, 0, 0, 0, 1.0),
            # Identical degrees
            (4, 1, 4, 1, 1.0),
            # Swapped endpoints (should still be perfect match)
            (4, 1, 1, 4, 1.0),
            # Dead ends (degree 1) vs intersections (degree 4)
            (1, 1, 4, 4, 0.4),  # diff=6, max=10 -> 1-(6/10)=0.4
            # Moderate difference: T-junction vs 4-way
            (3, 1, 2, 2, 0.75),  # min(|3-2|+|1-2|, |3-2|+|1-2|)=2, max=8 -> 1-(2/8)=0.75
            # Single dead end vs normal segment
            (1, 2, 1, 1, 0.8),  # diff=1, max=5 -> 1-(1/5)=0.8
            # Both are dead ends
            (1, 1, 1, 1, 1.0),
            # High degree intersections
            (5, 5, 5, 5, 1.0),
            # Mixed: one matching, one not
            (3, 3, 3, 1, 0.8),  # diff=2, max=10 -> 1-(2/10)=0.8
        ],
        ids=[
            "all_zeros",
            "identical",
            "swapped_endpoints",
            "dead_ends_vs_intersections",
            "t_junction_vs_4way",
            "single_dead_end_diff",
            "both_dead_ends",
            "high_degree_match",
            "mixed_partial_match",
        ],
    )
    def test_degree_match_score(self, ref_from, ref_to, target_from, target_to, expected):
        """Test degree match scoring with various topology combinations."""
        score = compute_degree_match_score(ref_from, ref_to, target_from, target_to)
        assert score == pytest.approx(expected, abs=0.01)

    def test_score_symmetry(self):
        """Score should be symmetric in ref/target swap."""
        score1 = compute_degree_match_score(3, 2, 4, 1)
        score2 = compute_degree_match_score(4, 1, 3, 2)
        assert score1 == pytest.approx(score2)

    def test_score_in_valid_range(self):
        """Score should always be in [0, 1]."""
        import random

        random.seed(42)
        for _ in range(100):
            ref_from = random.randint(0, 10)
            ref_to = random.randint(0, 10)
            target_from = random.randint(0, 10)
            target_to = random.randint(0, 10)

            score = compute_degree_match_score(ref_from, ref_to, target_from, target_to)
            assert 0.0 <= score <= 1.0


class TestUnionFindParameterized:
    """Parameterized tests for UnionFind clustering correctness."""

    @pytest.mark.parametrize(
        "n,unions,expected_groups",
        [
            # Chain -> single group
            (4, [(0, 1), (1, 2), (2, 3)], [{0, 1, 2, 3}]),
            # Three disjoint pairs
            (6, [(0, 1), (2, 3), (4, 5)], [{0, 1}, {2, 3}, {4, 5}]),
            # Disjoint sets that merge
            (4, [(0, 1), (2, 3), (1, 2)], [{0, 1, 2, 3}]),
            # Star pattern: all connect to center
            (5, [(0, 1), (0, 2), (0, 3), (0, 4)], [{0, 1, 2, 3, 4}]),
            # No unions -> all singletons
            (3, [], [{0}, {1}, {2}]),
            # Binary tree pattern
            (7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)], [{0, 1, 2, 3, 4, 5, 6}]),
        ],
        ids=[
            "chain_single_group",
            "three_disjoint_pairs",
            "disjoint_then_merge",
            "star_pattern",
            "no_unions",
            "binary_tree",
        ],
    )
    def test_union_find_clustering(self, n, unions, expected_groups):
        """Test that Union-Find produces correct clustering."""
        uf = UnionFind(n)

        # Perform unions
        for x, y in unions:
            uf.union(x, y)

        # Build actual groups from find results
        actual_groups = {}
        for i in range(n):
            root = uf.find(i)
            if root not in actual_groups:
                actual_groups[root] = set()
            actual_groups[root].add(i)

        # Convert to list of sets for comparison
        actual_group_list = list(actual_groups.values())

        # Verify same number of groups
        assert len(actual_group_list) == len(expected_groups)

        # Verify each expected group exists in actual
        for expected in expected_groups:
            assert expected in actual_group_list

    def test_path_compression(self):
        """Test that path compression works (find returns consistent roots)."""
        uf = UnionFind(10)

        # Create a long chain
        for i in range(9):
            uf.union(i, i + 1)

        # After finding root of first element, path should be compressed
        root = uf.find(0)

        # All elements should have same root
        for i in range(10):
            assert uf.find(i) == root

    def test_union_by_rank(self):
        """Test that union by rank keeps tree balanced."""
        uf = UnionFind(8)

        # Create two separate trees
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(0, 2)  # Merge two size-2 trees

        uf.union(4, 5)
        uf.union(6, 7)
        uf.union(4, 6)  # Merge two size-2 trees

        # Merge the two size-4 trees
        uf.union(0, 4)

        # All should be in same set
        root = uf.find(0)
        for i in range(8):
            assert uf.find(i) == root


class TestComputeDegreeSignatureSimilarity:
    """Parameterized tests for compute_degree_signature_similarity function."""

    @pytest.mark.parametrize(
        "sig_a,sig_b,expected",
        [
            # Identical signatures
            ((1, 2), (1, 2), 1.0),
            # Completely different
            ((1, 1), (3, 3), 0.0),
            # Same multiset (order doesn't matter in Jaccard)
            ((1, 2, 3), (3, 2, 1), 1.0),
            # Partial overlap
            ((1, 2, 3), (1, 2, 4), 2 / 4),  # intersection=2, union=4
            # Repeated elements
            ((1, 1, 2), (1, 2, 2), 2 / 4),  # intersection=2 (one 1, one 2), union=4
            # Empty signature a
            ((), (1, 2), 0.0),
            # Empty signature b
            ((1, 2), (), 0.0),
            # Both empty
            ((), (), 0.0),
            # Single element match
            ((2,), (2,), 1.0),
            # Single element mismatch
            ((1,), (2,), 0.0),
        ],
        ids=[
            "identical",
            "completely_different",
            "same_multiset_reordered",
            "partial_overlap",
            "repeated_elements",
            "empty_a",
            "empty_b",
            "both_empty",
            "single_match",
            "single_mismatch",
        ],
    )
    def test_degree_signature_similarity(self, sig_a, sig_b, expected):
        """Test degree signature similarity with various inputs."""
        score = compute_degree_signature_similarity(sig_a, sig_b)
        assert score == pytest.approx(expected, abs=0.01)

    def test_symmetry(self):
        """Similarity should be symmetric."""
        sig_a = (1, 2, 3, 4)
        sig_b = (2, 3, 4, 5)

        score1 = compute_degree_signature_similarity(sig_a, sig_b)
        score2 = compute_degree_signature_similarity(sig_b, sig_a)
        assert score1 == pytest.approx(score2)

    def test_score_in_valid_range(self):
        """Score should always be in [0, 1]."""
        import random

        random.seed(42)
        for _ in range(50):
            len_a = random.randint(0, 5)
            len_b = random.randint(0, 5)
            sig_a = tuple(random.randint(1, 5) for _ in range(len_a))
            sig_b = tuple(random.randint(1, 5) for _ in range(len_b))

            score = compute_degree_signature_similarity(sig_a, sig_b)
            assert 0.0 <= score <= 1.0


class TestEndpointClusteringPerformance:
    """Performance regression tests for endpoint clustering.

    These tests ensure the clustering algorithm remains fast on dense datasets.
    The thresholds are based on expected performance with scipy's cKDTree.
    """

    def test_build_from_gdf_10k_segments_under_5_seconds(self):
        """Building spatial index from 10k segments should complete in under 5 seconds.

        This tests the full build_from_gdf pipeline including:
        - Endpoint extraction from geometries
        - Clustering with Union-Find
        - Building the STRtree index

        The Boston dataset has ~10k segments.
        """
        import time

        import numpy as np

        # Generate 10k random line segments
        np.random.seed(42)
        n_segments = 10_000

        # Create line segments as pairs of points
        starts = np.random.uniform(0, 1000, size=(n_segments, 2))
        # End points are 50-100m away from start
        angles = np.random.uniform(0, 2 * np.pi, size=n_segments)
        lengths = np.random.uniform(50, 100, size=n_segments)
        ends = starts + np.column_stack([lengths * np.cos(angles), lengths * np.sin(angles)])

        geometries = [LineString([starts[i], ends[i]]) for i in range(n_segments)]
        gdf = gpd.GeoDataFrame(
            {"id": [f"seg_{i}" for i in range(n_segments)], "geometry": geometries},
            crs="EPSG:32610",
        )

        start = time.perf_counter()
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance_m=5.0)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"build_from_gdf took {elapsed:.2f}s, expected < 5.0s"
        assert len(ctx.endpoint_coords) == 2 * n_segments

    def test_compute_all_topology_10k_segments_under_5_seconds(self):
        """Computing topology for 10k segments should complete in under 5 seconds."""
        import time

        import numpy as np

        np.random.seed(42)
        n_segments = 10_000

        starts = np.random.uniform(0, 1000, size=(n_segments, 2))
        angles = np.random.uniform(0, 2 * np.pi, size=n_segments)
        lengths = np.random.uniform(50, 100, size=n_segments)
        ends = starts + np.column_stack([lengths * np.cos(angles), lengths * np.sin(angles)])

        geometries = [LineString([starts[i], ends[i]]) for i in range(n_segments)]
        gdf = gpd.GeoDataFrame(
            {"id": [f"seg_{i}" for i in range(n_segments)], "geometry": geometries},
            crs="EPSG:32610",
        )

        start = time.perf_counter()
        topology = compute_all_topology(gdf, id_column="id", tolerance_m=5.0)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"compute_all_topology took {elapsed:.2f}s, expected < 5.0s"
        assert len(topology) == n_segments

    def test_clustering_20k_endpoints_under_1_second(self):
        """Clustering 20k endpoints should complete in under 1 second.

        This simulates a dense urban dataset like Boston streets.
        With cKDTree.query_pairs(), this should take ~0.1-0.2 seconds.
        We use 1 second as the threshold to allow for CI variability.
        """
        import time

        import numpy as np

        # Generate 20k endpoints in a 1km x 1km area (dense urban)
        # This is similar to the Boston streets dataset
        np.random.seed(42)
        n_endpoints = 20_000
        endpoint_coords = np.random.uniform(0, 1000, size=(n_endpoints, 2))

        tolerance = 5.0  # 5 meter snap tolerance

        start = time.perf_counter()
        uf = _cluster_endpoints_fast(endpoint_coords, tolerance)
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"Clustering took {elapsed:.2f}s, expected < 1.0s"

        # Verify it actually did work (should have some clusters)
        roots = {uf.find(i) for i in range(n_endpoints)}
        assert len(roots) < n_endpoints, "Should have formed some clusters"

    def test_clustering_50k_endpoints_under_3_seconds(self):
        """Clustering 50k endpoints should complete in under 3 seconds.

        This tests scaling behavior for larger datasets.
        """
        import time

        import numpy as np

        np.random.seed(42)
        n_endpoints = 50_000
        endpoint_coords = np.random.uniform(0, 2000, size=(n_endpoints, 2))

        tolerance = 5.0

        start = time.perf_counter()
        uf = _cluster_endpoints_fast(endpoint_coords, tolerance)
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, f"Clustering took {elapsed:.2f}s, expected < 3.0s"

        # Verify it actually did work
        roots = {uf.find(i) for i in range(n_endpoints)}
        assert len(roots) < n_endpoints, "Should have formed some clusters"

    def test_clustering_correctness_with_known_clusters(self):
        """Verify clustering produces correct results on known data."""
        import numpy as np

        # Create 4 tight clusters at corners of a 100x100 grid
        # Points within each cluster are 1m apart, clusters are 50m apart
        cluster_centers = [(0, 0), (0, 100), (100, 0), (100, 100)]
        points_per_cluster = 10

        all_points = []
        for cx, cy in cluster_centers:
            for i in range(points_per_cluster):
                # Points within 2m of cluster center
                all_points.append([cx + (i % 3) * 0.5, cy + (i // 3) * 0.5])

        endpoint_coords = np.array(all_points)
        tolerance = 5.0  # Should cluster points within same corner

        uf = _cluster_endpoints_fast(endpoint_coords, tolerance)

        # Count distinct clusters
        roots = {uf.find(i) for i in range(len(endpoint_coords))}
        assert len(roots) == 4, f"Expected 4 clusters, got {len(roots)}"

        # Verify points in same cluster have same root
        for cluster_idx, (_cx, _cy) in enumerate(cluster_centers):
            start_idx = cluster_idx * points_per_cluster
            end_idx = start_idx + points_per_cluster
            cluster_roots = {uf.find(i) for i in range(start_idx, end_idx)}
            assert len(cluster_roots) == 1, f"Cluster {cluster_idx} should have 1 root"

    def test_build_from_gdf_with_geographic_crs(self):
        """Verify build_from_gdf handles geographic CRS correctly.

        This test catches a critical bug where meter-based tolerance was
        interpreted as degrees when data is in EPSG:4326, causing
        O(N²) pair generation and massive slowdowns.
        """
        import time

        import numpy as np

        # Create segments in geographic coordinates (EPSG:4326)
        # Boston area: ~42.36°N, ~-71.06°E
        np.random.seed(42)
        n_segments = 1000

        # Create line segments ~100m long in geographic coords
        # 0.001 degrees ≈ 111m at this latitude
        base_lon, base_lat = -71.06, 42.36
        starts_lon = base_lon + np.random.uniform(-0.01, 0.01, n_segments)
        starts_lat = base_lat + np.random.uniform(-0.01, 0.01, n_segments)
        ends_lon = starts_lon + np.random.uniform(-0.001, 0.001, n_segments)
        ends_lat = starts_lat + np.random.uniform(-0.001, 0.001, n_segments)

        geometries = [
            LineString([(starts_lon[i], starts_lat[i]), (ends_lon[i], ends_lat[i])])
            for i in range(n_segments)
        ]
        gdf = gpd.GeoDataFrame(
            {"id": [f"seg_{i}" for i in range(n_segments)], "geometry": geometries},
            crs="EPSG:4326",  # Geographic CRS!
        )

        # This should complete quickly (< 2 seconds) because it projects to UTM
        # Before the fix, 5m tolerance was interpreted as 5 degrees, causing
        # N² pairs and taking minutes
        start = time.perf_counter()
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance_m=5.0)  # 5 meters
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"build_from_gdf with geographic CRS took {elapsed:.2f}s"
        assert len(ctx.endpoint_coords) == 2 * n_segments


class TestConnectorGraphAndAlignment:
    """Tests for connector-based graph building and alignment-aware graphlet similarity."""

    @pytest.fixture
    def sample_segment_with_connectors(self):
        """Create a sample GeoDataFrame with Overture-style connectors."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg_1", "seg_2", "seg_3"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),  # Horizontal segment
                    LineString([(100, 0), (200, 0)]),  # Connected horizontally
                    LineString([(100, 0), (100, 100)]),  # T-junction at (100, 0)
                ],
                "connectors": [
                    # seg_1: connectors at start, middle, and end
                    [
                        {"at": 0.0, "connector_id": "conn_a"},
                        {"at": 0.5, "connector_id": "conn_b"},
                        {"at": 1.0, "connector_id": "conn_c"},
                    ],
                    # seg_2: connectors at start and end, sharing conn_c with seg_1
                    [
                        {"at": 0.0, "connector_id": "conn_c"},
                        {"at": 1.0, "connector_id": "conn_d"},
                    ],
                    # seg_3: T-junction, shares conn_c at start
                    [
                        {"at": 0.0, "connector_id": "conn_c"},
                        {"at": 1.0, "connector_id": "conn_e"},
                    ],
                ],
            },
            crs="EPSG:32618",
        )
        return gdf

    def test_build_connector_graph_creates_correct_nodes(self, sample_segment_with_connectors):
        """build_connector_graph should create a node for each unique connector."""
        from matcher.features.spatial_context import build_connector_graph

        G, seg_to_connectors, features = build_connector_graph(
            sample_segment_with_connectors,
            id_column="id",
            connectors_column="connectors",
            degrees_only=False,  # Need graph for node count assertion
        )

        # Should have 5 unique connectors (conn_a, conn_b, conn_c, conn_d, conn_e)
        assert G.number_of_nodes() == 5

    def test_build_connector_graph_creates_correct_edges(self, sample_segment_with_connectors):
        """build_connector_graph should create edges between consecutive connectors."""
        from matcher.features.spatial_context import build_connector_graph

        G, seg_to_connectors, features = build_connector_graph(
            sample_segment_with_connectors,
            id_column="id",
            connectors_column="connectors",
            degrees_only=False,  # Need graph for edge count assertion
        )

        # seg_1: conn_a--conn_b--conn_c (2 edges)
        # seg_2: conn_c--conn_d (1 edge)
        # seg_3: conn_c--conn_e (1 edge)
        # Total: 4 edges
        assert G.number_of_edges() == 4

    def test_seg_to_connectors_mapping(self, sample_segment_with_connectors):
        """seg_to_connectors should map segment IDs to connector positions."""
        from matcher.features.spatial_context import build_connector_graph

        G, seg_to_connectors, features = build_connector_graph(
            sample_segment_with_connectors,
            id_column="id",
            connectors_column="connectors",
            degrees_only=False,  # Test with full features
        )

        assert "seg_1" in seg_to_connectors
        assert len(seg_to_connectors["seg_1"]) == 3  # Three connectors

        # Connectors should be sorted by position
        positions = [pos for pos, _ in seg_to_connectors["seg_1"]]
        assert positions == [0.0, 0.5, 1.0]

    def test_find_nearest_connector_exact_match(self):
        """find_nearest_connector should find exact matches."""
        from matcher.features.spatial_context import find_nearest_connector

        connectors = [(0.0, 10), (0.5, 20), (1.0, 30)]

        assert find_nearest_connector(connectors, 0.0) == 10
        assert find_nearest_connector(connectors, 0.5) == 20
        assert find_nearest_connector(connectors, 1.0) == 30

    def test_find_nearest_connector_interpolation(self):
        """find_nearest_connector should find nearest for intermediate positions."""
        from matcher.features.spatial_context import find_nearest_connector

        connectors = [(0.0, 10), (0.5, 20), (1.0, 30)]

        # Position 0.3 is closer to 0.5 (distance 0.2) than to 0.0 (distance 0.3)
        assert find_nearest_connector(connectors, 0.3) == 20
        # Position 0.2 is closer to 0.0 (distance 0.2) than to 0.5 (distance 0.3)
        assert find_nearest_connector(connectors, 0.2) == 10
        # Position 0.8 is closer to 1.0 (distance 0.2) than to 0.5 (distance 0.3)
        assert find_nearest_connector(connectors, 0.8) == 30

    def test_find_nearest_connector_empty_list(self):
        """find_nearest_connector should return None for empty list."""
        from matcher.features.spatial_context import find_nearest_connector

        assert find_nearest_connector([], 0.5) is None

    def test_get_alignment_connectors(self, sample_segment_with_connectors):
        """get_alignment_connectors should return correct nodes for alignment positions."""
        from matcher.features.spatial_context import (
            build_connector_graph,
            get_alignment_connectors,
        )

        G, seg_to_connectors, features = build_connector_graph(
            sample_segment_with_connectors,
            id_column="id",
            connectors_column="connectors",
            degrees_only=False,
        )

        # Full segment (0.0 to 1.0) should return start and end connectors
        start_node, end_node = get_alignment_connectors("seg_1", seg_to_connectors, 0.0, 1.0)
        assert start_node is not None
        assert end_node is not None
        assert start_node != end_node

        # Partial alignment (0.3 to 0.7) should find nearest connectors
        # 0.3 is closer to 0.5, and 0.7 is also closer to 0.5
        start_node2, end_node2 = get_alignment_connectors("seg_1", seg_to_connectors, 0.3, 0.7)
        assert start_node2 == end_node2  # Both map to the 0.5 connector

    def test_graphlet_similarity_with_alignment_full_segment(self, sample_segment_with_connectors):
        """graphlet_similarity_with_alignment should work for full segments."""
        from matcher.features.spatial_context import (
            build_connector_graph,
            graphlet_similarity_with_alignment,
        )

        G, seg_to_connectors, features = build_connector_graph(
            sample_segment_with_connectors,
            id_column="id",
            connectors_column="connectors",
            degrees_only=False,
        )

        # Compare seg_1 with seg_2 (they share connector conn_c)
        sim = graphlet_similarity_with_alignment(
            "seg_1",
            "seg_2",
            features,
            features,
            seg_to_connectors,
            seg_to_connectors,
            ref_start_frac=0.0,
            ref_end_frac=1.0,
            target_start_frac=0.0,
            target_end_frac=1.0,
        )

        assert "graphlet_similarity" in sim
        assert "endpoint_degree_similarity" in sim
        assert 0.0 <= sim["graphlet_similarity"] <= 1.0
        assert 0.0 <= sim["endpoint_degree_similarity"] <= 1.0

    def test_graphlet_similarity_with_alignment_partial_match(self, sample_segment_with_connectors):
        """graphlet_similarity_with_alignment should handle partial alignment."""
        from matcher.features.spatial_context import (
            build_connector_graph,
            graphlet_similarity_with_alignment,
        )

        G, seg_to_connectors, features = build_connector_graph(
            sample_segment_with_connectors,
            id_column="id",
            connectors_column="connectors",
            degrees_only=False,
        )

        # Compare partial seg_1 (0.4 to 0.6, around the middle connector)
        # with full seg_2
        sim = graphlet_similarity_with_alignment(
            "seg_1",
            "seg_2",
            features,
            features,
            seg_to_connectors,
            seg_to_connectors,
            ref_start_frac=0.4,
            ref_end_frac=0.6,
            target_start_frac=0.0,
            target_end_frac=1.0,
        )

        assert "graphlet_similarity" in sim
        assert 0.0 <= sim["graphlet_similarity"] <= 1.0
