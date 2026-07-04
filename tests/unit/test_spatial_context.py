"""Tests for spatial context indexing and topology computation."""

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString

from matcher.features.spatial_context import (
    SpatialContextIndex,
    TopologySpatialIndex,
    UnionFind,
    _cluster_endpoints_fast,
    compute_aligned_endpoint_features,
    compute_aligned_endpoint_features_batch,
    compute_aligned_topology_at_position,
    compute_aligned_topology_features,
    compute_all_topology,
    compute_degree_match_score,
    compute_degree_signature_similarity,
    compute_topology_features,
    find_nearest_connector_position,
    sample_topology_along_segment,
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
            ((), (1, 2), float("nan")),
            # Empty signature b
            ((1, 2), (), float("nan")),
            # Both empty
            ((), (), float("nan")),
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
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(score), f"Expected NaN for ({sig_a}, {sig_b}), got {score}"
        else:
            assert score == pytest.approx(expected, abs=0.01)

    def test_symmetry(self):
        """Similarity should be symmetric."""
        sig_a = (1, 2, 3, 4)
        sig_b = (2, 3, 4, 5)

        score1 = compute_degree_signature_similarity(sig_a, sig_b)
        score2 = compute_degree_signature_similarity(sig_b, sig_a)
        assert score1 == pytest.approx(score2)

    def test_score_in_valid_range(self):
        """Score should always be in [0, 1] or NaN (for empty signatures)."""
        import random

        random.seed(42)
        for _ in range(50):
            len_a = random.randint(0, 5)
            len_b = random.randint(0, 5)
            sig_a = tuple(random.randint(1, 5) for _ in range(len_a))
            sig_b = tuple(random.randint(1, 5) for _ in range(len_b))

            score = compute_degree_signature_similarity(sig_a, sig_b)
            if len_a == 0 or len_b == 0:
                assert math.isnan(score), f"Expected NaN for empty sig, got {score}"
            else:
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
        assert G.n_nodes == 5

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
        assert G.n_edges == 4

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


class TestComputeAllTopologyExplicit:
    """Tests for compute_all_topology_explicit function using explicit connector data."""

    def test_returns_none_when_no_connectors_column(self):
        """Should return None when connectors column doesn't exist."""
        from matcher.features.spatial_context import compute_all_topology_explicit

        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1"],
                "geometry": [LineString([(0, 0), (100, 0)])],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology_explicit(gdf, id_column="id", connectors_column="connectors")
        assert result is None

    def test_returns_none_when_connectors_all_null(self):
        """Should return None when all connectors are null."""
        from matcher.features.spatial_context import compute_all_topology_explicit

        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (200, 0)]),
                ],
                "connectors": [None, None],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology_explicit(gdf, id_column="id", connectors_column="connectors")
        assert result is None

    def test_single_segment_with_connectors(self):
        """Single segment with unique connectors has degree 1 at both ends."""
        from matcher.features.spatial_context import compute_all_topology_explicit

        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1"],
                "geometry": [LineString([(0, 0), (100, 0)])],
                "connectors": [
                    [
                        {"at": 0.0, "connector_id": "conn_a"},
                        {"at": 1.0, "connector_id": "conn_b"},
                    ]
                ],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology_explicit(gdf, id_column="id", connectors_column="connectors")

        assert result is not None
        assert "seg1" in result
        assert result["seg1"]["from_degree"] == 1
        assert result["seg1"]["to_degree"] == 1
        assert result["seg1"]["is_dead_end"] is True
        assert result["seg1"]["is_intersection"] is False

    def test_two_connected_segments_via_shared_connector(self):
        """Two segments sharing a connector should have degree 2 at that point."""
        from matcher.features.spatial_context import compute_all_topology_explicit

        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (100, 100)]),
                ],
                "connectors": [
                    [
                        {"at": 0.0, "connector_id": "conn_a"},
                        {"at": 1.0, "connector_id": "conn_shared"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "conn_shared"},
                        {"at": 1.0, "connector_id": "conn_b"},
                    ],
                ],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology_explicit(gdf, id_column="id", connectors_column="connectors")

        assert result is not None
        # seg1's end and seg2's start share conn_shared
        assert result["seg1"]["to_degree"] == 2
        assert result["seg2"]["from_degree"] == 2
        # Other endpoints are dead ends
        assert result["seg1"]["from_degree"] == 1
        assert result["seg2"]["to_degree"] == 1

    def test_t_junction_with_shared_connector(self):
        """T-junction with shared connector should have degree 3."""
        from matcher.features.spatial_context import compute_all_topology_explicit

        gdf = gpd.GeoDataFrame(
            {
                "id": ["main_left", "main_right", "side"],
                "geometry": [
                    LineString([(0, 0), (50, 0)]),
                    LineString([(50, 0), (100, 0)]),
                    LineString([(50, 50), (50, 0)]),
                ],
                "connectors": [
                    [
                        {"at": 0.0, "connector_id": "conn_west"},
                        {"at": 1.0, "connector_id": "conn_junction"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "conn_junction"},
                        {"at": 1.0, "connector_id": "conn_east"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "conn_north"},
                        {"at": 1.0, "connector_id": "conn_junction"},
                    ],
                ],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology_explicit(gdf, id_column="id", connectors_column="connectors")

        assert result is not None
        # Junction at conn_junction should have degree 3
        assert result["main_left"]["to_degree"] == 3
        assert result["main_right"]["from_degree"] == 3
        assert result["side"]["to_degree"] == 3
        # Outer endpoints are dead ends
        assert result["main_left"]["from_degree"] == 1
        assert result["main_right"]["to_degree"] == 1
        assert result["side"]["from_degree"] == 1
        # All segments touch an intersection
        assert result["main_left"]["is_intersection"] is True
        assert result["main_right"]["is_intersection"] is True
        assert result["side"]["is_intersection"] is True

    def test_cross_intersection_with_shared_connector(self):
        """4-way intersection with shared connector should have degree 4."""
        from matcher.features.spatial_context import compute_all_topology_explicit

        gdf = gpd.GeoDataFrame(
            {
                "id": ["north", "south", "east", "west"],
                "geometry": [
                    LineString([(50, 50), (50, 100)]),
                    LineString([(50, 50), (50, 0)]),
                    LineString([(50, 50), (100, 50)]),
                    LineString([(50, 50), (0, 50)]),
                ],
                "connectors": [
                    [
                        {"at": 0.0, "connector_id": "conn_center"},
                        {"at": 1.0, "connector_id": "conn_north"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "conn_center"},
                        {"at": 1.0, "connector_id": "conn_south"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "conn_center"},
                        {"at": 1.0, "connector_id": "conn_east"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "conn_center"},
                        {"at": 1.0, "connector_id": "conn_west"},
                    ],
                ],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology_explicit(gdf, id_column="id", connectors_column="connectors")

        assert result is not None
        # All segments share the center connector - degree 4
        for seg_id in ["north", "south", "east", "west"]:
            assert result[seg_id]["from_degree"] == 4
            assert result[seg_id]["to_degree"] == 1
            assert result[seg_id]["is_intersection"] is True

    def test_mid_segment_connector_counted_correctly(self):
        """Mid-segment connectors should not affect endpoint degrees."""
        from matcher.features.spatial_context import compute_all_topology_explicit

        # seg1 has a mid-segment connector (at=0.5) that is shared with seg2's start
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(50, 0), (50, 50)]),
                ],
                "connectors": [
                    [
                        {"at": 0.0, "connector_id": "conn_a"},
                        {"at": 0.5, "connector_id": "conn_mid"},  # mid-segment
                        {"at": 1.0, "connector_id": "conn_b"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "conn_mid"},  # shares with seg1's mid
                        {"at": 1.0, "connector_id": "conn_c"},
                    ],
                ],
            },
            crs="EPSG:32610",
        )
        result = compute_all_topology_explicit(gdf, id_column="id", connectors_column="connectors")

        assert result is not None
        # seg1's endpoints are conn_a and conn_b - only seg1 uses them
        assert result["seg1"]["from_degree"] == 1
        assert result["seg1"]["to_degree"] == 1
        # seg2's start is conn_mid (shared with seg1's mid) - degree 2
        # But seg1 doesn't count conn_mid as an endpoint
        assert result["seg2"]["from_degree"] == 2  # shared with seg1's mid-segment connector
        assert result["seg2"]["to_degree"] == 1


class TestAlignedTopologyFeatures:
    """Tests for alignment-aware topology computation.

    These tests verify that topology features are computed at the correct
    positions for partial overlaps, using the connector infrastructure.
    """

    @pytest.fixture
    def connector_graph_data(self):
        """Create sample connector data for testing aligned topology."""
        # seg_1: Linear segment with connectors at 0.0, 0.5, 1.0
        # Connector degrees: conn_a=1 (dead end), conn_b=3 (T-junction), conn_c=2 (through)
        seg_to_connectors = {
            "seg_1": [(0.0, 1), (0.5, 2), (1.0, 3)],  # connector positions and node IDs
            "seg_2": [(0.0, 3), (1.0, 4)],  # shares node 3 with seg_1
            "seg_3": [(0.0, 2), (1.0, 5)],  # shares node 2 with seg_1's middle
        }
        node_features = {
            1: 1,  # dead end
            2: 3,  # T-junction (connected to seg_1 and seg_3)
            3: 2,  # through connection (seg_1 end to seg_2 start)
            4: 1,  # dead end
            5: 1,  # dead end
        }
        return seg_to_connectors, node_features

    def test_compute_aligned_topology_at_position_exact_match(self, connector_graph_data):
        """Should return correct degree when position exactly matches a connector."""
        seg_to_connectors, node_features = connector_graph_data

        # At position 0.0, should return degree of node 1 (dead end)
        degree = compute_aligned_topology_at_position(
            seg_to_connectors, node_features, "seg_1", 0.0
        )
        assert degree == 1

        # At position 0.5, should return degree of node 2 (T-junction)
        degree = compute_aligned_topology_at_position(
            seg_to_connectors, node_features, "seg_1", 0.5
        )
        assert degree == 3

        # At position 1.0, should return degree of node 3 (through)
        degree = compute_aligned_topology_at_position(
            seg_to_connectors, node_features, "seg_1", 1.0
        )
        assert degree == 2

    def test_compute_aligned_topology_at_position_interpolated(self, connector_graph_data):
        """Should return degree of nearest connector for intermediate positions."""
        seg_to_connectors, node_features = connector_graph_data

        # Position 0.3 is closer to 0.5 (node 2, degree 3) than to 0.0 (node 1, degree 1)
        degree = compute_aligned_topology_at_position(
            seg_to_connectors, node_features, "seg_1", 0.3
        )
        assert degree == 3

        # Position 0.2 is closer to 0.0 (node 1, degree 1) than to 0.5 (node 2, degree 3)
        degree = compute_aligned_topology_at_position(
            seg_to_connectors, node_features, "seg_1", 0.2
        )
        assert degree == 1

        # Position 0.8 is closer to 1.0 (node 3, degree 2) than to 0.5 (node 2, degree 3)
        degree = compute_aligned_topology_at_position(
            seg_to_connectors, node_features, "seg_1", 0.8
        )
        assert degree == 2

    def test_compute_aligned_topology_at_position_missing_segment(self, connector_graph_data):
        """Should return 1 (dead end) for missing segment."""
        seg_to_connectors, node_features = connector_graph_data

        degree = compute_aligned_topology_at_position(
            seg_to_connectors, node_features, "nonexistent_seg", 0.5
        )
        assert degree == 1  # Dead end default

    def test_compute_aligned_topology_at_position_empty_connectors(self):
        """Should return 1 (dead end) when segment has no connectors."""
        seg_to_connectors = {"seg_1": []}  # Empty connector list
        node_features = {}

        degree = compute_aligned_topology_at_position(
            seg_to_connectors, node_features, "seg_1", 0.5
        )
        assert degree == 1  # Dead end default

    def test_compute_aligned_topology_features_full_segment(self, connector_graph_data):
        """Should return correct topology features for full segment (0.0 to 1.0)."""
        seg_to_connectors, node_features = connector_graph_data

        features = compute_aligned_topology_features(
            "seg_1", seg_to_connectors, node_features, 0.0, 1.0
        )

        assert features["from_degree"] == 1  # node 1 at position 0.0
        assert features["to_degree"] == 2  # node 3 at position 1.0
        assert features["is_dead_end"] is True  # min(1, 2) == 1
        assert features["is_intersection"] is False  # max(1, 2) = 2, not > 2
        assert features["degree_signature"] == (1, 2)

    def test_compute_aligned_topology_features_partial_overlap_at_intersection(
        self, connector_graph_data
    ):
        """Should return correct topology for partial overlap at T-junction."""
        seg_to_connectors, node_features = connector_graph_data

        # Partial overlap from 0.3 to 0.7 - both endpoints map to node 2 (T-junction)
        features = compute_aligned_topology_features(
            "seg_1", seg_to_connectors, node_features, 0.3, 0.7
        )

        # Both endpoints map to the 0.5 connector (node 2, degree 3)
        assert features["from_degree"] == 3
        assert features["to_degree"] == 3
        assert features["is_dead_end"] is False  # min(3, 3) = 3, not 1
        assert features["is_intersection"] is True  # max(3, 3) = 3 > 2
        assert features["degree_signature"] == (3, 3)

    def test_compute_aligned_topology_features_partial_overlap_spanning_connectors(
        self, connector_graph_data
    ):
        """Should use nearest connectors for partial overlap spanning multiple."""
        seg_to_connectors, node_features = connector_graph_data

        # Partial overlap from 0.2 to 0.8
        # 0.2 is closer to 0.0 (node 1, degree 1)
        # 0.8 is closer to 1.0 (node 3, degree 2)
        features = compute_aligned_topology_features(
            "seg_1", seg_to_connectors, node_features, 0.2, 0.8
        )

        assert features["from_degree"] == 1
        assert features["to_degree"] == 2
        assert features["is_dead_end"] is True
        assert features["is_intersection"] is False

    def test_aligned_topology_differs_from_full_geometry(self):
        """Verify that aligned topology can differ from full geometry topology.

        This is the key test case from the investigation: a 438m segment that
        only overlaps 43% with a 186m reference segment should use the degrees
        at the aligned endpoints, not the full geometry endpoints.
        """
        # Simulated scenario:
        # Full segment has connectors at 0.0 (intersection, degree 4) and 1.0 (dead end, degree 1)
        # But the alignment only covers 0.0 to 0.43 (43% overlap)
        # The 0.43 position is closest to a mid-segment connector at 0.5 (T-junction, degree 3)

        seg_to_connectors = {
            "long_segment": [
                (0.0, 1),  # Start: intersection
                (0.5, 2),  # Middle: T-junction
                (1.0, 3),  # End: dead end
            ]
        }
        node_features = {
            1: 4,  # intersection degree
            2: 3,  # T-junction degree
            3: 1,  # dead end degree
        }

        # Full segment topology (incorrect for partial overlap)
        full_features = compute_aligned_topology_features(
            "long_segment", seg_to_connectors, node_features, 0.0, 1.0
        )
        assert full_features["from_degree"] == 4
        assert full_features["to_degree"] == 1

        # Aligned topology (correct for 43% overlap from start)
        # Position 0.43 maps to connector at 0.5 (distance 0.07) vs 0.0 (distance 0.43)
        aligned_features = compute_aligned_topology_features(
            "long_segment", seg_to_connectors, node_features, 0.0, 0.43
        )
        assert aligned_features["from_degree"] == 4  # Start is still at intersection
        assert aligned_features["to_degree"] == 3  # End maps to T-junction, not dead end!

        # The key difference: aligned end degree (3) != full end degree (1)
        assert aligned_features["to_degree"] != full_features["to_degree"]


class TestEndpointFeaturesCRSConsistency:
    """Tests for CRS consistency in endpoint feature computation.

    These tests verify that the spatial index CRS matches the query geometry CRS.
    A CRS mismatch causes endpoint proximity features to return infinity because
    the R-tree query coordinates are in different units (e.g., UTM meters vs WGS84 degrees).
    """

    def test_endpoint_features_with_projected_index_and_projected_query(self):
        """Endpoint features should work when index and query use same projected CRS."""
        from matcher.features.spatial_context import SpatialContextIndex, compute_endpoint_features

        # Create segments in projected CRS (UTM)
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2", "seg3"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (100, 100)]),  # Shares endpoint with seg1
                    LineString([(200, 200), (300, 200)]),  # Isolated
                ],
            },
            crs="EPSG:32610",  # UTM zone 10N
        )

        # Build index from projected data
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance_m=5.0)

        # Query with geometry from the same projected CRS
        query_geom = gdf.geometry.iloc[0]  # seg1
        features = compute_endpoint_features(query_geom, ctx, exclude_segment_idx=0)

        # seg1 shares endpoint (100,0) with seg2, so min proximity should be ~0
        assert features["min_endpoint_proximity_m"] < 1.0, (
            f"Expected small proximity (shared endpoint), got {features['min_endpoint_proximity_m']}"
        )
        assert features["shared_endpoint_count"] >= 1

    def test_endpoint_features_with_geographic_index_and_geographic_query(self):
        """Endpoint features should work when index and query use same geographic CRS."""
        from matcher.features.spatial_context import SpatialContextIndex, compute_endpoint_features

        # Create segments in geographic CRS (WGS84)
        # Boston area: ~42.36°N, ~-71.06°E
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(-71.06, 42.36), (-71.05, 42.36)]),  # ~800m segment
                    LineString([(-71.05, 42.36), (-71.05, 42.37)]),  # Shares endpoint
                ],
            },
            crs="EPSG:4326",  # WGS84
        )

        # Build index from geographic data (will internally project to UTM)
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance_m=5.0)

        # Query with geometry from WGS84 (same as input)
        query_geom = gdf.geometry.iloc[0]  # seg1
        features = compute_endpoint_features(query_geom, ctx, exclude_segment_idx=0)

        # seg1 shares endpoint with seg2, so min proximity should be ~0
        assert features["min_endpoint_proximity_m"] < 5.0, (
            f"Expected small proximity (shared endpoint), got {features['min_endpoint_proximity_m']}"
        )
        assert features["shared_endpoint_count"] >= 1

    def test_endpoint_features_crs_mismatch_returns_infinity(self):
        """CRS mismatch between index and query should be avoided by design.

        This test documents the bug that was fixed: if the spatial index is built
        from WGS84 data but queries use projected (UTM) coordinates, the R-tree
        query fails because coordinates are in different units.

        The fix ensures both index and query use the same CRS.
        """

        from matcher.features.spatial_context import SpatialContextIndex, compute_endpoint_features

        # Build index from WGS84 data
        gdf_wgs84 = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(-71.06, 42.36), (-71.05, 42.36)]),
                    LineString([(-71.05, 42.36), (-71.05, 42.37)]),
                ],
            },
            crs="EPSG:4326",
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf_wgs84, id_column="id", snap_tolerance_m=5.0)

        # Create a query geometry in UTM coordinates (wrong CRS!)
        # These coordinates are what seg1's endpoint would be in UTM zone 19N
        # If we accidentally query with UTM coords against a WGS84-based index,
        # the query will find nothing because the numbers are in different scales
        wrong_crs_geom = LineString([(330000, 4690000), (331000, 4690000)])

        # This demonstrates the bug: querying with mismatched CRS returns infinity
        # because the R-tree can't find any nearby points in the wrong coordinate space
        features = compute_endpoint_features(wrong_crs_geom, ctx, exclude_segment_idx=None)

        # With CRS mismatch, proximity should be infinity (no nearby endpoints found)
        # This is the symptom of the bug - if you see infinity in production, check CRS!
        assert features["min_endpoint_proximity_m"] > 9000, (
            "CRS mismatch should result in very large proximity (no matches found)"
        )

    def test_compute_features_only_uses_consistent_crs(self):
        """compute_features_only should use consistent CRS for index and query.

        This is an integration test that verifies the fix in data_loader.py.
        Before the fix, target_candidates_only was built from unprojected data
        but queries used projected geometries, causing CRS mismatch.
        """
        from matcher.labeling.data_loader import compute_features_only

        # Create small test datasets in WGS84
        reference = gpd.GeoDataFrame(
            {
                "id": ["ref1", "ref2"],
                "geometry": [
                    LineString([(-71.06, 42.36), (-71.05, 42.36)]),
                    LineString([(-71.05, 42.36), (-71.04, 42.36)]),
                ],
                "names": [{"primary": "Main St"}, {"primary": "Main St"}],
                "class": ["primary", "primary"],
            },
            crs="EPSG:4326",
        )

        target = gpd.GeoDataFrame(
            {
                "id": ["target1", "target2"],
                "geometry": [
                    # Overlaps with ref1
                    LineString([(-71.06, 42.3601), (-71.05, 42.3601)]),
                    # Overlaps with ref2
                    LineString([(-71.05, 42.3601), (-71.04, 42.3601)]),
                ],
                "names": [{"primary": "Main Street"}, {"primary": "Main Street"}],
                "class": ["primary", "primary"],
            },
            crs="EPSG:4326",
        )

        # Generate features
        df = compute_features_only(
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            ref_name_column="names",
            target_name_column="names",
            ref_class_column="class",
            target_class_column="class",
        )

        if len(df) > 0:
            # Endpoint features should have reasonable values, not all infinity
            # The targets share an endpoint at (-71.05, 42.3601)
            finite_min = (df["min_endpoint_proximity_m"] < 9000).sum()
            total = len(df)

            # At least some pairs should have finite endpoint proximity
            # (not all infinity, which would indicate CRS mismatch bug)
            assert finite_min > 0 or total == 0, (
                f"All {total} pairs have infinite endpoint proximity - CRS mismatch bug?"
            )


class TestComputeAllTopologyWithConnectors:
    """Tests for compute_all_topology with connectors_column parameter."""

    def test_uses_explicit_when_connectors_available(self):
        """Should use explicit topology when connectors column is available."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (100, 100)]),
                ],
                "connectors": [
                    [
                        {"at": 0.0, "connector_id": "conn_a"},
                        {"at": 1.0, "connector_id": "conn_shared"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "conn_shared"},
                        {"at": 1.0, "connector_id": "conn_b"},
                    ],
                ],
            },
            crs="EPSG:32610",
        )

        result = compute_all_topology(gdf, id_column="id", connectors_column="connectors")

        # Should produce same results as explicit computation
        assert result["seg1"]["to_degree"] == 2
        assert result["seg2"]["from_degree"] == 2

    def test_falls_back_to_geometry_when_no_connectors(self):
        """Should fall back to geometry inference when connectors not available."""
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

        # Even with connectors_column specified, should work (falls back)
        result = compute_all_topology(
            gdf, id_column="id", tolerance_m=5.0, connectors_column="connectors"
        )

        # Should produce correct results from geometry inference
        assert result["seg1"]["to_degree"] == 2
        assert result["seg2"]["from_degree"] == 2

    def test_connectors_column_none_uses_geometry(self):
        """When connectors_column is None, should use geometry inference."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (100, 100)]),
                ],
                "connectors": [
                    [
                        {"at": 0.0, "connector_id": "conn_a"},
                        {"at": 1.0, "connector_id": "conn_b"},  # Different from seg2!
                    ],
                    [
                        {"at": 0.0, "connector_id": "conn_c"},  # Different connector
                        {"at": 1.0, "connector_id": "conn_d"},
                    ],
                ],
            },
            crs="EPSG:32610",
        )

        # With connectors_column=None, should use geometry (which finds connection)
        result = compute_all_topology(gdf, id_column="id", tolerance_m=5.0, connectors_column=None)

        # Geometry inference finds the connection
        assert result["seg1"]["to_degree"] == 2
        assert result["seg2"]["from_degree"] == 2


class TestFindNearestConnectorPosition:
    """Tests for find_nearest_connector_position helper."""

    def test_exact_match_returns_same_position(self):
        """Exact match on a connector position returns that position."""
        connectors = [(0.0, 1), (0.5, 2), (1.0, 3)]
        result = find_nearest_connector_position(connectors, 0.5)
        assert result == 0.5

    def test_snap_to_nearest(self):
        """Position 0.43 should snap to nearest connector at 0.42."""
        connectors = [(0.0, 1), (0.42, 2), (1.0, 3)]
        result = find_nearest_connector_position(connectors, 0.43)
        assert result == 0.42

    def test_empty_list_returns_none(self):
        """Empty connector list returns None."""
        result = find_nearest_connector_position([], 0.5)
        assert result is None

    def test_single_connector(self):
        """Single connector is always the nearest."""
        connectors = [(0.7, 42)]
        result = find_nearest_connector_position(connectors, 0.1)
        assert result == 0.7

    def test_equidistant_picks_first(self):
        """When equidistant to two connectors, picks the first encountered."""
        connectors = [(0.4, 1), (0.6, 2)]
        result = find_nearest_connector_position(connectors, 0.5)
        # Both are 0.1 away; first encountered (0.4) wins
        assert result == 0.4


class TestEndpointProximityContinuity:
    """Endpoint proximity must be a continuous distance, not pinned at the sentinel.

    Regression for the #253-deferred degeneracy: the old bounded radius query
    (query_ball_point at r = 2*tolerance ≈ 10 m) collapsed any endpoint whose
    nearest neighbour sat beyond that radius to MAX_DISTANCE_METERS, pinning
    ~87-91% of pairs. The k-NN redesign returns the true nearest distance.
    """

    def test_proximity_beyond_radius_is_continuous_not_sentinel(self):
        from matcher.config import MAX_DISTANCE_METERS
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            compute_aligned_endpoint_features,
        )

        # "iso" end (100,0) is 50 m from "near" start (150,0) — well beyond the
        # ~10 m bounded radius that used to force the sentinel.
        gdf = gpd.GeoDataFrame(
            {
                "id": ["iso", "near"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(150, 0), (250, 0)]),
                ],
            },
            crs="EPSG:32610",
        )
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance_m=5.0)

        result = compute_aligned_endpoint_features(
            LineString([(0, 0), (100, 0)]),
            ctx,
            start_frac=0.0,
            end_frac=1.0,
            exclude_segment_idx=0,  # exclude "iso"'s own two endpoints
        )

        # Nearest non-self endpoint to (100,0) is (150,0) → 50 m; to (0,0) → 150 m.
        assert result["min_endpoint_proximity_m"] == pytest.approx(50.0, abs=1e-6)
        assert result["max_endpoint_proximity_m"] == pytest.approx(150.0, abs=1e-6)
        # Crucially: no longer pinned at the 10 km sentinel.
        assert result["min_endpoint_proximity_m"] < MAX_DISTANCE_METERS

    def test_connected_endpoint_reports_near_zero(self):
        """A shared junction endpoint still reports ~0 proximity (connectivity)."""
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            compute_aligned_endpoint_features,
        )

        # "a" ends at (100,0) where "b" and "c" also meet → connected.
        gdf = gpd.GeoDataFrame(
            {
                "id": ["a", "b", "c"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (200, 0)]),
                    LineString([(100, 0), (100, 100)]),
                ],
            },
            crs="EPSG:32610",
        )
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(gdf, id_column="id", snap_tolerance_m=5.0)

        result = compute_aligned_endpoint_features(
            LineString([(0, 0), (100, 0)]),
            ctx,
            start_frac=0.0,
            end_frac=1.0,
            exclude_segment_idx=0,
        )
        # End (100,0) coincides with b/c endpoints → ~0 m.
        assert result["min_endpoint_proximity_m"] == pytest.approx(0.0, abs=1e-6)


class TestAlignedEndpointFeaturesWithConnectors:
    """Tests for compute_aligned_endpoint_features with connector snapping."""

    @pytest.fixture
    def simple_context(self):
        """Build a SpatialContextIndex from a simple set of target segments."""
        # Three segments forming a T-junction:
        # seg_a: (0,0) -> (100,0)
        # seg_b: (100,0) -> (200,0)
        # seg_c: (100,0) -> (100,100)
        segments = gpd.GeoDataFrame(
            {
                "id": ["a", "b", "c"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (200, 0)]),
                    LineString([(100, 0), (100, 100)]),
                ],
            }
        )
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(segments, id_column="id")
        return ctx

    def test_with_connectors_snaps_fraction(self, simple_context):
        """When connector data is provided, fractions should snap to connector positions.

        Geometry: seg_a is (0,0)->(200,0) with a connector at position 0.5 = (100,0).
        The context has segment endpoints at (100,0) from seg_b and seg_c.
        Raw fraction 0.43 interpolates to (86,0) which is far from any endpoint.
        Snapped fraction 0.5 interpolates to (100,0) which is RIGHT at the junction.
        """
        # Use a 200m line so 0.5 = (100,0) which is at the T-junction
        geom = LineString([(0, 0), (200, 0)])

        # Without connectors: raw fraction 0.43 → point (86,0), not near any endpoint
        result_raw = compute_aligned_endpoint_features(
            geom,
            simple_context,
            start_frac=0.0,
            end_frac=0.43,
        )

        # With connectors: 0.43 snaps to connector at 0.5 → point (100,0),
        # which is right at the T-junction endpoint shared by seg_b and seg_c
        connectors = {"seg_a": [(0.0, 1), (0.5, 2), (1.0, 3)]}
        result_snapped = compute_aligned_endpoint_features(
            geom,
            simple_context,
            start_frac=0.0,
            end_frac=0.43,
            seg_id="seg_a",
            seg_to_connectors=connectors,
        )

        # The snapped version's end point (100,0) is at the junction, so
        # max_endpoint_proximity should be much smaller than the raw version
        assert result_snapped["max_endpoint_proximity_m"] < result_raw["max_endpoint_proximity_m"]

    def test_without_connectors_uses_defaults(self, simple_context):
        """Without connector data, default and explicit None produce same result."""
        geom = LineString([(0, 0), (100, 0)])

        result_default = compute_aligned_endpoint_features(
            geom,
            simple_context,
            start_frac=0.0,
            end_frac=0.5,
        )

        # Explicitly passing None should give same result
        result_none = compute_aligned_endpoint_features(
            geom,
            simple_context,
            start_frac=0.0,
            end_frac=0.5,
            seg_id=None,
            seg_to_connectors=None,
        )

        assert result_default == result_none

    def test_no_connectors_for_seg_id(self, simple_context):
        """When seg_id not in seg_to_connectors, no snapping occurs."""
        geom = LineString([(0, 0), (100, 0)])

        # Provide connectors but not for this seg_id
        connectors = {"other_seg": [(0.0, 1), (0.5, 2), (1.0, 3)]}
        result_with = compute_aligned_endpoint_features(
            geom,
            simple_context,
            start_frac=0.0,
            end_frac=0.43,
            seg_id="seg_a",
            seg_to_connectors=connectors,
        )

        result_without = compute_aligned_endpoint_features(
            geom,
            simple_context,
            start_frac=0.0,
            end_frac=0.43,
        )

        assert result_with == result_without

    def test_full_segment_match_unaffected(self, simple_context):
        """Full segment match (0.0, 1.0) is unaffected by connector snapping."""
        geom = LineString([(0, 0), (100, 0)])

        # Connectors at 0.0 and 1.0 match the raw fractions exactly
        connectors = {"seg_a": [(0.0, 1), (0.5, 2), (1.0, 3)]}

        result_without = compute_aligned_endpoint_features(
            geom, simple_context, start_frac=0.0, end_frac=1.0
        )
        result_with = compute_aligned_endpoint_features(
            geom,
            simple_context,
            start_frac=0.0,
            end_frac=1.0,
            seg_id="seg_a",
            seg_to_connectors=connectors,
        )

        assert result_without == result_with


class TestAlignedEndpointFeaturesBatch:
    """Tests for compute_aligned_endpoint_features_batch vectorized implementation."""

    @pytest.fixture
    def t_junction_context(self):
        """Build a SpatialContextIndex from a T-junction of three segments."""
        # seg_a: (0,0) -> (100,0)
        # seg_b: (100,0) -> (200,0)
        # seg_c: (100,0) -> (100,100)
        segments = gpd.GeoDataFrame(
            {
                "id": ["a", "b", "c"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (200, 0)]),
                    LineString([(100, 0), (100, 100)]),
                ],
            }
        )
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(segments, id_column="id")
        return ctx

    def _make_alignment(self, start_frac, end_frac):
        """Create a minimal alignment-like object for testing."""
        from dataclasses import dataclass

        @dataclass
        class _Alignment:
            dataset_start_frac: float
            dataset_end_frac: float

        return _Alignment(start_frac, end_frac)

    def test_batch_matches_per_pair(self, t_junction_context):
        """Batch output should match per-pair compute_aligned_endpoint_features."""
        geom = LineString([(0, 0), (100, 0)])
        ctx = t_junction_context

        # Per-pair result
        per_pair = compute_aligned_endpoint_features(
            geom, ctx, start_frac=0.0, end_frac=1.0, exclude_segment_idx=0
        )

        # Batch result
        alignments = {(0, 0): self._make_alignment(0.0, 1.0)}
        batch_result = compute_aligned_endpoint_features_batch(
            alignments=alignments,
            target_geoms=np.array([geom]),
            target_ids=np.array(["a"]),
            target_index=ctx,
            original_to_filtered={0: 0},
        )

        assert (0, 0) in batch_result
        for key in (
            "min_endpoint_proximity_m",
            "max_endpoint_proximity_m",
            "shared_endpoint_count",
        ):
            assert batch_result[(0, 0)][key] == pytest.approx(per_pair[key], abs=1e-6), (
                f"{key}: batch={batch_result[(0, 0)][key]} != per_pair={per_pair[key]}"
            )

    def test_batch_multiple_pairs(self, t_junction_context):
        """Batch should process multiple pairs correctly."""
        ctx = t_junction_context
        geoms = np.array(
            [
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (200, 0)]),
            ]
        )
        target_ids = np.array(["a", "b"])

        alignments = {
            (0, 0): self._make_alignment(0.0, 1.0),
            (1, 1): self._make_alignment(0.0, 1.0),
        }
        result = compute_aligned_endpoint_features_batch(
            alignments=alignments,
            target_geoms=geoms,
            target_ids=target_ids,
            target_index=ctx,
            original_to_filtered={0: 0, 1: 1},
        )

        assert len(result) == 2
        assert (0, 0) in result
        assert (1, 1) in result
        for key in result:
            assert "min_endpoint_proximity_m" in result[key]
            assert "max_endpoint_proximity_m" in result[key]
            assert "shared_endpoint_count" in result[key]

    def test_batch_empty_alignments(self, t_junction_context):
        """Empty alignments should return empty dict."""
        result = compute_aligned_endpoint_features_batch(
            alignments={},
            target_geoms=np.array([]),
            target_ids=np.array([]),
            target_index=t_junction_context,
            original_to_filtered={},
        )
        assert result == {}

    def test_batch_empty_endpoint_index(self):
        """Empty endpoint index should return defaults for all pairs."""
        from matcher.config import MAX_DISTANCE_METERS

        empty_ctx = SpatialContextIndex()
        geom = LineString([(0, 0), (100, 0)])

        alignments = {(0, 0): self._make_alignment(0.0, 1.0)}
        result = compute_aligned_endpoint_features_batch(
            alignments=alignments,
            target_geoms=np.array([geom]),
            target_ids=np.array(["a"]),
            target_index=empty_ctx,
            original_to_filtered={},
        )

        assert (0, 0) in result
        assert result[(0, 0)]["min_endpoint_proximity_m"] == MAX_DISTANCE_METERS
        assert result[(0, 0)]["max_endpoint_proximity_m"] == MAX_DISTANCE_METERS
        assert result[(0, 0)]["shared_endpoint_count"] == 0

    def test_batch_with_connectors(self, t_junction_context):
        """Batch with connectors should match per-pair with connectors."""
        geom = LineString([(0, 0), (200, 0)])
        ctx = t_junction_context
        connectors = {"seg_a": [(0.0, 1), (0.5, 2), (1.0, 3)]}

        # Per-pair
        per_pair = compute_aligned_endpoint_features(
            geom,
            ctx,
            start_frac=0.0,
            end_frac=0.43,
            seg_id="seg_a",
            seg_to_connectors=connectors,
        )

        # Batch
        alignments = {(0, 0): self._make_alignment(0.0, 0.43)}
        batch_result = compute_aligned_endpoint_features_batch(
            alignments=alignments,
            target_geoms=np.array([geom]),
            target_ids=np.array(["seg_a"]),
            target_index=ctx,
            original_to_filtered={},
            seg_to_connectors=connectors,
        )

        for key in (
            "min_endpoint_proximity_m",
            "max_endpoint_proximity_m",
            "shared_endpoint_count",
        ):
            assert batch_result[(0, 0)][key] == pytest.approx(per_pair[key], abs=1e-6), (
                f"{key}: batch={batch_result[(0, 0)][key]} != per_pair={per_pair[key]}"
            )

    def test_batch_invalid_geometry_skipped(self, t_junction_context):
        """Invalid/empty geometries should be skipped in batch results."""
        ctx = t_junction_context
        geoms = np.array(
            [
                LineString([(0, 0), (100, 0)]),
                LineString(),  # empty
            ]
        )

        alignments = {
            (0, 0): self._make_alignment(0.0, 1.0),
            (1, 1): self._make_alignment(0.0, 1.0),
        }
        result = compute_aligned_endpoint_features_batch(
            alignments=alignments,
            target_geoms=geoms,
            target_ids=np.array(["a", "b"]),
            target_index=ctx,
            original_to_filtered={0: 0},
        )

        assert (0, 0) in result
        assert (1, 1) not in result


class TestTopologySpatialIndex:
    """Tests for TopologySpatialIndex and return_spatial_index parameter."""

    def test_compute_all_topology_return_spatial_index(self):
        """compute_all_topology with return_spatial_index=True returns tuple."""
        # Simple T-junction: one long segment, two shorter segments meeting it
        gdf = gpd.GeoDataFrame(
            {
                "id": ["long", "arm1", "arm2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),  # Horizontal
                    LineString([(50, 0), (50, 50)]),  # Vertical, meets at midpoint
                    LineString([(50, 0), (50, -50)]),  # Vertical, other direction
                ],
            },
            crs="EPSG:32632",
        )
        result = compute_all_topology(gdf, tolerance_m=5.0, return_spatial_index=True)

        # Should return tuple
        assert isinstance(result, tuple)
        topo_dict, spatial_index = result

        # Check topology dict
        assert isinstance(topo_dict, dict)
        assert len(topo_dict) == 3

        # Check spatial index
        assert isinstance(spatial_index, TopologySpatialIndex)
        assert len(spatial_index.centroids) > 0
        assert len(spatial_index.degrees) == len(spatial_index.centroids)
        assert spatial_index.tree is not None

    def test_spatial_index_captures_junction_degree(self):
        """Spatial index should capture high degree at junction points."""
        # 4-way intersection: 4 segments meeting at (50, 50)
        gdf = gpd.GeoDataFrame(
            {
                "id": ["n", "s", "e", "w"],
                "geometry": [
                    LineString([(50, 50), (50, 100)]),  # North
                    LineString([(50, 50), (50, 0)]),  # South
                    LineString([(50, 50), (100, 50)]),  # East
                    LineString([(50, 50), (0, 50)]),  # West
                ],
            },
            crs="EPSG:32632",
        )
        _, spatial_index = compute_all_topology(gdf, tolerance_m=5.0, return_spatial_index=True)

        # One cluster at (50,50) with degree 4
        assert 4 in spatial_index.degrees

    def test_without_return_spatial_index_returns_dict_only(self):
        """Default behavior should just return dict (not tuple)."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["a"],
                "geometry": [LineString([(0, 0), (100, 0)])],
            },
            crs="EPSG:32632",
        )
        result = compute_all_topology(gdf, tolerance_m=5.0)
        assert isinstance(result, dict)


class TestSampleTopologyAlongSegment:
    """Tests for sample_topology_along_segment()."""

    def _make_intersection_index(self):
        """Create a spatial index with a 4-way intersection at (50, 50)."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["n", "s", "e", "w"],
                "geometry": [
                    LineString([(50, 50), (50, 100)]),
                    LineString([(50, 50), (50, 0)]),
                    LineString([(50, 50), (100, 50)]),
                    LineString([(50, 50), (0, 50)]),
                ],
            },
            crs="EPSG:32632",
        )
        _, spatial_index = compute_all_topology(gdf, tolerance_m=5.0, return_spatial_index=True)
        return spatial_index

    def test_sample_returns_connectors_and_node_features(self):
        """Should return list of connectors and dict of node features."""
        spatial_index = self._make_intersection_index()

        # A segment passing through the intersection
        geom = LineString([(0, 50), (100, 50)])
        connectors, node_features = sample_topology_along_segment(
            geom, spatial_index, sample_interval_m=50.0, tolerance_m=10.0
        )

        assert len(connectors) > 0
        assert len(node_features) > 0

        # All connectors should have (frac, node_id) format
        for frac, node_id in connectors:
            assert 0.0 <= frac <= 1.0
            assert node_id in node_features

    def test_includes_endpoints(self):
        """Should always include frac=0.0 and frac=1.0."""
        spatial_index = self._make_intersection_index()
        geom = LineString([(0, 50), (100, 50)])
        connectors, _ = sample_topology_along_segment(geom, spatial_index, sample_interval_m=50.0)

        fracs = [c[0] for c in connectors]
        assert 0.0 in fracs
        assert 1.0 in fracs

    def test_detects_junction_along_segment(self):
        """Segment passing through a junction should have degree > 1 at that sample."""
        spatial_index = self._make_intersection_index()

        # Segment passing through the junction at (50, 50)
        geom = LineString([(0, 50), (100, 50)])
        connectors, node_features = sample_topology_along_segment(
            geom, spatial_index, sample_interval_m=25.0, tolerance_m=10.0
        )

        # At least one sample should have degree > 1 (the junction)
        degrees = [node_features[nid] for _, nid in connectors]
        assert max(degrees) > 1, f"Expected junction detection, got degrees: {degrees}"

    def test_t_junction_detection(self):
        """Sampling should detect branches meeting at an interior point.

        The topology spatial index clusters segment ENDPOINTS. When two branch
        segments have endpoints at the same point (200,0) that's on the main
        road's interior, sampling the main road picks up that cluster (degree=2).

        Note: The cluster degree is 2 (branches only), not 3 (branches + main),
        because the main road has no endpoint at (200,0). Full T-junction degree
        would require mid-segment crossing detection (future enhancement).
        """
        # Build a network with a T-junction at approximately (200, 0)
        gdf = gpd.GeoDataFrame(
            {
                "id": ["main", "branch_n", "branch_s"],
                "geometry": [
                    LineString([(0, 0), (400, 0)]),  # Main road
                    LineString([(200, 0), (200, 100)]),  # Branch north
                    LineString([(200, 0), (200, -100)]),  # Branch south
                ],
            },
            crs="EPSG:32632",
        )
        _, spatial_index = compute_all_topology(gdf, tolerance_m=5.0, return_spatial_index=True)

        # Sample along the main road
        main_geom = gdf.geometry.iloc[0]
        connectors, node_features = sample_topology_along_segment(
            main_geom, spatial_index, sample_interval_m=50.0, tolerance_m=10.0
        )

        # The junction cluster at (200,0) has degree 2 (branch_n + branch_s endpoints)
        # Sample point near (200,0) should pick up degree > 1
        degrees = [node_features[nid] for _, nid in connectors]
        assert max(degrees) >= 2, (
            f"Branch junction at (200,0) should give degree >= 2, got: {degrees}"
        )

    def test_output_feeds_into_compute_aligned_topology_features(self):
        """Output format should work with compute_aligned_topology_features()."""
        spatial_index = self._make_intersection_index()

        geom = LineString([(0, 50), (100, 50)])
        connectors, node_features = sample_topology_along_segment(
            geom, spatial_index, sample_interval_m=25.0, tolerance_m=10.0
        )

        # Build the connector dict format expected by compute_aligned_topology_features
        seg_to_connectors = {"test_seg": connectors}

        # Should work without error
        result = compute_aligned_topology_features(
            "test_seg", seg_to_connectors, node_features, 0.0, 1.0
        )

        assert "from_degree" in result
        assert "to_degree" in result
        assert "is_dead_end" in result
        assert "is_intersection" in result
        assert "degree_signature" in result

    def test_empty_geometry_returns_empty(self):
        """Empty or None geometry should return empty results."""
        spatial_index = self._make_intersection_index()

        connectors, node_features = sample_topology_along_segment(None, spatial_index)
        assert connectors == []
        assert node_features == {}

    def test_short_segment_only_endpoints(self):
        """A segment shorter than sample_interval should only get endpoints."""
        spatial_index = self._make_intersection_index()

        # 10m segment (shorter than 50m interval)
        geom = LineString([(45, 50), (55, 50)])
        connectors, _ = sample_topology_along_segment(geom, spatial_index, sample_interval_m=50.0)

        fracs = [c[0] for c in connectors]
        assert fracs == [0.0, 1.0]


class TestTargetDegreeSemanticsUnification:
    """Degree-semantics unification: target topology derived from Overture
    connectors (projected onto the target segment) must agree with the ref side.

    Regression for the bug where target degrees came from endpoint-only
    Union-Find clustering: a road passing THROUGH a junction contributes no
    endpoint there, so the junction degree is undercounted and through-junctions
    are missed entirely — making intersection_match / dead_end_match /
    degree_match_score anti-informative on true matches.

    Repro geometry (a '+' crossroads):
        - Horizontal road H runs straight THROUGH the center as one segment.
        - Two vertical stubs (N, S) each END at the center.
    On the ref (Overture) side the center is one connector referenced by all
    three segments -> high degree, is_intersection True. On the target side the
    endpoint-cluster at the center sees only the two stub endpoints (H passes
    through) -> undercounted degree, is_intersection False.
    """

    @staticmethod
    def _build_plus_crossroads():
        """Build ref (Overture w/ connectors) + target (spaghetti) '+' crossroads.

        Returns a dict with everything needed to exercise the ref path, the old
        endpoint-cluster target path, and the new Overture-projected target path.
        """
        from matcher.features.compute import precompute_graphlet_features
        from matcher.features.spatial_context import (
            build_overture_connector_spatial_index,
            find_overture_connectors_for_targets,
            sample_topology_batch,
        )

        # Ref: Overture segments with explicit connectors.
        # Connector ids: cW, cE (H endpoints), cN, cS (stub tips), cC (center).
        ref_gdf = gpd.GeoDataFrame(
            {
                "id": ["H", "N", "S"],
                "geometry": [
                    LineString([(-100, 0), (100, 0)]),  # through road
                    LineString([(0, 0), (0, 100)]),  # north stub (starts at center)
                    LineString([(0, 0), (0, -100)]),  # south stub (starts at center)
                ],
                "connectors": [
                    [
                        {"at": 0.0, "connector_id": "cW"},
                        {"at": 0.5, "connector_id": "cC"},
                        {"at": 1.0, "connector_id": "cE"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "cC"},
                        {"at": 1.0, "connector_id": "cN"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "cC"},
                        {"at": 1.0, "connector_id": "cS"},
                    ],
                ],
            },
            crs="EPSG:32610",
        )

        # Target: same physical geometry but no explicit connectors. The
        # horizontal road is a single unsplit THROUGH segment.
        target_gdf = gpd.GeoDataFrame(
            {
                "id": ["tH", "tN", "tS"],
                "geometry": [
                    LineString([(-100, 0), (100, 0)]),
                    LineString([(0, 0), (0, 100)]),
                    LineString([(0, 0), (0, -100)]),
                ],
            },
            crs="EPSG:32610",
        )

        # Ref graphlet data (connector-based) -> node degrees in the Overture
        # connector id space.
        _, ref_s2c, ref_node_features, _ = precompute_graphlet_features(
            ref_gdf, connectors_column="connectors"
        )

        # Old target path: endpoint-cluster spatial index -> synthetic connectors.
        _, target_topo_index = compute_all_topology(
            target_gdf, id_column="id", return_spatial_index=True
        )
        target_geoms_by_id = {
            str(target_gdf["id"].iloc[i]): target_gdf.geometry.iloc[i]
            for i in range(len(target_gdf))
        }
        old_connectors, old_node_features = sample_topology_batch(
            list(target_geoms_by_id.values()),
            list(target_geoms_by_id.keys()),
            target_topo_index,
        )

        # New target path: Overture connectors projected onto target segments,
        # scored against the SAME ref node degrees.
        ref_geoms_by_id = {
            str(ref_gdf["id"].iloc[i]): ref_gdf.geometry.iloc[i] for i in range(len(ref_gdf))
        }
        conn_index = build_overture_connector_spatial_index(ref_s2c, ref_geoms_by_id)
        overture_connectors = find_overture_connectors_for_targets(target_geoms_by_id, conn_index)

        return {
            "ref_s2c": ref_s2c,
            "ref_node_features": ref_node_features,
            "old_connectors": old_connectors,
            "old_node_features": old_node_features,
            "overture_connectors": overture_connectors,
        }

    def test_center_connector_high_degree_on_ref(self):
        """Sanity: the shared center connector has degree 4 in the ref graph."""
        ctx = self._build_plus_crossroads()
        # The north stub starts (frac 0.0) at the center connector.
        ref_topo = compute_aligned_topology_features(
            "N", ctx["ref_s2c"], ctx["ref_node_features"], 0.0, 1.0
        )
        # Center touches W, E, N, S -> degree 4; north tip is a dead end.
        assert ref_topo["from_degree"] == 4
        assert ref_topo["is_intersection"] is True

    def test_old_endpoint_cluster_path_undercounts(self):
        """Documents the bug: endpoint clustering misses the through-junction."""
        ctx = self._build_plus_crossroads()
        old_topo = compute_aligned_topology_features(
            "tN", ctx["old_connectors"], ctx["old_node_features"], 0.0, 1.0
        )
        # Only the two stub endpoints cluster at the center (H passes through),
        # so the center reads as degree 2 and the stub is NOT an intersection.
        assert old_topo["from_degree"] == 2
        assert old_topo["is_intersection"] is False

    def test_overture_projected_target_agrees_with_ref(self):
        """The fix: target topology from projected Overture connectors matches ref."""
        ctx = self._build_plus_crossroads()

        ref_topo = compute_aligned_topology_features(
            "N", ctx["ref_s2c"], ctx["ref_node_features"], 0.0, 1.0
        )
        new_topo = compute_aligned_topology_features(
            "tN", ctx["overture_connectors"], ctx["ref_node_features"], 0.0, 1.0
        )

        # Degree at the shared center now matches the ref side exactly.
        assert new_topo["from_degree"] == ref_topo["from_degree"] == 4
        # Both sides agree the stub touches an intersection.
        assert new_topo["is_intersection"] is True
        assert new_topo["is_intersection"] == ref_topo["is_intersection"]
        assert new_topo["is_dead_end"] == ref_topo["is_dead_end"]

    def test_reversed_alignment_with_projected_overture_degrees(self):
        """Combined test: projected-Overture target degrees + is_reversed swap.

        End-to-end through the real pipeline (prepare_worker_data ->
        _compute_feature_chunk), so it exercises the actual interaction between
        the degree-semantics unification (this PR) and the orientation fix
        (is_reversed swap from PR #251) in compute.py.

        Geometry: ref segment H runs from a 4-way crossroads center (cC,
        degree 4) to a dead end (cE, degree 1) -> asymmetric (from=4, to=1).
        The target copy tH is digitized in the REVERSE direction (dead end
        first). The projected-Overture derivation reads degrees in the
        target's own coordinate order (1 at coord[0], 4 at coord[-1]); the
        downstream is_reversed swap must then re-pair them physically so
        from_degree_target matches the degree at the ref's FROM end (4).
        """
        from matcher.blocking.spatial_index import CandidatePair
        from matcher.features.pipeline import prepare_worker_data
        from matcher.matching.ml import _compute_feature_chunk, _init_worker

        # Ref: crossroads at (0, 0) with four arms; H is the eastbound arm
        # ending at a dead end cE. cC is referenced by all four arms -> degree 4.
        ref_gdf = gpd.GeoDataFrame(
            {
                "id": ["H", "N", "S", "W"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),  # center -> dead end
                    LineString([(0, 0), (0, 100)]),
                    LineString([(0, 0), (0, -100)]),
                    LineString([(0, 0), (-100, 0)]),
                ],
                "connectors": [
                    [
                        {"at": 0.0, "connector_id": "cC"},
                        {"at": 1.0, "connector_id": "cE"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "cC"},
                        {"at": 1.0, "connector_id": "cN"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "cC"},
                        {"at": 1.0, "connector_id": "cS"},
                    ],
                    [
                        {"at": 0.0, "connector_id": "cC"},
                        {"at": 1.0, "connector_id": "cW"},
                    ],
                ],
            },
            crs="EPSG:32610",
        )

        # Target: same physical roads; tH digitized REVERSED (dead end first).
        # The arms are included so the candidates-only ref connector graph
        # sees all four segments at cC (degree 4).
        target_gdf = gpd.GeoDataFrame(
            {
                "id": ["tH", "tN", "tS", "tW"],
                "geometry": [
                    LineString([(100, 0), (0, 0)]),  # REVERSED vs ref H
                    LineString([(0, 0), (0, 100)]),
                    LineString([(0, 0), (0, -100)]),
                    LineString([(0, 0), (-100, 0)]),
                ],
            },
            crs="EPSG:32610",
        )

        candidates = [
            CandidatePair(
                ref_id=r,
                ref_idx=ri,
                target_id=t,
                target_idx=ti,
                distance_estimate=0.0,
                heading_diff=0.0,
            )
            for r, ri, t, ti in [
                ("H", 0, "tH", 0),
                ("N", 1, "tN", 1),
                ("S", 2, "tS", 2),
                ("W", 3, "tW", 3),
            ]
        ]

        result = prepare_worker_data(
            candidates=candidates,
            reference=ref_gdf,
            target=target_gdf,
            n_jobs=1,
        )

        # Sanity: the pipeline's alignment must flag tH as reversed.
        h_alignment = result.worker_data["alignments"].get((0, 0))
        assert h_alignment is not None and h_alignment.is_reversed is True
        # Sanity: projected Overture connectors reached the target segment,
        # so the topology block takes the new projected-Overture branch.
        assert result.worker_data["target_overture_connectors"].get("tH")

        _init_worker(result.worker_data)
        features_list, _errors = _compute_feature_chunk([(0, 0)])
        feats = features_list[0]
        assert feats is not None and not feats.get("_error")

        # Ref: from end at the crossroads center (degree 4), to end dead (1).
        assert feats["from_degree_ref"] == 4
        assert feats["to_degree_ref"] == 1
        # Target degrees must pair with the SAME physical ends as ref, i.e.
        # the raw coordinate-order derivation (1, 4) swapped by is_reversed.
        assert feats["from_degree_target"] == feats["from_degree_ref"] == 4
        assert feats["to_degree_target"] == feats["to_degree_ref"] == 1
        # Derived agreement flags follow.
        assert feats["intersection_match"] == 1.0
        assert feats["dead_end_match"] == 1.0
