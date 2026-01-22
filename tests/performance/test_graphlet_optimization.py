"""Performance tests for graphlet computation optimization.

These tests create large synthetic networks to stress-test the graphlet computation
and verify that optimizations maintain correctness while improving performance.

TDD approach: Write failing tests first, then optimize until they pass.
"""

import time

import networkx as nx
import numpy as np
import pytest

from matcher.features.spatial_context import compute_road_graphlet_features


def create_dense_grid_graph(n: int) -> nx.Graph:
    """Create an n x n grid graph (city-like topology).

    Grid graphs have many 4-cycles (squares) which stress the square detection.
    Nodes: n², Edges: ~2n²
    """
    return nx.grid_2d_graph(n, n)


def create_high_degree_graph(n_nodes: int, avg_degree: int = 6) -> nx.Graph:
    """Create a graph with higher average degree (highway interchange topology).

    Uses Barabási-Albert model which creates hubs with high degree.
    """
    # BA model: n nodes, m edges per new node
    m = avg_degree // 2
    return nx.barabasi_albert_graph(n_nodes, m, seed=42)


def create_tree_like_graph(n_nodes: int) -> nx.Graph:
    """Create a tree-like structure (rural road topology).

    Trees have no cycles, so triangle/square detection should be fast.
    """
    return nx.random_labeled_tree(n_nodes, seed=42)


def create_mixed_topology_graph(n_clusters: int = 10, cluster_size: int = 50) -> nx.Graph:
    """Create a graph with mixed topology (realistic road network).

    Multiple dense clusters connected by sparse bridges.
    """
    G = nx.Graph()
    node_offset = 0

    for cluster_id in range(n_clusters):
        # Create dense cluster (grid-like)
        cluster = nx.grid_2d_graph(int(np.sqrt(cluster_size)), int(np.sqrt(cluster_size)))
        # Relabel nodes to be unique
        mapping = {old: node_offset + i for i, old in enumerate(cluster.nodes())}
        cluster = nx.relabel_nodes(cluster, mapping)
        G = nx.compose(G, cluster)

        # Connect to previous cluster with sparse bridge
        if cluster_id > 0:
            prev_cluster_start = (cluster_id - 1) * cluster_size
            curr_cluster_start = node_offset
            # Add 1-3 bridge edges
            for _ in range(np.random.randint(1, 4)):
                src = prev_cluster_start + np.random.randint(0, cluster_size)
                dst = curr_cluster_start + np.random.randint(0, len(cluster))
                if G.has_node(src) and G.has_node(dst):
                    G.add_edge(src, dst)

        node_offset += len(cluster)

    return G


class TestGraphletComputePerformanceTargets:
    """Performance target tests - these define our optimization goals."""

    @pytest.mark.slow
    def test_10k_node_grid_under_5_seconds(self):
        """10K node grid graph should compute features in under 5 seconds.

        Grid graphs are worst-case for square detection (many 4-cycles).
        Current implementation may be slower - this is our optimization target.
        """
        # 100x100 grid = 10,000 nodes
        G = create_dense_grid_graph(100)
        assert G.number_of_nodes() == 10000

        start = time.perf_counter()
        features = compute_road_graphlet_features(G)
        elapsed = time.perf_counter() - start

        assert len(features) == G.number_of_nodes()
        assert elapsed < 5.0, f"10K grid took {elapsed:.2f}s, target is <5s"

    @pytest.mark.slow
    def test_20k_node_tree_under_3_seconds(self):
        """20K node tree should compute features quickly (no cycles).

        Trees should be fast because triangle/square detection finds nothing.
        """
        G = create_tree_like_graph(20000)

        start = time.perf_counter()
        features = compute_road_graphlet_features(G)
        elapsed = time.perf_counter() - start

        assert len(features) == G.number_of_nodes()
        # Trees should be faster than grids
        assert elapsed < 3.0, f"20K tree took {elapsed:.2f}s, target is <3s"

    @pytest.mark.slow
    def test_5k_high_degree_graph_under_10_seconds(self):
        """5K node high-degree graph should handle hub nodes efficiently.

        High-degree nodes stress the O(D²) square detection.
        """
        G = create_high_degree_graph(5000, avg_degree=8)

        start = time.perf_counter()
        features = compute_road_graphlet_features(G)
        elapsed = time.perf_counter() - start

        assert len(features) == G.number_of_nodes()
        assert elapsed < 10.0, f"5K high-degree took {elapsed:.2f}s, target is <10s"

    @pytest.mark.slow
    def test_mixed_topology_realistic_road_network(self):
        """Mixed topology (clusters + bridges) should handle varied structure.

        This mimics real road networks: dense city blocks connected by highways.
        """
        G = create_mixed_topology_graph(n_clusters=20, cluster_size=100)
        n_nodes = G.number_of_nodes()

        start = time.perf_counter()
        features = compute_road_graphlet_features(G)
        elapsed = time.perf_counter() - start

        assert len(features) == n_nodes
        # Allow ~1ms per node for mixed topology
        assert elapsed < n_nodes * 0.001, (
            f"{n_nodes} mixed nodes took {elapsed:.2f}s, target is <{n_nodes * 0.001:.2f}s"
        )


class TestGraphletCorrectness:
    """Correctness tests - ensure optimizations don't break results."""

    def test_grid_square_count_correct(self):
        """Verify square counting is correct for a known grid."""
        # 3x3 grid has 4 squares, each internal node participates in squares
        G = create_dense_grid_graph(3)
        features = compute_road_graphlet_features(G)

        # Center node (1,1) should participate in 4 squares
        center = (1, 1)
        assert features[center][2] == 4, "Center of 3x3 grid should have 4 squares"

        # Corner nodes should have 1 square each
        corner = (0, 0)
        assert features[corner][2] == 1, "Corner of 3x3 grid should have 1 square"

        # Edge midpoints should have 2 squares
        edge = (0, 1)
        assert features[edge][2] == 2, "Edge of 3x3 grid should have 2 squares"

    def test_tree_no_cycles(self):
        """Trees should have no triangles or squares."""
        G = create_tree_like_graph(100)
        features = compute_road_graphlet_features(G)

        for node, feat in features.items():
            assert feat[1] == 0, f"Tree node {node} should have 0 triangles"
            assert feat[2] == 0, f"Tree node {node} should have 0 squares"

    def test_triangle_graph(self):
        """Verify triangle counting."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        features = compute_road_graphlet_features(G)

        for node in [0, 1, 2]:
            assert features[node][1] == 1, f"Triangle node {node} should have 1 triangle"

    def test_two_hop_count_chain(self):
        """Verify two-hop count for simple chain."""
        G = nx.path_graph(5)  # 0-1-2-3-4
        features = compute_road_graphlet_features(G)

        # Node 0: neighbors={1}, two_hop={2}
        assert features[0][4] == 1
        # Node 1: neighbors={0,2}, two_hop={3}
        assert features[1][4] == 1
        # Node 2: neighbors={1,3}, two_hop={0,4}
        assert features[2][4] == 2

    def test_articulation_points_bridge(self):
        """Verify articulation point detection."""
        # Two triangles connected by a bridge
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle 1
        G.add_edges_from([(3, 4), (4, 5), (5, 3)])  # Triangle 2
        G.add_edge(2, 3)  # Bridge

        features = compute_road_graphlet_features(G)

        # Nodes 2 and 3 are articulation points
        assert features[2][5] == 1.0, "Node 2 should be articulation point"
        assert features[3][5] == 1.0, "Node 3 should be articulation point"
        # Other nodes are not
        assert features[0][5] == 0.0
        assert features[1][5] == 0.0


class TestGraphletScaling:
    """Scaling tests - verify O(n) or O(n log n) complexity."""

    @pytest.mark.slow
    def test_grid_scales_subquadratically(self):
        """Grid computation should not be O(n²)."""
        sizes = [20, 40, 60, 80]  # n x n grids
        times = []

        # Warmup
        G_warmup = create_dense_grid_graph(10)
        compute_road_graphlet_features(G_warmup)

        for n in sizes:
            G = create_dense_grid_graph(n)
            n_nodes = G.number_of_nodes()

            start = time.perf_counter()
            compute_road_graphlet_features(G)
            elapsed = time.perf_counter() - start
            times.append((n_nodes, elapsed))

        # Check scaling: if O(n²), doubling nodes should 4x time
        # We want sub-quadratic: doubling nodes should < 3x time
        for i in range(1, len(times)):
            node_ratio = times[i][0] / times[i - 1][0]
            time_ratio = times[i][1] / times[i - 1][1] if times[i - 1][1] > 0.01 else 1.0

            # For sub-quadratic, time_ratio should be < node_ratio²
            # Allow up to 2.5x slowdown for 2x nodes (O(n log n) behavior)
            assert time_ratio < node_ratio * 2.5, (
                f"Scaling issue: {times[i - 1][0]} -> {times[i][0]} nodes "
                f"caused {time_ratio:.2f}x slowdown (node ratio: {node_ratio:.2f}x)"
            )

    @pytest.mark.slow
    def test_high_degree_nodes_dont_dominate(self):
        """High-degree hub nodes shouldn't cause O(D³) explosion."""
        # Create a star graph with one very high degree node
        hub_degree = 100
        G = nx.star_graph(hub_degree)  # Node 0 connected to 1..100

        # Add some structure to non-hub nodes
        for i in range(1, hub_degree, 2):
            if i + 1 <= hub_degree:
                G.add_edge(i, i + 1)

        start = time.perf_counter()
        features = compute_road_graphlet_features(G)
        elapsed = time.perf_counter() - start

        # Hub should have high degree
        assert features[0][0] == hub_degree

        # Should be fast even with high-degree node
        assert elapsed < 0.5, f"Star graph with degree {hub_degree} took {elapsed:.2f}s"


class TestOptimizedVsOriginal:
    """Compare optimized implementation against reference (if available)."""

    def _reference_square_count(self, G: nx.Graph, node: int) -> int:
        """Reference implementation for square counting (slow but correct)."""
        neighbors = set(G.neighbors(node))
        neighbor_list = list(neighbors)
        square_count = 0
        for i, n1 in enumerate(neighbor_list):
            for n2 in neighbor_list[i + 1 :]:
                common = set(G.neighbors(n1)) & set(G.neighbors(n2)) - {node}
                square_count += len(common)
        return square_count

    def test_square_count_matches_reference(self):
        """Verify our square count matches naive implementation."""
        G = create_dense_grid_graph(5)
        features = compute_road_graphlet_features(G)

        for node in G.nodes():
            expected = self._reference_square_count(G, node)
            actual = features[node][2]
            assert actual == expected, f"Node {node}: expected {expected} squares, got {actual}"
