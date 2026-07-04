"""Tests for graphlet features in spatial_context.py."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString


class TestBuildInferredGraph:
    """Tests for build_inferred_graph function."""

    def test_empty_geodataframe(self):
        """Empty GeoDataFrame returns empty graph."""
        from matcher.features.spatial_context import build_inferred_graph

        gdf = gpd.GeoDataFrame({"id": [], "geometry": []}, crs="EPSG:4326")
        G, seg_to_start, seg_to_end = build_inferred_graph(gdf, "id")

        assert G.n_nodes == 0
        assert G.n_edges == 0
        assert seg_to_start == {}
        assert seg_to_end == {}

    def test_single_segment(self):
        """Single segment creates 2 nodes and 1 edge."""
        from matcher.features.spatial_context import build_inferred_graph

        gdf = gpd.GeoDataFrame(
            {"id": ["r1"], "geometry": [LineString([(0, 0), (10, 0)])]},
            crs="EPSG:4326",
        )
        G, seg_to_start, seg_to_end = build_inferred_graph(gdf, "id", tolerance_m=1.0)

        assert G.n_nodes == 2
        assert G.n_edges == 1
        assert "r1" in seg_to_start
        assert "r1" in seg_to_end

    @pytest.mark.parametrize(
        "gap_meters,tolerance,expected_nodes,expected_edges",
        [
            # 6 endpoints total: r1 start, r1 end, r2 start (=r3 start), r2 end, r3 end
            # Roads 2 and 3 share exact start point, so always clustered = 5 unique endpoints
            (5.0, 0.5, 5, 3),  # Gap=5m, tolerance=0.5m: no additional clustering
            (5.0, 10.0, 4, 3),  # Gap=5m, tolerance=10m: r1 end clusters with r2/r3 start
            (5.0, 20.0, 4, 3),  # Gap=5m, tolerance=20m: same clustering
        ],
    )
    def test_tolerance_affects_clustering(
        self, gap_meters, tolerance, expected_nodes, expected_edges
    ):
        """Tolerance parameter controls endpoint clustering."""
        from matcher.features.spatial_context import build_inferred_graph

        # Three roads meeting at an intersection (with small gaps)
        # Using projected coordinates (EPSG:32632 - UTM zone 32N) so units are meters
        lines = [
            LineString([(500000, 0), (500100, 0)]),  # Road 1: 100m long
            LineString([(500100 + gap_meters, 0), (500200, 0)]),  # Road 2 (gap from Road 1)
            LineString(
                [(500100 + gap_meters, 0), (500100 + gap_meters, 100)]
            ),  # Road 3 (branches from Road 2 start)
        ]
        gdf = gpd.GeoDataFrame(
            {"id": ["r1", "r2", "r3"], "geometry": lines},
            crs="EPSG:32632",  # Use projected CRS - units are meters
        )
        G, _, _ = build_inferred_graph(gdf, "id", tolerance_m=tolerance)

        assert G.n_nodes == expected_nodes
        assert G.n_edges == expected_edges

    def test_t_intersection_topology(self):
        """T-intersection creates correct topology."""
        from matcher.features.spatial_context import build_inferred_graph

        # T-intersection: main road + side street
        lines = [
            LineString([(0, 0), (10, 0)]),  # Main road west
            LineString([(10, 0), (20, 0)]),  # Main road east
            LineString([(10, 0), (10, 10)]),  # Side street
        ]
        gdf = gpd.GeoDataFrame(
            {"id": ["main_w", "main_e", "side"], "geometry": lines},
            crs="EPSG:4326",
        )
        G, seg_to_start, seg_to_end = build_inferred_graph(gdf, "id", tolerance_m=1.0)

        # Should have 4 nodes: 3 dead-ends + 1 intersection
        assert G.n_nodes == 4
        assert G.n_edges == 3

        # Find the intersection node (degree 3)
        intersection_nodes = [n for n in G.node_ids if G.degree(n) == 3]
        assert len(intersection_nodes) == 1

    def test_loop_topology(self):
        """Closed loop creates correct graph structure."""
        from matcher.features.spatial_context import build_inferred_graph

        # Square loop made of 4 segments
        lines = [
            LineString([(0, 0), (10, 0)]),  # Bottom
            LineString([(10, 0), (10, 10)]),  # Right
            LineString([(10, 10), (0, 10)]),  # Top
            LineString([(0, 10), (0, 0)]),  # Left
        ]
        gdf = gpd.GeoDataFrame(
            {"id": ["bottom", "right", "top", "left"], "geometry": lines},
            crs="EPSG:4326",
        )
        G, _, _ = build_inferred_graph(gdf, "id", tolerance_m=1.0)

        # 4 corners = 4 nodes, 4 edges
        assert G.n_nodes == 4
        assert G.n_edges == 4

        # All nodes should have degree 2 (2 roads meeting at each corner)
        degrees = [G.degree(n) for n in G.node_ids]
        assert all(d == 2 for d in degrees)


class TestComputeRoadGraphletFeatures:
    """Tests for compute_road_graphlet_features function."""

    def test_empty_graph(self):
        """Empty graph returns empty features dict."""
        from scipy.sparse import csr_matrix

        from matcher.features.spatial_context import compute_road_graphlet_features
        from matcher.topology.sparse_graph import SparseGraph

        G = SparseGraph(
            adjacency=csr_matrix((0, 0), dtype=np.int32),
            node_ids=[],
            node_to_idx={},
        )
        features = compute_road_graphlet_features(G)

        assert features == {}

    def test_single_node(self):
        """Single isolated node has degree 0."""
        from matcher.features.spatial_context import compute_road_graphlet_features
        from matcher.topology.sparse_graph import build_graph_from_edges

        # Create graph with single isolated node
        G = build_graph_from_edges([], node_attrs={0: {}})
        features = compute_road_graphlet_features(G)

        assert len(features) == 1
        assert features[0][0] == 0  # degree
        assert features[0][1] == 0  # triangles

    @pytest.mark.parametrize(
        "edges,node,expected_degree",
        [
            ([(0, 1)], 0, 1),  # Dead end
            ([(0, 1), (0, 2)], 0, 2),  # Two-way intersection
            ([(0, 1), (0, 2), (0, 3)], 0, 3),  # Three-way intersection
            ([(0, 1), (0, 2), (0, 3), (0, 4)], 0, 4),  # Four-way intersection
        ],
    )
    def test_node_degree(self, edges, node, expected_degree):
        """Node degree is computed correctly."""
        from matcher.features.spatial_context import compute_road_graphlet_features
        from matcher.topology.sparse_graph import build_graph_from_edges

        G = build_graph_from_edges(edges)
        features = compute_road_graphlet_features(G)

        assert features[node][0] == expected_degree

    def test_triangle_detection(self):
        """Triangles are detected correctly."""
        from matcher.features.spatial_context import compute_road_graphlet_features
        from matcher.topology.sparse_graph import build_graph_from_edges

        # Triangle: 0-1-2-0
        G = build_graph_from_edges([(0, 1), (1, 2), (2, 0)])
        features = compute_road_graphlet_features(G)

        # All nodes participate in 1 triangle
        for node in [0, 1, 2]:
            assert features[node][1] == 1  # triangles

    def test_square_detection(self):
        """4-cycles (squares) are detected correctly."""
        from matcher.features.spatial_context import compute_road_graphlet_features
        from matcher.topology.sparse_graph import build_graph_from_edges

        # Square: 0-1-2-3-0 (no diagonals)
        G = build_graph_from_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
        features = compute_road_graphlet_features(G)

        # All corner nodes participate in squares
        for node in [0, 1, 2, 3]:
            assert features[node][2] > 0  # squares

    def test_articulation_point_detection(self):
        """Articulation points are correctly identified."""
        from matcher.features.spatial_context import compute_road_graphlet_features
        from matcher.topology.sparse_graph import build_graph_from_edges

        # Bridge topology: 0-1-2 where 1 is articulation point
        G = build_graph_from_edges([(0, 1), (1, 2)])
        features = compute_road_graphlet_features(G)

        # Node 1 is an articulation point (bridges the two ends)
        assert features[1][5] == 1.0  # is_articulation
        # Endpoints are not articulation points
        assert features[0][5] == 0.0
        assert features[2][5] == 0.0

    def test_two_hop_count(self):
        """Two-hop neighbor count is computed correctly."""
        from matcher.features.spatial_context import compute_road_graphlet_features
        from matcher.topology.sparse_graph import build_graph_from_edges

        # Chain: 0-1-2-3
        G = build_graph_from_edges([(0, 1), (1, 2), (2, 3)])
        features = compute_road_graphlet_features(G)

        # Node 0: neighbors={1}, two-hop={2}
        assert features[0][4] == 1  # two_hop_count

        # Node 1: neighbors={0,2}, two-hop={3}
        assert features[1][4] == 1


class TestGraphletSegmentSimilarity:
    """Tests for graphlet_segment_similarity function."""

    def test_identical_features_perfect_similarity(self):
        """Identical endpoint features give similarity of 1.0."""
        from matcher.features.spatial_context import graphlet_segment_similarity

        # Create identical features for both segments
        ref_features = {0: np.array([2.0, 0.0, 0.0, 0.0, 1.0, 0.0])}
        target_features = {1: np.array([2.0, 0.0, 0.0, 0.0, 1.0, 0.0])}

        ref_seg_to_nodes = ({"r1": 0}, {"r1": 0})  # Same node at both ends (loop)
        target_seg_to_nodes = ({"t1": 1}, {"t1": 1})

        result = graphlet_segment_similarity(
            "r1",
            "t1",
            ref_features,
            target_features,
            ref_seg_to_nodes,
            target_seg_to_nodes,
        )

        assert result["graphlet_similarity"] == 1.0
        assert result["endpoint_degree_similarity"] == 1.0

    def test_different_degrees_lower_similarity(self):
        """Different endpoint degrees result in lower similarity."""
        from matcher.features.spatial_context import graphlet_segment_similarity

        # Reference has degree 4 (4-way intersection)
        ref_features = {0: np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        # Target has degree 1 (dead end)
        target_features = {1: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])}

        ref_seg_to_nodes = ({"r1": 0}, {"r1": 0})
        target_seg_to_nodes = ({"t1": 1}, {"t1": 1})

        result = graphlet_segment_similarity(
            "r1",
            "t1",
            ref_features,
            target_features,
            ref_seg_to_nodes,
            target_seg_to_nodes,
        )

        # Degree difference of 3 maps to 0.0 under the road-appropriate discrete
        # scale (exact=1.0, ±1=0.67, ±2=0.33, larger=0.0). The old linear
        # 1 - |Δdeg|/10 mapping compressed this to 0.7 — a value that could barely
        # move across the whole 1-6 degree range (see _road_degree_similarity).
        assert result["endpoint_degree_similarity"] == pytest.approx(0.0, abs=1e-9)

    def test_missing_segments_return_nan(self):
        """Missing segment IDs -> NaN, not a fabricated perfect/default match.

        SEMANTICS CHANGE: previously this returned default (degree-1) feature
        vectors, which made two segments with UNKNOWN topology score a valid
        (often perfect) similarity. Unknown topology is now undefined -> NaN so it
        passes through to XGBoost as a missing value rather than fake signal.
        """
        from matcher.features.spatial_context import graphlet_segment_similarity

        ref_features = {}  # No features
        target_features = {}

        ref_seg_to_nodes = ({}, {})  # No mappings
        target_seg_to_nodes = ({}, {})

        result = graphlet_segment_similarity(
            "nonexistent_ref",
            "nonexistent_target",
            ref_features,
            target_features,
            ref_seg_to_nodes,
            target_seg_to_nodes,
        )

        assert np.isnan(result["graphlet_similarity"])
        assert np.isnan(result["endpoint_degree_similarity"])

    @pytest.mark.parametrize(
        "ref_start_degree,ref_end_degree,target_start_degree,target_end_degree,expected_orientation",
        [
            (1, 3, 1, 3, "forward"),  # Same orientation
            (1, 3, 3, 1, "reverse"),  # Reversed orientation
            (2, 2, 2, 2, "either"),  # Symmetric
        ],
    )
    def test_orientation_handling(
        self,
        ref_start_degree,
        ref_end_degree,
        target_start_degree,
        target_end_degree,
        expected_orientation,
    ):
        """Best orientation is selected for similarity computation."""
        from matcher.features.spatial_context import graphlet_segment_similarity

        # Create features with specified degrees
        ref_features = {
            0: np.array([ref_start_degree, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1: np.array([ref_end_degree, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        target_features = {
            2: np.array([target_start_degree, 0.0, 0.0, 0.0, 0.0, 0.0]),
            3: np.array([target_end_degree, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }

        ref_seg_to_nodes = ({"r1": 0}, {"r1": 1})
        target_seg_to_nodes = ({"t1": 2}, {"t1": 3})

        result = graphlet_segment_similarity(
            "r1",
            "t1",
            ref_features,
            target_features,
            ref_seg_to_nodes,
            target_seg_to_nodes,
        )

        # Similarity should be high regardless of orientation
        if expected_orientation in ("forward", "reverse"):
            assert result["graphlet_similarity"] == 1.0
        else:  # symmetric
            assert result["graphlet_similarity"] == 1.0


class TestGraphletIntegration:
    """Integration tests for the full graphlet pipeline."""

    def test_end_to_end_graphlet_computation(self):
        """Full pipeline from GeoDataFrame to segment similarity."""
        from matcher.features.spatial_context import (
            build_inferred_graph,
            compute_road_graphlet_features,
            graphlet_segment_similarity,
        )

        # Create two similar road networks
        ref_lines = [
            LineString([(0, 0), (10, 0)]),  # Main road
            LineString([(10, 0), (20, 0)]),  # Continuation
            LineString([(10, 0), (10, 10)]),  # Side street
        ]
        ref_gdf = gpd.GeoDataFrame(
            {"id": ["ref_main", "ref_cont", "ref_side"], "geometry": ref_lines},
            crs="EPSG:4326",
        )

        target_lines = [
            LineString([(0.1, 0.1), (10.1, 0.1)]),  # Slightly offset main road
            LineString([(10.1, 0.1), (20.1, 0.1)]),  # Continuation
            LineString([(10.1, 0.1), (10.1, 10.1)]),  # Side street
        ]
        target_gdf = gpd.GeoDataFrame(
            {"id": ["tgt_main", "tgt_cont", "tgt_side"], "geometry": target_lines},
            crs="EPSG:4326",
        )

        # Build graphs
        ref_G, ref_start, ref_end = build_inferred_graph(ref_gdf, "id", tolerance_m=1.0)
        target_G, target_start, target_end = build_inferred_graph(target_gdf, "id", tolerance_m=1.0)

        # Compute graphlet features
        ref_features = compute_road_graphlet_features(ref_G)
        target_features = compute_road_graphlet_features(target_G)

        # Compare corresponding segments - should have high similarity
        result = graphlet_segment_similarity(
            "ref_main",
            "tgt_main",
            ref_features,
            target_features,
            (ref_start, ref_end),
            (target_start, target_end),
        )

        # Similar topology should give high similarity
        assert result["graphlet_similarity"] > 0.8
        assert result["endpoint_degree_similarity"] > 0.8


class TestRoadDegreeSimilarity:
    """Tests for the rescaled, road-appropriate degree similarity (defect #4).

    The old mapping was sim = 1 - |Δdeg| / 10. Road-network node degrees are
    almost all in 1-6, so that compressed every output into [0.5, 1.0] and piled
    ~44% of pairs at exactly 0.8 (AUC ~0.50). The new discrete mapping spreads the
    signal: exact = 1.0, ±1 = 0.67, ±2 = 0.33, larger = 0.0.
    """

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (3, 3, 1.0),
            (2, 3, 0.67),
            (4, 3, 0.67),
            (1, 3, 0.33),
            (5, 3, 0.33),
            (1, 4, 0.0),
            (2, 6, 0.0),
        ],
    )
    def test_discrete_scale(self, a, b, expected):
        from matcher.features.spatial_context import _road_degree_similarity

        assert _road_degree_similarity(a, b) == pytest.approx(expected, abs=1e-9)

    def test_endpoint_degree_similarity_uses_new_scale(self):
        """endpoint_degree_similarity reflects the discrete mapping, not 1-|Δ|/10."""
        from matcher.features.spatial_context import graphlet_segment_similarity

        # ref degrees (2, 3), target degrees (3, 4): each endpoint is off-by-one.
        ref_features = {
            0: np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1: np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        target_features = {
            2: np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            3: np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        result = graphlet_segment_similarity(
            "r1",
            "t1",
            ref_features,
            target_features,
            ({"r1": 0}, {"r1": 1}),
            ({"t1": 2}, {"t1": 3}),
        )
        # Both endpoints off-by-one -> 0.67 each -> average 0.67.
        assert result["endpoint_degree_similarity"] == pytest.approx(0.67, abs=1e-9)


class TestUnknownTopologyReturnsNaN:
    """Unknown endpoint topology -> NaN, not a fabricated match (defect #2)."""

    def test_alignment_missing_connectors_returns_nan(self):
        """graphlet_similarity_with_alignment returns NaN when a lookup misses."""
        from matcher.features.spatial_context import graphlet_similarity_with_alignment

        ref_features = {0: np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        target_features = {1: np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])}

        # Ref has connectors; target's connector map is empty -> target lookup misses.
        ref_seg_to_connectors = {"r1": [(0.0, 0), (1.0, 0)]}
        target_seg_to_connectors: dict = {}

        result = graphlet_similarity_with_alignment(
            "r1",
            "t1",
            ref_features,
            target_features,
            ref_seg_to_connectors,
            target_seg_to_connectors,
        )
        assert np.isnan(result["graphlet_similarity"])
        assert np.isnan(result["endpoint_degree_similarity"])

    def test_alignment_both_known_is_finite(self):
        """When both sides resolve, similarity is a finite value (not NaN)."""
        from matcher.features.spatial_context import graphlet_similarity_with_alignment

        ref_features = {0: np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        target_features = {1: np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        ref_seg_to_connectors = {"r1": [(0.0, 0), (1.0, 0)]}
        target_seg_to_connectors = {"t1": [(0.0, 1), (1.0, 1)]}

        result = graphlet_similarity_with_alignment(
            "r1",
            "t1",
            ref_features,
            target_features,
            ref_seg_to_connectors,
            target_seg_to_connectors,
        )
        assert not np.isnan(result["graphlet_similarity"])
        assert result["endpoint_degree_similarity"] == pytest.approx(1.0, abs=1e-9)


class TestClusteringUndefinedIsNaN:
    """Clustering coefficient is NaN (not 0.0) for degree < 2 nodes (defect #3)."""

    def test_degree_lt_2_nodes_are_nan(self):
        from matcher.features.spatial_context import build_inferred_graph
        from matcher.topology.sparse_graph import compute_clustering

        # A path graph: two dead-ends (degree 1) and one middle node (degree 2).
        lines = [
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
        ]
        gdf = gpd.GeoDataFrame({"id": ["a", "b"], "geometry": lines}, crs="EPSG:32632")
        G, _, _ = build_inferred_graph(gdf, "id", tolerance_m=1.0)

        clustering = compute_clustering(G)
        for node in G.node_ids:
            if G.degree(node) < 2:
                assert np.isnan(clustering[node]), (
                    f"degree-{G.degree(node)} node should be NaN, got {clustering[node]}"
                )
            else:
                # degree-2 node with no triangle -> defined, equals 0.0
                assert clustering[node] == 0.0


class TestGraphletDistributionSmoke:
    """Guard against re-degeneration: features must not collapse to one value."""

    @staticmethod
    def _grid(offset: float, id_prefix: str) -> gpd.GeoDataFrame:
        """Build a 4x4 grid of road segments (24 segments, varied node degrees)."""
        lines = []
        ids = []
        for j in range(4):
            for i in range(3):
                lines.append(LineString([(i + offset, j + offset), (i + 1 + offset, j + offset)]))
                ids.append(f"{id_prefix}_h_{i}_{j}")
        for i in range(4):
            for j in range(3):
                lines.append(LineString([(i + offset, j + offset), (i + offset, j + 1 + offset)]))
                ids.append(f"{id_prefix}_v_{i}_{j}")
        return gpd.GeoDataFrame({"id": ids, "geometry": lines}, crs="EPSG:32632")

    def test_similarity_not_single_valued(self):
        from matcher.features.spatial_context import (
            build_inferred_graph,
            compute_road_graphlet_features,
            graphlet_segment_similarity,
        )

        ref_gdf = self._grid(0.0, "ref")
        target_gdf = self._grid(0.05, "tgt")

        ref_G, ref_start, ref_end = build_inferred_graph(ref_gdf, "id", tolerance_m=0.5)
        tgt_G, tgt_start, tgt_end = build_inferred_graph(target_gdf, "id", tolerance_m=0.5)
        ref_feats = compute_road_graphlet_features(ref_G)
        tgt_feats = compute_road_graphlet_features(tgt_G)

        graphlet_vals = []
        degree_vals = []
        # Score a diagonal-ish cross section of pairs (all-pairs would be 576).
        for r_id in ref_gdf["id"]:
            for t_id in target_gdf["id"]:
                res = graphlet_segment_similarity(
                    r_id,
                    t_id,
                    ref_feats,
                    tgt_feats,
                    (ref_start, ref_end),
                    (tgt_start, tgt_end),
                )
                graphlet_vals.append(round(res["graphlet_similarity"], 4))
                degree_vals.append(round(res["endpoint_degree_similarity"], 4))

        def modal_fraction(vals):
            import collections

            counts = collections.Counter(vals)
            return max(counts.values()) / len(vals)

        assert modal_fraction(graphlet_vals) < 0.9, (
            f"graphlet_similarity is >90% single-valued: {modal_fraction(graphlet_vals):.2%}"
        )
        assert modal_fraction(degree_vals) < 0.9, (
            f"endpoint_degree_similarity is >90% single-valued: {modal_fraction(degree_vals):.2%}"
        )
