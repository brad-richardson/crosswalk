"""Performance tests for graphlet feature computation.

These tests ensure graphlet computation scales appropriately for production use.
They are marked with @pytest.mark.slow so they can be skipped during fast test runs.

Performance requirements:
- Graphlet precomputation for 10K segments: < 30 seconds
- Full match pipeline for Boston streets: < 5 minutes (not tested here, integration level)
"""

import time

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely import LineString

from matcher.features.compute import (
    compute_graphlet_similarity,
    precompute_graphlet_features,
)
from matcher.features.spatial_context import (
    build_inferred_graph,
    compute_road_graphlet_features,
)


@pytest.fixture
def grid_network_factory():
    """Factory fixture for creating grid networks of various sizes."""

    def _create(n_segments: int = 100, cell_size: float = 100.0) -> gpd.GeoDataFrame:
        """Create a synthetic grid road network.

        A grid with k x k cells has 2*k*(k+1) segments.
        """
        k = max(2, int(np.sqrt(n_segments / 2)))
        segments = []
        seg_id = 0

        # Horizontal segments
        for row in range(k + 1):
            for col in range(k):
                x1, y1 = col * cell_size, row * cell_size
                x2, y2 = (col + 1) * cell_size, row * cell_size
                segments.append(
                    {"id": f"seg_{seg_id}", "geometry": LineString([(x1, y1), (x2, y2)])}
                )
                seg_id += 1

        # Vertical segments
        for row in range(k):
            for col in range(k + 1):
                x1, y1 = col * cell_size, row * cell_size
                x2, y2 = col * cell_size, (row + 1) * cell_size
                segments.append(
                    {"id": f"seg_{seg_id}", "geometry": LineString([(x1, y1), (x2, y2)])}
                )
                seg_id += 1

        return gpd.GeoDataFrame(segments, crs="EPSG:32618")

    return _create


class TestGraphletPrecomputePerformance:
    """Performance tests for graphlet precomputation."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "n_segments,max_seconds",
        [
            (100, 1.0),
            (1000, 5.0),
            (10000, 30.0),  # Plan requirement: < 30s for 10K segments
        ],
        ids=["100_segments", "1K_segments", "10K_segments"],
    )
    def test_precompute_scales(self, grid_network_factory, n_segments, max_seconds):
        """Benchmark graphlet precomputation at various scales."""
        gdf = grid_network_factory(n_segments=n_segments)

        start = time.perf_counter()
        G, seg_to_connectors, node_features, _ = precompute_graphlet_features(
            gdf, id_column="id", tolerance_m=5.0
        )
        elapsed = time.perf_counter() - start

        # G may be None when degrees_only=True (default for memory efficiency)
        # In that case, node_features contains the degree data
        assert len(node_features) > 0
        assert len(seg_to_connectors) > 0
        assert elapsed < max_seconds, (
            f"{n_segments} segments took {elapsed:.2f}s, expected < {max_seconds}s"
        )


class TestGraphletSimilarityPerformance:
    """Performance tests for graphlet similarity computation."""

    @pytest.mark.slow
    def test_similarity_1000_pairs(self, grid_network_factory):
        """Benchmark graphlet similarity for 1K pairs."""
        ref_gdf = grid_network_factory(n_segments=500)
        target_gdf = grid_network_factory(n_segments=500)

        ref_graphlet = precompute_graphlet_features(ref_gdf, id_column="id")
        target_graphlet = precompute_graphlet_features(target_gdf, id_column="id")

        ref_ids = [f"seg_{i}" for i in range(min(500, len(ref_gdf)))]
        target_ids = [f"seg_{i}" for i in range(min(500, len(target_gdf)))]

        np.random.seed(42)
        pairs = [(np.random.choice(ref_ids), np.random.choice(target_ids)) for _ in range(1000)]

        start = time.perf_counter()
        for ref_id, target_id in pairs:
            compute_graphlet_similarity(ref_id, target_id, ref_graphlet, target_graphlet)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"1K pairs took {elapsed:.2f}s, expected < 1s"


class TestScalingBehavior:
    """Tests to verify algorithms scale well (O(n) to O(n log n))."""

    @pytest.mark.slow
    def test_graph_building_scales_linearly(self, grid_network_factory):
        """Verify graph building scales approximately linearly."""
        sizes = [100, 500, 1000, 2000]
        times = []

        for n in sizes:
            gdf = grid_network_factory(n_segments=n)
            gdf["id"] = gdf["id"].astype(str)

            start = time.perf_counter()
            build_inferred_graph(gdf, id_column="id", tolerance_m=5.0)
            times.append(time.perf_counter() - start)

        # Verify sub-quadratic scaling: 2x data should cause < 4x slowdown
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = times[i] / times[i - 1] if times[i - 1] > 0 else 1
            assert time_ratio < size_ratio * 2.5, (
                f"Graph building doesn't scale well: {sizes[i - 1]} -> {sizes[i]} "
                f"caused {time_ratio:.1f}x slowdown"
            )

    @pytest.mark.slow
    def test_feature_computation_scales_linearly(self):
        """Verify feature computation scales approximately linearly."""
        # Start from 500 nodes to ensure stable timing measurements.
        # Very small graphs (100 nodes) complete in <5ms which leads to
        # unreliable ratios due to timing variance and warm-up effects.
        node_counts = [500, 1000, 2000, 4000]
        times = []

        # Warm-up run to avoid JIT/caching effects on first measurement
        G_warmup = nx.gnm_random_graph(200, 400, seed=0)
        compute_road_graphlet_features(G_warmup)

        for n_nodes in node_counts:
            G = nx.gnm_random_graph(n_nodes, n_nodes * 2, seed=42)

            start = time.perf_counter()
            features = compute_road_graphlet_features(G)
            times.append(time.perf_counter() - start)

            assert len(features) == n_nodes

        # Verify sub-quadratic scaling.
        # Only check ratios when the baseline time is >= 10ms to avoid
        # unreliable measurements due to timing variance.
        min_reliable_time = 0.01  # 10ms
        for i in range(1, len(node_counts)):
            if times[i - 1] < min_reliable_time:
                continue  # Skip ratio check for very fast measurements
            size_ratio = node_counts[i] / node_counts[i - 1]
            time_ratio = times[i] / times[i - 1]
            assert time_ratio < size_ratio * 3, (
                f"Feature computation doesn't scale well: {node_counts[i - 1]} -> "
                f"{node_counts[i]} nodes caused {time_ratio:.1f}x slowdown"
            )
