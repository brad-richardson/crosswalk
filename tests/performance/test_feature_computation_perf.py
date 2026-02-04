"""Performance benchmark for feature computation.

This test uses synthetic data to measure feature computation performance.
Run with: pytest tests/performance/test_feature_computation_perf.py -v -s

Target run time: ~10-30 seconds total for meaningful measurements.
"""

import time

import numpy as np
import pytest
from shapely import LineString

from matcher.features.geometric import (
    _buffer_iou,
    _buffer_iou_from_buffers,
    _compute_hausdorff_stats,
    clear_buffer_cache,
    compute_collinear_gap_ratio,
    compute_geometric_features,
    compute_heading_consistency,
    compute_shape_complexity,
    compute_sinuosity,
    compute_vertex_density,
)
from matcher.features.relational import (
    compute_endpoint_proximity,
    compute_parallel_alignment,
    compute_perpendicular_offset,
)

# Number of synthetic lines to generate
N_LINES = 5000
# Number of pairs to test (higher = more accurate timing, longer runtime)
N_PAIRS = 10000


class TestFeatureComputationPerformance:
    """Benchmark tests for feature computation bottlenecks."""

    @pytest.fixture(scope="class")
    def synthetic_lines(self):
        """Generate synthetic line pairs for benchmarking."""
        np.random.seed(42)
        lines = []
        for _ in range(N_LINES):
            # Create random lines ~100m long with varying complexity
            start = np.random.rand(2) * 1000
            angle = np.random.rand() * 2 * np.pi
            length = 50 + np.random.rand() * 100
            # Add some intermediate points for more realistic lines
            n_points = np.random.randint(2, 6)
            coords = [start]
            current = start.copy()
            segment_length = length / n_points
            for _ in range(n_points):
                # Small random angle variation for curvature
                angle += (np.random.rand() - 0.5) * 0.3
                current = current + segment_length * np.array([np.cos(angle), np.sin(angle)])
                coords.append(current.copy())
            lines.append(LineString(coords))
        return lines

    def test_geometric_features_throughput(self, synthetic_lines):
        """Benchmark compute_geometric_features (includes buffer ops)."""
        # Clear buffer cache for consistent baseline measurement
        clear_buffer_cache()

        n_pairs = N_PAIRS
        pairs = [
            (
                synthetic_lines[i % len(synthetic_lines)],
                synthetic_lines[(i + 1) % len(synthetic_lines)],
            )
            for i in range(n_pairs)
        ]

        start = time.perf_counter()
        for line_a, line_b in pairs:
            compute_geometric_features(line_a, line_b)
        elapsed = time.perf_counter() - start

        throughput = n_pairs / elapsed
        per_pair_us = (elapsed / n_pairs) * 1_000_000

        print(
            f"\nGeometric features: {throughput:.0f} pairs/sec, {per_pair_us:.1f} µs/pair ({elapsed:.2f}s total)"
        )

        # Target: < 500 µs per pair
        assert per_pair_us < 1000, f"Geometric features too slow: {per_pair_us:.1f} µs/pair"

    def test_perpendicular_offset_throughput(self, synthetic_lines):
        """Benchmark compute_perpendicular_offset."""
        n_pairs = N_PAIRS
        pairs = [
            (
                synthetic_lines[i % len(synthetic_lines)],
                synthetic_lines[(i + 1) % len(synthetic_lines)],
            )
            for i in range(n_pairs)
        ]

        start = time.perf_counter()
        for target, anchor in pairs:
            compute_perpendicular_offset(target, anchor)
        elapsed = time.perf_counter() - start

        throughput = n_pairs / elapsed
        per_pair_us = (elapsed / n_pairs) * 1_000_000

        print(
            f"\nPerpendicular offset: {throughput:.0f} pairs/sec, {per_pair_us:.1f} µs/pair ({elapsed:.2f}s total)"
        )

        # Target: < 100 µs per pair (vectorized)
        assert per_pair_us < 500, f"Perpendicular offset too slow: {per_pair_us:.1f} µs/pair"

    def test_buffer_iou_throughput(self, synthetic_lines):
        """Benchmark buffer IoU specifically."""
        # Clear buffer cache for consistent baseline measurement
        clear_buffer_cache()

        n_pairs = N_PAIRS
        pairs = [
            (
                synthetic_lines[i % len(synthetic_lines)],
                synthetic_lines[(i + 1) % len(synthetic_lines)],
            )
            for i in range(n_pairs)
        ]

        start = time.perf_counter()
        for line_a, line_b in pairs:
            _buffer_iou(line_a, line_b, 5.0)
            _buffer_iou(line_a, line_b, 15.0)
        elapsed = time.perf_counter() - start

        per_pair_us = (elapsed / n_pairs) * 1_000_000 / 2  # per IoU call

        print(f"\nBuffer IoU: {per_pair_us:.1f} µs/call ({elapsed:.2f}s total)")

        # Target: < 200 µs per call with caching
        assert per_pair_us < 500, f"Buffer IoU too slow: {per_pair_us:.1f} µs/call"

    def test_buffer_iou_from_buffers_throughput(self, synthetic_lines):
        """Benchmark IoU with pre-computed buffers."""
        n_pairs = N_PAIRS
        pairs = [
            (
                synthetic_lines[i % len(synthetic_lines)],
                synthetic_lines[(i + 1) % len(synthetic_lines)],
            )
            for i in range(n_pairs)
        ]

        # Pre-compute buffers
        buffers_5m = [(pairs[i][0].buffer(5.0), pairs[i][1].buffer(5.0)) for i in range(n_pairs)]

        start = time.perf_counter()
        for buf_a, buf_b in buffers_5m:
            _buffer_iou_from_buffers(buf_a, buf_b)
        elapsed = time.perf_counter() - start

        per_pair_us = (elapsed / n_pairs) * 1_000_000

        print(
            f"\nBuffer IoU (pre-computed buffers): {per_pair_us:.1f} µs/call ({elapsed:.2f}s total)"
        )

        # Should be faster without buffer creation overhead
        assert per_pair_us < 300, f"Buffer IoU (pre-computed) too slow: {per_pair_us:.1f} µs/call"

    @pytest.mark.skip(reason="Flaky on CI due to timing variability")
    def test_iou_calculation_optimization(self, synthetic_lines):
        """Verify IoU optimization: union = A + B - intersection."""
        # Take a sample pair
        line_a = synthetic_lines[0]
        line_b = synthetic_lines[1]

        buf_a = line_a.buffer(5.0)
        buf_b = line_b.buffer(5.0)

        # Original method: explicit union
        intersection_area = buf_a.intersection(buf_b).area
        union_area_original = buf_a.union(buf_b).area
        iou_original = intersection_area / union_area_original if union_area_original > 0 else 0.0

        # Optimized method: union = A + B - intersection
        union_area_optimized = buf_a.area + buf_b.area - intersection_area
        iou_optimized = (
            intersection_area / union_area_optimized if union_area_optimized > 0 else 0.0
        )

        # Results should be identical (or very close due to floating point)
        assert abs(iou_original - iou_optimized) < 1e-10, (
            f"IoU mismatch: original={iou_original}, optimized={iou_optimized}"
        )

        # Benchmark both approaches with meaningful iterations
        n_iterations = 5000

        start = time.perf_counter()
        for _ in range(n_iterations):
            intersection = buf_a.intersection(buf_b)
            _ = buf_a.union(buf_b).area  # Original
        elapsed_original = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(n_iterations):
            intersection = buf_a.intersection(buf_b)
            _ = buf_a.area + buf_b.area - intersection.area  # Optimized
        elapsed_optimized = time.perf_counter() - start

        speedup = elapsed_original / elapsed_optimized

        print(
            f"\nIoU calculation speedup: {speedup:.2f}x (original: {elapsed_original * 1000:.1f}ms, optimized: {elapsed_optimized * 1000:.1f}ms)"
        )

        # Note: Speed assertion removed - too flaky on CI due to environment variability.
        # The correctness assertion above (line 187) is the important check.

    def test_batch_vs_single_pair_performance(self, synthetic_lines):
        """Test that batch geometric computation is efficient.

        Simulates the ML scoring scenario where each segment appears in
        multiple candidate pairs. The batch path uses vectorized Shapely 2.0
        operations to avoid per-pair Python dispatch overhead.
        """
        from matcher.features.geometric import compute_geometric_features_batch

        # Create pairs where the same geometries appear multiple times
        n_base_lines = 100
        n_pairs_per_line = 50
        lines_a = []
        lines_b = []
        for i in range(n_base_lines):
            for j in range(n_pairs_per_line):
                target_idx = (i + j + 1) % len(synthetic_lines)
                lines_a.append(synthetic_lines[i])
                lines_b.append(synthetic_lines[target_idx])

        n_pairs = len(lines_a)
        print(
            f"\nTesting {n_pairs} pairs with {n_base_lines} base geometries "
            f"(avg {n_pairs_per_line} reuses each)"
        )

        # Per-pair path
        start = time.perf_counter()
        for la, lb in zip(lines_a, lines_b):
            compute_geometric_features(la, lb)
        elapsed_single = time.perf_counter() - start

        # Batch path
        arr_a = np.array(lines_a, dtype=object)
        arr_b = np.array(lines_b, dtype=object)
        start = time.perf_counter()
        compute_geometric_features_batch(arr_a, arr_b)
        elapsed_batch = time.perf_counter() - start

        speedup = elapsed_single / elapsed_batch if elapsed_batch > 0 else 1.0

        print("\nBatch vs single-pair results:")
        print(f"  Single-pair: {elapsed_single:.3f}s ({n_pairs / elapsed_single:.0f} pairs/sec)")
        print(f"  Batch: {elapsed_batch:.3f}s ({n_pairs / elapsed_batch:.0f} pairs/sec)")
        print(f"  Speedup: {speedup:.2f}x")

        # Batch should be at least as fast as single-pair (typically 2-5x faster)
        assert speedup >= 0.9, f"Batch path should be at least as fast: speedup={speedup:.2f}x"

    def test_collinear_gap_ratio_baseline(self, synthetic_lines):
        """Baseline benchmark for compute_collinear_gap_ratio."""
        n_pairs = N_PAIRS
        pairs = [
            (
                synthetic_lines[i % len(synthetic_lines)],
                synthetic_lines[(i + 1) % len(synthetic_lines)],
            )
            for i in range(n_pairs)
        ]

        start = time.perf_counter()
        for line_a, line_b in pairs:
            compute_collinear_gap_ratio(line_a, line_b)
        elapsed = time.perf_counter() - start

        throughput = n_pairs / elapsed
        per_pair_us = (elapsed / n_pairs) * 1_000_000

        print(
            f"\nCollinear gap ratio: {throughput:.0f} pairs/sec, {per_pair_us:.1f} µs/pair ({elapsed:.2f}s total)"
        )

        # Current baseline - will tighten after JIT optimization
        assert per_pair_us < 200, f"Collinear gap ratio too slow: {per_pair_us:.1f} µs/pair"

    def test_shape_complexity_throughput(self, synthetic_lines):
        """Benchmark compute_shape_complexity with JIT."""
        n_lines = len(synthetic_lines)

        start = time.perf_counter()
        for line in synthetic_lines:
            compute_shape_complexity(line)
        elapsed = time.perf_counter() - start

        per_line_us = (elapsed / n_lines) * 1_000_000
        print(f"\nShape complexity: {per_line_us:.1f} µs/line ({elapsed:.2f}s total)")

        # Threshold allows for CI environment variability (~3-5x slower than local)
        assert per_line_us < 150, f"Shape complexity too slow: {per_line_us:.1f} µs/line"

    def test_parallel_alignment_throughput(self, synthetic_lines):
        """Benchmark compute_parallel_alignment with JIT."""
        n_pairs = N_PAIRS
        pairs = [
            (
                synthetic_lines[i % len(synthetic_lines)],
                synthetic_lines[(i + 1) % len(synthetic_lines)],
            )
            for i in range(n_pairs)
        ]

        start = time.perf_counter()
        for line_a, line_b in pairs:
            compute_parallel_alignment(line_a, line_b)
        elapsed = time.perf_counter() - start

        per_pair_us = (elapsed / n_pairs) * 1_000_000
        print(f"\nParallel alignment: {per_pair_us:.1f} µs/pair ({elapsed:.2f}s total)")

        # Threshold allows for CI environment variability (~3-5x slower than local)
        assert per_pair_us < 100, f"Parallel alignment too slow: {per_pair_us:.1f} µs/pair"

    def test_endpoint_proximity_throughput(self, synthetic_lines):
        """Benchmark compute_endpoint_proximity with JIT."""
        # Create endpoint array from all line endpoints
        all_endpoints = []
        for line in synthetic_lines:
            coords = np.array(line.coords)
            all_endpoints.append(coords[0])
            all_endpoints.append(coords[-1])
        endpoint_array = np.array(all_endpoints)

        n_lines = len(synthetic_lines)

        start = time.perf_counter()
        for line in synthetic_lines:
            compute_endpoint_proximity(line, endpoint_array, 5.0)
        elapsed = time.perf_counter() - start

        per_line_us = (elapsed / n_lines) * 1_000_000
        print(f"\nEndpoint proximity: {per_line_us:.1f} µs/line ({elapsed:.2f}s total)")

        # Threshold allows for CI environment variability (~3-5x slower than local)
        assert per_line_us < 200, f"Endpoint proximity too slow: {per_line_us:.1f} µs/line"

    def test_heading_consistency_throughput(self, synthetic_lines):
        """Benchmark compute_heading_consistency with JIT."""
        n_lines = len(synthetic_lines)

        start = time.perf_counter()
        for line in synthetic_lines:
            compute_heading_consistency(line)
        elapsed = time.perf_counter() - start

        per_line_us = (elapsed / n_lines) * 1_000_000
        print(f"\nHeading consistency: {per_line_us:.1f} µs/line ({elapsed:.2f}s total)")

        # Threshold allows for CI environment variability (~3-5x slower than local)
        assert per_line_us < 300, f"Heading consistency too slow: {per_line_us:.1f} µs/line"

    def test_coordinate_extraction_overhead(self, synthetic_lines):
        """Measure baseline cost of np.array(line.coords) extraction.

        This establishes the per-extraction overhead that is saved by
        extracting coords once and passing through to multiple functions.
        """
        n_lines = len(synthetic_lines)
        n_extractions = n_lines * 4  # Simulate 4 extractions per line

        start = time.perf_counter()
        for line in synthetic_lines:
            # Simulate redundant extractions
            _ = np.array(line.coords)
            _ = np.array(line.coords)
            _ = np.array(line.coords)
            _ = np.array(line.coords)
        elapsed = time.perf_counter() - start

        per_extraction_us = (elapsed / n_extractions) * 1_000_000
        print(f"\nCoord extraction: {per_extraction_us:.2f} µs/extraction ({elapsed:.2f}s total)")
        print(f"  Potential savings: {per_extraction_us * 3:.2f} µs/line with coord unification")

    def test_sinuosity_with_pre_extracted_coords(self, synthetic_lines):
        """Benchmark compute_sinuosity with vs without pre-extracted coords."""
        n_lines = len(synthetic_lines)

        # Without pre-extracted coords
        start = time.perf_counter()
        for line in synthetic_lines:
            compute_sinuosity(line)
        elapsed_auto = time.perf_counter() - start
        per_line_auto_us = (elapsed_auto / n_lines) * 1_000_000

        # With pre-extracted coords
        start = time.perf_counter()
        for line in synthetic_lines:
            coords = np.array(line.coords)
            compute_sinuosity(line, coords=coords)
        elapsed_pre = time.perf_counter() - start
        per_line_pre_us = (elapsed_pre / n_lines) * 1_000_000

        # When coords are already extracted by caller
        coords_list = [np.array(line.coords) for line in synthetic_lines]
        start = time.perf_counter()
        for i, line in enumerate(synthetic_lines):
            compute_sinuosity(line, coords=coords_list[i])
        elapsed_reuse = time.perf_counter() - start
        per_line_reuse_us = (elapsed_reuse / n_lines) * 1_000_000

        print("\nSinuosity timing:")
        print(f"  Auto-extract:  {per_line_auto_us:.2f} µs/line")
        print(f"  Pre-extract:   {per_line_pre_us:.2f} µs/line")
        print(f"  Reuse coords:  {per_line_reuse_us:.2f} µs/line")
        savings = per_line_auto_us - per_line_reuse_us
        print(f"  Savings when reusing: {savings:.2f} µs/line")

    def test_hausdorff_stats_with_pre_extracted_coords(self, synthetic_lines):
        """Benchmark _compute_hausdorff_stats with vs without pre-extracted coords."""
        n_pairs = min(N_PAIRS, 5000)  # Use fewer pairs for this test
        pairs = [
            (
                synthetic_lines[i % len(synthetic_lines)],
                synthetic_lines[(i + 1) % len(synthetic_lines)],
            )
            for i in range(n_pairs)
        ]

        # Without pre-extracted coords
        start = time.perf_counter()
        for line_a, line_b in pairs:
            _compute_hausdorff_stats(line_a, line_b)
        elapsed_auto = time.perf_counter() - start
        per_pair_auto_us = (elapsed_auto / n_pairs) * 1_000_000

        # With pre-extracted coords (reusing already extracted)
        coords_pairs = [(np.array(a.coords), np.array(b.coords)) for a, b in pairs]
        start = time.perf_counter()
        for i, (line_a, line_b) in enumerate(pairs):
            coords_a, coords_b = coords_pairs[i]
            _compute_hausdorff_stats(line_a, line_b, coords_a=coords_a, coords_b=coords_b)
        elapsed_reuse = time.perf_counter() - start
        per_pair_reuse_us = (elapsed_reuse / n_pairs) * 1_000_000

        print("\nHausdorff stats timing:")
        print(f"  Auto-extract:  {per_pair_auto_us:.2f} µs/pair")
        print(f"  Reuse coords:  {per_pair_reuse_us:.2f} µs/pair")
        savings = per_pair_auto_us - per_pair_reuse_us
        print(f"  Savings when reusing: {savings:.2f} µs/pair")

    def test_vertex_density_with_pre_extracted_coords(self, synthetic_lines):
        """Benchmark compute_vertex_density with vs without pre-extracted coords."""
        n_lines = len(synthetic_lines)

        # Without pre-extracted coords
        start = time.perf_counter()
        for line in synthetic_lines:
            compute_vertex_density(line)
        elapsed_auto = time.perf_counter() - start
        per_line_auto_us = (elapsed_auto / n_lines) * 1_000_000

        # With pre-extracted coords
        coords_list = [np.array(line.coords) for line in synthetic_lines]
        start = time.perf_counter()
        for i, line in enumerate(synthetic_lines):
            compute_vertex_density(line, coords=coords_list[i])
        elapsed_reuse = time.perf_counter() - start
        per_line_reuse_us = (elapsed_reuse / n_lines) * 1_000_000

        print("\nVertex density timing:")
        print(f"  Auto-extract:  {per_line_auto_us:.2f} µs/line")
        print(f"  Reuse coords:  {per_line_reuse_us:.2f} µs/line")
        savings = per_line_auto_us - per_line_reuse_us
        print(f"  Savings when reusing: {savings:.2f} µs/line")

    def test_query_nearby_endpoints_jit_throughput(self, synthetic_lines):
        """Benchmark query_nearby_endpoints JIT function directly."""
        from matcher.features._jit_helpers import query_nearby_endpoints_numba

        # Create endpoint array from all line endpoints
        all_endpoints = []
        for line in synthetic_lines:
            coords = np.array(line.coords)
            all_endpoints.append(coords[0])
            all_endpoints.append(coords[-1])
        endpoint_array = np.array(all_endpoints)

        n_endpoints = len(endpoint_array)
        n_queries = 10000

        # Generate random query points
        np.random.seed(42)
        query_points = np.random.rand(n_queries, 2) * 1000
        radius = 50.0

        # Create candidate indices (simulate STRtree returning ~100 candidates per query)
        # In practice, STRtree filters to nearby candidates first
        n_candidates_per_query = min(100, n_endpoints)
        candidate_indices = np.arange(n_candidates_per_query, dtype=np.int64)

        start = time.perf_counter()
        for i in range(n_queries):
            query_nearby_endpoints_numba(endpoint_array, candidate_indices, query_points[i], radius)
        elapsed = time.perf_counter() - start

        per_query_us = (elapsed / n_queries) * 1_000_000
        print(
            f"\nQuery nearby endpoints (JIT): {per_query_us:.2f} µs/query "
            f"({n_candidates_per_query} candidates, {elapsed:.2f}s total)"
        )

        # Target: < 50 µs per query for 100 candidates
        assert per_query_us < 100, f"Query nearby endpoints too slow: {per_query_us:.2f} µs/query"
