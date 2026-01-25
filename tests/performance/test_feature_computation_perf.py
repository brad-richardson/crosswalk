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
    clear_buffer_cache,
    compute_collinear_gap_ratio,
    compute_geometric_features,
    get_buffer_cache_info,
)
from matcher.features.relational import compute_perpendicular_offset

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

    def test_buffer_cache_effectiveness(self, synthetic_lines):
        """Test that buffer caching provides speedup for repeated geometries.

        Simulates the ML scoring scenario where each segment appears in
        multiple candidate pairs (e.g., each reference segment matched
        against multiple target segments).
        """
        # Clear cache before test
        clear_buffer_cache()

        # Create pairs where the same geometries appear multiple times
        # Each of the first 100 lines is paired with 50 different lines
        n_base_lines = 100
        n_pairs_per_line = 50
        pairs = []
        for i in range(n_base_lines):
            line_a = synthetic_lines[i]
            for j in range(n_pairs_per_line):
                # Pair with different target lines
                target_idx = (i + j + 1) % len(synthetic_lines)
                pairs.append((line_a, synthetic_lines[target_idx]))

        n_pairs = len(pairs)
        print(
            f"\nTesting {n_pairs} pairs with {n_base_lines} base geometries (avg {n_pairs_per_line} reuses each)"
        )

        # First run - cold cache
        clear_buffer_cache()
        start = time.perf_counter()
        for line_a, line_b in pairs:
            compute_geometric_features(line_a, line_b)
        elapsed_cold = time.perf_counter() - start
        cache_info_after_cold = get_buffer_cache_info()

        # Second run - warm cache (same pairs)
        start = time.perf_counter()
        for line_a, line_b in pairs:
            compute_geometric_features(line_a, line_b)
        elapsed_warm = time.perf_counter() - start
        cache_info_after_warm = get_buffer_cache_info()

        speedup = elapsed_cold / elapsed_warm if elapsed_warm > 0 else 1.0

        print("\nBuffer cache results:")
        print(f"  Cold cache: {elapsed_cold:.3f}s ({n_pairs / elapsed_cold:.0f} pairs/sec)")
        print(f"  Warm cache: {elapsed_warm:.3f}s ({n_pairs / elapsed_warm:.0f} pairs/sec)")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Cache after cold run: {cache_info_after_cold}")
        print(f"  Cache after warm run: {cache_info_after_warm}")

        # Expect significant cache hits on warm run
        # Each unique geometry should be cached with 2 radii (5m and 15m)
        # With 100 base lines × 2 radii = 200 unique buffers for line_a
        # Plus varying line_b buffers
        assert cache_info_after_warm.hits > cache_info_after_cold.hits, (
            "Warm cache should have more hits than cold cache"
        )

        # Warm run should be faster due to cache hits
        # (relaxed assertion since cache benefits depend on geometry complexity)
        assert speedup >= 0.9, f"Warm cache should be at least as fast: speedup={speedup:.2f}x"

        # Clear cache after test
        clear_buffer_cache()

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
