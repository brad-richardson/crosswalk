"""Performance tests for candidate generation.

These tests benchmark candidate generation and verify scaling behavior.
They are marked with @pytest.mark.slow so they can be skipped during fast test runs.

Performance requirements:
- 1K x 1K segments: < 2 seconds
- 5K x 5K segments: < 15 seconds
- 10K x 10K segments: < 60 seconds
"""

import time

import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString

from matcher.blocking.spatial_index import (
    CandidateBatch,
    CandidatePair,
    generate_candidates,
)


@pytest.fixture
def random_road_network_factory():
    """Factory fixture for creating random road networks of various sizes."""

    def _create(
        n_segments: int = 100,
        bbox: tuple[float, float, float, float] = (0, 0, 10000, 10000),
        avg_length: float = 100.0,
        seed: int = 42,
    ) -> gpd.GeoDataFrame:
        """Create a synthetic random road network.

        Args:
            n_segments: Number of road segments
            bbox: Bounding box (minx, miny, maxx, maxy) in meters
            avg_length: Average segment length in meters
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        minx, miny, maxx, maxy = bbox

        segments = []
        for i in range(n_segments):
            # Random start point
            x1 = np.random.uniform(minx, maxx)
            y1 = np.random.uniform(miny, maxy)

            # Random heading and length
            heading = np.random.uniform(0, 2 * np.pi)
            length = np.random.exponential(avg_length)

            # End point
            x2 = x1 + length * np.cos(heading)
            y2 = y1 + length * np.sin(heading)

            segments.append(
                {
                    "id": f"seg_{i}",
                    "name": f"Road {i % 100}",
                    "geometry": LineString([(x1, y1), (x2, y2)]),
                }
            )

        return gpd.GeoDataFrame(segments, crs="EPSG:32618")

    return _create


@pytest.fixture
def overlapping_networks_factory(random_road_network_factory):
    """Factory for creating two overlapping networks (simulates reference/target)."""

    def _create(
        n_reference: int = 100,
        n_target: int = 100,
        offset_m: float = 10.0,
        noise_m: float = 5.0,
        seed: int = 42,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Create reference and target networks that overlap.

        The target network is derived from reference with offset and noise,
        simulating typical real-world matching scenarios.
        """
        # Create reference with max of both sizes to ensure enough geometries
        max_size = max(n_reference, n_target)
        full_reference = random_road_network_factory(n_segments=max_size, seed=seed)

        # Slice to requested reference size
        reference = full_reference.iloc[:n_reference].copy()
        reference = reference.reset_index(drop=True)

        # Create target by copying geometries with offset and noise
        np.random.seed(seed + 1)
        source_geoms = full_reference.geometry.iloc[:n_target]
        target_geoms = []
        for geom in source_geoms:
            coords = np.array(geom.coords)
            # Add lateral offset
            dx = np.random.uniform(-offset_m, offset_m)
            dy = np.random.uniform(-offset_m, offset_m)
            # Add noise to each vertex
            noise = np.random.normal(0, noise_m, coords.shape)
            new_coords = coords + noise + np.array([dx, dy])
            target_geoms.append(LineString(new_coords))

        target = gpd.GeoDataFrame(
            {
                "local_id": [f"local_{i}" for i in range(n_target)],
                "name": [f"Local Road {i % 100}" for i in range(n_target)],
                "geometry": target_geoms,
            },
            crs="EPSG:32618",
        )

        return reference, target

    return _create


class TestCandidateGenerationPerformance:
    """Performance tests for candidate generation."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "n_ref,n_target,max_seconds",
        [
            (100, 100, 0.5),
            (1000, 1000, 2.0),
            (5000, 5000, 15.0),
            (10000, 10000, 60.0),
        ],
        ids=["100x100", "1Kx1K", "5Kx5K", "10Kx10K"],
    )
    def test_candidate_generation_scales(
        self, overlapping_networks_factory, n_ref, n_target, max_seconds
    ):
        """Benchmark candidate generation at various scales."""
        reference, target = overlapping_networks_factory(n_reference=n_ref, n_target=n_target)

        start = time.perf_counter()
        candidates = generate_candidates(
            reference,
            target,
            buffer_distance_m=50.0,
            ref_id_column="id",
            target_id_column="local_id",
        )
        elapsed = time.perf_counter() - start

        # Should find candidates (overlapping networks)
        assert len(candidates) > 0
        assert all(isinstance(c, CandidatePair) for c in candidates)

        assert elapsed < max_seconds, (
            f"{n_ref}x{n_target} segments took {elapsed:.2f}s, expected < {max_seconds}s"
        )


class TestCandidateGenerationScaling:
    """Tests to verify candidate generation scales sub-quadratically."""

    @pytest.mark.slow
    def test_scales_with_reference_size(self, overlapping_networks_factory):
        """Verify scaling as reference size increases."""
        target_size = 1000
        ref_sizes = [500, 1000, 2000, 4000]
        times = []

        for n_ref in ref_sizes:
            reference, target = overlapping_networks_factory(
                n_reference=n_ref, n_target=target_size
            )

            start = time.perf_counter()
            candidates = generate_candidates(
                reference,
                target,
                buffer_distance_m=50.0,
                ref_id_column="id",
                target_id_column="local_id",
            )
            times.append(time.perf_counter() - start)

            assert len(candidates) > 0

        # Verify sub-quadratic scaling
        for i in range(1, len(ref_sizes)):
            size_ratio = ref_sizes[i] / ref_sizes[i - 1]
            time_ratio = times[i] / times[i - 1] if times[i - 1] > 0.001 else 1
            assert time_ratio < size_ratio * 2.5, (
                f"Reference scaling: {ref_sizes[i - 1]} -> {ref_sizes[i]} "
                f"caused {time_ratio:.1f}x slowdown (expected < {size_ratio * 2.5:.1f}x)"
            )

    @pytest.mark.slow
    @pytest.mark.skip(reason="Flaky on CI due to timing variance")
    def test_scales_with_target_size(self, overlapping_networks_factory):
        """Verify scaling as target size increases."""
        ref_size = 1000
        target_sizes = [500, 1000, 2000, 4000]
        times = []

        for n_target in target_sizes:
            reference, target = overlapping_networks_factory(
                n_reference=ref_size, n_target=n_target
            )

            start = time.perf_counter()
            candidates = generate_candidates(
                reference,
                target,
                buffer_distance_m=50.0,
                ref_id_column="id",
                target_id_column="local_id",
            )
            times.append(time.perf_counter() - start)

            assert len(candidates) > 0

        # Verify sub-quadratic scaling
        for i in range(1, len(target_sizes)):
            size_ratio = target_sizes[i] / target_sizes[i - 1]
            time_ratio = times[i] / times[i - 1] if times[i - 1] > 0.001 else 1
            assert time_ratio < size_ratio * 2.5, (
                f"Target scaling: {target_sizes[i - 1]} -> {target_sizes[i]} "
                f"caused {time_ratio:.1f}x slowdown (expected < {size_ratio * 2.5:.1f}x)"
            )


class TestCandidatePairConstructionPerformance:
    """Micro-benchmarks for CandidatePair construction overhead."""

    @pytest.mark.slow
    def test_candidate_pair_construction_overhead(self):
        """Benchmark CandidatePair dataclass construction."""
        n_pairs = 100000

        # Simulate data from sjoin result
        ref_ids = [f"ref_{i}" for i in range(n_pairs)]
        ref_idxs = list(range(n_pairs))
        target_ids = [f"target_{i}" for i in range(n_pairs)]
        target_idxs = list(range(n_pairs))
        distances = np.random.uniform(0, 50, n_pairs)
        heading_diffs = np.random.uniform(0, 90, n_pairs)
        length_ratios = np.random.uniform(0.1, 1.0, n_pairs)

        start = time.perf_counter()
        candidates = [
            CandidatePair(
                ref_id=ref_ids[i],
                ref_idx=ref_idxs[i],
                target_id=target_ids[i],
                target_idx=target_idxs[i],
                distance_estimate=distances[i],
                heading_diff=heading_diffs[i],
                length_ratio=length_ratios[i],
            )
            for i in range(n_pairs)
        ]
        elapsed = time.perf_counter() - start

        assert len(candidates) == n_pairs
        # 100K pairs should take < 0.5s with list comprehension
        assert elapsed < 0.5, f"100K CandidatePair construction took {elapsed:.2f}s"


class TestHighDensityScenarios:
    """Performance tests for high-density scenarios with many candidates per segment."""

    @pytest.mark.slow
    def test_high_density_candidates(self, random_road_network_factory):
        """Test performance when buffer catches many candidates per segment."""
        # Small area with many segments = high density
        reference = random_road_network_factory(
            n_segments=2000,
            bbox=(0, 0, 1000, 1000),  # Dense: 2 segments per 1000m²
            avg_length=50.0,
            seed=42,
        )
        target = random_road_network_factory(
            n_segments=2000,
            bbox=(0, 0, 1000, 1000),
            avg_length=50.0,
            seed=43,
        )

        start = time.perf_counter()
        candidates = generate_candidates(
            reference,
            target,
            buffer_distance_m=100.0,  # Large buffer = many candidates
            ref_id_column="id",
            target_id_column="id",
        )
        elapsed = time.perf_counter() - start

        # Should generate many candidates in dense scenario
        assert len(candidates) > 10000
        # Should still complete in reasonable time
        assert elapsed < 30.0, f"High density scenario took {elapsed:.2f}s"


class TestMemoryEfficiency:
    """Tests to verify memory-efficient candidate generation."""

    @pytest.mark.slow
    def test_large_dataset_completes(self, random_road_network_factory):
        """Verify large datasets complete without memory issues."""
        reference = random_road_network_factory(n_segments=10000, seed=42)
        target = random_road_network_factory(n_segments=10000, seed=43)

        # Should complete without running out of memory
        candidates = generate_candidates(
            reference,
            target,
            buffer_distance_m=50.0,
            ref_id_column="id",
            target_id_column="id",
        )

        # Basic sanity check - generate_candidates now returns CandidateBatch
        assert isinstance(candidates, CandidateBatch)
        assert len(candidates) > 0


class TestCandidateGenerationProfiling:
    """Detailed profiling tests to identify optimization targets."""

    @pytest.mark.slow
    def test_profile_generation_steps(self, overlapping_networks_factory):
        """Profile individual steps of candidate generation."""
        import geopandas as gpd

        from matcher.blocking.spatial_index import (
            _angle_diff_vectorized,
            _compute_headings_vectorized,
            _create_local_projection_crs,
        )

        n_ref, n_target = 5000, 5000
        reference, target = overlapping_networks_factory(n_reference=n_ref, n_target=n_target)
        buffer_distance_m = 50.0

        timings = {}

        # Step 1: CRS projection
        start = time.perf_counter()
        local_crs = _create_local_projection_crs(target)
        if local_crs is not None:
            target_proj = target.to_crs(local_crs)
            reference_proj = reference.to_crs(local_crs)
        else:
            target_proj = target
            reference_proj = reference
        timings["projection"] = time.perf_counter() - start

        # Step 2: Prepare target (copy, add columns)
        start = time.perf_counter()
        target_prep = target_proj.copy()
        target_prep["_target_idx"] = range(len(target))
        target_prep["_target_heading"] = _compute_headings_vectorized(target_proj.geometry)
        target_prep["_target_length"] = target_proj.geometry.length
        target_prep["_target_id"] = target["local_id"]
        target_prep["_target_geom"] = target_prep.geometry
        timings["target_prep"] = time.perf_counter() - start

        # Step 3: Buffer target geometries
        start = time.perf_counter()
        target_prep = target_prep.set_geometry(target_prep.geometry.buffer(buffer_distance_m))
        timings["buffering"] = time.perf_counter() - start

        # Step 4: Prepare reference
        start = time.perf_counter()
        reference_prep = reference_proj.copy()
        reference_prep["_ref_idx"] = range(len(reference))
        reference_prep["_ref_heading"] = _compute_headings_vectorized(reference_proj.geometry)
        reference_prep["_ref_length"] = reference_proj.geometry.length
        reference_prep["_ref_id"] = reference["id"]
        reference_prep["_ref_geom"] = reference_prep.geometry
        timings["reference_prep"] = time.perf_counter() - start

        # Step 5: Spatial join
        start = time.perf_counter()
        ref_cols = ["geometry", "_ref_idx", "_ref_heading", "_ref_length", "_ref_id", "_ref_geom"]
        joined = gpd.sjoin(
            target_prep,
            reference_prep[ref_cols],
            how="inner",
            predicate="intersects",
        )
        timings["spatial_join"] = time.perf_counter() - start

        n_candidates = len(joined)

        # Step 6: Heading/length ratio computation
        start = time.perf_counter()
        heading_diff = _angle_diff_vectorized(
            joined["_target_heading"].values,
            joined["_ref_heading"].values,
        )
        min_len = np.minimum(joined["_target_length"].values, joined["_ref_length"].values)
        max_len = np.maximum(joined["_target_length"].values, joined["_ref_length"].values)
        length_ratio = max_len / np.maximum(min_len, 0.1)
        timings["heading_length"] = time.perf_counter() - start

        # Step 7: Centroid distances
        start = time.perf_counter()
        target_centroids = joined["_target_geom"].centroid
        ref_centroids = joined["_ref_geom"].centroid
        distances = target_centroids.distance(ref_centroids).values
        timings["centroids"] = time.perf_counter() - start

        # Step 8: CandidatePair construction with iterrows (current approach)
        start = time.perf_counter()
        candidates_iterrows = []
        for i, (_idx, row) in enumerate(joined.iterrows()):
            candidates_iterrows.append(
                CandidatePair(
                    ref_id=row["_ref_id"],
                    ref_idx=int(row["_ref_idx"]),
                    target_id=row["_target_id"],
                    target_idx=int(row["_target_idx"]),
                    distance_estimate=distances[i],
                    heading_diff=heading_diff[i],
                    length_ratio=1.0 / length_ratio[i],
                )
            )
        timings["iterrows_construction"] = time.perf_counter() - start

        # Step 8b: CandidatePair construction with list comprehension (optimized)
        start = time.perf_counter()
        ref_ids = joined["_ref_id"].values
        ref_idxs = joined["_ref_idx"].values.astype(int)
        target_ids = joined["_target_id"].values
        target_idxs = joined["_target_idx"].values.astype(int)
        inv_length_ratios = 1.0 / length_ratio

        candidates_list_comp = [
            CandidatePair(
                ref_id=ref_ids[i],
                ref_idx=ref_idxs[i],
                target_id=target_ids[i],
                target_idx=target_idxs[i],
                distance_estimate=distances[i],
                heading_diff=heading_diff[i],
                length_ratio=inv_length_ratios[i],
            )
            for i in range(len(joined))
        ]
        timings["listcomp_construction"] = time.perf_counter() - start

        total = sum(timings.values()) - timings["listcomp_construction"]  # Don't double-count

        print(
            f"\n=== Candidate Generation Profile ({n_ref}x{n_target}, {n_candidates} candidates) ==="
        )
        for step, t in timings.items():
            pct = (t / total) * 100 if total > 0 else 0
            print(f"  {step:25s}: {t:6.3f}s ({pct:5.1f}%)")
        print(f"  {'TOTAL':25s}: {total:6.3f}s")
        print(
            f"\n  iterrows speedup from listcomp: {timings['iterrows_construction'] / timings['listcomp_construction']:.1f}x"
        )

        # Verify both methods produce same results
        assert len(candidates_iterrows) == len(candidates_list_comp)

        # Assertions for reasonable timing distribution
        assert timings["spatial_join"] < 5.0, "Spatial join too slow"
        assert timings["iterrows_construction"] > timings["listcomp_construction"], (
            "List comp should be faster than iterrows"
        )

    @pytest.mark.slow
    def test_profile_dataframe_copy_overhead(self, overlapping_networks_factory):
        """Profile DataFrame copy vs assign overhead."""
        from matcher.blocking.spatial_index import _compute_headings_vectorized

        n_ref, n_target = 5000, 5000
        reference, _ = overlapping_networks_factory(n_reference=n_ref, n_target=n_target)

        timings = {}

        # Approach 1: Current - copy() then add columns
        start = time.perf_counter()
        for _ in range(3):  # Average over 3 runs
            df_copy = reference.copy()
            df_copy["_idx"] = range(len(reference))
            df_copy["_heading"] = _compute_headings_vectorized(reference.geometry)
            df_copy["_length"] = reference.geometry.length
            df_copy["_geom"] = df_copy.geometry
        timings["copy_then_add"] = (time.perf_counter() - start) / 3

        # Approach 2: assign() - creates new df with added columns
        start = time.perf_counter()
        for _ in range(3):
            df_assign = reference.assign(
                _idx=range(len(reference)),
                _heading=_compute_headings_vectorized(reference.geometry),
                _length=reference.geometry.length,
            )
            df_assign["_geom"] = df_assign.geometry
        timings["assign"] = (time.perf_counter() - start) / 3

        # Approach 3: Direct column addition (modifies original - not safe!)
        test_df = reference.copy()  # Make a safe copy first
        start = time.perf_counter()
        for _ in range(3):
            test_df["_idx2"] = range(len(test_df))
            test_df["_heading2"] = _compute_headings_vectorized(test_df.geometry)
            test_df["_length2"] = test_df.geometry.length
        timings["direct_add"] = (time.perf_counter() - start) / 3

        print(f"\n=== DataFrame Copy Overhead ({n_ref} rows) ===")
        for approach, t in timings.items():
            print(f"  {approach:20s}: {t * 1000:6.1f}ms")

        speedup = timings["copy_then_add"] / timings["assign"]
        print(f"\n  assign() speedup: {speedup:.2f}x")

        # The approaches should produce equivalent results
        assert len(df_copy) == len(df_assign) == len(reference)

    @pytest.mark.slow
    def test_actual_generate_candidates_timing(self, overlapping_networks_factory):
        """Time the actual generate_candidates function after optimization."""
        sizes = [(1000, 1000), (5000, 5000), (10000, 10000)]

        print("\n=== generate_candidates() Timing (after optimization) ===")
        for n_ref, n_target in sizes:
            reference, target = overlapping_networks_factory(n_reference=n_ref, n_target=n_target)

            # Warm-up run
            _ = generate_candidates(
                reference,
                target,
                buffer_distance_m=50.0,
                ref_id_column="id",
                target_id_column="local_id",
            )

            # Timed run
            start = time.perf_counter()
            candidates = generate_candidates(
                reference,
                target,
                buffer_distance_m=50.0,
                ref_id_column="id",
                target_id_column="local_id",
            )
            elapsed = time.perf_counter() - start

            print(f"  {n_ref}x{n_target}: {elapsed:.3f}s ({len(candidates)} candidates)")

            # Performance assertions
            if n_ref == 1000:
                assert elapsed < 1.0, f"1Kx1K too slow: {elapsed:.2f}s"
            elif n_ref == 5000:
                assert elapsed < 3.0, f"5Kx5K too slow: {elapsed:.2f}s"
            elif n_ref == 10000:
                assert elapsed < 10.0, f"10Kx10K too slow: {elapsed:.2f}s"
