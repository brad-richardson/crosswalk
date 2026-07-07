"""Parity tests for the fast-path optimizations in per-pair feature code.

These guard the invariant that the optimized implementations are
VALUE-IDENTICAL (bitwise where applicable) to the straightforward
numpy/Shapely formulations they replaced:

1. percentile_sorted == np.percentile (linear method), bitwise
2. _offset_stats == the original mean/np.percentile formulation, bitwise
3. find_parallel_sibling with precomputed context arrays == without
4. _compute_crossing_angle fast path (precomputed tiers/headings) == the
   per-neighbor geometry path
"""

import dataclasses

import numpy as np
import pytest
from shapely import LineString

from crosswalk.features._exact_stats import percentile_sorted
from crosswalk.features.compute import _compute_crossing_angle
from crosswalk.features.relational import (
    _offset_stats,
    build_sibling_search_context,
    find_parallel_sibling,
)


class TestPercentileSorted:
    def test_matches_numpy_percentile_bitwise(self):
        rng = np.random.default_rng(42)
        for _ in range(500):
            n = int(rng.integers(1, 120))
            scale = float(rng.choice([1e-6, 1.0, 100.0, 1e6]))
            values = rng.random(n) * scale
            sorted_values = np.sort(values)
            for q in (0.0, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0):
                expected = float(np.percentile(values, q))
                actual = percentile_sorted(sorted_values, q)
                # Bitwise equality, not just approximate
                assert actual == expected, f"n={n} q={q}: {actual!r} != {expected!r}"

    def test_single_element(self):
        assert percentile_sorted(np.array([3.5]), 95.0) == 3.5

    def test_two_elements(self):
        values = np.array([1.0, 2.0])
        for q in (25.0, 75.0, 95.0):
            assert percentile_sorted(values, q) == float(np.percentile(values, q))


class TestOffsetStats:
    @staticmethod
    def _reference_stats(offsets, return_percentile=None):
        """The original formulation _offset_stats replaced."""
        mean_offset = float(np.mean(offsets))
        p25, p75 = np.percentile(offsets, [25, 75])
        offset_iqr = float(p75 - p25)
        offset_p95 = float(np.percentile(offsets, 95))
        offset_pn = (
            float(np.percentile(offsets, return_percentile))
            if return_percentile is not None
            else float("inf")
        )
        return mean_offset, offset_iqr, offset_p95, offset_pn

    def test_matches_original_formulation_bitwise(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            offsets = rng.random(int(rng.integers(1, 80))) * 50.0
            assert _offset_stats(offsets) == self._reference_stats(offsets)
            assert _offset_stats(offsets, 25) == self._reference_stats(offsets, 25)

    def test_nan_fallback_matches_numpy(self):
        offsets = np.array([1.0, np.nan, 3.0])
        with np.errstate(invalid="ignore"):
            expected = self._reference_stats(offsets)
            actual = _offset_stats(offsets)
        for e, a in zip(expected, actual):
            assert (np.isnan(e) and np.isnan(a)) or e == a


def _grid_context():
    """Synthetic network: parallel carriageways + crossing footways."""
    geometries = [
        LineString([(0, 0), (100, 0)]),  # main EB
        LineString([(0, 12), (100, 12)]),  # main WB (parallel sibling)
        LineString([(50, -20), (50, 20)]),  # crossing footway
        LineString([(0, 50), (100, 55)]),  # unrelated road
        LineString([(20, -5), (20, 5)]),  # crossing cycleway
        LineString([(0, 3), (40, 3)]),  # partial parallel service road
    ]
    ids = ["eb", "wb", "xw", "far", "cy", "svc"]
    names = ["Main St", "Main St", None, "Far Rd", None, None]
    classes = ["primary", "primary", "footway", "residential", "cycleway", "service"]
    return build_sibling_search_context(
        geometries=geometries, segment_ids=ids, names=names, classes=classes
    )


class TestSiblingContextFastPath:
    def test_context_and_no_context_paths_identical(self):
        ctx = _grid_context()
        for idx, (seg_id, name, cls) in enumerate(ctx.segment_data):
            geom = ctx.spatial_index.geometries[idx]
            slow = find_parallel_sibling(
                segment=geom,
                segment_id=seg_id,
                segment_name=name,
                segment_class=cls,
                spatial_index=ctx.spatial_index,
                segment_data=ctx.segment_data,
            )
            fast = find_parallel_sibling(
                segment=geom,
                segment_id=seg_id,
                segment_name=name,
                segment_class=cls,
                spatial_index=ctx.spatial_index,
                segment_data=ctx.segment_data,
                context=ctx,
            )
            assert slow == fast, f"sibling mismatch for {seg_id}: {slow} vs {fast}"

    def test_finds_expected_sibling(self):
        ctx = _grid_context()
        result = find_parallel_sibling(
            segment=ctx.spatial_index.geometries[0],
            segment_id="eb",
            segment_name="Main St",
            segment_class="primary",
            spatial_index=ctx.spatial_index,
            segment_data=ctx.segment_data,
            context=ctx,
        )
        assert result.has_sibling
        assert result.sibling_distance == pytest.approx(12.0, abs=0.5)


class TestCrossingAngleFastPath:
    def test_fast_and_slow_paths_identical(self):
        ctx = _grid_context()
        # Context without precomputed arrays forces the per-neighbor path
        ctx_slow = dataclasses.replace(
            ctx,
            segment_coords=None,
            segment_valid=None,
            segment_lengths=None,
            segment_tiers=None,
            segment_headings=None,
        )
        for idx, (seg_id, _name, cls) in enumerate(ctx.segment_data):
            geom = ctx.spatial_index.geometries[idx]
            fast = _compute_crossing_angle(geom, cls, seg_id, ctx)
            slow = _compute_crossing_angle(geom, cls, seg_id, ctx_slow)
            assert set(fast) == set(slow)
            for key in fast:
                f, s = fast[key], slow[key]
                assert (np.isnan(f) and np.isnan(s)) or f == s, (
                    f"crossing mismatch for {seg_id}.{key}: {f} vs {s}"
                )

    def test_footway_crossing_road_is_transverse(self):
        ctx = _grid_context()
        result = _compute_crossing_angle(ctx.spatial_index.geometries[2], "footway", "xw", ctx)
        # Different-tier neighbors of the footway: four vehicle segments
        # (eb/wb/svc/far, all perpendicular at ~90 deg) and the cycleway
        # (parallel, 0 deg) -> 4/5 transverse, min angle ~0 (parallel cycleway).
        assert result["transverse_neighbor_fraction"] == pytest.approx(0.8, abs=1e-9)
        assert result["crossing_angle_min"] == pytest.approx(0.0, abs=1.0)
