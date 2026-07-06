"""Tests for stitch-batch spatial context filling.

Regression tests for the optimizer "under-selection" root cause: the old
``_fill_spatial_context`` clipped every group to a centroid-centered ~500m box
and DELETED out-of-box group segments/edges before the pack/UI ever saw them.
The fix bounds only the CONTEXT layer; the group's own segments, edges,
optimizer_assignment and alternatives must always survive intact.
"""

from __future__ import annotations

import copy

import geopandas as gpd
from shapely.geometry import LineString, mapping

from crosswalk.cli.data import (
    CONTEXT_MIN_HALF_M,
    _add_spatial_context_to_group,
    _compute_context_envelope,
)

# Boston-ish origin (WGS84 lon/lat). ~111 km / degree of latitude.
LON0, LAT0 = -71.06, 42.36
DEG_PER_M_LAT = 1.0 / 111_000.0


def _vline(lat_start_m: float, lat_end_m: float, lon: float = LON0) -> LineString:
    """Vertical line between two north offsets (in meters) from LAT0."""
    return LineString(
        [
            (lon, LAT0 + lat_start_m * DEG_PER_M_LAT),
            (lon, LAT0 + lat_end_m * DEG_PER_M_LAT),
        ]
    )


def _large_group() -> dict:
    """A single 1500m ref chained to 4 tiling targets (well over the 500m box)."""
    ref = _vline(0, 1500)
    targets = {
        "T_a": _vline(0, 400),
        "T_b": _vline(400, 800),
        "T_c": _vline(800, 1200),
        "T_d": _vline(1200, 1500),
    }
    edges = [{"ref_id": "R1", "target_id": tid, "confidence": 0.95} for tid in targets]
    return {
        "group_id": "big1",
        "match_type": "1:N",
        "ref_ids": ["R1"],
        "target_ids": list(targets),
        "ref_geometries": {"R1": mapping(ref)},
        "target_geometries": {tid: mapping(g) for tid, g in targets.items()},
        "edges": edges,
        "optimizer_assignment": [dict(e) for e in edges],
        "alternatives": [
            {"option_index": 0, "edges": [dict(e) for e in edges], "confidence": 3.8},
            {"option_index": 1, "edges": [dict(edges[0])], "confidence": 0.95},
        ],
    }


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"id": [], "geometry": [], "names": [], "class": []},
        crs="EPSG:4326",
    )


def _run(group: dict, ref_gdf=None, target_gdf=None) -> dict:
    ref_gdf = ref_gdf if ref_gdf is not None else _empty_gdf()
    target_gdf = target_gdf if target_gdf is not None else _empty_gdf()
    _ = ref_gdf.sindex
    _ = target_gdf.sindex
    _add_spatial_context_to_group(
        group, ref_gdf, target_gdf, names_column="names", class_column="class"
    )
    return group


class TestGroupPreservation:
    def test_large_group_keeps_all_segments_and_edges(self):
        group = _large_group()
        before_refs = list(group["ref_ids"])
        before_targets = list(group["target_ids"])
        before_edges = copy.deepcopy(group["edges"])
        before_ref_geoms = copy.deepcopy(group["ref_geometries"])
        before_target_geoms = copy.deepcopy(group["target_geometries"])

        _run(group)

        assert group["ref_ids"] == before_refs
        assert group["target_ids"] == before_targets
        assert group["edges"] == before_edges
        # Group geometry is left byte-for-byte intact (never clipped/rounded).
        assert group["ref_geometries"] == before_ref_geoms
        assert group["target_geometries"] == before_target_geoms

    def test_audit_metric_reports_no_clipping(self):
        group = _run(_large_group())
        assert group["n_edges_full"] == 4
        assert group["n_edges_rendered"] == 4
        assert group["context_clipped"] is False

    def test_optimizer_assignment_and_alternatives_in_sync(self):
        """The #238 regression class: options must reference only surviving edges
        and must be untouched when nothing is clipped."""
        group = _large_group()
        before_opt = copy.deepcopy(group["optimizer_assignment"])
        before_alts = copy.deepcopy(group["alternatives"])

        _run(group)

        assert group["optimizer_assignment"] == before_opt
        assert group["alternatives"] == before_alts
        # Every option edge references a still-present group edge.
        group_edges = {(e["ref_id"], e["target_id"]) for e in group["edges"]}
        for alt in group["alternatives"]:
            for e in alt["edges"]:
                assert (e["ref_id"], e["target_id"]) in group_edges

    def test_envelope_contains_full_group(self):
        group = _run(_large_group())
        from shapely.geometry import shape

        env = shape(group["envelope"])
        for gj in list(group["ref_geometries"].values()) + list(
            group["target_geometries"].values()
        ):
            assert env.covers(shape(gj)), "envelope must fully contain every group segment"


class TestContextBounding:
    def test_context_still_bounded_to_envelope(self):
        """Context segments are added within the envelope; far-away ones excluded,
        and a straddling context segment is clipped to the box."""
        group = _large_group()
        # A near context segment (~30m east of the ref, inside the envelope).
        near = LineString(
            [
                (LON0 + 0.0004, LAT0 + 200 * DEG_PER_M_LAT),
                (LON0 + 0.0004, LAT0 + 600 * DEG_PER_M_LAT),
            ]
        )
        # A far context segment ~5km north — well outside any reasonable envelope.
        far = _vline(6000, 6400, lon=LON0 + 0.0004)
        ref_gdf = gpd.GeoDataFrame(
            {
                "id": ["CTX_near", "CTX_far"],
                "geometry": [near, far],
                "names": ["Near St", "Far Rd"],
                "class": ["residential", "residential"],
            },
            crs="EPSG:4326",
        )

        _run(group, ref_gdf=ref_gdf)

        assert "CTX_near" in group["context_ref_ids"]
        assert "CTX_far" not in group["context_ref_ids"]
        # Context geometry is stored (clipped to envelope) and never mixed into group.
        assert "CTX_near" in group["context_ref_geometries"]
        assert "CTX_near" not in group["ref_ids"]

    def test_group_segments_excluded_from_context(self):
        """A raw-data row that IS a group member must not be re-added as context."""
        group = _large_group()
        ref_gdf = gpd.GeoDataFrame(
            {
                "id": ["R1"],  # same id as the group's ref
                "geometry": [_vline(0, 1500)],
                "names": ["Circuit Dr"],
                "class": ["residential"],
            },
            crs="EPSG:4326",
        )
        _run(group, ref_gdf=ref_gdf)
        assert "R1" not in group["context_ref_ids"]


class TestComputeContextEnvelope:
    def test_small_group_padded_to_minimum(self):
        """A tiny group envelope is padded out to at least 2*CONTEXT_MIN_HALF_M."""
        from shapely.geometry import shape

        tiny = shape(mapping(_vline(0, 20)))  # 20m segment
        env = _compute_context_envelope([tiny])
        minx, miny, maxx, maxy = env.bounds
        lat_span_m = (maxy - miny) * 111_000.0
        assert lat_span_m >= 2 * CONTEXT_MIN_HALF_M - 1.0

    def test_large_group_not_capped(self):
        """A 1500m group produces an envelope taller than the 500m minimum box."""
        from shapely.geometry import shape

        big = shape(mapping(_vline(0, 1500)))
        env = _compute_context_envelope([big])
        minx, miny, maxx, maxy = env.bounds
        lat_span_m = (maxy - miny) * 111_000.0
        assert lat_span_m > 1500.0  # contains the full chain plus margin
