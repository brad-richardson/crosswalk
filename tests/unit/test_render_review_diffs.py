"""Unit tests for scripts/render_review_diffs.py.

Covers the pure logic (label<->optimizer diff, confidence-lookup precedence,
optimizer-source fallback, geometry coercion, selected-edge parsing, summary
formatting) plus a smoke render of the overview and zoom PNGs against a small
synthetic fixture. No image-content assertions -- we only assert files land.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render_review_diffs.py"
_spec = importlib.util.spec_from_file_location("render_review_diffs", _SCRIPT)
rrd = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve the module via sys.modules.
sys.modules[_spec.name] = rrd
_spec.loader.exec_module(rrd)


# ---------------------------------------------------------------------------
# Synthetic fixture: a group with 2 refs, 2 targets, cache + sidecar edges
# ---------------------------------------------------------------------------
def _line(x0, y0, x1, y1):
    return {"type": "LineString", "coordinates": [[x0, y0], [x1, y1]]}


@pytest.fixture
def cache_group():
    return {
        "group_id": "abcd1234ef",
        "match_type": "M:N",
        "ref_geometries": {
            "refA": _line(0.0, 42.0, 0.001, 42.0),
            "refB": _line(0.001, 42.0, 0.002, 42.0),
        },
        "target_geometries": {
            "tgt_1_h": _line(0.0, 42.00001, 0.001, 42.00001),
            "tgt_2_h": _line(0.001, 42.00001, 0.002, 42.00001),
        },
        "ref_names": {"refA": "Main St", "refB": "Main St"},
        "target_names": {"tgt_1_h": "MAIN STREET", "tgt_2_h": "MAIN STREET"},
        "edges": [
            {"ref_id": "refA", "target_id": "tgt_1_h", "confidence": 0.90, "selected": True},
            {"ref_id": "refB", "target_id": "tgt_2_h", "confidence": 0.30, "selected": False},
        ],
        "optimizer_assignment": [
            {"ref_id": "refA", "target_id": "tgt_1_h", "confidence": 0.11},
        ],
    }


@pytest.fixture
def sidecar_group():
    return {
        "group_id": "abcd1234ef",
        "edges": [
            {"ref_id": "refA", "target_id": "tgt_1_h", "confidence": 0.95, "selected": True},
        ],
        "rejected_edges": [
            {"ref_id": "refB", "target_id": "tgt_2_h", "confidence": 0.42, "selected": False},
        ],
    }


# ---------------------------------------------------------------------------
# Geometry coercion
# ---------------------------------------------------------------------------
def test_coerce_coords_linestring():
    g = _line(0, 0, 1, 1)
    assert rrd.coerce_coords(g) == [[[0, 0], [1, 1]]]


def test_coerce_coords_multilinestring():
    g = {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]}
    assert rrd.coerce_coords(g) == [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]


def test_coerce_coords_none_and_unknown():
    assert rrd.coerce_coords(None) == []
    assert rrd.coerce_coords({}) == []
    assert rrd.coerce_coords({"type": "Point", "coordinates": [0, 0]}) == []


def test_geom_points_flattens_multipart():
    g = {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1]], [[2, 2]]]}
    assert rrd.geom_points(g) == [[0, 0], [1, 1], [2, 2]]


def test_geom_midpoint_first_part():
    g = {"type": "LineString", "coordinates": [[0, 0], [1, 1], [2, 2]]}
    assert rrd.geom_midpoint(g) == [1, 1]
    assert rrd.geom_midpoint(None) is None


# ---------------------------------------------------------------------------
# Selected-edge parsing
# ---------------------------------------------------------------------------
def test_parse_selected_edges():
    raw = "[{'ref_id': 'r1', 'target_id': 't1'}, {'ref_id': 'r2', 'target_id': 't2'}]"
    assert rrd.parse_selected_edges(raw) == {("r1", "t1"), ("r2", "t2")}


def test_parse_selected_edges_empty():
    assert rrd.parse_selected_edges("") == set()
    assert rrd.parse_selected_edges(None) == set()


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------
def test_compute_diff():
    label = {("a", "1"), ("b", "2"), ("c", "3")}
    opt = {("b", "2"), ("c", "3"), ("d", "4")}
    added, removed = rrd.compute_diff(label, opt)
    assert added == {("a", "1")}
    assert removed == {("d", "4")}


def test_compute_diff_identical():
    s = {("a", "1")}
    added, removed = rrd.compute_diff(s, set(s))
    assert added == set() and removed == set()


# ---------------------------------------------------------------------------
# Optimizer-source fallback
# ---------------------------------------------------------------------------
def test_resolve_optimizer_prefers_sidecar(cache_group, sidecar_group):
    pairs, from_sidecar = rrd.resolve_optimizer(cache_group, sidecar_group)
    assert from_sidecar is True
    assert pairs == {("refA", "tgt_1_h")}  # only selected sidecar edge


def test_resolve_optimizer_falls_back_to_cache(cache_group):
    pairs, from_sidecar = rrd.resolve_optimizer(cache_group, None)
    assert from_sidecar is False
    assert pairs == {("refA", "tgt_1_h")}  # optimizer_assignment


def test_resolve_optimizer_missing_group():
    pairs, from_sidecar = rrd.resolve_optimizer(None, None)
    assert pairs == set() and from_sidecar is False


# ---------------------------------------------------------------------------
# Confidence-lookup precedence: cache edges > sidecar edges > rejected > opt_assign
# ---------------------------------------------------------------------------
def test_confidence_precedence_cache_edges_win(cache_group, sidecar_group):
    confs = rrd.build_confidence_lookup(cache_group, sidecar_group)
    # cache edge conf (0.90) wins over sidecar edge conf (0.95)
    assert confs[("refA", "tgt_1_h")] == 0.90
    # refB/tgt2 present in cache edges (0.30) wins over rejected (0.42)
    assert confs[("refB", "tgt_2_h")] == 0.30


def test_confidence_falls_through_to_rejected(sidecar_group):
    # No cache group: sidecar edges then rejected supply confidences.
    confs = rrd.build_confidence_lookup(None, sidecar_group)
    assert confs[("refA", "tgt_1_h")] == 0.95
    assert confs[("refB", "tgt_2_h")] == 0.42


def test_confidence_uses_optimizer_assignment_last():
    cache = {
        "edges": [],
        "optimizer_assignment": [
            {"ref_id": "x", "target_id": "y", "confidence": 0.7},
        ],
    }
    confs = rrd.build_confidence_lookup(cache, None)
    assert confs[("x", "y")] == 0.7


def test_confidence_skips_none():
    cache = {"edges": [{"ref_id": "x", "target_id": "y", "confidence": None}]}
    assert rrd.build_confidence_lookup(cache, None) == {}


# ---------------------------------------------------------------------------
# Cross-product artifact detection (pure set arithmetic)
# ---------------------------------------------------------------------------
def test_candidate_universe_union(cache_group, sidecar_group):
    uni = rrd.candidate_universe(cache_group, sidecar_group)
    # cache edges (refA/t1, refB/t2) ∪ sidecar edges (refA/t1) ∪ rejected (refB/t2)
    assert uni == {("refA", "tgt_1_h"), ("refB", "tgt_2_h")}


def test_crossproduct_within_universe():
    # label refs {rA, rB} × targets {t1, t2} = 4 combos, universe drops one.
    label = {("rA", "t1"), ("rB", "t2")}
    universe = {("rA", "t1"), ("rA", "t2"), ("rB", "t1"), ("rB", "t2")}
    xp = rrd.crossproduct_within_universe(label, universe)
    assert xp == universe  # full grid present in universe


def test_is_crossproduct_artifact_flags_overexpansion():
    # Reviewer meant rA/t1 + rB/t2, but submit stored the full 2x2 grid.
    universe = {("rA", "t1"), ("rA", "t2"), ("rB", "t1"), ("rB", "t2")}
    label = set(universe)  # stored cross-product
    opt = {("rA", "t1"), ("rB", "t2")}  # optimizer's diagonal
    assert rrd.is_crossproduct_artifact(label, opt, universe) is True


def test_is_crossproduct_artifact_not_flagged_when_intentional():
    # Label adds a pair but is NOT the full cross-product -> deliberate.
    universe = {("rA", "t1"), ("rA", "t2"), ("rB", "t1"), ("rB", "t2")}
    label = {("rA", "t1"), ("rA", "t2"), ("rB", "t2")}  # missing rB/t1
    opt = {("rA", "t1")}
    assert rrd.is_crossproduct_artifact(label, opt, universe) is False


def test_is_crossproduct_artifact_not_flagged_without_additions():
    # Matches optimizer exactly -> no additions -> never an artifact.
    universe = {("rA", "t1")}
    label = {("rA", "t1")}
    assert rrd.is_crossproduct_artifact(label, label, universe) is False
    # Pure exclusion (label ⊂ opt) also never flags.
    assert (
        rrd.is_crossproduct_artifact({("rA", "t1")}, {("rA", "t1"), ("rB", "t2")}, universe)
        is False
    )


def test_is_crossproduct_artifact_empty_label():
    assert rrd.is_crossproduct_artifact(set(), {("rA", "t1")}, set()) is False


# ---------------------------------------------------------------------------
# build_review integration + reject-all detection
# ---------------------------------------------------------------------------
def test_build_review_added_pair(cache_group, sidecar_group):
    row = {
        "group_id": "abcd1234ef",
        "match_type": "M:N",
        "session_id": "deanchored_v1",
        "selected_edges": "[{'ref_id': 'refA', 'target_id': 'tgt_1_h'}, {'ref_id': 'refB', 'target_id': 'tgt_2_h'}]",
    }
    rv = rrd.build_review(row, cache_group, sidecar_group)
    assert rv.added == {("refB", "tgt_2_h")}
    assert rv.removed == set()
    assert rv.from_sidecar is True
    assert rv.has_diff and not rv.is_reject_all


def test_build_review_stamps_crossproduct_artifact(cache_group, sidecar_group):
    # label refs {refA,refB} × tgts {t1,t2} ∩ universe == stored pairs, adds refB/t2.
    row = {
        "group_id": "abcd1234ef",
        "match_type": "M:N",
        "session_id": "deanchored_v1",
        "selected_edges": "[{'ref_id': 'refA', 'target_id': 'tgt_1_h'}, {'ref_id': 'refB', 'target_id': 'tgt_2_h'}]",
    }
    rv = rrd.build_review(row, cache_group, sidecar_group)
    assert rv.crossproduct_artifact is True
    assert "xprod?" in rrd.format_summary([rv])


def test_build_review_reject_all(cache_group, sidecar_group):
    row = {"group_id": "abcd1234ef", "match_type": "M:N", "session_id": "", "selected_edges": ""}
    rv = rrd.build_review(row, cache_group, sidecar_group)
    assert rv.is_reject_all
    assert rv.removed == {("refA", "tgt_1_h")}


# ---------------------------------------------------------------------------
# gid matching / filter helpers
# ---------------------------------------------------------------------------
def test_gid_matches_prefix():
    assert rrd.gid_matches("abcd1234ef", None) is True
    assert rrd.gid_matches("abcd1234ef", ["abcd1234"]) is True
    assert rrd.gid_matches("abcd1234ef", ["abcd1234ffff"]) is True  # same 8-char key
    assert rrd.gid_matches("abcd1234ef", ["deadbeef"]) is False


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------
def test_format_summary_flags_old_grouping(cache_group):
    row = {
        "group_id": "abcd1234ef",
        "match_type": "M:N",
        "session_id": "deanchored_v1",
        "selected_edges": "[{'ref_id': 'refA', 'target_id': 'tgt_1_h'}]",
    }
    rv = rrd.build_review(row, cache_group, None)  # no sidecar -> old grouping
    out = rrd.format_summary([rv])
    assert "abcd1234" in out
    assert "old-grp" in out


def test_format_summary_empty():
    assert "no matching" in rrd.format_summary([]).lower()


# ---------------------------------------------------------------------------
# Smoke renders (files land, no content assertions)
# ---------------------------------------------------------------------------
def test_smoke_render_overview(cache_group, sidecar_group, tmp_path):
    row = {
        "group_id": "abcd1234ef",
        "match_type": "M:N",
        "session_id": "deanchored_v1",
        "selected_edges": "[{'ref_id': 'refA', 'target_id': 'tgt_1_h'}, {'ref_id': 'refB', 'target_id': 'tgt_2_h'}]",
    }
    rv = rrd.build_review(row, cache_group, sidecar_group)
    out = tmp_path / "review_abcd1234.png"
    assert rrd.render_overview(rv, cache_group, out) == out
    assert out.exists() and out.stat().st_size > 0


def test_smoke_render_zoom(cache_group, sidecar_group, tmp_path):
    row = {
        "group_id": "abcd1234ef",
        "match_type": "M:N",
        "session_id": "deanchored_v1",
        "selected_edges": "[{'ref_id': 'refA', 'target_id': 'tgt_1_h'}, {'ref_id': 'refB', 'target_id': 'tgt_2_h'}]",
    }
    rv = rrd.build_review(row, cache_group, sidecar_group)
    out = tmp_path / "zoom_abcd1234.png"
    assert rrd.render_zoom(rv, cache_group, sidecar_group, out) == out
    assert out.exists() and out.stat().st_size > 0


def test_smoke_render_zoom_none_when_no_diff(cache_group, sidecar_group, tmp_path):
    row = {
        "group_id": "abcd1234ef",
        "match_type": "M:N",
        "session_id": "deanchored_v1",
        "selected_edges": "[{'ref_id': 'refA', 'target_id': 'tgt_1_h'}]",
    }
    rv = rrd.build_review(row, cache_group, sidecar_group)  # matches optimizer exactly
    assert rrd.render_zoom(rv, cache_group, sidecar_group, tmp_path / "z.png") is None


def test_smoke_render_reject_all_overview(cache_group, tmp_path):
    row = {"group_id": "abcd1234ef", "match_type": "M:N", "session_id": "", "selected_edges": ""}
    rv = rrd.build_review(row, cache_group, None)
    out = tmp_path / "review_reject.png"
    rrd.render_overview(rv, cache_group, out)
    assert out.exists() and out.stat().st_size > 0
