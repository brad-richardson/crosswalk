"""Regression tests for M2 candidate-graph persistence in the groups sidecar.

Covers:
- rejected (non-selected) candidate edges are persisted in a SEPARATE
  ``rejected_edges`` list with ``selected: false`` and the structural layer;
- the in-product ``edges`` list and every gate-relevant field are UNCHANGED
  whether or not rejected persistence is enabled (the stitch-gate invariance
  contract — see runner.py::_export_groups_sidecar);
- the per-group cap + truncation accounting;
- the confidence-drop prune's effect is recorded (``pruned`` / ``n_pruned``);
- the ``stitch_eval`` label->group mapping (a gate consumer) ignores the new
  ``rejected_edges`` list entirely.
"""

from __future__ import annotations

import json

import geopandas as gpd
from shapely import LineString

from crosswalk.config import settings
from crosswalk.matching.types import MatchDecision, MatchResult
from crosswalk.pipeline.runner import _export_groups_sidecar

# Metric CRS so sliver/structure computation runs on meters (as in production).
_CRS = "EPSG:32619"


def _ref_gdf():
    # Two collinear ref segments (a corridor) + a far-away distractor ref.
    return gpd.GeoDataFrame(
        {"id": ["R1", "R2", "R3"]},
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(0, 500), (100, 500)]),
        ],
        crs=_CRS,
    )


def _tgt_gdf():
    # One long local target covering the corridor + a second nearby target.
    return gpd.GeoDataFrame(
        {"id": ["T1", "T2"]},
        geometry=[
            LineString([(0, 1), (200, 1)]),
            LineString([(0, 3), (200, 3)]),
        ],
        crs=_CRS,
    )


def _mr(ref, tgt, conf, ref_idx, tgt_idx, gid=None):
    feats = {"group_id": gid} if gid else {}
    return MatchResult(
        ref_id=ref,
        target_id=tgt,
        decision=MatchDecision.MATCH,
        confidence=conf,
        score_breakdown={},
        features=feats,
        ref_idx=ref_idx,
        target_idx=tgt_idx,
        gers_start_frac=0.0,
        gers_end_frac=1.0,
        local_start_frac=0.0,
        local_end_frac=1.0,
    )


def _scenario():
    """N:1 group {R1,R2}->T1, with rejected candidates R1->T2 and R3->T1."""
    ref, tgt = _ref_gdf(), _tgt_gdf()
    selected = [
        _mr("R1", "T1", 0.98, 0, 0, gid="g1"),
        _mr("R2", "T1", 0.95, 1, 0, gid="g1"),
    ]
    # Non-selected raw candidates the optimizer saw (>= min_confidence):
    rejected_cands = [
        _mr("R1", "T2", 0.42, 0, 1),  # R1 alt target (out of group)
        _mr("R3", "T1", 0.30, 2, 0),  # alt ref for T1 (out of group)
    ]
    results = selected + rejected_cands
    return ref, tgt, results, selected


def _load(path):
    return json.loads(path.read_text())["groups"]


def _export(tmp_path, results, optimized, ref, tgt, pruned_pairs=None, pruned_group_ids=None):
    out = tmp_path / "bridge.parquet"
    p = _export_groups_sidecar(
        results=results,
        optimized=optimized,
        output_path=out,
        reference=ref,
        target=tgt,
        min_confidence=0.1,
        ref_id_column="id",
        target_id_column="id",
        reference_proj=ref,
        target_proj=tgt,
        pruned_pairs=pruned_pairs,
        pruned_group_ids=pruned_group_ids,
    )
    return _load(p)


def test_rejected_edges_persisted_with_selected_false(tmp_path):
    ref, tgt, results, selected = _scenario()
    groups = _export(tmp_path, results, selected, ref, tgt)
    assert len(groups) == 1
    g = groups[0]
    # in-product edges are the selected assignment
    assert {(e["ref_id"], e["target_id"]) for e in g["edges"]} == {("R1", "T1"), ("R2", "T1")}
    assert all(e["selected"] for e in g["edges"])
    # rejected candidates are persisted separately, all selected=False
    rej = {(e["ref_id"], e["target_id"]) for e in g["rejected_edges"]}
    assert ("R1", "T2") in rej
    assert ("R3", "T1") in rej
    assert all(e["selected"] is False for e in g["rejected_edges"])
    assert g["n_rejected_edges"] == len(g["rejected_edges"]) == g["n_rejected_total"]
    assert g["rejected_truncated"] is False
    # rejected edges carry the structural layer (degree etc.)
    for e in g["rejected_edges"]:
        assert "degree_ref" in e and "is_bridge" in e and "confidence" in e


def test_edges_and_gate_fields_invariant_to_rejected_persistence(tmp_path):
    """Toggling stitch_persist_rejected_edges must not perturb `edges` or any
    gate-relevant field — only add/remove the sibling rejected list."""
    ref, tgt, results, selected = _scenario()
    groups_on = _export(tmp_path / "on", results, selected, ref, tgt)

    orig = settings.stitch_persist_rejected_edges
    try:
        settings.stitch_persist_rejected_edges = False
        groups_off = _export(tmp_path / "off", results, selected, ref, tgt)
    finally:
        settings.stitch_persist_rejected_edges = orig

    g_on, g_off = groups_on[0], groups_off[0]
    for key in (
        "edges",
        "ref_geometries",
        "target_geometries",
        "ref_ids",
        "target_ids",
        "match_type",
        "n_edges",
        "n_corridors",
        "n_assignment_components",
        "largest_biconnected_block",
        "oversized_group",
    ):
        assert json.dumps(g_on[key], sort_keys=True) == json.dumps(g_off[key], sort_keys=True), key
    # OFF yields no rejected edges; ON does.
    assert g_off["rejected_edges"] == [] and g_off["n_rejected_edges"] == 0
    assert g_on["n_rejected_edges"] > 0


def test_rejected_cap_and_truncation(tmp_path):
    ref, tgt, results, selected = _scenario()
    orig = settings.stitch_rejected_edges_max_per_group
    try:
        settings.stitch_rejected_edges_max_per_group = 1
        groups = _export(tmp_path, results, selected, ref, tgt)
    finally:
        settings.stitch_rejected_edges_max_per_group = orig
    g = groups[0]
    assert g["n_rejected_edges"] == 1
    assert g["n_rejected_total"] == 2
    assert g["rejected_truncated"] is True
    # the retained one is the highest-confidence rejected candidate (R1->T2, 0.42)
    assert (g["rejected_edges"][0]["ref_id"], g["rejected_edges"][0]["target_id"]) == ("R1", "T2")


def test_prune_effect_recorded(tmp_path):
    """When a selected edge is pruned, the sidecar marks it and counts it."""
    ref, tgt, results, selected = _scenario()
    # Simulate the prune having dropped R2->T1: it leaves the assignment and the
    # sidecar should record its effect.
    kept = [r for r in selected if (r.ref_id, r.target_id) != ("R2", "T1")]
    pruned_pairs = {("R2", "T1")}
    groups = _export(tmp_path, results, kept, ref, tgt, pruned_pairs=pruned_pairs)
    g = groups[0]
    assert g["n_pruned"] >= 1
    pruned_marked = [e for e in g["edges"] + g["rejected_edges"] if e.get("pruned")]
    assert any((e["ref_id"], e["target_id"]) == ("R2", "T1") for e in pruned_marked)


def test_no_pruned_key_when_no_prune(tmp_path):
    ref, tgt, results, selected = _scenario()
    groups = _export(tmp_path, results, selected, ref, tgt)
    g = groups[0]
    assert g["n_pruned"] == 0
    assert all("pruned" not in e for e in g["edges"] + g["rejected_edges"])


def test_pruned_edge_exempt_from_truncation_cap(tmp_path):
    """A pruned edge collected as a rejected candidate must never be dropped by
    the per-group cap — else ``n_pruned`` undercounts. With the cap at 0, every
    NON-pruned rejected candidate is truncated, but the pruned one survives."""
    ref, tgt, results, selected = _scenario()  # N:1 {R1,R2}->T1
    # Prune drops R2->T1 (T1 survives via R1->T1), so it becomes a rejected
    # candidate incident to T1 alongside non-pruned R1->T2 and R3->T1.
    kept = [r for r in selected if (r.ref_id, r.target_id) != ("R2", "T1")]
    pruned_pairs = {("R2", "T1")}
    orig = settings.stitch_rejected_edges_max_per_group
    try:
        settings.stitch_rejected_edges_max_per_group = 0
        groups = _export(tmp_path, results, kept, ref, tgt, pruned_pairs=pruned_pairs)
    finally:
        settings.stitch_rejected_edges_max_per_group = orig
    g = groups[0]
    # non-pruned rejected candidates are truncated (cap 0) ...
    assert g["rejected_truncated"] is True
    # ... but the pruned edge survives and is counted exactly.
    assert g["n_pruned"] == 1
    pruned_marked = [e for e in g["edges"] + g["rejected_edges"] if e.get("pruned")]
    assert any((e["ref_id"], e["target_id"]) == ("R2", "T1") for e in pruned_marked)
    # no non-pruned rejected candidate survived the cap
    assert all(e.get("pruned") for e in g["rejected_edges"])


def test_pruned_pendant_edge_recorded(tmp_path):
    """A pruned edge whose BOTH endpoints leave the group post-prune (a pendant)
    is recorded and counted, attributed via its pre-prune group_id, so n_pruned
    is exact. Without the attribution it would silently drop from the sidecar."""
    ref, tgt = _ref_gdf(), _tgt_gdf()  # R1,R2,R3 ; T1,T2
    # Pre-prune M:N group g1 = {R1,R2} x {T1,T2} with two selected components:
    # R1->T1 (kept top) and R2->T2 (pruned) -> R2 and T2 both leave the group.
    r1t1 = _mr("R1", "T1", 0.98, 0, 0, gid="g1")
    r2t2 = _mr("R2", "T2", 0.30, 1, 1, gid="g1")
    results = [r1t1, r2t2]
    kept = [r1t1]
    groups = _export(
        tmp_path,
        results,
        kept,
        ref,
        tgt,
        pruned_pairs={("R2", "T2")},
        pruned_group_ids={("R2", "T2"): "g1"},
    )
    g = next(gg for gg in groups if gg["group_id"] == "g1")
    # the pendant is neither an in-product edge nor incident to a surviving node
    assert ("R2", "T2") not in {(e["ref_id"], e["target_id"]) for e in g["edges"]}
    assert g["n_pruned"] == 1
    all_edges = g["edges"] + g["rejected_edges"]
    assert any((e["ref_id"], e["target_id"]) == ("R2", "T2") and e.get("pruned") for e in all_edges)
    # accounting invariant holds after recovery
    assert g["n_rejected_edges"] <= g["n_rejected_total"]


def test_pruned_pair_counted_once_across_corridor_subgroups(tmp_path):
    """A pruned pair must be counted in EXACTLY its pre-prune (owner) group, even
    when its surviving endpoint anchors a DIFFERENT corridor sub-group. The owner
    records it (pendant recovery); the foreign group sees it only as a non-pruned
    incident alternative. Global sum(n_pruned) == number of pruned pairs (no
    double-count). Regresses the cross-group double-count the adversarial review
    of #288 found."""
    import geopandas as gpd
    from shapely import LineString

    # R1,R2,R4 refs ; T1,T2,T3 targets. R2's pruned edge R2->T2 belongs to g1
    # (with R1->T1); R2 also SURVIVES in a separate sub-group g2 via R2->T3.
    ref = gpd.GeoDataFrame(
        {"id": ["R1", "R2", "R4"]},
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 500), (100, 500)]),
            LineString([(100, 500), (200, 500)]),
        ],
        crs=_CRS,
    )
    tgt = gpd.GeoDataFrame(
        {"id": ["T1", "T2", "T3"]},
        geometry=[
            LineString([(0, 1), (100, 1)]),
            LineString([(0, 3), (100, 3)]),
            LineString([(0, 501), (200, 501)]),
        ],
        crs=_CRS,
    )
    r1t1 = _mr("R1", "T1", 0.98, 0, 0, gid="g1")  # keeps g1 alive (top edge)
    r2t3 = _mr("R2", "T3", 0.97, 1, 2, gid="g2")  # R2 survives here
    r4t3 = _mr("R4", "T3", 0.95, 2, 2, gid="g2")
    r2t2 = _mr("R2", "T2", 0.30, 1, 1, gid="g1")  # pruned; pendant w.r.t. g1
    results = [r1t1, r2t3, r4t3, r2t2]
    kept = [r1t1, r2t3, r4t3]
    groups = _export(
        tmp_path,
        results,
        kept,
        ref,
        tgt,
        pruned_pairs={("R2", "T2")},
        pruned_group_ids={("R2", "T2"): "g1"},
    )
    g1 = next(g for g in groups if g["group_id"] == "g1")
    g2 = next(g for g in groups if g["group_id"] == "g2")
    # owner group g1 records the pruned pendant exactly once
    assert g1["n_pruned"] == 1
    # foreign group g2 (where R2 survives) sees R2->T2 as a plain incident
    # candidate, NOT a pruned record
    assert g2["n_pruned"] == 0
    g2_r2t2 = [e for e in g2["rejected_edges"] if (e["ref_id"], e["target_id"]) == ("R2", "T2")]
    assert g2_r2t2 and not g2_r2t2[0].get("pruned")
    # global exactness: one pruned pair -> total n_pruned == 1
    assert sum(g["n_pruned"] for g in groups) == 1


def test_pendant_pruned_edge_lost_without_attribution(tmp_path):
    """Contrast: without the group_id attribution (pre-fix behaviour), the same
    pendant edge is invisible and n_pruned undercounts — documents why the
    attribution is required."""
    ref, tgt = _ref_gdf(), _tgt_gdf()
    r1t1 = _mr("R1", "T1", 0.98, 0, 0, gid="g1")
    r2t2 = _mr("R2", "T2", 0.30, 1, 1, gid="g1")
    groups = _export(tmp_path, [r1t1, r2t2], [r1t1], ref, tgt, pruned_pairs={("R2", "T2")})
    g = next(gg for gg in groups if gg["group_id"] == "g1")
    assert g["n_pruned"] == 0  # pendant dropped when unattributed


def test_stitch_eval_mapping_ignores_rejected_edges(tmp_path):
    """recover_labeled_groups (a gate consumer) maps labels by the `edges`
    candidate set only — the new rejected_edges list must not change mapping."""
    import pandas as pd

    from crosswalk.agent_labeling.stitch_eval import recover_labeled_groups

    ref, tgt, results, selected = _scenario()
    groups = _export(tmp_path, results, selected, ref, tgt)
    human = pd.DataFrame(
        [
            {
                "group_id": "hg1",
                "selected_edges": json.dumps(
                    [{"ref_id": "R1", "target_id": "T1"}, {"ref_id": "R2", "target_id": "T1"}]
                ),
            }
        ]
    )
    rec = recover_labeled_groups(groups, human)
    # A rejected edge (R1,T2)/(R3,T1) must NOT create a spurious mapping target;
    # the human label maps cleanly to the one real group by its selected edges.
    assert rec["clean"] == [("hg1", "g1")]
    assert rec["split"] == []


def test_nan_confidence_excluded_and_json_stays_strict(tmp_path):
    """A NaN-confidence candidate must not leak a bare `NaN` token into the
    sidecar JSON (invalid strict JSON for non-Python consumers). NaN
    comparisons are always False, so a naive `r.confidence < min_confidence`
    floor check lets it through; the fix treats NaN as failing the floor, so
    the candidate is excluded from the sidecar entirely (not merely
    re-encoded as null)."""
    ref, tgt, results, selected = _scenario()
    # Corrupt the R3->T1 rejected candidate's confidence to NaN (e.g. an
    # upstream feature-computation glitch).
    results = [
        _mr("R3", "T1", float("nan"), 2, 0) if (r.ref_id, r.target_id) == ("R3", "T1") else r
        for r in results
    ]
    out = tmp_path / "bridge.parquet"
    path = _export_groups_sidecar(
        results=results,
        optimized=selected,
        output_path=out,
        reference=ref,
        target=tgt,
        min_confidence=0.1,
        ref_id_column="id",
        target_id_column="id",
        reference_proj=ref,
        target_proj=tgt,
    )
    raw = path.read_text()

    def _reject_constant(token):
        raise ValueError(f"non-finite constant leaked into JSON: {token}")

    # Strict round-trip: no bare NaN/Infinity tokens anywhere in the file.
    data = json.loads(raw, parse_constant=_reject_constant)

    g = next(gg for gg in data["groups"] if gg["group_id"] == "g1")
    # The NaN-confidence candidate is dropped outright, not re-encoded as null.
    assert ("R3", "T1") not in {(e["ref_id"], e["target_id"]) for e in g["rejected_edges"]}
    # The other (finite-confidence) rejected candidate is unaffected.
    assert ("R1", "T2") in {(e["ref_id"], e["target_id"]) for e in g["rejected_edges"]}
