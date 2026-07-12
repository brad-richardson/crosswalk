"""Tests for FULL candidate-graph persistence in the groups sidecar.

Learned-resolver flip condition #1 (docs/SCALING_ROADMAP.md): the sidecar must
persist EVERY candidate edge in a group's component (pre-selection, with ML
confidences), marking which the optimizer selected — otherwise optimizer
under-selection is unlearnable (research/learned_group_resolver_prototype.md).

Covers:
- ``candidate_edges`` round-trips with correct ``selected`` flags;
- a candidate the optimizer dropped is recorded with ``selected: false``;
- deterministic ordering (sorted by (ref_id, target_id)) independent of the
  input results order;
- attribution: each component candidate appears in EXACTLY ONE group;
- ``selected_elsewhere`` marks pairs the optimizer selected in another
  group / as a 1:1 so the resolver never learns them as drops;
- purely additive: toggling the flag leaves every pre-existing key
  byte-identical, and existing consumers (build_stitch_options, the queue
  refresh pair-set contract) ignore the new key.
"""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely import LineString

from crosswalk.config import FEATURE_COLUMNS, FEATURE_VERSION, settings
from crosswalk.filenames import candidates_sidecar_path
from crosswalk.matching.stitch_options import build_stitch_options
from crosswalk.matching.stitch_queue_refresh import (
    check_queue_optimizer_parity,
    optimizer_pair_set,
    selected_pair_set,
)
from crosswalk.matching.types import MatchDecision, MatchResult
from crosswalk.pipeline.runner import _export_groups_sidecar, _signed_lateral_offset_m

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
    return gpd.GeoDataFrame(
        {"id": ["T1", "T2"]},
        geometry=[
            LineString([(0, 1), (200, 1)]),
            LineString([(0, 3), (200, 3)]),
        ],
        crs=_CRS,
    )


def _mr(ref, tgt, conf, ref_idx, tgt_idx, gid=None):
    feats = {"hausdorff_distance_m": float(ref_idx + tgt_idx + 1)}
    if gid:
        feats["group_id"] = gid
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
    """N:1 group {R1,R2}->T1, with dropped candidates R1->T2 and R3->T1."""
    ref, tgt = _ref_gdf(), _tgt_gdf()
    selected = [
        _mr("R1", "T1", 0.98, 0, 0, gid="g1"),
        _mr("R2", "T1", 0.95, 1, 0, gid="g1"),
    ]
    dropped = [
        _mr("R1", "T2", 0.42, 0, 1),  # R1 alt target the optimizer dropped
        _mr("R3", "T1", 0.30, 2, 0),  # alt ref for T1 the optimizer dropped
    ]
    return ref, tgt, selected + dropped, selected


def _load(path):
    return json.loads(path.read_text())["groups"]


def _export(tmp_path, results, optimized, ref, tgt, **kwargs):
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
        dataset_id="toy",
        **kwargs,
    )
    return _load(p)


def test_candidate_edges_round_trip_with_selected_flags(tmp_path):
    ref, tgt, results, selected = _scenario()
    groups = _export(tmp_path, results, selected, ref, tgt)
    assert len(groups) == 1
    g = groups[0]
    cand = g["candidate_edges"]
    # The FULL component candidate graph: selected assignment + dropped pairs.
    assert {(e["ref_id"], e["target_id"]) for e in cand} == {
        ("R1", "T1"),
        ("R2", "T1"),
        ("R1", "T2"),
        ("R3", "T1"),
    }
    by_pair = {(e["ref_id"], e["target_id"]): e for e in cand}
    assert by_pair[("R1", "T1")]["selected"] is True
    assert by_pair[("R2", "T1")]["selected"] is True
    assert by_pair[("R1", "T2")]["selected"] is False
    assert by_pair[("R3", "T1")]["selected"] is False
    # Confidences are the raw ML scores of the candidates.
    assert by_pair[("R1", "T2")]["confidence"] == 0.42
    assert by_pair[("R3", "T1")]["confidence"] == 0.3
    assert g["n_candidate_edges"] == 4
    # No pair here was selected in another group / as a 1:1.
    assert all("selected_elsewhere" not in e for e in cand)


def test_typed_candidate_parquet_covers_full_graph_with_runtime_features(tmp_path):
    ref, tgt, results, selected = _scenario()
    out = tmp_path / "bridge.parquet"
    sidecar = _export_groups_sidecar(
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
        dataset_id="toy",
    )
    groups = _load(sidecar)
    frame = pd.read_parquet(candidates_sidecar_path(out))

    json_keys = {
        (group["group_id"], edge["ref_id"], edge["target_id"])
        for group in groups
        for edge in group["candidate_edges"]
    }
    parquet_keys = set(
        frame[["group_id", "ref_id", "target_id"]].itertuples(index=False, name=None)
    )
    assert parquet_keys == json_keys
    assert len(frame) == 4
    assert frame["dataset_id"].eq("toy").all()
    assert set(FEATURE_COLUMNS) <= set(frame.columns)
    assert frame[FEATURE_COLUMNS].dtypes.map(lambda dtype: dtype.kind).eq("f").all()
    assert frame["feature_version"].eq(FEATURE_VERSION).all()
    assert frame["schema_version"].eq("1.0").all()
    assert frame["model_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert frame["ref_idx"].notna().all()
    assert frame["target_idx"].notna().all()
    assert frame["hausdorff_distance_m"].notna().all()
    selected_rows = frame.set_index(["ref_id", "target_id"])
    assert selected_rows.loc[("R1", "T1"), "lateral_offset_signed_m"] > 0
    assert selected_rows.loc[("R3", "T1"), "lateral_offset_signed_m"] < 0
    assert bool(selected_rows.loc[("R1", "T1"), "selected"])
    assert selected_rows.loc[("R1", "T2"), "optimizer_decision"] == "rejected"
    assert selected_rows.loc[("R1", "T2"), "decision_reason"] == "optimizer_rejected"


def test_signed_lateral_offset_uses_reference_orientation():
    target = LineString([(0, 2), (10, 2)])
    assert _signed_lateral_offset_m(LineString([(0, 0), (10, 0)]), target) == 2.0
    assert _signed_lateral_offset_m(LineString([(10, 0), (0, 0)]), target) == -2.0


def test_candidate_parquet_flag_is_independent_and_clears_stale_file(tmp_path):
    ref, tgt, results, selected = _scenario()
    out = tmp_path / "bridge.parquet"
    _export_groups_sidecar(
        results,
        selected,
        out,
        ref,
        tgt,
        min_confidence=0.1,
        reference_proj=ref,
        target_proj=tgt,
    )
    candidate_path = candidates_sidecar_path(out)
    assert candidate_path.exists()

    original = settings.stitch_persist_candidates
    try:
        settings.stitch_persist_candidates = False
        _export_groups_sidecar(
            results,
            selected,
            out,
            ref,
            tgt,
            min_confidence=0.1,
            reference_proj=ref,
            target_proj=tgt,
        )
    finally:
        settings.stitch_persist_candidates = original
    assert not candidate_path.exists()


def test_dropped_candidate_recorded_even_when_rejected_persistence_off(tmp_path):
    """candidate_edges is the authoritative pre-selection record: a dropped
    candidate is present with selected=false even with rejected_edges off."""
    ref, tgt, results, selected = _scenario()
    orig = settings.stitch_persist_rejected_edges
    try:
        settings.stitch_persist_rejected_edges = False
        groups = _export(tmp_path, results, selected, ref, tgt)
    finally:
        settings.stitch_persist_rejected_edges = orig
    g = groups[0]
    assert g["rejected_edges"] == []
    dropped = [e for e in g["candidate_edges"] if not e["selected"]]
    assert {(e["ref_id"], e["target_id"]) for e in dropped} == {("R1", "T2"), ("R3", "T1")}


def test_candidate_edges_ordering_deterministic(tmp_path):
    ref, tgt, results, selected = _scenario()
    groups_fwd = _export(tmp_path / "fwd", results, selected, ref, tgt)
    groups_rev = _export(tmp_path / "rev", list(reversed(results)), selected, ref, tgt)
    cand_fwd = groups_fwd[0]["candidate_edges"]
    cand_rev = groups_rev[0]["candidate_edges"]
    # Sorted by (ref_id, target_id) and independent of input results order.
    keys = [(e["ref_id"], e["target_id"]) for e in cand_fwd]
    assert keys == sorted(keys)
    assert cand_fwd == cand_rev


def test_candidate_edges_superset_of_assignment(tmp_path):
    ref, tgt, results, selected = _scenario()
    g = _export(tmp_path, results, selected, ref, tgt)[0]
    assignment = {(e["ref_id"], e["target_id"]) for e in g["optimizer_assignment"]}
    selected_cands = {(e["ref_id"], e["target_id"]) for e in g["candidate_edges"] if e["selected"]}
    assert selected_cands == assignment


def test_selected_elsewhere_marks_one_to_one_winner(tmp_path):
    """A component pair the optimizer selected as a 1:1 (no group_id) is
    attributed to the group with selected=false but flagged selected_elsewhere,
    so a resolver never learns a genuinely-selected pair as an optimizer drop."""
    ref, tgt, results, selected = _scenario()
    # R3->T2 selected as a plain 1:1 (no group). R3 and T2 are pulled into the
    # same component by the dropped candidates R3->T1 and R1->T2.
    r3t2 = _mr("R3", "T2", 0.88, 2, 1)
    results = results + [r3t2]
    optimized = selected + [r3t2]
    g = _export(tmp_path, results, optimized, ref, tgt)[0]
    by_pair = {(e["ref_id"], e["target_id"]): e for e in g["candidate_edges"]}
    e = by_pair[("R3", "T2")]
    assert e["selected"] is False
    assert e["selected_elsewhere"] is True
    # The flag is emitted only when true.
    assert all(
        "selected_elsewhere" not in ee for pair, ee in by_pair.items() if pair != ("R3", "T2")
    )


def test_cross_group_candidate_attributed_to_exactly_one_group(tmp_path):
    """Two groups welded into one component by a weak cross candidate: the
    cross edge appears in exactly one group's candidate_edges (ref-endpoint
    group), and each group's own assignment stays in its own list."""
    ref = gpd.GeoDataFrame(
        {"id": ["R1", "R2", "R4", "R5"]},
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(0, 500), (100, 500)]),
            LineString([(100, 500), (200, 500)]),
        ],
        crs=_CRS,
    )
    tgt = gpd.GeoDataFrame(
        {"id": ["T1", "T3"]},
        geometry=[
            LineString([(0, 1), (200, 1)]),
            LineString([(0, 501), (200, 501)]),
        ],
        crs=_CRS,
    )
    g1 = [_mr("R1", "T1", 0.98, 0, 0, gid="g1"), _mr("R2", "T1", 0.95, 1, 0, gid="g1")]
    g2 = [_mr("R4", "T3", 0.97, 2, 1, gid="g2"), _mr("R5", "T3", 0.94, 3, 1, gid="g2")]
    cross = _mr("R2", "T3", 0.2, 1, 1)  # welds the two groups' component
    groups = _export(tmp_path, g1 + g2 + [cross], g1 + g2, ref, tgt)
    by_gid = {g["group_id"]: g for g in groups}
    assert set(by_gid) == {"g1", "g2"}
    occurrences = [
        gid
        for gid, g in by_gid.items()
        for e in g["candidate_edges"]
        if (e["ref_id"], e["target_id"]) == ("R2", "T3")
    ]
    # Attributed to exactly one group: the ref endpoint's group (rule 3).
    assert occurrences == ["g1"]
    # No candidate pair is duplicated across groups.
    all_pairs = [(e["ref_id"], e["target_id"]) for g in groups for e in g["candidate_edges"]]
    assert len(all_pairs) == len(set(all_pairs))
    # Each group's selected candidate set equals its own assignment.
    for g in by_gid.values():
        sel = {(e["ref_id"], e["target_id"]) for e in g["candidate_edges"] if e["selected"]}
        assert sel == {(e["ref_id"], e["target_id"]) for e in g["optimizer_assignment"]}


def test_pruned_pendant_candidate_restored_to_pre_prune_owner(tmp_path):
    """A confidence-pruned edge whose endpoints both left its surviving group
    remains in the resolver universe, owned exactly once by its pre-prune gid."""
    ref, tgt = _ref_gdf(), _tgt_gdf()
    kept = _mr("R1", "T1", 0.98, 0, 0, gid="g1")
    pruned = _mr("R2", "T2", 0.30, 1, 1, gid="g1")

    original = settings.stitch_persist_rejected_edges
    try:
        # The canonical candidate graph/parquet must not depend on the legacy
        # rejected-edge recovery path that already knew how to retain pendants.
        settings.stitch_persist_rejected_edges = False
        groups = _export(
            tmp_path,
            [kept, pruned],
            [kept],
            ref,
            tgt,
            pruned_pairs={("R2", "T2")},
            pruned_group_ids={("R2", "T2"): "g1"},
        )
        reversed_groups = _export(
            tmp_path / "reversed",
            [pruned, kept],
            [kept],
            ref,
            tgt,
            pruned_pairs={("R2", "T2")},
            pruned_group_ids={("R2", "T2"): "g1"},
        )
    finally:
        settings.stitch_persist_rejected_edges = original
    group = next(g for g in groups if g["group_id"] == "g1")
    assert group["rejected_edges"] == []
    reversed_group = next(g for g in reversed_groups if g["group_id"] == "g1")
    assert group["candidate_edges"] == reversed_group["candidate_edges"]
    by_pair = {(e["ref_id"], e["target_id"]): e for e in group["candidate_edges"]}

    assert by_pair[("R2", "T2")] == {
        "ref_id": "R2",
        "target_id": "T2",
        "confidence": 0.3,
        "selected": False,
        "pruned": True,
    }
    occurrences = [
        g["group_id"]
        for g in groups
        for e in g["candidate_edges"]
        if (e["ref_id"], e["target_id"]) == ("R2", "T2")
    ]
    assert occurrences == ["g1"]

    frame = pd.read_parquet(candidates_sidecar_path(tmp_path / "bridge.parquet"))
    row = frame.set_index(["group_id", "ref_id", "target_id"]).loc[("g1", "R2", "T2")]
    assert bool(row["pruned"])
    assert not bool(row["selected"])
    assert row["optimizer_decision"] == "pruned"
    assert row["decision_reason"] == "confidence_drop_prune"


def test_pruned_candidate_ownership_overrides_foreign_endpoint_attribution(tmp_path):
    """If a pruned pair touches a surviving foreign group, rule 0 moves it back
    to its snapshotted owner instead of teaching that foreign group a drop."""
    ref = gpd.GeoDataFrame(
        {"id": ["R1", "R2", "R3"]},
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
    owner_kept = _mr("R1", "T1", 0.98, 0, 0, gid="g1")
    foreign_a = _mr("R2", "T3", 0.97, 1, 2, gid="g2")
    foreign_b = _mr("R3", "T3", 0.95, 2, 2, gid="g2")
    pruned = _mr("R2", "T2", 0.30, 1, 1, gid="g1")
    groups = _export(
        tmp_path,
        [owner_kept, foreign_a, foreign_b, pruned],
        [owner_kept, foreign_a, foreign_b],
        ref,
        tgt,
        pruned_pairs={("R2", "T2")},
        pruned_group_ids={("R2", "T2"): "g1"},
    )
    by_gid = {g["group_id"]: g for g in groups}
    owners = [
        gid
        for gid, group in by_gid.items()
        for edge in group["candidate_edges"]
        if (edge["ref_id"], edge["target_id"]) == ("R2", "T2")
    ]
    assert owners == ["g1"]
    edge = next(
        e for e in by_gid["g1"]["candidate_edges"] if (e["ref_id"], e["target_id"]) == ("R2", "T2")
    )
    assert edge["pruned"] is True
    assert edge["selected"] is False
    assert "selected_elsewhere" not in edge
    assert sum(g["n_pruned"] for g in groups) == 1
    assert sum(bool(e.get("pruned")) for g in groups for e in g["candidate_edges"]) == 1

    frame = pd.read_parquet(candidates_sidecar_path(tmp_path / "bridge.parquet"))
    json_keys = {
        (g["group_id"], e["ref_id"], e["target_id"]) for g in groups for e in g["candidate_edges"]
    }
    parquet_keys = set(
        frame[["group_id", "ref_id", "target_id"]].itertuples(index=False, name=None)
    )
    assert parquet_keys == json_keys
    parquet_edge = frame.set_index(["group_id", "ref_id", "target_id"]).loc[("g1", "R2", "T2")]
    assert bool(parquet_edge["pruned"])
    assert parquet_edge["optimizer_decision"] == "pruned"


@pytest.mark.parametrize(
    ("pruned_pair", "owner", "message"),
    [
        (("R1", "T1"), "g1", "both selected and pruned"),
        (("R2", "T2"), "missing", "did not survive"),
        (("X", "Y"), "g1", "without a floor-passing"),
    ],
)
def test_invalid_pruned_candidate_metadata_fails_closed(tmp_path, pruned_pair, owner, message):
    ref, tgt, scenario_results, selected = _scenario()
    with pytest.raises(ValueError, match=message):
        _export(
            tmp_path,
            scenario_results,
            selected,
            ref,
            tgt,
            pruned_pairs={pruned_pair},
            pruned_group_ids={pruned_pair: owner},
        )


@pytest.mark.parametrize(
    ("pruned_pairs", "pruned_group_ids"),
    [
        ({("R1", "T2")}, {}),
        (set(), {("R1", "T2"): "g1"}),
    ],
)
def test_pruned_pair_and_owner_map_must_match(tmp_path, pruned_pairs, pruned_group_ids):
    ref, tgt, results, selected = _scenario()
    with pytest.raises(ValueError, match="attribution mismatch"):
        _export(
            tmp_path,
            results,
            selected,
            ref,
            tgt,
            pruned_pairs=pruned_pairs,
            pruned_group_ids=pruned_group_ids,
        )


def test_existing_keys_invariant_to_candidate_graph_flag(tmp_path):
    """Toggling stitch_persist_candidate_graph must not perturb any
    pre-existing sidecar key — purely additive."""
    ref, tgt, results, selected = _scenario()
    groups_on = _export(tmp_path / "on", results, selected, ref, tgt)
    orig = settings.stitch_persist_candidate_graph
    try:
        settings.stitch_persist_candidate_graph = False
        groups_off = _export(tmp_path / "off", results, selected, ref, tgt)
    finally:
        settings.stitch_persist_candidate_graph = orig
    g_on, g_off = groups_on[0], groups_off[0]
    for key in g_off:
        if key in ("candidate_edges", "n_candidate_edges"):
            continue
        assert json.dumps(g_on[key], sort_keys=True) == json.dumps(g_off[key], sort_keys=True), key
    assert g_off["candidate_edges"] == [] and g_off["n_candidate_edges"] == 0
    assert g_on["n_candidate_edges"] == 4


def test_queue_refresh_pair_sets_ignore_candidate_edges(tmp_path):
    """The queue-refresh stability contract compares SELECTED pair-sets only
    (optimizer_assignment / edges[selected]); candidate_edges — which carries
    volatile confidences and non-proposed pairs — must not leak into it."""
    ref, tgt, results, selected = _scenario()
    g = _export(tmp_path, results, selected, ref, tgt)[0]
    assert g["candidate_edges"]  # the new key is present...
    expected = {("R1", "T1"), ("R2", "T1")}
    # ...but both canonical pair-sets still reflect only the selection.
    assert selected_pair_set(g) == expected
    assert optimizer_pair_set(g) == expected
    # A queue entry snapshotted WITHOUT candidate_edges is still in parity.
    queue_entry = {k: v for k, v in g.items() if k != "candidate_edges"}
    assert check_queue_optimizer_parity([queue_entry], {g["group_id"]: g}) == []


def test_build_stitch_options_unaffected(tmp_path):
    ref, tgt, results, selected = _scenario()
    g = _export(tmp_path, results, selected, ref, tgt)[0]
    ctx = build_stitch_options(g)
    g_no_cand = {k: v for k, v in g.items() if k not in ("candidate_edges", "n_candidate_edges")}
    ctx_no_cand = build_stitch_options(g_no_cand)
    assert ctx == ctx_no_cand
