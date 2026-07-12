"""Tests for persisting optimizer demotion reasons into the groups sidecar.

PR #371 (Mode A: #367) added ``PARALLEL_SIBLING_REVIEW_FLAG`` — set on a
``MatchResult.features`` dict when the optimizer demotes a contested
small-span stub edge from MATCH to REVIEW (see
``optimizer._contested_small_span_review_pairs`` / ``_create_group_results``).
A follow-up code review found the flag had ZERO consumers: the groups sidecar
serialized edges without any trace of it, so demoted edges were invisible
outside the bridge parquet.

This module covers the fix: ``_export_groups_sidecar`` now reads the flag off
each group's optimizer assignment and stamps a per-edge, additive
``review_reason`` key (currently only ``"parallel_sibling"``) on exactly the
demoted edge(s); every other edge carries no such key. UI surfacing is
deliberately deferred (Brad: "persist now, UI later").
"""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely import LineString

from crosswalk.filenames import candidates_sidecar_path
from crosswalk.matching.optimizer import (
    DECOMPOSED_SINGLETON_REVIEW_FLAG,
    LOW_CONFIDENCE_ADDITION_REVIEW_FLAG,
    PARALLEL_SIBLING_REVIEW_FLAG,
    PRUNED_SINGLETON_REVIEW_FLAG,
    _contested_small_span_review_pairs,
    _create_group_results,
)
from crosswalk.matching.types import MatchDecision, MatchResult, MatchType
from crosswalk.pipeline.runner import _export_groups_sidecar

_CRS = "EPSG:32619"


def _edge(ref_id, target_id, conf, ref_span, tgt_span):
    """MatchResult with alignment fractions producing the given spans (from 0).

    Mirrors ``test_optimizer.py::_edge`` (kept local so this module has no
    cross-test-file dependency).
    """
    return MatchResult(
        ref_id,
        target_id,
        MatchDecision.MATCH,
        conf,
        {},
        {},
        gers_start_frac=0.0,
        gers_end_frac=ref_span,
        local_start_frac=0.0,
        local_end_frac=tgt_span,
    )


def _len_geoms(edges, length=100.0):
    """``length``-meter LineString geoms for every ref/target id in ``edges``."""
    line = LineString([(0.0, 0.0), (0.0, length)])
    refs = {e.ref_id: line for e in edges}
    tgts = {e.target_id: line for e in edges}
    return refs, tgts


def _gdf(id_geom: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"id": list(id_geom.keys())}, geometry=list(id_geom.values()), crs=_CRS)


def _contested_stub_scenario():
    """The #367 "193ac00f shape": a contested small-span stub demoted to REVIEW.

    r_b covers ~88% of its ref to t_a (conf 1.0); r_a covers ~90% of t_b (conf
    0.998); the stub r_b->t_b covers only ~11%/8% of ref/target (conf 0.97)
    and is contested on both its ref (r_b) and target (t_b) — the
    parallel-sibling crossing stub that Mode A demotes.
    """
    edges = [
        _edge("r_a", "t_b", 0.998, 0.986, 0.906),
        _edge("r_b", "t_a", 1.0, 0.881, 1.0),
        _edge("r_b", "t_b", 0.97, 0.108, 0.079),
    ]
    ref_geoms, tgt_geoms = _len_geoms(edges)
    review_pairs = _contested_small_span_review_pairs(edges, ref_geoms, tgt_geoms)
    assert review_pairs == {("r_b", "t_b")}, "scenario must demote exactly the stub edge"
    tagged = _create_group_results(edges, MatchType.M_TO_N, review_pairs=review_pairs)
    ref_gdf = _gdf(ref_geoms)
    tgt_gdf = _gdf(tgt_geoms)
    return edges, tagged, ref_gdf, tgt_gdf


def _export(tmp_path, results, optimized, ref, tgt):
    out = tmp_path / "bridge.parquet"
    path = _export_groups_sidecar(
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
    )
    return path


def test_demoted_edge_carries_review_reason(tmp_path):
    """An optimizer run that demotes an edge -> the sidecar carries the reason
    on exactly that edge; sibling (non-demoted) edges carry no such key."""
    edges, tagged, ref_gdf, tgt_gdf = _contested_stub_scenario()
    # Sanity: the tagged optimizer output really did demote the stub, mirroring
    # the PARALLEL_SIBLING_REVIEW_FLAG contract PR #371 pinned in test_optimizer.py.
    by_pair = {(r.ref_id, r.target_id): r for r in tagged}
    assert by_pair[("r_b", "t_b")].decision == MatchDecision.REVIEW
    assert by_pair[("r_b", "t_b")].features.get(PARALLEL_SIBLING_REVIEW_FLAG) == 1.0

    path = _export(tmp_path, edges, tagged, ref_gdf, tgt_gdf)
    assert path is not None
    data = json.loads(path.read_text())
    assert len(data["groups"]) == 1
    group = data["groups"][0]

    edges_by_pair = {(e["ref_id"], e["target_id"]): e for e in group["edges"]}
    assert set(edges_by_pair) == {("r_a", "t_b"), ("r_b", "t_a"), ("r_b", "t_b")}

    demoted = edges_by_pair[("r_b", "t_b")]
    assert demoted["review_reason"] == "parallel_sibling"

    # Non-demoted edges carry no review_reason key at all (not null, absent).
    for pair in (("r_a", "t_b"), ("r_b", "t_a")):
        assert "review_reason" not in edges_by_pair[pair]


def test_non_demoted_group_has_no_review_reason_key(tmp_path):
    """A group the optimizer never demoted has zero ``review_reason`` keys
    anywhere in its serialized edges — the key is purely additive."""
    ref = gpd.GeoDataFrame(
        {"id": ["R1", "R2"]},
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
        ],
        crs=_CRS,
    )
    tgt = gpd.GeoDataFrame(
        {"id": ["T1"]},
        geometry=[LineString([(0, 1), (200, 1)])],
        crs=_CRS,
    )

    def _mr(ref_id, tgt_id, conf, ref_idx, tgt_idx):
        return MatchResult(
            ref_id=ref_id,
            target_id=tgt_id,
            decision=MatchDecision.MATCH,
            confidence=conf,
            score_breakdown={},
            features={"group_id": "g1"},
            ref_idx=ref_idx,
            target_idx=tgt_idx,
        )

    selected = [
        _mr("R1", "T1", 0.98, 0, 0),
        _mr("R2", "T1", 0.95, 1, 0),
    ]
    path = _export(tmp_path, selected, selected, ref, tgt)
    assert path is not None
    data = json.loads(path.read_text())
    group = data["groups"][0]
    assert len(group["edges"]) == 2
    for e in group["edges"]:
        assert "review_reason" not in e


def test_sidecar_stays_strict_json_with_review_reason(tmp_path):
    """The additive key does not break strict JSON round-tripping (no bare
    NaN/Infinity tokens -- mirrors the existing sidecar strictness contract)."""
    edges, tagged, ref_gdf, tgt_gdf = _contested_stub_scenario()
    path = _export(tmp_path, edges, tagged, ref_gdf, tgt_gdf)
    raw = path.read_text()

    def _reject_constant(token):
        raise ValueError(f"non-finite constant leaked into JSON: {token}")

    data = json.loads(raw, parse_constant=_reject_constant)
    group = data["groups"][0]
    edges_by_pair = {(e["ref_id"], e["target_id"]): e for e in group["edges"]}
    assert edges_by_pair[("r_b", "t_b")]["review_reason"] == "parallel_sibling"


@pytest.mark.parametrize(
    ("review_flag", "expected_reason"),
    [
        (LOW_CONFIDENCE_ADDITION_REVIEW_FLAG, "low_confidence_addition"),
        (PRUNED_SINGLETON_REVIEW_FLAG, "pruned_singleton"),
        (DECOMPOSED_SINGLETON_REVIEW_FLAG, "decomposed_singleton"),
    ],
)
def test_review_reason_reaches_every_optimizer_provenance_view(
    tmp_path,
    review_flag,
    expected_reason,
):
    """New review modes stay visible in JSON assignment views and parquet."""
    edges = [
        _edge("r_a", "t_a", 0.95, 1.0, 0.5),
        _edge("r_b", "t_a", 0.6, 1.0, 0.5),
    ]
    ref_geoms, tgt_geoms = _len_geoms(edges)
    optimized = _create_group_results(
        edges,
        MatchType.N_TO_ONE,
        review_pairs={("r_b", "t_a")},
        review_flag=review_flag,
    )
    bridge_path = tmp_path / "bridge.parquet"

    groups_path = _export_groups_sidecar(
        results=edges,
        optimized=optimized,
        output_path=bridge_path,
        reference=_gdf(ref_geoms),
        target=_gdf(tgt_geoms),
        min_confidence=0.1,
        ref_id_column="id",
        target_id_column="id",
        reference_proj=_gdf(ref_geoms),
        target_proj=_gdf(tgt_geoms),
        dataset_id="toy",
    )

    group = json.loads(groups_path.read_text())["groups"][0]
    for source in ("edges", "optimizer_assignment", "candidate_edges"):
        by_pair = {(edge["ref_id"], edge["target_id"]): edge for edge in group[source]}
        reviewed = by_pair[("r_b", "t_a")]
        assert reviewed["decision"] == "review"
        assert reviewed["review_reason"] == expected_reason

    candidates = pd.read_parquet(candidates_sidecar_path(bridge_path))
    reviewed_row = candidates[
        (candidates["ref_id"] == "r_b") & (candidates["target_id"] == "t_a")
    ].iloc[0]
    assert reviewed_row["optimizer_decision"] == "review"
    assert reviewed_row["decision_reason"] == expected_reason


def test_duplicate_pair_export_uses_canonical_optimizer_row(tmp_path):
    """Equal-score duplicate rows cannot make sidecar provenance order-sensitive."""
    bad = MatchResult(
        "r_a",
        "t",
        MatchDecision.MATCH,
        0.9,
        {},
        {},
        ref_idx=0,
        target_idx=0,
        gers_start_frac=0.0,
        gers_end_frac=0.2,
        local_start_frac=0.0,
        local_end_frac=0.2,
    )
    good = MatchResult(
        "r_a",
        "t",
        MatchDecision.MATCH,
        0.9,
        {},
        {},
        ref_idx=1,
        target_idx=0,
        gers_start_frac=0.0,
        gers_end_frac=0.8,
        local_start_frac=0.0,
        local_end_frac=0.8,
    )
    sibling = MatchResult(
        "r_b",
        "t",
        MatchDecision.MATCH,
        0.9,
        {},
        {},
        ref_idx=2,
        target_idx=0,
        gers_start_frac=0.0,
        gers_end_frac=1.0,
        local_start_frac=0.8,
        local_end_frac=1.0,
    )
    optimized = _create_group_results([good, sibling], MatchType.N_TO_ONE)
    ref = _gdf(
        {
            "r_a": LineString([(0, 0), (80, 0)]),
            "r_b": LineString([(80, 0), (100, 0)]),
        }
    )
    target = _gdf({"t": LineString([(0, 1), (100, 1)])})

    serialized = []
    for index, raw in enumerate(([bad, good, sibling], [good, bad, sibling])):
        bridge_path = tmp_path / str(index) / "bridge.parquet"
        bridge_path.parent.mkdir()
        groups_path = _export_groups_sidecar(
            results=raw,
            optimized=optimized,
            output_path=bridge_path,
            reference=ref,
            target=target,
            min_confidence=0.1,
            ref_id_column="id",
            target_id_column="id",
            reference_proj=ref,
            target_proj=target,
            dataset_id="toy",
        )
        group = json.loads(groups_path.read_text())["groups"][0]
        edge = next(
            edge for edge in group["edges"] if (edge["ref_id"], edge["target_id"]) == ("r_a", "t")
        )
        candidate = (
            pd.read_parquet(candidates_sidecar_path(bridge_path))
            .query("ref_id == 'r_a' and target_id == 't'")
            .iloc[0]
        )
        serialized.append((edge["gers_end_frac"], int(candidate["ref_idx"])))

    assert serialized == [(0.8, 1), (0.8, 1)]
