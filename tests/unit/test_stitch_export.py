"""Tests for `crosswalk agent stitch-export` (panel-consensus -> stitching labels).

Fixtures are synthetic (the real batch dirs are gitignored) so these tests are
self-contained and exercise every export gate plus idempotency and schema
round-trip through the real `StitchingLabelStore`.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from loguru import logger

from crosswalk.agent_labeling.matching_rubric import MATCHING_RUBRIC_VERSION
from crosswalk.agent_labeling.stitch_evidence import generate_group_evidence
from crosswalk.agent_labeling.stitch_export import (
    PANEL_LABELER,
    REASON_ABLATION_VARIANT,
    REASON_CLASS_MISMATCH,
    REASON_CONTAINS_SLIVER,
    REASON_EMPTY_SELECTION,
    REASON_EXPORTED,
    REASON_HUMAN_PRECEDENCE,
    REASON_OVER_MAX,
    REASON_STRUCTURAL_TANGLE,
    GroupExport,
    plan_exports,
    write_exports,
    write_vote_provenance,
)
from crosswalk.agent_labeling.stitch_provenance import (
    build_evidence_record,
    consensus_policy_signature,
    load_evidence_manifest,
    sha256_file,
    write_evidence_manifest,
)
from crosswalk.config import settings
from crosswalk.labeling.stitching_store import STITCHING_LABEL_COLUMNS, StitchingLabelStore

VOTES_COLUMNS = [
    "group_id",
    "provider",
    "model",
    "choice",
    "confidence",
    "reasoning",
    "edge_set",
    "latency_s",
    "timestamp",
    "error",
    "invocation_route",
    "pack_feedback",
]


def _write_votes(batch_dir: Path, rows: list[dict]) -> None:
    """Write a synthetic per-batch votes.csv (raw panel ballots)."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    with (batch_dir / "votes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VOTES_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in VOTES_COLUMNS})


CONSENSUS_COLUMNS = [
    "group_id",
    "consensus",
    "choice",
    "edge_set",
    "routing",
    "n_votes",
    "n_valid",
    "minority",
    "mean_confidence",
    "route_reason",
]


def _line(lat0: float, dlat: float) -> dict:
    """A 2-point LineString GeoJSON; dlat=0.0005deg ~= 55m."""
    return {
        "type": "LineString",
        "coordinates": [[-71.06, lat0], [-71.06, lat0 + dlat]],
    }


def _edge(ref: str, tgt: str, span: float = 1.0) -> dict:
    """A candidate edge dict with symmetric ref/target aligned spans."""
    return {
        "ref_id": ref,
        "target_id": tgt,
        "confidence": 0.9,
        "gers_start_frac": 0.0,
        "gers_end_frac": span,
        "local_start_frac": 0.0,
        "local_end_frac": span,
    }


# Blessed voter compositions, written out literally (not imported) so a panel
# change that forgets the provenance decision fails these tests loudly.
_V3_VOTERS = [
    ("claude", "claude-opus-4-8"),
    ("codex", "gpt-5.5"),
    ("agy", "Gemini 3.5 Flash (Medium)"),
]
_V4_VOTERS = [
    ("claude", "claude-opus-4-8"),
    ("codex", "gpt-5.6-terra"),
    ("kimi", "openrouter/moonshotai/kimi-k2.6"),
]
# The v5 quad (2026-07-10 bless): the v4 trio plus muse/Muse Spark 1.1.
_V5_VOTERS = [
    ("claude", "claude-opus-4-8"),
    ("codex", "gpt-5.6-terra"),
    ("kimi", "openrouter/moonshotai/kimi-k2.6"),
    ("muse", "meta/muse-spark-1.1"),
]
_V6_VOTERS = [
    ("claude", "claude-opus-4-8"),
    ("codex", "gpt-5.6-terra"),
    ("muse", "meta/muse-spark-1.1"),
]
_V7_VOTERS = [
    ("claude", "claude-opus-4-8"),
    ("codex", "gpt-5.6-sol"),
    ("muse", "meta/muse-spark-1.1"),
]
# The 2026-07-07 transport-swap composition (commit 80dbe1f): Gemini via
# opencode instead of agy. Committed v3 labels trace to it, so era resolution
# must keep stamping it v3 (while the export gate still flags it).
_HISTORICAL_GEMINI_TRANSPORT_VOTERS = [
    ("claude", "claude-opus-4-8"),
    ("codex", "gpt-5.5"),
    ("opencode", "openrouter/google/gemini-3.5-flash"),
]


def _write_votes_csv(
    batch_dir: Path, voters: list[tuple[str, str]], include_model_col: bool = True
) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    with open(batch_dir / "votes.csv", "w", newline="") as f:
        w = csv.writer(f)
        if include_model_col:
            w.writerow(["group_id", "provider", "model", "choice"])
            for p, m in voters:
                w.writerow(["g1", p, m, "A"])
        else:
            w.writerow(["group_id", "provider", "choice"])
            for p, _m in voters:
                w.writerow(["g1", p, "A"])


def _stamp_matching_rubric(
    batch_dir: Path,
    group_id: str = "g1",
) -> None:
    """Bind a synthetic voted group to a valid current-rubric evidence pack."""
    group_dir = batch_dir / group_id
    group_dir.mkdir(parents=True, exist_ok=True)
    edge = {"ref_id": "r1", "target_id": "t1", "confidence": 0.9}
    options = {
        "optimizer_letter": "A",
        "options": [{"letter": "A", "is_optimizer": True, "edges": [edge]}],
    }
    evidence = build_evidence_record(
        {"group_id": group_id, "edges": [edge]},
        options,
    )
    metadata = {
        "group_id": group_id,
        "segments": {
            "reference": [{"label": "R1", "id": "r1"}],
            "target": [{"label": "T1", "id": "t1"}],
        },
        "optimizer_letter": "A",
        "options": [
            {
                "letter": "A",
                "is_optimizer": True,
                "edges": [{"ref": "R1", "target": "T1"}],
            }
        ],
    }
    (group_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    manifest = write_evidence_manifest(group_dir, evidence)

    links = {
        "evidence_id": evidence["evidence_id"],
        "evidence_pack_sha256": manifest["evidence_pack_sha256"],
    }
    votes_path = batch_dir / "votes.csv"
    votes = pd.read_csv(votes_path, dtype={"group_id": str})
    votes["group_id"] = group_id
    for column, value in links.items():
        votes[column] = value
    votes.to_csv(votes_path, index=False)

    consensus_path = batch_dir / "consensus.csv"
    if consensus_path.exists():
        consensus = pd.read_csv(consensus_path, dtype={"group_id": str})
        mask = consensus["group_id"] == group_id
        assert mask.any(), f"synthetic consensus has no row for {group_id}"
        for column, value in links.items():
            consensus.loc[mask, column] = value
    else:
        consensus = pd.DataFrame([{"group_id": group_id, **links}])
    consensus.to_csv(consensus_path, index=False)


def _tamper_rubric_stamp(batch_dir: Path, group_id: str = "g1") -> None:
    path = batch_dir / group_id / "evidence.json"
    manifest = json.loads(path.read_text())
    manifest["evidence"]["matching_rubric_version"] = f"{MATCHING_RUBRIC_VERSION}-tampered"
    path.write_text(json.dumps(manifest))


def make_batch(
    batch_dir: Path,
    dataset: str,
    groups: list[dict],
    voters: list[tuple[str, str]] | None = _V5_VOTERS,
) -> Path:
    """Write a synthetic batch dir (consensus.csv, batch.json, metadata.yaml).

    Each ``groups`` entry:
      group_id, match_type, routing, choice, mean_confidence,
      edges: list of (ref, tgt) chosen edges (the consensus edge_set),
      ref_classes / target_classes: {id: class},
      sliver_edges: set of (ref, tgt) to encode as tiny-span (sliver) edges,
      candidate_edges: optional extra (ref, tgt) present in batch.json but not chosen.

    ``voters`` writes a votes.csv with those (provider, model) ballots so the
    batch resolves to a labeler era (default: the blessed v5 quad —
    write_exports refuses era-less batches). Pass ``voters=None`` to omit
    votes.csv and exercise the era-less path. Consensus rows default to a
    fully unanimous 4/4 vote (n_votes/n_valid overridable per group, e.g. for
    quorum rows).
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    if voters is not None:
        _write_votes_csv(batch_dir, voters)

    # consensus.csv
    with (batch_dir / "consensus.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONSENSUS_COLUMNS)
        w.writeheader()
        for g in groups:
            w.writerow(
                {
                    "group_id": g["group_id"],
                    "consensus": g.get("consensus", "unanimous"),
                    "choice": g.get("choice", "A"),
                    "edge_set": json.dumps([[r, t] for r, t in g["edges"]]),
                    "routing": g["routing"],
                    "n_votes": g.get("n_votes", 4),
                    "n_valid": g.get("n_valid", 4),
                    "minority": g.get("minority", ""),
                    "mean_confidence": g.get("mean_confidence", 0.9),
                    "route_reason": g.get("route_reason", "unanimous_non_none_small"),
                }
            )

    # batch.json + per-group metadata.yaml
    json_groups = []
    for g in groups:
        ref_classes = g.get("ref_classes", {})
        tgt_classes = g.get("target_classes", {})
        sliver_edges = set(g.get("sliver_edges", set()))
        all_edges = list(g["edges"]) + list(g.get("candidate_edges", []))

        ref_ids = sorted({r for r, _ in all_edges})
        tgt_ids = sorted({t for _, t in all_edges})

        edge_dicts = []
        for r, t in all_edges:
            span = 0.03 if (r, t) in sliver_edges else 1.0
            edge_dicts.append(_edge(r, t, span=span))

        ref_geoms = {r: _line(42.28 + 0.001 * i, 0.0005) for i, r in enumerate(ref_ids)}
        tgt_geoms = {t: _line(42.29 + 0.001 * i, 0.0005) for i, t in enumerate(tgt_ids)}

        json_group = {
            "group_id": g["group_id"],
            "match_type": g.get("match_type", "M:N"),
            "ref_ids": ref_ids,
            "target_ids": tgt_ids,
            "edges": edge_dicts,
            "ref_geometries": ref_geoms,
            "target_geometries": tgt_geoms,
            "ref_classes": {r: ref_classes.get(r, "residential") for r in ref_ids},
            "target_classes": {t: tgt_classes.get(t, "residential") for t in tgt_ids},
        }
        # Optional structure fields (drives the structural export gate). Omit
        # them to exercise the flat-max_edges fallback path.
        for k in ("n_edges", "n_corridors", "n_assignment_components"):
            if k in g:
                json_group[k] = g[k]
        json_groups.append(json_group)

        # metadata.yaml
        meta = {
            "group_id": g["group_id"],
            "match_type": g.get("match_type", "M:N"),
            "segments": {
                "reference": [
                    {
                        "label": f"R{i + 1}",
                        "id": r,
                        "name": r,
                        "class": ref_classes.get(r, "residential"),
                    }
                    for i, r in enumerate(ref_ids)
                ],
                "target": [
                    {
                        "label": f"T{i + 1}",
                        "id": t,
                        "name": t,
                        "class": tgt_classes.get(t, "residential"),
                    }
                    for i, t in enumerate(tgt_ids)
                ],
            },
            "options": [],
        }
        gdir = batch_dir / g["group_id"]
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "metadata.yaml").write_text(yaml.safe_dump(meta))

    (batch_dir / "batch.json").write_text(
        json.dumps({"dataset_id": dataset, "groups": json_groups})
    )
    return batch_dir


@pytest.fixture
def labels_dir(tmp_path):
    return tmp_path / "stitching"


DATASET = "test_ds"


def _plan(batch_dirs, labels_dir, **kw):
    return plan_exports([Path(b) for b in batch_dirs], DATASET, Path(labels_dir), **kw)


def _by_gid(report):
    return {g.group_id: g for g in report.groups}


def test_only_auto_accept_exported(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {"group_id": "g_auto", "routing": "auto_accept", "edges": [("r1", "t1")]},
            {"group_id": "g_human", "routing": "human_review", "edges": [("r2", "t2")]},
        ],
    )
    report = _plan([b], labels_dir)
    gids = _by_gid(report)
    assert report.n_auto_accept == 1
    assert "g_human" not in gids  # not a candidate at all
    assert gids["g_auto"].reason == REASON_EXPORTED


def test_over_max_edges_rejected(tmp_path, labels_dir):
    many = [(f"r{i}", f"t{i}") for i in range(25)]
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [{"group_id": "big", "routing": "auto_accept", "edges": many}],
    )
    report = _plan([b], labels_dir, max_edges=20)
    g = _by_gid(report)["big"]
    assert not g.exported
    assert g.reason == REASON_OVER_MAX
    assert g.n_edges_raw == 25


def test_structural_gate_single_corridor_exports_above_flat_cap(tmp_path, labels_dir):
    """A clean 30-edge single corridor passes even though it exceeds max_edges=20."""
    edges = [(f"r{i}", f"t{i}") for i in range(30)]
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "corridor",
                "routing": "auto_accept",
                "edges": edges,
                "n_edges": 30,
                "n_corridors": 1,
                "n_assignment_components": 1,
            }
        ],
    )
    report = _plan([b], labels_dir, max_edges=20)
    g = _by_gid(report)["corridor"]
    assert g.exported, f"single corridor should export, got {g.reason}"
    assert g.reason == REASON_EXPORTED


def test_structural_gate_small_tangle_blocked(tmp_path, labels_dir):
    """A small multi-corridor tangle is blocked even below the flat cap."""
    edges = [(f"r{i}", f"t{i}") for i in range(10)]
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "tangle",
                "routing": "auto_accept",
                "edges": edges,
                "n_edges": 10,
                "n_corridors": 3,
                "n_assignment_components": 3,
            }
        ],
    )
    report = _plan([b], labels_dir, max_edges=20)
    g = _by_gid(report)["tangle"]
    assert not g.exported
    assert g.reason == REASON_STRUCTURAL_TANGLE


def test_structural_gate_backstop_blocks_giant_single_corridor(tmp_path, labels_dir):
    """Even a single corridor is blocked above the hard backstop ceiling."""
    edges = [(f"r{i}", f"t{i}") for i in range(45)]
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "giant",
                "routing": "auto_accept",
                "edges": edges,
                "n_edges": 45,
                "n_corridors": 1,
                "n_assignment_components": 1,
            }
        ],
    )
    report = _plan([b], labels_dir, max_edges=20, backstop_max_edges=40)
    g = _by_gid(report)["giant"]
    assert not g.exported
    assert g.reason == REASON_OVER_MAX


def test_legacy_fallback_blocks_over_backstop_candidates(tmp_path, labels_dir):
    """calib0709 shape: no structure fields, 45 candidates, 18 selected -> BLOCKED.

    The legacy fallback capped only the SELECTED edge set (18 <= max_edges), so
    an over-backstop group with a small selection exported — minting a panel
    label AND marking the group reviewed (removing it from the human queue),
    resurrecting the size-routing void. The backstop must bind the group's
    CANDIDATE count on the legacy path too (real shapes: calib0709 0cbcf706
    45/18, 2414344b 44/19).
    """
    selected = [(f"r{i}", f"t{i}") for i in range(18)]
    extra = [(f"r{i}", f"tx{i}") for i in range(27)]  # 45 candidates total
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "legacy_monster",
                "routing": "auto_accept",
                "edges": selected,
                "candidate_edges": extra,
                # No n_edges / n_corridors / n_assignment_components: exercises
                # the no-structure-fields fallback.
            }
        ],
    )
    report = _plan([b], labels_dir, max_edges=20, backstop_max_edges=40)
    g = _by_gid(report)["legacy_monster"]
    assert not g.exported
    assert g.reason == REASON_OVER_MAX
    assert g.n_edges_raw == 18


def test_legacy_fallback_under_backstop_still_exports(tmp_path, labels_dir):
    """No structure fields, candidates within the backstop -> still exports."""
    selected = [(f"r{i}", f"t{i}") for i in range(18)]
    extra = [(f"r{i}", f"tx{i}") for i in range(12)]  # 30 candidates total
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "legacy_ok",
                "routing": "auto_accept",
                "edges": selected,
                "candidate_edges": extra,
            }
        ],
    )
    report = _plan([b], labels_dir, max_edges=20, backstop_max_edges=40)
    g = _by_gid(report)["legacy_ok"]
    assert g.exported, f"under-backstop legacy group should export, got {g.reason}"
    assert g.reason == REASON_EXPORTED


def test_class_mismatch_rejected(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "xmode",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "ref_classes": {"r1": "residential"},
                "target_classes": {"t1": "footway"},
            }
        ],
    )
    g = _by_gid(_plan([b], labels_dir))["xmode"]
    assert not g.exported
    assert g.reason == REASON_CLASS_MISMATCH


def test_road_cycleway_class_mismatch_rejected(tmp_path, labels_dir):
    # road<->cycleway is cross-mode on the export path too (co_bogota_bike_network
    # shape: road-class ref, cycleway target) — must not auto-export.
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "bikemode",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "ref_classes": {"r1": "primary"},
                "target_classes": {"t1": "cycleway"},
            }
        ],
    )
    g = _by_gid(_plan([b], labels_dir))["bikemode"]
    assert not g.exported
    assert g.reason == REASON_CLASS_MISMATCH


def test_cycleway_cycleway_exported(tmp_path, labels_dir):
    # cycleway<->cycleway is same-mode and stays exportable.
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "bikebike",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "ref_classes": {"r1": "cycleway"},
                "target_classes": {"t1": "cycleway"},
            }
        ],
    )
    g = _by_gid(_plan([b], labels_dir))["bikebike"]
    assert g.exported
    assert g.reason == REASON_EXPORTED


def test_selected_sliver_blocks_export_without_mutating_selection(tmp_path, labels_dir):
    from crosswalk.agent_labeling.panel_routing import (
        attach_panel_route_reasons,
        panel_failed_group_ids,
    )

    batches_root = tmp_path / "batches"
    b = make_batch(
        batches_root / DATASET,
        DATASET,
        [
            {
                "group_id": "mixed",
                "routing": "auto_accept",
                "edges": [("r1", "t1"), ("r2", "t2")],
                "sliver_edges": {("r2", "t2")},
            }
        ],
    )
    g = _by_gid(_plan([b], labels_dir))["mixed"]
    assert not g.exported
    assert g.reason == REASON_CONTAINS_SLIVER
    assert g.n_slivers_dropped == 0
    assert g.n_edges_final == 2
    assert g.selected_edges == []
    # The export hold has a destination: read-time routing surfaces the exact
    # selection to human review instead of letting the auto-accept disappear.
    assert panel_failed_group_ids(DATASET, batches_root) == {"mixed"}
    groups = [{"group_id": "mixed"}]
    assert attach_panel_route_reasons(groups, DATASET, batches_root) == 1
    assert groups[0]["panel_route_reason"] == REASON_CONTAINS_SLIVER


def test_all_sliver_selection_blocks_export_without_minting_empty_truth(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "allsliver",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "sliver_edges": {("r1", "t1")},
            }
        ],
    )
    g = _by_gid(_plan([b], labels_dir))["allsliver"]
    assert not g.exported
    assert g.reason == REASON_CONTAINS_SLIVER
    assert g.n_slivers_dropped == 0
    assert g.n_edges_final == 1


def test_human_precedence_by_group_id(tmp_path, labels_dir):
    store = StitchingLabelStore(DATASET, labels_dir=labels_dir)
    store.add("g1", [{"ref_id": "rX", "target_id": "tX"}], "1:N", 1, 1, "brad", "s1")
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [{"group_id": "g1", "routing": "auto_accept", "edges": [("r1", "t1")]}],
    )
    g = _by_gid(_plan([b], labels_dir))["g1"]
    assert not g.exported
    assert g.reason == REASON_HUMAN_PRECEDENCE


def test_human_precedence_by_edge_overlap(tmp_path, labels_dir):
    # Human label has a DIFFERENT group_id but shares a candidate edge.
    store = StitchingLabelStore(DATASET, labels_dir=labels_dir)
    store.add(
        "old_hash",
        [{"ref_id": "r1", "target_id": "t1"}],
        "M:N",
        1,
        1,
        "brad",
        "s1",
    )
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "new_hash",
                "routing": "auto_accept",
                "edges": [("r1", "t1"), ("r2", "t2")],
            }
        ],
    )
    g = _by_gid(_plan([b], labels_dir))["new_hash"]
    assert not g.exported
    assert g.reason == REASON_HUMAN_PRECEDENCE
    assert g.human_group_id == "old_hash"


def test_panel_v1_rows_do_not_confer_human_precedence(tmp_path, labels_dir):
    # A prior-generation panel label (v1) covers the group, but it is NOT human,
    # so it must not block a v2 export (prefix match, not exact-tag match).
    store = StitchingLabelStore(DATASET, labels_dir=labels_dir)
    store.add("g1", [{"ref_id": "r1", "target_id": "t1"}], "1:N", 1, 1, "panel_unanimous_v1", "s1")
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [{"group_id": "g1", "routing": "auto_accept", "edges": [("r1", "t1")]}],
    )
    g = _by_gid(_plan([b], labels_dir))["g1"]
    assert g.exported


def test_missing_labeler_does_not_crash_precedence(tmp_path, labels_dir):
    # A hand-edited/legacy row with an empty labeler must be kept as human
    # (na=False) rather than crashing the prefix filter with ~NaN.
    import pandas as pd

    store = StitchingLabelStore(DATASET, labels_dir=labels_dir)
    store.add("g1", [{"ref_id": "r1", "target_id": "t1"}], "1:N", 1, 1, "brad", "s1")
    csv_path = store.csv_path
    df = pd.read_csv(csv_path)
    df.loc[df["group_id"] == "g1", "labeler"] = None
    df.to_csv(csv_path, index=False)
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [{"group_id": "g1", "routing": "auto_accept", "edges": [("r1", "t1")]}],
    )
    g = _by_gid(_plan([b], labels_dir))["g1"]
    assert not g.exported
    assert g.reason == REASON_HUMAN_PRECEDENCE


def test_phase3_supersedes_phase2(tmp_path, labels_dir):
    b2 = make_batch(
        tmp_path / "phase2",
        DATASET,
        [
            {
                "group_id": "shared",
                "routing": "human_review",  # phase2 not accepted
                "edges": [("r9", "t9")],
            }
        ],
    )
    b3 = make_batch(
        tmp_path / "phase3",
        DATASET,
        [
            {
                "group_id": "shared",
                "routing": "auto_accept",  # phase3 accepts, different edges
                "edges": [("r1", "t1")],
            }
        ],
    )
    # phase2 then phase3 -> phase3 wins
    g = _by_gid(_plan([b2, b3], labels_dir))["shared"]
    assert g.exported
    assert g.source_batch == "phase3"
    assert {(e["ref_id"], e["target_id"]) for e in g.selected_edges} == {("r1", "t1")}


def test_gate_ordering_over_max_beats_class(tmp_path, labels_dir):
    # A group that fails BOTH the edge-count and class gates is reported by the
    # earlier gate (edge count).
    many = [(f"r{i}", f"t{i}") for i in range(25)]
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "both",
                "routing": "auto_accept",
                "edges": many,
                "ref_classes": {"r0": "residential"},
                "target_classes": {"t0": "footway"},
            }
        ],
    )
    g = _by_gid(_plan([b], labels_dir, max_edges=20))["both"]
    assert g.reason == REASON_OVER_MAX


def test_class_gate_runs_without_metadata_yaml(tmp_path, labels_dir):
    # When metadata.yaml is absent, classes are recovered from batch.json so the
    # class-consistency gate still fires (no silent degradation).
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "xmode",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "ref_classes": {"r1": "residential"},
                "target_classes": {"t1": "footway"},
            }
        ],
    )
    # Remove the per-group metadata.yaml, keeping batch.json.
    (b / "xmode" / "metadata.yaml").unlink()
    g = _by_gid(_plan([b], labels_dir))["xmode"]
    assert not g.exported
    assert g.reason == REASON_CLASS_MISMATCH


def _set_experiment_variant(batch_dir: Path, variant: str) -> None:
    """Inject an ``experiment.variant`` into a batch's batch.json.

    Mirrors the shape written by ``scripts/build_physical_stitch_wave.py`` — the
    2x2 physical/coincidence ablation wave stamps every batch with an
    ``experiment`` block whose ``variant`` is one of enriched / no_physical /
    no_coincidence / minimal.
    """
    path = Path(batch_dir) / "batch.json"
    payload = json.loads(path.read_text())
    payload["experiment"] = {
        "wave": "physical_context_test",
        "variant": variant,
        "physical_metadata_visible": variant in ("enriched", "no_coincidence"),
        "same_side_coincidence_visible": variant in ("enriched", "no_physical"),
    }
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "variant",
    ["minimal", "no_physical", "no_coincidence", "future_unknown_ablation"],
)
def test_ablation_variant_batch_mints_nothing(tmp_path, labels_dir, variant):
    """A context-stripped ablation-variant batch never mints, even on auto_accept.

    Prevents the v8 ``1b90f03b/minimal`` misfire: an auto_accept produced by a
    context-blinded panel cell must be skipped as ``ablation_variant`` rather
    than promoted to a durable panel label. "Anything present and != enriched"
    is treated as ablation, so an unknown future variant is gated too.
    """
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [{"group_id": "g_auto", "routing": "auto_accept", "edges": [("r1", "t1")]}],
    )
    _set_experiment_variant(b, variant)
    report = _plan([b], labels_dir)
    g = _by_gid(report)["g_auto"]
    assert not g.exported
    assert g.reason == REASON_ABLATION_VARIANT
    assert report.skipped_by_reason().get(REASON_ABLATION_VARIANT) == 1
    # And it mints nothing through the write path.
    assert write_exports(report, DATASET, Path(labels_dir)) == 0
    assert StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET).empty


def test_enriched_variant_batch_mints_normally(tmp_path, labels_dir):
    """An enriched (full-context) variant batch mints exactly like production."""
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [{"group_id": "g_auto", "routing": "auto_accept", "edges": [("r1", "t1")]}],
    )
    _set_experiment_variant(b, "enriched")
    report = _plan([b], labels_dir)
    g = _by_gid(report)["g_auto"]
    assert g.exported
    assert g.reason == REASON_EXPORTED
    assert write_exports(report, DATASET, Path(labels_dir)) == 1


def test_no_variant_field_mints_normally(tmp_path, labels_dir):
    """An ordinary production batch (no experiment block) mints — no regression."""
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [{"group_id": "g_auto", "routing": "auto_accept", "edges": [("r1", "t1")]}],
    )
    # make_batch writes batch.json without an experiment block, i.e. no variant.
    report = _plan([b], labels_dir)
    g = _by_gid(report)["g_auto"]
    assert g.exported
    assert g.reason == REASON_EXPORTED
    assert write_exports(report, DATASET, Path(labels_dir)) == 1


def test_unreadable_batch_json_fails_closed(tmp_path, labels_dir):
    """An auto_accept whose batch.json is present-but-corrupt must NOT mint.

    ``_merge_consensus`` only hard-requires consensus.csv, so a batch with a
    valid consensus row but an unparseable batch.json could otherwise reach the
    minting gates. Provenance we cannot verify is gated (fail closed) with the
    ablation_variant reason, since we cannot confirm it is the enriched variant.
    """
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [{"group_id": "g_auto", "routing": "auto_accept", "edges": [("r1", "t1")]}],
    )
    (b / "batch.json").write_text("{ this is not valid json ]")
    report = _plan([b], labels_dir)
    g = _by_gid(report)["g_auto"]
    assert not g.exported
    assert g.reason == REASON_ABLATION_VARIANT
    assert write_exports(report, DATASET, Path(labels_dir)) == 0
    assert StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET).empty


def test_recomposed_parent_in_ablation_batch_mints_nothing(tmp_path, labels_dir):
    """The recomposition-path (#367 Mode B) ablation gate blocks a parent too.

    A decomposed parent whose ONLY consumed sub-verdict lives in an
    ablation-variant batch dir must be skipped as ablation_variant — the union
    of context-blinded sub-decisions must never mint. The parent has no DIRECT
    consensus row here, so its outcome comes solely from recomposition: this
    exercises the recomposition gate, not the direct-loop gate.
    """
    b = tmp_path / "b1"
    b.mkdir(parents=True)
    # consensus.csv: only the sub-problem is directly voted (auto_accept).
    with (b / "consensus.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONSENSUS_COLUMNS)
        w.writeheader()
        w.writerow(
            {
                "group_id": "sub1",
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": json.dumps([["r1", "t1"]]),
                "routing": "auto_accept",
                "n_votes": 4,
                "n_valid": 4,
                "minority": "",
                "mean_confidence": 0.9,
                "route_reason": "unanimous_non_none_small",
            }
        )
    _write_votes_csv(b, _V5_VOTERS)

    def _seg(gid: str, *, parent: bool) -> dict:
        node = {
            "group_id": gid,
            "match_type": "M:N",
            "ref_ids": ["r1"],
            "target_ids": ["t1"],
            "edges": [_edge("r1", "t1")],
            "ref_geometries": {"r1": _line(42.28, 0.0005)},
            "target_geometries": {"t1": _line(42.29, 0.0005)},
            "ref_classes": {"r1": "residential"},
            "target_classes": {"t1": "residential"},
        }
        if parent:
            node["decomposed_parent"] = True
            node["subproblem_ids"] = ["sub1"]
        else:
            node["parent_group_id"] = "parent"
        return node

    (b / "batch.json").write_text(
        json.dumps(
            {
                "dataset_id": DATASET,
                "experiment": {
                    "wave": "physical_context_test",
                    "variant": "minimal",
                    "physical_metadata_visible": False,
                    "same_side_coincidence_visible": False,
                },
                "groups": [_seg("parent", parent=True), _seg("sub1", parent=False)],
            }
        )
    )
    report = _plan([b], labels_dir)
    g = _by_gid(report)["parent"]
    assert g.from_decomposition
    assert not g.exported
    assert g.reason == REASON_ABLATION_VARIANT
    assert write_exports(report, DATASET, Path(labels_dir)) == 0
    assert StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET).empty


def test_idempotent_write(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {"group_id": "g1", "routing": "auto_accept", "edges": [("r1", "t1")]},
            {"group_id": "g2", "routing": "auto_accept", "edges": [("r2", "t2")]},
        ],
    )
    report = _plan([b], labels_dir)
    n1 = write_exports(report, DATASET, Path(labels_dir))
    assert n1 == 2

    # Re-plan (now the store has the panel rows) and re-write: no duplicates,
    # and the previously-written panel rows are NOT treated as human precedence.
    report2 = _plan([b], labels_dir)
    assert len(report2.exported) == 2
    assert all(g.reason == REASON_EXPORTED for g in report2.exported)
    write_exports(report2, DATASET, Path(labels_dir))

    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert len(df) == 2
    assert df["group_id"].duplicated().sum() == 0


def test_schema_round_trip(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "g1",
                "match_type": "N:1",
                "routing": "auto_accept",
                "edges": [("r1", "t1"), ("r2", "t1")],
                "mean_confidence": 0.812,
            }
        ],
    )
    report = _plan([b], labels_dir)
    write_exports(report, DATASET, Path(labels_dir))

    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert list(df.columns) == [c for c in STITCHING_LABEL_COLUMNS if c in df.columns]
    row = df.iloc[0]
    assert row["group_id"] == "g1"
    assert row["labeler"] == PANEL_LABELER
    assert row["match_type"] == "N:1"
    assert row["num_refs"] == 2
    assert row["num_targets"] == 1
    assert row["session_id"] == "b1"  # source batch name recorded
    edges = json.loads(row["selected_edges"])
    assert {(e["ref_id"], e["target_id"]) for e in edges} == {("r1", "t1"), ("r2", "t1")}


# --- vote provenance archival -------------------------------------------------


def _voter_rows(group_id: str, choice: str = "A") -> list[dict]:
    """Three raw ballots (claude/codex/agy) for one group."""
    return [
        {
            "group_id": group_id,
            "provider": p,
            "model": m,
            "choice": choice,
            "confidence": 0.9,
            "edge_set": "[]",
        }
        for p, m in (("claude", "opus"), ("codex", "gpt"), ("agy", "gemini"))
    ]


_VOTE_LINK_COLUMNS = [
    "evidence_id",
    "evidence_pack_sha256",
    "displayed_candidate_universe_sha256",
    "option_menu_sha256",
    "chosen_option_id",
    "panel_invocation_sha256",
]
_CONSENSUS_LINK_COLUMNS = [*_VOTE_LINK_COLUMNS, "consensus_policy_sha256"]


def _write_linked_panel_rows(batch: Path, group_id: str, *, choice: str = "A") -> dict:
    """Write runner-shaped ballots/consensus linked to the current evidence pack."""
    manifest = load_evidence_manifest(batch / group_id, allow_legacy=False)
    evidence = manifest["evidence"]
    options = {option["letter"]: option for option in evidence["option_menu"]}
    option = options[choice]
    edges = json.dumps([[edge["ref_id"], edge["target_id"]] for edge in option["edges"]])
    invocation = "a" * 64
    shared = {
        "evidence_id": evidence["evidence_id"],
        "evidence_pack_sha256": manifest["evidence_pack_sha256"],
        "displayed_candidate_universe_sha256": evidence["displayed_candidate_universe_sha256"],
        "option_menu_sha256": evidence["option_menu_sha256"],
        "chosen_option_id": option["option_id"],
        "panel_invocation_sha256": invocation,
    }
    vote_rows = [
        {
            "group_id": group_id,
            "provider": provider,
            "model": model,
            "choice": choice,
            "confidence": 0.9,
            "edge_set": edges,
            **shared,
        }
        for provider, model in (("claude", "opus"), ("codex", "gpt"), ("agy", "gemini"))
    ]
    with (batch / "votes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*VOTES_COLUMNS, *_VOTE_LINK_COLUMNS])
        writer.writeheader()
        for row in vote_rows:
            writer.writerow({column: row.get(column, "") for column in writer.fieldnames})

    with (batch / "consensus.csv").open(newline="") as f:
        consensus_rows = list(csv.DictReader(f))
    assert len(consensus_rows) == 1
    consensus_rows[0].update(
        choice=choice,
        edge_set=edges,
        n_votes="3",
        n_valid="3",
        route_reason="unanimous",
        **shared,
        consensus_policy_sha256=consensus_policy_signature(
            max_edges=settings.stitch_export_backstop_max_edges,
            min_voter_confidence=settings.stitch_min_voter_confidence,
            runtime_contract_sha256=sha256_file(
                Path(__file__).parents[2] / "src/crosswalk/agent_labeling/stitch_runner.py"
            ),
        ),
    )
    with (batch / "consensus.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*CONSENSUS_COLUMNS, *_CONSENSUS_LINK_COLUMNS])
        writer.writeheader()
        writer.writerows(consensus_rows)
    return manifest


def _make_linked_evidence_batch(tmp_path: Path, name: str, group_id: str) -> tuple[Path, dict]:
    batch = tmp_path / name
    make_batch(
        batch,
        DATASET,
        [
            {
                "group_id": group_id,
                "match_type": "1:N",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
            }
        ],
    )
    group = {
        "group_id": group_id,
        "match_type": "1:N",
        "ref_ids": ["r1"],
        "target_ids": ["t1", "t2"],
        "edges": [
            {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
            {"ref_id": "r1", "target_id": "t2", "confidence": 0.6},
        ],
        "optimizer_assignment": [{"ref_id": "r1", "target_id": "t1"}],
        "alternatives": [],
        "ref_geometries": {"r1": {"type": "LineString", "coordinates": [[0.0, 0.0], [0.01, 0.0]]}},
        "target_geometries": {
            "t1": {"type": "LineString", "coordinates": [[0.0, 0.0], [0.01, 0.0]]},
            "t2": {
                "type": "LineString",
                "coordinates": [[0.0, 0.001], [0.01, 0.001]],
            },
        },
    }
    generate_group_evidence(group, batch / group_id)
    (batch / "batch.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_id": DATASET,
                "source_artifacts": {"status": "unavailable"},
                "batch_generation_source": {"status": "unavailable"},
                "groups": [group],
            }
        )
    )
    _write_linked_panel_rows(batch, group_id)
    return batch, group


def test_vote_provenance_archived_from_batches(tmp_path):
    """votes.csv + consensus.csv are snapshotted into a tracked labels/votes tree."""
    b1 = tmp_path / "b1"
    b2 = tmp_path / "b2"
    make_batch(
        b1,
        DATASET,
        [
            {
                "group_id": "g1",
                "match_type": "1:1",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
            }
        ],
    )
    make_batch(
        b2,
        DATASET,
        [
            {
                "group_id": "g2",
                "match_type": "1:1",
                "routing": "auto_accept",
                "edges": [("r2", "t2")],
            }
        ],
    )
    _write_votes(b1, _voter_rows("g1"))
    _write_votes(b2, _voter_rows("g2"))

    votes_dir = tmp_path / "votes"
    n_votes, n_consensus = write_vote_provenance([b1, b2], DATASET, votes_dir=votes_dir)

    assert (n_votes, n_consensus) == (6, 2)  # 3 ballots x 2 groups, 1 consensus row x 2
    out = votes_dir / f"dataset={DATASET}"
    votes = list(csv.DictReader((out / "votes.csv").open()))
    cons = list(csv.DictReader((out / "consensus.csv").open()))
    assert len(votes) == 6 and len(cons) == 2
    # source_batch column ties every row back to its originating batch
    assert {v["source_batch"] for v in votes} == {"b1", "b2"}
    assert {c["source_batch"] for c in cons} == {"b1", "b2"}
    assert {v["provider"] for v in votes} == {"claude", "codex", "agy"}


def test_vote_provenance_archives_exact_displayed_menu(tmp_path):
    batch = tmp_path / "evidence-wave"
    make_batch(
        batch,
        DATASET,
        [
            {
                "group_id": "g1",
                "match_type": "1:N",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
            }
        ],
    )
    group = {
        "group_id": "g1",
        "match_type": "1:N",
        "ref_ids": ["r1"],
        "target_ids": ["t1", "t2"],
        "edges": [
            {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
            {"ref_id": "r1", "target_id": "t2", "confidence": 0.6},
        ],
        "optimizer_assignment": [{"ref_id": "r1", "target_id": "t1"}],
        "alternatives": [{"edges": [{"ref_id": "r1", "target_id": "t2"}]}],
        "ref_geometries": {"r1": {"type": "LineString", "coordinates": [[0.0, 0.0], [0.01, 0.0]]}},
        "target_geometries": {
            "t1": {"type": "LineString", "coordinates": [[0.0, 0.0], [0.01, 0.0]]},
            "t2": {"type": "LineString", "coordinates": [[0.0, 0.001], [0.01, 0.001]]},
        },
    }
    generate_group_evidence(
        group,
        batch / "g1",
        source_artifacts={"groups_sidecar": {"available": True, "sha256": "b" * 64}},
    )
    _write_votes(batch, _voter_rows("g1"))

    write_vote_provenance([batch], DATASET, votes_dir=tmp_path / "votes")

    evidence_path = tmp_path / "votes" / f"dataset={DATASET}" / "evidence.csv"
    rows = list(csv.DictReader(evidence_path.open()))
    assert len(rows) == 1
    evidence = json.loads(rows[0]["evidence"])
    assert evidence["selectable_choices"] == ["A", "B", "NONE"]
    assert evidence["displayed_candidate_count"] == 2
    assert {(edge["ref_id"], edge["target_id"]) for edge in evidence["displayed_edges"]} == {
        ("r1", "t1"),
        ("r1", "t2"),
    }
    assert evidence["source_artifacts"]["groups_sidecar"]["sha256"] == "b" * 64


def test_vote_provenance_quad_panel_keeps_four_rows_per_group(tmp_path):
    """A 4-voter (quad-candidate) batch archives FOUR ballots per group.

    The provenance dedupe keys on (source_batch, group_id, provider). Kimi and
    Muse ride the same opencode transport but carry distinct provider names
    ("kimi"/"muse"); if they shared one name their ballots would collapse into a
    single (source_batch, group_id, provider) key — silent vote loss. The
    distinct names keep all four ballots, which is exactly what a calibration
    wave needs archived intact.
    """
    b = tmp_path / "quad"
    make_batch(
        b,
        DATASET,
        [
            {
                "group_id": "g1",
                "match_type": "1:1",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
            }
        ],
    )
    quad_rows = [
        {
            "group_id": "g1",
            "provider": p,
            "model": m,
            "choice": "A",
            "confidence": 0.9,
            "edge_set": "[]",
        }
        for p, m in (
            ("claude", "claude-opus-4-8"),
            ("codex", "gpt-5.6-terra"),
            ("kimi", "openrouter/moonshotai/kimi-k2.6"),
            ("muse", "meta/muse-spark-1.1"),
        )
    ]
    _write_votes(b, quad_rows)

    votes_dir = tmp_path / "votes"
    n_votes, n_consensus = write_vote_provenance([b], DATASET, votes_dir=votes_dir)

    assert n_votes == 4  # all four ballots survive the (…, provider) dedupe
    out = votes_dir / f"dataset={DATASET}"
    votes = list(csv.DictReader((out / "votes.csv").open()))
    g1_rows = [v for v in votes if v["group_id"] == "g1"]
    assert len(g1_rows) == 4
    assert {v["provider"] for v in g1_rows} == {"claude", "codex", "kimi", "muse"}
    # Kimi and Muse carry their own model strings on their own rows.
    by_prov = {v["provider"]: v["model"] for v in g1_rows}
    assert by_prov["kimi"] == "openrouter/moonshotai/kimi-k2.6"
    assert by_prov["muse"] == "meta/muse-spark-1.1"


def test_vote_provenance_idempotent(tmp_path):
    """Re-archiving the same batches rewrites identical, deduplicated files."""
    b1 = tmp_path / "b1"
    make_batch(
        b1,
        DATASET,
        [
            {
                "group_id": "g1",
                "match_type": "1:1",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
            }
        ],
    )
    _write_votes(b1, _voter_rows("g1"))
    votes_dir = tmp_path / "votes"

    first = write_vote_provenance([b1], DATASET, votes_dir=votes_dir)
    second = write_vote_provenance([b1], DATASET, votes_dir=votes_dir)
    assert first == second == (3, 1)
    rows = list(csv.DictReader((votes_dir / f"dataset={DATASET}" / "votes.csv").open()))
    assert len(rows) == 3  # no duplication on re-run


def test_vote_provenance_tolerates_missing_votes_csv(tmp_path):
    """A batch with consensus but no votes.csv archives consensus and skips ballots."""
    b1 = tmp_path / "b1"
    make_batch(
        b1,
        DATASET,
        [
            {
                "group_id": "g1",
                "match_type": "1:1",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
            }
        ],
        voters=None,  # deliberately no votes.csv
    )
    votes_dir = tmp_path / "votes"
    n_votes, n_consensus = write_vote_provenance([b1], DATASET, votes_dir=votes_dir)
    assert n_votes == 0 and n_consensus == 1
    assert not (votes_dir / f"dataset={DATASET}" / "votes.csv").exists()
    assert (votes_dir / f"dataset={DATASET}" / "consensus.csv").exists()


def test_vote_provenance_strict_mode_refuses_missing_evidence_pack(tmp_path):
    batch = tmp_path / "no-pack"
    make_batch(
        batch,
        DATASET,
        [
            {
                "group_id": "g1",
                "match_type": "1:1",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
            }
        ],
    )
    _write_votes(batch, _voter_rows("g1"))
    shutil.rmtree(batch / "g1")

    with pytest.raises(ValueError, match="missing evidence pack"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )
    assert not (tmp_path / "votes" / f"dataset={DATASET}" / "votes.csv").exists()


def test_vote_provenance_strict_mode_links_ballots_consensus_and_menu(tmp_path):
    batch, _group = _make_linked_evidence_batch(tmp_path, "linked", "g1")
    votes_dir = tmp_path / "votes"

    n_votes, n_consensus = write_vote_provenance(
        [batch],
        DATASET,
        votes_dir=votes_dir,
        require_evidence=True,
    )

    assert (n_votes, n_consensus) == (3, 1)
    archive = votes_dir / f"dataset={DATASET}"
    assert {path.name for path in archive.glob("*.csv")} == {
        "votes.csv",
        "consensus.csv",
        "evidence.csv",
    }


def _stamp_stale_policy(batch: Path, group_id: str, sha: str = "deadbeef" * 8) -> str:
    """Overwrite the batch consensus row's policy sha with a stale signature."""
    path = batch / "consensus.csv"
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        if str(row.get("group_id")) == group_id:
            row["consensus_policy_sha256"] = sha
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return sha


def _current_policy_signature() -> str:
    return consensus_policy_signature(
        max_edges=settings.stitch_export_backstop_max_edges,
        min_voter_confidence=settings.stitch_min_voter_confidence,
        runtime_contract_sha256=sha256_file(
            Path(__file__).parents[2] / "src/crosswalk/agent_labeling/stitch_runner.py"
        ),
    )


def test_vote_provenance_strict_mode_refuses_stale_consensus_policy(tmp_path):
    """A stale consensus_policy_sha256 is a hard refusal by default."""
    batch, _group = _make_linked_evidence_batch(tmp_path, "stale-policy", "g1")
    _stamp_stale_policy(batch, "g1")

    with pytest.raises(ValueError, match="consensus policy linkage is stale"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )
    # Fail-closed: nothing archived when the preflight raised.
    assert not (tmp_path / "votes" / f"dataset={DATASET}" / "consensus.csv").exists()


def test_vote_provenance_allow_stale_policy_mints_with_warning(tmp_path):
    """--allow-stale-policy downgrades the stale-policy refusal to a warning."""
    batch, _group = _make_linked_evidence_batch(tmp_path, "stale-policy-override", "g1")
    stale = _stamp_stale_policy(batch, "g1")

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        n_votes, n_consensus = write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
            allow_stale_policy=True,
        )
    finally:
        logger.remove(sink_id)

    assert (n_votes, n_consensus) == (3, 1)
    warning = "\n".join(messages)
    # The warning must be auditable: name the group and both signatures.
    assert "consensus policy linkage is stale" in warning
    assert "stale-policy-override/g1" in warning
    assert stale in warning
    assert _current_policy_signature() in warning


def test_vote_provenance_allow_stale_policy_records_stored_policy_sha(tmp_path):
    """The mint records the batch's OWN stored policy sha, not the current one."""
    batch, _group = _make_linked_evidence_batch(tmp_path, "stale-policy-recorded", "g1")
    stale = _stamp_stale_policy(batch, "g1")
    votes_dir = tmp_path / "votes"

    write_vote_provenance(
        [batch],
        DATASET,
        votes_dir=votes_dir,
        require_evidence=True,
        allow_stale_policy=True,
    )

    cons = list(csv.DictReader((votes_dir / f"dataset={DATASET}" / "consensus.csv").open()))
    assert len(cons) == 1
    # Recorded under the stored historical policy — never re-stamped to current.
    assert cons[0]["consensus_policy_sha256"] == stale
    assert cons[0]["consensus_policy_sha256"] != _current_policy_signature()


def test_vote_provenance_strict_mode_passes_mixed_attempt_group(tmp_path):
    """A seat-filled (mixed-attempt) group still passes the strict export gates.

    Seat-level retries leave a group whose ballots were drawn in different panel
    rounds (``attempt`` differs per seat) but which share one
    panel_invocation_sha256 and one provider each. The fail-closed export gates
    key on the (provider, model) set and the provenance hashes — never on the
    attempt round — so a mixed-attempt group exports exactly like a
    single-draw one.
    """
    batch, _group = _make_linked_evidence_batch(tmp_path, "mixed-attempt", "g1")
    votes_path = batch / "votes.csv"
    votes = pd.read_csv(votes_path)
    # Mark agy as filled in a later retry round; claude/codex are first-draw.
    votes["attempt"] = [1, 1, 2]
    votes.to_csv(votes_path, index=False)

    n_votes, n_consensus = write_vote_provenance(
        [batch],
        DATASET,
        votes_dir=tmp_path / "votes",
        require_evidence=True,
    )
    assert (n_votes, n_consensus) == (3, 1)


def test_vote_provenance_strict_mode_requires_known_gemini_invocation_route(tmp_path):
    from crosswalk.agent_labeling.stitch_runner import GEMINI_ROUTE_OPENROUTER_FLEX

    batch, _group = _make_linked_evidence_batch(tmp_path, "gemini-route", "g1")
    votes_path = batch / "votes.csv"
    votes = pd.read_csv(votes_path)
    votes["invocation_route"] = votes["invocation_route"].fillna("").astype(object)
    agy = votes["provider"] == "agy"
    votes.loc[agy, "provider"] = "gemini"
    votes.loc[agy, "model"] = "google/gemini-3.5-flash"
    votes.loc[agy, "invocation_route"] = ""
    votes.to_csv(votes_path, index=False)

    with pytest.raises(ValueError, match="invocation_route"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes-bad",
            require_evidence=True,
        )

    votes.loc[agy, "invocation_route"] = GEMINI_ROUTE_OPENROUTER_FLEX
    votes.to_csv(votes_path, index=False)
    assert write_vote_provenance(
        [batch],
        DATASET,
        votes_dir=tmp_path / "votes-good",
        require_evidence=True,
    )[:2] == (3, 1)


def test_vote_provenance_strict_mode_refuses_regenerated_menu_after_ballot(tmp_path):
    batch, group = _make_linked_evidence_batch(tmp_path, "regenerated", "g1")
    group["alternatives"] = [{"edges": [{"ref_id": "r1", "target_id": "t2"}]}]
    generate_group_evidence(group, batch / "g1")
    batch_payload = json.loads((batch / "batch.json").read_text())
    batch_payload["groups"] = [group]
    (batch / "batch.json").write_text(json.dumps(batch_payload))

    with pytest.raises(ValueError, match="ballot evidence_id does not match"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )
    assert not (tmp_path / "votes" / f"dataset={DATASET}" / "votes.csv").exists()


def test_vote_provenance_strict_mode_never_downgrades_v2_pack_to_legacy(tmp_path):
    batch, _group = _make_linked_evidence_batch(tmp_path, "v2-missing-manifest", "g1")
    batch_payload = json.loads((batch / "batch.json").read_text())
    batch_payload["schema_version"] = 2
    (batch / "batch.json").write_text(json.dumps(batch_payload))
    (batch / "g1" / "evidence.json").unlink()
    # Even stripping the embedded evidence marker from metadata cannot make a
    # schema-v2 batch masquerade as a pre-provenance legacy pack.
    metadata_path = batch / "g1" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())
    metadata.pop("evidence", None)
    metadata_path.write_text(yaml.safe_dump(metadata))

    with pytest.raises(ValueError, match="missing .*evidence.json"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )


def test_vote_provenance_strict_mode_rejects_downgraded_batch_contract(tmp_path):
    batch, group = _make_linked_evidence_batch(tmp_path, "downgraded-batch", "g1")
    changed_group = dict(group)
    changed_group["match_type"] = "M:N"
    payload = json.loads((batch / "batch.json").read_text())
    payload["schema_version"] = 1
    payload["groups"] = [changed_group]
    (batch / "batch.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="current evidence manifest requires a schema-v2"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )


def test_vote_provenance_strict_mode_requires_batch_for_current_manifest(tmp_path):
    batch, _group = _make_linked_evidence_batch(tmp_path, "missing-batch", "g1")
    (batch / "batch.json").unlink()

    with pytest.raises(ValueError, match="current evidence manifest requires a schema-v2"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )


def test_vote_provenance_strict_mode_binds_pack_to_v2_batch_group(tmp_path):
    batch, group = _make_linked_evidence_batch(tmp_path, "v2-changed-group", "g1")
    changed_group = dict(group)
    changed_group["match_type"] = "M:N"
    (batch / "batch.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_id": DATASET,
                "source_artifacts": {"status": "unavailable"},
                "batch_generation_source": {"status": "unavailable"},
                "groups": [changed_group],
            }
        )
    )

    with pytest.raises(ValueError, match="source group does not match"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )


def test_vote_provenance_strict_mode_rejects_v2_dataset_mismatch(tmp_path):
    batch, group = _make_linked_evidence_batch(tmp_path, "v2-wrong-dataset", "g1")
    (batch / "batch.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_id": "different_dataset",
                "source_artifacts": {"status": "unavailable"},
                "batch_generation_source": {"status": "unavailable"},
                "groups": [group],
            }
        )
    )

    with pytest.raises(ValueError, match="batch dataset mismatch"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )


def test_vote_provenance_strict_mode_refuses_edge_set_not_in_chosen_option(tmp_path):
    batch, _group = _make_linked_evidence_batch(tmp_path, "wrong-edge", "g1")
    with (batch / "votes.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["edge_set"] = json.dumps([["r1", "t2"]])
    with (batch / "votes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*VOTES_COLUMNS, *_VOTE_LINK_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="edge_set does not match choice"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )


def test_vote_provenance_strict_mode_replays_consensus_from_ballots(tmp_path):
    batch, _group = _make_linked_evidence_batch(tmp_path, "edited-consensus", "g1")
    consensus = pd.read_csv(batch / "consensus.csv")
    consensus.loc[0, "routing"] = "human_review"
    consensus.to_csv(batch / "consensus.csv", index=False)

    with pytest.raises(ValueError, match="not derivable from ballots"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )


def test_vote_provenance_strict_mode_rejects_out_of_range_confidence(tmp_path):
    batch, _group = _make_linked_evidence_batch(tmp_path, "bad-confidence", "g1")
    votes = pd.read_csv(batch / "votes.csv")
    votes["confidence"] = 1.1
    votes.to_csv(batch / "votes.csv", index=False)

    with pytest.raises(ValueError, match="ballot confidence is invalid"):
        write_vote_provenance(
            [batch],
            DATASET,
            votes_dir=tmp_path / "votes",
            require_evidence=True,
        )


def test_vote_provenance_nonstrict_refresh_drops_stale_evidence_row(tmp_path):
    batch, _group = _make_linked_evidence_batch(tmp_path, "nonstrict-refresh", "g1")
    votes_dir = tmp_path / "votes"
    write_vote_provenance(
        [batch],
        DATASET,
        votes_dir=votes_dir,
        require_evidence=True,
    )
    shutil.rmtree(batch / "g1")

    write_vote_provenance([batch], DATASET, votes_dir=votes_dir, require_evidence=False)

    evidence_path = votes_dir / f"dataset={DATASET}" / "evidence.csv"
    assert pd.read_csv(evidence_path).empty


def test_vote_provenance_late_prepare_failure_preserves_all_archives(tmp_path, monkeypatch):
    first, _group = _make_linked_evidence_batch(tmp_path, "first", "g1")
    votes_dir = tmp_path / "votes"
    write_vote_provenance(
        [first],
        DATASET,
        votes_dir=votes_dir,
        require_evidence=True,
    )
    archive = votes_dir / f"dataset={DATASET}"
    before = {path.name: path.read_bytes() for path in archive.glob("*.csv")}
    second, _group = _make_linked_evidence_batch(tmp_path, "second", "g2")

    # votes use a three-column dedupe key; consensus and evidence both use the
    # two-column key. Fail the second two-column call, which is the evidence
    # frame after votes and consensus have already been fully prepared.
    original_drop_duplicates = pd.DataFrame.drop_duplicates
    two_key_calls = 0

    def fail_during_evidence_prepare(frame, *args, **kwargs):
        nonlocal two_key_calls
        subset = kwargs.get("subset", args[0] if args else None)
        if subset == ["source_batch", "group_id"]:
            two_key_calls += 1
            if two_key_calls == 2:
                raise RuntimeError("synthetic late evidence preparation failure")
        return original_drop_duplicates(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "drop_duplicates", fail_during_evidence_prepare)

    with pytest.raises(RuntimeError, match="late evidence preparation"):
        write_vote_provenance(
            [second],
            DATASET,
            votes_dir=votes_dir,
            require_evidence=True,
        )

    after = {path.name: path.read_bytes() for path in archive.glob("*.csv")}
    assert after == before


def test_vote_provenance_commit_failure_rolls_back_all_archives(tmp_path, monkeypatch):
    first, _group = _make_linked_evidence_batch(tmp_path, "first", "g1")
    votes_dir = tmp_path / "votes"
    write_vote_provenance(
        [first],
        DATASET,
        votes_dir=votes_dir,
        require_evidence=True,
    )
    archive = votes_dir / f"dataset={DATASET}"
    before = {path.name: path.read_bytes() for path in archive.glob("*.csv")}
    second, _group = _make_linked_evidence_batch(tmp_path, "second", "g2")

    original_replace = Path.replace

    def fail_consensus_commit(path, target):
        target = Path(target)
        if path.name.endswith(".tmp") and target.name == "consensus.csv":
            raise OSError("synthetic consensus commit failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_consensus_commit)
    with pytest.raises(OSError, match="consensus commit failure"):
        write_vote_provenance(
            [second],
            DATASET,
            votes_dir=votes_dir,
            require_evidence=True,
        )

    after = {path.name: path.read_bytes() for path in archive.glob("*.csv")}
    assert after == before


def _make_group_batch(bd, group_id, edges):
    """A one-group batch (consensus + batch.json) plus its 3 raw ballots."""
    make_batch(
        bd,
        DATASET,
        [{"group_id": group_id, "match_type": "1:1", "routing": "auto_accept", "edges": edges}],
    )
    _write_votes(bd, _voter_rows(group_id))


def test_vote_provenance_accumulates_across_invocations(tmp_path):
    """Separate exports of disjoint batches must not drop earlier ballots.

    write_exports upserts labels per run (older labels persist), so provenance
    must accumulate the same way — otherwise the audit trail stops covering
    every exported label.
    """
    b1, b2 = tmp_path / "b1", tmp_path / "b2"
    _make_group_batch(b1, "g1", [("r1", "t1")])
    _make_group_batch(b2, "g2", [("r2", "t2")])
    votes_dir = tmp_path / "votes"

    write_vote_provenance([b1], DATASET, votes_dir=votes_dir)
    n_votes, n_consensus = write_vote_provenance([b2], DATASET, votes_dir=votes_dir)

    out = votes_dir / f"dataset={DATASET}"
    votes = list(csv.DictReader((out / "votes.csv").open()))
    cons = list(csv.DictReader((out / "consensus.csv").open()))
    # BOTH batches survive the second, disjoint invocation.
    assert {v["source_batch"] for v in votes} == {"b1", "b2"}
    assert {c["source_batch"] for c in cons} == {"b1", "b2"}
    assert len(votes) == 6 and len(cons) == 2
    assert (n_votes, n_consensus) == (6, 2)  # returns the accumulated totals


def test_vote_provenance_rejects_duplicate_basenames(tmp_path):
    """Two batch dirs sharing a basename would collapse under one source_batch."""
    p2 = tmp_path / "phase2" / "us_boston"
    p3 = tmp_path / "phase3" / "us_boston"
    _make_group_batch(p2, "g1", [("r1", "t1")])
    _make_group_batch(p3, "g2", [("r2", "t2")])
    with pytest.raises(ValueError, match="duplicate basenames"):
        write_vote_provenance([p2, p3], DATASET, votes_dir=tmp_path / "votes")


def test_vote_provenance_best_effort_on_malformed_votes(tmp_path):
    """An empty votes.csv is skipped, not fatal; consensus still archives."""
    b1 = tmp_path / "b1"
    make_batch(
        b1,
        DATASET,
        [
            {
                "group_id": "g1",
                "match_type": "1:1",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
            }
        ],
    )
    (b1 / "votes.csv").write_text("")  # 0-byte -> EmptyDataError, must be tolerated
    votes_dir = tmp_path / "votes"

    n_votes, n_consensus = write_vote_provenance([b1], DATASET, votes_dir=votes_dir)
    assert n_votes == 0 and n_consensus == 1
    assert (votes_dir / f"dataset={DATASET}" / "consensus.csv").exists()


def test_vote_provenance_rearchive_replaces_batch_wholesale(tmp_path):
    """Re-archiving a batch REPLACES all of its previous rows: a batch re-run
    with a different panel composition must not leave stale ballots from the
    old composition lingering under the same source_batch (a 4-voter chimera:
    archived agy rows surviving a v4-panel re-run). Other batches' rows are
    untouched, and a listed batch whose votes.csv is unreadable keeps its
    archive (never delete ballots without a replacement)."""
    votes_dir = tmp_path / "votes"
    grp = {"group_id": "g1", "match_type": "1:1", "routing": "auto_accept", "edges": [("r1", "t1")]}
    wave = tmp_path / "wave1"
    make_batch(wave, DATASET, [grp])
    other = tmp_path / "wave_other"
    make_batch(other, DATASET, [dict(grp, group_id="g9")])

    # First archive: wave1 voted by the old claude/codex/agy composition.
    _write_votes(wave, _voter_rows("g1"))  # providers claude/codex/agy
    _write_votes(other, _voter_rows("g9"))
    write_vote_provenance([wave, other], DATASET, votes_dir=votes_dir)

    # Re-run wave1 with the v4 panel (agy replaced by kimi) and re-archive.
    v4_rows = [
        dict(r, provider=p, model=m)
        for r, (p, m) in zip(_voter_rows("g1"), _V4_VOTERS, strict=True)
    ]
    _write_votes(wave, v4_rows)
    n_votes, _ = write_vote_provenance([wave], DATASET, votes_dir=votes_dir)

    out = votes_dir / f"dataset={DATASET}"
    votes = list(csv.DictReader((out / "votes.csv").open()))
    wave1_rows = [v for v in votes if v["source_batch"] == "wave1"]
    # Exactly the new composition — no lingering agy chimera row.
    assert {(v["provider"], v["model"]) for v in wave1_rows} == set(_V4_VOTERS)
    # The other batch's archive is untouched.
    assert {v["provider"] for v in votes if v["source_batch"] == "wave_other"} == {
        "claude",
        "codex",
        "agy",
    }
    assert n_votes == 6  # 3 refreshed wave1 rows + 3 untouched wave_other rows

    # A listed batch whose votes.csv is unreadable keeps its archived rows.
    (wave / "votes.csv").write_text("")  # EmptyDataError -> no replacement rows
    n_votes, _ = write_vote_provenance([wave], DATASET, votes_dir=votes_dir)
    assert n_votes == 6  # nothing deleted, nothing added
    votes = list(csv.DictReader((out / "votes.csv").open()))
    assert {(v["provider"], v["model"]) for v in votes if v["source_batch"] == "wave1"} == set(
        _V4_VOTERS
    )


def test_vote_provenance_header_only_votes_preserves_archive(tmp_path):
    """A HEADER-ONLY (readable, zero-row) votes.csv is as empty as a 0-byte
    one: it must not mark the batch as contributing in the wholesale
    replacement, or it would delete the batch's archived ballots with no
    replacement rows (#398 re-review, finding 2)."""
    votes_dir = tmp_path / "votes"
    grp = {"group_id": "g1", "match_type": "1:1", "routing": "auto_accept", "edges": [("r1", "t1")]}
    wave = tmp_path / "wave1"
    make_batch(wave, DATASET, [grp])
    _write_votes(wave, _voter_rows("g1"))
    write_vote_provenance([wave], DATASET, votes_dir=votes_dir)

    # Re-archive with a header-only votes.csv: archived ballots must survive.
    _write_votes(wave, [])  # header row only, zero data rows
    n_votes, _ = write_vote_provenance([wave], DATASET, votes_dir=votes_dir)
    assert n_votes == 3  # nothing deleted, nothing added
    out = votes_dir / f"dataset={DATASET}"
    votes = list(csv.DictReader((out / "votes.csv").open()))
    assert {v["provider"] for v in votes if v["source_batch"] == "wave1"} == {
        "claude",
        "codex",
        "agy",
    }


# ---------------------------------------------------------------------------
# Nonstandard-panel export guard (provenance: PANEL_LABELER is composition-bound,
# keyed on (provider, model) VOTER pairs since the v4 bless)
# ---------------------------------------------------------------------------


def test_standard_v3_v4_v5_batches_pass(tmp_path):
    """All blessed eras validate: older history is never retroactively flagged."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    v3 = tmp_path / "batch_v3"
    _write_votes_csv(v3, _V3_VOTERS)
    v4 = tmp_path / "batch_v4"
    _write_votes_csv(v4, _V4_VOTERS)
    v5 = tmp_path / "batch_v5"
    _write_votes_csv(v5, _V5_VOTERS)
    assert nonstandard_panel_batches([v3, v4, v5]) == {}


def test_pre_rubric_v6_roster_is_era_less(tmp_path):
    """A roster alone cannot retroactively claim the refined prompt contract."""
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    batch = tmp_path / "pre_rubric_v6"
    _write_votes_csv(batch, _V6_VOTERS)
    assert batch_panel_era(batch) is None


def test_current_rubric_v6_candidate_has_own_era_but_remains_nonstandard(tmp_path):
    """Known stamping and production blessing are separate decisions."""
    from crosswalk.agent_labeling.stitch_export import (
        PANEL_VOTERS_V6,
        batch_panel_era,
        nonstandard_panel_batches,
    )

    v6 = tmp_path / "batch_v6_candidate"
    _write_votes_csv(v6, _V6_VOTERS)
    _stamp_matching_rubric(v6)
    assert frozenset(_V6_VOTERS) == PANEL_VOTERS_V6
    assert batch_panel_era(v6) == "v6"
    assert nonstandard_panel_batches([v6]) == {"batch_v6_candidate": set(_V6_VOTERS)}
    assert nonstandard_panel_batches([v6], expected=PANEL_VOTERS_V6) == {}


def test_pre_rubric_v7_roster_is_era_less(tmp_path):
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    batch = tmp_path / "pre_rubric_v7"
    _write_votes_csv(batch, _V7_VOTERS)
    assert batch_panel_era(batch) is None


def test_current_rubric_v7_batch_is_blessed_and_standard(tmp_path):
    """The 2026-07-18 bless: v7 + canonical rubric passes WITHOUT any override."""
    from crosswalk.agent_labeling.stitch_export import (
        PANEL_VOTERS_V7,
        STANDARD_PANEL_VOTERS_CURRENT_RUBRIC,
        batch_panel_era,
        nonstandard_panel_batches,
    )

    v7 = tmp_path / "batch_v7"
    _write_votes_csv(v7, _V7_VOTERS)
    _stamp_matching_rubric(v7)
    assert frozenset(_V7_VOTERS) == PANEL_VOTERS_V7
    assert STANDARD_PANEL_VOTERS_CURRENT_RUBRIC["v7"] == PANEL_VOTERS_V7
    assert batch_panel_era(v7) == "v7"
    assert nonstandard_panel_batches([v7]) == {}
    assert nonstandard_panel_batches([v7], expected=PANEL_VOTERS_V7) == {}


def test_current_rubric_near_miss_composition_still_refuses(tmp_path):
    """Blessing keys on exact (provider, model) pairs: the v5-quad codex model
    (terra) swapped into the v7 roster is NOT the blessed sol trio."""
    from crosswalk.agent_labeling.stitch_export import (
        batch_panel_era,
        nonstandard_panel_batches,
    )

    near = tmp_path / "batch_v7_terra_near_miss"
    near_voters = [
        ("claude", "claude-opus-4-8"),
        ("codex", "gpt-5.6-terra"),
        ("muse", "meta/muse-spark-1.1"),
    ]
    _write_votes_csv(near, near_voters)
    _stamp_matching_rubric(near)
    # terra+muse trio IS the v6 candidate roster: stampable, still nonstandard.
    assert batch_panel_era(near) == "v6"
    assert nonstandard_panel_batches([near]) == {"batch_v7_terra_near_miss": set(near_voters)}


def test_current_rubric_v5_quad_remains_flagged_after_v7_bless(tmp_path):
    """The v7 bless must not loosen the gate for other current-rubric rosters."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    v5_current = tmp_path / "batch_v5_current_rubric"
    _write_votes_csv(v5_current, _V5_VOTERS)
    _stamp_matching_rubric(v5_current)
    assert nonstandard_panel_batches([v5_current]) == {"batch_v5_current_rubric": set(_V5_VOTERS)}


def test_current_rubric_with_v5_roster_does_not_reuse_historical_era(tmp_path):
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    batch = tmp_path / "refined_v5_roster"
    _write_votes_csv(batch, _V5_VOTERS)
    _stamp_matching_rubric(batch)
    assert batch_panel_era(batch) is None


@pytest.mark.parametrize("manifest", [[], "not-an-object", {"evidence": []}])
def test_malformed_rubric_manifest_fails_era_resolution_closed(tmp_path, manifest):
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    batch = tmp_path / "malformed_manifest"
    _write_votes_csv(batch, _V6_VOTERS)
    _stamp_matching_rubric(batch)
    group_dir = batch / "g1"
    (group_dir / "evidence.json").write_text(json.dumps(manifest))
    assert batch_panel_era(batch) is None


def test_tampered_rubric_stamp_fails_era_resolution_closed(tmp_path, labels_dir):
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    batch = tmp_path / "tampered_manifest"
    _write_votes_csv(batch, _V6_VOTERS)
    _stamp_matching_rubric(batch)
    _tamper_rubric_stamp(batch)
    assert batch_panel_era(batch) is None
    with pytest.raises(ValueError, match="invalid matching-rubric provenance"):
        plan_exports([batch], DATASET, labels_dir, stamp_era="v6")


def test_regenerated_pack_does_not_reattribute_stale_ballots(tmp_path, labels_dir):
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    batch = tmp_path / "regenerated_pack"
    _write_votes_csv(batch, _V6_VOTERS)
    _stamp_matching_rubric(batch)
    assert batch_panel_era(batch) == "v6"
    group_dir = batch / "g1"
    old_manifest = load_evidence_manifest(group_dir, allow_legacy=False)
    (group_dir / "prompt.txt").write_text("regenerated after voting")
    new_manifest = write_evidence_manifest(group_dir, old_manifest["evidence"])
    assert new_manifest["evidence_pack_sha256"] != old_manifest["evidence_pack_sha256"]
    assert batch_panel_era(batch) is None
    with pytest.raises(ValueError, match="invalid matching-rubric provenance"):
        plan_exports([batch], DATASET, labels_dir, stamp_era="v6")


def test_unsafe_vote_group_id_cannot_escape_batch_for_rubric_stamp(tmp_path, labels_dir):
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    batch = tmp_path / "unsafe_group"
    batch.mkdir()
    with (batch / "votes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group_id",
                "provider",
                "model",
                "choice",
                "evidence_id",
                "evidence_pack_sha256",
            ]
        )
        for provider, model in _V6_VOTERS:
            writer.writerow(["../outside", provider, model, "A", "e" * 64, "p" * 64])
    pd.DataFrame(
        [
            {
                "group_id": "../outside",
                "evidence_id": "e" * 64,
                "evidence_pack_sha256": "p" * 64,
            }
        ]
    ).to_csv(batch / "consensus.csv", index=False)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evidence.json").write_text(
        json.dumps({"evidence": {"matching_rubric_version": MATCHING_RUBRIC_VERSION}})
    )
    assert batch_panel_era(batch) is None
    with pytest.raises(ValueError, match="invalid matching-rubric provenance"):
        plan_exports([batch], DATASET, labels_dir, stamp_era="v6")


def test_current_rubric_rejects_consensus_only_group(tmp_path, labels_dir):
    """A linked group cannot lend its verified v6 identity to an unvoted row."""
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    batch = tmp_path / "consensus_only_group"
    _write_votes_csv(batch, _V6_VOTERS)
    _stamp_matching_rubric(batch)
    consensus = pd.read_csv(batch / "consensus.csv", dtype={"group_id": str})
    consensus = pd.concat(
        [consensus, pd.DataFrame([{"group_id": "g2", "routing": "auto_accept"}])],
        ignore_index=True,
    )
    consensus.to_csv(batch / "consensus.csv", index=False)
    assert batch_panel_era(batch) is None
    with pytest.raises(ValueError, match="invalid matching-rubric provenance"):
        plan_exports([batch], DATASET, labels_dir, stamp_era="v6")


def test_current_rubric_rejects_null_vote_group_id(tmp_path, labels_dir):
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    batch = tmp_path / "null_vote_group"
    _write_votes_csv(batch, _V6_VOTERS)
    _stamp_matching_rubric(batch)
    votes = pd.read_csv(batch / "votes.csv", dtype={"group_id": str})
    votes.loc[0, "group_id"] = None
    votes.to_csv(batch / "votes.csv", index=False)
    assert batch_panel_era(batch) is None
    with pytest.raises(ValueError, match="invalid matching-rubric provenance"):
        plan_exports([batch], DATASET, labels_dir, stamp_era="v6")


def test_current_rubric_rejects_missing_vote_group_column(tmp_path, labels_dir):
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    batch = tmp_path / "missing_vote_group_column"
    _write_votes_csv(batch, _V6_VOTERS)
    _stamp_matching_rubric(batch)
    votes = pd.read_csv(batch / "votes.csv").drop(columns=["group_id"])
    votes.to_csv(batch / "votes.csv", index=False)
    assert batch_panel_era(batch) is None
    with pytest.raises(ValueError, match="invalid matching-rubric provenance"):
        plan_exports([batch], DATASET, labels_dir, stamp_era="v6")


def test_explicitly_approved_v6_candidate_export_uses_v6_labeler(tmp_path, labels_dir):
    from crosswalk.agent_labeling.stitch_export import PANEL_LABELER_V6, batch_panel_era

    batch = make_batch(
        tmp_path / "v6_candidate",
        DATASET,
        [
            {
                "group_id": "g_v6",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "route_reason": "unanimous",
            }
        ],
        voters=_V6_VOTERS,
    )
    _stamp_matching_rubric(batch, group_id="g_v6")
    assert batch_panel_era(batch) == "v6"
    report = plan_exports([batch], DATASET, labels_dir)
    assert [(g.group_id, g.panel_era) for g in report.exported] == [("g_v6", "v6")]
    assert write_exports(report, DATASET, labels_dir) == 1
    stored = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert list(stored["labeler"]) == [PANEL_LABELER_V6]


def test_blessed_v7_export_uses_v7_labeler(tmp_path, labels_dir):
    from crosswalk.agent_labeling.stitch_export import PANEL_LABELER_V7, batch_panel_era

    batch = make_batch(
        tmp_path / "v7_candidate",
        DATASET,
        [
            {
                "group_id": "g_v7",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "route_reason": "unanimous",
            }
        ],
        voters=_V7_VOTERS,
    )
    _stamp_matching_rubric(batch, group_id="g_v7")
    assert batch_panel_era(batch) == "v7"
    report = plan_exports([batch], DATASET, labels_dir)
    assert [(g.group_id, g.panel_era) for g in report.exported] == [("g_v7", "v7")]
    assert write_exports(report, DATASET, labels_dir) == 1
    stored = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert list(stored["labeler"]) == [PANEL_LABELER_V7]


def test_v7_voter_set_matches_runner_candidate_panel():
    from crosswalk.agent_labeling.stitch_export import PANEL_VOTERS_V7
    from crosswalk.agent_labeling.stitch_runner import PANEL_V7_CANDIDATE

    assert frozenset(_V7_VOTERS) == PANEL_VOTERS_V7
    assert frozenset((p.name, p.model) for p in PANEL_V7_CANDIDATE) == PANEL_VOTERS_V7
    assert [p.effort for p in PANEL_V7_CANDIDATE] == ["high", "high", "high"]


def test_default_panel_voters_match_runner_default_panel():
    """stitch_export's blessed v5 set stays in lockstep with the live DEFAULT_PANEL.

    A composition change in stitch_runner without a provenance decision here
    (bump the labeler + the blessed set) must fail CI, not silently drift.
    """
    from crosswalk.agent_labeling.stitch_export import (
        DEFAULT_PANEL_VOTERS,
        PANEL_VOTERS_V5,
    )
    from crosswalk.agent_labeling.stitch_runner import DEFAULT_PANEL

    assert DEFAULT_PANEL_VOTERS == PANEL_VOTERS_V5
    assert frozenset((p.name, p.model) for p in DEFAULT_PANEL) == DEFAULT_PANEL_VOTERS
    # The v5 quad is written out literally here too, so a silent edit of BOTH
    # constants still fails loudly.
    assert frozenset(_V5_VOTERS) == DEFAULT_PANEL_VOTERS


def test_nonstandard_panel_batches_flags_swapped_voter(tmp_path):
    """A no-agy batch (opencode/Qwen swapped in) is flagged with its voter pairs."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    ok = tmp_path / "batch_default"
    _write_votes_csv(ok, _V4_VOTERS)
    bad = tmp_path / "batch_noagy"
    noagy = [v for v in _V3_VOTERS if v[0] != "agy"] + [
        ("opencode", "openrouter/qwen/qwen3-vl-235b-a22b-instruct")
    ]
    _write_votes_csv(bad, noagy)

    offending = nonstandard_panel_batches([ok, bad])
    assert set(offending) == {"batch_noagy"}
    assert offending["batch_noagy"] == set(noagy)


def test_nonstandard_panel_batches_flags_wrong_model_same_provider(tmp_path):
    """The whole point of (provider, model) keying: an opencode-transport voter
    driving Gemini (the historical transport-swap model) is not the blessed v4
    panel (which seats kimi/Kimi K2.6) -> flagged. The residual "opencode" seat
    name is shared with the v3-era Qwen voter, so name alone cannot bless it;
    the model pins it."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    b = tmp_path / "batch_opencode_gemini"
    voters = [v for v in _V4_VOTERS if v[0] != "kimi"] + [
        ("opencode", "openrouter/google/gemini-3.5-flash")
    ]
    _write_votes_csv(b, voters)
    assert set(nonstandard_panel_batches([b])) == {"batch_opencode_gemini"}


def test_nonstandard_panel_batches_flags_mixed_era_composition(tmp_path):
    """A cross-era mix (v3 codex model + v4 Kimi voter) matches neither blessed
    set exactly -> flagged (this is the historical v4-candidate composition)."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    b = tmp_path / "batch_v4_candidate"
    voters = [v for v in _V3_VOTERS if v[0] != "agy"] + [
        ("kimi", "openrouter/moonshotai/kimi-k2.6")
    ]
    _write_votes_csv(b, voters)
    assert set(nonstandard_panel_batches([b])) == {"batch_v4_candidate"}


def test_nonstandard_panel_batches_flags_blank_model(tmp_path):
    """A blank/NaN model on any vote row reads as (provider, "") -> flagged,
    never a crash and never mistakable for the blessed panel."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    b = tmp_path / "batch_blank_model"
    voters = [v for v in _V4_VOTERS if v[0] != "kimi"] + [("kimi", "")]
    _write_votes_csv(b, voters)
    offending = nonstandard_panel_batches([b])
    assert set(offending) == {"batch_blank_model"}
    assert ("kimi", "") in offending["batch_blank_model"]


def test_nonstandard_panel_batches_flags_missing_model_column(tmp_path):
    """A votes.csv with NO model column (pre-provenance format) is flagged —
    incomplete provenance must not pass as the blessed panel."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    b = tmp_path / "batch_no_model_col"
    _write_votes_csv(b, _V4_VOTERS, include_model_col=False)
    offending = nonstandard_panel_batches([b])
    assert set(offending) == {"batch_no_model_col"}
    assert all(m == "" for _p, m in offending["batch_no_model_col"])


def test_nonstandard_panel_batches_flags_subset_panel(tmp_path):
    """A degraded 2-voter batch is also nonstandard (missing voter)."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    b = tmp_path / "batch_two"
    _write_votes_csv(b, _V4_VOTERS[:2])
    assert set(nonstandard_panel_batches([b])) == {"batch_two"}


def test_nonstandard_panel_batches_explicit_expected_pins_one_composition(tmp_path):
    """Passing expected= pins a single composition: a v3 batch then flags."""
    from crosswalk.agent_labeling.stitch_export import (
        PANEL_VOTERS_V4,
        nonstandard_panel_batches,
    )

    v3 = tmp_path / "batch_v3"
    _write_votes_csv(v3, _V3_VOTERS)
    assert set(nonstandard_panel_batches([v3], expected=PANEL_VOTERS_V4)) == {"batch_v3"}
    assert nonstandard_panel_batches([v3], expected=frozenset(_V3_VOTERS)) == {}


def test_nonstandard_panel_batches_skips_missing_votes(tmp_path):
    """No votes.csv -> best-effort skip, not flagged (consensus gate lives in CLI)."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    b = tmp_path / "batch_novotes"
    b.mkdir()
    assert nonstandard_panel_batches([b]) == {}


def test_batch_panel_era_resolution(tmp_path):
    """batch_panel_era: v3 -> "v3", v4 -> "v4", v5 -> "v5", known-historical ->
    its era, anything else -> None (no silent default)."""
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    v3 = tmp_path / "b_v3"
    _write_votes_csv(v3, _V3_VOTERS)
    v4 = tmp_path / "b_v4"
    _write_votes_csv(v4, _V4_VOTERS)
    v5 = tmp_path / "b_v5"
    _write_votes_csv(v5, _V5_VOTERS)
    hist = tmp_path / "b_hist"
    _write_votes_csv(hist, _HISTORICAL_GEMINI_TRANSPORT_VOTERS)
    odd = tmp_path / "b_odd"
    _write_votes_csv(odd, _V4_VOTERS[:2])
    none = tmp_path / "b_none"
    none.mkdir()

    assert batch_panel_era(v3) == "v3"
    assert batch_panel_era(v4) == "v4"
    assert batch_panel_era(v5) == "v5"
    assert batch_panel_era(hist) == "v3"
    assert batch_panel_era(odd) is None
    assert batch_panel_era(none) is None


def _one_group_batch(root: Path, name: str, gid: str, **make_kw) -> Path:
    """A minimal exportable single-group batch dir."""
    return make_batch(
        root / name,
        DATASET,
        [
            {
                "group_id": gid,
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "n_edges": 1,
                "n_corridors": 1,
                "n_assignment_components": 1,
            }
        ],
        **make_kw,
    )


def test_write_exports_stamps_labeler_by_batch_era(tmp_path, labels_dir):
    """Era-scoped labeler stamping: a v3-era batch mints panel_unanimous_v3, a
    v4 batch panel_unanimous_v4, and a v5 batch panel_unanimous_v5.
    Re-exporting committed older-era history must never silently rewrite its
    provenance to the current era."""
    from crosswalk.agent_labeling.stitch_export import PANEL_LABELER_V3, PANEL_LABELER_V4

    b_v3 = _one_group_batch(tmp_path, "b_v3", "g_v3", voters=_V3_VOTERS)
    b_v4 = _one_group_batch(tmp_path, "b_v4", "g_v4", voters=_V4_VOTERS)
    b_v5 = _one_group_batch(tmp_path, "b_v5", "g_v5", voters=_V5_VOTERS)

    report = plan_exports([b_v3, b_v4, b_v5], DATASET, labels_dir)
    assert {g.group_id: g.panel_era for g in report.groups} == {
        "g_v3": "v3",
        "g_v4": "v4",
        "g_v5": "v5",
    }
    assert write_exports(report, DATASET, labels_dir) == 3

    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    labelers = dict(zip(df["group_id"], df["labeler"], strict=True))
    assert labelers == {
        "g_v3": PANEL_LABELER_V3,
        "g_v4": PANEL_LABELER_V4,
        "g_v5": PANEL_LABELER,
    }


def test_historical_transport_swap_batch_stamps_v3_but_stays_gated(tmp_path, labels_dir):
    """THE reviewer-proven regression (w0707 wave, commit 80dbe1f): the
    claude + codex/gpt-5.5 + opencode/Gemini transport-swap composition minted
    committed v3 labels via --allow-nonstandard-panel. Its era must resolve to
    v3 (a re-export re-stamps panel_unanimous_v3, never v4) while the export
    gate STILL flags it as nonstandard, exactly as at original export time."""
    from crosswalk.agent_labeling.stitch_export import (
        PANEL_LABELER_V3,
        nonstandard_panel_batches,
    )

    b = _one_group_batch(tmp_path, "b_w0707", "g_hist", voters=_HISTORICAL_GEMINI_TRANSPORT_VOTERS)

    # Gate behavior unchanged: still nonstandard (needs --allow-nonstandard-panel).
    assert set(nonstandard_panel_batches([b])) == {"b_w0707"}

    # Stamping: era resolves v3 and a (re-)export mints the v3 tag.
    report = plan_exports([b], DATASET, labels_dir)
    assert [g.panel_era for g in report.groups] == ["v3"]
    assert write_exports(report, DATASET, labels_dir) == 1
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert list(df["labeler"]) == [PANEL_LABELER_V3]


def test_write_exports_refuses_era_less_batches(tmp_path, labels_dir):
    """An exported group from a batch with NO resolvable era (unknown
    composition or no votes.csv) makes write_exports raise — never a silent
    current-era stamp."""
    b_unknown = _one_group_batch(tmp_path, "b_unknown", "g_unknown", voters=None)

    report = plan_exports([b_unknown], DATASET, labels_dir)
    assert [g.panel_era for g in report.groups] == [""]
    with pytest.raises(ValueError, match="no known panel era.*stamp-era"):
        write_exports(report, DATASET, labels_dir)
    # Nothing was written.
    assert StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET).empty


def test_stamp_era_fills_in_era_less_batches_only(tmp_path, labels_dir):
    """plan_exports(stamp_era=...) is a FILL-IN, not a whole-run override: an
    era-less batch exports under the declared generation's tags, while a batch
    that genuinely resolves to an era in the SAME run keeps its own — passing
    --stamp-era v3 for one unknown batch must never re-stamp a blessed-v4
    batch's labels as v3 (#398 re-review, finding 1)."""
    from crosswalk.agent_labeling.stitch_export import PANEL_LABELER_V3

    b_unknown = _one_group_batch(tmp_path, "b_unknown", "g_unknown", voters=None)

    report = plan_exports([b_unknown], DATASET, labels_dir, stamp_era="v3")
    assert [g.panel_era for g in report.groups] == ["v3"]
    assert write_exports(report, DATASET, labels_dir) == 1
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert list(df["labeler"]) == [PANEL_LABELER_V3]

    # Mixed set: the blessed-v5 batch keeps v5; only the era-less one fills in.
    b_v5 = _one_group_batch(tmp_path, "b_v5", "g_v5", voters=_V5_VOTERS)
    report = plan_exports([b_unknown, b_v5], DATASET, labels_dir, stamp_era="v3")
    assert {g.group_id: g.panel_era for g in report.groups} == {
        "g_unknown": "v3",
        "g_v5": "v5",
    }
    assert write_exports(report, DATASET, labels_dir) == 2
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    labelers = dict(zip(df["group_id"], df["labeler"], strict=True))
    assert labelers["g_v5"] == PANEL_LABELER  # never re-stamped by the fill-in
    assert labelers["g_unknown"] == PANEL_LABELER_V3

    with pytest.raises(ValueError, match="stamp_era"):
        plan_exports([b_unknown], DATASET, labels_dir, stamp_era="v9")


def test_plan_exports_refuses_duplicate_batch_basenames(tmp_path, labels_dir):
    """Era stamping keys batches by basename: two dirs with the same basename
    would mis-attribute one dir's era to the other (last-dir-wins). Refused up
    front, before planning."""
    b1 = _one_group_batch(tmp_path / "left", "same_name", "g1")
    b2 = _one_group_batch(tmp_path / "right", "same_name", "g2")
    with pytest.raises(ValueError, match="duplicate basenames"):
        plan_exports([b1, b2], DATASET, labels_dir)


# ---------------------------------------------------------------------------
# v5 quorum provenance: a 4/4 unanimous accept and a 3-of-4 quorum accept mint
# DISTINCT labelers (panel_unanimous_v5 vs panel_quorum_v5), end-to-end.
# ---------------------------------------------------------------------------


def _quorum_group(group_id: str, edges, **kw) -> dict:
    """An auto-accepted QUORUM group (v5 rule): 3 valid agree, 1 abstained."""
    return {
        "group_id": group_id,
        "routing": "auto_accept",
        "consensus": "quorum",
        "edges": list(edges),
        "n_votes": 4,
        "n_valid": 3,
        "route_reason": "quorum",
        **kw,
    }


def test_quorum_accept_mints_quorum_labeler(tmp_path, labels_dir):
    """PROVENANCE DISTINCTION: in one v5 batch, a fully unanimous accept mints
    panel_unanimous_v5 while a quorum accept (one abstention) mints the
    DISTINCT panel_quorum_v5."""
    from crosswalk.agent_labeling.stitch_export import PANEL_QUORUM_LABELER

    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "g_unanimous",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "route_reason": "unanimous",
            },
            _quorum_group("g_quorum", [("r2", "t2")]),
        ],
    )
    report = plan_exports([b], DATASET, labels_dir)
    by_gid = {g.group_id: g for g in report.groups}
    assert by_gid["g_unanimous"].is_quorum is False
    assert by_gid["g_quorum"].is_quorum is True
    assert write_exports(report, DATASET, labels_dir) == 2

    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    labelers = dict(zip(df["group_id"], df["labeler"], strict=True))
    assert labelers == {"g_unanimous": PANEL_LABELER, "g_quorum": PANEL_QUORUM_LABELER}


def test_quorum_accept_detected_from_counts_without_stamp(tmp_path, labels_dir):
    """CONSERVATIVE detection: an auto_accept row with n_valid < n_votes is a
    quorum accept even without the tier/reason stamps — abstention evidence
    always downgrades the claim, never launders into unanimous provenance."""
    from crosswalk.agent_labeling.stitch_export import PANEL_QUORUM_LABELER

    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "g_bare",
                "routing": "auto_accept",
                "consensus": "unanimous",  # stale tier
                "edges": [("r1", "t1")],
                "n_votes": 4,
                "n_valid": 3,
                "route_reason": "",  # no stamp
            }
        ],
    )
    report = plan_exports([b], DATASET, labels_dir)
    assert [g.is_quorum for g in report.groups] == [True]
    assert write_exports(report, DATASET, labels_dir) == 1
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert list(df["labeler"]) == [PANEL_QUORUM_LABELER]


def test_contradicting_unanimous_stamp_still_downgrades_to_quorum(tmp_path, labels_dir):
    """Counts downgrade an accept stamp; panel NONE still stays in review."""
    from crosswalk.agent_labeling.stitch_export import PANEL_QUORUM_LABELER

    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "g_stamped_unan",
                "routing": "auto_accept",
                "consensus": "unanimous",
                "edges": [("r1", "t1")],
                "n_votes": 4,
                "n_valid": 3,
                "route_reason": "unanimous",  # stale stamp contradicted by counts
            },
            {
                "group_id": "g_stamped_none",
                "routing": "human_review",
                "consensus": "unanimous",
                "choice": "NONE",
                "edges": [],
                "candidate_edges": [("r2", "t2")],
                "n_votes": 4,
                "n_valid": 3,
                "route_reason": "unanimous_none",  # stale stamp contradicted by counts
            },
        ],
    )
    report = plan_exports([b], DATASET, labels_dir)
    assert {g.group_id: g.is_quorum for g in report.groups} == {"g_stamped_unan": True}
    assert write_exports(report, DATASET, labels_dir) == 1
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    labelers = dict(zip(df["group_id"], df["labeler"], strict=True))
    assert labelers == {"g_stamped_unan": PANEL_QUORUM_LABELER}


def test_quorum_none_requires_human_confirmation(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "g_qnone",
                "routing": "human_review",
                "consensus": "quorum",
                "choice": "NONE",
                "edges": [],
                "candidate_edges": [("r1", "t1"), ("r2", "t2")],
                "n_votes": 4,
                "n_valid": 3,
                "route_reason": "quorum_none",
            }
        ],
    )
    report = plan_exports([b], DATASET, labels_dir)
    assert report.groups == []
    assert report.n_unanimous_none == 0
    assert write_exports(report, DATASET, labels_dir) == 0


def test_quorum_none_below_quorum_not_exported(tmp_path, labels_dir):
    """quorum_none still requires >=3 valid votes: a 2-valid NONE row (however
    stamped) must not mint reject ground truth."""
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "g_2none",
                "routing": "human_review",
                "consensus": "quorum",
                "choice": "NONE",
                "edges": [],
                "candidate_edges": [("r1", "t1")],
                "n_votes": 4,
                "n_valid": 2,
                "route_reason": "quorum_none",
            }
        ],
    )
    report = plan_exports([b], DATASET, labels_dir)
    assert report.n_unanimous_none == 0
    assert write_exports(report, DATASET, labels_dir) == 0


def test_quorum_verdict_in_pre_quorum_era_refused(tmp_path, labels_dir):
    """A quorum-flagged verdict attributed to a pre-v5 era is a provenance
    anomaly (a 3-voter panel cannot accept over an abstention): write_exports
    refuses rather than blurring it into that era's unanimous tag."""
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [_quorum_group("g_anomaly", [("r1", "t1")])],
        voters=_V4_VOTERS,  # batch resolves to era v4, which has no quorum tags
    )
    report = plan_exports([b], DATASET, labels_dir)
    assert [(g.panel_era, g.is_quorum) for g in report.groups] == [("v4", True)]
    with pytest.raises(ValueError, match="predates the quorum rule"):
        write_exports(report, DATASET, labels_dir)
    assert StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET).empty


def test_quorum_era_anomaly_writes_nothing_even_after_valid_rows(tmp_path, labels_dir):
    """PARTIAL-WRITE GUARD (#405): a valid row FOLLOWED by a quorum-era-violation
    row must write NOTHING — labeler resolution happens in a pre-pass, so the
    anomaly raises before any row hits the store (no half-exported state)."""
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            # Sorts first -> pre-fix it would be written before the anomaly raises.
            {
                "group_id": "g_aaa_valid",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "route_reason": "unanimous",
            },
            # A quorum accept in v4 (no quorum labelers) -> resolution raises.
            _quorum_group("g_zzz_anomaly", [("r2", "t2")]),
        ],
        voters=_V4_VOTERS,
    )
    report = plan_exports([b], DATASET, labels_dir)
    assert {g.group_id for g in report.exported} == {"g_aaa_valid", "g_zzz_anomaly"}
    with pytest.raises(ValueError, match="predates the quorum rule"):
        write_exports(report, DATASET, labels_dir)
    # NOTHING written — not even the valid row that precedes the anomaly.
    assert StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET).empty


# ---------------------------------------------------------------------------
# 2026-07-10 quadcal0710 calibration batches (awareness item): batches voted
# entirely post-#404 carry voter sets EXACTLY equal to PANEL_VOTERS_V5, so they
# resolve to era v5 and pass the gate — safe only because every calibration
# group corresponds to an existing human label (human precedence blocks the
# export). The Boston batch mixes sol+terra codex ballots -> era-less-> refused.
# ---------------------------------------------------------------------------


def test_quadcal_shaped_batch_resolves_v5_but_human_precedence_blocks(tmp_path, labels_dir):
    """A calibration-shaped batch (exact v5 voter set, auto_accept verdict on a
    group a HUMAN already labeled) resolves to era v5 and passes the panel
    gate, but exports nothing: human precedence decides the group."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    b = make_batch(
        tmp_path / "cal_b",
        DATASET,
        [
            {
                "group_id": "g_cal",
                "routing": "auto_accept",
                "edges": [("r1", "t1")],
                "n_edges": 1,
                "n_corridors": 1,
                "n_assignment_components": 1,
            }
        ],
        voters=_V5_VOTERS,
    )
    # The quad composition is the blessed v5 set: standard, era v5.
    assert nonstandard_panel_batches([b]) == {}

    # A prior HUMAN label on the same group (exact group_id) takes precedence.
    store = StitchingLabelStore(DATASET, labels_dir=labels_dir)
    store.add(
        group_id="g_cal",
        selected_edges=[{"ref_id": "r1", "target_id": "t1"}],
        match_type="1:1",
        num_refs=1,
        num_targets=1,
        labeler="brad",
        session_id="human-session",
    )
    report = plan_exports([b], DATASET, labels_dir)
    g = next(g for g in report.groups if g.group_id == "g_cal")
    assert g.exported is False
    assert g.reason == REASON_HUMAN_PRECEDENCE
    assert write_exports(report, DATASET, labels_dir) == 0
    # The human row is untouched.
    df = store.load(DATASET)
    assert list(df["labeler"]) == ["brad"]


def test_mixed_codex_model_batch_is_era_less_and_refused(tmp_path, labels_dir):
    """The Boston quadcal batch shape: BOTH sol and terra codex ballots (5
    distinct voter pairs) matches no blessed set -> flagged nonstandard AND
    era-less, so an export is refused outright (needs --allow-nonstandard-panel
    AND --stamp-era)."""
    from crosswalk.agent_labeling.stitch_export import (
        batch_panel_era,
        nonstandard_panel_batches,
    )

    boston_shape = [*_V5_VOTERS, ("codex", "gpt-5.6-sol")]
    b = _one_group_batch(tmp_path, "b_boston_cal", "g1", voters=boston_shape)
    assert set(nonstandard_panel_batches([b])) == {"b_boston_cal"}
    assert batch_panel_era(b) is None
    report = plan_exports([b], DATASET, labels_dir)
    assert [g.panel_era for g in report.groups] == [""]
    with pytest.raises(ValueError, match="no known panel era"):
        write_exports(report, DATASET, labels_dir)


# ---------------------------------------------------------------------------
# Panel NONE -> human confirmation (never direct reject-all truth)
# ---------------------------------------------------------------------------


def _none_group(group_id: str, candidate_edges, **kw) -> dict:
    """A NONE verdict routed to review with an empty consensus edge set."""
    return {
        "group_id": group_id,
        "routing": "human_review",
        "consensus": "unanimous",
        "choice": "NONE",
        "edges": [],  # empty consensus edge set (reject-all)
        "candidate_edges": list(candidate_edges),
        "route_reason": kw.pop("route_reason", "unanimous_none"),
        **kw,
    }


def test_panel_none_is_not_an_export_candidate(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            _none_group("g_unanimous", [("r1", "t1")]),
            _none_group(
                "g_quorum",
                [("r2", "t2")],
                consensus="quorum",
                n_valid=3,
                route_reason="quorum_none",
            ),
            {"group_id": "g_accept", "routing": "auto_accept", "edges": [("r3", "t3")]},
        ],
    )
    report = _plan([b], labels_dir)
    assert set(_by_gid(report)) == {"g_accept"}
    assert [g.group_id for g in report.exported] == ["g_accept"]
    assert report.n_unanimous_none == 0
    assert report.exported_empty == []


def test_malformed_auto_accept_with_empty_edge_set_routes_to_review(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {
                "group_id": "g_empty_accept",
                "routing": "auto_accept",
                "edges": [],
                "candidate_edges": [("r1", "t1")],
            }
        ],
    )
    group = _by_gid(_plan([b], labels_dir))["g_empty_accept"]
    assert not group.exported
    assert group.reason == REASON_EMPTY_SELECTION


def test_legacy_empty_set_opt_in_fails_closed(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [_none_group("gnone", [("r1", "t1")])],
    )
    with pytest.raises(ValueError, match="confirm the empty set in human review"):
        _plan([b], labels_dir, export_empty_set=True)


def test_write_exports_refuses_constructed_empty_group_atomically(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [{"group_id": "g_accept", "routing": "auto_accept", "edges": [("r1", "t1")]}],
    )
    report = _plan([b], labels_dir)
    report.groups.append(
        GroupExport(
            group_id="g_none",
            source_batch="b1",
            exported=True,
            reason=REASON_EXPORTED,
            panel_era="v5",
        )
    )
    with pytest.raises(ValueError, match="empty panel selections"):
        write_exports(report, DATASET, Path(labels_dir))
    assert StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET).empty


# ---------------------------------------------------------------------------
# Accept-floor guard (#405): an auto_accept row with n_valid present and < 3
# contradicts the quorum floor (compute_consensus only mints accept at
# n_valid >= 3). It must never mint accept ground truth at export.
# ---------------------------------------------------------------------------


def test_sub_quorum_accept_not_exported(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {"group_id": "g_ok", "routing": "auto_accept", "edges": [("r1", "t1")]},
            # auto_accept but only 2 valid votes -> below the n_valid >= 3 floor.
            {
                "group_id": "g_bad",
                "routing": "auto_accept",
                "edges": [("r2", "t2")],
                "n_votes": 2,
                "n_valid": 2,
            },
        ],
    )
    report = _plan([b], labels_dir)
    exported = {g.group_id for g in report.exported}
    # The clean accept exports; the sub-quorum accept is not even a candidate.
    assert "g_ok" in exported
    assert "g_bad" not in {g.group_id for g in report.groups}
    write_exports(report, DATASET, Path(labels_dir))
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert set(df["group_id"]) == {"g_ok"}


# ---------------------------------------------------------------------------
# .no-export marker (#405): a batch dir carrying a .no-export marker is skipped
# by export, but panel_routing / queue building still reads it.
# ---------------------------------------------------------------------------


def test_no_export_marker_skips_batch(tmp_path, labels_dir):
    from crosswalk.agent_labeling.stitch_export import NO_EXPORT_MARKER

    marked = _one_group_batch(tmp_path, "b_cal", "g_cal")
    (marked / NO_EXPORT_MARKER).write_text("")
    normal = _one_group_batch(tmp_path, "b_real", "g_real")

    report = _plan([marked, normal], labels_dir)
    # Only the unmarked batch's group is planned/exported.
    assert {g.group_id for g in report.groups} == {"g_real"}
    write_exports(report, DATASET, Path(labels_dir))
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert set(df["group_id"]) == {"g_real"}


def test_filter_exportable_batch_dirs_drops_marked(tmp_path):
    from crosswalk.agent_labeling.stitch_export import (
        NO_EXPORT_MARKER,
        filter_exportable_batch_dirs,
    )

    marked = tmp_path / "b_cal"
    marked.mkdir()
    (marked / NO_EXPORT_MARKER).write_text("")
    normal = tmp_path / "b_real"
    normal.mkdir()
    assert filter_exportable_batch_dirs([marked, normal]) == [normal]


def test_no_export_marker_still_feeds_review_queue(tmp_path):
    # The queue path (panel_routing) does NOT honor the marker: a calibration
    # batch's contested groups still reach the human review queue.
    from crosswalk.agent_labeling.panel_routing import panel_failed_group_ids
    from crosswalk.agent_labeling.stitch_export import NO_EXPORT_MARKER

    root = tmp_path / "batches"
    b = make_batch(
        root / "ds_cal",
        "ds_cal",
        [{"group_id": "g_contested", "routing": "human_review", "edges": [("r1", "t1")]}],
    )
    (b / NO_EXPORT_MARKER).write_text("")
    # Despite the marker, the failed group surfaces to the queue.
    assert panel_failed_group_ids("ds_cal", root) == {"g_contested"}
