"""Tests for `crosswalk agent stitch-export` (panel-consensus -> stitching labels).

Fixtures are synthetic (the real batch dirs are gitignored) so these tests are
self-contained and exercise every export gate plus idempotency and schema
round-trip through the real `StitchingLabelStore`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from crosswalk.agent_labeling.stitch_export import (
    PANEL_LABELER,
    REASON_CLASS_MISMATCH,
    REASON_EMPTIED_BY_SLIVER,
    REASON_EXPORTED,
    REASON_HUMAN_PRECEDENCE,
    REASON_OVER_MAX,
    REASON_STRUCTURAL_TANGLE,
    plan_exports,
    write_exports,
    write_vote_provenance,
)
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


def make_batch(batch_dir: Path, dataset: str, groups: list[dict]) -> Path:
    """Write a synthetic batch dir (consensus.csv, batch.json, metadata.yaml).

    Each ``groups`` entry:
      group_id, match_type, routing, choice, mean_confidence,
      edges: list of (ref, tgt) chosen edges (the consensus edge_set),
      ref_classes / target_classes: {id: class},
      sliver_edges: set of (ref, tgt) to encode as tiny-span (sliver) edges,
      candidate_edges: optional extra (ref, tgt) present in batch.json but not chosen.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)

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
                    "n_votes": 3,
                    "n_valid": 3,
                    "minority": "",
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


def test_sliver_dropped_but_group_exported(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
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
    assert g.exported
    assert g.n_slivers_dropped == 1
    assert g.n_edges_final == 1
    assert {(e["ref_id"], e["target_id"]) for e in g.selected_edges} == {("r1", "t1")}


def test_all_slivers_empties_group(tmp_path, labels_dir):
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
    assert g.reason == REASON_EMPTIED_BY_SLIVER
    assert g.n_slivers_dropped == 1


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
    )
    # deliberately no _write_votes(b1)
    votes_dir = tmp_path / "votes"
    n_votes, n_consensus = write_vote_provenance([b1], DATASET, votes_dir=votes_dir)
    assert n_votes == 0 and n_consensus == 1
    assert not (votes_dir / f"dataset={DATASET}" / "votes.csv").exists()
    assert (votes_dir / f"dataset={DATASET}" / "consensus.csv").exists()


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


# ---------------------------------------------------------------------------
# Nonstandard-panel export guard (provenance: PANEL_LABELER is composition-bound)
# ---------------------------------------------------------------------------


def _write_votes_csv(batch_dir: Path, providers: list[str]) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    with open(batch_dir / "votes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "provider", "model", "choice"])
        for p in providers:
            w.writerow(["g1", p, "m", "A"])


def test_nonstandard_panel_batches_flags_swapped_voter(tmp_path):
    """A no-agy batch (opencode swapped in) is flagged; a default batch is not."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    ok = tmp_path / "batch_default"
    _write_votes_csv(ok, ["claude", "codex", "agy"])
    bad = tmp_path / "batch_noagy"
    _write_votes_csv(bad, ["claude", "codex", "opencode"])

    offending = nonstandard_panel_batches([ok, bad])
    assert set(offending) == {"batch_noagy"}
    assert offending["batch_noagy"] == {"claude", "codex", "opencode"}


def test_nonstandard_panel_batches_flags_subset_panel(tmp_path):
    """A degraded 2-provider batch is also nonstandard (missing voter)."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    b = tmp_path / "batch_two"
    _write_votes_csv(b, ["claude", "codex"])
    assert set(nonstandard_panel_batches([b])) == {"batch_two"}


def test_nonstandard_panel_batches_skips_missing_votes(tmp_path):
    """No votes.csv -> best-effort skip, not flagged (consensus gate lives in CLI)."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    b = tmp_path / "batch_novotes"
    b.mkdir()
    assert nonstandard_panel_batches([b]) == {}
