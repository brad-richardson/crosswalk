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
    PANEL_NONE_LABELER,
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


def make_batch(
    batch_dir: Path,
    dataset: str,
    groups: list[dict],
    voters: list[tuple[str, str]] | None = _V4_VOTERS,
) -> Path:
    """Write a synthetic batch dir (consensus.csv, batch.json, metadata.yaml).

    Each ``groups`` entry:
      group_id, match_type, routing, choice, mean_confidence,
      edges: list of (ref, tgt) chosen edges (the consensus edge_set),
      ref_classes / target_classes: {id: class},
      sliver_edges: set of (ref, tgt) to encode as tiny-span (sliver) edges,
      candidate_edges: optional extra (ref, tgt) present in batch.json but not chosen.

    ``voters`` writes a votes.csv with those (provider, model) ballots so the
    batch resolves to a labeler era (default: the blessed v4 composition —
    write_exports refuses era-less batches). Pass ``voters=None`` to omit
    votes.csv and exercise the era-less path.
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
                    "n_votes": g.get("n_votes", 3),
                    "n_valid": g.get("n_valid", 3),
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


def test_standard_v3_and_v4_batches_pass(tmp_path):
    """Both blessed eras validate: v3 history is never retroactively flagged."""
    from crosswalk.agent_labeling.stitch_export import nonstandard_panel_batches

    v3 = tmp_path / "batch_v3"
    _write_votes_csv(v3, _V3_VOTERS)
    v4 = tmp_path / "batch_v4"
    _write_votes_csv(v4, _V4_VOTERS)
    assert nonstandard_panel_batches([v3, v4]) == {}


def test_default_panel_voters_match_runner_default_panel():
    """stitch_export's blessed v4 set stays in lockstep with the live DEFAULT_PANEL.

    A composition change in stitch_runner without a provenance decision here
    (bump the labeler + the blessed set) must fail CI, not silently drift.
    """
    from crosswalk.agent_labeling.stitch_export import (
        DEFAULT_PANEL_VOTERS,
        PANEL_VOTERS_V4,
    )
    from crosswalk.agent_labeling.stitch_runner import DEFAULT_PANEL

    assert DEFAULT_PANEL_VOTERS == PANEL_VOTERS_V4
    assert frozenset((p.name, p.model) for p in DEFAULT_PANEL) == DEFAULT_PANEL_VOTERS


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
    """batch_panel_era: v3 -> "v3", v4 -> "v4", known-historical -> its era,
    anything else -> None (no silent default)."""
    from crosswalk.agent_labeling.stitch_export import batch_panel_era

    v3 = tmp_path / "b_v3"
    _write_votes_csv(v3, _V3_VOTERS)
    v4 = tmp_path / "b_v4"
    _write_votes_csv(v4, _V4_VOTERS)
    hist = tmp_path / "b_hist"
    _write_votes_csv(hist, _HISTORICAL_GEMINI_TRANSPORT_VOTERS)
    odd = tmp_path / "b_odd"
    _write_votes_csv(odd, _V4_VOTERS[:2])
    none = tmp_path / "b_none"
    none.mkdir()

    assert batch_panel_era(v3) == "v3"
    assert batch_panel_era(v4) == "v4"
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
    """Era-scoped labeler stamping: a v3-era batch mints panel_unanimous_v3 and
    a v4 batch mints panel_unanimous_v4. Re-exporting committed v3 history must
    never silently rewrite its provenance to v4."""
    from crosswalk.agent_labeling.stitch_export import PANEL_LABELER_V3

    b_v3 = _one_group_batch(tmp_path, "b_v3", "g_v3", voters=_V3_VOTERS)
    b_v4 = _one_group_batch(tmp_path, "b_v4", "g_v4", voters=_V4_VOTERS)

    report = plan_exports([b_v3, b_v4], DATASET, labels_dir)
    assert {g.group_id: g.panel_era for g in report.groups} == {"g_v3": "v3", "g_v4": "v4"}
    assert write_exports(report, DATASET, labels_dir) == 2

    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    labelers = dict(zip(df["group_id"], df["labeler"], strict=True))
    assert labelers == {"g_v3": PANEL_LABELER_V3, "g_v4": PANEL_LABELER}


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

    # Mixed set: the blessed-v4 batch keeps v4; only the era-less one fills in.
    b_v4 = _one_group_batch(tmp_path, "b_v4", "g_v4", voters=_V4_VOTERS)
    report = plan_exports([b_unknown, b_v4], DATASET, labels_dir, stamp_era="v3")
    assert {g.group_id: g.panel_era for g in report.groups} == {
        "g_unknown": "v3",
        "g_v4": "v4",
    }
    assert write_exports(report, DATASET, labels_dir) == 2
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    labelers = dict(zip(df["group_id"], df["labeler"], strict=True))
    assert labelers["g_v4"] == PANEL_LABELER  # never re-stamped by the fill-in
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
# Unanimous-NONE -> empty-set (reject-all) export (resolver L1)
# ---------------------------------------------------------------------------


def _none_group(group_id: str, candidate_edges, **kw) -> dict:
    """A unanimous-NONE group: routed to human_review, empty consensus edge set.

    ``candidate_edges`` are the group's real candidate pairs (present in
    batch.json), while the consensus ``edges`` (chosen set) is empty -- the panel
    rejected every option. ``route_reason`` defaults to the fresh stamp; pass
    ``route_reason=""`` to exercise the historical-derivation path.
    """
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


def test_unanimous_none_exported_as_empty_set(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [_none_group("gnone", [("r1", "t1"), ("r2", "t2")])],
    )
    report = _plan([b], labels_dir)
    g = _by_gid(report)["gnone"]
    assert g.exported
    assert g.is_empty_set
    assert g.reason == REASON_EXPORTED
    assert g.selected_edges == []
    assert g.n_edges_final == 0
    assert report.n_unanimous_none == 1
    assert report.exported_empty == [g]
    assert report.exported_nonempty == []


def test_unanimous_none_derived_without_stamp(tmp_path, labels_dir):
    # A historical wave with a blank route_reason is still recognized as
    # unanimous-NONE from consensus=unanimous + choice=NONE.
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [_none_group("gnone", [("r1", "t1")], route_reason="")],
    )
    g = _by_gid(_plan([b], labels_dir))["gnone"]
    assert g.exported and g.is_empty_set


def test_unanimous_none_below_quorum_not_exported(tmp_path, labels_dir):
    # Defense-in-depth: a hand-edited / hypothetical pre-quorum-rule historical
    # row claiming consensus=unanimous + choice=NONE with n_valid < 3 must NOT
    # mint reject ground truth — with or without the route_reason stamp
    # (contradicting n_valid evidence wins over the stamp).
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            _none_group("g_derived", [("r1", "t1")], n_valid=2, route_reason=""),
            _none_group("g_stamped", [("r2", "t2")], n_valid=2),
        ],
    )
    report = _plan([b], labels_dir)
    assert "g_derived" not in _by_gid(report)
    assert "g_stamped" not in _by_gid(report)
    assert report.n_unanimous_none == 0


def test_unanimous_none_missing_n_valid_requires_stamp(tmp_path, labels_dir):
    # No n_valid evidence at all: only the compute_consensus route_reason stamp
    # (which enforces the quorum at write time) is trusted; derivation from the
    # consensus/choice columns alone is not.
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            _none_group("g_stamped", [("r1", "t1")], n_valid="", n_votes=""),
            _none_group("g_derived", [("r2", "t2")], n_valid="", n_votes="", route_reason=""),
        ],
    )
    report = _plan([b], labels_dir)
    gids = _by_gid(report)
    assert gids["g_stamped"].exported and gids["g_stamped"].is_empty_set
    assert "g_derived" not in gids
    assert report.n_unanimous_none == 1


def test_unanimous_none_writes_reject_all_row(tmp_path, labels_dir):
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [_none_group("gnone", [("r1", "t1")], match_type="M:N")],
    )
    report = _plan([b], labels_dir)
    written = write_exports(report, DATASET, Path(labels_dir))
    assert written == 1

    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    row = df.iloc[0]
    # Same on-disk shape as a human reject-all: PAIR semantics, empty edges, 0/0.
    assert row["labeler"] == PANEL_NONE_LABELER
    assert row["labeler"].startswith("panel_")  # stays non-human for all consumers
    assert json.loads(row["selected_edges"]) == []
    assert row["num_refs"] == 0 and row["num_targets"] == 0
    assert str(row.get("label_semantics") or "pair") == "pair"
    assert row["session_id"] == "b1"


def test_non_unanimous_none_not_exported(tmp_path, labels_dir):
    # A NONE that carries a dissent (majority, not unanimous) is NOT a candidate.
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            _none_group(
                "gmaj",
                [("r1", "t1")],
                consensus="majority",
                minority="codex=A",
                route_reason="",  # force derivation -> dissent, not unanimous_none
            )
        ],
    )
    report = _plan([b], labels_dir)
    assert "gmaj" not in _by_gid(report)  # not a candidate at all
    assert report.n_unanimous_none == 0


def test_empty_set_human_precedence_by_group_id(tmp_path, labels_dir):
    # A prior human reject-all on the same group_id must not be overwritten.
    store = StitchingLabelStore(DATASET, labels_dir=labels_dir)
    store.add("gnone", [], "M:N", 0, 0, "brad", "s1")  # human reject-all
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [_none_group("gnone", [("r1", "t1")])],
    )
    g = _by_gid(_plan([b], labels_dir))["gnone"]
    assert not g.exported
    assert g.reason == REASON_HUMAN_PRECEDENCE
    assert g.is_empty_set


def test_empty_set_human_precedence_by_edge_overlap(tmp_path, labels_dir):
    # A human ACCEPT label sharing a candidate edge blocks the empty-set export.
    store = StitchingLabelStore(DATASET, labels_dir=labels_dir)
    store.add("old_hash", [{"ref_id": "r1", "target_id": "t1"}], "M:N", 1, 1, "brad", "s1")
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [_none_group("new_hash", [("r1", "t1"), ("r2", "t2")])],
    )
    g = _by_gid(_plan([b], labels_dir))["new_hash"]
    assert not g.exported
    assert g.reason == REASON_HUMAN_PRECEDENCE
    assert g.human_group_id == "old_hash"


def test_empty_set_cross_mode_still_exported(tmp_path, labels_dir):
    # The class gate is vacuous on an empty set, so a cross-mode reject
    # (road ref + cycleway target, the co_bogota_bike_network shape) still
    # exports -- this is the whole point of empty-set labels.
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            _none_group(
                "xmode_none",
                [("r1", "t1")],
                ref_classes={"r1": "primary"},
                target_classes={"t1": "cycleway"},
            )
        ],
    )
    g = _by_gid(_plan([b], labels_dir))["xmode_none"]
    assert g.exported and g.is_empty_set


def test_empty_set_multi_corridor_tangle_still_exported(tmp_path, labels_dir):
    # The corridor/assignment-tangle sub-gate (which blocks small tangles on the
    # accept path) is NOT applied to empty sets: a 3-corridor reject exports.
    edges = [(f"r{i}", f"t{i}") for i in range(6)]
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            _none_group(
                "tangle_none",
                edges,
                n_edges=6,
                n_corridors=3,
                n_assignment_components=3,
            )
        ],
    )
    g = _by_gid(_plan([b], labels_dir))["tangle_none"]
    assert g.exported and g.is_empty_set


def test_empty_set_over_backstop_blocked(tmp_path, labels_dir):
    # A genuine monster reject (candidate size over the hard backstop) still
    # routes to a human rather than auto-committing a blanket NONE.
    edges = [(f"r{i}", f"t{i}") for i in range(45)]
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [_none_group("giant_none", edges, n_edges=45)],
    )
    g = _by_gid(_plan([b], labels_dir, backstop_max_edges=40))["giant_none"]
    assert not g.exported
    assert g.reason == REASON_OVER_MAX
    assert g.is_empty_set


def test_empty_set_over_flat_max_without_structure(tmp_path, labels_dir):
    # Without structure fields, the empty-set ceiling is the flat max_edges on the
    # group's candidate count.
    edges = [(f"r{i}", f"t{i}") for i in range(25)]
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [_none_group("big_none", edges)],  # no n_edges field -> flat cap
    )
    g = _by_gid(_plan([b], labels_dir, max_edges=20))["big_none"]
    assert not g.exported
    assert g.reason == REASON_OVER_MAX


def test_empty_set_disabled_flag(tmp_path, labels_dir):
    # export_empty_set=False: unanimous-NONE is not a candidate; accepts still are.
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            _none_group("gnone", [("r1", "t1")]),
            {"group_id": "gyes", "routing": "auto_accept", "edges": [("r2", "t2")]},
        ],
    )
    report = _plan([b], labels_dir, export_empty_set=False)
    gids = _by_gid(report)
    assert "gnone" not in gids
    assert gids["gyes"].exported
    assert report.n_unanimous_none == 0


def test_empty_set_round_trip_through_store(tmp_path, labels_dir):
    # Store -> load -> recovery: the panel empty-set label is recovered exactly
    # like a human reject-all (empty bucket / verbatim group_id).
    from crosswalk.agent_labeling.stitch_eval import (
        recover_empty_reject_all,
        recover_labeled_groups,
    )

    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [_none_group("gnone", [("r1", "t1")])],
    )
    write_exports(_plan([b], labels_dir), DATASET, Path(labels_dir))
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)

    groups = [{"group_id": "gnone", "edges": [{"ref_id": "r1", "target_id": "t1"}]}]
    rec = recover_labeled_groups(groups, df)
    assert rec["empty"] == ["gnone"]  # classified as reject-all, not set/clean/lost
    assert rec["clean"] == [] and rec["set"] == []

    emp = recover_empty_reject_all(groups, df)
    assert emp["recovered"] == ["gnone"] and emp["unrecoverable"] == []


def test_empty_and_accept_coexist(tmp_path, labels_dir):
    # A batch with both an accept and a reject-all group exports both, each with
    # its own labeler.
    b = make_batch(
        tmp_path / "b1",
        DATASET,
        [
            {"group_id": "gyes", "routing": "auto_accept", "edges": [("r1", "t1")]},
            _none_group("gno", [("r2", "t2")]),
        ],
    )
    report = _plan([b], labels_dir)
    assert len(report.exported) == 2
    assert {g.group_id for g in report.exported_empty} == {"gno"}
    assert {g.group_id for g in report.exported_nonempty} == {"gyes"}

    write_exports(report, DATASET, Path(labels_dir))
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    labelers = dict(zip(df["group_id"], df["labeler"]))
    assert labelers["gyes"] == PANEL_LABELER
    assert labelers["gno"] == PANEL_NONE_LABELER
