"""Tests for recomposed export of decomposed groups (#367 Mode B).

Synthetic batch dirs mirror what ``stitch-batch --decompose`` writes: sub-problem
groups (``parent_group_id``) plus a pack-less parent entry
(``decomposed_parent`` + ``subproblem_ids`` roster) in batch.json, and consensus
rows keyed by sub-problem id. The suite pins the conservative recomposition
rule: a whole-group label is minted ONLY when every roster sub-problem resolved
as a unanimous accept; any failed or unvoted sub-problem blocks the group, and
sub-problem rows never export individually.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from crosswalk.agent_labeling.stitch_export import (
    PANEL_DECOMPOSED_LABELER,
    REASON_CLASS_MISMATCH,
    REASON_EXPORTED,
    REASON_HUMAN_PRECEDENCE,
    REASON_SUBPROBLEM_FAILED,
    REASON_SUBPROBLEMS_UNVOTED,
    plan_exports,
    write_exports,
)
from crosswalk.labeling.stitching_store import StitchingLabelStore
from crosswalk.matching.group_decomposition import subproblem_id

from .test_stitch_export import CONSENSUS_COLUMNS, _edge, _line

DATASET = "test_ds"
PARENT = "beef0001"


@pytest.fixture
def labels_dir(tmp_path):
    return tmp_path / "stitching"


def make_decomposed_batch(
    batch_dir: Path,
    sub_specs: list[dict],
    parent_gid: str = PARENT,
    ref_classes: dict | None = None,
    target_classes: dict | None = None,
    sliver_edges: set | None = None,
    roster_extra: list[str] | None = None,
) -> tuple[Path, list[str]]:
    """Write a synthetic decomposed batch dir.

    Each ``sub_specs`` entry describes one sub-problem:
      edges: candidate (ref, tgt) pairs (also the sub-group's membership),
      chosen: pairs the panel selected (defaults to all edges),
      routing: consensus routing (default auto_accept),
      voted: False to omit the consensus row entirely (unvoted sub-problem),
      mean_confidence: consensus confidence.

    ``roster_extra`` appends extra ids to the parent's roster (e.g. a size-gated
    oversized sub-problem that was never packed or voted). Returns the batch dir
    and the ordered sub-problem ids.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    ref_classes = ref_classes or {}
    target_classes = target_classes or {}
    sliver_edges = sliver_edges or set()

    all_edges: list[tuple[str, str]] = []
    for spec in sub_specs:
        all_edges += list(spec["edges"])
    ref_ids = sorted({r for r, _ in all_edges})
    tgt_ids = sorted({t for _, t in all_edges})
    ref_geoms = {r: _line(42.28 + 0.001 * i, 0.0005) for i, r in enumerate(ref_ids)}
    tgt_geoms = {t: _line(42.29 + 0.001 * i, 0.0005) for i, t in enumerate(tgt_ids)}

    def _edge_dicts(pairs):
        return [_edge(r, t, span=(0.03 if (r, t) in sliver_edges else 1.0)) for r, t in pairs]

    sub_ids: list[str] = []
    json_groups: list[dict] = []
    consensus_rows: list[dict] = []
    for spec in sub_specs:
        pairs = list(spec["edges"])
        sid = subproblem_id(parent_gid, pairs)
        sub_ids.append(sid)
        srefs = sorted({r for r, _ in pairs})
        stgts = sorted({t for _, t in pairs})
        json_groups.append(
            {
                "group_id": sid,
                "parent_group_id": parent_gid,
                "match_type": "M:N",
                "ref_ids": srefs,
                "target_ids": stgts,
                "edges": _edge_dicts(pairs),
                "ref_geometries": {r: ref_geoms[r] for r in srefs},
                "target_geometries": {t: tgt_geoms[t] for t in stgts},
                "ref_classes": {r: ref_classes.get(r, "residential") for r in srefs},
                "target_classes": {t: target_classes.get(t, "residential") for t in stgts},
                "n_edges": len(pairs),
            }
        )
        if not spec.get("voted", True):
            continue
        chosen = spec.get("chosen", pairs)
        consensus_rows.append(
            {
                "group_id": sid,
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": json.dumps([[r, t] for r, t in chosen]),
                "routing": spec.get("routing", "auto_accept"),
                "n_votes": 3,
                "n_valid": 3,
                "minority": "",
                "mean_confidence": spec.get("mean_confidence", 0.9),
                "route_reason": spec.get("route_reason", "unanimous"),
            }
        )

    roster = sub_ids + list(roster_extra or [])
    json_groups.append(
        {
            "group_id": parent_gid,
            "decomposed_parent": True,
            "subproblem_ids": roster,
            "decompose_max_edges": 40,
            "match_type": "M:N",
            "ref_ids": ref_ids,
            "target_ids": tgt_ids,
            "edges": _edge_dicts(all_edges),
            "ref_geometries": ref_geoms,
            "target_geometries": tgt_geoms,
            "ref_classes": {r: ref_classes.get(r, "residential") for r in ref_ids},
            "target_classes": {t: target_classes.get(t, "residential") for t in tgt_ids},
            "n_edges": len(all_edges),
        }
    )

    with (batch_dir / "consensus.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONSENSUS_COLUMNS)
        w.writeheader()
        for row in consensus_rows:
            w.writerow(row)
    (batch_dir / "batch.json").write_text(
        json.dumps({"dataset_id": DATASET, "groups": json_groups})
    )
    return batch_dir, sub_ids


def _plan(batch_dirs, labels_dir, **kw):
    return plan_exports([Path(b) for b in batch_dirs], DATASET, Path(labels_dir), **kw)


def _by_gid(report):
    return {g.group_id: g for g in report.groups}


def test_all_subproblems_accept_exports_union(tmp_path, labels_dir):
    b, sub_ids = make_decomposed_batch(
        tmp_path / "b1",
        [
            {"edges": [("r1", "t1"), ("r2", "t1")]},
            {"edges": [("r3", "t2")]},
        ],
    )
    report = _plan([b], labels_dir)
    gids = _by_gid(report)
    assert report.n_decomposed_parents == 1
    assert report.n_subproblem_rows == 2
    # Sub-problem rows are consumed, never exported individually.
    assert not any(sid in gids for sid in sub_ids)
    parent = gids[PARENT]
    assert parent.exported is True
    assert parent.reason == REASON_EXPORTED
    assert parent.from_decomposition is True
    assert parent.n_subproblems == 2
    assert parent.n_subproblems_resolved == 2
    assert sorted((e["ref_id"], e["target_id"]) for e in parent.selected_edges) == [
        ("r1", "t1"),
        ("r2", "t1"),
        ("r3", "t2"),
    ]


def test_recomposed_export_stamped_decomposed_labeler(tmp_path, labels_dir):
    b, _ = make_decomposed_batch(
        tmp_path / "b1", [{"edges": [("r1", "t1")]}, {"edges": [("r2", "t2")]}]
    )
    report = _plan([b], labels_dir)
    assert write_exports(report, DATASET, labels_dir) == 1
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert list(df["group_id"]) == [PARENT]
    assert list(df["labeler"]) == [PANEL_DECOMPOSED_LABELER]
    stored = json.loads(df.iloc[0]["selected_edges"])
    assert len(stored) == 2


def test_one_failed_subproblem_blocks_group(tmp_path, labels_dir):
    b, sub_ids = make_decomposed_batch(
        tmp_path / "b1",
        [
            {"edges": [("r1", "t1")]},
            {"edges": [("r2", "t2")], "routing": "human_review", "chosen": []},
        ],
    )
    report = _plan([b], labels_dir)
    parent = _by_gid(report)[PARENT]
    assert parent.exported is False
    assert parent.reason == REASON_SUBPROBLEM_FAILED
    assert parent.n_subproblems_resolved == 1
    assert write_exports(report, DATASET, labels_dir) == 0


def test_unanimous_none_subproblem_blocks_and_mints_nothing(tmp_path, labels_dir):
    # A unanimous-NONE SUB verdict must neither complete the group nor mint an
    # empty-set label for the sub-problem id (empty-set export is on by default).
    b, sub_ids = make_decomposed_batch(
        tmp_path / "b1",
        [
            {"edges": [("r1", "t1")]},
            {
                "edges": [("r2", "t2")],
                "routing": "human_review",
                "chosen": [],
                "route_reason": "unanimous_none",
            },
        ],
    )
    report = _plan([b], labels_dir)
    gids = _by_gid(report)
    assert gids[PARENT].reason == REASON_SUBPROBLEM_FAILED
    assert not any(sid in gids for sid in sub_ids)
    assert report.n_unanimous_none == 0
    assert write_exports(report, DATASET, labels_dir) == 0


def test_unvoted_subproblem_blocks_group(tmp_path, labels_dir):
    b, _ = make_decomposed_batch(
        tmp_path / "b1",
        [
            {"edges": [("r1", "t1")]},
            {"edges": [("r2", "t2")], "voted": False},
        ],
    )
    report = _plan([b], labels_dir)
    parent = _by_gid(report)[PARENT]
    assert parent.exported is False
    assert parent.reason == REASON_SUBPROBLEMS_UNVOTED


def test_size_gated_roster_entry_blocks_group(tmp_path, labels_dir):
    # An oversized irreducible sub-problem is in the roster but never packed or
    # voted: the group must stay blocked until a human resolves it.
    b, _ = make_decomposed_batch(
        tmp_path / "b1",
        [{"edges": [("r1", "t1")]}],
        roster_extra=[subproblem_id(PARENT, [("rX", "tX")])],
    )
    report = _plan([b], labels_dir)
    parent = _by_gid(report)[PARENT]
    assert parent.reason == REASON_SUBPROBLEMS_UNVOTED
    assert parent.n_subproblems == 2
    assert parent.n_subproblems_resolved == 1


def test_union_class_gate_blocks_cross_mode(tmp_path, labels_dir):
    # Sub-verdicts each pass their own vote-time gates, but the union-level
    # class gate still guards the recomposed label (defense in depth).
    b, _ = make_decomposed_batch(
        tmp_path / "b1",
        [{"edges": [("r1", "t1")]}, {"edges": [("r2", "t2")]}],
        ref_classes={"r2": "footway"},
        target_classes={"t2": "residential"},
    )
    report = _plan([b], labels_dir)
    parent = _by_gid(report)[PARENT]
    assert parent.exported is False
    assert parent.reason == REASON_CLASS_MISMATCH


def test_union_sliver_edges_dropped(tmp_path, labels_dir):
    b, _ = make_decomposed_batch(
        tmp_path / "b1",
        [{"edges": [("r1", "t1"), ("r2", "t1")]}, {"edges": [("r3", "t2")]}],
        sliver_edges={("r2", "t1")},
    )
    report = _plan([b], labels_dir)
    parent = _by_gid(report)[PARENT]
    assert parent.exported is True
    assert parent.n_slivers_dropped == 1
    assert parent.n_edges_final == 2
    assert all(e["ref_id"] != "r2" for e in parent.selected_edges)


def test_human_precedence_blocks_recomposed_export(tmp_path, labels_dir):
    store = StitchingLabelStore(DATASET, labels_dir=labels_dir)
    store.add(
        group_id=PARENT,
        selected_edges=[{"ref_id": "r1", "target_id": "t1"}],
        match_type="M:N",
        num_refs=1,
        num_targets=1,
        labeler="brad",
        session_id="s1",
    )
    b, _ = make_decomposed_batch(
        tmp_path / "b1", [{"edges": [("r1", "t1")]}, {"edges": [("r2", "t2")]}]
    )
    report = _plan([b], labels_dir)
    parent = _by_gid(report)[PARENT]
    assert parent.exported is False
    assert parent.reason == REASON_HUMAN_PRECEDENCE
    assert parent.human_group_id == PARENT


def test_partial_chosen_subsets_union_correctly(tmp_path, labels_dir):
    # A sub-problem's panel may choose a strict subset of its candidate edges;
    # the union is over CHOSEN edges only.
    b, _ = make_decomposed_batch(
        tmp_path / "b1",
        [
            {"edges": [("r1", "t1"), ("r2", "t1")], "chosen": [("r1", "t1")]},
            {"edges": [("r3", "t2")]},
        ],
    )
    report = _plan([b], labels_dir)
    parent = _by_gid(report)[PARENT]
    assert parent.exported is True
    assert sorted((e["ref_id"], e["target_id"]) for e in parent.selected_edges) == [
        ("r1", "t1"),
        ("r3", "t2"),
    ]


def test_weakest_link_confidence_reported(tmp_path, labels_dir):
    b, _ = make_decomposed_batch(
        tmp_path / "b1",
        [
            {"edges": [("r1", "t1")], "mean_confidence": 0.95},
            {"edges": [("r2", "t2")], "mean_confidence": 0.62},
        ],
    )
    report = _plan([b], labels_dir)
    assert _by_gid(report)[PARENT].mean_confidence == pytest.approx(0.62)


def test_idempotent_recomposed_write(tmp_path, labels_dir):
    b, _ = make_decomposed_batch(
        tmp_path / "b1", [{"edges": [("r1", "t1")]}, {"edges": [("r2", "t2")]}]
    )
    report = _plan([b], labels_dir)
    write_exports(report, DATASET, labels_dir)
    write_exports(_plan([b], labels_dir), DATASET, labels_dir)
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert len(df) == 1  # upsert by group_id: no duplicates


def test_superseded_sub_row_never_exports_individually(tmp_path, labels_dir):
    # An older wave decomposed the parent differently; its sub-problem id is no
    # longer in the newest roster. Its (auto_accept) row must still be consumed
    # as a sub-problem row — never exported under the orphaned sub-problem id.
    b1, old_ids = make_decomposed_batch(tmp_path / "b1", [{"edges": [("r1", "t1"), ("r2", "t1")]}])
    b2, new_ids = make_decomposed_batch(
        tmp_path / "b2", [{"edges": [("r1", "t1")]}, {"edges": [("r2", "t1")]}]
    )
    assert set(old_ids).isdisjoint(new_ids)
    report = _plan([b1, b2], labels_dir)
    gids = _by_gid(report)
    assert not any(sid in gids for sid in old_ids + new_ids)
    # The newest roster wins: parent recomposes from b2's two sub-problems.
    parent = gids[PARENT]
    assert parent.exported is True
    assert parent.n_subproblems == 2
    assert report.n_subproblem_rows == 3  # 1 orphaned + 2 current
    assert write_exports(report, DATASET, labels_dir) == 1
    df = StitchingLabelStore(DATASET, labels_dir=labels_dir).load(DATASET)
    assert list(df["group_id"]) == [PARENT]
