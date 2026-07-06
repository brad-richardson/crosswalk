"""Tests for the stitching option-menu expressibility metric."""

from __future__ import annotations

import json

import pandas as pd

from crosswalk.agent_labeling.stitch_expressibility import (
    measure_expressibility,
    settled_labels,
)


def _edge(ref, tgt, conf=0.8):
    return {"ref_id": ref, "target_id": tgt, "confidence": conf}


def _group(gid, edges, match_type="M:N"):
    return {"group_id": gid, "match_type": match_type, "edges": edges}


def _label_row(gid, edges, match_type="M:N", labeler="brad", **extra):
    selected = json.dumps([{"ref_id": r, "target_id": t} for r, t in edges])
    row = {
        "group_id": gid,
        "dataset_id": "ds",
        "selected_edges": selected,
        "match_type": match_type,
        "labeler": labeler,
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# settled_labels filtering
# ---------------------------------------------------------------------------


def test_settled_labels_drops_reject_all():
    df = pd.DataFrame(
        [
            _label_row("g1", [("r1", "t1")]),
            _label_row("g2", []),  # reject-all
        ]
    )
    out = settled_labels(df)
    assert list(out["group_id"]) == ["g1"]


def test_settled_labels_drops_set_semantics_when_column_present():
    df = pd.DataFrame(
        [
            _label_row("g1", [("r1", "t1")], label_semantics="pair"),
            _label_row("g2", [("r2", "t2")], label_semantics="set"),
        ]
    )
    out = settled_labels(df)
    assert list(out["group_id"]) == ["g1"]


def test_settled_labels_treats_missing_semantics_as_pair():
    df = pd.DataFrame([_label_row("g1", [("r1", "t1")])])
    assert len(settled_labels(df)) == 1


# ---------------------------------------------------------------------------
# measure_expressibility
# ---------------------------------------------------------------------------


def test_full_set_label_is_expressible_via_seed():
    """A label equal to the group's full candidate set is covered by the
    full-set seed even when the per-target enumeration cannot build it."""
    # t1 spans 4 refs (> MAX_REF_CHAIN_LEN); the full set is only reachable via
    # the full-candidate-set seed.
    edges = [_edge(f"r{i}", "t1") for i in range(4)] + [_edge("r0", "t2")]
    groups = [_group("G_full", edges)]
    labels = pd.DataFrame(
        [_label_row("G_full", [(f"r{i}", "t1") for i in range(4)] + [("r0", "t2")])]
    )
    rep = measure_expressibility("ds", groups, labels, k=5)
    assert rep.n_recoverable == 1
    assert rep.n_covered == 1
    assert rep.expressibility == 1.0


def test_inexpressible_subset_is_a_miss():
    """A recoverable label whose set is neither enumerable, the full set, nor
    the selected set counts as an inexpressible miss."""
    # No geometries -> no chains: t1 mapping to BOTH r1 and r2 is not enumerable.
    # Label drops t2 entirely, so it is not the full set either.
    edges = [_edge("r1", "t1"), _edge("r2", "t1"), _edge("r3", "t2")]
    groups = [_group("G_miss", edges)]
    labels = pd.DataFrame([_label_row("G_miss", [("r1", "t1"), ("r2", "t1")])])
    rep = measure_expressibility("ds", groups, labels, k=10)
    assert rep.n_recoverable == 1
    assert rep.n_covered == 0
    assert rep.expressibility == 0.0
    assert len(rep.misses) == 1
    assert rep.misses[0].label_group_id == "G_miss"


def test_split_label_is_not_recoverable():
    """A label whose edges span two current groups is excluded from the rate."""
    g1 = _group("G1", [_edge("r1", "t1"), _edge("r2", "t1")])
    g2 = _group("G2", [_edge("rX", "tY")])
    # Label references one edge from each group -> not clean-recoverable.
    labels = pd.DataFrame([_label_row("Gsplit", [("r1", "t1"), ("rX", "tY")])])
    rep = measure_expressibility("ds", [g1, g2], labels, k=5)
    assert rep.n_settled == 1
    assert rep.n_recoverable == 0
    assert rep.expressibility is None


def test_recovery_by_edge_overlap_when_group_id_differs():
    """Recovery is by edge overlap, not group_id, so a drifted id still maps."""
    edges = [_edge("r1", "t1"), _edge("r2", "t1")]
    groups = [_group("CURRENT_ID", edges)]
    labels = pd.DataFrame([_label_row("OLD_ID", [("r1", "t1"), ("r2", "t1")], match_type="N:1")])
    rep = measure_expressibility("ds", groups, labels, k=5)
    assert rep.n_recoverable == 1
    assert rep.per_label[0].sidecar_group_id == "CURRENT_ID"
    # N:1 enumerates the full ref power set, so this two-ref set is expressible.
    assert rep.n_covered == 1


def test_summary_shape():
    edges = [_edge("r1", "t1"), _edge("r2", "t1")]
    groups = [_group("G", edges, match_type="N:1")]
    labels = pd.DataFrame([_label_row("G", [("r1", "t1")], match_type="N:1")])
    s = measure_expressibility("ds", groups, labels, k=5).summary()
    assert set(s) == {
        "dataset",
        "k",
        "n_settled",
        "n_recoverable",
        "n_covered",
        "expressibility",
        "n_misses",
    }
    assert s["dataset"] == "ds"
