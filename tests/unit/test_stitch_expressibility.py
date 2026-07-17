"""Tests for the stitching option-menu expressibility metric."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from crosswalk.agent_labeling.stitch_expressibility import (
    measure_expressibility,
    set_settled_labels,
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


def _set_label_row(gid, ref_ids, target_ids, match_type="M:N", labeler="brad", **extra):
    """Build a SET-semantics label row: membership only, no edges (#3c3e6853)."""
    row = {
        "group_id": gid,
        "dataset_id": "ds",
        "selected_edges": "[]",
        "match_type": match_type,
        "labeler": labeler,
        "label_semantics": "set",
        "ref_ids": json.dumps(sorted(ref_ids)),
        "target_ids": json.dumps(sorted(target_ids)),
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
# set_settled_labels filtering (companion to settled_labels: selects what that
# function drops)
# ---------------------------------------------------------------------------


def test_set_settled_labels_selects_only_set_rows_with_membership():
    df = pd.DataFrame(
        [
            _label_row("g1", [("r1", "t1")], label_semantics="pair"),
            _set_label_row("g2", ["r2"], ["t2"]),
            _set_label_row("g3", [], []),  # empty membership -> analogue of reject-all
        ]
    )
    out = set_settled_labels(df)
    assert list(out["group_id"]) == ["g2"]


def test_set_settled_labels_empty_when_no_semantics_column():
    df = pd.DataFrame([_label_row("g1", [("r1", "t1")])])
    out = set_settled_labels(df)
    assert len(out) == 0


def test_set_settled_labels_empty_frame():
    out = set_settled_labels(pd.DataFrame())
    assert len(out) == 0


def test_settled_labels_and_set_settled_labels_are_disjoint():
    """The two filters must never double-count a row (each row belongs to
    exactly one of pair-settled / set-settled / dropped)."""
    df = pd.DataFrame(
        [
            _label_row("g1", [("r1", "t1")]),
            _label_row("g2", []),  # reject-all -> dropped by both
            _set_label_row("g3", ["r2"], ["t2"]),
            _set_label_row("g4", [], []),  # empty membership -> dropped by both
        ]
    )
    pair_ids = set(settled_labels(df)["group_id"])
    set_ids = set(set_settled_labels(df)["group_id"])
    assert pair_ids == {"g1"}
    assert set_ids == {"g3"}
    assert pair_ids.isdisjoint(set_ids)
    assert len(pair_ids) + len(set_ids) == 2  # g2, g4 dropped by both


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
    # Label drops t2 entirely, so it is not the full set either. Edges are
    # uniformly strong and all optimizer-selected, so the "base minus flagged
    # edge" seeds flag nothing (no weak/sliver/unselected edge to drop) and the
    # r3/t2 exclusion stays inexpressible -- exactly the miss this test asserts.
    edges = [_edge("r1", "t1", 0.99), _edge("r2", "t1", 0.99), _edge("r3", "t2", 0.99)]
    for e in edges:
        e["selected"] = True
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
        "n_set_settled",
        "n_set_recoverable",
        "n_set_covered",
        "set_expressibility",
        "n_set_misses",
        "set_mean_best_boundary_precision",
        "set_mean_best_coverage",
        "n_dropped",
    }
    assert s["dataset"] == "ds"


# ---------------------------------------------------------------------------
# measure_expressibility: SET-semantics labels (#3c3e6853 bug fix)
#
# Previously settled_labels() silently dropped label_semantics == "set" rows,
# so expressibility never noticed when the option generator structurally
# could not express a human's ref/target exclusions on a large group. These
# labels are now scored separately (membership/boundary/coverage against the
# best generated option, via the existing set_label_metrics machinery) and
# reported alongside the pair-semantics results.
# ---------------------------------------------------------------------------


def test_set_label_no_option_honors_exclusions():
    """Mirrors co_bogota_roads group 3c3e6853 (see #367/#368 stitching-label
    work): the human kept only a subset of refs/targets, but every generated
    option -- organic top-K AND the full-set seed -- assigns every target, so
    none can express the exclusion.

    k=1 pins this down deterministically: with exactly one ref per target and
    positive confidences, the single highest-confidence assignment is always
    "assign every target" (leaving any target unassigned strictly lowers total
    confidence), so it is the ONLY option generated -- the full-set seed
    dedups into it. This is the small-fixture analogue of the real bug: no
    option in the menu ever omits a ref/target the human excluded.

    The edges are uniformly strong and all optimizer-selected, so the "base
    minus flagged edge" seeds flag nothing (no weak/sliver/unselected edge to
    drop): the r3/t3 exclusion stays inexpressible, preserving the miss this
    test documents.
    """
    edges = [_edge("r1", "t1", 0.99), _edge("r2", "t2", 0.99), _edge("r3", "t3", 0.99)]
    for e in edges:
        e["selected"] = True
    groups = [_group("G_set", edges)]
    # Human excludes r3 / t3 -- keeps only r1, r2, t1, t2.
    labels = pd.DataFrame([_set_label_row("G_set", ["r1", "r2"], ["t1", "t2"])])
    rep = measure_expressibility("ds", groups, labels, k=1)

    assert rep.n_set_settled == 1
    assert rep.n_set_recoverable == 1
    assert rep.n_set_covered == 0
    assert rep.set_expressibility == 0.0
    assert len(rep.set_misses) == 1

    miss = rep.set_misses[0]
    assert miss.label_group_id == "G_set"
    assert miss.sidecar_group_id == "G_set"
    assert miss.recoverable is True
    assert miss.covered is False
    # The lone generated option is the full 3-edge set: 2 of its 3 edges honor
    # the asserted membership (the r3/t3 edge does not) -> boundary precision
    # is capped below 1.0, exactly the "0.806"-style gap the bug report flags.
    assert miss.best_boundary_precision == pytest.approx(2 / 3)
    assert miss.n_options == 1
    # The report-level aggregate is rounded to 4dp; the per-label value is not.
    assert rep.set_mean_best_boundary_precision == pytest.approx(2 / 3, abs=1e-4)

    # Pair metric is untouched: this store has no pair-semantics rows.
    assert rep.n_settled == 0
    assert rep.n_covered == 0


def test_pair_only_store_leaves_set_metrics_empty():
    """A pure pair-semantics store scores exactly as before this fix, with the
    new SET counters reporting empty/zero rather than being silently absent."""
    edges = [_edge("r1", "t1"), _edge("r2", "t1")]
    groups = [_group("G", edges, match_type="N:1")]
    labels = pd.DataFrame([_label_row("G", [("r1", "t1")], match_type="N:1")])
    rep = measure_expressibility("ds", groups, labels, k=5)

    # Pair scoring unchanged.
    assert rep.n_settled == 1
    assert rep.n_recoverable == 1
    assert rep.n_covered == 1
    assert rep.expressibility == 1.0

    # No set-semantics rows in the store.
    assert rep.n_set_settled == 0
    assert rep.n_set_recoverable == 0
    assert rep.n_set_covered == 0
    assert rep.set_expressibility is None
    assert rep.set_mean_best_boundary_precision is None
    assert rep.set_mean_best_coverage is None
    assert rep.n_dropped == 0


def test_mixed_store_scores_both_semantics_with_correct_counts():
    """A store with a settled pair row, a reject-all pair row, a settled set
    row, and an empty-membership set row: both semantics are scored
    independently, and n_dropped accounts for exactly the two rows that
    assert nothing."""
    pair_group = _group("G1", [_edge("r1", "t1")])
    set_group = _group("G3", [_edge("rA", "tA"), _edge("rB", "tB")])
    groups = [pair_group, set_group]

    labels = pd.DataFrame(
        [
            _label_row("G1", [("r1", "t1")]),  # settled pair, expressible
            _label_row("G2", []),  # reject-all pair -> dropped
            _set_label_row("G3", ["rA", "rB"], ["tA", "tB"]),  # settled set, full membership
            _set_label_row("G4", [], []),  # empty membership -> dropped
        ]
    )
    rep = measure_expressibility("ds", groups, labels, k=1)

    assert rep.n_settled == 1
    assert rep.n_recoverable == 1
    assert rep.n_covered == 1

    assert rep.n_set_settled == 1
    assert rep.n_set_recoverable == 1
    assert rep.n_set_covered == 1
    assert rep.set_expressibility == 1.0

    assert rep.n_dropped == 2

    s = rep.summary()
    assert s["n_settled"] + s["n_set_settled"] + s["n_dropped"] == len(labels)
