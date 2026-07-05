"""Tests for reinterpreting historical cross-product labels as SET labels.

Covers the shared decision (matcher.agent_labeling.xprod.reinterpret_row_to_set):
the cross-product signature, idempotency on already-set rows, panel-row safety,
and the guards that leave non-artifact rows untouched.
"""

from __future__ import annotations

import json

from matcher.agent_labeling.xprod import reinterpret_row_to_set


def _cache_group(candidate_pairs, optimizer_pairs):
    """A cache group: all candidates in ``edges``, optimizer picks in assignment."""
    return {
        "group_id": "g1234567abcd",
        "edges": [{"ref_id": r, "target_id": t} for r, t in candidate_pairs],
        "optimizer_assignment": [{"ref_id": r, "target_id": t} for r, t in optimizer_pairs],
    }


def _row(pairs, labeler="brad", label_semantics="pair", ref_ids="", target_ids=""):
    return {
        "group_id": "g1234567abcd",
        "labeler": labeler,
        "label_semantics": label_semantics,
        "selected_edges": json.dumps([{"ref_id": r, "target_id": t} for r, t in pairs]),
        "ref_ids": ref_ids,
        "target_ids": target_ids,
    }


# A 2x2 grid where the optimizer only kept the diagonal, so the label's stored
# pairs (the full grid) add the off-diagonal beyond the optimizer -> artifact.
_GRID = [("r1", "t1"), ("r1", "t2"), ("r2", "t1"), ("r2", "t2")]
_OPT = [("r1", "t1"), ("r2", "t2")]


def test_cross_product_artifact_converts():
    cache = _cache_group(_GRID, _OPT)
    decision = reinterpret_row_to_set(_row(_GRID), cache, None)
    assert decision == (["r1", "r2"], ["t1", "t2"])


def test_idempotent_on_already_set_row():
    """A row already tagged set is never reconverted (re-runs are no-ops)."""
    cache = _cache_group(_GRID, _OPT)
    row = _row(
        [],
        label_semantics="set",
        ref_ids=json.dumps(["r1", "r2"]),
        target_ids=json.dumps(["t1", "t2"]),
    )
    assert reinterpret_row_to_set(row, cache, None) is None


def test_panel_rows_never_converted():
    cache = _cache_group(_GRID, _OPT)
    for lab in ("panel_unanimous_v1", "panel_unanimous_v2", "panel_future"):
        assert reinterpret_row_to_set(_row(_GRID, labeler=lab), cache, None) is None


def test_explicit_ratification_not_a_grid_untouched():
    """A ratified option that is NOT the full cross-product stays a pair label."""
    cache = _cache_group(_GRID, _OPT)
    # Stored exactly the optimizer diagonal (no pairs beyond the optimizer).
    assert reinterpret_row_to_set(_row(_OPT), cache, None) is None


def test_single_ref_fan_not_flagged():
    """A 1:N fan (single ref) has no ref×target grid -> never an artifact."""
    fan = [("r1", "t1"), ("r1", "t2"), ("r1", "t3")]
    cache = _cache_group(fan, [("r1", "t1")])
    assert reinterpret_row_to_set(_row(fan), cache, None) is None


def test_no_universe_source_skips():
    """A row whose group is in NEITHER the cache NOR the sidecar is skipped."""
    assert reinterpret_row_to_set(_row(_GRID), None, None) is None


def test_sidecar_only_group_converts():
    """A group present only in the groups sidecar (not the review-queue cache)
    still supplies the candidate universe: sidecar edges[selected] is the
    optimizer set, rejected_edges extend the universe."""
    sidecar_group = {
        "group_id": "g1234567abcd",
        "edges": [
            {"ref_id": "r1", "target_id": "t1", "selected": True},
            {"ref_id": "r2", "target_id": "t2", "selected": True},
            {"ref_id": "r1", "target_id": "t2", "selected": False},
        ],
        "rejected_edges": [{"ref_id": "r2", "target_id": "t1"}],
    }
    decision = reinterpret_row_to_set(_row(_GRID), None, sidecar_group)
    assert decision == (["r1", "r2"], ["t1", "t2"])


def test_partial_grid_not_flagged():
    """Stored pairs are a strict subset of the grid (not the full cross-product)."""
    cache = _cache_group(_GRID, _OPT)
    partial = [("r1", "t1"), ("r1", "t2"), ("r2", "t1")]  # missing (r2,t2)
    assert reinterpret_row_to_set(_row(partial), cache, None) is None


def test_nan_and_malformed_selected_edges_do_not_crash():
    """A blank CSV cell reads back as float NaN (truthy!) and a hand-edited cell
    may not parse — one bad row must not abort a whole reinterpretation run."""
    from matcher.agent_labeling.xprod import parse_selected_edges

    assert parse_selected_edges(float("nan")) == set()
    assert parse_selected_edges(None) == set()
    assert parse_selected_edges("") == set()
    assert parse_selected_edges("   ") == set()
    assert parse_selected_edges("not-a-list") == set()
    assert parse_selected_edges("{'ref_id': 'r1'}") == set()  # not a list

    cache = _cache_group(_GRID, _OPT)
    row = _row(_GRID)
    row["selected_edges"] = float("nan")
    assert reinterpret_row_to_set(row, cache, None) is None  # empty -> no artifact
