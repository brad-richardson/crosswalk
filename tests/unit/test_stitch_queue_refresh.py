"""Regression tests for the stale-proposal (queue vs sidecar) invariant.

This is the FOURTH instance of the display-path corruption bug class (menus
#238, clipping #262, worktree labels, and now stale cache proposals): a value is
rendered / ratified from a snapshot that has drifted from the source of truth.

The guarded invariant: for every review-queue entry whose ``group_id`` is
present in the current groups sidecar, the queue entry's ``optimizer_assignment``
(what the UI pre-seeds and the "auto" option card shows) MUST equal the sidecar
group's selected edge set. Any drift means reviewers are being shown a stale
proposal. ``check_queue_optimizer_parity`` is the maintenance check the cache
rebuild path (``stitch-batch`` / ``stitch-refresh-queue``) calls.
"""

from crosswalk.matching.stitch_queue_refresh import (
    STALE_GROUPING_KEY,
    check_queue_optimizer_parity,
    optimizer_pair_set,
    plan_queue_refresh,
    selected_pair_set,
)


def _sidecar_group(gid, selected, rejected=()):
    """A sidecar group: edges carry a ``selected`` flag; optimizer_assignment
    lists the selected pairs (mirrors the real sidecar invariant)."""
    edges = [
        {"ref_id": r, "target_id": t, "confidence": 0.95, "selected": True} for r, t in selected
    ]
    edges += [
        {"ref_id": r, "target_id": t, "confidence": 0.30, "selected": False} for r, t in rejected
    ]
    return {
        "group_id": gid,
        "edges": edges,
        "optimizer_assignment": [
            {"ref_id": r, "target_id": t, "confidence": 0.95} for r, t in selected
        ],
    }


def _queue_entry(gid, proposed):
    """A queue entry proposing ``proposed`` as its optimizer_assignment."""
    return {
        "group_id": gid,
        "optimizer_assignment": [
            {"ref_id": r, "target_id": t, "confidence": 0.9} for r, t in proposed
        ],
        "edges": [
            {"ref_id": r, "target_id": t, "confidence": 0.9, "selected": True} for r, t in proposed
        ],
    }


class TestPairSets:
    def test_selected_prefers_selected_flag(self):
        g = _sidecar_group("g1", [("r1", "t1")], rejected=[("r2", "t2")])
        assert selected_pair_set(g) == {("r1", "t1")}

    def test_selected_empty_reject_all_not_optimizer_fallback(self):
        # Edges carry the flag but none selected → genuine reject-all; must NOT
        # fall back to a stale optimizer_assignment.
        g = {
            "group_id": "g1",
            "edges": [{"ref_id": "r1", "target_id": "t1", "selected": False}],
            "optimizer_assignment": [{"ref_id": "r1", "target_id": "t1"}],
        }
        assert selected_pair_set(g) == set()

    def test_selected_falls_back_to_optimizer_without_flag(self):
        g = {
            "group_id": "g1",
            "edges": [{"ref_id": "r1", "target_id": "t1"}],  # no 'selected' key
            "optimizer_assignment": [{"ref_id": "r1", "target_id": "t1"}],
        }
        assert selected_pair_set(g) == {("r1", "t1")}

    def test_optimizer_pair_set(self):
        assert optimizer_pair_set(_queue_entry("g1", [("r1", "t1"), ("r2", "t2")])) == {
            ("r1", "t1"),
            ("r2", "t2"),
        }


class TestCheckQueueOptimizerParity:
    def test_in_parity_returns_empty(self):
        sidecar = {"g1": _sidecar_group("g1", [("r1", "t1"), ("r2", "t2")])}
        queue = [_queue_entry("g1", [("r1", "t1"), ("r2", "t2")])]
        assert check_queue_optimizer_parity(queue, sidecar) == []

    def test_stale_extra_pair_detected(self):
        # Queue proposes 3 pairs; sidecar now selects only 2 (the third was
        # pruned/retrained away). This is the 56a0b1cd scenario.
        sidecar = {
            "g1": _sidecar_group("g1", [("r1", "t1"), ("r2", "t2")], rejected=[("r3", "t3")])
        }
        queue = [_queue_entry("g1", [("r1", "t1"), ("r2", "t2"), ("r3", "t3")])]
        drift = check_queue_optimizer_parity(queue, sidecar)
        assert len(drift) == 1
        assert drift[0]["group_id"] == "g1"
        assert drift[0]["queue_only"] == [("r3", "t3")]
        assert drift[0]["sidecar_only"] == []

    def test_missing_pair_detected(self):
        sidecar = {"g1": _sidecar_group("g1", [("r1", "t1"), ("r2", "t2")])}
        queue = [_queue_entry("g1", [("r1", "t1")])]
        drift = check_queue_optimizer_parity(queue, sidecar)
        assert drift[0]["sidecar_only"] == [("r2", "t2")]

    def test_old_grouping_entry_skipped(self):
        # A queue entry whose group is gone from the sidecar is NOT checkable.
        sidecar = {"g1": _sidecar_group("g1", [("r1", "t1")])}
        queue = [_queue_entry("g1", [("r1", "t1")]), _queue_entry("gone", [("rX", "tX")])]
        assert check_queue_optimizer_parity(queue, sidecar) == []


class TestPlanQueueRefresh:
    def test_classifies_refreshable_and_stale_preserving_order(self):
        sidecar = {"g1": _sidecar_group("g1", [("r1", "t1")]), "g3": _sidecar_group("g3", [])}
        queue = [
            _queue_entry("g1", [("r1", "t1")]),
            _queue_entry("gone", [("rX", "tX")]),
            _queue_entry("g3", []),
        ]
        refreshable, stale = plan_queue_refresh(queue, sidecar)
        assert refreshable == ["g1", "g3"]
        assert stale == ["gone"]

    def test_stale_key_constant(self):
        assert STALE_GROUPING_KEY == "stale_grouping"
