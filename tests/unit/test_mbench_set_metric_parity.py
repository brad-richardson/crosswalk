"""Parity guard: mbench's set-label metric core must match matcher's.

mbench is a standalone benchmarking package that cannot depend on matcher, so it
replicates the SET-semantics scoring core (membership / boundary / coverage) from
``matcher.agent_labeling.stitch_eval.set_label_metrics`` in
``mbench.eval.stitch_metrics.set_label_metrics``. This test asserts the two agree
across a representative grid of predicted-edge / membership inputs, so they cannot
drift silently (mirrors ``test_mbench_sliver_parity.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# mbench lives in its own package tree; add it to the path so the main test suite
# (which can import both) can compare the two implementations.
_MBENCH_SRC = Path(__file__).resolve().parents[2] / "mbench" / "src"
if str(_MBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(_MBENCH_SRC))

from mbench.eval.stitch_metrics import set_label_metrics as mbench_set_metrics  # noqa: E402

from matcher.agent_labeling.stitch_eval import (
    set_label_metrics as matcher_set_metrics,  # noqa: E402
)

# (pred_edges, ref_members, target_members) fixtures covering: exact match,
# boundary-crossing edges, partial coverage, empty prediction, empty membership,
# and supersets on either side.
_CASES = [
    # perfect: prediction covers exactly the asserted membership
    (frozenset({("r1", "t1"), ("r2", "t2")}), frozenset({"r1", "r2"}), frozenset({"t1", "t2"})),
    # boundary crossing: an edge reaches a non-member target
    (frozenset({("r1", "t1"), ("r1", "t9")}), frozenset({"r1"}), frozenset({"t1"})),
    # missing coverage: a member has no incident edge
    (frozenset({("r1", "t1")}), frozenset({"r1", "r2"}), frozenset({"t1", "t2"})),
    # empty prediction against non-empty membership
    (frozenset(), frozenset({"r1"}), frozenset({"t1"})),
    # empty membership (degenerate)
    (frozenset({("r1", "t1")}), frozenset(), frozenset()),
    # both empty
    (frozenset(), frozenset(), frozenset()),
    # fully within but extra members uncovered + one boundary edge
    (
        frozenset({("r1", "t1"), ("r2", "t3")}),
        frozenset({"r1", "r2"}),
        frozenset({"t1", "t2"}),
    ),
    # prediction membership is a superset of asserted (extra ref not a member)
    (frozenset({("r1", "t1"), ("r3", "t1")}), frozenset({"r1"}), frozenset({"t1"})),
]


@pytest.mark.parametrize("pred,ref_members,tgt_members", _CASES)
def test_set_metric_parity(pred, ref_members, tgt_members):
    expected = matcher_set_metrics(pred, ref_members, tgt_members)
    actual = mbench_set_metrics(pred, ref_members, tgt_members)
    assert actual == expected, (
        f"mismatch for pred={pred} ref={ref_members} tgt={tgt_members}: "
        f"matcher={expected} mbench={actual}"
    )


def test_set_metric_semantics():
    """Spot-check the three components carry the intended meaning."""
    # Perfect membership.
    exact, boundary, coverage = matcher_set_metrics(
        frozenset({("r1", "t1"), ("r2", "t2")}),
        frozenset({"r1", "r2"}),
        frozenset({"t1", "t2"}),
    )
    assert exact and boundary == 1.0 and coverage == 1.0

    # One of two predicted edges crosses into a non-member target -> boundary 0.5,
    # membership not exact (t9 present), coverage full for the asserted members.
    exact, boundary, coverage = matcher_set_metrics(
        frozenset({("r1", "t1"), ("r1", "t9")}),
        frozenset({"r1"}),
        frozenset({"t1"}),
    )
    assert not exact
    assert boundary == 0.5
    assert coverage == 1.0

    # Half the members uncovered.
    exact, boundary, coverage = matcher_set_metrics(
        frozenset({("r1", "t1")}),
        frozenset({"r1", "r2"}),
        frozenset({"t1", "t2"}),
    )
    assert not exact
    assert boundary == 1.0
    assert coverage == 0.5
