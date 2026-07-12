"""Focused regressions for pipeline orchestration summaries."""

from crosswalk.matching.types import MatchDecision, MatchResult
from crosswalk.pipeline.runner import _partition_target_decision_ids


def _result(ref_id: str, target_id: str, decision: MatchDecision) -> MatchResult:
    return MatchResult(ref_id, target_id, decision, 0.9, {}, {})


def test_target_decision_partition_gives_match_precedence_over_review() -> None:
    """A mixed-decision target belongs to exactly one summary category."""
    matches = [
        _result("r_match", "t_mixed", MatchDecision.MATCH),
        _result("r_review", "t_mixed", MatchDecision.REVIEW),
        _result("r_review_only", "t_review", MatchDecision.REVIEW),
    ]

    matched, review = _partition_target_decision_ids(matches)

    assert matched == {"t_mixed"}
    assert review == {"t_review"}
    assert matched.isdisjoint(review)
