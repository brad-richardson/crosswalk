"""Evaluation metrics for conflation benchmarking.

Supports two match levels:

- "target" (default): a match-labeled target counts as a TP if its target_id
  appears in *any* prediction, regardless of which reference segment was
  chosen. This avoids penalizing tools for picking a different reference
  segment that covers a different subsegment of the same target road.
  False positives are still counted at the pair level against explicit
  no_match labels, because a no_match label only asserts that one specific
  (ref_id, target_id) pair is wrong — predicting a *different* reference for
  that target is not evidence of error.
- "pair": exact (ref_id, target_id) pair matching. A prediction is a TP only
  if the exact pair appears in the ground truth.

Precision is computed only over predictions that hit labeled ground truth;
predictions on unlabeled pairs are excluded from FP. The `labeled_coverage`
field reports what fraction of predictions were actually evaluated — a low
value means precision is measured against a small labeled subset and should
be read with caution.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MATCH_LEVELS = ("target", "pair")


@dataclass
class EvalResult:
    """Metrics from evaluating predictions against ground truth."""

    true_positives: int
    false_positives: int
    false_negatives: int
    unlabeled_predictions: int
    precision: float
    recall: float
    f1: float
    total_labeled_matches: int
    total_labeled_non_matches: int
    total_predictions: int
    match_level: str = "pair"
    labeled_coverage: float = 0.0
    skipped_unsure: int = 0

    def to_dict(self) -> dict:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "unlabeled_predictions": self.unlabeled_predictions,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "total_labeled_matches": self.total_labeled_matches,
            "total_labeled_non_matches": self.total_labeled_non_matches,
            "total_predictions": self.total_predictions,
            "match_level": self.match_level,
            "labeled_coverage": self.labeled_coverage,
            "skipped_unsure": self.skipped_unsure,
        }


def evaluate(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    match_level: str = "target",
) -> EvalResult:
    """Compute precision/recall/F1 of predictions vs ground truth labels.

    Args:
        predictions: DataFrame with columns [ref_id, target_id, confidence].
        labels: DataFrame with columns [ref_id, target_id, label].
                label values: "match", "no_match", "unsure" (skipped).
        match_level: "target" (default) for target-level matching, or "pair"
                for exact (ref_id, target_id) pair matching. See module
                docstring for semantics.

    Returns:
        EvalResult with all metrics.
    """
    if match_level == "target":
        return _evaluate_target_level(predictions, labels)
    if match_level == "pair":
        return _evaluate_pair_level(predictions, labels)
    raise ValueError(f"Unknown match_level: {match_level!r} (expected one of {MATCH_LEVELS})")


def _prepare_sets(
    predictions: pd.DataFrame, labels: pd.DataFrame
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]], int]:
    """Extract (true_matches, true_non_matches, predicted_set, skipped_unsure)."""
    labels_str = labels.assign(
        ref_id=labels["ref_id"].astype(str), target_id=labels["target_id"].astype(str)
    )
    match_mask = labels_str["label"] == "match"
    no_match_mask = labels_str["label"] == "no_match"
    skipped_unsure = int((labels_str["label"] == "unsure").sum())

    true_matches = set(
        zip(labels_str.loc[match_mask, "ref_id"], labels_str.loc[match_mask, "target_id"])
    )
    true_non_matches = set(
        zip(labels_str.loc[no_match_mask, "ref_id"], labels_str.loc[no_match_mask, "target_id"])
    )
    predicted_set = set(
        zip(predictions["ref_id"].astype(str), predictions["target_id"].astype(str))
    )
    return true_matches, true_non_matches, predicted_set, skipped_unsure


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _evaluate_pair_level(predictions: pd.DataFrame, labels: pd.DataFrame) -> EvalResult:
    """Exact (ref_id, target_id) pair evaluation."""
    true_matches, true_non_matches, predicted_set, skipped_unsure = _prepare_sets(
        predictions, labels
    )

    tp = len(predicted_set & true_matches)
    fp = len(predicted_set & true_non_matches)
    fn = len(true_matches - predicted_set)
    unlabeled = len(predicted_set - true_matches - true_non_matches)

    precision, recall, f1 = _prf(tp, fp, fn)
    total_predictions = len(predicted_set)
    labeled_coverage = (
        (total_predictions - unlabeled) / total_predictions if total_predictions > 0 else 0.0
    )

    return EvalResult(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        unlabeled_predictions=unlabeled,
        precision=precision,
        recall=recall,
        f1=f1,
        total_labeled_matches=len(true_matches),
        total_labeled_non_matches=len(true_non_matches),
        total_predictions=total_predictions,
        match_level="pair",
        labeled_coverage=labeled_coverage,
        skipped_unsure=skipped_unsure,
    )


def _evaluate_target_level(predictions: pd.DataFrame, labels: pd.DataFrame) -> EvalResult:
    """Target-level evaluation.

    A match-labeled target is a TP if its target_id appears in any prediction
    (regardless of which reference segment was chosen), and an FN if it
    appears in no prediction. TP/FN therefore count unique labeled targets,
    not pairs.

    False positives remain pair-exact: a no_match label asserts that one
    specific (ref_id, target_id) pair is wrong, so only a prediction of that
    exact pair counts as an FP. Predicting a different reference for the same
    target is not evidence of error.
    """
    true_matches, true_non_matches, predicted_set, skipped_unsure = _prepare_sets(
        predictions, labels
    )

    matched_targets = {t for _, t in true_matches}
    predicted_targets = {t for _, t in predicted_set}

    tp = len(matched_targets & predicted_targets)
    fn = len(matched_targets - predicted_targets)
    fp = len(predicted_set & true_non_matches)

    # A prediction counts toward labeled coverage if it hits a match-labeled
    # target or an explicitly labeled no_match pair.
    unlabeled = sum(
        1
        for pair in predicted_set
        if pair[1] not in matched_targets and pair not in true_non_matches
    )

    precision, recall, f1 = _prf(tp, fp, fn)
    total_predictions = len(predicted_set)
    labeled_coverage = (
        (total_predictions - unlabeled) / total_predictions if total_predictions > 0 else 0.0
    )

    return EvalResult(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        unlabeled_predictions=unlabeled,
        precision=precision,
        recall=recall,
        f1=f1,
        total_labeled_matches=len(matched_targets),
        total_labeled_non_matches=len(true_non_matches),
        total_predictions=total_predictions,
        match_level="target",
        labeled_coverage=labeled_coverage,
        skipped_unsure=skipped_unsure,
    )
