"""Evaluation metrics for conflation benchmarking.

Two evaluation modes:

- **pair**: Exact (ref_id, target_id) pair matching. A prediction is a TP only
  if the exact pair appears in the ground truth. This penalizes the matcher when
  it matches a target to a different (but equally valid) reference segment.

- **target**: Target-level matching. A labeled match target is a TP if the target
  appears in *any* prediction, regardless of which ref_id was chosen. This avoids
  penalizing tools for picking a different reference segment that covers a different
  subsegment of the same target road.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class MatchLevel(Enum):
    """Level at which to evaluate match predictions."""

    PAIR = "pair"
    TARGET = "target"


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
        }


def evaluate(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    match_level: MatchLevel = MatchLevel.PAIR,
) -> EvalResult:
    """Compute precision/recall/F1 of predictions vs ground truth labels.

    Args:
        predictions: DataFrame with columns [ref_id, target_id, confidence].
        labels: DataFrame with columns [ref_id, target_id, label].
                label values: "match", "no_match", "unsure" (skipped).
        match_level: PAIR for exact pair matching, TARGET for target-level matching.

    Returns:
        EvalResult with all metrics.
    """
    if match_level == MatchLevel.TARGET:
        return _evaluate_target_level(predictions, labels)
    return _evaluate_pair_level(predictions, labels)


def _evaluate_pair_level(predictions: pd.DataFrame, labels: pd.DataFrame) -> EvalResult:
    """Exact (ref_id, target_id) pair evaluation."""
    labels_str = labels.assign(
        ref_id=labels["ref_id"].astype(str), target_id=labels["target_id"].astype(str)
    )
    match_mask = labels_str["label"] == "match"
    no_match_mask = labels_str["label"] == "no_match"

    true_matches = set(
        zip(labels_str.loc[match_mask, "ref_id"], labels_str.loc[match_mask, "target_id"])
    )
    true_non_matches = set(
        zip(labels_str.loc[no_match_mask, "ref_id"], labels_str.loc[no_match_mask, "target_id"])
    )

    predicted_set = set(
        zip(predictions["ref_id"].astype(str), predictions["target_id"].astype(str))
    )

    tp = len(predicted_set & true_matches)
    fp = len(predicted_set & true_non_matches)
    fn = len(true_matches - predicted_set)
    unlabeled = len(predicted_set - true_matches - true_non_matches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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
        total_predictions=len(predicted_set),
        match_level="pair",
    )


def _evaluate_target_level(predictions: pd.DataFrame, labels: pd.DataFrame) -> EvalResult:
    """Target-level evaluation: did the tool match the target to *any* ref?

    A labeled match target counts as TP if the target_id appears in any prediction.
    A labeled no_match target counts as FP if the target_id appears in any prediction.
    This avoids penalizing tools for picking a different ref segment that covers a
    different subsegment of the same target road.
    """
    labels_str = labels.assign(
        ref_id=labels["ref_id"].astype(str), target_id=labels["target_id"].astype(str)
    )
    match_mask = labels_str["label"] == "match"
    no_match_mask = labels_str["label"] == "no_match"

    # Targets that should be matched (at least one "match" label)
    match_targets = set(labels_str.loc[match_mask, "target_id"])
    # Targets that should NOT be matched (only "no_match" labels, no "match" labels)
    no_match_only_targets = set(labels_str.loc[no_match_mask, "target_id"]) - match_targets

    # Targets that were predicted
    predicted_targets = set(predictions["target_id"].astype(str))

    tp = len(predicted_targets & match_targets)
    fp = len(predicted_targets & no_match_only_targets)
    fn = len(match_targets - predicted_targets)
    unlabeled = len(predicted_targets - match_targets - no_match_only_targets)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvalResult(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        unlabeled_predictions=unlabeled,
        precision=precision,
        recall=recall,
        f1=f1,
        total_labeled_matches=len(match_targets),
        total_labeled_non_matches=len(no_match_only_targets),
        total_predictions=len(predicted_targets),
        match_level="target",
    )
