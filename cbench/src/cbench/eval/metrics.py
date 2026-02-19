"""Evaluation metrics for conflation benchmarking.

Uses exact (ref_id, target_id) pair matching. A prediction is a TP only
if the exact pair appears in the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
        }


def evaluate(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> EvalResult:
    """Compute precision/recall/F1 of predictions vs ground truth labels.

    Uses exact (ref_id, target_id) pair matching.

    Args:
        predictions: DataFrame with columns [ref_id, target_id, confidence].
        labels: DataFrame with columns [ref_id, target_id, label].
                label values: "match", "no_match", "unsure" (skipped).

    Returns:
        EvalResult with all metrics.
    """
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
    )
