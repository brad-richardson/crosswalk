"""Tests for evaluation metrics."""

import pandas as pd

from cbench.eval.metrics import EvalResult, evaluate


def test_evaluate_perfect_predictions(sample_predictions, sample_labels):
    """When predictions exactly match ground truth matches."""
    # Predictions: r1-t1, r2-t2 are matches; r3-t3 is no_match; r4-t4 unlabeled
    result = evaluate(sample_predictions, sample_labels)

    assert result.true_positives == 2  # r1-t1, r2-t2
    assert result.false_positives == 1  # r3-t3 (labeled no_match)
    assert result.false_negatives == 1  # r5-t5 (match not predicted)
    assert result.unlabeled_predictions == 1  # r4-t4
    assert result.total_labeled_matches == 3  # r1-t1, r2-t2, r5-t5
    assert result.total_labeled_non_matches == 1  # r3-t3
    assert result.total_predictions == 4


def test_evaluate_precision():
    preds = pd.DataFrame({"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "confidence": [1, 1]})
    labels = pd.DataFrame(
        {"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "label": ["match", "no_match"]}
    )
    result = evaluate(preds, labels)
    assert result.precision == 0.5
    assert result.recall == 1.0


def test_evaluate_empty_predictions():
    preds = pd.DataFrame({"ref_id": [], "target_id": [], "confidence": []})
    labels = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["match"]})
    result = evaluate(preds, labels)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.false_negatives == 1


def test_evaluate_no_labels():
    preds = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame({"ref_id": [], "target_id": [], "label": []})
    result = evaluate(preds, labels)
    assert result.unlabeled_predictions == 1
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_evaluate_unsure_labels_skipped():
    """Unsure labels should not affect precision/recall."""
    preds = pd.DataFrame({"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "confidence": [1, 1]})
    labels = pd.DataFrame(
        {"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "label": ["match", "unsure"]}
    )
    result = evaluate(preds, labels)
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.unlabeled_predictions == 1  # r2-t2 is unsure, treated as unlabeled


def test_eval_result_to_dict():
    r = EvalResult(
        true_positives=10,
        false_positives=2,
        false_negatives=3,
        unlabeled_predictions=5,
        precision=0.833,
        recall=0.769,
        f1=0.8,
        total_labeled_matches=13,
        total_labeled_non_matches=2,
        total_predictions=17,
    )
    d = r.to_dict()
    assert d["f1"] == 0.8
    assert d["true_positives"] == 10
    assert len(d) == 10
