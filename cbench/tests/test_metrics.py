"""Tests for evaluation metrics."""

import pandas as pd

from cbench.eval.metrics import EvalResult, MatchLevel, evaluate

# ---------------------------------------------------------------------------
# Pair-level evaluation (default legacy behavior)
# ---------------------------------------------------------------------------


def test_evaluate_pair_level(sample_predictions, sample_labels):
    """When predictions exactly match ground truth matches."""
    result = evaluate(sample_predictions, sample_labels, match_level=MatchLevel.PAIR)

    assert result.true_positives == 2  # r1-t1, r2-t2
    assert result.false_positives == 1  # r3-t3 (labeled no_match)
    assert result.false_negatives == 1  # r5-t5 (match not predicted)
    assert result.unlabeled_predictions == 1  # r4-t4
    assert result.total_labeled_matches == 3  # r1-t1, r2-t2, r5-t5
    assert result.total_labeled_non_matches == 1  # r3-t3
    assert result.total_predictions == 4
    assert result.match_level == "pair"


def test_evaluate_pair_precision():
    preds = pd.DataFrame({"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "confidence": [1, 1]})
    labels = pd.DataFrame(
        {"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "label": ["match", "no_match"]}
    )
    result = evaluate(preds, labels, match_level=MatchLevel.PAIR)
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


# ---------------------------------------------------------------------------
# Target-level evaluation
# ---------------------------------------------------------------------------


def test_evaluate_target_level_different_ref():
    """Target matched to different ref than label should still be TP."""
    preds = pd.DataFrame({"ref_id": ["r_other"], "target_id": ["t1"], "confidence": [0.99]})
    labels = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["match"]})
    # Pair-level: FN (r1-t1 not predicted) + unlabeled (r_other-t1)
    pair_result = evaluate(preds, labels, match_level=MatchLevel.PAIR)
    assert pair_result.true_positives == 0
    assert pair_result.false_negatives == 1

    # Target-level: TP (t1 is predicted and labeled match)
    target_result = evaluate(preds, labels, match_level=MatchLevel.TARGET)
    assert target_result.true_positives == 1
    assert target_result.false_negatives == 0
    assert target_result.match_level == "target"


def test_evaluate_target_level_multiple_refs_one_target():
    """Target labeled as matching multiple refs, prediction picks one."""
    preds = pd.DataFrame({"ref_id": ["r2"], "target_id": ["t1"], "confidence": [0.95]})
    labels = pd.DataFrame(
        {
            "ref_id": ["r1", "r2", "r3"],
            "target_id": ["t1", "t1", "t1"],
            "label": ["match", "match", "match"],
        }
    )
    # Pair-level: 1 TP (r2-t1), 2 FN (r1-t1, r3-t1)
    pair_result = evaluate(preds, labels, match_level=MatchLevel.PAIR)
    assert pair_result.true_positives == 1
    assert pair_result.false_negatives == 2

    # Target-level: 1 TP (t1 is matched), 0 FN
    target_result = evaluate(preds, labels, match_level=MatchLevel.TARGET)
    assert target_result.true_positives == 1
    assert target_result.false_negatives == 0
    assert target_result.total_labeled_matches == 1  # 1 unique target


def test_evaluate_target_level_no_match_target():
    """Target with only no_match labels should be FP if predicted."""
    preds = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["no_match"]})
    result = evaluate(preds, labels, match_level=MatchLevel.TARGET)
    assert result.false_positives == 1
    assert result.true_positives == 0


def test_evaluate_target_level_mixed_labels():
    """Target with both match and no_match labels should count as match target."""
    preds = pd.DataFrame({"ref_id": ["r_new"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame(
        {
            "ref_id": ["r1", "r2"],
            "target_id": ["t1", "t1"],
            "label": ["match", "no_match"],
        }
    )
    # t1 has at least one match label, so it's a match target
    result = evaluate(preds, labels, match_level=MatchLevel.TARGET)
    assert result.true_positives == 1
    assert result.false_positives == 0


def test_evaluate_target_level_fn():
    """Target with match label but not in predictions is FN."""
    preds = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame(
        {
            "ref_id": ["r1", "r2"],
            "target_id": ["t1", "t2"],
            "label": ["match", "match"],
        }
    )
    result = evaluate(preds, labels, match_level=MatchLevel.TARGET)
    assert result.true_positives == 1  # t1
    assert result.false_negatives == 1  # t2 not predicted


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


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
    assert d["match_level"] == "pair"
    assert len(d) == 11
