"""Tests for evaluation metrics."""

import pandas as pd
import pytest

from cbench.eval.metrics import EvalResult, evaluate


def test_evaluate_pair_level(sample_predictions, sample_labels):
    """When predictions exactly match ground truth matches."""
    result = evaluate(sample_predictions, sample_labels, match_level="pair")

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
    result = evaluate(preds, labels, match_level="pair")
    assert result.precision == 0.5
    assert result.recall == 1.0


def test_evaluate_empty_predictions():
    preds = pd.DataFrame({"ref_id": [], "target_id": [], "confidence": []})
    labels = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["match"]})
    result = evaluate(preds, labels, match_level="pair")
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.false_negatives == 1
    assert result.labeled_coverage == 0.0


def test_evaluate_no_labels():
    preds = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame({"ref_id": [], "target_id": [], "label": []})
    result = evaluate(preds, labels, match_level="pair")
    assert result.unlabeled_predictions == 1
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.labeled_coverage == 0.0


def test_evaluate_unsure_labels_skipped():
    """Unsure labels should not affect precision/recall."""
    preds = pd.DataFrame({"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "confidence": [1, 1]})
    labels = pd.DataFrame(
        {"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "label": ["match", "unsure"]}
    )
    result = evaluate(preds, labels, match_level="pair")
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.unlabeled_predictions == 1  # r2-t2 is unsure, treated as unlabeled
    assert result.skipped_unsure == 1


def test_evaluate_default_is_target_level():
    """The default match_level matches the documented default: target."""
    preds = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["match"]})
    result = evaluate(preds, labels)
    assert result.match_level == "target"


def test_evaluate_invalid_match_level():
    preds = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["match"]})
    with pytest.raises(ValueError, match="match_level"):
        evaluate(preds, labels, match_level="bogus")


class TestTargetLevel:
    def test_tp_with_different_ref_segment(self):
        """A match-labeled target predicted with a DIFFERENT ref is still a TP."""
        preds = pd.DataFrame({"ref_id": ["r9"], "target_id": ["t1"], "confidence": [0.9]})
        labels = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["match"]})
        result = evaluate(preds, labels, match_level="target")
        assert result.true_positives == 1
        assert result.false_negatives == 0
        assert result.false_positives == 0
        assert result.recall == 1.0
        # Pair-level would call this a miss
        pair = evaluate(preds, labels, match_level="pair")
        assert pair.true_positives == 0
        assert pair.false_negatives == 1

    def test_fn_when_target_never_predicted(self):
        """A match-labeled target absent from all predictions is an FN."""
        preds = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
        labels = pd.DataFrame(
            {"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "label": ["match", "match"]}
        )
        result = evaluate(preds, labels, match_level="target")
        assert result.true_positives == 1
        assert result.false_negatives == 1
        assert result.recall == 0.5

    def test_fp_requires_exact_no_match_pair(self):
        """FP only when the predicted pair exactly hits a labeled no_match pair."""
        preds = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
        labels = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["no_match"]})
        result = evaluate(preds, labels, match_level="target")
        assert result.false_positives == 1
        assert result.true_positives == 0
        assert result.unlabeled_predictions == 0

    def test_different_ref_for_no_match_target_is_not_fp(self):
        """Predicting a DIFFERENT ref for a no_match-labeled target is not an FP.

        The no_match label is per-pair; it says nothing about other refs.
        """
        preds = pd.DataFrame({"ref_id": ["r9"], "target_id": ["t1"], "confidence": [0.9]})
        labels = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["no_match"]})
        result = evaluate(preds, labels, match_level="target")
        assert result.false_positives == 0
        assert result.true_positives == 0
        assert result.unlabeled_predictions == 1

    def test_tp_counts_unique_targets_not_pairs(self):
        """Multiple predictions for the same match-labeled target count as one TP."""
        preds = pd.DataFrame(
            {"ref_id": ["r1", "r2"], "target_id": ["t1", "t1"], "confidence": [0.9, 0.8]}
        )
        labels = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["match"]})
        result = evaluate(preds, labels, match_level="target")
        assert result.true_positives == 1
        assert result.total_labeled_matches == 1
        assert result.total_predictions == 2
        assert result.unlabeled_predictions == 0

    def test_target_level_fixture(self, sample_predictions, sample_labels):
        """Fixture data: distinct targets, so target level mirrors pair level here."""
        result = evaluate(sample_predictions, sample_labels, match_level="target")
        assert result.match_level == "target"
        assert result.true_positives == 2  # t1, t2
        assert result.false_positives == 1  # r3-t3 exact no_match pair
        assert result.false_negatives == 1  # t5
        assert result.unlabeled_predictions == 1  # r4-t4
        assert result.skipped_unsure == 1  # r6-t6


class TestTransparencyFields:
    def test_labeled_coverage_pair_level(self):
        """coverage = (TP + FP) / total_predictions at pair level."""
        preds = pd.DataFrame(
            {
                "ref_id": ["r1", "r2", "r3", "r4"],
                "target_id": ["t1", "t2", "t3", "t4"],
                "confidence": [1, 1, 1, 1],
            }
        )
        labels = pd.DataFrame(
            {"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "label": ["match", "no_match"]}
        )
        result = evaluate(preds, labels, match_level="pair")
        assert result.unlabeled_predictions == 2
        assert result.labeled_coverage == pytest.approx(0.5)
        assert result.labeled_coverage == pytest.approx(
            (result.true_positives + result.false_positives) / result.total_predictions
        )

    def test_labeled_coverage_target_level(self):
        """Coverage counts predictions touching labeled targets or no_match pairs."""
        preds = pd.DataFrame(
            {
                "ref_id": ["r9", "r2", "r3"],
                "target_id": ["t1", "t2", "t3"],
                "confidence": [1, 1, 1],
            }
        )
        labels = pd.DataFrame(
            {"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "label": ["match", "no_match"]}
        )
        result = evaluate(preds, labels, match_level="target")
        # r9-t1 covers match target t1; r2-t2 hits no_match pair; r3-t3 unlabeled
        assert result.unlabeled_predictions == 1
        assert result.labeled_coverage == pytest.approx(2 / 3)

    def test_fields_in_to_dict(self, sample_predictions, sample_labels):
        d = evaluate(sample_predictions, sample_labels).to_dict()
        assert d["match_level"] == "target"
        assert "labeled_coverage" in d
        assert "unlabeled_predictions" in d
        assert "skipped_unsure" in d


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
        match_level="pair",
        labeled_coverage=0.706,
        skipped_unsure=1,
    )
    d = r.to_dict()
    assert d["f1"] == 0.8
    assert d["true_positives"] == 10
    assert d["match_level"] == "pair"
    assert d["labeled_coverage"] == 0.706
    assert d["skipped_unsure"] == 1
    assert len(d) == 13
