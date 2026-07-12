"""Model evaluation regression test - ensure F1 score meets quality threshold.

This test evaluates the trained model on a held-out test dataset and verifies
that the F1 score is at least 0.90. This prevents model quality regressions.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from crosswalk.config import METRIC_AVERAGE, bundled_model_path, settings
from crosswalk.labeling.label_store import LabelStore
from crosswalk.matching.ml import MLMatcher, segment_aware_split

# Minimum acceptable F1 scores on the held-out test set. The production gate is
# the important one; raw XGBoost classification remains visible for diagnosis.
MIN_PRODUCTION_F1_SCORE = 0.90
MIN_RAW_F1_SCORE = 0.90
MIN_CROSS_PLATFORM_DECISION_AGREEMENT = 0.97
MAX_CROSS_PLATFORM_F1_DELTA = 0.02


@pytest.fixture
def trained_model_path() -> Path:
    """Return path to the trained model.

    The model is trained by the CI workflow before running tests.
    """
    path = Path(__file__).parent.parent.parent / "data" / "models" / "matcher_model_combined.joblib"
    if not path.exists():
        pytest.skip(f"Model not found at {path}. Run 'crosswalk train' first.")
    return path


@pytest.fixture
def labels_dir() -> Path:
    """Return path to labels directory."""
    path = Path(__file__).parent.parent.parent / "labels"
    if not path.exists():
        pytest.skip(f"Labels directory not found at {path}")
    return path


class TestModelEvaluation:
    """Test that model meets F1 score quality threshold on held-out test data."""

    def test_f1_score_meets_threshold(self, trained_model_path, labels_dir):
        """Calibrated deployment predictions should achieve F1 >= 0.90.

        This test loads all labels, splits into train/test using segment-aware
        splitting (same as training), then evaluates both raw XGBoost classes
        and the actual production path: calibrated probabilities thresholded at
        ``settings.scoring_match_threshold``.
        """
        # CI trains one artifact from scratch; the second is the exact artifact
        # shipped to fresh installs. Both must clear the deployment gate and
        # agree on provenance and scores.
        matchers = {
            "ci-trained": MLMatcher(model_path=str(trained_model_path)),
            "bundled": MLMatcher(model_path=str(bundled_model_path())),
        }
        matcher = matchers["ci-trained"]

        # Load all labels
        all_labels = LabelStore.load_all(labels_dir, skip_errors=False)

        if len(all_labels) == 0:
            pytest.skip("No labels found for evaluation")

        # Filter to valid labels (match/no_match)
        valid_labels = {"match", "no_match"}
        df = all_labels[all_labels["label"].isin(valid_labels)].copy()
        matcher._check_feature_versions(df, allow_stale_features=False)
        df = matcher._validate_training_pairs(df, max_hausdorff_m=1000.0)

        if len(df) < 100:
            pytest.skip(f"Not enough labels for evaluation (found {len(df)}, need >= 100)")

        # Segment-aware split (same as training) to get test set
        # Uses same parameters as training: test_size=0.2, random_state=42
        _, test_idx = segment_aware_split(df, test_size=0.2, random_state=42)
        test_df = df.iloc[test_idx].copy()

        # Extract features and labels (same as training evaluation)
        X_test, y_test = matcher._extract_features_and_labels(test_df, binary=True)
        X_test = matcher._cap_infinities(X_test)

        feature_records = test_df.reindex(columns=matcher.feature_names).to_dict("records")
        production_decisions = {}
        production_f1_scores = {}
        for artifact_name, artifact_matcher in matchers.items():
            assert artifact_matcher.feature_names == matcher.feature_names

            # Raw diagnostic: XGBoost's built-in class prediction.
            y_pred_raw = artifact_matcher.model.predict(X_test)
            raw_f1 = f1_score(y_test, y_pred_raw, average=METRIC_AVERAGE)

            # Deployment gate: exercise MLMatcher.predict(), which applies the
            # loaded isotonic calibrator when production calibration is enabled.
            assert artifact_matcher.calibration_active, (
                f"{artifact_name} model regression must exercise calibrated "
                "production predictions; the model has no active calibrator or "
                "calibration was disabled."
            )
            production_probs = artifact_matcher.predict(feature_records)
            y_pred_production = (production_probs >= settings.scoring_match_threshold).astype(int)
            production_f1 = f1_score(y_test, y_pred_production, average=METRIC_AVERAGE)
            production_decisions[artifact_name] = y_pred_production
            production_f1_scores[artifact_name] = production_f1

            assert production_f1 >= MIN_PRODUCTION_F1_SCORE, (
                f"{artifact_name} production F1 {production_f1:.3f} below threshold "
                f"{MIN_PRODUCTION_F1_SCORE} at scoring_match_threshold="
                f"{settings.scoring_match_threshold:.3f} (raw F1={raw_f1:.3f}). "
                "Deployment-path model quality has regressed."
            )
            assert raw_f1 >= MIN_RAW_F1_SCORE, (
                f"{artifact_name} raw XGBoost F1 {raw_f1:.3f} below diagnostic "
                f"threshold {MIN_RAW_F1_SCORE} (production F1={production_f1:.3f})."
            )

        ci_metadata = matchers["ci-trained"].training_metadata or {}
        bundled_metadata = matchers["bundled"].training_metadata or {}
        assert ci_metadata.get("fingerprints") == bundled_metadata.get("fingerprints"), (
            "The bundled model was not trained from the same labeled data and split "
            "as CI's fresh artifact. Retrain and reship the bundled model."
        )
        # XGBoost histogram construction and OOF isotonic knots are not
        # bit-identical across Linux/macOS or Python runtimes even with the same
        # data, split, params, and package versions. Gate the deployment
        # semantics instead of every floating-point probability.
        decision_agreement = np.mean(
            production_decisions["ci-trained"] == production_decisions["bundled"]
        )
        assert decision_agreement >= MIN_CROSS_PLATFORM_DECISION_AGREEMENT, (
            f"Bundled and freshly trained deployment decisions agree on only "
            f"{decision_agreement:.1%} of holdout rows; expected at least "
            f"{MIN_CROSS_PLATFORM_DECISION_AGREEMENT:.1%}."
        )
        f1_delta = abs(production_f1_scores["ci-trained"] - production_f1_scores["bundled"])
        assert f1_delta <= MAX_CROSS_PLATFORM_F1_DELTA, (
            f"Bundled and freshly trained production F1 differ by {f1_delta:.3f}; "
            f"maximum allowed cross-platform delta is {MAX_CROSS_PLATFORM_F1_DELTA:.3f}."
        )
