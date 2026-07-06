"""Model evaluation regression test - ensure F1 score meets quality threshold.

This test evaluates the trained model on a held-out test dataset and verifies
that the F1 score is at least 0.90. This prevents model quality regressions.
"""

from pathlib import Path

import pytest
from sklearn.metrics import f1_score

from crosswalk.config import METRIC_AVERAGE
from crosswalk.labeling.label_store import LabelStore
from crosswalk.matching.ml import MLMatcher, segment_aware_split

# Minimum acceptable F1 score on held-out test set
MIN_TEST_F1_SCORE = 0.90


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
    path = Path(__file__).parent.parent.parent / "labels" / "data"
    if not path.exists():
        pytest.skip(f"Labels directory not found at {path}")
    return path


class TestModelEvaluation:
    """Test that model meets F1 score quality threshold on held-out test data."""

    def test_f1_score_meets_threshold(self, trained_model_path, labels_dir):
        """Model should achieve F1 >= 0.90 on held-out test set.

        This test loads all labels, splits into train/test using segment-aware
        splitting (same as training), and evaluates the model's F1 score on the
        held-out test set.
        """
        # Load trained model
        matcher = MLMatcher()
        matcher.load_model(str(trained_model_path))

        # Load all labels
        all_labels = LabelStore.load_all(labels_dir)

        if len(all_labels) == 0:
            pytest.skip("No labels found for evaluation")

        # Filter to valid labels (match/no_match)
        valid_labels = {"match", "no_match"}
        df = all_labels[all_labels["label"].isin(valid_labels)].copy()

        if len(df) < 100:
            pytest.skip(f"Not enough labels for evaluation (found {len(df)}, need >= 100)")

        # Segment-aware split (same as training) to get test set
        # Uses same parameters as training: test_size=0.2, random_state=42
        _, test_idx = segment_aware_split(df, test_size=0.2, random_state=42)
        test_df = df.iloc[test_idx].copy()

        # Extract features and labels (same as training evaluation)
        X_test, y_test = matcher._extract_features_and_labels(test_df, binary=True)
        X_test = matcher._cap_infinities(X_test)

        # Predict on test set - use model.predict for class labels (0/1)
        y_pred = matcher.model.predict(X_test)

        # Calculate F1 score (binary: positive class only)
        test_f1 = f1_score(y_test, y_pred, average=METRIC_AVERAGE)

        # Assert F1 meets threshold
        assert test_f1 >= MIN_TEST_F1_SCORE, (
            f"Test F1 score {test_f1:.3f} below threshold {MIN_TEST_F1_SCORE}. "
            "Model quality has regressed."
        )
