"""Model evaluation regression test - ensure F1 score meets quality threshold.

This test evaluates the trained model on a held-out test dataset and verifies
that the F1 score is at least 0.85. This prevents model quality regressions.
"""

from pathlib import Path

import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from matcher.config import FEATURE_COLUMNS
from matcher.labeling.label_store import LabelStore
from matcher.matching.ml import MLMatcher

# Minimum acceptable F1 score on held-out test set
MIN_TEST_F1_SCORE = 0.85


@pytest.fixture
def trained_model_path() -> Path:
    """Return path to the trained model.

    The model is trained by the CI workflow before running tests.
    """
    path = Path(__file__).parent.parent.parent / "data" / "models" / "matcher_model_combined.joblib"
    if not path.exists():
        pytest.skip(f"Model not found at {path}. Run 'matcher train' first.")
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
        """Model should achieve F1 >= 0.85 on held-out test set.

        This test loads all labels, splits into train/test, and evaluates
        the model's F1 score on the held-out test set.
        """
        # Load trained model
        matcher = MLMatcher(str(trained_model_path))

        # Load all labels
        all_labels = LabelStore.load_all(labels_dir)

        if len(all_labels) == 0:
            pytest.skip("No labels found for evaluation")

        # Filter to binary classification (match vs no_match)
        # Binary: 0 = no_match, 1 = match
        df = all_labels[all_labels["label"].isin([0, 1])].copy()

        if len(df) < 100:
            pytest.skip(f"Not enough labels for evaluation (found {len(df)}, need >= 100)")

        # Split into train/test (same seed and split as training for consistency)
        X = df[FEATURE_COLUMNS]
        y = df["label"]

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # Predict on test set
        y_pred = matcher.predict(X_test)

        # Calculate F1 score (weighted average for multiclass, binary here)
        test_f1 = f1_score(y_test, y_pred, average="weighted")

        # Assert F1 meets threshold
        assert test_f1 >= MIN_TEST_F1_SCORE, (
            f"Test F1 score {test_f1:.3f} below threshold {MIN_TEST_F1_SCORE}. "
            "Model quality has regressed."
        )
