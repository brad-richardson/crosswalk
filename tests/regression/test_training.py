"""Training regression test - verify model quality hasn't degraded."""

from pathlib import Path

import pytest

from matcher.matching.ml import train_model

# Minimum acceptable thresholds
MIN_TEST_ACCURACY = 0.85
MIN_CV_F1_MEAN = 0.80


class TestTrainingRegression:
    """Regression tests for ML model training."""

    @pytest.fixture
    def labels_dir(self):
        """Path to labels directory."""
        return Path(__file__).parent.parent.parent / "labels"

    def test_training_meets_accuracy_threshold(self, labels_dir, tmp_path):
        """Model trained on Boston data should meet accuracy threshold."""
        results = train_model(
            labels_dir=str(labels_dir),
            output_path=str(tmp_path / "test_model.joblib"),
        )

        assert results["test_accuracy"] >= MIN_TEST_ACCURACY, (
            f"Test accuracy {results['test_accuracy']:.3f} below threshold {MIN_TEST_ACCURACY}"
        )

    def test_training_meets_f1_threshold(self, labels_dir, tmp_path):
        """Model trained on Boston data should meet F1 threshold."""
        results = train_model(
            labels_dir=str(labels_dir),
            output_path=str(tmp_path / "test_model.joblib"),
        )

        assert results["cv_f1_mean"] >= MIN_CV_F1_MEAN, (
            f"CV F1 mean {results['cv_f1_mean']:.3f} below threshold {MIN_CV_F1_MEAN}"
        )
