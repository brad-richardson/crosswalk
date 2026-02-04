"""Training regression test - verify model quality hasn't degraded."""

from matcher.matching.ml import train_model

# Minimum acceptable thresholds
MIN_TEST_ACCURACY = 0.88
MIN_CV_F1_MEAN = 0.88


class TestTrainingRegression:
    """Regression tests for ML model training."""

    def test_training_meets_quality_thresholds(self, labels_dir, tmp_path):
        """Model trained on Boston data should meet accuracy and F1 thresholds.

        Trains the model once and verifies both metrics to avoid redundant
        expensive training operations.
        """
        results = train_model(
            labels_dir=str(labels_dir),
            output_path=str(tmp_path / "test_model.joblib"),
        )

        # Check accuracy threshold
        assert results["test_accuracy"] >= MIN_TEST_ACCURACY, (
            f"Test accuracy {results['test_accuracy']:.3f} below threshold {MIN_TEST_ACCURACY}"
        )

        # Check F1 threshold
        assert results["cv_f1_mean"] >= MIN_CV_F1_MEAN, (
            f"CV F1 mean {results['cv_f1_mean']:.3f} below threshold {MIN_CV_F1_MEAN}"
        )
