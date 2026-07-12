"""Integration test for export-spark-model command.

Trains, exports, and verifies the Spark-portable XGBoost model can be loaded
and used for inference as it would be in the tf-data-platform Spark job.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from crosswalk.config import SPARK_PORTABLE_FEATURES, SPARK_PORTABLE_XGB_PARAMS


@pytest.fixture
def exported_model(tmp_path):
    """Train and export a Spark-portable model to a temp directory."""
    from crosswalk.config import FEATURE_COLUMNS
    from crosswalk.matching.ml import MLMatcher
    from crosswalk.model_export import build_spark_model_manifest

    labels_dir = Path("labels")
    if not labels_dir.exists():
        pytest.skip("Labels directory not found — run from repo root")

    exclude_features = [f for f in FEATURE_COLUMNS if f not in SPARK_PORTABLE_FEATURES]

    matcher = MLMatcher()
    matcher.train(
        labels_dir=labels_dir,
        test_size=0.2,
        binary=True,
        exclude_features=exclude_features,
        **SPARK_PORTABLE_XGB_PARAMS,
    )

    # Export
    model_path = tmp_path / "model.json"
    matcher.model.get_booster().save_model(str(model_path))

    manifest = build_spark_model_manifest(matcher)
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return model_path, manifest_path, matcher


class TestExportSparkModel:
    """Test the full export-spark-model pipeline."""

    def test_exported_model_loads_as_booster(self, exported_model):
        """model.json should be loadable as a raw XGBoost Booster."""
        model_path, _, _ = exported_model
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        # Should have the right number of features
        assert booster.num_features() == len(SPARK_PORTABLE_FEATURES)

    def test_manifest_has_correct_features(self, exported_model):
        """manifest.json should list exactly the Spark-portable features."""
        _, manifest_path, _ = exported_model
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert set(manifest["features"]) == set(SPARK_PORTABLE_FEATURES)
        assert manifest["n_features"] == len(SPARK_PORTABLE_FEATURES)
        assert "source_commit" in manifest["training_metadata"]

    def test_booster_predicts_with_feature_names(self, exported_model):
        """Booster should accept DMatrix with feature names and return probabilities."""
        model_path, manifest_path, _ = exported_model
        with open(manifest_path) as f:
            features = json.load(f)["features"]

        booster = xgb.Booster()
        booster.load_model(str(model_path))

        # Simulate a batch of 10 candidate pairs
        X = np.random.rand(10, len(features)).astype(np.float32)
        dmat = xgb.DMatrix(X, feature_names=features)
        preds = booster.predict(dmat)

        assert preds.shape == (10,)
        assert all(0 <= p <= 1 for p in preds), "Predictions should be probabilities"

    def test_booster_handles_nan_features(self, exported_model):
        """XGBoost should handle NaN features natively (missing names, etc.)."""
        model_path, manifest_path, _ = exported_model
        with open(manifest_path) as f:
            features = json.load(f)["features"]

        booster = xgb.Booster()
        booster.load_model(str(model_path))

        # All NaN — simulates a pair with no computable features
        X = np.full((5, len(features)), np.nan, dtype=np.float32)
        dmat = xgb.DMatrix(X, feature_names=features)
        preds = booster.predict(dmat)

        assert preds.shape == (5,)
        assert all(np.isfinite(p) for p in preds), "Should predict even with all NaN"

    def test_booster_matches_sklearn_predictions(self, exported_model):
        """Exported booster should produce same predictions as the sklearn model."""
        model_path, manifest_path, matcher = exported_model
        with open(manifest_path) as f:
            features = json.load(f)["features"]

        booster = xgb.Booster()
        booster.load_model(str(model_path))

        X = np.random.rand(20, len(features)).astype(np.float32)

        # Sklearn prediction
        sklearn_proba = matcher.model.predict_proba(X)[:, 1]

        # Booster prediction
        dmat = xgb.DMatrix(X, feature_names=features)
        booster_proba = booster.predict(dmat)

        np.testing.assert_allclose(
            sklearn_proba,
            booster_proba,
            atol=1e-5,
            err_msg="Booster predictions should match sklearn model",
        )

    def test_model_uses_tuned_hyperparams(self, exported_model):
        """Model should use the Spark-portable tuned hyperparams, not defaults."""
        _, _, matcher = exported_model
        params = matcher.model.get_params()
        assert params["n_estimators"] == SPARK_PORTABLE_XGB_PARAMS["n_estimators"]
        assert params["max_depth"] == SPARK_PORTABLE_XGB_PARAMS["max_depth"]
        assert abs(params["learning_rate"] - SPARK_PORTABLE_XGB_PARAMS["learning_rate"]) < 1e-10
