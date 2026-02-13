"""Tests for feature version tracking in ML models."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from loguru import logger
from sklearn.tree import DecisionTreeClassifier

from matcher.config import FEATURE_COLUMNS, FEATURE_VERSION
from matcher.matching.ml import MLMatcher


def _make_simple_model():
    """Create a simple picklable model for testing."""
    rng = np.random.default_rng(42)
    clf = DecisionTreeClassifier(random_state=42)
    X = rng.random((20, len(FEATURE_COLUMNS)))
    y = np.array([0] * 10 + [1] * 10)
    clf.fit(X, y)
    return clf


def _save_model_dict(path, extra=None):
    """Save a model dict to disk with optional extra keys."""
    data = {
        "model": _make_simple_model(),
        "feature_names": FEATURE_COLUMNS.copy(),
        "label_encoder": {"match": 1, "no_match": 0, "associated": 2},
        "label_decoder": {1: "match", 0: "no_match", 2: "associated"},
        "is_binary": True,
    }
    if extra:
        data.update(extra)
    joblib.dump(data, path)
    return data


@pytest.fixture
def log_capture():
    """Capture loguru output for assertion."""
    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}", level="WARNING")
    yield sink
    logger.remove(handler_id)


class TestSaveModelIncludesFeatureVersion:
    def test_save_model_includes_feature_version(self, tmp_path):
        """Saved model should contain feature_version key."""
        matcher = MLMatcher()
        matcher.model = _make_simple_model()
        matcher.feature_version = FEATURE_VERSION

        model_path = tmp_path / "model.joblib"
        matcher.save_model(str(model_path))

        data = joblib.load(model_path)
        assert "feature_version" in data
        assert data["feature_version"] == FEATURE_VERSION


class TestLoadModelVersionWarnings:
    def test_load_old_model_without_version_warns(self, tmp_path, log_capture):
        """Loading a model without feature_version should warn."""
        model_path = tmp_path / "old_model.joblib"
        _save_model_dict(model_path)  # No feature_version key

        matcher = MLMatcher()
        matcher.load_model(str(model_path))

        output = log_capture.getvalue()
        assert "pre-versioning" in output.lower()
        assert matcher.feature_version is None

    def test_load_mismatched_version_warns(self, tmp_path, log_capture):
        """Loading a model with stale feature_version should warn."""
        model_path = tmp_path / "stale_model.joblib"
        stale_version = "2020-01-01"
        _save_model_dict(model_path, extra={"feature_version": stale_version})

        matcher = MLMatcher()
        matcher.load_model(str(model_path))

        output = log_capture.getvalue()
        assert "does not match" in output
        assert stale_version in output
        assert matcher.feature_version == stale_version

    def test_load_matching_version_no_warning(self, tmp_path, log_capture):
        """Loading a model with matching feature_version should not warn."""
        model_path = tmp_path / "current_model.joblib"
        _save_model_dict(model_path, extra={"feature_version": FEATURE_VERSION})

        matcher = MLMatcher()
        matcher.load_model(str(model_path))

        output = log_capture.getvalue()
        assert "does not match" not in output
        assert "pre-versioning" not in output
        assert matcher.feature_version == FEATURE_VERSION


class TestTrainVersionChecks:
    def _make_labels(self, labels_dir, feature_versions):
        """Create label files in normalized format for testing."""
        n_samples = len(feature_versions)
        rng = np.random.default_rng(123)

        # Create human labels in labels/human/dataset=test_ds/data.csv
        human_dir = labels_dir / "human" / "dataset=test_ds"
        human_dir.mkdir(parents=True)
        human_data = {
            "gers_id": [f"ref_{i}" for i in range(n_samples)],
            "target_id": [f"target_{i}" for i in range(n_samples)],
            "label": ["match"] * (n_samples // 2) + ["no_match"] * (n_samples - n_samples // 2),
            "labeler": ["test"] * n_samples,
            "labeled_at": ["2026-01-01T00:00:00"] * n_samples,
            "session_id": ["test"] * n_samples,
        }
        pd.DataFrame(human_data).to_csv(human_dir / "data.csv", index=False)

        # Create features in labels/features/dataset=test_ds/data.parquet
        features_dir = labels_dir / "features" / "dataset=test_ds"
        features_dir.mkdir(parents=True)
        features_data = {
            "gers_id": [f"ref_{i}" for i in range(n_samples)],
            "target_id": [f"target_{i}" for i in range(n_samples)],
            "feature_version": feature_versions,
        }
        for col in FEATURE_COLUMNS:
            features_data[col] = rng.random(n_samples).tolist()

        pd.DataFrame(features_data).to_parquet(features_dir / "data.parquet", index=False)

    def test_train_sets_feature_version(self, tmp_path):
        """Training should set feature_version to current FEATURE_VERSION."""
        labels_dir = tmp_path / "labels"
        self._make_labels(labels_dir, [FEATURE_VERSION] * 20)

        matcher = MLMatcher()
        try:
            matcher.train(labels_dir=str(labels_dir), test_size=0.0)
        except ImportError:
            pytest.skip("XGBoost not installed")

        assert matcher.feature_version == FEATURE_VERSION

    def test_train_mixed_versions_warns(self, tmp_path, log_capture):
        """Training on labels with mixed feature_versions should warn."""
        labels_dir = tmp_path / "labels"
        versions = [FEATURE_VERSION] * 10 + ["2020-01-01"] * 10
        self._make_labels(labels_dir, versions)

        matcher = MLMatcher()
        try:
            matcher.train(labels_dir=str(labels_dir), test_size=0.0)
        except ImportError:
            pytest.skip("XGBoost not installed")

        output = log_capture.getvalue()
        assert "different feature versions" in output
