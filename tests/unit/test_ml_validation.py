"""Tests for training-time feature validation in MLMatcher.

Validates that _validate_training_pairs correctly detects and removes
pairs with implausible feature values (corrupted geometry lookups, stale backfills).
"""

import io

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from matcher.config import FEATURE_COLUMNS
from matcher.matching.ml import MLMatcher


@pytest.fixture
def matcher():
    """Create an MLMatcher instance with feature_names set."""
    m = MLMatcher.__new__(MLMatcher)
    m.feature_names = FEATURE_COLUMNS.copy()
    return m


@pytest.fixture
def log_capture():
    """Capture loguru output for assertion."""
    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}", level="DEBUG")
    yield sink
    logger.remove(handler_id)


def _make_pair(
    dataset: str = "test_dataset",
    centroid_distance_m: float = 20.0,
    hausdorff_distance_m: float = 15.0,
    buffer_overlap_iou: float = 0.5,
    label: str = "match",
    **overrides,
) -> dict:
    """Create a synthetic training pair with plausible defaults."""
    row = {
        "gers_id": "test_gers_1",
        "target_id": "test_target_1",
        "dataset": dataset,
        "label": label,
        "centroid_distance_m": centroid_distance_m,
        "hausdorff_distance_m": hausdorff_distance_m,
        "buffer_overlap_iou": buffer_overlap_iou,
    }
    # Fill remaining feature columns with reasonable defaults
    for col in FEATURE_COLUMNS:
        if col not in row:
            row[col] = 0.5
    row.update(overrides)
    return row


def _make_df(pairs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(pairs)


class TestValidateTrainingPairs:
    """Unit tests for _validate_training_pairs."""

    def test_normal_pair_passes(self, matcher):
        """A pair with plausible features is kept."""
        df = _make_df([_make_pair(centroid_distance_m=20.0, hausdorff_distance_m=15.0)])
        result = matcher._validate_training_pairs(df)
        assert len(result) == 1

    def test_distant_centroid_rejected(self, matcher):
        """A pair with centroid_distance_m > 500m is dropped."""
        df = _make_df([_make_pair(centroid_distance_m=5000.0)])
        result = matcher._validate_training_pairs(df)
        assert len(result) == 0

    def test_high_hausdorff_rejected(self, matcher):
        """A pair with hausdorff_distance_m > 1000m is dropped."""
        df = _make_df([_make_pair(hausdorff_distance_m=15000.0)])
        result = matcher._validate_training_pairs(df)
        assert len(result) == 0

    def test_all_nan_features_rejected(self, matcher):
        """A pair where every feature column is NaN is dropped."""
        pair = _make_pair()
        for col in FEATURE_COLUMNS:
            pair[col] = np.nan
        df = _make_df([pair])
        result = matcher._validate_training_pairs(df)
        assert len(result) == 0

    def test_nan_centroid_passes(self, matcher):
        """A pair with NaN centroid_distance_m is kept (let imputation handle it)."""
        df = _make_df([_make_pair(centroid_distance_m=np.nan)])
        result = matcher._validate_training_pairs(df)
        assert len(result) == 1

    def test_nan_hausdorff_passes(self, matcher):
        """A pair with NaN hausdorff_distance_m is kept."""
        df = _make_df([_make_pair(hausdorff_distance_m=np.nan)])
        result = matcher._validate_training_pairs(df)
        assert len(result) == 1

    def test_mixed_dataset_logging(self, matcher, log_capture):
        """Verify per-dataset logging counts for mixed valid/invalid pairs."""
        pairs = [
            _make_pair(dataset="good_dataset", centroid_distance_m=10.0),
            _make_pair(dataset="good_dataset", centroid_distance_m=20.0),
            _make_pair(dataset="bad_dataset", centroid_distance_m=5000.0),
            _make_pair(dataset="bad_dataset", centroid_distance_m=6000.0),
            _make_pair(dataset="bad_dataset", centroid_distance_m=15.0),
        ]
        df = _make_df(pairs)
        result = matcher._validate_training_pairs(df)

        assert len(result) == 3
        log_text = log_capture.getvalue()
        assert "bad_dataset: dropped 2/3 pairs" in log_text
        assert "good_dataset" not in log_text  # no drops from good_dataset

    def test_warning_threshold_triggered(self, matcher, log_capture):
        """>20% drop rate for a dataset triggers a warning."""
        pairs = [
            _make_pair(dataset="problematic", centroid_distance_m=5000.0),
            _make_pair(dataset="problematic", centroid_distance_m=6000.0),
            _make_pair(dataset="problematic", centroid_distance_m=15.0),
        ]
        df = _make_df(pairs)
        matcher._validate_training_pairs(df)

        log_text = log_capture.getvalue()
        assert "possible systematic data issue" in log_text

    def test_custom_thresholds(self, matcher):
        """Custom thresholds override defaults."""
        df = _make_df([_make_pair(centroid_distance_m=200.0, hausdorff_distance_m=600.0)])
        # With defaults (500m/1000m), this passes
        result = matcher._validate_training_pairs(df)
        assert len(result) == 1
        # With stricter thresholds, it fails
        result = matcher._validate_training_pairs(
            df, max_centroid_distance_m=100.0, max_hausdorff_m=500.0
        )
        assert len(result) == 0

    def test_boundary_values(self, matcher):
        """Values exactly at thresholds are kept."""
        df = _make_df([_make_pair(centroid_distance_m=500.0, hausdorff_distance_m=1000.0)])
        result = matcher._validate_training_pairs(df)
        assert len(result) == 1

    def test_multiple_reasons_single_drop(self, matcher):
        """A pair failing multiple checks is only counted once."""
        df = _make_df([_make_pair(centroid_distance_m=5000.0, hausdorff_distance_m=15000.0)])
        result = matcher._validate_training_pairs(df)
        assert len(result) == 0

    def test_empty_dataframe(self, matcher):
        """Empty DataFrame passes through without error."""
        df = _make_df([])
        # Ensure columns exist even with empty df
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = pd.Series(dtype=float)
        result = matcher._validate_training_pairs(df)
        assert len(result) == 0

    def test_all_pairs_valid(self, matcher, log_capture):
        """When all pairs are valid, logs success message."""
        df = _make_df(
            [
                _make_pair(centroid_distance_m=10.0, hausdorff_distance_m=5.0),
                _make_pair(centroid_distance_m=30.0, hausdorff_distance_m=20.0),
            ]
        )
        result = matcher._validate_training_pairs(df)
        assert len(result) == 2
        log_text = log_capture.getvalue()
        assert "all training pairs passed plausibility checks" in log_text
