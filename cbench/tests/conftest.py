"""Shared test fixtures for cbench."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_predictions() -> pd.DataFrame:
    """Sample predictions DataFrame."""
    return pd.DataFrame(
        {
            "ref_id": ["r1", "r2", "r3", "r4"],
            "target_id": ["t1", "t2", "t3", "t4"],
            "confidence": [0.95, 0.80, 0.70, 0.60],
        }
    )


@pytest.fixture
def sample_labels() -> pd.DataFrame:
    """Sample ground truth labels."""
    return pd.DataFrame(
        {
            "ref_id": ["r1", "r2", "r3", "r5", "r6"],
            "target_id": ["t1", "t2", "t3", "t5", "t6"],
            "label": ["match", "match", "no_match", "match", "unsure"],
        }
    )


@pytest.fixture
def labels_dir(tmp_path: Path) -> Path:
    """Create a temporary labels directory with test data."""
    dataset_dir = tmp_path / "dataset=test_city"
    dataset_dir.mkdir()
    labels = pd.DataFrame(
        {
            "gers_id": ["r1", "r2", "r3"],
            "target_id": ["t1", "t2", "t3"],
            "label": ["match", "no_match", "match"],
        }
    )
    labels.to_csv(dataset_dir / "data.csv", index=False)
    return tmp_path
