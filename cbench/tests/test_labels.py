"""Tests for label loading."""

import json

import pandas as pd
import pytest

from cbench.eval.labels import list_datasets, load_labels, load_stitch_labels


def test_load_labels_csv(labels_dir):
    """Load labels from CSV file with column renaming."""
    df = load_labels(labels_dir, "test_city")
    assert len(df) == 3
    assert list(df.columns) == ["ref_id", "target_id", "label"]
    assert df["ref_id"].iloc[0] == "r1"


def test_load_labels_parquet(tmp_path):
    """Load labels from parquet file."""
    dataset_dir = tmp_path / "dataset=parquet_city"
    dataset_dir.mkdir()
    labels = pd.DataFrame(
        {
            "gers_id": ["r1", "r2"],
            "target_id": ["t1", "t2"],
            "label": ["match", "no_match"],
        }
    )
    labels.to_parquet(dataset_dir / "data.parquet", index=False)

    df = load_labels(tmp_path, "parquet_city")
    assert len(df) == 2
    assert "ref_id" in df.columns


def test_load_labels_custom_columns(tmp_path):
    """Load labels with custom column names."""
    dataset_dir = tmp_path / "dataset=custom"
    dataset_dir.mkdir()
    labels = pd.DataFrame(
        {
            "source_id": ["r1"],
            "dest_id": ["t1"],
            "label": ["match"],
        }
    )
    labels.to_csv(dataset_dir / "data.csv", index=False)

    df = load_labels(tmp_path, "custom", ref_id_column="source_id", target_id_column="dest_id")
    assert df["ref_id"].iloc[0] == "r1"
    assert df["target_id"].iloc[0] == "t1"


def test_load_labels_missing_dataset(labels_dir):
    with pytest.raises(FileNotFoundError, match="No labels found"):
        load_labels(labels_dir, "nonexistent")


def test_load_labels_missing_columns(tmp_path):
    dataset_dir = tmp_path / "dataset=bad"
    dataset_dir.mkdir()
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(dataset_dir / "data.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_labels(tmp_path, "bad")


def test_list_datasets(labels_dir):
    datasets = list_datasets(labels_dir)
    assert "test_city" in datasets
    assert datasets["test_city"] == 3


def test_list_datasets_empty(tmp_path):
    datasets = list_datasets(tmp_path)
    assert datasets == {}


def test_list_datasets_nonexistent(tmp_path):
    datasets = list_datasets(tmp_path / "nonexistent")
    assert datasets == {}


def test_load_stitch_labels(tmp_path):
    """Load stitching labels from hive-partitioned CSV."""
    dataset_dir = tmp_path / "dataset=test_city"
    dataset_dir.mkdir()
    labels = pd.DataFrame(
        {
            "group_id": ["abc123"],
            "dataset_id": ["test_city"],
            "selected_edges": [json.dumps([{"ref_id": "r1", "target_id": "t1"}])],
            "match_type": ["1:1"],
            "num_refs": [1],
            "num_targets": [1],
            "labeler": ["test"],
            "labeled_at": ["2026-02-21T00:00:00Z"],
            "session_id": ["s1"],
        }
    )
    labels.to_csv(dataset_dir / "data.csv", index=False)

    result = load_stitch_labels(tmp_path, "test_city")
    assert len(result) == 1
    assert "selected_edges" in result.columns


def test_load_stitch_labels_missing_returns_none(tmp_path):
    """Return None when no stitch labels exist."""
    result = load_stitch_labels(tmp_path, "nonexistent")
    assert result is None
