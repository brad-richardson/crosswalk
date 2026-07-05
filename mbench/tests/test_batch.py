"""Tests for batch runner and dataset config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from mbench.adapters.base import EvalMode, ToolOutput
from mbench.cli import load_datasets_config
from mbench.runner import run_single


@dataclass
class _FakeAdapter:
    name: str = "fake"
    eval_mode: EvalMode = EvalMode.STITCH

    def run(self, reference, target, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "bridge.parquet"
        out.touch()
        return out

    def parse_output(self, output_path):
        matches = pd.DataFrame(
            {
                "ref_id": ["r1", "r2"],
                "target_id": ["t1", "t2"],
                "confidence": [0.95, 0.80],
            }
        )
        return ToolOutput(matches=matches, metadata={"test": True})


class TestRunSingle:
    def test_missing_reference_raises(self, tmp_path: Path):
        adapter = _FakeAdapter()
        with pytest.raises(FileNotFoundError, match="Reference file not found"):
            run_single(
                adapter=adapter,
                dataset="test",
                reference=tmp_path / "nonexistent.parquet",
                target=tmp_path / "also_missing.parquet",
                labels_dir=tmp_path,
                output_dir=tmp_path / "output",
            )

    def test_missing_target_raises(self, tmp_path: Path):
        ref = tmp_path / "ref.parquet"
        ref.touch()
        adapter = _FakeAdapter()
        with pytest.raises(FileNotFoundError, match="Target file not found"):
            run_single(
                adapter=adapter,
                dataset="test",
                reference=ref,
                target=tmp_path / "nonexistent.parquet",
                labels_dir=tmp_path,
                output_dir=tmp_path / "output",
            )

    def test_missing_labels_dir_raises(self, tmp_path: Path):
        ref = tmp_path / "ref.parquet"
        ref.touch()
        tgt = tmp_path / "tgt.parquet"
        tgt.touch()
        adapter = _FakeAdapter()
        with pytest.raises(FileNotFoundError, match="Labels directory not found"):
            run_single(
                adapter=adapter,
                dataset="test",
                reference=ref,
                target=tgt,
                labels_dir=tmp_path / "no_labels",
                output_dir=tmp_path / "output",
            )

    def test_successful_run(self, tmp_path: Path):
        ref = tmp_path / "ref.parquet"
        ref.touch()
        tgt = tmp_path / "tgt.parquet"
        tgt.touch()

        # Create labels
        labels_dir = tmp_path / "labels"
        dataset_dir = labels_dir / "dataset=test_ds"
        dataset_dir.mkdir(parents=True)
        labels = pd.DataFrame(
            {
                "ref_id": ["r1", "r2", "r3"],
                "target_id": ["t1", "t2", "t3"],
                "label": ["match", "match", "no_match"],
            }
        )
        labels.to_csv(dataset_dir / "data.csv", index=False)

        adapter = _FakeAdapter()
        result = run_single(
            adapter=adapter,
            dataset="test_ds",
            reference=ref,
            target=tgt,
            labels_dir=labels_dir,
            output_dir=tmp_path / "output",
        )

        assert result.tool == "fake"
        assert result.dataset == "test_ds"
        assert result.eval_result.true_positives == 2
        assert result.eval_result.false_negatives == 0
        assert result.eval_result.f1 == pytest.approx(1.0)
        assert result.metadata == {"test": True}

    def test_saves_to_results_file(self, tmp_path: Path):
        ref = tmp_path / "ref.parquet"
        ref.touch()
        tgt = tmp_path / "tgt.parquet"
        tgt.touch()

        labels_dir = tmp_path / "labels"
        dataset_dir = labels_dir / "dataset=test_ds"
        dataset_dir.mkdir(parents=True)
        labels = pd.DataFrame(
            {
                "ref_id": ["r1"],
                "target_id": ["t1"],
                "label": ["match"],
            }
        )
        labels.to_csv(dataset_dir / "data.csv", index=False)

        results_file = tmp_path / "results.jsonl"
        adapter = _FakeAdapter()
        run_single(
            adapter=adapter,
            dataset="test_ds",
            reference=ref,
            target=tgt,
            labels_dir=labels_dir,
            output_dir=tmp_path / "output",
            results_file=results_file,
        )

        assert results_file.exists()
        lines = results_file.read_text().strip().split("\n")
        assert len(lines) == 1


class TestLoadDatasetsConfig:
    def test_load_valid_config(self, tmp_path: Path):
        config_path = tmp_path / "datasets.toml"
        config_path.write_text(
            """
[defaults]
data_dir = "../data/raw"
labels_dir = "../labels/human"

[datasets.test_city]
reference = "test_city_overture_segments_v1.0.parquet"
target = "test_city_v1.0.parquet"
connectors = "test_city_overture_connectors_v1.0.parquet"
"""
        )

        cfg = load_datasets_config(config_path)
        assert "defaults" in cfg
        assert "datasets" in cfg
        assert "test_city" in cfg["datasets"]
        ds = cfg["datasets"]["test_city"]
        assert ds["reference"] == "test_city_overture_segments_v1.0.parquet"
        assert ds["target"] == "test_city_v1.0.parquet"
        assert ds["connectors"] == "test_city_overture_connectors_v1.0.parquet"

    def test_missing_config_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_datasets_config(tmp_path / "nonexistent.toml")

    def test_missing_datasets_section_raises(self, tmp_path: Path):
        config_path = tmp_path / "bad.toml"
        config_path.write_text("[defaults]\ndata_dir = '.'")
        with pytest.raises(ValueError, match="missing \\[datasets\\] section"):
            load_datasets_config(config_path)

    def test_multiple_datasets(self, tmp_path: Path):
        config_path = tmp_path / "datasets.toml"
        config_path.write_text(
            """
[datasets.city_a]
reference = "a_ref.parquet"
target = "a_tgt.parquet"

[datasets.city_b]
reference = "b_ref.parquet"
target = "b_tgt.parquet"
"""
        )

        cfg = load_datasets_config(config_path)
        assert len(cfg["datasets"]) == 2
        assert "city_a" in cfg["datasets"]
        assert "city_b" in cfg["datasets"]
