"""Tests for batch runner and dataset config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from mbench.adapters.base import EvalMode, ToolOutput
from mbench.cli import load_datasets_config
from mbench.runner import _decision_views, run_single


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


@dataclass
class _RecordingAdapter(_FakeAdapter):
    """Fake adapter that records the kwargs forwarded to ``run``."""

    name: str = "recording"
    last_kwargs: dict = None

    def run(self, reference, target, output_dir, **kwargs):
        self.last_kwargs = dict(kwargs)
        return super().run(reference, target, output_dir, **kwargs)


@dataclass
class _DecisionAdapter(_FakeAdapter):
    name: str = "decision_fake"
    decision_aware: bool = True

    def parse_output(self, output_path):
        matches = pd.DataFrame(
            {
                "ref_id": ["r1", "r2", "r3"],
                "target_id": ["t1", "t2", "t3"],
                "confidence": [0.95, 0.60, 0.05],
                "match_decision": ["match", "review", "no_match"],
            }
        )
        return ToolOutput(matches=matches)


def _make_labeled_inputs(tmp_path: Path, dataset: str) -> tuple[Path, Path, Path]:
    ref = tmp_path / "ref.parquet"
    ref.touch()
    tgt = tmp_path / "tgt.parquet"
    tgt.touch()
    labels_dir = tmp_path / "labels"
    dataset_dir = labels_dir / f"dataset={dataset}"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "label": ["match"]}).to_csv(
        dataset_dir / "data.csv", index=False
    )
    return ref, tgt, labels_dir


class TestRunSingle:
    def test_forwards_dataset_name_to_adapter(self, tmp_path: Path):
        """run_single injects the dataset identity into adapter.run kwargs (#372).

        Crosswalk's resolver-prune allowlist keys on the dataset NAME, so the
        adapter needs it to run the same (pruned) code path production/the gate
        floor was calibrated on.
        """
        ref, tgt, labels_dir = _make_labeled_inputs(tmp_path, "us_boston_streets")
        adapter = _RecordingAdapter()
        run_single(
            adapter=adapter,
            dataset="us_boston_streets",
            reference=ref,
            target=tgt,
            labels_dir=labels_dir,
            output_dir=tmp_path / "output",
        )
        assert adapter.last_kwargs.get("dataset") == "us_boston_streets"

    def test_forwards_dataset_alongside_other_opts(self, tmp_path: Path):
        """The dataset injection coexists with other tool kwargs (e.g. --opt)."""
        ref, tgt, labels_dir = _make_labeled_inputs(tmp_path, "test_ds")
        adapter = _RecordingAdapter()
        run_single(
            adapter=adapter,
            dataset="test_ds",
            reference=ref,
            target=tgt,
            labels_dir=labels_dir,
            output_dir=tmp_path / "output",
            model="xgboost",
        )
        assert adapter.last_kwargs.get("dataset") == "test_ds"
        assert adapter.last_kwargs.get("model") == "xgboost"

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
        assert result.metadata["test"] is True
        assert "provenance" in result.metadata

    def test_decision_output_uses_accepted_as_headline(self, tmp_path: Path):
        ref, tgt, labels_dir = _make_labeled_inputs(tmp_path, "test_ds")
        dataset_dir = labels_dir / "dataset=test_ds"
        pd.DataFrame(
            {
                "ref_id": ["r1", "r2"],
                "target_id": ["t1", "t2"],
                "label": ["match", "match"],
            }
        ).to_csv(dataset_dir / "data.csv", index=False)

        result = run_single(
            adapter=_DecisionAdapter(),
            dataset="test_ds",
            reference=ref,
            target=tgt,
            labels_dir=labels_dir,
            output_dir=tmp_path / "output",
        )

        # Top-level metrics are the production/published accepted set.
        assert result.eval_result.total_predictions == 1
        assert result.eval_result.recall == pytest.approx(0.5)
        assert result.decision_results["accepted"].recall == pytest.approx(0.5)
        assert result.decision_results["review"].total_predictions == 1
        assert result.decision_results["proposal"].recall == pytest.approx(1.0)
        dm = result.bench_result.metrics["decision_metrics"]
        assert dm["headline"] == "accepted"
        assert dm["proposal"]["total_predictions"] == 2
        assert dm["excluded_no_match_count"] == 1
        assert result.bench_result.prediction_view == "accepted"
        assert result.bench_result.metric_schema_version == 2

    def test_adapter_without_decisions_keeps_combined_headline(self, tmp_path: Path):
        ref, tgt, labels_dir = _make_labeled_inputs(tmp_path, "test_ds")
        result = run_single(
            adapter=_FakeAdapter(),
            dataset="test_ds",
            reference=ref,
            target=tgt,
            labels_dir=labels_dir,
            output_dir=tmp_path / "output",
        )
        assert result.eval_result.total_predictions == 2
        assert result.decision_results == {}
        assert "decision_metrics" not in result.bench_result.metrics
        assert result.bench_result.prediction_view == "combined"

    @pytest.mark.parametrize(
        ("values", "message"),
        [(["match", None], "null"), (["match", "unknown"], "unknown values")],
    )
    def test_decision_validation_rejects_null_and_unknown(self, values, message):
        matches = pd.DataFrame(
            {
                "ref_id": ["r1", "r2"],
                "target_id": ["t1", "t2"],
                "confidence": [0.9, 0.8],
                "match_decision": values,
            }
        )
        with pytest.raises(ValueError, match=message):
            _decision_views(matches, decision_aware=True)

    def test_decision_aware_output_requires_decision_column(self):
        matches = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
        with pytest.raises(ValueError, match="missing match_decision"):
            _decision_views(matches, decision_aware=True)

    def test_non_decision_aware_adapter_ignores_decision_column(self):
        """Decision handling is an adapter capability, not a column sniff.

        A third-party adapter that never declared decision_aware must keep its
        historical combined behavior even when its output carries a
        match_decision column with a foreign vocabulary.
        """
        matches = pd.DataFrame(
            {
                "ref_id": ["r1", "r2"],
                "target_id": ["t1", "t2"],
                "confidence": [0.9, 0.8],
                "match_decision": ["accept", "reject"],
            }
        )
        headline, views = _decision_views(matches, decision_aware=False)
        assert headline is matches
        assert views == {}

    def test_stitch_eval_scores_full_selection_not_accepted_view(self, tmp_path: Path):
        """Stitch-level eval is decision-agnostic.

        The optimizer's edge selection includes edges that end up as `review`
        decisions, and the armed gate floors were calibrated on that full
        selection. Scoring accepted-only rows here would silently consume the
        calibrated floor margins.
        """
        ref, tgt, labels_dir = _make_labeled_inputs(tmp_path, "test_ds")
        stitch_dir = labels_dir.parent / "stitching" / "dataset=test_ds"
        stitch_dir.mkdir(parents=True)
        # Curate exactly the review-decision edge (r2, t2).
        pd.DataFrame(
            {
                "group_id": ["g1"],
                "selected_edges": ['[{"ref_id": "r2", "target_id": "t2"}]'],
                "labeler": ["brad"],
            }
        ).to_csv(stitch_dir / "data.csv", index=False)

        result = run_single(
            adapter=_DecisionAdapter(),
            dataset="test_ds",
            reference=ref,
            target=tgt,
            labels_dir=labels_dir,
            output_dir=tmp_path / "output",
        )

        assert result.stitch_result is not None
        # The review edge counts as selected: accepted-only scoring would be 0.
        assert result.stitch_result.f1 == pytest.approx(1.0)

    def test_provenance_failure_does_not_discard_run(self, tmp_path: Path, monkeypatch):
        ref, tgt, labels_dir = _make_labeled_inputs(tmp_path, "test_ds")

        def explode(**_kwargs):
            raise TypeError("mixed-type keys")

        monkeypatch.setattr("mbench.runner.collect_provenance", explode)
        result = run_single(
            adapter=_FakeAdapter(),
            dataset="test_ds",
            reference=ref,
            target=tgt,
            labels_dir=labels_dir,
            output_dir=tmp_path / "output",
        )
        assert result.eval_result.total_predictions == 2
        assert "mixed-type keys" in result.metadata["provenance"]["error"]

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
