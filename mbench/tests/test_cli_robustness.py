"""Failure-mode tests for benchmark execution and requested gates."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mbench.cli import _apply_stitch_gate, app


def _write_config(path: Path, gate: str = "") -> None:
    path.write_text(
        f"""
[defaults]
data_dir = "."
labels_dir = "."

{gate}

[datasets.demo]
reference = "ref.parquet"
target = "target.parquet"
"""
    )


def test_requested_gate_fails_closed_on_malformed_config(tmp_path: Path):
    config = tmp_path / "datasets.toml"
    _write_config(
        config,
        """
[gate.demo]
min_mapped_groups = 30
f1_filtered_floor = 0.8
""",
    )
    assert _apply_stitch_gate(config, [("demo", None)]) is True


def test_requested_gate_fails_closed_when_no_floors_exist(tmp_path: Path):
    config = tmp_path / "datasets.toml"
    _write_config(config)
    assert _apply_stitch_gate(config, [("demo", None)]) is True


def test_run_batch_exits_nonzero_on_execution_failure(tmp_path: Path, monkeypatch):
    config = tmp_path / "datasets.toml"
    _write_config(config)

    def fail_run(**_kwargs):
        raise RuntimeError("intentional adapter failure")

    monkeypatch.setattr("mbench.runner.run_single", fail_run)
    result = CliRunner().invoke(app, ["run-batch", "crosswalk", "--config", str(config)])

    assert result.exit_code == 1
    assert "FAILED: intentional adapter failure" in result.stdout
