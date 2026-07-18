"""Tests for the agent-panel voting hold (``panel_hold:`` dataset flag).

Two seams:

* ``dataset_panel_hold`` — the declarative helper that reads the ``panel_hold:``
  block from a dataset YAML (present / absent / missing-file / malformed /
  unquoted-date), mirroring ``factory.publish.dataset_quality_hold``.
* ``crosswalk agent stitch-batch`` — the CLI gate that refuses evidence-pack
  generation for a held dataset, the ``--override-hold`` escape hatch, and the
  guarantee that an unheld dataset is unaffected.

The gate is AGENT-PANEL only: the human review queue (``crosswalk data
stitch-batch``) and publishing (``quality_hold``) are deliberately not gated.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from crosswalk.cli import app
from crosswalk.datasets.schema import dataset_panel_hold

runner = CliRunner()

# The real dataset this PR holds — used to exercise the CLI gate end-to-end
# against the shipped config (which carries the panel_hold block).
HELD_DATASET = "ch_grand_geneva_cycle_schema"


def _write_yaml(datasets_dir, name, body):
    (datasets_dir / f"{name}.yaml").write_text(body)


# ---------------------------------------------------------------------------
# dataset_panel_hold helper
# ---------------------------------------------------------------------------
class TestDatasetPanelHold:
    def test_present_hold_returns_reason_and_since(self, tmp_path):
        _write_yaml(
            tmp_path,
            "ds",
            "name: ds\npanel_hold:\n  reason: route overlay noise\n  since: '2026-07-18'\n",
        )
        assert dataset_panel_hold("ds", tmp_path) == {
            "reason": "route overlay noise",
            "since": "2026-07-18",
        }

    def test_absent_block_returns_none(self, tmp_path):
        _write_yaml(tmp_path, "ds", "name: ds\ntype: bike\n")
        assert dataset_panel_hold("ds", tmp_path) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert dataset_panel_hold("nonexistent_ds", tmp_path) is None

    def test_malformed_block_still_holds(self, tmp_path):
        """Fail-safe: any truthy panel_hold value holds — a known-noise dataset
        must not slip into a wave on a parsing technicality."""
        _write_yaml(tmp_path, "ds", "name: ds\npanel_hold: true\n")
        assert dataset_panel_hold("ds", tmp_path) == {
            "reason": "unspecified (malformed panel_hold block)",
            "since": None,
        }

    def test_reason_without_since(self, tmp_path):
        _write_yaml(tmp_path, "ds", "name: ds\npanel_hold:\n  reason: noisy\n")
        assert dataset_panel_hold("ds", tmp_path) == {"reason": "noisy", "since": None}

    def test_normalizes_unquoted_yaml_date(self, tmp_path):
        """An unquoted YAML date parses to datetime.date — normalize to ISO string."""
        _write_yaml(
            tmp_path,
            "ds",
            "name: ds\npanel_hold:\n  reason: broken\n  since: 2026-07-18\n",
        )
        assert dataset_panel_hold("ds", tmp_path) == {
            "reason": "broken",
            "since": "2026-07-18",
        }

    def test_unparseable_yaml_returns_none(self, tmp_path):
        _write_yaml(tmp_path, "ds", "name: ds\n  : : bad indent\n:::\n")
        assert dataset_panel_hold("ds", tmp_path) is None


# ---------------------------------------------------------------------------
# Schema round-trip: the typed field must survive a save/load cycle
# ---------------------------------------------------------------------------
def test_panel_hold_survives_schema_roundtrip(tmp_path):
    from crosswalk.datasets.schema import (
        DatasetConfig,
        PanelHoldConfig,
        load_dataset_config,
        save_dataset_config,
    )

    config = DatasetConfig(
        name="held_ds",
        panel_hold=PanelHoldConfig(reason="route overlay noise", since="2026-07-18"),
    )
    path = tmp_path / "held_ds.yaml"
    save_dataset_config(config, path)
    loaded = load_dataset_config(path)
    assert loaded.panel_hold is not None
    assert loaded.panel_hold.reason == "route overlay noise"
    assert loaded.panel_hold.since == "2026-07-18"


# ---------------------------------------------------------------------------
# CLI gate: crosswalk agent stitch-batch
# ---------------------------------------------------------------------------
class TestStitchBatchPanelHoldGate:
    def test_shipped_dataset_carries_hold(self):
        """The dataset this PR holds must actually declare the block, so the CLI
        gate below exercises real config (not just a fixture)."""
        hold = dataset_panel_hold(HELD_DATASET)
        assert hold is not None
        assert "route" in hold["reason"]
        assert hold["since"] == "2026-07-18"

    def test_held_dataset_refuses_pack_generation(self):
        result = runner.invoke(app, ["agent", "stitch-batch", HELD_DATASET])
        assert result.exit_code == 1
        assert "held from agent-panel voting" in result.stdout
        # No pack work happened (would require a groups sidecar).
        assert "Loaded" not in result.stdout

    def test_override_hold_proceeds_past_gate(self):
        """--override-hold prints a yellow acknowledgment and continues past the
        gate (then exits on the missing sidecar — NOT on the hold refusal)."""
        result = runner.invoke(app, ["agent", "stitch-batch", HELD_DATASET, "--override-hold"])
        assert "override-hold: proceeding" in result.stdout
        assert "held from agent-panel voting" not in result.stdout

    def test_unheld_dataset_is_unaffected(self):
        """A dataset with no panel_hold never hits the gate — it proceeds to the
        (missing) sidecar check, never the hold refusal."""
        result = runner.invoke(app, ["agent", "stitch-batch", "some_unheld_dataset"])
        assert "held from agent-panel voting" not in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
