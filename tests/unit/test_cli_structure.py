"""Tests for the CLI command structure after refactoring.

These tests verify that all command groups are properly wired up
and accessible at the expected paths.
"""

import re

from typer.testing import CliRunner

from crosswalk.cli import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


class TestTopLevelCommands:
    """Test that top-level commands are accessible."""

    def test_stitch_help(self):
        """Test stitch command is at top level."""
        result = runner.invoke(app, ["stitch", "--help"])
        assert result.exit_code == 0
        assert "stitch pipeline" in result.output.lower()

    def test_train_help(self):
        """Test train command is at top level."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train" in result.output

    def test_eval_help(self):
        """Test eval command is at top level."""
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "Evaluate" in result.output

    def test_backfill_help(self):
        """Test backfill command is at top level."""
        result = runner.invoke(app, ["backfill", "--help"])
        assert result.exit_code == 0
        assert "Recompute" in result.output or "features" in result.output.lower()

    def test_ui_help(self):
        """Test ui command is at top level."""
        result = runner.invoke(app, ["ui", "--help"])
        assert result.exit_code == 0
        assert "ui" in result.output.lower() or "labeling" in result.output.lower()

    def test_version(self):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "crosswalk version" in result.output.lower()


class TestDataCommandGroup:
    """Test the data command group structure."""

    def test_data_help(self):
        """Test data group shows subcommands."""
        result = runner.invoke(app, ["data", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "topology" in output.lower()
        assert "repair" in output.lower()
        assert "quality" in output.lower()
        assert "validate" in output.lower()
        assert "fetch" in output.lower()
        assert "cache" in output.lower()

    def test_data_fetch_help(self):
        """Test data fetch subgroup shows commands."""
        result = runner.invoke(app, ["data", "fetch", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "target" in output.lower()
        assert "reference" in output.lower()
        assert "overture" in output.lower()
        assert "osm" in output.lower()
        assert "all" in output.lower()
        assert "list" in output.lower()
        assert "verify" in output.lower()

    def test_data_topology_help(self):
        """Test data topology command."""
        result = runner.invoke(app, ["data", "topology", "--help"])
        assert result.exit_code == 0
        assert "topology" in result.output.lower()

    def test_data_repair_help(self):
        """Test data repair command."""
        result = runner.invoke(app, ["data", "repair", "--help"])
        assert result.exit_code == 0
        assert "repair" in result.output.lower()

    def test_data_quality_help(self):
        """Test data quality command."""
        result = runner.invoke(app, ["data", "quality", "--help"])
        assert result.exit_code == 0
        assert "quality" in result.output.lower() or "fingerprint" in result.output.lower()

    def test_data_validate_help(self):
        """Test data validate command."""
        result = runner.invoke(app, ["data", "validate", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.output.lower()

    def test_data_cache_help(self):
        """Test data cache command."""
        result = runner.invoke(app, ["data", "cache", "--help"])
        assert result.exit_code == 0
        assert "features" in result.output.lower() or "cache" in result.output.lower()


class TestAnalyzeCommandGroup:
    """Test the analyze command group structure."""

    def test_analyze_help(self):
        """Test analyze group shows subcommands."""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "bridge" in output.lower()
        assert "screen" in output.lower()
        assert "errors" in output.lower()
        assert "labels" in output.lower()
        assert "integrate" in output.lower()
        assert "validate" in output.lower()

    def test_analyze_bridge_help(self):
        """Test analyze bridge command."""
        result = runner.invoke(app, ["analyze", "bridge", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "bridge" in output.lower() or "quality" in output.lower()

    def test_analyze_screen_help(self):
        """Test analyze screen command."""
        result = runner.invoke(app, ["analyze", "screen", "--help"])
        assert result.exit_code == 0
        assert "screen" in result.output.lower()

    def test_analyze_errors_help(self):
        """Test analyze errors command."""
        result = runner.invoke(app, ["analyze", "errors", "--help"])
        assert result.exit_code == 0
        assert "error" in result.output.lower()

    def test_analyze_labels_help(self):
        """Test analyze labels command."""
        result = runner.invoke(app, ["analyze", "labels", "--help"])
        assert result.exit_code == 0
        assert "label" in result.output.lower() or "statistic" in result.output.lower()

    def test_analyze_integrate_help(self):
        """Test analyze integrate command."""
        result = runner.invoke(app, ["analyze", "integrate", "--help"])
        assert result.exit_code == 0
        assert "Integrate" in result.output

    def test_analyze_validate_help(self):
        """Test analyze validate command."""
        result = runner.invoke(app, ["analyze", "validate", "--help"])
        assert result.exit_code == 0
        assert "validation" in result.output.lower() or "experiment" in result.output.lower()


class TestClassCommandGroup:
    """Test the class command group structure."""

    def test_class_help(self):
        """Test class group shows subcommands."""
        result = runner.invoke(app, ["class", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "discover" in output.lower()
        assert "analyze" in output.lower()
        assert "detect-non-roads" in output.lower()
        assert "train-predictor" in output.lower()
        assert "predict" in output.lower()

    def test_class_discover_help(self):
        """Test class discover command."""
        result = runner.invoke(app, ["class", "discover", "--help"])
        assert result.exit_code == 0
        assert "Discover" in result.output

    def test_class_analyze_help(self):
        """Test class analyze command."""
        result = runner.invoke(app, ["class", "analyze", "--help"])
        assert result.exit_code == 0
        assert "Analyze" in result.output or "class" in result.output.lower()

    def test_class_detect_non_roads_help(self):
        """Test class detect-non-roads command."""
        result = runner.invoke(app, ["class", "detect-non-roads", "--help"])
        assert result.exit_code == 0
        assert "non-road" in result.output.lower()

    def test_class_train_predictor_help(self):
        """Test class train-predictor command."""
        result = runner.invoke(app, ["class", "train-predictor", "--help"])
        assert result.exit_code == 0
        assert "Train" in result.output

    def test_class_predict_help(self):
        """Test class predict command."""
        result = runner.invoke(app, ["class", "predict", "--help"])
        assert result.exit_code == 0
        assert "Apply" in result.output or "predict" in result.output.lower()

    def test_class_analyze_train_predictor_option(self):
        """Test class analyze --train-predictor option (merged from analyze-predictor)."""
        result = runner.invoke(app, ["class", "analyze", "--help"])
        assert result.exit_code == 0
        # Use strip_ansi because rich/typer ANSI codes can split the option string
        assert "--train-predictor" in strip_ansi(result.output)

    def test_class_update_mappings_help(self):
        """Test class update-mappings command."""
        result = runner.invoke(app, ["class", "update-mappings", "--help"])
        assert result.exit_code == 0
        assert "Update" in result.output or "mappings" in result.output.lower()

    def test_class_suggest_mapping_help(self):
        """Test class suggest-mapping command."""
        result = runner.invoke(app, ["class", "suggest-mapping", "--help"])
        assert result.exit_code == 0
        assert "Suggest" in result.output


class TestAgentCommandGroup:
    """Test the agent command group structure."""

    def test_agent_help(self):
        """Test agent group shows subcommands."""
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "batch" in output.lower()
        assert "test-batch" in output.lower()
        assert "sweep" in output.lower()
        assert "run" in output.lower()
        assert "import" in output.lower()
        assert "consensus" in output.lower()

    def test_agent_batch_help(self):
        """Test agent batch command."""
        result = runner.invoke(app, ["agent", "batch", "--help"])
        assert result.exit_code == 0
        assert "Generate" in result.output or "batch" in result.output.lower()

    def test_agent_test_batch_help(self):
        """Test agent test-batch command."""
        result = runner.invoke(app, ["agent", "test-batch", "--help"])
        assert result.exit_code == 0
        assert "test batch" in result.output.lower() or "human labels" in result.output.lower()

    def test_agent_sweep_help(self):
        """Test agent sweep command."""
        result = runner.invoke(app, ["agent", "sweep", "--help"])
        assert result.exit_code == 0
        assert "sweep" in result.output.lower()

    def test_agent_run_help(self):
        """Test agent run command."""
        result = runner.invoke(app, ["agent", "run", "--help"])
        assert result.exit_code == 0
        assert "Run" in result.output

    def test_agent_import_help(self):
        """Test agent import command."""
        result = runner.invoke(app, ["agent", "import", "--help"])
        assert result.exit_code == 0
        assert "Import" in result.output

    def test_agent_consensus_help(self):
        """Test agent consensus command."""
        result = runner.invoke(app, ["agent", "consensus", "--help"])
        assert result.exit_code == 0
        assert "consensus" in result.output.lower()

    def test_agent_eval_sweep_help(self):
        """Test agent eval-sweep command."""
        result = runner.invoke(app, ["agent", "eval-sweep", "--help"])
        assert result.exit_code == 0
        assert "Evaluate" in result.output or "sweep" in result.output.lower()


class TestMainAppStructure:
    """Test the main app structure and help output."""

    def test_main_help_shows_all_groups(self):
        """Test that main help shows all command groups."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)

        # Top-level commands
        assert "stitch" in output.lower()
        assert "train" in output.lower()
        assert "eval" in output.lower()
        assert "backfill" in output.lower()
        assert "ui" in output.lower()
        assert "version" in output.lower()

        # Command groups
        assert "data" in output.lower()
        assert "analyze" in output.lower()
        assert "class" in output.lower()
        assert "agent" in output.lower()

    def test_no_args_shows_help(self):
        """Test that running without args shows help."""
        result = runner.invoke(app, [])
        # no_args_is_help=True shows help with exit code 0 or 2 depending on Typer version
        assert result.exit_code in (0, 2)
        assert "Usage:" in result.output
