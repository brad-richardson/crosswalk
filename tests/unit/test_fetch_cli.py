"""Tests for the data fetch CLI subcommands."""

import re
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from matcher.cli import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


class TestFetchTargetCommand:
    """Tests for the data fetch target command."""

    def test_fetch_target_help(self):
        """Test data fetch target help output."""
        result = runner.invoke(app, ["data", "fetch", "target", "--help"])
        assert result.exit_code == 0
        assert "target" in result.output.lower()
        assert "gis portals" in result.output.lower()

    def test_fetch_target_no_args(self):
        """Test data fetch target with no arguments shows error."""
        result = runner.invoke(app, ["data", "fetch", "target"])
        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("matcher.fetch.target.fetch_dataset")
    def test_fetch_target_single_dataset(self, mock_fetch):
        """Test fetching a single dataset."""
        mock_fetch.return_value = Path("data/raw/test_v1.0.parquet")

        result = runner.invoke(app, ["data", "fetch", "target", "test_dataset"])

        assert result.exit_code == 0
        mock_fetch.assert_called_once()

    @patch("matcher.fetch.target.fetch_datasets_by_prefix")
    def test_fetch_target_by_prefix(self, mock_fetch_by_prefix):
        """Test fetching datasets by prefix."""
        mock_fetch_by_prefix.return_value = {
            "test_a": Path("a.parquet"),
            "test_b": Path("b.parquet"),
        }

        result = runner.invoke(app, ["data", "fetch", "target", "--prefix", "test_"])

        assert result.exit_code == 0
        mock_fetch_by_prefix.assert_called_once()


class TestFetchReferenceCommand:
    """Tests for the data fetch reference command."""

    def test_fetch_reference_help(self):
        """Test data fetch reference help output."""
        result = runner.invoke(app, ["data", "fetch", "reference", "--help"])
        assert result.exit_code == 0
        assert "reference" in result.output.lower()
        assert "Overture" in result.output

    @patch("matcher.datasets.schema.get_dataset_config")
    @patch("matcher.datasets.schema.list_dataset_configs")
    def test_fetch_reference_missing_dataset(self, mock_list, mock_get_config):
        """Test error when dataset doesn't exist."""
        mock_get_config.return_value = None
        mock_list.return_value = []

        result = runner.invoke(app, ["data", "fetch", "reference", "nonexistent"])

        assert result.exit_code == 1
        assert "Could not find" in result.output


class TestFetchAllCommand:
    """Tests for the data fetch all command."""

    def test_fetch_all_help(self):
        """Test data fetch all help output."""
        result = runner.invoke(app, ["data", "fetch", "all", "--help"])
        assert result.exit_code == 0
        assert "target and reference" in result.output.lower()


class TestFetchListCommand:
    """Tests for the data fetch list command."""

    def test_fetch_list_help(self):
        """Test data fetch list help output."""
        result = runner.invoke(app, ["data", "fetch", "list", "--help"])
        assert result.exit_code == 0
        assert "List available datasets" in result.output

    @patch("matcher.fetch.target.print_datasets")
    def test_fetch_list_all(self, mock_print):
        """Test listing all datasets."""
        result = runner.invoke(app, ["data", "fetch", "list"])

        assert result.exit_code == 0
        mock_print.assert_called_once_with(None)

    @patch("matcher.fetch.target.print_datasets")
    def test_fetch_list_with_prefix(self, mock_print):
        """Test listing datasets with prefix filter."""
        result = runner.invoke(app, ["data", "fetch", "list", "--prefix", "us_"])

        assert result.exit_code == 0
        mock_print.assert_called_once_with("us_")


class TestFetchVerifyCommand:
    """Tests for the data fetch verify command."""

    def test_fetch_verify_help(self):
        """Test data fetch verify help output."""
        result = runner.invoke(app, ["data", "fetch", "verify", "--help"])
        assert result.exit_code == 0
        assert "Verify" in result.output
        assert "dataset" in result.output.lower()

    def test_fetch_verify_no_args(self):
        """Test data fetch verify with no arguments shows error."""
        result = runner.invoke(app, ["data", "fetch", "verify"])
        assert result.exit_code == 1
        assert "Error" in result.output or "Must specify" in result.output

    @patch("matcher.datasets.schema.get_dataset_config")
    def test_fetch_verify_missing_dataset(self, mock_get_config):
        """Test error when dataset doesn't exist."""
        mock_get_config.return_value = None

        result = runner.invoke(app, ["data", "fetch", "verify", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "Could not find" in result.output


class TestFetchOvertureCommand:
    """Tests for the data fetch overture command."""

    def test_fetch_overture_help(self):
        """Test data fetch overture help output."""
        result = runner.invoke(app, ["data", "fetch", "overture", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "Overture" in output
        assert "--all" in output
        assert "--prefix" in output

    def test_fetch_overture_no_args(self):
        """Test data fetch overture with no arguments shows error."""
        result = runner.invoke(app, ["data", "fetch", "overture"])
        assert result.exit_code == 1
        assert "Must specify" in result.output or "Error" in result.output


class TestFetchOsmCommand:
    """Tests for the data fetch osm command."""

    def test_fetch_osm_help(self):
        """Test data fetch osm help output."""
        result = runner.invoke(app, ["data", "fetch", "osm", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "OSM" in output
        assert "--all" in output
        assert "--prefix" in output

    def test_fetch_osm_no_args(self):
        """Test data fetch osm with no arguments shows error."""
        result = runner.invoke(app, ["data", "fetch", "osm"])
        assert result.exit_code == 1
        assert "Must specify" in result.output or "Error" in result.output


class TestFetchSubcommands:
    """Test the data fetch subcommand structure."""

    def test_fetch_shows_subcommands(self):
        """Test that data fetch command shows available subcommands."""
        result = runner.invoke(app, ["data", "fetch", "--help"])
        assert result.exit_code == 0
        assert "target" in result.output
        assert "reference" in result.output
        assert "all" in result.output
        assert "list" in result.output
        assert "verify" in result.output
        assert "overture" in result.output
        assert "osm" in result.output
