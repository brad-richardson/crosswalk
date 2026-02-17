"""Tests for tool adapters."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cbench.adapters.base import EvalMode, ToolOutput
from cbench.adapters.matcher import MatcherAdapter


class TestToolOutput:
    def test_valid_output(self):
        df = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
        out = ToolOutput(matches=df)
        assert len(out.matches) == 1

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"]})
        with pytest.raises(ValueError, match="missing required columns"):
            ToolOutput(matches=df)


class TestMatcherAdapter:
    def test_name_and_eval_mode(self):
        adapter = MatcherAdapter()
        assert adapter.name == "matcher"
        assert adapter.eval_mode == EvalMode.RAW_MATCH

    @patch("cbench.adapters.matcher.subprocess.run")
    def test_run_calls_matcher_cli(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        # Create fake output
        bridge_path = tmp_path / "bridge.parquet"
        pd.DataFrame(
            {
                "gers_id": ["r1"],
                "local_id": ["t1"],
                "confidence": [0.9],
                "match_type": ["1:1"],
            }
        ).to_parquet(bridge_path)

        adapter = MatcherAdapter()
        result = adapter.run(
            reference=tmp_path / "ref.parquet",
            target=tmp_path / "tgt.parquet",
            output_dir=tmp_path,
        )

        assert result == bridge_path
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "matcher"
        assert cmd[1] == "match"

    def test_parse_output(self, tmp_path):
        bridge_path = tmp_path / "bridge.parquet"
        pd.DataFrame(
            {
                "gers_id": ["r1", "r2"],
                "local_id": ["t1", "t2"],
                "confidence": [0.95, 0.80],
                "match_type": ["1:1", "1:N"],
            }
        ).to_parquet(bridge_path)

        adapter = MatcherAdapter()
        output = adapter.parse_output(bridge_path)

        assert len(output.matches) == 2
        assert list(output.matches.columns) == ["ref_id", "target_id", "confidence"]
        assert output.matches["ref_id"].iloc[0] == "r1"
        assert output.matches["target_id"].iloc[0] == "t1"
        assert output.metadata["total_rows"] == 2


try:
    import geopandas  # noqa: F401

    _has_geopandas = True
except ImportError:
    _has_geopandas = False


@pytest.mark.skipif(not _has_geopandas, reason="geopandas required for hootenanny adapter")
class TestHootAdapter:
    def test_name_and_eval_mode(self):
        from cbench.adapters.hootenanny import HootAdapter

        adapter = HootAdapter()
        assert adapter.name == "hootenanny"
        assert adapter.eval_mode == EvalMode.RAW_MATCH
