"""Tests for the eval-bridge CLI command with ground truth evaluation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from matcher.cli import app


class TestEvalBridgeCommand:
    """Tests for matcher eval-bridge command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def bridge_file(self):
        """Create a sample bridge file."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df = pd.DataFrame(
                {
                    "gers_id": ["g1", "g2", "g3", "g4", "g5"],
                    "local_id": ["t1", "t2", "t3", "t4", "t5"],
                    "confidence": [0.95, 0.85, 0.75, 0.60, 0.45],
                    "match_type": ["1:1", "1:1", "1:1", "1:1", "1:1"],
                }
            )
            df.to_parquet(f.name)
            yield Path(f.name)
            Path(f.name).unlink()

    @pytest.fixture
    def ground_truth_file(self):
        """Create a sample ground truth file."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {
                    "gers_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
                    "target_id": ["t1", "t2", "t3", "t4", "t5", "t6"],
                    # g1-t1: match (TP), g2-t2: no_match (FP), g3-t3: match (TP)
                    # g4-t4: no_match (FP), g5-t5: match (TP), g6-t6: match (FN - not in predictions)
                    "label": ["match", "no_match", "match", "no_match", "match", "match"],
                }
            )
            df.to_csv(f.name, index=False)
            yield Path(f.name)
            Path(f.name).unlink()

    def test_eval_bridge_without_ground_truth(self, runner, bridge_file):
        """eval-bridge command shows basic stats without ground truth."""
        result = runner.invoke(app, ["eval-bridge", str(bridge_file)])

        assert result.exit_code == 0
        assert "Total matches: 5" in result.output
        assert "Mean confidence:" in result.output
        assert "Confidence distribution:" in result.output

    def test_eval_bridge_with_ground_truth(self, runner, bridge_file, ground_truth_file):
        """eval-bridge command computes precision/recall/F1 with ground truth."""
        result = runner.invoke(
            app, ["eval-bridge", str(bridge_file), "--ground-truth", str(ground_truth_file)]
        )

        assert result.exit_code == 0
        assert "Ground Truth Evaluation" in result.output
        assert "Total labeled pairs: 6" in result.output
        # Should show TP=3 (g1, g3, g5 are match and predicted)
        assert "True Positives: 3" in result.output
        # Should show FP=2 (g2, g4 are no_match but predicted)
        assert "False Positives: 2" in result.output
        # Should show FN=1 (g6 is match but not predicted)
        assert "False Negatives: 1" in result.output
        # Precision = 3 / (3+2) = 0.6
        assert "Precision: 0.600" in result.output
        # Recall = 3 / (3+1) = 0.75
        assert "Recall: 0.750" in result.output
        # F1 = 2 * 0.6 * 0.75 / (0.6 + 0.75) = 0.667
        assert "F1 Score: 0.667" in result.output

    def test_eval_bridge_nonexistent_ground_truth(self, runner, bridge_file):
        """eval-bridge command errors on nonexistent ground truth file."""
        result = runner.invoke(
            app, ["eval-bridge", str(bridge_file), "--ground-truth", "/nonexistent/file.csv"]
        )

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_eval_bridge_parquet_ground_truth(self, runner, bridge_file):
        """eval-bridge command can read parquet ground truth files."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df = pd.DataFrame(
                {
                    "gers_id": ["g1", "g2"],
                    "target_id": ["t1", "t2"],
                    "label": ["match", "match"],
                }
            )
            df.to_parquet(f.name)
            gt_path = Path(f.name)

        try:
            result = runner.invoke(
                app, ["eval-bridge", str(bridge_file), "--ground-truth", str(gt_path)]
            )

            assert result.exit_code == 0
            assert "Ground Truth Evaluation" in result.output
        finally:
            gt_path.unlink()

    @pytest.mark.parametrize(
        "predictions,ground_truth_labels,expected_tp,expected_fp,expected_fn",
        [
            # All predictions are true positives
            (
                [("g1", "t1"), ("g2", "t2")],
                [("g1", "t1", "match"), ("g2", "t2", "match")],
                2,
                0,
                0,
            ),
            # All predictions are false positives
            (
                [("g1", "t1"), ("g2", "t2")],
                [("g1", "t1", "no_match"), ("g2", "t2", "no_match")],
                0,
                2,
                0,
            ),
            # All ground truth matches are false negatives (no predictions)
            (
                [],
                [("g1", "t1", "match"), ("g2", "t2", "match")],
                0,
                0,
                2,
            ),
            # Mixed case
            (
                [("g1", "t1"), ("g2", "t2"), ("g3", "t3")],
                [
                    ("g1", "t1", "match"),  # TP
                    ("g2", "t2", "no_match"),  # FP
                    ("g4", "t4", "match"),  # FN (not predicted)
                ],
                1,
                1,
                1,
            ),
        ],
    )
    def test_various_metric_scenarios(
        self,
        runner,
        predictions,
        ground_truth_labels,
        expected_tp,
        expected_fp,
        expected_fn,
    ):
        """Test various precision/recall scenarios."""
        # Create bridge file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            if predictions:
                bridge_df = pd.DataFrame(
                    {
                        "gers_id": [p[0] for p in predictions],
                        "local_id": [p[1] for p in predictions],
                        "confidence": [0.9] * len(predictions),
                        "match_type": ["1:1"] * len(predictions),
                    }
                )
            else:
                bridge_df = pd.DataFrame(
                    {"gers_id": [], "local_id": [], "confidence": [], "match_type": []}
                )
            bridge_df.to_parquet(f.name)
            bridge_path = Path(f.name)

        # Create ground truth file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            gt_df = pd.DataFrame(
                {
                    "gers_id": [g[0] for g in ground_truth_labels],
                    "target_id": [g[1] for g in ground_truth_labels],
                    "label": [g[2] for g in ground_truth_labels],
                }
            )
            gt_df.to_csv(f.name, index=False)
            gt_path = Path(f.name)

        try:
            result = runner.invoke(
                app, ["eval-bridge", str(bridge_path), "--ground-truth", str(gt_path)]
            )

            assert result.exit_code == 0
            assert f"True Positives: {expected_tp}" in result.output
            assert f"False Positives: {expected_fp}" in result.output
            assert f"False Negatives: {expected_fn}" in result.output
        finally:
            bridge_path.unlink()
            gt_path.unlink()
