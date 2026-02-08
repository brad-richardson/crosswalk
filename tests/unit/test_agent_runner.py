"""Tests for agent_labeling.runner module (batch mode)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from matcher.agent_labeling.runner import (
    IMAGE_DESCRIPTIONS,
    VARIANT_CONFIG,
    prepare_batch_prompt,
    run_agent_batch,
    select_few_shot_examples,
    validate_output_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(tmp_path, batch_name, candidates, ground_truth_rows=None):
    """Create a minimal batch directory structure for testing.

    Args:
        tmp_path: pytest tmp_path fixture.
        batch_name: Name for the batch directory.
        candidates: List of (ref_id, target_id) tuples.
        ground_truth_rows: Optional list of dicts for ground_truth/data.csv.

    Returns:
        Path to the created batch directory.
    """
    batch_dir = tmp_path / batch_name
    cand_dir = batch_dir / "candidates"
    cand_dir.mkdir(parents=True)

    for ref_id, target_id in candidates:
        d = cand_dir / f"{ref_id}__{target_id}"
        d.mkdir()
        (d / "metadata.yaml").write_text(
            f"candidate:\n  ref_id: {ref_id}\n  target_id: {target_id}\n"
        )
        (d / "geometry_only.png").write_bytes(b"fake png")

    if ground_truth_rows:
        gt_dir = batch_dir / "labels" / "ground_truth"
        gt_dir.mkdir(parents=True)
        gt_df = pd.DataFrame(ground_truth_rows)
        gt_df.to_csv(gt_dir / "data.csv", index=False)

    return batch_dir


# ---------------------------------------------------------------------------
# TestSelectFewShotExamples
# ---------------------------------------------------------------------------


class TestSelectFewShotExamples:
    """Tests for few-shot example selection."""

    def test_balanced_selection(self, tmp_path):
        """Should select balanced match/no_match examples."""
        batches_dir = tmp_path / "batches"

        # Source batch with ground truth
        _make_batch(
            batches_dir,
            "source_batch",
            [("r1", "t1"), ("r2", "t2"), ("r3", "t3"), ("r4", "t4")],
            ground_truth_rows=[
                {"ref_id": "r1", "target_id": "t1", "label": "match"},
                {"ref_id": "r2", "target_id": "t2", "label": "match"},
                {"ref_id": "r3", "target_id": "t3", "label": "no_match"},
                {"ref_id": "r4", "target_id": "t4", "label": "no_match"},
            ],
        )

        # Current batch (should be excluded)
        current = _make_batch(
            batches_dir,
            "current_batch",
            [("r5", "t5")],
            ground_truth_rows=[
                {"ref_id": "r5", "target_id": "t5", "label": "match"},
            ],
        )

        examples = select_few_shot_examples(
            batch_dir=current,
            variant="geometry_only",
            n_examples=4,
        )

        assert len(examples) == 4
        labels = [e["label"] for e in examples]
        assert labels.count("match") == 2
        assert labels.count("no_match") == 2

    def test_excludes_current_batch(self, tmp_path):
        """Should not use examples from the current batch."""
        batches_dir = tmp_path / "batches"

        current = _make_batch(
            batches_dir,
            "only_batch",
            [("r1", "t1")],
            ground_truth_rows=[
                {"ref_id": "r1", "target_id": "t1", "label": "match"},
            ],
        )

        examples = select_few_shot_examples(
            batch_dir=current,
            variant="geometry_only",
            n_examples=4,
        )

        # No other batches exist, so no examples
        assert len(examples) == 0

    def test_empty_ground_truth(self, tmp_path):
        """Should return empty list when no ground truth exists."""
        batches_dir = tmp_path / "batches"

        current = _make_batch(batches_dir, "empty_batch", [("r1", "t1")])

        examples = select_few_shot_examples(
            batch_dir=current,
            variant="geometry_only",
            n_examples=4,
        )

        assert len(examples) == 0

    def test_fills_from_one_side_when_unbalanced(self, tmp_path):
        """Should fill extra examples from the available side when one is short."""
        batches_dir = tmp_path / "batches"

        # Source batch with only matches
        _make_batch(
            batches_dir,
            "source_batch",
            [("r1", "t1"), ("r2", "t2"), ("r3", "t3")],
            ground_truth_rows=[
                {"ref_id": "r1", "target_id": "t1", "label": "match"},
                {"ref_id": "r2", "target_id": "t2", "label": "match"},
                {"ref_id": "r3", "target_id": "t3", "label": "match"},
            ],
        )

        current = _make_batch(batches_dir, "current_batch", [("r4", "t4")])

        examples = select_few_shot_examples(
            batch_dir=current,
            variant="geometry_only",
            n_examples=4,
        )

        # Should get all available matches (no no_match available)
        assert len(examples) == 3
        assert all(e["label"] == "match" for e in examples)


# ---------------------------------------------------------------------------
# TestPrepareBatchPrompt
# ---------------------------------------------------------------------------


class TestPrepareBatchPrompt:
    """Tests for batch prompt building."""

    def test_contains_required_sections(self):
        prompt = prepare_batch_prompt(
            batch_dir=Path("/fake/batch"),
            variant="geometry_only",
            candidates=["r1__t1", "r2__t2"],
            few_shot_examples=[],
            output_path="labels/test/data.csv",
        )

        assert "transportation network segment matches" in prompt
        assert "LABELS:" in prompt
        assert "match:" in prompt
        assert "no_match:" in prompt
        assert "unsure:" in prompt
        assert "CRITICAL RULES:" in prompt
        assert "GEOMETRY FIRST" in prompt
        assert "IMAGE VARIANT:" in prompt
        assert "geometry_only.png" in prompt
        assert "BATCH PROCESSING INSTRUCTIONS:" in prompt
        assert "labels/test/data.csv" in prompt
        assert "r1__t1" in prompt
        assert "r2__t2" in prompt
        assert "Total candidates to process: 2" in prompt

    def test_few_shot_examples_formatted(self):
        examples = [
            {
                "ref_id": "ex1",
                "target_id": "tx1",
                "label": "match",
                "metadata_content": "test metadata",
                "source_batch_dir": Path("/fake"),
            },
        ]
        prompt = prepare_batch_prompt(
            batch_dir=Path("/fake/batch"),
            variant="road_context",
            candidates=["r1__t1"],
            few_shot_examples=examples,
            output_path="labels/test/data.csv",
        )

        assert "FEW-SHOT EXAMPLES:" in prompt
        assert "Example 1: ex1__tx1" in prompt
        assert "examples/ex1__tx1/road_context.png" in prompt
        assert "ex1,tx1,match,1.0,ground truth example" in prompt

    def test_variant_image_description_included(self):
        for variant_name in VARIANT_CONFIG:
            prompt = prepare_batch_prompt(
                batch_dir=Path("/fake"),
                variant=variant_name,
                candidates=["r1__t1"],
                few_shot_examples=[],
                output_path="out.csv",
            )
            desc = IMAGE_DESCRIPTIONS[variant_name]
            # The description text should appear in the prompt
            assert desc.split(":")[0] in prompt

    def test_no_explore_files_instruction_removed(self):
        """Should NOT contain the old 'Do NOT explore files' instruction."""
        prompt = prepare_batch_prompt(
            batch_dir=Path("/fake"),
            variant="geometry_only",
            candidates=["r1__t1"],
            few_shot_examples=[],
            output_path="out.csv",
        )
        assert "Do NOT explore files" not in prompt

    def test_batch_instructions_present(self):
        """Should contain agent self-direction instructions."""
        prompt = prepare_batch_prompt(
            batch_dir=Path("/fake"),
            variant="geometry_only",
            candidates=["r1__t1", "r2__t2", "r3__t3"],
            few_shot_examples=[],
            output_path="out.csv",
        )
        assert "Read `candidates/" in prompt
        assert "View `candidates/" in prompt
        assert "Write ALL results" in prompt
        assert "do NOT use commas in the reasoning" in prompt


# ---------------------------------------------------------------------------
# TestValidateOutputCsv
# ---------------------------------------------------------------------------


class TestValidateOutputCsv:
    """Tests for CSV output validation."""

    def test_valid_csv(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "ref_id,target_id,label,confidence,reasoning\n"
            "r1,t1,match,0.9,good overlap\n"
            "r2,t2,no_match,0.8,parallel roads\n"
        )

        expected = {("r1", "t1"), ("r2", "t2")}
        df, warnings = validate_output_csv(csv_path, expected)

        assert df is not None
        assert len(df) == 2
        assert len(warnings) == 0

    def test_missing_file(self, tmp_path):
        csv_path = tmp_path / "missing.csv"
        df, warnings = validate_output_csv(csv_path, {("r1", "t1")})

        assert df is None
        assert any("not found" in w for w in warnings)

    def test_missing_candidates(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("ref_id,target_id,label,confidence,reasoning\nr1,t1,match,0.9,test\n")

        expected = {("r1", "t1"), ("r2", "t2")}
        df, warnings = validate_output_csv(csv_path, expected)

        assert df is not None
        assert len(df) == 1
        assert any("1 candidates missing" in w for w in warnings)

    def test_invalid_label(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "ref_id,target_id,label,confidence,reasoning\nr1,t1,invalid_label,0.9,test\n"
        )

        df, warnings = validate_output_csv(csv_path, {("r1", "t1")})

        assert df is not None
        assert any("invalid label" in w for w in warnings)

    def test_invalid_confidence(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("ref_id,target_id,label,confidence,reasoning\nr1,t1,match,1.5,test\n")

        df, warnings = validate_output_csv(csv_path, {("r1", "t1")})

        assert df is not None
        assert any("invalid" in w.lower() for w in warnings)

    def test_duplicate_handling(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "ref_id,target_id,label,confidence,reasoning\n"
            "r1,t1,match,0.9,first\n"
            "r1,t1,no_match,0.8,second\n"
        )

        df, warnings = validate_output_csv(csv_path, {("r1", "t1")})

        assert df is not None
        assert len(df) == 1
        # Should keep last
        assert df.iloc[0]["label"] == "no_match"
        assert any("duplicate" in w.lower() for w in warnings)

    def test_missing_columns(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("ref_id,target_id,label\nr1,t1,match\n")

        df, warnings = validate_output_csv(csv_path, {("r1", "t1")})

        assert df is None
        assert any("Missing columns" in w for w in warnings)


# ---------------------------------------------------------------------------
# TestRunAgentBatch
# ---------------------------------------------------------------------------


class TestRunAgentBatch:
    """Tests for the batch runner (mocked subprocess)."""

    def test_basic_flow(self, tmp_path):
        """Should invoke Claude and produce output."""
        batches_dir = tmp_path / "batches"
        batch = _make_batch(
            batches_dir,
            "test_batch",
            [("r1", "t1"), ("r2", "t2")],
        )

        # Create a mock chunk CSV that the "agent" would write
        def mock_subprocess_run(cmd, **kwargs):
            # Simulate the agent writing a CSV file
            cwd = Path(kwargs.get("cwd", "."))
            # Find the output path from the prompt
            prompt_text = kwargs.get("input", "")
            # Extract output path from prompt
            for line in prompt_text.splitlines():
                if line.startswith("Output file:"):
                    rel_path = line.split(":", 1)[1].strip()
                    out_path = cwd / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        "ref_id,target_id,label,confidence,reasoning\n"
                        "r1,t1,match,0.9,good overlap\n"
                        "r2,t2,no_match,0.8,parallel\n"
                    )
                    break
            return MagicMock(returncode=0, stdout="Done", stderr="")

        with patch("matcher.agent_labeling.runner.subprocess.run", side_effect=mock_subprocess_run):
            run_agent_batch(
                model="opus",
                variant="geometry_only",
                batch_dir=batch,
                overwrite=True,
            )

        output = batch / "labels" / "claude_opus_geometry_only" / "data.csv"
        assert output.exists()
        df = pd.read_csv(output, dtype=str)
        assert len(df) == 2

    def test_resume_skips_labeled(self, tmp_path):
        """Existing labels should be skipped in resume mode."""
        batches_dir = tmp_path / "batches"
        batch = _make_batch(
            batches_dir,
            "test_batch",
            [("r1", "t1"), ("r2", "t2")],
        )

        # Pre-populate output with r1 already labeled
        labels_dir = batch / "labels" / "claude_sonnet_geometry_only"
        labels_dir.mkdir(parents=True)
        (labels_dir / "data.csv").write_text(
            "ref_id,target_id,label,confidence,reasoning\nr1,t1,match,0.9,test\n"
        )
        (labels_dir / "run.log").write_text("")
        (labels_dir / "raw_responses.log").write_text("")

        call_count = 0

        def mock_subprocess_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            cwd = Path(kwargs.get("cwd", "."))
            prompt_text = kwargs.get("input", "")
            for line in prompt_text.splitlines():
                if line.startswith("Output file:"):
                    rel_path = line.split(":", 1)[1].strip()
                    out_path = cwd / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        "ref_id,target_id,label,confidence,reasoning\n"
                        "r2,t2,no_match,0.8,different\n"
                    )
                    break
            return MagicMock(returncode=0, stdout="Done", stderr="")

        with patch("matcher.agent_labeling.runner.subprocess.run", side_effect=mock_subprocess_run):
            run_agent_batch(
                model="sonnet",
                variant="geometry_only",
                batch_dir=batch,
            )

        # Should only have invoked once (for r2 only)
        assert call_count == 1

        # Final output should have both labels
        output = batch / "labels" / "claude_sonnet_geometry_only" / "data.csv"
        df = pd.read_csv(output, dtype=str)
        assert len(df) == 2

    def test_prompt_file_written(self, tmp_path):
        """Should write prompt.txt for inspection."""
        batches_dir = tmp_path / "batches"
        batch = _make_batch(
            batches_dir,
            "test_batch",
            [("r1", "t1")],
        )

        def mock_subprocess_run(cmd, **kwargs):
            cwd = Path(kwargs.get("cwd", "."))
            prompt_text = kwargs.get("input", "")
            for line in prompt_text.splitlines():
                if line.startswith("Output file:"):
                    rel_path = line.split(":", 1)[1].strip()
                    out_path = cwd / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        "ref_id,target_id,label,confidence,reasoning\nr1,t1,match,0.9,test\n"
                    )
                    break
            return MagicMock(returncode=0, stdout="Done", stderr="")

        with patch("matcher.agent_labeling.runner.subprocess.run", side_effect=mock_subprocess_run):
            run_agent_batch(
                model="opus",
                variant="geometry_only",
                batch_dir=batch,
                overwrite=True,
            )

        prompt_file = batch / "labels" / "claude_opus_geometry_only" / "prompt.txt"
        assert prompt_file.exists()
        content = prompt_file.read_text()
        assert "BATCH PROCESSING INSTRUCTIONS" in content
        assert "r1__t1" in content


# ---------------------------------------------------------------------------
# TestVariantConfigCompleteness
# ---------------------------------------------------------------------------


class TestVariantConfigCompleteness:
    """Tests for configuration consistency."""

    def test_all_variants_have_descriptions(self):
        """Every variant in VARIANT_CONFIG should have an IMAGE_DESCRIPTIONS entry."""
        for variant_name in VARIANT_CONFIG:
            assert variant_name in IMAGE_DESCRIPTIONS, (
                f"Variant '{variant_name}' missing from IMAGE_DESCRIPTIONS"
            )

    def test_all_descriptions_have_config(self):
        """Every IMAGE_DESCRIPTIONS entry should have a VARIANT_CONFIG entry."""
        for desc_name in IMAGE_DESCRIPTIONS:
            assert desc_name in VARIANT_CONFIG, (
                f"Description '{desc_name}' missing from VARIANT_CONFIG"
            )
