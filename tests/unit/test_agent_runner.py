"""Tests for agent_labeling.runner module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from matcher.agent_labeling.runner import (
    IMAGE_DESCRIPTIONS,
    VARIANT_CONFIG,
    build_prompt,
    parse_response,
    run_agent_batch,
)


class TestBuildPrompt:
    """Tests for prompt building."""

    def test_build_prompt_with_variant(self):
        prompt = build_prompt(
            "ref_1",
            "target_1",
            "metadata: test",
            variant="subline_geometry_only",
        )
        assert "subline_geometry_only.png" in prompt
        assert "alignment view" in prompt
        assert "ref_1" in prompt
        assert "target_1" in prompt
        assert "metadata: test" in prompt

    def test_build_prompt_legacy(self):
        prompt = build_prompt("ref_1", "target_1", "metadata: test", variant=None)
        assert "satellite.png" in prompt
        assert "geometry.png" in prompt
        assert "ref_1" in prompt

    def test_build_prompt_svg_variant(self):
        prompt = build_prompt(
            "ref_1",
            "target_1",
            "metadata: test",
            variant="road_context_svg",
            svg_content='<svg xmlns="http://www.w3.org/2000/svg"></svg>',
        )
        assert "SVG content inline below" in prompt
        assert "<svg" in prompt

    def test_build_prompt_all_variants(self):
        """Every variant in VARIANT_CONFIG should produce a valid prompt."""
        for variant_name in VARIANT_CONFIG:
            prompt = build_prompt("r", "t", "meta", variant=variant_name)
            assert "r,t" in prompt
            assert "meta" in prompt


class TestParseResponse:
    """Tests for response parsing."""

    def test_parse_response_csv_line(self):
        raw = "Some preamble text\nref_1,target_1,match,0.9,lines overlap well\nExtra output"
        result = parse_response(raw, "ref_1", "target_1")
        assert result == "ref_1,target_1,match,0.9,lines overlap well"

    def test_parse_response_keyword_fallback(self):
        raw = "Based on the geometry, this is clearly a no_match."
        result = parse_response(raw, "ref_1", "target_1")
        assert result == "ref_1,target_1,no_match,0.5,parsed"

    def test_parse_response_match_keyword(self):
        raw = "The segments match."
        result = parse_response(raw, "ref_1", "target_1")
        assert result == "ref_1,target_1,match,0.5,parsed"

    def test_parse_response_no_label(self):
        raw = "I cannot determine the relationship between these segments."
        result = parse_response(raw, "ref_1", "target_1")
        assert result is None

    def test_parse_response_empty(self):
        assert parse_response(None, "r", "t") is None
        assert parse_response("", "r", "t") is None


class TestInvokeAgent:
    """Tests for agent invocation (mocked subprocess)."""

    @patch("matcher.agent_labeling.runner.subprocess.run")
    def test_invoke_agent_claude(self, mock_run):
        from matcher.agent_labeling.runner import invoke_agent

        mock_run.return_value = MagicMock(
            stdout="ref_1,target_1,match,0.9,test",
            stderr="",
            returncode=0,
        )

        result = invoke_agent(
            agent="claude",
            model="sonnet",
            prompt="Test prompt",
            candidate_dir=Path("/tmp/test"),
            variant_filename="geometry_only.png",
        )

        assert result is not None
        assert "ref_1" in result
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "claude" in cmd
        assert "-p" in cmd
        assert "--model" in cmd
        assert "sonnet" in cmd

    @patch("matcher.agent_labeling.runner.subprocess.run")
    def test_invoke_agent_gemini(self, mock_run):
        from matcher.agent_labeling.runner import invoke_agent

        mock_run.return_value = MagicMock(
            stdout="ref_1,target_1,match,0.8,gemini result",
            stderr="",
            returncode=0,
        )

        result = invoke_agent(
            agent="gemini",
            model="flash",
            prompt="Test prompt",
            image_path=Path("/tmp/test/geometry_only.png"),
            candidate_dir=Path("/tmp/test"),
        )

        assert result is not None
        mock_run.assert_called_once()


class TestRunAgentBatch:
    """Tests for batch runner logic."""

    def test_run_agent_batch_resume(self, tmp_path):
        """Existing labels should be skipped in resume mode."""
        batch_dir = tmp_path / "test_batch"
        candidates_dir = batch_dir / "candidates"
        candidates_dir.mkdir(parents=True)

        # Create 2 candidates
        for pair in ["ref_1__target_1", "ref_2__target_2"]:
            cdir = candidates_dir / pair
            cdir.mkdir()
            (cdir / "metadata.yaml").write_text("candidate: test")
            (cdir / "geometry_only.png").write_bytes(b"fake png")

        # Pre-populate output with ref_1 already labeled
        labels_dir = batch_dir / "labels" / "claude_sonnet_geometry_only"
        labels_dir.mkdir(parents=True)
        (labels_dir / "data.csv").write_text(
            "ref_id,target_id,label,confidence,reasoning\nref_1,target_1,match,0.9,test\n"
        )

        # Mock the agent invocation to track calls
        with patch("matcher.agent_labeling.runner.invoke_agent") as mock_invoke:
            mock_invoke.return_value = "ref_2,target_2,no_match,0.8,different roads"

            run_agent_batch(
                agent="claude",
                model="sonnet",
                variant="geometry_only",
                batch_dir=batch_dir,
            )

            # Should only invoke for ref_2 (ref_1 was already labeled)
            assert mock_invoke.call_count == 1

    def test_run_agent_batch_bail_after(self, tmp_path):
        """Should bail after N consecutive failures."""
        batch_dir = tmp_path / "test_batch"
        candidates_dir = batch_dir / "candidates"
        candidates_dir.mkdir(parents=True)

        # Create 5 candidates
        for i in range(5):
            cdir = candidates_dir / f"ref_{i}__target_{i}"
            cdir.mkdir()
            (cdir / "metadata.yaml").write_text("candidate: test")
            (cdir / "geometry_only.png").write_bytes(b"fake png")

        with patch("matcher.agent_labeling.runner.invoke_agent") as mock_invoke:
            # Return gibberish (unparseable) to trigger failures
            mock_invoke.return_value = "total gibberish output"

            run_agent_batch(
                agent="claude",
                model="sonnet",
                variant="geometry_only",
                batch_dir=batch_dir,
                bail_after=2,
                overwrite=True,
            )

            # Should bail after 2 consecutive failures
            assert mock_invoke.call_count == 2


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
