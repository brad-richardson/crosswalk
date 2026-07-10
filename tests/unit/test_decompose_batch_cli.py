"""Follow-up guards for ``agent stitch-batch --decompose`` (#367 Mode B, #388).

Covers two seams flagged in the #388 adversarial review:

* ``--decompose-max-edges`` above the export backstop is a silent mini-void
  (sub-problems sized above the backstop pass the panel but size-gate at export,
  blocking the parent forever) — the CLI must fail loud.
* an evidence pack that silently fails to render for a decomposed sub-problem
  permanently blocks its parent's recomposition — ``missing_evidence_packs``
  must surface it so the CLI can refuse.
"""

from __future__ import annotations

from typer.testing import CliRunner

from crosswalk.agent_labeling.stitch_evidence import missing_evidence_packs
from crosswalk.cli import app
from crosswalk.config import settings

runner = CliRunner()


class TestDecomposeMaxEdgesValidation:
    def test_above_backstop_is_hard_error(self):
        backstop = settings.stitch_export_backstop_max_edges
        result = runner.invoke(
            app,
            [
                "agent",
                "stitch-batch",
                "some_dataset",
                "--decompose",
                "--decompose-max-edges",
                str(backstop + 20),
            ],
        )
        assert result.exit_code == 1
        assert "exceeds the export backstop" in result.stdout

    def test_zero_or_negative_budget_rejected(self):
        result = runner.invoke(
            app,
            [
                "agent",
                "stitch-batch",
                "some_dataset",
                "--decompose",
                "--decompose-max-edges",
                "-3",
            ],
        )
        assert result.exit_code == 1
        assert "must be >= 1" in result.stdout

    def test_budget_within_backstop_passes_validation(self):
        # A budget at/under the backstop clears validation; the command then
        # exits on the missing sidecar (exit 1) — but NOT with the mini-void
        # message, which is what this test pins.
        backstop = settings.stitch_export_backstop_max_edges
        result = runner.invoke(
            app,
            [
                "agent",
                "stitch-batch",
                "definitely_not_a_real_dataset",
                "--decompose",
                "--decompose-max-edges",
                str(backstop),
            ],
        )
        assert "exceeds the export backstop" not in result.stdout


class TestMissingEvidencePacks:
    def test_missing_subproblem_pack_is_flagged(self):
        packable = [
            {"group_id": "beef0001__pabc", "parent_group_id": "beef0001"},
            {"group_id": "beef0001__pdef", "parent_group_id": "beef0001"},
        ]
        # Only the first sub-problem got a pack.
        missing_subs, missing_other = missing_evidence_packs(packable, ["beef0001__pabc"])
        assert missing_subs == [("beef0001__pdef", "beef0001")]
        assert missing_other == []

    def test_missing_plain_group_is_other(self):
        packable = [{"group_id": "plain1"}, {"group_id": "plain2"}]
        missing_subs, missing_other = missing_evidence_packs(packable, ["plain1"])
        assert missing_subs == []
        assert missing_other == ["plain2"]

    def test_all_packed_reports_nothing(self):
        packable = [
            {"group_id": "beef0001__pabc", "parent_group_id": "beef0001"},
            {"group_id": "plain1"},
        ]
        missing_subs, missing_other = missing_evidence_packs(packable, ["beef0001__pabc", "plain1"])
        assert missing_subs == []
        assert missing_other == []
