"""Tests for per-voter panel bias monitoring (crosswalk.agent_labeling.panel_monitor).

Covers the synthetic-defect matrix (position anchor, constant confidence, n-floor
boundary, NONE/ABSTAIN exclusion), the wave-time surfacing, the CLI smoke path, and
a real-data integration sanity: on the committed vote provenance, voter ``agy`` must
trip POSITION_ANCHOR (11/12 valid ballots on the first-listed option "A"). That is
committed history — deterministic — so it doubles as the monitor's regression anchor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from crosswalk.agent_labeling.panel_monitor import (
    CONSTANT_CONFIDENCE,
    POSITION_ANCHOR,
    VoterStats,
    _position,
    compute_voter_stats,
    constant_confidence_tripped,
    load_vote_provenance,
    position_anchor_tripped,
    wave_position_anchor_warnings,
)
from crosswalk.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _votes(rows: list[dict]) -> pd.DataFrame:
    """Build a votes DataFrame, filling defaults for optional columns."""
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "group_id": r.get("group_id", f"g{i}"),
                "provider": r["provider"],
                "model": r.get("model", "m"),
                "choice": r["choice"],
                "confidence": r.get("confidence", 0.8),
                "error": r.get("error", None),
            }
        )
    return pd.DataFrame(out)


def _one(stats: list[VoterStats], provider: str) -> VoterStats:
    return next(s for s in stats if s.provider == provider)


# ---------------------------------------------------------------------------
# _position helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "choice,expected",
    [
        ("A", 0),
        ("B", 1),
        ("Z", 25),
        ("NONE", None),
        ("ABSTAIN", None),
        ("", None),
        ("a", None),  # lowercase is not a valid option letter
        ("AB", None),
        (None, None),
        (1.0, None),
    ],
)
def test_position_maps_letters_only(choice, expected):
    assert _position(choice) == expected


# ---------------------------------------------------------------------------
# POSITION_ANCHOR
# ---------------------------------------------------------------------------


def test_always_A_voter_trips_position_anchor():
    # 11 A + 1 B, varied confidence so ONLY the position defect is at play.
    rows = [{"provider": "agy", "choice": "A", "confidence": 0.5 + 0.01 * i} for i in range(11)]
    rows.append({"provider": "agy", "choice": "B", "confidence": 0.9})
    stat = _one(compute_voter_stats(_votes(rows)), "agy")

    assert stat.n_valid == 12
    assert stat.modal_position == 0
    assert stat.modal_letter == "A"
    assert stat.modal_position_share == pytest.approx(11 / 12)
    assert POSITION_ANCHOR in stat.alarms
    assert CONSTANT_CONFIDENCE not in stat.alarms  # confidence was varied


def test_healthy_voter_does_not_trip():
    # Even split across positions with varied confidence — a real voter.
    rows = []
    for i in range(6):
        rows.append({"provider": "claude", "choice": "A", "confidence": 0.4 + 0.05 * i})
    for i in range(6):
        rows.append({"provider": "claude", "choice": "B", "confidence": 0.5 + 0.05 * i})
    stat = _one(compute_voter_stats(_votes(rows)), "claude")

    assert stat.modal_position_share == pytest.approx(0.5)
    assert stat.alarms == []


def test_position_anchor_n_floor_boundary():
    # 9 all-A ballots: below the aggregate floor (10) -> no alarm even at share 1.0.
    stat9 = _one(
        compute_voter_stats(
            _votes(
                [{"provider": "v", "choice": "A", "confidence": 0.3 + 0.05 * i} for i in range(9)]
            )
        ),
        "v",
    )
    assert stat9.modal_position_share == pytest.approx(1.0)
    assert POSITION_ANCHOR not in stat9.alarms
    assert not position_anchor_tripped(stat9)

    # 10 all-A ballots: exactly at the floor -> trips.
    stat10 = _one(
        compute_voter_stats(
            _votes(
                [{"provider": "v", "choice": "A", "confidence": 0.3 + 0.05 * i} for i in range(10)]
            )
        ),
        "v",
    )
    assert POSITION_ANCHOR in stat10.alarms
    assert position_anchor_tripped(stat10)


# ---------------------------------------------------------------------------
# CONSTANT_CONFIDENCE
# ---------------------------------------------------------------------------


def test_constant_confidence_trips():
    # Confidence pinned at 0.95, but positions spread so ONLY the confidence
    # defect is at play (share below the anchor threshold).
    rows = []
    for _ in range(6):
        rows.append({"provider": "rs", "choice": "A", "confidence": 0.95})
    for _ in range(6):
        rows.append({"provider": "rs", "choice": "B", "confidence": 0.95})
    stat = _one(compute_voter_stats(_votes(rows)), "rs")

    assert stat.conf_std == pytest.approx(0.0)
    assert CONSTANT_CONFIDENCE in stat.alarms
    assert POSITION_ANCHOR not in stat.alarms
    assert constant_confidence_tripped(stat)


def test_varied_confidence_does_not_trip_constant():
    rows = [
        {"provider": "v", "choice": "A" if i % 2 else "B", "confidence": 0.4 + 0.04 * i}
        for i in range(12)
    ]
    stat = _one(compute_voter_stats(_votes(rows)), "v")
    assert stat.conf_std > 0.02
    assert CONSTANT_CONFIDENCE not in stat.alarms


# ---------------------------------------------------------------------------
# NONE / ABSTAIN exclusion + confidence over valid ballots only
# ---------------------------------------------------------------------------


def test_none_and_abstain_excluded_from_position_stats():
    rows = [{"provider": "v", "choice": "A", "confidence": 0.9} for _ in range(10)]
    rows += [{"provider": "v", "choice": "NONE", "confidence": 0.1} for _ in range(3)]
    rows += [
        {"provider": "v", "choice": "ABSTAIN", "confidence": 0.0, "error": "timeout after 240s"}
        for _ in range(2)
    ]
    stat = _one(compute_voter_stats(_votes(rows)), "v")

    assert stat.n_ballots == 15
    assert stat.n_valid == 10  # only the 10 A ballots
    assert stat.n_abstain == 5
    assert stat.position_counts == {0: 10}  # no position entry for NONE/ABSTAIN
    assert stat.modal_position_share == pytest.approx(1.0)
    # Confidence stats over VALID ballots only: the forced 0.0s must not drag the mean.
    assert stat.conf_mean == pytest.approx(0.9)
    assert stat.conf_std == pytest.approx(0.0)


def test_errored_row_is_not_valid_even_with_letter_choice():
    rows = [{"provider": "v", "choice": "A", "confidence": 0.9} for _ in range(3)]
    rows += [{"provider": "v", "choice": "A", "confidence": 0.9, "error": "parse/validation: boom"}]
    stat = _one(compute_voter_stats(_votes(rows)), "v")
    assert stat.n_valid == 3
    assert stat.n_abstain == 1


# ---------------------------------------------------------------------------
# Dissent + calibration proxy (consensus join)
# ---------------------------------------------------------------------------


def test_dissent_and_calibration_from_consensus():
    votes = _votes(
        [
            {"provider": "v", "group_id": "g1", "choice": "A", "confidence": 0.9},
            {"provider": "v", "group_id": "g2", "choice": "A", "confidence": 0.6},  # dissent
            {"provider": "v", "group_id": "g3", "choice": "B", "confidence": 0.8},
            {
                "provider": "v",
                "group_id": "g4",
                "choice": "A",
                "confidence": 0.7,
            },  # no-consensus grp
        ]
    )
    consensus = pd.DataFrame(
        [
            {"group_id": "g1", "choice": "A"},
            {"group_id": "g2", "choice": "B"},  # v dissented here
            {"group_id": "g3", "choice": "B"},
            {"group_id": "g4", "choice": ""},  # undecided -> excluded from dissent denom
        ]
    )
    stat = _one(compute_voter_stats(votes, consensus), "v")

    assert stat.n_decided == 3  # g4 excluded (no consensus letter)
    assert stat.n_dissent == 1  # only g2
    assert stat.dissent_rate == pytest.approx(1 / 3)
    assert stat.conf_on_agree == pytest.approx((0.9 + 0.8) / 2)
    assert stat.conf_on_dissent == pytest.approx(0.6)
    assert stat.calibration_gap == pytest.approx((0.85) - 0.6)


def test_stats_work_without_consensus():
    stat = _one(
        compute_voter_stats(_votes([{"provider": "v", "choice": "A"} for _ in range(10)])),
        "v",
    )
    assert stat.n_decided == 0
    assert stat.n_dissent == 0
    assert POSITION_ANCHOR in stat.alarms  # position alarm needs no consensus


def test_quad_panel_kimi_and_muse_are_distinct_voter_rows():
    """A 4-voter (quad-candidate) wave produces FOUR distinct per-voter rows.

    Kimi (opencode) and Muse (muse) ride the same transport but carry distinct
    provider names, so ``compute_voter_stats`` (grouped by provider) must NOT pool
    them into one row — otherwise the monitor would silently average a Meta model
    with a Moonshot model. Every voter votes "A" on every group here, so pooling
    would still yield one row; the assertion that BOTH names surface as their own
    rows is what catches a regression.
    """
    rows = []
    for i in range(6):
        for prov, model in [
            ("claude", "claude-opus-4-8"),
            ("codex", "gpt-5.6-sol"),
            ("opencode", "openrouter/moonshotai/kimi-k2.6"),
            ("muse", "meta/muse-spark-1.1"),
        ]:
            rows.append({"provider": prov, "model": model, "group_id": f"g{i}", "choice": "A"})
    stats = compute_voter_stats(_votes(rows))
    providers = {s.provider for s in stats}
    assert {"claude", "codex", "opencode", "muse"} <= providers
    # Kimi and Muse are SEPARATE rows carrying their own model strings (not pooled).
    assert _one(stats, "opencode").model == "openrouter/moonshotai/kimi-k2.6"
    assert _one(stats, "muse").model == "meta/muse-spark-1.1"


def test_cli_panel_stats_shows_muse_as_its_own_row(tmp_path, monkeypatch):
    """`crosswalk agent panel-stats` renders the Muse voter as its own row on a
    4-voter provenance snapshot — distinct from the opencode/Kimi seat."""
    # Widen the rich console so the 8-char "opencode" voter cell isn't truncated
    # to an ellipsis in the narrow 12-column table (default non-tty width is 80).
    monkeypatch.setenv("COLUMNS", "400")
    rows = []
    for i in range(6):
        for prov, model in [
            ("claude", "claude-opus-4-8"),
            ("codex", "gpt-5.6-sol"),
            ("opencode", "openrouter/moonshotai/kimi-k2.6"),
            ("muse", "meta/muse-spark-1.1"),
        ]:
            rows.append(
                {
                    "provider": prov,
                    "model": model,
                    "group_id": f"g{i}",
                    "choice": "A",
                    "confidence": 0.5 + 0.05 * i,
                }
            )
    votes = _votes(rows)
    consensus = pd.DataFrame([{"group_id": f"g{i}", "choice": "A"} for i in range(6)])
    _write_provenance(tmp_path, "quad_demo", votes, consensus)

    result = runner.invoke(app, ["agent", "panel-stats", "--data-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    # Both opencode/Kimi and muse surface as their own voter rows (not pooled).
    assert "muse" in result.output
    assert "opencode" in result.output


# ---------------------------------------------------------------------------
# Wave-time surfacing
# ---------------------------------------------------------------------------


def test_wave_uses_lower_n_floor():
    # 8 all-A ballots: below the aggregate floor (10) but at/above the wave floor (8).
    rows = [{"provider": "agy", "choice": "A", "confidence": 0.3 + 0.05 * i} for i in range(8)]
    votes = _votes(rows)

    stat = _one(compute_voter_stats(votes), "agy")
    assert POSITION_ANCHOR not in stat.alarms  # aggregate view stays quiet at n=8

    warnings = wave_position_anchor_warnings(votes)
    assert len(warnings) == 1
    assert "agy" in warnings[0]
    assert POSITION_ANCHOR in warnings[0]


def test_wave_quiet_for_healthy_and_small_voters():
    rows = []
    for i in range(4):
        rows.append({"provider": "claude", "choice": "A", "confidence": 0.4 + 0.05 * i})
    for i in range(4):
        rows.append({"provider": "claude", "choice": "B", "confidence": 0.6 + 0.05 * i})
    # A tiny voter with a perfect anchor but too few ballots to count.
    rows += [{"provider": "tiny", "choice": "A", "confidence": 0.9} for _ in range(3)]
    assert wave_position_anchor_warnings(_votes(rows)) == []


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def _write_provenance(root: Path, dataset: str, votes: pd.DataFrame, consensus: pd.DataFrame):
    d = root / "labels" / "votes" / f"dataset={dataset}"
    d.mkdir(parents=True, exist_ok=True)
    votes.to_csv(d / "votes.csv", index=False)
    consensus.to_csv(d / "consensus.csv", index=False)


def test_cli_panel_stats_smoke(tmp_path):
    votes = _votes(
        [
            {"provider": "agy", "group_id": f"g{i}", "choice": "A", "confidence": 0.95}
            for i in range(12)
        ]
        + [
            {
                "provider": "claude",
                "group_id": f"g{i}",
                "choice": "A" if i % 2 else "B",
                "confidence": 0.4 + 0.03 * i,
            }
            for i in range(12)
        ]
    )
    consensus = pd.DataFrame([{"group_id": f"g{i}", "choice": "A"} for i in range(12)])
    _write_provenance(tmp_path, "demo", votes, consensus)

    result = runner.invoke(app, ["agent", "panel-stats", "--data-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "agy" in result.output
    assert POSITION_ANCHOR in result.output
    assert CONSTANT_CONFIDENCE in result.output

    # --strict turns a tripped alarm into a nonzero exit (for CI/cron).
    strict = runner.invoke(app, ["agent", "panel-stats", "--data-root", str(tmp_path), "--strict"])
    assert strict.exit_code == 1, strict.output


def test_cli_panel_stats_no_alarms_exit_zero_under_strict(tmp_path):
    votes = _votes(
        [
            {
                "provider": "claude",
                "group_id": f"g{i}",
                "choice": "A" if i % 2 else "B",
                "confidence": 0.4 + 0.03 * i,
            }
            for i in range(12)
        ]
    )
    consensus = pd.DataFrame([{"group_id": f"g{i}", "choice": "A"} for i in range(12)])
    _write_provenance(tmp_path, "demo", votes, consensus)

    result = runner.invoke(app, ["agent", "panel-stats", "--data-root", str(tmp_path), "--strict"])
    assert result.exit_code == 0, result.output
    assert "No alarms tripped" in result.output


def test_cli_panel_stats_empty(tmp_path):
    result = runner.invoke(app, ["agent", "panel-stats", "--data-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "No committed votes" in result.output


# ---------------------------------------------------------------------------
# Real-data integration sanity (committed provenance is deterministic history)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (REPO_ROOT / "labels" / "votes").exists(),
    reason="committed vote provenance not present",
)
def test_real_data_agy_trips_position_anchor():
    votes_df, consensus_df = load_vote_provenance(REPO_ROOT)
    # Freeze the sample to the batches agy actually voted in (agy was retired from
    # the default panel afterwards), so later waves can't shift these exact counts.
    agy_era = {"boston_test5", "de_berlin_roads_w0707", "us_seattle_sidewalks_w0707"}
    votes_df = votes_df[votes_df["source_batch"].isin(agy_era)]
    consensus_df = consensus_df[consensus_df["source_batch"].isin(agy_era)]
    stats = compute_voter_stats(votes_df, consensus_df)
    by_provider = {s.provider: s for s in stats}

    assert "agy" in by_provider, "expected agy in committed panel provenance"
    agy = by_provider["agy"]
    # Committed history: agy voted the first-listed option "A" in 11/12 valid ballots.
    assert agy.modal_letter == "A"
    assert agy.n_valid == 12
    assert agy.modal_position_share > 0.6
    assert POSITION_ANCHOR in agy.alarms
    assert CONSTANT_CONFIDENCE in agy.alarms  # flat 0.95

    # The calibrated voters must NOT trip the position anchor.
    for name in ("claude", "codex"):
        if name in by_provider:
            assert POSITION_ANCHOR not in by_provider[name].alarms
