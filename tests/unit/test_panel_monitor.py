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
    OPTIMIZER_ANCHOR,
    POSITION_ANCHOR,
    VoterStats,
    _position,
    compute_voter_stats,
    constant_confidence_tripped,
    load_evidence_provenance,
    load_vote_provenance,
    optimizer_anchor_tripped,
    position_anchor_tripped,
    wave_optimizer_anchor_warnings,
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


def _evidence(rows: list[dict]) -> pd.DataFrame:
    """Build a per-group evidence frame (see load_evidence_provenance)."""
    return pd.DataFrame(
        [
            {
                "group_id": r["group_id"],
                "optimizer_letter": r.get("optimizer_letter"),
                "option_shuffled": r.get("option_shuffled", False),
            }
            for r in rows
        ]
    )


def test_position_anchor_suppressed_for_shuffled_era_ballots():
    """Shuffled-era letters are content-free: they must not feed POSITION stats.

    12 all-A ballots would trip POSITION_ANCHOR — but when every pack was
    shuffled, "A" carries no positional meaning, so the alarm must stay quiet
    and the exclusion must be visible in the era-aware counters.
    """
    rows = [
        {"provider": "v", "group_id": f"g{i}", "choice": "A", "confidence": 0.3 + 0.05 * i}
        for i in range(12)
    ]
    evidence = _evidence(
        [{"group_id": f"g{i}", "optimizer_letter": "B", "option_shuffled": True} for i in range(12)]
    )
    stat = _one(compute_voter_stats(_votes(rows), evidence=evidence), "v")

    assert stat.n_valid == 12  # still valid ballots for everything else
    assert stat.n_position == 0  # ...but none are position-eligible
    assert stat.n_shuffled == 12
    assert stat.position_counts == {}
    assert stat.modal_position is None
    assert POSITION_ANCHOR not in stat.alarms
    assert not position_anchor_tripped(stat)
    assert wave_position_anchor_warnings(_votes(rows), evidence=evidence) == []


def test_position_anchor_mixed_eras_uses_only_unshuffled_ballots():
    """Mixed shuffled/unshuffled pools: position stats run over the unshuffled
    subset only, and the n-floor applies to THAT subset (a voter with mostly
    shuffled-era ballots cannot trip on a handful of eligible ones)."""
    # 6 unshuffled all-A + 30 shuffled all-A: without the era split this would
    # scream POSITION_ANCHOR at share 1.0 over n=36.
    rows = [
        {"provider": "v", "group_id": f"g{i}", "choice": "A", "confidence": 0.3 + 0.01 * i}
        for i in range(36)
    ]
    evidence = _evidence(
        [
            {"group_id": f"g{i}", "option_shuffled": i >= 6, "optimizer_letter": "A"}
            for i in range(36)
        ]
    )
    stat = _one(compute_voter_stats(_votes(rows), evidence=evidence), "v")

    assert stat.n_valid == 36
    assert stat.n_position == 6
    assert stat.n_shuffled == 30
    assert stat.modal_position_share == pytest.approx(1.0)  # over the 6 eligible
    # Aggregate floor is 10 position-eligible ballots: 6 does not clear it.
    assert POSITION_ANCHOR not in stat.alarms


def test_position_stats_unchanged_without_evidence():
    """No evidence frame -> every ballot counts as unshuffled (legacy behavior)."""
    rows = [
        {"provider": "v", "group_id": f"g{i}", "choice": "A", "confidence": 0.3 + 0.05 * i}
        for i in range(12)
    ]
    stat = _one(compute_voter_stats(_votes(rows)), "v")
    assert stat.n_position == stat.n_valid == 12
    assert stat.n_shuffled == 0
    assert POSITION_ANCHOR in stat.alarms
    # And optimizer stats stay unknown rather than fabricated.
    assert stat.n_optimizer_known == 0
    assert stat.optimizer_agree_share != stat.optimizer_agree_share  # NaN
    assert OPTIMIZER_ANCHOR not in stat.alarms


# ---------------------------------------------------------------------------
# OPTIMIZER_ANCHOR
# ---------------------------------------------------------------------------


def _optimizer_rows(n_agree: int, n_disagree: int, provider: str = "v") -> tuple:
    """n_agree ballots on the optimizer letter + n_disagree off it, with evidence."""
    rows, ev = [], []
    for i in range(n_agree + n_disagree):
        opt_letter = "A" if i % 2 else "B"  # optimizer letter varies across groups
        choice = opt_letter if i < n_agree else ("B" if opt_letter == "A" else "A")
        rows.append(
            {
                "provider": provider,
                "group_id": f"g{i}",
                "choice": choice,
                "confidence": 0.3 + 0.01 * i,
            }
        )
        ev.append({"group_id": f"g{i}", "optimizer_letter": opt_letter})
    return _votes(rows), _evidence(ev)


def test_optimizer_anchor_trips_on_rubber_stamp():
    """A voter agreeing with the optimizer on ~92% of ballots trips the alarm,
    even though its LETTER positions are split (POSITION_ANCHOR stays quiet)."""
    votes, evidence = _optimizer_rows(11, 1)
    stat = _one(compute_voter_stats(votes, evidence=evidence), "v")

    assert stat.n_optimizer_known == 12
    assert stat.n_optimizer_agree == 11
    assert stat.optimizer_agree_share == pytest.approx(11 / 12)
    assert OPTIMIZER_ANCHOR in stat.alarms
    assert optimizer_anchor_tripped(stat)
    assert POSITION_ANCHOR not in stat.alarms  # letters alternate A/B


def test_optimizer_anchor_quiet_for_healthy_agreement():
    """Base-rate agreement (the optimizer IS right most of the time) must not
    trip: ~72% agreement — the committed healthy-seat level — stays quiet."""
    votes, evidence = _optimizer_rows(13, 5)  # 13/18 ~ 0.72
    stat = _one(compute_voter_stats(votes, evidence=evidence), "v")

    assert stat.optimizer_agree_share == pytest.approx(13 / 18)
    assert OPTIMIZER_ANCHOR not in stat.alarms
    assert not optimizer_anchor_tripped(stat)


def test_optimizer_anchor_n_floor():
    """Below the aggregate floor (10 known-optimizer ballots) the alarm holds."""
    votes, evidence = _optimizer_rows(9, 0)
    stat = _one(compute_voter_stats(votes, evidence=evidence), "v")
    assert stat.n_optimizer_known == 9
    assert stat.optimizer_agree_share == pytest.approx(1.0)
    assert OPTIMIZER_ANCHOR not in stat.alarms
    assert not optimizer_anchor_tripped(stat)


def test_optimizer_anchor_survives_shuffle():
    """The alarm keys on the pack's recorded optimizer letter, so an optimizer
    rubber stamp is caught even when every pack was shuffled (where
    POSITION_ANCHOR is structurally blind)."""
    rows, ev = [], []
    for i in range(12):
        opt_letter = "A" if i % 2 else "B"
        rows.append(
            {
                "provider": "v",
                "group_id": f"g{i}",
                "choice": opt_letter,
                "confidence": 0.3 + 0.01 * i,
            }
        )
        ev.append({"group_id": f"g{i}", "optimizer_letter": opt_letter, "option_shuffled": True})
    stat = _one(compute_voter_stats(_votes(rows), evidence=_evidence(ev)), "v")

    assert stat.n_position == 0  # POSITION view is (correctly) blind here
    assert POSITION_ANCHOR not in stat.alarms
    assert stat.optimizer_agree_share == pytest.approx(1.0)
    assert OPTIMIZER_ANCHOR in stat.alarms


def test_wave_optimizer_anchor_uses_lower_n_floor():
    """Wave-time surfacing mirrors POSITION_ANCHOR: the lower wave floor (8)
    applies, and without an evidence frame no warning can fire."""
    votes, evidence = _optimizer_rows(8, 0)
    stat = _one(compute_voter_stats(votes, evidence=evidence), "v")
    assert OPTIMIZER_ANCHOR not in stat.alarms  # aggregate floor (10) not met

    warnings = wave_optimizer_anchor_warnings(votes, evidence=evidence)
    assert len(warnings) == 1
    assert OPTIMIZER_ANCHOR in warnings[0]
    assert "v" in warnings[0]

    assert wave_optimizer_anchor_warnings(votes) == []  # no evidence -> no signal


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


def test_none_excluded_from_position_but_abstain_split_out():
    """NONE is a decisive verdict (own accounting + confidence), ABSTAIN a failure.

    NONE occupies no letter slot so it stays out of POSITION statistics, but its
    real confidence joins the confidence view; ABSTAIN's synthetic 0.0 stays out.
    """
    rows = [{"provider": "v", "choice": "A", "confidence": 0.9} for _ in range(10)]
    rows += [{"provider": "v", "choice": "NONE", "confidence": 0.1} for _ in range(3)]
    rows += [
        {"provider": "v", "choice": "ABSTAIN", "confidence": 0.0, "error": "timeout after 240s"}
        for _ in range(2)
    ]
    stat = _one(compute_voter_stats(_votes(rows)), "v")

    assert stat.n_ballots == 15
    assert stat.n_valid == 10  # only the 10 A letter ballots
    assert stat.n_none == 3  # NONE broken out from failures
    assert stat.n_abstain == 2  # only the ABSTAIN/error rows
    assert stat.n_ballots == stat.n_valid + stat.n_none + stat.n_abstain
    assert stat.position_counts == {0: 10}  # no position entry for NONE/ABSTAIN
    assert stat.modal_position_share == pytest.approx(1.0)  # denominator is n_valid
    # Confidence spans the 13 CAST ballots (letters + NONE); the forced ABSTAIN 0.0
    # does not drag it, but the real NONE 0.1s do.
    assert stat.n_scored == 13
    assert stat.conf_mean == pytest.approx((10 * 0.9 + 3 * 0.1) / 13)
    assert stat.conf_std > 0.0


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


def test_none_counts_as_a_decisive_dissenting_or_agreeing_ballot():
    """NONE is a first-class verdict in the decided/dissent view.

    - NONE against a letter consensus  -> dissent
    - a letter against a NONE consensus -> dissent
    - NONE agreeing with a NONE consensus -> agreement
    """
    votes = _votes(
        [
            {"provider": "v", "group_id": "g1", "choice": "NONE", "confidence": 0.7},  # vs A
            {"provider": "v", "group_id": "g2", "choice": "A", "confidence": 0.6},  # vs NONE
            {"provider": "v", "group_id": "g3", "choice": "NONE", "confidence": 0.9},  # vs NONE
            {"provider": "v", "group_id": "g4", "choice": "A", "confidence": 0.8},  # vs A (agree)
        ]
    )
    consensus = pd.DataFrame(
        [
            {"group_id": "g1", "choice": "A"},
            {"group_id": "g2", "choice": "NONE"},
            {"group_id": "g3", "choice": "NONE"},
            {"group_id": "g4", "choice": "A"},
        ]
    )
    stat = _one(compute_voter_stats(votes, consensus), "v")

    assert stat.n_none == 2  # the two NONE ballots are cast verdicts
    assert stat.n_decided == 4  # all four groups reached a verdict (letter or NONE)
    assert stat.n_dissent == 2  # g1 (NONE vs A) and g2 (A vs NONE)
    assert stat.dissent_rate == pytest.approx(0.5)
    # Confidence/calibration span NONE too: agree = g3,g4 (0.9,0.8); dissent = g1,g2.
    assert stat.conf_on_agree == pytest.approx((0.9 + 0.8) / 2)
    assert stat.conf_on_dissent == pytest.approx((0.7 + 0.6) / 2)
    assert stat.calibration_gap == pytest.approx(0.85 - 0.65)


def test_none_heavy_voter_does_not_trip_position_anchor():
    """A voter that reject-alls (NONE) most groups but spreads its few letters must
    NOT trip POSITION_ANCHOR: NONE holds no letter slot, so it cannot make the voter
    look slot-anchored."""
    rows = [{"provider": "v", "choice": "NONE", "confidence": 0.3 + 0.01 * i} for i in range(12)]
    rows += [{"provider": "v", "choice": "A", "confidence": 0.5}]
    rows += [{"provider": "v", "choice": "B", "confidence": 0.6}]
    stat = _one(compute_voter_stats(_votes(rows)), "v")

    assert stat.n_none == 12
    assert stat.n_valid == 2  # only the two letter ballots feed POSITION stats
    assert stat.position_counts == {0: 1, 1: 1}
    assert stat.modal_position_share == pytest.approx(0.5)
    assert POSITION_ANCHOR not in stat.alarms


def test_constant_confidence_catches_flat_reject_all_voter():
    """A voter that reject-alls (NONE) at a flat confidence is a rubber stamp and must
    trip CONSTANT_CONFIDENCE even with zero letter ballots — the n-floor is scored
    (cast) ballots, not letters."""
    rows = [{"provider": "v", "choice": "NONE", "confidence": 0.95} for _ in range(12)]
    stat = _one(compute_voter_stats(_votes(rows)), "v")

    assert stat.n_valid == 0
    assert stat.n_none == 12
    assert stat.n_scored == 12
    assert stat.conf_std == pytest.approx(0.0)
    assert CONSTANT_CONFIDENCE in stat.alarms
    assert constant_confidence_tripped(stat)


def test_abstain_still_excluded_from_confidence():
    """ABSTAIN's synthetic 0.0 confidence never enters the confidence view."""
    rows = [{"provider": "v", "choice": "A", "confidence": 0.9} for _ in range(6)]
    rows += [
        {"provider": "v", "choice": "ABSTAIN", "confidence": 0.0, "error": "timeout"}
        for _ in range(4)
    ]
    stat = _one(compute_voter_stats(_votes(rows)), "v")

    assert stat.n_abstain == 4
    assert stat.n_none == 0
    assert stat.n_scored == 6  # only the six real letter ballots
    assert stat.conf_mean == pytest.approx(0.9)
    assert stat.conf_std == pytest.approx(0.0)


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

    Kimi (kimi) and Muse (muse) ride the same transport but carry distinct
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
            ("codex", "gpt-5.6-terra"),
            ("kimi", "openrouter/moonshotai/kimi-k2.6"),
            ("muse", "meta/muse-spark-1.1"),
        ]:
            rows.append({"provider": prov, "model": model, "group_id": f"g{i}", "choice": "A"})
    stats = compute_voter_stats(_votes(rows))
    providers = {s.provider for s in stats}
    assert {"claude", "codex", "kimi", "muse"} <= providers
    # Kimi and Muse are SEPARATE rows carrying their own model strings (not pooled).
    assert _one(stats, "kimi").model == "openrouter/moonshotai/kimi-k2.6"
    assert _one(stats, "muse").model == "meta/muse-spark-1.1"


def test_cli_panel_stats_shows_muse_as_its_own_row(tmp_path, monkeypatch):
    """`crosswalk agent panel-stats` renders the Muse voter as its own row on a
    4-voter provenance snapshot — distinct from the kimi/Kimi seat."""
    # Widen the rich console so the voter/model cells aren't truncated to an
    # ellipsis in the narrow 12-column table (default non-tty width is 80).
    monkeypatch.setenv("COLUMNS", "400")
    rows = []
    for i in range(6):
        for prov, model in [
            ("claude", "claude-opus-4-8"),
            ("codex", "gpt-5.6-terra"),
            ("kimi", "openrouter/moonshotai/kimi-k2.6"),
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
    # Both kimi/Kimi and muse surface as their own voter rows (not pooled).
    assert "muse" in result.output
    assert "kimi" in result.output


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


def _write_provenance(
    root: Path,
    dataset: str,
    votes: pd.DataFrame,
    consensus: pd.DataFrame,
    evidence: pd.DataFrame | None = None,
):
    d = root / "labels" / "votes" / f"dataset={dataset}"
    d.mkdir(parents=True, exist_ok=True)
    votes.to_csv(d / "votes.csv", index=False)
    consensus.to_csv(d / "consensus.csv", index=False)
    if evidence is not None:
        evidence.to_csv(d / "evidence.csv", index=False)


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
# Evidence provenance loading (optimizer letters + shuffled-era flags)
# ---------------------------------------------------------------------------


def test_load_evidence_provenance_parses_optimizer_and_era(tmp_path):
    import json

    d = tmp_path / "labels" / "votes" / "dataset=demo"
    d.mkdir(parents=True)
    rows = [
        # Unshuffled era: no option_order key at all (pre-shuffle pack).
        {
            "source_batch": "b1",
            "group_id": "g1",
            "evidence": json.dumps({"optimizer_letter": "A"}),
        },
        # Shuffled era: option_order.shuffled stamped by the opt-in shuffle.
        {
            "source_batch": "b1",
            "group_id": "g2",
            "evidence": json.dumps(
                {"optimizer_letter": "B", "option_order": {"shuffled": True, "permutation": [1, 0]}}
            ),
        },
        # Unparseable rows are skipped, never a crash (best-effort monitoring).
        {"source_batch": "b1", "group_id": "g3", "evidence": "not-json"},
    ]
    pd.DataFrame(rows).to_csv(d / "evidence.csv", index=False)

    df = load_evidence_provenance(tmp_path)

    assert set(df["group_id"]) == {"g1", "g2"}
    assert (df["dataset"] == "demo").all()
    g1 = df[df["group_id"] == "g1"].iloc[0]
    assert g1["optimizer_letter"] == "A"
    assert bool(g1["option_shuffled"]) is False  # key absence == unshuffled era
    g2 = df[df["group_id"] == "g2"].iloc[0]
    assert g2["optimizer_letter"] == "B"
    assert bool(g2["option_shuffled"]) is True


def test_load_evidence_provenance_missing_returns_empty(tmp_path):
    df = load_evidence_provenance(tmp_path)
    assert len(df) == 0


def test_cli_panel_stats_surfaces_optimizer_anchor_and_era_note(tmp_path, monkeypatch):
    """End-to-end CLI: an optimizer rubber stamp on a fully SHUFFLED wave trips
    OPTIMIZER_ANCHOR (not POSITION_ANCHOR — those ballots are position-excluded,
    which the output annotates)."""
    import json

    monkeypatch.setenv("COLUMNS", "400")
    n = 12
    letters = ["A" if i % 2 else "B" for i in range(n)]  # optimizer letter varies
    votes = _votes(
        [
            # Voter "v" always votes the optimizer's letter; positions split A/B.
            {
                "provider": "v",
                "group_id": f"g{i}",
                "choice": letters[i],
                "confidence": 0.4 + 0.02 * i,
            }
            for i in range(n)
        ]
    )
    consensus = pd.DataFrame([{"group_id": f"g{i}", "choice": letters[i]} for i in range(n)])
    evidence = pd.DataFrame(
        [
            {
                "source_batch": "b1",
                "group_id": f"g{i}",
                "evidence": json.dumps(
                    {"optimizer_letter": letters[i], "option_order": {"shuffled": True}}
                ),
            }
            for i in range(n)
        ]
    )
    _write_provenance(tmp_path, "demo", votes, consensus, evidence)

    result = runner.invoke(app, ["agent", "panel-stats", "--data-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert OPTIMIZER_ANCHOR in result.output
    assert "shuffled-era" in result.output  # mixed/shuffled-era annotation
    # POSITION_ANCHOR fires for no voter (the ballots are position-excluded);
    # the alarm name itself may still appear in the explanatory era note.
    assert f"{POSITION_ANCHOR} v" not in result.output

    strict = runner.invoke(app, ["agent", "panel-stats", "--data-root", str(tmp_path), "--strict"])
    assert strict.exit_code == 1, strict.output


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
