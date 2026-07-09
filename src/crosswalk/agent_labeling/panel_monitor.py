"""Per-voter bias monitoring for the stitch consensus panel.

Make voter defects LOUD instead of discovering them by accident.

Motivating evidence (committed panel provenance): voter ``agy`` (Gemini Flash via
CLI) voted the FIRST-listed option ``A`` in 11 of 12 valid ballots at a CONSTANT
0.95 confidence, and every one of its dissents was ``agy=A`` — a position-anchored
rubber stamp that inflated unanimity whenever the first option happened to be
correct and drove ~1/3 of panel failures in its waves. ``opencode`` (same model
via OpenRouter) shows confidence inflation but healthy dissent positions;
``claude`` is well calibrated. Nothing in the pipeline surfaced any of this. This
module does.

Deliberate design choice (Brad): this is a MONITOR, not a mitigation. Shuffling the
option letters per ballot would *hide* the very defect it detects — a position
anchor is only visible because the letter order is stable across a voter's
ballots. We keep the order fixed and make the anchor loud.

Two defects are flagged (thresholds live in ``config.py`` as ``panel_monitor_*``,
tunable via ``CROSSWALK_PANEL_MONITOR_*`` env vars):

    POSITION_ANCHOR      A voter lands on its single most-common choice POSITION
                         (letter slot, ``NONE``/``ABSTAIN`` excluded) more often
                         than merit would predict — it is picking by slot.
    CONSTANT_CONFIDENCE  A voter reports a near-constant confidence: the number
                         carries no information (a rubber stamp).

Confidence statistics are computed over VALID ballots only — a real letter choice
with no error. A forced ABSTAIN carries a synthetic 0.0 confidence that is the
*system's*, not the model's, and would corrupt the std / calibration view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings

# Alarm identifiers (stable strings — CLI/tests/log lines key on these).
POSITION_ANCHOR = "POSITION_ANCHOR"
CONSTANT_CONFIDENCE = "CONSTANT_CONFIDENCE"

# Non-letter ballots: excluded from position/confidence/dissent statistics.
_NON_CHOICES = frozenset({"NONE", "ABSTAIN"})

# Columns used, in priority order, to join a vote to its group's consensus row.
# Whichever of these are present in BOTH frames form the join key: the committed
# provenance carries all three (multiple datasets/waves concatenated); a live
# single-wave ``votes_df``/``consensus_df`` from ``run_batch`` carries only
# ``group_id`` (one batch, no collisions).
_JOIN_CANDIDATES = ("dataset", "source_batch", "group_id")


def _position(choice: object) -> int | None:
    """Letter index of a choice (``A``->0, ``B``->1, ...) or ``None``.

    ``NONE``, ``ABSTAIN``, blanks, and anything that is not a single A-Z letter
    return ``None`` and are excluded from position statistics.
    """
    if not isinstance(choice, str):
        return None
    c = choice.strip()
    if len(c) == 1 and "A" <= c <= "Z":
        return ord(c) - ord("A")
    return None


def _is_letter(value: object) -> bool:
    return _position(value) is not None


@dataclass
class VoterStats:
    """Per-voter (provider x model) summary over a set of ballots."""

    provider: str
    model: str
    n_ballots: int  # every recorded row for this voter (incl. abstains/errors)
    n_valid: int  # letter choice with no error
    n_abstain: int  # ABSTAIN/NONE or error rows
    abstain_rate: float
    position_counts: dict[int, int]  # letter index -> count (valid ballots only)
    modal_position: int | None
    modal_position_share: float  # modal count / n_valid (0.0 when n_valid == 0)
    n_decided: int  # valid ballots on groups that reached a consensus choice
    n_dissent: int  # decided ballots whose choice != the consensus choice
    dissent_rate: float  # n_dissent / n_decided (NaN when n_decided == 0)
    conf_mean: float
    conf_std: float  # POPULATION std (ddof=0) over valid ballots
    conf_min: float
    conf_max: float
    conf_on_agree: float  # mean confidence when the voter agreed with consensus
    conf_on_dissent: float  # mean confidence when the voter dissented
    calibration_gap: float  # conf_on_agree - conf_on_dissent (>0 == calibrated)
    alarms: list[str] = field(default_factory=list)

    @property
    def modal_letter(self) -> str:
        return "-" if self.modal_position is None else chr(ord("A") + self.modal_position)


# ---------------------------------------------------------------------------
# Alarm predicates (shared by the offline CLI and the wave-time surfacing)
# ---------------------------------------------------------------------------


def position_anchor_tripped(
    stat: VoterStats,
    *,
    share: float | None = None,
    min_n: int | None = None,
) -> bool:
    """True when the voter's modal-position share clears the anchor threshold.

    ``min_n`` guards against small-sample noise; the offline monitor uses the
    aggregate floor (``panel_monitor_position_anchor_min_n``) while wave-time
    surfacing passes the lower ``panel_monitor_wave_min_n``.
    """
    share = settings.panel_monitor_position_anchor_share if share is None else share
    min_n = settings.panel_monitor_position_anchor_min_n if min_n is None else min_n
    return stat.n_valid >= min_n and stat.modal_position_share > share


def constant_confidence_tripped(
    stat: VoterStats,
    *,
    std: float | None = None,
    min_n: int | None = None,
) -> bool:
    """True when confidence is near-constant over enough valid ballots."""
    std = settings.panel_monitor_constant_confidence_std if std is None else std
    min_n = settings.panel_monitor_constant_confidence_min_n if min_n is None else min_n
    return stat.n_valid >= min_n and not np.isnan(stat.conf_std) and stat.conf_std < std


def default_alarms(stat: VoterStats) -> list[str]:
    """Alarms tripped under the configured aggregate thresholds."""
    alarms = []
    if position_anchor_tripped(stat):
        alarms.append(POSITION_ANCHOR)
    if constant_confidence_tripped(stat):
        alarms.append(CONSTANT_CONFIDENCE)
    return alarms


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _join_keys(votes: pd.DataFrame, consensus: pd.DataFrame) -> list[str]:
    return [k for k in _JOIN_CANDIDATES if k in votes.columns and k in consensus.columns]


def _with_consensus_choice(votes: pd.DataFrame, consensus: pd.DataFrame | None) -> pd.DataFrame:
    """Attach a ``_cons_choice`` column (the group's consensus choice, or NaN)."""
    out = votes.copy()
    if consensus is None or len(consensus) == 0 or "choice" not in consensus.columns:
        out["_cons_choice"] = np.nan
        return out
    keys = _join_keys(votes, consensus)
    if not keys:
        out["_cons_choice"] = np.nan
        return out
    right = consensus[[*keys, "choice"]].rename(columns={"choice": "_cons_choice"})
    right = right.drop_duplicates(subset=keys)
    return out.merge(right, on=keys, how="left")


def compute_voter_stats(
    votes: pd.DataFrame,
    consensus: pd.DataFrame | None = None,
) -> list[VoterStats]:
    """Per-voter statistics + configured alarms over a set of ballots.

    ``votes`` needs at least ``provider`` and ``choice`` columns; ``model``,
    ``confidence``, ``error``, and the join keys are used when present. ``consensus``
    (optional) supplies the per-group agreed choice used for dissent / calibration;
    without it those fields are NaN but the position and confidence alarms still work.
    """
    if len(votes) == 0:
        return []

    df = _with_consensus_choice(votes, consensus)
    if "model" not in df.columns:
        df["model"] = ""
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    if "error" not in df.columns:
        df["error"] = np.nan

    df["_pos"] = df["choice"].map(_position)
    has_error = df["error"].notna() & (df["error"].astype(str).str.strip() != "")
    df["_valid"] = df["_pos"].notna() & ~has_error
    df["_decided"] = df["_cons_choice"].map(_is_letter)

    stats: list[VoterStats] = []
    group_cols = ["provider", "model"]
    for (provider, model), g in df.groupby(group_cols, dropna=False, sort=True):
        valid = g[g["_valid"]]
        n_ballots = len(g)
        n_valid = len(valid)
        n_abstain = n_ballots - n_valid

        position_counts: dict[int, int] = {
            int(k): int(v) for k, v in valid["_pos"].value_counts().sort_index().items()
        }
        if position_counts:
            modal_position = max(position_counts, key=lambda k: (position_counts[k], -k))
            modal_share = position_counts[modal_position] / n_valid
        else:
            modal_position = None
            modal_share = 0.0

        conf = pd.to_numeric(valid["confidence"], errors="coerce").to_numpy(dtype=float)
        conf = conf[~np.isnan(conf)]
        if conf.size:
            conf_mean = float(np.mean(conf))
            conf_std = float(np.std(conf, ddof=0))
            conf_min = float(np.min(conf))
            conf_max = float(np.max(conf))
        else:
            conf_mean = conf_std = conf_min = conf_max = float("nan")

        decided = valid[valid["_decided"]]
        n_decided = len(decided)
        dissent_mask = decided["choice"] != decided["_cons_choice"]
        n_dissent = int(dissent_mask.sum())
        dissent_rate = (n_dissent / n_decided) if n_decided else float("nan")

        agree_conf = pd.to_numeric(decided[~dissent_mask]["confidence"], errors="coerce")
        dissent_conf = pd.to_numeric(decided[dissent_mask]["confidence"], errors="coerce")
        conf_on_agree = float(agree_conf.mean()) if len(agree_conf) else float("nan")
        conf_on_dissent = float(dissent_conf.mean()) if len(dissent_conf) else float("nan")
        calibration_gap = conf_on_agree - conf_on_dissent

        stat = VoterStats(
            provider=str(provider),
            model=str(model) if model == model else "",  # NaN-safe
            n_ballots=n_ballots,
            n_valid=n_valid,
            n_abstain=n_abstain,
            abstain_rate=(n_abstain / n_ballots) if n_ballots else 0.0,
            position_counts=position_counts,
            modal_position=modal_position,
            modal_position_share=modal_share,
            n_decided=n_decided,
            n_dissent=n_dissent,
            dissent_rate=dissent_rate,
            conf_mean=conf_mean,
            conf_std=conf_std,
            conf_min=conf_min,
            conf_max=conf_max,
            conf_on_agree=conf_on_agree,
            conf_on_dissent=conf_on_dissent,
            calibration_gap=calibration_gap,
        )
        stat.alarms = default_alarms(stat)
        stats.append(stat)

    return stats


# ---------------------------------------------------------------------------
# Committed provenance loading
# ---------------------------------------------------------------------------


def load_vote_provenance(
    data_root: str | Path = ".",
    dataset: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load committed vote/consensus provenance from ``labels/votes/dataset=*``.

    Returns ``(votes_df, consensus_df)``, each carrying a ``dataset`` column. When
    ``dataset`` is given, only that partition is read. Missing partitions yield
    empty frames.
    """
    root = Path(data_root) / "labels" / "votes"
    if dataset is not None:
        vote_dirs = [root / f"dataset={dataset}"]
    else:
        vote_dirs = sorted(p for p in root.glob("dataset=*") if p.is_dir())

    vote_frames: list[pd.DataFrame] = []
    cons_frames: list[pd.DataFrame] = []
    for d in vote_dirs:
        name = d.name.replace("dataset=", "")
        vpath = d / "votes.csv"
        cpath = d / "consensus.csv"
        if vpath.exists():
            vdf = pd.read_csv(vpath, dtype={"group_id": str})
            vdf["dataset"] = name
            vote_frames.append(vdf)
        if cpath.exists():
            cdf = pd.read_csv(cpath, dtype={"group_id": str})
            cdf["dataset"] = name
            cons_frames.append(cdf)

    votes_df = pd.concat(vote_frames, ignore_index=True) if vote_frames else pd.DataFrame()
    consensus_df = pd.concat(cons_frames, ignore_index=True) if cons_frames else pd.DataFrame()
    return votes_df, consensus_df


# ---------------------------------------------------------------------------
# Wave-time surfacing (used by stitch_runner.run_batch)
# ---------------------------------------------------------------------------


def wave_position_anchor_warnings(
    votes: pd.DataFrame,
    consensus: pd.DataFrame | None = None,
    *,
    min_n: int | None = None,
) -> list[str]:
    """One warning line per voter tripping POSITION_ANCHOR within a single wave.

    Uses the lower wave n-floor (``panel_monitor_wave_min_n``) so a defect shows up
    on a wave before the aggregate view has accumulated enough ballots. Reuses the
    same stats + predicate as the offline monitor — no duplicated logic.
    """
    min_n = settings.panel_monitor_wave_min_n if min_n is None else min_n
    lines: list[str] = []
    for stat in compute_voter_stats(votes, consensus):
        if position_anchor_tripped(stat, min_n=min_n):
            lines.append(
                f"{stat.provider} ({stat.model}): {POSITION_ANCHOR} — "
                f"{stat.modal_position_share:.0%} of {stat.n_valid} valid ballots on "
                f"position {stat.modal_letter}; the voter may be picking by slot, not merit"
            )
    return lines
