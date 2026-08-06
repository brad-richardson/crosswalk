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

Deliberate design choice (Brad): monitoring is the DEFAULT posture, not a
mitigation. A position anchor is only visible while the letter order is stable
across a voter's ballots, so by default the order stays fixed (option A = the
optimizer's proposal) and the anchor is made loud. Pack-level option-order
shuffling exists as an opt-in mitigation (``settings.stitch_panel_shuffle_options``,
OFF by default); on shuffled-era packs the letters are content-free, so this
module EXCLUDES shuffled-era ballots from every position statistic (a
POSITION_ANCHOR over shuffled letters would be meaningless noise) and keys the
anchoring signal on the pack's recorded optimizer letter instead
(OPTIMIZER_ANCHOR), which works identically in both modes. Era identification
comes from the archived evidence records (``option_order.shuffled``; key absent
== unshuffled era), so mixed shuffled/unshuffled pools stay well-defined.

Three defects are flagged (thresholds live in ``config.py`` as ``panel_monitor_*``,
tunable via ``CROSSWALK_PANEL_MONITOR_*`` env vars):

    POSITION_ANCHOR      A voter lands on its single most-common choice POSITION
                         (letter slot, ``NONE``/``ABSTAIN`` excluded; shuffled-era
                         ballots excluded — their letters are content-free) more
                         often than merit would predict — it is picking by slot.
    OPTIMIZER_ANCHOR     A voter agrees with the optimizer's proposed option more
                         often than the base rate of optimizer correctness can
                         explain — it is rubber-stamping the optimizer. Joined to
                         each pack's recorded ``optimizer_letter`` (evidence
                         records), so it survives option-order shuffling.
    CONSTANT_CONFIDENCE  A voter reports a near-constant confidence: the number
                         carries no information (a rubber stamp).

``NONE`` vs ``ABSTAIN`` — two very different non-letter rows:

    ``NONE`` is a DECISIVE reject-all verdict: a first-class ballot choice with a
    real, model-reported confidence. It is counted as a cast ballot everywhere the
    verdict and its confidence matter — decided/dissent counting (a ``NONE`` vote
    against a letter consensus is dissent; a letter vote against a ``NONE``
    consensus is dissent; ``NONE`` agreeing with a ``NONE`` consensus is agreement),
    confidence stats, calibration, and CONSTANT_CONFIDENCE detection. It is EXCLUDED
    only from letter-POSITION statistics (POSITION_ANCHOR), because it occupies no
    letter slot.

    ``ABSTAIN`` (and blank / errored rows) is a FAILURE, not a verdict. It carries a
    synthetic 0.0 confidence that is the *system's*, not the model's, and would
    corrupt the std / calibration view, so it is excluded from every decided,
    confidence, position, and dissent statistic and only counted in ``n_abstain``.

Field accounting: ``n_ballots == n_valid + n_none + n_abstain``. ``n_valid`` is the
letter-choice count (the POSITION-stat denominator); ``n_none`` the reject-all
verdicts; ``n_abstain`` the ABSTAIN/blank/error failures. Confidence statistics are
computed over CAST ballots (letters + ``NONE``); ``n_scored`` is that sample size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings

# Alarm identifiers (stable strings — CLI/tests/log lines key on these).
POSITION_ANCHOR = "POSITION_ANCHOR"
OPTIMIZER_ANCHOR = "OPTIMIZER_ANCHOR"
CONSTANT_CONFIDENCE = "CONSTANT_CONFIDENCE"

# The decisive reject-all verdict: a real ballot with real confidence, excluded
# only from letter-POSITION statistics (it occupies no letter slot). Failure
# rows (no verdict — blank, error, ABSTAIN) are anything that is neither a
# single letter nor ``NONE``; they are excluded from every decided/confidence/
# position/dissent statistic and counted only in ``n_abstain``.
_NONE = "NONE"

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


def _norm_choice(value: object) -> str:
    """Uppercased, stripped string form of a choice (``""`` for non-strings)."""
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def _is_none(value: object) -> bool:
    """True for the decisive reject-all ``NONE`` verdict."""
    return _norm_choice(value) == _NONE


def _is_decision(value: object) -> bool:
    """True for a real verdict — a letter slot or the reject-all ``NONE``.

    Used both for a voter's cast ballot and for a group's consensus choice: a
    group is "decided" when its consensus is a letter OR ``NONE``; an empty /
    non-verdict consensus is undecided.
    """
    return _is_letter(value) or _is_none(value)


@dataclass
class VoterStats:
    """Per-voter (provider x model) summary over a set of ballots."""

    provider: str
    model: str
    n_ballots: int  # every recorded row for this voter (incl. NONE/abstains/errors)
    n_valid: int  # letter choice with no error (POSITION-stat denominator)
    n_none: int  # decisive reject-all NONE verdicts (cast ballots, no letter slot)
    n_abstain: int  # ABSTAIN/blank/error FAILURE rows (n_ballots - n_valid - n_none)
    n_scored: int  # cast ballots (letters + NONE) with a finite confidence
    abstain_rate: float  # n_abstain / n_ballots — the failure rate
    # POSITION statistics run over letter ballots from UNSHUFFLED-era packs only
    # (shuffled letters are content-free; without evidence metadata every ballot
    # counts as unshuffled, so n_position == n_valid and behavior is unchanged).
    position_counts: dict[int, int]  # letter index -> count (position-eligible only)
    modal_position: int | None
    modal_position_share: float  # modal count / n_position (0.0 when n_position == 0)
    n_decided: int  # cast ballots (letters + NONE) on groups that reached a verdict
    n_dissent: int  # decided ballots whose verdict != the consensus verdict
    dissent_rate: float  # n_dissent / n_decided (NaN when n_decided == 0)
    conf_mean: float
    conf_std: float  # POPULATION std (ddof=0) over cast ballots (letters + NONE)
    conf_min: float
    conf_max: float
    conf_on_agree: float  # mean confidence when the voter agreed with consensus
    conf_on_dissent: float  # mean confidence when the voter dissented
    calibration_gap: float  # conf_on_agree - conf_on_dissent (>0 == calibrated)
    # Era-aware position denominators (defaults keep older direct constructions
    # working; compute_voter_stats always sets them).
    n_position: int = 0  # letter ballots eligible for POSITION stats (unshuffled era)
    n_shuffled: int = 0  # letter ballots from shuffled-era packs (position-excluded)
    # Optimizer-agreement statistics (era-independent: joined to each pack's
    # recorded optimizer letter, so they survive option-order shuffling).
    n_optimizer_known: int = 0  # letter ballots whose pack records an optimizer letter
    n_optimizer_agree: int = 0  # ...of those, ballots that chose the optimizer's option
    optimizer_agree_share: float = float("nan")  # agree / known (NaN when known == 0)
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
    surfacing passes the lower ``panel_monitor_wave_min_n``. The floor applies
    to ``n_position`` — the POSITION-eligible (unshuffled-era) ballots the share
    is actually computed over — so a voter whose ballots are mostly shuffled-era
    cannot trip on a handful of eligible ones, and a fully shuffled-era voter
    (``n_position == 0``) can never trip.
    """
    share = settings.panel_monitor_position_anchor_share if share is None else share
    min_n = settings.panel_monitor_position_anchor_min_n if min_n is None else min_n
    return stat.n_position >= min_n and stat.modal_position_share > share


def optimizer_anchor_tripped(
    stat: VoterStats,
    *,
    share: float | None = None,
    min_n: int | None = None,
) -> bool:
    """True when the voter's optimizer-agreement share clears the anchor threshold.

    Keyed on ballots whose pack records an optimizer letter (``n_optimizer_known``),
    so the signal works whether or not option presentation order was shuffled —
    this is the anchoring alarm that survives the opt-in shuffle. The default
    share threshold (``panel_monitor_optimizer_anchor_share``) is calibrated to
    committed base rates: the optimizer is genuinely right most of the time, so
    healthy agreement sits well above POSITION_ANCHOR's 0.6 (see config.py).
    ``min_n`` reuses the POSITION_ANCHOR floors (aggregate / wave).
    """
    share = settings.panel_monitor_optimizer_anchor_share if share is None else share
    min_n = settings.panel_monitor_position_anchor_min_n if min_n is None else min_n
    return (
        stat.n_optimizer_known >= min_n
        and not np.isnan(stat.optimizer_agree_share)
        and stat.optimizer_agree_share > share
    )


def constant_confidence_tripped(
    stat: VoterStats,
    *,
    std: float | None = None,
    min_n: int | None = None,
) -> bool:
    """True when confidence is near-constant over enough scored ballots.

    Keyed on ``n_scored`` (cast ballots — letters + ``NONE`` — carrying a finite
    confidence), the exact sample the ``conf_std`` is computed over, so a voter who
    reject-alls at a flat confidence is still caught.
    """
    std = settings.panel_monitor_constant_confidence_std if std is None else std
    min_n = settings.panel_monitor_constant_confidence_min_n if min_n is None else min_n
    return stat.n_scored >= min_n and not np.isnan(stat.conf_std) and stat.conf_std < std


def default_alarms(stat: VoterStats) -> list[str]:
    """Alarms tripped under the configured aggregate thresholds."""
    alarms = []
    if position_anchor_tripped(stat):
        alarms.append(POSITION_ANCHOR)
    if optimizer_anchor_tripped(stat):
        alarms.append(OPTIMIZER_ANCHOR)
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


def _with_evidence_context(votes: pd.DataFrame, evidence: pd.DataFrame | None) -> pd.DataFrame:
    """Attach ``_opt_letter`` / ``_shuffled`` columns from per-group evidence.

    ``evidence`` carries one row per voted group with ``optimizer_letter`` and
    ``option_shuffled`` (see :func:`load_evidence_provenance`), joined on the
    same key columns as the consensus join. Without evidence (or without a join
    key) every ballot reads as unshuffled with an unknown optimizer letter —
    the exact pre-evidence behavior.
    """
    out = votes.copy()
    out["_opt_letter"] = ""
    out["_shuffled"] = False
    if evidence is None or len(evidence) == 0:
        return out
    keys = _join_keys(votes, evidence)
    if not keys:
        return out
    cols = {}
    if "optimizer_letter" in evidence.columns:
        cols["optimizer_letter"] = "_ev_opt_letter"
    if "option_shuffled" in evidence.columns:
        cols["option_shuffled"] = "_ev_shuffled"
    if not cols:
        return out
    right = evidence[[*keys, *cols.keys()]].rename(columns=cols).drop_duplicates(subset=keys)
    out = out.merge(right, on=keys, how="left")
    if "_ev_opt_letter" in out.columns:
        out["_opt_letter"] = out.pop("_ev_opt_letter").map(_norm_choice)
    if "_ev_shuffled" in out.columns:
        # NaN (group absent from evidence) reads as unshuffled: absence of the
        # option_order key is the archived signal for the unshuffled era.
        out["_shuffled"] = out.pop("_ev_shuffled").fillna(False).astype(bool)
    return out


def compute_voter_stats(
    votes: pd.DataFrame,
    consensus: pd.DataFrame | None = None,
    evidence: pd.DataFrame | None = None,
) -> list[VoterStats]:
    """Per-voter statistics + configured alarms over a set of ballots.

    ``votes`` needs at least ``provider`` and ``choice`` columns; ``model``,
    ``confidence``, ``error``, and the join keys are used when present. ``consensus``
    (optional) supplies the per-group agreed choice used for dissent / calibration;
    without it those fields are NaN but the position and confidence alarms still work.
    ``evidence`` (optional; see :func:`load_evidence_provenance`) supplies each
    group's recorded ``optimizer_letter`` and shuffled-era flag: with it the
    OPTIMIZER_ANCHOR alarm activates and shuffled-era ballots (content-free
    letters) are excluded from POSITION statistics; without it optimizer stats
    stay unknown and every ballot counts as unshuffled (pre-evidence behavior).
    """
    if len(votes) == 0:
        return []

    df = _with_consensus_choice(votes, consensus)
    df = _with_evidence_context(df, evidence)
    if "model" not in df.columns:
        df["model"] = ""
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    if "error" not in df.columns:
        df["error"] = np.nan

    df["_pos"] = df["choice"].map(_position)
    has_error = df["error"].notna() & (df["error"].astype(str).str.strip() != "")
    df["_is_none"] = df["choice"].map(_is_none)
    # A cast ballot is a real verdict with no error: a letter slot OR reject-all NONE.
    df["_letter"] = df["_pos"].notna() & ~has_error
    df["_ballot"] = (df["_pos"].notna() | df["_is_none"]) & ~has_error
    df["_decided"] = df["_cons_choice"].map(_is_decision)
    # POSITION eligibility: letter ballots from unshuffled-era packs only.
    df["_pos_eligible"] = df["_letter"] & ~df["_shuffled"]
    # Optimizer agreement: letter ballots whose pack records an optimizer letter
    # (era-independent — the optimizer letter is the pack's own, shuffled or not).
    df["_opt_known"] = df["_letter"] & df["_opt_letter"].map(_is_letter)
    df["_opt_agree"] = df["_opt_known"] & (df["choice"].map(_norm_choice) == df["_opt_letter"])

    stats: list[VoterStats] = []
    group_cols = ["provider", "model"]
    for (provider, model), g in df.groupby(group_cols, dropna=False, sort=True):
        letters = g[g["_letter"]]  # letter ballots
        pos_eligible = g[g["_pos_eligible"]]  # POSITION statistics sample
        ballots = g[g["_ballot"]]  # cast verdicts: letters + NONE
        n_ballots = len(g)
        n_valid = len(letters)
        n_position = len(pos_eligible)
        n_shuffled = n_valid - n_position
        n_none = int((ballots["_is_none"]).sum())
        n_abstain = n_ballots - n_valid - n_none

        position_counts: dict[int, int] = {
            int(k): int(v) for k, v in pos_eligible["_pos"].value_counts().sort_index().items()
        }
        if position_counts:
            modal_position = max(position_counts, key=lambda k: (position_counts[k], -k))
            modal_share = position_counts[modal_position] / n_position
        else:
            modal_position = None
            modal_share = 0.0

        n_optimizer_known = int(g["_opt_known"].sum())
        n_optimizer_agree = int(g["_opt_agree"].sum())
        optimizer_agree_share = (
            (n_optimizer_agree / n_optimizer_known) if n_optimizer_known else float("nan")
        )

        # Confidence over CAST ballots (letters + NONE): a NONE verdict's confidence
        # is the model's own and belongs in the mean / std / calibration view.
        conf = pd.to_numeric(ballots["confidence"], errors="coerce").to_numpy(dtype=float)
        conf = conf[~np.isnan(conf)]
        n_scored = int(conf.size)
        if conf.size:
            conf_mean = float(np.mean(conf))
            conf_std = float(np.std(conf, ddof=0))
            conf_min = float(np.min(conf))
            conf_max = float(np.max(conf))
        else:
            conf_mean = conf_std = conf_min = conf_max = float("nan")

        decided = ballots[ballots["_decided"]]
        n_decided = len(decided)
        # Compare verdict identities (letter slot or NONE), case/space-normalized, so
        # NONE-vs-letter and NONE-vs-NONE resolve correctly against the consensus.
        dissent_mask = decided["choice"].map(_norm_choice) != decided["_cons_choice"].map(
            _norm_choice
        )
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
            n_none=n_none,
            n_abstain=n_abstain,
            n_scored=n_scored,
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
            n_position=n_position,
            n_shuffled=n_shuffled,
            n_optimizer_known=n_optimizer_known,
            n_optimizer_agree=n_optimizer_agree,
            optimizer_agree_share=optimizer_agree_share,
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


def _evidence_row(record: dict) -> dict:
    """Monitoring-relevant fields of one archived evidence record."""
    option_order = record.get("option_order") or {}
    return {
        "optimizer_letter": record.get("optimizer_letter"),
        # Key absence == unshuffled era (pre-shuffle packs never wrote it, and
        # shuffle-off packs deliberately omit it to stay byte-identical).
        "option_shuffled": bool(option_order.get("shuffled", False)),
    }


def load_evidence_provenance(
    data_root: str | Path = ".",
    dataset: str | None = None,
) -> pd.DataFrame:
    """Load per-group evidence context from ``labels/votes/dataset=*/evidence.csv``.

    Returns one row per archived (dataset, source_batch, group_id) with the
    monitoring-relevant evidence fields parsed out of the archived evidence
    JSON: ``optimizer_letter`` (which displayed option was the optimizer's
    proposal) and ``option_shuffled`` (whether the pack's presentation order
    was shuffled — ``option_order.shuffled``, with key absence meaning the
    unshuffled era). Feed the frame to :func:`compute_voter_stats` to activate
    OPTIMIZER_ANCHOR and era-aware POSITION statistics. Missing partitions or
    unparseable rows are skipped (monitoring is best-effort, never a crash).
    """
    root = Path(data_root) / "labels" / "votes"
    if dataset is not None:
        dirs = [root / f"dataset={dataset}"]
    else:
        dirs = sorted(p for p in root.glob("dataset=*") if p.is_dir())

    frames: list[pd.DataFrame] = []
    for d in dirs:
        path = d / "evidence.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype={"group_id": str})
        if "evidence" not in df.columns:
            continue
        rows = []
        for _, row in df.iterrows():
            try:
                record = json.loads(row["evidence"])
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "dataset": d.name.replace("dataset=", ""),
                    "source_batch": row.get("source_batch"),
                    "group_id": row.get("group_id"),
                    **_evidence_row(record),
                }
            )
        if rows:
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Wave-time surfacing (used by stitch_runner.run_batch)
# ---------------------------------------------------------------------------


def wave_position_anchor_warnings(
    votes: pd.DataFrame,
    consensus: pd.DataFrame | None = None,
    evidence: pd.DataFrame | None = None,
    *,
    min_n: int | None = None,
) -> list[str]:
    """One warning line per voter tripping POSITION_ANCHOR within a single wave.

    Uses the lower wave n-floor (``panel_monitor_wave_min_n``) so a defect shows up
    on a wave before the aggregate view has accumulated enough ballots. Reuses the
    same stats + predicate as the offline monitor — no duplicated logic. Passing
    the wave's ``evidence`` frame suppresses the alarm for shuffled-era ballots
    (their letters are content-free); on a fully shuffled wave no POSITION_ANCHOR
    warning can fire.
    """
    min_n = settings.panel_monitor_wave_min_n if min_n is None else min_n
    lines: list[str] = []
    for stat in compute_voter_stats(votes, consensus, evidence):
        if position_anchor_tripped(stat, min_n=min_n):
            lines.append(
                f"{stat.provider} ({stat.model}): {POSITION_ANCHOR} — "
                f"{stat.modal_position_share:.0%} of {stat.n_position} position-eligible "
                f"ballots on position {stat.modal_letter}; the voter may be picking by "
                f"slot, not merit"
            )
    return lines


def wave_optimizer_anchor_warnings(
    votes: pd.DataFrame,
    consensus: pd.DataFrame | None = None,
    evidence: pd.DataFrame | None = None,
    *,
    min_n: int | None = None,
) -> list[str]:
    """One warning line per voter tripping OPTIMIZER_ANCHOR within a single wave.

    The shuffle-proof anchoring alarm: keyed on each pack's recorded optimizer
    letter (``evidence`` frame), so it fires identically on shuffled and
    unshuffled waves. Uses the lower wave n-floor, mirroring
    :func:`wave_position_anchor_warnings`. Without an ``evidence`` frame no
    optimizer letters are known and no warning can fire.
    """
    min_n = settings.panel_monitor_wave_min_n if min_n is None else min_n
    lines: list[str] = []
    for stat in compute_voter_stats(votes, consensus, evidence):
        if optimizer_anchor_tripped(stat, min_n=min_n):
            lines.append(
                f"{stat.provider} ({stat.model}): {OPTIMIZER_ANCHOR} — agreed with the "
                f"optimizer's option on {stat.optimizer_agree_share:.0%} of "
                f"{stat.n_optimizer_known} ballots; the voter may be rubber-stamping "
                f"the optimizer, not judging the geometry"
            )
    return lines
