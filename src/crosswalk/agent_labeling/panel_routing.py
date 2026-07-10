"""Resolve the agent panel's per-group routing for a dataset.

The stitching panel (see :mod:`stitch_runner`) votes on M:N groups
and writes a ``consensus.csv`` per batch dir with a ``routing`` column whose
values are ``auto_accept`` (all valid votes agree at quorum — full unanimity,
or a v5 quorum accept over an abstention; safe to promote) or ``human_review``
(dissent, below quorum, NONE consensus, cross-mode flags, oversize, ...). A
group that the panel could not auto-accept is exactly the kind of decision
worth a human's 1-2 minutes.

This module discovers a dataset's panel batch dirs (``{dataset}`` or
``{dataset}_*`` under ``data/agents/stitching/batches``), resolves the MOST
RECENT vote per group (a group may be voted across several waves), and exposes
the set of group ids the panel routed to ``human_review``. The human
``/stitching-review`` queue is gated to that set so it only ever contains
panel failures — never never-voted curiosity/calibration samples.

Recency is determined by each ``consensus.csv``'s mtime (older waves first, so
the newest wave's routing wins on conflict; the dir name breaks ties). That
mirrors the precedence-ordered merge used by the label exporter
(:func:`stitch_export._merge_consensus`) without requiring the caller to know
the wave order.

Size-gate overlay: the export path has a hard backstop
(``settings.stitch_export_backstop_max_edges``) — no verdict on a group whose
candidate-edge count exceeds it can ever mint a label
(:func:`stitch_export._gate_group` enforces the backstop on the candidate
count on BOTH the structural path and the legacy no-structure-fields fallback;
:func:`stitch_export._gate_empty_group` likewise for reject-alls). An ``auto_accept``
verdict on such a group would therefore vanish: not in the human queue (it did
not route to ``human_review``) and blocked at export. :func:`compute_consensus
<stitch_runner.compute_consensus>` now demotes those verdicts at vote time, and
this module applies the SAME gate when reading historical waves (rows stamped
before the gate existed), so past monster votes surface in the human queue with
``route_reason="size_gated"`` instead of disappearing. Groups the panel NEVER
voted on stay out of the queue even when over-backstop: batch selection no
longer feeds them to panel waves, and reviewing a monster whole is exactly what
the #367 Mode-B decomposition flow is being built to avoid (``--include-unvoted``
remains the manual escape).

Low-confidence overlay: a panel review found the two Gemini-based voters report
pathologically inflated confidence (one pinned at 0.95, the other a ~1.0 median
on coin-flips) while the calibrated voters' self-reports separated wrong
unanimous verdicts from clean accepts. So an ``auto_accept`` whose MINIMUM
confidence across valid votes is below ``settings.stitch_min_voter_confidence``
is demoted to ``human_review`` / ``route_reason="low_confidence"`` — at vote time
in :func:`compute_consensus <stitch_runner.compute_consensus>`, and as a
read-time overlay here for waves voted before the gate existed (unlike a
size-gated verdict, a low-confidence one is NOT blocked at export, so without the
overlay it would silently auto-export on the next ``stitch-export``). The
low-confidence overlay runs AFTER the size gate — ``size_gated`` wins when both
apply.

Quorum-floor overlay: :func:`compute_consensus <stitch_runner.compute_consensus>`
only mints ``auto_accept`` at ``n_valid >= 3``; a historical/hand-edited/corrupt
row claiming ``auto_accept`` with ``n_valid`` present and < 3 is demoted here to
``human_review`` (``route_reason="below_quorum:<n>"``) so a sub-quorum accept can
never be treated as accepted. Runs BEFORE the size/low-confidence gates (the
floor is the more fundamental invariant).

Residual decomposition void (#403 follow-up): a monster that went straight to
decomposition has no direct whole-group vote, and one of its roster
sub-problems may have been skipped as an irreducible oversized block (no
consensus row). Such a parent can never recompose (all-or-nothing) yet has no
failed sub-vote to fold onto the queue — a silent void. :func:`panel_failed_group_ids`
surfaces these via :func:`unvoted_decomposed_parents` (which reads the batch.json
roster) and :func:`attach_panel_route_reasons` annotates them
``route_reason="oversized_unvoted"``.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from ..config import settings
from ..filenames import PROJECT_ROOT
from ..matching.group_decomposition import parent_group_id_of

logger = logging.getLogger(__name__)


def _raise_csv_field_limit() -> None:
    """Lift csv's per-field size cap so large ``edge_set`` cells parse.

    ``consensus.csv`` stores each group's selected ``edge_set`` as a JSON array
    inline; for large M:N groups this single field exceeds Python's default
    131072-char limit and ``csv.DictReader`` raises "field larger than field
    limit". Raise the limit to the largest C long the platform accepts (backing
    off on the Windows OverflowError where ``sys.maxsize`` overflows a C long).
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()

# Root under which each panel wave writes ``<batch_name>/consensus.csv``.
STITCH_BATCHES_DIR = PROJECT_ROOT / "data" / "agents" / "stitching" / "batches"

ROUTING_HUMAN_REVIEW = "human_review"
ROUTING_AUTO_ACCEPT = "auto_accept"

# ---------------------------------------------------------------------------
# route_reason codes
#
# Compact machine-readable codes covering every routing decision the panel
# makes (see stitch_runner.compute_consensus, which stamps these into
# consensus.csv going forward). derive_route_reason() reconstructs the same
# code from an existing consensus row's columns when route_reason is blank
# (historical waves predate the stamp).
# ---------------------------------------------------------------------------

#: Unanimous non-NONE verdict (every recorded vote valid and agreeing) — the
#: auto-accept path.
REASON_UNANIMOUS = "unanimous"
#: All panelists voted NONE — every offered option was rejected.
REASON_UNANIMOUS_NONE = "unanimous_none"
#: QUORUM accept (v5 rule): all VALID votes agree, >=3 are valid, and >=1
#: panelist abstained (e.g. 3-of-4 with one abstention) — the auto-accept
#: path's second tier. Distinct from ``unanimous`` so a quorum accept stays
#: distinguishable end-to-end (consensus.csv and the ``panel_quorum_*`` export
#: labelers). Only reachable with >=4 voters; quorum forgives abstention only,
#: never a dissenting valid vote.
REASON_QUORUM = "quorum"
#: All valid votes were NONE at quorum with >=1 abstention — the quorum analog
#: of ``unanimous_none`` (routes to ``human_review``; exportable as a
#: reject-all empty-set label under the quorum-NONE labeler).
REASON_QUORUM_NONE = "quorum_none"
#: Majority with dissenting valid vote(s); suffix is the minority summary,
#: e.g. ``dissent:codex=B`` or ``dissent:codex=F,agy=A``.
REASON_DISSENT_PREFIX = "dissent:"
#: All valid votes agree but fewer than 3 were valid (quorum for unanimity);
#: suffix is n_valid, e.g. ``below_quorum:2``.
REASON_BELOW_QUORUM_PREFIX = "below_quorum:"
#: LEGACY (pre-v5): >=3 valid votes all agree, but an abstention blocked full
#: unanimity under the old ``agree == len(votes)`` rule (only reachable with a
#: 4-voter panel, e.g. the 2026-07-10 quad calibration waves). The v5 quorum
#: rule AUTO-ACCEPTS exactly this case (stamped ``quorum``), so this code is
#: retired from live routing — kept only so historical consensus.csv rows keep
#: deriving/rendering faithfully.
REASON_ABSTENTION = "abstention"
#: No choice reached 2 votes — the panel split.
REASON_NO_MAJORITY = "no_majority"
#: Every panelist abstained (parse failure / provider error on all).
REASON_ALL_ABSTAINED = "all_abstained"
#: Class-consistency gate demoted a unanimous verdict whose chosen edges
#: include a cross-mode (pedestrian/vehicular/bike) pair. Value kept as the
#: historical on-disk spelling stamped by the gate since it shipped.
REASON_CLASS_MISMATCH = "class-mismatch"
#: Size gate demoted a verdict on a group whose candidate-edge count exceeds
#: the export backstop (``settings.stitch_export_backstop_max_edges``): no
#: verdict on such a group can mint a label, so it always needs a human.
#: Same spelling as the legacy phase-2 size gate's stamp.
REASON_SIZE_GATED = "size_gated"
#: Low-confidence gate demoted a unanimous verdict whose MINIMUM confidence
#: across valid votes fell below ``settings.stitch_min_voter_confidence``. The
#: two Gemini-based voters report near-constant inflated confidence, so the
#: minimum is effectively the calibrated voter's self-report — a low value there
#: flagged wrong unanimous verdicts the panel review found.
REASON_LOW_CONFIDENCE = "low_confidence"
#: A decomposed group (#367 Mode B) at least one of whose sub-problems the panel
#: routed to ``human_review``. The sub-problem verdicts key on sub-problem ids
#: (``{parent}__p...``) that are not sidecar groups, so the PARENT is surfaced to
#: the human review queue on its behalf: recomposition is all-or-nothing, so a
#: single failed sub-problem blocks the whole-group label and the reviewer must
#: adjudicate the parent. Synthesized by :func:`attach_panel_route_reasons` for a
#: parent that has no direct consensus row of its own (a decompose-first monster).
REASON_SUBPROBLEM_FAILED = "subproblem_failed"
#: A decompose-first parent (#367 Mode B) at least one of whose roster
#: sub-problems was NEVER voted — skipped as an irreducible oversized block, so it
#: has no consensus row. Such a parent has no failed sub-vote to fold and no direct
#: parent vote, so without this it would neither auto-accept (recomposition is
#: all-or-nothing, and an unvoted sub blocks it) nor queue: a silent void. It is
#: surfaced to the human review queue on its own by :func:`panel_failed_group_ids`
#: and annotated here by :func:`attach_panel_route_reasons`.
REASON_OVERSIZED_UNVOTED = "oversized_unvoted"

#: Legacy phase-2 stamps that merely echo the ``consensus`` column ("majority",
#: "none"). They carry strictly less information than what the row's own
#: columns derive (who dissented, quorum), so they are treated as blank and
#: re-derived. Informative legacy codes (e.g. ``size_gated``, which is NOT
#: derivable from the columns) are preserved as-is.
_LEGACY_TIER_ECHOES = frozenset({"majority", "none"})


def _clean(val) -> str:
    """Normalize a consensus-row cell to a stripped string ('' for NaN/None)."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


def _int_or_none(val) -> int | None:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def counts_show_abstention(n_valid: int | None, n_votes: int | None) -> bool:
    """True when a row's vote counts prove >=1 abstention (``n_valid < n_votes``).

    Count evidence must always be able to DOWNGRADE a verdict's tier to the
    WEAKER quorum claim; ``n_valid == n_votes`` (no abstention) is the only shape
    that supports the stronger unanimous claim.

    Logically impossible counts — ``n_valid > n_votes`` (more valid than total),
    or a negative — cannot support unanimity either, yet the naive
    ``0 < n_valid < n_votes`` check falls through to False on them and would let a
    corrupt/hand-edited row mint the STRONGER unanimous claim its data cannot
    support. So impossible counts take the conservative path: treated as
    abstention-present (the weakest claim, quorum) with a warning logged so the
    anomaly is visible. Missing/unparseable counts are no evidence (False).
    """
    if n_valid is None or n_votes is None:
        return False
    if n_valid < 0 or n_votes < 0 or n_valid > n_votes:
        logger.warning(
            "impossible vote counts n_valid=%s n_votes=%s — treating as quorum "
            "(weakest claim) rather than unanimous",
            n_valid,
            n_votes,
        )
        return True
    return 0 < n_valid < n_votes


def derive_route_reason(row: Mapping) -> str:
    """Reason code for a consensus row's routing decision.

    Returns the row's own ``route_reason`` when it carries an informative one
    (the stamp, or a legacy code such as ``class-mismatch`` / ``size_gated``);
    otherwise derives the code from the columns every historical wave has
    (consensus/choice/routing/minority/n_votes/n_valid). Phase-2's bare
    tier-echo stamps ("majority"/"none") are re-derived, and its
    ``unanimous_NONE`` spelling is normalized. Pure — safe on rows from
    csv.DictReader or pandas. Returns ``""`` only for rows too malformed to
    classify.
    """
    existing = _clean(row.get("route_reason"))
    if existing == "unanimous_NONE":  # legacy spelling of the same code
        return REASON_UNANIMOUS_NONE
    if existing and existing not in _LEGACY_TIER_ECHOES:
        return existing

    routing = _clean(row.get("routing"))
    consensus = _clean(row.get("consensus"))
    choice = _clean(row.get("choice"))
    minority = _clean(row.get("minority"))
    n_valid = _int_or_none(row.get("n_valid"))
    n_votes = _int_or_none(row.get("n_votes"))

    if routing == ROUTING_AUTO_ACCEPT:
        # A QUORUM accept (v5 rule: all valid votes agree over >=1 abstention)
        # is distinguishable from full unanimity by its tier stamp, or — for a
        # row missing the tier — by the vote counts. Pre-v5 auto_accept rows
        # always have n_valid == n_votes (the old rule required
        # agree == len(votes)), so their derivation is unchanged.
        if consensus == "quorum":
            return REASON_QUORUM
        # Any abstention evidence (or impossible counts, handled conservatively)
        # downgrades the claim to quorum; only n_valid == n_votes is unanimous.
        if counts_show_abstention(n_valid, n_votes):
            return REASON_QUORUM
        return REASON_UNANIMOUS
    if consensus in ("unanimous", "quorum"):
        if choice == "NONE":
            return REASON_QUORUM_NONE if consensus == "quorum" else REASON_UNANIMOUS_NONE
        # All-valid agreement on a non-NONE choice yet routed to human review:
        # only the class gate does that (historical rows predating the gate's
        # stamp; the quorum branch mirrors the unanimous one for shape-parity —
        # quorum rows postdate the stamp, so it is unreachable in practice).
        return REASON_CLASS_MISMATCH
    if consensus == "majority":
        if minority:
            return REASON_DISSENT_PREFIX + minority.replace("; ", ",").replace(" ", "")
        if n_valid is not None and n_valid >= 3:
            # LEGACY pre-v5 shape: all valid votes agreed at quorum but the old
            # rule blocked the accept on the abstention. The v5 rule never
            # mints this row (it auto-accepts as "quorum"); historical rows
            # must keep deriving their original code.
            return REASON_ABSTENTION
        return f"{REASON_BELOW_QUORUM_PREFIX}{n_valid if n_valid is not None else '?'}"
    if consensus == "none":
        if n_valid == 0 or minority == "all providers abstained":
            return REASON_ALL_ABSTAINED
        return REASON_NO_MAJORITY
    return ""


def humanize_route_reason(code: str) -> str:
    """Short human-readable variant of a route-reason code for the review UI."""
    code = _clean(code)
    if not code:
        return ""
    fixed = {
        REASON_UNANIMOUS: "panel unanimous — auto-accepted",
        REASON_UNANIMOUS_NONE: "panel unanimous: none of the options fit",
        REASON_QUORUM: "panel quorum — all valid votes agree (abstention forgiven); auto-accepted",
        REASON_QUORUM_NONE: "panel quorum: none of the options fit (abstention forgiven)",
        # LEGACY (pre-v5 rows only): the v5 quorum rule auto-accepts this case.
        REASON_ABSTENTION: "an abstention blocked unanimity",
        REASON_NO_MAJORITY: "panel split — no majority choice",
        REASON_ALL_ABSTAINED: "all panelists abstained",
        REASON_CLASS_MISMATCH: "cross-mode edge (e.g. footway↔road) blocked auto-accept",
        # Size gate: stamped by compute_consensus (and the read-time overlay in
        # latest_panel_consensus) on over-backstop groups; also the legacy
        # phase-2 gate's spelling.
        REASON_SIZE_GATED: "over the size gate — too large to auto-accept",
        # Low-confidence gate: stamped by compute_consensus (and the read-time
        # overlay) when the calibrated voter's confidence was too low.
        REASON_LOW_CONFIDENCE: "low panel confidence — below the auto-accept floor",
        # Decomposition (#367 Mode B): a sub-problem the panel could not
        # auto-accept blocks the whole-group label — review the group as a whole.
        REASON_SUBPROBLEM_FAILED: (
            "a decomposed sub-problem could not be auto-accepted — review the whole group"
        ),
        # Decomposition (#367 Mode B): a roster sub-problem was never voted
        # (oversized/irreducible), so the whole group can neither recompose nor
        # auto-accept — review it as a whole.
        REASON_OVERSIZED_UNVOTED: (
            "a decomposed sub-problem was too large to vote — review the whole group"
        ),
    }
    if code in fixed:
        return fixed[code]
    if code.startswith(REASON_DISSENT_PREFIX):
        frags = []
        for part in code[len(REASON_DISSENT_PREFIX) :].split(","):
            if not part:
                continue
            provider, _, choice = part.partition("=")
            frags.append(
                f"{provider} dissented — voted {choice}" if choice else f"{provider} dissented"
            )
        return "; ".join(frags) or code
    if code.startswith(REASON_BELOW_QUORUM_PREFIX):
        n = code[len(REASON_BELOW_QUORUM_PREFIX) :]
        return f"only {n} valid vote{'' if n == '1' else 's'} — below quorum"
    # Legacy/unknown codes (e.g. phase-2's "majority") pass through readably.
    return code.replace("_", " ")


def candidate_edge_count(meta: Mapping) -> int | None:
    """A group's candidate-edge count from its evidence-pack metadata.

    ``n_edges_full`` is the group's full candidate-edge count
    (:func:`stitch_evidence.build_metadata` stamps it on every pack; it always
    equals ``n_edges_rendered`` post-clipping-fix, which serves as the
    fallback). Returns ``None`` when neither is present — size gating is then
    skipped for the group.
    """
    n = _int_or_none(meta.get("n_edges_full"))
    if n is None:
        n = _int_or_none(meta.get("n_edges_rendered"))
    return n


def _pack_candidate_edge_count(batch_dir: Path, group_id: str) -> int | None:
    """Candidate-edge count of a voted group, read from its evidence pack.

    Returns ``None`` when the pack's ``metadata.yaml`` is missing or
    unparseable (e.g. a batch dir kept only for its CSVs) — callers treat that
    as "no size evidence" and leave the row's routing untouched.
    """
    meta_path = batch_dir / group_id / "metadata.yaml"
    if not meta_path.is_file():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text())
    except (yaml.YAMLError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(meta, Mapping):
        return None
    return candidate_edge_count(meta)


def _float_or_nan(val) -> float:
    """Parse a confidence cell; a blank/unparseable value becomes ``nan``.

    ``nan`` is the deliberate "missing self-report" sentinel — the low-confidence
    gate treats it as below any positive floor (a NaN comparison must never
    silently pass).
    """
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _batch_min_confidences(batch_dir: Path) -> dict[str, float | None]:
    """Per-group minimum confidence across valid votes, from ``votes.csv``.

    Mirrors :func:`stitch_runner.compute_consensus`'s validity rule: a vote whose
    ``choice`` is ``ABSTAIN`` does not count. For each group the value is:

    * ``None`` — no valid (non-abstaining) vote row (or no/unreadable
      ``votes.csv``): no confidence evidence, so the gate leaves the row
      untouched (the same "no evidence -> untouched" stance as the size gate).
    * ``float('nan')`` — a valid vote's confidence is blank/unparseable
      (conservatively below any positive floor).
    * else the numeric minimum across the group's valid votes.
    """
    vpath = batch_dir / "votes.csv"
    if not vpath.is_file():
        return {}
    per_group: dict[str, list[float]] = {}
    try:
        with open(vpath, newline="") as fh:
            for row in csv.DictReader(fh):
                gid = str(row.get("group_id", "") or "").strip()
                if not gid:
                    continue
                if str(row.get("choice", "") or "").strip() == "ABSTAIN":
                    continue
                per_group.setdefault(gid, []).append(_float_or_nan(row.get("confidence")))
    except (OSError, csv.Error, UnicodeDecodeError):
        return {}
    out: dict[str, float | None] = {}
    for gid, confs in per_group.items():
        if any(math.isnan(c) for c in confs):
            out[gid] = float("nan")  # a missing self-report gates conservatively
        else:
            out[gid] = min(confs)
    return out


def _dataset_batch_dirs(dataset: str, batches_root: Path) -> list[Path]:
    """Batch dirs belonging to ``dataset``, oldest consensus.csv first.

    A dir belongs to the dataset when its name equals the dataset or begins with
    ``{dataset}_`` (e.g. ``us_seattle_sidewalks_phase2``). Only dirs that carry a
    ``consensus.csv`` are returned. Ordering is by consensus mtime ascending
    (name as tie-breaker) so a later wave supersedes an earlier one when both
    voted the same group.
    """
    if not dataset or not batches_root.exists():
        return []
    dirs = [
        d
        for d in batches_root.iterdir()
        if d.is_dir()
        and (d.name == dataset or d.name.startswith(dataset + "_"))
        and (d / "consensus.csv").is_file()
    ]
    dirs.sort(key=lambda d: ((d / "consensus.csv").stat().st_mtime, d.name))
    return dirs


def latest_panel_consensus(
    dataset: str,
    batches_root: Path = STITCH_BATCHES_DIR,
) -> dict[str, dict]:
    """Map each voted ``group_id`` to its most-recent consensus row (as a dict).

    Later waves (by consensus mtime) overwrite earlier ones. Returns an empty
    dict when the dataset has no panel batches. Rows with a blank group_id are
    skipped.

    Size-gate overlay (see module docstring): an ``auto_accept`` row on a group
    whose candidate-edge count exceeds the export backstop is returned with its
    EFFECTIVE routing — ``human_review`` / ``route_reason="size_gated"`` (the
    same code :func:`stitch_runner.compute_consensus` stamps going forward).
    The export backstop blocks such a verdict from ever minting a label, so
    treating it as accepted would let it vanish, reviewed by no one. The
    on-disk CSV is never modified; rows whose evidence pack is missing are left
    untouched (no size evidence).

    Low-confidence overlay: an ``auto_accept`` row whose MINIMUM confidence
    across valid votes (read from the batch's ``votes.csv``) is below
    ``settings.stitch_min_voter_confidence`` is likewise returned as
    ``human_review`` / ``route_reason="low_confidence"`` — the same code
    :func:`stitch_runner.compute_consensus` stamps going forward — so a wave
    voted before the gate existed surfaces in the human queue for adjudication
    instead of auto-exporting on the next ``stitch-export``. Applied AFTER the
    size gate (``size_gated`` wins if both). Rows already exported as labels are
    filtered out of the queue by the reviewed-id check in the caller, so this
    never re-surfaces a minted label. A blank/NaN confidence on a valid vote
    counts as below the floor; a group with no confidence evidence is left
    untouched.
    """
    rows: dict[str, dict] = {}
    origins: dict[str, Path] = {}
    for batch_dir in _dataset_batch_dirs(dataset, batches_root):
        with open(batch_dir / "consensus.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                gid = str(row.get("group_id", "") or "").strip()
                if not gid:
                    continue
                rows[gid] = row
                origins[gid] = batch_dir

    backstop = settings.stitch_export_backstop_max_edges
    conf_floor = settings.stitch_min_voter_confidence
    min_conf_by_batch: dict[Path, dict[str, float | None]] = {}
    for gid, row in rows.items():
        if str(row.get("routing", "") or "").strip() != ROUTING_AUTO_ACCEPT:
            continue  # already human-routed; keep the (more specific) reason
        # Quorum floor first: compute_consensus only mints auto_accept at
        # n_valid >= 3, so a row claiming auto_accept with n_valid PRESENT and < 3
        # is a hand-edit / corrupt / pre-quorum-rule artifact. Never treat it as
        # accepted (it would then neither export cleanly nor reach the queue) —
        # surface it as a below-quorum human_review. n_valid missing is no
        # evidence (historical rows lacking the column keep their routing).
        n_valid = _int_or_none(row.get("n_valid"))
        if n_valid is not None and n_valid < 3:
            row["routing"] = ROUTING_HUMAN_REVIEW
            row["route_reason"] = f"{REASON_BELOW_QUORUM_PREFIX}{n_valid}"
            continue
        batch_dir = origins[gid]
        # Size gate first: the structural export block is the decisive fact.
        n_edges = _pack_candidate_edge_count(batch_dir, gid)
        if n_edges is not None and n_edges > backstop:
            row["routing"] = ROUTING_HUMAN_REVIEW
            row["route_reason"] = REASON_SIZE_GATED
            continue
        # Low-confidence gate.
        if conf_floor > 0.0:
            if batch_dir not in min_conf_by_batch:
                min_conf_by_batch[batch_dir] = _batch_min_confidences(batch_dir)
            min_conf = min_conf_by_batch[batch_dir].get(gid)
            if min_conf is not None and (math.isnan(min_conf) or min_conf < conf_floor):
                row["routing"] = ROUTING_HUMAN_REVIEW
                row["route_reason"] = REASON_LOW_CONFIDENCE
    return rows


def latest_panel_routing(
    dataset: str,
    batches_root: Path = STITCH_BATCHES_DIR,
) -> dict[str, str]:
    """Map each voted ``group_id`` to its most-recent panel ``routing`` value.

    Later waves (by consensus mtime) overwrite earlier ones. Returns an empty
    dict when the dataset has no panel batches. Rows with a blank group_id are
    skipped; a blank routing is preserved as-is (it is simply not
    ``human_review``, so it will not gate anything in).
    """
    return {
        gid: str(row.get("routing", "") or "").strip()
        for gid, row in latest_panel_consensus(dataset, batches_root).items()
    }


def _decomposition_rosters(
    dataset: str,
    batches_root: Path = STITCH_BATCHES_DIR,
) -> dict[str, list[str]]:
    """Each decompose-first parent's full sub-problem roster, from ``batch.json``.

    Maps ``{parent_group_id: [subproblem_id, ...]}`` for every ``batch.json``
    group carrying ``decomposed_parent`` + ``subproblem_ids`` (#367 Mode B),
    merged across the dataset's batch dirs (later wave supersedes earlier per
    parent, mirroring the consensus precedence). The roster is the completeness
    contract — it includes oversized sub-problems that were never packed/voted —
    so it is the only source that reveals a parent whose voted-sub set is
    INCOMPLETE. Batch dirs are matched by the same name rule as
    :func:`_dataset_batch_dirs` but keyed on ``batch.json`` presence (a
    decompose-first parent may have no ``consensus.csv`` row of its own). Returns
    ``{}`` when nothing decomposed.
    """
    if not dataset or not batches_root.exists():
        return {}
    dirs = [
        d
        for d in batches_root.iterdir()
        if d.is_dir()
        and (d.name == dataset or d.name.startswith(dataset + "_"))
        and (d / "batch.json").is_file()
    ]
    dirs.sort(key=lambda d: ((d / "batch.json").stat().st_mtime, d.name))
    rosters: dict[str, list[str]] = {}
    for d in dirs:
        try:
            batch = json.loads((d / "batch.json").read_text())
        except (ValueError, OSError, UnicodeDecodeError):
            continue
        if not isinstance(batch, Mapping):
            continue
        for grp in batch.get("groups", []):
            if not isinstance(grp, Mapping):
                continue
            if grp.get("decomposed_parent") and grp.get("subproblem_ids"):
                rosters[str(grp.get("group_id"))] = [str(s) for s in grp["subproblem_ids"]]
    return rosters


def unvoted_decomposed_parents(
    dataset: str,
    batches_root: Path = STITCH_BATCHES_DIR,
) -> set[str]:
    """Decompose-first parents with an unvoted (oversized) roster sub-problem.

    A monster that went straight to decomposition has no direct whole-group vote;
    when one of its roster sub-problems was skipped as an irreducible oversized
    block it has NO consensus row, so recomposition is permanently incomplete
    (all-or-nothing) — the group can never auto-accept — yet no failed sub-vote
    exists to fold it onto the queue (see :func:`panel_failed_group_ids`). That is
    a residual queue-void (#403 follow-up): the group would be reviewed by no one.

    Returns each such parent id: it has a batch.json roster, no direct consensus
    row of its own, and at least one roster sub-problem with no consensus row.
    Parents whose entire roster was voted are excluded (they either recompose/
    export cleanly or already surface via a failed sub-problem).
    """
    rosters = _decomposition_rosters(dataset, batches_root)
    if not rosters:
        return set()
    voted = set(latest_panel_consensus(dataset, batches_root).keys())
    void: set[str] = set()
    for parent, roster in rosters.items():
        if parent in voted:
            continue  # a direct (e.g. size-gated) parent row already routes it
        if any(sid not in voted for sid in roster):
            void.add(parent)
    return void


def panel_failed_group_ids(
    dataset: str,
    batches_root: Path = STITCH_BATCHES_DIR,
) -> set[str]:
    """Group ids whose most-recent panel vote routed to ``human_review``.

    This is the allow-list for the human review queue: the groups the agent
    panel could not auto-accept. Groups never voted by the panel are absent (they
    do not enter the human queue by default).

    Decomposition fold (#367 Mode B): a decomposed group's panel votes key on
    sub-problem ids (``{parent}__p...``), which are NOT sidecar groups, so a
    sub-problem the panel failed would be filtered out of the queue against the
    sidecar and — for a monster that went STRAIGHT to decomposition (no
    size-gated whole-group vote of its own) — the parent would be invisible in
    every review queue. Each failed sub-problem is therefore folded onto its
    PARENT group id (the sidecar entry the reviewer adjudicates): recomposition
    is all-or-nothing, so one failed sub-problem blocks the whole-group label.

    Residual void (#403 follow-up): a decompose-first parent whose roster has an
    UNVOTED (oversized) sub-problem has neither a failed sub-vote to fold nor a
    direct parent vote — it can never auto-accept but nothing surfaces it. Such
    parents (see :func:`unvoted_decomposed_parents`) are added directly so they
    reach the human queue instead of vanishing.
    """
    failed: set[str] = set()
    for gid, routing in latest_panel_routing(dataset, batches_root).items():
        if routing != ROUTING_HUMAN_REVIEW:
            continue
        # A failed sub-problem surfaces its PARENT (the sidecar group); a plain
        # group surfaces itself.
        failed.add(parent_group_id_of(gid) or gid)
    # Decompose-first parents blocked only by an unvoted oversized sub-problem
    # have no human_review row above — surface them explicitly.
    failed |= unvoted_decomposed_parents(dataset, batches_root)
    return failed


def attach_panel_route_reasons(
    groups: list[dict],
    dataset: str,
    batches_root: Path = STITCH_BATCHES_DIR,
) -> int:
    """Annotate review-queue groups with WHY the panel routed them to a human.

    For each group dict whose ``group_id`` has a panel consensus row, sets
    ``panel_route_reason`` (machine-readable code, see :func:`derive_route_reason`)
    and ``panel_route_reason_human`` (short display string). Annotation only —
    never touches routing or selection; groups the panel never voted on are left
    untouched. Returns the number of groups annotated.

    Decomposition fold (#367 Mode B): a decompose-first monster has no direct
    consensus row of its own — only its sub-problems were voted. When such a
    parent surfaces in the queue because a sub-problem failed (see
    :func:`panel_failed_group_ids`), it is annotated with
    :data:`REASON_SUBPROBLEM_FAILED` so the reviewer sees WHY the whole group is
    up for review even though the panel never voted it directly.
    """
    rows = latest_panel_consensus(dataset, batches_root)
    unvoted_parents = unvoted_decomposed_parents(dataset, batches_root)
    if not rows and not unvoted_parents:
        return 0
    # Parents with >=1 sub-problem the panel routed to human_review. A direct
    # consensus row (below) always wins over this synthesized reason.
    failed_sub_parents: set[str] = set()
    for gid, row in rows.items():
        if str(row.get("routing", "") or "").strip() != ROUTING_HUMAN_REVIEW:
            continue
        parent = parent_group_id_of(gid)
        if parent is not None:
            failed_sub_parents.add(parent)
    n = 0
    for group in groups:
        gid = str(group.get("group_id", "") or "").strip()
        row = rows.get(gid)
        if row is not None:
            code = derive_route_reason(row)
        elif gid in failed_sub_parents:
            # A concrete failed sub-vote is more actionable than an unvoted one,
            # so it wins when a parent has both.
            code = REASON_SUBPROBLEM_FAILED
        elif gid in unvoted_parents:
            code = REASON_OVERSIZED_UNVOTED
        else:
            continue
        if not code:
            continue
        group["panel_route_reason"] = code
        group["panel_route_reason_human"] = humanize_route_reason(code)
        n += 1
    return n
