"""Resolve the agent panel's per-group routing for a dataset.

The 3-provider stitching panel (see :mod:`stitch_runner`) votes on M:N groups
and writes a ``consensus.csv`` per batch dir with a ``routing`` column whose
values are ``auto_accept`` (unanimous, safe to promote) or ``human_review``
(non-unanimous vote, NONE consensus, cross-mode flags, oversize, ...). A group
that the panel could not auto-accept is exactly the kind of decision worth a
human's 1-2 minutes.

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
candidate-edge count exceeds it can ever mint a label. An ``auto_accept``
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
"""

from __future__ import annotations

import csv
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from ..config import settings
from ..filenames import PROJECT_ROOT


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

#: Unanimous non-NONE verdict — the auto-accept path.
REASON_UNANIMOUS = "unanimous"
#: All panelists voted NONE — every offered option was rejected.
REASON_UNANIMOUS_NONE = "unanimous_none"
#: Majority with dissenting valid vote(s); suffix is the minority summary,
#: e.g. ``dissent:codex=B`` or ``dissent:codex=F,agy=A``.
REASON_DISSENT_PREFIX = "dissent:"
#: All valid votes agree but fewer than 3 were valid (quorum for unanimity);
#: suffix is n_valid, e.g. ``below_quorum:2``.
REASON_BELOW_QUORUM_PREFIX = "below_quorum:"
#: >=3 valid votes all agree, but an abstention blocked full unanimity
#: (only reachable with a 4-voter panel).
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

    if routing == ROUTING_AUTO_ACCEPT:
        return REASON_UNANIMOUS
    if consensus == "unanimous":
        if choice == "NONE":
            return REASON_UNANIMOUS_NONE
        # Unanimous non-NONE yet routed to human review: only the class gate
        # does that (historical rows predating the gate's stamp).
        return REASON_CLASS_MISMATCH
    if consensus == "majority":
        if minority:
            return REASON_DISSENT_PREFIX + minority.replace("; ", ",").replace(" ", "")
        if n_valid is not None and n_valid >= 3:
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
        REASON_ABSTENTION: "an abstention blocked unanimity",
        REASON_NO_MAJORITY: "panel split — no majority choice",
        REASON_ALL_ABSTAINED: "all panelists abstained",
        REASON_CLASS_MISMATCH: "cross-mode edge (e.g. footway↔road) blocked auto-accept",
        # Size gate: stamped by compute_consensus (and the read-time overlay in
        # latest_panel_consensus) on over-backstop groups; also the legacy
        # phase-2 gate's spelling.
        REASON_SIZE_GATED: "over the size gate — too large to auto-accept",
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
    for gid, row in rows.items():
        if str(row.get("routing", "") or "").strip() != ROUTING_AUTO_ACCEPT:
            continue  # already human-routed; keep the (more specific) reason
        n_edges = _pack_candidate_edge_count(origins[gid], gid)
        if n_edges is not None and n_edges > backstop:
            row["routing"] = ROUTING_HUMAN_REVIEW
            row["route_reason"] = REASON_SIZE_GATED
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


def panel_failed_group_ids(
    dataset: str,
    batches_root: Path = STITCH_BATCHES_DIR,
) -> set[str]:
    """Group ids whose most-recent panel vote routed to ``human_review``.

    This is the allow-list for the human review queue: the groups the agent
    panel could not auto-accept. Groups never voted by the panel are absent (they
    do not enter the human queue by default).
    """
    return {
        gid
        for gid, routing in latest_panel_routing(dataset, batches_root).items()
        if routing == ROUTING_HUMAN_REVIEW
    }


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
    """
    rows = latest_panel_consensus(dataset, batches_root)
    if not rows:
        return 0
    n = 0
    for group in groups:
        row = rows.get(str(group.get("group_id", "") or "").strip())
        if row is None:
            continue
        code = derive_route_reason(row)
        if not code:
            continue
        group["panel_route_reason"] = code
        group["panel_route_reason_human"] = humanize_route_reason(code)
        n += 1
    return n
