"""Export accepted LLM-panel consensus into human-equivalent stitching labels.

The stitching panel (see :mod:`stitch_runner`) votes on M:N group edge
selections and writes a ``consensus.csv`` per batch. This module promotes the
subset of those verdicts that are safe to treat as durable labels into
``labels/stitching`` alongside the human labels, tagged with a ``panel_*``
labeler so their provenance stays visible (v1 tagged the earlier
sonnet/gpt-5.4/Gemini-Flash-Low panel; v2 the Opus 4.8/gpt-5.5/Gemini-3.5-Flash
panel on pre-enrichment packs; v3 that composition on #302-enriched packs; v4
the 2026-07-09 bless — Opus 4.8 / gpt-5.6-terra / Kimi K2.6 (the codex model was
swapped gpt-5.6-sol -> gpt-5.6-terra in place on 2026-07-10, no era bump, as v4
had minted no committed rows); v5 the 2026-07-10 quad bless — the v4 trio PLUS
muse/Muse Spark 1.1, paired with the quorum consensus rule. The tag is bumped
whenever the panel composition OR its pack inputs change, and each batch is
stamped with ITS OWN era's tag — see :data:`STANDARD_PANEL_VOTERS`).

One directly voted panel verdict class is promoted:

  * **Accept** (routing == ``auto_accept``) -> a normal pair label with the
    panel's chosen edge set. Since v5 the accept tier is part of provenance:
    a FULLY UNANIMOUS accept (every recorded vote valid and agreeing, e.g.
    4/4) is tagged ``panel_unanimous_v5`` while a QUORUM accept (all valid
    votes agree over >=1 abstention, e.g. 3-of-4) is tagged the DISTINCT
    ``panel_quorum_v5`` — the two must stay distinguishable end-to-end.
    v4/v3-era batches keep their ``panel_unanimous_v4``/``_v3`` tags (their
    3-voter rule could not produce a quorum accept).
Panel ``NONE`` is never promoted directly. It deliberately covers three
different panel outcomes: every edge is a no-match, no offered option is exact,
or the evidence is insufficient. Consensus routes every ``NONE`` to human
review, where an explicit human reject-all writes the unambiguous empty-set
label if appropriate. Historical ``panel_*_none_*`` labels remain readable,
but this exporter cannot mint new ones.

A second, recomposed class covers DECOMPOSED groups (#367 Mode B, ``stitch-batch
--decompose``): an over-backstop group split into panel-sized sub-problems is
recomposed here — a whole-group label (the union of the sub-selections) is
minted ONLY when every sub-problem in the batch.json roster resolved as a
panel accept; any failed or unvoted
sub-problem blocks the group (``subproblem_failed`` / ``subproblems_unvoted``),
as does a sub-verdict set whose contributing batch dirs resolve to different
panel eras (``subproblem_era_mixed`` — a mixed-composition union must not be
stamped under a single era). The recomposed labeler is
``panel_unanimous_decomposed_v5`` only when EVERY consumed sub-verdict was
fully unanimous; if ANY sub-problem was quorum-accepted the whole
recomposition is conservatively tagged ``panel_quorum_decomposed_v5`` (quorum
taints the union — the label's weakest link names it). Sub-problem consensus
rows are consumed by that recomposition and never export individually. See
:mod:`crosswalk.matching.group_decomposition`.

Historical panel empty-set labels use the same on-disk representation as a
human reject-all review and remain readable by existing consumers. This is
backward compatibility only; new reject-all truth comes from human review.

Gates (applied in order; the first failing gate decides the group and is
reported):

  a. routing gate -- only an ``auto_accept`` row is a candidate. ``NONE`` and
     every other human-review outcome are never panel-exportable.
  b. size gate -- huge/tangled groups stay for human review (structural gate,
     or flat ``max_edges`` fallback).
  c. class-consistency gate -- reuses the panel runner's cross-mode rule
     (:func:`stitch_runner.has_cross_mode_edge`) on the chosen edge set
     (pedestrian / vehicular / bike; any two different modes are cross-mode).
  d. sliver exactness gate -- if the exact voted set contains a geometry-tagged
     sliver, the group is skipped for human review. Export never silently edits
     an exact panel selection by deleting edges.
  e. human precedence -- a group already covered by a *human* label (by exact
     group_id or by edge-overlap, reusing :func:`stitch_eval.map_human_labels_to_groups`)
     is left untouched. Applies to direct and recomposed paths.

Writing is idempotent: rows are upserted by ``group_id`` under the appropriate
``panel_*`` labeler, so re-running never duplicates and always refreshes to the
latest consensus. Previously exported panel rows (any ``panel_`` prefix) are
excluded from the human-precedence check (they are not human), so re-runs stay
accurate.
"""

from __future__ import annotations

import json
import math
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import settings
from ..labeling.stitching_store import StitchingLabelStore
from ..matching.group_decomposition import (
    STATUS_FAILED,
    STATUS_UNVOTED,
    Recomposition,
    recompose_subproblem_verdicts,
)
from ..matching.optimizer import group_is_structurally_simple
from ..matching.sliver import annotate_group_sliver_flags
from .matching_rubric import MATCHING_RUBRIC_VERSION
from .panel_routing import (
    REASON_CONTAINS_SLIVER,
    REASON_QUORUM,
    _int_or_none,
    candidate_edge_count,
    counts_show_abstention,
    derive_route_reason,
)
from .stitch_eval import (
    _is_set_label,
    _load_group_metadata,
    _map_set_labels_to_groups,
    map_human_labels_to_groups,
)
from .stitch_provenance import (
    canonical_json,
    consensus_policy_signature,
    load_evidence_manifest,
    safe_group_id,
    sha256_file,
    validate_manifest_against_batch,
)
from .stitch_runner import (
    GEMINI_ROUTE_AGY,
    GEMINI_ROUTE_OPENROUTER_FLEX,
    Vote,
    _edge_classes_for,
    _load_group_context,
    _segment_class_maps,
    compute_consensus,
    has_cross_mode_edge,
)

# Bumped v1 -> v2 when the panel composition changed (Opus 4.8 / gpt-5.5 /
# Gemini 3.5 Flash Medium); v2 -> v3 when the evidence-pack inputs changed
# (#302 enrichment: per-edge overlap meters, BORDERLINE tags, junction zoom
# crops -- votes are not comparable across pack versions, see
# research/panel_enriched_ab.md); v3 -> v4 when the composition changed again
# (2026-07-09 bless: agy/Gemini replaced by kimi/Kimi K2.6, codex bumped
# gpt-5.5 -> gpt-5.6-sol; validated in #397). The v4 codex model was later
# swapped gpt-5.6-sol -> gpt-5.6-terra IN PLACE (2026-07-10, no era bump: v4 had
# minted no committed rows); v4 -> v5 when the composition changed again
# (2026-07-10 quad bless: the v4 trio PLUS muse/Muse Spark 1.1, blessed
# together with the quorum consensus rule). Existing v1/v2/v3 labels stay
# untouched — the era-suffixed constants below remain the write-time tags for
# their eras (see :data:`STANDARD_PANEL_VOTERS` era scoping) — and new default-
# panel waves are tagged v5. Any labeler with the PANEL_LABELER_PREFIX is a
# panel (non-human) label and is excluded from the human-precedence check below.
PANEL_LABELER_V3 = "panel_unanimous_v3"
PANEL_NONE_LABELER_V3 = "panel_unanimous_none_v3"
PANEL_DECOMPOSED_LABELER_V3 = "panel_unanimous_decomposed_v3"

# v4-era tags. The v4 3-voter rule could not produce a quorum accept (all-valid
# agreement among 3 voters IS full unanimity), so v4 has no quorum variants.
PANEL_LABELER_V4 = "panel_unanimous_v4"
PANEL_NONE_LABELER_V4 = "panel_unanimous_none_v4"
PANEL_DECOMPOSED_LABELER_V4 = "panel_unanimous_decomposed_v4"

# Current-era (v5) tags. The unsuffixed names always track the CURRENT era.
PANEL_LABELER = "panel_unanimous_v5"
PANEL_LABELER_PREFIX = "panel_"

# QUORUM-accept labeler (v5+): all valid votes agreed but >=1 panelist
# abstained (e.g. 3-of-4). A *separate* tag (rather than reusing PANEL_LABELER)
# keeps the provenance distinction end-to-end: a 4/4 unanimous accept and a
# 3-of-4 quorum accept are different evidentiary claims, and per-labeler eval
# must be able to slice them apart (e.g. to audit whether quorum accepts are
# noisier). Kept under the ``panel_`` prefix (non-human).
PANEL_QUORUM_LABELER = "panel_quorum_v5"

# Historical reject-all labelers remain defined so committed rows stay readable
# and attributable. The exporter no longer mints these tags: panel NONE is
# overloaded and must be confirmed by a human before becoming empty-set truth.
PANEL_NONE_LABELER = "panel_unanimous_none_v5"
PANEL_QUORUM_NONE_LABELER = "panel_quorum_none_v5"

# Distinct labeler for a RECOMPOSED whole-group label (#367 Mode B): the union
# of accepted per-sub-problem verdicts from a decomposed over-backstop group.
# No single panel saw the whole group, so this is a different labeling process
# than PANEL_LABELER and must stay sliceable on its own in per-labeler eval.
# Kept under the ``panel_`` prefix (non-human); version suffix tracks
# PANEL_LABELER (same panel composition / pack inputs per sub-problem). The
# unanimous variant requires EVERY consumed sub-verdict to be fully unanimous;
# if ANY sub-problem was quorum-accepted the whole recomposition is
# conservatively stamped with the quorum variant (quorum taints the union —
# the label's weakest link names it).
PANEL_DECOMPOSED_LABELER = "panel_unanimous_decomposed_v5"
PANEL_QUORUM_DECOMPOSED_LABELER = "panel_quorum_decomposed_v5"

# v6-candidate tags. These exist before promotion so an explicitly approved
# candidate export can never be mislabeled as v5. The composition remains OUT
# of STANDARD_PANEL_VOTERS until calibration passes, so stitch-export still
# requires --allow-nonstandard-panel. DEFAULT_PANEL and the unsuffixed labeler
# constants continue to point at the blessed v5 production panel.
PANEL_LABELER_V6 = "panel_unanimous_v6"
PANEL_QUORUM_LABELER_V6 = "panel_quorum_v6"
PANEL_NONE_LABELER_V6 = "panel_unanimous_none_v6"
PANEL_QUORUM_NONE_LABELER_V6 = "panel_quorum_none_v6"
PANEL_DECOMPOSED_LABELER_V6 = "panel_unanimous_decomposed_v6"
PANEL_QUORUM_DECOMPOSED_LABELER_V6 = "panel_quorum_decomposed_v6"

# v7-candidate tags. V7 is a distinct generation rather than an in-place v6
# edit: it changes Codex Terra -> Sol and records high effort across the lean
# Claude/Codex/Muse panel. It remains nonstandard until the canonical-rubric
# replay is manually reviewed and explicitly promoted.
PANEL_LABELER_V7 = "panel_unanimous_v7"
PANEL_QUORUM_LABELER_V7 = "panel_quorum_v7"
PANEL_NONE_LABELER_V7 = "panel_unanimous_none_v7"
PANEL_QUORUM_NONE_LABELER_V7 = "panel_quorum_none_v7"
PANEL_DECOMPOSED_LABELER_V7 = "panel_unanimous_decomposed_v7"
PANEL_QUORUM_DECOMPOSED_LABELER_V7 = "panel_quorum_decomposed_v7"

#: Blessed (provider, model) voter compositions, keyed by labeler era. The gate
#: keys on the PAIR, not the provider name alone: the opencode transport has
#: driven Gemini Flash (no-agy quota-outage waves) and Qwen3-VL (v3-candidate)
#: under the SAME ``opencode`` provider name, and a provider-name-only set cannot
#: tell them apart. (The blessed v4 Kimi voter now carries its own ``kimi``
#: provider name; older opencode-transport rows still key on ``opencode``.)
PANEL_VOTERS_V3 = frozenset(
    {
        ("claude", "claude-opus-4-8"),
        ("codex", "gpt-5.5"),
        ("agy", "Gemini 3.5 Flash (Medium)"),
    }
)
PANEL_VOTERS_V4 = frozenset(
    {
        ("claude", "claude-opus-4-8"),
        # In-place codex swap gpt-5.6-sol -> gpt-5.6-terra (2026-07-10, Brad
        # waived an era bump): v4 has minted ZERO committed rows (the only codex
        # model in committed labels/votes is v3-era gpt-5.5), so editing the
        # blessed pair rewrites nothing on disk — same argument as the #402 kimi
        # rename. Both are the same gpt-5.6 family; terra is a lower quota class,
        # more apples-to-apples with opus/medium (sol showed 8/8 choice-agreement
        # with opus at >=0.86 conf in the 2026-07-10 sol-anchor wave — no
        # decorrelated signal to justify the premium quota).
        ("codex", "gpt-5.6-terra"),
        ("kimi", "openrouter/moonshotai/kimi-k2.6"),
    }
)

#: The v5 quad (2026-07-10 bless): the v4 trio plus muse/Muse Spark 1.1.
#: Introducing v5 carries NO committed-provenance constraint: zero committed
#: labels/votes rows reference any gpt-5.6, kimi/Moonshot, or muse model (v4
#: minted nothing committed; the only committed codex model is v3-era gpt-5.5),
#: verified against the tracked ``labels/`` tree at bless time. The 2026-07-10
#: ``*_quadcal0710`` calibration batches DO match this set exactly (they were
#: voted with the quad composition) and thus resolve to era v5 on export — safe
#: because every calibration group corresponds to an existing human label and
#: human precedence blocks the export (verified; the Boston batch additionally
#: mixes sol+terra codex ballots -> era-less -> refused outright).
PANEL_VOTERS_V5 = frozenset(
    {
        ("claude", "claude-opus-4-8"),
        ("codex", "gpt-5.6-terra"),
        ("kimi", "openrouter/moonshotai/kimi-k2.6"),
        ("muse", "meta/muse-spark-1.1"),
    }
)

# Candidate v6: lean three-seat Claude/Codex/Muse. Calibration found no routing
# lift from replacing Kimi with Gemini, so Gemini remains experimental and is
# not part of this labeler generation's voter identity.
PANEL_VOTERS_V6 = frozenset(
    {
        ("claude", "claude-opus-4-8"),
        ("codex", "gpt-5.6-terra"),
        ("muse", "meta/muse-spark-1.1"),
    }
)

# Candidate v7: the canonical-rubric high-effort replay roster. Effort is
# preserved by panel_invocation_sha256; the era's voter identity remains keyed
# on (provider, model), matching every historical export generation.
PANEL_VOTERS_V7 = frozenset(
    {
        ("claude", "claude-opus-4-8"),
        ("codex", "gpt-5.6-sol"),
        ("muse", "meta/muse-spark-1.1"),
    }
)

#: Era -> blessed voter set. A batch matching an era's set exactly is STANDARD
#: for that era: it passes the export gate and its labels are stamped with that
#: era's labeler tags (v3-era batches keep minting ``*_v3`` labels on
#: re-export — the committed v3 history is never retroactively flagged as
#: nonstandard, nor silently re-stamped with the current era's tags).
STANDARD_PANEL_VOTERS: dict[str, frozenset[tuple[str, str]]] = {
    "v3": PANEL_VOTERS_V3,
    "v4": PANEL_VOTERS_V4,
    "v5": PANEL_VOTERS_V5,
}

# Known candidate compositions paired with the current matching rubric resolve
# to their own labeler generation but do NOT pass nonstandard_panel_batches.
# This separates "we know how to stamp it" from "it passed calibration".
CANDIDATE_ERA_VOTERS: dict[frozenset[tuple[str, str]], str] = {
    PANEL_VOTERS_V6: "v6",
    PANEL_VOTERS_V7: "v7",
}

#: STAMPING-ONLY historical compositions -> labeler era. These compositions
#: were NEVER a blessed default — they still fail the export gate and require
#: ``--allow-nonstandard-panel``, exactly as when they were first exported —
#: but they DID mint committed labels under a specific era via an explicit
#: operator decision, so era resolution must keep attributing them to that
#: era: a re-export must re-stamp the SAME tag, never drift to the current one.
#:
#: * 2026-07-07 8-dataset wave (commit 80dbe1f, #356): the Gemini third voter
#:   ran via the opencode transport instead of agy (Brad-approved swap during
#:   the agy quota outage) — same model, different provider string. Six
#:   datasets' committed ``*_v3`` labels trace to this composition (pairs
#:   verified against ``labels/votes/dataset=*/votes.csv``).
HISTORICAL_ERA_VOTERS: dict[frozenset[tuple[str, str]], str] = {
    frozenset(
        {
            ("claude", "claude-opus-4-8"),
            ("codex", "gpt-5.5"),
            ("opencode", "openrouter/google/gemini-3.5-flash"),
        }
    ): "v3",
}

#: The (provider, model) composition of the CURRENT default panel — must stay
#: in lockstep with ``stitch_runner.DEFAULT_PANEL`` (asserted in
#: tests/unit/test_stitch_export.py, so a panel change without a provenance
#: decision here fails CI). Replaces the provider-name-only
#: ``DEFAULT_PANEL_PROVIDERS``.
DEFAULT_PANEL_VOTERS = PANEL_VOTERS_V5


def _batch_voters(batch_dir: Path) -> set[tuple[str, str]] | None:
    """Read a batch's ``votes.csv`` into its set of (provider, model) voters.

    Returns ``None`` when ``votes.csv`` is missing, unreadable, or has no rows
    (the CLI already hard-requires ``consensus.csv``; provenance for such
    batches is best-effort). A missing ``model`` column or a blank/NaN model
    cell yields pairs with ``model == ""`` — deliberately KEPT rather than
    skipped, so incomplete provenance reads as a composition mismatch (flagged
    by :func:`nonstandard_panel_batches`), never as the blessed panel and never
    as a crash.
    """
    votes_path = Path(batch_dir) / "votes.csv"
    if not votes_path.exists():
        return None
    try:
        df = pd.read_csv(votes_path)
        providers = df["provider"].fillna("").astype(str).str.strip()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError):
        return None
    if "model" in df.columns:
        models = df["model"].fillna("").astype(str).str.strip()
    else:
        models = pd.Series([""] * len(df), index=df.index, dtype=str)
    voters = set(zip(providers, models, strict=True))
    return voters or None


def _batch_matching_rubric_versions(batch_dir: Path) -> frozenset[str]:
    """Return rubric versions bound to the batch's voted evidence packs.

    ``""`` denotes a legacy/missing rubric stamp. ``"<invalid>"`` denotes a
    present but unverifiable pack or ballot/consensus linkage. A non-empty
    rubric stamp counts only when the complete pack verifies and every ballot
    plus the consensus row names its exact evidence and pack hashes.
    """
    batch_dir = Path(batch_dir)
    votes_path = batch_dir / "votes.csv"
    try:
        votes = pd.read_csv(votes_path, dtype={"group_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        return frozenset({""})
    if "group_id" not in votes.columns:
        return frozenset({"<invalid>"})
    consensus_path = batch_dir / "consensus.csv"
    try:
        consensus = pd.read_csv(consensus_path, dtype={"group_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        consensus = pd.DataFrame()
    if consensus_path.exists() and "group_id" not in consensus.columns:
        return frozenset({"<invalid>"})

    def _group_ids(frame: pd.DataFrame) -> set[str]:
        if "group_id" not in frame.columns:
            return set()
        ids: set[str] = set()
        for value in frame["group_id"]:
            if pd.isna(value):
                raise ValueError("group_id is null")
            group_id = safe_group_id(str(value).strip())
            ids.add(group_id)
        return ids

    try:
        vote_group_ids = _group_ids(votes)
        consensus_group_ids = _group_ids(consensus)
    except (TypeError, ValueError):
        return frozenset({"<invalid>"})

    link_fields = {"evidence_id", "evidence_pack_sha256"}
    any_link_fields = bool(
        link_fields.intersection(votes.columns) or link_fields.intersection(consensus.columns)
    )
    full_linkage = link_fields.issubset(votes.columns) and link_fields.issubset(consensus.columns)

    def _rows_bind_manifest(frame: pd.DataFrame, group_id: str, manifest: dict) -> bool:
        rows = frame[frame["group_id"].astype(str) == group_id]
        if rows.empty:
            return False
        expected = {
            "evidence_id": str(manifest["evidence"]["evidence_id"]),
            "evidence_pack_sha256": str(manifest["evidence_pack_sha256"]),
        }
        for column, value in expected.items():
            cells = rows[column].fillna("").astype(str).str.strip()
            if not cells.eq(value).all():
                return False
        return True

    manifests: dict[str, dict] = {}
    versions: dict[str, str] = {}
    all_group_ids = vote_group_ids | consensus_group_ids
    for group_id in sorted(all_group_ids):
        group_dir = batch_dir / group_id
        if not (group_dir / "evidence.json").exists():
            versions[group_id] = ""
            continue
        try:
            manifest = load_evidence_manifest(group_dir, allow_legacy=False)
            evidence = manifest.get("evidence")
            if not isinstance(evidence, dict):
                raise ValueError("evidence record is not an object")
            versions[group_id] = str(evidence.get("matching_rubric_version") or "")
            manifests[group_id] = manifest
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return frozenset({"<invalid>"})

    # A stamped rubric is a batch-wide relational contract, not merely a field
    # found on whichever ballot groups happen to be convenient to inspect.
    # Once any pack declares a rubric (or either CSV carries linkage fields),
    # require exact vote/consensus group agreement, one consensus row per group,
    # and matching links on every ballot and consensus row. This prevents a
    # consensus-only group from borrowing another group's verified v6 identity.
    linked_profile = any(versions.values()) or any_link_fields
    if not linked_profile:
        return frozenset({""})
    if (
        not full_linkage
        or not vote_group_ids
        or vote_group_ids != consensus_group_ids
        or any(not versions.get(group_id) for group_id in vote_group_ids)
    ):
        return frozenset({"<invalid>"})
    for group_id in sorted(vote_group_ids):
        if int((consensus["group_id"].astype(str).str.strip() == group_id).sum()) != 1:
            return frozenset({"<invalid>"})
        manifest = manifests[group_id]
        if not _rows_bind_manifest(votes, group_id, manifest) or not _rows_bind_manifest(
            consensus, group_id, manifest
        ):
            return frozenset({"<invalid>"})
    return frozenset(versions.values())


def batch_panel_era(batch_dir: Path) -> str | None:
    """Return the labeler era for a batch's composition *and* rubric profile.

    Resolution order: the blessed sets (:data:`STANDARD_PANEL_VOTERS`), then the
    stamping-only historical map (:data:`HISTORICAL_ERA_VOTERS`) — compositions
    that minted committed labels under an era via an explicit operator decision
    but were never a blessed default. The historical map affects STAMPING only;
    the export gate (:func:`nonstandard_panel_batches`) still flags those
    batches, exactly as it did when they were first exported.

    ``None`` means the batch is attributable to no known era: an unknown
    composition, no readable ``votes.csv``, or a valid rubric profile that is
    not a known candidate generation. Non-invalid era-less batches are refused
    at write time unless the operator declares an era explicitly
    (``plan_exports(stamp_era=...)`` / CLI ``--stamp-era``). Invalid rubric
    provenance is stricter: :func:`plan_exports` rejects it before any explicit
    stamp can apply. Neither case silently defaults to the current era.
    """
    voters = _batch_voters(batch_dir)
    if voters is None:
        return None
    rubric_versions = _batch_matching_rubric_versions(batch_dir)
    current_rubric = rubric_versions == frozenset({MATCHING_RUBRIC_VERSION})
    # Candidate v6/v7 tags are defined by both their lean voter composition and
    # the refined canonical rubric. Pre-rubric canaries with either roster are
    # deliberately era-less and cannot be exported under the new meaning.
    candidate_era = CANDIDATE_ERA_VOTERS.get(frozenset(voters))
    if candidate_era is not None:
        return candidate_era if current_rubric else None
    # v3-v5 are historical prompt eras. Any non-legacy rubric profile (current,
    # unknown, mixed, or invalid) is a different/unattributable process.
    if rubric_versions != frozenset({""}):
        return None
    for era, blessed in STANDARD_PANEL_VOTERS.items():
        if voters == blessed:
            return era
    frozen = frozenset(voters)
    return HISTORICAL_ERA_VOTERS.get(frozen)


def nonstandard_panel_batches(
    batch_dirs: list[Path],
    expected: frozenset[tuple[str, str]] | None = None,
) -> dict[str, set[tuple[str, str]]]:
    """Return ``{batch_name: voter_pairs}`` for batches with a nonstandard panel.

    Reads each batch's ``votes.csv`` (provider, model) pairs and flags any batch
    whose voter set is not a blessed composition. With the default
    ``expected=None``, a batch passes when it exactly matches ANY era's blessed
    set (:data:`STANDARD_PANEL_VOTERS`): historical v3 batches (claude+codex+agy
    with their v3 models) stay standard instead of being retroactively flagged,
    and v4 batches validate against the v4 set. Current/unknown/invalid rubric
    profiles remain candidates and are flagged until a new production process
    is blessed.
    Pass an explicit ``expected`` frozenset of (provider, model) pairs to pin
    and explicitly accept a single composition.

    Keying on the pair (not the provider name) is the point of this gate: the
    two unblessed opencode-transport voters opencode/Gemini and opencode/Qwen
    (which share the ``opencode`` provider name) are distinguishable from each
    other and from the blessed kimi/Kimi voter by their model. A vote row with a
    blank/missing model reads as ``(provider, "")`` and therefore flags — it is
    never treated as standard and never a crash. Batches with a
    missing/unreadable ``votes.csv`` are skipped (the CLI already hard-requires
    ``consensus.csv``; provenance for such batches is best-effort).
    """
    accepted = (
        [frozenset(expected)] if expected is not None else list(STANDARD_PANEL_VOTERS.values())
    )
    offending: dict[str, set[tuple[str, str]]] = {}
    for bd in batch_dirs:
        voters = _batch_voters(bd)
        if voters is None:
            continue
        rubric_versions = _batch_matching_rubric_versions(bd)
        legacy_rubric = rubric_versions == frozenset({""})
        current_rubric = rubric_versions == frozenset({MATCHING_RUBRIC_VERSION})
        rubric_blocked = (not legacy_rubric and not current_rubric) or (
            expected is None and current_rubric
        )
        if rubric_blocked or not any(voters == blessed for blessed in accepted):
            offending[Path(bd).name] = voters
    return offending


# Per-group outcome reasons (stable strings for reporting/tests).
REASON_EXPORTED = "exported"
REASON_OVER_MAX = "over_max_edges"
REASON_STRUCTURAL_TANGLE = "structural_tangle"
REASON_CLASS_MISMATCH = "class_mismatch"
# Historical report reason retained for compatibility with archived output.
REASON_EMPTIED_BY_SLIVER = "emptied_by_sliver"
REASON_EMPTY_SELECTION = "empty_selection"
REASON_HUMAN_PRECEDENCE = "human_precedence"
# A batch whose ``experiment.variant`` is a context-stripped ablation cell
# (anything present and != ``enriched``). Its ballots are experiment data and
# must never mint a production label, even when a group routes ``auto_accept``.
REASON_ABLATION_VARIANT = "ablation_variant"
# Decomposed-group (recomposition) outcomes: a sub-problem the panel could not
# unanimously accept, or one never voted (including size-gated irreducible
# blocks), blocks the whole-group label.
REASON_SUBPROBLEM_FAILED = "subproblem_failed"
REASON_SUBPROBLEMS_UNVOTED = "subproblems_unvoted"
# A recomposition that resolved COMPLETE (every sub-problem unanimously accepted)
# but whose union of accepted selections is EMPTY. Structurally unreachable via a
# normal panel (a non-NONE auto_accept selects >=1 edge; a unanimous-NONE routes
# to human_review, not auto_accept), so this is defense-in-depth: an empty
# accepted union is NOT a real whole-group label and must route to review rather
# than mint an empty label or be mis-attributed to the sliver gate.
REASON_EMPTY_RECOMPOSITION = "empty_recomposition"
# The batch dirs contributing a parent's consumed sub-problem verdicts resolve
# to DIFFERENT panel eras (e.g. a v3-era wave completed by post-bless v4
# re-votes): the union label would mix compositions under a single era tag, so
# the group is blocked rather than stamped.
REASON_SUBPROBLEM_ERA_MIXED = "subproblem_era_mixed"


@dataclass
class GroupExport:
    """Outcome of running the export gates on a single consensus group."""

    group_id: str
    source_batch: str
    exported: bool
    reason: str
    match_type: str = ""
    n_edges_raw: int = 0
    n_slivers_dropped: int = 0
    n_edges_final: int = 0
    mean_confidence: float = 0.0
    selected_edges: list[dict] = field(default_factory=list)
    human_group_id: str = ""
    # Historical compatibility flag. New plans never export panel empty sets;
    # write_exports also rejects a manually constructed empty outcome.
    is_empty_set: bool = False
    # True for a recomposed decomposed-group outcome (#367 Mode B): the group's
    # verdict is the union of per-sub-problem panel votes; an export is stamped
    # PANEL_DECOMPOSED_LABELER.
    from_decomposition: bool = False
    n_subproblems: int = 0
    n_subproblems_resolved: int = 0
    # True when the verdict is a QUORUM one (v5 rule; see _is_quorum_accept):
    # a direct accept with >=1 abstention among the panel, or a recomposition ANY
    # of whose consumed sub-verdicts was quorum-accepted (quorum taints the union).
    # write_exports mints the panel_quorum_* labeler variants for these so the
    # unanimous/quorum provenance distinction survives end-to-end.
    is_quorum: bool = False
    # Labeler era of the SOURCE BATCH ("v3"/"v4"/"v5", from batch_panel_era or
    # an explicit stamp_era), or "" when the batch matches no known composition.
    # write_exports stamps each era's own labeler tags and REFUSES to write an
    # exported group with era "" — an unknown composition never silently mints
    # the current era's provenance (declare one via plan_exports(stamp_era=...)
    # / CLI --stamp-era). Recomposed groups resolve era across every batch dir
    # contributing a consumed sub-problem verdict.
    panel_era: str = ""


@dataclass
class ExportReport:
    """Full plan of what would be / was written, plus tallies for reporting."""

    dataset: str
    n_total_groups: int
    n_auto_accept: int
    groups: list[GroupExport]
    # Historical compatibility field. New export plans never include panel
    # NONE candidates, so this is always zero.
    n_unanimous_none: int = 0
    # Decomposition (#367 Mode B): consensus rows consumed as sub-problem
    # verdicts (never exported individually), and parents with a roster.
    n_subproblem_rows: int = 0
    n_decomposed_parents: int = 0

    @property
    def exported(self) -> list[GroupExport]:
        return [g for g in self.groups if g.exported]

    @property
    def exported_empty(self) -> list[GroupExport]:
        """Exported reject-all (empty-set) groups."""
        return [g for g in self.groups if g.exported and g.is_empty_set]

    @property
    def exported_nonempty(self) -> list[GroupExport]:
        """Exported normal (non-empty accept) groups."""
        return [g for g in self.groups if g.exported and not g.is_empty_set]

    @property
    def skipped(self) -> list[GroupExport]:
        return [g for g in self.groups if not g.exported]

    def skipped_by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for g in self.skipped:
            out[g.reason] = out.get(g.reason, 0) + 1
        return out

    def total_slivers_dropped(self) -> int:
        return sum(g.n_slivers_dropped for g in self.groups)


def _parse_edge_set_pairs(raw) -> list[tuple[str, str]]:
    """Parse a consensus ``edge_set`` JSON string ``[[ref,tgt],...]`` to pairs."""
    if raw is None or isinstance(raw, float):
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [(str(a), str(b)) for a, b in data]


def _merge_consensus(batch_dirs: list[Path]) -> dict[str, tuple[Path, dict]]:
    """Merge each batch's ``consensus.csv``; later batch dirs supersede earlier.

    Returns ``{group_id: (batch_dir, consensus_row_dict)}``.
    """
    merged: dict[str, tuple[Path, dict]] = {}
    for bd in batch_dirs:
        cpath = Path(bd) / "consensus.csv"
        if not cpath.exists():
            raise FileNotFoundError(f"No consensus.csv in batch dir: {bd}")
        df = pd.read_csv(cpath, dtype={"group_id": str})
        for _, row in df.iterrows():
            merged[str(row["group_id"])] = (Path(bd), row.to_dict())
    return merged


def _load_batch_groups(batch_dir: Path) -> dict[str, dict]:
    """Load ``batch.json`` groups keyed by group_id (geometries + edges)."""
    batch_path = Path(batch_dir) / "batch.json"
    if not batch_path.exists():
        return {}
    try:
        batch = json.loads(batch_path.read_text())
    except (ValueError, OSError):
        return {}
    return {str(g.get("group_id")): g for g in batch.get("groups", [])}


#: The one experiment variant whose ballots are full-context and MAY mint panel
#: labels. See :func:`_is_ablation_variant`.
ENRICHED_VARIANT = "enriched"

#: Sentinel returned by :func:`_batch_experiment_variant` when a batch.json is
#: PRESENT but unreadable/unparseable (bad JSON, non-object payload, or a
#: non-object ``experiment`` block). It is non-empty and != ENRICHED_VARIANT, so
#: :func:`_is_ablation_variant` GATES it: a batch whose provenance we cannot
#: verify must never mint (fail closed). An ABSENT batch.json is different — it
#: reads as an ordinary (non-experimental) production batch and mints normally.
UNREADABLE_VARIANT = "<unreadable_batch_json>"


def _batch_experiment_variant(batch_dir: Path) -> str:
    """Return the ``experiment.variant`` stamped in a batch's ``batch.json``.

    The physical/coincidence ablation wave (``scripts/build_physical_stitch_wave.py``)
    stamps every batch with an ``experiment`` block whose ``variant`` is one of
    ``enriched`` / ``no_physical`` / ``no_coincidence`` / ``minimal``. Ordinary
    (non-experimental) production batches carry NO ``experiment`` block.

    Three cases, deliberately distinguished so the gate can fail CLOSED on
    corruption (an ablation batch with a valid ``consensus.csv`` but a corrupt
    ``batch.json`` must not slip through as mintable):

      * batch.json ABSENT, or present with no ``experiment`` block / no
        ``variant`` -> ``""`` (reads as production; mints).
      * batch.json PRESENT but unreadable/unparseable (bad JSON, non-object
        payload, or a non-object ``experiment``) -> :data:`UNREADABLE_VARIANT`
        (gated; never mints).
      * a stamped variant -> that stripped variant string.
    """
    batch_path = Path(batch_dir) / "batch.json"
    if not batch_path.exists():
        return ""
    try:
        batch = json.loads(batch_path.read_text())
    except (ValueError, OSError):
        return UNREADABLE_VARIANT
    if not isinstance(batch, dict):
        return UNREADABLE_VARIANT
    if "experiment" not in batch:
        return ""  # ordinary production batch
    experiment = batch["experiment"]
    if not isinstance(experiment, dict):
        return UNREADABLE_VARIANT  # present but corrupt experiment block -> gate
    variant = experiment.get("variant")
    return "" if variant is None else str(variant).strip()


def _is_ablation_variant(variant: str) -> bool:
    """True when ``variant`` must NOT mint a production label.

    Ablation-variant ballots (``no_physical`` / ``no_coincidence`` / ``minimal``,
    and any FUTURE variant name) are experiment data: stripping physical or
    coincidence context can make the panel confidently wrong (the v8
    ``1b90f03b/minimal`` misfire — a context-blinded auto_accept that every
    informed cell unanimously rejected in favor of the correct seeded option),
    so such ballots must NEVER mint. The :data:`UNREADABLE_VARIANT` sentinel (an
    unverifiable batch.json) is likewise gated — fail closed. Only the
    full-context ``enriched`` variant may mint. An empty variant (no experiment
    block / ordinary production batch) is NOT gated and mints normally; treating
    "anything present and != enriched" as ablation gates unknown future ablation
    names too.
    """
    return bool(variant) and variant != ENRICHED_VARIANT


def _group_sliver_pairs(group: dict) -> set[tuple[str, str]]:
    """Return the set of ``(ref_id, target_id)`` pairs the group classifies as slivers."""
    if not group:
        return set()
    annotated, _ = annotate_group_sliver_flags(group)
    return {(str(e["ref_id"]), str(e["target_id"])) for e in annotated if e.get("is_sliver")}


def _meta_from_group(grp: dict) -> dict | None:
    """Synthesize a ``metadata.yaml``-shaped dict from a ``batch.json`` group.

    Fallback for packs that carry ``batch.json`` (ids + classes) but no per-group
    ``metadata.yaml``, so the class-consistency gate and the human edge-overlap
    mapping can still run instead of silently degrading. Returns ``None`` when the
    group has no usable segment ids.
    """
    if not grp:
        return None
    ref_classes = grp.get("ref_classes") or {}
    tgt_classes = grp.get("target_classes") or {}
    ref_ids = grp.get("ref_ids") or sorted({str(e["ref_id"]) for e in grp.get("edges", [])})
    tgt_ids = grp.get("target_ids") or sorted({str(e["target_id"]) for e in grp.get("edges", [])})
    if not ref_ids and not tgt_ids:
        return None
    return {
        "match_type": grp.get("match_type", ""),
        "segments": {
            "reference": [
                {"label": f"R{i + 1}", "id": str(r), "class": ref_classes.get(str(r), "") or ""}
                for i, r in enumerate(ref_ids)
            ],
            "target": [
                {"label": f"T{i + 1}", "id": str(t), "class": tgt_classes.get(str(t), "") or ""}
                for i, t in enumerate(tgt_ids)
            ],
        },
    }


def _counts_show_abstention(row: dict) -> bool:
    """True when a row's vote counts prove >=1 abstention (``n_valid < n_votes``).

    Count evidence must always be able to DOWNGRADE a verdict's tier to quorum,
    even against a contradicting ``unanimous`` stamp (which
    ``derive_route_reason`` would otherwise return verbatim as an informative
    existing reason). Missing/unparseable
    counts are no evidence (False); logically impossible counts are handled
    conservatively (treated as abstention-present, the weaker quorum claim) by
    the shared :func:`panel_routing.counts_show_abstention`.
    """
    return counts_show_abstention(
        _int_or_none(row.get("n_valid")), _int_or_none(row.get("n_votes"))
    )


def _accept_below_quorum(row: dict) -> bool:
    """True when an ``auto_accept`` row's own counts contradict the quorum floor.

    ``compute_consensus`` only mints ``auto_accept`` at ``n_valid >= 3``; a row
    with routing ``auto_accept`` but ``n_valid`` present and < 3 is a hand-edit /
    corrupt / pre-quorum-rule artifact that must NOT mint accept ground truth.
    ``n_valid`` missing is no evidence — historical rows lacking the column keep
    exporting (the conservative "no evidence -> untouched" stance).
    """
    n_valid = _int_or_none(row.get("n_valid"))
    return n_valid is not None and n_valid < 3


def _is_quorum_accept(row: dict) -> bool:
    """True when an ``auto_accept`` consensus row is a QUORUM accept (v5 rule).

    A quorum accept is an all-valid agreement over >=1 abstention (e.g. 3-of-4
    with one abstain) — a weaker evidentiary claim than full unanimity, so it
    must mint the distinct ``panel_quorum_*`` labelers. Detection is
    deliberately CONSERVATIVE toward the quorum tag: the shared derivation
    (tier/reason stamp, or ``n_valid < n_votes`` on stamp-less rows) decides,
    and the raw count check ALSO runs independently so a stale ``unanimous``
    stamp with contradicting counts still downgrades — any abstention evidence
    on an accept row downgrades the claim to quorum. Mislabeling a unanimous
    accept as quorum is safe (a weaker claim); the reverse would launder an
    abstention into full-unanimity provenance. Pre-v5 auto_accept rows always
    have ``n_valid == n_votes`` -> False.
    """
    if str(row.get("routing")) != "auto_accept":
        return False
    return derive_route_reason(row) == REASON_QUORUM or _counts_show_abstention(row)


#: A batch dir carrying this marker file must never mint labels. Its verdicts
#: still feed the human review queue (panel_routing reads it directly), so a
#: calibration batch whose contested groups legitimately need human eyes is not
#: lost — only label EXPORT is blocked. Used by :func:`filter_exportable_batch_dirs`.
NO_EXPORT_MARKER = ".no-export"


def filter_exportable_batch_dirs(batch_dirs: list[Path]) -> list[Path]:
    """Drop batch dirs carrying a ``.no-export`` marker; keep order otherwise.

    A ``.no-export`` marker file means the batch is a calibration wave whose
    verdicts must not become durable labels — but whose contested groups still
    legitimately feed the human review queue (``panel_routing`` reads the batch
    dir directly and does NOT honor the marker). Only the export path skips these
    dirs; each skip is logged with the dir name for traceability.
    """
    kept: list[Path] = []
    for bd in batch_dirs:
        bd = Path(bd)
        if (bd / NO_EXPORT_MARKER).exists():
            logger.info(
                f"Skipping export for batch dir {bd.name}: {NO_EXPORT_MARKER} marker present"
            )
            continue
        kept.append(bd)
    return kept


def plan_exports(
    batch_dirs: list[Path],
    dataset: str,
    labels_dir: Path,
    max_edges: int = 20,
    max_assignment_components: int | None = None,
    soft_max_edges: int | None = None,
    backstop_max_edges: int | None = None,
    export_empty_set: bool = False,
    stamp_era: str | None = None,
) -> ExportReport:
    """Run the export gates over merged consensus and return a full plan.

    Pure w.r.t. the label store: reads human labels but writes nothing. Call
    :func:`write_exports` with the returned report to persist.

    ``stamp_era`` ("v3"/"v4"/"v5"/"v6"/"v7") fills non-invalid era-less batches only: each
    batch is first resolved via :func:`batch_panel_era`, and ``stamp_era``
    applies solely to batches whose composition resolves to NO era (unknown
    compositions, which :func:`write_exports` otherwise refuses). A batch that
    genuinely resolves to an era always keeps it — a mixed run of one
    era-less and one blessed-v4 batch with ``stamp_era="v3"`` stamps v3 only
    on the era-less one, never re-stamping the v4 batch. It should accompany
    ``--allow-nonstandard-panel``-style explicit provenance decisions only.
    ``None`` (default) leaves era-less batches unresolved. An invalid linked
    rubric profile (tampered pack, stale ballot links, unsafe/mismatched group
    roster) is never era-less: planning rejects it before an explicit stamp can
    override the failed integrity check.

    The size gate is *structural*, not a flat edge count: a group auto-exports
    when it is a single corridor-pair OR has few assignment-components within a
    soft edge budget (so a clean 30-edge single corridor is exportable, while a
    small two-highway tangle is blocked). A hard backstop ceiling blocks
    anything larger regardless — a defence against a structure-detection bug,
    not the primary gate. When a group lacks structure fields (an older
    batch.json predating the structure sidecar) the gate falls back to the flat
    ``max_edges`` cap on the selected edge set PLUS the hard backstop on the
    group's candidate count, so the backstop invariant (no over-backstop group
    ever auto-exports) holds on the legacy path too.

    ``export_empty_set`` is a retained compatibility argument and must remain
    False. Panel ``NONE`` is semantically overloaded and cannot become reject-all
    truth without human confirmation; passing True therefore fails closed.
    """
    if export_empty_set:
        raise ValueError(
            "panel NONE cannot be exported as reject-all ground truth; "
            "confirm the empty set in human review"
        )
    # Calibration batches (a ``.no-export`` marker) must not mint labels — drop
    # them up front so nothing downstream plans against them. panel_routing still
    # reads them, so their contested groups reach the human queue.
    batch_dirs = filter_exportable_batch_dirs(batch_dirs)
    if max_assignment_components is None:
        max_assignment_components = settings.stitch_export_max_assignment_components
    if soft_max_edges is None:
        soft_max_edges = settings.stitch_export_soft_max_edges
    if backstop_max_edges is None:
        backstop_max_edges = settings.stitch_export_backstop_max_edges
    if stamp_era is not None and stamp_era not in LABELERS_BY_ERA:
        raise ValueError(f"stamp_era must be one of {sorted(LABELERS_BY_ERA)}, got {stamp_era!r}")

    # Era stamping (and vote-provenance archival) keys batches by BASENAME, so
    # duplicate basenames would mis-attribute one dir's era/rows to another
    # (last-dir-wins). Refuse up front rather than plan on ambiguous identity.
    names = [Path(bd).name for bd in batch_dirs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"batch dirs have duplicate basenames {dupes}; era stamping and "
            "vote-provenance archival key on the basename and would collapse "
            "them — pass uniquely-named batch dirs."
        )
    invalid_rubric_batches = sorted(
        Path(bd).name
        for bd in batch_dirs
        if "<invalid>" in _batch_matching_rubric_versions(Path(bd))
    )
    if invalid_rubric_batches:
        raise ValueError(
            "invalid matching-rubric provenance in batches "
            f"{invalid_rubric_batches}; refusing to plan exports even with stamp_era"
        )
    merged = _merge_consensus(batch_dirs)

    # Load batch.json groups (geometries/edges) once per distinct batch dir,
    # preserving the caller's precedence order (later supersedes earlier).
    # batch.json is the ONLY source of geometries (sliver gate) and candidate
    # edges (overlap precedence); warn loudly if a batch dir lacks it, since
    # those gates then degrade rather than fail.
    batch_groups: dict[Path, dict[str, dict]] = {}
    for bd in dict.fromkeys(Path(b) for b in batch_dirs):
        if not (bd / "batch.json").exists():
            logger.warning(
                f"No batch.json in {bd}: sliver detection and edge-overlap "
                "precedence cannot run for its groups (gates degrade)."
            )
        batch_groups[bd] = _load_batch_groups(bd)

    # Ablation-variant gate: a batch stamped with a context-stripped experiment
    # variant (``no_physical`` / ``no_coincidence`` / ``minimal``, or any future
    # non-``enriched`` variant) is experiment data. Stripping physical/coincidence
    # context can make the panel confidently wrong (the v8 ``1b90f03b/minimal``
    # misfire), and the auto_accept flag is variant-blind, so such a batch must
    # NEVER mint a label — its auto_accept groups are skipped as
    # ``ablation_variant`` below. Enriched batches (``variant == "enriched"``) and
    # ordinary production batches (no experiment block) mint normally.
    variant_by_dir: dict[Path, str] = {bd: _batch_experiment_variant(bd) for bd in batch_groups}
    for bd, variant in variant_by_dir.items():
        if _is_ablation_variant(variant):
            logger.info(
                f"Batch {bd.name} is ablation variant {variant!r}: its groups are "
                "experiment data and will NOT mint panel labels (auto_accept skipped)."
            )

    # Decomposition rosters (#367 Mode B): a parent entry written by
    # ``stitch-batch --decompose`` carries the FULL sub-problem roster
    # (including size-gated oversized sub-problems that were never packed).
    # Later batch dirs supersede earlier ones per parent, mirroring
    # ``_merge_consensus`` precedence. Every roster sub-problem id maps to its
    # parent so its consensus rows are consumed by recomposition below and are
    # never exported individually (a sub-problem id is not a sidecar group).
    rosters: dict[str, tuple[Path, dict]] = {}
    for bd in batch_groups:
        for gid, grp in batch_groups[bd].items():
            if grp.get("decomposed_parent") and grp.get("subproblem_ids"):
                rosters[gid] = (bd, grp)
    sub_to_parent: dict[str, str] = {
        str(sid): parent_gid
        for parent_gid, (_bd, grp) in rosters.items()
        for sid in grp["subproblem_ids"]
    }
    # Defense in depth: ANY batch group carrying ``parent_group_id`` is a
    # sub-problem, even when a newer decomposition's roster no longer lists it
    # (e.g. the budget changed between waves). Such an orphaned sub row is
    # consumed (skipped) rather than falling through to the normal loop, where
    # its auto_accept could mint a label under a non-sidecar sub-problem id.
    for bd in batch_groups:
        for gid, grp in batch_groups[bd].items():
            if grp.get("parent_group_id"):
                sub_to_parent.setdefault(str(gid), str(grp["parent_group_id"]))

    # Human labels for precedence. Exclude our own previously-exported panel rows
    # (they are not human) so re-runs stay idempotent and accurately reported.
    store = StitchingLabelStore(dataset, labels_dir=labels_dir)
    human_df = store.load(dataset)
    if not human_df.empty and "labeler" in human_df.columns:
        # Exclude ALL panel labelers (v1, v2, ...) — none are human, so they must
        # not confer human precedence. Matching the prefix keeps re-runs idempotent
        # across a labeler-tag bump. astype("string") + na=False: a missing
        # labeler (hand-edited/legacy CSV, possibly an all-NaN float column) is
        # not a panel label — keep the row instead of raising.
        labeler = human_df["labeler"].astype("string")
        human_df = human_df[~labeler.str.startswith(PANEL_LABELER_PREFIX, na=False)]
    human_gids = set(human_df["group_id"].astype(str)) if not human_df.empty else set()

    # Metadata + candidate edges for auto-accept candidates (for the class gate
    # and the human edge-overlap mapping, reusing the eval module's approach).
    candidate_metas: dict[str, dict] = {}
    candidate_edges: dict[str, frozenset] = {}
    for gid, (bd, row) in merged.items():
        if gid in sub_to_parent:
            continue  # sub-problem rows are recomposition inputs, not candidates
        is_accept = str(row.get("routing")) == "auto_accept" and not _accept_below_quorum(row)
        if not is_accept:
            continue
        grp = batch_groups.get(bd, {}).get(gid)
        # Prefer per-group metadata.yaml; fall back to batch.json (ids + classes)
        # so the class gate and overlap mapping still run when it is absent.
        gdir = bd / gid
        if (gdir / "metadata.yaml").exists():
            candidate_metas[gid] = _load_group_metadata(gdir)
        else:
            meta = _meta_from_group(grp) if grp is not None else None
            if meta is not None:
                candidate_metas[gid] = meta
        if grp is not None:
            candidate_edges[gid] = frozenset(
                (str(e["ref_id"]), str(e["target_id"])) for e in grp.get("edges", [])
            )

    # Decomposed parents are recomposition candidates: include their metadata
    # (class gate) and candidate edges (human edge-overlap / set-membership
    # precedence) exactly like directly-voted candidate groups. Parents carry
    # no evidence pack, so batch.json is the metadata source.
    for parent_gid, (bd, grp) in sorted(rosters.items()):
        gdir = bd / parent_gid
        if (gdir / "metadata.yaml").exists():
            candidate_metas[parent_gid] = _load_group_metadata(gdir)
        else:
            meta = _meta_from_group(grp)
            if meta is not None:
                candidate_metas[parent_gid] = meta
        candidate_edges[parent_gid] = frozenset(
            (str(e["ref_id"]), str(e["target_id"])) for e in grp.get("edges", [])
        )

    overlap_map: dict[str, str] = {}
    if not human_df.empty:
        overlap_map = map_human_labels_to_groups(human_df, candidate_metas, candidate_edges)

    # SET-semantics human labels carry no edges, so the edge-overlap mapping above
    # cannot see them. A set label still means "a human reviewed this group", so
    # it MUST confer precedence: map set rows to candidate groups by MEMBERSHIP
    # overlap and merge the result (an edge-overlap match wins on a clash — it is
    # the stronger signal). Precedence is about the human having reviewed the
    # group, not about pair-level detail.
    if not human_df.empty and "label_semantics" in human_df.columns:
        set_rows = human_df[human_df.apply(_is_set_label, axis=1)]
        if not set_rows.empty:
            group_members = {
                gid: frozenset(r for r, _ in edges) | frozenset(t for _, t in edges)
                for gid, edges in candidate_edges.items()
            }
            # {human_gid: panel_gid} -> invert to {panel_gid: human_gid}, not
            # clobbering an existing (edge-overlap) mapping for that panel group.
            for hgid, pgid in _map_set_labels_to_groups(set_rows, group_members).items():
                overlap_map.setdefault(pgid, hgid)

    groups: list[GroupExport] = []
    n_auto = 0
    n_sub_rows = 0
    for gid, (bd, row) in sorted(merged.items()):
        # Sub-problem rows (#367 Mode B) are consumed by the recomposition path
        # below — a sub-problem id is not a sidecar group and must never mint a
        # label of its own (not even an empty-set one).
        if gid in sub_to_parent:
            n_sub_rows += 1
            continue
        # Gate (a): only auto_accept is an export candidate. Every NONE remains
        # in human review because it does not uniquely imply reject-all truth.
        # A sub-quorum auto_accept (n_valid present and < 3) is also skipped; the
        # read-time overlay in panel_routing surfaces it to the human queue.
        if str(row.get("routing")) == "auto_accept" and not _accept_below_quorum(row):
            n_auto += 1
            # Ablation-variant gate (batch-level): a context-stripped experiment
            # cell must never mint, even on auto_accept. Skip before any minting
            # gate so a context-blinded artifact can never reach production.
            if _is_ablation_variant(variant_by_dir.get(bd, "")):
                grp = batch_groups.get(bd, {}).get(gid, {})
                edges_pairs = _parse_edge_set_pairs(row.get("edge_set"))
                groups.append(
                    GroupExport(
                        group_id=gid,
                        source_batch=bd.name,
                        exported=False,
                        reason=REASON_ABLATION_VARIANT,
                        match_type=str(grp.get("match_type") or ""),
                        n_edges_raw=len(edges_pairs),
                        n_edges_final=len(edges_pairs),
                    )
                )
                continue
            ge = _gate_group(
                gid=gid,
                bd=bd,
                row=row,
                grp=batch_groups.get(bd, {}).get(gid, {}),
                meta=candidate_metas.get(gid),
                human_gids=human_gids,
                overlap_map=overlap_map,
                max_edges=max_edges,
                max_assignment_components=max_assignment_components,
                soft_max_edges=soft_max_edges,
                backstop_max_edges=backstop_max_edges,
            )
            ge.is_quorum = _is_quorum_accept(row)
            groups.append(ge)

    # Per-dir labeler era ("v3"/"v4"/"v5"; "" for unattributable). stamp_era is a
    # FILL-IN for dirs that resolve to no era — a genuinely-resolved batch
    # always keeps its own era (resolution first), so an operator passing
    # --stamp-era v3 for one era-less batch can never re-stamp a blessed-v4
    # batch in the same run. Resolved once; used for both the direct groups
    # (by source batch) and the recomposition era-mix check below.
    era_by_dir: dict[Path, str] = {
        bd: (batch_panel_era(bd) or stamp_era or "") for bd in batch_groups
    }

    # Recomposition (#367 Mode B): one outcome per decomposed parent, from its
    # sub-problem verdicts. Conservative all-or-nothing rule — see
    # :func:`_gate_recomposed_group`. When an OLD wave also voted the parent
    # directly, its (size-gated) outcome above coexists in the report; a later
    # recomposed export for the same group_id supersedes it at write time
    # (``write_exports`` upserts by group_id in report order).
    for parent_gid, (bd, grp) in sorted(rosters.items()):
        roster_ids = [str(s) for s in grp["subproblem_ids"]]
        verdicts: dict[str, tuple[str, list[tuple[str, str]]]] = {}
        sub_confs: list[float] = []
        for sid in roster_ids:
            if sid not in merged:
                continue
            srow = merged[sid][1]
            # A sub-quorum auto_accept sub-verdict (n_valid present and < 3) is
            # not a valid accept — demote it to human_review so recomposition
            # treats it as a failed sub (the union can never launder a sub-quorum
            # accept into a whole-group label).
            sub_routing = str(srow.get("routing", ""))
            if sub_routing == "auto_accept" and _accept_below_quorum(srow):
                sub_routing = "human_review"
            verdicts[sid] = (sub_routing, _parse_edge_set_pairs(srow.get("edge_set")))
            if sub_routing == "auto_accept":
                with suppress(ValueError, TypeError):
                    sub_confs.append(float(srow.get("mean_confidence") or 0.0))
        rec = recompose_subproblem_verdicts(parent_gid, roster_ids, verdicts)
        # Weakest-link confidence: the minimum accepted sub-panel mean, so a
        # reviewer sees the least-certain sub-decision.
        min_conf = min(sub_confs) if sub_confs else 0.0
        # Weakest-link provenance (v5 quorum rule): if ANY consumed sub-problem
        # was quorum-accepted (an abstention among its panel), the whole
        # recomposed label is conservatively a QUORUM label — one abstention
        # anywhere in the union must not be laundered into full-unanimity
        # provenance (panel_quorum_decomposed_* vs panel_unanimous_decomposed_*).
        sub_quorum = any(_is_quorum_accept(merged[sid][1]) for sid in roster_ids if sid in merged)

        # Era of the recomposed label = the era of the batch dirs whose
        # consensus rows the recomposition CONSUMED (merged precedence), not
        # the roster dir's: a v3-era wave whose failed sub-problems were
        # re-voted post-bless in a v4 batch dir would otherwise stamp a
        # mixed-composition union under a single era. On a mismatch the group
        # is blocked (subproblem_era_mixed) — never stamped.
        sub_dirs = {merged[sid][0] for sid in roster_ids if sid in merged}
        # Ablation-variant gate (defense in depth on the recomposition path): if
        # the roster dir OR any dir contributing a consumed sub-verdict is a
        # context-stripped experiment cell, the union must not mint either.
        if any(_is_ablation_variant(variant_by_dir.get(d, "")) for d in ({bd} | sub_dirs)):
            groups.append(
                GroupExport(
                    group_id=parent_gid,
                    source_batch=bd.name,
                    exported=False,
                    reason=REASON_ABLATION_VARIANT,
                    match_type=str(grp.get("match_type") or ""),
                    n_edges_raw=len(rec.union_edges),
                    mean_confidence=min_conf,
                    from_decomposition=True,
                    n_subproblems=rec.n_subproblems,
                    n_subproblems_resolved=rec.n_resolved,
                    is_quorum=sub_quorum,
                )
            )
            continue
        sub_eras = {era_by_dir.get(d, "") for d in (sub_dirs or {bd})}
        if len(sub_eras) > 1:
            groups.append(
                GroupExport(
                    group_id=parent_gid,
                    source_batch=bd.name,
                    exported=False,
                    reason=REASON_SUBPROBLEM_ERA_MIXED,
                    match_type=str(grp.get("match_type") or ""),
                    n_edges_raw=len(rec.union_edges),
                    mean_confidence=min_conf,
                    from_decomposition=True,
                    n_subproblems=rec.n_subproblems,
                    n_subproblems_resolved=rec.n_resolved,
                    is_quorum=sub_quorum,
                )
            )
            continue
        ge = _gate_recomposed_group(
            gid=parent_gid,
            bd=bd,
            grp=grp,
            rec=rec,
            meta=candidate_metas.get(parent_gid),
            human_gids=human_gids,
            overlap_map=overlap_map,
            mean_confidence=min_conf,
        )
        ge.panel_era = next(iter(sub_eras))
        ge.is_quorum = sub_quorum
        groups.append(ge)

    # Stamp every DIRECT outcome with its source batch's era so write_exports
    # can tag each batch with its own labeler generation. Recomposed groups
    # were already stamped above from their CONTRIBUTING dirs (which may differ
    # from the roster dir this loop would key on) — leave them untouched.
    era_by_name = {bd.name: era for bd, era in era_by_dir.items()}
    for g in groups:
        if not g.from_decomposition:
            g.panel_era = era_by_name.get(g.source_batch, "")

    return ExportReport(
        dataset=dataset,
        n_total_groups=len(merged),
        n_auto_accept=n_auto,
        groups=groups,
        n_unanimous_none=0,
        n_subproblem_rows=n_sub_rows,
        n_decomposed_parents=len(rosters),
    )


def _gate_group(
    gid: str,
    bd: Path,
    row: dict,
    grp: dict,
    meta: dict | None,
    human_gids: set[str],
    overlap_map: dict[str, str],
    max_edges: int,
    max_assignment_components: int,
    soft_max_edges: int,
    backstop_max_edges: int,
) -> GroupExport:
    """Apply gates (b)-(e) to one auto-accept group and return its outcome.

    The first failing gate decides the group. Gate (a) (auto_accept) is applied
    by the caller before this is invoked.
    """
    match_type = str(grp.get("match_type") or (meta.get("match_type") if meta else "") or "")
    edges_pairs = _parse_edge_set_pairs(row.get("edge_set"))
    n_raw = len(edges_pairs)
    try:
        mean_conf = float(row.get("mean_confidence") or 0.0)
    except (ValueError, TypeError):
        mean_conf = 0.0

    def _mk(reason: str, **kw) -> GroupExport:
        return GroupExport(
            group_id=gid,
            source_batch=bd.name,
            exported=(reason == REASON_EXPORTED),
            reason=reason,
            match_type=match_type,
            n_edges_raw=n_raw,
            mean_confidence=mean_conf,
            **kw,
        )

    # A valid accepted option always has at least one edge. An auto_accept row
    # with an empty/malformed edge set is not reject-all truth and must return to
    # review rather than fall through as an empty panel accept.
    if not edges_pairs:
        return _mk(REASON_EMPTY_SELECTION, n_edges_final=0)

    # Gate (b): size gate. Prefer the structural gate when the group carries
    # structure fields (single corridor / few assignment-components within a
    # soft budget, under a hard backstop). Fall back to the flat edge cap on the
    # selected edge set for older batch.json packs without structure fields —
    # ALSO enforcing the hard backstop on the group's CANDIDATE count there.
    # Without it, a legacy over-backstop group
    # with a small selected set (e.g. calib0709's 0cbcf706: 45 candidates, 18
    # selected) slipped past the flat cap and minted a label, resurrecting the
    # size-routing void the consensus/read-time size gates close.
    n_group_edges = grp.get("n_edges")
    n_corridors = grp.get("n_corridors")
    n_assign = grp.get("n_assignment_components")
    if n_group_edges is None:
        n_group_edges = len(grp.get("edges", [])) or n_raw
    if n_corridors is not None and n_assign is not None:
        simple = group_is_structurally_simple(
            int(n_corridors),
            int(n_assign),
            int(n_group_edges),
            max_assignment_components,
            soft_max_edges,
            backstop_max_edges,
        )
        if not simple:
            reason = (
                REASON_OVER_MAX
                if int(n_group_edges) > backstop_max_edges
                else REASON_STRUCTURAL_TANGLE
            )
            return _mk(reason, n_edges_final=n_raw)
    elif n_raw > max_edges or int(n_group_edges) > backstop_max_edges:
        return _mk(REASON_OVER_MAX, n_edges_final=n_raw)

    # Gate (c): class-consistency (any cross-mode pair: pedestrian/vehicular/bike).
    if meta is not None:
        ref_c, tgt_c = _segment_class_maps(meta)
        edge_classes = _edge_classes_for(frozenset(edges_pairs), ref_c, tgt_c)
        if has_cross_mode_edge(edge_classes):
            return _mk(REASON_CLASS_MISMATCH, n_edges_final=n_raw)

    # Gate (d): exactness-preserving sliver gate.  A SLIVER tag is evidence,
    # not semantic truth; silently deleting a voted edge would mutate the exact
    # option the panel selected.  Hold the whole group for human review instead.
    sliver_pairs = _group_sliver_pairs(grp)
    if sliver_pairs.intersection(edges_pairs):
        return _mk(REASON_CONTAINS_SLIVER, n_edges_final=n_raw)

    # Gate (e): human precedence (exact group_id or edge-overlap).
    if gid in human_gids or gid in overlap_map:
        return _mk(
            REASON_HUMAN_PRECEDENCE,
            n_edges_final=n_raw,
            human_group_id=(gid if gid in human_gids else overlap_map[gid]),
        )

    selected = [{"ref_id": r, "target_id": t} for r, t in edges_pairs]
    return _mk(
        REASON_EXPORTED,
        n_edges_final=n_raw,
        selected_edges=selected,
    )


def _gate_recomposed_group(
    gid: str,
    bd: Path,
    grp: dict,
    rec: Recomposition,
    meta: dict | None,
    human_gids: set[str],
    overlap_map: dict[str, str],
    mean_confidence: float,
) -> GroupExport:
    """Gate a decomposed group's recomposed verdict into a whole-group export.

    Conservative all-or-nothing rule (#367 Mode B): a label is minted ONLY when
    every sub-problem in the roster resolved as a unanimous panel accept — any
    failed sub-problem (``subproblem_failed``) or unvoted/size-gated one
    (``subproblems_unvoted``) blocks the group, and the failing sub-problems
    stay routed to human review. Mixed human+panel recomposition (a human
    sub-verdict completing a partially-panel-resolved group) is deferred.

    The remaining gates mirror :func:`_gate_group` on the UNION selection, with
    one deliberate exception — there is NO size gate: each sub-decision was
    within the panel envelope by construction, and the union spans the whole
    over-backstop parent, which is exactly what decomposition exists to label.
    Class-consistency, the exactness-preserving sliver gate, and human precedence apply to
    the union / parent group unchanged.
    """
    match_type = str(grp.get("match_type") or (meta.get("match_type") if meta else "") or "")
    n_raw = len(rec.union_edges)

    def _mk(reason: str, **kw) -> GroupExport:
        return GroupExport(
            group_id=gid,
            source_batch=bd.name,
            exported=(reason == REASON_EXPORTED),
            reason=reason,
            match_type=match_type,
            n_edges_raw=n_raw,
            mean_confidence=mean_confidence,
            from_decomposition=True,
            n_subproblems=rec.n_subproblems,
            n_subproblems_resolved=rec.n_resolved,
            **kw,
        )

    if rec.status == STATUS_FAILED:
        return _mk(REASON_SUBPROBLEM_FAILED, n_edges_final=n_raw)
    if rec.status == STATUS_UNVOTED:
        return _mk(REASON_SUBPROBLEMS_UNVOTED, n_edges_final=n_raw)

    # Empty-union guard: a COMPLETE recomposition whose accepted selections union
    # to nothing is not a real whole-group label. Refuse it explicitly rather
    # than letting an empty union fall through as a successful exact selection.
    if not rec.union_edges:
        return _mk(REASON_EMPTY_RECOMPOSITION, n_edges_final=0)

    # Class-consistency gate on the union (same rule as _gate_group).
    if meta is not None:
        ref_c, tgt_c = _segment_class_maps(meta)
        edge_classes = _edge_classes_for(frozenset(rec.union_edges), ref_c, tgt_c)
        if has_cross_mode_edge(edge_classes):
            return _mk(REASON_CLASS_MISMATCH, n_edges_final=n_raw)

    # Exactness-preserving sliver gate on the union.  Never mutate a recomposed
    # exact selection by silently deleting one of its accepted edges.
    sliver_pairs = _group_sliver_pairs(grp)
    if sliver_pairs.intersection(rec.union_edges):
        return _mk(REASON_CONTAINS_SLIVER, n_edges_final=n_raw)

    # Human precedence (exact group_id or edge-overlap): never overwrite a human.
    if gid in human_gids or gid in overlap_map:
        return _mk(
            REASON_HUMAN_PRECEDENCE,
            n_edges_final=n_raw,
            human_group_id=(gid if gid in human_gids else overlap_map[gid]),
        )

    selected = [{"ref_id": r, "target_id": t} for r, t in rec.union_edges]
    return _mk(
        REASON_EXPORTED,
        n_edges_final=n_raw,
        selected_edges=selected,
    )


@dataclass(frozen=True)
class EraLabelers:
    """One era's labeler tags for :func:`write_exports`.

    The ``*_quorum`` variants exist only from v5 on (the quorum consensus rule
    shipped with the v5 bless; a 3-voter era cannot produce a quorum accept —
    all-valid agreement among 3 voters IS full unanimity). ``None`` makes
    :func:`write_exports` REFUSE a quorum-flagged group in that era: a quorum
    verdict attributed to a pre-quorum era is a provenance anomaly (hand-edited
    rows / tampered votes.csv), never something to blur into a unanimous tag.
    """

    accept: str
    decomposed: str
    accept_quorum: str | None = None
    decomposed_quorum: str | None = None


#: Era -> labeler tags for write_exports. v3/v4-era batches keep minting their
#: own era's tags on (re-)export; v5-v7 batches use their generation-specific
#: tags (with quorum variants for quorum verdicts). There is deliberately NO fallback
#: entry: an era-less group ("" — unknown composition, or no readable
#: votes.csv) makes write_exports refuse rather than silently mint the current
#: era's provenance.
LABELERS_BY_ERA: dict[str, EraLabelers] = {
    "v3": EraLabelers(PANEL_LABELER_V3, PANEL_DECOMPOSED_LABELER_V3),
    "v4": EraLabelers(PANEL_LABELER_V4, PANEL_DECOMPOSED_LABELER_V4),
    "v5": EraLabelers(
        PANEL_LABELER,
        PANEL_DECOMPOSED_LABELER,
        accept_quorum=PANEL_QUORUM_LABELER,
        decomposed_quorum=PANEL_QUORUM_DECOMPOSED_LABELER,
    ),
    "v6": EraLabelers(
        PANEL_LABELER_V6,
        PANEL_DECOMPOSED_LABELER_V6,
        accept_quorum=PANEL_QUORUM_LABELER_V6,
        decomposed_quorum=PANEL_QUORUM_DECOMPOSED_LABELER_V6,
    ),
    "v7": EraLabelers(
        PANEL_LABELER_V7,
        PANEL_DECOMPOSED_LABELER_V7,
        accept_quorum=PANEL_QUORUM_LABELER_V7,
        decomposed_quorum=PANEL_QUORUM_DECOMPOSED_LABELER_V7,
    ),
}


def write_exports(
    report: ExportReport,
    dataset: str,
    labels_dir: Path,
) -> int:
    """Persist the report's exported groups as ``panel_*`` stitching labels.

    Accept groups are stamped ``panel_unanimous_v5`` with their chosen edge set
    (``panel_quorum_v5`` for quorum accepts — a 4/4 and a 3-of-4 verdict must
    stay distinguishable); panel reject-all (empty-set) groups are refused and
    must be confirmed through human review; recomposed decomposed-group verdicts
    (#367 Mode B) are stamped ``panel_unanimous_decomposed_v5`` with the union of
    their sub-problem selections — or ``panel_quorum_decomposed_v5`` when ANY
    consumed sub-verdict was a quorum accept. Groups whose source batch is an
    older-era panel (``panel_era`` "v3"/"v4", see :func:`batch_panel_era`) are
    stamped with that era's variants instead — re-exporting committed history
    never rewrites its provenance to the current era — and a group with NO
    resolvable era raises ``ValueError``
    (declare one explicitly via ``plan_exports(stamp_era=...)`` / CLI
    ``--stamp-era``) instead of silently minting current-era provenance. A
    QUORUM-flagged group in an era without quorum labelers (pre-v5) also raises:
    that combination is a provenance anomaly, never blurred into a unanimous tag.

    The anomaly check is ATOMIC over the run: labelers are resolved in a
    pre-pass, so a raise writes NOTHING — which also means a single anomalous
    row (e.g. an impossible-counts row that
    :func:`panel_routing.counts_show_abstention` conservatively downgrades to a
    quorum claim inside a pre-quorum-era batch) blocks the WHOLE export run.
    That is by design — the same raise-on-provenance-anomaly stance as the
    quorum-in-pre-quorum-era refusal (#405) — and the operational escape hatch
    is a ``.no-export`` marker on the offending batch dir, which excludes it
    from export while its groups keep feeding the human review queue.
    Upserts by ``group_id`` (the store replaces an
    existing row for the same group_id), so this is idempotent. The source batch
    name is recorded in the ``session_id`` field for provenance.
    Returns the number of rows written.
    """
    empty_groups = sorted(g.group_id for g in report.exported if not g.selected_edges)
    if empty_groups:
        raise ValueError(
            f"empty panel selections {empty_groups} cannot be exported as durable truth; "
            "confirm reject-all outcomes in human review and investigate empty accepts"
        )
    era_less = sorted(
        {g.source_batch for g in report.exported if g.panel_era not in LABELERS_BY_ERA}
    )
    if era_less:
        raise ValueError(
            f"batches {era_less} match no known panel era (blessed or historical) — "
            f"refusing to stamp panel_* labelers on unknown provenance. Declare the "
            f"era explicitly with plan_exports(stamp_era=...) / CLI --stamp-era "
            f"{{{', '.join(sorted(LABELERS_BY_ERA))}}}."
        )
    # Resolve every row's labeler in a PRE-PASS, before any write. A quorum
    # verdict attributed to a pre-quorum (no-quorum-labelers) era is a provenance
    # anomaly that must raise — but with NOTHING written, not mid-loop after
    # earlier rows already hit disk (that partial write left the store in an
    # inconsistent, half-exported state). Resolution is pure, so any anomaly is
    # caught here before the store is touched.
    resolved: list[tuple[GroupExport, str]] = []
    for g in report.exported:
        tags = LABELERS_BY_ERA[g.panel_era]
        if g.from_decomposition:
            labeler = tags.decomposed_quorum if g.is_quorum else tags.decomposed
        else:
            labeler = tags.accept_quorum if g.is_quorum else tags.accept
        if labeler is None:
            raise ValueError(
                f"group {g.group_id} (batch {g.source_batch}) is a QUORUM verdict "
                f"but its era {g.panel_era!r} predates the quorum rule — a "
                f"{g.panel_era}-era panel cannot have produced an accept with an "
                f"abstention. Refusing to blur quorum/unanimous provenance; "
                f"investigate the batch's consensus.csv."
            )
        resolved.append((g, labeler))

    store = StitchingLabelStore(dataset, labels_dir=labels_dir)
    written = 0
    for g, labeler in resolved:
        ref_ids = {e["ref_id"] for e in g.selected_edges}
        tgt_ids = {e["target_id"] for e in g.selected_edges}
        store.add(
            group_id=g.group_id,
            selected_edges=g.selected_edges,
            match_type=g.match_type,
            num_refs=len(ref_ids),
            num_targets=len(tgt_ids),
            labeler=labeler,
            session_id=g.source_batch,
        )
        written += 1
    return written


def write_vote_provenance(
    batch_dirs: list[Path],
    dataset: str,
    votes_dir: Path = Path("labels/votes"),
    *,
    require_evidence: bool = False,
    allow_stale_policy: bool = False,
) -> tuple[int, int]:
    """Snapshot raw panel ballots + consensus into a git-tracked location.

    The panel writes ``votes.csv`` (every raw ballot) and ``consensus.csv`` per
    batch, but those live under the batch dir in the git-ignored ``data/`` tree,
    so the audit trail behind every exported label is never committed. This
    copies them into ``labels/votes/dataset=<dataset>/`` — which *is* tracked —
    tagging each row with a ``source_batch`` column for cross-batch traceability.
    It also archives one compact ``evidence.csv`` row per voted group containing
    the exact option menu/displayed edge universe and its pack hashes.

    **Accumulates** like the label store: the existing archived files are read
    back and merged with the current batches, so exporting batches across
    *separate* invocations (``stitch-export -b b1`` then later ``-b b2``) never
    drops earlier ballots — matching ``write_exports``, which only upserts the
    labels for the groups in each run. A re-archived batch REPLACES all of its
    previous rows wholesale (keyed by ``source_batch``, only when the batch
    contributes a readable file in this call): per-row upsert would leave stale
    ballots from a different panel composition lingering under the same batch
    name (e.g. archived agy rows surviving a v4-panel re-run — a 4-voter
    chimera batch). Rows are still deduped within a run as defense in depth.

    ``source_batch`` is the batch dir *basename*, so duplicate basenames in one
    call would collapse distinct ballots — we refuse that rather than lose data.
    Field-level best-effort: an empty or malformed batch CSV is skipped, not
    fatal. When ``require_evidence`` is true (the label-export CLI uses it),
    every consensus group must also have a verifiable evidence pack; otherwise
    the call raises before a panel label can be minted without its menu.

    ``allow_stale_policy`` is a narrow, logged operator escape: when set, a
    consensus row whose stored ``consensus_policy_sha256`` no longer equals the
    signature recomputed from current code is downgraded from a hard refusal to
    a warning (naming the group and the stored-vs-expected signatures), and the
    mint proceeds. It relaxes ONLY that signature-equality check — every other
    strict gate (routing/auto_accept, size, class-consistency, sliver,
    human-precedence, panel composition) stays in force — and the archived
    consensus row keeps its OWN stored policy sha and real era stamp; nothing is
    re-stamped to the current signature. Use it only to mint a genuine,
    rubric-stable historical auto-accept under its recorded policy.
    Returns ``(n_vote_rows, n_consensus_rows)``.
    """
    out_dir = Path(votes_dir) / f"dataset={dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [Path(bd).name for bd in batch_dirs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"batch dirs have duplicate basenames {dupes}; the source_batch tag "
            "would collapse them and lose ballots — pass uniquely-named batch dirs."
        )

    def _read_csv(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, dtype={"group_id": str})
        except pd.errors.EmptyDataError:
            return None
        # A header-only (zero-row) file is as empty as a 0-byte one: it must
        # not mark a batch as "contributing" in _collect's wholesale
        # replacement, or it would delete the batch's archived ballots with no
        # replacement rows.
        return None if df.empty else df

    def _cell_text(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value).strip()

    def _strict_edge_set(value, *, where: str) -> frozenset[tuple[str, str]]:
        text = _cell_text(value)
        if not text:
            data = []
        else:
            try:
                data = json.loads(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{where} has malformed edge_set JSON") from exc
        if not isinstance(data, list):
            raise ValueError(f"{where} edge_set must be a JSON list")
        pairs: list[tuple[str, str]] = []
        for item in data:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"{where} edge_set contains a malformed edge")
            pairs.append((str(item[0]), str(item[1])))
        if len(pairs) != len(set(pairs)):
            raise ValueError(f"{where} edge_set contains duplicate edges")
        return frozenset(pairs)

    def _validate_choice_link(
        row: pd.Series,
        *,
        option_by_letter: dict[str, tuple[str, frozenset[tuple[str, str]]]],
        where: str,
        allow_abstain: bool,
    ) -> None:
        choice = _cell_text(row.get("choice"))
        if choice in option_by_letter:
            expected_option, expected_edges = option_by_letter[choice]
        elif choice == "NONE":
            expected_option, expected_edges = "NONE", frozenset()
        elif allow_abstain and choice == "ABSTAIN":
            expected_option, expected_edges = "ABSTAIN", frozenset()
        elif not choice and not allow_abstain:
            # ``compute_consensus`` writes an empty choice only when all voters
            # abstained; there is no selected option or edge in that outcome.
            expected_option, expected_edges = "", frozenset()
        else:
            raise ValueError(f"{where} references unknown choice {choice!r}")
        if _cell_text(row.get("chosen_option_id")) != expected_option:
            raise ValueError(f"{where} chosen_option_id does not match choice {choice!r}")
        if _strict_edge_set(row.get("edge_set"), where=where) != expected_edges:
            raise ValueError(f"{where} edge_set does not match choice {choice!r}")

    def _validate_strict_group(
        bd: Path,
        group_id: str,
        votes: pd.DataFrame | None,
        consensus_group: pd.DataFrame,
        manifest: dict,
    ) -> None:
        where = f"{bd.name}/{group_id}"
        evidence = manifest.get("evidence") or {}
        identities = {
            _cell_text(manifest.get("group_id")),
            _cell_text(evidence.get("group_id")),
        }
        if group_id in {"", ".", ".."} or Path(group_id).name != group_id:
            raise ValueError(f"unsafe evidence group_id for {where}")
        if identities != {group_id}:
            raise ValueError(f"evidence manifest group_id does not match {where}")
        if len(consensus_group) != 1:
            raise ValueError(f"strict provenance requires exactly one consensus row for {where}")
        if votes is None or "group_id" not in votes:
            raise ValueError(f"strict provenance requires ballots for {where}")
        vote_group = votes[votes["group_id"].astype(str) == group_id]
        if vote_group.empty:
            raise ValueError(f"strict provenance requires ballots for {where}")
        if "provider" not in vote_group or vote_group["provider"].astype(str).duplicated().any():
            raise ValueError(f"strict provenance requires one ballot per provider for {where}")
        gemini_ballots = vote_group[vote_group["provider"].astype(str) == "gemini"]
        if not gemini_ballots.empty:
            if "invocation_route" not in gemini_ballots:
                raise ValueError(f"Gemini ballot is missing invocation_route for {where}")
            actual_routes = set(gemini_ballots["invocation_route"].map(_cell_text))
            allowed_routes = {GEMINI_ROUTE_AGY, GEMINI_ROUTE_OPENROUTER_FLEX}
            if not actual_routes or "" in actual_routes or not actual_routes <= allowed_routes:
                raise ValueError(
                    f"Gemini ballot has invalid invocation_route for {where}: "
                    f"{sorted(actual_routes)}"
                )

        vote_fields = {
            "evidence_id",
            "evidence_pack_sha256",
            "displayed_candidate_universe_sha256",
            "option_menu_sha256",
            "chosen_option_id",
            "panel_invocation_sha256",
        }
        consensus_fields = vote_fields | {"consensus_policy_sha256"}
        missing_votes = sorted(vote_fields - set(vote_group.columns))
        missing_consensus = sorted(consensus_fields - set(consensus_group.columns))
        if missing_votes or missing_consensus:
            raise ValueError(
                f"strict provenance linkage fields missing for {where}: "
                f"votes={missing_votes}, consensus={missing_consensus}"
            )

        expected = {
            "evidence_id": _cell_text(evidence.get("evidence_id")),
            "evidence_pack_sha256": _cell_text(manifest.get("evidence_pack_sha256")),
            "displayed_candidate_universe_sha256": _cell_text(
                evidence.get("displayed_candidate_universe_sha256")
            ),
            "option_menu_sha256": _cell_text(evidence.get("option_menu_sha256")),
        }
        consensus_row = consensus_group.iloc[0]
        for field_name, expected_value in expected.items():
            if not expected_value:
                raise ValueError(f"verified evidence is missing {field_name} for {where}")
            if set(vote_group[field_name].map(_cell_text)) != {expected_value}:
                raise ValueError(f"ballot {field_name} does not match evidence for {where}")
            if _cell_text(consensus_row.get(field_name)) != expected_value:
                raise ValueError(f"consensus {field_name} does not match evidence for {where}")

        invocation_values = set(vote_group["panel_invocation_sha256"].map(_cell_text))
        consensus_invocation = _cell_text(consensus_row.get("panel_invocation_sha256"))
        if "" in invocation_values or invocation_values != {consensus_invocation}:
            raise ValueError(f"panel invocation linkage does not match for {where}")
        expected_policy = consensus_policy_signature(
            max_edges=settings.stitch_export_backstop_max_edges,
            min_voter_confidence=settings.stitch_min_voter_confidence,
            runtime_contract_sha256=sha256_file(Path(__file__).with_name("stitch_runner.py")),
        )
        stored_policy = _cell_text(consensus_row.get("consensus_policy_sha256"))
        if stored_policy != expected_policy:
            if not allow_stale_policy:
                raise ValueError(f"consensus policy linkage is stale or missing for {where}")
            # Operator override: mint under the recorded historical policy,
            # explicitly acknowledged. The archived consensus row keeps its OWN
            # stored consensus_policy_sha256 and its real era stamp — this only
            # relaxes the signature-equality check, never re-stamps to current.
            logger.warning(
                "consensus policy linkage is stale for {where}; --allow-stale-policy "
                "is set, so minting under the recorded historical policy "
                "(stored={stored}, expected={expected}). Verify this is a "
                "rubric-stable historical auto-accept before trusting the label.",
                where=where,
                stored=stored_policy or "<missing>",
                expected=expected_policy,
            )

        try:
            recorded_n_votes = int(consensus_row.get("n_votes"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"consensus n_votes is invalid for {where}") from exc
        if recorded_n_votes != len(vote_group):
            raise ValueError(f"consensus n_votes does not match ballots for {where}")

        option_by_letter: dict[str, tuple[str, frozenset[tuple[str, str]]]] = {}
        for option in evidence.get("option_menu", []):
            letter = _cell_text(option.get("letter"))
            option_id = _cell_text(option.get("option_id"))
            edges = frozenset(
                (str(edge["ref_id"]), str(edge["target_id"])) for edge in option.get("edges", [])
            )
            if not letter or not option_id or letter in option_by_letter:
                raise ValueError(f"verified evidence has a malformed option menu for {where}")
            option_by_letter[letter] = (option_id, edges)
        for index, vote in vote_group.iterrows():
            _validate_choice_link(
                vote,
                option_by_letter=option_by_letter,
                where=f"ballot {where} row {index}",
                allow_abstain=True,
            )
        _validate_choice_link(
            consensus_row,
            option_by_letter=option_by_letter,
            where=f"consensus {where}",
            allow_abstain=False,
        )

        # The consensus row is an executable claim about these ballots, not an
        # independent annotation. Replay the current, signature-bound policy so
        # edited routing/tier/confidence fields cannot mint a label even when
        # their menu hashes still look valid.
        replay_votes: list[Vote] = []
        for _, vote in vote_group.iterrows():
            confidence = float(vote.get("confidence"))
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"ballot confidence is invalid for {where}")
            replay_votes.append(
                Vote(
                    group_id=group_id,
                    provider=_cell_text(vote.get("provider")),
                    model=_cell_text(vote.get("model")),
                    choice=_cell_text(vote.get("choice")),
                    confidence=confidence,
                    reasoning=_cell_text(vote.get("reasoning")),
                    edge_set=_strict_edge_set(vote.get("edge_set"), where=where),
                )
            )
        metadata = _load_group_context(bd / group_id)[2]
        ref_class, target_class = _segment_class_maps(metadata)
        base = compute_consensus(replay_votes)
        replay = compute_consensus(
            replay_votes,
            edge_classes=_edge_classes_for(base.edge_set, ref_class, target_class),
            n_candidate_edges=candidate_edge_count(metadata),
            min_voter_confidence=settings.stitch_min_voter_confidence,
        )
        try:
            recorded_mean = float(consensus_row.get("mean_confidence"))
            recorded_n_valid = int(consensus_row.get("n_valid"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"consensus statistics are invalid for {where}") from exc
        if not math.isfinite(recorded_mean) or not 0.0 <= recorded_mean <= 1.0:
            raise ValueError(f"consensus mean confidence is invalid for {where}")
        if not all(
            (
                _cell_text(consensus_row.get("consensus")) == replay.consensus,
                _cell_text(consensus_row.get("choice")) == replay.choice,
                _strict_edge_set(consensus_row.get("edge_set"), where=where) == replay.edge_set,
                _cell_text(consensus_row.get("routing")) == replay.routing,
                recorded_n_votes == replay.n_votes,
                recorded_n_valid == replay.n_valid,
                _cell_text(consensus_row.get("minority")) == replay.minority,
                math.isclose(recorded_mean, replay.mean_confidence, abs_tol=1e-12),
                _cell_text(consensus_row.get("route_reason")) == replay.route_reason,
            )
        ):
            raise ValueError(f"consensus row is not derivable from ballots for {where}")

    verified_evidence: dict[tuple[Path, str], dict] = {}
    verified_votes: dict[Path, pd.DataFrame | None] = {}
    verified_consensus: dict[Path, pd.DataFrame] = {}
    if require_evidence:
        # Validate every referenced pack before rewriting any tracked archive.
        # This keeps strict export fail-closed as one preflight instead of
        # discovering a bad menu after votes.csv was already replaced.
        for bd in batch_dirs:
            bd = Path(bd)
            consensus = _read_csv(bd / "consensus.csv")
            if consensus is None:
                continue
            if "group_id" not in consensus:
                raise ValueError(f"consensus.csv has no group_id column in {bd.name}")
            batch_path = bd / "batch.json"
            batch_schema = 0
            batch_payload: dict | None = None
            if batch_path.exists():
                try:
                    batch_payload = json.loads(batch_path.read_text())
                    batch_schema = int(batch_payload.get("schema_version", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid batch.json provenance in {bd.name}: {exc}") from exc
                if batch_schema >= 2 and _cell_text(batch_payload.get("dataset_id")) != dataset:
                    raise ValueError(
                        f"schema-v2 batch dataset mismatch in {bd.name}: "
                        f"expected {dataset!r}, got {batch_payload.get('dataset_id')!r}"
                    )
            votes = _read_csv(bd / "votes.csv")
            for raw_group_id in sorted(set(consensus["group_id"].astype(str))):
                try:
                    group_id = safe_group_id(raw_group_id)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid evidence group_id in {bd.name}: {raw_group_id!r}"
                    ) from exc
                group_dir = bd / group_id
                if not group_dir.is_dir():
                    raise ValueError(f"missing evidence pack directory for {bd.name}/{group_id}")
                try:
                    manifest = load_evidence_manifest(
                        group_dir,
                        allow_legacy=batch_schema < 2,
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid evidence provenance for {bd.name}/{group_id}: {exc}"
                    ) from exc
                try:
                    validate_manifest_against_batch(manifest, batch_payload or {}, group_id)
                except ValueError as exc:
                    raise ValueError(
                        f"evidence pack does not match batch.json for {bd.name}/{group_id}: {exc}"
                    ) from exc
                file_names = {item["path"] for item in manifest.get("files", [])}
                required_files = {"metadata.yaml", "prompt.txt", "overview.png"}
                if not required_files <= file_names:
                    raise ValueError(
                        f"incomplete evidence pack for {bd.name}/{group_id}: "
                        f"missing {sorted(required_files - file_names)}"
                    )
                _validate_strict_group(
                    bd,
                    group_id,
                    votes,
                    consensus[consensus["group_id"].astype(str) == group_id],
                    manifest,
                )
                verified_evidence[(bd, group_id)] = manifest
            verified_votes[bd] = votes
            verified_consensus[bd] = consensus

    def _batch_csv(bd: Path, filename: str) -> pd.DataFrame | None:
        """Reuse strict-preflight snapshots so validated rows cannot be swapped."""
        bd = Path(bd)
        if require_evidence and bd in verified_consensus:
            if filename == "votes.csv":
                return verified_votes[bd]
            if filename == "consensus.csv":
                return verified_consensus[bd]
        return _read_csv(bd / filename)

    def _prepare_archive(filename: str, dedupe_on: list[str]) -> pd.DataFrame | None:
        # Current batches first: only a batch that contributes a READABLE file
        # in this call supersedes its archived rows — a listed batch whose CSV
        # is missing/empty keeps its archive untouched (never delete ballots
        # without a replacement).
        new_frames: list[pd.DataFrame] = []
        contributing: set[str] = set()
        for bd in batch_dirs:
            df = _batch_csv(Path(bd), filename)
            if df is None:
                continue
            if "source_batch" in df.columns:  # a re-archived tree; re-tag cleanly
                df = df.drop(columns="source_batch")
            df.insert(0, "source_batch", Path(bd).name)
            contributing.add(Path(bd).name)
            new_frames.append(df)
        frames = []
        existing = _read_csv(out_dir / filename)
        if existing is not None:
            # Wholesale replacement per re-archived batch: drop ALL previously
            # archived rows for the batches refreshed in this call, so stale
            # ballots from an earlier panel composition (e.g. agy rows under a
            # batch re-run with the v4 panel) can never linger beside the new
            # rows as a chimera composition.
            if "source_batch" in existing.columns and contributing:
                existing = existing[~existing["source_batch"].astype(str).isin(contributing)]
            if not existing.empty:
                frames.append(existing)
        frames.extend(new_frames)
        if not frames:
            return None
        merged = pd.concat(frames, ignore_index=True)
        keys = [c for c in dedupe_on if c in merged.columns]
        if keys:
            merged = merged.drop_duplicates(subset=keys, keep="last")
        return merged

    votes_archive = _prepare_archive("votes.csv", ["source_batch", "group_id", "provider"])
    consensus_archive = _prepare_archive("consensus.csv", ["source_batch", "group_id"])

    # Archive the menu itself, not just a hash pointing back into ignored data/.
    # Restrict to groups with consensus rows so unused packs do not masquerade as
    # observed panel evidence. Like the ballot archive, a refreshed batch
    # replaces its previous evidence rows wholesale.
    evidence_frames: list[pd.DataFrame] = []
    contributing_evidence: set[str] = set()
    for bd in batch_dirs:
        consensus = _batch_csv(Path(bd), "consensus.csv")
        if consensus is None:
            continue
        contributing_evidence.add(Path(bd).name)
        rows: list[dict] = []
        for group_id in sorted(set(consensus["group_id"].astype(str))):
            group_dir = Path(bd) / group_id
            if not group_dir.is_dir():
                if require_evidence:
                    raise ValueError(
                        f"missing evidence pack directory for {Path(bd).name}/{group_id}"
                    )
                continue
            try:
                manifest = verified_evidence.get((Path(bd), group_id)) or load_evidence_manifest(
                    group_dir
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                if require_evidence:
                    raise ValueError(
                        f"invalid evidence provenance for {Path(bd).name}/{group_id}: {exc}"
                    ) from exc
                logger.warning(
                    "Skipping evidence provenance for {}/{}: {}",
                    Path(bd).name,
                    group_id,
                    exc,
                )
                continue
            evidence = manifest["evidence"]
            rows.append(
                {
                    "source_batch": Path(bd).name,
                    "group_id": group_id,
                    "evidence_id": evidence["evidence_id"],
                    "evidence_pack_sha256": manifest["evidence_pack_sha256"],
                    "displayed_candidate_universe_sha256": evidence[
                        "displayed_candidate_universe_sha256"
                    ],
                    "option_menu_sha256": evidence["option_menu_sha256"],
                    "displayed_candidate_count": evidence["displayed_candidate_count"],
                    "evidence": canonical_json(evidence),
                }
            )
        if rows:
            evidence_frames.append(pd.DataFrame(rows))

    existing_evidence = _read_csv(out_dir / "evidence.csv")
    merged_evidence: list[pd.DataFrame] = []
    empty_evidence_archive: pd.DataFrame | None = None
    if existing_evidence is not None:
        if "source_batch" in existing_evidence and contributing_evidence:
            existing_evidence = existing_evidence[
                ~existing_evidence["source_batch"].astype(str).isin(contributing_evidence)
            ]
        if not existing_evidence.empty:
            merged_evidence.append(existing_evidence)
        elif contributing_evidence:
            # A refreshed non-strict batch with no readable pack must remove
            # its old menu row instead of pairing new ballots with stale evidence.
            empty_evidence_archive = existing_evidence
    merged_evidence.extend(evidence_frames)
    evidence_archive: pd.DataFrame | None = empty_evidence_archive
    if merged_evidence:
        evidence_df = pd.concat(merged_evidence, ignore_index=True)
        evidence_archive = evidence_df.drop_duplicates(
            subset=["source_batch", "group_id"], keep="last"
        )

    # Prepare and validate all three outputs before touching any tracked file.
    # Then stage every CSV beside its destination and replace as one rollback-
    # protected unit: a late evidence/schema or I/O failure cannot leave a new
    # votes.csv paired with an old consensus/evidence archive.
    archives = {
        out_dir / "votes.csv": votes_archive,
        out_dir / "consensus.csv": consensus_archive,
        out_dir / "evidence.csv": evidence_archive,
    }
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    originally_present = {path: path.exists() for path in archives}
    commit_complete = False
    rollback_complete = True
    try:
        for destination, frame in archives.items():
            if frame is None:
                continue
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=out_dir,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                staged[destination] = Path(tmp.name)
                frame.to_csv(tmp, index=False)
        for destination in staged:
            if destination.exists():
                with tempfile.NamedTemporaryFile(
                    dir=out_dir,
                    prefix=f".{destination.name}.",
                    suffix=".bak",
                    delete=False,
                ) as tmp:
                    backup = Path(tmp.name)
                backup.unlink()
                backups[destination] = backup
                destination.replace(backup)
        for destination, temporary in staged.items():
            temporary.replace(destination)
        commit_complete = True
    except Exception as commit_error:
        rollback_errors: list[str] = []
        for destination in reversed(list(staged)):
            backup = backups.get(destination)
            try:
                if backup is not None and backup.exists():
                    backup.replace(destination)
                elif not originally_present[destination] and destination.exists():
                    destination.unlink()
            except OSError as rollback_error:
                rollback_complete = False
                rollback_errors.append(f"{destination}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "vote-provenance archive commit failed and rollback was incomplete; "
                f"preserved backups: {rollback_errors}"
            ) from commit_error
        raise
    finally:
        for temporary in staged.values():
            with suppress(OSError):
                temporary.unlink()
        if commit_complete or rollback_complete:
            for backup in backups.values():
                with suppress(OSError):
                    backup.unlink()

    n_votes = 0 if votes_archive is None else len(votes_archive)
    n_consensus = 0 if consensus_archive is None else len(consensus_archive)
    return n_votes, n_consensus
