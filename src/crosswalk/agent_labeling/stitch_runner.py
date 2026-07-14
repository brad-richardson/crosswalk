"""Consensus-panel runner for agent stitching-group labeling.

Runs a heterogeneous 4-provider panel (claude + codex + kimi/Kimi K2.6 +
muse/Muse Spark 1.1 since the 2026-07-10 v5 bless; previously the 3-seat
claude + codex + kimi v4 panel, and claude + codex + agy before that) on each
group's evidence pack, in parallel. Each provider returns a JSON option pick;
votes are validated (choice must be a real option letter or NONE), retried once
on garbage, and recorded as audit data. A consensus rule routes each group: ALL
VALID (non-abstaining) votes agreeing with at least 3 valid votes auto-accepts
— full 4/4 unanimity and a 3-of-4 quorum accept (one abstention) stay
distinguishable end-to-end (see :func:`compute_consensus`).

Votes are audit data and are stored under the batch dir (``votes.csv``),
deliberately separate from ``labels/``. This module writes NOTHING into
``labels/stitching/`` — export policy is decided after the validation gate.

The opt-in ``v6-candidate`` is the lean three-seat Claude/Codex-Terra/Muse
panel used for the first breadth wave. ``v7-candidate`` keeps that lean roster,
returns Codex to gpt-5.6-sol, and records high effort for all three seats under
the canonical matching rubric. The route-aware Gemini 3.5 Flash voter remains
available through explicit agy and OpenRouter AI Studio flex calibration
panels, but is not a v6 or v7 production seat.
"""

from __future__ import annotations

import errno
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from ..config import settings
from .panel_monitor import wave_position_anchor_warnings
from .panel_routing import (
    REASON_ALL_ABSTAINED,
    REASON_CLASS_MISMATCH,
    REASON_LOW_CONFIDENCE,
    REASON_SIZE_GATED,
    candidate_edge_count,
    derive_route_reason,
)
from .stitch_provenance import (
    DELIVERY_MODE_NATIVE_ATTACHMENT,
    DELIVERY_MODE_PROMPT_PATH,
    EvidenceProvenanceError,
    batch_group_map,
    build_evidence_delivery_record,
    canonical_evidence_delivery_json,
    consensus_policy_signature,
    invocation_signature,
    load_evidence_manifest,
    managed_image_descriptors,
    sha256_file,
    validate_evidence_delivery_record,
    validate_manifest_against_batch,
)

# OSError errnos that are deterministic for a fixed command: retrying identically
# fails identically, so they hard-fail immediately rather than consuming the
# backoff budget. E2BIG = argument list too long (the large-prompt bug); ENOENT =
# missing binary; EACCES = permission denied; ENOEXEC/ENAMETOOLONG = malformed exec.
_FATAL_ERRNOS = frozenset(
    {errno.E2BIG, errno.ENOENT, errno.EACCES, errno.ENOEXEC, errno.ENAMETOOLONG}
)

# ---------------------------------------------------------------------------
# Provider panel configuration
# ---------------------------------------------------------------------------


#: Global default per-provider vote timeout (seconds). A :class:`ProviderSpec`
#: may carry its own ``timeout`` (e.g. Kimi's long thinking on large packs); an
#: EXPLICIT caller/CLI ``--timeout`` overrides both. See :func:`resolve_timeout`.
DEFAULT_VOTE_TIMEOUT_S = 240


@dataclass
class ProviderSpec:
    """A panel member: how to invoke a provider CLI on an evidence pack."""

    name: str  # short id used in votes.csv (e.g. "claude")
    model: str  # model string recorded in votes
    effort: str = ""  # reasoning/thinking effort where the CLI supports it
    # Per-spec vote timeout (seconds). None -> DEFAULT_VOTE_TIMEOUT_S. An
    # explicitly passed caller/CLI timeout beats this (resolve_timeout).
    timeout: int | None = None
    # opencode-only: name of an opencode agent to run under (``opencode run
    # --agent <x>``). None -> opencode's default ``build`` agent (the residual
    # v3-era Qwen seat, whose invocation stays byte-identical). BOTH the Kimi and
    # Muse voters set this to a tool-less ``vote`` agent (defined in the repo-root
    # ``opencode.json``, model-agnostic): under the default ``build`` agent the
    # model's tool loop (auto-rejected ls/cat/read calls) is the prime stall
    # suspect — Muse burned its turn on it, and Kimi timed out on 7/30 groups
    # under ``build`` vs 0/30 for Muse under ``vote`` on the same evidence packs.
    # ZERO tools forces a pure-text vote from the already-attached pack. Ignored
    # by every non-opencode invoker.
    opencode_agent: str | None = None
    # Ordered physical routes behind a logical voter. Empty for ordinary
    # single-transport voters. This is invocation provenance (hashed by
    # invocation_signature), not a second panel seat: a fallback must never
    # contribute an additional correlated ballot.
    routes: tuple[str, ...] = ()


def resolve_timeout(spec: ProviderSpec, timeout: int | None) -> int:
    """Resolve one provider's effective vote timeout (seconds).

    Precedence: an EXPLICITLY passed caller/CLI timeout (``--timeout``; not
    ``None``) wins over the spec's own ``timeout``, which wins over the global
    :data:`DEFAULT_VOTE_TIMEOUT_S`. Per-spec timeouts exist because voters
    differ structurally in latency — Kimi K2.6's thinking ran past the 240s
    default on 3/6 smoke votes (up to ~390s), which is a property of the model,
    not of a particular wave — while the explicit flag stays the operator's
    override for experiments.
    """
    if timeout is not None:
        return timeout
    if spec.timeout is not None:
        return spec.timeout
    return DEFAULT_VOTE_TIMEOUT_S


# Panel v3 (the FORMER production default: v2 was the same composition — the
# v2 -> v3 labeler bump tracked the #302 pack-input enrichment, not a voter
# change). Kept as named fallback panels ("v3"/"v2") so historical batches can
# be reproduced. Effort is CLI-specific: claude takes --effort; codex takes
# model_reasoning_effort; agy encodes it in the model name ("... (Medium)").
PANEL_V3 = [
    ProviderSpec(name="claude", model="claude-opus-4-8", effort="medium"),
    ProviderSpec(name="codex", model="gpt-5.5", effort="low"),
    ProviderSpec(name="agy", model="Gemini 3.5 Flash (Medium)"),
]

# A candidate FOURTH voter (default OFF): opencode driving an OpenRouter-hosted
# Qwen3-VL model. Deliberately a distinct model family from the three v3
# incumbents (Claude / GPT / Gemini) so its vote is decorrelated, adding real
# signal to the quorum rather than echoing an existing voice. opencode carries
# all knobs (reasoning etc.) in the model string, so ``effort`` is unused for
# it — like agy.
OPENCODE_QWEN = ProviderSpec(name="opencode", model="openrouter/qwen/qwen3-vl-235b-a22b-instruct")

# The v4 third voter: opencode driving OpenRouter-hosted Kimi K2.6 (Moonshot) —
# an open-weight native-multimodal flagship and a FOURTH model family (vs the
# Claude/GPT/Gemini incumbents), so its errors are decorrelated from the rest
# of the panel. Unlike agy (which must proactively read the pack images
# itself), the opencode invoker force-attaches every PNG, so this voter is
# guaranteed to see the full visual evidence. Kimi's thinking runs long on
# large packs (observed 11-386s/vote; 3/6 smoke votes exceeded the 240s
# default), so the spec carries its own 480s timeout — an explicit --timeout
# still overrides it (see resolve_timeout).
#
# Runs under the tool-less ``vote`` agent (``opencode_agent="vote"``, the same
# agent the Muse seat uses; defined in the repo-root ``opencode.json`` and
# model-agnostic — the model comes from ``-m``). A voter with the evidence-pack
# PNGs already force-attached needs no tools; under opencode's default ``build``
# agent, Kimi's tool loop is the prime stall suspect. In the 2026-07-10
# quad-candidate calibration wave Kimi timed out (480s) on 7/30 groups while its
# SUCCESSFUL votes had median latency 37s / max 172s — bimodal
# answer-fast-or-stall-forever, not slow thinking. Muse, on the SAME opencode
# transport with identical evidence packs but under this tool-less ``vote``
# agent, had 0/30 timeouts (median 19s), so we run Kimi tool-less too. The
# ``--agent`` threading keys on the RESOLVED INVOKER (``invoke_opencode``), not
# the name, so this is forwarded as ``--agent vote`` with no other change.
# Era-gate safe: the stitch-export gate keys voter identity on (provider, model)
# pairs only, so ``opencode_agent`` is invocation plumbing (like ``timeout``,
# #398 precedent) and the blessed v4 composition is unchanged.
#
# Provider NAME is ``"kimi"``, NOT the transport name ``"opencode"`` — a KEYING
# field (see the MUSE note below for the full list of provider-keyed sites). The
# ``quad-candidate`` panel seats BOTH this voter and Muse on the SAME opencode
# transport, so a transport-named ``"opencode"`` seat would be ambiguous when
# Muse shares the wave; the distinct ``"kimi"`` name keeps it individually
# addressable (provenance dedupe, monitor stats, ``--kimi-model``). Its invoker
# is resolved via ``_INVOKERS["kimi"] -> invoke_opencode`` (same transport). No
# committed votes reference the Kimi model — every on-disk ``provider="opencode"``
# row is the historical Gemini/Qwen transport-swap era — so the ``opencode`` ->
# ``kimi`` rename rewrites nothing on disk.
OPENCODE_KIMI = ProviderSpec(
    name="kimi", model="openrouter/moonshotai/kimi-k2.6", timeout=480, opencode_agent="vote"
)

# The v5 FOURTH voter (blessed 2026-07-10; formerly the meta-candidate /
# quad-candidate prototype seat): opencode driving Meta's "Muse Spark 1.1" via
# Meta's OpenAI-compatible developer API (api.meta.ai/v1), wired as an opencode
# CUSTOM provider — see the repo-root ``opencode.json`` (baseURL + apiKey via
# ``{env:META_API_KEY}``, never inlined). A FIFTH model family (Meta, vs the
# Anthropic/OpenAI/Google/Moonshot voices already in the quorum), so its errors
# are decorrelated; vision is confirmed working and the opencode invoker
# force-attaches every pack PNG, so it sees the full visual evidence like Kimi.
#
# Bless evidence (2026-07-10 quad calibration wave: 53 groups / 212 ballots on
# human-labeled groups): Muse had the TOP exact accuracy vs the human label
# (~67%; claude 65%, codex 63%, kimi 57%), 0/53 abstains, at ~$0.03/vote — and
# it exact-matched the human label on Boston group 0e3e10ad where the
# claude+codex majority got it wrong (a decorrelated voice paying rent). Known
# caveat, monitored: a recall-leaning A-bias (all 5 sole dissents leaned toward
# the inclusive option A; 2 Bogotá over-inclusions) — structurally CONTAINED by
# the consensus rules, because a dissent only ever blocks an auto-accept and
# routes the group to a human; a single voter can never mint a label.
#
# Muse is a REASONING model with hidden reasoning tokens: with a low output
# budget its JSON answer truncates mid-object (observed finish_reason "length"
# + null content when max_tokens starved the answer of room after the reasoning
# trace), so ``opencode.json`` gives the model a generous ``limit.output``; the
# spec carries the same 480s timeout as Kimi (reasoning models run long on
# large packs). An explicit ``--timeout`` still overrides it (resolve_timeout).
#
# Provider NAME is deliberately ``"muse"`` — distinct from the Kimi seat's
# ``"kimi"`` — even though Muse rides the SAME opencode transport as Kimi. The
# provider string is a KEYING field in several places: vote-provenance dedupe
# (``write_vote_provenance`` keys rows on (source_batch, group_id, provider)); the
# panel monitor's per-voter stats (``compute_voter_stats`` groups by provider); the
# ``provider=letter`` minority / route_reason strings; the resume-consistency
# provider-set check; and the ``--*-model`` CLI overrides. The ``quad-candidate``
# panel seats BOTH Kimi and Muse on that shared transport, so giving them the same
# name (e.g. both ``"opencode"``) would put two indistinguishable voters in one
# wave — their provenance rows would collapse to one (silent vote loss), their
# monitor stats would pool, and a single ``--*-model`` override would ambiguously
# hit both. Distinct ``"kimi"``/``"muse"`` names keep every voter individually
# addressable. Its invoker is resolved via ``_INVOKERS["muse"] -> invoke_opencode``
# (same transport), and the ``--agent`` threading keys on the RESOLVED INVOKER (not
# the name), so Muse still gets its tool-less ``vote`` agent. No committed votes
# carry the old ``("opencode", "meta/muse-spark-1.1")`` pair (smoke votes were never
# committed), so the rename rewrites nothing on disk.
MUSE = ProviderSpec(name="muse", model="meta/muse-spark-1.1", timeout=480, opencode_agent="vote")

# V7 records the already-configured Muse high reasoning policy explicitly in
# the panel invocation signature. ``invoke_opencode`` does not translate the
# generic effort field into a CLI flag; the actual Muse setting remains pinned
# by opencode.json's model-scoped ``reasoningEffort: high``. Keep the historical
# MUSE spec unchanged so v5/v6 invocation signatures remain reproducible.
MUSE_HIGH_EFFORT = ProviderSpec(
    name="muse",
    model="meta/muse-spark-1.1",
    effort="high",
    timeout=480,
    opencode_agent="vote",
)

# Experimental Gemini calibration voter: one LOGICAL Gemini 3.5 Flash seat with
# two physical routes. agy uses the user's Google quota first; a provider-scoped failure
# opens a wave-local circuit and falls back to the paid OpenRouter AI Studio
# flex endpoint. The route actually used is recorded on the Vote. Keeping the
# canonical model independent of transport makes the panel composition stable,
# while ``routes`` makes the ordered policy part of panel_invocation_sha256.
GEMINI_MODEL = "google/gemini-3.5-flash"
GEMINI_AGY_MODEL = "Gemini 3.5 Flash (Medium)"
GEMINI_ROUTE_AGY = "agy/google-ai-studio"
GEMINI_ROUTE_OPENROUTER_FLEX = "openrouter/google-ai-studio/flex"
_GEMINI_FLEX_PROVIDER_POLICY = {
    "only": ["google-ai-studio/flex"],
    "allow_fallbacks": False,
}
GEMINI = ProviderSpec(
    name="gemini",
    model=GEMINI_MODEL,
    opencode_agent="vote",
    routes=(GEMINI_ROUTE_AGY, GEMINI_ROUTE_OPENROUTER_FLEX),
)
GEMINI_AGY_ONLY = ProviderSpec(
    name="gemini",
    model=GEMINI_MODEL,
    routes=(GEMINI_ROUTE_AGY,),
)
GEMINI_FLEX_ONLY = ProviderSpec(
    name="gemini",
    model=GEMINI_MODEL,
    opencode_agent="vote",
    routes=(GEMINI_ROUTE_OPENROUTER_FLEX,),
)

# Panel v4 — the FORMER production default (2026-07-09 bless, #397; superseded
# by the v5 quad below on 2026-07-10). Kept as a named panel ("v4") so v4-era
# waves can be reproduced exactly, like "v3"/"v2":
#
#   * agy/Gemini Flash is REPLACED by kimi/Kimi K2.6. agy position-anchored
#     (11/12 votes "A" at constant 0.95 confidence in the w0707 waves) and only
#     reads pack images when it chooses to; Kimi agreed 6/6 with settled panel
#     verdicts across four different letters in the #397 smoke test, with
#     varied confidence and evidence-citing reasoning, at ~$0.04/vote.
#   * The codex voter bumps gpt-5.5 (low) -> gpt-5.6-sol (medium). Model id
#     verified 2026-07-09 against the codex CLI's server-fetched model listing
#     AND a live `codex exec -m gpt-5.6-sol` smoke test through the
#     invoke_codex invocation shape.
#
# Amendment (2026-07-10, Brad waived an era bump): the codex model is swapped
# gpt-5.6-sol -> gpt-5.6-terra IN PLACE (effort stays medium). In the 10-group
# sol anchor of the 2026-07-10 quad wave, sol showed 8/8 choice-agreement with
# claude/opus at >=0.86 confidence — no decorrelated signal to justify its
# premium quota; terra is the same gpt-5.6 family at a lower quota class, more
# apples-to-apples with opus/medium. Verified live via `codex exec -m
# gpt-5.6-terra` through the invoke_codex invocation shape. Safe as an in-place
# edit (no era bump): v4 has minted ZERO committed rows — the only codex model
# in committed labels/votes is v3-era gpt-5.5 — so it rewrites nothing on disk,
# same argument as the #402 kimi rename. PANEL_VOTERS_V4 moves in lockstep.
PANEL_V4 = [
    ProviderSpec(name="claude", model="claude-opus-4-8", effort="medium"),
    ProviderSpec(name="codex", model="gpt-5.6-terra", effort="medium"),
    OPENCODE_KIMI,
]

# Panel v5 — the production DEFAULT since the 2026-07-10 bless: the 4-seat
# "quad" composition (the v4 trio PLUS muse/Muse Spark 1.1 as a decorrelated
# fourth voice), blessed TOGETHER with the quorum consensus rule
# (compute_consensus: auto-accept when all valid votes agree and >=3 are
# valid), which is what makes a 4th seat pay for itself — an abstaining voter
# no longer blocks an otherwise-clean 3-of-4 agreement, it merely downgrades
# the accept's provenance from ``panel_unanimous_v5`` to ``panel_quorum_v5``.
#
# Bless evidence (2026-07-10 quad calibration wave, 53 groups / 212 ballots on
# human-labeled groups): muse top exact accuracy ~67% (claude 65%, codex 63%,
# kimi 57%), 0/53 muse abstains, ~$0.03/vote; muse exact-matched the human
# label on Boston 0e3e10ad where the claude+codex majority got it wrong. Known
# muse caveat (see the MUSE spec note): recall-leaning A-bias — contained,
# since a dissent only blocks auto-accept and can never mint a label.
# PANEL_VOTERS_V5 in stitch_export moves in lockstep (CI-asserted).
DEFAULT_PANEL = [*PANEL_V4, MUSE]

# Explicitly opt-in until the calibration gate in
# research/kimi_openrouter_routing_2026-07-12.md passes. The smoke replays found
# no auto-accept lift from a fourth Gemini seat, so v6 removes Kimi without a
# replacement and keeps the inexpensive Muse diversity voter.
PANEL_V6_CANDIDATE = [
    ProviderSpec(name="claude", model="claude-opus-4-8", effort="medium"),
    ProviderSpec(name="codex", model="gpt-5.6-terra", effort="medium"),
    MUSE,
]
# Route experiments deliberately remain four-wide so Gemini's marginal panel
# effect can be compared against the lean v6 baseline without changing the
# established Claude/Codex/Muse seats.
PANEL_V6_AGY_CALIBRATION = [*PANEL_V6_CANDIDATE[:2], GEMINI_AGY_ONLY, MUSE]
PANEL_V6_FLEX_CALIBRATION = [*PANEL_V6_CANDIDATE[:2], GEMINI_FLEX_ONLY, MUSE]

# Candidate v7: a new provenance era for the canonical-rubric replay. It keeps
# the lean three-family roster, moves Codex Terra -> Sol, and raises Claude and
# Codex to high effort. Muse was already invoked with high reasoning through
# opencode.json; MUSE_HIGH_EFFORT makes that policy explicit in the invocation
# signature without rewriting historical v5/v6 specs.
PANEL_V7_CANDIDATE = [
    ProviderSpec(name="claude", model="claude-opus-4-8", effort="high"),
    ProviderSpec(name="codex", model="gpt-5.6-sol", effort="high"),
    MUSE_HIGH_EFFORT,
]

# Named panel configurations. DEFAULT_PANEL (v5) is the default; historical
# compositions stay addressable so old batches can be reproduced exactly.
#
# ``no-agy`` is a QUOTA-OUTAGE fallback for v3-era reruns (observed 2026-07-06:
# agy silently returns exit 0 + empty output when its daily cap is hit): it
# swaps agy for the opencode/Qwen voter so a wave can proceed 3-wide. NOTE:
# panel composition is part of export-label provenance — stitch-export keys its
# gate on (provider, model) pairs, so labels from any non-blessed composition
# (no-agy, v3-candidate, v4-candidate, meta-candidate, and the v6/v7 candidates)
# are refused without --allow-nonstandard-panel.
PANELS: dict[str, list[ProviderSpec]] = {
    "default": DEFAULT_PANEL,
    "v5": DEFAULT_PANEL,
    # The FORMER 3-seat default (2026-07-09 bless). Kept addressable so v4-era
    # waves can be re-run/reproduced; its exports are stamped with the v4
    # labelers by stitch_export's era scoping.
    "v4": PANEL_V4,
    # v3-era compositions (v2 == v3 composition; the labeler bump was pack
    # inputs). Kept so v3-era batches can be re-run/reproduced; their exports
    # are stamped with the v3 labelers by stitch_export's era scoping.
    "v3": PANEL_V3,
    "v2": PANEL_V3,
    "v3-candidate": [*PANEL_V3, OPENCODE_QWEN],
    "no-agy": [*(p for p in PANEL_V3 if p.name != "agy"), OPENCODE_QWEN],
    # The #397 validation composition: the v3 panel with agy swapped for Kimi
    # (codex still gpt-5.5/low). SUPERSEDED by the v4 bless (which also bumped
    # the codex model) and then by v5 — so this remains NONSTANDARD to the
    # stitch-export (provider, model) gate; kept only to reproduce the
    # calibration/validation waves.
    "v4-candidate": [*(p for p in PANEL_V3 if p.name != "agy"), OPENCODE_KIMI],
    # Third-voter REPLACEMENT prototype on the v4 trio: kimi/Kimi -> muse/Muse
    # Spark 1.1 (Meta API). SUPERSEDED by v5 (which seats muse as a FOURTH
    # voter alongside kimi rather than replacing it); remains NONSTANDARD to
    # the stitch-export (provider, model) gate — kept only to reproduce the
    # Muse validation waves. Filter drops the v4 trio's Kimi seat.
    "meta-candidate": [*(p for p in PANEL_V4 if p.name != "kimi"), MUSE],
    # The 4-seat CALIBRATION composition that ran the 2026-07-10 quadcal0710
    # waves under the pre-quorum rules and was then blessed AS v5 (same
    # (provider, model) seats). Kept as an alias of the default so wave
    # scripts/docs that pinned --panel quad-candidate keep working and the
    # calibration waves stay reproducible; SUPERSEDED by "v5" as the name to
    # use going forward.
    "quad-candidate": DEFAULT_PANEL,
    "v6-candidate": PANEL_V6_CANDIDATE,
    "v6-agy-calibration": PANEL_V6_AGY_CALIBRATION,
    "v6-flex-calibration": PANEL_V6_FLEX_CALIBRATION,
    "v7-candidate": PANEL_V7_CANDIDATE,
}


def get_panel(name: str | None) -> list[ProviderSpec]:
    """Resolve a named panel config; an empty/None name means DEFAULT_PANEL.

    An UNKNOWN name is a hard error, not a silent default: panel choice is
    era-load-bearing (it decides which export labeler generation a wave's
    labels are stamped with), so a typo like ``--panel v3-candiate`` quietly
    running the v5 default would corrupt a wave's intended provenance.
    """
    if not name:
        return DEFAULT_PANEL
    try:
        return PANELS[name]
    except KeyError:
        raise ValueError(
            f"unknown panel {name!r}; valid panels: {', '.join(sorted(PANELS))}"
        ) from None


class AbstainReason(StrEnum):
    """Why a provider produced an ABSTAIN vote — drives the run_batch circuit breaker.

    ``TIMEOUT`` deliberately covers BOTH timeout flavors, which are the SAME
    failure (no answer before the deadline) surfaced two different ways:

    * a subprocess ``TimeoutExpired`` — the caller's ``timeout`` SIGKILLs the CLI; and
    * agy's CLI-internal response timeout (#343) — agy's own ``--print-timeout``,
      derived as strictly less than the subprocess timeout so it fires FIRST, and
      the CLI then exits nonzero with a clean ``timeout waiting for response``
      that ``_check_exit`` classifies as a :class:`GroupScopedProviderError`.

    Both MUST count toward the consecutive-timeout breaker: a provider network-
    blackholed mid-wave fails every group with the CLI-internal flavor, so if that
    flavor reset the counter the breaker (built for exactly this #334 silent-
    degradation mode) would never trip. Every OTHER abstain reason (context
    overflow, parse/validation) RESETS the breaker — those are a property of the
    group, not of provider health.
    """

    UNSET = ""
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    PARSE = "parse"


@dataclass
class Vote:
    """One provider's vote on one group."""

    group_id: str
    provider: str
    model: str
    choice: str  # option letter, "NONE", or "ABSTAIN"
    confidence: float
    reasoning: str
    edge_set: frozenset = field(default_factory=frozenset)  # mapped (ref,tgt) pairs
    latency_s: float = 0.0
    timestamp: str = ""
    raw: str = ""
    error: str = ""
    # Structured abstain classification (runtime-only; not serialized to votes.csv).
    # The circuit breaker in run_batch keys on this instead of string-matching the
    # free-text ``error`` — see :class:`AbstainReason`.
    abstain_reason: AbstainReason = AbstainReason.UNSET
    pack_feedback: str = ""  # diagnostic self-report JSON (wave-local; usually "")
    evidence_id: str = ""
    evidence_pack_sha256: str = ""
    displayed_candidate_universe_sha256: str = ""
    option_menu_sha256: str = ""
    chosen_option_id: str = ""
    panel_invocation_sha256: str = ""
    # Physical transport/endpoint that produced this logical voter's ballot.
    # Blank for legacy/single-route voters; required for Gemini calibration seats.
    invocation_route: str = ""
    # Canonical JSON describing the exact manifest-hashed images made
    # addressable to this invocation and the CLI delivery mechanism. This is a
    # delivery assertion, never a claim that the remote model consumed them.
    evidence_delivery: str = ""


@dataclass
class ProviderRouteState:
    """Wave-local circuit state for logical voters with fallback routes."""

    unavailable: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class InvocationResult:
    """Raw provider output plus the physical route that produced it."""

    raw: str
    route: str = ""


# ---------------------------------------------------------------------------
# Vote parsing / validation
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _scan_json_object(text: str) -> dict | None:
    """Scan text for the first balanced, parseable JSON object.

    Uses ``json.JSONDecoder.raw_decode`` from each ``{`` position instead of a
    regex: a greedy ``{.*}`` swallows trailing braces / concatenated objects,
    while a non-greedy ``{.*?}`` truncates when the reasoning string itself
    contains ``}``. ``raw_decode`` respects string escaping and nesting, so it
    returns exactly the first complete object.
    """
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            parsed, _end = decoder.raw_decode(text, idx)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
        idx = text.find("{", idx + 1)
    return None


def _extract_json_object(text: str) -> dict | None:
    """Extract the first JSON object from arbitrary text.

    Handles: raw JSON, ```json fenced blocks, JSON embedded in prose, and
    reasoning strings that themselves contain braces. Returns None if nothing
    parseable is found. Fenced content is preferred (a provider that fences
    its answer means THAT to be the answer).
    """
    if not text:
        return None
    m = _FENCE_RE.search(text)
    if m:
        obj = _scan_json_object(m.group(1))
        if obj is not None:
            return obj
    return _scan_json_object(text)


def parse_vote(raw_text: str, valid_letters: set[str]) -> tuple[str, float, str]:
    """Parse and validate a provider's raw output.

    Args:
        raw_text: The provider's stdout / response text.
        valid_letters: Set of acceptable option letters for this group.

    Returns:
        (choice, confidence, reasoning) where choice is a valid letter or "NONE".

    Raises:
        ValueError if no valid choice can be extracted.
    """
    obj = _extract_json_object(raw_text)
    if obj is None:
        raise ValueError("no JSON object found in output")

    choice = obj.get("choice")
    if choice is None:
        raise ValueError("missing 'choice' field")
    choice = str(choice).strip().upper()
    # Tolerate "OPTION A" / "A." style answers.
    choice = choice.replace("OPTION", "").strip().strip(".").strip()

    allowed = {ltr.upper() for ltr in valid_letters} | {"NONE"}
    if choice not in allowed:
        raise ValueError(f"choice {choice!r} not in {sorted(allowed)}")

    try:
        confidence = float(obj.get("confidence", 0.5))
    except (ValueError, TypeError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(obj.get("reasoning", "")).replace("\n", " ").strip()
    return choice, confidence, reasoning


# ---------------------------------------------------------------------------
# Diagnostic pack-feedback instrumentation (wave-local; default OFF).
#
# When enabled via the --pack-feedback flag, each panelist is asked to append a
# structured self-report to its JSON answer so we can mine WHAT EVIDENCE the
# pack failed to provide. This is a diagnostic-only augmentation: the default
# production prompt is untouched unless the flag is passed.
# ---------------------------------------------------------------------------

PACK_FEEDBACK_INSTRUCTION = (
    "\n\nADDITIONALLY (diagnostic — does not change your choice): include a "
    'fourth key "pack_feedback" in the SAME JSON object. It is an object with '
    "three keys:\n"
    '  "missing_info": list of short strings — information you needed to decide '
    "but the pack did not provide (e.g. a needed zoom level, a name, a class, an "
    "angle, connectivity).\n"
    '  "ambiguities": list of short strings — what was genuinely ambiguous or '
    "hard to disambiguate from the evidence shown.\n"
    '  "confidence_basis": short string — what your confidence chiefly rests on '
    "(or what would raise it).\n"
    "Use empty lists / empty string if nothing applies. Example: "
    '{"choice": "A", "confidence": 0.8, "reasoning": "...", "pack_feedback": '
    '{"missing_info": ["no street name on T1"], "ambiguities": ["R2/T1 overlap '
    'unclear at this zoom"], "confidence_basis": "clear parallel geometry"}}'
)


def augment_prompt_with_feedback(prompt: str) -> str:
    """Append the diagnostic pack-feedback request to a group prompt."""
    return prompt + PACK_FEEDBACK_INSTRUCTION


def _extract_pack_feedback(raw_text: str) -> str:
    """Pull the optional ``pack_feedback`` object from a provider's raw output.

    Returns a compact JSON string, or "" when absent/unparseable. Never raises —
    a missing self-report must not invalidate an otherwise-good vote.
    """
    obj = _extract_json_object(raw_text)
    if not obj or "pack_feedback" not in obj:
        return ""
    fb = obj.get("pack_feedback")
    if fb in (None, "", [], {}):
        return ""
    try:
        return json.dumps(fb, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(fb)


def choice_to_edge_set(
    choice: str, options_by_letter: dict[str, list[tuple[str, str]]]
) -> frozenset:
    """Map a choice letter to its (ref_id, target_id) edge set. NONE -> empty."""
    if choice == "NONE":
        return frozenset()
    return frozenset(options_by_letter.get(choice, []))


# ---------------------------------------------------------------------------
# Provider invocations
# ---------------------------------------------------------------------------

_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "choice": {"type": "string"},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
            # Optional diagnostic self-report (only requested when the
            # --pack-feedback flag augments the prompt). Declared here so a
            # schema-enforcing provider (claude) may emit it; omitted otherwise.
            "pack_feedback": {"type": "object", "additionalProperties": True},
        },
        "required": ["choice", "confidence", "reasoning"],
        "additionalProperties": False,
    }
)


def _image_paths(group_dir: Path, letters: list[str]) -> list[str]:
    imgs = [str(group_dir / "overview.png")]
    for ltr in letters:
        p = group_dir / f"option_{ltr}.png"
        if p.exists():
            imgs.append(str(p))
    # Junction zoom crops (#302 enrichment). claude/agy read these by the
    # paths embedded in the prompt, but codex only sees images attached via
    # -i, so they must be listed here or codex votes blind to the crops
    # (observed in the enriched_ab1 wave: codex pack_feedback "junction zooms
    # were referenced but not shown inline").
    imgs.extend(sorted(str(p) for p in group_dir.glob("zoom_*.png")))
    return imgs


#: Substrings that identify a context-window overflow in a provider CLI's
#: output. Deterministic for the (group, provider) pair: the same prompt will
#: overflow on every retry, but other groups remain servable, so this must be
#: classified as group-scoped (abstain + continue), never provider-down (halt).
#: Scanned over the FULL stdout+stderr, not the 500-char error snippet — codex
#: prints a long banner + echoed prompt before its overflow line.
_CONTEXT_OVERFLOW_MARKERS = (
    "context window",
    "context length",
    "context_length_exceeded",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "maximum context",
)

#: CLI-internal response-timeout messages (the CLI exits nonzero after its own
#: timer fires, e.g. agy's ``Error: timeout waiting for response``). Same fate as
#: a subprocess timeout in BOTH senses: abstain on this group AND count toward the
#: consecutive-timeout breaker (they are the same "no answer in time" failure —
#: agy's ``--print-timeout`` is derived to fire before the subprocess kill, #343).
#: Classified as ``GroupScopedProviderError(kind=AbstainReason.TIMEOUT)`` so the
#: breaker in ``run_batch`` counts it rather than resetting on it.
_CLI_TIMEOUT_MARKERS = ("timeout waiting for response",)

#: Consecutive timeout-abstentions from ONE provider before ``run_batch``
#: promotes it to the #334 provider-down halt. One or two slow oversized groups
#: abstain and the wave survives; a provider hanging on every group in a row is
#: provider health, not group size.
_TIMEOUT_BREAKER_N = 3


class GroupScopedProviderError(RuntimeError):
    """A provider failure that is deterministic for THIS group, not provider health.

    E.g. the group's prompt exceeds the model's context window, or the response
    timed out on an oversized group. Retrying the same group cannot succeed and
    halting the run would let one monster group kill a whole wave — instead the
    runner records an ABSTAIN vote (with the error trail) and continues; the
    group then routes to human review via the abstention/below-quorum path.

    ``kind`` carries the structured abstain classification (see
    :class:`AbstainReason`) so the caller can stamp it on the abstain vote without
    re-parsing the message. It defaults to ``CONTEXT_OVERFLOW`` (breaker-resetting):
    only a CLI-internal response timeout, raised explicitly with
    ``kind=AbstainReason.TIMEOUT``, counts toward the consecutive-timeout breaker.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: AbstainReason = AbstainReason.CONTEXT_OVERFLOW,
        invocation_route: str = "",
    ):
        super().__init__(message)
        self.kind = kind
        self.invocation_route = invocation_route


def _check_exit(provider: str, result: subprocess.CompletedProcess) -> None:
    """Raise on non-zero CLI exit so failures don't masquerade as parse errors.

    Group-scoped failures (context overflow, CLI-internal response timeout) are
    classified from the full output and raised as :class:`GroupScopedProviderError`
    with the matching :class:`AbstainReason` ``kind``; everything else raises
    ``RuntimeError`` with truncated stderr, which the runner treats as potential
    provider-down (backoff, then halt).
    """
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        combined = f"{result.stdout or ''}\n{stderr}".lower()
        if any(m in combined for m in _CONTEXT_OVERFLOW_MARKERS):
            raise GroupScopedProviderError(
                f"{provider} context overflow: prompt exceeds the model's context window",
                kind=AbstainReason.CONTEXT_OVERFLOW,
            )
        if any(m in combined for m in _CLI_TIMEOUT_MARKERS):
            raise GroupScopedProviderError(
                f"{provider} CLI-internal response timeout",
                kind=AbstainReason.TIMEOUT,
            )
        raise RuntimeError(f"{provider} exited with code {result.returncode}: {stderr[:500]}")


class ProviderInvocationError(RuntimeError):
    """A voter's CLI/API failed and did not recover within the retry budget.

    Raised for invocation/API failures (nonzero exit, quota exhaustion,
    rate-limit, network) that persist through the backoff window.
    It propagates out of ``run_batch`` to HALT the run rather than silently
    degrading the panel to fewer voters — a quota-exhausted provider abstaining
    would quietly weaken quorum on every subsequent group. Parse/validation
    failures do NOT raise this (a single malformed response still abstains).
    Neither do group-scoped failures — context overflow and timeouts abstain on
    that group and keep the run going (see :class:`GroupScopedProviderError`);
    only genuine provider-down conditions halt. The run is resumable: completed
    groups are already flushed to ``votes.partial.csv``.
    """


def invoke_claude(
    prompt: str,
    group_dir: Path,
    letters: list[str],
    model: str,
    timeout: int = 240,
    effort: str = "",
) -> str:
    """Invoke the claude CLI. Prompt via stdin, Read tool for images, JSON schema.

    Runs from a neutral tmp cwd to avoid inheriting a project CLAUDE.md.
    """
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--allowedTools",
        "Read",
        "--json-schema",
        _JSON_SCHEMA,
    ]
    if effort:
        cmd += ["--effort", effort]
    with tempfile.TemporaryDirectory() as neutral_cwd:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=neutral_cwd,
        )
    _check_exit("claude", result)
    return result.stdout


def invoke_codex(
    prompt: str,
    group_dir: Path,
    letters: list[str],
    model: str,
    timeout: int = 240,
    effort: str = "",
) -> str:
    """Invoke the codex CLI. Native multi-image via -i, JSON written to -o file.

    The prompt is piped via STDIN (``codex exec ... -``), not passed as an argv
    string: a large group's prompt (>128 KB) exceeds the OS single-argument limit
    (Linux MAX_ARG_STRLEN) and fails with E2BIG. codex reads instructions from
    stdin when ``-`` is given. codex's read-only sandbox blocks reading a context
    file, so stdin (not a file pointer) is the mechanism here.

    stdout is a transcript, so the answer is read from the -o output file.
    """
    imgs = _image_paths(group_dir, letters)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out.json"
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "--ephemeral",
            "-m",
            model,
            "-c",
            f"model_reasoning_effort={effort or 'low'}",
        ]
        for img in imgs:
            cmd += ["-i", img]
        cmd += ["-o", str(out_path), "-"]  # '-' => read the prompt from stdin
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        _check_exit("codex", result)
        if out_path.exists():
            txt = out_path.read_text().strip()
            if txt:
                return txt
    return result.stdout


# agy's CLI-internal response timer (``--print-timeout``) must fire strictly
# BEFORE the subprocess SIGKILL so agy exits with its own clean "timeout waiting
# for response" (a group-scoped abstain, #343) rather than being killed — a kill
# surfaces as a spurious provider-down. We aim to LEAD the subprocess deadline by
# this margin.
_AGY_TIMEOUT_MARGIN_S = 15
# Floor for the normal range (agy needs a few seconds to read the pack and answer).
# The strictly-less-than guarantee still wins for tiny caller timeouts, capping
# the derived timer below this floor when the floor itself would exceed the deadline.
_AGY_TIMEOUT_FLOOR_S = 10


def _agy_print_timeout(timeout: int) -> int:
    """Derive agy's ``--print-timeout`` (seconds) from the caller's subprocess timeout.

    The internal timer MUST be strictly less than ``timeout`` so agy fails first
    with a clean error (classified group-scoped and abstained, and counted by the
    consecutive-timeout breaker) rather than being SIGKILLed as a spurious
    provider-down. We lead the deadline by ``_AGY_TIMEOUT_MARGIN_S``, floored at
    ``_AGY_TIMEOUT_FLOOR_S`` for the normal range; the final ``min(..., timeout - 1)``
    guarantees strictly-less-than for ALL caller timeouts.

    Small-timeout behavior: at/under ~25s the 15s margin collapses to the floor,
    and under ~11s even the floor is capped down to ``timeout - 1`` — the clean-
    error window shrinks to seconds and the subprocess kill is the effective
    backstop. Production runs at 240s (-> 225s), unaffected.

    (Before this fix the derivation was ``max(30, timeout - 15)``, which INVERTED
    for caller timeouts <= 30: the internal timer met or exceeded the subprocess
    deadline, so the kill always won first and the clean-error design — the whole
    point of deriving the timer — never engaged.)
    """
    derived = max(timeout - _AGY_TIMEOUT_MARGIN_S, _AGY_TIMEOUT_FLOOR_S)
    return min(derived, timeout - 1)


def invoke_agy(
    prompt: str,
    group_dir: Path,
    letters: list[str],
    model: str,
    timeout: int = 240,
    effort: str = "",
) -> str:
    """Invoke the agy (Antigravity) CLI via a FILE POINTER.

    agy takes the prompt only as an argv value (``-p``) and does not read it from
    stdin, so a large group's prompt (>128 KB) would exceed the OS single-argument
    limit (Linux MAX_ARG_STRLEN) and fail with E2BIG. Instead the prompt is
    written to ``panel_prompt.txt`` in the (scratch, per-invocation) group dir and
    agy is given a tiny instruction to read it. agy is an agentic CLI that reads
    files — and the images the prompt references — from its cwd;
    ``--dangerously-skip-permissions`` avoids an interactive permission prompt in
    non-interactive mode. (Unlike codex, agy is not sandboxed away from the file.)

    Effort is encoded in the model name (e.g. "Gemini 3.5 Flash (Medium)"), so the
    ``effort`` argument is accepted for a uniform invoker signature but unused.
    """
    tmp_ctx = tempfile.TemporaryDirectory() if group_dir is None else None
    work = Path(tmp_ctx.name) if tmp_ctx is not None else Path(group_dir)
    try:
        prompt_file = work / "panel_prompt.txt"
        prompt_file.write_text(prompt)
        # agy's internal response timer is derived from the caller's timeout so it
        # fires with a clean error strictly BEFORE the subprocess kill (see
        # _agy_print_timeout): the old hardcoded 2m silently starved large groups
        # that Gemini needs >120s to answer, surfacing as a spurious "provider down".
        print_timeout_s = _agy_print_timeout(timeout)
        cmd = [
            "agy",
            f"--print-timeout={print_timeout_s}s",
            f"--model={model}",
            "--dangerously-skip-permissions",
            "-p",
            f"Read the file {prompt_file.name} in the current directory (and any "
            "image files it references), then output ONLY the answer it requests — "
            "no other text.",
        ]
        result = subprocess.run(
            cmd,
            input="",
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work),
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
    _check_exit("agy", result)
    return result.stdout


def invoke_opencode(
    prompt: str,
    group_dir: Path,
    letters: list[str],
    model: str,
    timeout: int = 240,
    effort: str = "",
    agent: str = "",
    config_content: dict | None = None,
) -> str:
    """Invoke the opencode CLI (OpenRouter-backed). Reads images by path via -f.

    The prompt is piped via STDIN (no positional message) to avoid the OS
    single-argument size limit on large groups (see ``invoke_codex``). ``-f``
    attaches images by path; multiple ``-f`` flags attach multiple images.

    The model string carries everything opencode needs (provider/model/knobs),
    so ``effort`` is accepted for a uniform invoker signature but unused. opencode
    prints the assistant's answer to stdout (TUI framing goes to stderr), so the
    raw stdout is returned for JSON extraction.

    ``agent`` (empty by default) selects an opencode agent via ``--agent``. Left
    empty for the residual v3-era Qwen voter, which runs under opencode's default
    ``build`` agent (its invocation stays byte-identical to before this knob
    existed). Both the Kimi and Muse voters pass ``agent="vote"`` — a tool-less
    agent (defined in the repo-root ``opencode.json``) that forces a pure-text
    answer instead of the agentic ls/cat/read loop that stalls a voter under
    ``build`` (Muse burned its turn on it; Kimi timed out on it).
    """
    imgs = _image_paths(group_dir, letters)
    # Pipe the prompt via STDIN (no positional message) rather than as an argv
    # string: a large group's prompt (>128 KB) exceeds the OS single-argument
    # limit (Linux MAX_ARG_STRLEN) and fails with E2BIG. opencode reads the
    # message from stdin when no positional message is given; -f still attaches
    # images by path.
    cmd = ["opencode", "run", "-m", model]
    if agent:
        cmd += ["--agent", agent]
    for img in imgs:
        cmd += ["-f", img]
    # Give each invocation its OWN opencode sqlite DB. opencode persists session
    # history to ~/.opencode/opencode.db by default, so concurrent vote
    # invocations (kimi + muse in one group, multiplied by the parallel wave
    # lanes) contend on that single sqlite and log "database is locked" retries
    # (observed in the 2026-07-10 wave). The opencode binary honors OPENCODE_DB:
    # pointing it at a per-invocation temp path moves the DB (and its -shm/-wal
    # siblings) off the shared file. Vote runs are one-shot, so their session
    # history is disposable. Auth is untouched (it stays in ~/.opencode/auth.json).
    # Copy os.environ and ADD the override — the subprocess still needs the rest of
    # the environment (META_API_KEY, OpenRouter creds, PATH, ...).
    db_dir = tempfile.mkdtemp(prefix="opencode_db_")
    env = {**os.environ, "OPENCODE_DB": str(Path(db_dir) / "opencode.db")}
    if config_content is not None:
        # OPENCODE_CONFIG_CONTENT is merged at highest precedence. The
        # route-aware Gemini caller supplies every ballot-changing key so a
        # stale project/user config cannot silently reroute a ballot while
        # provenance still claims AI Studio flex. Unrelated config may remain.
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config_content)
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    finally:
        # rmtree clears opencode.db plus its -shm/-wal siblings in one shot; the
        # finally guarantees cleanup even when subprocess.run raises (e.g. the
        # TimeoutExpired that _attempt_provider turns into a CLI-timeout abstain).
        shutil.rmtree(db_dir, ignore_errors=True)
    _check_exit("opencode", result)
    return result.stdout


def _gemini_flex_opencode_config(model: str) -> dict:
    """Complete isolated OpenCode config for the provenanced flex route."""
    model_key = model.removeprefix("openrouter/")
    return {
        "model": f"openrouter/{model_key}",
        "provider": {
            "openrouter": {
                "models": {
                    model_key: {
                        "options": {"provider": _GEMINI_FLEX_PROVIDER_POLICY},
                    }
                }
            }
        },
        "agent": {
            "vote": {
                "description": (
                    "Tool-less stitch-panel voter: answer ONLY from the attached "
                    "evidence pack, never call tools."
                ),
                "mode": "primary",
                "tools": {
                    name: False
                    for name in (
                        "write",
                        "edit",
                        "bash",
                        "read",
                        "glob",
                        "grep",
                        "list",
                        "webfetch",
                        "task",
                        "todowrite",
                        "todoread",
                        "patch",
                    )
                },
            }
        },
    }


def invoke_gemini(
    prompt: str,
    group_dir: Path,
    letters: list[str],
    model: str,
    timeout: int = 240,
    effort: str = "",
    agent: str = "vote",
    route_state: ProviderRouteState | None = None,
    routes: tuple[str, ...] = GEMINI.routes,
    evidence_manifest: dict | None = None,
) -> InvocationResult:
    """Invoke one logical Gemini voter with agy -> OpenRouter flex fallback.

    A provider-scoped agy failure opens a wave-local circuit so later groups go
    straight to OpenRouter instead of repeatedly paying for a known exhausted
    quota or broken CLI. Context overflow is group-scoped and therefore does
    not poison agy for later groups, but the current group still gets a chance
    through OpenRouter's larger context. OpenRouter errors propagate to the
    runner's existing backed-off retry/halt path; dual-route failure never
    silently degrades the panel.
    """
    if model != GEMINI_MODEL:
        raise ValueError(f"route-aware Gemini seat is pinned to {GEMINI_MODEL!r}, got {model!r}")
    supported = {GEMINI_ROUTE_AGY, GEMINI_ROUTE_OPENROUTER_FLEX}
    if not routes or not set(routes) <= supported or len(routes) != len(set(routes)):
        raise ValueError(f"invalid Gemini route policy: {routes!r}")

    state = route_state if route_state is not None else ProviderRouteState()
    last_error: BaseException | None = None
    last_route = ""
    for index, route in enumerate(routes):
        # The first, agentic route may write files into its scratch directory.
        # Re-verify the exact managed attachment set before EVERY physical route
        # so a fallback can never submit mutated or newly-created images while
        # the ballot later claims the original manifest bytes.
        if evidence_manifest is not None:
            _preflight_native_attachment_assets(group_dir, letters, evidence_manifest)
        last_route = route
        has_fallback = index + 1 < len(routes)
        if route in state.unavailable:
            last_error = RuntimeError(f"Gemini route circuit is open: {route}")
            continue

        if route == GEMINI_ROUTE_AGY:
            try:
                raw = invoke_agy(prompt, group_dir, letters, GEMINI_AGY_MODEL, timeout, effort)
                if not raw or not raw.strip():
                    raise RuntimeError("empty output (exit 0) — provider likely quota-capped")
                return InvocationResult(raw=raw, route=route)
            except GroupScopedProviderError as exc:
                last_error = exc
                # Context overflow is tied to this pack; CLI response timeout is a
                # provider-health signal and opens the circuit.
                if exc.kind == AbstainReason.TIMEOUT:
                    state.unavailable.add(route)
            except subprocess.TimeoutExpired as exc:
                last_error = exc
                state.unavailable.add(route)
            except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
                last_error = exc
                state.unavailable.add(route)
        else:
            openrouter_model = model if model.startswith("openrouter/") else f"openrouter/{model}"
            try:
                raw = invoke_opencode(
                    prompt,
                    group_dir,
                    letters,
                    openrouter_model,
                    timeout,
                    effort,
                    agent=agent,
                    config_content=_gemini_flex_opencode_config(openrouter_model),
                )
                return InvocationResult(raw=raw, route=route)
            except (
                GroupScopedProviderError,
                subprocess.SubprocessError,
                OSError,
                RuntimeError,
            ) as exc:
                last_error = exc

        if has_fallback:
            logger.warning(
                f"gemini route {route} failed ({last_error}); falling back to {routes[index + 1]}"
            )

    if last_error is not None:
        # Preserve the physical route on a group-scoped failure so an ABSTAIN
        # ballot can still carry honest delivery provenance. Exception objects
        # are mutable, including subprocess.TimeoutExpired.
        last_error.invocation_route = last_route  # type: ignore[attr-defined]
        raise last_error
    raise RuntimeError(f"all Gemini routes unavailable: {routes!r}")


_INVOKERS = {
    "claude": invoke_claude,
    "codex": invoke_codex,
    "agy": invoke_agy,
    # Three seats resolve to the opencode transport. "opencode" is the residual
    # v3-era seat (the Qwen voter in v3-candidate/no-agy, which historical batches
    # still reproduce). "kimi" and "muse" ride the SAME transport but carry
    # DISTINCT provider names so every provider-keyed site (provenance dedupe,
    # monitor stats, minority strings, resume check, CLI --*-model overrides)
    # addresses them separately — the quad-candidate panel seats both. The
    # ``--agent`` threading in ``_attempt_provider`` keys on the RESOLVED invoker
    # (``invoker is invoke_opencode``), not the name, so Muse's tool-less ``vote``
    # agent is still forwarded under its distinct name.
    "opencode": invoke_opencode,
    "kimi": invoke_opencode,
    "muse": invoke_opencode,
    "gemini": invoke_gemini,
}


# ---------------------------------------------------------------------------
# Per-group panel run
# ---------------------------------------------------------------------------


_PROMPT_PNG_FIELD_RE = re.compile(
    r"^(?P<prefix>[ \t]*(?:-[ \t]+)?(?:overview|junction zoom|image):[ \t]*)"
    r"(?P<path>.*?\.png)(?P<suffix>[ \t]*)$",
    re.IGNORECASE | re.MULTILINE,
)
_PROMPT_PNG_TOKEN_RE = re.compile(
    r"(?P<path>(?:(?:[A-Za-z]:)?[\\/]|\.{1,2}[\\/])?"
    r"[^\s\"'`<>{}\[\](),;]+?\.png)",
    re.IGNORECASE,
)
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _managed_image_name(name: str) -> bool:
    """Whether ``name`` belongs to the generated evidence-image namespace."""
    return (
        name == "overview.png"
        or (name.startswith("option_") and name.endswith(".png"))
        or (name.startswith("zoom_") and name.endswith(".png"))
    )


def _png_ref_name(ref: str) -> str:
    """Return a PNG reference's basename for both POSIX and Windows prompts."""
    return ref.strip().replace("\\", "/").rsplit("/", 1)[-1]


def _rewrite_prompt_png_refs(prompt: str, scratch: Path, managed_names: set[str]) -> str:
    """Relocate references to this pack's managed PNGs into ``scratch``.

    Packs are sometimes renamed after generation, so the absolute directory
    embedded in a historical prompt is not a reliable identity. Managed image
    basenames are pack-local and unique; rewriting by basename supports stale
    absolute paths as well as the relative names used in prompt prose. Explicit
    image fields are handled separately so paths containing spaces remain
    supported.
    """

    def relocate(ref: str) -> str:
        if _URI_RE.match(ref.strip()):
            return ref
        name = _png_ref_name(ref)
        if name in managed_names:
            return str(scratch / name)
        return ref

    def field_repl(match: re.Match) -> str:
        return f"{match.group('prefix')}{relocate(match.group('path'))}{match.group('suffix')}"

    rewritten = _PROMPT_PNG_FIELD_RE.sub(field_repl, prompt)
    return _PROMPT_PNG_TOKEN_RE.sub(lambda match: relocate(match.group("path")), rewritten)


def _prompt_png_refs(prompt: str) -> set[str]:
    """Extract explicit and inline local/remote PNG references from a prompt."""
    refs: set[str] = set()
    for line in prompt.splitlines():
        field = _PROMPT_PNG_FIELD_RE.fullmatch(line)
        if field:
            # Do not token-scan a field as well: for a path containing spaces,
            # the token matcher would see a false relative suffix.
            refs.add(field.group("path").strip())
        else:
            refs.update(match.group("path") for match in _PROMPT_PNG_TOKEN_RE.finditer(line))
    return refs


def _preflight_prompt_png_refs(prompt: str, scratch: Path, managed_names: set[str]) -> None:
    """Fail closed when a prompt's local PNG reference cannot be read.

    Managed images must resolve to the isolated scratch copy, never to a stale
    canonical/renamed directory. Other PNGs remain unmanaged: they are not
    copied, but an explicit absolute reference is allowed when it really exists.
    """
    unresolved: list[str] = []
    for ref in sorted(_prompt_png_refs(prompt)):
        if _URI_RE.match(ref):
            continue
        name = _png_ref_name(ref)
        expected = scratch / name
        path = Path(ref).expanduser()
        if not path.is_absolute():
            path = scratch / path
        if _managed_image_name(name):
            if name not in managed_names or path != expected or not expected.is_file():
                unresolved.append(ref)
        elif not path.is_file():
            unresolved.append(ref)
    if unresolved:
        refs = ", ".join(repr(ref) for ref in unresolved)
        raise ValueError(
            f"prompt contains unresolved local PNG reference(s) after scratch rewrite: {refs}"
        )


def _scratch_managed_image_descriptors(scratch: Path) -> list[dict]:
    """Hash the exact managed image set copied into one invocation scratch."""
    return sorted(
        (
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in scratch.iterdir()
            if path.is_file() and _managed_image_name(path.name)
        ),
        key=lambda item: item["path"],
    )


def _preflight_manifest_delivery_assets(prompt: str, scratch: Path, manifest: dict) -> None:
    """Bind scratch bytes and prompt references to a verified pack manifest."""
    expected = managed_image_descriptors(manifest)
    actual = _scratch_managed_image_descriptors(scratch)
    if actual != expected:
        raise EvidenceProvenanceError(
            "invocation scratch image bytes/count/set do not match the evidence manifest"
        )

    expected_names = {asset["path"] for asset in expected}
    referenced_names: set[str] = set()
    unexpected_refs: list[str] = []
    for ref in sorted(_prompt_png_refs(prompt)):
        if _URI_RE.match(ref):
            unexpected_refs.append(ref)
            continue
        name = _png_ref_name(ref)
        if name in expected_names:
            referenced_names.add(name)
        else:
            unexpected_refs.append(ref)
    if unexpected_refs or referenced_names != expected_names:
        missing = sorted(expected_names - referenced_names)
        raise EvidenceProvenanceError(
            "prompt image references do not match the evidence manifest: "
            f"missing={missing}, unexpected={unexpected_refs}"
        )


def _scratch_pack(
    group_dir: Path,
    prompt: str,
    evidence_manifest: dict | None = None,
) -> tuple[Path, str, tempfile.TemporaryDirectory]:
    """Copy a pack's images into an isolated scratch dir and rewrite the prompt.

    Some providers (agy) write derived crops next to the images they inspect.
    Giving each invocation an isolated copy keeps the canonical evidence dir
    pristine and bounds any provider scratch to an auto-cleaned temp dir. Prompt
    PNG references are relocated by managed basename rather than canonical-dir
    string, so a pack remains runnable after its batch directory is renamed.
    """
    tmp = tempfile.TemporaryDirectory(prefix="stitch_pack_")
    scratch = Path(tmp.name)
    try:
        managed_images = [group_dir / "overview.png"]
        managed_images.extend(sorted(group_dir.glob("option_*.png")))
        managed_images.extend(sorted(group_dir.glob("zoom_*.png")))
        managed_images = [img for img in managed_images if img.is_file()]
        for img in managed_images:
            shutil.copy2(img, scratch / img.name)
        managed_names = {img.name for img in managed_images}
        rewritten = _rewrite_prompt_png_refs(prompt, scratch, managed_names)
        _preflight_prompt_png_refs(rewritten, scratch, managed_names)
        if evidence_manifest is not None:
            _preflight_manifest_delivery_assets(rewritten, scratch, evidence_manifest)
        return scratch, rewritten, tmp
    except BaseException:
        tmp.cleanup()
        raise


def _load_group_context(group_dir: Path) -> tuple[list[str], dict, dict]:
    """Load option letters + edge-set map + metadata for a generated pack."""
    meta = yaml.safe_load((group_dir / "metadata.yaml").read_text())
    letters = [o["letter"] for o in meta["options"]]
    # Map letter -> [(ref_id, target_id)] using the metadata segment tables.
    ref_by_label = {s["label"]: s["id"] for s in meta["segments"]["reference"]}
    tgt_by_label = {s["label"]: s["id"] for s in meta["segments"]["target"]}
    options_by_letter: dict[str, list[tuple[str, str]]] = {}
    for opt in meta["options"]:
        pairs = []
        for e in opt["edges"]:
            rid = ref_by_label.get(e["ref"], e["ref"])
            tid = tgt_by_label.get(e["target"], e["target"])
            pairs.append((rid, tid))
        options_by_letter[opt["letter"]] = pairs
    return letters, options_by_letter, meta


def _segment_class_maps(meta: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Build per-side {segment_id: class} maps from a pack's metadata.

    Ref and target IDs come from different namespaces, so the maps are kept
    separate — a shared dict could let an ID collision misclassify edges.
    """
    ref_out: dict[str, str] = {}
    tgt_out: dict[str, str] = {}
    for side, out in (("reference", ref_out), ("target", tgt_out)):
        for s in meta.get("segments", {}).get(side, []):
            out[str(s["id"])] = s.get("class", "") or ""
    return ref_out, tgt_out


def _edge_classes_for(
    edge_set: frozenset,
    ref_class: dict[str, str],
    tgt_class: dict[str, str],
) -> list[tuple[str, str]]:
    """Map a chosen (ref_id, target_id) edge set to (ref_class, target_class)."""
    return [(ref_class.get(str(r), ""), tgt_class.get(str(t), "")) for r, t in edge_set]


def _delivery_mode_transport(provider: str, invocation_route: str = "") -> tuple[str, str]:
    """Resolve the evidence-delivery contract for an actual physical route."""
    if provider == "claude":
        return DELIVERY_MODE_PROMPT_PATH, "claude:Read"
    if provider == "agy":
        return DELIVERY_MODE_PROMPT_PATH, "agy:agent-read"
    if provider == "codex":
        return DELIVERY_MODE_NATIVE_ATTACHMENT, "codex:-i"
    if provider in {"opencode", "kimi", "muse"}:
        return DELIVERY_MODE_NATIVE_ATTACHMENT, "opencode:-f"
    if provider == "gemini":
        if invocation_route == GEMINI_ROUTE_AGY:
            return DELIVERY_MODE_PROMPT_PATH, "agy:agent-read"
        if invocation_route == GEMINI_ROUTE_OPENROUTER_FLEX:
            return DELIVERY_MODE_NATIVE_ATTACHMENT, "opencode:-f"
        raise EvidenceProvenanceError(
            "route-aware Gemini ballot is missing a known physical invocation_route"
        )
    raise EvidenceProvenanceError(f"no evidence-delivery contract for provider {provider!r}")


def _preflight_native_attachment_assets(
    scratch: Path,
    letters: list[str],
    manifest: dict,
) -> None:
    """Verify the shared ``_image_paths`` set equals the manifested image set.

    Claude/agy use prompt paths, but checking this for every provider also
    proves either route of the dynamic Gemini voter is ready before invocation.
    Codex and OpenCode build their actual ``-i``/``-f`` argv from this same
    helper, so the recorded assets and native attachment set cannot drift.
    """
    attached = sorted(
        (
            {
                "path": Path(raw_path).name,
                "bytes": Path(raw_path).stat().st_size,
                "sha256": sha256_file(Path(raw_path)),
            }
            for raw_path in _image_paths(scratch, letters)
        ),
        key=lambda item: item["path"],
    )
    if attached != managed_image_descriptors(manifest):
        raise EvidenceProvenanceError(
            "native attachment image bytes/count/set do not match the evidence manifest"
        )


def run_provider_on_group(
    provider: ProviderSpec,
    group_id: str,
    group_dir: Path | None,
    prompt: str,
    letters: list[str],
    options_by_letter: dict[str, list[tuple[str, str]]],
    timeout: int | None = None,
    retries: int = 1,
    collect_feedback: bool = False,
    invocation_budget_s: float = 300.0,
    route_state: ProviderRouteState | None = None,
    evidence_manifest: dict | None = None,
) -> Vote:
    """Run one provider on one group; abstain on bad output, hard-fail if down.

    ``timeout=None`` (the default) resolves per provider: the spec's own
    ``timeout`` if set (e.g. 480s for the kimi/Kimi voter), else
    :data:`DEFAULT_VOTE_TIMEOUT_S`. An explicit value overrides both — see
    :func:`resolve_timeout`.

    Parse/validation failures retry ``retries`` times then abstain. Invocation/
    API failures (nonzero exit, quota, rate-limit, timeout) back off and retry
    within ``invocation_budget_s``, then raise :class:`ProviderInvocationError`.

    Each invocation runs against an isolated scratch copy of the pack so a
    provider that writes derived files (e.g. agy crops) never pollutes the
    canonical evidence dir. Skipped when group_dir is None (unit tests).
    Production callers pass the already-verified ``evidence_manifest``; direct
    helper calls may omit it and retain the legacy blank-delivery behavior.
    """
    invoker = _INVOKERS[provider.name]
    valid = set(letters)
    timeout = resolve_timeout(provider, timeout)

    scratch_dir = group_dir
    run_prompt = prompt
    tmp_ctx = None
    try:
        if evidence_manifest is not None and group_dir is None:
            raise EvidenceProvenanceError(
                "evidence delivery provenance requires a group directory for preflight"
            )
        if group_dir is not None:
            scratch_dir, run_prompt, tmp_ctx = _scratch_pack(
                group_dir,
                prompt,
                evidence_manifest=evidence_manifest,
            )

        vote = _attempt_provider(
            provider,
            group_id,
            scratch_dir,
            run_prompt,
            letters,
            options_by_letter,
            valid,
            invoker,
            timeout,
            retries,
            collect_feedback,
            invocation_budget_s=invocation_budget_s,
            route_state=route_state,
            evidence_manifest=evidence_manifest,
        )
        if evidence_manifest is not None:
            delivery_mode, transport = _delivery_mode_transport(
                provider.name,
                vote.invocation_route,
            )
            vote.evidence_delivery = canonical_evidence_delivery_json(
                build_evidence_delivery_record(
                    evidence_manifest,
                    delivery_mode=delivery_mode,
                    transport=transport,
                )
            )
        return vote
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def _attempt_provider(
    provider,
    group_id,
    group_dir,
    prompt,
    letters,
    options_by_letter,
    valid,
    invoker,
    timeout,
    retries,
    collect_feedback=False,
    invocation_budget_s: float = 300.0,
    route_state: ProviderRouteState | None = None,
    evidence_manifest: dict | None = None,
) -> Vote:
    """Run one provider, distinguishing two failure classes with opposite fates:

    * **Invocation/API failure** (nonzero exit incl. quota, rate-limit, network,
      missing binary) — back off (exponential, capped 60s) and retry until
      ``invocation_budget_s`` (default 5 min) is spent, then raise
      :class:`ProviderInvocationError` to HALT the run rather than silently
      degrade the panel to fewer voters. Only *expected external* failures are
      caught — subprocess/OS errors and the nonzero-exit ``RuntimeError`` from
      ``_check_exit``; an unexpected exception type (a programming bug inside an
      invoker) propagates immediately instead of masquerading as a quota error
      and burning the whole budget.
    * **Group-scoped failure** (context-window overflow, response timeout) —
      ABSTAIN immediately and keep the run going. These are deterministic for
      the (group, provider) pair: retrying burns budget without hope (overflow)
      or another full timeout window (timeout), and halting would let one
      monster group kill a whole wave. The abstention carries the error trail,
      and the group routes to human review via abstention/below-quorum.
    * **Parse/validation failure** (malformed output) — retry up to ``retries``
      times, then ABSTAIN. A single bad response should not kill a whole sweep.
    """

    def _abstain(error: str, raw: str = "", reason: AbstainReason = AbstainReason.UNSET) -> Vote:
        return Vote(
            group_id=group_id,
            provider=provider.name,
            model=provider.model,
            choice="ABSTAIN",
            confidence=0.0,
            reasoning="",
            edge_set=frozenset(),
            latency_s=0.0,
            timestamp=datetime.now(UTC).isoformat(),
            raw=raw[:2000],
            error=error,
            abstain_reason=reason,
            invocation_route=last_route,
        )

    last_parse_err = ""
    last_raw = ""
    last_route = ""
    parse_attempts = 0
    deadline = time.monotonic() + invocation_budget_s
    backoff = 5.0
    # The OpenCode invoker and route-aware Gemini wrapper accept an ``agent``.
    # Pass it only to those resolved callables so other invokers retain their
    # historical six-argument signature. Key on the callable, not provider name:
    # Kimi and Muse use distinct logical names on the same OpenCode transport.
    extra_kwargs = {}
    if invoker in (invoke_opencode, invoke_gemini) and provider.opencode_agent:
        extra_kwargs["agent"] = provider.opencode_agent
    if invoker is invoke_gemini:
        extra_kwargs["route_state"] = route_state
        extra_kwargs["routes"] = provider.routes
        extra_kwargs["evidence_manifest"] = evidence_manifest
    while True:
        # Parse retries and backed-off provider retries reuse one scratch dir.
        # Re-hash the native attachment set immediately before every attempt so
        # a prior agentic attempt cannot mutate the evidence behind a later
        # successful ballot.
        if evidence_manifest is not None:
            _preflight_native_attachment_assets(group_dir, letters, evidence_manifest)
        start = time.monotonic()
        try:
            result = invoker(
                prompt, group_dir, letters, provider.model, timeout, provider.effort, **extra_kwargs
            )
            if isinstance(result, InvocationResult):
                raw = result.raw
                last_route = result.route
            else:
                raw = result
                last_route = ""
        except GroupScopedProviderError as e:
            # Deterministic for this (group, provider): same prompt, same fate.
            # Abstain loudly and let the run continue. The exception's ``kind``
            # (context overflow vs CLI-internal timeout) rides onto the abstain so
            # run_batch's breaker counts CLI-internal timeouts but not overflows.
            last_route = e.invocation_route or last_route
            logger.warning(
                f"{provider.name} group {group_id}: {e}; abstaining (group-scoped, "
                f"not retryable) and continuing the run"
            )
            return _abstain(f"group-scoped: {e}", reason=e.kind)
        except subprocess.TimeoutExpired as e:
            # Retrying a timeout just burns another full timeout window, and a
            # group whose prompt is too slow for this provider is a property of
            # the group, not provider health -> abstain, don't halt the wave. The
            # consecutive-timeout breaker in run_batch still catches a provider
            # that times out on EVERY group (via AbstainReason.TIMEOUT).
            last_route = getattr(e, "invocation_route", "") or last_route
            logger.warning(
                f"{provider.name} group {group_id}: timeout after {timeout}s; "
                f"abstaining and continuing the run"
            )
            return _abstain(f"timeout after {timeout}s", reason=AbstainReason.TIMEOUT)
        except (subprocess.SubprocessError, OSError, RuntimeError) as e:
            # Expected external failure (nonzero exit incl. quota/auth, missing
            # binary, network). Unexpected exception types are NOT caught
            # here — a programming bug must fail fast, not retry as a "quota" error.
            err = f"invocation error: {e}"
            # Deterministic OS errors (arg-list-too-long, missing binary, perms)
            # will fail identically on every retry — hard-fail immediately instead
            # of burning the whole backoff budget on a no-hope loop.
            if isinstance(e, OSError) and e.errno in _FATAL_ERRNOS:
                raise ProviderInvocationError(
                    f"{provider.name} on group {group_id}: {err} (errno "
                    f"{e.errno}={errno.errorcode.get(e.errno, '?')}, not retryable) — "
                    f"halting the run."
                ) from e
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderInvocationError(
                    f"{provider.name} on group {group_id}: {err}; gave up after "
                    f"{invocation_budget_s:.0f}s of backed-off retries. Halting the run "
                    f"rather than degrading the panel — fix quota/credentials and resume."
                ) from e
            sleep_s = min(backoff, remaining)
            logger.warning(
                f"{provider.name} group {group_id}: {err}; retrying in {sleep_s:.0f}s "
                f"({remaining:.0f}s budget left)"
            )
            time.sleep(sleep_s)
            backoff = min(backoff * 2, 60.0)
            continue
        latency = time.monotonic() - start
        if not raw or not raw.strip():
            # Empty output from a zero-exit CLI is PROVIDER failure, not model
            # output: observed live when agy hits its daily quota cap — every
            # call returns exit 0 with empty stdout/stderr. Routing it through
            # the parse path would abstain per group and silently degrade the
            # panel on every remaining group (the exact #334 failure mode,
            # invisible to both the nonzero-exit halt and the timeout breaker).
            # Classify as an invocation failure: backoff, then halt.
            err = "invocation error: empty output (exit 0) — provider likely quota-capped"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderInvocationError(
                    f"{provider.name} on group {group_id}: {err}; gave up after "
                    f"{invocation_budget_s:.0f}s of backed-off retries. Halting the run "
                    f"rather than degrading the panel — fix quota/credentials and resume."
                )
            sleep_s = min(backoff, remaining)
            logger.warning(
                f"{provider.name} group {group_id}: {err}; retrying in {sleep_s:.0f}s "
                f"({remaining:.0f}s budget left)"
            )
            time.sleep(sleep_s)
            backoff = min(backoff * 2, 60.0)
            continue
        last_raw = raw
        try:
            choice, confidence, reasoning = parse_vote(raw, valid)
            return Vote(
                group_id=group_id,
                provider=provider.name,
                model=provider.model,
                choice=choice,
                confidence=confidence,
                reasoning=reasoning,
                edge_set=choice_to_edge_set(choice, options_by_letter),
                latency_s=round(latency, 2),
                timestamp=datetime.now(UTC).isoformat(),
                raw=raw[:2000],
                pack_feedback=_extract_pack_feedback(raw) if collect_feedback else "",
                invocation_route=last_route,
            )
        except ValueError as e:
            parse_attempts += 1
            last_parse_err = f"parse/validation: {e}"
            logger.warning(f"{provider.name} group {group_id} parse attempt {parse_attempts}: {e}")
            if parse_attempts > retries:
                break  # exhausted parse retries -> abstain (keep the panel going)

    # Parse retries exhausted -> abstention. Parse failures are a property of the
    # response, not provider health, so this reason resets the timeout breaker.
    return _abstain(last_parse_err, raw=last_raw, reason=AbstainReason.PARSE)


def run_panel_on_group(
    group_id: str,
    group_dir: Path,
    panel: list[ProviderSpec],
    timeout: int | None = None,
    collect_feedback: bool = False,
    invocation_budget_s: float = 300.0,
    route_state: ProviderRouteState | None = None,
    evidence_manifest: dict | None = None,
) -> list[Vote]:
    """Run the full panel on one group in parallel (one thread per provider).

    ``timeout=None`` resolves per provider (spec timeout, else the global
    default); an explicit value applies to every provider (resolve_timeout).

    A provider that stays down past ``invocation_budget_s`` raises
    ProviderInvocationError, which propagates out to halt ``run_batch``. The halt
    is not instantaneous: ``ThreadPoolExecutor.__exit__`` waits for the other
    providers (possibly mid-backoff) before re-raising — acceptable since we are
    aborting anyway and completed groups are already flushed for ``--resume``.
    """
    letters, options_by_letter, _meta = _load_group_context(group_dir)
    prompt = (group_dir / "prompt.txt").read_text()
    if collect_feedback:
        prompt = augment_prompt_with_feedback(prompt)

    def _run(p: ProviderSpec) -> Vote:
        return run_provider_on_group(
            p,
            group_id,
            group_dir,
            prompt,
            letters,
            options_by_letter,
            timeout,
            collect_feedback=collect_feedback,
            invocation_budget_s=invocation_budget_s,
            route_state=route_state,
            evidence_manifest=evidence_manifest,
        )

    with ThreadPoolExecutor(max_workers=len(panel)) as ex:
        votes = list(ex.map(_run, panel))
    return votes


# ---------------------------------------------------------------------------
# Class-consistency gate
# ---------------------------------------------------------------------------
#
# Deterministic mitigation for a specific auto-accept failure mode: a reference
# segment of one travel MODE (e.g. a pedestrian footway/sidewalk, or a bike
# cycleway) matched to a target segment of a DIFFERENT mode (e.g. a vehicular
# road), or vice-versa. A sidewalk or a separated cycleway that runs alongside a
# road is a DIFFERENT physical feature than the road, so such a cross-mode edge
# is almost never a true same-traveled-way correspondence — but its geometry is
# parallel and nearby, which can fool a geometry-only panel.
#
# The gate demotes any auto-accept candidate whose CHOSEN edge set contains a
# cross-mode edge to human review. It only fires when BOTH sides are
# unambiguously classified into DIFFERENT modes: same-mode pairs, and any pair
# where a class is missing/unknown/ambiguous (neutral), pass (we do not
# over-gate on absent data).
#
# Mode-set membership (derived from the OSM `class` values that appear in the
# stitch batches — footway/path/pedestrian/steps for pedestrian, cycleway for
# bike, and motorway..living_street for vehicular):
#
#   * PEDESTRIAN_CLASSES — foot-only ways. `sidewalk` and `crossing` are
#     included for completeness (standard OSM footway subtypes) even though the
#     current Boston data tags them as `footway`.
#   * VEHICULAR_CLASSES — the drivable road hierarchy.
#   * CYCLEWAY_CLASSES — the bike mode. `cycleway` is its OWN mode (no longer
#     neutral): on co_bogota_bike_network the reference is road-class centerlines
#     while the targets are `cycleway`, so a road↔cycleway edge is a genuine
#     cross-mode mismatch that must route to human review rather than auto-accept
#     (Brad's decision, 2026-07-05). Modes are compared pairwise (any two
#     DIFFERENT non-neutral modes are cross-mode), so pedestrian↔cycleway is ALSO
#     treated as cross-mode — the conservative default (Brad ruled only on
#     road↔cycleway; flagged as a reviewer-checkable default in the PR body).
#     cycleway↔cycleway stays same-mode and remains auto-acceptable.
#   * Everything else (track, alley, unknown, "", None) is treated as NEUTRAL and
#     never triggers the gate. `track` is deliberately neutral: it is genuinely
#     ambiguous (a farm/service track drivable or not), and the gate's job is to
#     catch clear cross-mode mismatches, not to adjudicate ambiguous classes.
#     `alley` does not appear in the data at all.

PEDESTRIAN_CLASSES = frozenset({"footway", "sidewalk", "path", "pedestrian", "steps", "crossing"})
VEHICULAR_CLASSES = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "service",
        "unclassified",
        "living_street",
        "driveway",
        "road",
    }
)
CYCLEWAY_CLASSES = frozenset({"cycleway"})


def road_class_mode(cls: str | None) -> str:
    """Classify an OSM road class into a travel mode.

    Returns "pedestrian", "vehicular", "bike", or "neutral"
    (unknown/ambiguous/missing).
    """
    c = (cls or "").strip().lower()
    if c in PEDESTRIAN_CLASSES:
        return "pedestrian"
    if c in VEHICULAR_CLASSES:
        return "vehicular"
    if c in CYCLEWAY_CLASSES:
        return "bike"
    return "neutral"


def is_cross_mode_edge(ref_class: str | None, target_class: str | None) -> bool:
    """True iff the two sides are unambiguously classified into DIFFERENT modes.

    Modes are pedestrian / vehicular / bike (see :func:`road_class_mode`). A pair
    is cross-mode iff neither side is neutral AND the two modes differ — so
    road↔cycleway and pedestrian↔cycleway are cross-mode, while cycleway↔cycleway
    (and any same-mode pair) is not. Any pair involving a neutral/unknown/missing
    class returns False (it passes the gate — we do not over-gate on absent data).
    """
    a = road_class_mode(ref_class)
    b = road_class_mode(target_class)
    if a == "neutral" or b == "neutral":
        return False
    return a != b


def has_cross_mode_edge(edge_classes: list[tuple[str | None, str | None]]) -> bool:
    """True iff any (ref_class, target_class) pair in the set is cross-mode."""
    return any(is_cross_mode_edge(rc, tc) for rc, tc in edge_classes)


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


@dataclass
class Consensus:
    group_id: str
    # Agreement tier. "unanimous": every recorded vote is valid and agrees.
    # "quorum": every VALID vote agrees, >=3 are valid, but >=1 panelist
    # abstained (reachable only with >=4 voters; introduced with the v5 quorum
    # rule — pre-v5 rows never carry it). "majority"/"none": no all-valid
    # agreement.
    consensus: str  # "unanimous" | "quorum" | "majority" | "none"
    choice: str  # agreed choice letter/NONE, or "" when none
    edge_set: frozenset
    routing: str  # "auto_accept" | "human_review"
    n_votes: int
    n_valid: int
    minority: str  # summary of dissenting votes
    mean_confidence: float
    # Machine-readable code for WHY the group routed the way it did, stamped on
    # every row (codes enumerated in panel_routing: unanimous, quorum,
    # unanimous_none, quorum_none, dissent:<provider>=<choice>,
    # below_quorum:<n>, no_majority, all_abstained, class-mismatch, size_gated,
    # low_confidence; plus the legacy pre-v5 "abstention").
    route_reason: str = ""
    evidence_id: str = ""
    evidence_pack_sha256: str = ""
    displayed_candidate_universe_sha256: str = ""
    option_menu_sha256: str = ""
    chosen_option_id: str = ""
    panel_invocation_sha256: str = ""
    consensus_policy_sha256: str = ""


def compute_consensus(
    votes: list[Vote],
    edge_classes: list[tuple[str | None, str | None]] | None = None,
    n_candidate_edges: int | None = None,
    min_voter_confidence: float | None = None,
) -> Consensus:
    """Apply the quorum consensus rule over a group's votes.

    Abstentions do not count toward agreement. A group auto-accepts when ALL
    VALID (non-abstaining) votes agree AND at least 3 votes are valid
    (``agree == n_valid >= 3``); everything else routes to human review. Two
    accept tiers stay distinguishable end-to-end:

    * ``unanimous`` — every recorded vote is valid and agrees (e.g. 4/4);
    * ``quorum`` — all valid votes agree with >=1 abstention (e.g. 3-of-4
      with one abstain). Quorum forgives ABSTENTION only, never disagreement:
      any dissent among valid votes still routes to human review.

    For any 3-voter panel (v2/v3/v4 composition re-runs) this is
    routing-identical to the pre-v5 rule (``agree == len(votes) >= 3``): with 3
    voters, all-valid agreement at quorum IS full unanimity, and any abstention
    drops ``n_valid`` below 3 (``below_quorum``). The regression sweep
    ``test_quorum_rule_is_noop_for_3voter_panels`` proves the equivalence over
    every 3-vote combination. A NONE verdict never auto-accepts on either tier;
    an all-valid-NONE at quorum is stamped ``quorum_none`` (the quorum analog
    of ``unanimous_none``) and remains in human review. NONE is overloaded (all
    edges wrong, no exact offered option, or insufficient evidence), so only an
    explicitly human-confirmed empty selection can become reject-all truth.

    Size gate: when ``n_candidate_edges`` (the group's candidate-edge count) is
    supplied and exceeds the export backstop
    (``settings.stitch_export_backstop_max_edges``), an otherwise-auto-accept
    verdict is demoted to ``human_review`` with ``route_reason="size_gated"``.
    No verdict on such a group can ever mint a label — ``stitch_export``
    enforces the backstop on the candidate count on both its structural gate
    and its legacy no-structure-fields fallback — so letting it auto-accept
    would make it vanish: not in the human queue, not exported, reviewed by
    no one. Non-auto-accept outcomes
    already route to a human and keep their (more specific) reason, so every
    over-backstop group ends up ``human_review`` regardless of vote outcome.
    ``None`` disables the gate (callers without size metadata get the pre-gate
    behavior).

    Class-consistency gate: when ``edge_classes`` (the (ref_class, target_class)
    pairs of the *chosen* edge set) is supplied, an otherwise-auto-accept
    verdict whose chosen edges include a cross-mode pedestrian↔vehicular edge is
    demoted to ``human_review`` with ``route_reason="class-mismatch"``. Passing
    no ``edge_classes`` disables the gate (callers without class metadata get
    the pre-gate behavior). The size gate runs first: when both would demote,
    the structural export block is the decisive fact and its reason wins.

    Low-confidence gate: when ``min_voter_confidence`` is a positive floor, an
    otherwise-auto-accept verdict whose MINIMUM confidence across valid votes is
    below it is demoted to ``human_review`` with ``route_reason="low_confidence"``.
    The two Gemini-based voters report near-constant inflated confidence, so the
    minimum is effectively the calibrated voter's self-report — a review found low
    values there flagged wrong unanimous verdicts. A blank/NaN confidence on a
    valid vote counts as BELOW the floor (a missing self-report must not silently
    pass). Runs AFTER the size and class gates, so their more-structural reasons
    win when several would demote. ``None`` or a non-positive floor disables the
    gate (callers without the setting get the pre-gate behavior).
    """
    group_id = votes[0].group_id if votes else ""
    valid = [v for v in votes if v.choice != "ABSTAIN"]
    n_valid = len(valid)

    # Tally by choice letter.
    tally: dict[str, list[Vote]] = {}
    for v in valid:
        tally.setdefault(v.choice, []).append(v)

    if not tally:
        return Consensus(
            group_id,
            "none",
            "",
            frozenset(),
            "human_review",
            len(votes),
            0,
            "all providers abstained",
            0.0,
            route_reason=REASON_ALL_ABSTAINED,
        )

    top_choice = max(tally, key=lambda c: len(tally[c]))
    top_votes = tally[top_choice]
    agree = len(top_votes)

    minority_votes = [v for v in valid if v.choice != top_choice]
    minority = "; ".join(f"{v.provider}={v.choice}" for v in minority_votes)
    mean_conf = round(sum(v.confidence for v in top_votes) / len(top_votes), 3)
    edge_set = top_votes[0].edge_set

    # Quorum rule (v5): auto-accept when ALL valid votes agree and >=3 are
    # valid. Full unanimity (no abstentions) keeps the "unanimous" tier; an
    # all-valid agreement over >=1 abstention is the distinct "quorum" tier so
    # a 4/4 accept and a 3-of-4 accept stay distinguishable in consensus.csv
    # and in the export labelers. agree == n_valid implies no dissenting valid
    # vote, so quorum forgives abstention only — never disagreement. For any
    # 3-voter panel this routes byte-identically to the pre-v5
    # agree == len(votes) rule (see the docstring and the sweep test).
    route_reason = ""
    if agree == n_valid and n_valid >= 3:
        consensus = "unanimous" if n_valid == len(votes) else "quorum"
        routing = "auto_accept" if top_choice != "NONE" else "human_review"
    elif agree >= 2:
        consensus = "majority"
        routing = "human_review"
    else:
        consensus = "none"
        routing = "human_review"

    # Size gate: demote an auto-accept on a group whose candidate-edge count
    # exceeds the export backstop — its verdict can never export, so it must
    # land in the human queue instead of vanishing. Runs before the class gate
    # so the structural export block's reason wins when both would demote.
    if (
        routing == "auto_accept"
        and n_candidate_edges is not None
        and n_candidate_edges > settings.stitch_export_backstop_max_edges
    ):
        routing = "human_review"
        route_reason = REASON_SIZE_GATED

    # Class-consistency gate: demote an auto-accept whose chosen edge set
    # contains a cross-mode pedestrian↔vehicular edge. An empty list means
    # "supplied, nothing to gate on" — only None disables the gate.
    if routing == "auto_accept" and edge_classes is not None and has_cross_mode_edge(edge_classes):
        routing = "human_review"
        route_reason = REASON_CLASS_MISMATCH

    # Low-confidence gate: demote an auto-accept whose calibrated-voter
    # confidence is too low. The Gemini-based voters report near-constant
    # inflated confidence, so min(valid) is effectively the calibrated voter's
    # self-report — a review found low values there flagged wrong verdicts. A
    # blank/NaN confidence counts as below the floor (NaN must not silently
    # pass). Runs after the size and class gates so their reasons win.
    if routing == "auto_accept" and min_voter_confidence is not None and min_voter_confidence > 0.0:
        valid_confs = [v.confidence for v in valid]
        if any(math.isnan(c) for c in valid_confs) or min(valid_confs) < min_voter_confidence:
            routing = "human_review"
            route_reason = REASON_LOW_CONFIDENCE

    # Stamp a reason on EVERY row (not just gate demotions) so consensus.csv
    # says why each group routed the way it did. The derivation is shared with
    # the historical-row reader (panel_routing.derive_route_reason), so the
    # stamp and the derived-for-history codes can never diverge.
    if not route_reason:
        route_reason = derive_route_reason(
            {
                "consensus": consensus,
                "choice": top_choice,
                "routing": routing,
                "minority": minority,
                "n_votes": len(votes),
                "n_valid": n_valid,
            }
        )

    return Consensus(
        group_id=group_id,
        consensus=consensus,
        choice=top_choice,
        edge_set=edge_set,
        routing=routing,
        n_votes=len(votes),
        n_valid=n_valid,
        minority=minority,
        mean_confidence=mean_conf,
        route_reason=route_reason,
    )


# ---------------------------------------------------------------------------
# Batch driver + persistence
# ---------------------------------------------------------------------------

VOTES_COLUMNS = [
    "group_id",
    "provider",
    "model",
    "choice",
    "confidence",
    "reasoning",
    "edge_set",
    "latency_s",
    "timestamp",
    "error",
    "invocation_route",
    "evidence_delivery",
    "pack_feedback",
    "evidence_id",
    "evidence_pack_sha256",
    "displayed_candidate_universe_sha256",
    "option_menu_sha256",
    "chosen_option_id",
    "panel_invocation_sha256",
]
CONSENSUS_COLUMNS = [
    "group_id",
    "consensus",
    "choice",
    "edge_set",
    "routing",
    "n_votes",
    "n_valid",
    "minority",
    "mean_confidence",
    "route_reason",
    "evidence_id",
    "evidence_pack_sha256",
    "displayed_candidate_universe_sha256",
    "option_menu_sha256",
    "chosen_option_id",
    "panel_invocation_sha256",
    "consensus_policy_sha256",
]


def _edge_set_str(es: frozenset) -> str:
    return json.dumps(sorted([list(e) for e in es]))


def _vote_row(v: Vote) -> dict:
    return {
        "group_id": v.group_id,
        "provider": v.provider,
        "model": v.model,
        "choice": v.choice,
        "confidence": v.confidence,
        "reasoning": v.reasoning,
        "edge_set": _edge_set_str(v.edge_set),
        "latency_s": v.latency_s,
        "timestamp": v.timestamp,
        "error": v.error,
        "invocation_route": v.invocation_route,
        "evidence_delivery": v.evidence_delivery,
        "pack_feedback": v.pack_feedback,
        "evidence_id": v.evidence_id,
        "evidence_pack_sha256": v.evidence_pack_sha256,
        "displayed_candidate_universe_sha256": v.displayed_candidate_universe_sha256,
        "option_menu_sha256": v.option_menu_sha256,
        "chosen_option_id": v.chosen_option_id,
        "panel_invocation_sha256": v.panel_invocation_sha256,
    }


def _consensus_row(c: Consensus) -> dict:
    return {
        "group_id": c.group_id,
        "consensus": c.consensus,
        "choice": c.choice,
        "edge_set": _edge_set_str(c.edge_set),
        "routing": c.routing,
        "n_votes": c.n_votes,
        "n_valid": c.n_valid,
        "minority": c.minority,
        "mean_confidence": c.mean_confidence,
        "route_reason": c.route_reason,
        "evidence_id": c.evidence_id,
        "evidence_pack_sha256": c.evidence_pack_sha256,
        "displayed_candidate_universe_sha256": c.displayed_candidate_universe_sha256,
        "option_menu_sha256": c.option_menu_sha256,
        "chosen_option_id": c.chosen_option_id,
        "panel_invocation_sha256": c.panel_invocation_sha256,
        "consensus_policy_sha256": c.consensus_policy_sha256,
    }


def run_batch(
    batch_dir: Path,
    panel: list[ProviderSpec] | None = None,
    group_ids: list[str] | None = None,
    timeout: int | None = None,
    limit: int = 0,
    collect_feedback: bool = False,
    resume: bool = False,
    invocation_budget_s: float = 300.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the panel over every generated group in a batch dir.

    Expects evidence packs under ``batch_dir/{group_id}/``. Writes
    ``batch_dir/votes.csv`` and ``batch_dir/consensus.csv``. Returns the two
    DataFrames. ``timeout=None`` resolves per provider (spec timeout, else
    :data:`DEFAULT_VOTE_TIMEOUT_S`); an explicit value overrides both.

    Resumable per-group driver: rows are flushed to ``votes.partial.csv`` /
    ``consensus.partial.csv`` after EACH group so an interrupted run (timeout,
    provider cap, crash) loses at most the in-flight group. With ``resume=True``
    the driver reloads those partials and skips any group already recorded,
    resuming from where it stopped. This is panel-size agnostic — it works
    identically for the 4-voter v5 default and the 3-voter era panels.

    Resume safety: partial rows are only reused when the recorded provider set
    matches the CURRENT panel — resuming a ``v3-candidate`` run over partials
    written by the 3-voter default would otherwise silently return cached
    3-voter votes with the 4th voter never invoked. On a panel mismatch the
    partials are ignored and every group re-runs. Carried-forward groups are
    also restricted to the current ``group_ids``/``limit`` selection so a
    filtered resume never leaks unrequested groups into the final output.
    """
    batch_dir = Path(batch_dir)
    panel = panel or DEFAULT_PANEL
    votes_partial = batch_dir / "votes.partial.csv"
    consensus_partial = batch_dir / "consensus.partial.csv"

    group_dirs = sorted(
        d for d in batch_dir.iterdir() if d.is_dir() and (d / "prompt.txt").exists()
    )
    allow_legacy_evidence = True
    batch: dict | None = None
    batch_path = batch_dir / "batch.json"
    if batch_path.exists():
        batch = json.loads(batch_path.read_text())
        if int(batch.get("schema_version") or 0) >= 2:
            allow_legacy_evidence = False
            roster = set(batch_group_map(batch))
            stale = sorted(d.name for d in group_dirs if d.name not in roster)
            if stale:
                raise ValueError(
                    "evidence packs are outside the current schema-v2 batch roster: "
                    f"{stale}; regenerate or remove the stale packs"
                )
            group_dirs = [d for d in group_dirs if d.name in roster]
    if group_ids:
        wanted = set(group_ids)
        group_dirs = [d for d in group_dirs if d.name in wanted]
    if limit > 0:
        group_dirs = group_dirs[:limit]
    selected_ids = {d.name for d in group_dirs}
    group_dirs_by_id = {d.name: d for d in group_dirs}
    evidence_by_id = {
        d.name: load_evidence_manifest(d, allow_legacy=allow_legacy_evidence) for d in group_dirs
    }
    for group_id, manifest in evidence_by_id.items():
        validate_manifest_against_batch(manifest, batch or {}, group_id)
    panel_invocation_sha = invocation_signature(
        panel,
        timeout=timeout,
        collect_feedback=collect_feedback,
        invocation_budget_s=invocation_budget_s,
        effective_timeouts=[resolve_timeout(provider, timeout) for provider in panel],
        runtime_contract_sha256=sha256_file(Path(__file__)),
    )
    policy_sha = consensus_policy_signature(
        max_edges=settings.stitch_export_backstop_max_edges,
        min_voter_confidence=settings.stitch_min_voter_confidence,
        runtime_contract_sha256=sha256_file(Path(__file__)),
    )

    # Resume: carry forward already-completed groups from the partial files and
    # skip re-running them. A group only counts as done when (a) it is present
    # in BOTH partials, (b) it is in the current selection, and (c) the partials
    # were written by the SAME panel (provider-set match).
    done_ids: set[str] = set()
    vote_rows: list[dict] = []
    consensus_out: list[dict] = []
    # Rows for previously-done groups OUTSIDE the current selection: excluded
    # from the final output but preserved in the partials, so a filtered resume
    # never destroys another group's crash-recovery data.
    unselected_votes: list[dict] = []
    unselected_cons: list[dict] = []
    if resume and votes_partial.exists() and consensus_partial.exists():
        prev_votes = pd.read_csv(votes_partial, dtype={"group_id": str})
        prev_cons = pd.read_csv(consensus_partial, dtype={"group_id": str})
        recorded = set(prev_votes["group_id"].astype(str)) & set(prev_cons["group_id"].astype(str))
        expected_voters = {(p.name, p.model) for p in panel}
        for gid in sorted(recorded & selected_ids):
            vote_group = prev_votes[prev_votes["group_id"].astype(str) == gid]
            cons_group = prev_cons[prev_cons["group_id"].astype(str) == gid]
            manifest = evidence_by_id[gid]
            evidence = manifest["evidence"]
            actual_voters = set(
                zip(
                    vote_group.get("provider", pd.Series(dtype=str)).astype(str),
                    vote_group.get("model", pd.Series(dtype=str)).astype(str),
                    strict=False,
                )
            )
            complete = (
                len(vote_group) == len(panel)
                and not vote_group.duplicated(subset=["provider"]).any()
                and actual_voters == expected_voters
                and len(cons_group) == 1
            )
            gemini_rows = (
                vote_group[vote_group["provider"].astype(str) == "gemini"]
                if "provider" in vote_group
                else vote_group.iloc[0:0]
            )
            if not gemini_rows.empty:
                gemini_spec = next((p for p in panel if p.name == "gemini"), None)
                allowed_routes = set(gemini_spec.routes) if gemini_spec is not None else set()
                recorded_routes = (
                    set(gemini_rows["invocation_route"].fillna("").astype(str))
                    if "invocation_route" in gemini_rows
                    else set()
                )
                complete = (
                    complete
                    and "invocation_route" in gemini_rows
                    and bool(allowed_routes)
                    and recorded_routes <= allowed_routes
                    and "" not in recorded_routes
                )

            def _all_equal(frame: pd.DataFrame, column: str, expected: str) -> bool:
                return (
                    column in frame
                    and len(frame) > 0
                    and set(frame[column].astype(str)) == {expected}
                )

            option_ids = {
                str(option["letter"]): str(option["option_id"])
                for option in evidence["option_menu"]
            }
            option_edges = {
                str(option["letter"]): frozenset(
                    (str(edge["ref_id"]), str(edge["target_id"])) for edge in option["edges"]
                )
                for option in evidence["option_menu"]
            }
            option_ids.update({"NONE": "NONE", "ABSTAIN": "ABSTAIN", "": ""})
            option_edges.update({"NONE": frozenset(), "ABSTAIN": frozenset(), "": frozenset()})

            def _stored_edges(value: object) -> frozenset[tuple[str, str]] | None:
                try:
                    parsed = json.loads(str(value))
                    return frozenset((str(edge[0]), str(edge[1])) for edge in parsed)
                except (json.JSONDecodeError, TypeError, IndexError):
                    return None

            def _text(value: object) -> str:
                return "" if pd.isna(value) else str(value)

            def _rows_match_menu(
                frame: pd.DataFrame,
                option_ids: dict[str, str] = option_ids,
                option_edges: dict[str, frozenset[tuple[str, str]]] = option_edges,
            ) -> bool:
                required = {"choice", "chosen_option_id", "edge_set"}
                if not required <= set(frame.columns):
                    return False
                for row in frame.to_dict("records"):
                    choice = _text(row.get("choice"))
                    if choice not in option_ids:
                        return False
                    if _text(row.get("chosen_option_id")) != option_ids[choice]:
                        return False
                    if _stored_edges(row.get("edge_set")) != option_edges[choice]:
                        return False
                return True

            def _rows_match_delivery(
                frame: pd.DataFrame,
                manifest: dict = manifest,
            ) -> bool:
                if "evidence_delivery" not in frame.columns:
                    return False
                for row in frame.to_dict("records"):
                    provider_name = _text(row.get("provider"))
                    route = _text(row.get("invocation_route"))
                    try:
                        delivery_mode, transport = _delivery_mode_transport(provider_name, route)
                        validate_evidence_delivery_record(
                            _text(row.get("evidence_delivery")),
                            manifest,
                            expected_delivery_mode=delivery_mode,
                            expected_transport=transport,
                        )
                    except (EvidenceProvenanceError, TypeError, ValueError):
                        return False
                return True

            compatible = complete and all(
                (
                    _all_equal(vote_group, "evidence_id", evidence["evidence_id"]),
                    _all_equal(
                        vote_group,
                        "evidence_pack_sha256",
                        manifest["evidence_pack_sha256"],
                    ),
                    _all_equal(
                        vote_group,
                        "panel_invocation_sha256",
                        panel_invocation_sha,
                    ),
                    _all_equal(
                        vote_group,
                        "displayed_candidate_universe_sha256",
                        evidence["displayed_candidate_universe_sha256"],
                    ),
                    _all_equal(vote_group, "option_menu_sha256", evidence["option_menu_sha256"]),
                    _rows_match_menu(vote_group),
                    _rows_match_delivery(vote_group),
                    _all_equal(cons_group, "evidence_id", evidence["evidence_id"]),
                    _all_equal(
                        cons_group,
                        "evidence_pack_sha256",
                        manifest["evidence_pack_sha256"],
                    ),
                    _all_equal(
                        cons_group,
                        "panel_invocation_sha256",
                        panel_invocation_sha,
                    ),
                    _all_equal(
                        cons_group,
                        "displayed_candidate_universe_sha256",
                        evidence["displayed_candidate_universe_sha256"],
                    ),
                    _all_equal(cons_group, "option_menu_sha256", evidence["option_menu_sha256"]),
                    _rows_match_menu(cons_group),
                    _all_equal(cons_group, "consensus_policy_sha256", policy_sha),
                )
            )
            if compatible:
                replay_votes = [
                    Vote(
                        group_id=gid,
                        provider=str(row["provider"]),
                        model=str(row["model"]),
                        choice=_text(row["choice"]),
                        confidence=float(row["confidence"]),
                        reasoning=_text(row.get("reasoning")),
                        edge_set=option_edges[_text(row["choice"])],
                        invocation_route=_text(row.get("invocation_route")),
                    )
                    for row in vote_group.to_dict("records")
                ]
                meta = _load_group_context(group_dirs_by_id[gid])[2]
                ref_class, tgt_class = _segment_class_maps(meta)
                base = compute_consensus(replay_votes)
                replay = compute_consensus(
                    replay_votes,
                    edge_classes=_edge_classes_for(base.edge_set, ref_class, tgt_class),
                    n_candidate_edges=candidate_edge_count(meta),
                    min_voter_confidence=settings.stitch_min_voter_confidence,
                )
                stored = cons_group.iloc[0]
                compatible = all(
                    (
                        str(stored["consensus"]) == replay.consensus,
                        _text(stored.get("choice")) == replay.choice,
                        _stored_edges(stored["edge_set"]) == replay.edge_set,
                        str(stored["routing"]) == replay.routing,
                        int(stored["n_votes"]) == replay.n_votes,
                        int(stored["n_valid"]) == replay.n_valid,
                        _text(stored.get("minority")) == replay.minority,
                        math.isclose(
                            float(stored["mean_confidence"]),
                            replay.mean_confidence,
                            abs_tol=1e-12,
                        ),
                        _text(stored.get("route_reason")) == replay.route_reason,
                    )
                )
            if compatible:
                done_ids.add(gid)
            else:
                logger.warning(
                    f"resume: group {gid} has incomplete or stale panel/evidence/policy "
                    f"provenance; re-running it"
                )

        vote_rows = prev_votes[prev_votes["group_id"].astype(str).isin(done_ids)].to_dict("records")
        consensus_out = prev_cons[prev_cons["group_id"].astype(str).isin(done_ids)].to_dict(
            "records"
        )
        carry = recorded - selected_ids
        unselected_votes = prev_votes[prev_votes["group_id"].astype(str).isin(carry)].to_dict(
            "records"
        )
        unselected_cons = prev_cons[prev_cons["group_id"].astype(str).isin(carry)].to_dict(
            "records"
        )
        if done_ids:
            logger.info(f"resume: skipping {len(done_ids)} already-completed groups")

    def _flush() -> None:
        pd.DataFrame(unselected_votes + vote_rows, columns=VOTES_COLUMNS).to_csv(
            votes_partial, index=False
        )
        pd.DataFrame(unselected_cons + consensus_out, columns=CONSENSUS_COLUMNS).to_csv(
            consensus_partial, index=False
        )

    pending = [d for d in group_dirs if d.name not in done_ids]
    # Circuit breaker: a provider whose votes are timeout-abstentions on
    # _TIMEOUT_BREAKER_N consecutive groups is treated as genuinely hung
    # (network blackhole with no fast error) and promotes to the #334
    # provider-down halt. Per-group timeout abstains keep a wave alive when one
    # oversized group is slow; a hang on EVERY group is provider health, and
    # letting it degrade the panel silently is exactly what #334 forbids.
    # BOTH timeout flavors count (AbstainReason.TIMEOUT): the subprocess-kill
    # timeout AND agy's CLI-internal response timeout (#343). The latter is the
    # common case for a network-blackholed agy — its own --print-timeout fires
    # first with a clean error — so counting only the subprocess flavor would
    # leave the breaker unreachable for the very provider it was built for.
    # Context-overflow and parse abstains do NOT count: overflow is a property of
    # the group's prompt size and parse failures a property of the response, and
    # several in a row say nothing about the provider. Any successful vote (or a
    # non-timeout abstain) resets the count.
    consecutive_timeouts: dict[str, int] = {}
    # One state object spans the wave, so a quota-capped/broken primary route
    # is tried once and then bypassed for every later group. It is deliberately
    # process-local: a fresh --resume run probes the primary route again in case
    # quota or infrastructure recovered.
    route_state = ProviderRouteState()
    for i, gdir in enumerate(pending):
        gid = gdir.name
        logger.info(f"[{i + 1}/{len(pending)}] panel on group {gid}")
        votes = run_panel_on_group(
            gid,
            gdir,
            panel,
            timeout,
            collect_feedback=collect_feedback,
            invocation_budget_s=invocation_budget_s,
            route_state=route_state,
            evidence_manifest=evidence_by_id[gid],
        )
        manifest = evidence_by_id[gid]
        evidence = manifest["evidence"]
        option_ids = {
            str(option["letter"]): str(option["option_id"]) for option in evidence["option_menu"]
        }
        option_ids.update({"NONE": "NONE", "ABSTAIN": "ABSTAIN"})
        for vote in votes:
            vote.evidence_id = evidence["evidence_id"]
            vote.evidence_pack_sha256 = manifest["evidence_pack_sha256"]
            vote.displayed_candidate_universe_sha256 = evidence[
                "displayed_candidate_universe_sha256"
            ]
            vote.option_menu_sha256 = evidence["option_menu_sha256"]
            vote.chosen_option_id = option_ids[vote.choice]
            vote.panel_invocation_sha256 = panel_invocation_sha
        for v in votes:
            if v.choice == "ABSTAIN" and v.abstain_reason == AbstainReason.TIMEOUT:
                consecutive_timeouts[v.provider] = consecutive_timeouts.get(v.provider, 0) + 1
                if consecutive_timeouts[v.provider] >= _TIMEOUT_BREAKER_N:
                    _flush()  # keep completed groups resumable past the halt
                    raise ProviderInvocationError(
                        f"{v.provider}: timed out on {_TIMEOUT_BREAKER_N} consecutive "
                        f"groups (last: {gid}) — treating as provider-down and halting "
                        f"the run. Completed groups were flushed; fix the provider and "
                        f"re-run with --resume."
                    )
            else:
                consecutive_timeouts[v.provider] = 0
        vote_rows.extend(_vote_row(v) for v in votes)
        # Derive the chosen edge set's classes so the class-consistency gate can
        # demote cross-mode auto-accepts. compute_consensus is pure, so a first
        # (gate-less) call gives the chosen edge_set to look up classes for.
        # The pack metadata also carries the group's candidate-edge count for
        # the size gate (an over-backstop group's verdict can never export, so
        # it must route to a human instead of vanishing).
        meta = _load_group_context(gdir)[2]
        ref_class, tgt_class = _segment_class_maps(meta)
        base = compute_consensus(votes)
        edge_classes = _edge_classes_for(base.edge_set, ref_class, tgt_class)
        cons = compute_consensus(
            votes,
            edge_classes=edge_classes,
            n_candidate_edges=candidate_edge_count(meta),
            min_voter_confidence=settings.stitch_min_voter_confidence,
        )
        cons.evidence_id = evidence["evidence_id"]
        cons.evidence_pack_sha256 = manifest["evidence_pack_sha256"]
        cons.displayed_candidate_universe_sha256 = evidence["displayed_candidate_universe_sha256"]
        cons.option_menu_sha256 = evidence["option_menu_sha256"]
        cons.chosen_option_id = option_ids.get(cons.choice, "")
        cons.panel_invocation_sha256 = panel_invocation_sha
        cons.consensus_policy_sha256 = policy_sha
        consensus_out.append(_consensus_row(cons))
        logger.info(
            f"  -> {cons.consensus} choice={cons.choice} routing={cons.routing} "
            f"({'/'.join(v.provider + ':' + v.choice for v in votes)})"
        )
        _flush()  # persist after each group so an interrupted run is resumable

    votes_df = pd.DataFrame(vote_rows, columns=VOTES_COLUMNS)
    consensus_df = pd.DataFrame(consensus_out, columns=CONSENSUS_COLUMNS)

    # Per-voter bias monitoring: make a position-anchored voter LOUD within its own
    # wave (a lower n-floor than the aggregate offline monitor). This does NOT touch
    # the breaker or consensus semantics — it only inspects the completed rows.
    for _warning in wave_position_anchor_warnings(votes_df, consensus_df):
        logger.warning(f"panel bias: {_warning}")

    votes_df.to_csv(batch_dir / "votes.csv", index=False)
    consensus_df.to_csv(batch_dir / "consensus.csv", index=False)
    return votes_df, consensus_df
