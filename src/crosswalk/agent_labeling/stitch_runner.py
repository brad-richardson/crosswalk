"""Consensus-panel runner for agent stitching-group labeling.

Runs a heterogeneous 3-provider panel (claude + codex + agy) on each group's
evidence pack, in parallel. Each provider returns a JSON option pick; votes are
validated (choice must be a real option letter or NONE), retried once on
garbage, and recorded as audit data. A consensus rule routes each group.

Votes are audit data and are stored under the batch dir (``votes.csv``),
deliberately separate from ``labels/``. This module writes NOTHING into
``labels/stitching/`` — export policy is decided after the validation gate.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from .panel_routing import REASON_ALL_ABSTAINED, REASON_CLASS_MISMATCH, derive_route_reason

# ---------------------------------------------------------------------------
# Provider panel configuration
# ---------------------------------------------------------------------------


@dataclass
class ProviderSpec:
    """A panel member: how to invoke a provider CLI on an evidence pack."""

    name: str  # short id used in votes.csv (e.g. "claude")
    model: str  # model string recorded in votes
    effort: str = ""  # reasoning/thinking effort where the CLI supports it


# Panel v2 (composition change from v1 -> bump the export labeler; the labeler
# was bumped again to panel_unanimous_v3 when the #302 pack enrichment changed
# the panel's inputs). Effort is CLI-specific: claude takes --effort; codex takes
# model_reasoning_effort; agy encodes it in the model name ("... (Medium)").
DEFAULT_PANEL = [
    ProviderSpec(name="claude", model="claude-opus-4-8", effort="medium"),
    ProviderSpec(name="codex", model="gpt-5.5", effort="low"),
    ProviderSpec(name="agy", model="Gemini 3.5 Flash (Medium)"),
]

# A candidate FOURTH voter (default OFF): opencode driving an OpenRouter-hosted
# Qwen3-VL model. Deliberately a distinct model family from the three incumbents
# (Claude / GPT / Gemini) so its vote is decorrelated, adding real signal to the
# quorum rather than echoing an existing voice. opencode carries all knobs
# (reasoning etc.) in the model string, so ``effort`` is unused for it — like agy.
OPENCODE_QWEN = ProviderSpec(name="opencode", model="openrouter/qwen/qwen3-vl-235b-a22b-instruct")

# Named panel configurations. DEFAULT_PANEL (the 3-voter production panel) is the
# default; the 4th voter ships behind the opt-in ``v3-candidate`` panel only, so
# production waves are unaffected until the export rule is validated and flipped.
PANELS: dict[str, list[ProviderSpec]] = {
    "default": DEFAULT_PANEL,
    "v2": DEFAULT_PANEL,
    "v3-candidate": [*DEFAULT_PANEL, OPENCODE_QWEN],
}


def get_panel(name: str | None) -> list[ProviderSpec]:
    """Resolve a named panel config; unknown/empty names fall back to DEFAULT_PANEL."""
    if not name:
        return DEFAULT_PANEL
    return PANELS.get(name, DEFAULT_PANEL)


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
    pack_feedback: str = ""  # diagnostic self-report JSON (wave-local; usually "")


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


def _check_exit(provider: str, result: subprocess.CompletedProcess) -> None:
    """Raise on non-zero CLI exit so failures don't masquerade as parse errors.

    Includes truncated stderr in the message; the runner records it in the
    abstention error trail.
    """
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:500]
        raise RuntimeError(f"{provider} exited with code {result.returncode}: {stderr}")


class ProviderInvocationError(RuntimeError):
    """A voter's CLI/API failed and did not recover within the retry budget.

    Raised for invocation/API failures (nonzero exit, quota exhaustion,
    rate-limit, network, timeout) that persist through the backoff window.
    It propagates out of ``run_batch`` to HALT the run rather than silently
    degrading the panel to fewer voters — a quota-exhausted provider abstaining
    would quietly weaken quorum on every subsequent group. Parse/validation
    failures do NOT raise this (a single malformed response still abstains);
    only genuine provider-down conditions do. The run is resumable: completed
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
        cmd += ["-o", str(out_path), prompt]
        result = subprocess.run(
            cmd,
            input="",
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


def invoke_agy(
    prompt: str,
    group_dir: Path,
    letters: list[str],
    model: str,
    timeout: int = 240,
    effort: str = "",
) -> str:
    """Invoke the agy (Antigravity) CLI. =-form flags only; reads images by path.

    Effort is encoded in the model name (e.g. "Gemini 3.5 Flash (Medium)"), so
    the ``effort`` argument is accepted for a uniform invoker signature but unused.
    """
    cmd = [
        "agy",
        "--print-timeout=2m",
        f"--model={model}",
        "-p",
        prompt,
    ]
    result = subprocess.run(
        cmd,
        input="",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    _check_exit("agy", result)
    return result.stdout


def invoke_opencode(
    prompt: str,
    group_dir: Path,
    letters: list[str],
    model: str,
    timeout: int = 240,
    effort: str = "",
) -> str:
    """Invoke the opencode CLI (OpenRouter-backed). Reads images by path via -f.

    CRITICAL ordering: the prompt MUST be the positional argument immediately
    after ``run`` and BEFORE any ``-f`` flags. ``-f`` takes an array of file
    paths, so a prompt placed after it is swallowed as another filename and the
    model receives no instruction. Multiple ``-f`` flags attach multiple images.

    The model string carries everything opencode needs (provider/model/knobs),
    so ``effort`` is accepted for a uniform invoker signature but unused. opencode
    prints the assistant's answer to stdout (TUI framing goes to stderr), so the
    raw stdout is returned for JSON extraction.
    """
    imgs = _image_paths(group_dir, letters)
    cmd = ["opencode", "run", prompt, "-m", model]
    for img in imgs:
        cmd += ["-f", img]
    result = subprocess.run(
        cmd,
        input="",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    _check_exit("opencode", result)
    return result.stdout


_INVOKERS = {
    "claude": invoke_claude,
    "codex": invoke_codex,
    "agy": invoke_agy,
    "opencode": invoke_opencode,
}


# ---------------------------------------------------------------------------
# Per-group panel run
# ---------------------------------------------------------------------------


def _scratch_pack(group_dir: Path, prompt: str) -> tuple[Path, str, tempfile.TemporaryDirectory]:
    """Copy a pack's images into an isolated scratch dir and rewrite the prompt.

    Some providers (agy) write derived crops next to the images they inspect.
    Giving each invocation an isolated copy keeps the canonical evidence dir
    pristine and bounds any provider scratch to an auto-cleaned temp dir. The
    prompt's absolute canonical paths are rewritten to the scratch dir.
    """
    import shutil

    tmp = tempfile.TemporaryDirectory(prefix="stitch_pack_")
    scratch = Path(tmp.name)
    for img in group_dir.glob("*.png"):
        shutil.copy2(img, scratch / img.name)
    canonical = str(group_dir.resolve())
    rewritten = prompt.replace(canonical, str(scratch))
    return scratch, rewritten, tmp


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


def run_provider_on_group(
    provider: ProviderSpec,
    group_id: str,
    group_dir: Path | None,
    prompt: str,
    letters: list[str],
    options_by_letter: dict[str, list[tuple[str, str]]],
    timeout: int = 240,
    retries: int = 1,
    collect_feedback: bool = False,
    invocation_budget_s: float = 300.0,
) -> Vote:
    """Run one provider on one group; abstain on bad output, hard-fail if down.

    Parse/validation failures retry ``retries`` times then abstain. Invocation/
    API failures (nonzero exit, quota, rate-limit, timeout) back off and retry
    within ``invocation_budget_s``, then raise :class:`ProviderInvocationError`.

    Each invocation runs against an isolated scratch copy of the pack so a
    provider that writes derived files (e.g. agy crops) never pollutes the
    canonical evidence dir. Skipped when group_dir is None (unit tests).
    """
    invoker = _INVOKERS[provider.name]
    valid = set(letters)

    scratch_dir = group_dir
    run_prompt = prompt
    tmp_ctx = None
    if group_dir is not None:
        scratch_dir, run_prompt, tmp_ctx = _scratch_pack(group_dir, prompt)

    try:
        return _attempt_provider(
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
        )
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
) -> Vote:
    """Run one provider, distinguishing two failure classes with opposite fates:

    * **Invocation/API failure** (nonzero exit, quota, rate-limit, network,
      timeout) — back off (exponential, capped) and retry within
      ``invocation_budget_s`` (default 5 min) so a transient rate-limit
      self-heals. If the budget is exhausted the provider is treated as down and
      we raise :class:`ProviderInvocationError` to HALT the run rather than
      silently degrade the panel to fewer voters.
    * **Parse/validation failure** (malformed output) — retry up to ``retries``
      times, then ABSTAIN. A single bad response should not kill a whole sweep.
    """
    last_parse_err = ""
    last_raw = ""
    parse_attempts = 0
    deadline = time.monotonic() + invocation_budget_s
    backoff = 5.0
    while True:
        start = time.monotonic()
        try:
            raw = invoker(prompt, group_dir, letters, provider.model, timeout, provider.effort)
        except Exception as e:  # noqa: BLE001 - invocation/API failure (incl. timeout)
            err = (
                f"timeout after {timeout}s"
                if isinstance(e, subprocess.TimeoutExpired)
                else f"invocation error: {e}"
            )
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
            )
        except ValueError as e:
            parse_attempts += 1
            last_parse_err = f"parse/validation: {e}"
            logger.warning(f"{provider.name} group {group_id} parse attempt {parse_attempts}: {e}")
            if parse_attempts > retries:
                break  # exhausted parse retries -> abstain (keep the panel going)

    # Parse retries exhausted -> abstention.
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
        raw=last_raw[:2000],
        error=last_parse_err,
    )


def run_panel_on_group(
    group_id: str,
    group_dir: Path,
    panel: list[ProviderSpec],
    timeout: int = 240,
    collect_feedback: bool = False,
) -> list[Vote]:
    """Run the full panel on one group in parallel (one thread per provider)."""
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
    consensus: str  # "unanimous" | "majority" | "none"
    choice: str  # agreed choice letter/NONE, or "" when none
    edge_set: frozenset
    routing: str  # "auto_accept" | "human_review"
    n_votes: int
    n_valid: int
    minority: str  # summary of dissenting votes
    mean_confidence: float
    # Machine-readable code for WHY the group routed the way it did, stamped on
    # every row (codes enumerated in panel_routing: unanimous, unanimous_none,
    # dissent:<provider>=<choice>, below_quorum:<n>, abstention, no_majority,
    # all_abstained, class-mismatch).
    route_reason: str = ""


def compute_consensus(
    votes: list[Vote],
    edge_classes: list[tuple[str | None, str | None]] | None = None,
) -> Consensus:
    """Apply the 3/3, 2/3, else routing rule over a group's votes.

    Abstentions do not count toward agreement. Only unanimous agreement among
    all (>=3) valid votes auto-accepts; everything else routes to human review.

    Class-consistency gate: when ``edge_classes`` (the (ref_class, target_class)
    pairs of the *chosen* edge set) is supplied, an otherwise-auto-accept
    verdict whose chosen edges include a cross-mode pedestrian↔vehicular edge is
    demoted to ``human_review`` with ``route_reason="class-mismatch"``. Passing
    no ``edge_classes`` disables the gate (callers without class metadata get
    the pre-gate behavior).
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

    # Unanimous requires all 3 panelists valid AND agreeing.
    route_reason = ""
    if agree == len(votes) and agree >= 3 and not minority_votes:
        consensus = "unanimous"
        routing = "auto_accept" if top_choice != "NONE" else "human_review"
    elif agree >= 2:
        consensus = "majority"
        routing = "human_review"
    else:
        consensus = "none"
        routing = "human_review"

    # Class-consistency gate: demote an auto-accept whose chosen edge set
    # contains a cross-mode pedestrian↔vehicular edge. An empty list means
    # "supplied, nothing to gate on" — only None disables the gate.
    if routing == "auto_accept" and edge_classes is not None and has_cross_mode_edge(edge_classes):
        routing = "human_review"
        route_reason = REASON_CLASS_MISMATCH

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
    "pack_feedback",
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
        "pack_feedback": v.pack_feedback,
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
    }


def run_batch(
    batch_dir: Path,
    panel: list[ProviderSpec] | None = None,
    group_ids: list[str] | None = None,
    timeout: int = 240,
    limit: int = 0,
    collect_feedback: bool = False,
    resume: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the panel over every generated group in a batch dir.

    Expects evidence packs under ``batch_dir/{group_id}/``. Writes
    ``batch_dir/votes.csv`` and ``batch_dir/consensus.csv``. Returns the two
    DataFrames.

    Resumable per-group driver: rows are flushed to ``votes.partial.csv`` /
    ``consensus.partial.csv`` after EACH group so an interrupted run (timeout,
    provider cap, crash) loses at most the in-flight group. With ``resume=True``
    the driver reloads those partials and skips any group already recorded,
    resuming from where it stopped. This is panel-size agnostic — it works
    identically for the 3-voter default and the 4-voter ``v3-candidate`` panel.

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
    if group_ids:
        wanted = set(group_ids)
        group_dirs = [d for d in group_dirs if d.name in wanted]
    if limit > 0:
        group_dirs = group_dirs[:limit]
    selected_ids = {d.name for d in group_dirs}

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
        panel_names = {p.name for p in panel}
        prev_names = set(prev_votes["provider"].astype(str).unique())
        if prev_names != panel_names:
            logger.warning(
                f"resume: partials were written by a different panel "
                f"({sorted(prev_names)} != {sorted(panel_names)}); ignoring them "
                f"and re-running all groups"
            )
        else:
            recorded = set(prev_votes["group_id"]) & set(prev_cons["group_id"])
            done_ids = recorded & selected_ids
            vote_rows = prev_votes[prev_votes["group_id"].isin(done_ids)].to_dict("records")
            consensus_out = prev_cons[prev_cons["group_id"].isin(done_ids)].to_dict("records")
            carry = recorded - selected_ids
            unselected_votes = prev_votes[prev_votes["group_id"].isin(carry)].to_dict("records")
            unselected_cons = prev_cons[prev_cons["group_id"].isin(carry)].to_dict("records")
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
    for i, gdir in enumerate(pending):
        gid = gdir.name
        logger.info(f"[{i + 1}/{len(pending)}] panel on group {gid}")
        votes = run_panel_on_group(gid, gdir, panel, timeout, collect_feedback=collect_feedback)
        vote_rows.extend(_vote_row(v) for v in votes)
        # Derive the chosen edge set's classes so the class-consistency gate can
        # demote cross-mode auto-accepts. compute_consensus is pure, so a first
        # (gate-less) call gives the chosen edge_set to look up classes for.
        ref_class, tgt_class = _segment_class_maps(_load_group_context(gdir)[2])
        base = compute_consensus(votes)
        edge_classes = _edge_classes_for(base.edge_set, ref_class, tgt_class)
        cons = compute_consensus(votes, edge_classes=edge_classes)
        consensus_out.append(_consensus_row(cons))
        logger.info(
            f"  -> {cons.consensus} choice={cons.choice} routing={cons.routing} "
            f"({'/'.join(v.provider + ':' + v.choice for v in votes)})"
        )
        _flush()  # persist after each group so an interrupted run is resumable

    votes_df = pd.DataFrame(vote_rows, columns=VOTES_COLUMNS)
    consensus_df = pd.DataFrame(consensus_out, columns=CONSENSUS_COLUMNS)

    votes_df.to_csv(batch_dir / "votes.csv", index=False)
    consensus_df.to_csv(batch_dir / "consensus.csv", index=False)
    return votes_df, consensus_df
