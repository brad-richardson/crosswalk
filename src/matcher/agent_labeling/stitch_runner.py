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

# ---------------------------------------------------------------------------
# Provider panel configuration
# ---------------------------------------------------------------------------


@dataclass
class ProviderSpec:
    """A panel member: how to invoke a provider CLI on an evidence pack."""

    name: str  # short id used in votes.csv (e.g. "claude")
    model: str  # model string recorded in votes


DEFAULT_PANEL = [
    ProviderSpec(name="claude", model="sonnet"),
    ProviderSpec(name="codex", model="gpt-5.4"),
    ProviderSpec(name="agy", model="Gemini 3.5 Flash (Low)"),
]


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
    return imgs


def _check_exit(provider: str, result: subprocess.CompletedProcess) -> None:
    """Raise on non-zero CLI exit so failures don't masquerade as parse errors.

    Includes truncated stderr in the message; the runner records it in the
    abstention error trail.
    """
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:500]
        raise RuntimeError(f"{provider} exited with code {result.returncode}: {stderr}")


def invoke_claude(
    prompt: str, group_dir: Path, letters: list[str], model: str, timeout: int = 240
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
    prompt: str, group_dir: Path, letters: list[str], model: str, timeout: int = 240
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
            "model_reasoning_effort=low",
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
    prompt: str, group_dir: Path, letters: list[str], model: str, timeout: int = 240
) -> str:
    """Invoke the agy (Antigravity) CLI. =-form flags only; reads images by path."""
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


_INVOKERS = {
    "claude": invoke_claude,
    "codex": invoke_codex,
    "agy": invoke_agy,
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


def run_provider_on_group(
    provider: ProviderSpec,
    group_id: str,
    group_dir: Path | None,
    prompt: str,
    letters: list[str],
    options_by_letter: dict[str, list[tuple[str, str]]],
    timeout: int = 240,
    retries: int = 1,
) -> Vote:
    """Run one provider on one group, validate, retry once on invalid output.

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
) -> Vote:
    last_err = ""
    last_raw = ""
    for attempt in range(retries + 1):
        start = time.monotonic()
        try:
            raw = invoker(prompt, group_dir, letters, provider.model, timeout)
        except subprocess.TimeoutExpired:
            # Retrying a timeout just burns another full timeout window; abstain.
            last_err = f"timeout after {timeout}s"
            last_raw = ""
            break
        except Exception as e:  # noqa: BLE001 - record any invocation failure
            last_err = f"invocation error: {e}"
            last_raw = ""
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
            )
        except ValueError as e:
            last_err = f"parse/validation: {e}"
            logger.warning(f"{provider.name} group {group_id} attempt {attempt}: {e}")

    # All attempts failed -> abstention
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
        error=last_err,
    )


def run_panel_on_group(
    group_id: str,
    group_dir: Path,
    panel: list[ProviderSpec],
    timeout: int = 240,
) -> list[Vote]:
    """Run the full panel on one group in parallel (one thread per provider)."""
    letters, options_by_letter, _meta = _load_group_context(group_dir)
    prompt = (group_dir / "prompt.txt").read_text()

    def _run(p: ProviderSpec) -> Vote:
        return run_provider_on_group(
            p, group_id, group_dir, prompt, letters, options_by_letter, timeout
        )

    with ThreadPoolExecutor(max_workers=len(panel)) as ex:
        votes = list(ex.map(_run, panel))
    return votes


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


def compute_consensus(votes: list[Vote]) -> Consensus:
    """Apply the 3/3, 2/3, else routing rule over a group's votes.

    Abstentions do not count toward agreement. Only unanimous agreement among
    all (>=3) valid votes auto-accepts; everything else routes to human review.
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
        )

    top_choice = max(tally, key=lambda c: len(tally[c]))
    top_votes = tally[top_choice]
    agree = len(top_votes)

    minority_votes = [v for v in valid if v.choice != top_choice]
    minority = "; ".join(f"{v.provider}={v.choice}" for v in minority_votes)
    mean_conf = round(sum(v.confidence for v in top_votes) / len(top_votes), 3)
    edge_set = top_votes[0].edge_set

    # Unanimous requires all 3 panelists valid AND agreeing.
    if agree == len(votes) and agree >= 3 and not minority_votes:
        consensus = "unanimous"
        routing = "auto_accept" if top_choice != "NONE" else "human_review"
    elif agree >= 2:
        consensus = "majority"
        routing = "human_review"
    else:
        consensus = "none"
        routing = "human_review"

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
]


def _edge_set_str(es: frozenset) -> str:
    return json.dumps(sorted([list(e) for e in es]))


def run_batch(
    batch_dir: Path,
    panel: list[ProviderSpec] | None = None,
    group_ids: list[str] | None = None,
    timeout: int = 240,
    limit: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the panel over every generated group in a batch dir.

    Expects evidence packs under ``batch_dir/{group_id}/``. Writes
    ``batch_dir/votes.csv`` and ``batch_dir/consensus.csv``. Returns the two
    DataFrames.
    """
    batch_dir = Path(batch_dir)
    panel = panel or DEFAULT_PANEL

    group_dirs = sorted(
        d for d in batch_dir.iterdir() if d.is_dir() and (d / "prompt.txt").exists()
    )
    if group_ids:
        wanted = set(group_ids)
        group_dirs = [d for d in group_dirs if d.name in wanted]
    if limit > 0:
        group_dirs = group_dirs[:limit]

    all_votes: list[Vote] = []
    consensus_rows: list[Consensus] = []

    for i, gdir in enumerate(group_dirs):
        gid = gdir.name
        logger.info(f"[{i + 1}/{len(group_dirs)}] panel on group {gid}")
        votes = run_panel_on_group(gid, gdir, panel, timeout)
        all_votes.extend(votes)
        cons = compute_consensus(votes)
        consensus_rows.append(cons)
        logger.info(
            f"  -> {cons.consensus} choice={cons.choice} routing={cons.routing} "
            f"({'/'.join(v.provider + ':' + v.choice for v in votes)})"
        )

    votes_df = pd.DataFrame(
        [
            {
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
            }
            for v in all_votes
        ],
        columns=VOTES_COLUMNS,
    )
    consensus_df = pd.DataFrame(
        [
            {
                "group_id": c.group_id,
                "consensus": c.consensus,
                "choice": c.choice,
                "edge_set": _edge_set_str(c.edge_set),
                "routing": c.routing,
                "n_votes": c.n_votes,
                "n_valid": c.n_valid,
                "minority": c.minority,
                "mean_confidence": c.mean_confidence,
            }
            for c in consensus_rows
        ],
        columns=CONSENSUS_COLUMNS,
    )

    votes_df.to_csv(batch_dir / "votes.csv", index=False)
    consensus_df.to_csv(batch_dir / "consensus.csv", index=False)
    return votes_df, consensus_df
