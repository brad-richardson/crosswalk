"""Agent runner for labeling pipeline.

Invokes AI agent CLIs (Claude, Gemini, Codex, Ollama) on candidate
packages and collects structured label responses.

Replaces the shell-based run_agent.sh with a Python implementation
accessible via ``matcher run-agent``.
"""

import csv
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------

VARIANT_CONFIG: dict[str, dict] = {
    "geometry_only": {"filename": "geometry_only.png", "is_svg": False},
    "carto_positron": {"filename": "carto_positron.png", "is_svg": False},
    "road_context": {"filename": "road_context.png", "is_svg": False},
    "road_context_svg": {"filename": "road_context.svg", "is_svg": True},
    "subline_geometry_only": {"filename": "subline_geometry_only.png", "is_svg": False},
    "subline_road_context": {"filename": "subline_road_context.png", "is_svg": False},
}

IMAGE_DESCRIPTIONS: dict[str, str] = {
    "geometry_only": (
        "geometry_only.png: clean geometry view on white background "
        "(blue circles=reference, red circles=target)"
    ),
    "carto_positron": (
        "carto_positron.png: geometry overlay on CartoDB light map "
        "(blue circles=reference, red circles=target)"
    ),
    "road_context": (
        "road_context.png: geometry with nearby roads shown as gray lines for context "
        "(blue circles=reference, red circles=target)"
    ),
    "road_context_svg": (
        "SVG content inline below: geometry with nearby roads shown as gray lines "
        "for context (blue=reference, red=target)"
    ),
    "subline_geometry_only": (
        "subline_geometry_only.png: alignment view on white background. "
        "Faint dashed lines show full segments (light blue=reference, light red=target). "
        "Bright solid lines with circles show aligned/overlapping portions "
        "(blue=reference, red=target)"
    ),
    "subline_road_context": (
        "subline_road_context.png: alignment view with road context. "
        "Gray dashed lines show nearby roads (for context only, ignore them). "
        "Faint dashed colored lines show full segments "
        "(light blue=reference, light red=target). "
        "Bright solid lines with circles show aligned/overlapping portions "
        "(blue=reference, red=target)"
    ),
}

# CSV header for output labels
LABEL_HEADER = "ref_id,target_id,label,confidence,reasoning"

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(
    ref_id: str,
    target_id: str,
    metadata_content: str,
    variant: str | None = None,
    svg_content: str | None = None,
) -> str:
    """Build the labeling prompt with static prefix + variable suffix.

    Args:
        ref_id: Reference segment ID
        target_id: Target segment ID
        metadata_content: YAML metadata string
        variant: Image variant name, or None for legacy (satellite + geometry)
        svg_content: SVG markup to inline for SVG variants

    Returns:
        Complete prompt string
    """
    # Static prefix - keep core task first for prefix caching
    if variant and variant in IMAGE_DESCRIPTIONS:
        img_desc = f"- {IMAGE_DESCRIPTIONS[variant]}"
        static_prefix = f"""You are labeling transportation segment matches. Do NOT explore files or use tools.

TASK: Do the blue and red segments represent the SAME PHYSICAL FEATURE?
Output exactly ONE CSV line: ref_id,target_id,LABEL,CONFIDENCE,REASON

LABELS:
- match: Same physical feature with >=10% overlap (roads match roads, sidewalks match sidewalks)
- no_match: Different features (parallel roads, road vs sidewalk, perpendicular streets)
- unsure: Ambiguous cases

CRITICAL RULES:
1. GEOMETRY FIRST: If lines clearly overlap/follow the same path, it's likely a match
2. CLASS LABELS ARE WEAK EVIDENCE: Different classes (footway vs tertiary, residential vs secondary) don't preclude match - datasets classify the same road differently
3. LENGTH DIFFERENCES OK: One segment can be longer (subsegment matches count as match)
4. PARALLEL BUT SEPARATE = NO MATCH: Lines that run SIDE BY SIDE (visually offset) are different features
5. SMALL OFFSET OK: 3-5m offset from GPS/digitization error is acceptable IF lines follow same path
6. NAMES ARE SECONDARY: Same name doesn't guarantee match; different names don't prevent match

NO_MATCH EXAMPLES:
- Two lines running parallel but visually offset from each other (separate infrastructure)
- Northbound vs southbound lanes of divided highway
- Road centerline vs sidewalk that runs 5m to the side
- Main road vs adjacent service road/alley
- Perpendicular/intersecting segments

MATCH EXAMPLES:
- Lines that overlap on the same path (even with different class labels like footway/tertiary)
- Same road with 3-5m digitization offset along its length
- Segments of different lengths that share overlapping portions
- Same feature with different names or abbreviations

Image:
{img_desc}

CONFIDENCE: 0.0-1.0 (how certain you are)
REASON: Brief explanation with no commas

---
"""
    else:
        static_prefix = """You are labeling transportation segment matches. Do NOT explore files or use tools.

TASK: Do the blue and red segments represent the SAME PHYSICAL FEATURE?
Output exactly ONE CSV line: ref_id,target_id,LABEL,CONFIDENCE,REASON

LABELS:
- match: Same physical feature with >=10% overlap (roads match roads, sidewalks match sidewalks)
- no_match: Different features (parallel roads, road vs sidewalk, perpendicular streets)
- unsure: Ambiguous cases

CRITICAL RULES:
1. GEOMETRY FIRST: If lines clearly overlap/follow the same path, it's likely a match
2. CLASS LABELS ARE WEAK EVIDENCE: Different classes (footway vs tertiary, residential vs secondary) don't preclude match - datasets classify the same road differently
3. LENGTH DIFFERENCES OK: One segment can be longer (subsegment matches count as match)
4. PARALLEL BUT SEPARATE = NO MATCH: Lines that run SIDE BY SIDE (visually offset) are different features
5. SMALL OFFSET OK: 3-5m offset from GPS/digitization error is acceptable IF lines follow same path
6. NAMES ARE SECONDARY: Same name doesn't guarantee match; different names don't prevent match

NO_MATCH EXAMPLES:
- Two lines running parallel but visually offset from each other (separate infrastructure)
- Northbound vs southbound lanes of divided highway
- Road centerline vs sidewalk that runs 5m to the side
- Main road vs adjacent service road/alley
- Perpendicular/intersecting segments

MATCH EXAMPLES:
- Lines that overlap on the same path (even with different class labels like footway/tertiary)
- Same road with 3-5m digitization offset along its length
- Segments of different lengths that share overlapping portions
- Same feature with different names or abbreviations

Images:
- satellite.png: aerial view with geometry overlay (blue=reference, red=target)
- geometry.png: clean geometry on white background (same colors)

CONFIDENCE: 0.0-1.0 (how certain you are)
REASON: Brief explanation with no commas

---
"""

    # Variable suffix
    variable_suffix = f"Candidate: {ref_id},{target_id}\n\nMetadata:\n{metadata_content}"

    # Optional inline SVG
    if svg_content:
        variable_suffix += f"\n\nSVG Image:\n{svg_content}"

    return static_prefix + variable_suffix


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


def invoke_agent(
    agent: str,
    model: str,
    prompt: str,
    image_path: Path | None = None,
    candidate_dir: Path | None = None,
    variant_is_svg: bool = False,
    variant_filename: str | None = None,
) -> str | None:
    """Dispatch to agent-specific CLI via subprocess.

    Args:
        agent: Agent name (claude, gemini, codex, ollama)
        model: Model variant (e.g. "sonnet", "flash")
        prompt: Complete prompt text
        image_path: Path to image file (PNG), if applicable
        candidate_dir: Path to candidate directory
        variant_is_svg: Whether the variant is SVG (content inlined in prompt)
        variant_filename: Filename of the variant image

    Returns:
        Raw text response from agent, or None on failure
    """
    try:
        if agent == "claude":
            return _invoke_claude(model, prompt, candidate_dir, variant_is_svg, variant_filename)
        elif agent == "gemini":
            return _invoke_gemini(model, prompt, image_path, candidate_dir)
        elif agent == "codex":
            return _invoke_codex(prompt, image_path)
        elif agent == "ollama":
            return _invoke_ollama(model, prompt, image_path)
        else:
            logger.error(f"Unknown agent: {agent}")
            return None
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        return None


def _invoke_claude(
    model: str,
    prompt: str,
    candidate_dir: Path | None,
    variant_is_svg: bool,
    variant_filename: str | None,
) -> str | None:
    """Invoke Claude Code CLI."""
    cmd = ["claude", "-p"]
    if model:
        cmd.extend(["--model", model])

    # Build read instruction
    if variant_filename and not variant_is_svg:
        read_instruction = f"First read {variant_filename}, then answer:"
    elif variant_is_svg:
        read_instruction = "Answer the following (SVG image is inlined in the prompt):"
    else:
        read_instruction = "First read satellite.png and geometry.png, then answer:"

    full_prompt = f"{read_instruction}\n\n{prompt}"
    cmd.append(full_prompt)

    cwd = str(candidate_dir) if candidate_dir else None
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=cwd,
    )
    if result.returncode != 0:
        logger.warning(
            "Claude CLI returned non-zero exit code: {}. stderr: {}",
            result.returncode,
            result.stderr[:200] if result.stderr else "(empty)",
        )
        return None
    return result.stdout + result.stderr


def _invoke_gemini(
    model: str,
    prompt: str,
    image_path: Path | None,
    candidate_dir: Path | None,
) -> str | None:
    """Invoke Gemini CLI with sandbox."""
    with tempfile.TemporaryDirectory() as sandbox:
        sandbox_path = Path(sandbox)

        # Copy image to sandbox
        if image_path and image_path.exists():
            import shutil

            dest = sandbox_path / image_path.name
            shutil.copy2(image_path, dest)

        # Write prompt to file
        prompt_file = sandbox_path / "prompt.txt"
        prompt_file.write_text(prompt)

        cmd = ["gemini", "--sandbox", "--yolo", "-o", "text"]
        if model:
            cmd.extend(["-m", model])

        # Build command differently depending on whether an image is available
        if image_path and (sandbox_path / image_path.name).exists():
            cmd.append(f"Analyze {image_path.name} using instructions in prompt.txt")
            cmd.append("prompt.txt")
            cmd.append(str(sandbox_path / image_path.name))
        else:
            # Text-only mode (e.g., SVG variants where content is inlined in prompt)
            cmd.append("Use instructions in prompt.txt")
            cmd.append("prompt.txt")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(sandbox_path),
        )
        return result.stdout


def _invoke_codex(prompt: str, image_path: Path | None) -> str | None:
    """Invoke OpenAI Codex CLI."""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("")
        tmpout = f.name

    cmd = ["codex", "exec"]
    if image_path and image_path.exists():
        cmd.extend(["-i", str(image_path)])
    cmd.extend(["-o", tmpout, "--", prompt])

    subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    try:
        return Path(tmpout).read_text()
    except Exception as e:
        logger.debug(f"Could not read codex output file: {e}")
        return None
    finally:
        Path(tmpout).unlink(missing_ok=True)


def _invoke_ollama(model: str, prompt: str, image_path: Path | None) -> str | None:
    """Invoke local Ollama via HTTP API."""
    import base64
    import json

    model_name = model or "llava"
    images = []
    if image_path and image_path.exists():
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        images.append(b64)

    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "images": images,
            "stream": False,
        }
    )

    result = subprocess.run(
        ["curl", "-s", "http://localhost:11434/api/generate", "-d", payload],
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        data = json.loads(result.stdout)
        return data.get("response") or data.get("error") or "No response"
    except (json.JSONDecodeError, KeyError):
        return result.stdout


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_response(raw: str | None, ref_id: str, target_id: str) -> str | None:
    """Extract CSV label line from raw agent output.

    Tries two strategies:
    1. Look for a line starting with ``ref_id,``
    2. Grep for match/no_match/unsure keyword and construct a CSV line

    Args:
        raw: Raw text output from agent
        ref_id: Expected reference ID
        target_id: Expected target ID

    Returns:
        CSV line string, or None if parsing fails
    """
    if not raw:
        return None

    # Strategy 1: look for CSV line starting with ref_id
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{ref_id},"):
            return stripped

    # Strategy 2: keyword fallback
    label_match = re.search(r"\b(no_match|match|unsure)\b", raw)
    if label_match:
        label = label_match.group(1)
        return f"{ref_id},{target_id},{label},0.5,parsed"

    return None


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_agent_batch(
    agent: str,
    model: str,
    variant: str,
    batch_dir: Path,
    limit: int = 0,
    overwrite: bool = False,
    bail_after: int = 2,
) -> None:
    """Main loop: iterate candidates, invoke agent, collect labels.

    Args:
        agent: Agent name (claude, gemini, codex, ollama)
        model: Model variant
        variant: Image variant name
        batch_dir: Path to batch directory
        limit: Maximum candidates to process (0 = no limit)
        overwrite: Start fresh (True) or resume from existing output (False)
        bail_after: Stop after N consecutive failures (0 = never bail)
    """
    if agent not in ("claude", "codex", "gemini", "ollama"):
        logger.error(f"Unknown agent: {agent}. Use claude, gemini, codex, or ollama.")
        return

    batch_dir = Path(batch_dir)
    candidates_dir = batch_dir / "candidates"
    if not candidates_dir.exists():
        logger.error(f"Candidates directory not found: {candidates_dir}")
        return

    # Resolve variant config
    if variant and variant in VARIANT_CONFIG:
        vcfg = VARIANT_CONFIG[variant]
    elif variant:
        # Unknown variant - try as filename
        vcfg = {"filename": f"{variant}.png", "is_svg": False}
        logger.warning(f"Unknown variant '{variant}', trying {vcfg['filename']}")
    else:
        vcfg = None  # Legacy mode

    # Build output directory name
    output_name = agent
    if model:
        output_name = f"{agent}_{model}"
    if variant:
        output_name = f"{output_name}_{variant}"

    output_dir = batch_dir / "labels" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "data.csv"
    log_file = output_dir / "run.log"
    raw_output_file = output_dir / "raw_responses.log"

    # Handle resume vs overwrite
    existing_pairs: set[tuple[str, str]] = set()
    if not overwrite and output_file.exists():
        try:
            with open(output_file) as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        existing_pairs.add((row[0], row[1]))
        except Exception as e:
            logger.debug(f"Could not read existing output file for resume: {e}")
        logger.info(f"Resuming: found {len(existing_pairs)} existing labels")
    else:
        output_file.write_text(LABEL_HEADER + "\n")
        raw_output_file.write_text("")
        log_file.write_text("")  # Truncate log on overwrite

    _log(log_file, f"=== {agent} Run Started: {datetime.now(UTC).isoformat()} ===")

    logger.info(f"Agent: {agent}")
    if model:
        logger.info(f"Model: {model}")
    if variant:
        logger.info(f"Variant: {variant}")
    logger.info(f"Batch: {batch_dir}")
    logger.info(f"Output: {output_file}")

    # Enumerate candidates
    candidate_dirs = sorted(
        [d for d in candidates_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    count = 0
    failed = 0
    skipped = 0
    consecutive_fails = 0
    actually_processed = 0

    for cand_dir in candidate_dirs:
        if limit > 0 and actually_processed >= limit:
            logger.info(f"Reached limit of {limit} candidates")
            break

        dir_name = cand_dir.name
        parts = dir_name.split("__", 1)
        if len(parts) != 2:
            logger.warning(f"Skipping unexpected directory name: {dir_name}")
            continue

        ref_id, target_id = parts

        # Skip if already labeled (resume mode)
        if (ref_id, target_id) in existing_pairs:
            skipped += 1
            continue

        metadata_path = cand_dir / "metadata.yaml"
        if not metadata_path.exists():
            logger.warning(f"No metadata in {cand_dir}")
            continue

        metadata_content = metadata_path.read_text()

        # Determine image path and SVG content
        image_path = None
        svg_content = None
        variant_filename = None

        if vcfg:
            variant_filename = vcfg["filename"]
            img_file = cand_dir / variant_filename
            if vcfg["is_svg"]:
                if img_file.exists():
                    svg_content = img_file.read_text()
            else:
                image_path = img_file if img_file.exists() else None
        else:
            # Legacy mode: satellite.png and geometry.png
            geo_path = cand_dir / "geometry.png"
            image_path = geo_path if geo_path.exists() else None
            # Legacy prompt references both files; Claude reads them from cwd

        # Build prompt
        prompt = build_prompt(
            ref_id, target_id, metadata_content, variant=variant, svg_content=svg_content
        )

        _log(log_file, f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {ref_id}")

        # Invoke agent
        raw = invoke_agent(
            agent=agent,
            model=model,
            prompt=prompt,
            image_path=image_path,
            candidate_dir=cand_dir,
            variant_is_svg=vcfg["is_svg"] if vcfg else False,
            variant_filename=variant_filename,
        )

        # Log raw response
        _append(raw_output_file, f"=== {ref_id} ===\n{raw or '(no response)'}\n")

        # Parse response
        result = parse_response(raw, ref_id, target_id)

        if result:
            _append(output_file, result + "\n")
            count += 1
            consecutive_fails = 0
            label = result.split(",")[2] if len(result.split(",")) > 2 else "?"
            logger.info(f"{ref_id}: {label} [{count}]")
        else:
            failed += 1
            consecutive_fails += 1
            _log(log_file, f"  FAIL: {(raw or '(no response)')[:200]}")
            logger.warning(f"{ref_id}: FAIL [{count + failed}]")

            if bail_after > 0 and consecutive_fails >= bail_after:
                logger.error(
                    f"BAILING OUT: {consecutive_fails} consecutive failures. "
                    f"Completed: {count} success, {failed} failed, {skipped} skipped"
                )
                break

        actually_processed += 1

    logger.info(f"Complete: {count} success, {failed} failed, {skipped} skipped")
    logger.info(f"Output: {output_file}")


def _log(path: Path, message: str) -> None:
    """Append a log line to a file."""
    with open(path, "a") as f:
        f.write(message + "\n")


def _append(path: Path, text: str) -> None:
    """Append text to a file."""
    with open(path, "a") as f:
        f.write(text)
