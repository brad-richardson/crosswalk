"""Batch agent runner for labeling pipeline.

Invokes Claude Code CLI in batch mode with few-shot examples.
A single CLI call processes multiple candidates, reading images
and metadata itself, writing results to a CSV file.
"""

import csv
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------

VARIANT_CONFIG: dict[str, dict] = {
    "subline_geometry_only": {"filename": "subline_geometry_only.png", "is_svg": False},
    "subline_road_context": {"filename": "subline_road_context.png", "is_svg": False},
    "subline_carto_positron": {"filename": "subline_carto_positron.png", "is_svg": False},
}

IMAGE_DESCRIPTIONS: dict[str, str] = {
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
    "subline_carto_positron": (
        "subline_carto_positron.png: alignment view on CartoDB light map tiles. "
        "Faint dashed lines show full segments (light blue=reference, light red=target). "
        "Bright solid lines with circles show aligned/overlapping portions "
        "(blue=reference, red=target)"
    ),
}

# CSV header for output labels
LABEL_HEADER = "ref_id,target_id,label,confidence,reasoning"


# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------


def select_few_shot_examples(
    batch_dir: Path,
    variant: str,
    n_examples: int = 4,
    exclude_batch: Path | None = None,
    few_shot_source: Path | None = None,
) -> list[dict]:
    """Select balanced few-shot examples from ground truth in other batches.

    Scans ``data/agents/batches/*/labels/ground_truth/data.csv`` for ground
    truth labels.  Excludes the current batch to avoid leaking test answers.
    Selects balanced examples (half match, half no_match).

    Args:
        batch_dir: Current batch directory (will be excluded).
        variant: Image variant name (to verify image exists).
        n_examples: Total number of examples to select.
        exclude_batch: Explicit batch to exclude (defaults to *batch_dir*).
        few_shot_source: If provided, only use examples from this batch.

    Returns:
        List of dicts with keys: ref_id, target_id, label,
        metadata_content, source_batch_dir.
    """
    if exclude_batch is None:
        exclude_batch = batch_dir

    # Resolve to absolute for reliable comparison
    exclude_batch = exclude_batch.resolve()

    if few_shot_source is not None:
        # Only scan the specified source batch
        gt_file = Path(few_shot_source) / "labels" / "ground_truth" / "data.csv"
        gt_files = [gt_file] if gt_file.exists() else []
    else:
        batches_root = batch_dir.parent  # data/agents/batches/
        gt_files = sorted(batches_root.glob("*/labels/ground_truth/data.csv"))

    matches: list[dict] = []
    no_matches: list[dict] = []

    vcfg = VARIANT_CONFIG.get(variant, {})
    variant_filename = vcfg.get("filename", f"{variant}.png")

    for gt_path in gt_files:
        source_batch = gt_path.parent.parent.parent  # batch dir
        if source_batch.resolve() == exclude_batch:
            continue

        try:
            df = pd.read_csv(gt_path, dtype=str)
        except Exception:
            continue

        required = {"ref_id", "target_id", "label"}
        if not required.issubset(df.columns):
            continue

        for _, row in df.iterrows():
            ref_id = str(row["ref_id"])
            target_id = str(row["target_id"])
            label = str(row["label"])

            if label not in ("match", "no_match"):
                continue

            cand_dir = source_batch / "candidates" / f"{ref_id}__{target_id}"
            metadata_path = cand_dir / "metadata.yaml"
            image_path = cand_dir / variant_filename

            if not metadata_path.exists():
                continue
            if not vcfg.get("is_svg", False) and not image_path.exists():
                continue

            entry = {
                "ref_id": ref_id,
                "target_id": target_id,
                "label": label,
                "metadata_content": metadata_path.read_text(),
                "source_batch_dir": source_batch,
            }

            if label == "match":
                matches.append(entry)
            else:
                no_matches.append(entry)

    # Balanced selection: half match, half no_match
    n_match = n_examples // 2
    n_no_match = n_examples - n_match

    # Cap at available
    n_match = min(n_match, len(matches))
    n_no_match = min(n_no_match, len(no_matches))

    # If one side is short, fill from the other
    if n_match < n_examples // 2:
        n_no_match = min(n_examples - n_match, len(no_matches))
    elif n_no_match < n_examples - n_examples // 2:
        n_match = min(n_examples - n_no_match, len(matches))

    selected = matches[:n_match] + no_matches[:n_no_match]
    return selected


# ---------------------------------------------------------------------------
# Few-shot directory preparation
# ---------------------------------------------------------------------------


def prepare_few_shot_dir(
    batch_dir: Path,
    few_shot_examples: list[dict],
    variant: str,
) -> Path:
    """Create an ``examples/`` directory with symlinks to few-shot candidates.

    For each example, symlinks the candidate directory from its source batch
    into ``{batch_dir}/examples/{ref_id}__{target_id}``.

    Args:
        batch_dir: Current batch directory.
        few_shot_examples: List from :func:`select_few_shot_examples`.
        variant: Image variant name (unused here, kept for API consistency).

    Returns:
        Path to the examples directory.
    """
    examples_dir = batch_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    for ex in few_shot_examples:
        link_name = examples_dir / f"{ex['ref_id']}__{ex['target_id']}"
        source_cand = ex["source_batch_dir"] / "candidates" / f"{ex['ref_id']}__{ex['target_id']}"

        if link_name.is_symlink() or link_name.is_file():
            link_name.unlink()
        elif link_name.is_dir():
            import shutil

            shutil.rmtree(link_name)

        if source_cand.exists():
            os.symlink(source_cand.resolve(), link_name)

    return examples_dir


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def prepare_batch_prompt(
    batch_dir: Path,
    variant: str,
    candidates: list[str],
    few_shot_examples: list[dict],
    output_path: str,
) -> str:
    """Build a comprehensive batch prompt with few-shot examples.

    Args:
        batch_dir: Batch directory (for resolving relative paths).
        variant: Image variant name.
        candidates: List of candidate directory names to process.
        few_shot_examples: Few-shot example dicts.
        output_path: Relative path where the agent should write the CSV.

    Returns:
        Complete prompt string.
    """
    vcfg = VARIANT_CONFIG.get(variant, {})
    variant_filename = vcfg.get("filename", f"{variant}.png")
    img_desc = IMAGE_DESCRIPTIONS.get(variant, f"{variant_filename}: image variant")

    # Section 1: Task description
    prompt = """You are labeling transportation network segment matches in batch mode.
Segments may be roads, sidewalks, bike lanes, trails, or other features in road/pedestrian/cycling networks.

TASK: For each candidate pair, determine whether the blue (reference) and red (target) segments represent the SAME PHYSICAL FEATURE.

A "match" means: this reference segment best represents this target segment. They cover the same physical movement space, even if the datasets differ in segmentation, naming, or classification.

A "no_match" means: either a better option exists for this target segment, or these are genuinely different features.

"""

    # Section 2: Label definitions
    prompt += """LABELS:
- match: Same physical feature with >=10% spatial overlap of either segment's length
- no_match: Different features, or segments on the same road that do not spatially overlap
- unsure: Ambiguous cases where reasonable people would disagree

"""

    # Section 3: Critical rules
    prompt += """CRITICAL RULES:
1. GEOMETRY FIRST: Always start with the image. If lines clearly overlap/follow the same path, it's likely a match. If they don't visually overlap, it's likely no_match regardless of names or metadata.
2. CLASS LABELS ARE WEAK EVIDENCE: Different classes (footway vs tertiary, residential vs secondary) don't preclude match - datasets classify the same road differently.
3. LENGTH DIFFERENCES OK: One segment can be longer (subsegment matches count as match).
4. PARALLEL BUT SEPARATE = NO MATCH: Lines that run SIDE BY SIDE (visually offset) are different features even if they share a street name.
5. SMALL OFFSET OK: 3-5m offset from GPS/digitization error is acceptable IF lines follow same path.
6. NAMES ARE SUPPORTING EVIDENCE ONLY: Same name does NOT guarantee match. Two segments can share a street name but be different spans of that street or different features alongside it. Different names don't prevent match either.
7. ML FEATURES ARE CONTEXT ONLY: The metadata includes computed ML features (hausdorff_distance, buffer_iou, etc.) for context, but do not rely on them as the primary basis for your decision. Many of these candidates are subjective cases where the features may not be well-tuned for the dataset. Focus on the image and primary attributes (geometry, names, classes) over raw feature values.

DATASET REPRESENTATION DIFFERENCES:
- The reference dataset (Overture) often uses a single centerline for divided roads, while local datasets may have separate segments for each carriageway (split carriageways).
- A centerline matched to one carriageway of a split road = match (same physical road, different representation).
- Opposite carriageways of a divided road matched to each other = no_match (physically separate lanes; each should match to its own reference segment).

BIKE LANE DECISION GUIDE:
- Painted bike lane / sharrows / flexpost-separated lane (same pavement surface) = same feature as road → match to road, no_match to cycleway
- Raised/curbed bike lane or separated cycle track (different surface/grade) = separate feature → no_match to road, match to cycleway
- If facility type is unclear from data and image, prefer unsure over guessing

ML FEATURE REFERENCE (rough thresholds for context):
- buffer_iou: >0.7 suggests match, <0.3 suggests no_match
- overlap_ratio: >0.8 suggests match, <0.3 suggests no_match
- hausdorff_distance: <15m suggests match, >50m suggests no_match
- heading_delta: <10 degrees suggests match, >45 degrees suggests no_match
These are guidelines only - always defer to the image over raw numbers.

NO_MATCH EXAMPLES:
- Two lines running parallel but visually offset (separate infrastructure)
- Northbound vs southbound lanes of divided highway
- Road centerline vs adjacent sidewalk (different surface and grade - raised curb separates them)
- Main road vs adjacent service road/alley
- Perpendicular/intersecting segments
- Segments that share an intersection endpoint but continue in different directions

MATCH EXAMPLES:
- Lines that overlap on the same path (even with different class labels)
- Same road with 3-5m digitization offset along its length
- Segments of different lengths that share overlapping portions (>=10%)
- Centerline matched to one carriageway of a divided road
- Same feature with different names or abbreviations

"""

    # Section 4: Image variant description
    prompt += f"""IMAGE VARIANT:
- {img_desc}
- For each candidate, view the file `{variant_filename}` in that candidate's directory.

"""

    # Section 5: Few-shot examples
    if few_shot_examples:
        prompt += """---
FEW-SHOT EXAMPLES:

Below are labeled examples for you to learn from. For each example, read its metadata.yaml and view its image to understand the labeling pattern.

"""
        for i, ex in enumerate(few_shot_examples, 1):
            prompt += f"""Example {i}: {ex["ref_id"]}__{ex["target_id"]}
Directory: examples/{ex["ref_id"]}__{ex["target_id"]}/
Image: examples/{ex["ref_id"]}__{ex["target_id"]}/{variant_filename}
Read the metadata.yaml in that directory, then view the image above.

Correct label for this example:
{ex["ref_id"]},{ex["target_id"]},{ex["label"]},1.0,ground truth example

"""

    # Section 6: Batch processing instructions
    prompt += f"""---
BATCH PROCESSING INSTRUCTIONS:

You must process the following {len(candidates)} candidates and write ALL results to a CSV file.

Output file: {output_path}
CSV header: ref_id,target_id,label,confidence,reasoning

For each candidate listed below:
1. Read `candidates/{{candidate_dir}}/metadata.yaml`
2. View `candidates/{{candidate_dir}}/{variant_filename}`
3. Determine the label (match, no_match, or unsure)
4. Assign a confidence score (0.0-1.0)
5. Write a brief reasoning (do NOT use commas in the reasoning field)

IMPORTANT:
- Process EVERY candidate listed below. Do not skip any.
- Write ALL results to the CSV file at `{output_path}`.
- First write the header line, then one line per candidate.
- Total candidates to process: {len(candidates)}

CANDIDATES TO PROCESS:
"""
    for cand in candidates:
        prompt += f"- {cand}\n"

    return prompt


# ---------------------------------------------------------------------------
# Claude CLI invocation
# ---------------------------------------------------------------------------


def invoke_claude_batch(
    model: str,
    prompt_path: Path,
    batch_dir: Path,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Invoke Claude Code CLI with a prompt file for batch processing.

    Pipes the prompt via stdin.  Uses ``--allowedTools`` to restrict the
    agent to Read, Write, and Glob (no Bash, no Edit, no web).

    Args:
        model: Model name (e.g. ``opus``, ``sonnet``).
        prompt_path: Path to the prompt text file.
        batch_dir: Working directory for the CLI invocation.
        timeout: Timeout in seconds.

    Returns:
        :class:`subprocess.CompletedProcess` result.
    """
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--allowedTools",
        "Read,Write,Glob",
        "--dangerously-skip-permissions",
    ]

    prompt_text = prompt_path.read_text()

    result = subprocess.run(
        cmd,
        input=prompt_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(batch_dir),
    )

    return result


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def validate_output_csv(
    output_path: Path,
    expected_candidates: set[tuple[str, str]],
) -> tuple[pd.DataFrame | None, list[str]]:
    """Validate the agent-written CSV output.

    Checks that the file exists, has the right header, and contains valid
    rows.  Reports missing candidates and duplicates.

    Args:
        output_path: Path to the CSV file the agent wrote.
        expected_candidates: Set of (ref_id, target_id) tuples expected.

    Returns:
        Tuple of (DataFrame or None, list of warning strings).
    """
    warnings: list[str] = []

    if not output_path.exists():
        warnings.append(f"Output file not found: {output_path}")
        return None, warnings

    try:
        df = pd.read_csv(output_path, dtype=str)
    except Exception as e:
        warnings.append(f"Could not parse CSV: {e}")
        return None, warnings

    # Validate header
    required_cols = {"ref_id", "target_id", "label", "confidence", "reasoning"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        warnings.append(f"Missing columns: {missing_cols}")
        return None, warnings

    valid_labels = {"match", "no_match", "unsure"}
    invalid_rows = []
    for idx, row in df.iterrows():
        label = str(row["label"]).strip()
        if label not in valid_labels:
            invalid_rows.append(idx)
            continue

        try:
            conf = float(row["confidence"])
            if not (0.0 <= conf <= 1.0):
                invalid_rows.append(idx)
        except (ValueError, TypeError):
            invalid_rows.append(idx)

    if invalid_rows:
        warnings.append(f"{len(invalid_rows)} rows with invalid label or confidence (dropped)")
        df = df.drop(index=invalid_rows).reset_index(drop=True)

    # Check for missing candidates
    found_pairs = set()
    for _, row in df.iterrows():
        found_pairs.add((str(row["ref_id"]).strip(), str(row["target_id"]).strip()))

    missing = expected_candidates - found_pairs
    if missing:
        warnings.append(f"{len(missing)} candidates missing from output")

    # Check for duplicates (keep last)
    dups = df.duplicated(subset=["ref_id", "target_id"], keep="last")
    n_dups = dups.sum()
    if n_dups > 0:
        warnings.append(f"{n_dups} duplicate rows (keeping last)")
        df = df.drop_duplicates(subset=["ref_id", "target_id"], keep="last")

    return df, warnings


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_agent_batch(
    model: str,
    variant: str,
    batch_dir: Path,
    limit: int = 0,
    overwrite: bool = False,
    n_few_shot: int = 4,
    few_shot_source: Path | None = None,
    timeout: int = 600,
    chunk_size: int = 25,
) -> None:
    """Run Claude agent in batch mode on a set of labeling candidates.

    Selects few-shot examples, builds a batch prompt, and invokes the
    Claude Code CLI once per chunk (~25 candidates).  The agent reads
    images and metadata itself and writes results to a CSV.

    Args:
        model: Model name (e.g. ``opus``, ``sonnet``).
        variant: Image variant name.
        batch_dir: Path to batch directory.
        limit: Maximum candidates to process (0 = no limit).
        overwrite: Start fresh (True) or resume from existing output (False).
        n_few_shot: Number of few-shot examples to include.
        few_shot_source: Explicit batch to source examples from.
        timeout: Timeout per chunk in seconds.
        chunk_size: Candidates per CLI invocation.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    if n_few_shot < 0:
        raise ValueError(f"n_few_shot must be non-negative, got {n_few_shot}")

    batch_dir = Path(batch_dir)
    candidates_dir = batch_dir / "candidates"
    if not candidates_dir.exists():
        logger.error(f"Candidates directory not found: {candidates_dir}")
        return

    # Resolve variant config
    if variant and variant in VARIANT_CONFIG:
        vcfg = VARIANT_CONFIG[variant]
    elif variant:
        vcfg = {"filename": f"{variant}.png", "is_svg": False}
        logger.warning(f"Unknown variant '{variant}', trying {vcfg['filename']}")
    else:
        logger.error("Variant is required for batch mode")
        return

    # Build output directory name: claude_{model}_{variant}
    output_name = "claude"
    if model:
        output_name = f"claude_{model}"
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
        log_file.write_text("")

    _log(log_file, f"=== Batch Run Started: {datetime.now(UTC).isoformat()} ===")

    logger.info(f"Model: {model}")
    logger.info(f"Variant: {variant}")
    logger.info(f"Batch: {batch_dir}")
    logger.info(f"Output: {output_file}")

    # Enumerate candidate directories
    candidate_dirs = sorted(
        [d for d in candidates_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    # Build list of (ref_id, target_id, dir_name) for all candidates
    all_candidates: list[tuple[str, str, str]] = []
    for cand_dir in candidate_dirs:
        dir_name = cand_dir.name
        parts = dir_name.split("__", 1)
        if len(parts) != 2:
            logger.warning(f"Skipping unexpected directory name: {dir_name}")
            continue
        ref_id, target_id = parts
        all_candidates.append((ref_id, target_id, dir_name))

    # Apply limit to total candidates (before filtering out existing)
    if limit > 0:
        all_candidates = all_candidates[:limit]

    # Filter out already-labeled candidates (resume)
    if existing_pairs:
        remaining = [(r, t, d) for r, t, d in all_candidates if (r, t) not in existing_pairs]
        logger.info(
            f"Filtered: {len(all_candidates)} total, "
            f"{len(all_candidates) - len(remaining)} already labeled, "
            f"{len(remaining)} remaining"
        )
        all_candidates = remaining

    if not all_candidates:
        logger.info("No candidates to process")
        return

    logger.info(f"Processing {len(all_candidates)} candidates in chunks of {chunk_size}")

    # Select few-shot examples
    few_shot_examples = select_few_shot_examples(
        batch_dir=batch_dir,
        variant=variant,
        n_examples=n_few_shot,
        few_shot_source=few_shot_source,
    )
    if few_shot_examples:
        logger.info(
            f"Selected {len(few_shot_examples)} few-shot examples "
            f"({sum(1 for e in few_shot_examples if e['label'] == 'match')} match, "
            f"{sum(1 for e in few_shot_examples if e['label'] == 'no_match')} no_match)"
        )
        examples_dir = prepare_few_shot_dir(batch_dir, few_shot_examples, variant)
    else:
        logger.warning("No few-shot examples found; proceeding without examples")
        examples_dir = None

    # Chunk candidates and invoke
    chunks = [all_candidates[i : i + chunk_size] for i in range(0, len(all_candidates), chunk_size)]

    total_success = 0
    total_missing = 0

    for chunk_idx, chunk in enumerate(chunks):
        chunk_dir_names = [d for _, _, d in chunk]
        chunk_expected = {(r, t) for r, t, _ in chunk}

        # Output path relative to batch_dir for the agent
        # For chunked operation, each chunk appends to a temp CSV,
        # then we merge into the main output
        chunk_output_name = f"_chunk_{chunk_idx}.csv"
        chunk_output_rel = f"labels/{output_name}/{chunk_output_name}"
        chunk_output_abs = output_dir / chunk_output_name

        # Remove any previous chunk file
        if chunk_output_abs.exists():
            chunk_output_abs.unlink()

        logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk)} candidates")

        # Build and write prompt
        prompt_text = prepare_batch_prompt(
            batch_dir=batch_dir,
            variant=variant,
            candidates=chunk_dir_names,
            few_shot_examples=few_shot_examples,
            output_path=chunk_output_rel,
        )

        prompt_path = output_dir / f"prompt_chunk_{chunk_idx}.txt"
        prompt_path.write_text(prompt_text)

        # Also save the latest prompt as prompt.txt for inspection
        if chunk_idx == 0:
            (output_dir / "prompt.txt").write_text(prompt_text)

        _log(log_file, f"[{datetime.now(UTC).strftime('%H:%M:%S')}] Chunk {chunk_idx + 1} started")

        # Invoke Claude
        try:
            result = invoke_claude_batch(
                model=model,
                prompt_path=prompt_path,
                batch_dir=batch_dir,
                timeout=timeout,
            )

            # Log raw output
            raw_text = f"=== Chunk {chunk_idx + 1} ===\n"
            raw_text += f"Exit code: {result.returncode}\n"
            raw_text += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                raw_text += f"STDERR:\n{result.stderr}\n"
            _append(raw_output_file, raw_text)

            if result.returncode != 0:
                logger.warning(
                    f"Chunk {chunk_idx + 1}: Claude exited with code {result.returncode}"
                )

        except subprocess.TimeoutExpired:
            logger.error(f"Chunk {chunk_idx + 1}: timed out after {timeout}s")
            _log(log_file, f"  TIMEOUT: chunk {chunk_idx + 1}")
            continue
        except Exception as e:
            logger.error(f"Chunk {chunk_idx + 1}: invocation failed: {e}")
            _log(log_file, f"  ERROR: {e}")
            continue

        # Validate and merge chunk output
        chunk_df, chunk_warnings = validate_output_csv(chunk_output_abs, chunk_expected)

        for w in chunk_warnings:
            logger.warning(f"Chunk {chunk_idx + 1}: {w}")
            _log(log_file, f"  WARNING: {w}")

        if chunk_df is not None and len(chunk_df) > 0:
            # Append to main output file
            chunk_df.to_csv(output_file, mode="a", header=False, index=False)
            total_success += len(chunk_df)
            logger.info(f"Chunk {chunk_idx + 1}: {len(chunk_df)} labels written")

            # Periodic backup
            if (chunk_idx + 1) % 5 == 0:
                backup_path = output_dir / f"data_backup_chunk{chunk_idx + 1}.csv"
                import shutil
                shutil.copy2(output_file, backup_path)
                logger.info(f"Backup saved: {backup_path}")

            missing_count = len(chunk_expected) - len(chunk_df)
            if missing_count > 0:
                total_missing += missing_count
        else:
            total_missing += len(chunk)
            logger.warning(f"Chunk {chunk_idx + 1}: no valid output")

        # Clean up chunk file
        if chunk_output_abs.exists():
            chunk_output_abs.unlink()

    # Clean up examples directory
    if examples_dir and examples_dir.exists():
        import shutil

        shutil.rmtree(examples_dir, ignore_errors=True)

    # Clean up chunk prompt files
    for chunk_idx in range(len(chunks)):
        p = output_dir / f"prompt_chunk_{chunk_idx}.txt"
        if p.exists() and chunk_idx > 0:
            p.unlink()

    logger.info(f"Complete: {total_success} labels written, {total_missing} missing/failed")
    logger.info(f"Output: {output_file}")

    _log(log_file, f"=== Run Complete: {datetime.now(UTC).isoformat()} ===")
    _log(log_file, f"Success: {total_success}, Missing: {total_missing}")


def _log(path: Path, message: str) -> None:
    """Append a log line to a file."""
    with open(path, "a") as f:
        f.write(message + "\n")


def _append(path: Path, text: str) -> None:
    """Append text to a file."""
    with open(path, "a") as f:
        f.write(text)
