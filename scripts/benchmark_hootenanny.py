#!/usr/bin/env python3
"""Benchmark Hootenanny conflation against ground truth labels.

This script:
1. Converts reference/target data to OSM format
2. Runs Hootenanny conflation
3. Extracts match pairs from Hootenanny's output
4. Compares against ground truth labels
5. Reports precision, recall, F1
"""

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

# Add matcher to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

HOOT_BIN = "/var/lib/hootenanny/bin/hoot"


def get_available_datasets() -> dict[str, int]:
    """List datasets that have labels with their label counts.

    Returns:
        Dictionary mapping dataset name to label count.
    """
    labels_dir = Path(__file__).parents[1] / "labels"
    datasets: dict[str, int] = {}

    # Prioritize Hive-partitioned labels (CSV)
    for d in labels_dir.glob("dataset=*"):
        csv_path = d / "data.csv"
        if csv_path.exists():
            name = d.name.replace("dataset=", "")
            datasets[name] = len(pd.read_csv(csv_path))

    # Check for Hive-partitioned labels (parquet)
    for d in labels_dir.glob("dataset=*"):
        parquet_path = d / "data.parquet"
        if parquet_path.exists():
            name = d.name.replace("dataset=", "")
            if name not in datasets:
                datasets[name] = len(pd.read_parquet(parquet_path))

    return datasets


def get_dataset_files(dataset_name: str) -> tuple[Path, Path, Path | None]:
    """Get reference and target files for a dataset using labeling app logic.

    Returns:
        (reference_segments, target_segments, reference_connectors)
    """
    raw_dir = Path(__file__).parents[1] / "data" / "raw"

    # Target file: {dataset_name}.parquet (e.g., us_boston_streets.parquet)
    target_file = raw_dir / f"{dataset_name}.parquet"
    if not target_file.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")

    # Find Overture reference using same logic as labeling app
    reference_file = _find_overture_reference(dataset_name, raw_dir)
    if reference_file is None:
        raise FileNotFoundError(f"Overture reference not found for {dataset_name}")

    reference_path = raw_dir / reference_file

    # Check for connectors
    connectors_path = reference_path.parent / reference_file.replace("_segments.", "_connectors.")
    if not connectors_path.exists():
        connectors_path = raw_dir / "overture_connectors.parquet"

    return (
        reference_path,
        target_file,
        connectors_path if connectors_path.exists() else None,
    )


def _find_overture_reference(dataset_name: str, raw_dir: Path) -> str | None:
    """Find the Overture reference file for a dataset (same as labeling app)."""
    parts = dataset_name.split("_")

    # Try progressively shorter prefixes
    for i in range(len(parts), 0, -1):
        region = "_".join(parts[:i])
        candidate = f"{region}_overture_segments.parquet"
        if (raw_dir / candidate).exists():
            return candidate

    # Fallback to generic
    if (raw_dir / "overture_segments.parquet").exists():
        return "overture_segments.parquet"

    return None


def load_labels(dataset_name: str | None = None) -> pd.DataFrame:
    """Load ground truth labels for a dataset (or all if None)."""
    labels_dir = Path(__file__).parents[1] / "labels"

    # Prioritize Hive-partitioned structure (current format)
    if dataset_name:
        # Try CSV first (more common)
        labels_path = labels_dir / f"dataset={dataset_name}" / "data.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            logger.info(f"Loaded {len(df)} labels for {dataset_name}")
            return df

        # Try parquet
        labels_path = labels_dir / f"dataset={dataset_name}" / "data.parquet"
        if labels_path.exists():
            df = pd.read_parquet(labels_path)
            logger.info(f"Loaded {len(df)} labels for {dataset_name}")
            return df

        # Show available datasets
        available = get_available_datasets()
        logger.warning(f"No labels found for '{dataset_name}'")
        logger.info(f"Available datasets: {list(available.keys())}")
        raise FileNotFoundError(f"No labels found for {dataset_name}")

    raise FileNotFoundError("No labels found")


def convert_to_osm(
    input_path: Path,
    output_path: Path,
    connectors_path: Path | None = None,
    source_tag: str | None = None,
) -> None:
    """Convert parquet to OSM format.

    Args:
        source_tag: If provided, use 'matcher:{source_tag}:id' tag (e.g., 'ref' or 'tgt')
    """
    cmd = [
        "python",
        str(Path(__file__).parent / "convert_to_osm.py"),
        str(input_path),
        "-o",
        str(output_path),
    ]
    if connectors_path:
        cmd.extend(["--connectors", str(connectors_path)])
    if source_tag:
        cmd.extend(["--source-tag", source_tag])

    logger.info(f"Converting {input_path.name} to OSM (source={source_tag or 'default'})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        raise RuntimeError(f"Failed to convert {input_path}")


def run_hootenanny_conflate(
    reference_osm: Path,
    target_osm: Path,
    output_osm: Path,
    hoot_dir: Path,
) -> None:
    """Run Hootenanny conflation."""
    import shutil

    # Copy files to hootenanny data dir
    hoot_data = hoot_dir / "data"
    hoot_data.mkdir(exist_ok=True)

    ref_dest = hoot_data / reference_osm.name
    tgt_dest = hoot_data / target_osm.name
    out_dest = hoot_data / output_osm.name

    shutil.copy2(reference_osm, ref_dest)
    shutil.copy2(target_osm, tgt_dest)

    logger.info("Running Hootenanny conflation...")

    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "core-services",
        HOOT_BIN,
        "conflate",
        # Use only highway matching
        "-D",
        "match.creators=HighwayMatchCreator",
        "-D",
        "merger.creators=HighwayMergerCreator",
        # Input/output files
        f"/var/lib/hootenanny/data/{reference_osm.name}",
        f"/var/lib/hootenanny/data/{target_osm.name}",
        f"/var/lib/hootenanny/data/{output_osm.name}",
    ]

    # Stream output in real-time for progress visibility
    process = subprocess.Popen(
        cmd,
        cwd=hoot_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
    )

    stderr_lines = []
    for line in process.stdout:
        line = line.rstrip()
        if line:
            # Show progress lines (STATUS) in real-time
            if "STATUS" in line or "ERROR" in line or "WARN" in line:
                logger.info(f"  {line}")
            stderr_lines.append(line)

    return_code = process.wait()

    # Check if output was created
    if not out_dest.exists():
        logger.error("Hootenanny failed to create output file")
        logger.error(f"Return code: {return_code}")
        if stderr_lines:
            logger.error("Output (last 20 lines):")
            for line in stderr_lines[-20:]:
                logger.error(f"  {line}")
        raise RuntimeError("Hootenanny conflation failed - no output created")

    # Copy result back
    shutil.copy2(out_dest, output_osm)
    logger.info(f"Conflation complete: {output_osm}")


def run_hootenanny_score_matches(
    reference_osm: Path,
    target_osm: Path,
    output_json: Path,
    hoot_dir: Path,
    threads: int | None = None,
) -> list[tuple[str, str, float]]:
    """Run Hootenanny score-matches to get match pairs with scores.

    This is more useful than conflate for benchmarking because it outputs
    the actual match decisions before merging.

    Returns:
        List of (ref_id, target_id, score) tuples
    """
    import json
    import os
    import shutil

    hoot_data = hoot_dir / "data"
    hoot_data.mkdir(exist_ok=True)

    ref_dest = hoot_data / reference_osm.name
    tgt_dest = hoot_data / target_osm.name
    out_dest = hoot_data / output_json.name

    shutil.copy2(reference_osm, ref_dest)
    shutil.copy2(target_osm, tgt_dest)

    if threads is None:
        threads = max(2, (os.cpu_count() or 4) // 2)

    logger.info(f"Running Hootenanny score-matches with {threads} threads...")
    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "core-services",
        HOOT_BIN,
        "score-matches",
        "-D",
        "match.creators=HighwayMatchCreator",
        "-D",
        f"job.thread.count={threads}",
        f"/var/lib/hootenanny/data/{reference_osm.name}",
        f"/var/lib/hootenanny/data/{target_osm.name}",
        f"/var/lib/hootenanny/data/{output_json.name}",
    ]

    result = subprocess.run(cmd, cwd=hoot_dir, capture_output=True, text=True)

    if result.returncode != 0:
        # score-matches might not exist in all versions, fall back
        logger.warning(f"score-matches failed: {result.stderr}")
        return []

    shutil.copy2(out_dest, output_json)

    # Parse the JSON output
    matches = []
    try:
        with open(output_json) as f:
            data = json.load(f)
            for match in data.get("matches", []):
                ref_id = match.get("id1") or match.get("ref_id")
                tgt_id = match.get("id2") or match.get("target_id")
                score = match.get("score", 1.0)
                if ref_id and tgt_id:
                    matches.append((str(ref_id), str(tgt_id), float(score)))
    except Exception as e:
        logger.warning(f"Failed to parse score-matches output: {e}")

    logger.info(f"score-matches found {len(matches)} potential matches")
    return matches


def extract_matches_from_conflated(
    conflated_osm: Path,
    reference_osm: Path,
    target_osm: Path,
) -> list[tuple[str, str, str]]:
    """Extract match pairs from Hootenanny's conflated output.

    Looks for:
    1. Review relations with member references
    2. hoot:review:* tags on features
    3. hoot:status tags (1=ref, 2=target, 3=merged)

    Returns:
        List of (reference_id, target_id, match_type) tuples
    """
    matches = []

    # Parse the conflated output
    tree = ET.parse(conflated_osm)
    root = tree.getroot()

    # Build ID maps from originals (osm_id -> matcher:id)
    ref_id_map = _build_id_map(reference_osm, source_tag="ref")
    tgt_id_map = _build_id_map(target_osm, source_tag="tgt")

    # Build sets for fast lookup
    ref_matcher_ids = set(ref_id_map.values())
    tgt_matcher_ids = set(tgt_id_map.values())

    # Look for review relations (Hootenanny creates these for uncertain matches)
    for relation in root.findall(".//relation"):
        tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}

        if tags.get("type") == "review" or "hoot:review" in tags:
            members = relation.findall("member")
            ref_ids = []
            tgt_ids = []

            for member in members:
                member_ref = member.get("ref")

                # Find the original ID for this member
                # We need to look up the way/node in the conflated file
                member_elem = root.find(f".//{member.get('type')}[@id='{member_ref}']")
                if member_elem is not None:
                    member_tags = {t.get("k"): t.get("v") for t in member_elem.findall("tag")}
                    status = member_tags.get("hoot:status")

                    # Look for our matcher tags (new format: matcher_ref_* or matcher_tgt_*)
                    for k, v in member_tags.items():
                        if not v:
                            continue
                        if k.startswith("matcher_ref_"):
                            ref_ids.append(v)
                        elif k.startswith("matcher_tgt_"):
                            tgt_ids.append(v)

                    # Fallback to old matcher:id format with hoot:status
                    if not ref_ids and not tgt_ids:
                        orig_id = member_tags.get("matcher:id")
                        if orig_id:
                            if status == "1":
                                ref_ids.append(orig_id)
                            elif status == "2":
                                tgt_ids.append(orig_id)

            # Create match pairs from review relation
            for ref_id in ref_ids:
                for tgt_id in tgt_ids:
                    matches.append((ref_id, tgt_id, "review"))

    # Look for merged ways with matcher_ref_* and matcher_tgt_* tags
    # Format: matcher_ref_<sanitized_id> = <original_id>
    for way in root.findall(".//way"):
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}

        ref_ids_found = []
        tgt_ids_found = []

        for k, v in tags.items():
            if not v:
                continue
            # New format: matcher_ref_xxx = original_id, matcher_tgt_xxx = original_id
            if k.startswith("matcher_ref_"):
                ref_ids_found.append(v)  # Value is the original ID
            elif k.startswith("matcher_tgt_"):
                tgt_ids_found.append(v)
            # Old format fallback
            elif k == "matcher:id":
                if v in ref_matcher_ids:
                    ref_ids_found.append(v)
                elif v in tgt_matcher_ids:
                    tgt_ids_found.append(v)

        # If we found IDs from both sources, this is a merge
        if ref_ids_found and tgt_ids_found:
            for ref_id in ref_ids_found:
                for tgt_id in tgt_ids_found:
                    matches.append((ref_id, tgt_id, "merge"))
            continue

        # Check for Hootenanny's source ID tags
        source1_id = tags.get("hoot:source:id:1") or tags.get("source:id:1")
        source2_id = tags.get("hoot:source:id:2") or tags.get("source:id:2")
        if source1_id and source2_id:
            matches.append((source1_id, source2_id, "merge"))

    logger.info(f"Extracted {len(matches)} match pairs from conflated output")
    return matches


def _build_id_map(osm_path: Path, source_tag: str | None = None) -> dict[str, str]:
    """Build map from OSM way ID to original matcher ID.

    Args:
        source_tag: If provided, look for values with '{source_tag}:' prefix
    """
    id_map = {}
    tree = ET.parse(osm_path)
    root = tree.getroot()

    value_prefix = f"{source_tag}:" if source_tag else None

    for way in root.findall(".//way"):
        way_id = way.get("id")
        for tag in way.findall("tag"):
            if tag.get("k") == "matcher:id":
                v = tag.get("v")
                # New format: matcher:id = ref:<id> or tgt:<id>
                if value_prefix and v.startswith(value_prefix):
                    id_map[way_id] = v[len(value_prefix) :]
                elif not value_prefix:
                    # Extract ID without prefix for general lookup
                    if v.startswith("ref:") or v.startswith("tgt:"):
                        id_map[way_id] = v.split(":", 1)[1]
                    else:
                        id_map[way_id] = v
                break

    return id_map


def extract_matches_via_diff(
    reference_osm: Path,
    target_osm: Path,
    conflated_osm: Path,
    hoot_dir: Path,
) -> list[tuple[str, str]]:
    """Alternative: use hoot diff to find what was matched.

    This compares the conflated output against the reference to see
    what features were modified (indicating they were merged with target).
    """
    # This is a fallback approach - parsing the conflated output directly is preferred
    pass


def compute_metrics(
    predicted_matches: list[tuple[str, str, str]],
    labels: pd.DataFrame,
) -> dict:
    """Compare predicted matches against ground truth labels.

    Args:
        predicted_matches: List of (ref_id, target_id, match_type) from Hootenanny
        labels: DataFrame with gers_id, target_id, label columns

    Returns:
        Dictionary with precision, recall, F1, etc.
    """
    # Convert labels to sets for lookup
    true_matches = set()
    true_non_matches = set()

    for _, row in labels.iterrows():
        pair = (str(row["gers_id"]), str(row["target_id"]))
        if row["label"] == "match":
            true_matches.add(pair)
        elif row["label"] == "no_match":
            true_non_matches.add(pair)
        # Skip 'unsure' labels

    # Convert predictions to set
    predicted_set = {(ref_id, tgt_id) for ref_id, tgt_id, _ in predicted_matches}

    # Calculate metrics
    true_positives = len(predicted_set & true_matches)
    false_positives = len(predicted_set & true_non_matches)
    false_negatives = len(true_matches - predicted_set)

    # Matches that weren't in our labeled set (can't evaluate)
    unlabeled_predictions = len(predicted_set - true_matches - true_non_matches)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "unlabeled_predictions": unlabeled_predictions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_labeled_matches": len(true_matches),
        "total_labeled_non_matches": len(true_non_matches),
        "total_predictions": len(predicted_set),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hootenanny against ground truth labels")
    parser.add_argument(
        "--dataset",
        help="Dataset name (must have labels). Use --list to see available.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets with labels",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Override reference parquet file",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="Override target parquet file",
    )
    parser.add_argument(
        "--hoot-dir",
        type=Path,
        default=Path(__file__).parents[1].parent / "hootenanny",
        help="Path to hootenanny repo (default: ../hootenanny)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "benchmark",
        help="Directory for intermediate files",
    )
    parser.add_argument(
        "--skip-conflate",
        action="store_true",
        help="Skip conflation, use existing output",
    )

    args = parser.parse_args()

    if args.list:
        datasets = get_available_datasets()
        print("Available datasets with labels:")
        for ds, count in sorted(datasets.items()):
            print(f"  - {ds}: {count} labels")
        return

    if not args.dataset:
        parser.error("--dataset is required (use --list to see available)")

    # Setup paths
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels = load_labels(args.dataset)

    # Find data files using same logic as labeling app
    if args.reference and args.target:
        ref_segments = args.reference
        tgt_segments = args.target
        ref_connectors = None
    else:
        ref_segments, tgt_segments, ref_connectors = get_dataset_files(args.dataset)

    logger.info(f"Reference: {ref_segments}")
    logger.info(f"Target: {tgt_segments}")

    # Convert to OSM
    ref_osm = args.output_dir / f"{args.dataset}_reference.osm"
    tgt_osm = args.output_dir / f"{args.dataset}_target.osm"
    out_osm = args.output_dir / f"{args.dataset}_conflated.osm"

    matches = []

    if not args.skip_conflate:
        # Skip connectors - shared connector nodes cause Hootenanny LinearSnapMerger bug
        # "No node ID specified for RemoveNodeByEid" during merge phase
        convert_to_osm(ref_segments, ref_osm, connectors_path=None, source_tag="ref")
        convert_to_osm(tgt_segments, tgt_osm, source_tag="tgt")

        # Run conflation with review mode to preserve match relationships
        run_hootenanny_conflate(ref_osm, tgt_osm, out_osm, args.hoot_dir)
        matches = extract_matches_from_conflated(out_osm, ref_osm, tgt_osm)
    else:
        if not out_osm.exists():
            logger.error(f"--skip-conflate specified but {out_osm} doesn't exist")
            sys.exit(1)
        matches = extract_matches_from_conflated(out_osm, ref_osm, tgt_osm)

    if not matches:
        logger.warning("No matches extracted! Hootenanny output may need different parsing.")
        logger.info("Checking conflated output structure...")

        # Debug: show what's in the output
        tree = ET.parse(out_osm)
        root = tree.getroot()

        status_counts = defaultdict(int)
        for way in root.findall(".//way"):
            for tag in way.findall("tag"):
                if tag.get("k") == "hoot:status":
                    status_counts[tag.get("v")] += 1

        logger.info(f"Status distribution in output: {dict(status_counts)}")

        # Try alternative extraction
        logger.info("Attempting alternative match extraction...")
        matches = extract_matches_alternative(out_osm, ref_osm, tgt_osm)

    # Compute metrics
    metrics = compute_metrics(matches, labels)

    print("\n" + "=" * 60)
    print(f"Hootenanny Benchmark Results: {args.dataset}")
    print("=" * 60)
    print(f"Total labeled matches:     {metrics['total_labeled_matches']}")
    print(f"Total labeled non-matches: {metrics['total_labeled_non_matches']}")
    print(f"Total predictions:         {metrics['total_predictions']}")
    print()
    print(f"True Positives:   {metrics['true_positives']}")
    print(f"False Positives:  {metrics['false_positives']}")
    print(f"False Negatives:  {metrics['false_negatives']}")
    print(f"Unlabeled preds:  {metrics['unlabeled_predictions']}")
    print()
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("=" * 60)


def extract_matches_alternative(
    conflated_osm: Path,
    reference_osm: Path,
    target_osm: Path,
) -> list[tuple[str, str, str]]:
    """Alternative extraction by comparing IDs between input and output.

    If a reference feature and target feature got merged, the output will have
    a way that contains geometry/tags from both. We can detect this by looking
    for ways in the output that have matcher:id matching the target but geometry
    similar to reference (or vice versa).
    """
    matches = []

    # Build maps of original features
    ref_ids = _get_all_matcher_ids(reference_osm, source_tag="ref")
    tgt_ids = _get_all_matcher_ids(target_osm, source_tag="tgt")
    out_ids = _get_all_matcher_ids(conflated_osm)  # Check all ID tags in output

    # IDs that were in input but not in output (merged away)
    ref_merged = ref_ids - out_ids
    tgt_merged = tgt_ids - out_ids

    logger.info(f"Reference IDs: {len(ref_ids)} total, {len(ref_merged)} merged away")
    logger.info(f"Target IDs: {len(tgt_ids)} total, {len(tgt_merged)} merged away")
    logger.info(f"Output IDs: {len(out_ids)} total")

    # Parse conflated output
    tree = ET.parse(conflated_osm)
    root = tree.getroot()

    # Look for ways that appear to be merged
    for way in root.findall(".//way"):
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}

        # Hootenanny may combine tags - look for patterns
        # Sometimes it creates alt_id or source:id tags
        all_ids = []
        for k, v in tags.items():
            if "id" in k.lower() and v:
                all_ids.append(v)

        # Check which source each ID came from
        found_ref = [id for id in all_ids if id in ref_ids]
        found_tgt = [id for id in all_ids if id in tgt_ids]

        if found_ref and found_tgt:
            for ref_id in found_ref:
                for tgt_id in found_tgt:
                    matches.append((ref_id, tgt_id, "merge"))

    # If no direct matches found, report stats about what was merged
    if not matches:
        # The merged features indicate matches happened, but we can't determine pairs
        # without geometry comparison (which is expensive)
        logger.warning(
            f"Cannot extract exact match pairs from conflated output. "
            f"Hootenanny merged {len(ref_merged)} ref + {len(tgt_merged)} target features, "
            f"but doesn't preserve pair information in output."
        )

    logger.info(f"Alternative extraction found {len(matches)} matches")
    return matches


def _get_all_matcher_ids(osm_path: Path, source_tag: str | None = None) -> set[str]:
    """Get all matcher ID values from an OSM file.

    Args:
        source_tag: If provided, look for 'matcher:{source_tag}:<id>' keys
    """
    ids = set()
    tree = ET.parse(osm_path)
    root = tree.getroot()

    prefix = f"matcher:{source_tag}:" if source_tag else None

    for way in root.findall(".//way"):
        for tag in way.findall("tag"):
            k = tag.get("k")
            # New format: matcher:ref:<id> = 1 (ID is in the key)
            if prefix and k.startswith(prefix):
                extracted_id = k[len(prefix) :]
                if extracted_id and extracted_id != "id":
                    ids.add(extracted_id)
            # Old format: matcher:id = <id>
            elif k == "matcher:id":
                ids.add(tag.get("v"))

    return ids


if __name__ == "__main__":
    main()
