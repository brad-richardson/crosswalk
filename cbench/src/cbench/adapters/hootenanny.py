"""Hootenanny tool adapter."""

from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from loguru import logger

from cbench.adapters.base import EvalMode, ToolOutput
from cbench.convert.osm import convert_parquet_to_osm

HOOT_BIN = "/var/lib/hootenanny/bin/hoot"


# ---------------------------------------------------------------------------
# Docker lifecycle
# ---------------------------------------------------------------------------


def _find_hoot_dir(hoot_dir: Path | None = None) -> Path:
    """Locate the Hootenanny repo directory."""
    if hoot_dir:
        return hoot_dir
    env = os.environ.get("HOOTENANNY_DIR")
    if env:
        return Path(env)
    # Default: sibling directory to CWD
    return Path.cwd().parent / "hootenanny"


def is_compose_running(hoot_dir: Path) -> bool:
    """Check if Hootenanny docker-compose services are running."""
    if not hoot_dir.exists():
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "-q", "core-services"],
            cwd=hoot_dir,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def ensure_compose_running(hoot_dir: Path) -> None:
    """Start Hootenanny docker-compose if not already running."""
    if not hoot_dir.exists():
        raise FileNotFoundError(
            f"Hootenanny repo not found at {hoot_dir}. "
            "Clone it with: git clone https://github.com/ngageoint/hootenanny.git"
        )
    if not is_compose_running(hoot_dir):
        logger.info("Starting Hootenanny services...")
        subprocess.run(["make", "-f", "Makefile.docker", "up"], cwd=hoot_dir, check=True)


# ---------------------------------------------------------------------------
# Match extraction from conflated OSM output
# ---------------------------------------------------------------------------


def _build_id_map(osm_path: Path, source_tag: str | None = None) -> dict[str, str]:
    """Build map from OSM way ID to original matcher ID."""
    id_map = {}
    tree = ET.parse(osm_path)
    root = tree.getroot()
    key_prefix = f"matcher_{source_tag}_" if source_tag else None

    for way in root.findall(".//way"):
        way_id = way.get("id")
        for tag in way.findall("tag"):
            k = tag.get("k")
            v = tag.get("v")
            if (key_prefix and k and k.startswith(key_prefix) and v) or (k == "matcher:id" and v):
                id_map[way_id] = v
                break

    return id_map


def _get_all_matcher_ids(osm_path: Path, source_tag: str | None = None) -> set[str]:
    """Get all matcher ID values from an OSM file."""
    ids: set[str] = set()
    tree = ET.parse(osm_path)
    root = tree.getroot()
    key_prefix = f"matcher_{source_tag}_" if source_tag else None

    for way in root.findall(".//way"):
        for tag in way.findall("tag"):
            k = tag.get("k")
            v = tag.get("v")
            if (key_prefix and k and k.startswith(key_prefix) and v) or (k == "matcher:id" and v):
                ids.add(v)

    return ids


def extract_matches_from_conflated(
    conflated_osm: Path,
    reference_osm: Path,
    target_osm: Path,
) -> list[tuple[str, str, str]]:
    """Extract match pairs from Hootenanny's conflated output.

    Looks for:
    1. Review relations with member references
    2. Merged ways with matcher_ref_* and matcher_tgt_* tags
    3. hoot:source:id tags on merged features

    Returns:
        List of (reference_id, target_id, match_type) tuples.
    """
    matches = []
    tree = ET.parse(conflated_osm)
    root = tree.getroot()

    ref_id_map = _build_id_map(reference_osm, source_tag="ref")
    tgt_id_map = _build_id_map(target_osm, source_tag="tgt")
    ref_matcher_ids = set(ref_id_map.values())
    tgt_matcher_ids = set(tgt_id_map.values())

    # Review relations
    for relation in root.findall(".//relation"):
        tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
        if tags.get("type") != "review" and "hoot:review" not in tags:
            continue

        members = relation.findall("member")
        ref_ids = []
        tgt_ids = []

        for member in members:
            member_ref = member.get("ref")
            member_elem = root.find(f".//{member.get('type')}[@id='{member_ref}']")
            if member_elem is None:
                continue

            member_tags = {t.get("k"): t.get("v") for t in member_elem.findall("tag")}
            status = member_tags.get("hoot:status")

            for k, v in member_tags.items():
                if not v:
                    continue
                if k.startswith("matcher_ref_"):
                    ref_ids.append(v)
                elif k.startswith("matcher_tgt_"):
                    tgt_ids.append(v)

            if not ref_ids and not tgt_ids:
                orig_id = member_tags.get("matcher:id")
                if orig_id:
                    if status == "1":
                        ref_ids.append(orig_id)
                    elif status == "2":
                        tgt_ids.append(orig_id)

        for ref_id in ref_ids:
            for tgt_id in tgt_ids:
                matches.append((ref_id, tgt_id, "review"))

    # Merged ways
    for way in root.findall(".//way"):
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        ref_ids_found = []
        tgt_ids_found = []

        for k, v in tags.items():
            if not v:
                continue
            if k.startswith("matcher_ref_"):
                ref_ids_found.append(v)
            elif k.startswith("matcher_tgt_"):
                tgt_ids_found.append(v)
            elif k == "matcher:id":
                if v in ref_matcher_ids:
                    ref_ids_found.append(v)
                elif v in tgt_matcher_ids:
                    tgt_ids_found.append(v)

        if ref_ids_found and tgt_ids_found:
            for ref_id in ref_ids_found:
                for tgt_id in tgt_ids_found:
                    matches.append((ref_id, tgt_id, "merge"))
            continue

        source1_id = tags.get("hoot:source:id:1") or tags.get("source:id:1")
        source2_id = tags.get("hoot:source:id:2") or tags.get("source:id:2")
        if source1_id and source2_id:
            matches.append((source1_id, source2_id, "merge"))

    logger.info(f"Extracted {len(matches)} match pairs from conflated output")
    return matches


def extract_matches_alternative(
    conflated_osm: Path,
    reference_osm: Path,
    target_osm: Path,
) -> list[tuple[str, str, str]]:
    """Alternative extraction by comparing IDs between input and output."""
    matches = []

    ref_ids = _get_all_matcher_ids(reference_osm, source_tag="ref")
    tgt_ids = _get_all_matcher_ids(target_osm, source_tag="tgt")

    tree = ET.parse(conflated_osm)
    root = tree.getroot()

    for way in root.findall(".//way"):
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        all_ids = [v for k, v in tags.items() if "id" in k.lower() and v]

        found_ref = [id for id in all_ids if id in ref_ids]
        found_tgt = [id for id in all_ids if id in tgt_ids]

        if found_ref and found_tgt:
            for ref_id in found_ref:
                for tgt_id in found_tgt:
                    matches.append((ref_id, tgt_id, "merge"))

    if not matches:
        ref_merged = ref_ids - _get_all_matcher_ids(conflated_osm)
        tgt_merged = tgt_ids - _get_all_matcher_ids(conflated_osm)
        logger.warning(
            f"Cannot extract exact match pairs. "
            f"Hootenanny merged {len(ref_merged)} ref + {len(tgt_merged)} target features."
        )

    logger.info(f"Alternative extraction found {len(matches)} matches")
    return matches


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class HootAdapter:
    """Adapter for Hootenanny conflation tool.

    Runs full conflation (match + optimize + merge) by default. We need the
    merge step because match results are only observable in the output through
    merged ways (with both ref/tgt provenance tags) and review relations.
    Hootenanny's conflate.match.only=true option skips optimize+merge, but
    also discards match results — so it's not useful for benchmarking.

    The optimization/merge phase can be slow on large datasets (London 873K
    ways took 60+ min). When STITCH/MERGE eval modes are added, this adapter
    can also evaluate merge quality in addition to match identification.

    TODO: Manage Docker lifecycle automatically:
    - Auto-detect hoot_dir (sibling dir, HOOTENANNY_DIR env, or prompt)
    - Start docker compose services if not running, tear down on completion
    - Stream conflation progress (parse STATUS lines for element counts,
      show a progress bar or periodic updates for long-running jobs)
    - Timeout protection for the optimization phase which can hang on
      large datasets
    """

    name: str = "hootenanny"
    eval_mode: EvalMode = EvalMode.PAIR_MATCH

    def run(self, reference: Path, target: Path, output_dir: Path, **kwargs) -> Path:
        """Convert inputs to OSM, run Hootenanny conflation, return output path.

        Args:
            reference: Path to reference parquet.
            target: Path to target parquet.
            output_dir: Working directory for intermediate and output files.
            **kwargs:
                hoot_dir: Path to Hootenanny repo.
                connectors: Path to connectors parquet.
                skip_conflate: If True, skip conflation and reuse existing output.

        Returns:
            Path to conflated OSM output.
        """
        hoot_dir = _find_hoot_dir(kwargs.get("hoot_dir"))
        connectors = kwargs.get("connectors")
        skip_conflate = kwargs.get("skip_conflate", False)

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = target.stem

        ref_osm = output_dir / f"{dataset_name}_reference.osm"
        tgt_osm = output_dir / f"{dataset_name}_target.osm"
        out_osm = output_dir / f"{dataset_name}_conflated.osm"

        if not skip_conflate:
            # Convert parquet -> OSM
            # TODO: Neither input gets proper topology right now:
            # - Reference: connectors are only passed if caller provides them via --opt.
            #   The old benchmark script intentionally skipped them because shared connector
            #   nodes triggered Hootenanny's LinearSnapMerger bug ("No node ID specified
            #   for RemoveNodeByEid"). Needs investigation on newer Hootenanny versions.
            # - Target: most non-Overture datasets lack a connectors column entirely.
            # - Even WITH connectors, non-connector vertices get unique node IDs (see
            #   OSMConverter._create_node TODO). This means Hootenanny sees disconnected
            #   ways at junctions, disabling its graph-based matching algorithms.
            # Net effect: benchmark measures geometry-only matching, not realistic conflation.
            convert_parquet_to_osm(reference, ref_osm, connectors_path=connectors, source_tag="ref")
            convert_parquet_to_osm(target, tgt_osm, source_tag="tgt")

            # Run conflation
            ensure_compose_running(hoot_dir)
            _run_conflate(ref_osm, tgt_osm, out_osm, hoot_dir)
        else:
            if not out_osm.exists():
                raise FileNotFoundError(f"--skip-conflate specified but {out_osm} doesn't exist")

        return out_osm

    def parse_output(self, output_path: Path) -> ToolOutput:
        """Parse Hootenanny conflated OSM output into match pairs.

        Expects ref/target OSM files in same directory (written by run()).
        """
        output_dir = output_path.parent
        dataset_name = output_path.stem.replace("_conflated", "")
        ref_osm = output_dir / f"{dataset_name}_reference.osm"
        tgt_osm = output_dir / f"{dataset_name}_target.osm"

        if not ref_osm.exists() or not tgt_osm.exists():
            raise FileNotFoundError(
                f"Reference/target OSM files not found alongside {output_path}. "
                f"Expected: {ref_osm.name}, {tgt_osm.name}"
            )

        raw_matches = extract_matches_from_conflated(output_path, ref_osm, tgt_osm)

        # Fallback to alternative extraction if primary found nothing
        if not raw_matches:
            logger.warning("Primary extraction found no matches, trying alternative...")
            raw_matches = extract_matches_alternative(output_path, ref_osm, tgt_osm)

        matches = pd.DataFrame(
            [(ref_id, tgt_id, 1.0) for ref_id, tgt_id, _ in raw_matches],
            columns=["ref_id", "target_id", "confidence"],
        )

        match_types = {}
        for _, _, mtype in raw_matches:
            match_types[mtype] = match_types.get(mtype, 0) + 1

        return ToolOutput(
            matches=matches,
            metadata={"match_type_counts": match_types, "total_raw_matches": len(raw_matches)},
        )


def _run_conflate(
    reference_osm: Path,
    target_osm: Path,
    output_osm: Path,
    hoot_dir: Path,
) -> None:
    """Run Hootenanny conflation via Docker compose."""
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
        "-D",
        "match.creators=HighwayMatchCreator",
        "-D",
        "merger.creators=HighwayMergerCreator",
        f"/var/lib/hootenanny/data/{reference_osm.name}",
        f"/var/lib/hootenanny/data/{target_osm.name}",
        f"/var/lib/hootenanny/data/{output_osm.name}",
    ]

    process = subprocess.Popen(
        cmd,
        cwd=hoot_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stderr_lines = []
    for line in process.stdout:
        line = line.rstrip()
        if line:
            if "STATUS" in line or "ERROR" in line or "WARN" in line:
                logger.info(f"  {line}")
            stderr_lines.append(line)

    return_code = process.wait()

    if not out_dest.exists():
        logger.error("Hootenanny failed to create output file")
        logger.error(f"Return code: {return_code}")
        if stderr_lines:
            logger.error("Output (last 20 lines):")
            for line in stderr_lines[-20:]:
                logger.error(f"  {line}")
        raise RuntimeError("Hootenanny conflation failed - no output created")

    shutil.copy2(out_dest, output_osm)
    logger.info(f"Conflation complete: {output_osm}")
