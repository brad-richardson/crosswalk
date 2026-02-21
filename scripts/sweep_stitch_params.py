#!/usr/bin/env python3
"""Parameter sweep for stitch-level precision/recall tuning.

Runs ML scoring once (expensive), then sweeps optimizer parameters and
post-hoc bridge filters (cheap) to explore precision/recall tradeoffs.

Usage:
    python scripts/sweep_stitch_params.py
    python scripts/sweep_stitch_params.py --dataset us_boston_streets
"""

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

# Add matcher and cbench to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "cbench" / "src"))

from cbench.eval.labels import load_stitch_labels
from cbench.eval.stitch_metrics import StitchEvalResult, evaluate_stitch_groups

from matcher.config import DEFAULT_SNAP_TOLERANCE_M
from matcher.matching.optimizer import optimize_matches_with_grouping
from matcher.matching.types import MatchDecision, MatchResult
from matcher.pipeline.runner import score_candidates_from_geodataframes
from matcher.utils.geometry import filter_to_linestrings

# Default paths (relative to project root)
DEFAULT_REF_PATH = Path("data/raw/us_boston_streets_overture_segments_v1.0.parquet")
DEFAULT_TARGET_PATH = Path("data/raw/us_boston_streets_v1.0.parquet")
DEFAULT_STITCH_LABELS_DIR = Path("labels/stitching")
DEFAULT_DATASET = "us_boston_streets"

# Parameter grid
MIN_CONFIDENCE_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
CONTIGUITY_TOLERANCE_VALUES = [3.0, 5.0, 7.0]
BRIDGE_MIN_CONFIDENCE_VALUES = [None, 0.3, 0.5, 0.7]

# Production defaults for highlighting
PROD_MIN_CONFIDENCE = 0.1
PROD_CONTIGUITY_TOLERANCE = DEFAULT_SNAP_TOLERANCE_M  # 5.0
PROD_BRIDGE_MIN_CONFIDENCE = 0.5


@dataclass
class SweepResult:
    """Result from one parameter combination."""

    min_confidence: float
    contiguity_tolerance: float
    bridge_min_confidence: float | None
    n_bridge_edges: int
    n_match: int
    n_review: int
    stitch_precision: float
    stitch_recall: float
    stitch_f1: float
    extra_edges: int
    groups_evaluated: int
    is_default: bool
    is_pareto_optimal: bool = False


def score_candidates(
    ref_path: Path,
    target_path: Path,
) -> tuple[list[MatchResult], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Phase 1: Load data, project, and score all candidates (runs once).

    Returns:
        Tuple of (match_results, reference_projected, target_projected)
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Scoring candidates (this is the expensive step)")
    logger.info("=" * 60)

    t0 = time.perf_counter()

    reference = gpd.read_parquet(ref_path)
    target = gpd.read_parquet(target_path)

    # Same preprocessing as runner.py
    reference = reference[~reference.geometry.isna()]
    target = target[~target.geometry.isna()]
    reference = filter_to_linestrings(reference, source_name="reference")
    target = filter_to_linestrings(target, source_name="target")

    logger.info(f"  Reference: {len(reference)} features")
    logger.info(f"  Target: {len(target)} features")

    results, projection_result = score_candidates_from_geodataframes(
        reference=reference,
        target=target,
    )

    elapsed = time.perf_counter() - t0
    logger.info(f"  Scored {len(results)} candidates in {elapsed:.1f}s")

    return results, projection_result.reference, projection_result.target


def build_bridge_df(
    optimized: list[MatchResult],
    bridge_min_confidence: float | None = None,
) -> pd.DataFrame:
    """Build bridge DataFrame from optimizer output.

    Mimics generate_bridge_file() filtering: includes MATCH + REVIEW,
    excludes NO_MATCH. Uses ref_id/target_id column names for
    evaluate_stitch_groups() compatibility.
    """
    records = []
    for r in optimized:
        if r.decision == MatchDecision.NO_MATCH:
            continue
        if bridge_min_confidence is not None and r.confidence < bridge_min_confidence:
            continue
        records.append(
            {
                "ref_id": str(r.ref_id),
                "target_id": str(r.target_id),
                "confidence": float(r.confidence),
            }
        )

    if not records:
        return pd.DataFrame(columns=["ref_id", "target_id", "confidence"])
    return pd.DataFrame(records)


def mark_pareto_optimal(results: list[SweepResult]) -> None:
    """Mark Pareto-optimal configs (no other config is better on both P and R)."""
    for i, a in enumerate(results):
        dominated = False
        for j, b in enumerate(results):
            if i == j:
                continue
            # b dominates a if b is >= on both metrics and strictly > on at least one
            if (
                b.stitch_precision >= a.stitch_precision
                and b.stitch_recall >= a.stitch_recall
                and (b.stitch_precision > a.stitch_precision or b.stitch_recall > a.stitch_recall)
            ):
                dominated = True
                break
        results[i].is_pareto_optimal = not dominated


def run_sweep(
    results: list[MatchResult],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    stitch_labels: pd.DataFrame,
) -> list[SweepResult]:
    """Phase 2: Sweep optimizer and bridge filter parameters."""
    logger.info("=" * 60)
    logger.info("Phase 2: Parameter sweep")
    logger.info("=" * 60)

    optimizer_combos = list(
        itertools.product(
            MIN_CONFIDENCE_VALUES,
            CONTIGUITY_TOLERANCE_VALUES,
        )
    )
    total_evals = len(optimizer_combos) * len(BRIDGE_MIN_CONFIDENCE_VALUES)
    logger.info(
        f"  {len(optimizer_combos)} optimizer configs x "
        f"{len(BRIDGE_MIN_CONFIDENCE_VALUES)} bridge filters = "
        f"{total_evals} evaluations"
    )

    all_results: list[SweepResult] = []
    t0_sweep = time.perf_counter()

    for i, (min_conf, contiguity_tol) in enumerate(optimizer_combos, 1):
        logger.info(
            f"  [{i}/{len(optimizer_combos)}] "
            f"min_confidence={min_conf}, contiguity={contiguity_tol}m"
        )

        t0 = time.perf_counter()
        optimized = optimize_matches_with_grouping(
            results,
            reference=reference,
            target=target,
            min_confidence=min_conf,
            contiguity_tolerance=contiguity_tol,
            ref_id_column="id",
            target_id_column="id",
        )
        opt_time = time.perf_counter() - t0

        n_match = sum(1 for r in optimized if r.decision == MatchDecision.MATCH)
        n_review = sum(1 for r in optimized if r.decision == MatchDecision.REVIEW)
        logger.info(
            f"    {len(optimized)} results ({n_match} match, {n_review} review) in {opt_time:.2f}s"
        )

        for bridge_min_conf in BRIDGE_MIN_CONFIDENCE_VALUES:
            bridge_df = build_bridge_df(optimized, bridge_min_conf)

            if bridge_df.empty:
                eval_result = StitchEvalResult(
                    groups_evaluated=0,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    total_curated_edges=0,
                    total_extra_edges=0,
                )
            else:
                eval_result = evaluate_stitch_groups(bridge_df, stitch_labels)

            is_default = (
                min_conf == PROD_MIN_CONFIDENCE
                and contiguity_tol == PROD_CONTIGUITY_TOLERANCE
                and bridge_min_conf == PROD_BRIDGE_MIN_CONFIDENCE
            )

            all_results.append(
                SweepResult(
                    min_confidence=min_conf,
                    contiguity_tolerance=contiguity_tol,
                    bridge_min_confidence=bridge_min_conf,
                    n_bridge_edges=len(bridge_df),
                    n_match=n_match,
                    n_review=n_review,
                    stitch_precision=eval_result.precision,
                    stitch_recall=eval_result.recall,
                    stitch_f1=eval_result.f1,
                    extra_edges=eval_result.total_extra_edges,
                    groups_evaluated=eval_result.groups_evaluated,
                    is_default=is_default,
                )
            )

    sweep_time = time.perf_counter() - t0_sweep
    logger.info(f"  Sweep complete: {total_evals} evaluations in {sweep_time:.1f}s")

    mark_pareto_optimal(all_results)
    return all_results


def print_results(results: list[SweepResult]) -> None:
    """Print formatted results table sorted by F1."""
    print("\n" + "=" * 110)
    print("STITCH PARAMETER SWEEP RESULTS")
    print("=" * 110)

    # Sort by F1 descending
    sorted_results = sorted(results, key=lambda r: -r.stitch_f1)

    # Header
    print(
        f"\n{'':>3} {'min_conf':>8} {'cont_m':>6} {'br_conf':>7} "
        f"{'P':>7} {'R':>7} {'F1':>7} {'Extra':>5} "
        f"{'Bridge':>6} {'Match':>5} {'Revw':>5} {'Flags':>12}"
    )
    print("-" * 110)

    for r in sorted_results:
        br_conf_str = f"{r.bridge_min_confidence:.1f}" if r.bridge_min_confidence else "None"

        flags = []
        if r.is_default:
            flags.append("DEFAULT")
        if r.is_pareto_optimal:
            flags.append("PARETO")
        flag_str = " ".join(flags)

        marker = ">>>" if r.is_default else "   "

        print(
            f"{marker} {r.min_confidence:>8.2f} {r.contiguity_tolerance:>6.1f} "
            f"{br_conf_str:>7} "
            f"{r.stitch_precision:>7.4f} {r.stitch_recall:>7.4f} "
            f"{r.stitch_f1:>7.4f} {r.extra_edges:>5d} "
            f"{r.n_bridge_edges:>6d} {r.n_match:>5d} {r.n_review:>5d} {flag_str:>12}"
        )

    # Summary
    print("\n" + "-" * 110)

    default = next((r for r in results if r.is_default), None)
    if default:
        print(
            f"\nCurrent default: P={default.stitch_precision:.4f} "
            f"R={default.stitch_recall:.4f} F1={default.stitch_f1:.4f} "
            f"Extra={default.extra_edges}"
        )

    best_f1 = sorted_results[0]
    print(
        f"Best F1:         P={best_f1.stitch_precision:.4f} "
        f"R={best_f1.stitch_recall:.4f} F1={best_f1.stitch_f1:.4f} "
        f"Extra={best_f1.extra_edges} "
        f"(min_conf={best_f1.min_confidence}, "
        f"cont_tol={best_f1.contiguity_tolerance}, "
        f"br_conf={best_f1.bridge_min_confidence})"
    )

    # Best precision while keeping recall >= 0.9
    high_recall = [r for r in sorted_results if r.stitch_recall >= 0.9]
    if high_recall:
        best_p_hr = max(high_recall, key=lambda r: r.stitch_precision)
        print(
            f"Best P (R>=0.9): P={best_p_hr.stitch_precision:.4f} "
            f"R={best_p_hr.stitch_recall:.4f} F1={best_p_hr.stitch_f1:.4f} "
            f"Extra={best_p_hr.extra_edges} "
            f"(min_conf={best_p_hr.min_confidence}, "
            f"cont_tol={best_p_hr.contiguity_tolerance}, "
            f"br_conf={best_p_hr.bridge_min_confidence})"
        )

    # Pareto frontier
    pareto = [r for r in sorted_results if r.is_pareto_optimal]
    print(f"\nPareto-optimal configs ({len(pareto)}):")
    for r in sorted(pareto, key=lambda r: -r.stitch_recall):
        br_str = f"{r.bridge_min_confidence:.1f}" if r.bridge_min_confidence else "None"
        print(
            f"  min_conf={r.min_confidence:<5} cont_tol={r.contiguity_tolerance:<5} "
            f"br_conf={br_str:<5} -> "
            f"P={r.stitch_precision:.4f} R={r.stitch_recall:.4f} "
            f"F1={r.stitch_f1:.4f} Extra={r.extra_edges}"
        )

    print("\n" + "=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for stitch precision/recall tuning"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REF_PATH,
        help=f"Reference parquet (default: {DEFAULT_REF_PATH})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET_PATH,
        help=f"Target parquet (default: {DEFAULT_TARGET_PATH})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name for stitch labels (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--stitch-labels",
        type=Path,
        default=DEFAULT_STITCH_LABELS_DIR,
        help=f"Stitch labels directory (default: {DEFAULT_STITCH_LABELS_DIR})",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.reference.exists():
        logger.error(f"Reference file not found: {args.reference}")
        sys.exit(1)
    if not args.target.exists():
        logger.error(f"Target file not found: {args.target}")
        sys.exit(1)

    # Load stitch labels
    stitch_labels = load_stitch_labels(args.stitch_labels, args.dataset)
    if stitch_labels is None:
        logger.error(f"No stitch labels found for dataset '{args.dataset}' in {args.stitch_labels}")
        sys.exit(1)
    logger.info(f"Loaded {len(stitch_labels)} stitch label groups")

    # Phase 1: Score candidates (expensive, runs once)
    results, reference, target = score_candidates(args.reference, args.target)

    if not results:
        logger.error("No candidates scored - nothing to sweep")
        sys.exit(1)

    # Phase 2: Sweep parameters (cheap, runs many times)
    sweep_results = run_sweep(results, reference, target, stitch_labels)

    # Phase 3: Print results
    print_results(sweep_results)


if __name__ == "__main__":
    main()
