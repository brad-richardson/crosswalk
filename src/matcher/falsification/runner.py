"""Falsification runner - orchestrates falsification tests on match results.

Runs registered falsification tests on bridge file matches and outputs
filtered results with a detailed report.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from loguru import logger

# Import tests to register them
from . import tests as _tests  # noqa: F401

del _tests  # Avoid unused variable warning; import is only for side effects

from .base import (
    FalsificationOutcome,
    FalsificationResult,
    FalsificationTest,
    MatchContext,
    get_registered_tests,
    get_test,
)


@dataclass
class FalsificationReport:
    """Report from running falsification tests."""

    total_matches: int
    passed: int
    failed: int
    warned: int
    skipped: int

    # Per-test breakdown
    test_results: dict[str, dict[str, int]] = field(default_factory=dict)

    # Details of failed matches
    failed_matches: list[dict[str, Any]] = field(default_factory=list)

    # Details of warned matches
    warned_matches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fail_rate(self) -> float:
        """Percentage of matches that failed."""
        if self.total_matches == 0:
            return 0.0
        return self.failed / self.total_matches

    @property
    def warn_rate(self) -> float:
        """Percentage of matches that warned."""
        if self.total_matches == 0:
            return 0.0
        return self.warned / self.total_matches

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "total_matches": self.total_matches,
            "passed": self.passed,
            "failed": self.failed,
            "warned": self.warned,
            "skipped": self.skipped,
            "fail_rate": round(self.fail_rate, 4),
            "warn_rate": round(self.warn_rate, 4),
            "test_results": self.test_results,
            "failed_matches": self.failed_matches,
            "warned_matches": self.warned_matches,
        }


def run_falsification(
    bridge_path: Path,
    ref_path: Path,
    target_path: Path,
    test_names: list[str] | None = None,
    output_path: Path | None = None,
    report_only: bool = False,
) -> tuple[gpd.GeoDataFrame | None, FalsificationReport]:
    """Run falsification tests on a bridge file.

    Loads the bridge file, reference, and target datasets, then runs
    specified (or all) falsification tests on each match. Outputs a
    filtered bridge file with failed matches removed.

    Args:
        bridge_path: Path to bridge parquet file with matches
        ref_path: Path to reference parquet file (Overture)
        target_path: Path to target parquet file
        test_names: List of test names to run (None = all registered tests)
        output_path: Path for output filtered bridge file (None = no output)
        report_only: If True, don't filter matches, just generate report

    Returns:
        Tuple of (filtered_gdf, report)
        - filtered_gdf: Bridge file with failed matches removed (None if report_only)
        - report: FalsificationReport with test results
    """
    logger.info(f"Running falsification on {bridge_path}")

    # Load data
    bridge_gdf = gpd.read_parquet(bridge_path)
    ref_gdf = gpd.read_parquet(ref_path)
    target_gdf = gpd.read_parquet(target_path)

    # Ensure consistent CRS (EPSG:4326 for falsification)
    if bridge_gdf.crs is not None and bridge_gdf.crs != "EPSG:4326":
        bridge_gdf = bridge_gdf.to_crs("EPSG:4326")
    if ref_gdf.crs is not None and ref_gdf.crs != "EPSG:4326":
        ref_gdf = ref_gdf.to_crs("EPSG:4326")
    if target_gdf.crs is not None and target_gdf.crs != "EPSG:4326":
        target_gdf = target_gdf.to_crs("EPSG:4326")

    # Get bounding box from combined geometries
    all_geoms = pd.concat([ref_gdf.geometry, target_gdf.geometry], ignore_index=True)
    bounds = all_geoms.total_bounds
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

    # Initialize tests
    if test_names is None:
        test_classes = list(get_registered_tests().values())
    else:
        test_classes = [get_test(name) for name in test_names]

    tests: list[FalsificationTest] = [cls() for cls in test_classes]
    logger.info(f"Running {len(tests)} falsification tests: {[t.name for t in tests]}")

    # Prepare all tests (fetch context data)
    for test in tests:
        logger.info(f"Preparing test: {test.name}")
        test.prepare(bbox)

    # Build lookup indices for geometries
    # Determine ID columns
    ref_id_col = _get_id_column(ref_gdf, "ref")
    target_id_col = _get_id_column(target_gdf, "target")

    ref_lookup = {row[ref_id_col]: row.geometry for _, row in ref_gdf.iterrows()}
    target_lookup = {row[target_id_col]: row.geometry for _, row in target_gdf.iterrows()}

    # Run tests on each match
    results_by_match: dict[int, list[FalsificationResult]] = {}
    failed_indices: set[int] = set()
    warned_indices: set[int] = set()

    # Determine bridge file ID columns
    bridge_ref_col = _get_bridge_ref_column(bridge_gdf)
    bridge_target_col = _get_bridge_target_column(bridge_gdf)

    for idx, row in bridge_gdf.iterrows():
        ref_id = row[bridge_ref_col]
        target_id = row[bridge_target_col]
        confidence = row.get("confidence", 1.0)

        # Get geometries
        ref_geom = ref_lookup.get(ref_id)
        target_geom = target_lookup.get(target_id)

        if ref_geom is None or target_geom is None:
            logger.warning(f"Missing geometry for match {idx}: ref={ref_id}, target={target_id}")
            continue

        # Create match context
        ctx = MatchContext(
            match_id=str(idx),
            ref_id=str(ref_id),
            target_id=str(target_id),
            ref_geom=ref_geom,
            target_geom=target_geom,
            confidence=confidence,
        )

        # Run all tests
        match_results = []
        match_failed = False
        match_warned = False

        for test in tests:
            result = test.test_match(ctx)
            match_results.append(result)

            if result.outcome == FalsificationOutcome.FAIL:
                match_failed = True
            elif result.outcome == FalsificationOutcome.WARN:
                match_warned = True

        results_by_match[idx] = match_results

        if match_failed:
            failed_indices.add(idx)
        elif match_warned:
            warned_indices.add(idx)

    # Build report
    report = _build_report(
        bridge_gdf=bridge_gdf,
        results_by_match=results_by_match,
        failed_indices=failed_indices,
        warned_indices=warned_indices,
        tests=tests,
        bridge_ref_col=bridge_ref_col,
        bridge_target_col=bridge_target_col,
    )

    logger.info(
        f"Falsification complete: {report.passed} passed, {report.failed} failed, "
        f"{report.warned} warned ({report.fail_rate:.2%} fail rate)"
    )

    # Filter output
    if report_only:
        return None, report

    filtered_gdf = bridge_gdf[~bridge_gdf.index.isin(failed_indices)].copy()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_gdf.to_parquet(output_path)
        logger.info(f"Saved filtered bridge file to {output_path}")

    return filtered_gdf, report


def _get_id_column(gdf: gpd.GeoDataFrame, name: str) -> str:
    """Determine the ID column for a GeoDataFrame."""
    for col in ["id", "ID", f"{name}_id"]:
        if col in gdf.columns:
            return col
    if gdf.index.name:
        return gdf.index.name
    raise ValueError(f"Could not determine ID column for {name}")


def _get_bridge_ref_column(gdf: gpd.GeoDataFrame) -> str:
    """Determine the reference ID column in bridge file."""
    for col in ["ref_id", "reference_id", "overture_id"]:
        if col in gdf.columns:
            return col
    raise ValueError("Could not determine reference ID column in bridge file")


def _get_bridge_target_column(gdf: gpd.GeoDataFrame) -> str:
    """Determine the target ID column in bridge file."""
    for col in ["target_id", "local_id"]:
        if col in gdf.columns:
            return col
    raise ValueError("Could not determine target ID column in bridge file")


def _build_report(
    bridge_gdf: gpd.GeoDataFrame,
    results_by_match: dict[int, list[FalsificationResult]],
    failed_indices: set[int],
    warned_indices: set[int],
    tests: list[FalsificationTest],
    bridge_ref_col: str,
    bridge_target_col: str,
) -> FalsificationReport:
    """Build falsification report from results."""
    total = len(bridge_gdf)
    failed = len(failed_indices)
    warned = len(warned_indices)
    passed = total - failed - warned

    # Count skipped (matches not in results_by_match)
    skipped = total - len(results_by_match)

    # Per-test breakdown
    test_results: dict[str, dict[str, int]] = {}
    for test in tests:
        test_results[test.name] = {
            "pass": 0,
            "fail": 0,
            "warn": 0,
            "skip": 0,
        }

    for match_results in results_by_match.values():
        for result in match_results:
            outcome_key = result.outcome.value
            if result.test_name in test_results:
                test_results[result.test_name][outcome_key] += 1

    # Failed match details
    failed_matches = []
    for idx in failed_indices:
        row = bridge_gdf.loc[idx]
        match_results = results_by_match.get(idx, [])
        fail_reasons = [r for r in match_results if r.outcome == FalsificationOutcome.FAIL]

        failed_matches.append(
            {
                "match_index": int(idx),
                "ref_id": str(row[bridge_ref_col]),
                "target_id": str(row[bridge_target_col]),
                "confidence": float(row.get("confidence", 1.0)),
                "fail_reasons": [
                    {"test": r.test_name, "reason": r.reason, "details": r.details}
                    for r in fail_reasons
                ],
            }
        )

    # Warned match details (limit to top 100)
    warned_matches = []
    for idx in list(warned_indices)[:100]:
        row = bridge_gdf.loc[idx]
        match_results = results_by_match.get(idx, [])
        warn_reasons = [r for r in match_results if r.outcome == FalsificationOutcome.WARN]

        warned_matches.append(
            {
                "match_index": int(idx),
                "ref_id": str(row[bridge_ref_col]),
                "target_id": str(row[bridge_target_col]),
                "confidence": float(row.get("confidence", 1.0)),
                "warn_reasons": [
                    {"test": r.test_name, "reason": r.reason, "details": r.details}
                    for r in warn_reasons
                ],
            }
        )

    return FalsificationReport(
        total_matches=total,
        passed=passed,
        failed=failed,
        warned=warned,
        skipped=skipped,
        test_results=test_results,
        failed_matches=failed_matches,
        warned_matches=warned_matches,
    )
