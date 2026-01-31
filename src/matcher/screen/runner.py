"""Screen runner - validates unmatched target segments.

Runs registered screen tests on unmatched target segments to identify
valid candidates for addition to the network.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
from loguru import logger

# Import tests to register them
from . import tests as _tests  # noqa: F401
from .base import (
    CandidateContext,
    ScreenOutcome,
    ScreenResult,
    ScreenTest,
    get_registered_tests,
    get_test,
)


@dataclass
class ScreenReport:
    """Report from running screen tests on unmatched candidates."""

    total_candidates: int
    passed: int
    failed: int
    warned: int
    skipped: int

    # Per-test breakdown
    test_results: dict[str, dict[str, int]] = field(default_factory=dict)

    # Details of failed candidates
    failed_candidates: list[dict[str, Any]] = field(default_factory=list)

    # Details of warned candidates
    warned_candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fail_rate(self) -> float:
        """Percentage of candidates that failed."""
        if self.total_candidates == 0:
            return 0.0
        return self.failed / self.total_candidates

    @property
    def warn_rate(self) -> float:
        """Percentage of candidates that warned."""
        if self.total_candidates == 0:
            return 0.0
        return self.warned / self.total_candidates

    @property
    def pass_rate(self) -> float:
        """Percentage of candidates that passed (valid for addition)."""
        if self.total_candidates == 0:
            return 0.0
        return self.passed / self.total_candidates

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "total_candidates": self.total_candidates,
            "passed": self.passed,
            "failed": self.failed,
            "warned": self.warned,
            "skipped": self.skipped,
            "pass_rate": round(self.pass_rate, 4),
            "fail_rate": round(self.fail_rate, 4),
            "warn_rate": round(self.warn_rate, 4),
            "test_results": self.test_results,
            "failed_candidates": self.failed_candidates,
            "warned_candidates": self.warned_candidates,
        }


def run_screen(
    target_path: Path,
    bridge_path: Path | None = None,
    test_names: list[str] | None = None,
    output_path: Path | None = None,
    report_only: bool = False,
) -> tuple[gpd.GeoDataFrame | None, ScreenReport]:
    """Run screen tests on unmatched target segments.

    Identifies target segments that don't appear in the bridge file (unmatched),
    then runs screen tests on each to determine if they're valid candidates
    for addition to the network.

    Args:
        target_path: Path to target parquet file
        bridge_path: Path to bridge parquet file with matches (None = screen all targets)
        test_names: List of test names to run (None = all registered tests)
        output_path: Path for output filtered candidates file (None = no output)
        report_only: If True, don't filter candidates, just generate report

    Returns:
        Tuple of (valid_candidates_gdf, report)
        - valid_candidates_gdf: Target segments that passed screening (None if report_only)
        - report: ScreenReport with test results
    """
    logger.info(f"Running screen tests on {target_path}")

    # Load target data
    target_gdf = gpd.read_parquet(target_path)

    # Ensure consistent CRS (EPSG:4326 for screen tests)
    if target_gdf.crs is not None and target_gdf.crs != "EPSG:4326":
        target_gdf = target_gdf.to_crs("EPSG:4326")

    # Determine target ID column
    target_id_col = _get_id_column(target_gdf, "target")

    # Find unmatched targets (not in bridge file)
    if bridge_path is not None:
        bridge_gdf = gpd.read_parquet(bridge_path)
        bridge_target_col = _get_bridge_target_column(bridge_gdf)
        matched_ids = set(bridge_gdf[bridge_target_col].astype(str))
        unmatched_mask = ~target_gdf[target_id_col].astype(str).isin(matched_ids)
        candidates_gdf = target_gdf[unmatched_mask].copy()
        logger.info(
            f"Found {len(candidates_gdf)} unmatched targets "
            f"(out of {len(target_gdf)} total, {len(matched_ids)} matched)"
        )
    else:
        candidates_gdf = target_gdf.copy()
        logger.info(f"Screening all {len(candidates_gdf)} targets (no bridge file provided)")

    if len(candidates_gdf) == 0:
        logger.info("No candidates to screen")
        return candidates_gdf if not report_only else None, ScreenReport(
            total_candidates=0,
            passed=0,
            failed=0,
            warned=0,
            skipped=0,
        )

    # Get bounding box from candidates
    bounds = candidates_gdf.total_bounds
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

    # Initialize tests
    if test_names is None:
        test_classes = list(get_registered_tests().values())
    else:
        test_classes = [get_test(name) for name in test_names]

    tests: list[ScreenTest] = [cls() for cls in test_classes]
    logger.info(f"Running {len(tests)} screen tests: {[t.name for t in tests]}")

    # Prepare all tests (fetch context data)
    for test in tests:
        logger.info(f"Preparing test: {test.name}")
        test.prepare(bbox)

    # Run tests on each candidate
    results_by_candidate: dict[Any, list[ScreenResult]] = {}
    failed_indices: set[Any] = set()
    warned_indices: set[Any] = set()

    for idx, row in candidates_gdf.iterrows():
        target_id = str(row[target_id_col])
        target_geom = row.geometry

        if target_geom is None or target_geom.is_empty:
            logger.warning(f"Empty geometry for candidate {target_id}")
            continue

        # Get road class if available
        road_class = row.get("road_class") or row.get("class") or row.get("highway")

        # Create candidate context
        ctx = CandidateContext(
            target_id=target_id,
            target_geom=target_geom,
            road_class=road_class,
        )

        # Run all tests
        candidate_results = []
        candidate_failed = False
        candidate_warned = False

        for test in tests:
            result = test.test_candidate(ctx)
            candidate_results.append(result)

            if result.outcome == ScreenOutcome.FAIL:
                candidate_failed = True
            elif result.outcome == ScreenOutcome.WARN:
                candidate_warned = True

        results_by_candidate[idx] = candidate_results

        if candidate_failed:
            failed_indices.add(idx)
        elif candidate_warned:
            warned_indices.add(idx)

    # Build report
    report = _build_report(
        candidates_gdf=candidates_gdf,
        results_by_candidate=results_by_candidate,
        failed_indices=failed_indices,
        warned_indices=warned_indices,
        tests=tests,
        target_id_col=target_id_col,
    )

    logger.info(
        f"Screen tests complete: {report.passed} passed, {report.failed} failed, "
        f"{report.warned} warned ({report.pass_rate:.2%} pass rate)"
    )

    # Filter output
    if report_only:
        return None, report

    valid_gdf = candidates_gdf[~candidates_gdf.index.isin(failed_indices)].copy()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        valid_gdf.to_parquet(output_path)
        logger.info(f"Saved valid candidates to {output_path}")

    return valid_gdf, report


def _get_id_column(gdf: gpd.GeoDataFrame, name: str) -> str:
    """Determine the ID column for a GeoDataFrame."""
    for col in ["id", "ID", f"{name}_id"]:
        if col in gdf.columns:
            return col
    if gdf.index.name:
        return gdf.index.name
    raise ValueError(f"Could not determine ID column for {name}")


def _get_bridge_target_column(gdf: gpd.GeoDataFrame) -> str:
    """Determine the target ID column in bridge file."""
    for col in ["target_id", "local_id"]:
        if col in gdf.columns:
            return col
    raise ValueError("Could not determine target ID column in bridge file")


def _build_report(
    candidates_gdf: gpd.GeoDataFrame,
    results_by_candidate: dict[Any, list[ScreenResult]],
    failed_indices: set[Any],
    warned_indices: set[Any],
    tests: list[ScreenTest],
    target_id_col: str,
) -> ScreenReport:
    """Build screen report from results."""
    total = len(candidates_gdf)
    failed = len(failed_indices)
    warned = len(warned_indices)
    passed = total - failed - warned

    # Count skipped (candidates not in results)
    skipped = total - len(results_by_candidate)

    # Per-test breakdown
    test_results: dict[str, dict[str, int]] = {}
    for test in tests:
        test_results[test.name] = {
            "pass": 0,
            "fail": 0,
            "warn": 0,
            "skip": 0,
        }

    for candidate_results in results_by_candidate.values():
        for result in candidate_results:
            outcome_key = result.outcome.value
            if result.test_name in test_results:
                test_results[result.test_name][outcome_key] += 1

    # Failed candidate details
    failed_candidates = []
    for idx in failed_indices:
        row = candidates_gdf.loc[idx]
        candidate_results = results_by_candidate.get(idx, [])
        fail_reasons = [r for r in candidate_results if r.outcome == ScreenOutcome.FAIL]

        failed_candidates.append(
            {
                "target_id": str(row[target_id_col]),
                "fail_reasons": [
                    {"test": r.test_name, "reason": r.reason, "details": r.details}
                    for r in fail_reasons
                ],
            }
        )

    # Warned candidate details (limit to top 100)
    warned_candidates = []
    for idx in list(warned_indices)[:100]:
        row = candidates_gdf.loc[idx]
        candidate_results = results_by_candidate.get(idx, [])
        warn_reasons = [r for r in candidate_results if r.outcome == ScreenOutcome.WARN]

        warned_candidates.append(
            {
                "target_id": str(row[target_id_col]),
                "warn_reasons": [
                    {"test": r.test_name, "reason": r.reason, "details": r.details}
                    for r in warn_reasons
                ],
            }
        )

    return ScreenReport(
        total_candidates=total,
        passed=passed,
        failed=failed,
        warned=warned,
        skipped=skipped,
        test_results=test_results,
        failed_candidates=failed_candidates,
        warned_candidates=warned_candidates,
    )
