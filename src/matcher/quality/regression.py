"""Quality regression detection for re-fetched datasets.

Compares key metrics of a freshly fetched GeoDataFrame against a saved
quality fingerprint to catch catastrophic regressions (e.g., name coverage
dropping from 80% to 0%).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import geopandas as gpd
import pandas as pd
from loguru import logger

from ..datasets.schema import QualityFingerprintConfig


@dataclass
class RegressionViolation:
    """A single quality regression violation."""

    metric: str
    expected: float
    actual: float
    threshold: float
    message: str


def check_quality_regression(
    gdf: gpd.GeoDataFrame,
    fingerprint: QualityFingerprintConfig,
    dataset_name: str,
) -> list[RegressionViolation]:
    """Check for catastrophic quality regressions against a saved fingerprint.

    Thresholds are intentionally moderate — designed to catch catastrophic
    regressions (e.g., name coverage dropping to 0) rather than minor
    fluctuations from data updates.

    Thresholds:
    - name_coverage_ratio drop > 30 percentage points -> fail
    - class_coverage_ratio drop > 30 percentage points -> fail
    - total_segments change > 50% (either direction) -> fail

    Args:
        gdf: Freshly fetched GeoDataFrame (Overture-compatible schema)
        fingerprint: Saved quality fingerprint from dataset YAML
        dataset_name: Name for logging

    Returns:
        List of violations (empty = all checks passed)
    """
    violations: list[RegressionViolation] = []

    total_segments = len(gdf)

    # --- Segment count check (±50%) ---
    expected_segments = fingerprint.total_segments
    if expected_segments > 0:
        pct_change = abs(total_segments - expected_segments) / expected_segments
        if pct_change > 0.50:
            direction = "increase" if total_segments > expected_segments else "decrease"
            violations.append(
                RegressionViolation(
                    metric="total_segments",
                    expected=expected_segments,
                    actual=total_segments,
                    threshold=0.50,
                    message=(
                        f"{dataset_name}: segment count {direction} of {pct_change:.0%} "
                        f"({expected_segments} -> {total_segments})"
                    ),
                )
            )

    # --- Name coverage check (drop > 30pp) ---
    if total_segments > 0 and "names" in gdf.columns:
        has_name = gdf["names"].apply(
            lambda x: x is not None and isinstance(x, dict) and bool(x.get("primary"))
        )
        actual_name_ratio = has_name.sum() / total_segments
    else:
        actual_name_ratio = 0.0

    expected_name_ratio = fingerprint.name_coverage_ratio
    name_drop = expected_name_ratio - actual_name_ratio
    if name_drop > 0.30:
        violations.append(
            RegressionViolation(
                metric="name_coverage_ratio",
                expected=expected_name_ratio,
                actual=actual_name_ratio,
                threshold=0.30,
                message=(
                    f"{dataset_name}: name coverage dropped {name_drop:.0%} "
                    f"({expected_name_ratio:.1%} -> {actual_name_ratio:.1%})"
                ),
            )
        )

    # --- Class coverage check (drop > 30pp) ---
    if total_segments > 0 and "class" in gdf.columns:
        has_class = gdf["class"].apply(
            lambda x: pd.notna(x) and str(x) not in ("", "unknown", "unclassified")
        )
        actual_class_ratio = has_class.sum() / total_segments
    else:
        actual_class_ratio = 0.0

    expected_class_ratio = fingerprint.class_coverage_ratio
    class_drop = expected_class_ratio - actual_class_ratio
    if class_drop > 0.30:
        violations.append(
            RegressionViolation(
                metric="class_coverage_ratio",
                expected=expected_class_ratio,
                actual=actual_class_ratio,
                threshold=0.30,
                message=(
                    f"{dataset_name}: class coverage dropped {class_drop:.0%} "
                    f"({expected_class_ratio:.1%} -> {actual_class_ratio:.1%})"
                ),
            )
        )

    if violations:
        logger.warning(
            f"Quality regression detected for {dataset_name}: {len(violations)} violation(s)"
        )
        for v in violations:
            logger.warning(f"  {v.message}")

    return violations


def compute_quick_fingerprint(gdf: gpd.GeoDataFrame) -> QualityFingerprintConfig:
    """Compute a lightweight quality fingerprint from a fetched GeoDataFrame.

    Captures the metrics used by check_quality_regression so the fingerprint
    can be auto-updated after each successful fetch.

    Args:
        gdf: Fetched GeoDataFrame (Overture-compatible schema)

    Returns:
        QualityFingerprintConfig with key metrics populated
    """
    total = len(gdf)

    name_ratio = 0.0
    if total > 0 and "names" in gdf.columns:
        has_name = gdf["names"].apply(
            lambda x: x is not None and isinstance(x, dict) and bool(x.get("primary"))
        )
        name_ratio = has_name.sum() / total

    class_ratio = 0.0
    if total > 0 and "class" in gdf.columns:
        has_class = gdf["class"].apply(
            lambda x: pd.notna(x) and str(x) not in ("", "unknown", "unclassified")
        )
        class_ratio = has_class.sum() / total

    return QualityFingerprintConfig(
        computed_at=datetime.now(UTC),
        total_segments=total,
        name_coverage_ratio=round(name_ratio, 4),
        class_coverage_ratio=round(class_ratio, 4),
    )
