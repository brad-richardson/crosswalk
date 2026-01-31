"""Class confusion analysis using labeled data and geometry metadata.

This module analyzes class mapping quality by examining:
- Tier violations in human-labeled matches (vehicle↔pedestrian pairs labeled as match)
- Low similarity matches (class_similarity < 0.3 but labeled match)
- High similarity no-matches (class_similarity > 0.8 but labeled no_match)
- Per-dataset breakdown of issues
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ..features.semantic import get_traffic_tier
from ..labeling.geometry_store import GeometryStore
from ..labeling.label_store import DEFAULT_LABELS_DIR


@dataclass
class ClassAnalysisReport:
    """Report from class confusion analysis."""

    # Summary counts
    total_labels: int = 0
    total_matches: int = 0
    total_no_matches: int = 0

    # Tier violations: vehicle↔pedestrian pairs labeled as match
    tier_violations: list[dict] = field(default_factory=list)
    tier_violation_count: int = 0

    # Low similarity matches: class_similarity < threshold but labeled match
    low_similarity_matches: list[dict] = field(default_factory=list)
    low_similarity_match_count: int = 0

    # High similarity no-matches: class_similarity > threshold but labeled no_match
    high_similarity_no_matches: list[dict] = field(default_factory=list)
    high_similarity_no_match_count: int = 0

    # Per-dataset breakdown
    per_dataset_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    # Class confusion matrix: ref_class -> target_class -> {match: N, no_match: M}
    confusion_matrix: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_labels": self.total_labels,
                "total_matches": self.total_matches,
                "total_no_matches": self.total_no_matches,
                "tier_violation_count": self.tier_violation_count,
                "low_similarity_match_count": self.low_similarity_match_count,
                "high_similarity_no_match_count": self.high_similarity_no_match_count,
            },
            "tier_violations": self.tier_violations,
            "low_similarity_matches": self.low_similarity_matches,
            "high_similarity_no_matches": self.high_similarity_no_matches,
            "per_dataset_stats": self.per_dataset_stats,
            "confusion_matrix": self.confusion_matrix,
        }


def _is_tier_incompatible(ref_class: str | None, target_class: str | None) -> bool:
    """Check if ref and target classes are vehicle↔pedestrian mismatch."""
    ref_tier = get_traffic_tier(ref_class)
    target_tier = get_traffic_tier(target_class)

    if ref_tier is None or target_tier is None:
        return False

    return {ref_tier, target_tier} == {"vehicle", "pedestrian"}


def analyze_class_confusion_from_labels(
    labels_dir: Path = DEFAULT_LABELS_DIR,
    geometries_dir: Path = Path("label_geometries"),
    low_similarity_threshold: float = 0.3,
    high_similarity_threshold: float = 0.8,
    max_examples: int = 100,
) -> ClassAnalysisReport:
    """Analyze class confusion using labeled data and geometry metadata.

    Uses:
        - Labels with human match/no_match judgments
        - Geometry store with raw class/subclass values
        - class_similarity feature scores

    Args:
        labels_dir: Directory containing Hive-partitioned label CSVs
        geometries_dir: Directory containing geometry companion files
        low_similarity_threshold: Threshold for identifying low similarity matches
        high_similarity_threshold: Threshold for identifying high similarity no-matches
        max_examples: Maximum number of examples to include per category

    Returns:
        ClassAnalysisReport with analysis results
    """
    report = ClassAnalysisReport()

    # Load all labels
    labels_dir = Path(labels_dir)
    geometries_dir = Path(geometries_dir)

    if not labels_dir.exists():
        logger.warning(f"Labels directory not found: {labels_dir}")
        return report

    # Find all label partitions
    partitions = list(labels_dir.glob("dataset=*/data.csv"))
    if not partitions:
        logger.warning(f"No label partitions found in {labels_dir}")
        return report

    logger.info(f"Analyzing {len(partitions)} label partitions...")

    for partition_path in partitions:
        dataset_name = partition_path.parent.name.replace("dataset=", "")
        logger.info(f"Processing {dataset_name}...")

        # Load labels for this dataset
        try:
            df = pd.read_csv(partition_path)
        except Exception as e:
            logger.warning(f"Failed to load {partition_path}: {e}")
            continue

        if len(df) == 0:
            continue

        # Load geometry store for raw class values
        geo_store = GeometryStore(dataset_name, geometries_dir=geometries_dir)

        # Initialize per-dataset stats
        dataset_stats = {
            "total": len(df),
            "matches": 0,
            "no_matches": 0,
            "tier_violations": 0,
            "low_similarity_matches": 0,
            "high_similarity_no_matches": 0,
        }

        # Process each label
        for _, row in df.iterrows():
            gers_id = str(row.get("gers_id", row.get("ref_id", "")))
            target_id = str(row["target_id"])
            label = row.get("label", "")
            class_similarity = row.get("class_similarity", 0.5)

            report.total_labels += 1

            is_match = label == "match"
            is_no_match = label == "no_match"

            if is_match:
                report.total_matches += 1
                dataset_stats["matches"] += 1
            elif is_no_match:
                report.total_no_matches += 1
                dataset_stats["no_matches"] += 1

            # Get raw class values from geometry store
            pair = geo_store.get_pair(gers_id, target_id)
            if pair is None:
                # Fall back to feature-based analysis without raw classes
                ref_class = None
                target_class = None
            else:
                ref_class = pair.get("ref_class")
                target_class = pair.get("target_class")

            # Update confusion matrix
            ref_class_key = ref_class or "unknown"
            target_class_key = target_class or "unknown"

            if ref_class_key not in report.confusion_matrix:
                report.confusion_matrix[ref_class_key] = {}
            if target_class_key not in report.confusion_matrix[ref_class_key]:
                report.confusion_matrix[ref_class_key][target_class_key] = {
                    "match": 0,
                    "no_match": 0,
                }

            if is_match:
                report.confusion_matrix[ref_class_key][target_class_key]["match"] += 1
            elif is_no_match:
                report.confusion_matrix[ref_class_key][target_class_key]["no_match"] += 1

            # Check for tier violations (matches with vehicle↔pedestrian)
            if is_match and _is_tier_incompatible(ref_class, target_class):
                report.tier_violation_count += 1
                dataset_stats["tier_violations"] += 1

                if len(report.tier_violations) < max_examples:
                    ref_name = pair.get("ref_name") if pair else None
                    target_name = pair.get("target_name") if pair else None

                    report.tier_violations.append(
                        {
                            "dataset": dataset_name,
                            "gers_id": gers_id,
                            "target_id": target_id,
                            "ref_class": ref_class,
                            "target_class": target_class,
                            "ref_tier": get_traffic_tier(ref_class),
                            "target_tier": get_traffic_tier(target_class),
                            "ref_name": ref_name,
                            "target_name": target_name,
                            "class_similarity": class_similarity,
                        }
                    )

            # Check for low similarity matches
            if is_match and class_similarity < low_similarity_threshold:
                report.low_similarity_match_count += 1
                dataset_stats["low_similarity_matches"] += 1

                if len(report.low_similarity_matches) < max_examples:
                    ref_name = pair.get("ref_name") if pair else None
                    target_name = pair.get("target_name") if pair else None

                    report.low_similarity_matches.append(
                        {
                            "dataset": dataset_name,
                            "gers_id": gers_id,
                            "target_id": target_id,
                            "ref_class": ref_class,
                            "target_class": target_class,
                            "ref_name": ref_name,
                            "target_name": target_name,
                            "class_similarity": class_similarity,
                        }
                    )

            # Check for high similarity no-matches
            if is_no_match and class_similarity > high_similarity_threshold:
                report.high_similarity_no_match_count += 1
                dataset_stats["high_similarity_no_matches"] += 1

                if len(report.high_similarity_no_matches) < max_examples:
                    ref_name = pair.get("ref_name") if pair else None
                    target_name = pair.get("target_name") if pair else None

                    report.high_similarity_no_matches.append(
                        {
                            "dataset": dataset_name,
                            "gers_id": gers_id,
                            "target_id": target_id,
                            "ref_class": ref_class,
                            "target_class": target_class,
                            "ref_name": ref_name,
                            "target_name": target_name,
                            "class_similarity": class_similarity,
                        }
                    )

        report.per_dataset_stats[dataset_name] = dataset_stats

    return report


def analyze_class_confusion_from_bridge(
    bridge_path: Path,
    reference_path: Path,
    target_path: Path,
    confidence_threshold: float = 0.75,
    low_similarity_threshold: float = 0.3,
    high_similarity_threshold: float = 0.8,
    max_examples: int = 100,
) -> ClassAnalysisReport:
    """Analyze class confusion using a bridge file and source data.

    Similar to analyze_class_confusion_from_labels but uses ML predictions
    from a bridge file rather than human labels.

    Args:
        bridge_path: Path to bridge parquet file with matches
        reference_path: Path to reference (Overture) parquet file
        target_path: Path to target dataset parquet file
        confidence_threshold: Minimum confidence to consider a match
        low_similarity_threshold: Threshold for low similarity matches
        high_similarity_threshold: Threshold for high similarity no-matches
        max_examples: Maximum examples per category

    Returns:
        ClassAnalysisReport with analysis results
    """
    import geopandas as gpd

    report = ClassAnalysisReport()

    # Load data
    logger.info(f"Loading bridge file: {bridge_path}")
    bridge = pd.read_parquet(bridge_path)

    logger.info(f"Loading reference: {reference_path}")
    ref_gdf = gpd.read_parquet(reference_path)
    ref_gdf["id"] = ref_gdf["id"].astype(str)
    ref_lookup = ref_gdf.set_index("id")

    logger.info(f"Loading target: {target_path}")
    target_gdf = gpd.read_parquet(target_path)
    target_gdf["id"] = target_gdf["id"].astype(str)
    target_lookup = target_gdf.set_index("id")

    logger.info(f"Analyzing {len(bridge)} bridge entries...")

    for _, row in bridge.iterrows():
        gers_id = str(row.get("gers_id", row.get("ref_id", "")))
        target_id = str(row.get("target_id", ""))
        confidence = row.get("confidence", 0.0)
        class_similarity = row.get("class_similarity", 0.5)

        report.total_labels += 1

        is_match = confidence >= confidence_threshold
        is_no_match = confidence < 0.5

        if is_match:
            report.total_matches += 1
        elif is_no_match:
            report.total_no_matches += 1

        # Get raw class values from source data
        ref_class = None
        target_class = None
        ref_name = None
        target_name = None

        if gers_id in ref_lookup.index:
            ref_row = ref_lookup.loc[gers_id]
            ref_class = ref_row.get("class") if hasattr(ref_row, "get") else None
            ref_name = ref_row.get("names") if hasattr(ref_row, "get") else None

        if target_id in target_lookup.index:
            target_row = target_lookup.loc[target_id]
            target_class = target_row.get("class") if hasattr(target_row, "get") else None
            target_name = target_row.get("names") if hasattr(target_row, "get") else None

        # Update confusion matrix
        ref_class_key = str(ref_class) if ref_class else "unknown"
        target_class_key = str(target_class) if target_class else "unknown"

        if ref_class_key not in report.confusion_matrix:
            report.confusion_matrix[ref_class_key] = {}
        if target_class_key not in report.confusion_matrix[ref_class_key]:
            report.confusion_matrix[ref_class_key][target_class_key] = {"match": 0, "no_match": 0}

        if is_match:
            report.confusion_matrix[ref_class_key][target_class_key]["match"] += 1
        elif is_no_match:
            report.confusion_matrix[ref_class_key][target_class_key]["no_match"] += 1

        # Check for tier violations
        if is_match and _is_tier_incompatible(ref_class, target_class):
            report.tier_violation_count += 1

            if len(report.tier_violations) < max_examples:
                report.tier_violations.append(
                    {
                        "gers_id": gers_id,
                        "target_id": target_id,
                        "ref_class": ref_class,
                        "target_class": target_class,
                        "ref_tier": get_traffic_tier(ref_class),
                        "target_tier": get_traffic_tier(target_class),
                        "ref_name": ref_name,
                        "target_name": target_name,
                        "class_similarity": class_similarity,
                        "confidence": confidence,
                    }
                )

        # Check for low similarity matches
        if is_match and class_similarity < low_similarity_threshold:
            report.low_similarity_match_count += 1

            if len(report.low_similarity_matches) < max_examples:
                report.low_similarity_matches.append(
                    {
                        "gers_id": gers_id,
                        "target_id": target_id,
                        "ref_class": ref_class,
                        "target_class": target_class,
                        "ref_name": ref_name,
                        "target_name": target_name,
                        "class_similarity": class_similarity,
                        "confidence": confidence,
                    }
                )

        # Check for high similarity no-matches
        if is_no_match and class_similarity > high_similarity_threshold:
            report.high_similarity_no_match_count += 1

            if len(report.high_similarity_no_matches) < max_examples:
                report.high_similarity_no_matches.append(
                    {
                        "gers_id": gers_id,
                        "target_id": target_id,
                        "ref_class": ref_class,
                        "target_class": target_class,
                        "ref_name": ref_name,
                        "target_name": target_name,
                        "class_similarity": class_similarity,
                        "confidence": confidence,
                    }
                )

    return report


def format_analysis_report(report: ClassAnalysisReport) -> str:
    """Format analysis report for console output."""
    lines = []
    lines.append(f"=== Class Confusion Analysis (from {report.total_labels:,} labels) ===")
    lines.append("")

    # Summary
    lines.append("Summary:")
    lines.append(f"  Total labels: {report.total_labels:,}")
    lines.append(f"  Matches: {report.total_matches:,}")
    lines.append(f"  No-matches: {report.total_no_matches:,}")
    lines.append("")

    # Tier violations
    lines.append("Tier Violations in Human-Labeled Matches:")
    lines.append(
        f"  {report.tier_violation_count} vehicle↔pedestrian pairs labeled 'match' "
        f"(potential mapping errors)"
    )
    if report.tier_violations:
        # Group by class pair
        class_pair_counts: dict[tuple[str, str], int] = {}
        for v in report.tier_violations:
            pair = (v["ref_class"] or "unknown", v["target_class"] or "unknown")
            class_pair_counts[pair] = class_pair_counts.get(pair, 0) + 1

        for (ref_cls, tgt_cls), count in sorted(class_pair_counts.items(), key=lambda x: -x[1])[
            :10
        ]:
            lines.append(f"  - {ref_cls} -> {tgt_cls}: {count} pairs")
    lines.append("")

    # Low similarity matches
    lines.append("Low Similarity Matches (class_similarity < 0.3):")
    lines.append(
        f"  {report.low_similarity_match_count} pairs where class differs but humans labeled match"
    )
    lines.append("  Suggests class weight may be too high or mapping wrong")
    lines.append("")

    # High similarity no-matches
    lines.append("High Similarity No-Matches (class_similarity > 0.8):")
    lines.append(
        f"  {report.high_similarity_no_match_count} pairs with similar classes labeled no_match"
    )
    lines.append("")

    # Per-dataset breakdown
    if report.per_dataset_stats:
        lines.append("Per-Dataset Breakdown:")
        sorted_datasets = sorted(
            report.per_dataset_stats.items(),
            key=lambda x: x[1].get("tier_violations", 0),
            reverse=True,
        )
        for dataset, stats in sorted_datasets[:10]:
            if stats.get("tier_violations", 0) > 0:
                lines.append(f"  {dataset}: {stats['tier_violations']} tier violations")
        lines.append("")

    # Recommendations
    lines.append("Recommendations:")
    if report.tier_violation_count > 0:
        # Find dataset with most violations
        worst_dataset = max(
            report.per_dataset_stats.items(),
            key=lambda x: x[1].get("tier_violations", 0),
            default=("unknown", {}),
        )[0]
        lines.append(f"  - Review class mappings for {worst_dataset}")
        lines.append("  - Consider enabling hard blocking for datasets with many violations")

    if report.low_similarity_match_count > report.total_matches * 0.1:
        lines.append("  - Class similarity appears to be a poor predictor for this data")
        lines.append("  - Consider reducing class_similarity weight or fixing mappings")

    return "\n".join(lines)
