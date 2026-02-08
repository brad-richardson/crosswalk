#!/usr/bin/env python3
"""Feature diagnostics: analyze feature health across all labeled datasets.

Reads labeled features parquet files and produces a markdown report of
feature health, flagging anomalies like constant features, degenerate
distributions, and dataset-specific failures.

Usage:
    python scripts/feature_diagnostics.py
    python scripts/feature_diagnostics.py --dataset br_sao_paulo_roads
    python scripts/feature_diagnostics.py --output report.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root so we can import matcher config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matcher.config import FEATURE_CATEGORIES, FEATURE_COLUMNS, MAX_DISTANCE_METERS

# Boundary/default values that indicate a feature may not have been computed meaningfully
BOUNDARY_VALUES = {
    0.0: "zero",
    1.0: "one",
    MAX_DISTANCE_METERS: "MAX_DIST",
    10.0: "sinuosity_cap",
    180.0: "180.0",
    0.5: "neutral_0.5",
}

# Thresholds for anomaly detection
BOUNDARY_WARN_PCT = 50.0  # Warn if >50% of values are at a boundary
CONSTANT_THRESHOLD = 1e-12  # Variance below this = constant feature


def load_all_features(labels_dir: Path, dataset_filter: str | None = None) -> pd.DataFrame:
    """Load all labeled features parquet files into a single DataFrame."""
    feature_dirs = sorted(labels_dir.glob("dataset=*/data.parquet"))

    frames = []
    for parquet_path in feature_dirs:
        dataset_name = parquet_path.parent.name.replace("dataset=", "")
        if dataset_filter and dataset_name != dataset_filter:
            continue
        df = pd.read_parquet(parquet_path)
        df["_dataset"] = dataset_name
        frames.append(df)

    if not frames:
        print(f"ERROR: No feature files found in {labels_dir}", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def compute_feature_stats(df: pd.DataFrame, feature: str) -> dict:
    """Compute statistics for a single feature column."""
    values = df[feature] if feature in df.columns else pd.Series(dtype=float)

    if values.empty:
        return {"missing": True}

    # Convert to numeric, coercing errors
    values = pd.to_numeric(values, errors="coerce")

    n_total = len(values)
    n_nan = int(values.isna().sum())
    n_inf = int(np.isinf(values.replace([np.nan], 0)).sum())
    valid = values.dropna()
    valid = valid[~np.isinf(valid)]

    if len(valid) == 0:
        return {
            "n_total": n_total,
            "n_nan": n_nan,
            "n_inf": n_inf,
            "n_valid": 0,
            "missing": False,
        }

    n_zero = int((valid == 0.0).sum())
    n_unique = int(valid.nunique())
    variance = float(valid.var()) if len(valid) > 1 else 0.0

    stats = {
        "n_total": n_total,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_zero": n_zero,
        "n_valid": len(valid),
        "n_unique": n_unique,
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()) if len(valid) > 1 else 0.0,
        "median": float(valid.median()),
        "variance": variance,
        "missing": False,
    }

    # Check boundary value percentages
    boundary_counts = {}
    for bval, label in BOUNDARY_VALUES.items():
        count = int((valid == bval).sum())
        if count > 0:
            pct = count / len(valid) * 100
            boundary_counts[label] = (count, pct)
    stats["boundary_counts"] = boundary_counts

    return stats


def detect_anomalies(
    all_stats: dict[str, dict[str, dict]],
) -> list[dict]:
    """Detect anomalies across datasets and features.

    Returns a list of anomaly dicts with keys: feature, dataset, type, detail.
    """
    anomalies = []

    for feature, dataset_stats in all_stats.items():
        # Collect means across datasets for distribution comparison
        means = {}
        for dataset, stats in dataset_stats.items():
            if stats.get("missing") or stats.get("n_valid", 0) == 0:
                anomalies.append(
                    {
                        "feature": feature,
                        "dataset": dataset,
                        "type": "NO_DATA",
                        "detail": f"No valid values (n_nan={stats.get('n_nan', '?')})",
                    }
                )
                continue

            means[dataset] = stats["mean"]

            # Check NaN/Inf
            if stats["n_nan"] > 0:
                pct = stats["n_nan"] / stats["n_total"] * 100
                anomalies.append(
                    {
                        "feature": feature,
                        "dataset": dataset,
                        "type": "HAS_NAN",
                        "detail": f"{stats['n_nan']}/{stats['n_total']} NaN ({pct:.1f}%)",
                    }
                )
            if stats["n_inf"] > 0:
                pct = stats["n_inf"] / stats["n_total"] * 100
                anomalies.append(
                    {
                        "feature": feature,
                        "dataset": dataset,
                        "type": "HAS_INF",
                        "detail": f"{stats['n_inf']}/{stats['n_total']} Inf ({pct:.1f}%)",
                    }
                )

            # Check constant features (zero variance)
            if stats["variance"] < CONSTANT_THRESHOLD and stats["n_valid"] > 1:
                anomalies.append(
                    {
                        "feature": feature,
                        "dataset": dataset,
                        "type": "CONSTANT",
                        "detail": f"All values = {stats['min']:.4f} (n={stats['n_valid']})",
                    }
                )

            # Check boundary value dominance
            for bval_label, (count, pct) in stats.get("boundary_counts", {}).items():
                if pct > BOUNDARY_WARN_PCT:
                    anomalies.append(
                        {
                            "feature": feature,
                            "dataset": dataset,
                            "type": "BOUNDARY_DOMINANT",
                            "detail": f"{pct:.1f}% at {bval_label} ({count}/{stats['n_valid']})",
                        }
                    )

        # Check for dramatic distribution differences across datasets
        if len(means) >= 2:
            mean_vals = list(means.values())
            global_mean = np.mean(mean_vals)
            global_std = np.std(mean_vals)
            if global_std > 0 and global_mean != 0:
                for dataset, m in means.items():
                    z = abs(m - global_mean) / global_std
                    if z > 2.5 and len(mean_vals) >= 5:
                        anomalies.append(
                            {
                                "feature": feature,
                                "dataset": dataset,
                                "type": "DISTRIBUTION_OUTLIER",
                                "detail": f"Mean={m:.4f} vs global mean={global_mean:.4f} (z={z:.1f})",
                            }
                        )

    return anomalies


def format_report(
    all_stats: dict[str, dict[str, dict]],
    anomalies: list[dict],
    df: pd.DataFrame,
) -> str:
    """Format the diagnostic results as a markdown report."""
    lines = []
    lines.append("# Feature Diagnostics Report\n")

    # Dataset overview
    datasets = sorted(df["_dataset"].unique())
    lines.append("## Dataset Overview\n")
    lines.append("| Dataset | Samples | Features |")
    lines.append("|---------|---------|----------|")
    for ds in datasets:
        n = len(df[df["_dataset"] == ds])
        # Count features present (non-null)
        ds_df = df[df["_dataset"] == ds]
        feature_cols = [c for c in FEATURE_COLUMNS if c in ds_df.columns]
        n_features = sum(1 for c in feature_cols if ds_df[c].notna().any())
        lines.append(f"| {ds} | {n} | {n_features}/{len(feature_cols)} |")
    lines.append("")

    # Anomaly summary
    lines.append("## Anomaly Summary\n")
    if not anomalies:
        lines.append("No anomalies detected.\n")
    else:
        # Group by type
        by_type: dict[str, list] = {}
        for a in anomalies:
            by_type.setdefault(a["type"], []).append(a)

        for atype, items in sorted(by_type.items()):
            lines.append(f"### {atype} ({len(items)} occurrences)\n")
            lines.append("| Feature | Dataset | Detail |")
            lines.append("|---------|---------|--------|")
            for item in sorted(items, key=lambda x: (x["feature"], x["dataset"])):
                lines.append(f"| {item['feature']} | {item['dataset']} | {item['detail']} |")
            lines.append("")

    # Per-category feature stats
    lines.append("## Feature Statistics by Category\n")
    for category, features in FEATURE_CATEGORIES.items():
        lines.append(f"### {category}\n")
        for feature in features:
            if feature not in all_stats:
                lines.append(f"**{feature}**: Not found in data\n")
                continue

            lines.append(f"**{feature}**\n")
            lines.append(
                "| Dataset | N | NaN | Zero | Unique | Min | Max | Mean | Std | Boundary Issues |"
            )
            lines.append(
                "|---------|---|-----|------|--------|-----|-----|------|-----|-----------------|"
            )

            for dataset in datasets:
                stats = all_stats[feature].get(dataset, {})
                if stats.get("missing") or stats.get("n_valid", 0) == 0:
                    lines.append(f"| {dataset} | 0 | - | - | - | - | - | - | - | NO DATA |")
                    continue

                # Format boundary issues compactly
                boundary_issues = []
                for bval_label, (_count, pct) in stats.get("boundary_counts", {}).items():
                    if pct > 20:  # Show if >20%
                        boundary_issues.append(f"{bval_label}:{pct:.0f}%")
                boundary_str = ", ".join(boundary_issues) if boundary_issues else "-"

                lines.append(
                    f"| {dataset} | {stats['n_valid']} | {stats['n_nan']} | "
                    f"{stats['n_zero']} | {stats['n_unique']} | "
                    f"{stats['min']:.4f} | {stats['max']:.4f} | "
                    f"{stats['mean']:.4f} | {stats['std']:.4f} | {boundary_str} |"
                )
            lines.append("")

    # Dataset health summary (for underperforming datasets)
    lines.append("## Dataset Health Summary\n")
    for ds in datasets:
        ds_df = df[df["_dataset"] == ds]
        n = len(ds_df)
        ds_anomalies = [a for a in anomalies if a["dataset"] == ds]
        n_constant = sum(1 for a in ds_anomalies if a["type"] == "CONSTANT")
        n_boundary = sum(1 for a in ds_anomalies if a["type"] == "BOUNDARY_DOMINANT")
        n_nan = sum(1 for a in ds_anomalies if a["type"] == "HAS_NAN")

        lines.append(f"### {ds} (n={n})\n")
        lines.append(f"- Constant features: {n_constant}")
        lines.append(f"- Boundary-dominant features: {n_boundary}")
        lines.append(f"- Features with NaN: {n_nan}")

        # Check heading_consistency for short-segment masking
        if "heading_consistency_ref" in ds_df.columns:
            hc_ones = (ds_df["heading_consistency_ref"] == 1.0).sum()
            hc_pct = hc_ones / n * 100 if n > 0 else 0
            lines.append(f"- heading_consistency_ref == 1.0: {hc_ones}/{n} ({hc_pct:.0f}%)")

        # Check angle_histogram for sparse geometry masking
        if "angle_histogram_similarity" in ds_df.columns:
            ah_ones = (ds_df["angle_histogram_similarity"] == 1.0).sum()
            ah_pct = ah_ones / n * 100 if n > 0 else 0
            lines.append(f"- angle_histogram_similarity == 1.0: {ah_ones}/{n} ({ah_pct:.0f}%)")

        # Check sinuosity for outliers and cap saturation
        if "sinuosity_ref" in ds_df.columns:
            sin_max = ds_df["sinuosity_ref"].max()
            sin_cap = (ds_df["sinuosity_ref"] >= 10.0).sum()
            lines.append(f"- sinuosity_ref max: {sin_max:.1f}, n>=10.0 (cap hits): {sin_cap}")

        # Check collinear_gap_ratio dominance
        if "collinear_gap_ratio" in ds_df.columns:
            cgr_ones = (ds_df["collinear_gap_ratio"] == 1.0).sum()
            cgr_pct = cgr_ones / n * 100 if n > 0 else 0
            lines.append(f"- collinear_gap_ratio == 1.0: {cgr_ones}/{n} ({cgr_pct:.0f}%)")

        # Check error rates via _error column
        if "_error" in ds_df.columns:
            n_errors = ds_df["_error"].notna().sum()
            lines.append(f"- Computation errors: {n_errors}/{n}")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Feature diagnostics for matcher datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter to a single dataset (e.g., br_sao_paulo_roads)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write report to file instead of stdout",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=str(PROJECT_ROOT / "labels" / "features"),
        help="Path to labels/features directory",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    print(f"Loading features from {labels_dir}...", file=sys.stderr)

    df = load_all_features(labels_dir, args.dataset)
    datasets = sorted(df["_dataset"].unique())
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    print(
        f"Loaded {len(df)} samples across {len(datasets)} datasets, "
        f"{len(feature_cols)}/{len(FEATURE_COLUMNS)} features present",
        file=sys.stderr,
    )

    # Compute per-dataset per-feature stats
    all_stats: dict[str, dict[str, dict]] = {}
    for feature in feature_cols:
        all_stats[feature] = {}
        for dataset in datasets:
            ds_df = df[df["_dataset"] == dataset]
            all_stats[feature][dataset] = compute_feature_stats(ds_df, feature)

    # Detect anomalies
    anomalies = detect_anomalies(all_stats)
    print(f"Found {len(anomalies)} anomalies", file=sys.stderr)

    # Format report
    report = format_report(all_stats, anomalies, df)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
