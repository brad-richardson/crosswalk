#!/usr/bin/env python3
"""Feature ablation study script for identifying noise or redundant features.

This script systematically tests the impact of removing features on model performance
to identify features that may be noise or no longer needed.

Usage:
    # Full study (baseline + single features + categories)
    python scripts/ablation_study.py

    # Single-feature ablations only
    python scripts/ablation_study.py --mode single

    # Category ablations only
    python scripts/ablation_study.py --mode category

    # Custom output
    python scripts/ablation_study.py --output benchmarks/ablation_2026_02_01
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

# Add matcher to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from matcher.config import FEATURE_COLUMNS
from matcher.labeling.label_store import LabelStore
from matcher.matching.ml import MLMatcher, segment_aware_split

# Feature categories - comprehensive grouping of all 60 features
FEATURE_CATEGORIES = {
    "geometric": [
        "hausdorff_distance_m",
        "mean_hausdorff_distance_m",
        "hausdorff_p95_m",
        "buffer_iou_5m",
        "buffer_iou_15m",
        "heading_delta",
        "length_ratio",
        "centroid_distance_m",
        "collinear_gap_ratio",
    ],
    "semantic_name": [
        "name_levenshtein",
        "name_jaro_winkler",
        "name_token_sort",
        "name_soundex",
        "name_metaphone",
        "has_name_ref",
        "has_name_target",
        "name_is_generic",
    ],
    "semantic_class": [
        "class_similarity",
    ],
    "endpoint": [
        "min_endpoint_proximity_m",
        "max_endpoint_proximity_m",
        "shared_endpoint_count",
    ],
    "lateral_offset": [
        "lateral_offset_m",
        "lateral_offset_iqr_m",
        "lateral_offset_p95_m",
    ],
    "topology": [
        "from_degree_ref",
        "to_degree_ref",
        "from_degree_target",
        "to_degree_target",
        "degree_match_score",
        "degree_signature_similarity",
        "is_dead_end_ref",
        "is_dead_end_target",
        "dead_end_match",
        "is_intersection_ref",
        "is_intersection_target",
        "intersection_match",
    ],
    "coverage": [
        "ref_coverage",
        "target_coverage",
        "min_coverage",
        "coverage_ratio",
    ],
    "graphlet": [
        "graphlet_similarity",
        "endpoint_degree_similarity",
    ],
    "sinuosity": [
        "sinuosity_ref",
        "sinuosity_target",
        "sinuosity_delta",
    ],
    "heading_consistency": [
        "heading_consistency_ref",
        "heading_consistency_target",
        "heading_consistency_delta",
    ],
    "vertex_density": [
        "vertex_density_ref",
        "vertex_density_target",
        "vertex_density_ratio",
    ],
    "length": [
        "min_length_m",
    ],
    "shape_complexity": [
        "shape_complexity_ref",
        "shape_complexity_target",
        "shape_complexity_delta",
    ],
    "name_numeric": [
        "name_numeric_match",
    ],
    "parallel_sibling": [
        "has_parallel_sibling_ref",
        "offset_vs_half_corridor_ratio",
        "offset_over_expected_halfwidth",
        "likely_representation_mismatch",
    ],
}

# Quality thresholds
ACCURACY_THRESHOLD = 0.90
CV_F1_THRESHOLD = 0.90

# Classification thresholds
NOISE_THRESHOLD = 0.0  # F1 delta >= 0 (removal helps or neutral)
REDUNDANT_THRESHOLD = -0.005  # F1 delta > -0.005
USEFUL_THRESHOLD = -0.01  # -0.01 < F1 delta <= -0.005
# IMPORTANT: F1 delta <= -0.01 is classified as "important"


def validate_feature_categories():
    """Validate that feature categories cover all features."""
    all_features_in_categories = set()
    for features in FEATURE_CATEGORIES.values():
        all_features_in_categories.update(features)

    feature_columns_set = set(FEATURE_COLUMNS)

    missing = feature_columns_set - all_features_in_categories
    extra = all_features_in_categories - feature_columns_set

    if missing:
        logger.warning(f"Features not in any category: {sorted(missing)}")
    if extra:
        logger.warning(f"Features in categories but not in FEATURE_COLUMNS: {sorted(extra)}")

    return len(missing) == 0 and len(extra) == 0


def classify_feature(f1_delta: float) -> str:
    """Classify a feature based on its F1 delta.

    Args:
        f1_delta: Change in F1 score when feature is removed (negative = removal hurts)

    Returns:
        Classification string: "noise", "redundant", "useful", or "important"
    """
    if f1_delta >= NOISE_THRESHOLD:
        return "noise"
    elif f1_delta > REDUNDANT_THRESHOLD:
        return "redundant"
    elif f1_delta > USEFUL_THRESHOLD:
        return "useful"
    else:
        return "important"


def train_and_evaluate(
    labels_dir: Path,
    exclude_features: list[str] | None = None,
    seed: int = 999,
    n_cv_folds: int = 5,
) -> dict:
    """Train a model with optional feature exclusions and evaluate.

    Args:
        labels_dir: Path to labels directory
        exclude_features: List of features to exclude (None = use all)
        seed: Random seed for reproducibility
        n_cv_folds: Number of cross-validation folds

    Returns:
        Dict with metrics: accuracy, f1, cv_f1_mean, cv_f1_std, n_features_used
    """
    # Load labels
    df = LabelStore.load_all(labels_dir)

    # Filter to valid labels
    valid_labels = {"match", "no_match"}
    df = df[df["label"].isin(valid_labels)].copy()

    if len(df) == 0:
        raise ValueError("No valid labels found")

    # Determine feature columns to use
    if exclude_features:
        feature_names = [f for f in FEATURE_COLUMNS if f not in exclude_features]
    else:
        feature_names = FEATURE_COLUMNS.copy()

    # Create matcher with specific features
    matcher = MLMatcher()
    matcher.feature_names = feature_names

    # Extract features
    X, y = matcher._extract_features_and_labels(df, binary=True)

    # Segment-aware split
    train_idx, test_idx, groups = segment_aware_split(
        df, test_size=0.3, random_state=seed, return_groups=True
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Compute imputation medians from training data
    matcher.feature_medians = {}
    for i, feat_name in enumerate(matcher.feature_names):
        col_vals = X_train[:, i]
        median_val = np.nanmedian(col_vals)
        matcher.feature_medians[feat_name] = median_val if not np.isnan(median_val) else 0.0

    # Impute missing values
    X_train = matcher._impute_missing(X_train)
    X_test = matcher._impute_missing(X_test)

    # Handle class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Train XGBoost with consistent parameters
    try:
        import xgboost as xgb
    except ImportError as err:
        raise ImportError("XGBoost is required. Install with: pip install xgboost") from err

    params = {
        "n_estimators": 912,
        "max_depth": 7,
        "learning_rate": 0.010636101749852585,
        "min_child_weight": 5,
        "subsample": 0.8761081830856152,
        "colsample_bytree": 0.9616639656253169,
        "gamma": 0.30396926808636227,
        "reg_alpha": 1.724985773632091,
        "reg_lambda": 2.7011840568401686,
        "max_bin": 147,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Cross-validation with segment-aware folding
    X_imputed = matcher._impute_missing(X.copy())
    n_groups = groups.nunique()

    if n_groups >= n_cv_folds:
        gkf = GroupKFold(n_splits=n_cv_folds)
        cv_scores = []
        for train_cv_idx, val_cv_idx in gkf.split(X_imputed, y, groups=groups):
            X_cv_train, X_cv_val = X_imputed[train_cv_idx], X_imputed[val_cv_idx]
            y_cv_train, y_cv_val = y[train_cv_idx], y[val_cv_idx]

            cv_model = xgb.XGBClassifier(**params)
            cv_model.fit(X_cv_train, y_cv_train, verbose=False)
            y_cv_pred = cv_model.predict(X_cv_val)
            cv_scores.append(f1_score(y_cv_val, y_cv_pred, average="weighted"))

        cv_f1_mean = np.mean(cv_scores)
        cv_f1_std = np.std(cv_scores)
    else:
        logger.warning(f"Not enough groups ({n_groups}) for {n_cv_folds}-fold CV")
        cv_f1_mean = f1
        cv_f1_std = 0.0

    return {
        "accuracy": accuracy,
        "f1": f1,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "n_features_used": len(feature_names),
    }


def run_ablation_study(
    labels_dir: Path,
    output_dir: Path,
    mode: str = "full",
    seed: int = 999,
) -> tuple[list[dict], dict]:
    """Run the ablation study.

    Args:
        labels_dir: Path to labels directory
        output_dir: Output directory for results
        mode: "full", "single", or "category"
        seed: Random seed

    Returns:
        Tuple of (results list, summary dict)
    """
    results = []
    baseline_metrics = None

    # Validate feature categories
    if not validate_feature_categories():
        logger.warning("Feature category validation failed - some features may be missing")

    # Step 1: Baseline (all features)
    logger.info("Running baseline training with all features...")
    baseline_metrics = train_and_evaluate(labels_dir, exclude_features=None, seed=seed)

    baseline_result = {
        "experiment_type": "baseline",
        "excluded_features": "",
        "excluded_category": "",
        "n_features_used": baseline_metrics["n_features_used"],
        "accuracy": baseline_metrics["accuracy"],
        "f1": baseline_metrics["f1"],
        "cv_f1_mean": baseline_metrics["cv_f1_mean"],
        "cv_f1_std": baseline_metrics["cv_f1_std"],
        "accuracy_delta": 0.0,
        "f1_delta": 0.0,
        "classification": "baseline",
    }
    results.append(baseline_result)

    logger.info(
        f"Baseline: accuracy={baseline_metrics['accuracy']:.4f}, "
        f"f1={baseline_metrics['f1']:.4f}, "
        f"cv_f1_mean={baseline_metrics['cv_f1_mean']:.4f}"
    )

    # Step 2: Single-feature ablations
    if mode in ("full", "single"):
        logger.info(f"\nRunning single-feature ablations ({len(FEATURE_COLUMNS)} features)...")

        for i, feature in enumerate(FEATURE_COLUMNS, 1):
            logger.info(f"  [{i}/{len(FEATURE_COLUMNS)}] Excluding: {feature}")

            try:
                metrics = train_and_evaluate(labels_dir, exclude_features=[feature], seed=seed)

                accuracy_delta = metrics["accuracy"] - baseline_metrics["accuracy"]
                f1_delta = metrics["f1"] - baseline_metrics["f1"]
                classification = classify_feature(f1_delta)

                result = {
                    "experiment_type": "single_feature",
                    "excluded_features": feature,
                    "excluded_category": "",
                    "n_features_used": metrics["n_features_used"],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "cv_f1_mean": metrics["cv_f1_mean"],
                    "cv_f1_std": metrics["cv_f1_std"],
                    "accuracy_delta": accuracy_delta,
                    "f1_delta": f1_delta,
                    "classification": classification,
                }
                results.append(result)

                logger.info(f"    -> f1_delta={f1_delta:+.4f} ({classification})")

            except Exception as e:
                logger.error(f"    -> Failed: {e}")
                results.append(
                    {
                        "experiment_type": "single_feature",
                        "excluded_features": feature,
                        "excluded_category": "",
                        "n_features_used": 0,
                        "accuracy": 0.0,
                        "f1": 0.0,
                        "cv_f1_mean": 0.0,
                        "cv_f1_std": 0.0,
                        "accuracy_delta": 0.0,
                        "f1_delta": 0.0,
                        "classification": "error",
                    }
                )

    # Step 3: Category ablations
    if mode in ("full", "category"):
        logger.info(f"\nRunning category ablations ({len(FEATURE_CATEGORIES)} categories)...")

        for category_name, features in FEATURE_CATEGORIES.items():
            logger.info(f"  Excluding category: {category_name} ({len(features)} features)")

            try:
                metrics = train_and_evaluate(labels_dir, exclude_features=features, seed=seed)

                accuracy_delta = metrics["accuracy"] - baseline_metrics["accuracy"]
                f1_delta = metrics["f1"] - baseline_metrics["f1"]
                classification = classify_feature(f1_delta)

                result = {
                    "experiment_type": "category",
                    "excluded_features": ",".join(features),
                    "excluded_category": category_name,
                    "n_features_used": metrics["n_features_used"],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "cv_f1_mean": metrics["cv_f1_mean"],
                    "cv_f1_std": metrics["cv_f1_std"],
                    "accuracy_delta": accuracy_delta,
                    "f1_delta": f1_delta,
                    "classification": classification,
                }
                results.append(result)

                logger.info(f"    -> f1_delta={f1_delta:+.4f} ({classification})")

            except Exception as e:
                logger.error(f"    -> Failed: {e}")
                results.append(
                    {
                        "experiment_type": "category",
                        "excluded_features": ",".join(features),
                        "excluded_category": category_name,
                        "n_features_used": 0,
                        "accuracy": 0.0,
                        "f1": 0.0,
                        "cv_f1_mean": 0.0,
                        "cv_f1_std": 0.0,
                        "accuracy_delta": 0.0,
                        "f1_delta": 0.0,
                        "classification": "error",
                    }
                )

    # Generate summary
    summary = generate_summary(results, baseline_metrics)

    return results, summary


def generate_summary(results: list[dict], baseline_metrics: dict) -> dict:
    """Generate summary from ablation results.

    Args:
        results: List of result dicts
        baseline_metrics: Baseline metrics dict

    Returns:
        Summary dict
    """
    # Filter to single-feature results for ranking
    single_feature_results = [r for r in results if r["experiment_type"] == "single_feature"]

    # Sort by F1 delta (most negative = most important)
    ranked_by_importance = sorted(single_feature_results, key=lambda x: x["f1_delta"])

    # Identify noise candidates (F1 delta >= 0)
    noise_candidates = [
        r["excluded_features"]
        for r in single_feature_results
        if r["f1_delta"] >= NOISE_THRESHOLD and r["classification"] != "error"
    ]

    # Identify redundant features
    redundant_candidates = [
        r["excluded_features"] for r in single_feature_results if r["classification"] == "redundant"
    ]

    # Important features (bottom of list, most negative F1 delta)
    important_features = [
        r["excluded_features"]
        for r in ranked_by_importance[:10]
        if r["classification"] == "important"
    ]

    # Category impact ranking
    category_results = [r for r in results if r["experiment_type"] == "category"]
    category_ranking = sorted(
        [(r["excluded_category"], r["f1_delta"]) for r in category_results], key=lambda x: x[1]
    )

    # Classification counts
    classification_counts = defaultdict(int)
    for r in single_feature_results:
        classification_counts[r["classification"]] += 1

    summary = {
        "run_date": datetime.now(UTC).isoformat(),
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "f1": baseline_metrics["f1"],
            "cv_f1_mean": baseline_metrics["cv_f1_mean"],
            "cv_f1_std": baseline_metrics["cv_f1_std"],
            "n_features": baseline_metrics["n_features_used"],
        },
        "quality_check": {
            "meets_accuracy_threshold": bool(baseline_metrics["accuracy"] >= ACCURACY_THRESHOLD),
            "meets_cv_f1_threshold": bool(baseline_metrics["cv_f1_mean"] >= CV_F1_THRESHOLD),
            "accuracy_threshold": ACCURACY_THRESHOLD,
            "cv_f1_threshold": CV_F1_THRESHOLD,
        },
        "classification_thresholds": {
            "noise": f">= {NOISE_THRESHOLD}",
            "redundant": f"> {REDUNDANT_THRESHOLD}",
            "useful": f"> {USEFUL_THRESHOLD}",
            "important": f"<= {USEFUL_THRESHOLD}",
        },
        "classification_counts": dict(classification_counts),
        "feature_ranking_by_importance": [
            {
                "feature": r["excluded_features"],
                "f1_delta": r["f1_delta"],
                "classification": r["classification"],
            }
            for r in ranked_by_importance
        ],
        "category_ranking_by_importance": [
            {"category": cat, "f1_delta": delta} for cat, delta in category_ranking
        ],
        "noise_candidates": noise_candidates,
        "redundant_candidates": redundant_candidates,
        "important_features": important_features,
        "recommendations": {
            "safe_to_remove": noise_candidates,
            "consider_removing": redundant_candidates,
            "keep": important_features,
        },
    }

    return summary


def save_results(
    results: list[dict],
    summary: dict,
    output_dir: Path,
):
    """Save results to CSV and JSON.

    Args:
        results: List of result dicts
        summary: Summary dict
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / "ablation_results.csv"
    fieldnames = [
        "experiment_type",
        "excluded_features",
        "excluded_category",
        "n_features_used",
        "accuracy",
        "f1",
        "cv_f1_mean",
        "cv_f1_std",
        "accuracy_delta",
        "f1_delta",
        "classification",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    logger.info(f"Saved results to {csv_path}")

    # Save JSON summary
    json_path = output_dir / "ablation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary to {json_path}")


def print_summary(summary: dict):
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)

    baseline = summary["baseline"]
    print("\nBaseline Performance:")
    print(f"  Accuracy:    {baseline['accuracy']:.4f}")
    print(f"  F1:          {baseline['f1']:.4f}")
    print(f"  CV F1 Mean:  {baseline['cv_f1_mean']:.4f} ± {baseline['cv_f1_std']:.4f}")
    print(f"  Features:    {baseline['n_features']}")

    qc = summary["quality_check"]
    print("\nQuality Check:")
    acc_status = "✓" if qc["meets_accuracy_threshold"] else "✗"
    cv_status = "✓" if qc["meets_cv_f1_threshold"] else "✗"
    print(f"  {acc_status} Accuracy >= {qc['accuracy_threshold']:.0%}")
    print(f"  {cv_status} CV F1 >= {qc['cv_f1_threshold']:.0%}")

    counts = summary["classification_counts"]
    print("\nFeature Classifications:")
    print(f"  Noise:      {counts.get('noise', 0)}")
    print(f"  Redundant:  {counts.get('redundant', 0)}")
    print(f"  Useful:     {counts.get('useful', 0)}")
    print(f"  Important:  {counts.get('important', 0)}")

    if summary["noise_candidates"]:
        print("\nNoise Candidates (safe to remove):")
        for feat in summary["noise_candidates"]:
            print(f"  - {feat}")

    if summary["redundant_candidates"]:
        print("\nRedundant Candidates (consider removing):")
        for feat in summary["redundant_candidates"][:5]:
            print(f"  - {feat}")
        if len(summary["redundant_candidates"]) > 5:
            print(f"  ... and {len(summary['redundant_candidates']) - 5} more")

    if summary["important_features"]:
        print("\nMost Important Features (DO NOT remove):")
        for feat in summary["important_features"][:10]:
            print(f"  - {feat}")

    if summary["category_ranking_by_importance"]:
        print("\nCategory Importance Ranking:")
        for item in summary["category_ranking_by_importance"]:
            print(f"  {item['category']}: F1 delta = {item['f1_delta']:+.4f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Feature ablation study for identifying noise/redundant features"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("labels"),
        help="Labels directory (default: labels)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks"),
        help="Output directory (default: benchmarks)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "single", "category"],
        default="full",
        help="Ablation mode: full (all), single (features only), category (categories only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=999,
        help="Random seed for reproducibility (default: 999)",
    )

    args = parser.parse_args()

    if not args.labels.exists():
        logger.error(f"Labels directory not found: {args.labels}")
        sys.exit(1)

    logger.info("Starting ablation study...")
    logger.info(f"  Labels: {args.labels}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Total features: {len(FEATURE_COLUMNS)}")
    logger.info(f"  Total categories: {len(FEATURE_CATEGORIES)}")

    # Run ablation study
    results, summary = run_ablation_study(
        labels_dir=args.labels,
        output_dir=args.output,
        mode=args.mode,
        seed=args.seed,
    )

    # Save results
    save_results(results, summary, args.output)

    # Print summary
    print_summary(summary)

    # Verify expected row count
    expected_rows = 1  # baseline
    if args.mode in ("full", "single"):
        expected_rows += len(FEATURE_COLUMNS)
    if args.mode in ("full", "category"):
        expected_rows += len(FEATURE_CATEGORIES)

    actual_rows = len(results)
    if actual_rows != expected_rows:
        logger.warning(
            f"Expected {expected_rows} rows but got {actual_rows} "
            f"(mode={args.mode}, features={len(FEATURE_COLUMNS)}, categories={len(FEATURE_CATEGORIES)})"
        )
    else:
        logger.info(f"Generated {actual_rows} results (expected: {expected_rows})")

    logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()
