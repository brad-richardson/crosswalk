#!/usr/bin/env python3
"""Feature ablation study script for identifying noise or redundant features.

This script tests feature value through paired grouped-CV removal ablations,
category-only models, permutation importance, and per-dataset availability. The
paired grouped-CV delta is the decision metric; the single holdout delta is retained
as a diagnostic but is not used to recommend feature removal.

Usage:
    # Full study (baseline + single features + categories)
    python scripts/ablation_study.py

    # Single-feature ablations only
    python scripts/ablation_study.py --mode single

    # Category ablations only
    python scripts/ablation_study.py --mode category

    # Each category by itself (standalone signal / redundancy check)
    python scripts/ablation_study.py --mode category-only

    # Missing/constant feature families by dataset (no model training)
    python scripts/ablation_study.py --mode coverage

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

from crosswalk.config import FEATURE_CATEGORIES as CONFIG_CATEGORIES
from crosswalk.config import FEATURE_COLUMNS, METRIC_AVERAGE
from crosswalk.labeling.label_store import LabelStore
from crosswalk.matching.ml import DEFAULT_XGB_PARAMS, MLMatcher, segment_aware_split


def _make_ablation_categories() -> dict[str, list[str]]:
    """Transform config FEATURE_CATEGORIES to snake_case keys for ablation.

    Also validates that all features exist in FEATURE_COLUMNS.
    """
    # Transform Title Case keys to snake_case
    categories = {}
    for key, features in CONFIG_CATEGORIES.items():
        snake_key = key.lower().replace(" ", "_").replace("/", "_")
        categories[snake_key] = features

    # Validate all features exist in FEATURE_COLUMNS
    all_category_features = set()
    for features in categories.values():
        all_category_features.update(features)

    missing = all_category_features - set(FEATURE_COLUMNS)
    if missing:
        raise ValueError(f"Features in categories but not in FEATURE_COLUMNS: {missing}")

    extra = set(FEATURE_COLUMNS) - all_category_features
    if extra:
        raise ValueError(f"Features in FEATURE_COLUMNS but not in categories: {extra}")

    return categories


# Feature categories derived from config.py (snake_case keys for ablation)
FEATURE_CATEGORIES = _make_ablation_categories()

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


def paired_cv_delta(
    candidate_scores: list[float], baseline_scores: list[float]
) -> tuple[float, float]:
    """Return mean/std of fold-paired candidate-minus-baseline F1 deltas."""
    candidate = np.asarray(candidate_scores, dtype=float)
    baseline = np.asarray(baseline_scores, dtype=float)
    if candidate.shape != baseline.shape or candidate.size == 0:
        raise ValueError(
            "paired CV requires non-empty score arrays with identical shapes; "
            f"got candidate={candidate.shape}, baseline={baseline.shape}"
        )
    deltas = candidate - baseline
    return float(deltas.mean()), float(deltas.std())


def build_feature_coverage(
    df, feature_categories: dict[str, list[str]] | None = None
) -> list[dict]:
    """Summarize missing and constant feature families for each dataset.

    A feature is usable only when it has at least two distinct non-null values
    in the slice. This is stricter than non-null coverage: an always-unknown
    class signal or always-empty name signal is stored but carries no
    within-dataset information.
    """
    categories = feature_categories or FEATURE_CATEGORIES
    valid = df[df["label"].isin({"match", "no_match"})].copy()
    slices = [("__all__", valid)]
    if "dataset" in valid:
        slices.extend((str(dataset), sub) for dataset, sub in valid.groupby("dataset"))

    rows = []
    for dataset, sub in slices:
        for category, features in categories.items():
            available = [feature for feature in features if feature in sub]
            missing_columns = sorted(set(features) - set(available))
            if available:
                values = sub[available].replace([np.inf, -np.inf], np.nan)
                observed_fraction = float(values.notna().to_numpy().mean())
                unique_counts = values.nunique(dropna=True)
                all_missing = sorted(unique_counts[unique_counts == 0].index.tolist())
                constant = sorted(unique_counts[unique_counts == 1].index.tolist())
                usable = sorted(unique_counts[unique_counts >= 2].index.tolist())
            else:
                observed_fraction = 0.0
                all_missing = []
                constant = []
                usable = []
            rows.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "rows": len(sub),
                    "positive_rows": int((sub["label"] == "match").sum()),
                    "features": len(features),
                    "observed_fraction": observed_fraction,
                    "usable_features": len(usable),
                    "usable_fraction": len(usable) / len(features) if features else 0.0,
                    "all_missing_features": ",".join(all_missing + missing_columns),
                    "constant_features": ",".join(constant),
                }
            )
    return rows


def _get_xgb_params(seed: int, scale_pos_weight: float) -> dict:
    """Return XGBoost hyperparameters from the shared default in ml.py."""
    try:
        import xgboost  # noqa: F401
    except ImportError as err:
        raise ImportError("XGBoost is required. Install with: pip install xgboost") from err

    return {
        **DEFAULT_XGB_PARAMS,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }


def _prepare_data(
    labels_dir: Path,
    seed: int,
    feature_names: list[str] | None = None,
):
    """Load labels, extract features, and split into train/test.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, X_all, y_all, groups, feature_names)
    """
    df = LabelStore.load_all(labels_dir)
    valid_labels = {"match", "no_match"}
    df = df[df["label"].isin(valid_labels)].copy()

    if len(df) == 0:
        raise ValueError("No valid labels found")

    if feature_names is None:
        feature_names = FEATURE_COLUMNS.copy()

    matcher = MLMatcher()
    matcher.feature_names = feature_names

    X, y = matcher._extract_features_and_labels(df, binary=True)

    train_idx, test_idx, groups = segment_aware_split(
        df, test_size=0.3, random_state=seed, return_groups=True
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Cap infinite values (XGBoost handles NaN natively but not inf)
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)
    X_all = np.where(np.isinf(X), np.nan, X)

    return X_train, X_test, y_train, y_test, X_all, y, groups, feature_names


def _train_model(X_train, y_train, X_test, y_test, seed: int):
    """Train an XGBoost model and return it with params."""
    import xgboost as xgb

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    params = _get_xgb_params(seed, scale_pos_weight)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model, params


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
    import xgboost as xgb

    if exclude_features:
        feature_names = [f for f in FEATURE_COLUMNS if f not in exclude_features]
    else:
        feature_names = None

    X_train, X_test, y_train, y_test, X_all, y_all, groups, feature_names = _prepare_data(
        labels_dir, seed, feature_names
    )

    model, params = _train_model(X_train, y_train, X_test, y_test, seed)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=METRIC_AVERAGE, zero_division=0)

    # Cross-validation with segment-aware folding
    n_groups = groups.nunique()

    if n_groups >= n_cv_folds:
        gkf = GroupKFold(n_splits=n_cv_folds)
        cv_scores = []
        for train_cv_idx, val_cv_idx in gkf.split(X_all, y_all, groups=groups):
            X_cv_train, X_cv_val = X_all[train_cv_idx], X_all[val_cv_idx]
            y_cv_train, y_cv_val = y_all[train_cv_idx], y_all[val_cv_idx]

            cv_model = xgb.XGBClassifier(**params)
            cv_model.fit(X_cv_train, y_cv_train, verbose=False)
            y_cv_pred = cv_model.predict(X_cv_val)
            cv_scores.append(f1_score(y_cv_val, y_cv_pred, average=METRIC_AVERAGE, zero_division=0))

        cv_f1_mean = np.mean(cv_scores)
        cv_f1_std = np.std(cv_scores)
    else:
        logger.warning(f"Not enough groups ({n_groups}) for {n_cv_folds}-fold CV")
        cv_scores = [float(f1)]
        cv_f1_mean = f1
        cv_f1_std = 0.0

    return {
        "accuracy": accuracy,
        "f1": f1,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "cv_f1_scores": [float(score) for score in cv_scores],
        "n_features_used": len(feature_names),
    }


def run_permutation_importance(
    labels_dir: Path,
    output_dir: Path,
    seed: int = 999,
    n_repeats: int = 10,
) -> list[dict]:
    """Run permutation importance analysis.

    Trains one baseline model, then uses sklearn's permutation_importance
    to measure per-feature importance by shuffling feature values on the test
    set. This avoids the redundancy masking problem of single-feature ablation.

    Args:
        labels_dir: Path to labels directory
        output_dir: Output directory for results
        seed: Random seed for reproducibility
        n_repeats: Number of permutation repeats per feature

    Returns:
        List of per-feature importance dicts
    """
    from sklearn.inspection import permutation_importance

    logger.info("Running permutation importance analysis...")

    X_train, X_test, y_train, y_test, _, _, _, feature_names = _prepare_data(labels_dir, seed)
    model, _ = _train_model(X_train, y_train, X_test, y_test, seed)

    # Baseline test F1
    y_pred = model.predict(X_test)
    baseline_f1 = f1_score(y_test, y_pred, average=METRIC_AVERAGE, zero_division=0)
    logger.info(f"Baseline test F1: {baseline_f1:.4f}")

    # Permutation importance
    logger.info(f"Computing permutation importance ({n_repeats} repeats)...")
    perm_result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="f1",
        n_jobs=-1,
    )

    # Load existing ablation results for cross-referencing (if available)
    ablation_classifications = {}
    ablation_csv = output_dir / "ablation_results.csv"
    if ablation_csv.exists():
        logger.info(f"Loading ablation results from {ablation_csv} for cross-reference")
        import csv as csv_module

        with open(ablation_csv) as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                if row["experiment_type"] == "single_feature":
                    ablation_classifications[row["excluded_features"]] = row["classification"]

    # Build results
    results = []
    for i, feat_name in enumerate(feature_names):
        result = {
            "feature": feat_name,
            "importance_mean": float(perm_result.importances_mean[i]),
            "importance_std": float(perm_result.importances_std[i]),
            "ablation_classification": ablation_classifications.get(feat_name, ""),
        }
        results.append(result)

    # Sort by importance (most important first)
    results.sort(key=lambda x: -x["importance_mean"])

    # Save CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "permutation_importance.csv"
    fieldnames = ["feature", "importance_mean", "importance_std", "ablation_classification"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    logger.info(f"Saved permutation importance to {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("PERMUTATION IMPORTANCE RESULTS")
    print("=" * 70)
    print(f"\nBaseline test F1: {baseline_f1:.4f}")
    print(f"Features: {len(feature_names)}, Repeats: {n_repeats}\n")

    print(f"{'Feature':<45} {'Importance':>12} {'Std':>8} {'Ablation':>12}")
    print("-" * 80)
    for r in results[:20]:
        print(
            f"{r['feature']:<45} {r['importance_mean']:>12.4f} "
            f"{r['importance_std']:>8.4f} {r['ablation_classification']:>12}"
        )
    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more features")

    # Flag potential false negatives: features classified as "noise" by ablation
    # but showing meaningful permutation importance
    if ablation_classifications:
        false_negatives = [
            r
            for r in results
            if r["ablation_classification"] in ("noise", "redundant")
            and r["importance_mean"] > 0.005
        ]
        if false_negatives:
            print(f"\nPotential false negatives ({len(false_negatives)} features):")
            print(
                "  (Classified as noise/redundant by ablation but meaningful permutation importance)"
            )
            for r in false_negatives:
                print(
                    f"  - {r['feature']}: importance={r['importance_mean']:.4f}, "
                    f"ablation={r['ablation_classification']}"
                )

    print("\n" + "=" * 70)

    return results


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
        mode: "full", "single", "category", or "category-only"
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
        "cv_f1_delta": 0.0,
        "cv_f1_delta_std": 0.0,
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
                cv_f1_delta, cv_f1_delta_std = paired_cv_delta(
                    metrics["cv_f1_scores"], baseline_metrics["cv_f1_scores"]
                )
                classification = classify_feature(cv_f1_delta)

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
                    "cv_f1_delta": cv_f1_delta,
                    "cv_f1_delta_std": cv_f1_delta_std,
                    "classification": classification,
                }
                results.append(result)

                logger.info(
                    f"    -> paired cv_f1_delta={cv_f1_delta:+.4f} "
                    f"± {cv_f1_delta_std:.4f} ({classification}); "
                    f"holdout={f1_delta:+.4f}"
                )

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
                        "cv_f1_delta": 0.0,
                        "cv_f1_delta_std": 0.0,
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
                cv_f1_delta, cv_f1_delta_std = paired_cv_delta(
                    metrics["cv_f1_scores"], baseline_metrics["cv_f1_scores"]
                )
                classification = classify_feature(cv_f1_delta)

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
                    "cv_f1_delta": cv_f1_delta,
                    "cv_f1_delta_std": cv_f1_delta_std,
                    "classification": classification,
                }
                results.append(result)

                logger.info(
                    f"    -> paired cv_f1_delta={cv_f1_delta:+.4f} "
                    f"± {cv_f1_delta_std:.4f} ({classification}); "
                    f"holdout={f1_delta:+.4f}"
                )

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
                        "cv_f1_delta": 0.0,
                        "cv_f1_delta_std": 0.0,
                        "classification": "error",
                    }
                )

    # Step 4: category-only models measure standalone predictive signal,
    # complementing removal ablation when correlated families mask one another.
    if mode == "category-only":
        logger.info(f"\nRunning category-only models ({len(FEATURE_CATEGORIES)} categories)...")
        for category_name, features in FEATURE_CATEGORIES.items():
            logger.info(f"  Including only: {category_name} ({len(features)} features)")
            excluded = [feature for feature in FEATURE_COLUMNS if feature not in features]
            try:
                metrics = train_and_evaluate(labels_dir, exclude_features=excluded, seed=seed)
                accuracy_delta = metrics["accuracy"] - baseline_metrics["accuracy"]
                f1_delta = metrics["f1"] - baseline_metrics["f1"]
                cv_f1_delta, cv_f1_delta_std = paired_cv_delta(
                    metrics["cv_f1_scores"], baseline_metrics["cv_f1_scores"]
                )
                results.append(
                    {
                        "experiment_type": "category_only",
                        "excluded_features": ",".join(excluded),
                        "excluded_category": "",
                        "included_category": category_name,
                        "n_features_used": metrics["n_features_used"],
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "cv_f1_mean": metrics["cv_f1_mean"],
                        "cv_f1_std": metrics["cv_f1_std"],
                        "accuracy_delta": accuracy_delta,
                        "f1_delta": f1_delta,
                        "cv_f1_delta": cv_f1_delta,
                        "cv_f1_delta_std": cv_f1_delta_std,
                        "classification": "standalone",
                    }
                )
            except Exception as e:
                logger.error(f"    -> Failed: {e}")
                results.append(
                    {
                        "experiment_type": "category_only",
                        "excluded_features": ",".join(excluded),
                        "excluded_category": "",
                        "included_category": category_name,
                        "n_features_used": 0,
                        "accuracy": 0.0,
                        "f1": 0.0,
                        "cv_f1_mean": 0.0,
                        "cv_f1_std": 0.0,
                        "accuracy_delta": 0.0,
                        "f1_delta": 0.0,
                        "cv_f1_delta": 0.0,
                        "cv_f1_delta_std": 0.0,
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

    # Sort by fold-paired grouped-CV delta (most negative = most important).
    ranked_by_importance = sorted(single_feature_results, key=lambda x: x["cv_f1_delta"])

    # Identify investigation candidates. A single run is never sufficient to
    # call a feature safe to remove; permutation, category-only, multi-seed,
    # and per-dataset evidence still need to agree.
    noise_candidates = [
        r["excluded_features"]
        for r in single_feature_results
        if r["cv_f1_delta"] >= NOISE_THRESHOLD and r["classification"] != "error"
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
        [(r["excluded_category"], r["cv_f1_delta"]) for r in category_results],
        key=lambda x: x[1],
    )
    category_only_results = [r for r in results if r["experiment_type"] == "category_only"]
    category_only_ranking = sorted(
        [
            (r["included_category"], r["cv_f1_mean"])
            for r in category_only_results
            if r["classification"] != "error"
        ],
        key=lambda x: x[1],
        reverse=True,
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
            "metric": "fold-paired grouped-CV F1 delta",
            "noise": f">= {NOISE_THRESHOLD}",
            "redundant": f"> {REDUNDANT_THRESHOLD}",
            "useful": f"> {USEFUL_THRESHOLD}",
            "important": f"<= {USEFUL_THRESHOLD}",
        },
        "classification_counts": dict(classification_counts),
        "feature_ranking_by_importance": [
            {
                "feature": r["excluded_features"],
                "cv_f1_delta": r["cv_f1_delta"],
                "cv_f1_delta_std": r["cv_f1_delta_std"],
                "holdout_f1_delta": r["f1_delta"],
                "classification": r["classification"],
            }
            for r in ranked_by_importance
        ],
        "category_ranking_by_importance": [
            {"category": cat, "cv_f1_delta": delta} for cat, delta in category_ranking
        ],
        "category_only_ranking": [
            {"category": cat, "cv_f1_mean": score} for cat, score in category_only_ranking
        ],
        "noise_candidates": noise_candidates,
        "redundant_candidates": redundant_candidates,
        "important_features": important_features,
        "recommendations": {
            "safe_to_remove": [],
            "investigate_removal": noise_candidates + redundant_candidates,
            "keep": important_features,
            "removal_gate": (
                "Require agreement across paired grouped-CV ablation, permutation importance, "
                "multi-seed stability, and per-dataset coverage before removing a feature."
            ),
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
        "included_category",
        "n_features_used",
        "accuracy",
        "f1",
        "cv_f1_mean",
        "cv_f1_std",
        "accuracy_delta",
        "f1_delta",
        "cv_f1_delta",
        "cv_f1_delta_std",
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
        print("\nNoise Candidates (investigate; not automatically safe to remove):")
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
        print("\nCategory Importance Ranking (paired grouped-CV):")
        for item in summary["category_ranking_by_importance"]:
            print(f"  {item['category']}: CV F1 delta = {item['cv_f1_delta']:+.4f}")

    if summary["category_only_ranking"]:
        print("\nCategory-only Standalone Ranking (grouped-CV):")
        for item in summary["category_only_ranking"]:
            print(f"  {item['category']}: CV F1 = {item['cv_f1_mean']:.4f}")

    print("\n" + "=" * 70)


def save_feature_coverage(labels_dir: Path, output_dir: Path) -> list[dict]:
    """Write per-dataset category availability and dead-zone diagnostics."""
    rows = build_feature_coverage(LabelStore.load_all(labels_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "feature_coverage.csv"
    fieldnames = [
        "dataset",
        "category",
        "rows",
        "positive_rows",
        "features",
        "observed_fraction",
        "usable_features",
        "usable_fraction",
        "all_missing_features",
        "constant_features",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved per-dataset feature coverage to {path}")
    return rows


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
        choices=["full", "single", "category", "category-only", "permutation", "coverage"],
        default="full",
        help=(
            "Ablation mode: full (removal ablations), single, category, "
            "category-only (standalone family signal), permutation, or coverage"
        ),
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

    # Availability is cheap and is required context for every model-based run:
    # a globally useful feature family can still be dead for one dataset.
    save_feature_coverage(args.labels, args.output)

    if args.mode == "coverage":
        logger.info("Feature coverage analysis complete!")
    elif args.mode == "permutation":
        # Permutation importance mode
        run_permutation_importance(
            labels_dir=args.labels,
            output_dir=args.output,
            seed=args.seed,
        )
        logger.info("Permutation importance analysis complete!")
    else:
        # Model-based ablation modes.
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
        if args.mode == "category-only":
            expected_rows += len(FEATURE_CATEGORIES)

        actual_rows = len(results)
        if actual_rows != expected_rows:
            logger.warning(
                f"Expected {expected_rows} rows but got {actual_rows} "
                f"(mode={args.mode}, features={len(FEATURE_COLUMNS)}, "
                f"categories={len(FEATURE_CATEGORIES)})"
            )
        else:
            logger.info(f"Generated {actual_rows} results (expected: {expected_rows})")

        logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()
