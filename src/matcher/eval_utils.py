"""Utilities for leave-one-out by type cross-validation.

Classifies labeled datasets into type groups (road_good, road_poor, sidewalk, other)
for LOO-by-type CV that tests cross-dataset generalization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .datasets.schema import get_dataset_config

if TYPE_CHECKING:
    import pandas as pd

# Manual type reclassifications for datasets whose YAML `type` field
# doesn't reflect their actual category for CV purposes.
TYPE_OVERRIDES: dict[str, str] = {
    "us_boston_bike_network": "bike",
}

# Minimum number of valid labels for a dataset to be included in LOO CV.
MIN_LOO_LABELS = 10

# Default threshold for splitting road datasets into good/poor quality.
DEFAULT_QUALITY_THRESHOLD = 0.5


def classify_dataset_type_group(
    dataset_name: str,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> str:
    """Classify a dataset into a type group for LOO CV.

    Groups:
        road_good: type=road AND min(name_cov, class_cov) >= threshold
        road_poor: type=road AND min(name_cov, class_cov) < threshold
        sidewalk:  type=sidewalk
        other:     type=bike/trail or anything else (merged)

    Args:
        dataset_name: Dataset name (matches YAML filename without extension)
        quality_threshold: Threshold for splitting road into good/poor

    Returns:
        One of: "road_good", "road_poor", "sidewalk", "other"
    """
    # Check for manual override first
    effective_type = TYPE_OVERRIDES.get(dataset_name)
    config = get_dataset_config(dataset_name)

    if effective_type is None:
        if config is None:
            return "other"
        effective_type = config.type

    # Non-road types
    if effective_type == "sidewalk":
        return "sidewalk"
    if effective_type in ("bike", "trail"):
        return "other"

    # For road type, split by quality fingerprint
    if effective_type == "road":
        if config is None or config.quality_fingerprint is None:
            return "road_poor"

        qf = config.quality_fingerprint
        min_cov = min(qf.name_coverage_ratio, qf.class_coverage_ratio)
        if min_cov >= quality_threshold:
            return "road_good"
        else:
            return "road_poor"

    # Unknown type -> other
    return "other"


def build_type_groups(
    dataset_names: list[str],
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> dict[str, list[str]]:
    """Group datasets by type for LOO CV.

    Args:
        dataset_names: List of dataset names to classify
        quality_threshold: Threshold for road good/poor split

    Returns:
        Dict mapping group name to list of dataset names in that group.
        Only groups with at least one dataset are included.
    """
    groups: dict[str, list[str]] = {}
    for name in dataset_names:
        group = classify_dataset_type_group(name, quality_threshold)
        groups.setdefault(group, []).append(name)
    return groups


@dataclass
class LooTypeCvResult:
    """Structured results from a LOO-by-type cross-validation run.

    ``rows`` contains one dict per (fold, held-out dataset) evaluation with
    keys: run_date, fold, dataset, type_group, n_train, n_test, n_match,
    n_no_match, accuracy, f1, precision, recall, seed, quality_threshold.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    type_groups: dict[str, list[str]] = field(default_factory=dict)
    excluded_datasets: list[str] = field(default_factory=list)
    n_total_labels: int = 0
    n_valid_labels: int = 0
    n_duplicates_dropped: int = 0
    cv_folds: int = 0
    seed: int = 0
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    run_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_frame(self) -> pd.DataFrame:
        """Return the per-(fold, dataset) rows as a DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.rows)

    def group_metrics(self) -> dict[str, dict[str, Any]]:
        """Per-type-group aggregates (macro means over per-dataset evals)."""
        results_df = self.to_frame()
        metrics: dict[str, dict[str, Any]] = {}
        if results_df.empty:
            return metrics
        for group in sorted(results_df["type_group"].unique()):
            group_df = results_df[results_df["type_group"] == group]
            metrics[group] = {
                "f1_mean": float(group_df["f1"].mean()),
                "f1_std": float(group_df["f1"].std(ddof=0)),
                "accuracy_mean": float(group_df["accuracy"].mean()),
                "accuracy_std": float(group_df["accuracy"].std(ddof=0)),
                "precision_mean": float(group_df["precision"].mean()),
                "recall_mean": float(group_df["recall"].mean()),
                "n_evals": int(len(group_df)),
                "n_labels": int(group_df["n_test"].sum()),
                "n_match": int(group_df["n_match"].sum()),
                "n_no_match": int(group_df["n_no_match"].sum()),
                "datasets": sorted(group_df["dataset"].unique()),
            }
        return metrics

    def overall_metrics(self) -> dict[str, Any]:
        """Overall aggregates (macro means over all per-dataset evals)."""
        results_df = self.to_frame()
        if results_df.empty:
            return {}
        return {
            "f1_mean": float(results_df["f1"].mean()),
            "f1_std": float(results_df["f1"].std(ddof=0)),
            "accuracy_mean": float(results_df["accuracy"].mean()),
            "accuracy_std": float(results_df["accuracy"].std(ddof=0)),
            "precision_mean": float(results_df["precision"].mean()),
            "recall_mean": float(results_df["recall"].mean()),
            "n_evals": int(len(results_df)),
            "n_labels": int(results_df["n_test"].sum()),
            "n_match": int(results_df["n_match"].sum()),
            "n_no_match": int(results_df["n_no_match"].sum()),
        }


def run_loo_by_type_cv(
    labels: pd.DataFrame | Path | str,
    cv_folds: int = 5,
    seed: int = 42,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    xgb_params: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> LooTypeCvResult:
    """Run leave-one-out by type cross-validation.

    Holds out one dataset per type group per fold (round-robin over groups
    shuffled with ``seed``), trains an XGBoost model on the remaining
    datasets, and evaluates on each held-out dataset independently.

    Args:
        labels: Either a labels directory (loaded via ``LabelStore.load_all``)
            or a pre-loaded labels DataFrame (as returned by ``load_all``).
        cv_folds: Number of round-robin folds.
        seed: Random seed for group shuffling and model training.
        quality_threshold: Threshold for the road_good/road_poor split.
        xgb_params: Optional XGBoost params override. Defaults to
            ``DEFAULT_XGB_PARAMS``. ``scale_pos_weight`` is always computed
            per fold from the training-class balance.
        log: Optional callable for progress output (e.g. ``console.print``).
            Receives rich-markup strings. No-op if None.

    Returns:
        LooTypeCvResult with per-(fold, dataset) rows plus per-group and
        overall aggregate helpers. ``rows`` is empty if no folds produced
        evaluations.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from xgboost import XGBClassifier

    from .config import METRIC_AVERAGE
    from .labeling.label_store import LabelStore
    from .matching.ml import DEFAULT_XGB_PARAMS, MLMatcher

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    params = dict(DEFAULT_XGB_PARAMS if xgb_params is None else xgb_params)
    run_date = datetime.now(UTC)

    if isinstance(labels, (str, Path)):
        labels_dir = Path(labels)
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        _log("[blue]Loading labels...[/blue]")
        all_labels = LabelStore.load_all(labels_dir)
    else:
        all_labels = labels

    n_total = len(all_labels)
    _log(f"  Total labels: {n_total}")

    valid_labels = {"match", "no_match"}
    all_labels = all_labels[all_labels["label"].isin(valid_labels)].copy()
    _log(f"  Valid labels (match/no_match): {len(all_labels)}")

    # Remove duplicates
    n_before = len(all_labels)
    all_labels = all_labels.drop_duplicates(subset=["gers_id", "target_id", "dataset"], keep="last")
    n_dropped = n_before - len(all_labels)
    if n_dropped > 0:
        _log(f"  [yellow]Dropped {n_dropped} duplicate pairs (keeping last)[/yellow]")

    # Filter datasets with too few labels
    dataset_counts = all_labels.groupby("dataset").size()
    valid_datasets = dataset_counts[dataset_counts >= MIN_LOO_LABELS].index.tolist()
    excluded_datasets = dataset_counts[dataset_counts < MIN_LOO_LABELS].index.tolist()
    if excluded_datasets:
        _log(
            f"  [yellow]Excluded {len(excluded_datasets)} dataset(s) with < "
            f"{MIN_LOO_LABELS} labels: {', '.join(excluded_datasets)}[/yellow]"
        )

    all_labels = all_labels[all_labels["dataset"].isin(valid_datasets)].copy()
    _log(f"  Datasets with >= {MIN_LOO_LABELS} labels: {len(valid_datasets)}")

    # Build type groups
    type_groups = build_type_groups(valid_datasets, quality_threshold)
    _log(f"\n[blue]Type groups (threshold={quality_threshold}):[/blue]")
    for group, datasets in sorted(type_groups.items()):
        _log(f"  {group}: {', '.join(sorted(datasets))}")

    # Initialize MLMatcher for feature extraction
    matcher = MLMatcher()

    # Build fold assignments via round-robin over shuffled groups
    rng = np.random.RandomState(seed)
    shuffled_groups: dict[str, list[str]] = {}
    for group, datasets in type_groups.items():
        shuffled = list(datasets)
        rng.shuffle(shuffled)
        shuffled_groups[group] = shuffled

    _log(f"\n[blue]Running {cv_folds}-fold LOO-by-type CV...[/blue]")

    # Collect per-fold, per-dataset results
    all_results: list[dict[str, Any]] = []

    for fold_idx in range(cv_folds):
        _log(f"\n  Fold {fold_idx + 1}/{cv_folds}:")

        # Select held-out datasets for this fold (round-robin)
        held_out: list[tuple[str, str]] = []  # (dataset, group)
        for group, datasets in sorted(shuffled_groups.items()):
            if fold_idx < len(datasets):
                held_out.append((datasets[fold_idx], group))

        if not held_out:
            _log("    [yellow]No datasets to hold out in this fold, skipping[/yellow]")
            continue

        held_out_names = {ds for ds, _ in held_out}
        train_df = all_labels[~all_labels["dataset"].isin(held_out_names)].copy()
        test_df = all_labels[all_labels["dataset"].isin(held_out_names)].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            _log("    [yellow]Empty train or test set, skipping[/yellow]")
            continue

        # Extract features and train
        X_train, y_train = matcher._extract_features_and_labels(train_df, binary=True)
        X_train = matcher._cap_infinities(X_train)

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())

        if n_pos == 0 or n_neg == 0:
            _log("    [yellow]Single-class training set, skipping fold[/yellow]")
            continue

        fold_spw = n_neg / n_pos

        model = XGBClassifier(
            **params,
            scale_pos_weight=fold_spw,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate on each held-out dataset independently
        for ds, group in held_out:
            ds_df = test_df[test_df["dataset"] == ds]
            if len(ds_df) == 0:
                continue

            X_ds, y_ds = matcher._extract_features_and_labels(ds_df, binary=True)
            X_ds = matcher._cap_infinities(X_ds)
            y_pred = model.predict(X_ds)

            n_match = int((y_ds == 1).sum())
            n_no_match = int((y_ds == 0).sum())
            acc = accuracy_score(y_ds, y_pred)
            f1 = f1_score(y_ds, y_pred, average=METRIC_AVERAGE, zero_division=0)
            prec = precision_score(y_ds, y_pred, average=METRIC_AVERAGE, zero_division=0)
            rec = recall_score(y_ds, y_pred, average=METRIC_AVERAGE, zero_division=0)

            _log(
                f"    {ds} ({group}): acc={acc:.3f}, f1={f1:.3f} "
                f"(n={len(ds_df)}, match={n_match}, no_match={n_no_match})"
            )

            all_results.append(
                {
                    "run_date": run_date.isoformat(),
                    "fold": fold_idx,
                    "dataset": ds,
                    "type_group": group,
                    "n_train": len(train_df),
                    "n_test": len(ds_df),
                    "n_match": n_match,
                    "n_no_match": n_no_match,
                    "accuracy": acc,
                    "f1": f1,
                    "precision": prec,
                    "recall": rec,
                    "seed": seed,
                    "quality_threshold": quality_threshold,
                }
            )

    return LooTypeCvResult(
        rows=all_results,
        type_groups=type_groups,
        excluded_datasets=list(excluded_datasets),
        n_total_labels=n_total,
        n_valid_labels=len(all_labels),
        n_duplicates_dropped=n_dropped,
        cv_folds=cv_folds,
        seed=seed,
        quality_threshold=quality_threshold,
        run_date=run_date,
    )
