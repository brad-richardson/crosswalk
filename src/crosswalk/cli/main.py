"""Top-level CLI commands."""

import csv
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import typer

from ..eval_utils import DEFAULT_QUALITY_THRESHOLD
from .utils import console


def _auto_fetch_overture(dataset: str, data_dir: Path) -> Path | None:
    """Auto-fetch Overture segments when missing, using dataset config bbox.

    Delegates to the existing fetch reference implementation to avoid
    duplicating the fetch logic.

    Returns the path to the fetched file, or None if no config found.
    """
    from ..datasets.schema import get_dataset_config
    from ..filenames import find_overture_segments
    from .data import _fetch_reference_impl
    from .utils import console

    config = get_dataset_config(dataset)
    if config is None or config.fetch is None or config.fetch.bbox is None:
        return None

    console.print(f"  [blue]Auto-fetching Overture data for {dataset}...[/blue]")
    _fetch_reference_impl(
        dataset_name=dataset,
        output_dir=data_dir,
        sources={"overture"},
    )

    return find_overture_segments(data_dir, dataset)


def _eval_existing_model(
    model: Path,
    labels_dir: Path,
    by_dataset: bool,
    dataset: list[str],
    seed: int,
    output_dir: Path,
    skip_save: bool,
) -> None:
    """Evaluate an existing trained model on labeled data using 20% holdout."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    from ..config import METRIC_AVERAGE
    from ..labeling.label_store import LabelStore
    from ..matching.ml import MLMatcher, segment_aware_split

    if not model.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    run_date = datetime.now(UTC)

    if dataset:
        console.print(f"[blue]Filtering to datasets: {', '.join(dataset)}[/blue]")

    console.print(f"[blue]Evaluating {model.name} on 20% holdout (seed={seed})...[/blue]")

    # Load model. Evaluating an existing (possibly older) model is the point of
    # this flow, so a feature_version mismatch warns rather than blocking.
    matcher = MLMatcher()
    matcher.load_model(str(model), allow_version_mismatch=True)

    # Load labels
    console.print("[blue]Loading labels...[/blue]")
    all_labels = LabelStore.load_all(labels_dir)
    console.print(f"  Total labels: {len(all_labels)}")

    # Filter to valid labels only
    valid_labels = {"match", "no_match"}
    all_labels = all_labels[all_labels["label"].isin(valid_labels)].copy()
    console.print(f"  Valid labels (match/no_match): {len(all_labels)}")

    # Filter to specific datasets if requested
    if dataset:
        all_labels = all_labels[all_labels["dataset"].isin(dataset)].copy()
        console.print(f"  Filtered to datasets {dataset}: {len(all_labels)}")

    # Segment-aware split to get test set (20% holdout)
    _, test_idx = segment_aware_split(all_labels, test_size=0.2, random_state=seed)
    test_df = all_labels.iloc[test_idx].copy()
    console.print(f"  Test set: {len(test_df)} samples")

    # Evaluate
    X_test, y_test = matcher._extract_features_and_labels(test_df, binary=True)
    X_test = matcher._cap_infinities(X_test)
    y_pred = matcher.model.predict(X_test)

    # Overall metrics
    overall_acc = accuracy_score(y_test, y_pred)
    overall_f1 = f1_score(y_test, y_pred, average=METRIC_AVERAGE, zero_division=0)
    overall_precision = precision_score(y_test, y_pred, average=METRIC_AVERAGE, zero_division=0)
    overall_recall = recall_score(y_test, y_pred, average=METRIC_AVERAGE, zero_division=0)

    console.print(f"\n{'=' * 60}")
    console.print(f"[bold]EVALUATION ON 20% HOLDOUT ({len(test_df)} samples)[/bold]")
    console.print("=" * 60)
    console.print("\nOverall:")
    console.print(f"  Accuracy:  {overall_acc:.3f}")
    console.print(f"  F1:        {overall_f1:.3f}")
    console.print(f"  Precision: {overall_precision:.3f}")
    console.print(f"  Recall:    {overall_recall:.3f}")

    # Feature importances (if available)
    top_features = []
    if hasattr(matcher.model, "feature_importances_"):
        feature_importances = dict(zip(matcher.feature_names, matcher.model.feature_importances_))
        top_features = sorted(feature_importances.items(), key=lambda x: -x[1])[:10]
        console.print("\nTop 10 features by importance:")
        for feat, imp in top_features:
            console.print(f"  {feat}: {imp:.3f}")

    # Per-dataset metrics
    results = {}
    if by_dataset:
        console.print("\nPer-dataset results:")
        for ds in sorted(test_df["dataset"].unique()):
            ds_test = test_df[test_df["dataset"] == ds]
            X_ds, y_ds = matcher._extract_features_and_labels(ds_test, binary=True)
            X_ds = matcher._cap_infinities(X_ds)
            y_ds_pred = matcher.model.predict(X_ds)

            ds_acc = accuracy_score(y_ds, y_ds_pred)
            ds_f1 = f1_score(y_ds, y_ds_pred, average=METRIC_AVERAGE)
            ds_precision = precision_score(y_ds, y_ds_pred, average=METRIC_AVERAGE, zero_division=0)
            ds_recall = recall_score(y_ds, y_ds_pred, average=METRIC_AVERAGE, zero_division=0)
            n_match = int((y_ds == 1).sum())
            n_no_match = int((y_ds == 0).sum())

            console.print(
                f"  {ds}: acc={ds_acc:.3f}, f1={ds_f1:.3f} "
                f"(n={len(ds_test)}, match={n_match}, no_match={n_no_match})"
            )

            results[ds] = {
                "n_samples": len(ds_test),
                "n_match": n_match,
                "n_no_match": n_no_match,
                "accuracy": ds_acc,
                "f1": ds_f1,
                "precision": ds_precision,
                "recall": ds_recall,
            }
    else:
        results["overall"] = {
            "n_samples": len(test_df),
            "n_match": int((y_test == 1).sum()),
            "n_no_match": int((y_test == 0).sum()),
            "accuracy": overall_acc,
            "f1": overall_f1,
            "precision": overall_precision,
            "recall": overall_recall,
        }

    # Save results to CSV
    if not skip_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "ml_eval_results.csv"

        fieldnames = [
            "run_date",
            "data_pull_date",
            "dataset",
            "n_train",
            "n_test",
            "train_size",
            "n_samples",
            "n_match",
            "n_no_match",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "split_seed",
            "model_name",
            "top1_feature",
            "top1_importance",
            "top2_feature",
            "top2_importance",
            "top3_feature",
            "top3_importance",
            "top4_feature",
            "top4_importance",
            "top5_feature",
            "top5_importance",
            "top6_feature",
            "top6_importance",
            "top7_feature",
            "top7_importance",
            "top8_feature",
            "top8_importance",
            "top9_feature",
            "top9_importance",
            "top10_feature",
            "top10_importance",
        ]

        write_header = not results_file.exists()

        with open(results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            for dataset_name, metrics in results.items():
                row = {
                    "run_date": run_date.isoformat(),
                    "data_pull_date": run_date.isoformat(),
                    "dataset": dataset_name,
                    "n_train": "N/A (existing model)",
                    "n_test": len(test_df),
                    "train_size": 0.8,  # Fixed 80/20 split for existing model eval
                    "n_samples": metrics.get("n_samples", 0),
                    "n_match": metrics.get("n_match", 0),
                    "n_no_match": metrics.get("n_no_match", 0),
                    "accuracy": f"{metrics.get('accuracy', 0):.4f}",
                    "f1": f"{metrics.get('f1', 0):.4f}",
                    "precision": f"{metrics.get('precision', 0):.4f}",
                    "recall": f"{metrics.get('recall', 0):.4f}",
                    "split_seed": seed,
                    "model_name": model.name,
                    **{
                        f"top{i + 1}_feature": top_features[i][0] if len(top_features) > i else ""
                        for i in range(10)
                    },
                    **{
                        f"top{i + 1}_importance": f"{top_features[i][1]:.4f}"
                        if len(top_features) > i
                        else ""
                        for i in range(10)
                    },
                }
                writer.writerow(row)

        console.print(f"\n[green]Results saved to {results_file}[/green]")

    console.print("[green]Evaluation complete[/green]")


def _cross_validate(
    labels_dir: Path,
    output_dir: Path,
    cv_folds: int,
    seed: int,
    skip_save: bool,
    by_dataset: bool,
    filter_datasets: list[str] | None,
) -> None:
    """Run k-fold cross-validation with segment-aware splitting."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import GroupKFold

    from ..config import METRIC_AVERAGE
    from ..labeling.label_store import LabelStore
    from ..matching.ml import DEFAULT_XGB_PARAMS, MLMatcher, create_segment_groups

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    run_date = datetime.now(UTC)

    console.print("[blue]Loading labels...[/blue]")
    all_labels = LabelStore.load_all(labels_dir)
    console.print(f"  Total labels: {len(all_labels)}")

    # Filter to valid labels only
    valid_labels = {"match", "no_match"}
    all_labels = all_labels[all_labels["label"].isin(valid_labels)].copy()
    console.print(f"  Valid labels (match/no_match): {len(all_labels)}")

    # Filter to specific datasets if requested
    if filter_datasets:
        all_labels = all_labels[all_labels["dataset"].isin(filter_datasets)].copy()
        console.print(f"  Filtered to datasets {filter_datasets}: {len(all_labels)}")

    # Remove duplicates
    n_before = len(all_labels)
    all_labels = all_labels.drop_duplicates(subset=["gers_id", "target_id", "dataset"], keep="last")
    n_dropped = n_before - len(all_labels)
    if n_dropped > 0:
        console.print(f"  [yellow]Dropped {n_dropped} duplicate pairs (keeping first)[/yellow]")

    # Create segment groups for segment-aware CV using Union-Find
    # Pairs sharing any segment (gers_id or target_id) are grouped together
    groups = create_segment_groups(all_labels).values

    n_groups = len(np.unique(groups))
    actual_folds = min(cv_folds, n_groups)
    if actual_folds < cv_folds:
        console.print(
            f"  [yellow]Reduced to {actual_folds} folds (only {n_groups} segment groups)[/yellow]"
        )

    console.print(f"\n[blue]Running {actual_folds}-fold cross-validation...[/blue]")

    # Initialize MLMatcher to get feature extraction methods
    matcher = MLMatcher()

    # Extract features and labels once (imputation is done per-fold below)
    X, y = matcher._extract_features_and_labels(all_labels, binary=True)

    # Track metrics across folds
    fold_metrics = {
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }

    # Per-dataset metrics across folds
    dataset_fold_metrics: dict[str, dict[str, list]] = {}

    gkf = GroupKFold(n_splits=actual_folds)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        console.print(f"  Fold {fold_idx + 1}/{actual_folds}...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Cap infinities (XGBoost handles NaN natively)
        X_train = matcher._cap_infinities(X_train)
        X_test = matcher._cap_infinities(X_test)

        # Train model for this fold
        from xgboost import XGBClassifier

        # Compute scale_pos_weight for this fold's class balance
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        fold_spw = n_neg / n_pos if n_pos > 0 else 1.0

        model = XGBClassifier(
            **DEFAULT_XGB_PARAMS,
            scale_pos_weight=fold_spw,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Predict on test fold
        y_pred = model.predict(X_test)

        # Compute overall metrics for this fold
        fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        fold_metrics["f1"].append(f1_score(y_test, y_pred, average=METRIC_AVERAGE, zero_division=0))
        fold_metrics["precision"].append(
            precision_score(y_test, y_pred, average=METRIC_AVERAGE, zero_division=0)
        )
        fold_metrics["recall"].append(
            recall_score(y_test, y_pred, average=METRIC_AVERAGE, zero_division=0)
        )

        # Per-dataset metrics for this fold
        if by_dataset:
            test_df = all_labels.iloc[test_idx]
            for ds in test_df["dataset"].unique():
                ds_mask = test_df["dataset"] == ds
                ds_indices = np.where(ds_mask.values)[0]
                if len(ds_indices) == 0:
                    continue

                y_ds = y_test[ds_indices]
                y_ds_pred = y_pred[ds_indices]

                if ds not in dataset_fold_metrics:
                    dataset_fold_metrics[ds] = {
                        "accuracy": [],
                        "f1": [],
                        "precision": [],
                        "recall": [],
                        "n_samples": [],
                    }

                dataset_fold_metrics[ds]["accuracy"].append(accuracy_score(y_ds, y_ds_pred))
                dataset_fold_metrics[ds]["f1"].append(
                    f1_score(y_ds, y_ds_pred, average=METRIC_AVERAGE, zero_division=0)
                )
                dataset_fold_metrics[ds]["precision"].append(
                    precision_score(y_ds, y_ds_pred, average=METRIC_AVERAGE, zero_division=0)
                )
                dataset_fold_metrics[ds]["recall"].append(
                    recall_score(y_ds, y_ds_pred, average=METRIC_AVERAGE, zero_division=0)
                )
                dataset_fold_metrics[ds]["n_samples"].append(len(ds_indices))

    # Compute mean and std for overall metrics
    console.print(f"\n{'=' * 60}")
    console.print(
        f"[bold]{actual_folds}-FOLD CROSS-VALIDATION RESULTS ({len(all_labels)} samples)[/bold]"
    )
    console.print("=" * 60)

    console.print("\nOverall (mean ± std):")
    overall_results = {}
    for metric in ["accuracy", "f1", "precision", "recall"]:
        mean_val = np.mean(fold_metrics[metric])
        std_val = np.std(fold_metrics[metric])
        console.print(f"  {metric.capitalize():12s} {mean_val:.3f} ± {std_val:.3f}")
        overall_results[f"{metric}_mean"] = mean_val
        overall_results[f"{metric}_std"] = std_val

    # Per-dataset results
    MIN_DATASET_SAMPLES = 10
    dataset_results = {}
    skipped_datasets = []
    if by_dataset and dataset_fold_metrics:
        console.print("\nPer-dataset results (mean ± std):")
        for ds in sorted(dataset_fold_metrics.keys()):
            ds_metrics = dataset_fold_metrics[ds]
            f1_mean = np.mean(ds_metrics["f1"])
            f1_std = np.std(ds_metrics["f1"])
            acc_mean = np.mean(ds_metrics["accuracy"])
            n_samples = sum(ds_metrics["n_samples"])

            if n_samples < MIN_DATASET_SAMPLES:
                skipped_datasets.append((ds, n_samples))
                continue

            console.print(
                f"  {ds}: F1={f1_mean:.3f}±{f1_std:.3f}, Acc={acc_mean:.3f} (n={n_samples})"
            )

            dataset_results[ds] = {
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "accuracy_mean": acc_mean,
                "accuracy_std": np.std(ds_metrics["accuracy"]),
                "precision_mean": np.mean(ds_metrics["precision"]),
                "recall_mean": np.mean(ds_metrics["recall"]),
                "n_samples": n_samples,
            }

        if skipped_datasets:
            console.print(
                f"\n  [yellow]Skipped {len(skipped_datasets)} dataset(s) with < "
                f"{MIN_DATASET_SAMPLES} samples:[/yellow]"
            )
            for ds, n in skipped_datasets:
                console.print(f"    {ds} (n={n})")

    # Save results to CSV
    if not skip_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "ml_cv_results.csv"

        fieldnames = [
            "run_date",
            "dataset",
            "cv_folds",
            "n_samples",
            "f1_mean",
            "f1_std",
            "accuracy_mean",
            "accuracy_std",
            "precision_mean",
            "recall_mean",
            "seed",
        ]

        write_header = not results_file.exists()

        with open(results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            # Write overall row
            writer.writerow(
                {
                    "run_date": run_date.isoformat(),
                    "dataset": "overall",
                    "cv_folds": actual_folds,
                    "n_samples": len(all_labels),
                    "f1_mean": f"{overall_results['f1_mean']:.4f}",
                    "f1_std": f"{overall_results['f1_std']:.4f}",
                    "accuracy_mean": f"{overall_results['accuracy_mean']:.4f}",
                    "accuracy_std": f"{overall_results['accuracy_std']:.4f}",
                    "precision_mean": f"{overall_results['precision_mean']:.4f}",
                    "recall_mean": f"{overall_results['recall_mean']:.4f}",
                    "seed": seed,
                }
            )

            # Write per-dataset rows
            for ds, metrics in dataset_results.items():
                writer.writerow(
                    {
                        "run_date": run_date.isoformat(),
                        "dataset": ds,
                        "cv_folds": actual_folds,
                        "n_samples": metrics["n_samples"],
                        "f1_mean": f"{metrics['f1_mean']:.4f}",
                        "f1_std": f"{metrics['f1_std']:.4f}",
                        "accuracy_mean": f"{metrics['accuracy_mean']:.4f}",
                        "accuracy_std": f"{metrics['accuracy_std']:.4f}",
                        "precision_mean": f"{metrics['precision_mean']:.4f}",
                        "recall_mean": f"{metrics['recall_mean']:.4f}",
                        "seed": seed,
                    }
                )

        console.print(f"\n[green]Results saved to {results_file}[/green]")

    console.print("\n[green]Cross-validation complete![/green]")


def _loo_type_cross_validate(
    labels_dir: Path,
    output_dir: Path,
    cv_folds: int,
    seed: int,
    skip_save: bool,
    quality_threshold: float,
) -> None:
    """Run leave-one-out by type cross-validation.

    Thin CLI wrapper around :func:`crosswalk.eval_utils.run_loo_by_type_cv`:
    delegates the CV loop, then prints the summary and appends results to CSV.
    """
    from ..eval_utils import run_loo_by_type_cv

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    cv_result = run_loo_by_type_cv(
        labels=labels_dir,
        cv_folds=cv_folds,
        seed=seed,
        quality_threshold=quality_threshold,
        log=console.print,
    )

    run_date = cv_result.run_date
    all_results = cv_result.rows

    if not all_results:
        console.print("[red]No results produced - check dataset labels and type groups[/red]")
        raise typer.Exit(1)

    # Print per-type summary
    results_df = cv_result.to_frame()

    console.print(f"\n{'=' * 60}")
    console.print(f"[bold]LOO-BY-TYPE CV SUMMARY ({cv_folds} folds)[/bold]")
    console.print("=" * 60)

    for group in sorted(results_df["type_group"].unique()):
        group_df = results_df[results_df["type_group"] == group]
        console.print(
            f"\n  {group}: F1={group_df['f1'].mean():.3f}±{group_df['f1'].std(ddof=0):.3f}, "
            f"Acc={group_df['accuracy'].mean():.3f}±{group_df['accuracy'].std(ddof=0):.3f} "
            f"({len(group_df)} evals)"
        )

    console.print(
        f"\n  Overall: F1={results_df['f1'].mean():.3f}±{results_df['f1'].std(ddof=0):.3f}, "
        f"Acc={results_df['accuracy'].mean():.3f}±{results_df['accuracy'].std(ddof=0):.3f}"
    )

    # Save results to CSV
    if not skip_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "ml_loo_cv_results.csv"

        fieldnames = [
            "run_date",
            "fold",
            "dataset",
            "type_group",
            "n_train",
            "n_test",
            "n_match",
            "n_no_match",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "seed",
            "quality_threshold",
        ]

        write_header = not results_file.exists()

        with open(results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            # Per-fold rows
            for row in all_results:
                fmt_row = {**row}
                fmt_row["accuracy"] = f"{row['accuracy']:.4f}"
                fmt_row["f1"] = f"{row['f1']:.4f}"
                fmt_row["precision"] = f"{row['precision']:.4f}"
                fmt_row["recall"] = f"{row['recall']:.4f}"
                writer.writerow(fmt_row)

            # Aggregate rows per type group
            for group in sorted(results_df["type_group"].unique()):
                group_df = results_df[results_df["type_group"] == group]
                writer.writerow(
                    {
                        "run_date": run_date.isoformat(),
                        "fold": -1,
                        "dataset": f"agg:{group}",
                        "type_group": group,
                        "n_train": "",
                        "n_test": int(group_df["n_test"].sum()),
                        "n_match": int(group_df["n_match"].sum()),
                        "n_no_match": int(group_df["n_no_match"].sum()),
                        "accuracy": f"{group_df['accuracy'].mean():.4f}",
                        "f1": f"{group_df['f1'].mean():.4f}",
                        "precision": f"{group_df['precision'].mean():.4f}",
                        "recall": f"{group_df['recall'].mean():.4f}",
                        "seed": seed,
                        "quality_threshold": quality_threshold,
                    }
                )

            # Overall aggregate
            writer.writerow(
                {
                    "run_date": run_date.isoformat(),
                    "fold": -1,
                    "dataset": "agg:overall",
                    "type_group": "",
                    "n_train": "",
                    "n_test": int(results_df["n_test"].sum()),
                    "n_match": int(results_df["n_match"].sum()),
                    "n_no_match": int(results_df["n_no_match"].sum()),
                    "accuracy": f"{results_df['accuracy'].mean():.4f}",
                    "f1": f"{results_df['f1'].mean():.4f}",
                    "precision": f"{results_df['precision'].mean():.4f}",
                    "recall": f"{results_df['recall'].mean():.4f}",
                    "seed": seed,
                    "quality_threshold": quality_threshold,
                }
            )

        console.print(f"\n[green]Results saved to {results_file}[/green]")

    console.print("\n[green]LOO-by-type cross-validation complete![/green]")


def register_commands(app: typer.Typer) -> None:
    """Register top-level commands on the given app."""

    @app.command()
    def stitch(
        dataset: str | None = typer.Argument(
            None,
            help="Dataset name (e.g. us_boston_streets)",
        ),
        all_datasets: bool = typer.Option(
            False,
            "--all",
            "-a",
            help="Process all available datasets",
        ),
        reference: Path | None = typer.Option(
            None,
            "--reference",
            "-r",
            help="Path to reference (Overture) parquet file. "
            "Overrides DatasetLoader lookup when provided with --target.",
        ),
        target: Path | None = typer.Option(
            None,
            "--target",
            "-t",
            help="Path to target (local) parquet file. "
            "Overrides DatasetLoader lookup when provided with --reference.",
        ),
        output: Path | None = typer.Option(
            None,
            "--output",
            "-o",
            help="Output bridge file path (default: data/output/{dataset}_bridge.parquet)",
        ),
        method: str = typer.Option(
            "xgboost",
            "--method",
            "-m",
            help="Matching method: xgboost",
        ),
        buffer_distance_m: float = typer.Option(
            50.0,
            "--buffer-m",
            "-b",
            help="Candidate search radius in meters",
        ),
        workers: int = typer.Option(
            -1,
            "--workers",
            "-w",
            help="Number of parallel workers (-1 for auto). Reduce for large datasets to save memory.",
        ),
        stitch_profile: str | None = typer.Option(
            None,
            "--stitch-profile",
            "-p",
            help="Stitch profile: recall (no filtering, high recall), "
            "balanced (default, bridge_min_confidence=0.5), "
            "precision (bridge_min_confidence=0.7). "
            "Overrides bridge_min_confidence setting.",
        ),
        profile: bool = typer.Option(
            False,
            "--profile",
            help="Enable per-feature timing breakdown (sets MATCHER_PROFILE=1)",
        ),
        allow_version_mismatch: bool = typer.Option(
            False,
            "--allow-version-mismatch",
            help="Load a model whose feature_version differs from the current "
            "FEATURE_VERSION (normally a hard error). Scores may be degraded.",
        ),
    ):
        """Run the stitch pipeline (pair matching + M:N optimization).

        Examples:
            crosswalk stitch us_boston_streets
            crosswalk stitch --all
            crosswalk stitch us_boston_streets -p recall
            crosswalk stitch -r ref.parquet -t target.parquet -o bridge.parquet
            crosswalk stitch us_boston_streets -r ref.parquet -t target.parquet
        """
        from ..config import STITCH_PROFILES, settings
        from ..datasets.loader import DatasetLoader
        from ..filenames import PROJECT_ROOT, bridge_filename
        from ..pipeline import run_pipeline

        if stitch_profile is not None:
            if stitch_profile not in STITCH_PROFILES:
                console.print(
                    f"[red]Unknown stitch profile '{stitch_profile}'. "
                    f"Available: {', '.join(STITCH_PROFILES)}[/red]"
                )
                raise typer.Exit(1)
            settings.bridge_min_confidence = STITCH_PROFILES[stitch_profile]

        if profile:
            os.environ["MATCHER_PROFILE"] = "1"

        output_dir = PROJECT_ROOT / "data" / "output"
        loader = DatasetLoader()

        # Build list of (dataset_name, ref_path, target_path, output_path)
        jobs: list[tuple[str, Path, Path, Path]] = []

        # Validate --reference and --target are provided together
        if (reference is not None) != (target is not None):
            console.print("[red]--reference and --target must be provided together[/red]")
            raise typer.Exit(1)

        if all_datasets:
            available = loader.list_available()
            if not available:
                console.print("[yellow]No datasets found in data/raw/[/yellow]")
                raise typer.Exit(0)
            console.print(f"[blue]Found {len(available)} datasets[/blue]")
            for ds in available:
                ref = loader.find_reference_path(ds)
                tgt = loader.find_target_path(ds)
                if ref and tgt:
                    out = output_dir / bridge_filename(ds)
                    jobs.append((ds, ref, tgt, out))
                else:
                    console.print(f"  [yellow]Skipping {ds}: missing files[/yellow]")
        elif reference is not None and target is not None:
            # Explicit file paths provided
            ds_name = dataset or reference.stem
            out = output or (output_dir / bridge_filename(ds_name))
            jobs.append((ds_name, reference, target, out))
        elif dataset:
            ref = loader.find_reference_path(dataset)
            tgt = loader.find_target_path(dataset)
            if not ref:
                console.print(
                    f"[red]Could not find reference (Overture) file for '{dataset}'[/red]"
                )
                raise typer.Exit(1)
            if not tgt:
                console.print(f"[red]Could not find target file for '{dataset}'[/red]")
                raise typer.Exit(1)
            out = output or (output_dir / bridge_filename(dataset))
            jobs.append((dataset, ref, tgt, out))
        else:
            console.print("[red]Provide a dataset name or --all or --reference/--target[/red]")
            raise typer.Exit(1)

        for ds_name, ref_path, tgt_path, out_path in jobs:
            console.print(f"\n[bold blue]Stitching {ds_name}...[/bold blue]")
            console.print(f"  Reference: {ref_path}")
            console.print(f"  Target: {tgt_path}")
            console.print(f"  Method: {method}")
            console.print(f"  Buffer: {buffer_distance_m}m")
            br_conf = settings.bridge_min_confidence
            if stitch_profile is not None:
                profile_label = stitch_profile
            else:
                # Reverse-lookup profile name from current bridge_min_confidence
                profile_label = next(
                    (name for name, val in STITCH_PROFILES.items() if val == br_conf),
                    None,
                )
            if profile_label is not None:
                console.print(f"  Profile: {profile_label} (bridge_min_confidence={br_conf})")
            else:
                console.print(f"  bridge_min_confidence={br_conf}")
            if workers != -1:
                console.print(f"  [yellow]Workers: {workers}[/yellow]")

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Plain status line instead of a rich spinner: rich's live display
            # runs a background refresher thread, and the parallel feature
            # phase only takes the fork/COW fast path from a single-threaded
            # process (features/pipeline.py::_should_use_fork) — the spinner
            # thread would silently force the slow pickle-per-worker path.
            # Pipeline logs already stream progress.
            console.print("Stitching...")

            result = run_pipeline(
                reference_path=ref_path,
                target_path=tgt_path,
                output_path=out_path,
                method=method,
                buffer_distance_m=buffer_distance_m,
                n_jobs=workers,
                allow_version_mismatch=allow_version_mismatch,
            )

            console.print(f"[green]Matched {result.n_matched} / {result.n_target} features[/green]")
            console.print(f"[green]Bridge file: {out_path}[/green]")

    @app.command("fetch-overture")
    def fetch_overture(
        bbox: str | None = typer.Option(
            None,
            "--bbox",
            help="Bounding box as 'xmin,ymin,xmax,ymax' in WGS84 (lon/lat) degrees",
        ),
        clip_target: Path | None = typer.Option(
            None,
            "--clip-target",
            help="Derive the bbox automatically from this target parquet's extent "
            "(alternative to --bbox; the zero-thought path)",
        ),
        output: Path = typer.Option(
            Path("overture_segments.parquet"),
            "--output",
            "-o",
            help="Output GeoParquet path for the Overture road segments",
        ),
        release: str | None = typer.Option(
            None,
            "--release",
            help="Overture Maps release to pin (e.g. 2026-06-18.0). "
            "Default: settings.overture_release, else the latest release. "
            "The release used is recorded in the .meta.yaml sidecar.",
        ),
        buffer_m: float | None = typer.Option(
            None,
            "--buffer-m",
            help="Expand the bbox by this many meters to capture edge topology "
            "(default: 1000; pass 0 to disable)",
        ),
        connectors: bool = typer.Option(
            False,
            "--connectors",
            help="Also fetch Overture connectors to a sibling *_connectors.parquet. "
            "Not needed for 'crosswalk stitch' (topology comes from the segments' "
            "connectors column).",
        ),
    ):
        """Fetch Overture road segments for a bbox — no dataset YAML needed.

        The YAML-free path to the reference half of a match: supply either an
        explicit --bbox or --clip-target (your local parquet, whose extent is
        used), and stitch the result directly.

        Examples:
            crosswalk fetch-overture --bbox -71.06,42.35,-71.03,42.37 -o ref.parquet
            crosswalk fetch-overture --clip-target my_roads.parquet -o ref.parquet
            crosswalk stitch -r ref.parquet -t my_roads.parquet -o bridge.parquet
        """
        from ..config import settings
        from ..fetch.overture import (
            DEFAULT_OVERTURE_BUFFER_M,
            BoundingBox,
            fetch_overture_connectors,
            fetch_overture_segments,
            get_buffered_bbox,
        )

        if (bbox is None) == (clip_target is None):
            console.print("[red]Provide exactly one of --bbox or --clip-target[/red]")
            raise typer.Exit(1)

        if bbox is not None:
            try:
                parts = [float(p) for p in bbox.split(",")]
            except ValueError:
                parts = []
            if len(parts) != 4:
                console.print(
                    "[red]--bbox must be 'xmin,ymin,xmax,ymax' (4 comma-separated numbers)[/red]"
                )
                raise typer.Exit(1)
            xmin, ymin, xmax, ymax = parts
        else:
            import geopandas as gpd

            if not clip_target.exists():
                console.print(f"[red]Target parquet not found: {clip_target}[/red]")
                raise typer.Exit(1)
            console.print(f"[blue]Deriving bbox from {clip_target}...[/blue]")
            target_gdf = gpd.read_parquet(clip_target)
            if len(target_gdf) == 0:
                console.print(f"[red]Target parquet is empty: {clip_target}[/red]")
                raise typer.Exit(1)
            if target_gdf.crs is None:
                console.print(
                    "[yellow]Target parquet has no CRS; assuming EPSG:4326 (WGS84). "
                    "If the data is in a projected CRS the derived bbox will be "
                    "wrong — set a CRS on the parquet.[/yellow]"
                )
            elif target_gdf.crs.to_epsg() != 4326:
                target_gdf = target_gdf.to_crs("EPSG:4326")
            xmin, ymin, xmax, ymax = (float(v) for v in target_gdf.total_bounds)

        valid_bbox = (
            xmin < xmax
            and ymin < ymax
            and xmin >= -180
            and xmax <= 180
            and ymin >= -90
            and ymax <= 90
        )
        if not valid_bbox:
            console.print(
                f"[red]Invalid WGS84 bbox: {xmin},{ymin},{xmax},{ymax} "
                "(expected xmin<xmax, ymin<ymax, lon in [-180,180], lat in [-90,90])[/red]"
            )
            raise typer.Exit(1)

        original_bbox = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        fetch_bbox, effective_buffer = get_buffered_bbox(
            original_bbox, buffer_m, DEFAULT_OVERTURE_BUFFER_M
        )
        effective_release = release or settings.overture_release

        console.print(
            f"[blue]Fetching Overture segments for bbox "
            f"({xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f})"
            + (f" +{effective_buffer:.0f}m buffer" if effective_buffer else "")
            + (f", release {effective_release}" if effective_release else ", latest release")
            + "...[/blue]"
        )

        seg_path = fetch_overture_segments(
            bbox=fetch_bbox,
            output_path=output,
            release=effective_release,
            original_bbox=original_bbox,
            buffer_m=effective_buffer,
        )
        console.print(f"[green]Saved Overture segments to {seg_path}[/green]")

        if connectors:
            conn_path = output.with_name(output.stem + "_connectors.parquet")
            conn_path = fetch_overture_connectors(
                bbox=fetch_bbox,
                output_path=conn_path,
                release=effective_release,
                original_bbox=original_bbox,
                buffer_m=effective_buffer,
            )
            console.print(f"[green]Saved Overture connectors to {conn_path}[/green]")

        console.print(
            "\nNext: [bold]crosswalk stitch -r "
            f"{output} -t <your_local.parquet> -o bridge.parquet[/bold]"
        )

    @app.command()
    def train(
        labels_dir: Path = typer.Option(
            Path("labels"),
            "--labels",
            "-l",
            help="Labels directory (Hive-partitioned CSV format)",
        ),
        output: Path = typer.Option(
            Path("data/models/matcher_model_combined.joblib"),
            "--output",
            "-o",
            help="Output path for trained model",
        ),
        exclude_semantic: bool = typer.Option(
            False,
            "--exclude-semantic",
            help="Exclude semantic features (name_*, class_similarity) for geometry-only model",
        ),
        exclude_dataset: list[str] = typer.Option(
            [],
            "--exclude-dataset",
            "-x",
            help="Dataset(s) to exclude from training (for leave-one-out evaluation). Can be repeated.",
        ),
        exclude_features: list[str] = typer.Option(
            [],
            "--exclude-features",
            "-e",
            help="Feature(s) to exclude from training (for feature importance analysis). Can be repeated.",
        ),
        agent_weight: float = typer.Option(
            0.0,
            "--agent-weight",
            help="Weight for agent labels (0.0=ignore, 1.0=equal to human). Enables weak supervision.",
        ),
        min_agent_confidence: float = typer.Option(
            0.0,
            "--min-agent-confidence",
            help="Minimum confidence for including agent labels (0.0-1.0).",
        ),
    ):
        """Train an ML model on labeled data.

        Loads labels from normalized format:
        - labels/human/dataset=*/data.csv (metadata)
        - labels/features/dataset=*/data.parquet (computed features)

        Examples:
            crosswalk train
            crosswalk train --labels labels -o data/models/my_model.joblib

            # Train geometry-only model (no name/class features)
            crosswalk train --exclude-semantic -o data/models/matcher_model_geom_only.joblib

            # Leave-one-out: train without Frisco labels to test generalization
            crosswalk train -x us_frisco_trails -o data/models/no_frisco.joblib

            # Train with weak supervision from agent labels
            crosswalk train --agent-weight 0.5 --min-agent-confidence 0.7
        """
        from ..labeling.label_store import LabelStore
        from ..matching.ml import MLMatcher

        if not labels_dir.exists():
            console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
            raise typer.Exit(1)

        # Check for normalized format (human labels + features)
        human_dir = labels_dir / "human"
        features_dir = labels_dir / "features"
        if not human_dir.exists() or not features_dir.exists():
            console.print(f"[red]Normalized label format not found in {labels_dir}[/red]")
            console.print(
                "[yellow]Expected normalized label format (labels/human/ + labels/features/).[/yellow]"
            )
            raise typer.Exit(1)

        console.print(f"[blue]Loading labels from {labels_dir}...[/blue]")
        df = LabelStore.load_all(labels_dir)
        console.print(f"  Found {len(df)} labels from {df['dataset'].nunique()} datasets")

        if exclude_dataset:
            console.print(f"[yellow]Excluding datasets: {', '.join(exclude_dataset)}[/yellow]")

        if exclude_features:
            console.print(f"[yellow]Excluding features: {', '.join(exclude_features)}[/yellow]")

        if agent_weight > 0:
            console.print(
                f"[yellow]Including agent labels with weight={agent_weight}, "
                f"min_confidence={min_agent_confidence}[/yellow]"
            )

        # Train model
        model_type = "geometry-only" if exclude_semantic else "full"
        console.print(f"[blue]Training {model_type} model...[/blue]")
        matcher = MLMatcher()
        metrics = matcher.train(
            labels_dir=labels_dir,
            test_size=0.2,
            binary=True,
            exclude_semantic=exclude_semantic,
            exclude_datasets=list(exclude_dataset) if exclude_dataset else None,
            exclude_features=list(exclude_features) if exclude_features else None,
            agent_weight=agent_weight,
            min_agent_confidence=min_agent_confidence,
        )

        # Save model
        output.parent.mkdir(parents=True, exist_ok=True)
        matcher.save_model(str(output))

        console.print(f"\n[green]Model saved to {output}[/green]")
        console.print(f"[green]Holdout accuracy: {metrics['test_accuracy']:.1%}[/green]")

    @app.command("export-spark-model")
    def export_spark_model(
        labels_dir: Path = typer.Option(
            Path("labels"),
            "--labels",
            "-l",
            help="Labels directory",
        ),
        output_dir: Path = typer.Option(
            Path("data/models/export"),
            "--output",
            "-o",
            help="Output directory for model.json + manifest.json",
        ),
    ):
        """Train and export a Spark-portable XGBoost model for Overture matching.

        Trains on the 28-feature subset computable from aligned geometry pairs
        (no topology, graph, or spatial-index features required). Exports as
        XGBoost-native JSON loadable by the Spark MatchLayerToNetworkV2 job.

        Produces:
        - model.json: XGBoost native model
        - manifest.json: Feature list, hyperparameters, and metadata

        Examples:
            crosswalk export-spark-model
            crosswalk export-spark-model --labels labels -o data/models/export/
        """
        import json
        import os

        import numpy as np
        import xgboost as xgb

        from ..config import (
            FEATURE_COLUMNS,
            SPARK_PORTABLE_FEATURES,
            SPARK_PORTABLE_XGB_PARAMS,
        )
        from ..matching.ml import MLMatcher

        if not labels_dir.exists():
            console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
            raise typer.Exit(1)

        # Determine features to exclude (everything not in SPARK_PORTABLE_FEATURES)
        exclude_features = [f for f in FEATURE_COLUMNS if f not in SPARK_PORTABLE_FEATURES]

        console.print(
            f"[blue]Training Spark-portable model ({len(SPARK_PORTABLE_FEATURES)} features)...[/blue]"
        )
        console.print(
            f"[dim]Excluding {len(exclude_features)} features requiring topology/graph/spatial-index[/dim]"
        )

        # Train with Spark-portable hyperparams (tuned for 28-feature subset)
        matcher = MLMatcher()
        metrics = matcher.train(
            labels_dir=labels_dir,
            test_size=0.2,
            binary=True,
            exclude_features=exclude_features,
            **SPARK_PORTABLE_XGB_PARAMS,
        )

        # Export
        output_dir.mkdir(parents=True, exist_ok=True)

        model_out = output_dir / "model.json"
        matcher.model.get_booster().save_model(str(model_out))
        model_size_kb = os.path.getsize(model_out) / 1024

        # Extract all JSON-serializable hyperparams for reproducibility
        xgb_params = matcher.model.get_params()
        hyperparams = {}
        for k, v in xgb_params.items():
            if v is None or callable(v):
                continue
            # Skip NaN (not valid JSON)
            if isinstance(v, float) and np.isnan(v):
                continue
            try:
                json.dumps(v)
                hyperparams[k] = v
            except (TypeError, ValueError):
                hyperparams[k] = str(v)

        manifest = {
            "features": matcher.feature_names,
            "n_features": len(matcher.feature_names),
            "n_estimators": xgb_params.get("n_estimators"),
            "threshold": 0.5,
            "is_binary": matcher.is_binary,
            "feature_version": matcher.feature_version,
            "label_encoder": matcher.label_encoder,
            "hyperparams": hyperparams,
        }
        # Isotonic calibration knots (piecewise-linear P(match) remap). Portable
        # by construction: the Spark scorer can apply it as interp(score, xs, ys)
        # with endpoint clipping. NOTE: emitting the table makes the artifact
        # calibration-ready; wiring the Spark MatchLayerToNetworkV2 job to
        # consume it (and re-fit its thresholds on calibrated scores) is a
        # tf-data-platform follow-up — see docs/EVAL_ROADMAP.md.
        if matcher.calibrator is not None:
            manifest["calibration"] = matcher.calibrator.to_knots()
            manifest["calibration"]["applied"] = False  # consumer wiring is follow-up
        manifest_out = output_dir / "manifest.json"
        with open(manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Verify the exported model loads and predicts
        console.print("[blue]Verifying exported model...[/blue]")
        booster = xgb.Booster()
        booster.load_model(str(model_out))
        test_data = xgb.DMatrix(
            np.zeros((1, len(matcher.feature_names)), dtype=np.float32),
            feature_names=matcher.feature_names,
        )
        pred = booster.predict(test_data)
        if len(pred) != 1:
            console.print(
                "[red]Model verification failed: prediction returned unexpected shape[/red]"
            )
            raise typer.Exit(1)

        console.print(f"\n[green]Exported to {output_dir}/[/green]")
        console.print(f"  model.json: {model_size_kb:.0f} KB")
        console.print(f"  manifest.json: {len(matcher.feature_names)} features")
        console.print(f"  CV F1: {metrics['cv_f1_mean']:.4f} ± {metrics.get('cv_f1_std', 0):.4f}")
        console.print(f"  Holdout accuracy: {metrics['test_accuracy']:.1%}")

    @app.command("eval")
    def eval_model(
        model: Path = typer.Option(
            None,
            "--model",
            "-m",
            help="Path to existing trained model (if not provided, uses cross-validation)",
        ),
        labels_dir: Path = typer.Option(
            Path("labels"),
            "--labels",
            "-l",
            help="Labels directory (Hive-partitioned CSV format)",
        ),
        output_dir: Path = typer.Option(
            Path("benchmarks"),
            "--output",
            "-o",
            help="Output directory for benchmark results",
        ),
        cv_folds: int = typer.Option(
            5,
            "--cv-folds",
            "-k",
            help="Number of cross-validation folds (default: 5)",
        ),
        seed: int = typer.Option(
            42,
            "--seed",
            "-s",
            help="Random seed for CV splits",
        ),
        skip_save: bool = typer.Option(
            False,
            "--skip-save",
            help="Skip saving results to CSV (just print)",
        ),
        by_dataset: bool = typer.Option(
            True,
            "--by-dataset/--overall",
            help="Show metrics broken down by dataset",
        ),
        dataset: list[str] = typer.Option(
            [],
            "--dataset",
            "-d",
            help="Only evaluate on specific dataset(s). Can be repeated.",
        ),
        loo: bool = typer.Option(
            False,
            "--loo",
            help="Run leave-one-out by type cross-validation (mutually exclusive with --model)",
        ),
        quality_threshold: float = typer.Option(
            DEFAULT_QUALITY_THRESHOLD,
            "--quality-threshold",
            help="Quality threshold for road_good/road_poor split (only used with --loo)",
        ),
    ):
        """Evaluate ML model performance using cross-validation.

        By default, runs k-fold cross-validation with segment-aware splitting to
        prevent data leakage. Reports mean ± std for all metrics.

        Use --model to evaluate an existing trained model on a holdout set instead.
        Use --loo for leave-one-out by type CV that tests cross-dataset generalization.

        Examples:
            # Cross-validation evaluation (default)
            crosswalk eval
            crosswalk eval --cv-folds 10
            crosswalk eval --seed 123 --skip-save

            # Evaluate an existing model (single holdout)
            crosswalk eval --model data/models/matcher_model.joblib
            crosswalk eval -m data/models/combined.joblib -d us_frisco_trails

            # Leave-one-out by type cross-validation
            crosswalk eval --loo --cv-folds 5 --seed 42
            crosswalk eval --loo --quality-threshold 0.3
        """
        if loo and model is not None:
            console.print("[red]--loo and --model are mutually exclusive[/red]")
            raise typer.Exit(1)

        if loo and dataset:
            console.print("[red]--loo and --dataset are mutually exclusive[/red]")
            raise typer.Exit(1)

        if loo:
            _loo_type_cross_validate(
                labels_dir=labels_dir,
                output_dir=output_dir,
                cv_folds=cv_folds,
                seed=seed,
                skip_save=skip_save,
                quality_threshold=quality_threshold,
            )
        elif model is not None:
            # Evaluate existing model on holdout
            _eval_existing_model(
                model=model,
                labels_dir=labels_dir,
                by_dataset=by_dataset,
                dataset=dataset,
                seed=seed,
                output_dir=output_dir,
                skip_save=skip_save,
            )
        else:
            # Cross-validation evaluation
            _cross_validate(
                labels_dir=labels_dir,
                output_dir=output_dir,
                cv_folds=cv_folds,
                seed=seed,
                skip_save=skip_save,
                by_dataset=by_dataset,
                filter_datasets=list(dataset) if dataset else None,
            )

    @app.command()
    def backfill(
        labels_dir: Path = typer.Option(
            Path("labels"),
            "--labels",
            "-l",
            help="Directory containing labels",
        ),
        data_dir: Path = typer.Option(
            Path("data/raw"),
            "--data-dir",
            "-d",
            help="Directory containing source data files",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Show what would be done without making changes",
        ),
        skip_missing: bool = typer.Option(
            True,
            "--skip-missing/--fail-missing",
            help="Skip pairs with missing source data",
        ),
        include_agent: bool = typer.Option(
            False,
            "--include-agent",
            help="Also backfill agent labels (by default only human labels are processed)",
        ),
        agent_only: bool = typer.Option(
            False,
            "--agent-only",
            help="Only backfill agent labels (skip human)",
        ),
        missing_only: bool = typer.Option(
            False,
            "--missing-only",
            help="Only compute features for labels without any features (skip existing)",
        ),
        require_stored_data: bool = typer.Option(
            False,
            "--require-stored-data",
            help="Reject pairs without stored geometries (no fallback to raw data)",
        ),
        only_datasets: list[str] = typer.Option(
            None,
            "--dataset",
            "-D",
            help="Only backfill these datasets (repeatable). Useful when some "
            "datasets need a slow/unavailable Overture auto-fetch.",
        ),
    ):
        """Recompute features for human labels using current feature computation code.

        By default, this recomputes ALL features for all human labels to ensure
        consistency when feature computation logic changes or new features are added.

        Use --include-agent to also process agent labels.
        Use --missing-only to only compute features for labels that don't have any.

        Use this after:
        - Adding new features to the ML pipeline
        - Changing feature computation logic
        - Adding new labels via the UI (with --missing-only)

        Examples:
            crosswalk backfill --dry-run       # Preview what would be recomputed
            crosswalk backfill                 # Recompute all human label features
            crosswalk backfill --include-agent # Also include agent labels
            crosswalk backfill --missing-only  # Only compute for labels without features
            crosswalk backfill -D us_boston_streets -D us_seattle_sidewalks
        """

        from ..labeling.feature_store import FeatureStore
        from ..labeling.label_store import LabelStore

        labels_dir = Path(labels_dir)
        human_dir = labels_dir / "human"
        agent_dir = labels_dir / "agent"
        features_dir = labels_dir / "features"

        # Determine what to process based on flags
        process_human = not agent_only
        process_agent = include_agent or agent_only

        # Collect all labels to process
        all_label_keys = set()
        label_sources = {}  # Track which source each key came from

        if process_human:
            if human_dir.exists():
                console.print("[blue]Loading human labels...[/blue]")
                human_labels = LabelStore.load_human_labels(human_dir)
                if len(human_labels) > 0:
                    human_keys = set(
                        zip(
                            human_labels["gers_id"],
                            human_labels["target_id"],
                            human_labels["dataset"],
                        )
                    )
                    console.print(
                        f"  Found {len(human_labels)} human labels across {human_labels['dataset'].nunique()} datasets"
                    )
                    all_label_keys.update(human_keys)
                    for k in human_keys:
                        label_sources[k] = "human"
                else:
                    console.print("  [yellow]No human labels found[/yellow]")
            else:
                console.print(f"  [yellow]Human labels directory not found: {human_dir}[/yellow]")

        if process_agent:
            if agent_dir.exists():
                console.print("[blue]Loading agent labels...[/blue]")
                agent_labels = LabelStore.load_agent_labels(agent_dir)
                if len(agent_labels) > 0:
                    agent_keys = set(
                        zip(
                            agent_labels["gers_id"],
                            agent_labels["target_id"],
                            agent_labels["dataset"],
                        )
                    )
                    console.print(
                        f"  Found {len(agent_labels)} agent labels across {agent_labels['dataset'].nunique()} datasets"
                    )
                    all_label_keys.update(agent_keys)
                    for k in agent_keys:
                        if k not in label_sources:  # Don't overwrite human
                            label_sources[k] = "agent"
                else:
                    console.print("  [yellow]No agent labels found[/yellow]")
            else:
                console.print(f"  [yellow]Agent labels directory not found: {agent_dir}[/yellow]")

        if len(all_label_keys) == 0:
            console.print("[yellow]No labels found to process.[/yellow]")
            raise typer.Exit(0)

        if only_datasets:
            wanted = set(only_datasets)
            available = {d for _, _, d in all_label_keys}
            unknown = wanted - available
            if unknown:
                console.print(
                    f"  [yellow]No labels found for: {', '.join(sorted(unknown))}[/yellow]"
                )
            all_label_keys = {k for k in all_label_keys if k[2] in wanted}
            console.print(
                f"  Filtered to {len(all_label_keys)} labels in {len(wanted & available)} datasets"
            )

        # Determine which labels to process
        if missing_only:
            # Only compute for labels without any features
            console.print("[blue]Loading existing features...[/blue]")
            existing_features = FeatureStore.load_all(features_dir)

            if len(existing_features) > 0:
                existing_keys = set(
                    zip(
                        existing_features["gers_id"],
                        existing_features["target_id"],
                        existing_features["dataset"],
                    )
                )
                console.print(f"  Found {len(existing_keys)} existing feature records")
            else:
                existing_keys = set()
                console.print("  No existing features found")

            keys_to_process = all_label_keys - existing_keys
            action_verb = "missing"
        else:
            # Recompute all features (default)
            keys_to_process = all_label_keys
            action_verb = "total"

        # Count by source
        human_count = sum(1 for k in keys_to_process if label_sources.get(k) == "human")
        agent_count = sum(1 for k in keys_to_process if label_sources.get(k) == "agent")
        console.print(
            f"  {len(keys_to_process)} {action_verb} labels to process ({human_count} human, {agent_count} agent)"
        )

        if len(keys_to_process) == 0:
            if missing_only:
                console.print("[green]All labels already have features.[/green]")
            else:
                console.print("[yellow]No labels to process.[/yellow]")
            raise typer.Exit(0)

        if dry_run:
            console.print("\n[yellow][DRY RUN] Would compute features for:[/yellow]")
            # Group by dataset for summary
            by_dataset_summary = {}
            for _gers_id, _target_id, ds in keys_to_process:
                by_dataset_summary[ds] = by_dataset_summary.get(ds, 0) + 1

            for ds, count in sorted(by_dataset_summary.items()):
                console.print(f"  {ds}: {count} pairs")
            console.print("\n[yellow]Run without --dry-run to compute features.[/yellow]")
            raise typer.Exit(0)

        # Import heavy dependencies only when needed
        import geopandas as gpd
        import pandas as pd

        from ..blocking.spatial_index import CandidatePair
        from ..features.compute import precompute_graphlet_features
        from ..features.pipeline import prepare_worker_data
        from ..filenames import find_overture_segments, find_target_file
        from ..matching.ml import _compute_feature_chunk, _init_worker
        from ..utils.geometry import filter_to_linestrings

        # Process by dataset - get unique datasets from keys to process
        datasets = sorted(set(d for _, _, d in keys_to_process))
        total_computed = 0
        total_skipped = 0
        total_errored = 0

        for ds in datasets:
            dataset_keys = [(g, t) for g, t, d in keys_to_process if d == ds]
            if not dataset_keys:
                continue

            console.print(f"\n[blue]Processing {ds} ({len(dataset_keys)} pairs)...[/blue]")

            # Load source data
            overture_path = find_overture_segments(data_dir, ds)
            target_path = find_target_file(data_dir, ds)

            if overture_path is None:
                # Auto-fetch Overture data using dataset config bbox
                overture_path = _auto_fetch_overture(ds, data_dir)
                if overture_path is None:
                    console.print(
                        f"  [red]Error: Overture data not found and auto-fetch failed "
                        f"(no dataset config for '{ds}')[/red]"
                    )
                    raise typer.Exit(1)

            if target_path is None and not skip_missing:
                console.print("  [red]Error: Target data not found[/red]")
                raise typer.Exit(1)

            # Load and prepare data
            console.print(f"  Loading Overture from {overture_path.name}...")
            ref_gdf = gpd.read_parquet(overture_path)
            ref_gdf = filter_to_linestrings(ref_gdf, source_name="reference")
            ref_gdf["id"] = ref_gdf["id"].astype(str)
            ref_lookup = ref_gdf.set_index("id")

            # Project reference to UTM
            if ref_gdf.crs is not None and ref_gdf.crs.is_geographic:
                utm_crs = ref_gdf.estimate_utm_crs()
                ref_gdf_proj = ref_gdf.to_crs(utm_crs)
            else:
                utm_crs = ref_gdf.crs
                ref_gdf_proj = ref_gdf

            # Load target raw data if available (not required when stored geometries exist)
            if target_path is not None:
                console.print(f"  Loading target from {target_path.name}...")
                target_gdf = gpd.read_parquet(target_path)
                target_gdf = filter_to_linestrings(target_gdf, source_name="target")
                target_gdf["id"] = target_gdf["id"].astype(str)
                target_lookup = target_gdf.set_index("id")
                target_gdf_proj = (
                    target_gdf.to_crs(utm_crs) if utm_crs != target_gdf.crs else target_gdf
                )
            else:
                console.print(
                    "  [yellow]Target data not found - using stored geometries only[/yellow]"
                )
                target_gdf = None
                target_gdf_proj = None
                target_lookup = None

            # Initialize feature store and data store for this dataset
            feature_store = FeatureStore(ds, features_dir=features_dir)

            # Load stored pair data (geometries captured at labeling time)
            # This is critical because target IDs are not stable across data refreshes
            from ..labeling.data_store import DataStore

            data_store = DataStore(ds, data_dir=labels_dir / "data")
            has_stored_data = len(data_store.gdf) > 0
            if has_stored_data:
                console.print(
                    f"  Using stored geometries from labels/data ({len(data_store.gdf)} pairs)"
                )
            else:
                console.print("  [yellow]No stored geometries - using raw data lookup[/yellow]")

            computed = 0
            skipped = 0
            errored = 0
            used_stored = 0
            used_lookup = 0
            no_stored_rejected = 0

            # --- Phase 1: Resolve geometries for all labeled pairs ---
            # Collect stored target geometries for building the augmented target GDF
            resolved_pairs = []  # list of (gers_id, target_id, pair_data)
            stored_target_overrides = {}  # target_id -> WGS84 geometry
            stored_target_attrs = {}  # target_id -> attribute dict (for segments not in raw data)
            stored_ref_overrides = {}  # gers_id -> WGS84 geometry (refs gone from current release)
            stored_ref_attrs = {}  # gers_id -> attribute dict for those refs

            for gers_id, target_id in dataset_keys:
                pair_data = None
                used_stored_for_pair = False

                if has_stored_data:
                    pair_data = data_store.get_pair(gers_id, target_id)
                    if pair_data is not None:
                        stored_ref = pair_data.get("ref_geometry")
                        stored_target = pair_data.get("target_geometry")
                        if stored_ref is not None and stored_target is not None:
                            used_stored_for_pair = True

                if not used_stored_for_pair:
                    if require_stored_data:
                        no_stored_rejected += 1
                        skipped += 1
                        continue
                    if (
                        gers_id not in ref_lookup.index
                        or target_lookup is None
                        or target_id not in target_lookup.index
                    ):
                        skipped += 1
                        continue
                    used_lookup += 1
                    logger.debug(
                        "Backfill fallback lookup: gers_id=%s target_id=%s", gers_id, target_id
                    )
                else:
                    used_stored += 1
                    stored_target_overrides[target_id] = pair_data["target_geometry"]
                    if target_lookup is None or target_id not in target_lookup.index:
                        target_names = pair_data.get("target_names")
                        stored_target_attrs[target_id] = {
                            "names": target_names,
                            "names_lr": pair_data.get("target_names_lr"),
                            "class": pair_data.get("target_class"),
                            "subclass": pair_data.get("target_subclass"),
                        }
                    # GERS ids churn across Overture releases: a labeled gers_id may
                    # no longer exist in the current release. Fall back to the stored
                    # reference geometry so the pair stays resolvable (mirrors the
                    # target-side augmentation below).
                    if str(gers_id) not in ref_lookup.index:
                        stored_ref_overrides[gers_id] = pair_data["ref_geometry"]
                        stored_ref_attrs[gers_id] = {
                            "names": pair_data.get("ref_names"),
                            "names_lr": pair_data.get("ref_names_lr"),
                            "class": pair_data.get("ref_class"),
                            "subclass": pair_data.get("ref_subclass"),
                        }

                resolved_pairs.append((gers_id, target_id, pair_data))

            if not resolved_pairs:
                reason = "IDs not in data"
                if no_stored_rejected > 0:
                    reason += f", {no_stored_rejected} rejected (no stored data)"
                console.print(f"  [yellow]Skipped all {skipped} pairs ({reason})[/yellow]")
                total_skipped += skipped
                continue

            # --- Phase 2: Build augmented target GeoDataFrame ---
            # Start with full raw target (for sibling contexts, spatial index, topology),
            # then override/append stored geometries for labeled pairs
            if target_gdf_proj is not None:
                augmented_target = target_gdf_proj.copy()
            else:
                augmented_target = gpd.GeoDataFrame(
                    {"id": pd.Series(dtype=str)},
                    geometry=gpd.GeoSeries([], crs=utm_crs),
                )

            target_id_set = (
                set(augmented_target["id"].astype(str)) if len(augmented_target) > 0 else set()
            )

            # Override geometry for stored-data targets already in raw data
            override_ids = [tid for tid in stored_target_overrides if tid in target_id_set]
            if override_ids:
                override_geoms = gpd.GeoSeries(
                    [stored_target_overrides[tid] for tid in override_ids],
                    crs="EPSG:4326",
                ).to_crs(utm_crs)
                for tid, geom in zip(override_ids, override_geoms):
                    mask = augmented_target["id"].astype(str) == tid
                    augmented_target.loc[mask, "geometry"] = geom

            # Append new rows for stored-data targets not in raw data
            append_ids = [tid for tid in stored_target_attrs if tid not in target_id_set]
            if append_ids:
                append_geoms = gpd.GeoSeries(
                    [stored_target_overrides[tid] for tid in append_ids],
                    crs="EPSG:4326",
                ).to_crs(utm_crs)
                append_rows = []
                for tid, geom in zip(append_ids, append_geoms):
                    row = {"id": tid, "geometry": geom}
                    row.update(stored_target_attrs[tid])
                    append_rows.append(row)
                new_gdf = gpd.GeoDataFrame(append_rows, geometry="geometry", crs=utm_crs)
                augmented_target = pd.concat([augmented_target, new_gdf], ignore_index=True)

            # --- Phase 2b: Augment reference GDF with stored geometries for
            # gers_ids that vanished from the current Overture release ---
            # Append-only: refs still present in the release keep their live
            # geometry/attributes; only churned ids fall back to stored data.
            if stored_ref_overrides:
                ref_append_geoms = gpd.GeoSeries(
                    list(stored_ref_overrides.values()),
                    crs="EPSG:4326",
                ).to_crs(utm_crs)
                ref_append_rows = []
                for gid, geom in zip(stored_ref_overrides, ref_append_geoms):
                    row = {"id": str(gid), "geometry": geom}
                    row.update(stored_ref_attrs[gid])
                    if "connectors" in ref_gdf_proj.columns:
                        # Explicit None (not concat-NaN): downstream connector
                        # iteration handles None but not float NaN.
                        row["connectors"] = None
                    ref_append_rows.append(row)
                ref_new_gdf = gpd.GeoDataFrame(ref_append_rows, geometry="geometry", crs=utm_crs)
                ref_gdf_proj = pd.concat([ref_gdf_proj, ref_new_gdf], ignore_index=True)
                console.print(
                    f"  [yellow]{len(ref_append_rows)} gers_id(s) missing from current "
                    f"Overture release - using stored reference geometries[/yellow]"
                )

            # --- Phase 3: Create CandidatePair objects ---
            ref_id_to_idx = {str(rid): idx for idx, rid in enumerate(ref_gdf_proj["id"])}
            target_id_to_idx = {str(tid): idx for idx, tid in enumerate(augmented_target["id"])}

            candidates = []
            candidate_metadata = []  # parallel list tracking (gers_id, target_id, pair_data)

            for gers_id, target_id, pair_data in resolved_pairs:
                ref_idx = ref_id_to_idx.get(str(gers_id))
                target_idx = target_id_to_idx.get(str(target_id))

                if ref_idx is None or target_idx is None:
                    skipped += 1
                    continue

                ref_geom = ref_gdf_proj.geometry.iloc[ref_idx]
                target_geom = augmented_target.geometry.iloc[target_idx]
                if (
                    ref_geom is None
                    or ref_geom.is_empty
                    or target_geom is None
                    or target_geom.is_empty
                ):
                    skipped += 1
                    continue

                # Blocking stats are unused by the pipeline; placeholders required by dataclass
                candidates.append(
                    CandidatePair(
                        ref_id=gers_id,
                        ref_idx=ref_idx,
                        target_id=target_id,
                        target_idx=target_idx,
                        distance_estimate=0.0,
                        heading_diff=0.0,
                    )
                )
                candidate_metadata.append((gers_id, target_id, pair_data))

            if not candidates:
                reason = "no valid geometries"
                if no_stored_rejected > 0:
                    reason += f", {no_stored_rejected} rejected (no stored data)"
                console.print(f"  [yellow]Skipped all pairs ({reason})[/yellow]")
                total_skipped += skipped
                continue

            # --- Phase 4: Prepare worker data through shared pipeline ---
            # One call replaces manual spatial index, graphlet, sibling context,
            # topology, alignment, and endpoint feature computation
            console.print("  Running shared feature pipeline...")
            pipeline_result = prepare_worker_data(
                candidates=candidates,
                reference=ref_gdf_proj,
                target=augmented_target,
                n_jobs=1,  # small dataset, no need for parallel alignment
                filter_physical_overlap=False,  # backfill: keep all labeled pairs
            )
            worker_data = pipeline_result.worker_data
            candidates = pipeline_result.candidates

            # --- Phase 4b: Ensure ref graphlet data uses full-network computation ---
            # The shared pipeline now builds graphlet graphs on the FULL ref/target
            # networks (not candidate-only subsets), so worker_data["ref_graphlet_data"]
            # is already full-network. We recompute it here explicitly so that Phase 4c
            # can rebuild the Overture connector anchoring using the FULL ref geometry
            # set (backfill has few candidate pairs, so anchoring to candidate-only ref
            # connectors would be impoverished).
            ref_has_connectors = "connectors" in ref_gdf_proj.columns
            worker_data["ref_graphlet_data"] = precompute_graphlet_features(
                ref_gdf_proj,
                connectors_column="connectors" if ref_has_connectors else None,
            )

            # --- Phase 4c: Rebuild derived indices from overridden graphlet data ---
            # The node IDs in the full-network graphlet differ from the candidate-only
            # graphlet that prepare_worker_data() used to build
            # target_overture_connectors. Rebuild it.
            from crosswalk.features.pipeline import rebuild_connector_indices

            unique_target_idxs = {c.target_idx for c in candidates}
            target_ids_arr = worker_data["target_ids"]
            geoms_by_ref_id = {
                str(ref_gdf_proj["id"].iloc[i]): ref_gdf_proj.geometry.iloc[i]
                for i in range(len(ref_gdf_proj))
                if ref_gdf_proj.geometry.iloc[i] is not None
                and not ref_gdf_proj.geometry.iloc[i].is_empty
            }
            geoms_by_target_id = {
                str(target_ids_arr[idx]): augmented_target.geometry.iloc[idx]
                for idx in unique_target_idxs
            }
            rebuild_connector_indices(worker_data, geoms_by_ref_id, geoms_by_target_id)

            # --- Phase 5: Override topology with stored values ---
            # 3-tier fallback: stored topology > computed by pipeline > NaN defaults
            # Also override synthetic connectors from stored sampled topology
            from crosswalk.labeling.data_store import reconstruct_topo_connectors_from_sampled

            for cand, (_gid, _tid, pair_data) in zip(candidates, candidate_metadata):
                if pair_data is not None:
                    stored_ref_topo = pair_data.get("ref_topology")
                    stored_target_topo = pair_data.get("target_topology")
                    if stored_ref_topo:
                        worker_data["ref_topology_full"][cand.ref_idx] = stored_ref_topo
                    if stored_target_topo:
                        worker_data["target_topology_full"][cand.target_idx] = stored_target_topo

                    # Override synthetic connectors from stored sampled topology
                    stored_sampled = pair_data.get("target_topo_sampled")
                    if stored_sampled:
                        seg_id = str(worker_data["target_ids"][cand.target_idx])
                        connectors, node_feats = reconstruct_topo_connectors_from_sampled(
                            stored_sampled
                        )
                        worker_data["target_topo_connectors"][seg_id] = connectors
                        worker_data["target_topo_node_features"].update(node_feats)

            # --- Phase 6: Compute features through shared code path ---
            # Uses the exact same _compute_feature_chunk() that inference uses,
            # including batch geometric computation and LR attribute extraction
            console.print("  Computing features...")
            _init_worker(worker_data)
            work_items = [(cand.ref_idx, cand.target_idx) for cand in candidates]
            results, _errors = _compute_feature_chunk(work_items)

            # --- Phase 7: Process results and persist ---
            for i, (result, (gers_id, target_id, pair_data)) in enumerate(
                zip(results, candidate_metadata)
            ):
                if result is None:
                    skipped += 1
                    continue

                if result.get("_error"):
                    errored += 1

                # Backfill topology into data store if not already stored
                if (
                    has_stored_data
                    and pair_data is not None
                    and pair_data.get("ref_topology") is None
                ):
                    ref_topo = worker_data["ref_topology_full"].get(candidates[i].ref_idx)
                    target_topo = worker_data["target_topology_full"].get(candidates[i].target_idx)
                    if ref_topo is not None and target_topo is not None:
                        data_store.update_topology(
                            gers_id,
                            target_id,
                            ref_topology=ref_topo,
                            target_topology=target_topo,
                        )

                # Backfill sampled target topology connectors into data store
                if (
                    has_stored_data
                    and pair_data is not None
                    and pair_data.get("target_topo_sampled") is None
                ):
                    seg_id = str(worker_data["target_ids"][candidates[i].target_idx])
                    sampled_connectors = worker_data.get("target_topo_connectors", {}).get(seg_id)
                    sampled_node_feats = worker_data.get("target_topo_node_features", {})
                    if sampled_connectors:
                        # Convert from (frac, node_id) to (frac, degree) for storage
                        sampled_for_storage = [
                            (frac, sampled_node_feats.get(nid, 1))
                            for frac, nid in sampled_connectors
                        ]
                        data_store.update_topo_sampled(
                            gers_id,
                            target_id,
                            target_topo_sampled=sampled_for_storage,
                        )

                # Backfill raw names structs into data store
                if has_stored_data and pair_data is not None:
                    ref_names_struct = (
                        worker_data["ref_names"][candidates[i].ref_idx]
                        if "ref_names" in worker_data
                        else None
                    )
                    target_names_struct = (
                        worker_data["target_names"][candidates[i].target_idx]
                        if "target_names" in worker_data
                        else None
                    )
                    if ref_names_struct is not None or target_names_struct is not None:
                        # Only write dicts (not flat strings)
                        ref_dict = ref_names_struct if isinstance(ref_names_struct, dict) else None
                        target_dict = (
                            target_names_struct if isinstance(target_names_struct, dict) else None
                        )
                        if ref_dict is not None or target_dict is not None:
                            data_store.update_names_raw(
                                gers_id,
                                target_id,
                                ref_names=ref_dict,
                                target_names=target_dict,
                            )

                feature_store.add(gers_id=gers_id, target_id=target_id, features=result)
                computed += 1

            # Save feature store and data store (topology backfill)
            if computed > 0:
                feature_store.save()
                if has_stored_data:
                    data_store.save()
                parts = [f"Computed {computed} features"]
                if used_stored > 0 or used_lookup > 0:
                    parts[0] += f" (stored={used_stored}, lookup={used_lookup})"
                if no_stored_rejected > 0:
                    parts.append(f"rejected={no_stored_rejected} (no stored data)")
                if errored > 0:
                    error_rate = errored / computed
                    parts.append(f"[red]errored={errored} ({error_rate:.0%})[/red]")
                parts.append(f"skipped={skipped}")
                console.print("  " + ", ".join(parts))
            else:
                reason = "IDs not in data"
                if no_stored_rejected > 0:
                    reason += f", {no_stored_rejected} rejected (no stored data)"
                console.print(f"  [yellow]Skipped all {skipped} pairs ({reason})[/yellow]")

            total_computed += computed
            total_skipped += skipped
            total_errored += errored

        # Report results
        console.print(
            f"\nBackfill complete: {total_computed} features computed, {total_skipped} skipped"
        )
        if total_errored > 0:
            error_rate = total_errored / total_computed if total_computed > 0 else 0
            console.print(
                f"[red]WARNING: {total_errored} pairs ({error_rate:.1%}) fell back to error features. "
                f"This likely indicates a bug in feature computation or bad input data.[/red]"
            )
            if error_rate > 0.05:
                console.print(
                    "[red]ERROR: Error rate exceeds 5% threshold. "
                    "Features were saved but should NOT be committed until the issue is resolved.[/red]"
                )
                raise typer.Exit(1)

    @app.command()
    def ui(
        port: int = typer.Option(8505, "--port", "-p", help="Server port"),
        host: str = typer.Option("0.0.0.0", "--host", "-H", help="Server host"),
        reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    ):
        """Launch the web UI (Label Creation, Label Review, Integration QA)."""
        import uvicorn

        display_host = "localhost" if host == "0.0.0.0" else host
        console.print(f"[blue]Starting crosswalk web UI on port {port}...[/blue]")
        console.print(f"[green]Open http://{display_host}:{port} in your browser[/green]")

        reload_kwargs = {}
        if reload:
            # Only watch source code — not data/, labels/, or cache files
            from importlib.util import find_spec

            spec = find_spec("crosswalk")
            if spec and spec.submodule_search_locations:
                reload_kwargs["reload_dirs"] = list(spec.submodule_search_locations)

        uvicorn.run(
            "crosswalk.web.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
            **reload_kwargs,
        )

    @app.command()
    def version():
        """Show version information."""
        from .. import __version__

        console.print(f"crosswalk version {__version__}")
