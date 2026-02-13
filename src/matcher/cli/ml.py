"""Machine learning commands."""

import csv
from datetime import UTC, datetime
from pathlib import Path

import typer

from .utils import console

# Create ml group
ml_app = typer.Typer(
    name="ml",
    help="Machine learning commands",
    no_args_is_help=True,
)


@ml_app.command("eval")
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
):
    """Evaluate ML model performance using cross-validation.

    By default, runs k-fold cross-validation with segment-aware splitting to
    prevent data leakage. Reports mean ± std for all metrics.

    Use --model to evaluate an existing trained model on a holdout set instead.

    Examples:
        # Cross-validation evaluation (default)
        matcher ml eval
        matcher ml eval --cv-folds 10
        matcher ml eval --seed 123 --skip-save

        # Evaluate an existing model (single holdout)
        matcher ml eval --model data/models/matcher_model.joblib
        matcher ml eval -m data/models/combined.joblib -d us_frisco_trails
    """
    if model is not None:
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

    # Load model
    matcher = MLMatcher()
    matcher.load_model(str(model))

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
    from ..matching.ml import MLMatcher, create_segment_groups

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

        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
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


@ml_app.command("errors")
def analyze_errors(
    model: Path = typer.Option(
        Path("data/models/matcher_model_combined.joblib"),
        "--model",
        "-m",
        help="Path to trained model",
    ),
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Labels directory (Hive-partitioned CSV format)",
    ),
    dataset: list[str] = typer.Option(
        [],
        "--dataset",
        "-d",
        help="Focus on specific dataset(s). Can be repeated. Default: all datasets.",
    ),
    top_n: int = typer.Option(
        20,
        "--top",
        "-n",
        help="Number of worst errors to show per dataset",
    ),
    output_dir: Path = typer.Option(
        Path("analysis"),
        "--output",
        "-o",
        help="Output directory for error analysis reports",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        "-c",
        help="Minimum model confidence to consider 'high confidence' errors",
    ),
):
    """Analyze prediction errors to diagnose model performance issues.

    This command identifies error patterns in model predictions on labeled data:

    1. **Confusion Matrix**: False positives vs false negatives breakdown
    2. **High-Confidence Errors**: Wrong predictions with model confidence >= threshold
    3. **Feature Analysis**: Which features correlate with errors
    4. **Error Export**: CSV of worst errors for review in labeling UI

    Use this to diagnose why certain datasets underperform and whether issues
    are model failures or potential label quality problems.

    Examples:
        # Analyze errors on all labeled datasets
        matcher ml errors

        # Focus on underperforming datasets
        matcher ml errors -d br_sao_paulo_roads -d us_fort_collins_streets

        # Show top 50 worst errors
        matcher ml errors --top 50

        # Use custom confidence threshold for "high confidence" errors
        matcher ml errors --min-confidence 0.8
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    from ..config import METRIC_AVERAGE
    from ..labeling.label_store import LabelStore
    from ..matching.ml import MLMatcher

    if not model.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Loading model: {model.name}[/blue]")
    matcher = MLMatcher()
    matcher.load_model(str(model))

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

    if len(all_labels) == 0:
        console.print("[yellow]No labels to analyze[/yellow]")
        raise typer.Exit(0)

    # Extract features and get predictions
    console.print("[blue]Computing predictions...[/blue]")
    X, y_true = matcher._extract_features_and_labels(all_labels, binary=True)
    X = matcher._cap_infinities(X)

    y_pred = matcher.model.predict(X)
    y_prob = matcher.model.predict_proba(X)

    # Get confidence scores (probability of predicted class)
    # For binary classification: column 0 = no_match, column 1 = match
    match_probs = y_prob[:, 1]  # Probability of being a match
    pred_confidence = np.where(y_pred == 1, match_probs, 1 - match_probs)

    # Add predictions to dataframe for analysis
    all_labels = all_labels.copy()
    all_labels["y_true"] = y_true
    all_labels["y_pred"] = y_pred
    all_labels["match_prob"] = match_probs
    all_labels["pred_confidence"] = pred_confidence
    all_labels["is_error"] = y_true != y_pred
    all_labels["is_fp"] = (y_pred == 1) & (y_true == 0)  # Predicted match, actually no_match
    all_labels["is_fn"] = (y_pred == 0) & (y_true == 1)  # Predicted no_match, actually match

    # Overall metrics
    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, average=METRIC_AVERAGE, zero_division=0)
    overall_cm = confusion_matrix(y_true, y_pred)

    console.print(f"\n{'=' * 70}")
    console.print("[bold]ERROR ANALYSIS SUMMARY[/bold]")
    console.print("=" * 70)

    console.print("\n[bold]Overall Performance[/bold]")
    console.print(f"  Accuracy: {overall_acc:.3f}")
    console.print(f"  F1 Score: {overall_f1:.3f}")
    console.print(f"  Total errors: {all_labels['is_error'].sum()} / {len(all_labels)}")

    console.print("\n[bold]Confusion Matrix[/bold]")
    console.print("                  Predicted")
    console.print("                  no_match  match")
    console.print(
        f"  Actual no_match   {overall_cm[0, 0]:5d}    {overall_cm[0, 1]:5d}  (FP rate: {overall_cm[0, 1] / overall_cm[0].sum():.1%})"
    )
    console.print(
        f"  Actual match      {overall_cm[1, 0]:5d}    {overall_cm[1, 1]:5d}  (FN rate: {overall_cm[1, 0] / overall_cm[1].sum():.1%})"
    )

    n_fp = all_labels["is_fp"].sum()
    n_fn = all_labels["is_fn"].sum()
    console.print(f"\n  False Positives (predicted match, was no_match): {n_fp}")
    console.print(f"  False Negatives (predicted no_match, was match): {n_fn}")

    # High-confidence errors (these are the most diagnostic)
    high_conf_errors = all_labels[
        (all_labels["is_error"]) & (all_labels["pred_confidence"] >= min_confidence)
    ]
    console.print(f"\n[bold]High-Confidence Errors (confidence >= {min_confidence})[/bold]")
    console.print(f"  Total: {len(high_conf_errors)} / {all_labels['is_error'].sum()} errors")

    if len(high_conf_errors) > 0:
        hc_fp = high_conf_errors["is_fp"].sum()
        hc_fn = high_conf_errors["is_fn"].sum()
        console.print(
            f"  High-conf FP: {hc_fp} (model confident it's a match, but labeled no_match)"
        )
        console.print(f"  High-conf FN: {hc_fn} (model confident it's no_match, but labeled match)")
        console.print("\n  [yellow]High-confidence errors suggest either:[/yellow]")
        console.print("    - Systematic model failure on certain patterns")
        console.print("    - Potential label quality issues (mislabeled examples)")

    # Per-dataset breakdown
    console.print("\n[bold]Per-Dataset Error Analysis[/bold]")
    console.print("-" * 70)

    dataset_errors = []
    for ds in sorted(all_labels["dataset"].unique()):
        ds_df = all_labels[all_labels["dataset"] == ds]
        ds_acc = accuracy_score(ds_df["y_true"], ds_df["y_pred"])
        ds_f1 = f1_score(ds_df["y_true"], ds_df["y_pred"], average=METRIC_AVERAGE, zero_division=0)
        ds_n_errors = ds_df["is_error"].sum()
        ds_n_fp = ds_df["is_fp"].sum()
        ds_n_fn = ds_df["is_fn"].sum()
        ds_hc_errors = len(
            ds_df[(ds_df["is_error"]) & (ds_df["pred_confidence"] >= min_confidence)]
        )

        dataset_errors.append(
            {
                "dataset": ds,
                "n": len(ds_df),
                "acc": ds_acc,
                "f1": ds_f1,
                "errors": ds_n_errors,
                "fp": ds_n_fp,
                "fn": ds_n_fn,
                "hc_errors": ds_hc_errors,
            }
        )

        console.print(
            f"  {ds}: F1={ds_f1:.3f}, Acc={ds_acc:.3f}, "
            f"Errors={ds_n_errors} (FP={ds_n_fp}, FN={ds_n_fn}), "
            f"HiConf={ds_hc_errors}"
        )

    # Feature correlation with errors
    console.print("\n[bold]Feature Analysis: Errors vs Correct[/bold]")
    console.print("-" * 70)

    # Compute mean feature values for errors vs correct predictions
    errors_df = all_labels[all_labels["is_error"]]
    correct_df = all_labels[~all_labels["is_error"]]

    feature_diffs = []
    for feat_name in matcher.feature_names:
        if feat_name in all_labels.columns:
            err_mean = errors_df[feat_name].mean()
            cor_mean = correct_df[feat_name].mean()
            # Handle inf/nan values and division by zero
            if np.isfinite(err_mean) and np.isfinite(cor_mean) and cor_mean != 0:
                pct_diff = (err_mean - cor_mean) / abs(cor_mean) * 100
            elif np.isfinite(err_mean) and np.isfinite(cor_mean):
                pct_diff = 0 if err_mean == cor_mean else float("inf")
            else:
                pct_diff = np.nan  # Skip features with inf/nan means
            feature_diffs.append(
                {
                    "feature": feat_name,
                    "error_mean": err_mean,
                    "correct_mean": cor_mean,
                    "pct_diff": pct_diff,
                }
            )

    # Sort by absolute percentage difference (filter out nan values for sorting)
    feature_diffs = [fd for fd in feature_diffs if np.isfinite(fd["pct_diff"])]
    feature_diffs.sort(key=lambda x: abs(x["pct_diff"]), reverse=True)

    console.print("  Features with largest difference between errors and correct predictions:")
    for fd in feature_diffs[:10]:
        direction = "↑" if fd["pct_diff"] > 0 else "↓"
        console.print(
            f"    {fd['feature']}: errors={fd['error_mean']:.3f}, "
            f"correct={fd['correct_mean']:.3f} ({direction}{abs(fd['pct_diff']):.1f}%)"
        )

    # Export worst errors for review
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export all errors with details
    error_export = all_labels[all_labels["is_error"]].copy()
    error_export = error_export.sort_values("pred_confidence", ascending=False)

    # Select relevant columns for export
    export_cols = [
        "dataset",
        "gers_id",
        "target_id",
        "label",
        "y_pred",
        "match_prob",
        "pred_confidence",
        "is_fp",
        "is_fn",
    ]
    # Add key features
    key_features = [
        "name_jaro_winkler",
        "buffer_iou_5m",
        "hausdorff_distance_m",
        "class_similarity",
        "lateral_offset_m",
        "min_coverage",
    ]
    for feat in key_features:
        if feat in error_export.columns:
            export_cols.append(feat)

    error_export = error_export[[c for c in export_cols if c in error_export.columns]]

    errors_path = output_dir / "prediction_errors.csv"
    error_export.to_csv(errors_path, index=False)
    console.print(f"\n[green]Exported {len(error_export)} errors to {errors_path}[/green]")

    # Export high-confidence errors separately (priority for review)
    if len(high_conf_errors) > 0:
        hc_export = high_conf_errors[[c for c in export_cols if c in high_conf_errors.columns]]
        hc_export = hc_export.sort_values("pred_confidence", ascending=False)
        hc_path = output_dir / "high_confidence_errors.csv"
        hc_export.to_csv(hc_path, index=False)
        console.print(
            f"[green]Exported {len(hc_export)} high-confidence errors to {hc_path}[/green]"
        )

    # Print top errors per dataset
    console.print(f"\n[bold]Top {top_n} Worst Errors Per Dataset[/bold]")
    console.print("(Sorted by model confidence - these are systematic failures or label issues)")
    console.print("-" * 70)

    for ds in sorted(all_labels["dataset"].unique()):
        ds_errors = all_labels[
            (all_labels["dataset"] == ds) & (all_labels["is_error"])
        ].sort_values("pred_confidence", ascending=False)

        if len(ds_errors) == 0:
            continue

        console.print(f"\n[bold]{ds}[/bold] ({len(ds_errors)} errors)")

        for i, (_, row) in enumerate(ds_errors.head(top_n).iterrows()):
            error_type = "FP" if row["is_fp"] else "FN"
            pred_label = "match" if row["y_pred"] == 1 else "no_match"
            true_label = row["label"]

            # Get key feature values
            name_sim = row.get("name_jaro_winkler", 0)
            buf_iou = row.get("buffer_iou_5m", 0)
            haus = row.get("hausdorff_distance_m", 0)

            console.print(
                f"  {i + 1}. [{error_type}] conf={row['pred_confidence']:.2f} "
                f"pred={pred_label} actual={true_label}"
            )
            console.print(f"      gers={row['gers_id'][:20]}... target={row['target_id'][:20]}...")
            console.print(f"      name_jw={name_sim:.2f} buf_iou={buf_iou:.2f} haus={haus:.1f}m")

    # Summary recommendations
    console.print(f"\n{'=' * 70}")
    console.print("[bold]RECOMMENDATIONS[/bold]")
    console.print("=" * 70)

    # Identify datasets that need attention
    worst_datasets = sorted(dataset_errors, key=lambda x: x["f1"])[:3]
    if worst_datasets:
        console.print("\n[bold]Priority Datasets for Review:[/bold]")
        for ds in worst_datasets:
            console.print(f"  - {ds['dataset']}: F1={ds['f1']:.3f}, {ds['errors']} errors")

    # Recommend actions based on error patterns
    total_hc = sum(d["hc_errors"] for d in dataset_errors)
    if total_hc > 10:
        console.print(f"\n[yellow]Found {total_hc} high-confidence errors.[/yellow]")
        console.print("  → Review these in the labeling UI to check if labels are correct")
        console.print(f"  → See: {output_dir / 'high_confidence_errors.csv'}")

    if n_fp > n_fn * 2:
        console.print("\n[yellow]Model has 2x more False Positives than False Negatives.[/yellow]")
        console.print("  → Model is too aggressive in predicting matches")
        console.print("  → Consider raising the match threshold or adding more no_match examples")
    elif n_fn > n_fp * 2:
        console.print("\n[yellow]Model has 2x more False Negatives than False Positives.[/yellow]")
        console.print("  → Model is too conservative in predicting matches")
        console.print("  → Consider lowering the match threshold or adding more match examples")

    console.print("\n[green]Error analysis complete![/green]")


@ml_app.command("features")
def compute_features(
    dataset: str = typer.Argument(
        None,
        help="Dataset ID to compute features for (e.g., us_boston_streets)",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Compute features for all datasets with fetched data",
    ),
    prefix: str = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Compute features for datasets matching this prefix",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Recompute even if cached (ignores existing feature cache)",
    ),
    workers: int = typer.Option(
        -1,
        "--workers",
        "-w",
        help="Number of parallel workers for feature computation (-1 for auto)",
    ),
    generate_candidates: bool = typer.Option(
        False,
        "--generate-candidates",
        "-c",
        help="Also generate scored candidates cache for labeling UI (runs ML scoring)",
    ),
):
    """Compute and cache features for dataset(s) without ML scoring.

    This pre-computes the expensive feature computation step (~90% of processing time)
    and caches the results. When the labeling UI loads, it can use the cached features
    and only run ML prediction (fast).

    Features are versioned - when feature computation logic changes, bump FEATURE_VERSION
    in config.py and old caches will be ignored.

    Use --generate-candidates to also run ML scoring and cache the final candidates,
    making the labeling UI load instantly.

    Examples:
        matcher ml features us_boston_streets           # Single dataset
        matcher ml features --prefix us_                # All US datasets
        matcher ml features --all                       # All datasets with data
        matcher ml features --all --force               # Recompute all
        matcher ml features us_boston_streets -w 4      # Limit workers
        matcher ml features --all --generate-candidates # Full precache for UI
    """
    from ..datasets.loader import DatasetLoader
    from ..labeling.data_loader import (
        build_views_from_feature_df,
        compute_features_only,
        get_feature_cache_info,
        load_feature_cache,
        save_candidates_to_cache,
        save_feature_cache,
    )

    loader = DatasetLoader()

    def compute_for_dataset(dataset_id: str) -> bool:
        """Compute features for a single dataset. Returns True if successful."""
        ref_path = loader.find_reference_path(dataset_id)
        target_path = loader.find_target_path(dataset_id)
        if ref_path is None or target_path is None:
            console.print(f"[yellow]Skipping {dataset_id}: missing data files[/yellow]")
            return False

        # Check cache
        cache_info = get_feature_cache_info(dataset_id, ref_path, target_path)
        feature_cache_exists = cache_info["exists"]

        if feature_cache_exists and not force and not generate_candidates:
            console.print(
                f"[blue]Skipping {dataset_id}: feature cache exists "
                f"({cache_info['candidate_count']:,} candidates, "
                f"version {cache_info['version']})[/blue]"
            )
            return True

        # Use standardized Overture-format column names for parquet files
        # (the fetch step transforms source columns to these during data ingestion)
        from ..config import CLASS_COLUMN, NAMES_COLUMN

        ref_name_column = NAMES_COLUMN
        target_name_column = NAMES_COLUMN
        ref_class_column = CLASS_COLUMN
        target_class_column = CLASS_COLUMN

        try:
            feature_df = None
            reference = None
            target = None

            # Load or compute features
            if feature_cache_exists and not force:
                console.print(f"[blue]Loading cached features for {dataset_id}...[/blue]")
                feature_df = load_feature_cache(dataset_id)
            else:
                console.print(f"[blue]Computing features for {dataset_id}...[/blue]")

                # Load data
                reference = loader._load_gdf(ref_path)
                target = loader._load_gdf(target_path)

                console.print(f"  Reference: {len(reference):,} segments")
                console.print(f"  Target: {len(target):,} segments")

                # Compute features
                feature_df = compute_features_only(
                    reference=reference,
                    target=target,
                    ref_name_column=ref_name_column,
                    target_name_column=target_name_column,
                    ref_class_column=ref_class_column,
                    target_class_column=target_class_column,
                    n_jobs=workers,
                )

                if len(feature_df) == 0:
                    console.print(f"[yellow]  No candidates generated for {dataset_id}[/yellow]")
                    return False

                # Save feature cache
                cache_path = save_feature_cache(dataset_id, feature_df)
                console.print(
                    f"[green]  Saved {len(feature_df):,} features to {cache_path.name}[/green]"
                )

            # Generate candidates cache if requested
            if generate_candidates and feature_df is not None:
                console.print(f"[blue]  Generating scored candidates for {dataset_id}...[/blue]")

                # Load geodataframes if not already loaded (when using cached features)
                if reference is None:
                    reference = loader._load_gdf(ref_path)
                if target is None:
                    target = loader._load_gdf(target_path)

                # Build views (runs ML scoring, filter to review band for labeling UI)
                views = build_views_from_feature_df(
                    feature_df=feature_df,
                    reference=reference,
                    target=target,
                    ref_id_column="id",
                    target_id_column="id",
                    ref_name_column=ref_name_column,
                    target_name_column=target_name_column,
                    ref_class_column=ref_class_column,
                    target_class_column=target_class_column,
                    filter_to_review_band=True,
                )

                if views:
                    candidates_path = save_candidates_to_cache(dataset_id, views)
                    console.print(
                        f"[green]  Saved {len(views):,} candidates to "
                        f"{candidates_path.name}[/green]"
                    )
                else:
                    console.print(f"[yellow]  No candidates to cache for {dataset_id}[/yellow]")

            return True

        except Exception as e:
            console.print(f"[red]  Error computing features for {dataset_id}: {e}[/red]")
            return False

    # Determine which datasets to process
    datasets_to_process: list[str] = []

    if all_datasets:
        # Find all datasets with both target and reference data
        datasets_to_process = loader.list_available()

        if not datasets_to_process:
            console.print("[yellow]No datasets found with fetched data[/yellow]")
            raise typer.Exit(1)

        console.print(f"[blue]Found {len(datasets_to_process)} datasets with data[/blue]")

    elif prefix:
        # Find datasets matching prefix
        for dataset_id in loader.list_available():
            if dataset_id.startswith(prefix):
                datasets_to_process.append(dataset_id)

        if not datasets_to_process:
            console.print(f"[yellow]No datasets found with prefix '{prefix}'[/yellow]")
            raise typer.Exit(1)

        console.print(
            f"[blue]Found {len(datasets_to_process)} datasets matching prefix '{prefix}'[/blue]"
        )

    elif dataset:
        datasets_to_process = [dataset]

    else:
        console.print("[red]Error: Provide a dataset name, --prefix, or --all[/red]")
        raise typer.Exit(1)

    # Process datasets sequentially
    success_count = 0
    skip_count = 0
    fail_count = 0

    for dataset_id in datasets_to_process:
        # Check cache BEFORE computation to distinguish skip vs compute
        cache_info_before = get_feature_cache_info(dataset_id)
        had_cache = cache_info_before.get("exists", False)

        result = compute_for_dataset(dataset_id)
        if not result:
            fail_count += 1
        elif had_cache and not force:
            # Had cache and didn't force recompute = skipped
            skip_count += 1
        else:
            # Newly computed (or force recomputed)
            success_count += 1

    # Summary
    console.print()
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Computed: {success_count}")
    console.print(f"  Skipped (cached): {skip_count}")
    if fail_count > 0:
        console.print(f"  [red]Failed: {fail_count}[/red]")
