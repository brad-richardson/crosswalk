"""Top-level CLI commands."""

import csv
import os
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

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


def register_commands(app: typer.Typer) -> None:
    """Register top-level commands on the given app."""

    @app.command()
    def match(
        reference: Path = typer.Argument(..., help="Reference edges (Overture)"),
        target: Path = typer.Argument(..., help="Target edges (local data)"),
        output: Path = typer.Option(
            Path("data/output/bridge.parquet"),
            "--output",
            "-o",
            help="Output bridge file path",
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
        profile: bool = typer.Option(
            False,
            "--profile",
            help="Enable per-feature timing breakdown (sets MATCHER_PROFILE=1)",
        ),
    ):
        """Run the full matching pipeline."""
        from ..pipeline import run_pipeline

        if profile:
            os.environ["MATCHER_PROFILE"] = "1"

        console.print("[blue]Running matching pipeline...[/blue]")
        console.print(f"  Reference: {reference}")
        console.print(f"  Target: {target}")
        console.print(f"  Method: {method}")
        console.print(f"  Buffer: {buffer_distance_m}m")
        if workers != -1:
            console.print(f"  [yellow]Workers: {workers}[/yellow]")

        output.parent.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Matching...", total=None)

            result = run_pipeline(
                reference_path=reference,
                target_path=target,
                output_path=output,
                method=method,
                buffer_distance_m=buffer_distance_m,
                n_jobs=workers,
            )

            progress.update(task, completed=True)

        console.print(f"[green]Matched {result.n_matched} / {result.n_target} features[/green]")
        console.print(f"[green]Bridge file: {output}[/green]")

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
            matcher train
            matcher train --labels labels -o data/models/my_model.joblib

            # Train geometry-only model (no name/class features)
            matcher train --exclude-semantic -o data/models/matcher_model_geom_only.joblib

            # Leave-one-out: train without Frisco labels to test generalization
            matcher train -x us_frisco_trails -o data/models/no_frisco.joblib

            # Train with weak supervision from agent labels
            matcher train --agent-weight 0.5 --min-agent-confidence 0.7
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
            console.print("[yellow]Expected: labels/human/ and labels/features/[/yellow]")
            console.print("[yellow]Expected normalized label format (labels/human/ + labels/features/).[/yellow]")
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
    ):
        """Evaluate ML model performance using cross-validation.

        By default, runs k-fold cross-validation with segment-aware splitting to
        prevent data leakage. Reports mean ± std for all metrics.

        Use --model to evaluate an existing trained model on a holdout set instead.

        Examples:
            # Cross-validation evaluation (default)
            matcher eval
            matcher eval --cv-folds 10
            matcher eval --seed 123 --skip-save

            # Evaluate an existing model (single holdout)
            matcher eval --model data/models/matcher_model.joblib
            matcher eval -m data/models/combined.joblib -d us_frisco_trails
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
            matcher backfill --dry-run       # Preview what would be recomputed
            matcher backfill                 # Recompute all human label features
            matcher backfill --include-agent # Also include agent labels
            matcher backfill --missing-only  # Only compute for labels without features
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
                else:
                    used_stored += 1
                    stored_target_overrides[target_id] = pair_data["target_geometry"]
                    if target_lookup is None or target_id not in target_lookup.index:
                        stored_target_attrs[target_id] = {
                            "names": pair_data.get("target_name"),
                            "names_lr": pair_data.get("target_names_lr"),
                            "class": pair_data.get("target_class"),
                            "subclass": pair_data.get("target_subclass"),
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
                        length_ratio=1.0,
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
            )
            worker_data = pipeline_result.worker_data

            # --- Phase 4b: Override graphlet data with full-network computation ---
            # The shared pipeline computes graphlets on candidate-only subsets (efficient
            # for inference with ~10K candidates). For backfill with ~100-200 labeled pairs,
            # the candidate-only graph is too sparse for meaningful clustering coefficients.
            # Recompute on full GDFs so clustering_coef reflects the actual network topology.
            ref_has_connectors = "connectors" in ref_gdf_proj.columns
            worker_data["ref_graphlet_data"] = precompute_graphlet_features(
                ref_gdf_proj,
                connectors_column="connectors" if ref_has_connectors else None,
            )
            worker_data["target_graphlet_data"] = precompute_graphlet_features(
                augmented_target,
            )

            # --- Phase 5: Override topology with stored values ---
            # 3-tier fallback: stored topology > computed by pipeline > NaN defaults
            for cand, (_gid, _tid, pair_data) in zip(candidates, candidate_metadata):
                if pair_data is not None:
                    stored_ref_topo = pair_data.get("ref_topology")
                    stored_target_topo = pair_data.get("target_topology")
                    if stored_ref_topo:
                        worker_data["ref_topology"][cand.ref_idx] = stored_ref_topo
                    if stored_target_topo:
                        worker_data["target_topology"][cand.target_idx] = stored_target_topo

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
                    ref_topo = worker_data["ref_topology"].get(candidates[i].ref_idx)
                    target_topo = worker_data["target_topology"].get(candidates[i].target_idx)
                    if ref_topo is not None and target_topo is not None:
                        data_store.update_topology(
                            gers_id,
                            target_id,
                            ref_topology=ref_topo,
                            target_topology=target_topo,
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
        console.print(f"[blue]Starting matcher web UI on port {port}...[/blue]")
        console.print(f"[green]Open http://{display_host}:{port} in your browser[/green]")

        reload_kwargs = {}
        if reload:
            # Only watch source code — not data/, labels/, or cache files
            from importlib.util import find_spec

            spec = find_spec("matcher")
            if spec and spec.submodule_search_locations:
                reload_kwargs["reload_dirs"] = list(spec.submodule_search_locations)

        uvicorn.run(
            "matcher.web.app:create_app",
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

        console.print(f"matcher version {__version__}")
