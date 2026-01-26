"""ML commands: train, eval-model, compute-features, benchmark."""

from pathlib import Path

import typer

from ._app import app, console


@app.command("compute-features")
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
        matcher compute-features us_boston_streets           # Single dataset
        matcher compute-features --prefix us_                # All US datasets
        matcher compute-features --all                       # All datasets with data
        matcher compute-features --all --force               # Recompute all
        matcher compute-features us_boston_streets -w 4      # Limit workers
        matcher compute-features --all --generate-candidates # Full precache for UI
    """
    from ..datasets.schema import get_dataset_config, list_dataset_configs
    from ..filenames import find_overture_segments, find_target_file
    from ..labeling.data_loader import (
        build_views_from_feature_df,
        compute_features_only,
        get_feature_cache_info,
        load_feature_cache,
        load_geodataframe,
        save_candidates_to_cache,
        save_feature_cache,
    )

    raw_dir = Path("data/raw")

    def get_dataset_files(dataset_id: str) -> tuple[Path, Path] | None:
        """Get reference and target file paths for a dataset."""
        config = get_dataset_config(dataset_id)
        if config is None:
            return None

        # Target file (with version suffix)
        target_path = find_target_file(raw_dir, dataset_id)
        if target_path is None:
            return None

        # Reference file (Overture, with version suffix)
        ref_path = find_overture_segments(raw_dir, dataset_id)
        if ref_path is None:
            return None

        return ref_path, target_path

    def compute_for_dataset(dataset_id: str) -> bool:
        """Compute features for a single dataset. Returns True if successful."""
        files = get_dataset_files(dataset_id)
        if files is None:
            console.print(f"[yellow]Skipping {dataset_id}: missing data files[/yellow]")
            return False

        ref_path, target_path = files

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
                reference = load_geodataframe(ref_path)
                target = load_geodataframe(target_path)

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
                    reference = load_geodataframe(ref_path)
                if target is None:
                    target = load_geodataframe(target_path)

                # Build views (runs ML scoring)
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
        for dataset_id in sorted(list_dataset_configs()):
            if get_dataset_files(dataset_id) is not None:
                datasets_to_process.append(dataset_id)

        if not datasets_to_process:
            console.print("[yellow]No datasets found with fetched data[/yellow]")
            raise typer.Exit(1)

        console.print(f"[blue]Found {len(datasets_to_process)} datasets with data[/blue]")

    elif prefix:
        # Find datasets matching prefix
        for dataset_id in sorted(list_dataset_configs()):
            if dataset_id.startswith(prefix) and get_dataset_files(dataset_id) is not None:
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
):
    """Train an ML model on labeled data.

    Loads labels from Hive-partitioned CSV format (labels/dataset=*/data.csv).

    Examples:
        matcher train
        matcher train --labels labels -o data/models/my_model.joblib

        # Train geometry-only model (no name/class features)
        matcher train --exclude-semantic -o data/models/matcher_model_geom_only.joblib

        # Leave-one-out: train without Frisco labels to test generalization
        matcher train -x us_frisco_trails -o data/models/no_frisco.joblib
    """
    from ..labeling.label_store import LabelStore
    from ..matching.ml import MLMatcher

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    # Check for dataset partitions
    partitions = list(labels_dir.glob("dataset=*/data.csv"))
    if not partitions:
        console.print(f"[red]No label partitions found in {labels_dir}[/red]")
        console.print("[yellow]Expected format: labels/dataset=*/data.csv[/yellow]")
        raise typer.Exit(1)

    console.print(f"[blue]Loading labels from {labels_dir}...[/blue]")
    df = LabelStore.load_all(labels_dir)
    console.print(f"  Found {len(df)} labels from {df['dataset'].nunique()} datasets")

    if exclude_dataset:
        console.print(f"[yellow]Excluding datasets: {', '.join(exclude_dataset)}[/yellow]")

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
    )

    # Save model
    output.parent.mkdir(parents=True, exist_ok=True)
    matcher.save_model(str(output))

    console.print(f"\n[green]Model saved to {output}[/green]")
    console.print(f"[green]Holdout accuracy: {metrics['test_accuracy']:.1%}[/green]")


@app.command("eval-model")
def eval_model(
    model: Path = typer.Argument(..., help="Path to trained model"),
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Labels directory (Hive-partitioned CSV format)",
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
    holdout: bool = typer.Option(
        True,
        "--holdout/--no-holdout",
        help="Use holdout set for evaluation (default: True for unbiased metrics)",
    ),
    holdout_pct: float = typer.Option(
        0.2,
        "--holdout-pct",
        help="Fraction of data to hold out for testing (default: 0.2 = 20%%)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for holdout split (use same seed for comparable results)",
    ),
):
    """Evaluate ML model performance on labeled data.

    By default, evaluates on a 20%% holdout set for unbiased metrics.
    Use --no-holdout to evaluate on ALL data (may include training data).

    Examples:
        matcher eval-model data/models/matcher_model.joblib
        matcher eval-model data/models/combined.joblib --no-holdout
        matcher eval-model data/models/combined.joblib --seed 123
        matcher eval-model data/models/combined.joblib --holdout-pct 0.3

        # Evaluate only on specific dataset (for leave-one-out testing)
        matcher eval-model data/models/no_frisco.joblib -d us_frisco_trails --no-holdout
    """
    from ..matching.ml import evaluate_by_dataset

    if not model.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    if dataset:
        console.print(f"[blue]Filtering to datasets: {', '.join(dataset)}[/blue]")

    if holdout:
        console.print(
            f"[blue]Evaluating {model.name} on {holdout_pct * 100:.0f}% holdout (seed={seed})...[/blue]"
        )
    else:
        console.print(
            f"[yellow]Evaluating {model.name} on all data (may include training data)...[/yellow]"
        )

    evaluate_by_dataset(
        str(model),
        str(labels_dir),
        show_by_dataset=by_dataset,
        holdout=holdout,
        holdout_pct=holdout_pct,
        seed=seed,
        filter_datasets=list(dataset) if dataset else None,
    )

    console.print("[green]Evaluation complete[/green]")


@app.command()
def benchmark(
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
    train_size: float = typer.Option(
        0.7,
        "--train-size",
        "-t",
        help="Fraction of data for training (default: 0.7 = 70/30 split)",
    ),
    seed: int = typer.Option(
        999,
        "--seed",
        "-s",
        help="Random seed for train/test split",
    ),
    skip_save: bool = typer.Option(
        False,
        "--skip-save",
        help="Skip saving results to CSV (just print)",
    ),
):
    """Run a benchmark: train on subset, evaluate on holdout.

    Uses segment-aware splitting to prevent data leakage - no segment
    appears in both train and test sets. Results are saved to
    benchmarks/model_performance.csv for tracking over time.

    Examples:
        matcher benchmark
        matcher benchmark --train-size 0.8
        matcher benchmark --seed 123 --skip-save
    """
    import csv
    from datetime import UTC, datetime

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    from ..labeling.label_store import LabelStore
    from ..matching.ml import MLMatcher, segment_aware_split

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    # Validate train size
    if not 0.1 <= train_size <= 0.9:
        console.print("[red]--train-size must be between 0.1 and 0.9[/red]")
        raise typer.Exit(1)

    run_date = datetime.now(UTC)
    test_pct = int((1 - train_size) * 100)
    train_pct = int(train_size * 100)

    console.print("[blue]Loading labels...[/blue]")
    all_labels = LabelStore.load_all(labels_dir)
    console.print(f"  Total labels: {len(all_labels)}")

    # Filter to valid labels only
    valid_labels = {"match", "no_match"}
    all_labels = all_labels[all_labels["label"].isin(valid_labels)].copy()
    console.print(f"  Valid labels (match/no_match): {len(all_labels)}")

    # Segment-aware split to prevent leakage
    console.print(f"\n[blue]Splitting {train_pct}/{test_pct} with segment-aware split...[/blue]")
    train_idx, test_idx = segment_aware_split(
        all_labels, test_size=1 - train_size, random_state=seed
    )

    train_df = all_labels.iloc[train_idx].copy()
    test_df = all_labels.iloc[test_idx].copy()

    console.print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    console.print(f"  Train labels: {train_df['label'].value_counts().to_dict()}")
    console.print(f"  Test labels: {test_df['label'].value_counts().to_dict()}")

    # Save train labels to temp directory for training
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save each dataset's train portion
        for dataset in train_df["dataset"].unique():
            ds_train = train_df[train_df["dataset"] == dataset]
            ds_dir = tmpdir / f"dataset={dataset}"
            ds_dir.mkdir(parents=True, exist_ok=True)
            ds_train.to_csv(ds_dir / "data.csv", index=False)

        # Train model on train set only (no internal split since we already split)
        console.print(f"\n[blue]Training model on {len(train_df)} samples...[/blue]")
        matcher = MLMatcher()
        matcher.train(labels_dir=str(tmpdir), binary=True, test_size=0.0)

        # Save the model
        model_dir = Path("data/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "matcher_model_combined.joblib"
        matcher.save_model(str(model_path))
        console.print(f"  Model saved to {model_path}")

        # Evaluate on test set (completely unseen during training)
        console.print(f"\n[blue]Evaluating on {len(test_df)} holdout samples...[/blue]")

        X_test, y_test = matcher._extract_features_and_labels(test_df, binary=True)
        X_test = matcher._impute_missing(X_test)
        y_pred = matcher.model.predict(X_test)

        # Overall metrics
        overall_acc = accuracy_score(y_test, y_pred)
        overall_f1 = f1_score(y_test, y_pred, average="weighted")
        overall_precision = precision_score(y_test, y_pred, average="weighted")
        overall_recall = recall_score(y_test, y_pred, average="weighted")

        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]EVALUATION ON {test_pct}% HOLDOUT ({len(test_df)} samples)[/bold]")
        console.print("=" * 60)
        console.print("\nOverall:")
        console.print(f"  Accuracy:  {overall_acc:.3f}")
        console.print(f"  F1:        {overall_f1:.3f}")
        console.print(f"  Precision: {overall_precision:.3f}")
        console.print(f"  Recall:    {overall_recall:.3f}")

        # Extract top 5 feature importances
        feature_importances = dict(zip(matcher.feature_names, matcher.model.feature_importances_))
        top_5_features = sorted(feature_importances.items(), key=lambda x: -x[1])[:5]

        console.print("\nTop 5 features by importance:")
        for feat, imp in top_5_features:
            console.print(f"  {feat}: {imp:.3f}")

        # Per-dataset metrics
        results = {}
        console.print("\nPer-dataset results:")
        for dataset in sorted(test_df["dataset"].unique()):
            ds_test = test_df[test_df["dataset"] == dataset]
            X_ds, y_ds = matcher._extract_features_and_labels(ds_test, binary=True)
            X_ds = matcher._impute_missing(X_ds)
            y_ds_pred = matcher.model.predict(X_ds)

            ds_acc = accuracy_score(y_ds, y_ds_pred)
            ds_f1 = f1_score(y_ds, y_ds_pred, average="weighted")
            ds_precision = precision_score(y_ds, y_ds_pred, average="weighted", zero_division=0)
            ds_recall = recall_score(y_ds, y_ds_pred, average="weighted", zero_division=0)
            n_match = int((y_ds == 1).sum())
            n_no_match = int((y_ds == 0).sum())

            console.print(
                f"  {dataset}: acc={ds_acc:.3f}, f1={ds_f1:.3f} "
                f"(n={len(ds_test)}, match={n_match}, no_match={n_no_match})"
            )

            results[dataset] = {
                "n_samples": len(ds_test),
                "n_match": n_match,
                "n_no_match": n_no_match,
                "accuracy": ds_acc,
                "f1": ds_f1,
                "precision": ds_precision,
                "recall": ds_recall,
            }

        # Save results to CSV
        if not skip_save:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / "model_performance.csv"

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
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        "train_size": train_size,
                        "n_samples": metrics.get("n_samples", 0),
                        "n_match": metrics.get("n_match", 0),
                        "n_no_match": metrics.get("n_no_match", 0),
                        "accuracy": f"{metrics.get('accuracy', 0):.4f}",
                        "f1": f"{metrics.get('f1', 0):.4f}",
                        "precision": f"{metrics.get('precision', 0):.4f}",
                        "recall": f"{metrics.get('recall', 0):.4f}",
                        "split_seed": seed,
                        "model_name": model_path.name,
                        "top1_feature": top_5_features[0][0] if len(top_5_features) > 0 else "",
                        "top1_importance": f"{top_5_features[0][1]:.4f}"
                        if len(top_5_features) > 0
                        else "",
                        "top2_feature": top_5_features[1][0] if len(top_5_features) > 1 else "",
                        "top2_importance": f"{top_5_features[1][1]:.4f}"
                        if len(top_5_features) > 1
                        else "",
                        "top3_feature": top_5_features[2][0] if len(top_5_features) > 2 else "",
                        "top3_importance": f"{top_5_features[2][1]:.4f}"
                        if len(top_5_features) > 2
                        else "",
                        "top4_feature": top_5_features[3][0] if len(top_5_features) > 3 else "",
                        "top4_importance": f"{top_5_features[3][1]:.4f}"
                        if len(top_5_features) > 3
                        else "",
                        "top5_feature": top_5_features[4][0] if len(top_5_features) > 4 else "",
                        "top5_importance": f"{top_5_features[4][1]:.4f}"
                        if len(top_5_features) > 4
                        else "",
                    }
                    writer.writerow(row)

            console.print(f"\n[green]Results saved to {results_file}[/green]")

    console.print("\n[green]Benchmark complete![/green]")
