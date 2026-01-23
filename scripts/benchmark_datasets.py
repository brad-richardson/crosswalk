#!/usr/bin/env python3
"""Benchmark script for tracking model performance over time.

This script runs a full benchmarking workflow:
1. Fetch fresh data for all labeled datasets (local + Overture reference)
2. Backfill features on labels using fresh data
3. Split labels 70/30 (train/test) with stratification
4. Train model on 70% training set
5. Evaluate on 30% holdout (completely unseen during training)
6. Save results to benchmarks/model_performance.csv

Usage:
    # Full workflow
    python scripts/benchmark_datasets.py

    # Fetch only (skip training/evaluation)
    python scripts/benchmark_datasets.py --fetch-only

    # Evaluate only (skip fetch/train) - uses existing model
    python scripts/benchmark_datasets.py --eval-only

    # Skip backfill (use existing features)
    python scripts/benchmark_datasets.py --skip-backfill

    # Process specific dataset
    python scripts/benchmark_datasets.py --dataset us_boston_streets

    # Dry run (show what would be done)
    python scripts/benchmark_datasets.py --dry-run

    # Custom train/test split (default: 0.7)
    python scripts/benchmark_datasets.py --train-size 0.8
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

# Labeled datasets and their region configurations
# Maps dataset name to region info for Overture fetch
LABELED_DATASETS = {
    "us_boston_streets": {
        "fetch_script": "fetch_boston.py",
        "region": "us_boston",
        "bbox": "-71.19,42.21,-70.92,42.40",
        "overture_file": "us_boston_overture_segments.parquet",
    },
    "us_boston_sidewalks": {
        "fetch_script": "fetch_boston.py",
        "region": "us_boston",
        "bbox": "-71.19,42.21,-70.92,42.40",
        "overture_file": "us_boston_overture_segments.parquet",
    },
    "us_boston_bike_network": {
        "fetch_script": "fetch_boston.py",
        "region": "us_boston",
        "bbox": "-71.19,42.21,-70.92,42.40",
        "overture_file": "us_boston_overture_segments.parquet",
    },
    "us_boston_streets_osm": {
        "fetch_script": None,  # Fetched via CLI
        "region": "us_boston",
        "bbox": "-71.19,42.21,-70.92,42.40",
        "overture_file": "us_boston_overture_segments.parquet",
        "osm_fetch": True,  # Uses matcher fetch -d osm
    },
    "us_fort_collins_streets": {
        "fetch_script": "fetch_fort_collins.py",
        "region": "us_fort_collins",
        "bbox": "-105.15,40.45,-104.95,40.65",
        "overture_file": "us_fort_collins_overture_segments.parquet",
    },
    "us_frisco_trails": {
        "fetch_script": "fetch_frisco.py",
        "region": "us_frisco",
        "bbox": "-96.95,33.08,-96.75,33.18",
        "overture_file": "us_frisco_overture_segments.parquet",
    },
}

# Group datasets by region for efficient fetching
REGIONS = {
    "us_boston": {
        "bbox": "-71.19,42.21,-70.92,42.40",
        "overture_output": "us_boston_overture_segments.parquet",
        "fetch_scripts": ["fetch_boston.py"],
    },
    "us_fort_collins": {
        "bbox": "-105.15,40.45,-104.95,40.65",
        "overture_output": "us_fort_collins_overture_segments.parquet",
        "fetch_scripts": ["fetch_fort_collins.py"],
    },
    "us_frisco": {
        "bbox": "-96.95,33.08,-96.75,33.18",
        "overture_output": "us_frisco_overture_segments.parquet",
        "fetch_scripts": ["fetch_frisco.py"],
    },
}

# Paths
REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data" / "raw"
LABELS_DIR = REPO_ROOT / "labels"
MODELS_DIR = REPO_ROOT / "data" / "models"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
RESULTS_FILE = BENCHMARKS_DIR / "model_performance.csv"

# Default benchmark settings
DEFAULT_TRAIN_SIZE = 0.7  # 70% train, 30% test (industry standard)
SPLIT_SEED = 999  # Seed for train/test split (different from any internal seeds)


def run_command(cmd: list[str], description: str, dry_run: bool = False) -> bool:
    """Run a shell command and log the result."""
    cmd_str = " ".join(cmd)
    if dry_run:
        logger.info(f"[DRY RUN] Would run: {cmd_str}")
        return True

    logger.info(f"Running: {description}")
    logger.debug(f"Command: {cmd_str}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False


def fetch_overture_for_region(region_name: str, region_config: dict, dry_run: bool = False) -> bool:
    """Fetch Overture data for a region."""
    bbox = region_config["bbox"]
    cmd = [
        "matcher",
        "fetch",
        "--bbox",
        bbox,
        "-d",
        "overture",
        "-o",
        str(DATA_DIR),
        "--name",
        region_name,  # e.g., boston_overture_segments.parquet
    ]
    return run_command(cmd, f"Fetching Overture data for {region_name}", dry_run)


def fetch_osm_for_region(region_name: str, bbox: str, dry_run: bool = False) -> bool:
    """Fetch OSM data for a region."""
    cmd = [
        "matcher",
        "fetch",
        "--bbox",
        bbox,
        "-d",
        "osm",
        "-o",
        str(DATA_DIR),
        "--name",
        region_name,  # e.g., boston_osm_segments.parquet
    ]
    return run_command(cmd, f"Fetching OSM data for {region_name}", dry_run)


def fetch_local_datasets(fetch_scripts: list[str], dry_run: bool = False) -> bool:
    """Run fetch scripts for local datasets."""
    success = True
    for script in fetch_scripts:
        script_path = SCRIPTS_DIR / script
        if not script_path.exists():
            logger.warning(f"Fetch script not found: {script_path}")
            continue

        cmd = ["python", str(script_path)]
        if not run_command(cmd, f"Fetching local data via {script}", dry_run):
            success = False
    return success


def fetch_all_data(datasets: list[str] | None = None, dry_run: bool = False) -> datetime | None:
    """Fetch fresh data for all datasets."""
    data_pull_date = datetime.now(UTC)

    # Determine which regions to fetch
    regions_to_fetch = set()
    osm_regions = set()

    if datasets:
        for ds_name in datasets:
            if ds_name not in LABELED_DATASETS:
                logger.warning(f"Unknown dataset: {ds_name}")
                continue
            config = LABELED_DATASETS[ds_name]
            regions_to_fetch.add(config["region"])
            if config.get("osm_fetch"):
                osm_regions.add(config["region"])
    else:
        regions_to_fetch = set(REGIONS.keys())
        for _ds_name, config in LABELED_DATASETS.items():
            if config.get("osm_fetch"):
                osm_regions.add(config["region"])

    logger.info(f"Fetching data for regions: {sorted(regions_to_fetch)}")

    for region_name in sorted(regions_to_fetch):
        region_config = REGIONS[region_name]

        if not fetch_overture_for_region(region_name, region_config, dry_run):
            logger.error(f"Failed to fetch Overture data for {region_name}")
            return None

        if not fetch_local_datasets(region_config["fetch_scripts"], dry_run):
            logger.warning(f"Some local fetches failed for {region_name}")

        if region_name in osm_regions:
            if not fetch_osm_for_region(region_name, region_config["bbox"], dry_run):
                logger.error(f"Failed to fetch OSM data for {region_name}")
                return None

    return data_pull_date


def run_backfill(datasets: list[str] | None = None, dry_run: bool = False) -> bool:
    """Run feature backfill for labeled datasets."""
    cmd = ["python", str(SCRIPTS_DIR / "backfill_features.py")]

    if datasets and len(datasets) == 1:
        cmd.extend(["--dataset", datasets[0]])

    if dry_run:
        cmd.append("--dry-run")

    return run_command(cmd, "Backfilling features on labels", dry_run=False)


def train_and_evaluate(
    train_size: float = DEFAULT_TRAIN_SIZE,
    dry_run: bool = False,
) -> dict | None:
    """Train model on train set and evaluate on test set.

    This does a proper train/test split BEFORE training, so the test set
    is completely unseen during training.

    Args:
        train_size: Fraction of data to use for training (default: 0.7)
        dry_run: If True, just log what would be done

    Returns:
        Dictionary with results per dataset and metadata, or None on failure
    """
    if dry_run:
        test_pct = int((1 - train_size) * 100)
        logger.info(
            f"[DRY RUN] Would train on {int(train_size * 100)}% and evaluate on {test_pct}% holdout"
        )
        return {"_meta": {"train_size": train_size, "test_size": 1 - train_size}}

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    from matcher.labeling.label_store import LabelStore
    from matcher.matching.ml import MLMatcher

    # Load all labels
    logger.info("Loading all labels...")
    all_labels = LabelStore.load_all(LABELS_DIR)
    logger.info(f"Total labels: {len(all_labels)}")

    # Filter to valid labels only
    valid_labels = {"match", "no_match"}
    all_labels = all_labels[all_labels["label"].isin(valid_labels)].copy()
    logger.info(f"Valid labels (match/no_match): {len(all_labels)}")

    # Split into train/test with stratification
    train_df, test_df = train_test_split(
        all_labels,
        train_size=train_size,
        random_state=SPLIT_SEED,
        stratify=all_labels["label"],
    )

    test_pct = int((1 - train_size) * 100)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)} ({test_pct}% holdout)")
    logger.info(f"Train labels: {train_df['label'].value_counts().to_dict()}")
    logger.info(f"Test labels: {test_df['label'].value_counts().to_dict()}")

    # Save train labels to temp directory for training
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save each dataset's train portion
        for dataset in train_df["dataset"].unique():
            ds_train = train_df[train_df["dataset"] == dataset]
            ds_dir = tmpdir / f"dataset={dataset}"
            ds_dir.mkdir(parents=True, exist_ok=True)
            ds_train.to_csv(ds_dir / "data.csv", index=False)

        # Train model on train set only
        logger.info(f"\nTraining model on {len(train_df)} samples...")
        matcher = MLMatcher()
        matcher.train(labels_dir=str(tmpdir), binary=True, test_size=0.2)

        # Save the model
        model_path = MODELS_DIR / "matcher_model_combined.joblib"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        matcher.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Evaluate on test set (completely unseen during training)
        logger.info(f"\nEvaluating on {len(test_df)} holdout samples...")

        X_test, y_test = matcher._extract_features_and_labels(test_df, binary=True)
        X_test = matcher._impute_missing(X_test)
        y_pred = matcher.model.predict(X_test)

        # Overall metrics
        overall_acc = accuracy_score(y_test, y_pred)
        overall_f1 = f1_score(y_test, y_pred, average="weighted")
        overall_precision = precision_score(y_test, y_pred, average="weighted")
        overall_recall = recall_score(y_test, y_pred, average="weighted")

        print(f"\n{'=' * 60}")
        print(f"EVALUATION ON {test_pct}% HOLDOUT ({len(test_df)} samples)")
        print("=" * 60)
        print("\nOverall:")
        print(f"  Accuracy:  {overall_acc:.3f}")
        print(f"  F1:        {overall_f1:.3f}")
        print(f"  Precision: {overall_precision:.3f}")
        print(f"  Recall:    {overall_recall:.3f}")

        # Per-dataset metrics
        results = {
            "_meta": {
                "train_size": train_size,
                "test_size": 1 - train_size,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "split_seed": SPLIT_SEED,
            },
            "_overall": {
                "n_samples": len(test_df),
                "n_match": int((y_test == 1).sum()),
                "n_no_match": int((y_test == 0).sum()),
                "accuracy": overall_acc,
                "f1": overall_f1,
                "precision": overall_precision,
                "recall": overall_recall,
            },
        }

        print("\nPer-dataset results:")
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

            print(
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

        return results


def save_results(
    results: dict,
    data_pull_date: datetime,
    run_date: datetime,
    model_path: str,
    train_size: float,
    dry_run: bool = False,
) -> None:
    """Save benchmark results to CSV."""
    if dry_run:
        logger.info(f"[DRY RUN] Would save results to {RESULTS_FILE}")
        return

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

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
        "model_path",
    ]

    write_header = not RESULTS_FILE.exists()

    meta = results.get("_meta", {})

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for dataset_name, metrics in results.items():
            if dataset_name.startswith("_"):
                continue

            row = {
                "run_date": run_date.isoformat(),
                "data_pull_date": data_pull_date.isoformat(),
                "dataset": dataset_name,
                "n_train": meta.get("n_train", 0),
                "n_test": meta.get("n_test", 0),
                "train_size": train_size,
                "n_samples": metrics.get("n_samples", 0),
                "n_match": metrics.get("n_match", 0),
                "n_no_match": metrics.get("n_no_match", 0),
                "accuracy": f"{metrics.get('accuracy', 0):.4f}",
                "f1": f"{metrics.get('f1', 0):.4f}",
                "precision": f"{metrics.get('precision', 0):.4f}",
                "recall": f"{metrics.get('recall', 0):.4f}",
                "split_seed": SPLIT_SEED,
                "model_path": model_path,
            }
            writer.writerow(row)

    logger.success(f"Saved benchmark results to {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ML model performance on labeled datasets"
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch data (skip backfill, train, eval)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate existing model (skip fetch, backfill, train)",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="Skip feature backfill",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetching",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process specific dataset only",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=DEFAULT_TRAIN_SIZE,
        help=f"Fraction of data for training (default: {DEFAULT_TRAIN_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    args = parser.parse_args()

    # Validate train size
    if not 0.1 <= args.train_size <= 0.9:
        logger.error("--train-size must be between 0.1 and 0.9")
        return 1

    # Parse dataset filter
    datasets = [args.dataset] if args.dataset else None

    # Validate dataset name
    if args.dataset and args.dataset not in LABELED_DATASETS:
        logger.error(f"Unknown dataset: {args.dataset}")
        logger.info(f"Available datasets: {', '.join(LABELED_DATASETS.keys())}")
        return 1

    # Record timestamps
    run_date = datetime.now(UTC)
    data_pull_date = run_date

    test_pct = int((1 - args.train_size) * 100)
    train_pct = int(args.train_size * 100)

    logger.info("=" * 60)
    logger.info("BENCHMARK WORKFLOW")
    logger.info("=" * 60)
    logger.info(f"Run date: {run_date.isoformat()}")
    logger.info(f"Train/Test split: {train_pct}/{test_pct}")
    if datasets:
        logger.info(f"Datasets: {', '.join(datasets)}")
    else:
        logger.info(f"Datasets: all ({len(LABELED_DATASETS)} total)")

    # Step 1: Fetch data
    if not args.eval_only and not args.skip_fetch:
        logger.info("\n--- Step 1: Fetching data ---")
        fetch_result = fetch_all_data(datasets, args.dry_run)
        if fetch_result:
            data_pull_date = fetch_result
        else:
            logger.error("Data fetch failed")
            return 1

        if args.fetch_only:
            logger.info("Fetch-only mode, stopping here")
            return 0

    # Step 2: Backfill features
    if not args.eval_only and not args.skip_backfill:
        logger.info("\n--- Step 2: Backfilling features ---")
        if not run_backfill(datasets, args.dry_run):
            logger.error("Feature backfill failed")
            return 1

    # Step 3 & 4: Train and Evaluate with proper split
    logger.info("\n--- Step 3: Train and Evaluate ---")
    results = train_and_evaluate(
        train_size=args.train_size,
        dry_run=args.dry_run,
    )
    if results is None:
        logger.error("Train/Evaluate failed")
        return 1

    # Step 5: Save results
    logger.info("\n--- Step 4: Saving results ---")
    model_path = str(MODELS_DIR / "matcher_model_combined.joblib")
    save_results(results, data_pull_date, run_date, model_path, args.train_size, args.dry_run)

    logger.info("\n" + "=" * 60)
    logger.success("Benchmark complete!")
    logger.info(f"Results saved to: {RESULTS_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
