#!/usr/bin/env python
"""Migrate labels from parquet to Hive-partitioned CSV format.

This script converts the existing label storage from:
    data/labels/labels_*.parquet -> labels/dataset=*/data.csv
    data/labels/integration_*.parquet -> data/labels/integration_*.csv

It also creates the datasets.csv registry at data/datasets.csv.

Usage:
    python scripts/migrate_labels_to_csv.py
    python scripts/migrate_labels_to_csv.py --dry-run  # Preview changes
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


def migrate_labels_to_csv(labels_dir: Path, dry_run: bool = False) -> None:
    """Migrate label parquet files to Hive-partitioned CSVs.

    Args:
        labels_dir: Path to data/labels directory
        dry_run: If True, only print what would be done
    """
    logger.info(f"Migrating labels in {labels_dir}")

    # Find all label parquet files
    label_files = list(labels_dir.glob("labels_*.parquet"))
    logger.info(f"Found {len(label_files)} label parquet files")

    if not label_files:
        logger.warning("No label files found to migrate")
        return

    # Create new labels directory structure
    new_labels_dir = labels_dir
    if not dry_run:
        new_labels_dir.mkdir(parents=True, exist_ok=True)

    # Dataset registry entries
    datasets = []

    for pq_file in sorted(label_files):
        # Extract dataset_id from filename (e.g., labels_boston_streets.parquet -> boston_streets)
        dataset_id = pq_file.stem.replace("labels_", "")

        # Skip temp/combined files
        if dataset_id.startswith("_"):
            logger.info(f"  Skipping temp file: {pq_file.name}")
            continue

        # Load parquet
        df = pd.read_parquet(pq_file)
        logger.info(f"  {dataset_id}: {len(df)} rows")

        if dry_run:
            logger.info(f"    Would create: labels/dataset={dataset_id}/data.csv")
            continue

        # Rename ref_id -> gers_id for consistency
        if "ref_id" in df.columns and "gers_id" not in df.columns:
            df = df.rename(columns={"ref_id": "gers_id"})

        # Drop the features dict column if present (features are in individual columns now)
        if "features" in df.columns:
            # Check if features are in a dict column - if so we need to expand them
            first_features = df["features"].iloc[0] if len(df) > 0 else None
            if first_features and isinstance(first_features, dict):
                # Expand features dict to columns
                feature_df = pd.json_normalize(df["features"])
                # Only add columns that don't already exist
                for col in feature_df.columns:
                    if col not in df.columns:
                        df[col] = feature_df[col]
            # Drop the features dict column
            df = df.drop(columns=["features"])

        # Create partition directory
        partition_dir = new_labels_dir / f"dataset={dataset_id}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV (dataset column is implicit from partition path)
        csv_path = partition_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"    Created: {csv_path}")

        # Add to registry
        datasets.append({
            "dataset_id": dataset_id,
            "name": dataset_id.replace("_", " ").title(),
            "type": infer_dataset_type(dataset_id),
            "fetch_url": "",
            "info_url": "",
            "metadata": "{}",
        })

    # Create datasets.csv registry
    if datasets:
        registry_path = labels_dir.parent / "datasets.csv"
        if dry_run:
            logger.info(f"Would create: {registry_path} with {len(datasets)} datasets")
        else:
            pd.DataFrame(datasets).to_csv(registry_path, index=False)
            logger.info(f"Created: {registry_path}")


def migrate_integration_files(labels_dir: Path, dry_run: bool = False) -> None:
    """Migrate integration QA parquet files to CSV.

    Args:
        labels_dir: Path to data/labels directory
        dry_run: If True, only print what would be done
    """
    logger.info("Migrating integration QA files")

    for name in ["integration_orphans", "integration_merged"]:
        pq_path = labels_dir / f"{name}.parquet"
        if not pq_path.exists():
            logger.info(f"  {name}.parquet not found, skipping")
            continue

        df = pd.read_parquet(pq_path)
        logger.info(f"  {name}: {len(df)} rows")

        if dry_run:
            logger.info(f"    Would create: {name}.csv")
            continue

        # Rename source_dataset -> dataset_id for consistency
        if "source_dataset" in df.columns and "dataset_id" not in df.columns:
            df = df.rename(columns={"source_dataset": "dataset_id"})

        csv_path = labels_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"    Created: {csv_path}")


def infer_dataset_type(dataset_id: str) -> str:
    """Infer dataset type from dataset_id."""
    if "bike" in dataset_id.lower():
        return "bike"
    elif "sidewalk" in dataset_id.lower():
        return "sidewalk"
    elif "transit" in dataset_id.lower():
        return "transit"
    else:
        return "road"


def main():
    parser = argparse.ArgumentParser(description="Migrate labels to CSV format")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/labels"),
        help="Path to labels directory",
    )
    args = parser.parse_args()

    labels_dir = args.labels_dir
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return

    logger.info("=" * 60)
    logger.info("LABEL MIGRATION: Parquet -> Hive-partitioned CSV")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN - no files will be written")
        logger.info("")

    migrate_labels_to_csv(labels_dir, dry_run=args.dry_run)
    migrate_integration_files(labels_dir, dry_run=args.dry_run)

    logger.info("")
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("DRY RUN complete - run without --dry-run to apply changes")
    else:
        logger.info("Migration complete!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Verify CSV files were created correctly")
        logger.info("  2. Test with: python -c 'from matcher.labeling.label_store import LabelStore; print(LabelStore.load_all())'")
        logger.info("  3. If all looks good, delete old parquet files:")
        logger.info("     rm data/labels/labels_*.parquet")
        logger.info("     rm data/labels/integration_*.parquet")
        logger.info("  4. Commit the new CSV structure")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
