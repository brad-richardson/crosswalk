#!/usr/bin/env python3
"""Migrate label files from old ID format to H3-suffixed IDs.

Old format: {prefix}_{upstreamID}        e.g. us_boston_streets_10067
New format: {prefix}_{upstreamID}_{h3sfx} e.g. us_boston_streets_10067_882a306603

The H3 suffix is computed from each segment's geometry using the same
compute_spatial_suffix() function used by the fetch pipeline.

This script:
1. Loads each target parquet (has old-format IDs and geometries)
2. Computes old_id → new_id mapping via H3 spatial suffix
3. Updates target_id in: human CSVs, agent CSVs, features parquets, data parquets
4. Creates .bak backups before modifying any file

Usage:
    python scripts/migrate_to_h3_ids.py --dry-run     # Preview changes
    python scripts/migrate_to_h3_ids.py                # Run migration
    python scripts/migrate_to_h3_ids.py --no-backup    # Skip backups (not recommended)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

# Add project root to path so we can import matcher modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matcher.utils.spatial_id import compute_spatial_suffix


def build_id_mapping(target_parquet: Path) -> dict[str, str]:
    """Build old_id → new_id mapping from a target parquet file.

    For each row, computes the H3 spatial suffix from the geometry
    and appends it to the existing ID.

    Returns:
        Dict mapping old IDs to new H3-suffixed IDs.
        Only includes entries where the ID actually changes
        (i.e., doesn't already have a suffix).
    """
    gdf = gpd.read_parquet(target_parquet)
    mapping = {}
    conflicts = 0

    for _idx, row in gdf.iterrows():
        old_id = row["id"]
        geom = row.geometry

        if geom is None or geom.is_empty:
            logger.warning(f"Skipping {old_id}: no geometry")
            continue

        suffix = compute_spatial_suffix(geom)
        new_id = f"{old_id}_{suffix}"

        # Skip if the ID already has a suffix (idempotent)
        if old_id == new_id or old_id.endswith(f"_{suffix}"):
            continue

        # Warn on duplicate old_id with different new_id (data quality issue)
        if old_id in mapping and mapping[old_id] != new_id:
            conflicts += 1
            if conflicts <= 5:
                logger.warning(
                    f"Duplicate old_id '{old_id}' maps to different new IDs: "
                    f"'{mapping[old_id]}' vs '{new_id}' (keeping first)"
                )
            continue

        mapping[old_id] = new_id

    if conflicts:
        logger.warning(f"{conflicts} duplicate old_id(s) with conflicting new IDs (kept first)")

    return mapping


def migrate_csv(csv_path: Path, mapping: dict[str, str], dry_run: bool, backup: bool) -> int:
    """Migrate target_id column in a CSV label file.

    Returns number of IDs updated.
    """
    df = pd.read_csv(csv_path)
    if "target_id" not in df.columns:
        logger.warning(f"No target_id column in {csv_path}")
        return 0

    updated = df["target_id"].isin(mapping)
    count = updated.sum()

    if count == 0:
        return 0

    if dry_run:
        logger.info(f"  Would update {count}/{len(df)} target_ids in {csv_path}")
        return count

    if backup:
        shutil.copy2(csv_path, csv_path.with_suffix(csv_path.suffix + ".bak"))

    df.loc[updated, "target_id"] = df.loc[updated, "target_id"].map(mapping)
    df.to_csv(csv_path, index=False)
    logger.info(f"  Updated {count}/{len(df)} target_ids in {csv_path}")
    return count


def migrate_parquet(
    parquet_path: Path, mapping: dict[str, str], dry_run: bool, backup: bool
) -> int:
    """Migrate target_id column in a parquet file.

    Returns number of IDs updated.
    """
    df = pd.read_parquet(parquet_path)
    if "target_id" not in df.columns:
        logger.warning(f"No target_id column in {parquet_path}")
        return 0

    updated = df["target_id"].isin(mapping)
    count = updated.sum()

    if count == 0:
        return 0

    if dry_run:
        logger.info(f"  Would update {count}/{len(df)} target_ids in {parquet_path}")
        return count

    if backup:
        shutil.copy2(parquet_path, parquet_path.with_suffix(parquet_path.suffix + ".bak"))

    df.loc[updated, "target_id"] = df.loc[updated, "target_id"].map(mapping)
    df.to_parquet(parquet_path, index=False)
    logger.info(f"  Updated {count}/{len(df)} target_ids in {parquet_path}")
    return count


def migrate_target_parquet(
    parquet_path: Path, mapping: dict[str, str], dry_run: bool, backup: bool
) -> int:
    """Migrate id column in a target data parquet (the source of truth).

    This updates the 'id' column (not 'target_id') in the raw target data.

    Returns number of IDs updated.
    """
    gdf = gpd.read_parquet(parquet_path)
    if "id" not in gdf.columns:
        logger.warning(f"No id column in {parquet_path}")
        return 0

    updated = gdf["id"].isin(mapping)
    count = updated.sum()

    if count == 0:
        return 0

    if dry_run:
        logger.info(f"  Would update {count}/{len(gdf)} IDs in {parquet_path}")
        return count

    if backup:
        shutil.copy2(parquet_path, parquet_path.with_suffix(parquet_path.suffix + ".bak"))

    gdf.loc[updated, "id"] = gdf.loc[updated, "id"].map(mapping)
    gdf.to_parquet(parquet_path, index=False)
    logger.info(f"  Updated {count}/{len(gdf)} IDs in {parquet_path}")
    return count


def find_datasets_with_labels(labels_dir: Path) -> set[str]:
    """Find dataset names that have label data."""
    datasets = set()
    for subdir in ["human", "agent", "features", "data"]:
        dir_path = labels_dir / subdir
        if dir_path.exists():
            for dataset_dir in dir_path.iterdir():
                if dataset_dir.is_dir() and dataset_dir.name.startswith("dataset="):
                    dataset_name = dataset_dir.name.replace("dataset=", "")
                    datasets.add(dataset_name)
    return datasets


def find_target_parquet(dataset_name: str, data_dir: Path) -> Path | None:
    """Find the target parquet file for a dataset."""
    # Try versioned pattern first: {dataset_name}_v*.parquet
    candidates = sorted(data_dir.glob(f"{dataset_name}_v*.parquet"), reverse=True)
    if candidates:
        return candidates[0]

    # Try exact match
    exact = data_dir / f"{dataset_name}.parquet"
    if exact.exists():
        return exact

    return None


def main():
    parser = argparse.ArgumentParser(description="Migrate label IDs to H3-suffixed format")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating .bak backups")
    parser.add_argument("--labels-dir", type=Path, default=Path("labels"), help="Labels directory")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw"), help="Raw data directory"
    )
    parser.add_argument("--dataset", type=str, help="Migrate only this dataset")
    args = parser.parse_args()

    backup = not args.no_backup
    labels_dir = args.labels_dir
    data_dir = args.data_dir

    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Find datasets that need migration
    if args.dataset:
        datasets = {args.dataset}
    else:
        datasets = find_datasets_with_labels(labels_dir)

    if not datasets:
        logger.info("No datasets with labels found")
        return

    logger.info(f"Found {len(datasets)} datasets with labels")

    total_updated = 0
    total_skipped = 0

    for dataset_name in sorted(datasets):
        logger.info(f"\n--- {dataset_name} ---")

        # Find target parquet to build ID mapping
        target_path = find_target_parquet(dataset_name, data_dir)
        if target_path is None:
            logger.warning(f"  No target parquet found for {dataset_name} in {data_dir}, skipping")
            total_skipped += 1
            continue

        # Build old → new ID mapping
        mapping = build_id_mapping(target_path)
        if not mapping:
            logger.info("  No IDs need migration (already migrated or no geometry)")
            continue

        logger.info(f"  Built mapping for {len(mapping)} IDs")

        # Show a few examples
        examples = list(mapping.items())[:3]
        for old_id, new_id in examples:
            logger.info(f"    {old_id} → {new_id}")
        if len(mapping) > 3:
            logger.info(f"    ... and {len(mapping) - 3} more")

        dataset_updated = 0

        # Migrate human labels
        human_csv = labels_dir / "human" / f"dataset={dataset_name}" / "data.csv"
        if human_csv.exists():
            dataset_updated += migrate_csv(human_csv, mapping, args.dry_run, backup)

        # Migrate agent labels
        agent_csv = labels_dir / "agent" / f"dataset={dataset_name}" / "data.csv"
        if agent_csv.exists():
            dataset_updated += migrate_csv(agent_csv, mapping, args.dry_run, backup)

        # Migrate features parquet
        features_pq = labels_dir / "features" / f"dataset={dataset_name}" / "data.parquet"
        if features_pq.exists():
            dataset_updated += migrate_parquet(features_pq, mapping, args.dry_run, backup)

        # Migrate data parquet (stored pair data)
        data_pq = labels_dir / "data" / f"dataset={dataset_name}" / "data.parquet"
        if data_pq.exists():
            dataset_updated += migrate_parquet(data_pq, mapping, args.dry_run, backup)

        # Migrate the raw target parquet itself
        dataset_updated += migrate_target_parquet(target_path, mapping, args.dry_run, backup)

        total_updated += dataset_updated

    action = "Would update" if args.dry_run else "Updated"
    logger.info(f"\n{action} {total_updated} total ID references across all files")
    if total_skipped:
        logger.info(f"Skipped {total_skipped} datasets (no target parquet found)")

    if args.dry_run:
        logger.info("\nRe-run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
