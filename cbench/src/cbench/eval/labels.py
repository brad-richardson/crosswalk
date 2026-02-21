"""Ground truth label loading for cbench."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def load_labels(
    labels_path: Path,
    dataset: str,
    ref_id_column: str = "gers_id",
    target_id_column: str = "target_id",
) -> pd.DataFrame:
    """Load ground truth labels for a dataset.

    Reads hive-partitioned labels from {labels_path}/dataset={dataset}/data.csv
    (or data.parquet), renames ID columns to ref_id/target_id.

    Args:
        labels_path: Root labels directory (e.g., labels/human).
        dataset: Dataset name (e.g., "us_boston_streets").
        ref_id_column: Column name for reference IDs in the source file.
        target_id_column: Column name for target IDs in the source file.

    Returns:
        DataFrame with columns [ref_id, target_id, label].
    """
    dataset_dir = labels_path / f"dataset={dataset}"

    # Try CSV first, then parquet
    csv_path = dataset_dir / "data.csv"
    parquet_path = dataset_dir / "data.parquet"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} labels from {csv_path}")
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} labels from {parquet_path}")
    else:
        available = list_datasets(labels_path)
        logger.warning(f"No labels found for '{dataset}' in {labels_path}")
        if available:
            logger.info(f"Available datasets: {list(available.keys())}")
        raise FileNotFoundError(f"No labels found for dataset '{dataset}' in {labels_path}")

    # Rename to generic columns
    rename = {}
    if ref_id_column != "ref_id" and ref_id_column in df.columns:
        rename[ref_id_column] = "ref_id"
    if target_id_column != "target_id" and target_id_column in df.columns:
        rename[target_id_column] = "target_id"
    if rename:
        df = df.rename(columns=rename)

    # Validate required columns
    required = {"ref_id", "target_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels file missing required columns: {missing}")

    return df[["ref_id", "target_id", "label"]]


def load_stitch_labels(
    labels_path: Path,
    dataset: str,
) -> pd.DataFrame | None:
    """Load stitching labels for a dataset, if they exist.

    Reads from {labels_path}/dataset={dataset}/data.csv.
    Returns None if no stitching labels exist for this dataset.

    Args:
        labels_path: Root stitching labels directory.
        dataset: Dataset name.

    Returns:
        DataFrame with stitching label columns, or None if not found.
    """
    csv_path = labels_path / f"dataset={dataset}" / "data.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} stitch labels from {csv_path}")

    required = {"group_id", "selected_edges"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(f"Stitch labels missing required columns: {missing}")
        return None

    return df


def list_datasets(labels_path: Path) -> dict[str, int]:
    """List available datasets with their label counts.

    Args:
        labels_path: Root labels directory.

    Returns:
        Dictionary mapping dataset name to label count.
    """
    datasets: dict[str, int] = {}

    if not labels_path.exists():
        return datasets

    for d in sorted(labels_path.glob("dataset=*")):
        name = d.name.replace("dataset=", "")

        csv_path = d / "data.csv"
        if csv_path.exists():
            try:
                datasets[name] = len(pd.read_csv(csv_path))
            except Exception as exc:
                logger.warning(f"Failed to read {csv_path}: {exc}")
            continue

        parquet_path = d / "data.parquet"
        if parquet_path.exists():
            try:
                datasets[name] = len(pd.read_parquet(parquet_path))
            except Exception as exc:
                logger.warning(f"Failed to read {parquet_path}: {exc}")

    return datasets
