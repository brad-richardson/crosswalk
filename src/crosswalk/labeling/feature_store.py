"""Feature persistence - Parquet storage for computed features.

This module provides the FeatureStore class which stores computed features
keyed by (gers_id, target_id). Features are versioned to track which
feature computation logic was used.
"""

from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ..config import FEATURE_COLUMNS, FEATURE_VERSION

# Default directory for feature storage
DEFAULT_FEATURES_DIR = Path("labels/features")

# Key columns (not features, but needed for joins)
FEATURE_KEY_COLUMNS = [
    "gers_id",
    "target_id",
    "feature_version",
]

# These keys define the canonical order of bulk-loaded training rows.  Sorting
# after concatenation makes model splits independent of filesystem iteration
# order while preserving the order of exact duplicate rows (mergesort is
# stable).
_BULK_SORT_COLUMNS = ["dataset", "gers_id", "target_id", "feature_version"]


def _sort_bulk_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return bulk-loaded rows in a stable, canonical order."""
    sort_columns = [column for column in _BULK_SORT_COLUMNS if column in df.columns]
    if not sort_columns or df.empty:
        return df.reset_index(drop=True)
    return df.sort_values(
        sort_columns,
        kind="mergesort",
        na_position="last",
        key=lambda values: values.astype("string"),
    ).reset_index(drop=True)


@dataclass
class FeatureStore:
    """Manages computed features in Parquet format.

    Features are keyed by (gers_id, target_id) composite key with
    feature_version tracking. This allows features to be recomputed
    independently of label metadata and raw data.

    Storage format:
        labels/features/dataset={dataset_id}/data.parquet

    The Parquet format is efficient for 56+ float columns and allows
    fast reads during training.
    """

    dataset_id: str
    features_dir: Path = DEFAULT_FEATURES_DIR
    _df: pd.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self):
        self.features_dir = Path(self.features_dir)
        self.partition_path = self.features_dir / f"dataset={self.dataset_id}"
        self.parquet_path = self.partition_path / "data.parquet"
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load DataFrame."""
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        """Load features from Parquet, returning empty DataFrame if file doesn't exist."""
        if not self.parquet_path.exists():
            return self._empty_dataframe()

        try:
            df = pd.read_parquet(self.parquet_path)
            # Ensure all expected columns exist
            for col in FEATURE_KEY_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            for col in FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = float("nan")
            return df
        except Exception as e:
            logger.warning(f"Failed to load feature store from {self.parquet_path}: {e}")
            return self._empty_dataframe()

    def _empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with correct schema."""
        columns = FEATURE_KEY_COLUMNS + FEATURE_COLUMNS
        df = pd.DataFrame(columns=columns)
        # Set appropriate dtypes
        df["gers_id"] = df["gers_id"].astype("object")
        df["target_id"] = df["target_id"].astype("object")
        df["feature_version"] = df["feature_version"].astype("object")
        for col in FEATURE_COLUMNS:
            df[col] = df[col].astype("float64")
        return df

    def add(
        self,
        gers_id: str,
        target_id: str,
        features: dict[str, float],
        feature_version: str | None = None,
    ) -> None:
        """Add or update features for a labeled pair.

        Deduplicates on (gers_id, target_id) composite key, keeping the latest.

        Args:
            gers_id: Overture reference segment ID
            target_id: Target segment ID
            features: Dict of feature name -> value (from compute_pair_features)
            feature_version: Feature computation version (defaults to FEATURE_VERSION)
        """
        if feature_version is None:
            feature_version = FEATURE_VERSION

        new_row = {
            "gers_id": str(gers_id),
            "target_id": str(target_id),
            "feature_version": feature_version,
        }

        # Add all feature columns
        for col in FEATURE_COLUMNS:
            new_row[col] = features.get(col, float("nan"))

        df = self.df

        # Remove existing entry for this pair (dedup on composite key)
        mask = (df["gers_id"] == str(gers_id)) & (df["target_id"] == str(target_id))
        if mask.any():
            df = df[~mask]

        self._df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    def get(self, gers_id: str, target_id: str) -> dict[str, Any] | None:
        """Get features for a labeled pair.

        Args:
            gers_id: Overture reference segment ID
            target_id: Target segment ID

        Returns:
            Dict with feature_version and all feature values, or None if not found.
        """
        df = self.df
        mask = (df["gers_id"] == str(gers_id)) & (df["target_id"] == str(target_id))
        matches = df[mask]

        if len(matches) == 0:
            return None

        row = matches.iloc[-1]  # Latest entry

        result = {
            "gers_id": row["gers_id"],
            "target_id": row["target_id"],
            "feature_version": row.get("feature_version"),
        }

        for col in FEATURE_COLUMNS:
            result[col] = row.get(col)

        return result

    def has_pair(self, gers_id: str, target_id: str) -> bool:
        """Check if features exist for a labeled pair."""
        df = self.df
        mask = (df["gers_id"] == str(gers_id)) & (df["target_id"] == str(target_id))
        return mask.any()

    def delete_pair(self, gers_id: str, target_id: str) -> bool:
        """Delete features for a labeled pair.

        Args:
            gers_id: Overture reference segment ID
            target_id: Target segment ID

        Returns:
            True if found and deleted, False if pair not found.
        """
        df = self.df
        mask = (df["gers_id"] == str(gers_id)) & (df["target_id"] == str(target_id))
        if not mask.any():
            return False
        self._df = df[~mask].reset_index(drop=True)
        self.save()
        return True

    def save(self) -> None:
        """Save features to Parquet atomically with backup.

        Uses write-to-temp-then-rename pattern for atomic writes.
        """
        self.partition_path.mkdir(parents=True, exist_ok=True)

        temp_path = self.parquet_path.with_suffix(".parquet.tmp")
        backup_path = self.parquet_path.with_suffix(".parquet.bak")

        # Write to temp file first
        self._df.to_parquet(temp_path, index=False, compression="zstd")

        # Backup existing file (if present)
        if self.parquet_path.exists():
            self.parquet_path.replace(backup_path)

        # Atomic replace temp to final
        temp_path.replace(self.parquet_path)

    @staticmethod
    def load_all(
        features_dir: Path = DEFAULT_FEATURES_DIR,
        skip_errors: bool = True,
        required_datasets: Collection[str] | None = None,
    ) -> pd.DataFrame:
        """Load all feature partitions.

        Uses Hive partitioning to read all dataset partitions
        and adds a 'dataset' column from the partition path.

        Args:
            features_dir: Directory containing Hive-partitioned feature parquets
            skip_errors: If True (default), warn and skip malformed partitions.
                If False, fail when a declared partition is missing its parquet
                file or cannot be read. Training and tuning should pass False.
            required_datasets: If provided, load only these dataset partitions.
                In strict mode every requested partition must exist and load;
                unrelated archival/orphan directories are ignored.

        Returns:
            DataFrame with all features and 'dataset' column
        """
        features_dir = Path(features_dir)
        columns = FEATURE_KEY_COLUMNS + FEATURE_COLUMNS + ["dataset"]

        if not features_dir.exists():
            if not skip_errors and required_datasets:
                raise FileNotFoundError(f"Feature partition directory is missing: {features_dir}")
            return pd.DataFrame(columns=columns)

        dfs = []

        if required_datasets is None:
            partition_dirs = sorted(features_dir.glob("dataset=*"), key=lambda path: path.name)
        else:
            partition_dirs = [
                features_dir / f"dataset={dataset_id}"
                for dataset_id in sorted({str(value) for value in required_datasets})
            ]
        for partition_dir in partition_dirs:
            if not partition_dir.is_dir():
                if not skip_errors:
                    raise FileNotFoundError(
                        f"Feature partition is missing or is not a directory: {partition_dir}"
                    )
                continue
            dataset_id = partition_dir.name.removeprefix("dataset=")
            parquet_path = partition_dir / "data.parquet"
            if not parquet_path.exists():
                if skip_errors:
                    logger.warning(f"Feature partition has no data.parquet: {partition_dir}")
                    continue
                raise FileNotFoundError(f"Feature partition has no data.parquet: {partition_dir}")

            try:
                df = pd.read_parquet(parquet_path)
                df["dataset"] = dataset_id
                dfs.append(df)
            except Exception as e:
                if skip_errors:
                    logger.warning(f"Failed to load {parquet_path}: {e}")
                else:
                    raise

        if dfs:
            return _sort_bulk_rows(pd.concat(dfs, ignore_index=True))
        return pd.DataFrame(columns=columns)
