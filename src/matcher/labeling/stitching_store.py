"""Stitching label persistence - CSV-backed storage for group-level review decisions.

Follows the LabelStore pattern: Hive-partitioned CSV with atomic writes,
composite key dedup, and backup recovery.

Storage: labels/stitching/dataset={id}/data.csv
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

DEFAULT_STITCHING_DIR = Path("labels/stitching")

STITCHING_LABEL_COLUMNS = [
    "group_id",
    "dataset_id",
    "selected_option_index",
    "selected_edges",  # JSON: [{ref_id, target_id, gers_start_frac, gers_end_frac, local_start_frac, local_end_frac}, ...]
    "match_type",
    "num_refs",
    "num_targets",
    "labeler",
    "labeled_at",
    "session_id",
]


@dataclass
class StitchingLabelStore:
    """Manages stitching review labels for a single dataset partition.

    CSV-backed with atomic writes and backup recovery, following the
    LabelStore pattern.
    """

    dataset_id: str
    labels_dir: Path = DEFAULT_STITCHING_DIR
    _df: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.labels_dir = Path(self.labels_dir)
        self.partition_path = self.labels_dir / f"dataset={self.dataset_id}"
        self.csv_path = self.partition_path / "data.csv"

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load dataframe."""
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        """Load labels from CSV with backup recovery.

        Raises if files exist but cannot be read (prevents silent data loss).
        Returns empty DataFrame only when no files exist.
        """
        backup_path = self.csv_path.with_suffix(".csv.bak")

        if self.csv_path.exists():
            try:
                return pd.read_csv(self.csv_path, dtype={"group_id": str})
            except Exception as primary_error:
                logger.warning(
                    f"Failed to load stitching labels from {self.csv_path}: {primary_error}"
                )
                if backup_path.exists():
                    try:
                        logger.info(f"Recovering from backup: {backup_path}")
                        return pd.read_csv(backup_path, dtype={"group_id": str})
                    except Exception as backup_error:
                        raise OSError(
                            f"Both primary and backup stitching label files are corrupted.\n"
                            f"Primary ({self.csv_path}): {primary_error}\n"
                            f"Backup ({backup_path}): {backup_error}"
                        ) from backup_error
                raise OSError(
                    f"Stitching label file is corrupted and no backup available: "
                    f"{self.csv_path}\nError: {primary_error}"
                ) from primary_error

        if backup_path.exists():
            try:
                return pd.read_csv(backup_path, dtype={"group_id": str})
            except Exception as e:
                raise OSError(
                    f"Backup stitching label file is corrupted: {backup_path}\nError: {e}"
                ) from e

        return pd.DataFrame(columns=STITCHING_LABEL_COLUMNS)

    def save(self) -> None:
        """Save labels to CSV atomically with backup."""
        self.partition_path.mkdir(parents=True, exist_ok=True)

        temp_path = self.csv_path.with_suffix(".csv.tmp")
        backup_path = self.csv_path.with_suffix(".csv.bak")

        self._df.to_csv(temp_path, index=False)

        if self.csv_path.exists():
            self.csv_path.replace(backup_path)

        temp_path.replace(self.csv_path)

    def add(
        self,
        group_id: str,
        selected_option_index: int,
        selected_edges: list[dict],
        match_type: str,
        num_refs: int,
        num_targets: int,
        labeler: str,
        session_id: str,
    ) -> None:
        """Add a stitching review label.

        If a label already exists for this group_id, it is replaced.
        Uses self.dataset_id for the dataset (partition path and stored column
        always match).

        Args:
            group_id: Deterministic group identifier
            selected_option_index: Which alternative was selected
            selected_edges: List of {ref_id, target_id} dicts
            match_type: "1:N", "N:1", or "M:N"
            num_refs: Number of reference segments in the group
            num_targets: Number of target segments in the group
            labeler: Name of the reviewer
            session_id: Session identifier
        """
        new_row = {
            "group_id": str(group_id),
            "dataset_id": self.dataset_id,
            "selected_option_index": selected_option_index,
            "selected_edges": json.dumps(selected_edges),
            "match_type": match_type,
            "num_refs": num_refs,
            "num_targets": num_targets,
            "labeler": labeler,
            "labeled_at": datetime.now(UTC).isoformat(),
            "session_id": session_id,
        }

        # Remove existing label for this group (re-review replaces)
        df = self.df
        if len(df) > 0:
            mask = df["group_id"] == str(group_id)
            if mask.any():
                self._df = df[~mask]

        self._df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.save()

    def get_reviewed_group_ids(self, dataset_id: str | None = None) -> set[str]:
        """Get set of already-reviewed group IDs.

        Args:
            dataset_id: If provided, filter to this dataset only.

        Returns:
            Set of group_id strings.
        """
        df = self.df
        if df.empty:
            return set()
        if dataset_id:
            df = df[df["dataset_id"] == dataset_id]
        return set(df["group_id"].astype(str))

    def load(self, dataset_id: str | None = None) -> pd.DataFrame:
        """Load all stitching labels, optionally filtered by dataset.

        Args:
            dataset_id: If provided, filter to this dataset.

        Returns:
            DataFrame with stitching labels.
        """
        df = self.df
        if dataset_id and not df.empty:
            df = df[df["dataset_id"] == dataset_id]
        return df

    @staticmethod
    def load_all(
        labels_dir: Path = DEFAULT_STITCHING_DIR,
    ) -> pd.DataFrame:
        """Load all stitching labels across all dataset partitions.

        Args:
            labels_dir: Base stitching labels directory

        Returns:
            DataFrame with all stitching labels and 'dataset' column
        """
        labels_dir = Path(labels_dir)
        if not labels_dir.exists():
            return pd.DataFrame(columns=STITCHING_LABEL_COLUMNS)

        dfs = []
        for partition_dir in labels_dir.glob("dataset=*"):
            if not partition_dir.is_dir():
                continue
            csv_path = partition_dir / "data.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, dtype={"group_id": str})
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {csv_path}: {e}")

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=STITCHING_LABEL_COLUMNS)
