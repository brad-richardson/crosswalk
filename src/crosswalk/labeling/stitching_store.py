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

# Label semantics (see docs/ARCHITECTURE.md "Stitching labels"):
#   "pair" (default) - selected_edges is the authoritative pair set the labeler
#       endorsed (explicit option ratifications; historical cross-product rows).
#   "set"            - the labeler asserted only group MEMBERSHIP: these refs and
#       these targets form one matched group. He did NOT adjudicate individual
#       pairings, so selected_edges is empty ("[]") and the membership lives in
#       ref_ids / target_ids. Eval scores these on membership/boundary/coverage,
#       not per-pair edge-F1.
LABEL_SEMANTICS_PAIR = "pair"
LABEL_SEMANTICS_SET = "set"

STITCHING_LABEL_COLUMNS = [
    "group_id",
    "dataset_id",
    "selected_edges",  # JSON: [{ref_id, target_id}, ...]  (empty for set rows)
    "match_type",
    "num_refs",
    "num_targets",
    "labeler",
    "labeled_at",
    "session_id",
    # Set-semantics columns (backwards-compatible: default to a pair label with
    # empty membership when absent). ref_ids / target_ids are JSON arrays of
    # segment ids, encoded like selected_edges for consistency.
    "label_semantics",
    "ref_ids",
    "target_ids",
]

# Columns added after the original schema; older CSVs lack them. Loaders fill
# these defaults so missing columns read as a normal pair label (NaN-safe).
_SCHEMA_DEFAULTS = {
    "label_semantics": LABEL_SEMANTICS_PAIR,
    "ref_ids": "",
    "target_ids": "",
}


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Project a loaded frame onto the current schema, filling new columns.

    - Drops columns not in the current schema (e.g. removed fields).
    - Adds any missing column with its default, so a CSV written before the
      set-semantics columns existed reads as a pair label with empty membership.
    - Backfills NaN/empty ``label_semantics`` to ``pair`` (a hand-edited or
      partially-migrated CSV must never leave the semantics undefined).
    """
    keep = [c for c in STITCHING_LABEL_COLUMNS if c in df.columns]
    df = df[keep].copy()
    for col, default in _SCHEMA_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    # NaN-safe: an all-empty float column (pandas reads a blank column as NaN)
    # or a per-row blank must resolve to the pair default.
    sem = df["label_semantics"].astype("string")
    df["label_semantics"] = sem.where(
        sem.isin([LABEL_SEMANTICS_PAIR, LABEL_SEMANTICS_SET]), LABEL_SEMANTICS_PAIR
    ).astype(str)
    for col in ("ref_ids", "target_ids"):
        df[col] = df[col].astype("string").fillna("").astype(str)
    # Preserve column order.
    return df[[c for c in STITCHING_LABEL_COLUMNS if c in df.columns]]


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
                df = pd.read_csv(self.csv_path, dtype={"group_id": str})
                # Drop any columns not in the current schema (e.g. removed fields)
                return _ensure_schema(df)
            except Exception as primary_error:
                logger.warning(
                    f"Failed to load stitching labels from {self.csv_path}: {primary_error}"
                )
                if backup_path.exists():
                    try:
                        logger.info(f"Recovering from backup: {backup_path}")
                        df = pd.read_csv(backup_path, dtype={"group_id": str})
                        return _ensure_schema(df)
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
                df = pd.read_csv(backup_path, dtype={"group_id": str})
                return _ensure_schema(df)
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
        selected_edges: list[dict],
        match_type: str,
        num_refs: int,
        num_targets: int,
        labeler: str,
        session_id: str,
        label_semantics: str = LABEL_SEMANTICS_PAIR,
        ref_ids: list[str] | None = None,
        target_ids: list[str] | None = None,
    ) -> None:
        """Add a stitching review label.

        If a label already exists for this group_id, it is replaced.
        Uses self.dataset_id for the dataset (partition path and stored column
        always match).

        Args:
            group_id: Deterministic group identifier
            selected_edges: List of {ref_id, target_id} dicts (empty for set rows)
            match_type: "1:N", "N:1", or "M:N"
            num_refs: Number of reference segments in the group
            num_targets: Number of target segments in the group
            labeler: Name of the reviewer
            session_id: Session identifier
            label_semantics: "pair" (default) or "set". A set label stores only
                group membership (ref_ids / target_ids) and leaves selected_edges
                empty, because the labeler asserted membership, not per-pair
                matches (see module docstring).
            ref_ids: Set-label reference membership (ignored for pair rows).
            target_ids: Set-label target membership (ignored for pair rows).
        """
        new_row = {
            "group_id": str(group_id),
            "dataset_id": self.dataset_id,
            "selected_edges": json.dumps(selected_edges),
            "match_type": match_type,
            "num_refs": num_refs,
            "num_targets": num_targets,
            "labeler": labeler,
            "labeled_at": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "label_semantics": label_semantics,
            "ref_ids": json.dumps(sorted(ref_ids)) if ref_ids else "",
            "target_ids": json.dumps(sorted(target_ids)) if target_ids else "",
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
                    df = _ensure_schema(df)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {csv_path}: {e}")

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=STITCHING_LABEL_COLUMNS)
