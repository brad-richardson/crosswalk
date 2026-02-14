"""Label persistence - Hive-partitioned CSV storage for labeled training data.

This module provides the LabelStore class which manages human labels in CSV format.
Labels are stored separately from features and raw data in a normalized architecture:

- labels/human/dataset={id}/data.csv - Human labels (metadata only)
- labels/agent/dataset={id}/data.csv - Agent labels (metadata only)
- labels/features/dataset={id}/data.parquet - Computed features
- labels/data/dataset={id}/data.parquet - Raw pair data (geometries + attributes)
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from shapely.geometry import LineString

from ..config import ALIGNMENT_FULL_TOLERANCE, FEATURE_COLUMNS, FEATURE_VERSION


class LabelLoadError(Exception):
    """Raised when labels cannot be loaded and recovery fails.

    This exception is raised when:
    - The primary CSV file exists but is corrupted
    - The backup file (if any) is also corrupted or missing
    - Both files fail to parse

    The UI should catch this and display an actionable error message.
    """

    pass


def get_data_version(file_path: Path) -> str:
    """Get a version identifier for a data file.

    The version identifier is based on the file's modification time and size,
    which together provide a reasonably unique fingerprint. This is faster than
    computing a full content hash while still detecting when files have changed.

    Args:
        file_path: Path to the data file (parquet, etc.)

    Returns:
        Version string in format "{mtime_ns}_{size}" (e.g., "1706050800000000000_12345678")
        Returns "unknown" if file doesn't exist or can't be stat'd.
    """
    try:
        file_path = Path(file_path)
        stat = file_path.stat()
        return f"{stat.st_mtime_ns}_{stat.st_size}"
    except (OSError, FileNotFoundError):
        return "unknown"


def is_subsegment_selection(
    ref_start: float,
    ref_end: float,
    target_start: float,
    target_end: float,
    tolerance: float | None = None,
) -> bool:
    """Check if the selection represents a sub-segment (not whole segment).

    Args:
        ref_start: Reference start percentage
        ref_end: Reference end percentage
        target_start: Target start percentage
        target_end: Target end percentage
        tolerance: Tolerance for floating point comparison (defaults to ALIGNMENT_FULL_TOLERANCE)

    Returns:
        True if this is a sub-segment selection (not 0-100% for both)
    """
    if tolerance is None:
        tolerance = ALIGNMENT_FULL_TOLERANCE
    ref_is_full = abs(ref_start) < tolerance and abs(ref_end - 1.0) < tolerance
    target_is_full = abs(target_start) < tolerance and abs(target_end - 1.0) < tolerance
    return not (ref_is_full and target_is_full)


# Default paths - using new normalized structure
DEFAULT_LABELS_DIR = Path("labels")
DEFAULT_HUMAN_LABELS_DIR = Path("labels/human")
DEFAULT_AGENT_LABELS_DIR = Path("labels/agent")

# Metadata columns for human labels (not features)
HUMAN_LABEL_COLUMNS = [
    "gers_id",  # Overture reference segment ID (renamed from ref_id)
    "target_id",
    "label",  # "match", "no_match", "unsure"
    "labeler",
    "labeled_at",  # ISO timestamp string
    "session_id",
    "original_decision",
    "original_confidence",
    # Sub-segment linear referencing (0.0-1.0 percentages)
    "ref_start_pct",
    "ref_end_pct",
    "target_start_pct",
    "target_end_pct",
    "is_subsegment",
]

# Agent label columns
AGENT_LABEL_COLUMNS = [
    "gers_id",
    "target_id",
    "label",  # "match", "no_match", "unsure"
    "confidence",  # Agent confidence (0.0-1.0)
    "reasoning",  # Agent explanation
    "labeler",  # Agent identifier
    "labeled_at",  # ISO timestamp string
]

# Default values for sub-segment columns
SUBSEGMENT_DEFAULTS = {
    "ref_start_pct": 0.0,
    "ref_end_pct": 1.0,
    "target_start_pct": 0.0,
    "target_end_pct": 1.0,
    "is_subsegment": False,
}

# Default values for version columns (None = pre-versioning label)
# Note: feature_version belongs in FeatureStore, not human labels
VERSION_DEFAULTS = {
    "ref_data_version": None,
    "target_data_version": None,
}


@dataclass
class LabelStore:
    """Manages labeled data storage for a single dataset partition.

    Uses normalized format with separate stores:
    - labels/human/dataset={id}/data.csv - Human labels (metadata only)
    - labels/features/dataset={id}/data.parquet - Computed features
    - labels/data/dataset={id}/data.parquet - Raw pair data (geometries)

    When adding labels, writes to all three stores.
    """

    dataset_id: str
    labels_dir: Path = DEFAULT_LABELS_DIR
    _df: pd.DataFrame | None = None

    def __post_init__(self):
        self.labels_dir = Path(self.labels_dir)
        # Use normalized human labels directory
        self.human_dir = self.labels_dir / "human"
        self.partition_path = self.human_dir / f"dataset={self.dataset_id}"
        self.csv_path = self.partition_path / "data.csv"
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load dataframe."""
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        """Load labels from CSV with backup recovery.

        Attempts to load from primary CSV first. If that fails and the file
        exists, tries to recover from backup (.csv.bak). Only returns an
        empty dataframe if no file exists; raises LabelLoadError if files
        exist but cannot be read.

        Returns:
            DataFrame with labels

        Raises:
            LabelLoadError: If files exist but cannot be loaded
        """
        backup_path = self.csv_path.with_suffix(".csv.bak")
        primary_exists = self.csv_path.exists()
        backup_exists = backup_path.exists()

        # Try primary file first
        if primary_exists:
            try:
                df = self._read_and_migrate(self.csv_path)
                return df
            except Exception as primary_error:
                logger.warning(f"Failed to load labels from {self.csv_path}: {primary_error}")

                # Try backup file
                if backup_exists:
                    try:
                        logger.info(f"Attempting recovery from backup: {backup_path}")
                        df = self._read_and_migrate(backup_path)
                        logger.info(f"Successfully recovered {len(df)} labels from backup")
                        return df
                    except Exception as backup_error:
                        logger.error(f"Backup recovery also failed: {backup_error}")
                        raise LabelLoadError(
                            f"Both primary and backup label files are corrupted.\n"
                            f"Primary ({self.csv_path}): {primary_error}\n"
                            f"Backup ({backup_path}): {backup_error}"
                        ) from backup_error

                # Primary exists but is corrupted, no backup available
                raise LabelLoadError(
                    f"Label file is corrupted and no backup available: {self.csv_path}\n"
                    f"Error: {primary_error}"
                ) from primary_error

        # No primary file - try backup as last resort
        if backup_exists:
            try:
                logger.info(f"Primary file missing, loading from backup: {backup_path}")
                df = self._read_and_migrate(backup_path)
                logger.info(f"Loaded {len(df)} labels from backup (primary missing)")
                return df
            except Exception as e:
                logger.warning(f"Backup file exists but failed to load: {e}")
                raise LabelLoadError(
                    f"Backup file exists but is corrupted: {backup_path}\nError: {e}"
                ) from e

        # No files exist - return empty dataframe (fresh start)
        return self._empty_dataframe()

    def _read_and_migrate(self, path: Path) -> pd.DataFrame:
        """Read CSV and apply backward compatibility migrations.

        Args:
            path: Path to CSV file

        Returns:
            Migrated DataFrame

        Raises:
            ValueError: If file is missing required columns (corrupted schema)
        """
        df = pd.read_csv(path)

        # Validate required columns exist (detect schema corruption)
        # Need either gers_id or ref_id (legacy) to be present
        has_gers_id = "gers_id" in df.columns
        has_ref_id = "ref_id" in df.columns
        has_target_id = "target_id" in df.columns
        has_label = "label" in df.columns

        if not (has_gers_id or has_ref_id):
            cols = list(df.columns)
            sample = f"{cols[:3]}...{cols[-2:]}" if len(cols) > 5 else str(cols)
            raise ValueError(f"Missing required column: gers_id (found {len(cols)} cols: {sample})")
        if not has_target_id:
            cols = list(df.columns)
            sample = f"{cols[:3]}...{cols[-2:]}" if len(cols) > 5 else str(cols)
            raise ValueError(
                f"Missing required column: target_id (found {len(cols)} cols: {sample})"
            )
        if not has_label:
            cols = list(df.columns)
            sample = f"{cols[:3]}...{cols[-2:]}" if len(cols) > 5 else str(cols)
            raise ValueError(f"Missing required column: label (found {len(cols)} cols: {sample})")

        # Handle backward compatibility - add missing subsegment columns
        for col, default_val in SUBSEGMENT_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default_val
        # Handle backward compatibility - add missing version columns
        for col, default_val in VERSION_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default_val
        # Handle ref_id -> gers_id rename for backward compatibility
        if "ref_id" in df.columns and "gers_id" not in df.columns:
            df = df.rename(columns={"ref_id": "gers_id"})
        return df

    def _empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with correct schema."""
        return pd.DataFrame(columns=HUMAN_LABEL_COLUMNS)

    def save(self) -> None:
        """Save labels to CSV atomically with backup.

        Uses a write-to-temp-then-rename pattern to prevent data loss
        if the process crashes during write:
        1. Write to temporary file (.csv.tmp)
        2. Backup existing file (.csv.bak) if present
        3. Atomic rename temp file to final path
        """
        self.partition_path.mkdir(parents=True, exist_ok=True)

        temp_path = self.csv_path.with_suffix(".csv.tmp")
        backup_path = self.csv_path.with_suffix(".csv.bak")

        # Write to temp file first
        self._df.to_csv(temp_path, index=False, float_format=lambda x: f"{x:.10g}")

        # Backup existing file (if present)
        if self.csv_path.exists():
            # Use replace() for cross-platform atomicity (rename() fails on Windows if dest exists)
            self.csv_path.replace(backup_path)

        # Atomic replace temp to final
        temp_path.replace(self.csv_path)

    def add(
        self,
        gers_id: str,
        target_id: str,
        label: str,
        labeler: str,
        session_id: str,
        original_decision: str,
        original_confidence: float,
        features: dict[str, float],
        ref_start_pct: float = 0.0,
        ref_end_pct: float = 1.0,
        target_start_pct: float = 0.0,
        target_end_pct: float = 1.0,
        ref_data_version: str | None = None,
        target_data_version: str | None = None,
        feature_version: str | None = None,
        ref_geometry: LineString | None = None,
        target_geometry: LineString | None = None,
        ref_name_raw: str | None = None,
        target_name_raw: str | None = None,
        ref_class_raw: str | None = None,
        target_class_raw: str | None = None,
        ref_subclass: str | None = None,
        target_subclass: str | None = None,
        ref_topology: dict | None = None,
        target_topology: dict | None = None,
    ) -> None:
        """Add a new label.

        Writes to all three normalized stores:
        - labels/human/: Label metadata in CSV
        - labels/data/: Raw geometries and attributes in GeoParquet
        - labels/features/: Computed features in Parquet

        Args:
            gers_id: Overture reference segment ID (GERS ID)
            target_id: Target segment ID
            label: Label value (match, no_match, unsure)
            labeler: Name of the labeler
            session_id: Unique session identifier
            original_decision: Rule-based matcher decision
            original_confidence: Rule-based matcher confidence
            features: Dict of feature values
            ref_start_pct: Start of reference sub-segment (0.0-1.0)
            ref_end_pct: End of reference sub-segment (0.0-1.0)
            target_start_pct: Start of target sub-segment (0.0-1.0)
            target_end_pct: End of target sub-segment (0.0-1.0)
            ref_data_version: Version identifier for reference data file
            target_data_version: Version identifier for target data file
            feature_version: Feature computation version (defaults to FEATURE_VERSION)
            ref_geometry: Reference geometry (WGS84) for persistence in data store
            target_geometry: Target geometry (WGS84) for persistence in data store
            ref_name_raw: Reference segment name for data store
            target_name_raw: Target segment name for data store
            ref_class_raw: Reference road class for data store
            target_class_raw: Target road class for data store
            ref_subclass: Reference road subclass for data store
            target_subclass: Target road subclass for data store
        """
        # Determine if this is a sub-segment selection
        is_subseg = is_subsegment_selection(
            ref_start_pct, ref_end_pct, target_start_pct, target_end_pct
        )

        # Build the new row with metadata only (no features)
        new_row = {
            "gers_id": str(gers_id),
            "target_id": str(target_id),
            "label": label,
            "labeler": labeler,
            "labeled_at": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "original_decision": original_decision,
            "original_confidence": original_confidence,
            "ref_start_pct": ref_start_pct,
            "ref_end_pct": ref_end_pct,
            "target_start_pct": target_start_pct,
            "target_end_pct": target_end_pct,
            "is_subsegment": is_subseg,
            "ref_data_version": ref_data_version,
            "target_data_version": target_data_version,
        }

        # Remove any existing label for this pair (re-labeling replaces, not duplicates)
        if self._df is not None and len(self._df) > 0:
            mask = (self._df["gers_id"] == str(gers_id)) & (self._df["target_id"] == str(target_id))
            if mask.any():
                self._df = self._df[~mask]

        self._df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.save()

        # Write to FeatureStore (always, if features provided)
        if features:
            from .feature_store import FeatureStore

            feature_store = FeatureStore(self.dataset_id, features_dir=self.labels_dir / "features")
            feature_store.add(
                gers_id=gers_id,
                target_id=target_id,
                features=features,
                feature_version=feature_version if feature_version else FEATURE_VERSION,
            )
            feature_store.save()

        # Write to DataStore (if geometries provided)
        if ref_geometry is not None and target_geometry is not None:
            from .data_store import DataStore

            data_store = DataStore(self.dataset_id, data_dir=self.labels_dir / "data")
            data_store.add(
                gers_id=gers_id,
                target_id=target_id,
                ref_geometry=ref_geometry,
                target_geometry=target_geometry,
                ref_name=ref_name_raw,
                target_name=target_name_raw,
                ref_class=ref_class_raw,
                target_class=target_class_raw,
                ref_subclass=ref_subclass,
                target_subclass=target_subclass,
                ref_topology=ref_topology,
                target_topology=target_topology,
            )
            data_store.save()

    def update_label(
        self,
        gers_id: str,
        target_id: str,
        new_label: str,
        labeler: str,
    ) -> bool:
        """Update an existing label's value.

        Preserves original metadata but updates label, labeler, and timestamp.

        Args:
            gers_id: Overture reference segment ID
            target_id: Target segment ID
            new_label: New label value (match, no_match, unsure)
            labeler: Name of the labeler making the update

        Returns:
            True if found and updated, False if pair not found.
        """
        df = self.df
        mask = (df["gers_id"] == str(gers_id)) & (df["target_id"] == str(target_id))
        if not mask.any():
            return False

        idx = df[mask].index[-1]  # Latest entry for this pair
        self._df.at[idx, "label"] = new_label
        self._df.at[idx, "labeler"] = labeler
        self._df.at[idx, "labeled_at"] = datetime.now(UTC).isoformat()
        self.save()
        return True

    def delete_label(
        self,
        gers_id: str,
        target_id: str,
    ) -> bool:
        """Delete a label and its associated feature/geometry data.

        Removes from all three stores (human labels, features, data).

        Args:
            gers_id: Overture reference segment ID
            target_id: Target segment ID

        Returns:
            True if found and deleted, False if pair not found.
        """
        from .data_store import DataStore
        from .feature_store import FeatureStore

        df = self.df
        mask = (df["gers_id"] == str(gers_id)) & (df["target_id"] == str(target_id))
        if not mask.any():
            return False

        # Remove from labels
        self._df = df[~mask].reset_index(drop=True)
        self.save()

        # Remove from feature store
        fs = FeatureStore(self.dataset_id, features_dir=self.labels_dir / "features")
        fs.delete_pair(gers_id, target_id)

        # Remove from data store
        ds = DataStore(self.dataset_id, data_dir=self.labels_dir / "data")
        ds.delete_pair(gers_id, target_id)

        return True

    def get_labeled_pairs(self, labeler: str | None = None) -> set[tuple[str, str]]:
        """Get set of already-labeled (gers_id, target_id) pairs.

        Args:
            labeler: If provided, only return pairs labeled by this labeler.
        """
        df = self.df
        if df is None or len(df) == 0:
            return set()

        if labeler:
            df = df[df["labeler"] == labeler]

        if len(df) == 0:
            return set()
        return set(zip(df["gers_id"], df["target_id"], strict=True))

    def get_stats(self) -> dict[str, Any]:
        """Get labeling statistics."""
        df = self.df
        if df is None or len(df) == 0:
            return {"total": 0, "match": 0, "no_match": 0, "skip": 0}

        return {
            "total": len(df),
            "match": (df["label"] == "match").sum(),
            "no_match": (df["label"] == "no_match").sum(),
            "skip": (df["label"] == "skip").sum(),
            "unsure": (df["label"] == "unsure").sum(),
        }

    def remove_last(self) -> dict | None:
        """Remove the last label (for undo). Returns removed row or None."""
        df = self.df
        if df is None or len(df) == 0:
            return None

        last_row = df.iloc[-1].to_dict()
        self._df = df.iloc[:-1].reset_index(drop=True)
        self.save()
        return last_row

    @staticmethod
    def load_all(
        labels_dir: Path = DEFAULT_LABELS_DIR,
        skip_errors: bool = True,
    ) -> pd.DataFrame:
        """Load all human labels joined with features for ML training.

        Loads from normalized format:
        - labels/human/dataset=*/data.csv - Human labels (metadata)
        - labels/features/dataset=*/data.parquet - Computed features

        Args:
            labels_dir: Base labels directory (contains human/, features/ subdirs)
            skip_errors: If True (default), skip partitions that fail to load.
                        If False, raise an error on any loading failure.

        Returns:
            DataFrame with all labels joined with features and 'dataset' column

        Raises:
            ValueError: If skip_errors=False and any partition fails to load
        """
        from .feature_store import FeatureStore

        labels_dir = Path(labels_dir)
        human_dir = labels_dir / "human"
        features_dir = labels_dir / "features"

        # Load human labels
        human_labels = LabelStore.load_human_labels(human_dir, skip_errors=skip_errors)
        if len(human_labels) == 0:
            return pd.DataFrame(columns=HUMAN_LABEL_COLUMNS + FEATURE_COLUMNS + ["dataset"])

        # Load features
        features = FeatureStore.load_all(features_dir)
        if len(features) == 0:
            logger.warning(f"No features found in {features_dir}")
            return pd.DataFrame(columns=HUMAN_LABEL_COLUMNS + FEATURE_COLUMNS + ["dataset"])

        # Drop feature_version from human labels if present — it belongs in
        # FeatureStore and having it on both sides creates _x/_y suffixes
        if "feature_version" in human_labels.columns:
            human_labels = human_labels.drop(columns=["feature_version"])

        # Join labels with features
        result = human_labels.merge(
            features,
            on=["gers_id", "target_id", "dataset"],
            how="inner",
        )

        if len(result) < len(human_labels):
            missing = len(human_labels) - len(result)
            logger.warning(f"{missing} labels missing features (run 'matcher labels backfill')")

        return result

    @staticmethod
    def load_human_labels(
        human_dir: Path = DEFAULT_HUMAN_LABELS_DIR,
        skip_errors: bool = True,
    ) -> pd.DataFrame:
        """Load human labels from normalized format.

        Args:
            human_dir: Directory containing human label CSVs (labels/human/)
            skip_errors: If True, skip partitions that fail to load.

        Returns:
            DataFrame with human labels and 'dataset' column
        """
        human_dir = Path(human_dir)
        if not human_dir.exists():
            return pd.DataFrame(columns=HUMAN_LABEL_COLUMNS + ["dataset"])

        dfs = []
        for partition_dir in human_dir.glob("dataset=*"):
            if not partition_dir.is_dir():
                continue
            dataset_id = partition_dir.name.split("=")[1]
            csv_path = partition_dir / "data.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(
                        csv_path,
                        dtype={"gers_id": str, "ref_id": str, "target_id": str},
                    )
                    df["dataset"] = dataset_id
                    # Handle ref_id -> gers_id rename
                    if "ref_id" in df.columns and "gers_id" not in df.columns:
                        df = df.rename(columns={"ref_id": "gers_id"})
                    dfs.append(df)
                except Exception as e:
                    if skip_errors:
                        logger.warning(f"Failed to load {csv_path}: {e}")
                    else:
                        raise

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=HUMAN_LABEL_COLUMNS + ["dataset"])

    @staticmethod
    def load_agent_labels(
        agent_dir: Path = DEFAULT_AGENT_LABELS_DIR,
        skip_errors: bool = True,
    ) -> pd.DataFrame:
        """Load agent labels from normalized format.

        Args:
            agent_dir: Directory containing agent label CSVs (labels/agent/)
            skip_errors: If True, skip partitions that fail to load.

        Returns:
            DataFrame with agent labels and 'dataset' column
        """
        agent_dir = Path(agent_dir)
        if not agent_dir.exists():
            return pd.DataFrame(columns=AGENT_LABEL_COLUMNS + ["dataset"])

        dfs = []
        for partition_dir in agent_dir.glob("dataset=*"):
            if not partition_dir.is_dir():
                continue
            dataset_id = partition_dir.name.split("=")[1]
            csv_path = partition_dir / "data.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(
                        csv_path,
                        dtype={"gers_id": str, "ref_id": str, "target_id": str},
                    )
                    df["dataset"] = dataset_id
                    # Handle ref_id -> gers_id rename
                    if "ref_id" in df.columns and "gers_id" not in df.columns:
                        df = df.rename(columns={"ref_id": "gers_id"})
                    dfs.append(df)
                except Exception as e:
                    if skip_errors:
                        logger.warning(f"Failed to load {csv_path}: {e}")
                    else:
                        raise

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=AGENT_LABEL_COLUMNS + ["dataset"])
