"""Label persistence - Hive-partitioned CSV storage for labeled training data."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from shapely.geometry import LineString

from ..config import ALIGNMENT_FULL_TOLERANCE, FEATURE_COLUMNS, FEATURE_VERSION
from ..features.semantic import _extract_name_string


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


# Default paths
DEFAULT_LABELS_DIR = Path("labels")

# Metadata columns for labels (not features)
LABEL_METADATA_COLUMNS = [
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
    # Data versioning columns (added 2026-01-23)
    "ref_data_version",  # Version identifier for reference data file
    "target_data_version",  # Version identifier for target data file
    "feature_version",  # Feature computation version (from config.FEATURE_VERSION)
]

# Column definitions for labels: metadata + all feature columns
# FEATURE_COLUMNS is imported from config.py - single source of truth
LABEL_COLUMNS = LABEL_METADATA_COLUMNS + FEATURE_COLUMNS

# Default values for sub-segment columns (for backward compatibility)
SUBSEGMENT_DEFAULTS = {
    "ref_start_pct": 0.0,
    "ref_end_pct": 1.0,
    "target_start_pct": 0.0,
    "target_end_pct": 1.0,
    "is_subsegment": False,
}

# Default values for version columns (None = pre-versioning label)
VERSION_DEFAULTS = {
    "ref_data_version": None,
    "target_data_version": None,
    "feature_version": None,
}


@dataclass
class LabelStore:
    """Manages labeled data storage for a single dataset partition."""

    dataset_id: str
    labels_dir: Path = DEFAULT_LABELS_DIR
    _df: pd.DataFrame | None = None

    def __post_init__(self):
        self.labels_dir = Path(self.labels_dir)
        self.partition_path = self.labels_dir / f"dataset={self.dataset_id}"
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
        return pd.DataFrame(columns=LABEL_COLUMNS)

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
    ) -> None:
        """Add a new label.

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
            ref_geometry: Reference geometry (WGS84) for persistence in companion file
            target_geometry: Target geometry (WGS84) for persistence in companion file
            ref_name_raw: Reference segment name for companion file
            target_name_raw: Target segment name for companion file
            ref_class_raw: Reference road class for companion file
            target_class_raw: Target road class for companion file
            ref_subclass: Reference road subclass for companion file
            target_subclass: Target road subclass for companion file
        """
        # Determine if this is a sub-segment selection
        is_subseg = is_subsegment_selection(
            ref_start_pct, ref_end_pct, target_start_pct, target_end_pct
        )

        new_row = {
            "gers_id": str(gers_id),
            "target_id": str(target_id),
            "label": label,
            "labeler": labeler,
            "labeled_at": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "original_decision": original_decision,
            "original_confidence": original_confidence,
            # Sub-segment fields
            "ref_start_pct": ref_start_pct,
            "ref_end_pct": ref_end_pct,
            "target_start_pct": target_start_pct,
            "target_end_pct": target_end_pct,
            "is_subsegment": is_subseg,
            # Data versioning fields
            "ref_data_version": ref_data_version,
            "target_data_version": target_data_version,
            "feature_version": feature_version if feature_version else FEATURE_VERSION,
            # Geometric features (9)
            "hausdorff_distance_m": features.get("hausdorff_distance_m", 0.0),
            "mean_hausdorff_distance_m": features.get("mean_hausdorff_distance_m", 0.0),
            "hausdorff_p95_m": features.get("hausdorff_p95_m", 0.0),
            "buffer_iou_5m": features.get("buffer_iou_5m", 0.0),
            "buffer_iou_15m": features.get("buffer_iou_15m", 0.0),
            "heading_delta": features.get("heading_delta", 0.0),
            "length_ratio": features.get("length_ratio", 0.0),
            "centroid_distance_m": features.get("centroid_distance_m", 0.0),
            "collinear_gap_ratio": features.get("collinear_gap_ratio", 1.0),
            # Semantic features - name (7)
            "name_levenshtein": features.get("name_levenshtein", 0.0),
            "name_jaro_winkler": features.get("name_jaro_winkler", 0.0),
            "name_token_sort": features.get("name_token_sort", 0.0),
            "name_soundex": features.get("name_soundex", 0.5),
            "name_metaphone": features.get("name_metaphone", 0.5),
            "has_name_ref": features.get("has_name_ref", 0.0),
            "has_name_target": features.get("has_name_target", 0.0),
            "name_is_generic": features.get("name_is_generic", 0.0),
            # Semantic features - class (1)
            "class_similarity": features.get("class_similarity", 0.0),
            # Endpoint/connectivity features (3) - direction-invariant min/max
            "min_endpoint_proximity_m": features.get("min_endpoint_proximity_m", 0.0),
            "max_endpoint_proximity_m": features.get("max_endpoint_proximity_m", 0.0),
            "shared_endpoint_count": features.get("shared_endpoint_count", 0),
            # Lateral offset features (3) - distance features use _m suffix
            "lateral_offset_m": features.get("lateral_offset_m", 0.0),
            "lateral_offset_iqr_m": features.get("lateral_offset_iqr_m", 0.0),
            "lateral_offset_p95_m": features.get("lateral_offset_p95_m", 0.0),
            # Topology features (12)
            "from_degree_ref": features.get("from_degree_ref", 0),
            "to_degree_ref": features.get("to_degree_ref", 0),
            "from_degree_target": features.get("from_degree_target", 0),
            "to_degree_target": features.get("to_degree_target", 0),
            # Topology defaults use 0.5 (neutral) to match _get_error_features() in compute.py
            # This ensures training/inference consistency when features are missing
            "degree_match_score": features.get("degree_match_score", 0.5),
            "degree_signature_similarity": features.get("degree_signature_similarity", 0.5),
            "is_dead_end_ref": features.get("is_dead_end_ref", 0.5),
            "is_dead_end_target": features.get("is_dead_end_target", 0.5),
            "dead_end_match": features.get("dead_end_match", 0.5),
            "is_intersection_ref": features.get("is_intersection_ref", 0.5),
            "is_intersection_target": features.get("is_intersection_target", 0.5),
            "intersection_match": features.get("intersection_match", 0.5),
            # Alignment coverage features (4)
            "ref_coverage": features.get("ref_coverage", 0.0),
            "target_coverage": features.get("target_coverage", 0.0),
            "min_coverage": features.get("min_coverage", 0.0),
            "coverage_ratio": features.get("coverage_ratio", 0.0),
            # Graphlet features (2)
            "graphlet_similarity": features.get("graphlet_similarity", 0.5),
            "endpoint_degree_similarity": features.get("endpoint_degree_similarity", 0.5),
            # Sinuosity features (3)
            "sinuosity_ref": features.get("sinuosity_ref", 1.0),
            "sinuosity_target": features.get("sinuosity_target", 1.0),
            "sinuosity_delta": features.get("sinuosity_delta", 0.0),
            # Heading consistency features (3)
            "heading_consistency_ref": features.get("heading_consistency_ref", 1.0),
            "heading_consistency_target": features.get("heading_consistency_target", 1.0),
            "heading_consistency_delta": features.get("heading_consistency_delta", 0.0),
            # Vertex density features (3)
            "vertex_density_ref": features.get("vertex_density_ref", 0.0),
            "vertex_density_target": features.get("vertex_density_target", 0.0),
            "vertex_density_ratio": features.get("vertex_density_ratio", 0.0),
            # Length features (1)
            "min_length_m": features.get("min_length_m", 0.0),
            # Shape complexity features (3)
            "shape_complexity_ref": features.get("shape_complexity_ref", 0),
            "shape_complexity_target": features.get("shape_complexity_target", 0),
            "shape_complexity_delta": features.get("shape_complexity_delta", 0),
            # Numeric route matching (1) - default 0.0 when neither has number
            "name_numeric_match": features.get("name_numeric_match", 0.0),
        }

        self._df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.save()

        # Persist geometries to companion file if provided
        if ref_geometry is not None and target_geometry is not None:
            from .geometry_store import GeometryStore

            geo_store = GeometryStore(self.dataset_id)
            geo_store.add(
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
            )
            geo_store.save()

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
            "associated": (df["label"] == "associated").sum(),
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
        """Load all label partitions for ML training.

        Uses PyArrow's Hive partitioning to read all dataset partitions
        and adds a 'dataset' column from the partition path.

        Args:
            labels_dir: Directory containing Hive-partitioned label CSVs
            skip_errors: If True (default), skip partitions that fail to load.
                        If False, raise an error on any loading failure.

        Returns:
            DataFrame with all labels and 'dataset' column

        Raises:
            ValueError: If skip_errors=False and any partition fails to load
        """
        labels_dir = Path(labels_dir)
        if not labels_dir.exists():
            if skip_errors:
                return pd.DataFrame(columns=LABEL_COLUMNS + ["dataset"])
            else:
                raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        # Always use manual loading for more control over error handling
        # PyArrow's Hive partitioning can fail on type mismatches
        dfs = []
        errors = []

        for partition_dir in labels_dir.glob("dataset=*"):
            if not partition_dir.is_dir():
                continue
            dataset_id = partition_dir.name.split("=")[1]
            csv_path = partition_dir / "data.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    df["dataset"] = dataset_id
                    dfs.append(df)
                except Exception as e:
                    if skip_errors:
                        logger.warning(f"Failed to load {csv_path}: {e}")
                    else:
                        errors.append((dataset_id, str(e)))

        if errors and not skip_errors:
            error_msg = "\n".join([f"  {ds}: {err}" for ds, err in errors])
            raise ValueError(f"Failed to load label partitions:\n{error_msg}")

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            if "ref_id" in result.columns and "gers_id" not in result.columns:
                result = result.rename(columns={"ref_id": "gers_id"})
            return result
        return pd.DataFrame(columns=LABEL_COLUMNS + ["dataset"])


# Backward compatibility aliases
def load_labels(path: Path) -> pd.DataFrame:
    """Load labels from a single partition (backward compatibility).

    For new code, use LabelStore(dataset_id).df instead.
    """
    # Extract dataset_id from path
    if path.name == "data.csv":
        # New format: labels/dataset=xxx/data.csv
        partition_name = path.parent.name
        if partition_name.startswith("dataset="):
            dataset_id = partition_name.split("=")[1]
            store = LabelStore(dataset_id, labels_dir=path.parent.parent)
            return store.df
    # Old format: data/labels/labels_xxx.parquet
    dataset_id = path.stem.replace("labels_", "")
    store = LabelStore(dataset_id)
    return store.df


def save_labels(df: pd.DataFrame, path: Path) -> None:
    """Save labels to a partition (backward compatibility).

    For new code, use LabelStore(dataset_id).save() instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=lambda x: f"{x:.10g}")


def backfill_features(
    labels_dir: Path = DEFAULT_LABELS_DIR,
    overture_path: Path | None = None,
    data_dir: Path | None = None,
    dry_run: bool = False,
    skip_missing: bool = False,
    report_only: bool = False,
    drop_orphaned: bool = False,
) -> dict[str, dict[str, int]]:
    """Recompute all features for existing labels using current computation logic.

    This function is needed after changes to feature computation (e.g., alignment-aware
    topology) to ensure existing training labels have consistent features.

    The function:
    1. Loads all existing labels from labels/dataset=*/data.csv
    2. Groups labels by dataset
    3. For each dataset, loads source geometries and builds spatial indexes
    4. Recomputes alignment and ALL features from scratch for each label
    5. Updates the label CSV files with new feature values
    6. Updates version tracking columns (ref_data_version, target_data_version, feature_version)

    Args:
        labels_dir: Directory containing Hive-partitioned label CSVs
        overture_path: Path to Overture segments parquet. If None, will look for
                      dataset-specific files like {dataset}_overture_segments.parquet
        data_dir: Directory containing target dataset parquet files (default: data/raw/)
        dry_run: If True, compute features but don't write to disk
        skip_missing: If True, skip datasets with missing data files. If False (default),
                     raise an error when any dataset is missing required data.
        report_only: If True, only report what can/cannot be backfilled without modifying
        drop_orphaned: If True, remove labels where IDs are not found in current data

    Returns:
        Dict with counts per dataset: {dataset_name: {"updated": n, "orphaned": n, "total": n}}

    Raises:
        FileNotFoundError: If skip_missing=False and any dataset is missing required data
    """
    import geopandas as gpd
    from loguru import logger

    from ..config import DEFAULT_SNAP_TOLERANCE_M
    from ..features.alignment import linestring_alignment
    from ..features.compute import (
        compute_graphlet_similarity,
        compute_pair_features,
        precompute_graphlet_features,
    )
    from ..features.spatial_context import SpatialContextIndex, compute_aligned_endpoint_features
    from ..utils.geometry import filter_to_linestrings

    # Set default paths
    if data_dir is None:
        data_dir = Path("data/raw")

    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Find all label partitions
    partitions = list(labels_dir.glob("dataset=*/data.csv"))
    if not partitions:
        raise FileNotFoundError(f"No label partitions found in {labels_dir}")

    logger.info(f"Found {len(partitions)} label partitions to backfill")

    # Cache for loaded reference data (keyed by overture file path)
    ref_cache = {}

    def get_reference_data(dataset_name: str):
        """Load reference data for a dataset, with caching."""
        from ..filenames import find_overture_segments

        # Determine Overture file path
        if overture_path is not None:
            ref_path = overture_path
        else:
            # Use find_overture_segments for progressive prefix matching
            # Handles versioned filenames and tries progressively shorter prefixes
            ref_path = find_overture_segments(data_dir, dataset_name)

        # Handle case where find_overture_segments returns None
        if ref_path is None:
            logger.warning(f"  No Overture file found for dataset: {dataset_name}")
            return None

        ref_path_str = str(ref_path)
        if ref_path_str in ref_cache:
            return ref_cache[ref_path_str]

        if not ref_path.exists():
            logger.warning(f"  Overture file not found: {ref_path}")
            return None

        logger.info(f"  Loading Overture segments from {ref_path}...")
        ref_gdf = gpd.read_parquet(ref_path)
        ref_gdf = filter_to_linestrings(ref_gdf, source_name="reference")
        ref_gdf["id"] = ref_gdf["id"].astype(str)
        ref_lookup = ref_gdf.set_index("id")

        # Project to meters for accurate feature computation
        if ref_gdf.crs is not None and ref_gdf.crs.is_geographic:
            utm_crs = ref_gdf.estimate_utm_crs()
            logger.info(f"  Projecting reference to {utm_crs}")
            ref_gdf_proj = ref_gdf.to_crs(utm_crs)
        else:
            ref_gdf_proj = ref_gdf
            utm_crs = ref_gdf.crs

        # Build graphlet data for reference (Overture has explicit connectors)
        logger.info("  Building reference graphlet data...")
        ref_has_connectors = "connectors" in ref_gdf.columns
        ref_graphlet_data = precompute_graphlet_features(
            ref_gdf_proj,
            id_column="id",
            tolerance_m=DEFAULT_SNAP_TOLERANCE_M,
            connectors_column="connectors" if ref_has_connectors else None,
        )

        result = (ref_gdf, ref_gdf_proj, ref_lookup, ref_graphlet_data, utm_crs, ref_path)
        ref_cache[ref_path_str] = result
        return result

    results = {}

    # Process each partition
    for partition_path in partitions:
        dataset_name = partition_path.parent.name.replace("dataset=", "")
        logger.info(f"\nProcessing {dataset_name}...")

        # Load labels for this dataset
        df = pd.read_csv(partition_path)

        # Cast feature columns to float64 to prevent FutureWarning when writing
        # float values into columns that pandas inferred as int64 (e.g.,
        # length_ratio stored as "1" in CSV gets read as int64, but backfill
        # writes 0.9999... which is float64).
        for col in FEATURE_COLUMNS:
            if col in df.columns and df[col].dtype != "float64":
                df[col] = df[col].astype("float64")

        original_count = len(df)

        if original_count == 0:
            logger.info(f"  Skipping {dataset_name}: no labels")
            results[dataset_name] = {"updated": 0, "orphaned": 0, "total": 0}
            continue

        # Load reference data for this dataset
        ref_data = get_reference_data(dataset_name)
        if ref_data is None:
            if skip_missing:
                logger.warning(f"  Skipping {dataset_name}: reference data not found")
                results[dataset_name] = {
                    "updated": 0,
                    "orphaned": 0,
                    "total": original_count,
                    "skipped": "ref_not_found",
                }
                continue
            else:
                # Determine what file was expected
                if overture_path is not None:
                    expected = overture_path
                else:
                    expected = data_dir / f"{dataset_name}_overture_segments.parquet"
                raise FileNotFoundError(
                    f"Reference data not found for {dataset_name}. "
                    f"Expected: {expected}\n"
                    f"Use --skip-missing to skip datasets with missing data."
                )

        ref_gdf, ref_gdf_proj, ref_lookup, ref_graphlet_data, utm_crs, ref_path = ref_data

        # Find target dataset file using centralized filename functions
        from ..filenames import find_osm_segments, find_target_file

        # For OSM datasets, use find_osm_segments
        if dataset_name.endswith("_osm"):
            base_name = dataset_name[:-4]  # Remove "_osm" suffix
            target_path = find_osm_segments(data_dir, base_name)
        else:
            target_path = find_target_file(data_dir, dataset_name)

        if target_path is None:
            if skip_missing:
                logger.warning(f"  Skipping {dataset_name}: target data not found")
                results[dataset_name] = {
                    "updated": 0,
                    "orphaned": 0,
                    "total": original_count,
                    "skipped": "target_not_found",
                }
                continue
            else:
                raise FileNotFoundError(
                    f"Target data not found for {dataset_name}. "
                    f"Tried: {data_dir / f'{dataset_name}.parquet'}, "
                    f"{data_dir / f'{dataset_name}_segments.parquet'}\n"
                    f"Use --skip-missing to skip datasets with missing data."
                )

        logger.info(f"  Loading target data from {target_path}")
        target_gdf = gpd.read_parquet(target_path)
        target_gdf = filter_to_linestrings(target_gdf, source_name="target")
        target_gdf["id"] = target_gdf["id"].astype(str)
        target_lookup = target_gdf.set_index("id")

        # Project target to same CRS as reference
        if target_gdf.crs != utm_crs:
            target_gdf_proj = target_gdf.to_crs(utm_crs)
        else:
            target_gdf_proj = target_gdf

        # Build spatial context for endpoint features
        logger.info("  Building spatial context for target...")
        target_context = SpatialContextIndex()
        target_context.build_from_gdf(target_gdf_proj, id_column="id")

        # Build graphlet data for target
        logger.info("  Building target graphlet data...")
        target_graphlet_data = precompute_graphlet_features(
            target_gdf_proj, id_column="id", tolerance_m=DEFAULT_SNAP_TOLERANCE_M
        )

        # Extract connector data for snapping endpoint fractions to real junctions
        _, target_seg_to_connectors_ep, _, _ = (
            target_graphlet_data if target_graphlet_data else (None, None, None, None)
        )

        # Get data versions for tracking
        ref_data_version = get_data_version(ref_path)
        target_data_version = get_data_version(target_path)

        # Load geometry store for persisted geometry recovery and persistence
        from .geometry_store import GeometryStore

        geo_store = GeometryStore(dataset_name)

        # Context feature columns to preserve from existing CSV row when using
        # persisted geometry (tier 2) — these require full dataset spatial index /
        # graph construction and can't be recomputed from geometry alone
        CONTEXT_FEATURES = {
            "min_endpoint_proximity_m",
            "max_endpoint_proximity_m",
            "shared_endpoint_count",
            "from_degree_ref",
            "to_degree_ref",
            "from_degree_target",
            "to_degree_target",
            "degree_match_score",
            "degree_signature_similarity",
            "is_dead_end_ref",
            "is_dead_end_target",
            "dead_end_match",
            "is_intersection_ref",
            "is_intersection_target",
            "intersection_match",
            "graphlet_similarity",
            "endpoint_degree_similarity",
        }

        # Process each label row
        updated = 0
        recovered = 0
        orphaned_indices = []
        for idx, row in df.iterrows():
            # Handle both gers_id and ref_id naming
            ref_id = str(row.get("gers_id", row.get("ref_id", "")))
            target_id = str(row["target_id"])

            ids_in_data = ref_id in ref_lookup.index and target_id in target_lookup.index

            if ids_in_data:
                # === TIER 1: IDs found in current data — compute ALL features ===
                ref_row = ref_lookup.loc[ref_id]
                target_row = target_lookup.loc[target_id]

                # Get projected geometries using label index (.loc), not positional (.iloc)
                ref_idx_in_gdf = ref_gdf[ref_gdf["id"] == ref_id].index[0]
                target_idx_in_gdf = target_gdf[target_gdf["id"] == target_id].index[0]

                ref_geom = ref_gdf_proj.geometry.loc[ref_idx_in_gdf]
                target_geom = target_gdf_proj.geometry.loc[target_idx_in_gdf]

                if ref_geom is None or ref_geom.is_empty:
                    continue
                if target_geom is None or target_geom.is_empty:
                    continue
                if not isinstance(ref_geom, LineString):
                    logger.debug(f"    Skipping {ref_id}: ref geometry is {ref_geom.geom_type}")
                    continue
                if not isinstance(target_geom, LineString):
                    logger.debug(
                        f"    Skipping {target_id}: target geometry is {target_geom.geom_type}"
                    )
                    continue

                # Compute alignment from scratch
                alignment = linestring_alignment(ref_geom, target_geom)

                # Compute endpoint features using aligned subline endpoints
                # This ensures endpoint proximity is measured at the alignment boundaries,
                # not at the full geometry endpoints
                target_filtered_idx = (
                    target_gdf_proj[target_gdf_proj["id"] == target_id].index[0]
                    if target_id in target_gdf_proj["id"].values
                    else None
                )
                endpoint_features = compute_aligned_endpoint_features(
                    target_geom,
                    target_context,
                    start_frac=alignment.dataset_start_frac,
                    end_frac=alignment.dataset_end_frac,
                    exclude_segment_idx=target_filtered_idx,
                    seg_id=target_id,
                    seg_to_connectors=target_seg_to_connectors_ep,
                )

                # Compute graphlet similarity
                graphlet_features = compute_graphlet_similarity(
                    ref_id,
                    target_id,
                    ref_graphlet_data,
                    target_graphlet_data,
                    alignment,
                )

                # Get names and classes
                ref_name = ref_row.get("names") if hasattr(ref_row, "get") else None
                target_name = target_row.get("names") if hasattr(target_row, "get") else None
                ref_class = ref_row.get("class") if hasattr(ref_row, "get") else None
                target_class = target_row.get("class") if hasattr(target_row, "get") else None
                ref_subclass = ref_row.get("subclass") if hasattr(ref_row, "get") else None
                target_subclass = target_row.get("subclass") if hasattr(target_row, "get") else None

                # Compute ALL features using the authoritative function
                features = compute_pair_features(
                    ref_geom,
                    target_geom,
                    ref_name,
                    target_name,
                    ref_class,
                    target_class,
                    ref_subclass,
                    target_subclass,
                    endpoint_features=endpoint_features,
                    alignment=alignment,
                    graphlet_features=graphlet_features,
                    ref_graphlet_data=ref_graphlet_data,
                    target_graphlet_data=target_graphlet_data,
                    ref_seg_id=ref_id,
                    target_seg_id=target_id,
                )

                # Update all feature columns in the dataframe
                # Add new columns if they don't exist (for newly added features)
                for feat_name, feat_value in features.items():
                    if feat_name not in df.columns:
                        df[feat_name] = None  # Initialize new column
                    df.at[idx, feat_name] = feat_value

                # Update alignment fractions
                df.at[idx, "ref_start_pct"] = alignment.overture_start_frac
                df.at[idx, "ref_end_pct"] = alignment.overture_end_frac
                df.at[idx, "target_start_pct"] = alignment.dataset_start_frac
                df.at[idx, "target_end_pct"] = alignment.dataset_end_frac
                df.at[idx, "is_subsegment"] = is_subsegment_selection(
                    alignment.overture_start_frac,
                    alignment.overture_end_frac,
                    alignment.dataset_start_frac,
                    alignment.dataset_end_frac,
                )

                # Persist geometry to companion file (WGS84)
                ref_geom_wgs84 = ref_gdf.geometry.loc[ref_idx_in_gdf]
                target_geom_wgs84 = target_gdf.geometry.loc[target_idx_in_gdf]
                geo_store.add(
                    gers_id=ref_id,
                    target_id=target_id,
                    ref_geometry=ref_geom_wgs84,
                    target_geometry=target_geom_wgs84,
                    ref_name=_extract_name_string(ref_name),
                    target_name=_extract_name_string(target_name),
                    ref_class=str(ref_class) if ref_class is not None else None,
                    target_class=str(target_class) if target_class is not None else None,
                    ref_subclass=str(ref_subclass) if ref_subclass is not None else None,
                    target_subclass=str(target_subclass) if target_subclass is not None else None,
                )

            else:
                # === TIER 2: IDs not in current data — try persisted geometry ===
                persisted = geo_store.get_pair(ref_id, target_id)
                if persisted is None:
                    # === TIER 3: Truly orphaned — no recovery possible ===
                    orphaned_indices.append(idx)
                    if not report_only:
                        logger.debug(
                            f"    Orphaned: {ref_id}/{target_id} - "
                            f"not in current data and no persisted geometry"
                        )
                    continue

                # Recover using persisted WGS84 geometry
                ref_geom_wgs84 = persisted["ref_geometry"]
                target_geom_wgs84 = persisted["target_geometry"]

                if not isinstance(ref_geom_wgs84, LineString) or not isinstance(
                    target_geom_wgs84, LineString
                ):
                    orphaned_indices.append(idx)
                    logger.debug(
                        f"    Orphaned: {ref_id}/{target_id} - persisted geometry is not LineString"
                    )
                    continue

                # Project persisted WGS84 geometries to UTM for feature computation
                import pyproj
                from shapely.ops import transform

                project = pyproj.Transformer.from_crs(
                    "EPSG:4326", utm_crs, always_xy=True
                ).transform
                ref_geom = transform(project, ref_geom_wgs84)
                target_geom = transform(project, target_geom_wgs84)

                if ref_geom.is_empty or target_geom.is_empty:
                    orphaned_indices.append(idx)
                    continue

                # Compute pairwise features from persisted geometry
                # Context features (endpoint, topology, graphlet) cannot be
                # recomputed without the full dataset, so they will get defaults
                ref_name = persisted["ref_name"]
                target_name = persisted["target_name"]
                ref_class = persisted["ref_class"]
                target_class = persisted["target_class"]
                ref_subclass = persisted["ref_subclass"]
                target_subclass = persisted["target_subclass"]

                features = compute_pair_features(
                    ref_geom,
                    target_geom,
                    ref_name,
                    target_name,
                    ref_class,
                    target_class,
                    ref_subclass,
                    target_subclass,
                    # No endpoint/topology/graphlet features — will use defaults
                )

                # Preserve previous CSV values for context features
                # (topology, endpoint, graphlet) that can't be recomputed
                from ..features.compute import _get_error_features

                error_defaults = _get_error_features()
                for feat in CONTEXT_FEATURES:
                    prev_value = row.get(feat)
                    if prev_value is not None and not (
                        isinstance(prev_value, float) and pd.isna(prev_value)
                    ):
                        features[feat] = prev_value
                    else:
                        features[feat] = error_defaults.get(feat, 0.5)

                # Update all feature columns in the dataframe
                for feat_name, feat_value in features.items():
                    if feat_name not in df.columns:
                        df[feat_name] = None
                    df.at[idx, feat_name] = feat_value

                # Alignment fractions: compute from persisted geometry
                alignment = linestring_alignment(ref_geom, target_geom)
                df.at[idx, "ref_start_pct"] = alignment.overture_start_frac
                df.at[idx, "ref_end_pct"] = alignment.overture_end_frac
                df.at[idx, "target_start_pct"] = alignment.dataset_start_frac
                df.at[idx, "target_end_pct"] = alignment.dataset_end_frac
                df.at[idx, "is_subsegment"] = is_subsegment_selection(
                    alignment.overture_start_frac,
                    alignment.overture_end_frac,
                    alignment.dataset_start_frac,
                    alignment.dataset_end_frac,
                )

                recovered += 1
                logger.warning(
                    f"  {ref_id}/{target_id}: IDs not in current data, "
                    f"pairwise features recovered from persisted geometry, "
                    f"context features preserved from previous values"
                )

            # Update version columns (shared by tier 1 and tier 2)
            if "ref_data_version" not in df.columns:
                df["ref_data_version"] = None
            if "target_data_version" not in df.columns:
                df["target_data_version"] = None
            if "feature_version" not in df.columns:
                df["feature_version"] = None

            df.at[idx, "ref_data_version"] = ref_data_version
            df.at[idx, "target_data_version"] = target_data_version
            df.at[idx, "feature_version"] = FEATURE_VERSION

            updated += 1
            if updated % 100 == 0:
                logger.info(f"    Processed {updated}/{original_count} labels...")

        orphaned_count = len(orphaned_indices)
        logger.info(
            f"  Updated {updated}/{original_count} labels "
            f"({recovered} recovered from persisted geometry), "
            f"{orphaned_count} orphaned"
        )

        # Save geometry store once per partition (batch save)
        if not dry_run and not report_only:
            geo_store.save()
            logger.info(f"  Saved geometry store to {geo_store.csv_path}")

        # Store detailed results
        results[dataset_name] = {
            "updated": updated,
            "recovered": recovered,
            "orphaned": orphaned_count,
            "total": original_count,
        }

        # Report mode - just show stats, don't modify
        if report_only:
            if orphaned_count > 0:
                logger.warning(
                    f"  [REPORT] {orphaned_count} labels would be orphaned "
                    f"(IDs not found in current data)"
                )
            continue

        # Drop orphaned labels if requested
        if drop_orphaned and orphaned_indices:
            logger.info(f"  Dropping {orphaned_count} orphaned labels...")
            df = df.drop(orphaned_indices)
            results[dataset_name]["dropped"] = orphaned_count

        # Write back to CSV atomically
        if not dry_run and (updated > 0 or (drop_orphaned and orphaned_indices)):
            temp_path = partition_path.with_suffix(".csv.tmp")
            backup_path = partition_path.with_suffix(".csv.bak")

            # Remove deprecated feature columns (not in current FEATURE_COLUMNS)
            # Keep: gers_id, target_id, label, notes, metadata columns, and current features
            keep_columns = {
                "gers_id",
                "target_id",
                "label",
                "notes",
                "dataset",
                "labeled_at",
                "labeler",
                "session_id",
                "original_confidence",
                "original_decision",
                "ref_start_pct",
                "ref_end_pct",
                "target_start_pct",
                "target_end_pct",
                "is_subsegment",
                "ref_data_version",
                "target_data_version",
                "feature_version",
            } | set(FEATURE_COLUMNS)

            deprecated_cols = [c for c in df.columns if c not in keep_columns]
            if deprecated_cols:
                logger.info(
                    f"  Removing {len(deprecated_cols)} deprecated columns: {deprecated_cols}"
                )
                df = df.drop(columns=deprecated_cols)

            # Write to temp file first
            df.to_csv(temp_path, index=False, float_format=lambda x: f"{x:.10g}")

            # Backup existing file
            if partition_path.exists():
                # Use replace() for cross-platform atomicity
                partition_path.replace(backup_path)

            # Atomic replace
            temp_path.replace(partition_path)
            logger.info(f"  Saved to {partition_path}")

    # Summary
    total_updated = sum(r["updated"] for r in results.values())
    total_recovered = sum(r.get("recovered", 0) for r in results.values())
    total_orphaned = sum(r["orphaned"] for r in results.values())
    logger.info(
        f"\nBackfill complete: {total_updated} labels updated "
        f"({total_recovered} recovered from persisted geometry), "
        f"{total_orphaned} orphaned across {len(results)} datasets"
    )

    return results
