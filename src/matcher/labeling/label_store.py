"""Label persistence - Hive-partitioned CSV storage for labeled training data."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as pa_ds
from loguru import logger

from .subsegment import is_subsegment_selection

# Default paths
DEFAULT_LABELS_DIR = Path("labels")

# Column definitions for labels
LABEL_COLUMNS = [
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
    # Geometric features (9)
    "hausdorff_distance",
    "mean_hausdorff_distance",
    "buffer_iou",
    "overlap_ratio",
    "heading_delta",
    "length_ratio",
    "projection_distance",
    "centroid_distance",
    "collinear_gap_ratio",
    # Semantic features - name (5)
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_soundex",
    "name_metaphone",
    # Semantic features - class (1)
    "class_similarity",
    # Endpoint/connectivity features (3)
    "start_endpoint_proximity",
    "end_endpoint_proximity",
    "shared_endpoint_count",
    # Lateral offset features (2)
    "lateral_offset",
    "lateral_offset_consistency",
    # Topology features (12)
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
    # Alignment coverage features (4)
    "ref_coverage",
    "target_coverage",
    "min_coverage",
    "coverage_ratio",
]

# Default values for sub-segment columns (for backward compatibility)
SUBSEGMENT_DEFAULTS = {
    "ref_start_pct": 0.0,
    "ref_end_pct": 1.0,
    "target_start_pct": 0.0,
    "target_end_pct": 1.0,
    "is_subsegment": False,
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
        """Load labels from CSV."""
        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path)
                # Handle backward compatibility - add missing columns
                for col, default_val in SUBSEGMENT_DEFAULTS.items():
                    if col not in df.columns:
                        df[col] = default_val
                # Handle ref_id -> gers_id rename for backward compatibility
                if "ref_id" in df.columns and "gers_id" not in df.columns:
                    df = df.rename(columns={"ref_id": "gers_id"})
                return df
            except Exception as e:
                logger.warning(f"Failed to load labels from {self.csv_path}: {e}")
        return self._empty_dataframe()

    def _empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with correct schema."""
        return pd.DataFrame(columns=LABEL_COLUMNS)

    def save(self) -> None:
        """Save labels to CSV."""
        self.partition_path.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.csv_path, index=False)

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
            # Geometric features (9)
            "hausdorff_distance": features.get("hausdorff_distance", 0.0),
            "mean_hausdorff_distance": features.get("mean_hausdorff_distance", 0.0),
            "buffer_iou": features.get("buffer_iou", 0.0),
            "overlap_ratio": features.get("overlap_ratio", 0.0),
            "heading_delta": features.get("heading_delta", 0.0),
            "length_ratio": features.get("length_ratio", 0.0),
            "projection_distance": features.get("projection_distance", 0.0),
            "centroid_distance": features.get("centroid_distance", 0.0),
            "collinear_gap_ratio": features.get("collinear_gap_ratio", 1.0),
            # Semantic features - name (5)
            "name_levenshtein": features.get("name_levenshtein", 0.0),
            "name_jaro_winkler": features.get("name_jaro_winkler", 0.0),
            "name_token_sort": features.get("name_token_sort", 0.0),
            "name_soundex": features.get("name_soundex", 0.5),
            "name_metaphone": features.get("name_metaphone", 0.5),
            # Semantic features - class (1)
            "class_similarity": features.get("class_similarity", 0.0),
            # Endpoint/connectivity features (3)
            "start_endpoint_proximity": features.get("start_endpoint_proximity", 0.0),
            "end_endpoint_proximity": features.get("end_endpoint_proximity", 0.0),
            "shared_endpoint_count": features.get("shared_endpoint_count", 0),
            # Lateral offset features (2)
            "lateral_offset": features.get("lateral_offset", 0.0),
            "lateral_offset_consistency": features.get("lateral_offset_consistency", 0.0),
            # Topology features (12)
            "from_degree_ref": features.get("from_degree_ref", 0),
            "to_degree_ref": features.get("to_degree_ref", 0),
            "from_degree_target": features.get("from_degree_target", 0),
            "to_degree_target": features.get("to_degree_target", 0),
            "degree_match_score": features.get("degree_match_score", 0.0),
            "degree_signature_similarity": features.get("degree_signature_similarity", 0.0),
            "is_dead_end_ref": features.get("is_dead_end_ref", 0.0),
            "is_dead_end_target": features.get("is_dead_end_target", 0.0),
            "dead_end_match": features.get("dead_end_match", 0.0),
            "is_intersection_ref": features.get("is_intersection_ref", 0.0),
            "is_intersection_target": features.get("is_intersection_target", 0.0),
            "intersection_match": features.get("intersection_match", 0.0),
            # Alignment coverage features (4)
            "ref_coverage": features.get("ref_coverage", 0.0),
            "target_coverage": features.get("target_coverage", 0.0),
            "min_coverage": features.get("min_coverage", 0.0),
            "coverage_ratio": features.get("coverage_ratio", 0.0),
        }

        self._df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.save()

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
        return set(zip(df["gers_id"], df["target_id"]))

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
    def load_all(labels_dir: Path = DEFAULT_LABELS_DIR) -> pd.DataFrame:
        """Load all label partitions for ML training.

        Uses PyArrow's Hive partitioning to read all dataset partitions
        and adds a 'dataset' column from the partition path.

        Returns:
            DataFrame with all labels and 'dataset' column
        """
        labels_dir = Path(labels_dir)
        if not labels_dir.exists():
            return pd.DataFrame(columns=LABEL_COLUMNS + ["dataset"])

        try:
            dataset = pa_ds.dataset(
                labels_dir,
                format="csv",
                partitioning="hive",
            )
            df = dataset.to_table().to_pandas()
            # Handle ref_id -> gers_id rename for backward compatibility
            if "ref_id" in df.columns and "gers_id" not in df.columns:
                df = df.rename(columns={"ref_id": "gers_id"})
            return df
        except Exception as e:
            # Fallback: manual loading if pyarrow dataset fails
            logger.warning(f"PyArrow dataset loading failed, using manual fallback: {e}")
            dfs = []
            for partition_dir in labels_dir.glob("dataset=*"):
                dataset_id = partition_dir.name.split("=")[1]
                csv_path = partition_dir / "data.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df["dataset"] = dataset_id
                    dfs.append(df)
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
    df.to_csv(path, index=False)
