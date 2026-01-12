"""Label persistence - parquet I/O for labeled training data."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


LABELS_SCHEMA = pa.schema([
    ("ref_id", pa.string()),
    ("target_id", pa.string()),
    ("label", pa.string()),  # "match", "no_match", "skip"
    ("labeler", pa.string()),
    ("labeled_at", pa.timestamp("us", tz="UTC")),
    ("session_id", pa.string()),
    ("original_decision", pa.string()),  # From rule-based matcher
    ("original_confidence", pa.float64()),
    # Geometric features
    ("hausdorff_distance", pa.float64()),
    ("frechet_distance", pa.float64()),
    ("buffer_iou", pa.float64()),
    ("heading_delta", pa.float64()),
    ("length_ratio", pa.float64()),
    ("projection_distance", pa.float64()),
    ("centroid_distance", pa.float64()),
    # Semantic features
    ("name_levenshtein", pa.float64()),
    ("name_jaro_winkler", pa.float64()),
    ("name_token_sort", pa.float64()),
    ("class_similarity", pa.float64()),
])


@dataclass
class LabelStore:
    """Manages labeled data storage."""

    path: Path
    _df: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.path = Path(self.path)
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load dataframe."""
        if self._df is None:
            self._df = load_labels(self.path)
        return self._df

    def add(
        self,
        ref_id: str,
        target_id: str,
        label: str,
        labeler: str,
        session_id: str,
        original_decision: str,
        original_confidence: float,
        features: dict[str, float],
    ) -> None:
        """Add a new label."""
        self._df = add_label(
            self.df,
            ref_id=ref_id,
            target_id=target_id,
            label=label,
            labeler=labeler,
            session_id=session_id,
            original_decision=original_decision,
            original_confidence=original_confidence,
            features=features,
        )
        self.save()

    def save(self) -> None:
        """Save labels to disk."""
        save_labels(self._df, self.path)

    def get_labeled_pairs(self, labeler: Optional[str] = None) -> set[tuple[str, str]]:
        """Get set of already-labeled (ref_id, target_id) pairs.

        Args:
            labeler: If provided, only return pairs labeled by this labeler.
                    If None, returns all labeled pairs.
        """
        # Use self.df to trigger lazy load
        df = self.df
        if df is None or len(df) == 0:
            return set()

        if labeler:
            df = df[df["labeler"] == labeler]

        if len(df) == 0:
            return set()
        return set(zip(df["ref_id"], df["target_id"]))

    def get_stats(self) -> dict[str, Any]:
        """Get labeling statistics."""
        # Use self.df to trigger lazy load
        df = self.df
        if df is None or len(df) == 0:
            return {"total": 0, "match": 0, "no_match": 0, "skip": 0}

        return {
            "total": len(df),
            "match": (df["label"].str.contains("match", na=False)).sum(),
            "no_match": (df["label"] == "no_match").sum(),
            "skip": (df["label"] == "skip").sum(),
            "associated": (df["label"] == "associated").sum(),
            "unsure": (df["label"] == "unsure").sum(),
        }

    def remove_last(self) -> Optional[dict]:
        """Remove the last label (for undo). Returns removed row or None."""
        # Use self.df to trigger lazy load
        df = self.df
        if df is None or len(df) == 0:
            return None

        last_row = df.iloc[-1].to_dict()
        self._df = df.iloc[:-1].reset_index(drop=True)
        self.save()
        return last_row


def load_labels(path: Path) -> pd.DataFrame:
    """Load existing labels from parquet file.

    Creates empty DataFrame with correct schema if file doesn't exist.
    """
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    # Return empty DataFrame with schema
    return pd.DataFrame({
        col.name: pd.Series(dtype=_pa_to_pd_dtype(col.type))
        for col in LABELS_SCHEMA
    })


def save_labels(df: pd.DataFrame, path: Path) -> None:
    """Save labels to parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure labeled_at is timezone-aware
    if "labeled_at" in df.columns and len(df) > 0:
        if df["labeled_at"].dt.tz is None:
            df["labeled_at"] = df["labeled_at"].dt.tz_localize("UTC")

    df.to_parquet(path, index=False)


def add_label(
    df: pd.DataFrame,
    ref_id: str,
    target_id: str,
    label: str,
    labeler: str,
    session_id: str,
    original_decision: str,
    original_confidence: float,
    features: dict[str, float],
) -> pd.DataFrame:
    """Add a new label to the dataframe."""
    new_row = {
        "ref_id": str(ref_id),
        "target_id": str(target_id),
        "label": label,
        "labeler": labeler,
        "labeled_at": datetime.now(timezone.utc),
        "session_id": session_id,
        "original_decision": original_decision,
        "original_confidence": original_confidence,
        "hausdorff_distance": features.get("hausdorff_distance", 0.0),
        "frechet_distance": features.get("frechet_distance", 0.0),
        "buffer_iou": features.get("buffer_iou", 0.0),
        "heading_delta": features.get("heading_delta", 0.0),
        "length_ratio": features.get("length_ratio", 0.0),
        "projection_distance": features.get("projection_distance", 0.0),
        "centroid_distance": features.get("centroid_distance", 0.0),
        "name_levenshtein": features.get("name_levenshtein", 0.0),
        "name_jaro_winkler": features.get("name_jaro_winkler", 0.0),
        "name_token_sort": features.get("name_token_sort", 0.0),
        "class_similarity": features.get("class_similarity", 0.0),
    }

    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def _pa_to_pd_dtype(pa_type):
    """Convert PyArrow type to pandas dtype."""
    if pa.types.is_string(pa_type):
        return "object"
    if pa.types.is_float64(pa_type):
        return "float64"
    if pa.types.is_timestamp(pa_type):
        return "datetime64[ns, UTC]"
    return "object"
