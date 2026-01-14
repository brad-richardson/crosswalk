"""Label persistence - parquet I/O for labeled training data."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa


LABELS_SCHEMA = pa.schema([
    ("ref_id", pa.string()),
    ("target_id", pa.string()),
    ("label", pa.string()),  # "match", "no_match", "associated", "unsure"
    ("labeler", pa.string()),
    ("labeled_at", pa.timestamp("us", tz="UTC")),
    ("session_id", pa.string()),
    ("original_decision", pa.string()),  # From rule-based matcher
    ("original_confidence", pa.float64()),
    # Sub-segment linear referencing (0.0-1.0 percentages)
    ("ref_start_pct", pa.float64()),  # 0.0 = start of reference line
    ("ref_end_pct", pa.float64()),  # 1.0 = end of reference line
    ("target_start_pct", pa.float64()),  # 0.0 = start of target line
    ("target_end_pct", pa.float64()),  # 1.0 = end of target line
    ("is_subsegment", pa.bool_()),  # True if not whole segment match
    # Geometric features
    ("hausdorff_distance", pa.float64()),
    ("mean_hausdorff_distance", pa.float64()),  # Robust to segmentation differences
    ("buffer_iou", pa.float64()),
    ("overlap_ratio", pa.float64()),  # Fraction of line_a in line_b's buffer
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
        ref_start_pct: float = 0.0,
        ref_end_pct: float = 1.0,
        target_start_pct: float = 0.0,
        target_end_pct: float = 1.0,
    ) -> None:
        """Add a new label.

        Args:
            ref_id: Reference segment ID
            target_id: Target segment ID
            label: Label value (match, no_match, associated, unsure)
            labeler: Name of the labeler
            session_id: Unique session identifier
            original_decision: Rule-based matcher decision
            original_confidence: Rule-based matcher confidence
            features: Dict of feature values
            ref_start_pct: Start of reference sub-segment (0.0-1.0, default 0.0)
            ref_end_pct: End of reference sub-segment (0.0-1.0, default 1.0)
            target_start_pct: Start of target sub-segment (0.0-1.0, default 0.0)
            target_end_pct: End of target sub-segment (0.0-1.0, default 1.0)
        """
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
            ref_start_pct=ref_start_pct,
            ref_end_pct=ref_end_pct,
            target_start_pct=target_start_pct,
            target_end_pct=target_end_pct,
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
            "match": (df["label"] == "match").sum(),
            "no_match": (df["label"] == "no_match").sum(),
            "skip": (df["label"] == "skip").sum(),
            "associated": (df["label"] == "associated").sum(),
            "unsure": (df["label"] == "unsure").sum(),
        }

    def remove_last(self) -> Optional[dict]:
        """Remove the last label (for undo). Returns removed row or None.

        NOTE: This is designed for single-user/single-session use. It removes the
        last row in the file, which may not be the user's own label if multiple
        labelers share the file concurrently. For multi-user scenarios, implement
        undo by tracking specific (ref_id, target_id, labeler) tuples instead.
        """
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
    Handles backward compatibility by adding default values for new columns.
    """
    if path.exists():
        try:
            df = pd.read_parquet(path)
            # Add missing sub-segment columns with defaults for backward compatibility
            for col, default_val in SUBSEGMENT_DEFAULTS.items():
                if col not in df.columns:
                    df[col] = default_val
            return df
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
    ref_start_pct: float = 0.0,
    ref_end_pct: float = 1.0,
    target_start_pct: float = 0.0,
    target_end_pct: float = 1.0,
) -> pd.DataFrame:
    """Add a new label to the dataframe.

    Args:
        df: Existing labels DataFrame
        ref_id: Reference segment ID
        target_id: Target segment ID
        label: Label value
        labeler: Name of labeler
        session_id: Session identifier
        original_decision: Rule-based decision
        original_confidence: Rule-based confidence
        features: Feature values dict
        ref_start_pct: Reference sub-segment start (0.0-1.0)
        ref_end_pct: Reference sub-segment end (0.0-1.0)
        target_start_pct: Target sub-segment start (0.0-1.0)
        target_end_pct: Target sub-segment end (0.0-1.0)

    Returns:
        Updated DataFrame with new label appended
    """
    # Determine if this is a sub-segment selection
    is_subsegment = not (
        abs(ref_start_pct) < 0.001
        and abs(ref_end_pct - 1.0) < 0.001
        and abs(target_start_pct) < 0.001
        and abs(target_end_pct - 1.0) < 0.001
    )

    new_row = {
        "ref_id": str(ref_id),
        "target_id": str(target_id),
        "label": label,
        "labeler": labeler,
        "labeled_at": datetime.now(timezone.utc),
        "session_id": session_id,
        "original_decision": original_decision,
        "original_confidence": original_confidence,
        # Sub-segment fields
        "ref_start_pct": ref_start_pct,
        "ref_end_pct": ref_end_pct,
        "target_start_pct": target_start_pct,
        "target_end_pct": target_end_pct,
        "is_subsegment": is_subsegment,
        # Geometric features
        "hausdorff_distance": features.get("hausdorff_distance", 0.0),
        "mean_hausdorff_distance": features.get("mean_hausdorff_distance", 0.0),
        "buffer_iou": features.get("buffer_iou", 0.0),
        "overlap_ratio": features.get("overlap_ratio", 0.0),
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
    if pa.types.is_boolean(pa_type):
        return "bool"
    return "object"
