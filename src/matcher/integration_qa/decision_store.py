"""Decision persistence for integration QA.

Stores QA decisions in parquet format for training data collection.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
from loguru import logger


# Schema for orphan decisions
ORPHAN_DECISION_SCHEMA = pa.schema([
    ("edge_id", pa.int64()),
    ("original_id", pa.string()),
    ("source_dataset", pa.string()),
    ("component_id", pa.int64()),
    ("decision", pa.string()),  # "keep", "discard"
    ("reason", pa.string()),  # "legitimate_new", "data_error", "out_of_scope"
    ("reviewer", pa.string()),
    ("reviewed_at", pa.timestamp("us", tz="UTC")),
    ("session_id", pa.string()),
    # Context features for future ML
    ("length_m", pa.float64()),
    ("road_class", pa.string()),
    ("nearest_main_dist_m", pa.float64()),
    ("component_size", pa.int64()),
])

# Schema for merged edge decisions
MERGED_DECISION_SCHEMA = pa.schema([
    ("edge_id", pa.int64()),
    ("original_id", pa.string()),
    ("source_dataset", pa.string()),
    ("source_type", pa.string()),  # "target_matched", "target_new"
    ("match_ref_id", pa.string()),
    ("decision", pa.string()),  # "correct", "incorrect"
    ("reason", pa.string()),  # "matching_error", "duplicate", "wrong_source"
    ("reviewer", pa.string()),
    ("reviewed_at", pa.timestamp("us", tz="UTC")),
    ("session_id", pa.string()),
    # Context features for future ML
    ("match_confidence", pa.float64()),
    ("length_m", pa.float64()),
    ("road_class", pa.string()),
])


@dataclass
class OrphanDecisionStore:
    """Manages orphan QA decisions."""

    path: Path
    _df: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.path = Path(self.path)
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load dataframe."""
        if self._df is None:
            self._df = _load_orphan_decisions(self.path)
        return self._df

    def add_decision(
        self,
        edge_id: int,
        original_id: str,
        source_dataset: str,
        component_id: int,
        decision: str,
        reason: str,
        reviewer: str,
        session_id: str,
        length_m: float = 0.0,
        road_class: str = "",
        nearest_main_dist_m: float = 0.0,
        component_size: int = 0,
    ) -> None:
        """Add a new orphan decision."""
        new_row = {
            "edge_id": edge_id,
            "original_id": str(original_id),
            "source_dataset": source_dataset,
            "component_id": component_id,
            "decision": decision,
            "reason": reason,
            "reviewer": reviewer,
            "reviewed_at": datetime.now(timezone.utc),
            "session_id": session_id,
            "length_m": length_m,
            "road_class": road_class,
            "nearest_main_dist_m": nearest_main_dist_m,
            "component_size": component_size,
        }
        self._df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self._save()

    def get_reviewed_edges(self, reviewer: Optional[str] = None) -> set[int]:
        """Get set of already-reviewed edge IDs."""
        df = self.df
        if df is None or len(df) == 0:
            return set()

        if reviewer:
            df = df[df["reviewer"] == reviewer]

        return set(df["edge_id"].tolist())

    def get_stats(self) -> dict[str, Any]:
        """Get decision statistics."""
        df = self.df
        if df is None or len(df) == 0:
            return {"total": 0, "keep": 0, "discard": 0}

        return {
            "total": len(df),
            "keep": (df["decision"] == "keep").sum(),
            "discard": (df["decision"] == "discard").sum(),
        }

    def remove_last(self) -> Optional[dict]:
        """Remove the last decision (for undo)."""
        df = self.df
        if df is None or len(df) == 0:
            return None

        last_row = df.iloc[-1].to_dict()
        self._df = df.iloc[:-1].reset_index(drop=True)
        self._save()
        return last_row

    def _save(self) -> None:
        """Save decisions to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if "reviewed_at" in self._df.columns and len(self._df) > 0:
            if self._df["reviewed_at"].dt.tz is None:
                self._df["reviewed_at"] = self._df["reviewed_at"].dt.tz_localize("UTC")
        self._df.to_parquet(self.path, index=False)


@dataclass
class MergedDecisionStore:
    """Manages merged edge QA decisions."""

    path: Path
    _df: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.path = Path(self.path)
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load dataframe."""
        if self._df is None:
            self._df = _load_merged_decisions(self.path)
        return self._df

    def add_decision(
        self,
        edge_id: int,
        original_id: str,
        source_dataset: str,
        source_type: str,
        match_ref_id: Optional[str],
        decision: str,
        reason: str,
        reviewer: str,
        session_id: str,
        match_confidence: float = 0.0,
        length_m: float = 0.0,
        road_class: str = "",
    ) -> None:
        """Add a new merged edge decision."""
        new_row = {
            "edge_id": edge_id,
            "original_id": str(original_id),
            "source_dataset": source_dataset,
            "source_type": source_type,
            "match_ref_id": match_ref_id,
            "decision": decision,
            "reason": reason,
            "reviewer": reviewer,
            "reviewed_at": datetime.now(timezone.utc),
            "session_id": session_id,
            "match_confidence": match_confidence,
            "length_m": length_m,
            "road_class": road_class,
        }
        self._df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self._save()

    def get_reviewed_edges(self, reviewer: Optional[str] = None) -> set[int]:
        """Get set of already-reviewed edge IDs."""
        df = self.df
        if df is None or len(df) == 0:
            return set()

        if reviewer:
            df = df[df["reviewer"] == reviewer]

        return set(df["edge_id"].tolist())

    def get_stats(self) -> dict[str, Any]:
        """Get decision statistics."""
        df = self.df
        if df is None or len(df) == 0:
            return {"total": 0, "correct": 0, "incorrect": 0}

        return {
            "total": len(df),
            "correct": (df["decision"] == "correct").sum(),
            "incorrect": (df["decision"] == "incorrect").sum(),
        }

    def remove_last(self) -> Optional[dict]:
        """Remove the last decision (for undo)."""
        df = self.df
        if df is None or len(df) == 0:
            return None

        last_row = df.iloc[-1].to_dict()
        self._df = df.iloc[:-1].reset_index(drop=True)
        self._save()
        return last_row

    def _save(self) -> None:
        """Save decisions to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if "reviewed_at" in self._df.columns and len(self._df) > 0:
            if self._df["reviewed_at"].dt.tz is None:
                self._df["reviewed_at"] = self._df["reviewed_at"].dt.tz_localize("UTC")
        self._df.to_parquet(self.path, index=False)


def _load_orphan_decisions(path: Path) -> pd.DataFrame:
    """Load existing orphan decisions."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    # Return empty DataFrame with schema
    return pd.DataFrame({
        col.name: pd.Series(dtype=_pa_to_pd_dtype(col.type))
        for col in ORPHAN_DECISION_SCHEMA
    })


def _load_merged_decisions(path: Path) -> pd.DataFrame:
    """Load existing merged edge decisions."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    # Return empty DataFrame with schema
    return pd.DataFrame({
        col.name: pd.Series(dtype=_pa_to_pd_dtype(col.type))
        for col in MERGED_DECISION_SCHEMA
    })


def _pa_to_pd_dtype(pa_type):
    """Convert PyArrow type to pandas dtype."""
    if pa.types.is_string(pa_type):
        return "object"
    if pa.types.is_float64(pa_type):
        return "float64"
    if pa.types.is_int64(pa_type):
        return "Int64"  # Nullable integer
    if pa.types.is_timestamp(pa_type):
        return "datetime64[ns, UTC]"
    return "object"
