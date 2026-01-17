"""Decision persistence for integration QA.

Stores QA decisions in CSV format for git-trackable storage.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger


# Default paths
DEFAULT_ORPHAN_PATH = Path("data/labels/integration_orphans.csv")
DEFAULT_MERGED_PATH = Path("data/labels/integration_merged.csv")

# Column definitions
ORPHAN_COLUMNS = [
    "edge_id",
    "original_id",
    "dataset_id",  # Changed from source_dataset for consistency
    "component_id",
    "decision",  # "keep", "discard"
    "reason",  # "legitimate_new", "data_error", "out_of_scope"
    "reviewer",
    "reviewed_at",  # ISO timestamp string
    "session_id",
    # Context features for future ML
    "length_m",
    "road_class",
    "nearest_main_dist_m",
    "component_size",
]

MERGED_COLUMNS = [
    "edge_id",
    "original_id",
    "dataset_id",  # Changed from source_dataset for consistency
    "source_type",  # "target_matched", "target_new"
    "match_ref_id",
    "decision",  # "correct", "incorrect"
    "reason",  # "matching_error", "duplicate", "wrong_source"
    "reviewer",
    "reviewed_at",  # ISO timestamp string
    "session_id",
    # Context features for future ML
    "match_confidence",
    "length_m",
    "road_class",
]


@dataclass
class OrphanDecisionStore:
    """Manages orphan QA decisions."""

    path: Path = DEFAULT_ORPHAN_PATH
    _df: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.path = Path(self.path)
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load dataframe."""
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        """Load decisions from CSV."""
        if self.path.exists():
            try:
                df = pd.read_csv(self.path)
                # Handle source_dataset -> dataset_id rename for backward compatibility
                if "source_dataset" in df.columns and "dataset_id" not in df.columns:
                    df = df.rename(columns={"source_dataset": "dataset_id"})
                return df
            except Exception:
                pass
        return pd.DataFrame(columns=ORPHAN_COLUMNS)

    def _save(self) -> None:
        """Save decisions to CSV."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.path, index=False)

    def add_decision(
        self,
        edge_id: int,
        original_id: str,
        dataset_id: str,
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
            "dataset_id": dataset_id,
            "component_id": component_id,
            "decision": decision,
            "reason": reason,
            "reviewer": reviewer,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
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


@dataclass
class MergedDecisionStore:
    """Manages merged edge QA decisions."""

    path: Path = DEFAULT_MERGED_PATH
    _df: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.path = Path(self.path)
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load dataframe."""
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        """Load decisions from CSV."""
        if self.path.exists():
            try:
                df = pd.read_csv(self.path)
                # Handle source_dataset -> dataset_id rename for backward compatibility
                if "source_dataset" in df.columns and "dataset_id" not in df.columns:
                    df = df.rename(columns={"source_dataset": "dataset_id"})
                return df
            except Exception:
                pass
        return pd.DataFrame(columns=MERGED_COLUMNS)

    def _save(self) -> None:
        """Save decisions to CSV."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.path, index=False)

    def add_decision(
        self,
        edge_id: int,
        original_id: str,
        dataset_id: str,
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
            "dataset_id": dataset_id,
            "source_type": source_type,
            "match_ref_id": match_ref_id,
            "decision": decision,
            "reason": reason,
            "reviewer": reviewer,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
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
