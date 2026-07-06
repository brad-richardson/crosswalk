"""Core types for matching results.

These types are shared across the matching module and used by the ML matcher,
optimizer, and resolution components.
"""

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any


class MatchDecision(Enum):
    """Match decision categories."""

    MATCH = "match"
    REVIEW = "review"
    NO_MATCH = "no_match"


class MatchType(StrEnum):
    """Match group cardinality types."""

    ONE_TO_ONE = "1:1"
    ONE_TO_N = "1:N"
    N_TO_ONE = "N:1"
    M_TO_N = "M:N"


@dataclass
class MatchResult:
    """Result of matching a candidate pair."""

    ref_id: Any
    target_id: Any
    decision: MatchDecision
    confidence: float
    score_breakdown: dict[str, float]
    features: dict[str, float]
    # Positional indices of the scored edge into the reference/target
    # GeoDataFrames. These disambiguate rows that share an id (e.g. an Overture
    # segment split into multiple edges carrying one GERS id) so downstream
    # geometry lookups resolve the exact scored geometry rather than collapsing
    # to a single id-keyed row. Optional: None for results built without them.
    ref_idx: int | None = None
    target_idx: int | None = None
    # Linear reference fields from alignment (optional)
    # These indicate where on each geometry the match alignment starts/ends
    gers_start_frac: float | None = None
    gers_end_frac: float | None = None
    local_start_frac: float | None = None
    local_end_frac: float | None = None
