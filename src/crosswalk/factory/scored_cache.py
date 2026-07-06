"""Scored-candidate cache: (de)serialize ``MatchResult`` lists to parquet.

The factory caches the scored candidates for a dataset so ``crosswalk factory
reoptimize`` can re-run grouping/optimization/sidecar (~2 s) without re-scoring
(~7 min for a large dataset). Scoring is 84% of pipeline wall time, so this is the
biggest iteration win for grouping/optimizer changes.

The cache is validity-keyed by the manifest's ``score_key`` (input fingerprints +
model hash + FEATURE_VERSION + buffer distance); optimizer/prune/export settings
are deliberately EXCLUDED from that key so they can change without invalidating
the scores. Callers must check ``score_key`` before trusting a cache file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..matching.types import MatchDecision, MatchResult

# Bump if the on-disk column layout changes incompatibly.
SCORED_CACHE_SCHEMA_VERSION = 1


def _json_default(obj: Any) -> Any:
    """Cast numpy scalars to native Python for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _clean_idx(val: Any) -> int | None:
    """Coerce a stored positional index back to ``int`` or ``None``."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return int(val)


def _clean_frac(val: Any) -> float | None:
    """Coerce a stored alignment fraction back to ``float`` or ``None``."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return float(val)


def write_scored_cache(results: list[MatchResult], path: Path) -> int:
    """Serialize ``results`` (in list order) to a parquet cache at ``path``.

    ``ref_id`` / ``target_id`` are written natively so a homogeneously-typed id
    column (the norm: GERS strings, H3-suffixed local strings) round-trips its
    exact type. ``features`` / ``score_breakdown`` dicts are JSON-encoded (NaN
    preserved via Python's ``allow_nan``). Returns the number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for r in results:
        records.append(
            {
                "ref_id": r.ref_id,
                "target_id": r.target_id,
                "decision": r.decision.value,
                "confidence": float(r.confidence),
                "ref_idx": r.ref_idx,
                "target_idx": r.target_idx,
                "gers_start_frac": r.gers_start_frac,
                "gers_end_frac": r.gers_end_frac,
                "local_start_frac": r.local_start_frac,
                "local_end_frac": r.local_end_frac,
                "features_json": json.dumps(r.features, default=_json_default),
                "score_breakdown_json": json.dumps(r.score_breakdown, default=_json_default),
            }
        )
    df = pd.DataFrame.from_records(
        records,
        columns=[
            "ref_id",
            "target_id",
            "decision",
            "confidence",
            "ref_idx",
            "target_idx",
            "gers_start_frac",
            "gers_end_frac",
            "local_start_frac",
            "local_end_frac",
            "features_json",
            "score_breakdown_json",
        ],
    )
    # Nullable integer dtype so None positional indices survive the round-trip
    # without being coerced to float NaN and losing exactness.
    df["ref_idx"] = df["ref_idx"].astype("Int64")
    df["target_idx"] = df["target_idx"].astype("Int64")
    df.to_parquet(path, index=False)
    return len(df)


def read_scored_cache(path: Path) -> list[MatchResult]:
    """Load a parquet cache written by :func:`write_scored_cache`.

    Reconstructs ``MatchResult`` objects in stored order (tie-break stability).
    """
    df = pd.read_parquet(path)
    results: list[MatchResult] = []
    for row in df.itertuples(index=False):
        results.append(
            MatchResult(
                ref_id=row.ref_id,
                target_id=row.target_id,
                decision=MatchDecision(row.decision),
                confidence=float(row.confidence),
                score_breakdown=json.loads(row.score_breakdown_json),
                features=json.loads(row.features_json),
                ref_idx=_clean_idx(row.ref_idx),
                target_idx=_clean_idx(row.target_idx),
                gers_start_frac=_clean_frac(row.gers_start_frac),
                gers_end_frac=_clean_frac(row.gers_end_frac),
                local_start_frac=_clean_frac(row.local_start_frac),
                local_end_frac=_clean_frac(row.local_end_frac),
            )
        )
    return results
