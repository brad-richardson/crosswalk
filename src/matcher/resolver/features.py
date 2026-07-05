"""Feature engineering for the per-edge keep/drop resolver prototype.

All features derive from the sidecar (per-edge confidence + structural layer +
alignment fractions) and cheap within-group aggregates. No pairwise-feature
parquet dependency (coverage is ~5% for group edges) and no geometry recompute.

The strongest single signal is edge confidence *relative to its group* — a
dropped edge is typically the low-confidence competitor for a shared target/ref
inside a dense over-merged component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Per-edge features used by the prototype classifier. Kept small and
# interpretable given the tiny label scale (~40-60 labeled groups).
FEATURE_COLUMNS: list[str] = [
    # raw confidence + within-group relative confidence
    "confidence",
    "conf_rel_max",
    "conf_rel_mean",
    "conf_rank_frac",
    "conf_is_group_min",
    # alignment spans (coverage-asymmetry aware)
    "gers_span",
    "local_span",
    "max_span",
    "min_span",
    # graph structure
    "degree_ref",
    "degree_tgt",
    "is_bridge",
    "is_sliver",
    # competition context
    "n_share_ref",
    "n_share_tgt",
    # group-level structure
    "n_edges",
    "n_corridors",
    "n_assignment_components",
    "largest_biconnected_block",
    "oversized_group",
    "num_refs",
    "num_targets",
    "match_type_1N",
    "match_type_N1",
    "match_type_MN",
]


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature columns to a raw edge table (see FEATURE_COLUMNS).

    Operates per ``group_id``; returns a copy with feature columns added.
    """
    out = df.copy()

    out["gers_span"] = out["gers_end_frac"] - out["gers_start_frac"]
    out["local_span"] = out["local_end_frac"] - out["local_start_frac"]
    out["max_span"] = out[["gers_span", "local_span"]].max(axis=1)
    out["min_span"] = out[["gers_span", "local_span"]].min(axis=1)

    out["is_bridge"] = out["is_bridge"].astype(int)
    out["is_sliver"] = out["is_sliver"].astype(int)
    out["oversized_group"] = out["oversized_group"].astype(int)

    out["match_type_1N"] = (out["match_type"] == "1:N").astype(int)
    out["match_type_N1"] = (out["match_type"] == "N:1").astype(int)
    out["match_type_MN"] = (out["match_type"] == "M:N").astype(int)

    # within-group relative-confidence + competition features
    grp = out.groupby("group_id")
    gmax = grp["confidence"].transform("max")
    gmean = grp["confidence"].transform("mean")
    gmin = grp["confidence"].transform("min")
    out["conf_rel_max"] = out["confidence"] - gmax
    out["conf_rel_mean"] = out["confidence"] - gmean
    out["conf_is_group_min"] = (out["confidence"] <= gmin + 1e-9).astype(int)
    # rank of confidence within group, 0 = highest, normalized to [0, 1]
    out["conf_rank_frac"] = grp["confidence"].transform(
        lambda s: s.rank(method="min", ascending=False) - 1
    ) / grp["confidence"].transform(lambda s: max(len(s) - 1, 1))

    # competition: how many edges in the group share this edge's ref / target
    out["n_share_ref"] = grp["ref_id"].transform("count")  # placeholder, replaced below
    out["n_share_ref"] = out.groupby(["group_id", "ref_id"])["ref_id"].transform("count")
    out["n_share_tgt"] = out.groupby(["group_id", "target_id"])["target_id"].transform("count")

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the model feature matrix (NaNs passed through to XGBoost)."""
    return df[FEATURE_COLUMNS].to_numpy(dtype=float)
