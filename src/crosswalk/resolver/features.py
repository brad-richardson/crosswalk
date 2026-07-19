"""Feature engineering for the per-edge keep/drop resolver prototype.

The default feature contract derives from the sidecar (per-edge confidence +
structural layer + alignment fractions) and cheap within-group aggregates.
Candidate-parquet pairwise families can now be added explicitly by the research
CLI; they remain opt-in so the production comparison and legacy artifacts keep
their original 33-feature contract.

The strongest single signal is edge confidence *relative to its group* — a
dropped edge is typically the low-confidence competitor for a shared target/ref
inside a dense over-merged component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Version string for the resolver's feature contract. This is INDEPENDENT of
# ``crosswalk.config.FEATURE_VERSION`` (which versions the pairwise matcher's
# candidate features). Bump whenever a resolver feature's NAME
# SET *or* SEMANTICS change — i.e. any edit to ``FEATURE_COLUMNS`` below, to the
# ``featurize`` derivations, or to how ``resolver/extract.py`` populates a raw
# column a feature reads from. A saved resolver model stamps this value and
# ``resolver/train.load_model`` refuses to load a model whose stamp differs, so
# a stale feature contract can never silently score.
#
# Format mirrors ``config.FEATURE_VERSION`` (``YYYY-MM-DD.minor``).
#
# Start at ``2026-07-15.1`` (not an initial ``1``/undated value) to acknowledge
# a semantics change that already shipped UNVERSIONED: commit 55caab4
# (2026-07-15) changed ``is_bridge`` in ``resolver/extract.py::_edge_row`` to
# prefer the ``candidate_graph_bridge`` field over the legacy ``is_bridge`` when
# both are present. Any resolver model trained before that commit carries a
# different ``is_bridge`` contract, so this version deliberately starts past it.
RESOLVER_FEATURE_VERSION = "2026-07-15.1"

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


def group_key_columns(df: pd.DataFrame) -> list[str]:
    """Columns that uniquely identify a resolver group in ``df``."""
    return ["dataset_id", "group_id"] if "dataset_id" in df.columns else ["group_id"]


def group_keys(df: pd.DataFrame) -> pd.Series:
    """Stable one-dimensional keys for CV splitting and group-level metrics."""
    if "dataset_id" not in df.columns:
        return df["group_id"].astype(str)
    return df["dataset_id"].astype(str) + "\x1f" + df["group_id"].astype(str)


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
    group_cols = group_key_columns(out)
    grp = out.groupby(group_cols)
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
    out["n_share_ref"] = out.groupby([*group_cols, "ref_id"])["ref_id"].transform("count")
    out["n_share_tgt"] = out.groupby([*group_cols, "target_id"])["target_id"].transform("count")

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the model feature matrix (NaNs passed through to XGBoost)."""
    return df[FEATURE_COLUMNS].to_numpy(dtype=float)
