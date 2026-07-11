"""Offline evaluation helpers for optimizer prune heuristics.

This module deliberately operates on persisted resolver edge tables rather than
inside the production optimizer.  A policy comparison is only meaningful when
every policy sees the same recovered human labels and candidate universe.
Grouping/glue experiments can change that universe, so their coverage must be
compared separately before their quality metrics are interpreted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EDGE_KEY = ("dataset_id", "group_id", "ref_id", "target_id")
EVAL_GROUP_KEY = ("dataset_id", "human_group_id")


@dataclass(frozen=True)
class PrunePolicy:
    """A post-optimizer confidence-prune policy used by the offline harness."""

    name: str
    threshold: float | None = None
    per_type_thresholds: Mapping[str, float] = field(default_factory=dict)
    margin: float | None = None
    preserve_bridge_backbone: bool = False


def reconstruct_preprune_selection(df: pd.DataFrame) -> pd.Series:
    """Recover the selected assignment immediately before confidence pruning.

    Fresh sidecars mark confidence-demoted edges with ``pruned=True``.  Their
    post-prune ``selected`` flag is false, so the pre-prune assignment is the
    union of those two flags.  Legacy sidecars have no ``pruned`` column and
    therefore fall back honestly to the persisted selection.
    """
    selected = df["selected"].fillna(False).astype(bool)
    if "pruned" not in df:
        return selected
    return selected | df["pruned"].fillna(False).astype(bool)


def _normalized_match_type(value: object) -> str:
    text = str(value)
    # MatchType enum stringification differs across callers; tolerate both.
    return (
        text.rsplit(".", 1)[-1]
        .replace("ONE_TO_ONE", "1:1")
        .replace("ONE_TO_N", "1:N")
        .replace("N_TO_ONE", "N:1")
        .replace("M_TO_N", "M:N")
    )


def apply_prune_policy(df: pd.DataFrame, policy: PrunePolicy) -> pd.Series:
    """Apply ``policy`` to a fixed edge table and return a selected-row mask.

    The behavior mirrors ``optimizer.apply_confidence_drop_prune``:

    * only edges in the reconstructed pre-prune assignment are considered;
    * 1:1 edges are untouched;
    * the first highest-confidence edge in every sidecar group is retained;
    * absolute and relative-margin drops combine with OR semantics; and
    * the optional bridge guard preserves an edge only when it is a graph bridge
      and both bipartite endpoint degrees equal one.

    Duplicate rows can occur when multiple historical labels recover to the same
    current group.  Decisions are therefore made once per candidate edge and
    joined back, so duplicated evaluation rows cannot receive different results.
    """
    missing = [column for column in EDGE_KEY if column not in df]
    if missing:
        raise ValueError(f"edge table is missing key columns: {missing}")

    work = df.copy()
    work["_preselected"] = reconstruct_preprune_selection(work).to_numpy()
    edges = work.drop_duplicates(list(EDGE_KEY), keep="first").copy()
    edges["_drop"] = False

    eligible = edges["_preselected"] & edges["group_id"].notna()
    if "match_type" in edges:
        match_types = edges["match_type"].map(_normalized_match_type)
        eligible &= match_types != "1:1"
    else:
        match_types = pd.Series("", index=edges.index)

    if policy.threshold is not None or policy.per_type_thresholds:
        thresholds = pd.Series(policy.threshold, index=edges.index, dtype=float)
        for match_type, threshold in policy.per_type_thresholds.items():
            thresholds.loc[match_types == match_type] = threshold
        has_threshold = thresholds.notna()
        edges.loc[eligible & has_threshold & (edges["confidence"] < thresholds), "_drop"] = True

    if policy.margin is not None:
        group_max = edges.groupby(["dataset_id", "group_id"], dropna=False)["confidence"].transform(
            "max"
        )
        edges.loc[eligible & (edges["confidence"] < group_max - policy.margin), "_drop"] = True

    if policy.preserve_bridge_backbone:
        is_bridge = (
            edges["is_bridge"] if "is_bridge" in edges else pd.Series(False, index=edges.index)
        )
        degree_ref = (
            edges["degree_ref"] if "degree_ref" in edges else pd.Series(0, index=edges.index)
        )
        degree_tgt = (
            edges["degree_tgt"] if "degree_tgt" in edges else pd.Series(0, index=edges.index)
        )
        bridge_backbone = (
            is_bridge.fillna(False).astype(bool)
            & (degree_ref.fillna(0) == 1)
            & (degree_tgt.fillna(0) == 1)
        )
        edges.loc[bridge_backbone, "_drop"] = False

    # Production never empties a group. idxmax is stable and keeps the first
    # occurrence on tied maxima, matching the optimizer's deterministic max().
    selected_edges = edges[edges["_preselected"]]
    if not selected_edges.empty:
        top_idx = selected_edges.groupby(["dataset_id", "group_id"], dropna=False, sort=False)[
            "confidence"
        ].idxmax()
        edges.loc[top_idx, "_drop"] = False

    decisions = (
        edges.set_index(list(EDGE_KEY))["_preselected"] & ~edges.set_index(list(EDGE_KEY))["_drop"]
    )
    row_index = pd.MultiIndex.from_frame(work[list(EDGE_KEY)])
    return pd.Series(decisions.reindex(row_index).fillna(False).to_numpy(), index=df.index)


def _prf(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    pred = prediction.astype(bool)
    actual = truth.astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_predictions(
    df: pd.DataFrame,
    prediction: pd.Series | np.ndarray,
    *,
    policy: str,
    slice_name: str,
) -> dict[str, object]:
    """Return edge and exact-group metrics for one fixed-universe slice."""
    pred = np.asarray(prediction, dtype=bool)
    truth = df["keep"].to_numpy(dtype=bool)
    precision, recall, f1 = _prf(pred, truth)

    exact_frame = df.loc[:, list(EVAL_GROUP_KEY)].copy()
    exact_frame["correct"] = pred == truth
    exact = exact_frame.groupby(list(EVAL_GROUP_KEY), dropna=False)["correct"].all()

    return {
        "policy": policy,
        "slice": slice_name,
        "edges": len(df),
        "groups": int(exact.size),
        "truth_keep": int(truth.sum()),
        "predicted_keep": int(pred.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "group_exact": float(exact.mean()) if len(exact) else 0.0,
    }


def evaluation_slices(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the standard pooled, provenance, match-type, and dataset slices."""
    slices = {"all": pd.Series(True, index=df.index)}
    if "provenance" in df:
        for provenance in sorted(df["provenance"].dropna().unique()):
            slices[f"provenance={provenance}"] = df["provenance"] == provenance
    if "match_type" in df:
        normalized = df["match_type"].map(_normalized_match_type)
        for match_type in sorted(normalized.dropna().unique()):
            slices[f"match_type={match_type}"] = normalized == match_type
    for dataset in sorted(df["dataset_id"].dropna().unique()):
        slices[f"dataset={dataset}"] = df["dataset_id"] == dataset
    return slices


def evaluate_policies(df: pd.DataFrame, policies: list[PrunePolicy]) -> pd.DataFrame:
    """Evaluate persisted production selection plus policies on fixed slices."""
    if df.empty:
        return pd.DataFrame()
    predictions: dict[str, pd.Series] = {
        "optimizer_current": df["selected"].fillna(False).astype(bool),
    }
    predictions.update({policy.name: apply_prune_policy(df, policy) for policy in policies})

    rows: list[dict[str, object]] = []
    for slice_name, mask in evaluation_slices(df).items():
        sub = df.loc[mask]
        for name, prediction in predictions.items():
            rows.append(
                evaluate_predictions(
                    sub,
                    prediction.loc[mask],
                    policy=name,
                    slice_name=slice_name,
                )
            )
    result = pd.DataFrame(rows)
    baseline = result[result["policy"] == "optimizer_current"].set_index("slice")
    result["f1_delta"] = result.apply(
        lambda row: row["f1"] - baseline.loc[row["slice"], "f1"], axis=1
    )
    result["group_exact_delta"] = result.apply(
        lambda row: row["group_exact"] - baseline.loc[row["slice"], "group_exact"], axis=1
    )
    return result
