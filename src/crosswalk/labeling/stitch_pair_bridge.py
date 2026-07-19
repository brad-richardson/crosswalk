"""Audited bridge from stitch decisions to pair-matcher supervision.

The two tasks do not share a negative complement: a stitch edge can be dropped
for graph context while still representing the same physical feature.  This
module therefore exports only:

* explicit ``edge_dispositions.identity`` values from exact-identity reviews;
* selected stitch edges as weaker positive pair labels; and
* never an unselected edge as ``no_match`` unless identity was explicitly set.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd

from ..config import FEATURE_COLUMNS, FEATURE_VERSION

PAIR_BRIDGE_COLUMNS = [
    "dataset",
    "gers_id",
    "target_id",
    "label",
    "confidence",
    "reasoning",
    "labeler",
    "labeled_at",
    "source_group_id",
    "source_kind",
    "resolution",
]


def _json_list(raw: object) -> list:
    if raw is None or isinstance(raw, float) or not str(raw).strip():
        return []
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return []
    return value if isinstance(value, list) else []


def derive_stitch_pair_labels(
    stitching_labels: pd.DataFrame,
    dataset_id: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Derive safe pair labels and quarantine contradictory exact identities."""
    candidates: list[dict] = []
    stats = {
        "stitch_rows": len(stitching_labels),
        "explicit_identity": 0,
        "weak_selected_positive": 0,
        "unsure_skipped": 0,
        "conflicting_pairs": 0,
    }
    for _, row in stitching_labels.iterrows():
        scope = str(row.get("adjudication_scope") or "")
        dispositions = _json_list(row.get("edge_dispositions"))
        common = {
            "dataset": str(dataset_id),
            "labeler": f"stitch:{row.get('labeler') or 'unknown'}",
            "labeled_at": str(row.get("labeled_at") or ""),
            "source_group_id": str(row.get("group_id") or ""),
        }
        if scope == "exact_identity" and dispositions:
            for edge in dispositions:
                identity = str(edge.get("identity") or "")
                if identity == "unsure":
                    stats["unsure_skipped"] += 1
                    continue
                if identity not in {"match", "no_match"}:
                    continue
                candidates.append(
                    {
                        **common,
                        "gers_id": str(edge.get("ref_id")),
                        "target_id": str(edge.get("target_id")),
                        "label": identity,
                        "confidence": 1.0,
                        "reasoning": "Explicit stitch UI physical-identity adjudication",
                        "source_kind": "stitch_exact_identity",
                        "resolution": str(edge.get("resolution") or ""),
                        "_priority": 2,
                    }
                )
                stats["explicit_identity"] += 1
            continue

        # Selected resolution edges safely imply identity=match.  Their
        # complement does not imply no_match and is deliberately absent.
        for edge in _json_list(row.get("selected_edges")):
            if not isinstance(edge, dict) or not edge.get("ref_id") or not edge.get("target_id"):
                continue
            candidates.append(
                {
                    **common,
                    "gers_id": str(edge["ref_id"]),
                    "target_id": str(edge["target_id"]),
                    "label": "match",
                    "confidence": 0.7,
                    "reasoning": "Selected stitch edge; weak positive identity implication",
                    "source_kind": "stitch_selected_positive",
                    "resolution": "keep",
                    "_priority": 1,
                }
            )
            stats["weak_selected_positive"] += 1

    if not candidates:
        return pd.DataFrame(columns=PAIR_BRIDGE_COLUMNS), stats

    frame = pd.DataFrame(candidates)
    kept: list[pd.Series] = []
    for _, same_pair in frame.groupby(["dataset", "gers_id", "target_id"], sort=True):
        strongest = same_pair[same_pair["_priority"] == same_pair["_priority"].max()]
        if strongest["label"].nunique() > 1:
            stats["conflicting_pairs"] += 1
            continue
        # Stable latest decision within the strongest provenance tier.
        kept.append(strongest.sort_values("labeled_at", kind="stable").iloc[-1])
    if not kept:
        return pd.DataFrame(columns=PAIR_BRIDGE_COLUMNS), stats
    out = pd.DataFrame(kept).drop(columns=["_priority"])
    return out[PAIR_BRIDGE_COLUMNS].reset_index(drop=True), stats


def filter_against_human_truth(
    derived: pd.DataFrame,
    human_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove redundant/conflicting derived rows when pair human truth exists."""
    stats = {"human_redundant": 0, "human_conflicts": 0}
    if derived.empty or human_labels.empty:
        return derived.copy(), stats
    human = human_labels.copy()
    if "gers_id" not in human and "ref_id" in human:
        human = human.rename(columns={"ref_id": "gers_id"})
    truth = {
        (str(row["gers_id"]), str(row["target_id"])): str(row["label"])
        for _, row in human.iterrows()
        if str(row.get("label")) in {"match", "no_match"}
    }
    keep = []
    for idx, row in derived.iterrows():
        existing = truth.get((str(row["gers_id"]), str(row["target_id"])))
        if existing is None:
            keep.append(idx)
        elif existing == row["label"]:
            stats["human_redundant"] += 1
        else:
            stats["human_conflicts"] += 1
    return derived.loc[keep].reset_index(drop=True), stats


def candidate_features_for_pairs(
    candidates: pd.DataFrame,
    pair_keys: Iterable[tuple[str, str]],
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Materialize matcher feature rows from the typed candidate parquet."""
    keys = {(str(ref), str(target)) for ref, target in pair_keys}
    if not keys:
        return pd.DataFrame(
            columns=["gers_id", "target_id", "feature_version", *FEATURE_COLUMNS]
        ), []
    frame = candidates.copy()
    if "gers_id" not in frame and "ref_id" in frame:
        frame = frame.rename(columns={"ref_id": "gers_id"})
    missing_columns = [name for name in FEATURE_COLUMNS if name not in frame]
    if missing_columns:
        raise ValueError(f"candidate parquet missing matcher features: {missing_columns}")
    mask = [
        (str(ref), str(target)) in keys
        for ref, target in zip(frame["gers_id"], frame["target_id"], strict=False)
    ]
    selected = frame.loc[mask, ["gers_id", "target_id", *FEATURE_COLUMNS]].copy()
    selected["gers_id"] = selected["gers_id"].astype(str)
    selected["target_id"] = selected["target_id"].astype(str)
    selected = selected.drop_duplicates(["gers_id", "target_id"], keep="last")
    selected.insert(2, "feature_version", FEATURE_VERSION)
    found = set(zip(selected["gers_id"], selected["target_id"], strict=False))
    return selected.reset_index(drop=True), sorted(keys - found)
