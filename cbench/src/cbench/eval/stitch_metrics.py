"""Stitch-level evaluation metrics.

Compares bridge output against curated stitching labels to measure
whether the optimizer selects the correct edges within M:N groups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd


@dataclass
class StitchEvalResult:
    """Metrics from evaluating bridge edges against stitching labels."""

    groups_evaluated: int
    precision: float
    recall: float
    f1: float
    total_curated_edges: int
    total_extra_edges: int

    def to_dict(self) -> dict:
        return {
            "groups_evaluated": self.groups_evaluated,
            "stitch_precision": self.precision,
            "stitch_recall": self.recall,
            "stitch_f1": self.f1,
            "total_curated_edges": self.total_curated_edges,
            "total_extra_edges": self.total_extra_edges,
        }


def evaluate_stitch_groups(
    bridge: pd.DataFrame,
    stitch_labels: pd.DataFrame,
) -> StitchEvalResult:
    """Compare bridge output against curated stitching labels.

    For each labeled group:
    1. Extract curated edges from selected_edges JSON
    2. Find all bridge edges involving the group's segment IDs
    3. Compute edge precision and recall

    Args:
        bridge: DataFrame with columns [ref_id, target_id, confidence].
        stitch_labels: DataFrame with columns [group_id, selected_edges, ...].

    Returns:
        StitchEvalResult with aggregate and per-group metrics.
    """
    if stitch_labels.empty:
        return StitchEvalResult(
            groups_evaluated=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            total_curated_edges=0,
            total_extra_edges=0,
        )

    bridge_edges = set(
        zip(bridge["ref_id"].astype(str), bridge["target_id"].astype(str))
    )

    precisions = []
    recalls = []
    total_curated = 0
    total_extra = 0

    for _, row in stitch_labels.iterrows():
        selected = json.loads(row["selected_edges"])
        curated = {(str(e["ref_id"]), str(e["target_id"])) for e in selected}

        if not curated:
            continue

        # Collect all segment IDs in this group
        group_ref_ids = {r for r, _ in curated}
        group_target_ids = {t for _, t in curated}

        # Find bridge edges involving any of these IDs
        bridge_for_group = {
            (r, t)
            for r, t in bridge_edges
            if r in group_ref_ids or t in group_target_ids
        }

        found = curated & bridge_for_group
        extra = bridge_for_group - curated

        recall = len(found) / len(curated) if curated else 0.0
        precision = len(found) / len(bridge_for_group) if bridge_for_group else 0.0

        recalls.append(recall)
        precisions.append(precision)
        total_curated += len(curated)
        total_extra += len(extra)

    n = len(precisions)
    if n == 0:
        return StitchEvalResult(
            groups_evaluated=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            total_curated_edges=0,
            total_extra_edges=0,
        )

    avg_p = sum(precisions) / n
    avg_r = sum(recalls) / n
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

    return StitchEvalResult(
        groups_evaluated=n,
        precision=avg_p,
        recall=avg_r,
        f1=f1,
        total_curated_edges=total_curated,
        total_extra_edges=total_extra,
    )
