"""Extract a per-edge keep/drop training table from group sidecars + labels.

Data reality (verified 2026-07, see the prototype writeup):

* The production ``*_groups.json`` sidecar persists only the optimizer's
  **selected** assignment per group (``selected`` is True for ~99.9% of edges;
  the handful of False edges are junction slivers). It does NOT persist the
  edges the optimizer rejected. So a keep/drop task defined over the persisted
  edges is really an *over-selection correction* task: which optimizer-kept
  edges should actually be dropped. Recall of the keep-all baseline is 1.0 by
  construction; the only errors are false positives.
* The 78 pairwise ML features live in ``labels/features/`` keyed by
  ``(gers_id, target_id)`` but only for *pair*-labeled pairs — coverage of
  group edges is ~5%. So per-edge features come from the sidecar itself
  (confidence + structural layer + alignment fractions), not the pairwise
  parquet.
* Ground truth = curated ``labels/stitching/`` ``selected_edges``. Labels map
  to current sidecar groups by edge overlap (group_id churns on any component
  shift) via ``stitch_eval.recover_labeled_groups`` — reused here verbatim.

Provenance is preserved on every row so the eval can slice by dataset,
labeler, and clean-vs-split mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from matcher.agent_labeling.stitch_eval import recover_labeled_groups

EDGE_LABEL_COL = "keep"


def _human_edge_set(selected_edges_raw) -> frozenset[tuple[str, str]]:
    """Parse a stitching label's ``selected_edges`` JSON to an edge frozenset.

    Local copy of the tiny parser (rather than importing the private
    ``stitch_eval._human_edge_set``) so this research harness does not depend on
    an internal API that may change without notice.
    """
    try:
        edges = json.loads(selected_edges_raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset((str(e["ref_id"]), str(e["target_id"])) for e in edges)


def load_sidecar_groups(path: str | Path) -> list[dict]:
    """Load the ``groups`` list from a ``*_groups.json`` sidecar."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        return data.get("groups", [])
    return data


def load_stitching_labels(path: str | Path) -> pd.DataFrame:
    """Load a curated stitching label CSV (group_id kept as str)."""
    return pd.read_csv(path, dtype={"group_id": str})


def _edge_key(edge: dict) -> tuple[str, str]:
    return (str(edge["ref_id"]), str(edge["target_id"]))


def build_edge_table(
    groups: list[dict],
    human_df: pd.DataFrame,
    dataset_id: str,
    include_split: bool = True,
) -> pd.DataFrame:
    """Build a per-edge training table for one dataset.

    Each row is one candidate edge inside a *labeled* group, with the raw
    sidecar fields, group-context columns, the ``keep`` label (1 iff the edge is
    in the human's selected set), the ``selected`` optimizer baseline, and
    provenance columns.

    Args:
        groups: sidecar groups (from :func:`load_sidecar_groups`).
        human_df: curated stitching labels for this dataset.
        dataset_id: dataset identifier stored on every row.
        include_split: if False, only ``clean`` labels (all selected edges in
            one group) are emitted; ``split`` labels (human edge set spans
            multiple groups, so the within-group keep set is partial and the
            drop label is noisy) are dropped.

    Returns:
        DataFrame with one row per (group, edge). Empty if no labels map.
    """
    rec = recover_labeled_groups(groups, human_df)
    gmap = {g["group_id"]: g for g in groups}
    human_by = {str(r["group_id"]): r for _, r in human_df.iterrows()}

    mapped: list[tuple[str, str, str]] = [(hgid, bg, "clean") for hgid, bg in rec["clean"]]
    if include_split:
        mapped += [(hgid, bg, "split") for hgid, bg, _, _ in rec["split"]]

    rows: list[dict] = []
    for hgid, bg, provenance in mapped:
        group = gmap.get(bg)
        if group is None:
            continue
        hrow = human_by[hgid]
        human_es = _human_edge_set(hrow["selected_edges"])
        edges = group.get("edges", [])
        for edge in edges:
            key = _edge_key(edge)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "group_id": group["group_id"],
                    "human_group_id": hgid,
                    "labeler": hrow.get("labeler", ""),
                    "provenance": provenance,
                    "match_type": group.get("match_type", ""),
                    "ref_id": key[0],
                    "target_id": key[1],
                    EDGE_LABEL_COL: int(key in human_es),
                    "selected": bool(edge.get("selected", True)),
                    # raw per-edge sidecar fields
                    "confidence": float(edge.get("confidence", float("nan"))),
                    "degree_ref": int(edge.get("degree_ref", 0)),
                    "degree_tgt": int(edge.get("degree_tgt", 0)),
                    "is_bridge": bool(edge.get("is_bridge", False)),
                    "is_sliver": bool(edge.get("is_sliver", False)),
                    "biconnected_block": int(edge.get("biconnected_block", -1)),
                    "corridor_ref": int(edge.get("corridor_ref", -1)),
                    "corridor_tgt": int(edge.get("corridor_tgt", -1)),
                    "gers_start_frac": float(edge.get("gers_start_frac", float("nan"))),
                    "gers_end_frac": float(edge.get("gers_end_frac", float("nan"))),
                    "local_start_frac": float(edge.get("local_start_frac", float("nan"))),
                    "local_end_frac": float(edge.get("local_end_frac", float("nan"))),
                    # per-group structural fields
                    "n_edges": int(group.get("n_edges", len(edges))),
                    "n_corridors": int(group.get("n_corridors", 1)),
                    "n_assignment_components": int(group.get("n_assignment_components", 1)),
                    "largest_biconnected_block": int(group.get("largest_biconnected_block", 1)),
                    "oversized_group": bool(group.get("oversized_group", False)),
                    "num_refs": len(group.get("ref_ids", [])),
                    "num_targets": len(group.get("target_ids", [])),
                }
            )
    return pd.DataFrame(rows)


def build_multi_dataset_table(
    specs: list[tuple[str, str | Path, str | Path]],
    include_split: bool = True,
) -> pd.DataFrame:
    """Build and concatenate per-edge tables for several datasets.

    Args:
        specs: list of ``(dataset_id, groups_json_path, labels_csv_path)``.
        include_split: passed through to :func:`build_edge_table`.
    """
    frames = []
    for dataset_id, groups_path, labels_path in specs:
        groups = load_sidecar_groups(groups_path)
        human_df = load_stitching_labels(labels_path)
        frames.append(build_edge_table(groups, human_df, dataset_id, include_split=include_split))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
