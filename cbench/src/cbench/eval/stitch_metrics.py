"""Stitch-level evaluation metrics.

Compares bridge output against curated stitching labels to measure whether the
optimizer selects the correct edges within M:N groups. This is the cbench analog
of the matcher-side ``matcher agent stitch-eval`` and shares its key machinery:

- **Group mapping robust to group_id churn** (``map_labels_to_groups``): grouping
  hashes change whenever the pipeline regroups, so labels are mapped to current
  groups by exact group_id AND by edge-overlap against the groups sidecar.
- **Sliver-aware metrics**: both raw and sliver-filtered agreement are reported.
  Junction-sliver edges (near-zero physical overlap) are dropped from BOTH the
  predicted and curated edge sets before the filtered comparison, so agreement is
  not distorted by whether either side happened to include an artifact edge. The
  sliver rule is replicated standalone in ``cbench.eval.sliver`` (parity-tested
  against ``matcher.config.is_sliver_edge``).
- **Exact-match rate** per group, alongside edge precision / recall / F1.
- **Per-labeler breakdown**: curated labels now carry a ``labeler`` column
  (human user ids and ``panel_*`` agent-panel auto-accepts); counts and
  human-vs-panel metrics are reported so panel labels don't silently dominate.

Non-blocking by design: callers treat a raise here as "skip stitch eval", never
a run failure.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from cbench.eval.sliver import group_sliver_edges

Edge = tuple[str, str]
EdgeSet = frozenset[Edge]


@dataclass
class StitchEvalResult:
    """Metrics from evaluating bridge edges against stitching labels.

    Raw metrics use all curated edges; ``*_filtered`` metrics drop junction
    slivers from both sides. Precision/recall/F1 are macro-averaged over mapped
    groups (F1 is the harmonic mean of the averaged precision and recall, kept
    for backward compatibility with earlier results).
    """

    groups_evaluated: int
    precision: float
    recall: float
    f1: float
    exact_match_rate: float
    total_curated_edges: int
    total_extra_edges: int
    # Sliver-filtered variants.
    precision_filtered: float = 0.0
    recall_filtered: float = 0.0
    f1_filtered: float = 0.0
    exact_match_rate_filtered: float = 0.0
    groups_sliver_affected: int = 0
    # Provenance breakdown.
    label_counts_by_labeler: dict = field(default_factory=dict)
    metrics_by_labeler: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "groups_evaluated": self.groups_evaluated,
            "stitch_precision": self.precision,
            "stitch_recall": self.recall,
            "stitch_f1": self.f1,
            "stitch_exact_match_rate": self.exact_match_rate,
            "total_curated_edges": self.total_curated_edges,
            "total_extra_edges": self.total_extra_edges,
            "stitch_precision_filtered": self.precision_filtered,
            "stitch_recall_filtered": self.recall_filtered,
            "stitch_f1_filtered": self.f1_filtered,
            "stitch_exact_match_rate_filtered": self.exact_match_rate_filtered,
            "stitch_groups_sliver_affected": self.groups_sliver_affected,
            "stitch_label_counts_by_labeler": self.label_counts_by_labeler,
            "stitch_metrics_by_labeler": self.metrics_by_labeler,
        }


def _empty_result() -> StitchEvalResult:
    return StitchEvalResult(
        groups_evaluated=0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        exact_match_rate=0.0,
        total_curated_edges=0,
        total_extra_edges=0,
    )


def _curated_edge_set(selected_edges_raw: str) -> EdgeSet:
    """Parse a stitching label's ``selected_edges`` JSON into an edge set."""
    if not selected_edges_raw or isinstance(selected_edges_raw, float):
        return frozenset()
    try:
        edges = json.loads(selected_edges_raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset((str(e["ref_id"]), str(e["target_id"])) for e in edges)


def _labeler_class(labeler: object) -> str:
    """Bucket a labeler value into 'panel' (agent auto-accept) or 'human'."""
    s = "" if labeler is None else str(labeler)
    return "panel" if s.startswith("panel") else "human"


def _edge_prf(pred: EdgeSet, truth: EdgeSet) -> tuple[float, float, float]:
    """Edge-level precision / recall / F1. Two empty sets score a perfect match."""
    if not pred and not truth:
        return 1.0, 1.0, 1.0
    tp = len(pred & truth)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def map_labels_to_groups(
    stitch_labels: pd.DataFrame,
    group_candidate_edges: dict[str, EdgeSet],
) -> dict[int, str]:
    """Map each labeled row to a current sidecar group_id.

    Robust to group_id churn: grouping hashes are hashes of the exact ref/target
    id sets, so component shifts break exact-id recovery. A label maps to the
    current group containing the most of its selected edges (edge-overlap), with
    a verbatim group_id match preferred when that group still exists and shares an
    edge. Labels whose edges no longer survive in any group are dropped.

    Returns ``{row_index: group_id}``.
    """
    edge_groups: dict[Edge, set[str]] = defaultdict(set)
    for gid, edges in group_candidate_edges.items():
        for e in edges:
            edge_groups[e].add(gid)

    mapping: dict[int, str] = {}
    for idx, row in stitch_labels.iterrows():
        hes = _curated_edge_set(row.get("selected_edges"))
        if not hes:
            continue  # reject-all labels carry no edges to recover by overlap
        counts: dict[str, int] = defaultdict(int)
        for e in hes:
            for gid in edge_groups.get(e, ()):
                counts[gid] += 1
        if not counts:
            continue  # edges no longer survive in any current group
        best = max(counts, key=counts.get)
        hgid = str(row.get("group_id"))
        # Prefer a verbatim group_id match when it still overlaps this label.
        target = hgid if counts.get(hgid, 0) > 0 else best
        mapping[idx] = target
    return mapping


def _aggregate(
    records: list[tuple[EdgeSet, EdgeSet]],
) -> tuple[float, float, float, float]:
    """Macro precision/recall/F1 (F1-of-averages) and exact-match rate."""
    n = len(records)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    ps, rs, exact = [], [], 0
    for pred, truth in records:
        p, r, _ = _edge_prf(pred, truth)
        ps.append(p)
        rs.append(r)
        exact += int(pred == truth)
    avg_p = sum(ps) / n
    avg_r = sum(rs) / n
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0
    return avg_p, avg_r, f1, exact / n


def _legacy_evaluate(bridge: pd.DataFrame, stitch_labels: pd.DataFrame) -> StitchEvalResult:
    """Segment-id membership eval when no groups sidecar is available.

    Retains the original (pre-parity) behaviour: the predicted set for a group is
    every bridge edge touching one of its segment ids. Sliver filtering is not
    possible without geometries, so filtered metrics equal raw.
    """
    bridge_edges = set(zip(bridge["ref_id"].astype(str), bridge["target_id"].astype(str)))
    records: list[tuple[EdgeSet, EdgeSet]] = []
    total_curated = 0
    total_extra = 0
    for _, row in stitch_labels.iterrows():
        curated = _curated_edge_set(row.get("selected_edges"))
        if not curated:
            continue
        group_ref_ids = {r for r, _ in curated}
        group_target_ids = {t for _, t in curated}
        pred = frozenset(
            (r, t) for r, t in bridge_edges if r in group_ref_ids or t in group_target_ids
        )
        records.append((pred, curated))
        total_curated += len(curated)
        total_extra += len(pred - curated)

    avg_p, avg_r, f1, exact = _aggregate(records)
    return StitchEvalResult(
        groups_evaluated=len(records),
        precision=avg_p,
        recall=avg_r,
        f1=f1,
        exact_match_rate=exact,
        total_curated_edges=total_curated,
        total_extra_edges=total_extra,
        precision_filtered=avg_p,
        recall_filtered=avg_r,
        f1_filtered=f1,
        exact_match_rate_filtered=exact,
        groups_sliver_affected=0,
        label_counts_by_labeler=_label_counts(stitch_labels),
    )


def _label_counts(stitch_labels: pd.DataFrame) -> dict:
    """Count of ALL curated labels by their raw labeler value."""
    if "labeler" not in stitch_labels.columns:
        return {}
    return {str(k): int(v) for k, v in stitch_labels["labeler"].value_counts().items()}


def evaluate_stitch_groups(
    bridge: pd.DataFrame,
    stitch_labels: pd.DataFrame,
    groups: list[dict] | None = None,
) -> StitchEvalResult:
    """Compare bridge output against curated stitching labels.

    Args:
        bridge: DataFrame with columns [ref_id, target_id, confidence].
        stitch_labels: DataFrame with [group_id, selected_edges, labeler, ...].
        groups: The groups sidecar (list of group dicts with ``edges`` and
            ``ref_geometries`` / ``target_geometries``). When provided, labels are
            mapped by edge-overlap and sliver-filtered metrics are computed. When
            None, falls back to legacy segment-id membership (filtered == raw).

    Returns:
        StitchEvalResult with raw and sliver-filtered aggregate metrics plus a
        per-labeler breakdown.
    """
    if stitch_labels.empty:
        return _empty_result()

    if not groups:
        return _legacy_evaluate(bridge, stitch_labels)

    bridge_edges = set(zip(bridge["ref_id"].astype(str), bridge["target_id"].astype(str)))

    group_candidate_edges: dict[str, EdgeSet] = {}
    group_slivers: dict[str, EdgeSet] = {}
    for g in groups:
        gid = str(g.get("group_id"))
        group_candidate_edges[gid] = frozenset(
            (str(e["ref_id"]), str(e["target_id"])) for e in g.get("edges", [])
        )
        group_slivers[gid] = group_sliver_edges(g)

    mapping = map_labels_to_groups(stitch_labels, group_candidate_edges)

    raw_records: list[tuple[EdgeSet, EdgeSet]] = []
    filt_records: list[tuple[EdgeSet, EdgeSet]] = []
    by_labeler_raw: dict[str, list[tuple[EdgeSet, EdgeSet]]] = defaultdict(list)
    total_curated = 0
    total_extra = 0
    n_sliver_affected = 0

    for idx, gid in mapping.items():
        row = stitch_labels.loc[idx]
        curated = _curated_edge_set(row.get("selected_edges"))
        candidate = group_candidate_edges[gid]
        # Predicted edge set for this group = bridge selections among the group's
        # candidate edges (i.e. what the optimizer kept within this component).
        pred = frozenset(e for e in candidate if e in bridge_edges)

        raw_records.append((pred, curated))
        by_labeler_raw[_labeler_class(row.get("labeler"))].append((pred, curated))
        total_curated += len(curated)
        total_extra += len(pred - curated)

        slivers = group_slivers.get(gid, frozenset())
        pred_f = pred - slivers
        curated_f = curated - slivers
        filt_records.append((pred_f, curated_f))
        if pred_f != pred or curated_f != curated:
            n_sliver_affected += 1

    avg_p, avg_r, f1, exact = _aggregate(raw_records)
    avg_pf, avg_rf, f1f, exact_f = _aggregate(filt_records)

    metrics_by_labeler: dict[str, dict] = {}
    for cls, recs in sorted(by_labeler_raw.items()):
        p, r, f, ex = _aggregate(recs)
        metrics_by_labeler[cls] = {
            "n": len(recs),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "exact_match_rate": round(ex, 4),
        }

    return StitchEvalResult(
        groups_evaluated=len(raw_records),
        precision=avg_p,
        recall=avg_r,
        f1=f1,
        exact_match_rate=exact,
        total_curated_edges=total_curated,
        total_extra_edges=total_extra,
        precision_filtered=avg_pf,
        recall_filtered=avg_rf,
        f1_filtered=f1f,
        exact_match_rate_filtered=exact_f,
        groups_sliver_affected=n_sliver_affected,
        label_counts_by_labeler=_label_counts(stitch_labels),
        metrics_by_labeler=metrics_by_labeler,
    )
