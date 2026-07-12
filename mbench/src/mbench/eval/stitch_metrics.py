"""Stitch-level evaluation metrics.

Compares bridge output against curated stitching labels to measure whether the
optimizer selects the correct edges within M:N groups. This is the mbench analog
of the crosswalk-side ``crosswalk agent stitch-eval`` and shares its key machinery:

- **Decomposition-aware mapping** (``map_labels_to_fragments``): grouping hashes
  change whenever the pipeline regroups, so pair labels are recovered across all
  current sidecar groups and pure 1:1 bridge fragments touched by curated edges.
- **Sliver-aware metrics**: both raw and sliver-filtered agreement are reported.
  Junction-sliver edges (near-zero physical overlap) are dropped from BOTH the
  predicted and curated edge sets before the filtered comparison, so agreement is
  not distorted by whether either side happened to include an artifact edge. The
  sliver rule is replicated standalone in ``mbench.eval.sliver`` (parity-tested
  against ``crosswalk.config.is_sliver_edge``).
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

from mbench.eval.sliver import group_sliver_edges

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
    # SET-semantics metrics. Set labels (label_semantics == "set") assert only
    # group membership, so they are EXCLUDED from the edge-F1 pools above and
    # scored on membership/boundary/coverage instead. ``set_groups_evaluated`` is
    # the count of set labels mapped to a current group.
    set_groups_evaluated: int = 0
    set_membership_exact_rate: float = 0.0
    set_boundary_precision: float = 0.0
    set_coverage: float = 0.0
    set_metrics_by_labeler: dict = field(default_factory=dict)
    # Mapping-health diagnostics for decomposition-aware pair-label recovery.
    mapping_diagnostics_available: bool = False
    pair_labels_total: int = 0
    pair_labels_mapped: int = 0
    pair_labels_mapped_clean: int = 0
    pair_labels_mapped_partial: int = 0
    pair_labels_mapped_split: int = 0
    pair_labels_lost: int = 0
    reject_all_labels_total: int = 0
    reject_all_labels_mapped: int = 0
    reject_all_labels_unrecoverable: int = 0
    pair_label_mapping_rate: float = 0.0
    set_labels_total: int = 0
    set_labels_mapped: int = 0
    set_labels_lost: int = 0

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
            "stitch_set_groups_evaluated": self.set_groups_evaluated,
            "stitch_set_membership_exact_rate": self.set_membership_exact_rate,
            "stitch_set_boundary_precision": self.set_boundary_precision,
            "stitch_set_coverage": self.set_coverage,
            "stitch_set_metrics_by_labeler": self.set_metrics_by_labeler,
            "stitch_mapping_diagnostics_available": self.mapping_diagnostics_available,
            "stitch_pair_labels_total": self.pair_labels_total,
            "stitch_pair_labels_mapped": self.pair_labels_mapped,
            "stitch_pair_labels_mapped_clean": self.pair_labels_mapped_clean,
            "stitch_pair_labels_mapped_partial": self.pair_labels_mapped_partial,
            "stitch_pair_labels_mapped_split": self.pair_labels_mapped_split,
            "stitch_pair_labels_lost": self.pair_labels_lost,
            "stitch_reject_all_labels_total": self.reject_all_labels_total,
            "stitch_reject_all_labels_mapped": self.reject_all_labels_mapped,
            "stitch_reject_all_labels_unrecoverable": self.reject_all_labels_unrecoverable,
            "stitch_pair_label_mapping_rate": self.pair_label_mapping_rate,
            "stitch_set_labels_total": self.set_labels_total,
            "stitch_set_labels_mapped": self.set_labels_mapped,
            "stitch_set_labels_lost": self.set_labels_lost,
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


def _curated_edge_set(selected_edges_raw: object) -> EdgeSet:
    """Strictly parse a stitching label's ``selected_edges`` JSON.

    Only an explicit JSON array may represent the selection; only ``[]`` means
    reject-all. Missing, blank, malformed, or structurally invalid values are
    label errors and must never be silently reinterpreted as reject-all.
    """
    if selected_edges_raw is None or (
        not isinstance(selected_edges_raw, (str, list, dict)) and pd.isna(selected_edges_raw)
    ):
        raise ValueError("selected_edges is null; reject-all must be explicit JSON []")
    if not isinstance(selected_edges_raw, str) or not selected_edges_raw.strip():
        raise ValueError("selected_edges must be a non-blank JSON array")
    try:
        edges = json.loads(selected_edges_raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"selected_edges is invalid JSON: {exc}") from exc
    if not isinstance(edges, list):
        raise ValueError("selected_edges must decode to a JSON array")

    parsed: list[Edge] = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ValueError(f"selected_edges[{index}] must be an object")
        missing = {"ref_id", "target_id"} - set(edge)
        if missing:
            raise ValueError(f"selected_edges[{index}] missing keys: {sorted(missing)}")
        ref_id = edge["ref_id"]
        target_id = edge["target_id"]
        if ref_id is None or target_id is None or str(ref_id) == "" or str(target_id) == "":
            raise ValueError(f"selected_edges[{index}] contains a null/blank id")
        parsed.append((str(ref_id), str(target_id)))
    return frozenset(parsed)


def _row_edge_set(row, idx: object) -> EdgeSet:
    """Parse one label row's ``selected_edges``, naming the row on failure.

    A malformed label is curation corruption and must abort evaluation (never
    silently reinterpreted), but the error has to identify WHICH row so the
    label file can be fixed without bisecting it.
    """
    try:
        return _curated_edge_set(row.get("selected_edges"))
    except ValueError as exc:
        raise ValueError(
            f"invalid stitching label (row {idx}, group_id={row.get('group_id')}, "
            f"labeler={row.get('labeler')}): {exc}"
        ) from exc


def _labeler_class(labeler: object) -> str:
    """Bucket a labeler value into 'panel' (agent auto-accept) or 'human'."""
    s = "" if labeler is None else str(labeler)
    return "panel" if s.startswith("panel") else "human"


def _is_set_label(row) -> bool:
    """True when a curated row uses SET semantics (membership, not pairs)."""
    return str(row.get("label_semantics") or "pair") == "set"


def _parse_id_list(raw) -> frozenset[str]:
    """Parse a JSON id array (``ref_ids`` / ``target_ids``) into a string set."""
    if raw is None or isinstance(raw, float) or not str(raw).strip():
        return frozenset()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset(str(x) for x in data)


def set_label_metrics(
    pred_edges: EdgeSet,
    ref_members: frozenset[str],
    target_members: frozenset[str],
) -> tuple[bool, float, float]:
    """Score a predicted edge set against a SET label's membership.

    THE PARITY-CRITICAL CORE. Replicated verbatim in
    ``crosswalk.agent_labeling.stitch_eval.set_label_metrics`` and guarded by
    ``tests/unit/test_mbench_set_metric_parity.py`` — keep the two in lockstep.

    Args:
        pred_edges: the predicted (optimizer/panel) SELECTED edges within the
            group mapped to this label.
        ref_members / target_members: the labeler's asserted group membership.

    Returns ``(membership_exact, boundary_precision, coverage)``:
      * membership_exact - the segments incident to the predicted edges are
        EXACTLY this membership (both sides).
      * boundary_precision - fraction of predicted edges whose BOTH endpoints are
        members (no edge crosses into a non-member); 1.0 when nothing is
        predicted (vacuously, no boundary is crossed).
      * coverage - fraction of members with >= 1 incident predicted edge; 1.0
        when the membership is empty.
    """
    pred_ref = frozenset(r for r, _ in pred_edges)
    pred_tgt = frozenset(t for _, t in pred_edges)
    membership_exact = (pred_ref == ref_members) and (pred_tgt == target_members)

    n_edges = len(pred_edges)
    within = sum(1 for (r, t) in pred_edges if r in ref_members and t in target_members)
    boundary_precision = within / n_edges if n_edges else 1.0

    n_members = len(ref_members) + len(target_members)
    covered = len(ref_members & pred_ref) + len(target_members & pred_tgt)
    coverage = covered / n_members if n_members else 1.0
    return membership_exact, boundary_precision, coverage


def _aggregate_set(
    records: list[tuple[bool, float, float]],
) -> tuple[float, float, float]:
    """Mean (membership_exact_rate, boundary_precision, coverage)."""
    n = len(records)
    if n == 0:
        return 0.0, 0.0, 0.0
    exact = sum(1 for m, _, _ in records if m) / n
    boundary = sum(b for _, b, _ in records) / n
    coverage = sum(c for _, _, c in records) / n
    return exact, boundary, coverage


def _edge_prf(pred: EdgeSet, truth: EdgeSet) -> tuple[float, float, float]:
    """Edge-level precision / recall / F1. Two empty sets score a perfect match."""
    if not pred and not truth:
        return 1.0, 1.0, 1.0
    tp = len(pred & truth)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _group_source_edges(group: dict, source: str) -> EdgeSet:
    """Read and validate one edge-list source from a groups-sidecar record."""
    raw_edges = group.get(source) or []
    if not isinstance(raw_edges, list):
        raise ValueError(f"group {group.get('group_id')} {source} must be a list")
    parsed: list[Edge] = []
    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, dict):
            raise ValueError(f"group {group.get('group_id')} {source}[{index}] must be an object")
        ref_id = edge.get("ref_id")
        target_id = edge.get("target_id")
        if (
            ref_id is None
            or target_id is None
            or not str(ref_id).strip()
            or not str(target_id).strip()
        ):
            raise ValueError(f"group {group.get('group_id')} {source}[{index}] has a null/blank id")
        parsed.append((str(ref_id), str(target_id)))
    return frozenset(parsed)


def _edge_fragment_index(fragment_edges: dict[str, EdgeSet]) -> dict[Edge, frozenset[str]]:
    index: dict[Edge, set[str]] = defaultdict(set)
    for fragment_id, edges in fragment_edges.items():
        for edge in edges:
            index[edge].add(fragment_id)
    return {edge: frozenset(fragment_ids) for edge, fragment_ids in index.items()}


def _effective_edge_fragments(
    edge: Edge,
    fragment_indexes: tuple[dict[Edge, frozenset[str]], ...],
) -> frozenset[str]:
    """Return owners from the first recovery source that contains the edge."""
    for index in fragment_indexes:
        if owners := index.get(edge):
            return owners
    return frozenset()


def map_labels_to_fragments(
    stitch_labels: pd.DataFrame,
    fragment_indexes: tuple[dict[Edge, frozenset[str]], ...],
    real_group_ids: frozenset[str],
) -> dict[int, frozenset[str]]:
    """Map each pair label to every current assignment fragment it touches.

    A historical group may decompose into several current M:N groups and/or
    ordinary 1:1 bridge rows. Selecting one "best" current group loses correct
    fragments and makes tied mappings depend on unrelated group ids. Non-empty
    labels therefore map to the UNION of fragments touched by their curated
    edges. Ownership precedence is selected edge / bridge singleton, then the
    uncapped candidate graph, then legacy rejected edges. This prevents repeated
    rejected-edge records from pulling unrelated groups into a label footprint.

    Reject-all labels (empty ``selected_edges``, i.e. "no edges should be
    selected") still survive ONLY when their original ``group_id`` exists
    verbatim in the real sidecar. Synthetic singleton fragments must never make
    an edge-less label look recoverable.

    Returns ``{row_index: frozenset(fragment_id, ...)}``.
    """
    mapping: dict[int, frozenset[str]] = {}
    for idx, row in stitch_labels.iterrows():
        hes = _row_edge_set(row, idx)
        hgid = str(row.get("group_id"))
        if not hes:
            # Reject-all label: recoverable only by verbatim group_id match.
            if hgid in real_group_ids:
                mapping[idx] = frozenset({hgid})
            continue
        fragments = frozenset(
            fragment_id
            for edge in hes
            for fragment_id in _effective_edge_fragments(edge, fragment_indexes)
        )
        if fragments:
            mapping[idx] = fragments
    return mapping


def map_set_labels_to_groups(
    set_labels: pd.DataFrame,
    group_members: dict[str, frozenset[str]],
) -> dict[int, str]:
    """Map each SET-semantics row to a current group by MEMBERSHIP overlap.

    Set labels carry no edges (``selected_edges`` empty), so edge-overlap
    recovery is impossible. Instead a set label maps to the current group whose
    segment membership overlaps its ref_ids ∪ target_ids the most, preferring a
    verbatim ``group_id`` match when that group still shares a segment. Labels
    whose segments no longer appear in any current group are dropped (mirrors the
    pair mapper's "edges no longer survive" case).

    Returns ``{row_index: group_id}``.
    """
    seg_groups: dict[str, set[str]] = defaultdict(set)
    for gid, members in group_members.items():
        for s in members:
            seg_groups[s].add(gid)

    mapping: dict[int, str] = {}
    for idx, row in set_labels.iterrows():
        members = _parse_id_list(row.get("ref_ids")) | _parse_id_list(row.get("target_ids"))
        hgid = str(row.get("group_id"))
        if not members:
            # Empty-membership set row (degenerate reject-all): verbatim only.
            if hgid in group_members:
                mapping[idx] = hgid
            continue
        counts: dict[str, int] = defaultdict(int)
        for s in members:
            for gid in seg_groups.get(s, ()):
                counts[gid] += 1
        if not counts:
            continue
        # Deterministic tie-break for membership-only SET recovery (#367):
        # sort before max() so ties resolve to the smallest group_id.
        best = max(sorted(counts), key=counts.get)
        mapping[idx] = hgid if counts.get(hgid, 0) > 0 else best
    return mapping


def _pair_mapping_health(
    pair_labels: pd.DataFrame,
    fragment_indexes: tuple[dict[Edge, frozenset[str]], ...],
    mapping: dict[int, frozenset[str]],
) -> dict[str, int | float]:
    """Classify pair-label survival against the decomposition-aware fragments."""
    clean = partial = split = reject_all_total = reject_all_unrecoverable = 0
    for idx, row in pair_labels.iterrows():
        curated = _row_edge_set(row, idx)
        if not curated:
            reject_all_total += 1
            if idx not in mapping:
                reject_all_unrecoverable += 1
            continue

        surviving = [edge for edge in curated if _effective_edge_fragments(edge, fragment_indexes)]
        current_groups = {
            fragment_id
            for edge in surviving
            for fragment_id in _effective_edge_fragments(edge, fragment_indexes)
        }
        if not surviving:
            continue
        if len(current_groups) > 1:
            split += 1
        elif len(surviving) < len(curated):
            partial += 1
        else:
            clean += 1

    total = len(pair_labels)
    mapped = len(mapping)
    return {
        "pair_labels_total": total,
        "pair_labels_mapped": mapped,
        "pair_labels_mapped_clean": clean,
        "pair_labels_mapped_partial": partial,
        "pair_labels_mapped_split": split,
        "pair_labels_lost": total - mapped,
        "reject_all_labels_total": reject_all_total,
        "reject_all_labels_mapped": reject_all_total - reject_all_unrecoverable,
        "reject_all_labels_unrecoverable": reject_all_unrecoverable,
        "pair_label_mapping_rate": mapped / total if total else 0.0,
    }


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
    for idx, row in stitch_labels.iterrows():
        curated = _row_edge_set(row, idx)
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

    group_selected_edges: dict[str, EdgeSet] = {}
    group_members: dict[str, frozenset[str]] = {}
    fragment_primary_edges: dict[str, EdgeSet] = {}
    fragment_candidate_edges: dict[str, EdgeSet] = {}
    fragment_rejected_edges: dict[str, EdgeSet] = {}
    fragment_predictions: dict[str, EdgeSet] = {}
    fragment_slivers: dict[str, EdgeSet] = {}
    for g in groups:
        gid = str(g.get("group_id"))
        selected = _group_source_edges(g, "edges")
        candidate = _group_source_edges(g, "candidate_edges")
        rejected = _group_source_edges(g, "rejected_edges")
        slivers = group_sliver_edges(g)
        group_selected_edges[gid] = selected
        group_members[gid] = frozenset(r for r, _ in selected) | frozenset(t for _, t in selected)
        fragment_primary_edges[gid] = selected
        fragment_candidate_edges[gid] = candidate
        fragment_rejected_edges[gid] = rejected
        fragment_predictions[gid] = selected & bridge_edges
        fragment_slivers[gid] = slivers

    # The groups sidecar intentionally omits pure 1:1 assignments. Represent
    # each EXPLICITLY TYPED 1:1 bridge row outside sidecar `edges` as a singleton
    # fragment so a decomposed group remains evaluable. Never infer singleton
    # status from absence alone: that would let a partial/broken sidecar turn
    # missing N:M groups into apparently valid 1:1 recovery.
    sidecar_selected_edges = frozenset(
        edge for selected in group_selected_edges.values() for edge in selected
    )
    if "match_type" in bridge.columns:
        one_to_one_mask = bridge["match_type"].astype("string").str.strip().eq("1:1").fillna(False)
        one_to_one_edges = set(
            zip(
                bridge.loc[one_to_one_mask, "ref_id"].astype(str),
                bridge.loc[one_to_one_mask, "target_id"].astype(str),
            )
        )
    else:
        one_to_one_edges = set()
    used_fragment_ids = set(fragment_primary_edges)
    for position, edge in enumerate(sorted(one_to_one_edges - sidecar_selected_edges)):
        fragment_id = f"__bridge_singleton__:{position}"
        while fragment_id in used_fragment_ids:
            fragment_id += ":"
        used_fragment_ids.add(fragment_id)
        singleton = frozenset({edge})
        fragment_primary_edges[fragment_id] = singleton
        fragment_predictions[fragment_id] = singleton
        fragment_slivers[fragment_id] = frozenset()

    # Split by semantics: SET labels are scored on membership/boundary/coverage
    # and MUST NOT enter the edge-F1 pools (they assert no pair-level truth).
    if "label_semantics" in stitch_labels.columns:
        is_set = stitch_labels.apply(_is_set_label, axis=1)
        pair_labels = stitch_labels[~is_set]
        set_labels = stitch_labels[is_set]
    else:
        pair_labels = stitch_labels
        set_labels = stitch_labels.iloc[0:0]

    fragment_indexes = tuple(
        _edge_fragment_index(fragment_edges)
        for fragment_edges in (
            fragment_primary_edges,
            fragment_candidate_edges,
            fragment_rejected_edges,
        )
    )
    mapping = map_labels_to_fragments(
        pair_labels,
        fragment_indexes,
        frozenset(group_selected_edges),
    )
    mapping_health = _pair_mapping_health(
        pair_labels,
        fragment_indexes,
        mapping,
    )

    raw_records: list[tuple[EdgeSet, EdgeSet]] = []
    filt_records: list[tuple[EdgeSet, EdgeSet]] = []
    by_labeler_raw: dict[str, list[tuple[EdgeSet, EdgeSet]]] = defaultdict(list)
    total_curated = 0
    total_extra = 0
    n_sliver_affected = 0

    for idx, fragment_ids in mapping.items():
        row = stitch_labels.loc[idx]
        curated = _row_edge_set(row, idx)
        # Score the union of every current fragment touched by the curated edge
        # set. This is invariant to a historical group splitting into several
        # sidecar groups and/or pure 1:1 bridge assignments.
        pred = frozenset(
            edge for fragment_id in fragment_ids for edge in fragment_predictions[fragment_id]
        )

        raw_records.append((pred, curated))
        by_labeler_raw[_labeler_class(row.get("labeler"))].append((pred, curated))
        total_curated += len(curated)
        total_extra += len(pred - curated)

        slivers = frozenset(
            edge for fragment_id in fragment_ids for edge in fragment_slivers[fragment_id]
        )
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

    # --- SET-semantics metrics (membership / boundary / coverage) ---
    set_mapping = map_set_labels_to_groups(set_labels, group_members)
    set_records: list[tuple[bool, float, float]] = []
    set_by_labeler: dict[str, list[tuple[bool, float, float]]] = defaultdict(list)
    for idx, gid in set_mapping.items():
        row = set_labels.loc[idx]
        ref_members = _parse_id_list(row.get("ref_ids"))
        tgt_members = _parse_id_list(row.get("target_ids"))
        candidate = group_selected_edges[gid]
        pred = frozenset(e for e in candidate if e in bridge_edges)
        rec = set_label_metrics(pred, ref_members, tgt_members)
        set_records.append(rec)
        set_by_labeler[_labeler_class(row.get("labeler"))].append(rec)

    set_exact, set_boundary, set_coverage = _aggregate_set(set_records)
    set_metrics_by_labeler: dict[str, dict] = {}
    for cls, recs in sorted(set_by_labeler.items()):
        ex, bd, cov = _aggregate_set(recs)
        set_metrics_by_labeler[cls] = {
            "n": len(recs),
            "membership_exact_rate": round(ex, 4),
            "boundary_precision": round(bd, 4),
            "coverage": round(cov, 4),
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
        set_groups_evaluated=len(set_records),
        set_membership_exact_rate=set_exact,
        set_boundary_precision=set_boundary,
        set_coverage=set_coverage,
        set_metrics_by_labeler=set_metrics_by_labeler,
        mapping_diagnostics_available=True,
        **mapping_health,
        set_labels_total=len(set_labels),
        set_labels_mapped=len(set_mapping),
        set_labels_lost=len(set_labels) - len(set_mapping),
    )
