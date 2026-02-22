"""Post-optimizer graph consistency validation.

Builds lightweight segment-adjacency graphs from reference and target
GeoDataFrame endpoints, then checks whether accepted matches are
topologically consistent with their graph neighbourhoods.

Design principles:
- Precision-biased: demotes suspicious matches from MATCH → REVIEW
  (never hard rejects, never promotes).
- Runs after the optimizer, before bridge generation.
- O(N × avg_degree) ≈ O(N) for sparse road networks.

Three checks are applied, each producing a per-match contradiction score:

1. **Junction contradiction**: For each reference junction where matched
   ref segments meet, verify that the matched target segments also share
   a junction in the target graph.

2. **Neighbourhood coherence**: For a match (ref_A, target_X), look at
   ref_A's graph neighbours.  If most neighbours have accepted matches
   but none of those targets connect to target_X's neighbourhood, the
   match is isolated — suspicious.

3. **Degree excess**: At a reference junction of degree K, there should be
   at most K matched target "legs".  More than that signals conflicting
   assignments.
"""

from collections import defaultdict
from typing import Any

import geopandas as gpd
import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

from ..config import DEFAULT_SNAP_TOLERANCE_M
from ..topology.sparse_graph import SparseGraph, build_graph_from_edges
from .types import MatchDecision, MatchResult

# ---------------------------------------------------------------------------
# Graph construction from GeoDataFrame endpoints
# ---------------------------------------------------------------------------


def _build_segment_adjacency_graph(
    gdf: gpd.GeoDataFrame,
    id_column: str,
    tolerance: float,
) -> SparseGraph:
    """Build an undirected segment-adjacency graph from endpoint proximity.

    Two segments are adjacent if any endpoint of one is within *tolerance*
    metres of any endpoint of the other.

    Args:
        gdf: GeoDataFrame with LineString geometries (projected CRS).
        id_column: Column containing segment IDs.
        tolerance: Snap tolerance in metres.

    Returns:
        SparseGraph whose nodes are segment IDs and edges connect adjacent
        segments.
    """
    if id_column in gdf.columns:
        seg_ids = gdf[id_column].values
    else:
        seg_ids = gdf.index.values

    # Extract start/end coordinates for every segment.
    endpoints: list[tuple[float, float]] = []
    ep_to_seg_idx: list[int] = []  # maps endpoint array index → seg index

    for seg_idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        coords = geom.coords
        if len(coords) < 2:
            continue
        start = coords[0][:2]
        end = coords[-1][:2]
        endpoints.append(start)
        ep_to_seg_idx.append(seg_idx)
        endpoints.append(end)
        ep_to_seg_idx.append(seg_idx)

    if len(endpoints) < 2:
        return build_graph_from_edges([], node_attrs={n: {} for n in seg_ids})

    ep_array = np.array(endpoints)
    tree = cKDTree(ep_array)
    close_pairs = tree.query_pairs(tolerance)

    edges: list[tuple[Any, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for i, j in close_pairs:
        si = ep_to_seg_idx[i]
        sj = ep_to_seg_idx[j]
        if si == sj:
            continue
        a, b = seg_ids[si], seg_ids[sj]
        key = (min(a, b), max(a, b)) if a != b else None
        if key and key not in seen:
            seen.add(key)
            edges.append(key)

    return build_graph_from_edges(
        edges,
        node_attrs={sid: {} for sid in seg_ids},
    )


# ---------------------------------------------------------------------------
# Match index helpers
# ---------------------------------------------------------------------------


def _build_match_maps(
    accepted: list[MatchResult],
) -> tuple[dict[Any, set[Any]], dict[Any, set[Any]]]:
    """Build ref→targets and target→refs lookup dicts from accepted matches."""
    ref_to_targets: dict[Any, set[Any]] = defaultdict(set)
    target_to_refs: dict[Any, set[Any]] = defaultdict(set)
    for r in accepted:
        ref_to_targets[r.ref_id].add(r.target_id)
        target_to_refs[r.target_id].add(r.ref_id)
    return dict(ref_to_targets), dict(target_to_refs)


# ---------------------------------------------------------------------------
# Check 1 — Junction contradiction
# ---------------------------------------------------------------------------


def _junction_contradiction_scores(
    accepted: list[MatchResult],
    ref_graph: SparseGraph,
    target_graph: SparseGraph,
    ref_to_targets: dict[Any, set[Any]],
) -> dict[tuple[Any, Any], float]:
    """Score each match by how many shared reference junctions lack target-side confirmation.

    For a match (ref_A, target_X), consider every *matched* reference
    neighbour ref_B (i.e. ref_B is adjacent to ref_A in the ref graph and
    ref_B has accepted matches).  If none of ref_B's matched targets are
    adjacent to (or equal to) target_X in the target graph, that junction
    is contradicted.  Equality handles N:1 groups where multiple ref
    segments map to the same target.

    The score is:  contradicted_junctions / matched_ref_neighbours
    Range [0, 1]; 0 = fully consistent, 1 = every junction contradicted.
    Matches with no matched neighbours get score 0 (benefit of the doubt).
    """
    scores: dict[tuple[Any, Any], float] = {}

    for r in accepted:
        ref_id, target_id = r.ref_id, r.target_id
        ref_neighbours = ref_graph.neighbors(ref_id)

        # Only consider ref neighbours that themselves have accepted matches.
        matched_ref_nbrs = [n for n in ref_neighbours if n in ref_to_targets]
        if not matched_ref_nbrs:
            scores[(ref_id, target_id)] = 0.0
            continue

        target_nbrs = set(target_graph.neighbors(target_id)) | {target_id}
        contradicted = 0
        for ref_nbr in matched_ref_nbrs:
            nbr_targets = ref_to_targets[ref_nbr]
            # Is *any* of ref_nbr's matched targets adjacent to (or equal to) target_id?
            if not nbr_targets & target_nbrs:
                contradicted += 1

        scores[(ref_id, target_id)] = contradicted / len(matched_ref_nbrs)

    return scores


# ---------------------------------------------------------------------------
# Check 2 — Neighbourhood coherence
# ---------------------------------------------------------------------------


def _neighbourhood_coherence_scores(
    accepted: list[MatchResult],
    ref_graph: SparseGraph,
    target_graph: SparseGraph,
    ref_to_targets: dict[Any, set[Any]],
) -> dict[tuple[Any, Any], float]:
    """Score how isolated a match is within its local neighbourhood.

    For match (ref_A, target_X):
    - Look at ref_A's graph neighbours in the ref graph.
    - Count how many of those neighbours are matched (n_matched_nbrs).
    - Of those, count how many have a matched target that is a neighbour
      of (or equal to) target_X in the target graph (n_coherent).
      Equality handles N:1 groups where multiple ref segments share a target.

    coherence = n_coherent / n_matched_nbrs  (1 = all coherent, 0 = isolated)
    Returned score = 1 - coherence  (so higher = worse, matching convention).

    Matches whose ref has no matched neighbours get score 0.
    """
    scores: dict[tuple[Any, Any], float] = {}

    for r in accepted:
        ref_id, target_id = r.ref_id, r.target_id
        ref_neighbours = ref_graph.neighbors(ref_id)
        matched_ref_nbrs = [n for n in ref_neighbours if n in ref_to_targets]

        if not matched_ref_nbrs:
            scores[(ref_id, target_id)] = 0.0
            continue

        target_nbrs = set(target_graph.neighbors(target_id)) | {target_id}

        coherent = 0
        for ref_nbr in matched_ref_nbrs:
            nbr_targets = ref_to_targets[ref_nbr]
            if nbr_targets & target_nbrs:
                coherent += 1

        scores[(ref_id, target_id)] = 1.0 - (coherent / len(matched_ref_nbrs))

    return scores


# ---------------------------------------------------------------------------
# Check 3 — Degree excess
# ---------------------------------------------------------------------------


def _build_junction_nodes(
    gdf: gpd.GeoDataFrame,
    id_column: str,
    tolerance: float,
) -> dict[int, set[Any]]:
    """Map junction node IDs → the set of segment IDs incident at that junction.

    A junction node is identified by clustering segment endpoints within
    *tolerance* (same approach as _build_segment_adjacency_graph).  We
    return only junctions with degree ≥ 2 (i.e. shared by ≥ 2 segments).
    """
    if id_column in gdf.columns:
        seg_ids = gdf[id_column].values
    else:
        seg_ids = gdf.index.values

    endpoints: list[tuple[float, float]] = []
    ep_to_seg: list[Any] = []

    for seg_idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        coords = geom.coords
        if len(coords) < 2:
            continue
        endpoints.append(coords[0][:2])
        ep_to_seg.append(seg_ids[seg_idx])
        endpoints.append(coords[-1][:2])
        ep_to_seg.append(seg_ids[seg_idx])

    if len(endpoints) < 2:
        return {}

    ep_array = np.array(endpoints)
    tree = cKDTree(ep_array)

    # Union-find to cluster endpoints within tolerance.
    parent = list(range(len(endpoints)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in tree.query_pairs(tolerance):
        union(i, j)

    cluster_to_segs: dict[int, set[Any]] = defaultdict(set)
    for ep_idx in range(len(endpoints)):
        cluster_to_segs[find(ep_idx)].add(ep_to_seg[ep_idx])

    # Keep only junctions (≥ 2 distinct segments).
    return {k: v for k, v in cluster_to_segs.items() if len(v) >= 2}


def _degree_excess_scores(
    accepted: list[MatchResult],
    ref_junctions: dict[int, set[Any]],
    ref_to_targets: dict[Any, set[Any]],
) -> dict[tuple[Any, Any], float]:
    """Score matches involved in junctions where target-side degree exceeds ref-side degree.

    At each reference junction with K incident ref segments, count how many
    distinct target segments are matched to those K refs.  If that count
    exceeds K, there's a degree excess.

    The per-match score is:  max(0, (target_count - ref_count) / ref_count)
    for the worst junction that match participates in.
    """
    # Pre-compute excess per junction.
    junction_excess: dict[int, float] = {}
    junction_for_ref: dict[Any, list[int]] = defaultdict(list)

    for junc_id, ref_seg_ids in ref_junctions.items():
        ref_count = len(ref_seg_ids)
        target_seg_ids: set[Any] = set()
        for rid in ref_seg_ids:
            if rid in ref_to_targets:
                target_seg_ids.update(ref_to_targets[rid])
        target_count = len(target_seg_ids)
        excess = max(0.0, (target_count - ref_count) / ref_count)
        junction_excess[junc_id] = excess
        for rid in ref_seg_ids:
            junction_for_ref[rid].append(junc_id)

    scores: dict[tuple[Any, Any], float] = {}
    for r in accepted:
        juncs = junction_for_ref.get(r.ref_id, [])
        if not juncs:
            scores[(r.ref_id, r.target_id)] = 0.0
        else:
            scores[(r.ref_id, r.target_id)] = max(junction_excess[j] for j in juncs)

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_graph_consistency(
    results: list[MatchResult],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    snap_tolerance: float = DEFAULT_SNAP_TOLERANCE_M,
    contradiction_threshold: float = 0.5,
    coherence_threshold: float = 0.75,
    degree_excess_threshold: float = 0.5,
) -> list[MatchResult]:
    """Validate optimizer output for topological consistency.

    Builds segment-adjacency graphs from reference and target geometries,
    then runs three checks on each accepted match.  Matches that fail any
    check are demoted from MATCH → REVIEW.

    Only MATCH-decision results are checked; REVIEW and NO_MATCH results
    pass through unchanged.

    Args:
        results: Optimized MatchResult list (output of optimizer).
        reference: Reference GeoDataFrame (projected CRS).
        target: Target GeoDataFrame (projected CRS).
        ref_id_column: Reference ID column.
        target_id_column: Target ID column.
        snap_tolerance: Endpoint snap tolerance in metres.
        contradiction_threshold: Junction contradiction score above which
            a match is demoted.  Range [0, 1].
        coherence_threshold: Neighbourhood incoherence score above which
            a match is demoted.  Range [0, 1].
        degree_excess_threshold: Degree excess ratio above which a match
            is demoted.  Range [0, ∞).

    Returns:
        New list of MatchResult with demotions applied.
    """
    import time

    if not results:
        return results

    t0 = time.perf_counter()

    # Partition into accepted (MATCH) and rest.
    accepted = [r for r in results if r.decision == MatchDecision.MATCH]
    if not accepted:
        logger.info("  Graph consistency: no MATCH results to validate")
        return results

    # ------------------------------------------------------------------
    # Build graphs
    # ------------------------------------------------------------------
    ref_graph = _build_segment_adjacency_graph(reference, ref_id_column, snap_tolerance)
    target_graph = _build_segment_adjacency_graph(target, target_id_column, snap_tolerance)
    logger.debug(
        f"  Graph consistency: ref graph {ref_graph.n_nodes} nodes / {ref_graph.n_edges} edges, "
        f"target graph {target_graph.n_nodes} nodes / {target_graph.n_edges} edges"
    )

    ref_to_targets, target_to_refs = _build_match_maps(accepted)

    # ------------------------------------------------------------------
    # Run checks
    # ------------------------------------------------------------------
    junction_scores = _junction_contradiction_scores(
        accepted, ref_graph, target_graph, ref_to_targets
    )
    coherence_scores = _neighbourhood_coherence_scores(
        accepted, ref_graph, target_graph, ref_to_targets
    )
    ref_junctions = _build_junction_nodes(reference, ref_id_column, snap_tolerance)
    degree_scores = _degree_excess_scores(accepted, ref_junctions, ref_to_targets)

    # ------------------------------------------------------------------
    # Determine demotions
    # ------------------------------------------------------------------
    demote_keys: set[tuple[Any, Any]] = set()
    demote_reasons: dict[tuple[Any, Any], list[str]] = defaultdict(list)

    for key in junction_scores:
        if junction_scores[key] > contradiction_threshold:
            demote_keys.add(key)
            demote_reasons[key].append(f"junction_contradiction={junction_scores[key]:.2f}")
        if coherence_scores.get(key, 0.0) > coherence_threshold:
            demote_keys.add(key)
            demote_reasons[key].append(f"neighbourhood_incoherence={coherence_scores[key]:.2f}")
        if degree_scores.get(key, 0.0) > degree_excess_threshold:
            demote_keys.add(key)
            demote_reasons[key].append(f"degree_excess={degree_scores[key]:.2f}")

    # ------------------------------------------------------------------
    # Apply demotions
    # ------------------------------------------------------------------
    if not demote_keys:
        t1 = time.perf_counter()
        logger.info(f"  Graph consistency: all {len(accepted)} matches consistent ({t1 - t0:.2f}s)")
        return results

    validated: list[MatchResult] = []
    for r in results:
        key = (r.ref_id, r.target_id)
        if key in demote_keys and r.decision == MatchDecision.MATCH:
            reasons = demote_reasons[key]
            validated.append(
                MatchResult(
                    ref_id=r.ref_id,
                    target_id=r.target_id,
                    decision=MatchDecision.REVIEW,
                    confidence=r.confidence,
                    score_breakdown=r.score_breakdown,
                    features={
                        **r.features,
                        "graph_consistency_flag": 1.0,
                        "graph_consistency_reasons": "; ".join(reasons),
                    },
                    gers_start_frac=r.gers_start_frac,
                    gers_end_frac=r.gers_end_frac,
                    local_start_frac=r.local_start_frac,
                    local_end_frac=r.local_end_frac,
                )
            )
        else:
            validated.append(r)

    n_demoted = len(demote_keys)
    t1 = time.perf_counter()
    logger.info(
        f"  Graph consistency: demoted {n_demoted}/{len(accepted)} matches "
        f"to REVIEW ({t1 - t0:.2f}s)"
    )

    return validated
