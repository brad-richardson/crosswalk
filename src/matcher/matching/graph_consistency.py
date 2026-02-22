"""Post-optimizer graph consistency validation.

Builds lightweight segment-adjacency graphs from reference and target
GeoDataFrame endpoints, then checks whether accepted matches are
topologically consistent with their graph neighbourhoods.

Design principles:
- Precision-biased: demotes suspicious matches from MATCH → REVIEW
  (never hard rejects, never promotes).
- Runs after the optimizer, before bridge generation.
- O(N × avg_degree) ≈ O(N) for sparse road networks.

One check is applied, producing a per-match contradiction score:

1. **Junction contradiction**: For each reference junction where matched
   ref segments meet, verify that the matched target segments also share
   a junction in the target graph.

Note: Degree excess was removed because it's provably zero when correctly
accounting for M:N match groups (each group contains at least one ref
at the junction, so the group count can never exceed the junction degree).
Junction contradiction already catches the topological inconsistencies
that degree excess was intended to detect.
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
# Public API
# ---------------------------------------------------------------------------


def validate_graph_consistency(
    results: list[MatchResult],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    snap_tolerance: float = DEFAULT_SNAP_TOLERANCE_M,
    contradiction_threshold: float = 0.99,
) -> list[MatchResult]:
    """Validate optimizer output for topological consistency.

    Builds segment-adjacency graphs from reference and target geometries,
    then checks each accepted match for junction contradiction.  Matches
    that fail are demoted from MATCH → REVIEW.

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

    ref_to_targets, _target_to_refs = _build_match_maps(accepted)

    # ------------------------------------------------------------------
    # Run junction contradiction check
    # ------------------------------------------------------------------
    junction_scores = _junction_contradiction_scores(
        accepted, ref_graph, target_graph, ref_to_targets
    )

    # ------------------------------------------------------------------
    # Apply confidence penalties
    # ------------------------------------------------------------------
    # Matches with junction contradiction above the threshold get their
    # confidence scaled down by the contradiction score.  This causes
    # them to drop below bridge_min_confidence and be filtered out of
    # the bridge file, rather than just being relabeled REVIEW.
    penalised_keys: dict[tuple[Any, Any], float] = {}
    for key, score in junction_scores.items():
        if score > contradiction_threshold:
            penalised_keys[key] = score

    if not penalised_keys:
        t1 = time.perf_counter()
        logger.info(f"  Graph consistency: all {len(accepted)} matches consistent ({t1 - t0:.2f}s)")
        return results

    validated: list[MatchResult] = []
    for r in results:
        key = (r.ref_id, r.target_id)
        if key in penalised_keys and r.decision == MatchDecision.MATCH:
            score = penalised_keys[key]
            # Scale confidence by (1 - contradiction_score)
            new_confidence = r.confidence * (1.0 - score)
            validated.append(
                MatchResult(
                    ref_id=r.ref_id,
                    target_id=r.target_id,
                    decision=MatchDecision.REVIEW,
                    confidence=new_confidence,
                    score_breakdown=r.score_breakdown,
                    features={
                        **r.features,
                        "graph_consistency_flag": 1.0,
                        "graph_consistency_reasons": f"junction_contradiction={score:.2f}",
                        "graph_consistency_original_confidence": r.confidence,
                    },
                    gers_start_frac=r.gers_start_frac,
                    gers_end_frac=r.gers_end_frac,
                    local_start_frac=r.local_start_frac,
                    local_end_frac=r.local_end_frac,
                )
            )
        else:
            validated.append(r)

    n_penalised = len(penalised_keys)
    t1 = time.perf_counter()
    logger.info(
        f"  Graph consistency: penalised {n_penalised}/{len(accepted)} matches ({t1 - t0:.2f}s)"
    )

    return validated
