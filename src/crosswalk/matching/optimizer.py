"""Global match optimization with M:N group detection.

Resolves conflicts using a bipartite connected-components approach
that handles 1:1, 1:N, N:1, and M:N match groups in one pass.

Algorithm:
1. Filter results above min_confidence
2. Build bipartite graph: nodes = (ref_ids ∪ target_ids), edges = match pairs
3. Find connected components via BFS → list of component edge-lists
4. For each component, classify as 1:1, 1:N, N:1, or M:N based on
   contiguity of the multi-side
5. Run greedy 1:1 optimization on unclaimed leftovers
6. Return group_results + optimized_1to1
"""

import hashlib
from collections import defaultdict, deque
from typing import Any

import geopandas as gpd
import numpy as np
from loguru import logger
from scipy.spatial import cKDTree
from shapely import LineString

from ..config import DEFAULT_SNAP_TOLERANCE_M, MAX_ALIGNMENT_OVERLAP_M, NAMES_COLUMN, settings
from .sliver import sliver_edges_for_match_results
from .types import MatchDecision, MatchResult, MatchType


def _geom_lookup(gdf: gpd.GeoDataFrame, id_column: str) -> dict[Any, LineString]:
    """Build an id -> geometry lookup, falling back to the index if needed."""
    if id_column in gdf.columns:
        return dict(zip(gdf[id_column], gdf.geometry))
    return dict(zip(gdf.index, gdf.geometry))


def _name_lookup(gdf: gpd.GeoDataFrame, id_column: str) -> dict[Any, Any]:
    """Build an id -> name lookup from ``NAMES_COLUMN`` (empty if absent)."""
    if NAMES_COLUMN not in gdf.columns:
        return {}
    keys = gdf[id_column] if id_column in gdf.columns else gdf.index
    return dict(zip(keys, gdf[NAMES_COLUMN]))


def compute_sliver_candidate_edges(
    results: list[MatchResult],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
) -> set[tuple[Any, Any]]:
    """Classify candidate edges as junction slivers (shared hybrid rule).

    Thin wrapper over :func:`matching.sliver.sliver_edges_for_match_results`
    that builds the geometry lookups and detects whether the GeoDataFrames are
    in a metric (projected) CRS. Used by both the grouping optimizer and the
    groups-sidecar export so they agree on the exact same sliver set.
    """
    ref_geoms = _geom_lookup(reference, ref_id_column)
    target_geoms = _geom_lookup(target, target_id_column)
    metric = not (reference.crs is not None and reference.crs.is_geographic)
    return sliver_edges_for_match_results(results, ref_geoms, target_geoms, metric=metric)


def compute_group_id(ref_ids: set, target_ids: set) -> str:
    """Compute a deterministic short ID for a match group.

    Hashes sorted ref_ids + target_ids into an 8-character hex string.
    The same set of IDs always produces the same group_id.

    Args:
        ref_ids: Set of reference segment IDs in the group
        target_ids: Set of target segment IDs in the group

    Returns:
        8-character hex string uniquely identifying this group
    """
    key = (
        "|".join(sorted(str(r) for r in ref_ids))
        + "||"
        + "|".join(sorted(str(t) for t in target_ids))
    )
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def optimize_matches_greedy(
    results: list[MatchResult],
    min_confidence: float = 0.5,
) -> list[MatchResult]:
    """Greedy 1:1 assignment for large datasets.

    Sorts candidates by confidence and greedily assigns matches,
    ensuring each ref and target is matched at most once.

    Time complexity: O(n log n) for sorting
    Space complexity: O(n) where n = number of candidates

    Args:
        results: List of MatchResult objects
        min_confidence: Minimum confidence to consider a match

    Returns:
        List of greedily-selected MatchResult objects (1:1 assignments)
    """
    import time

    t0 = time.perf_counter()
    logger.info(f"Optimizing {len(results)} match results with greedy algorithm...")

    # Filter by minimum confidence
    valid_results = [r for r in results if r.confidence >= min_confidence]
    logger.info(f"  {len(valid_results)} results above min_confidence={min_confidence}")

    if not valid_results:
        return []

    # Sort by confidence (highest first). Break ties on stable string ids so the
    # greedy assignment is independent of input list order (which upstream steps
    # can permute) and of Python set/dict hash-seed iteration order — otherwise
    # equal-confidence candidates competing for a shared node resolve arbitrarily.
    sorted_results = sorted(
        valid_results, key=lambda r: (-r.confidence, str(r.ref_id), str(r.target_id))
    )

    assigned_refs: set = set()
    assigned_targets: set = set()
    optimal_matches = []

    for result in sorted_results:
        if result.ref_id not in assigned_refs and result.target_id not in assigned_targets:
            optimal_matches.append(result)
            assigned_refs.add(result.ref_id)
            assigned_targets.add(result.target_id)

    logger.info(
        f"  Found {len(optimal_matches)} greedy 1:1 matches in {time.perf_counter() - t0:.2f}s"
    )

    return optimal_matches


def find_match_components(
    results: list[MatchResult],
    min_confidence: float,
    sliver_edges: set[tuple[Any, Any]] | None = None,
    glue_min_confidence: float | None = None,
) -> list[list[MatchResult]]:
    """Find connected components in the bipartite ref-target match graph.

    Builds an undirected bipartite graph with namespaced nodes
    ("ref", id) / ("target", id) and edges from match pairs.
    Returns one list of MatchResult per connected component.

    Args:
        results: All match results
        min_confidence: Minimum confidence threshold
        sliver_edges: Optional set of ``(ref_id, target_id)`` pairs classified
            as junction slivers (see ``matching.sliver``). Sliver edges
            contribute NO adjacency when building components, so a junction
            kiss can never weld two otherwise-independent groups into one
            monster component. A sliver edge is attached to a component's edge
            list only when BOTH its endpoints already landed in that same
            component via non-sliver edges (it remains an ordinary group
            member there). A sliver whose endpoints end up in different
            components — or whose endpoints have no non-sliver edge at all —
            is dropped from grouping entirely: the only evidence tying those
            segments together is a junction artifact.
        glue_min_confidence: Optional grouping-only confidence prune. Candidate
            edges with ``min_confidence <= confidence < glue_min_confidence``
            are treated like slivers for adjacency (they never weld components
            together) but remain in a component's edge list when their
            endpoints co-land via stronger edges. Defaults to ``min_confidence``
            (no prune).

    Returns:
        List of components, each a list of MatchResult edges
    """
    valid = [r for r in results if r.confidence >= min_confidence]
    if not valid:
        return []

    sliver_edges = sliver_edges or set()
    glue_min = min_confidence if glue_min_confidence is None else glue_min_confidence

    # Split structural (component-building) edges from non-gluing edges. A
    # non-gluing edge is a sliver OR a weak edge below ``glue_min`` (grouping-
    # only confidence prune): neither builds adjacency, both only attach to a
    # component when their endpoints already co-land there via structural
    # edges. All are deduplicated to the highest-confidence result per pair.
    structural: list[MatchResult] = []
    sliver_best: dict[tuple[Any, Any], MatchResult] = {}
    for r in valid:
        pair = (r.ref_id, r.target_id)
        if pair in sliver_edges or r.confidence < glue_min:
            if pair not in sliver_best or r.confidence > sliver_best[pair].confidence:
                sliver_best[pair] = r
        else:
            structural.append(r)

    # Build adjacency list with namespaced nodes (structural edges only)
    adj: dict[tuple[str, Any], set[tuple[str, Any]]] = defaultdict(set)
    # Track which edges (MatchResults) connect each node pair
    edge_lookup: dict[tuple[tuple[str, Any], tuple[str, Any]], MatchResult] = {}

    for r in structural:
        ref_node = ("ref", r.ref_id)
        tgt_node = ("target", r.target_id)
        adj[ref_node].add(tgt_node)
        adj[tgt_node].add(ref_node)
        # Keep highest confidence if duplicate pair
        key = (ref_node, tgt_node)
        if key not in edge_lookup or r.confidence > edge_lookup[key].confidence:
            edge_lookup[key] = r

    # BFS to find connected components
    visited: set[tuple[str, Any]] = set()
    components: list[list[MatchResult]] = []
    node_component: dict[tuple[str, Any], int] = {}

    for start_node in adj:
        if start_node in visited:
            continue

        # BFS from start_node
        component_nodes: set[tuple[str, Any]] = set()
        queue = deque([start_node])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component_nodes.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        # Collect edges for this component. Sort node/neighbor iteration by id so
        # the emitted edge order is independent of set iteration order (which is
        # hash-seed dependent for str ids). The component's edge LIST order flows
        # downstream into grouping and greedy tie-breaks, so it must be canonical.
        component_edges: list[MatchResult] = []
        ref_nodes = sorted((n for n in component_nodes if n[0] == "ref"), key=lambda n: str(n[1]))
        tgt_node_set = {n for n in component_nodes if n[0] == "target"}

        for rn in ref_nodes:
            for tn in sorted(adj[rn], key=lambda n: str(n[1])):
                if tn in tgt_node_set:
                    key = (rn, tn)
                    if key in edge_lookup:
                        component_edges.append(edge_lookup[key])

        if component_edges:
            for n in component_nodes:
                node_component[n] = len(components)
            components.append(component_edges)

    # Attach sliver edges whose endpoints both landed in the SAME component via
    # structural edges. Cross-component (or unattached) slivers are dropped.
    n_attached = 0
    n_dropped = 0
    for (rid, tid), r in sliver_best.items():
        ci = node_component.get(("ref", rid))
        if ci is not None and ci == node_component.get(("target", tid)):
            components[ci].append(r)
            n_attached += 1
        else:
            n_dropped += 1
    if sliver_best:
        logger.debug(
            f"  Sliver edges: {n_attached} kept as in-component candidates, "
            f"{n_dropped} dropped (would have glued independent components)"
        )

    return components


def _normalize_name(name: Any) -> str:
    """Normalize a segment name to a lowercased comparison key.

    Accepts plain strings and Overture-style name dicts (``{"primary": ...}``).
    Returns ``""`` for missing / unusable names so callers can treat empty as
    "no name evidence" rather than a match.
    """
    if name is None:
        return ""
    if isinstance(name, dict):
        for key in ("primary", "common", "name", "value"):
            v = name.get(key)
            if isinstance(v, str) and v:
                name = v
                break
        else:
            name = next((v for v in name.values() if isinstance(v, str) and v), "")
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def _endpoints_are_collinear(
    ep_a: tuple[float, float],
    nb_a: tuple[float, float],
    ep_b: tuple[float, float],
    nb_b: tuple[float, float],
    max_turn_deg: float,
) -> bool:
    """Test whether two segments form a collinear continuation at a shared endpoint.

    ``ep_*`` is the shared (touching) endpoint of each segment and ``nb_*`` its
    adjacent interior vertex, so ``nb_* - ep_*`` is the direction the segment
    *leaves* the shared point. Two collinear continuations leave in nearly
    opposite directions (a straight through-path), so the deflection from
    straight is ``180 - angle(vec_a, vec_b)``. Returns True when that deflection
    is within ``max_turn_deg`` (a genuine corridor) rather than a sharp turn (a
    perpendicular junction kiss between two different streets).
    """
    va = (nb_a[0] - ep_a[0], nb_a[1] - ep_a[1])
    vb = (nb_b[0] - ep_b[0], nb_b[1] - ep_b[1])
    na = (va[0] ** 2 + va[1] ** 2) ** 0.5
    nb = (vb[0] ** 2 + vb[1] ** 2) ** 0.5
    if na == 0.0 or nb == 0.0:
        return False
    cos = (va[0] * vb[0] + va[1] * vb[1]) / (na * nb)
    cos = max(-1.0, min(1.0, cos))
    angle = np.degrees(np.arccos(cos))
    turn = 180.0 - angle
    return turn <= max_turn_deg


def build_contiguity_adjacency(
    ids: list[Any],
    geom_lookup: dict[Any, LineString],
    tolerance: float,
    require_collinear: bool = False,
    max_turn_deg: float = 40.0,
    name_lookup: dict[Any, Any] | None = None,
) -> dict[Any, set[Any]]:
    """Build an endpoint-proximity adjacency map over a set of IDs.

    Two geometries are adjacent if one's endpoint is within ``tolerance`` of the
    other's endpoint. Uses a KD-tree for O(n log n) proximity detection.

    This is the single shared primitive for endpoint-based contiguity: it backs
    both ``_find_contiguous_id_groups`` (connected components) here in the
    optimizer and the contiguous ref-chain enumeration in
    ``matching/alternatives.py`` — so assignment *options* can express the same
    multi-ref spans the optimizer itself can produce.

    Args:
        ids: List of IDs to check (duplicates collapse onto the same key).
        geom_lookup: Dictionary mapping ID to LineString geometry, in the same
            coordinate units as ``tolerance``.
        tolerance: Maximum endpoint distance to consider contiguous.
        require_collinear: When True, a proximity pair only becomes adjacent if
            the two segments are a collinear continuation at the shared endpoint
            (deflection <= ``max_turn_deg``) OR share a normalized name (the
            "same-street rescue" for gently curving named corridors). This is
            the corridor-aware gate used by the M:N branch so perpendicular
            junction kisses do not chain independent corridors together.
        max_turn_deg: Deflection threshold (degrees) for the collinearity gate.
        name_lookup: Optional ID -> name map (string or Overture name dict) for
            the same-name rescue. Ignored when ``require_collinear`` is False.

    Returns:
        Dict mapping each input ID to the set of IDs it is contiguous with.
        Every input ID is a key (empty set if it has no neighbour or has
        missing / degenerate geometry).
    """
    adjacency: dict[Any, set[Any]] = {id_: set() for id_ in ids}

    all_endpoints: list = []
    endpoint_to_id: list = []  # which id does each endpoint belong to
    endpoint_neighbor: list = []  # adjacent interior vertex (for direction)
    for id_ in ids:
        geom = geom_lookup.get(id_)
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        all_endpoints.append(coords[0][:2])
        endpoint_to_id.append(id_)
        endpoint_neighbor.append(coords[1][:2])
        all_endpoints.append(coords[-1][:2])
        endpoint_to_id.append(id_)
        endpoint_neighbor.append(coords[-2][:2])

    if len(all_endpoints) < 2:
        return adjacency

    name_cache: dict[Any, str] = {}

    def _name(id_: Any) -> str:
        if id_ not in name_cache:
            name_cache[id_] = _normalize_name(name_lookup.get(id_)) if name_lookup else ""
        return name_cache[id_]

    tree = cKDTree(np.array(all_endpoints))
    for ep_i, ep_j in tree.query_pairs(tolerance):
        a = endpoint_to_id[ep_i]
        b = endpoint_to_id[ep_j]
        if a == b:
            continue
        if require_collinear:
            collinear = _endpoints_are_collinear(
                all_endpoints[ep_i],
                endpoint_neighbor[ep_i],
                all_endpoints[ep_j],
                endpoint_neighbor[ep_j],
                max_turn_deg,
            )
            if not collinear:
                na, nb = _name(a), _name(b)
                if not (na and na == nb):
                    continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    return adjacency


def _find_contiguous_id_groups(
    ids: list[Any],
    geom_lookup: dict[Any, LineString],
    tolerance: float,
    require_collinear: bool = False,
    max_turn_deg: float = 40.0,
    name_lookup: dict[Any, Any] | None = None,
) -> list[list[Any]]:
    """Find groups of IDs whose geometries are contiguous.

    Two geometries are contiguous if one's endpoint is within tolerance
    of the other's endpoint. Uses KD-tree for O(n log n) proximity detection.

    Args:
        ids: List of IDs to check
        geom_lookup: Dictionary mapping ID to LineString geometry
        tolerance: Maximum endpoint distance to consider contiguous (meters)
        require_collinear: Gate contiguity on collinear continuation or same
            name (see :func:`build_contiguity_adjacency`).
        max_turn_deg: Deflection threshold (degrees) for the collinearity gate.
        name_lookup: Optional ID -> name map for the same-name rescue.

    Returns:
        List of groups, each a list of contiguous IDs.
        Single-element groups are included for IDs with no contiguous neighbors.
    """
    if len(ids) <= 1:
        return [ids] if ids else []

    adjacency = build_contiguity_adjacency(
        ids,
        geom_lookup,
        tolerance,
        require_collinear=require_collinear,
        max_turn_deg=max_turn_deg,
        name_lookup=name_lookup,
    )

    # Find connected components using BFS.
    visited: set[Any] = set()
    groups: list[list[Any]] = []
    for id_ in ids:
        if id_ in visited:
            continue
        group: list[Any] = []
        queue = deque([id_])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            group.append(node)
            # Sort neighbours by stable id: adjacency values are sets, whose
            # iteration order is hash-seed dependent for str ids. The BFS append
            # order determines each group's member order, which flows into the
            # optimizer's sub-component edge order — so it must be canonical.
            for neighbor in sorted(adjacency[node], key=str):
                if neighbor not in visited:
                    queue.append(neighbor)
        groups.append(group)

    return groups


def _create_group_results(
    results: list[MatchResult],
    match_type: MatchType,
    review_pairs: set[tuple[Any, Any]] | None = None,
) -> list[MatchResult]:
    """Tag MatchResult objects with group metadata.

    Preserves original MatchResult objects (including alignment fractions,
    confidence, score_breakdown) and adds group metadata to features.

    Sets group decision: MATCH if avg_confidence >= optimizer_review_threshold,
    else REVIEW.

    Args:
        results: MatchResult objects belonging to this group
        match_type: MatchType value (ONE_TO_N, N_TO_ONE, M_TO_N)
        review_pairs: Optional set of ``(ref_id, target_id)`` pairs whose
            decision is forced to REVIEW regardless of the group decision (the
            #367 anti-crossing demote-to-REVIEW; see
            ``_contested_small_span_review_pairs``). These edges stay in the
            group — only their decision is demoted — and carry the
            ``PARALLEL_SIBLING_REVIEW_FLAG`` feature.

    Returns:
        List of tagged MatchResult objects (same objects, mutated features)
    """
    if not results:
        return []

    review_pairs = review_pairs or set()

    avg_confidence = np.mean([r.confidence for r in results])
    group_decision = (
        MatchDecision.MATCH
        if avg_confidence >= settings.optimizer_review_threshold
        else MatchDecision.REVIEW
    )

    ref_ids = set(r.ref_id for r in results)
    target_ids = set(r.target_id for r in results)
    group_id = compute_group_id(ref_ids, target_ids)

    tagged: list[MatchResult] = []
    for r in results:
        demoted = (r.ref_id, r.target_id) in review_pairs
        decision = MatchDecision.REVIEW if demoted else group_decision
        features = {
            **r.features,
            "match_type": match_type,
            "group_id": group_id,
            "group_size": len(results),
            "group_ref_count": len(ref_ids),
            "group_target_count": len(target_ids),
        }
        if demoted:
            features[PARALLEL_SIBLING_REVIEW_FLAG] = 1.0
        # Create new MatchResult preserving all original fields
        tagged.append(
            MatchResult(
                ref_id=r.ref_id,
                target_id=r.target_id,
                decision=decision,
                confidence=r.confidence,
                score_breakdown=r.score_breakdown,
                features=features,
                ref_idx=r.ref_idx,
                target_idx=r.target_idx,
                gers_start_frac=r.gers_start_frac,
                gers_end_frac=r.gers_end_frac,
                local_start_frac=r.local_start_frac,
                local_end_frac=r.local_end_frac,
            )
        )

    return tagged


def _expand_greedy_matches(
    greedy_matches: list[MatchResult],
    all_candidates: list[MatchResult],
    ref_geoms: dict[Any, LineString],
    target_geoms: dict[Any, LineString],
    tolerance: float,
    min_confidence: float,
) -> list[MatchResult]:
    """Expand greedy 1:1 matches to 1:N/N:1 groups where contiguous candidates exist.

    After greedy 1:1 optimization, some refs/targets have additional high-confidence
    candidates for contiguous segments that were excluded by the 1:1 constraint.
    This post-processing step adds those contiguous candidates, expanding 1:1 matches
    to 1:N (one ref → multiple contiguous targets) or N:1 (multiple contiguous refs
    → one target) groups.

    This is safe because it only ADDS matches — existing assignments are never removed.

    Args:
        greedy_matches: 1:1 matches from greedy optimization
        all_candidates: All candidate MatchResults (pre-optimization pool)
        ref_geoms: Reference geometry lookup
        target_geoms: Target geometry lookup
        tolerance: Contiguity tolerance in meters
        min_confidence: Minimum confidence for expansion candidates

    Returns:
        Expanded list of MatchResult objects (original 1:1 + new group members)
    """
    if not greedy_matches:
        return []

    # Track which (ref, target) pairs already exist in greedy matches
    existing_pairs = {(r.ref_id, r.target_id) for r in greedy_matches}

    # Build candidate lookup from ALL candidates (not just greedy winners)
    candidates_by_ref: dict[Any, list[MatchResult]] = defaultdict(list)
    candidates_by_target: dict[Any, list[MatchResult]] = defaultdict(list)
    for r in all_candidates:
        if r.confidence >= min_confidence:
            candidates_by_ref[r.ref_id].append(r)
            candidates_by_target[r.target_id].append(r)

    expanded: list[MatchResult] = []
    expanded_pairs: set = set()

    # 1:N expansion: for each assigned ref, find contiguous candidate targets.
    # Targets already assigned to OTHER refs are allowed — this creates
    # legitimate N:1/M:N patterns where a target is covered by multiple refs.
    for match in greedy_matches:
        ref_id = match.ref_id
        assigned_target = match.target_id

        # Find other high-confidence candidates for this ref (any target, even claimed)
        other_targets = [
            c.target_id
            for c in candidates_by_ref[ref_id]
            if c.target_id != assigned_target
            and (ref_id, c.target_id) not in existing_pairs
            and (ref_id, c.target_id) not in expanded_pairs
        ]

        if not other_targets:
            continue

        # Check which are contiguous with the assigned target
        all_target_ids = [assigned_target] + other_targets
        target_groups = _find_contiguous_id_groups(all_target_ids, target_geoms, tolerance)

        # Find the group containing the assigned target
        for tg in target_groups:
            if assigned_target in tg and len(tg) > 1:
                new_target_ids = set(tg) - {assigned_target}
                for c in candidates_by_ref[ref_id]:
                    if (
                        c.target_id in new_target_ids
                        and (ref_id, c.target_id) not in expanded_pairs
                    ):
                        expanded.append(c)
                        expanded_pairs.add((ref_id, c.target_id))

    # N:1 expansion: for each assigned target, find contiguous candidate refs.
    # Refs already assigned to OTHER targets are allowed.
    for match in greedy_matches:
        target_id = match.target_id
        assigned_ref = match.ref_id

        other_refs = [
            c.ref_id
            for c in candidates_by_target[target_id]
            if c.ref_id != assigned_ref
            and (c.ref_id, target_id) not in existing_pairs
            and (c.ref_id, target_id) not in expanded_pairs
        ]

        if not other_refs:
            continue

        all_ref_ids = [assigned_ref] + other_refs
        ref_groups = _find_contiguous_id_groups(all_ref_ids, ref_geoms, tolerance)

        for rg in ref_groups:
            if assigned_ref in rg and len(rg) > 1:
                new_ref_ids = set(rg) - {assigned_ref}
                for c in candidates_by_target[target_id]:
                    if c.ref_id in new_ref_ids and (c.ref_id, target_id) not in expanded_pairs:
                        expanded.append(c)
                        expanded_pairs.add((c.ref_id, target_id))

    if not expanded:
        return greedy_matches

    # Group expanded by ref (for 1:N tagging) and by target (for N:1 tagging)
    expanded_by_ref: dict[Any, list[MatchResult]] = defaultdict(list)
    expanded_by_target: dict[Any, list[MatchResult]] = defaultdict(list)
    for r in expanded:
        expanded_by_ref[r.ref_id].append(r)
        expanded_by_target[r.target_id].append(r)

    # Re-tag original greedy matches that were expanded
    result: list[MatchResult] = []
    for match in greedy_matches:
        ref_expanded = match.ref_id in expanded_by_ref
        target_expanded = match.target_id in expanded_by_target

        if ref_expanded:
            # This ref got additional targets → tag as 1:N
            all_group = [match] + expanded_by_ref[match.ref_id]
            tagged = _create_group_results(all_group, MatchType.ONE_TO_N)
            result.extend(tagged)
            # Remove from expanded_by_ref so we don't double-count
            del expanded_by_ref[match.ref_id]
        elif target_expanded:
            # This target got additional refs → tag as N:1
            all_group = [match] + expanded_by_target[match.target_id]
            tagged = _create_group_results(all_group, MatchType.N_TO_ONE)
            result.extend(tagged)
            del expanded_by_target[match.target_id]
        else:
            result.append(match)

    return result


def _merge_singletons_by_alignment(
    contiguous_group_results: list[MatchResult],
    singleton_results: list[MatchResult],
    frac_start_key: str,  # "gers_start_frac" or "local_start_frac"
    frac_end_key: str,  # "gers_end_frac" or "local_end_frac"
    shared_segment_length_m: float,
    max_overlap_m: float = MAX_ALIGNMENT_OVERLAP_M,
) -> tuple[list[MatchResult], list[MatchResult]]:
    """Merge singletons into groups if their alignment fractions don't overlap.

    For each singleton, checks whether its coverage on the shared segment
    is compatible (non-overlapping) with the existing contiguous group.
    If so, merges it into the group.

    Args:
        contiguous_group_results: MatchResults already in a contiguous group
        singleton_results: MatchResults that were singletons from contiguity check
        frac_start_key: Attribute name for start fraction on the shared segment
        frac_end_key: Attribute name for end fraction on the shared segment
        shared_segment_length_m: Length of the shared segment in meters
        max_overlap_m: Maximum accepted overlap in meters

    Returns:
        Tuple of (merged_group_results, remaining_leftovers)
    """
    if not singleton_results or not contiguous_group_results:
        return contiguous_group_results, singleton_results

    def _get_frac_range(r: MatchResult) -> tuple[float, float] | None:
        start = getattr(r, frac_start_key)
        end = getattr(r, frac_end_key)
        if start is None or end is None:
            return None
        return (min(start, end), max(start, end))

    def _ranges_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Check if two fraction ranges overlap by more than max_overlap_m."""
        overlap_frac = min(a[1], b[1]) - max(a[0], b[0])
        if overlap_frac <= 0:
            return False
        return overlap_frac * shared_segment_length_m > max_overlap_m

    # Collect coverage ranges of existing group members
    group_ranges: list[tuple[float, float]] = []
    for r in contiguous_group_results:
        rng = _get_frac_range(r)
        if rng is not None:
            group_ranges.append(rng)

    merged = list(contiguous_group_results)
    remaining: list[MatchResult] = []

    for singleton in singleton_results:
        s_range = _get_frac_range(singleton)
        if s_range is None:
            # No alignment fractions → can't merge by alignment, keep as leftover
            remaining.append(singleton)
            continue

        # Check if singleton overlaps with ANY existing group member
        overlaps = any(_ranges_overlap(s_range, gr) for gr in group_ranges)

        if not overlaps:
            # Compatible — merge into the group
            merged.append(singleton)
            group_ranges.append(s_range)
        else:
            remaining.append(singleton)

    return merged, remaining


# Anti-crossing / mutual-exclusion guard for M:N groups (#367 "Mode A —
# parallel-sibling crossing swap"): two near-parallel refs are each plausible
# for two nearby targets, and a small segmentation-boundary-mismatch edge
# (covering only a sliver of BOTH its ref and its target) can still score
# confidently (~0.97-0.99) and survive the confidence-drop prune, over-merging
# the group with a spurious extra MATCH edge. See
# ``_contested_small_span_review_pairs``.
#
# Disposition is DEMOTE-TO-REVIEW, not drop: a flagged stub stays in the group
# (retained as a candidate edge, never silently removed) but its decision is
# demoted from MATCH to REVIEW — mirroring ``_validate_assignment_coverage`` —
# so it surfaces in /stitching-review for human adjudication instead of being
# auto-accepted into the production MATCH set. This keeps the guard safe on the
# genuine-but-contested N:1 fans it cannot geometrically distinguish from a true
# over-merge: those keep their real edge (as REVIEW), never losing coverage.
#
# The trigger is deliberately narrower than loosening ``SLIVER_SPAN_THRESHOLD``
# (config.py documents that as rejected: it risks dropping legitimate short
# matches). The span test only fires for edges that are ALSO contested — a rival
# edge sharing the same ref or target scores strictly higher — so an edge with no
# competing claim (the case the existing sliver / confidence machinery already
# governs) is never touched. And using max(ref_span, tgt_span) rather than min
# keeps every genuine asymmetric-coverage match (one side's span near 1.0 — e.g.
# a short ref fully consumed by a long target) exempt, no matter how small its
# OTHER side's span is.
MN_CONTESTED_EDGE_MAX_SPAN = 0.3

# Feature-dict flag set on a MatchResult whose decision this guard demoted to
# REVIEW, so downstream (review UI / audit) can identify the reason.
PARALLEL_SIBLING_REVIEW_FLAG = "parallel_sibling_stub_review"


def _edge_span_fracs(r: MatchResult) -> tuple[float, float]:
    """Alignment span fractions (ref_span, tgt_span) for a MatchResult.

    Mirrors ``matching.sliver.edge_span_fracs`` (which reads the same fractions
    off a serialized edge dict) for the in-memory ``MatchResult`` object.
    Missing fractions default to a full [0, 1] span so an unmeasurable edge is
    never mistaken for a small/contested one.
    """
    ref_span = 1.0
    if r.gers_start_frac is not None and r.gers_end_frac is not None:
        ref_span = abs(r.gers_end_frac - r.gers_start_frac)
    tgt_span = 1.0
    if r.local_start_frac is not None and r.local_end_frac is not None:
        tgt_span = abs(r.local_end_frac - r.local_start_frac)
    return ref_span, tgt_span


def _contested_small_span_review_pairs(
    component_results: list[MatchResult],
) -> set[tuple[Any, Any]]:
    """Identify contested small-span M:N edges to demote to REVIEW (module note above).

    An edge is flagged when BOTH:
      - it is small on both alignment axes: ``max(ref_span, tgt_span) <
        MN_CONTESTED_EDGE_MAX_SPAN``; and
      - it is contested: another edge sharing its ``ref_id`` OR its
        ``target_id`` has strictly higher confidence.

    Orphan-guard: a node (ref or target) is never left with zero MATCH edges. If
    demoting a flagged candidate would orphan its ref or its target (leave it
    with no un-flagged edge), the highest-confidence orphaning candidate is kept
    as MATCH instead (processed highest-confidence first, so at most one edge is
    rescued per orphaned node).

    Returns the set of ``(ref_id, target_id)`` pairs to demote to REVIEW. The
    edges themselves are NOT removed — the caller keeps them in the group and
    flips only their decision.
    """
    best_by_ref: dict[Any, float] = {}
    best_by_target: dict[Any, float] = {}
    for r in component_results:
        if r.confidence > best_by_ref.get(r.ref_id, -1.0):
            best_by_ref[r.ref_id] = r.confidence
        if r.confidence > best_by_target.get(r.target_id, -1.0):
            best_by_target[r.target_id] = r.confidence

    kept: list[MatchResult] = []
    candidate_demote: list[MatchResult] = []
    for r in component_results:
        ref_span, tgt_span = _edge_span_fracs(r)
        if max(ref_span, tgt_span) >= MN_CONTESTED_EDGE_MAX_SPAN:
            kept.append(r)
            continue
        contested = (
            best_by_ref[r.ref_id] > r.confidence or best_by_target[r.target_id] > r.confidence
        )
        if contested:
            candidate_demote.append(r)
        else:
            kept.append(r)

    if not candidate_demote:
        return set()

    kept_refs = {r.ref_id for r in kept}
    kept_targets = {r.target_id for r in kept}
    demote: set[tuple[Any, Any]] = set()
    for r in sorted(candidate_demote, key=lambda r: -r.confidence):
        if r.ref_id not in kept_refs or r.target_id not in kept_targets:
            # Rescue: demoting this would leave its ref or target with no MATCH
            # edge — keep it MATCH instead.
            kept_refs.add(r.ref_id)
            kept_targets.add(r.target_id)
        else:
            demote.add((r.ref_id, r.target_id))

    return demote


def _classify_and_resolve_component(
    component_results: list[MatchResult],
    ref_geoms: dict[Any, LineString],
    target_geoms: dict[Any, LineString],
    tolerance: float,
    corridor_aware: bool = False,
    max_turn_deg: float = 40.0,
    ref_name_lookup: dict[Any, Any] | None = None,
    target_name_lookup: dict[Any, Any] | None = None,
) -> tuple[list[MatchResult], list[MatchResult]]:
    """Classify a connected component and resolve it into groups.

    Classification rules:
    - |R|=1, |T|=1  → 1:1 candidate (pass to greedy optimizer)
    - |R|=1, |T|>1  → check target contiguity → contiguous subgroups = 1:N
    - |R|>1, |T|=1  → check ref contiguity → contiguous subgroups = N:1
    - |R|>1, |T|>1  → check BOTH sides fully contiguous → M:N, else decompose
                       into per-ref 1:N and per-target N:1 sub-problems

    Args:
        component_results: MatchResults in this component
        ref_geoms: Reference geometry lookup
        target_geoms: Target geometry lookup
        tolerance: Contiguity tolerance in meters

    Returns:
        Tuple of (group_results, leftover_1to1_candidates)
    """
    # Deduplicate to a CANONICAL order (sorted by stable string id) rather than
    # list(set(...)), whose order is hash-seed dependent for str ids. This id
    # order seeds contiguity grouping and the sub-component decomposition below,
    # so it must be deterministic to keep group membership/selection reproducible.
    ref_ids = sorted({r.ref_id for r in component_results}, key=str)
    target_ids = sorted({r.target_id for r in component_results}, key=str)

    n_refs = len(ref_ids)
    n_targets = len(target_ids)

    # Case 1: 1:1 — single ref, single target
    if n_refs == 1 and n_targets == 1:
        return [], component_results

    # Case 2: 1:N — single ref, multiple targets
    if n_refs == 1 and n_targets > 1:
        target_groups = _find_contiguous_id_groups(target_ids, target_geoms, tolerance)

        group_results: list[MatchResult] = []
        leftover: list[MatchResult] = []

        # Collect contiguous groups and singletons
        contiguous_group_matches: list[MatchResult] = []
        singleton_matches: list[MatchResult] = []

        for tg in target_groups:
            tg_set = set(tg)
            group_matches = [r for r in component_results if r.target_id in tg_set]

            if len(tg) > 1:
                contiguous_group_matches.extend(group_matches)
            else:
                singleton_matches.extend(group_matches)

        # Try to merge singletons into the contiguous group by alignment fractions
        if contiguous_group_matches and singleton_matches:
            ref_id = ref_ids[0]
            ref_length_m = ref_geoms[ref_id].length if ref_id in ref_geoms else 0.0
            contiguous_group_matches, singleton_matches = _merge_singletons_by_alignment(
                contiguous_group_matches,
                singleton_matches,
                frac_start_key="gers_start_frac",
                frac_end_key="gers_end_frac",
                shared_segment_length_m=ref_length_m,
            )

        if contiguous_group_matches:
            group_results.extend(
                _create_group_results(contiguous_group_matches, MatchType.ONE_TO_N)
            )
        leftover.extend(singleton_matches)

        return group_results, leftover

    # Case 3: N:1 — multiple refs, single target
    if n_refs > 1 and n_targets == 1:
        ref_groups = _find_contiguous_id_groups(ref_ids, ref_geoms, tolerance)

        group_results = []
        leftover = []

        # Collect contiguous groups and singletons
        contiguous_group_matches: list[MatchResult] = []
        singleton_matches: list[MatchResult] = []

        for rg in ref_groups:
            rg_set = set(rg)
            group_matches = [r for r in component_results if r.ref_id in rg_set]

            if len(rg) > 1:
                contiguous_group_matches.extend(group_matches)
            else:
                singleton_matches.extend(group_matches)

        # Try to merge singletons into the contiguous group by alignment fractions
        if contiguous_group_matches and singleton_matches:
            target_id = target_ids[0]
            target_length_m = target_geoms[target_id].length if target_id in target_geoms else 0.0
            contiguous_group_matches, singleton_matches = _merge_singletons_by_alignment(
                contiguous_group_matches,
                singleton_matches,
                frac_start_key="local_start_frac",
                frac_end_key="local_end_frac",
                shared_segment_length_m=target_length_m,
            )

        if contiguous_group_matches:
            group_results.extend(
                _create_group_results(contiguous_group_matches, MatchType.N_TO_ONE)
            )
        leftover.extend(singleton_matches)

        return group_results, leftover

    # Case 4: M:N — multiple refs AND multiple targets
    # Decompose using contiguity groups on both sides, then match sub-components.
    # Corridor-aware: gate contiguity on collinear continuation / same name so a
    # perpendicular junction kiss does not chain two different streets into one
    # over-merged "monster" corridor. Perpendicular corridors then fall into
    # separate contiguity groups and the per-(ref_group x target_group)
    # re-matching below decomposes the monster into per-corridor subgroups.
    ref_groups = _find_contiguous_id_groups(
        ref_ids,
        ref_geoms,
        tolerance,
        require_collinear=corridor_aware,
        max_turn_deg=max_turn_deg,
        name_lookup=ref_name_lookup,
    )
    target_groups = _find_contiguous_id_groups(
        target_ids,
        target_geoms,
        tolerance,
        require_collinear=corridor_aware,
        max_turn_deg=max_turn_deg,
        name_lookup=target_name_lookup,
    )

    refs_fully_contiguous = len(ref_groups) == 1 and len(ref_groups[0]) == n_refs
    targets_fully_contiguous = len(target_groups) == 1 and len(target_groups[0]) == n_targets

    if refs_fully_contiguous and targets_fully_contiguous:
        # Both sides fully contiguous → M:N group. Anti-crossing guard (#367
        # Mode A): demote contested small-span edges to REVIEW so a
        # confident-but-spurious boundary-mismatch edge doesn't over-merge the
        # MATCH set — the edge stays in the group for human adjudication (see
        # ``_contested_small_span_review_pairs``).
        review_pairs = _contested_small_span_review_pairs(component_results)
        return _create_group_results(component_results, MatchType.M_TO_N, review_pairs), []

    # Decompose into sub-components by matching contiguity groups on each side.
    # For each (ref_group, target_group) pair, collect connecting edges.
    # Build edge lookup for quick access
    edge_by_pair: dict[tuple, MatchResult] = {}
    for r in component_results:
        edge_by_pair[(r.ref_id, r.target_id)] = r

    group_results: list[MatchResult] = []
    leftover: list[MatchResult] = []
    used_edges: set[tuple] = set()

    for rg in ref_groups:
        rg_set = set(rg)
        for tg in target_groups:
            tg_set = set(tg)
            # Collect edges connecting this ref_group to this target_group
            sub_edges = [
                edge_by_pair[(rid, tid)] for rid in rg for tid in tg if (rid, tid) in edge_by_pair
            ]
            if not sub_edges:
                continue

            sub_ref_ids = {r.ref_id for r in sub_edges}
            sub_target_ids = {r.target_id for r in sub_edges}

            # Classify the sub-component
            if len(sub_ref_ids) == 1 and len(sub_target_ids) == 1:
                # 1:1 sub-component → leftover for greedy
                leftover.extend(sub_edges)
            elif len(sub_ref_ids) == 1 and len(sub_target_ids) > 1:
                # 1:N sub-component
                group_results.extend(_create_group_results(sub_edges, MatchType.ONE_TO_N))
            elif len(sub_ref_ids) > 1 and len(sub_target_ids) == 1:
                # N:1 sub-component
                group_results.extend(_create_group_results(sub_edges, MatchType.N_TO_ONE))
            else:
                # Smaller M:N sub-component. Same anti-crossing guard as the
                # single-corridor M:N case above (#367 Mode A): demote contested
                # small-span edges to REVIEW, keeping them in the group.
                sub_review_pairs = _contested_small_span_review_pairs(sub_edges)
                group_results.extend(
                    _create_group_results(sub_edges, MatchType.M_TO_N, sub_review_pairs)
                )

            for e in sub_edges:
                used_edges.add((e.ref_id, e.target_id))

    # Any edges not assigned to a sub-component go to leftover
    for r in component_results:
        if (r.ref_id, r.target_id) not in used_edges:
            leftover.append(r)

    return group_results, leftover


def _validate_assignment_coverage(
    results: list[MatchResult],
    ref_geoms: dict[Any, LineString],
    max_overlap_m: float = MAX_ALIGNMENT_OVERLAP_M,
) -> list[MatchResult]:
    """Detect conflicting alignment coverage in assigned matches.

    Checks per reference segment: no two targets should claim overlapping
    portions. When the overlap exceeds max_overlap_m, the lower-confidence
    match is demoted to REVIEW.

    Args:
        results: Optimized match results
        ref_geoms: Reference geometries for computing overlap in meters
        max_overlap_m: Maximum accepted overlap in meters

    Returns:
        Results with conflicting lower-confidence matches demoted to REVIEW
    """
    if not results:
        return results

    # Group results by ref_id
    by_ref: dict[Any, list[int]] = defaultdict(list)
    for i, r in enumerate(results):
        by_ref[r.ref_id].append(i)

    # Track indices that need demotion
    demote_indices: set[int] = set()

    for ref_id, indices in by_ref.items():
        if len(indices) < 2:
            continue

        ref_length_m = ref_geoms[ref_id].length if ref_id in ref_geoms else 0.0
        if ref_length_m <= 0:
            continue

        # Check all pairs for overlap on the ref side
        for a_pos in range(len(indices)):
            a_idx = indices[a_pos]
            a = results[a_idx]
            a_start = a.gers_start_frac
            a_end = a.gers_end_frac
            if a_start is None or a_end is None:
                continue
            a_lo, a_hi = min(a_start, a_end), max(a_start, a_end)

            for b_pos in range(a_pos + 1, len(indices)):
                b_idx = indices[b_pos]
                b = results[b_idx]
                b_start = b.gers_start_frac
                b_end = b.gers_end_frac
                if b_start is None or b_end is None:
                    continue
                b_lo, b_hi = min(b_start, b_end), max(b_start, b_end)

                # Compute overlap in meters
                overlap_frac = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
                if overlap_frac <= 0:
                    continue

                overlap_m = overlap_frac * ref_length_m
                if overlap_m > max_overlap_m:
                    # Conflict — demote the lower-confidence one
                    if a.confidence <= b.confidence:
                        demote_indices.add(a_idx)
                    else:
                        demote_indices.add(b_idx)

    if not demote_indices:
        return results

    demote_count = len(demote_indices)
    logger.info(f"  Coverage validation: demoting {demote_count} conflicting matches to REVIEW")

    validated: list[MatchResult] = []
    for i, r in enumerate(results):
        if i in demote_indices and r.decision != MatchDecision.REVIEW:
            validated.append(
                MatchResult(
                    ref_id=r.ref_id,
                    target_id=r.target_id,
                    decision=MatchDecision.REVIEW,
                    confidence=r.confidence,
                    score_breakdown=r.score_breakdown,
                    features={**r.features, "coverage_conflict": 1.0},
                    ref_idx=r.ref_idx,
                    target_idx=r.target_idx,
                    gers_start_frac=r.gers_start_frac,
                    gers_end_frac=r.gers_end_frac,
                    local_start_frac=r.local_start_frac,
                    local_end_frac=r.local_end_frac,
                )
            )
        else:
            validated.append(r)

    return validated


def optimize_matches_with_grouping(
    results: list[MatchResult],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    min_confidence: float = 0.5,
    contiguity_tolerance: float = DEFAULT_SNAP_TOLERANCE_M,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
    glue_min_confidence: float | None = None,
    corridor_aware: bool | None = None,
    corridor_max_turn_deg: float | None = None,
) -> list[MatchResult]:
    """Optimize matches with M:N group detection.

    Unified bipartite connected-components approach that handles
    1:1, 1:N, N:1, and M:N match groups in one pass.

    Args:
        results: List of MatchResult objects
        reference: Reference GeoDataFrame (for ref geometry lookup)
        target: Target GeoDataFrame (for target geometry lookup)
        min_confidence: Minimum confidence threshold
        contiguity_tolerance: Max distance for contiguity check (meters)
        ref_id_column: Column name for reference IDs
        target_id_column: Column name for target IDs
        glue_min_confidence: Grouping-only confidence prune (see
            :func:`find_match_components`). Defaults to
            ``settings.optimizer_glue_min_confidence``.
        corridor_aware: Gate M:N contiguity on collinear continuation / same
            name. Defaults to ``settings.optimizer_corridor_aware``.
        corridor_max_turn_deg: Collinearity deflection threshold (degrees).
            Defaults to ``settings.optimizer_corridor_max_turn_deg``.

    Returns:
        List of optimized MatchResult objects
    """
    import time

    if glue_min_confidence is None:
        glue_min_confidence = settings.optimizer_glue_min_confidence
    if corridor_aware is None:
        corridor_aware = settings.optimizer_corridor_aware
    if corridor_max_turn_deg is None:
        corridor_max_turn_deg = settings.optimizer_corridor_max_turn_deg
    # The prune only makes sense above the candidate floor.
    glue_min_confidence = max(glue_min_confidence, min_confidence)

    t0 = time.perf_counter()
    logger.info(f"Optimizing {len(results)} results with M:N grouping...")

    # Build geometry lookups
    ref_geoms = _geom_lookup(reference, ref_id_column)
    target_geoms = _geom_lookup(target, target_id_column)

    # Name lookups for the corridor-aware same-name rescue (opt-in). Missing
    # names collapse to "" downstream so they never act as a match.
    ref_name_lookup = _name_lookup(reference, ref_id_column) if corridor_aware else None
    target_name_lookup = _name_lookup(target, target_id_column) if corridor_aware else None

    # Step 0: Classify junction-sliver candidate edges (shared hybrid rule).
    # Slivers never contribute adjacency when building components (a junction
    # kiss must not weld independent groups together) and are never SELECTED
    # into assignments — they only remain visible as ordinary in-group
    # candidates when both endpoints land in the same component anyway.
    metric = not (reference.crs is not None and reference.crs.is_geographic)
    sliver_edges = sliver_edges_for_match_results(results, ref_geoms, target_geoms, metric=metric)
    if sliver_edges:
        logger.info(f"  Classified {len(sliver_edges)} candidate edges as junction slivers")

    # Step 1: Find connected components. Sliver edges and (via the grouping-only
    # prune) weak edges below ``glue_min_confidence`` are excluded from adjacency
    # so they never weld independent groups into a monster.
    components = find_match_components(
        results,
        min_confidence,
        sliver_edges=sliver_edges,
        glue_min_confidence=glue_min_confidence,
    )
    logger.info(f"  Found {len(components)} connected components")

    # Step 2: Classify and resolve each component. Slivers are filtered from
    # the resolution input so they can never be selected into an assignment;
    # they stay in the component itself for group-membership purposes. Weak
    # (pruned) edges are NOT filtered here — they remain scored candidates and
    # can be selected; they simply did not contribute gluing above.
    all_group_results: list[MatchResult] = []
    all_leftover: list[MatchResult] = []

    for component in components:
        structural = [r for r in component if (r.ref_id, r.target_id) not in sliver_edges]
        if not structural:
            continue
        group_results, leftover = _classify_and_resolve_component(
            structural,
            ref_geoms,
            target_geoms,
            contiguity_tolerance,
            corridor_aware=corridor_aware,
            max_turn_deg=corridor_max_turn_deg,
            ref_name_lookup=ref_name_lookup,
            target_name_lookup=target_name_lookup,
        )
        all_group_results.extend(group_results)
        all_leftover.extend(leftover)

    # Step 2b: Recover weak edges that the grouping-only prune left unattached
    # (both endpoints failed to co-land in any component). They are still valid
    # candidates, so feed them to the greedy 1:1 pool — the prune must not
    # reduce match coverage, only stop weak edges from GLUING monsters.
    if glue_min_confidence > min_confidence:
        in_component: set[tuple[Any, Any]] = {
            (r.ref_id, r.target_id) for comp in components for r in comp
        }
        best_unattached: dict[tuple[Any, Any], MatchResult] = {}
        for r in results:
            pair = (r.ref_id, r.target_id)
            if (
                min_confidence <= r.confidence < glue_min_confidence
                and pair not in sliver_edges
                and pair not in in_component
            ):
                if pair not in best_unattached or r.confidence > best_unattached[pair].confidence:
                    best_unattached[pair] = r
        if best_unattached:
            all_leftover.extend(best_unattached.values())

    # Step 3: Track claimed refs/targets from groups
    claimed_refs = {r.ref_id for r in all_group_results}
    claimed_targets = {r.target_id for r in all_group_results}

    # Filter leftover to exclude any already claimed
    unclaimed_leftover = [
        r
        for r in all_leftover
        if r.ref_id not in claimed_refs and r.target_id not in claimed_targets
    ]

    # Step 4: Run greedy 1:1 optimization on unclaimed leftovers
    if unclaimed_leftover:
        optimized_1to1 = optimize_matches_greedy(unclaimed_leftover, min_confidence)
    else:
        optimized_1to1 = []

    # Step 5: Post-expansion — expand 1:1 matches to 1:N/N:1 where
    # contiguous candidates exist. Uses ALL candidates (not just unclaimed
    # leftover) so refs can expand to targets claimed by other refs.
    # This only ADDS matches, never removes existing assignments.
    # Sliver candidates are excluded so expansion can never select one.
    if optimized_1to1:
        expandable = [r for r in results if (r.ref_id, r.target_id) not in sliver_edges]
        optimized_1to1 = _expand_greedy_matches(
            optimized_1to1,
            expandable,  # All non-sliver candidates, not just unclaimed
            ref_geoms,
            target_geoms,
            contiguity_tolerance,
            min_confidence,
        )

    # Combine results and validate alignment coverage
    final = _validate_assignment_coverage(all_group_results + optimized_1to1, ref_geoms)

    # Log summary — count match types across ALL results
    type_counts = defaultdict(int)
    for r in final:
        mt = r.features.get("match_type", MatchType.ONE_TO_ONE)
        type_counts[mt] += 1

    t1 = time.perf_counter()
    logger.info(
        f"  Optimization complete in {t1 - t0:.2f}s: "
        f"{type_counts.get(MatchType.ONE_TO_ONE, 0)} 1:1, "
        f"{type_counts.get(MatchType.ONE_TO_N, 0)} 1:N, "
        f"{type_counts.get(MatchType.N_TO_ONE, 0)} N:1, "
        f"{type_counts.get(MatchType.M_TO_N, 0)} M:N"
    )
    logger.info(f"  Total output: {len(final)} matches")

    return final


def apply_confidence_drop_prune(
    results: list[MatchResult],
    min_confidence: float,
) -> tuple[list[MatchResult], set[tuple[Any, Any]]]:
    """Drop low-confidence SELECTED group edges (M2 / resolver Phase 1).

    Post-optimizer prune: within each M:N / 1:N / N:1 group (a result carrying a
    ``group_id``), drop any selected edge whose confidence is below
    ``min_confidence``. This is the one-parameter confidence filter the #272
    resolver eval validated — an ABSOLUTE threshold, not group-relative
    (``evaluate.py::run_cv`` tuned raw ``confidence >= t`` and that baseline beat
    both keep-all and the learned per-edge model on the clean slice).

    Guarantees:
    - 1:1 matches (no ``group_id``) are never touched — the eval only covered
      M:N group selections.
    - Each group always retains its single highest-confidence edge, so a group is
      never fully emptied (respects the "keep the corridor's backbone edge"
      spirit of the Phase-1 single-corridor exemption).
    - Identity when nothing qualifies: the input list is returned unchanged and
      ``pruned_pairs`` is empty. When ``min_confidence <= 0`` nothing is dropped.

    Returns ``(kept_results, pruned_pairs)`` where ``pruned_pairs`` is the set of
    dropped ``(ref_id, target_id)`` pairs (raw id types).
    """
    if min_confidence <= 0:
        return results, set()

    by_gid: dict[Any, list[int]] = defaultdict(list)
    for i, r in enumerate(results):
        gid = r.features.get("group_id")
        if gid:
            by_gid[gid].append(i)

    pruned_idx: set[int] = set()
    for _gid, idxs in by_gid.items():
        # The single highest-confidence edge is always retained (never empty the
        # group). Ties break on first occurrence — deterministic given input order.
        keep_top = max(idxs, key=lambda i: results[i].confidence)
        for i in idxs:
            if i != keep_top and results[i].confidence < min_confidence:
                pruned_idx.add(i)

    if not pruned_idx:
        return results, set()

    pruned_pairs = {(results[i].ref_id, results[i].target_id) for i in pruned_idx}
    kept = [r for i, r in enumerate(results) if i not in pruned_idx]
    return kept, pruned_pairs


def group_is_structurally_simple(
    n_corridors: int,
    n_assignment_components: int,
    n_edges: int,
    max_assignment_components: int,
    soft_max_edges: int,
    backstop_max_edges: int,
) -> bool:
    """Structural export gate: is this group a clean, single-decision unit?

    Replaces a flat edge-count cap. A group is structurally simple when it is a
    single corridor-pair (one street matched to one street — fine even when long,
    e.g. a 30-edge Beacon-St corridor) OR it has few assignment-components and
    stays within a soft edge budget. A hard backstop ceiling blocks anything
    larger regardless — a defence against a structure-detection bug ever
    auto-exporting a monster, NOT the primary gate.

    Args:
        n_corridors: Number of distinct corridor-pairs in the candidate graph.
        n_assignment_components: Connected components of the selected assignment.
        n_edges: Candidate edge count.
        max_assignment_components: Max assignment-components for a simple group.
        soft_max_edges: Soft edge budget for a multi-component simple group.
        backstop_max_edges: Hard ceiling; nothing above this is ever simple.

    Returns:
        True if the group may auto-export under the structural gate.
    """
    if n_edges > backstop_max_edges:
        return False
    if n_corridors <= 1:
        return True
    return n_assignment_components <= max_assignment_components and n_edges <= soft_max_edges


def compute_group_structure(
    edges: list[tuple[Any, Any]],
    ref_ids: list[Any],
    target_ids: list[Any],
    assignment_pairs: set[tuple[Any, Any]],
    sliver_pairs: set[tuple[Any, Any]],
    ref_geoms: dict[Any, LineString],
    target_geoms: dict[Any, LineString],
    tolerance: float = DEFAULT_SNAP_TOLERANCE_M,
    corridor_aware: bool = True,
    max_turn_deg: float = 40.0,
    ref_name_lookup: dict[Any, Any] | None = None,
    target_name_lookup: dict[Any, Any] | None = None,
) -> tuple[dict[tuple[Any, Any], dict], dict]:
    """Compute candidate-graph structure features for a match group.

    Derives per-edge topology (degree, bridge, biconnected block, corridor ids,
    selected, sliver) and per-group counts (edges, corridors,
    assignment-components, largest biconnected block) from data already in the
    sidecar. These are cheap, purely structural signals persisted for the future
    learned group resolver (see the group-splitting design doc §6) and to drive
    the ``oversized_group`` flag and the structural export gate.

    Args:
        edges: All candidate ``(ref_id, target_id)`` pairs in the group.
        ref_ids / target_ids: The group's segment ids.
        assignment_pairs: Pairs selected by the optimizer assignment.
        sliver_pairs: Pairs classified as junction slivers.
        ref_geoms / target_geoms: Projected (metric) geometry lookups.
        tolerance: Contiguity tolerance (meters) for corridor detection.
        corridor_aware / max_turn_deg / *_name_lookup: Corridor gate config,
            matching the optimizer so corridor ids agree with the grouping.

    Returns:
        ``(per_edge, per_group)`` where ``per_edge`` maps each pair to its
        structure dict and ``per_group`` holds the group-level counts.
    """
    import networkx as nx

    # Corridor ids from same-side contiguity (collinear-gated), so a corridor is
    # a maximal collinear/same-name chain of segments.
    def _corridor_index(ids, geoms, names):
        groups = _find_contiguous_id_groups(
            list(ids),
            geoms,
            tolerance,
            require_collinear=corridor_aware,
            max_turn_deg=max_turn_deg,
            name_lookup=names,
        )
        return {sid: i for i, grp in enumerate(groups) for sid in grp}

    ref_corridor = _corridor_index(ref_ids, ref_geoms, ref_name_lookup)
    tgt_corridor = _corridor_index(target_ids, target_geoms, target_name_lookup)

    # Candidate bipartite graph (slivers excluded — they are junction artifacts,
    # not real adjacency).
    g = nx.Graph()
    g.add_nodes_from(("ref", r) for r in ref_ids)
    g.add_nodes_from(("target", t) for t in target_ids)
    structural_edges = [e for e in edges if e not in sliver_pairs]
    for rid, tid in structural_edges:
        g.add_edge(("ref", rid), ("target", tid))

    degree = dict(g.degree())
    bridge_set = set()
    block_of_edge: dict[frozenset, int] = {}
    largest_block = 0
    if g.number_of_edges() > 0:
        for a, b in nx.bridges(g):
            bridge_set.add(frozenset((a, b)))
        for i, block in enumerate(nx.biconnected_component_edges(g)):
            block = list(block)
            for a, b in block:
                block_of_edge[frozenset((a, b))] = i
            largest_block = max(largest_block, len(block))

    # Assignment-components: connected components of the selected subgraph.
    ga = nx.Graph()
    for rid, tid in assignment_pairs:
        ga.add_edge(("ref", rid), ("target", tid))
    n_assignment_components = nx.number_connected_components(ga) if ga.number_of_edges() else 0

    # Corridor-pairs actually connected by a (non-sliver) edge.
    corridor_pairs = {
        (ref_corridor.get(rid), tgt_corridor.get(tid)) for rid, tid in structural_edges
    }
    n_corridors = len(corridor_pairs) if corridor_pairs else 0

    per_edge: dict[tuple[Any, Any], dict] = {}
    for rid, tid in edges:
        rn, tn = ("ref", rid), ("target", tid)
        key = frozenset((rn, tn))
        is_sliver = (rid, tid) in sliver_pairs
        per_edge[(rid, tid)] = {
            "degree_ref": int(degree.get(rn, 0)),
            "degree_tgt": int(degree.get(tn, 0)),
            "is_bridge": (not is_sliver) and key in bridge_set,
            "biconnected_block": block_of_edge.get(key, -1) if not is_sliver else -1,
            "corridor_ref": ref_corridor.get(rid, -1),
            "corridor_tgt": tgt_corridor.get(tid, -1),
            "selected": (rid, tid) in assignment_pairs,
            "is_sliver": is_sliver,
        }

    per_group = {
        "n_edges": len(edges),
        "n_corridors": n_corridors,
        "n_assignment_components": n_assignment_components,
        "largest_biconnected_block": largest_block,
    }
    return per_edge, per_group
