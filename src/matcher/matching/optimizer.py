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

from ..config import DEFAULT_SNAP_TOLERANCE_M, MAX_ALIGNMENT_OVERLAP_M, settings
from .types import MatchDecision, MatchResult, MatchType


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

    # Sort by confidence (highest first)
    sorted_results = sorted(valid_results, key=lambda r: -r.confidence)

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
) -> list[list[MatchResult]]:
    """Find connected components in the bipartite ref-target match graph.

    Builds an undirected bipartite graph with namespaced nodes
    ("ref", id) / ("target", id) and edges from match pairs.
    Returns one list of MatchResult per connected component.

    Args:
        results: All match results
        min_confidence: Minimum confidence threshold

    Returns:
        List of components, each a list of MatchResult edges
    """
    valid = [r for r in results if r.confidence >= min_confidence]
    if not valid:
        return []

    # Build adjacency list with namespaced nodes
    adj: dict[tuple[str, Any], set[tuple[str, Any]]] = defaultdict(set)
    # Track which edges (MatchResults) connect each node pair
    edge_lookup: dict[tuple[tuple[str, Any], tuple[str, Any]], MatchResult] = {}

    for r in valid:
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

        # Collect edges for this component
        component_edges: list[MatchResult] = []
        ref_nodes = {n for n in component_nodes if n[0] == "ref"}
        tgt_nodes = {n for n in component_nodes if n[0] == "target"}

        for rn in ref_nodes:
            for tn in adj[rn]:
                if tn in tgt_nodes:
                    key = (rn, tn)
                    if key in edge_lookup:
                        component_edges.append(edge_lookup[key])

        if component_edges:
            components.append(component_edges)

    return components


def _find_contiguous_id_groups(
    ids: list[Any],
    geom_lookup: dict[Any, LineString],
    tolerance: float,
) -> list[list[Any]]:
    """Find groups of IDs whose geometries are contiguous.

    Two geometries are contiguous if one's endpoint is within tolerance
    of the other's endpoint. Uses KD-tree for O(n log n) proximity detection.

    Args:
        ids: List of IDs to check
        geom_lookup: Dictionary mapping ID to LineString geometry
        tolerance: Maximum endpoint distance to consider contiguous (meters)

    Returns:
        List of groups, each a list of contiguous IDs.
        Single-element groups are included for IDs with no contiguous neighbors.
    """
    if len(ids) <= 1:
        return [ids] if ids else []

    # Extract endpoints for each ID
    all_endpoints = []
    endpoint_to_idx = []  # which id-index does each endpoint belong to
    valid_id_indices = []

    for i, id_ in enumerate(ids):
        geom = geom_lookup.get(id_)
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        valid_id_indices.append(i)
        all_endpoints.append(coords[0][:2])
        all_endpoints.append(coords[-1][:2])
        endpoint_to_idx.append(i)
        endpoint_to_idx.append(i)

    if len(all_endpoints) < 2:
        return [[id_] for id_ in ids]

    # Build KD-tree for fast proximity queries
    endpoints_array = np.array(all_endpoints)
    tree = cKDTree(endpoints_array)
    pairs = tree.query_pairs(tolerance)

    # Build adjacency from KD-tree results
    adjacent: dict[int, set[int]] = defaultdict(set)
    for ep_i, ep_j in pairs:
        idx_i = endpoint_to_idx[ep_i]
        idx_j = endpoint_to_idx[ep_j]
        if idx_i != idx_j:
            adjacent[idx_i].add(idx_j)
            adjacent[idx_j].add(idx_i)

    # Find connected components using BFS
    visited: set[int] = set()
    groups: list[list[Any]] = []

    for i in range(len(ids)):
        if i in visited:
            continue

        group_indices: list[int] = []
        queue = deque([i])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            group_indices.append(node)
            for neighbor in adjacent[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        groups.append([ids[idx] for idx in group_indices])

    return groups


def _create_group_results(
    results: list[MatchResult],
    match_type: MatchType,
) -> list[MatchResult]:
    """Tag MatchResult objects with group metadata.

    Preserves original MatchResult objects (including alignment fractions,
    confidence, score_breakdown) and adds group metadata to features.

    Sets group decision: MATCH if avg_confidence >= optimizer_review_threshold,
    else REVIEW.

    Args:
        results: MatchResult objects belonging to this group
        match_type: MatchType value (ONE_TO_N, N_TO_ONE, M_TO_N)

    Returns:
        List of tagged MatchResult objects (same objects, mutated features)
    """
    if not results:
        return []

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
        # Create new MatchResult preserving all original fields
        tagged.append(
            MatchResult(
                ref_id=r.ref_id,
                target_id=r.target_id,
                decision=group_decision,
                confidence=r.confidence,
                score_breakdown=r.score_breakdown,
                features={
                    **r.features,
                    "match_type": match_type,
                    "group_id": group_id,
                    "group_size": len(results),
                    "group_ref_count": len(ref_ids),
                    "group_target_count": len(target_ids),
                },
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


def _classify_and_resolve_component(
    component_results: list[MatchResult],
    ref_geoms: dict[Any, LineString],
    target_geoms: dict[Any, LineString],
    tolerance: float,
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
    ref_ids = list(set(r.ref_id for r in component_results))
    target_ids = list(set(r.target_id for r in component_results))

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
    # Decompose using contiguity groups on both sides, then match sub-components
    ref_groups = _find_contiguous_id_groups(ref_ids, ref_geoms, tolerance)
    target_groups = _find_contiguous_id_groups(target_ids, target_geoms, tolerance)

    refs_fully_contiguous = len(ref_groups) == 1 and len(ref_groups[0]) == n_refs
    targets_fully_contiguous = len(target_groups) == 1 and len(target_groups[0]) == n_targets

    if refs_fully_contiguous and targets_fully_contiguous:
        # Both sides fully contiguous → M:N group
        return _create_group_results(component_results, MatchType.M_TO_N), []

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
                # Smaller M:N sub-component
                group_results.extend(_create_group_results(sub_edges, MatchType.M_TO_N))

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

    Returns:
        List of optimized MatchResult objects
    """
    import time

    t0 = time.perf_counter()
    logger.info(f"Optimizing {len(results)} results with M:N grouping...")

    # Build geometry lookups
    if ref_id_column in reference.columns:
        ref_geoms = dict(zip(reference[ref_id_column], reference.geometry))
    else:
        ref_geoms = dict(zip(reference.index, reference.geometry))

    if target_id_column in target.columns:
        target_geoms = dict(zip(target[target_id_column], target.geometry))
    else:
        target_geoms = dict(zip(target.index, target.geometry))

    # Step 1: Find connected components
    components = find_match_components(results, min_confidence)
    logger.info(f"  Found {len(components)} connected components")

    # Step 2: Classify and resolve each component
    all_group_results: list[MatchResult] = []
    all_leftover: list[MatchResult] = []

    for component in components:
        group_results, leftover = _classify_and_resolve_component(
            component, ref_geoms, target_geoms, contiguity_tolerance
        )
        all_group_results.extend(group_results)
        all_leftover.extend(leftover)

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
    if optimized_1to1:
        optimized_1to1 = _expand_greedy_matches(
            optimized_1to1,
            results,  # All candidates, not just unclaimed
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
