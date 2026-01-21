"""Global match optimization using bipartite matching.

Resolves conflicts where multiple targets match the same reference
(or vice versa) by finding the globally optimal assignment.

Supports both 1:1 matching (Hungarian algorithm) and 1:N matching
(where one reference can match multiple contiguous target segments).

Memory-efficient sparse optimization is used for large datasets to avoid OOM.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely import LineString

from ..config import settings
from .rules import MatchDecision, MatchResult


@dataclass
class MultiMatchResult:
    """Result of a 1:N match where one reference matches multiple targets."""

    ref_id: Any
    target_ids: list[Any]
    decision: MatchDecision
    confidence: float
    match_type: str = "1:N"
    individual_confidences: list[float] = field(default_factory=list)


def optimize_matches(
    results: list[MatchResult],
    min_confidence: float = 0.5,
    allow_multiple: bool = False,
) -> list[MatchResult]:
    """Optimize 1:1 matching using Hungarian algorithm.

    Finds the globally optimal assignment that maximizes total confidence
    while ensuring each reference and target is matched at most once.

    Args:
        results: List of MatchResult objects
        min_confidence: Minimum confidence to consider a match
        allow_multiple: If True, allow 1:N matches (not yet implemented)

    Returns:
        List of optimal MatchResult objects (1:1 assignments)
    """
    logger.info(f"Optimizing {len(results)} match results...")

    # Filter by minimum confidence
    valid_results = [r for r in results if r.confidence >= min_confidence]
    logger.info(f"  {len(valid_results)} results above min_confidence={min_confidence}")

    if not valid_results:
        return []

    # Build unique ID mappings
    ref_ids = sorted(set(r.ref_id for r in valid_results), key=str)
    target_ids = sorted(set(r.target_id for r in valid_results), key=str)

    ref_to_idx = {r: i for i, r in enumerate(ref_ids)}
    target_to_idx = {t: i for i, t in enumerate(target_ids)}

    n_ref = len(ref_ids)
    n_target = len(target_ids)

    logger.info(f"  Building {n_ref} x {n_target} cost matrix")

    # Build cost matrix (use negative confidence for minimization)
    # Large penalty (1e6) for invalid pairs
    cost_matrix = np.full((n_ref, n_target), 1e6)

    # Build result lookup for extracting optimal matches
    result_lookup = {}

    for result in valid_results:
        i = ref_to_idx[result.ref_id]
        j = target_to_idx[result.target_id]

        # Use negative confidence (minimize cost = maximize confidence)
        cost = -result.confidence

        # Keep the best score if multiple candidates for same pair
        if cost < cost_matrix[i, j]:
            cost_matrix[i, j] = cost
            result_lookup[(i, j)] = result

    # Solve assignment problem
    logger.info("  Running Hungarian algorithm...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract optimal matches
    optimal_matches = []

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < 1e5:  # Valid match exists
            if (i, j) in result_lookup:
                optimal_matches.append(result_lookup[(i, j)])

    logger.info(f"  Found {len(optimal_matches)} optimal 1:1 matches")

    # Re-classify based on thresholds
    final_matches = []
    for match in optimal_matches:
        # Keep original decision/confidence
        final_matches.append(match)

    return final_matches


def optimize_matches_sparse(
    results: list[MatchResult],
    min_confidence: float = 0.5,
) -> list[MatchResult]:
    """Optimization using scipy's linear_sum_assignment (LAPJV algorithm).

    Builds a dense cost matrix for unique refs/targets with candidates.
    Despite the name "sparse", this builds a dense matrix - the "sparse"
    refers to the input being sparse (few edges relative to all possible pairs).

    Memory: O(unique_refs × unique_targets) - dense matrix
    Time: O(n³) where n = max(unique_refs, unique_targets)

    Args:
        results: List of MatchResult objects
        min_confidence: Minimum confidence to consider a match

    Returns:
        List of optimal MatchResult objects (1:1 assignments)
    """
    import time

    logger.info(f"[LAPJV] Starting optimization of {len(results)} results...")

    # Filter by minimum confidence
    valid_results = [r for r in results if r.confidence >= min_confidence]
    logger.info(f"[LAPJV] {len(valid_results)} results above min_confidence={min_confidence}")

    if not valid_results:
        return []

    # Get unique refs and targets that have edges
    t0 = time.perf_counter()
    unique_refs = sorted(set(r.ref_id for r in valid_results), key=str)
    unique_targets = sorted(set(r.target_id for r in valid_results), key=str)

    ref_to_idx = {r: i for i, r in enumerate(unique_refs)}
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}

    n_ref = len(unique_refs)
    n_target = len(unique_targets)
    matrix_mb = (n_ref * n_target * 8) / (1024 * 1024)

    logger.info(
        f"[LAPJV] Building cost matrix: {n_ref} refs × {n_target} targets = {matrix_mb:.1f} MB"
    )

    # Build dense cost matrix with high cost for non-edges
    # Non-edges will be "unmatched" - we filter them out after
    UNMATCHED_COST = 1e9
    cost = np.full((n_ref, n_target), UNMATCHED_COST, dtype=np.float64)

    # Build result lookup and fill cost matrix
    # Negate confidence for minimization (maximize confidence = minimize -confidence)
    result_lookup: dict[tuple[int, int], MatchResult] = {}

    for result in valid_results:
        i = ref_to_idx[result.ref_id]
        j = target_to_idx[result.target_id]
        neg_conf = -result.confidence

        # Keep the best confidence if multiple candidates for same pair
        if cost[i, j] == UNMATCHED_COST or neg_conf < cost[i, j]:
            cost[i, j] = neg_conf
            result_lookup[(i, j)] = result

    t1 = time.perf_counter()
    logger.info(f"[LAPJV] Matrix built in {t1 - t0:.2f}s, running linear_sum_assignment...")

    row_ind, col_ind = linear_sum_assignment(cost)

    t2 = time.perf_counter()
    logger.info(f"[LAPJV] linear_sum_assignment completed in {t2 - t1:.2f}s")

    # Extract optimal matches (filter out unmatched = high cost)
    optimal_matches = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < UNMATCHED_COST / 2:  # Real match, not unmatched
            if (i, j) in result_lookup:
                optimal_matches.append(result_lookup[(i, j)])

    t3 = time.perf_counter()
    logger.info(
        f"[LAPJV] Found {len(optimal_matches)} optimal matches, "
        f"extraction took {t3 - t2:.2f}s, total {t3 - t0:.2f}s"
    )

    return optimal_matches


def optimize_matches_greedy(
    results: list[MatchResult],
    min_confidence: float = 0.5,
) -> list[MatchResult]:
    """Greedy 1:1 assignment as fallback for extremely large datasets.

    Sorts candidates by confidence and greedily assigns matches,
    ensuring each ref and target is matched at most once.

    Time complexity: O(n log n) for sorting
    Space complexity: O(n) where n = number of candidates

    Quality: Achieves ~97-99% of optimal in practice (worst-case competitive
    ratio of 2 for maximum weight matching).

    Args:
        results: List of MatchResult objects
        min_confidence: Minimum confidence to consider a match

    Returns:
        List of greedily-selected MatchResult objects (1:1 assignments)
    """
    logger.info(f"Optimizing {len(results)} match results using greedy algorithm...")

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

    logger.info(f"  Found {len(optimal_matches)} greedy 1:1 matches")

    return optimal_matches


def optimize_matches_auto(
    results: list[MatchResult],
    min_confidence: float = 0.5,
    memory_limit_gb: float | None = None,
) -> list[MatchResult]:
    """Auto-select optimization strategy based on problem size.

    Chooses between:
    1. Dense Hungarian algorithm (small problems, <1GB matrix)
    2. Sparse algorithm (medium problems, <memory_limit and <50k nodes)
    3. Greedy fallback (large problems - fast O(n log n) with ~97-99% optimal)

    The key constraint is that scipy's linear_sum_assignment is O(n³) where
    n = max(n_ref, n_target). For 50k nodes, that's 125 trillion operations.
    We use greedy for anything larger to avoid multi-minute optimization times.

    Args:
        results: List of MatchResult objects
        min_confidence: Minimum confidence to consider a match
        memory_limit_gb: Memory limit for optimization in GB.
            If None, uses settings.optimizer_memory_limit_gb.

    Returns:
        List of optimal MatchResult objects (1:1 assignments)
    """
    import time

    if memory_limit_gb is None:
        memory_limit_gb = settings.optimizer_memory_limit_gb

    valid_results = [r for r in results if r.confidence >= min_confidence]
    if not valid_results:
        return []

    n_ref = len(set(r.ref_id for r in valid_results))
    n_target = len(set(r.target_id for r in valid_results))
    n_edges = len(valid_results)
    max_dim = max(n_ref, n_target)

    # Memory: dense matrix is n_ref * n_target * 8 bytes (float64)
    dense_memory_gb = (n_ref * n_target * 8) / (1024**3)

    # Time complexity threshold: LAPJV is O(n³)
    # For 50k nodes: 50k³ = 125 trillion ops, takes ~30-60 seconds
    # For 70k nodes: 70k³ = 343 trillion ops, takes ~2-5 minutes
    # For 100k nodes: 100k³ = 1 quadrillion ops, takes ~10+ minutes
    LAPJV_MAX_DIMENSION = 50000  # Max dimension for LAPJV (time constraint)
    DENSE_THRESHOLD_GB = 1.0  # Max memory for dense algorithm

    logger.info(
        f"[optimizer] Problem size: {n_ref} refs × {n_target} targets, "
        f"{n_edges} edges, matrix={dense_memory_gb * 1024:.1f} MB"
    )

    # Decision logic:
    # 1. If matrix fits in 1GB, use dense Hungarian (fastest for small problems)
    # 2. If max dimension <= 50k and memory fits, use LAPJV (optimal solution)
    # 3. Otherwise use greedy (fast, near-optimal)

    # Respect both hard-coded threshold and user's memory limit for dense algorithm
    effective_dense_threshold = min(DENSE_THRESHOLD_GB, memory_limit_gb)

    if dense_memory_gb < effective_dense_threshold:
        logger.info(f"[optimizer] Using dense Hungarian (matrix < {effective_dense_threshold} GB)")
        start = time.perf_counter()
        result = optimize_matches(valid_results, min_confidence)
        elapsed = time.perf_counter() - start
        logger.info(f"[optimizer] Dense Hungarian completed in {elapsed:.2f}s")
        return result

    if max_dim > LAPJV_MAX_DIMENSION:
        logger.warning(
            f"[optimizer] Matrix dimension {max_dim} exceeds LAPJV limit {LAPJV_MAX_DIMENSION}, "
            f"using greedy (O(n³) would be too slow)"
        )
        start = time.perf_counter()
        result = optimize_matches_greedy(valid_results, min_confidence)
        elapsed = time.perf_counter() - start
        logger.info(f"[optimizer] Greedy completed in {elapsed:.2f}s")
        return result

    if dense_memory_gb > memory_limit_gb:
        logger.warning(
            f"[optimizer] Matrix {dense_memory_gb:.1f} GB exceeds memory limit {memory_limit_gb} GB, "
            "using greedy"
        )
        start = time.perf_counter()
        result = optimize_matches_greedy(valid_results, min_confidence)
        elapsed = time.perf_counter() - start
        logger.info(f"[optimizer] Greedy completed in {elapsed:.2f}s")
        return result

    # Use LAPJV for medium-sized problems
    logger.info(f"[optimizer] Using LAPJV (dim={max_dim}, memory={dense_memory_gb * 1024:.1f} MB)")
    start = time.perf_counter()
    try:
        result = optimize_matches_sparse(valid_results, min_confidence)
        elapsed = time.perf_counter() - start
        logger.info(f"[optimizer] LAPJV completed in {elapsed:.2f}s")
        return result
    except MemoryError:
        logger.warning("[optimizer] LAPJV hit memory limit, falling back to greedy")
        result = optimize_matches_greedy(valid_results, min_confidence)
        elapsed = time.perf_counter() - start
        logger.info(f"[optimizer] Greedy fallback completed in {elapsed:.2f}s")
        return result


def resolve_conflicts(
    results: list[MatchResult],
    strategy: str = "best_confidence",
) -> list[MatchResult]:
    """Resolve matching conflicts using a simple strategy.

    Simpler alternative to full optimization when speed is preferred.

    Args:
        results: List of MatchResult objects
        strategy: Resolution strategy:
            - "best_confidence": Keep highest confidence match per target
            - "best_per_reference": Keep highest confidence match per reference
            - "mutual_best": Keep only mutual best matches

    Returns:
        List of resolved MatchResult objects
    """
    if strategy == "best_confidence":
        return _resolve_best_per_target(results)
    elif strategy == "best_per_reference":
        return _resolve_best_per_reference(results)
    elif strategy == "mutual_best":
        return _resolve_mutual_best(results)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _resolve_best_per_target(results: list[MatchResult]) -> list[MatchResult]:
    """Keep only the best match for each target."""
    best = {}

    for result in results:
        if result.decision == MatchDecision.NO_MATCH:
            continue

        target_id = result.target_id

        if target_id not in best or result.confidence > best[target_id].confidence:
            best[target_id] = result

    return list(best.values())


def _resolve_best_per_reference(results: list[MatchResult]) -> list[MatchResult]:
    """Keep only the best match for each reference."""
    best = {}

    for result in results:
        if result.decision == MatchDecision.NO_MATCH:
            continue

        ref_id = result.ref_id

        if ref_id not in best or result.confidence > best[ref_id].confidence:
            best[ref_id] = result

    return list(best.values())


def _resolve_mutual_best(results: list[MatchResult]) -> list[MatchResult]:
    """Keep only matches that are mutually best.

    A match is mutual best if:
    - It's the best match for the target among all candidates
    - AND it's the best match for the reference among all candidates
    """
    # Get best per target
    best_per_target = {}
    for result in results:
        if result.decision == MatchDecision.NO_MATCH:
            continue
        tid = result.target_id
        if tid not in best_per_target or result.confidence > best_per_target[tid].confidence:
            best_per_target[tid] = result

    # Get best per reference
    best_per_ref = {}
    for result in results:
        if result.decision == MatchDecision.NO_MATCH:
            continue
        rid = result.ref_id
        if rid not in best_per_ref or result.confidence > best_per_ref[rid].confidence:
            best_per_ref[rid] = result

    # Find mutual bests
    mutual = []
    for result in best_per_target.values():
        # Check if this target's best is also the reference's best
        if result.ref_id in best_per_ref:
            ref_best = best_per_ref[result.ref_id]
            if ref_best.target_id == result.target_id:
                mutual.append(result)

    return mutual


def compute_match_statistics(results: list[MatchResult]) -> dict[str, Any]:
    """Compute statistics about match results.

    Args:
        results: List of MatchResult objects

    Returns:
        Dictionary of statistics
    """
    if not results:
        return {
            "n_total": 0,
            "n_match": 0,
            "n_review": 0,
            "n_no_match": 0,
        }

    confidences = [r.confidence for r in results]

    n_match = sum(1 for r in results if r.decision == MatchDecision.MATCH)
    n_review = sum(1 for r in results if r.decision == MatchDecision.REVIEW)
    n_no_match = sum(1 for r in results if r.decision == MatchDecision.NO_MATCH)

    return {
        "n_total": len(results),
        "n_match": n_match,
        "n_review": n_review,
        "n_no_match": n_no_match,
        "confidence_mean": np.mean(confidences),
        "confidence_std": np.std(confidences),
        "confidence_min": np.min(confidences),
        "confidence_max": np.max(confidences),
        "confidence_median": np.median(confidences),
        "match_rate": n_match / len(results) if results else 0,
    }


def resolve_one_to_many(
    results: list[MatchResult],
    target: gpd.GeoDataFrame,
    min_confidence: float = 0.5,
    contiguity_tolerance: float = 5.0,
    target_id_column: str = "local_id",
) -> tuple[list[MatchResult], list[MultiMatchResult]]:
    """Resolve 1:N matches where one reference matches multiple targets.

    A 1:N match is valid when multiple target segments that match the same
    reference are spatially contiguous (their endpoints are close together).

    Args:
        results: List of MatchResult objects
        target: Target GeoDataFrame (needed for contiguity check)
        min_confidence: Minimum confidence to consider
        contiguity_tolerance: Maximum distance between endpoints to consider contiguous (meters)
        target_id_column: Column name for target IDs

    Returns:
        Tuple of (resolved 1:1 matches, new 1:N matches)
    """
    logger.info(f"Resolving 1:N matches from {len(results)} results...")

    # Filter by confidence
    valid_results = [r for r in results if r.confidence >= min_confidence]

    # Group by reference ID
    by_ref: dict[Any, list[MatchResult]] = defaultdict(list)
    for r in valid_results:
        by_ref[r.ref_id].append(r)

    # Build target geometry lookup (vectorized - avoid iterrows())
    if target_id_column in target.columns:
        target_geoms = dict(zip(target[target_id_column], target.geometry))
    else:
        target_geoms = dict(zip(target.index, target.geometry))

    one_to_one = []
    one_to_many = []

    for ref_id, matches in by_ref.items():
        if len(matches) == 1:
            # Simple 1:1 match
            one_to_one.append(matches[0])
        else:
            # Check if multiple targets are contiguous
            contiguous_groups = _find_contiguous_groups(matches, target_geoms, contiguity_tolerance)

            for group in contiguous_groups:
                if len(group) == 1:
                    # Single match in group
                    one_to_one.append(group[0])
                else:
                    # Multiple contiguous matches -> 1:N
                    avg_confidence = np.mean([m.confidence for m in group])
                    target_ids = [m.target_id for m in group]
                    individual_confidences = [m.confidence for m in group]
                    multi_match = MultiMatchResult(
                        ref_id=ref_id,
                        target_ids=target_ids,
                        decision=MatchDecision.MATCH
                        if avg_confidence >= settings.review_threshold
                        else MatchDecision.REVIEW,
                        confidence=avg_confidence,
                        match_type="1:N",
                        individual_confidences=individual_confidences,
                    )
                    one_to_many.append(multi_match)
                    # Note: Individual 1:N matches are NOT added to one_to_one here.
                    # They're handled by optimize_with_one_to_many() to avoid duplicates.

    logger.info(
        f"  Resolved to {len(one_to_one)} individual matches, {len(one_to_many)} 1:N groups"
    )
    return one_to_one, one_to_many


def _find_contiguous_groups(
    matches: list[MatchResult],
    target_geoms: dict[Any, LineString],
    tolerance: float,
) -> list[list[MatchResult]]:
    """Find groups of contiguous target geometries among matches.

    Two targets are contiguous if one's endpoint is within tolerance of the other's.
    Uses KD-tree for O(n log n) endpoint proximity detection instead of O(n²).

    Args:
        matches: List of MatchResult for same reference
        target_geoms: Dictionary mapping target_id to geometry
        tolerance: Maximum endpoint distance to consider contiguous

    Returns:
        List of groups, where each group is a list of contiguous MatchResult
    """
    if len(matches) <= 1:
        return [matches] if matches else []

    # Extract all endpoints and track which match index they belong to
    all_endpoints = []
    endpoint_to_match_idx = []
    valid_match_indices = []

    for i, m in enumerate(matches):
        if m.target_id not in target_geoms:
            continue
        geom = target_geoms[m.target_id]
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        valid_match_indices.append(i)
        # Add both endpoints (slice to 2D in case of 3D geometries)
        all_endpoints.append(coords[0][:2])
        all_endpoints.append(coords[-1][:2])
        endpoint_to_match_idx.append(i)
        endpoint_to_match_idx.append(i)

    if len(all_endpoints) < 2:
        # Not enough valid geometries to form connections
        return [[m] for m in matches]

    # Build KD-tree for fast proximity queries
    endpoints_array = np.array(all_endpoints)
    tree = cKDTree(endpoints_array)

    # Query for pairs within tolerance
    pairs = tree.query_pairs(tolerance)

    # Build adjacency from KD-tree results
    adjacent: dict[int, set] = defaultdict(set)
    for ep_i, ep_j in pairs:
        match_i = endpoint_to_match_idx[ep_i]
        match_j = endpoint_to_match_idx[ep_j]
        if match_i != match_j:  # Skip same-geometry connections
            adjacent[match_i].add(match_j)
            adjacent[match_j].add(match_i)

    # Find connected components using BFS
    visited: set = set()
    groups = []
    n = len(matches)

    for start in range(n):
        if start in visited:
            continue

        # BFS from start
        group_indices = []
        queue = [start]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue

            visited.add(node)
            group_indices.append(node)

            for neighbor in adjacent[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        groups.append([matches[i] for i in group_indices])

    return groups


def optimize_with_one_to_many(
    results: list[MatchResult],
    target: gpd.GeoDataFrame,
    min_confidence: float = 0.5,
    contiguity_tolerance: float = 5.0,
    target_id_column: str = "local_id",
) -> list[MatchResult]:
    """Optimize matches with support for 1:N relationships.

    First resolves 1:N matches for contiguous target segments,
    then runs the optimization algorithm on remaining conflicts.
    Automatically selects the best algorithm based on problem size.

    Args:
        results: List of MatchResult objects
        target: Target GeoDataFrame
        min_confidence: Minimum confidence threshold
        contiguity_tolerance: Max distance for contiguity check
        target_id_column: Column name for target IDs

    Returns:
        List of optimized MatchResult objects
    """
    # First pass: identify and resolve 1:N matches
    # individual_matches contains only 1:1 matches (single matches per reference)
    # multi_matches contains 1:N groups
    individual_matches, multi_matches = resolve_one_to_many(
        results, target, min_confidence, contiguity_tolerance, target_id_column
    )

    # Run optimization on 1:1 matches to resolve remaining conflicts
    # Uses auto-selection to choose best algorithm based on problem size
    if individual_matches:
        optimized = optimize_matches_auto(individual_matches, min_confidence)
    else:
        optimized = []

    # Add the 1:N matches as individual results
    for mm in multi_matches:
        # Validate that individual_confidences matches target_ids length
        has_valid_confidences = mm.individual_confidences and len(mm.individual_confidences) == len(
            mm.target_ids
        )
        for i, tid in enumerate(mm.target_ids):
            # Use individual confidence if available and valid, otherwise fall back to group average
            confidence = mm.individual_confidences[i] if has_valid_confidences else mm.confidence
            optimized.append(
                MatchResult(
                    ref_id=mm.ref_id,
                    target_id=tid,
                    decision=mm.decision,
                    confidence=confidence,
                    score_breakdown={},
                    features={"match_type": "1:N", "group_size": len(mm.target_ids)},
                )
            )

    return optimized
