"""Global match optimization using bipartite matching.

Resolves conflicts where multiple targets match the same reference
(or vice versa) by finding the globally optimal assignment.

Supports both 1:1 matching (Hungarian algorithm) and 1:N matching
(where one reference can match multiple contiguous target segments).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment
from shapely import LineString, Point

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

    # Build target geometry lookup
    target_geoms = {}
    for idx, row in target.iterrows():
        tid = row.get(target_id_column, idx)
        target_geoms[tid] = row.geometry

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

    Args:
        matches: List of MatchResult for same reference
        target_geoms: Dictionary mapping target_id to geometry
        tolerance: Maximum endpoint distance to consider contiguous

    Returns:
        List of groups, where each group is a list of contiguous MatchResult
    """
    if len(matches) <= 1:
        return [matches] if matches else []

    # Get endpoints for each target (LineStrings only, MultiLineStrings filtered at ingest)
    endpoints = {}
    for m in matches:
        if m.target_id in target_geoms:
            geom = target_geoms[m.target_id]
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
                if len(coords) >= 2:
                    endpoints[m.target_id] = (Point(coords[0]), Point(coords[-1]))

    # Build adjacency based on endpoint proximity
    n = len(matches)
    adjacent = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            m_i, m_j = matches[i], matches[j]

            if m_i.target_id not in endpoints or m_j.target_id not in endpoints:
                continue

            eps_i = endpoints[m_i.target_id]
            eps_j = endpoints[m_j.target_id]

            # Check all endpoint combinations
            is_contiguous = False
            for ep_i in eps_i:
                for ep_j in eps_j:
                    if ep_i.distance(ep_j) <= tolerance:
                        is_contiguous = True
                        break
                if is_contiguous:
                    break

            if is_contiguous:
                adjacent[i].add(j)
                adjacent[j].add(i)

    # Find connected components using BFS
    visited = set()
    groups = []

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
    then runs Hungarian algorithm on remaining conflicts.

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

    # Run Hungarian algorithm on 1:1 matches to resolve remaining conflicts
    if individual_matches:
        optimized = optimize_matches(individual_matches, min_confidence)
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
