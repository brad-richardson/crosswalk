"""Global match optimization using bipartite matching.

Resolves conflicts where multiple targets match the same reference
(or vice versa) by finding the globally optimal assignment.
"""

from typing import Any

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from .rules import MatchDecision, MatchResult


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
