"""Rule-based baseline matcher.

Uses weighted combination of feature scores with thresholds
to classify candidate pairs as Match, Review, or No Match.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import geopandas as gpd
from loguru import logger

from ..blocking.spatial_index import CandidatePair
from ..config import settings
from ..features.geometric import GeometricFeatures, compute_geometric_features
from ..features.semantic import compute_class_similarity, compute_name_similarity


class MatchDecision(Enum):
    """Match decision categories."""

    MATCH = "match"
    REVIEW = "review"
    NO_MATCH = "no_match"


@dataclass
class MatchResult:
    """Result of matching a candidate pair."""

    ref_id: Any
    target_id: Any
    decision: MatchDecision
    confidence: float
    score_breakdown: dict[str, float]
    features: dict[str, float]


# Default feature weights - now configured via settings.matching_weights
# Kept for backwards compatibility
DEFAULT_WEIGHTS = {
    "hausdorff_norm": 0.20,  # Lower is better, normalized
    "frechet_norm": 0.10,  # Lower is better, normalized
    "buffer_iou": 0.20,  # Higher is better (0-1)
    "heading_norm": 0.10,  # Lower is better, normalized
    "length_ratio": 0.10,  # Higher is better (0-1)
    "projection_norm": 0.10,  # Lower is better, normalized
    "name_similarity": 0.15,  # Higher is better (0-1)
    "class_similarity": 0.05,  # Higher is better (0-1)
}


def _get_weights(weights: dict[str, float] = None) -> dict[str, float]:
    """Get matching weights from config or provided dict."""
    if weights is not None:
        return weights
    # Use configured weights if available
    if hasattr(settings, "matching_weights"):
        return settings.matching_weights
    return DEFAULT_WEIGHTS


def compute_match_score(
    ref_geom,
    target_geom,
    ref_name: Optional[str] = None,
    target_name: Optional[str] = None,
    ref_class: Optional[str] = None,
    target_class: Optional[str] = None,
    weights: dict[str, float] = None,
    buffer_radius: float = 10.0,
    distance_threshold: float = 50.0,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Compute weighted match score for a candidate pair.

    Args:
        ref_geom: Reference geometry (LineString)
        target_geom: Target geometry (LineString)
        ref_name: Reference road name
        target_name: Target road name
        ref_class: Reference road class
        target_class: Target road class
        weights: Feature weights (default if None)
        buffer_radius: Buffer radius for IoU calculation
        distance_threshold: Distance for normalization

    Returns:
        Tuple of (confidence, scores dict, raw features dict)
    """
    weights = _get_weights(weights)

    # Compute geometric features
    geom_features = compute_geometric_features(ref_geom, target_geom, buffer_radius)

    # Compute semantic features
    name_sim = compute_name_similarity(ref_name, target_name)
    class_sim = compute_class_similarity(ref_class, target_class)

    # Normalize geometric features to 0-1 (higher is better)
    scores = {
        "hausdorff_norm": max(0, 1 - geom_features.hausdorff_distance / distance_threshold),
        "frechet_norm": max(0, 1 - geom_features.frechet_distance / distance_threshold),
        "buffer_iou": geom_features.buffer_iou,
        "heading_norm": max(0, 1 - geom_features.heading_delta / 45.0),
        "length_ratio": geom_features.length_ratio,
        "projection_norm": max(0, 1 - geom_features.projection_distance / distance_threshold),
        "name_similarity": name_sim["token_sort_ratio"],
        "class_similarity": class_sim,
    }

    # Raw features for debugging
    raw_features = {
        "hausdorff_distance": geom_features.hausdorff_distance,
        "frechet_distance": geom_features.frechet_distance,
        "buffer_iou": geom_features.buffer_iou,
        "heading_delta": geom_features.heading_delta,
        "length_ratio": geom_features.length_ratio,
        "projection_distance": geom_features.projection_distance,
        "centroid_distance": geom_features.centroid_distance,
        "name_levenshtein": name_sim["levenshtein_ratio"],
        "name_jaro_winkler": name_sim["jaro_winkler"],
        "name_token_sort": name_sim["token_sort_ratio"],
        "class_similarity": class_sim,
    }

    # Weighted sum
    total_weight = sum(weights.values())
    confidence = sum(scores[k] * weights.get(k, 0) for k in scores) / total_weight

    return confidence, scores, raw_features


def classify_match(
    confidence: float,
    match_threshold: float = None,
    review_threshold: float = None,
) -> MatchDecision:
    """Classify match based on confidence score.

    Args:
        confidence: Match confidence (0-1)
        match_threshold: Threshold for automatic match
        review_threshold: Threshold for review (below = no match)

    Returns:
        MatchDecision
    """
    match_threshold = match_threshold or settings.match_threshold
    review_threshold = review_threshold or settings.review_threshold

    if confidence >= match_threshold:
        return MatchDecision.MATCH
    elif confidence >= review_threshold:
        return MatchDecision.REVIEW
    else:
        return MatchDecision.NO_MATCH


def score_candidates(
    candidates: list[CandidatePair],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_name_column: str = "name",
    target_name_column: str = "name",
    ref_class_column: str = "class",
    target_class_column: str = "road_class",
    weights: dict[str, float] = None,
) -> list[MatchResult]:
    """Score all candidate pairs.

    Args:
        candidates: List of candidate pairs
        reference: Reference GeoDataFrame
        target: Target GeoDataFrame
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        ref_class_column: Class column in reference
        target_class_column: Class column in target
        weights: Feature weights

    Returns:
        List of MatchResult objects
    """
    logger.info(f"Scoring {len(candidates)} candidates...")

    weights = _get_weights(weights)

    # Pre-extract arrays for faster access (avoid repeated .iloc calls)
    ref_geoms = reference.geometry.values
    target_geoms = target.geometry.values

    ref_names = (
        reference[ref_name_column].values
        if ref_name_column in reference.columns
        else [None] * len(reference)
    )
    target_names = (
        target[target_name_column].values
        if target_name_column in target.columns
        else [None] * len(target)
    )
    ref_classes = (
        reference[ref_class_column].values
        if ref_class_column in reference.columns
        else [None] * len(reference)
    )
    target_classes = (
        target[target_class_column].values
        if target_class_column in target.columns
        else [None] * len(target)
    )

    results = []

    for i, cand in enumerate(candidates):
        if i > 0 and i % 1000 == 0:
            logger.info(f"  Scored {i}/{len(candidates)} candidates")

        ref_geom = ref_geoms[cand.ref_idx]
        target_geom = target_geoms[cand.target_idx]

        ref_name = ref_names[cand.ref_idx]
        target_name = target_names[cand.target_idx]
        ref_class = ref_classes[cand.ref_idx]
        target_class = target_classes[cand.target_idx]

        # Handle NaN values
        if ref_name is not None and (isinstance(ref_name, float) and ref_name != ref_name):
            ref_name = None
        if target_name is not None and (isinstance(target_name, float) and target_name != target_name):
            target_name = None

        # Compute score
        confidence, scores, features = compute_match_score(
            ref_geom=ref_geom,
            target_geom=target_geom,
            ref_name=ref_name,
            target_name=target_name,
            ref_class=ref_class,
            target_class=target_class,
            weights=weights,
        )

        # Classify
        decision = classify_match(confidence)

        results.append(
            MatchResult(
                ref_id=cand.ref_id,
                target_id=cand.target_id,
                decision=decision,
                confidence=confidence,
                score_breakdown=scores,
                features=features,
            )
        )

    # Summary
    n_match = sum(1 for r in results if r.decision == MatchDecision.MATCH)
    n_review = sum(1 for r in results if r.decision == MatchDecision.REVIEW)
    n_no_match = sum(1 for r in results if r.decision == MatchDecision.NO_MATCH)

    logger.info(f"Scoring complete: {n_match} match, {n_review} review, {n_no_match} no match")

    return results


def get_best_match_per_target(
    results: list[MatchResult],
) -> dict[Any, MatchResult]:
    """Get the best match for each target ID.

    Args:
        results: List of all match results

    Returns:
        Dictionary mapping target_id to best MatchResult
    """
    best_matches = {}

    for result in results:
        target_id = result.target_id

        if target_id not in best_matches:
            best_matches[target_id] = result
        elif result.confidence > best_matches[target_id].confidence:
            best_matches[target_id] = result

    return best_matches


def get_best_match_per_reference(
    results: list[MatchResult],
) -> dict[Any, MatchResult]:
    """Get the best match for each reference ID.

    Args:
        results: List of all match results

    Returns:
        Dictionary mapping ref_id to best MatchResult
    """
    best_matches = {}

    for result in results:
        ref_id = result.ref_id

        if ref_id not in best_matches:
            best_matches[ref_id] = result
        elif result.confidence > best_matches[ref_id].confidence:
            best_matches[ref_id] = result

    return best_matches
