"""Rule-based baseline matcher.

Uses weighted combination of feature scores with thresholds
to classify candidate pairs as Match, Review, or No Match.

Scoring Philosophy:
------------------
All features are normalized to 0-1 where higher = better match likelihood.
Distance-based features are normalized as: max(0, 1 - distance / threshold)

The weighted sum produces a confidence score (0-1) which is then classified:
- confidence >= match_threshold (0.75) → MATCH (auto-accept)
- confidence >= review_threshold (0.5) → REVIEW (human review)
- confidence < review_threshold → NO_MATCH (auto-reject)

Weight Selection Rationale:
--------------------------
Weights balance geometric and semantic signals:

Geometric (60% total):
- hausdorff_norm (10%): Traditional metric, but sensitive to segmentation
- mean_hausdorff_norm (10%): Robust to segmentation, preferred for partial overlaps
- buffer_iou (15%): Good general-purpose overlap metric
- overlap_ratio (15%): Captures "how much actually overlaps"
- heading_norm (10%): Prevents matching parallel but different roads

Semantic (20% total):
- name_similarity (15%): Strong signal when available, but often missing
- class_similarity (5%): Weak signal, road classes vary between datasets

Other (20% total):
- length_ratio (10%): Helps detect segmentation mismatches
- projection_norm (10%): Average alignment quality

The ML model will learn optimal weights from labeled data; these defaults
provide a reasonable baseline for initial candidate scoring.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import geopandas as gpd
import pandas as pd
from loguru import logger

from ..blocking.spatial_index import CandidatePair
from ..config import settings
from ..features.geometric import compute_geometric_features
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
    # Linear reference fields from alignment (optional)
    # These indicate where on each geometry the match alignment starts/ends
    gers_start_frac: float | None = None
    gers_end_frac: float | None = None
    local_start_frac: float | None = None
    local_end_frac: float | None = None


# Default feature weights - can be overridden via settings.matching_weights
# All scores normalized 0-1, higher = better match
DEFAULT_WEIGHTS = {
    # Geometric features (63% total)
    "hausdorff_norm": 0.08,  # Max deviation - sensitive to segmentation, catches outliers
    "mean_hausdorff_norm": 0.10,  # Mean deviation - robust to partial overlaps
    "buffer_iou": 0.12,  # Overlap quality - robust general-purpose metric
    "overlap_ratio": 0.15,  # Overlap quantity - "how much actually matches?"
    "heading_norm": 0.10,  # Direction alignment - distinguishes parallel roads
    "collinear_gap_ratio": 0.08,  # Penalizes tip-to-tip collinear segments
    # Length/proximity (10% total)
    "length_ratio": 0.10,  # Similar lengths suggest same segment
    # Alignment quality (7% total)
    "projection_norm": 0.07,  # Average perpendicular distance
    # Semantic features (20% total)
    "name_similarity": 0.15,  # Strong signal when present, often missing
    "class_similarity": 0.05,  # Weak signal - classes vary between datasets
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
    ref_name: str | None = None,
    target_name: str | None = None,
    ref_class: str | None = None,
    target_class: str | None = None,
    ref_subclass: str | None = None,
    target_subclass: str | None = None,
    weights: dict[str, float] = None,
    buffer_radius: float = 10.0,
    distance_threshold: float = 50.0,
    precomputed_features: dict[str, float] | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Compute weighted match score for a candidate pair.

    Args:
        ref_geom: Reference geometry (LineString)
        target_geom: Target geometry (LineString)
        ref_name: Reference road name
        target_name: Target road name
        ref_class: Reference road class
        target_class: Target road class
        ref_subclass: Reference subclass (e.g., sidewalk, crosswalk)
        target_subclass: Target subclass
        weights: Feature weights (default if None)
        buffer_radius: Buffer radius for IoU calculation
        distance_threshold: Distance for normalization
        precomputed_features: Pre-computed features dict from compute_pair_features()
            If provided, skips recomputation for ~50% speedup

    Returns:
        Tuple of (confidence, scores dict, raw features dict)
    """
    weights = _get_weights(weights)

    if precomputed_features:
        # Use pre-computed features to avoid duplicate computation
        raw_features = {
            "hausdorff_distance_m": precomputed_features["hausdorff_distance_m"],
            "mean_hausdorff_distance_m": precomputed_features["mean_hausdorff_distance_m"],
            "buffer_iou": precomputed_features["buffer_iou"],
            "overlap_ratio": precomputed_features["overlap_ratio"],
            "heading_delta": precomputed_features["heading_delta"],
            "length_ratio": precomputed_features["length_ratio"],
            "projection_distance_m": precomputed_features["projection_distance_m"],
            "centroid_distance_m": precomputed_features["centroid_distance_m"],
            "collinear_gap_ratio": precomputed_features.get("collinear_gap_ratio", 1.0),
            "name_levenshtein": precomputed_features["name_levenshtein"],
            "name_jaro_winkler": precomputed_features["name_jaro_winkler"],
            "name_token_sort": precomputed_features["name_token_sort"],
            "class_similarity": precomputed_features["class_similarity"],
        }
    else:
        # Compute geometric features
        geom_features = compute_geometric_features(ref_geom, target_geom, buffer_radius)

        # Compute semantic features
        name_sim = compute_name_similarity(ref_name, target_name)
        class_sim = compute_class_similarity(ref_class, target_class, ref_subclass, target_subclass)

        # Raw features for debugging (distance features have _m suffix for clarity)
        raw_features = {
            "hausdorff_distance_m": geom_features.hausdorff_distance,
            "mean_hausdorff_distance_m": geom_features.mean_hausdorff_distance,
            "buffer_iou": geom_features.buffer_iou,
            "overlap_ratio": geom_features.overlap_ratio,
            "heading_delta": geom_features.heading_delta,
            "length_ratio": geom_features.length_ratio,
            "projection_distance_m": geom_features.projection_distance,
            "centroid_distance_m": geom_features.centroid_distance,
            "collinear_gap_ratio": geom_features.collinear_gap_ratio,
            "name_levenshtein": name_sim["levenshtein_ratio"],
            "name_jaro_winkler": name_sim["jaro_winkler"],
            "name_token_sort": name_sim["token_sort_ratio"],
            "class_similarity": class_sim,
        }

    # Normalize geometric features to 0-1 (higher is better)
    scores = {
        "hausdorff_norm": max(0, 1 - raw_features["hausdorff_distance_m"] / distance_threshold),
        "mean_hausdorff_norm": max(
            0, 1 - raw_features["mean_hausdorff_distance_m"] / distance_threshold
        ),
        "buffer_iou": raw_features["buffer_iou"],
        "overlap_ratio": raw_features["overlap_ratio"],  # Already 0-1
        "heading_norm": max(0, 1 - raw_features["heading_delta"] / 45.0),
        "collinear_gap_ratio": raw_features["collinear_gap_ratio"],  # Already 0-1
        "length_ratio": raw_features["length_ratio"],
        "projection_norm": max(0, 1 - raw_features["projection_distance_m"] / distance_threshold),
        "name_similarity": raw_features["name_token_sort"],
        "class_similarity": raw_features["class_similarity"],
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
    ref_subclass_column: str = "subclass",
    target_subclass_column: str = "subclass",
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
        ref_subclass_column: Subclass column in reference
        target_subclass_column: Subclass column in target
        weights: Feature weights

    Returns:
        List of MatchResult objects
    """
    logger.info(f"Scoring {len(candidates)} candidates...")

    weights = _get_weights(weights)

    # Project to meter-based CRS for accurate distance computations
    # All distance features will be in meters after this projection
    working_ref = reference
    working_target = target
    if reference.crs is not None and reference.crs.is_geographic:
        utm_crs = reference.estimate_utm_crs()
        logger.debug(f"Projecting to {utm_crs} for meter-based feature computation")
        working_ref = reference.to_crs(utm_crs)
        working_target = target.to_crs(utm_crs)

    # Pre-extract arrays for faster access (avoid repeated .iloc calls)
    ref_geoms = working_ref.geometry.values
    target_geoms = working_target.geometry.values

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
    ref_subclasses = (
        reference[ref_subclass_column].values
        if ref_subclass_column in reference.columns
        else [None] * len(reference)
    )
    target_subclasses = (
        target[target_subclass_column].values
        if target_subclass_column in target.columns
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
        ref_subclass = ref_subclasses[cand.ref_idx]
        target_subclass = target_subclasses[cand.target_idx]

        # Handle NaN values using pandas for clarity
        if pd.isna(ref_name):
            ref_name = None
        if pd.isna(target_name):
            target_name = None
        if pd.isna(ref_class):
            ref_class = None
        if pd.isna(target_class):
            target_class = None
        if pd.isna(ref_subclass):
            ref_subclass = None
        if pd.isna(target_subclass):
            target_subclass = None

        # Compute score
        confidence, scores, features = compute_match_score(
            ref_geom=ref_geom,
            target_geom=target_geom,
            ref_name=ref_name,
            target_name=target_name,
            ref_class=ref_class,
            target_class=target_class,
            ref_subclass=ref_subclass,
            target_subclass=target_subclass,
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

        if target_id not in best_matches or result.confidence > best_matches[target_id].confidence:
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

        if ref_id not in best_matches or result.confidence > best_matches[ref_id].confidence:
            best_matches[ref_id] = result

    return best_matches
