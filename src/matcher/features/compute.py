"""Shared feature computation for ML pipeline and labeling UI.

This module provides a unified interface for computing all features for candidate pairs,
including geometric, semantic, relational, and topology features.
"""

import time
from typing import Any

import geopandas as gpd
from loguru import logger

from .geometric import compute_geometric_features
from .relational import compute_perpendicular_offset
from .semantic import compute_class_similarity, compute_name_similarity
from .spatial_context import (
    SpatialContextIndex,
    compute_all_topology,
    compute_degree_match_score,
    compute_degree_signature_similarity,
    compute_endpoint_features,
)

# Maximum distance value for error cases (avoid infinity)
MAX_DISTANCE_METERS = 10000.0

# Default topology features for empty/missing geometries
DEFAULT_TOPOLOGY_FEATURES = {
    "from_degree": 1,
    "to_degree": 1,
    "is_dead_end": True,
    "is_intersection": False,
    "degree_signature": (1,),
}

# All feature columns computed by this module
ALL_FEATURE_COLUMNS = [
    # Geometric features (9)
    "hausdorff_distance",
    "mean_hausdorff_distance",
    "buffer_iou",
    "overlap_ratio",
    "heading_delta",
    "length_ratio",
    "projection_distance",
    "centroid_distance",
    "collinear_gap_ratio",
    # Semantic features (4)
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "class_similarity",
    # Endpoint/connectivity (3)
    "start_endpoint_proximity",
    "end_endpoint_proximity",
    "shared_endpoint_count",
    # Lateral offset (2)
    "lateral_offset",
    "lateral_offset_consistency",
    # Topology features (12)
    "from_degree_ref",
    "to_degree_ref",
    "from_degree_target",
    "to_degree_target",
    "degree_match_score",
    "degree_signature_similarity",
    "is_dead_end_ref",
    "is_dead_end_target",
    "dead_end_match",
    "is_intersection_ref",
    "is_intersection_target",
    "intersection_match",
]


def compute_pair_features(
    ref_geom,
    target_geom,
    ref_name: str | None,
    target_name: str | None,
    ref_class: str | None,
    target_class: str | None,
    ref_subclass: str | None = None,
    target_subclass: str | None = None,
    endpoint_features: dict[str, float] | None = None,
    ref_topology: dict[str, Any] | None = None,
    target_topology: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute all features for a single candidate pair.

    This is the core feature computation function used by both the ML pipeline
    and the labeling UI.

    Args:
        ref_geom: Reference geometry (LineString)
        target_geom: Target geometry (LineString)
        ref_name: Reference segment name
        target_name: Target segment name
        ref_class: Reference road class
        target_class: Target road class
        ref_subclass: Reference road subclass (optional)
        target_subclass: Target road subclass (optional)
        endpoint_features: Pre-computed endpoint proximity features (optional)
        ref_topology: Pre-computed topology features for reference (optional)
        target_topology: Pre-computed topology features for target (optional)

    Returns:
        Dictionary of feature name -> value
    """
    try:
        # Compute geometric features
        geom_features = compute_geometric_features(ref_geom, target_geom)

        # Compute semantic features
        name_sim = compute_name_similarity(ref_name, target_name)
        class_sim = compute_class_similarity(ref_class, target_class, ref_subclass, target_subclass)

        # Compute lateral offset
        lateral_offset, lateral_consistency = compute_perpendicular_offset(target_geom, ref_geom)

        # Use provided or default endpoint features
        if endpoint_features is None:
            endpoint_features = {
                "start_endpoint_proximity": MAX_DISTANCE_METERS,
                "end_endpoint_proximity": MAX_DISTANCE_METERS,
                "shared_endpoint_count": 0,
            }

        # Use provided or default topology features
        if ref_topology is None:
            ref_topology = DEFAULT_TOPOLOGY_FEATURES.copy()
        if target_topology is None:
            target_topology = DEFAULT_TOPOLOGY_FEATURES.copy()

        # Extract degree values
        from_degree_ref = ref_topology.get("from_degree", 1)
        to_degree_ref = ref_topology.get("to_degree", 1)
        from_degree_target = target_topology.get("from_degree", 1)
        to_degree_target = target_topology.get("to_degree", 1)

        # Compute degree match score
        degree_match = compute_degree_match_score(
            from_degree_ref, to_degree_ref, from_degree_target, to_degree_target
        )

        # Compute degree signature similarity
        ref_sig = ref_topology.get("degree_signature", (1,))
        target_sig = target_topology.get("degree_signature", (1,))
        sig_similarity = compute_degree_signature_similarity(ref_sig, target_sig)

        # Topology flags
        is_dead_end_ref = 1.0 if ref_topology.get("is_dead_end", True) else 0.0
        is_dead_end_target = 1.0 if target_topology.get("is_dead_end", True) else 0.0
        dead_end_match = 1.0 if is_dead_end_ref == is_dead_end_target else 0.0

        is_intersection_ref = 1.0 if ref_topology.get("is_intersection", False) else 0.0
        is_intersection_target = 1.0 if target_topology.get("is_intersection", False) else 0.0
        intersection_match = 1.0 if is_intersection_ref == is_intersection_target else 0.0

        return {
            # Geometric
            "hausdorff_distance": geom_features.hausdorff_distance,
            "mean_hausdorff_distance": geom_features.mean_hausdorff_distance,
            "buffer_iou": geom_features.buffer_iou,
            "overlap_ratio": geom_features.overlap_ratio,
            "heading_delta": geom_features.heading_delta,
            "length_ratio": geom_features.length_ratio,
            "projection_distance": geom_features.projection_distance,
            "centroid_distance": geom_features.centroid_distance,
            "collinear_gap_ratio": geom_features.collinear_gap_ratio,
            # Semantic
            "name_levenshtein": name_sim["levenshtein_ratio"],
            "name_jaro_winkler": name_sim["jaro_winkler"],
            "name_token_sort": name_sim["token_sort_ratio"],
            "class_similarity": class_sim,
            # Endpoint proximity
            "start_endpoint_proximity": endpoint_features.get(
                "start_endpoint_proximity", MAX_DISTANCE_METERS
            ),
            "end_endpoint_proximity": endpoint_features.get(
                "end_endpoint_proximity", MAX_DISTANCE_METERS
            ),
            "shared_endpoint_count": endpoint_features.get("shared_endpoint_count", 0),
            # Lateral offset
            "lateral_offset": min(lateral_offset, MAX_DISTANCE_METERS),
            "lateral_offset_consistency": min(lateral_consistency, MAX_DISTANCE_METERS),
            # Topology - Tier 1: Degree features
            "from_degree_ref": from_degree_ref,
            "to_degree_ref": to_degree_ref,
            "from_degree_target": from_degree_target,
            "to_degree_target": to_degree_target,
            "degree_match_score": degree_match,
            # Topology - Tier 2: Degree signature similarity
            "degree_signature_similarity": sig_similarity,
            # Topology - Tier 3: Topology flags
            "is_dead_end_ref": is_dead_end_ref,
            "is_dead_end_target": is_dead_end_target,
            "dead_end_match": dead_end_match,
            "is_intersection_ref": is_intersection_ref,
            "is_intersection_target": is_intersection_target,
            "intersection_match": intersection_match,
        }

    except Exception as e:
        logger.warning(f"Feature computation failed: {e}")
        # Return error values
        return _get_error_features()


def _get_error_features() -> dict[str, float]:
    """Return default feature values for error cases."""
    return {
        "hausdorff_distance": MAX_DISTANCE_METERS,
        "mean_hausdorff_distance": MAX_DISTANCE_METERS,
        "buffer_iou": 0.0,
        "overlap_ratio": 0.0,
        "heading_delta": 180.0,
        "length_ratio": 0.0,
        "projection_distance": MAX_DISTANCE_METERS,
        "centroid_distance": MAX_DISTANCE_METERS,
        "collinear_gap_ratio": 1.0,  # No penalty in error case (conservative)
        "name_levenshtein": 0.0,
        "name_jaro_winkler": 0.0,
        "name_token_sort": 0.0,
        "class_similarity": 0.0,
        "start_endpoint_proximity": MAX_DISTANCE_METERS,
        "end_endpoint_proximity": MAX_DISTANCE_METERS,
        "shared_endpoint_count": 0,
        "lateral_offset": MAX_DISTANCE_METERS,
        "lateral_offset_consistency": MAX_DISTANCE_METERS,
        "from_degree_ref": 0,
        "to_degree_ref": 0,
        "from_degree_target": 0,
        "to_degree_target": 0,
        "degree_match_score": 0.5,
        "degree_signature_similarity": 0.5,
        "is_dead_end_ref": 0.5,
        "is_dead_end_target": 0.5,
        "dead_end_match": 0.5,
        "is_intersection_ref": 0.5,
        "is_intersection_target": 0.5,
        "intersection_match": 0.5,
    }


def precompute_topology_and_endpoints(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_indices: set[int],
    target_indices: set[int],
    id_column: str = "id",
    tolerance: float = 5.0,
) -> tuple[dict, dict, dict]:
    """Pre-compute topology and endpoint features for efficiency.

    Args:
        reference: Reference GeoDataFrame
        target: Target GeoDataFrame
        ref_indices: Set of reference indices to compute features for
        target_indices: Set of target indices to compute features for
        id_column: Column name for segment IDs
        tolerance: Distance tolerance for topology computation (meters)

    Returns:
        Tuple of (target_endpoint_features, ref_topology_features, target_topology_features)
        Each is a dict mapping index -> feature dict
    """
    # Build spatial index for endpoint proximity features (target only)
    t_start = time.perf_counter()
    t0 = time.perf_counter()
    logger.info("Building spatial index for endpoint features...")
    target_index = SpatialContextIndex()
    target_index.build_from_gdf(target, id_column=id_column)
    logger.debug(f"[precompute] Built spatial index in {time.perf_counter() - t0:.2f}s")

    # Pre-compute endpoint features for target segments
    t0 = time.perf_counter()
    target_endpoint_features = {}
    logger.info(f"Pre-computing endpoint features for {len(target_indices)} target segments...")
    for target_idx in target_indices:
        target_geom = target.geometry.iloc[target_idx]
        if target_geom is not None and not target_geom.is_empty:
            ep_feats = compute_endpoint_features(
                target_geom, target_index, exclude_segment_idx=target_idx
            )
            target_endpoint_features[target_idx] = ep_feats
        else:
            target_endpoint_features[target_idx] = {
                "start_endpoint_proximity": MAX_DISTANCE_METERS,
                "end_endpoint_proximity": MAX_DISTANCE_METERS,
                "shared_endpoint_count": 0,
            }

    logger.debug(f"[precompute] Endpoint features in {time.perf_counter() - t0:.2f}s")

    # Get unique segment IDs
    target_ids = target[id_column].to_numpy()
    ref_ids = reference[id_column].to_numpy()
    unique_target_ids = {str(target_ids[idx]) for idx in target_indices}
    unique_ref_ids = {str(ref_ids[idx]) for idx in ref_indices}

    logger.info(
        f"Computing topology features for {len(unique_target_ids)} target "
        f"and {len(unique_ref_ids)} reference segments..."
    )

    # Compute topology for target and reference
    t0 = time.perf_counter()
    logger.debug("[precompute] Computing target topology...")
    target_topology_by_id = compute_all_topology(
        target, id_column=id_column, tolerance=tolerance, ids_to_compute=unique_target_ids
    )
    logger.debug(f"[precompute] Target topology in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    logger.debug("[precompute] Computing reference topology...")
    ref_topology_by_id = compute_all_topology(
        reference, id_column=id_column, tolerance=tolerance, ids_to_compute=unique_ref_ids
    )
    logger.debug(f"[precompute] Reference topology in {time.perf_counter() - t0:.2f}s")

    # Map topology from segment IDs to DataFrame indices
    target_topology_features = {}
    for target_idx in target_indices:
        seg_id = str(target_ids[target_idx])
        target_topology_features[target_idx] = target_topology_by_id.get(
            seg_id, DEFAULT_TOPOLOGY_FEATURES.copy()
        )

    ref_topology_features = {}
    for ref_idx in ref_indices:
        seg_id = str(ref_ids[ref_idx])
        ref_topology_features[ref_idx] = ref_topology_by_id.get(
            seg_id, DEFAULT_TOPOLOGY_FEATURES.copy()
        )

    logger.info(f"[precompute] Total precompute time: {time.perf_counter() - t_start:.2f}s")
    return target_endpoint_features, ref_topology_features, target_topology_features
