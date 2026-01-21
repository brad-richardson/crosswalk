"""Shared feature computation for ML pipeline and labeling UI.

This module provides a unified interface for computing all features for candidate pairs,
including geometric, semantic, relational, and topology features.
"""

import time
from typing import Any

import geopandas as gpd
from loguru import logger

from ..config import FEATURE_COLUMNS, MAX_DISTANCE_METERS
from .alignment import AlignmentResult, compute_coverage_features, create_subline
from .geometric import compute_geometric_features
from .relational import compute_perpendicular_offset
from .semantic import compute_class_similarity, compute_name_similarity
from .spatial_context import (
    SpatialContextIndex,
    build_connector_graph,
    compute_all_topology,
    compute_degree_match_score,
    compute_degree_signature_similarity,
    compute_endpoint_features,
    graphlet_similarity_with_alignment,
)

# Default topology features for empty/missing geometries
DEFAULT_TOPOLOGY_FEATURES = {
    "from_degree": 1,
    "to_degree": 1,
    "is_dead_end": True,
    "is_intersection": False,
    "degree_signature": (1,),
}

# Alias for backward compatibility - the authoritative list is in config.py
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS


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
    alignment: AlignmentResult | None = None,
    graphlet_features: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute all features for a single candidate pair.

    This is the AUTHORITATIVE source for feature computation. All consumers
    (ml.py scoring, backfill, labeling UI) MUST call this function to ensure
    consistency between training and inference.

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
        alignment: Pre-computed alignment result for using aligned sublines (optional)
        graphlet_features: Pre-computed graphlet similarity features (optional)

    Returns:
        Dictionary of feature name -> value. Keys match FEATURE_COLUMNS from config.py.
    """
    try:
        # Determine geometries for similarity features
        # If alignment is provided, extract sublines for computing similarity features
        # (hausdorff, buffer_iou, etc.) on comparable portions only.
        # Topology/endpoint features still use full geometries.
        if alignment is not None:
            ref_subline = create_subline(
                ref_geom, alignment.overture_start_frac, alignment.overture_end_frac
            )
            target_subline = create_subline(
                target_geom, alignment.dataset_start_frac, alignment.dataset_end_frac
            )
            # Use sublines if valid, otherwise fall back to full geometry
            geom_for_similarity_ref = ref_subline if ref_subline else ref_geom
            geom_for_similarity_target = target_subline if target_subline else target_geom
        else:
            geom_for_similarity_ref = ref_geom
            geom_for_similarity_target = target_geom

        # Compute geometric features on aligned sublines (or full geom if no alignment)
        geom_features = compute_geometric_features(
            geom_for_similarity_ref, geom_for_similarity_target
        )

        # Compute semantic features
        name_sim = compute_name_similarity(ref_name, target_name)
        class_sim = compute_class_similarity(ref_class, target_class, ref_subclass, target_subclass)

        # Compute lateral offset (now returns mean, iqr, p95)
        lateral_offset, lateral_iqr, lateral_p95 = compute_perpendicular_offset(
            target_geom, ref_geom
        )

        # Use provided or default endpoint features
        if endpoint_features is None:
            endpoint_features = {
                "min_endpoint_proximity_m": MAX_DISTANCE_METERS,
                "max_endpoint_proximity_m": MAX_DISTANCE_METERS,
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

        # Compute coverage features from alignment
        coverage_feats = compute_coverage_features(alignment)

        return {
            # Geometric (distance features use _m suffix to indicate meters)
            "hausdorff_distance_m": geom_features.hausdorff_distance,
            "mean_hausdorff_distance_m": geom_features.mean_hausdorff_distance,
            "hausdorff_p95_m": geom_features.hausdorff_p95_distance,
            "buffer_iou_5m": geom_features.buffer_iou_5m,
            "buffer_iou_15m": geom_features.buffer_iou_15m,
            "overlap_ratio": geom_features.overlap_ratio,
            "heading_delta": geom_features.heading_delta,
            "length_ratio": geom_features.length_ratio,
            "projection_distance_m": geom_features.projection_distance,
            "centroid_distance_m": geom_features.centroid_distance,
            "collinear_gap_ratio": geom_features.collinear_gap_ratio,
            # Semantic - name
            "name_levenshtein": name_sim["levenshtein_ratio"],
            "name_jaro_winkler": name_sim["jaro_winkler"],
            "name_token_sort": name_sim["token_sort_ratio"],
            "name_soundex": name_sim["soundex_match"],
            "name_metaphone": name_sim["metaphone_similarity"],
            "has_name_ref": name_sim["has_name_ref"],
            "has_name_target": name_sim["has_name_target"],
            "name_is_generic": name_sim["name_is_generic"],
            # Semantic - class
            "class_similarity": class_sim,
            # Endpoint proximity (direction-invariant min/max)
            "min_endpoint_proximity_m": endpoint_features.get(
                "min_endpoint_proximity_m", MAX_DISTANCE_METERS
            ),
            "max_endpoint_proximity_m": endpoint_features.get(
                "max_endpoint_proximity_m", MAX_DISTANCE_METERS
            ),
            "shared_endpoint_count": endpoint_features.get("shared_endpoint_count", 0),
            # Lateral offset (mean, IQR, P95 - robust to outliers)
            "lateral_offset_m": min(lateral_offset, MAX_DISTANCE_METERS),
            "lateral_offset_iqr_m": min(lateral_iqr, MAX_DISTANCE_METERS),
            "lateral_offset_p95_m": min(lateral_p95, MAX_DISTANCE_METERS),
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
            # Alignment coverage features
            "ref_coverage": coverage_feats["ref_coverage"],
            "target_coverage": coverage_feats["target_coverage"],
            "min_coverage": coverage_feats["min_coverage"],
            "coverage_ratio": coverage_feats["coverage_ratio"],
            # Graphlet features (network topology similarity)
            "graphlet_similarity": (
                graphlet_features.get("graphlet_similarity", 0.5) if graphlet_features else 0.5
            ),
            "endpoint_degree_similarity": (
                graphlet_features.get("endpoint_degree_similarity", 0.5)
                if graphlet_features
                else 0.5
            ),
        }

    except Exception as e:
        logger.warning(f"Feature computation failed: {e}", exc_info=True)
        # Return error values
        return _get_error_features()


def _get_error_features() -> dict[str, float]:
    """Return default feature values for error cases.

    Returns a dict with all features from FEATURE_COLUMNS, using neutral/default
    values that won't artificially inflate or deflate match scores.
    """
    return {
        # Geometric features
        "hausdorff_distance_m": MAX_DISTANCE_METERS,
        "mean_hausdorff_distance_m": MAX_DISTANCE_METERS,
        "hausdorff_p95_m": MAX_DISTANCE_METERS,
        "buffer_iou_5m": 0.0,
        "buffer_iou_15m": 0.0,
        "overlap_ratio": 0.0,
        "heading_delta": 180.0,
        "length_ratio": 0.0,
        "projection_distance_m": MAX_DISTANCE_METERS,
        "centroid_distance_m": MAX_DISTANCE_METERS,
        "collinear_gap_ratio": 1.0,  # No penalty in error case (conservative)
        # Semantic features - name
        "name_levenshtein": 0.0,
        "name_jaro_winkler": 0.0,
        "name_token_sort": 0.0,
        "name_soundex": 0.5,  # Neutral for missing names
        "name_metaphone": 0.5,  # Neutral for missing names
        "has_name_ref": 0.0,
        "has_name_target": 0.0,
        "name_is_generic": 0.0,
        # Semantic features - class
        "class_similarity": 0.0,
        # Endpoint proximity
        "min_endpoint_proximity_m": MAX_DISTANCE_METERS,
        "max_endpoint_proximity_m": MAX_DISTANCE_METERS,
        "shared_endpoint_count": 0,
        # Lateral offset
        "lateral_offset_m": MAX_DISTANCE_METERS,
        "lateral_offset_iqr_m": MAX_DISTANCE_METERS,
        "lateral_offset_p95_m": MAX_DISTANCE_METERS,
        # Topology features
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
        # Coverage features
        "ref_coverage": 0.0,
        "target_coverage": 0.0,
        "min_coverage": 0.0,
        "coverage_ratio": 0.0,
        # Graphlet features - neutral values for error case
        "graphlet_similarity": 0.5,
        "endpoint_degree_similarity": 0.5,
    }


def precompute_topology_and_endpoints(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_indices: set[int],
    target_indices: set[int],
    id_column: str = "id",
    tolerance_m: float = 5.0,
) -> tuple[dict, dict, dict]:
    """Pre-compute topology and endpoint features for efficiency.

    Args:
        reference: Reference GeoDataFrame
        target: Target GeoDataFrame
        ref_indices: Set of reference indices to compute features for
        target_indices: Set of target indices to compute features for
        id_column: Column name for segment IDs
        tolerance_m: Distance tolerance for topology computation (meters)

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
                "min_endpoint_proximity_m": MAX_DISTANCE_METERS,
                "max_endpoint_proximity_m": MAX_DISTANCE_METERS,
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
        target, id_column=id_column, tolerance_m=tolerance_m, ids_to_compute=unique_target_ids
    )
    logger.debug(f"[precompute] Target topology in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    logger.debug("[precompute] Computing reference topology...")
    ref_topology_by_id = compute_all_topology(
        reference, id_column=id_column, tolerance_m=tolerance_m, ids_to_compute=unique_ref_ids
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


def precompute_graphlet_features(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    tolerance_m: float = 5.0,
    connectors_column: str | None = None,
) -> tuple:
    """Pre-compute graphlet data for efficient per-pair lookups.

    Builds an inferred road graph from the GeoDataFrame and computes
    graphlet features for each node. This data can then be used to
    efficiently compute graphlet similarity for candidate pairs.

    If connectors_column is specified and present, uses explicit connector
    positions from Overture data for more accurate alignment-aware comparisons.
    Otherwise, falls back to endpoint-based graph inference.

    Args:
        gdf: GeoDataFrame with road segments
        id_column: Column name for segment IDs
        tolerance_m: Distance tolerance for endpoint snapping (meters)
        connectors_column: Column name for connectors array (Overture format).
                          If provided and column exists, builds connector-based graph.

    Returns:
        Tuple of (G, seg_to_connectors, node_features, use_connectors) where:
        - G: NetworkX graph of the road network
        - seg_to_connectors: Dict mapping segment ID -> list of (at, node_id) tuples
                            OR tuple of (seg_to_start, seg_to_end) for legacy format
        - node_features: Dict mapping node ID -> feature vector
        - use_connectors: Boolean indicating if connector-based graph was built
    """
    t0 = time.perf_counter()

    # Ensure ID column is string type for consistent lookups
    gdf_reset = gdf.reset_index(drop=True) if id_column not in gdf.columns else gdf
    if id_column in gdf_reset.columns:
        gdf_reset = gdf_reset.copy()
        gdf_reset[id_column] = gdf_reset[id_column].astype(str)

    # Check if we should use connector-based graph (Overture data)
    use_connectors = (
        connectors_column is not None
        and connectors_column in gdf_reset.columns
        # Check that at least some segments have connectors
        and gdf_reset[connectors_column].notna().any()
    )

    if use_connectors:
        logger.info("Building connector-based graph for graphlet features (explicit connectors)...")
        G, seg_to_connectors, node_features = build_connector_graph(
            gdf_reset,
            id_column=id_column,
            connectors_column=connectors_column,
            tolerance_m=tolerance_m,
        )
        logger.debug(
            f"[precompute] Built connector graph with {G.number_of_nodes()} nodes "
            f"in {time.perf_counter() - t0:.2f}s"
        )
        return G, seg_to_connectors, node_features, True
    else:
        # Use inferred connector graph that detects mid-segment crossings
        # This provides richer topology than endpoint-only inference
        from .spatial_context import build_inferred_connector_graph

        logger.info("Building inferred connector graph for graphlet features...")
        G, seg_to_connectors, node_features = build_inferred_connector_graph(
            gdf_reset, id_column=id_column, tolerance_m=tolerance_m
        )
        logger.debug(
            f"[precompute] Built inferred connector graph with {G.number_of_nodes()} nodes "
            f"in {time.perf_counter() - t0:.2f}s"
        )
        # Return True for use_connectors since we now have connector format
        return G, seg_to_connectors, node_features, True


def compute_graphlet_similarity(
    ref_seg_id: str,
    target_seg_id: str,
    ref_graphlet_data: tuple | None,
    target_graphlet_data: tuple | None,
    alignment: AlignmentResult | None = None,
) -> dict[str, float]:
    """Compute graphlet similarity features for a segment pair.

    Uses alignment-aware comparison that finds the nearest connectors to the
    aligned subline endpoints. For Overture data, uses explicit connectors.
    For spaghetti geometry, uses inferred connectors from spatial proximity
    (including mid-segment crossings).

    Args:
        ref_seg_id: Reference segment ID
        target_seg_id: Target segment ID
        ref_graphlet_data: Precomputed (G, seg_to_connectors, node_features, True)
        target_graphlet_data: Precomputed (G, seg_to_connectors, node_features, True)
        alignment: Optional alignment result for alignment-aware comparison

    Returns:
        Dict with 'graphlet_similarity' and 'endpoint_degree_similarity' keys
    """
    if ref_graphlet_data is None or target_graphlet_data is None:
        return {"graphlet_similarity": 0.5, "endpoint_degree_similarity": 0.5}

    try:
        # Unpack graphlet data - always connector format now
        _, ref_seg_to_connectors, ref_node_features, _ = ref_graphlet_data
        _, target_seg_to_connectors, target_node_features, _ = target_graphlet_data

        # Get alignment fractions (use full segment if no alignment)
        if alignment is not None:
            ref_start_frac = alignment.overture_start_frac
            ref_end_frac = alignment.overture_end_frac
            target_start_frac = alignment.dataset_start_frac
            target_end_frac = alignment.dataset_end_frac
        else:
            ref_start_frac, ref_end_frac = 0.0, 1.0
            target_start_frac, target_end_frac = 0.0, 1.0

        return graphlet_similarity_with_alignment(
            ref_seg_id,
            target_seg_id,
            ref_node_features,
            target_node_features,
            ref_seg_to_connectors,
            target_seg_to_connectors,
            ref_start_frac,
            ref_end_frac,
            target_start_frac,
            target_end_frac,
        )
    except Exception as exc:
        logger.debug(
            f"Graphlet similarity failed for ref={ref_seg_id}, target={target_seg_id}: {exc}"
        )
        return {"graphlet_similarity": 0.5, "endpoint_degree_similarity": 0.5}
