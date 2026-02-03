"""Shared feature computation for ML pipeline and labeling UI.

This module provides a unified interface for computing all features for candidate pairs,
including geometric, semantic, relational, and topology features.
"""

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np
from loguru import logger

# ============================================================================
# Performance Profiling Infrastructure
# ============================================================================
# Enable with MATCHER_PROFILE=1 environment variable
# Each worker accumulates timing stats and logs summaries periodically


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled via MATCHER_PROFILE env var."""
    return os.environ.get("MATCHER_PROFILE", "0") == "1"


@dataclass
class FeatureTimingStats:
    """Accumulated timing stats for feature computation."""

    counts: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)

    def record(self, name: str, elapsed: float) -> None:
        """Record a timing measurement."""
        self.counts[name] = self.counts.get(name, 0) + 1
        self.totals[name] = self.totals.get(name, 0.0) + elapsed

    def summary(self) -> str:
        """Generate a summary of timing stats."""
        lines = ["Feature Timing Breakdown:"]
        total = sum(self.totals.values())
        for name, elapsed in sorted(self.totals.items(), key=lambda x: -x[1]):
            count = self.counts[name]
            pct = elapsed / total * 100 if total > 0 else 0
            avg_us = elapsed / count * 1e6 if count > 0 else 0
            lines.append(
                f"  {name}: {elapsed:.2f}s ({pct:.1f}%) - {count} calls, {avg_us:.1f} us/call"
            )
        lines.append(f"  TOTAL: {total:.2f}s")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all stats."""
        self.counts.clear()
        self.totals.clear()


# Thread-local storage for timing stats (each worker process has its own)
_timing_stats = threading.local()
_call_counter = threading.local()


def get_timing_stats() -> FeatureTimingStats:
    """Get the thread-local timing stats."""
    if not hasattr(_timing_stats, "stats"):
        _timing_stats.stats = FeatureTimingStats()
    return _timing_stats.stats


def get_call_count() -> int:
    """Get the thread-local call counter."""
    if not hasattr(_call_counter, "count"):
        _call_counter.count = 0
    return _call_counter.count


def increment_call_count() -> int:
    """Increment and return the call counter."""
    if not hasattr(_call_counter, "count"):
        _call_counter.count = 0
    _call_counter.count += 1
    return _call_counter.count


@contextmanager
def timed_section(name: str):
    """Context manager to time a section and accumulate stats.

    Only active when MATCHER_PROFILE=1 environment variable is set.
    """
    if not is_profiling_enabled():
        yield
        return

    t0 = time.perf_counter()
    yield
    get_timing_stats().record(name, time.perf_counter() - t0)


def log_timing_summary_if_needed(interval: int = 5000) -> None:
    """Log timing summary every N calls (worker-side).

    Call this after each compute_pair_features() invocation.
    """
    if not is_profiling_enabled():
        return

    count = increment_call_count()
    if count % interval == 0:
        stats = get_timing_stats()
        logger.info(f"Worker timing after {count} pairs:\n{stats.summary()}")


from ..config import (
    DEFAULT_TOPOLOGY_FEATURES,
    FEATURE_COLUMNS,
    MAX_DISTANCE_METERS,
)
from .alignment import AlignmentResult, compute_coverage_features, create_subline
from .geometric import (
    GeometricFeatures,
    _compute_hausdorff_stats,
    compute_angle_histogram_similarity,
    compute_collinear_gap_ratio,
    compute_edge_distance_rmse,
    compute_geometric_features,
    compute_heading_consistency,
    compute_shape_complexity,
    compute_sinuosity,
    compute_vertex_density,
)
from .relational import (
    SiblingSearchContext,
    compute_perpendicular_offset,
    find_parallel_sibling,
    get_expected_half_width,
)
from .semantic import (
    compute_class_similarity,
    compute_name_numeric_match,
    compute_name_similarity,
    compute_route_prefix_match,
)
from .spatial_context import (
    build_connector_graph,
    compute_clustering_coefficient_features,
    compute_degree_match_score,
    compute_degree_signature_similarity,
    graphlet_similarity_with_alignment,
)

# Alias for backward compatibility - the authoritative list is in config.py
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS


def _compute_non_geometric_features(
    geom_sim_ref,
    geom_sim_target,
    coords_ref: "np.ndarray",
    coords_target: "np.ndarray",
    ref_name: str | None,
    target_name: str | None,
    ref_class: str | None,
    target_class: str | None,
    ref_subclass: str | None,
    target_subclass: str | None,
    endpoint_features: dict[str, float],
    ref_topology: dict | None,
    target_topology: dict | None,
    alignment: AlignmentResult | None,
    graphlet_features: dict[str, float] | None,
    ref_graphlet_data: tuple | None,
    target_graphlet_data: tuple | None,
    ref_seg_id: str | None,
    target_seg_id: str | None,
    geom_features: GeometricFeatures,
    precomputed_lateral_offset: tuple[float, float, float] | None = None,
    ref_sibling_context: SiblingSearchContext | None = None,
    target_sibling_context: SiblingSearchContext | None = None,
) -> dict[str, float]:
    """Compute all non-batchable features for a single candidate pair.

    This extracts the per-pair features that cannot be vectorized:
    - Hausdorff stats (mean/p95) - different vertex counts per pair
    - Collinear gap ratio - Numba JIT
    - Semantic features (name/class similarity)
    - Lateral offset
    - Sinuosity, heading consistency, vertex density, shape complexity
    - Endpoint proximity
    - Topology features
    - Graphlet features
    - Coverage features
    - Min length

    Args:
        geom_sim_ref: Reference geometry for similarity (may be subline)
        geom_sim_target: Target geometry for similarity (may be subline)
        coords_ref: Pre-extracted coordinates for geom_sim_ref
        coords_target: Pre-extracted coordinates for geom_sim_target
        ref_name: Reference segment name
        target_name: Target segment name
        ref_class: Reference road class
        target_class: Target road class
        ref_subclass: Reference road subclass
        target_subclass: Target road subclass
        endpoint_features: Pre-computed endpoint proximity features
        ref_topology: Topology features for reference
        target_topology: Topology features for target
        alignment: Alignment result
        graphlet_features: Pre-computed graphlet similarity features
        ref_graphlet_data: Graphlet data for reference
        target_graphlet_data: Graphlet data for target
        ref_seg_id: Reference segment ID
        target_seg_id: Target segment ID
        geom_features: Pre-computed geometric features (batchable fields filled in)
        precomputed_lateral_offset: Optional pre-computed (mean, iqr, p95) from batch.

    Returns:
        Dictionary of non-geometric feature name -> value, plus per-pair geometric
        fields (mean_hausdorff, p95, collinear_gap) merged in.
    """
    # Per-pair geometric features that can't be batched
    with timed_section("geom_hausdorff_stats"):
        mean_hausdorff, p95_hausdorff = _compute_hausdorff_stats(
            geom_sim_ref, geom_sim_target, coords_a=coords_ref, coords_b=coords_target
        )

    with timed_section("geom_collinear_gap"):
        collinear_gap_ratio = compute_collinear_gap_ratio(
            geom_sim_ref, geom_sim_target, coords_a=coords_ref, coords_b=coords_target
        )

    # Semantic features
    with timed_section("name_similarity"):
        name_sim = compute_name_similarity(ref_name, target_name)

    with timed_section("class_similarity"):
        class_sim = compute_class_similarity(ref_class, target_class, ref_subclass, target_subclass)

    # Lateral offset
    with timed_section("perpendicular_offset"):
        if precomputed_lateral_offset is not None:
            lateral_offset, lateral_iqr, lateral_p95 = precomputed_lateral_offset
        else:
            lateral_offset, lateral_iqr, lateral_p95 = compute_perpendicular_offset(
                geom_sim_target, geom_sim_ref
            )

    # Sinuosity
    with timed_section("sinuosity"):
        sinuosity_ref = compute_sinuosity(geom_sim_ref, coords=coords_ref)
        sinuosity_target = compute_sinuosity(geom_sim_target, coords=coords_target)
        sinuosity_delta = abs(sinuosity_ref - sinuosity_target)

    # Heading consistency
    with timed_section("heading_consistency"):
        heading_consistency_ref = compute_heading_consistency(geom_sim_ref)
        heading_consistency_target = compute_heading_consistency(geom_sim_target)
        heading_consistency_delta = abs(heading_consistency_ref - heading_consistency_target)

    # Vertex density
    with timed_section("vertex_density"):
        vertex_density_ref = compute_vertex_density(geom_sim_ref, coords=coords_ref)
        vertex_density_target = compute_vertex_density(geom_sim_target, coords=coords_target)
        if vertex_density_ref > 0 and vertex_density_target > 0:
            vertex_density_ratio = min(vertex_density_ref, vertex_density_target) / max(
                vertex_density_ref, vertex_density_target
            )
        else:
            vertex_density_ratio = 0.0

    # Min length
    with timed_section("length_computation"):
        ref_length = geom_sim_ref.length
        target_length = geom_sim_target.length
        min_length_m = min(ref_length, target_length)

    # Shape complexity
    with timed_section("shape_complexity"):
        shape_complexity_ref = compute_shape_complexity(geom_sim_ref, coords=coords_ref)
        shape_complexity_target = compute_shape_complexity(geom_sim_target, coords=coords_target)
        shape_complexity_delta = abs(shape_complexity_ref - shape_complexity_target)

    # Angle histogram similarity (shape fingerprint)
    with timed_section("angle_histogram"):
        angle_histogram_similarity = compute_angle_histogram_similarity(
            geom_sim_ref, geom_sim_target, coords_a=coords_ref, coords_b=coords_target
        )

    # Edge distance RMSE (Hootenanny's primary metric)
    with timed_section("edge_distance_rmse"):
        edge_distance_rmse_m = compute_edge_distance_rmse(geom_sim_ref, geom_sim_target)

    # Name numeric match
    with timed_section("name_numeric_match"):
        name_numeric_match = compute_name_numeric_match(ref_name, target_name)

    # Route prefix match
    with timed_section("route_prefix_match"):
        route_prefix_match = compute_route_prefix_match(ref_name, target_name)

    # Endpoint features
    with timed_section("endpoint_features_lookup"):
        if endpoint_features is None:
            raise ValueError(
                "endpoint_features is required - must be computed on aligned subline endpoints"
            )

    # Topology features
    with timed_section("aligned_topology"):
        use_aligned_topology = (
            alignment is not None
            and ref_graphlet_data is not None
            and target_graphlet_data is not None
            and ref_seg_id is not None
            and target_seg_id is not None
        )

        if use_aligned_topology:
            from .spatial_context import compute_aligned_topology_features

            _, ref_seg_to_connectors, ref_node_features, _ = ref_graphlet_data
            _, target_seg_to_connectors, target_node_features, _ = target_graphlet_data

            ref_aligned_topo = compute_aligned_topology_features(
                ref_seg_id,
                ref_seg_to_connectors,
                ref_node_features,
                alignment.overture_start_frac,
                alignment.overture_end_frac,
            )
            target_aligned_topo = compute_aligned_topology_features(
                target_seg_id,
                target_seg_to_connectors,
                target_node_features,
                alignment.dataset_start_frac,
                alignment.dataset_end_frac,
            )

            from_degree_ref = ref_aligned_topo["from_degree"]
            to_degree_ref = ref_aligned_topo["to_degree"]
            from_degree_target = target_aligned_topo["from_degree"]
            to_degree_target = target_aligned_topo["to_degree"]
            ref_topology = ref_aligned_topo
            target_topology = target_aligned_topo
        else:
            if ref_topology is None:
                ref_topology = DEFAULT_TOPOLOGY_FEATURES.copy()
            if target_topology is None:
                target_topology = DEFAULT_TOPOLOGY_FEATURES.copy()

            from_degree_ref = ref_topology.get("from_degree", 1)
            to_degree_ref = ref_topology.get("to_degree", 1)
            from_degree_target = target_topology.get("from_degree", 1)
            to_degree_target = target_topology.get("to_degree", 1)

    # Degree match score
    with timed_section("degree_match"):
        degree_match = compute_degree_match_score(
            from_degree_ref, to_degree_ref, from_degree_target, to_degree_target
        )

    # Degree signature similarity
    with timed_section("degree_signature"):
        ref_sig = ref_topology.get("degree_signature", (1,))
        target_sig = target_topology.get("degree_signature", (1,))
        sig_similarity = compute_degree_signature_similarity(ref_sig, target_sig)

    # Topology flags
    with timed_section("topology_flags"):
        is_dead_end_ref = 1.0 if ref_topology.get("is_dead_end", True) else 0.0
        is_dead_end_target = 1.0 if target_topology.get("is_dead_end", True) else 0.0
        dead_end_match = 1.0 if is_dead_end_ref == is_dead_end_target else 0.0

        is_intersection_ref = 1.0 if ref_topology.get("is_intersection", False) else 0.0
        is_intersection_target = 1.0 if target_topology.get("is_intersection", False) else 0.0
        intersection_match = 1.0 if is_intersection_ref == is_intersection_target else 0.0

    # Coverage features
    with timed_section("coverage_features"):
        coverage_feats = compute_coverage_features(alignment)

    # Parallel sibling features (detect split vs centerline representation)
    # Computed per-pair on the aligned sublines for accuracy with partial alignments
    with timed_section("sibling_features"):
        # Compute sibling detection on sublines (not precomputed full geometries)
        if ref_sibling_context is not None and ref_seg_id is not None:
            has_sibling_ref, sibling_dist_ref = find_parallel_sibling(
                segment=geom_sim_ref,  # Use subline, not full geometry
                segment_id=ref_seg_id,
                segment_name=ref_name,
                segment_class=ref_class,
                spatial_index=ref_sibling_context.spatial_index,
                segment_data=ref_sibling_context.segment_data,
            )
        else:
            if ref_sibling_context is None:
                logger.warning("ref_sibling_context is None - sibling detection disabled")
            has_sibling_ref, sibling_dist_ref = False, MAX_DISTANCE_METERS

        if target_sibling_context is not None and target_seg_id is not None:
            has_sibling_target, sibling_dist_target = find_parallel_sibling(
                segment=geom_sim_target,  # Use subline, not full geometry
                segment_id=target_seg_id,
                segment_name=target_name,
                segment_class=target_class,
                spatial_index=target_sibling_context.spatial_index,
                segment_data=target_sibling_context.segment_data,
            )
        else:
            if target_sibling_context is None:
                logger.warning("target_sibling_context is None - sibling detection disabled")
            has_sibling_target, sibling_dist_target = False, MAX_DISTANCE_METERS

        # Core sibling detection
        has_parallel_sibling_ref = float(has_sibling_ref)

        # Derived: likely representation mismatch (XOR: one has sibling, other doesn't)
        likely_representation_mismatch = float(has_sibling_ref != has_sibling_target)

        # Corridor-aware offset ratio
        # Use sibling distance from whichever side is split (has sibling)
        if has_sibling_ref and not has_sibling_target:
            corridor_width = sibling_dist_ref
        elif has_sibling_target and not has_sibling_ref:
            corridor_width = sibling_dist_target
        elif has_sibling_ref and has_sibling_target:
            corridor_width = min(sibling_dist_ref, sibling_dist_target)
        else:
            corridor_width = MAX_DISTANCE_METERS  # No corridor detected

        half_corridor = corridor_width / 2.0
        offset_vs_half = abs(lateral_offset - half_corridor)
        offset_vs_half_corridor_ratio = offset_vs_half / (corridor_width + 1e-6)

        # Class-based normalization
        half_width_ref = get_expected_half_width(ref_class)
        half_width_target = get_expected_half_width(target_class)
        expected_half_width = (half_width_ref + half_width_target) / 2.0
        offset_over_expected_halfwidth = lateral_offset / (expected_half_width + 1e-6)

    # Clustering coefficient features
    with timed_section("clustering_coef"):
        if (
            ref_graphlet_data is not None
            and target_graphlet_data is not None
            and ref_seg_id is not None
            and target_seg_id is not None
            and alignment is not None
        ):
            _, ref_seg_to_connectors, ref_node_features, _ = ref_graphlet_data
            _, target_seg_to_connectors, target_node_features, _ = target_graphlet_data
            clustering_feats = compute_clustering_coefficient_features(
                ref_seg_id,
                target_seg_id,
                ref_node_features,
                target_node_features,
                ref_seg_to_connectors,
                target_seg_to_connectors,
                alignment.overture_start_frac,
                alignment.overture_end_frac,
                alignment.dataset_start_frac,
                alignment.dataset_end_frac,
            )
        else:
            clustering_feats = {
                "clustering_coef_ref": 0.0,
                "clustering_coef_target": 0.0,
                "clustering_coef_delta": 0.0,
            }

    # Log timing summary periodically
    log_timing_summary_if_needed()

    return {
        # Per-pair geometric features (not batchable)
        "mean_hausdorff_distance_m": mean_hausdorff,
        "hausdorff_p95_m": p95_hausdorff,
        "collinear_gap_ratio": collinear_gap_ratio,
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
        # Endpoint proximity
        "min_endpoint_proximity_m": endpoint_features.get(
            "min_endpoint_proximity_m", MAX_DISTANCE_METERS
        ),
        "max_endpoint_proximity_m": endpoint_features.get(
            "max_endpoint_proximity_m", MAX_DISTANCE_METERS
        ),
        "shared_endpoint_count": endpoint_features.get("shared_endpoint_count", 0),
        # Lateral offset
        "lateral_offset_m": min(lateral_offset, MAX_DISTANCE_METERS),
        "lateral_offset_iqr_m": min(lateral_iqr, MAX_DISTANCE_METERS),
        "lateral_offset_p95_m": min(lateral_p95, MAX_DISTANCE_METERS),
        # Topology
        "from_degree_ref": from_degree_ref,
        "to_degree_ref": to_degree_ref,
        "from_degree_target": from_degree_target,
        "to_degree_target": to_degree_target,
        "degree_match_score": degree_match,
        "degree_signature_similarity": sig_similarity,
        "is_dead_end_ref": is_dead_end_ref,
        "is_dead_end_target": is_dead_end_target,
        "dead_end_match": dead_end_match,
        "is_intersection_ref": is_intersection_ref,
        "is_intersection_target": is_intersection_target,
        "intersection_match": intersection_match,
        # Alignment coverage
        "ref_coverage": coverage_feats["ref_coverage"],
        "target_coverage": coverage_feats["target_coverage"],
        "min_coverage": coverage_feats["min_coverage"],
        "coverage_ratio": coverage_feats["coverage_ratio"],
        # Graphlet features
        "graphlet_similarity": (
            graphlet_features.get("graphlet_similarity", 0.5) if graphlet_features else 0.5
        ),
        "endpoint_degree_similarity": (
            graphlet_features.get("endpoint_degree_similarity", 0.5) if graphlet_features else 0.5
        ),
        # Clustering coefficient features
        "clustering_coef_ref": clustering_feats["clustering_coef_ref"],
        "clustering_coef_target": clustering_feats["clustering_coef_target"],
        "clustering_coef_delta": clustering_feats["clustering_coef_delta"],
        # Sinuosity
        "sinuosity_ref": sinuosity_ref,
        "sinuosity_target": sinuosity_target,
        "sinuosity_delta": sinuosity_delta,
        # Heading consistency
        "heading_consistency_ref": heading_consistency_ref,
        "heading_consistency_target": heading_consistency_target,
        "heading_consistency_delta": heading_consistency_delta,
        # Vertex density
        "vertex_density_ref": vertex_density_ref,
        "vertex_density_target": vertex_density_target,
        "vertex_density_ratio": vertex_density_ratio,
        # Length
        "min_length_m": min_length_m,
        # Shape complexity
        "shape_complexity_ref": shape_complexity_ref,
        "shape_complexity_target": shape_complexity_target,
        "shape_complexity_delta": shape_complexity_delta,
        # Numeric route matching
        "name_numeric_match": name_numeric_match,
        # Route prefix matching
        "route_prefix_match": route_prefix_match,
        # Parallel sibling features (4) - detect split vs centerline representation
        "has_parallel_sibling_ref": has_parallel_sibling_ref,
        "offset_vs_half_corridor_ratio": offset_vs_half_corridor_ratio,
        "offset_over_expected_halfwidth": offset_over_expected_halfwidth,
        "likely_representation_mismatch": likely_representation_mismatch,
        # Shape/geometric features (new)
        "angle_histogram_similarity": angle_histogram_similarity,
        "edge_distance_rmse_m": edge_distance_rmse_m,
    }


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
    ref_graphlet_data: tuple | None = None,
    target_graphlet_data: tuple | None = None,
    ref_seg_id: str | None = None,
    target_seg_id: str | None = None,
    ref_sibling_context: SiblingSearchContext | None = None,
    target_sibling_context: SiblingSearchContext | None = None,
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
        ref_graphlet_data: Graphlet data for reference (G, seg_to_connectors, node_features, use_connectors)
        target_graphlet_data: Graphlet data for target (G, seg_to_connectors, node_features, use_connectors)
        ref_seg_id: Reference segment ID (required for aligned topology when using graphlet_data)
        target_seg_id: Target segment ID (required for aligned topology when using graphlet_data)

    Returns:
        Dictionary of feature name -> value. Keys match FEATURE_COLUMNS from config.py.
    """
    try:
        # Determine geometries for similarity features
        # If alignment is provided, extract sublines for computing similarity features
        # (hausdorff, buffer_iou, etc.) on comparable portions only.
        # Topology/endpoint features still use full geometries.
        #
        # Optimization: Skip subline extraction when coverage is >99.5%.
        # When alignment covers nearly the full geometry, extracting a subline
        # just creates a nearly-identical geometry with unnecessary overhead.
        # Note: This threshold must be very high (>99%) to avoid conflicting
        # with divergence detection (PR #81) which trims alignment at 95-99%
        # coverage — using full geometry at those levels re-introduces the
        # divergent portions that were deliberately trimmed.
        HIGH_COVERAGE_THRESHOLD = 0.995

        with timed_section("subline_extraction"):
            if alignment is not None:
                # Calculate coverage for each geometry
                ref_coverage = alignment.overture_end_frac - alignment.overture_start_frac
                target_coverage = alignment.dataset_end_frac - alignment.dataset_start_frac

                # Only extract subline if coverage is below threshold
                if ref_coverage >= HIGH_COVERAGE_THRESHOLD:
                    geom_for_similarity_ref = ref_geom  # Use original (cacheable)
                else:
                    ref_subline = create_subline(
                        ref_geom, alignment.overture_start_frac, alignment.overture_end_frac
                    )
                    geom_for_similarity_ref = ref_subline if ref_subline else ref_geom

                if target_coverage >= HIGH_COVERAGE_THRESHOLD:
                    geom_for_similarity_target = target_geom  # Use original (cacheable)
                else:
                    target_subline = create_subline(
                        target_geom, alignment.dataset_start_frac, alignment.dataset_end_frac
                    )
                    geom_for_similarity_target = target_subline if target_subline else target_geom
            else:
                geom_for_similarity_ref = ref_geom
                geom_for_similarity_target = target_geom

        # Extract coords once for functions that accept optional coords parameter
        # This eliminates redundant np.array(line.coords) calls (~4.2 µs each)
        with timed_section("coord_extraction"):
            coords_ref = np.array(geom_for_similarity_ref.coords)
            coords_target = np.array(geom_for_similarity_target.coords)

        # Compute geometric features on aligned sublines (or full geom if no alignment)
        with timed_section("geometric_features"):
            geom_features = compute_geometric_features(
                geom_for_similarity_ref,
                geom_for_similarity_target,
            )

        # Compute non-geometric features (semantic, topology, etc.)
        non_geom = _compute_non_geometric_features(
            geom_sim_ref=geom_for_similarity_ref,
            geom_sim_target=geom_for_similarity_target,
            coords_ref=coords_ref,
            coords_target=coords_target,
            ref_name=ref_name,
            target_name=target_name,
            ref_class=ref_class,
            target_class=target_class,
            ref_subclass=ref_subclass,
            target_subclass=target_subclass,
            endpoint_features=endpoint_features,
            ref_topology=ref_topology,
            target_topology=target_topology,
            alignment=alignment,
            graphlet_features=graphlet_features,
            ref_graphlet_data=ref_graphlet_data,
            target_graphlet_data=target_graphlet_data,
            ref_seg_id=ref_seg_id,
            target_seg_id=target_seg_id,
            geom_features=geom_features,
            ref_sibling_context=ref_sibling_context,
            target_sibling_context=target_sibling_context,
        )

        # Merge batchable geometric features with non-geometric features
        features = {
            # Batchable geometric features (from compute_geometric_features)
            "hausdorff_distance_m": geom_features.hausdorff_distance,
            "buffer_iou_5m": geom_features.buffer_iou_5m,
            "buffer_iou_15m": geom_features.buffer_iou_15m,
            "heading_delta": geom_features.heading_delta,
            "length_ratio": geom_features.length_ratio,
            "centroid_distance_m": geom_features.centroid_distance,
            # Non-geometric and per-pair geometric features
            **non_geom,
        }

        # Embed per-pair timing data in the feature dict for main-process aggregation
        if is_profiling_enabled():
            stats = get_timing_stats()
            for name, total in stats.totals.items():
                features[f"_t_{name}"] = total
            stats.reset()

        return features

    except Exception as e:
        # Log at warning level to catch bugs early (see TODO.md for planned improvements)
        logger.warning(f"Feature computation failed: {type(e).__name__}: {e}")
        # Return error values with metadata for tracking
        return _get_error_features(error=e, phase="compute_pair_features")


def _get_error_features(
    error: Exception | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    """Return default feature values for error cases.

    Args:
        error: Optional exception that caused the error
        phase: Optional phase where the error occurred (e.g., "compute_pair_features")

    Returns a dict with all features from FEATURE_COLUMNS, using neutral/default
    values that won't artificially inflate or deflate match scores. Also includes
    error metadata fields (_error, _error_type, _error_phase) when error is provided.
    """
    features = {
        # Geometric features
        "hausdorff_distance_m": MAX_DISTANCE_METERS,
        "mean_hausdorff_distance_m": MAX_DISTANCE_METERS,
        "hausdorff_p95_m": MAX_DISTANCE_METERS,
        "buffer_iou_5m": 0.0,
        "buffer_iou_15m": 0.0,
        "heading_delta": 180.0,
        "length_ratio": 0.0,
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
        # Clustering coefficient features - default to 0 (no clustering)
        "clustering_coef_ref": 0.0,
        "clustering_coef_target": 0.0,
        "clustering_coef_delta": 0.0,
        # Sinuosity features - default to straight line (1.0)
        "sinuosity_ref": 1.0,
        "sinuosity_target": 1.0,
        "sinuosity_delta": 0.0,
        # Heading consistency features - default to perfectly straight (1.0)
        "heading_consistency_ref": 1.0,
        "heading_consistency_target": 1.0,
        "heading_consistency_delta": 0.0,
        # Vertex density features - default to 0 (unknown)
        "vertex_density_ref": 0.0,
        "vertex_density_target": 0.0,
        "vertex_density_ratio": 0.0,
        # Length features
        "min_length_m": 0.0,
        # Shape complexity features - default to no turns
        "shape_complexity_ref": 0,
        "shape_complexity_target": 0,
        "shape_complexity_delta": 0,
        # Numeric route matching - 0.0 (no signal when neither has number)
        "name_numeric_match": 0.0,
        # Route prefix matching - 0.5 neutral (don't penalize non-routes)
        "route_prefix_match": 0.5,
        # Parallel sibling features - default to no sibling detected
        "has_parallel_sibling_ref": 0.0,
        "offset_vs_half_corridor_ratio": 1.0,
        "offset_over_expected_halfwidth": 0.0,
        "likely_representation_mismatch": 0.0,
        # Shape/geometric features - default to neutral values
        "angle_histogram_similarity": 1.0,  # Treat as compatible in error case
        "edge_distance_rmse_m": MAX_DISTANCE_METERS,
        # Road properties - default to neutral values
    }

    # Add error metadata only if error is provided
    if error is not None:
        features["_error"] = str(error)
        features["_error_type"] = type(error).__name__
        features["_error_phase"] = phase or "unknown"

    return features


def precompute_parallel_siblings(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    name_column: str = "names",
    class_column: str = "class",
    ids_to_compute: set[str] | None = None,
    spatial_index: Any | None = None,
) -> dict[str, tuple[bool, float]]:
    """Pre-compute parallel sibling info for segments in a GeoDataFrame.

    This function detects which segments are part of split carriageways
    (dual highways) by finding nearby parallel segments with matching names/classes.

    Args:
        gdf: GeoDataFrame with road segments (must be in projected CRS, meters)
        id_column: Column name for segment IDs
        name_column: Column name for segment names
        class_column: Column name for road class
        ids_to_compute: Optional set of segment IDs to compute sibling info for.
            If None, computes for all segments. Use this to filter to only
            labeled/candidate segments for efficiency.
        spatial_index: Optional pre-built STRtree over gdf.geometry. If provided,
            skips building a new one (saves O(N log N) construction time).

    Returns:
        Dict mapping segment_id -> (has_sibling, sibling_distance)
        where has_sibling is True if a parallel sibling was found,
        and sibling_distance is the lateral offset in meters (inf if no sibling).
    """
    from .relational import precompute_parallel_siblings as _precompute_siblings

    # Extract data from GeoDataFrame
    geometries = list(gdf.geometry)
    segment_ids = [str(sid) for sid in gdf[id_column]]

    # Handle name column - may be nested dict, list, or string
    names: list[str | None] = []
    for idx in range(len(gdf)):
        name_val = gdf.iloc[idx].get(name_column) if name_column in gdf.columns else None
        if name_val is None:
            names.append(None)
        elif isinstance(name_val, dict):
            # Overture format: {"primary": "Main St", "common": [...]}
            names.append(name_val.get("primary") or name_val.get("common", [None])[0])
        elif isinstance(name_val, list) and len(name_val) > 0:
            names.append(str(name_val[0]) if name_val[0] else None)
        else:
            names.append(str(name_val) if name_val else None)

    # Handle class column
    classes: list[str | None] = []
    if class_column in gdf.columns:
        for cls in gdf[class_column]:
            classes.append(str(cls) if cls else None)
    else:
        classes = [None] * len(gdf)

    return _precompute_siblings(
        geometries, segment_ids, names, classes, ids_to_compute, spatial_index
    )


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

    # Use full graphlet features for richer topology comparison
    # This computes: degree, triangles, squares, clustering, two_hop_count, is_articulation
    if use_connectors:
        logger.info("Building connector-based graph for graphlet features (explicit connectors)...")
        G, seg_to_connectors, node_features = build_connector_graph(
            gdf_reset,
            id_column=id_column,
            connectors_column=connectors_column,
            tolerance_m=tolerance_m,
            degrees_only=False,  # Full 6-feature vectors
        )
        n_nodes = len(node_features) if node_features else 0
        logger.debug(
            f"[precompute] Built connector graph with {n_nodes} nodes "
            f"in {time.perf_counter() - t0:.2f}s"
        )
        return G, seg_to_connectors, node_features, True
    else:
        # Use inferred connector graph that detects mid-segment crossings
        # This provides richer topology than endpoint-only inference
        from .spatial_context import build_inferred_connector_graph

        logger.info("Building inferred connector graph for graphlet features...")
        G, seg_to_connectors, node_features = build_inferred_connector_graph(
            gdf_reset, id_column=id_column, tolerance_m=tolerance_m, degrees_only=False
        )
        n_nodes = len(node_features) if node_features else 0
        logger.debug(
            f"[precompute] Built inferred connector graph with {n_nodes} nodes "
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
