"""Shared feature computation for ML pipeline and labeling UI.

This module provides a unified interface for computing all features for candidate pairs,
including geometric, semantic, relational, and topology features.
"""

import math
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
    DEFAULT_SNAP_TOLERANCE_M,
    FEATURE_COLUMNS,
    MAX_DISTANCE_METERS,
)


class MissingContextError(TypeError):
    """Required context parameters missing - programmer error, not bad data."""


from .alignment import AlignmentResult, compute_coverage_features, create_subline
from .geometric import (
    GeometricFeatures,
    _compute_hausdorff_stats,
    compute_angle_histogram_similarity,
    compute_collinear_gap_ratio,
    compute_crossing_angle_features,
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
    resolve_best_name_variant,
)
from .spatial_context import (
    build_connector_graph,
    compute_clustering_coefficient_features,
    compute_degree_match_score,
    compute_degree_signature_similarity,
    compute_interior_connector_features,
    compute_shared_anchor_features,
    graphlet_similarity_with_alignment,
)

# Alias for backward compatibility - the authoritative list is in config.py
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS


def _compute_non_geometric_features(
    ref_geom_aligned,
    target_geom_aligned,
    coords_aligned_ref: "np.ndarray",
    coords_aligned_target: "np.ndarray",
    ref_class: str | None,
    target_class: str | None,
    ref_subclass: str | None,
    target_subclass: str | None,
    endpoint_features: dict[str, float],
    ref_topology_full: dict | None,
    target_topology_full: dict | None,
    alignment: AlignmentResult | None,
    graphlet_features: dict[str, float] | None,
    ref_graphlet_data: tuple | None,
    target_graphlet_data: tuple | None,
    ref_seg_id: str | None,
    target_seg_id: str | None,
    geom_features: GeometricFeatures,
    precomputed_lateral_offset: tuple[float, float, float] | None = None,
    ref_sibling_context_full: SiblingSearchContext | None = None,
    target_sibling_context_full: SiblingSearchContext | None = None,
    ref_names_raw=None,
    target_names_raw=None,
    target_topo_connectors: dict[str, list[tuple[float, int]]] | None = None,
    target_topo_node_features: dict[int, int] | None = None,
    precomputed_sibling_ref: tuple[bool, float, float] | None = None,
    precomputed_sibling_target: tuple[bool, float] | None = None,
    precomputed_crossing_ref: dict[str, float] | None = None,
    precomputed_crossing_target: dict[str, float] | None = None,
    target_overture_connectors: dict[str, list[tuple[float, int]]] | None = None,
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
        ref_geom_aligned: Reference geometry for similarity (aligned portion,
            or full geometry when coverage >= 99.5%)
        target_geom_aligned: Target geometry for similarity (aligned portion,
            or full geometry when coverage >= 99.5%)
        coords_aligned_ref: Pre-extracted coordinates for ref_geom_aligned
        coords_aligned_target: Pre-extracted coordinates for target_geom_aligned
        ref_class: Reference road class
        target_class: Target road class
        ref_subclass: Reference road subclass
        target_subclass: Target road subclass
        endpoint_features: Pre-computed endpoint proximity features
        ref_topology_full: Topology features for reference (full segment)
        target_topology_full: Topology features for target (full segment)
        alignment: Alignment result
        graphlet_features: Pre-computed graphlet similarity features
        ref_graphlet_data: Graphlet data for reference
        target_graphlet_data: Graphlet data for target
        ref_seg_id: Reference segment ID
        target_seg_id: Target segment ID
        geom_features: Pre-computed geometric features (batchable fields filled in)
        precomputed_lateral_offset: Optional pre-computed (mean, iqr, p95) from batch.
        ref_sibling_context_full: Sibling search context for reference (built from
            full geometries of all segments, not just candidates)
        target_sibling_context_full: Sibling search context for target (built from
            full geometries of all segments, not just candidates)
        ref_names_raw: Raw Overture names dict for reference (all variants).
        target_names_raw: Raw target names dict (all variants).
        target_topo_connectors: Synthetic topology connectors for target segments,
            mapping seg_id -> [(frac, node_id), ...]. Used for alignment-aware
            target topology when graphlet_data is not available for the target.
        target_topo_node_features: Node features for synthetic connectors,
            mapping node_id -> degree.
        precomputed_sibling_ref: Cached (has_sibling, sibling_dist, parallel_fraction)
            for the ref segment. Used when aligned geom is the full geom (identity).
        precomputed_sibling_target: Cached (has_sibling, sibling_dist) for the target.
        precomputed_crossing_ref: Cached crossing angle result dict for ref.
        precomputed_crossing_target: Cached crossing angle result dict for target.

    Returns:
        Dictionary of non-geometric feature name -> value, plus per-pair geometric
        fields (mean_hausdorff, p95, collinear_gap) merged in.
    """
    # Per-pair geometric features that can't be batched
    with timed_section("geom_hausdorff_stats"):
        mean_hausdorff, p95_hausdorff = _compute_hausdorff_stats(
            ref_geom_aligned,
            target_geom_aligned,
            coords_a=coords_aligned_ref,
            coords_b=coords_aligned_target,
        )

    with timed_section("geom_collinear_gap"):
        collinear_gap_ratio = compute_collinear_gap_ratio(
            ref_geom_aligned,
            target_geom_aligned,
            coords_a=coords_aligned_ref,
            coords_b=coords_aligned_target,
        )

    # Semantic features
    # When raw names dicts are available with multiple language variants,
    # find the best-matching variant pair. This handles cross-script
    # comparisons (e.g., Chinese primary name + English alt vs English target).
    with timed_section("name_variant_resolution"):
        effective_ref_name, effective_target_name = resolve_best_name_variant(
            ref_names_raw, target_names_raw
        )

    with timed_section("name_similarity"):
        name_sim = compute_name_similarity(effective_ref_name, effective_target_name)

    with timed_section("class_similarity"):
        class_sim = compute_class_similarity(ref_class, target_class, ref_subclass, target_subclass)

    # Lateral offset
    with timed_section("perpendicular_offset"):
        if precomputed_lateral_offset is not None:
            lateral_offset, lateral_iqr, lateral_p95 = precomputed_lateral_offset
        else:
            lateral_offset, lateral_iqr, lateral_p95 = compute_perpendicular_offset(
                target_geom_aligned, ref_geom_aligned
            )

    # Sinuosity
    with timed_section("sinuosity"):
        sinuosity_ref = compute_sinuosity(ref_geom_aligned, coords=coords_aligned_ref)
        sinuosity_target = compute_sinuosity(target_geom_aligned, coords=coords_aligned_target)
        sinuosity_delta = abs(sinuosity_ref - sinuosity_target)

    # Heading consistency
    with timed_section("heading_consistency"):
        heading_consistency_ref = compute_heading_consistency(ref_geom_aligned)
        heading_consistency_target = compute_heading_consistency(target_geom_aligned)
        heading_consistency_delta = abs(heading_consistency_ref - heading_consistency_target)

    # Vertex density
    with timed_section("vertex_density"):
        vertex_density_ref = compute_vertex_density(ref_geom_aligned, coords=coords_aligned_ref)
        vertex_density_target = compute_vertex_density(
            target_geom_aligned, coords=coords_aligned_target
        )
        if vertex_density_ref > 0 and vertex_density_target > 0:
            vertex_density_ratio = min(vertex_density_ref, vertex_density_target) / max(
                vertex_density_ref, vertex_density_target
            )
        else:
            vertex_density_ratio = 0.0

    # Min length (of aligned geometries)
    with timed_section("length_computation"):
        ref_length_aligned = ref_geom_aligned.length
        target_length_aligned = target_geom_aligned.length
        min_length_m = min(ref_length_aligned, target_length_aligned)

    # Shape complexity
    with timed_section("shape_complexity"):
        shape_complexity_ref = compute_shape_complexity(ref_geom_aligned, coords=coords_aligned_ref)
        shape_complexity_target = compute_shape_complexity(
            target_geom_aligned, coords=coords_aligned_target
        )
        shape_complexity_delta = abs(shape_complexity_ref - shape_complexity_target)

    # Angle histogram similarity (shape fingerprint)
    with timed_section("angle_histogram"):
        angle_histogram_similarity = compute_angle_histogram_similarity(
            ref_geom_aligned,
            target_geom_aligned,
            coords_a=coords_aligned_ref,
            coords_b=coords_aligned_target,
        )

    # Edge distance RMSE (Hootenanny's primary metric)
    with timed_section("edge_distance_rmse"):
        edge_distance_rmse_m = compute_edge_distance_rmse(ref_geom_aligned, target_geom_aligned)

    # Name numeric match
    with timed_section("name_numeric_match"):
        name_numeric_match = compute_name_numeric_match(effective_ref_name, effective_target_name)

    # Route prefix match
    with timed_section("route_prefix_match"):
        route_prefix_match = compute_route_prefix_match(effective_ref_name, effective_target_name)

    # Endpoint features
    with timed_section("endpoint_features_lookup"):
        if endpoint_features is None:
            raise ValueError(
                "endpoint_features is required - must be computed on aligned portion endpoints"
            )

    # Topology features — unified code path via compute_aligned_topology_features()
    # Both sides use connector-based alignment-aware topology:
    #   - Ref: Overture explicit connectors (from ref_graphlet_data)
    #   - Target: Synthetic connectors sampled from full-network topology spatial index
    #     (from target_topo_connectors/target_topo_node_features)
    # Fallback: full-segment topology (ref_topology_full/target_topology_full) when
    # connector data is unavailable (e.g., labeling UI without full pipeline).
    with timed_section("aligned_topology"):
        from .spatial_context import compute_aligned_topology_features

        # --- Ref side topology ---
        ref_has_aligned = (
            alignment is not None and ref_graphlet_data is not None and ref_seg_id is not None
        )
        if ref_has_aligned:
            _, ref_seg_to_connectors, ref_node_features, _ = ref_graphlet_data
            ref_topo = compute_aligned_topology_features(
                ref_seg_id,
                ref_seg_to_connectors,
                ref_node_features,
                alignment.overture_start_frac,
                alignment.overture_end_frac,
            )
        elif ref_topology_full is not None:
            ref_topo = ref_topology_full
        else:
            raise MissingContextError(
                "ref_topology is required when aligned topology path is not active. "
                "Call compute_all_topology() and pass the result."
            )

        # --- Target side topology ---
        # Prefer synthetic connectors (sampled from full network spatial index)
        target_has_synthetic = (
            alignment is not None
            and target_topo_connectors is not None
            and target_topo_node_features is not None
            and target_seg_id is not None
        )
        # Fallback: graphlet-based connectors (candidates-only graph — less accurate)
        target_has_graphlet = (
            alignment is not None and target_graphlet_data is not None and target_seg_id is not None
        )

        if target_has_synthetic:
            target_topo = compute_aligned_topology_features(
                target_seg_id,
                target_topo_connectors,
                target_topo_node_features,
                alignment.dataset_start_frac,
                alignment.dataset_end_frac,
            )
        elif target_has_graphlet:
            _, target_seg_to_connectors, target_node_features, _ = target_graphlet_data
            target_topo = compute_aligned_topology_features(
                target_seg_id,
                target_seg_to_connectors,
                target_node_features,
                alignment.dataset_start_frac,
                alignment.dataset_end_frac,
            )
        elif target_topology_full is not None:
            target_topo = target_topology_full
        else:
            raise MissingContextError(
                "target_topology is required when aligned topology path is not active. "
                "Call compute_all_topology() and pass the result."
            )

        from_degree_ref = ref_topo.get("from_degree", float("nan"))
        to_degree_ref = ref_topo.get("to_degree", float("nan"))
        from_degree_target = target_topo.get("from_degree", float("nan"))
        to_degree_target = target_topo.get("to_degree", float("nan"))

    # NaN-propagation guard: if any degree value is NaN, all derived topology
    # features must be NaN too. This avoids truthy/falsy issues with NaN in
    # boolean comparisons (e.g. `if NaN` is truthy in Python).
    _topo_nan = any(
        isinstance(v, float) and math.isnan(v)
        for v in [from_degree_ref, to_degree_ref, from_degree_target, to_degree_target]
    )

    if _topo_nan:
        degree_match = float("nan")
        sig_similarity = float("nan")
        is_dead_end_ref = float("nan")
        is_dead_end_target = float("nan")
        dead_end_match = float("nan")
        is_intersection_ref = float("nan")
        is_intersection_target = float("nan")
        intersection_match = float("nan")
    else:
        # Degree match score
        with timed_section("degree_match"):
            degree_match = compute_degree_match_score(
                from_degree_ref, to_degree_ref, from_degree_target, to_degree_target
            )

        # Degree signature similarity
        with timed_section("degree_signature"):
            ref_sig = ref_topo.get("degree_signature", (1,))
            target_sig = target_topo.get("degree_signature", (1,))
            sig_similarity = compute_degree_signature_similarity(ref_sig, target_sig)

        # Topology flags
        with timed_section("topology_flags"):
            is_dead_end_ref = 1.0 if ref_topo.get("is_dead_end", True) else 0.0
            is_dead_end_target = 1.0 if target_topo.get("is_dead_end", True) else 0.0
            dead_end_match = 1.0 if is_dead_end_ref == is_dead_end_target else 0.0

            is_intersection_ref = 1.0 if ref_topo.get("is_intersection", False) else 0.0
            is_intersection_target = 1.0 if target_topo.get("is_intersection", False) else 0.0
            intersection_match = 1.0 if is_intersection_ref == is_intersection_target else 0.0

    # Coverage features
    with timed_section("coverage_features"):
        coverage_feats = compute_coverage_features(alignment)

    # Parallel sibling features (detect split vs centerline representation)
    # Computed per-pair on aligned portions for accuracy with partial alignments
    with timed_section("sibling_features"):
        # Use precomputed values when available (cache hit for full-geometry pairs)
        if precomputed_sibling_ref is not None:
            has_sibling_ref, sibling_dist_ref, parallel_fraction_ref = precomputed_sibling_ref
        elif ref_sibling_context_full is not None and ref_seg_id is not None:
            has_sibling_ref, sibling_dist_ref, parallel_fraction_ref = find_parallel_sibling(
                segment=ref_geom_aligned,
                segment_id=ref_seg_id,
                segment_name=effective_ref_name,
                segment_class=ref_class,
                spatial_index=ref_sibling_context_full.spatial_index,
                segment_data=ref_sibling_context_full.segment_data,
            )
        else:
            if ref_sibling_context_full is None:
                logger.warning("ref_sibling_context_full is None - sibling detection disabled")
            has_sibling_ref, sibling_dist_ref, parallel_fraction_ref = (
                False,
                MAX_DISTANCE_METERS,
                0.0,
            )

        if precomputed_sibling_target is not None:
            has_sibling_target, sibling_dist_target = precomputed_sibling_target
        elif target_sibling_context_full is not None and target_seg_id is not None:
            has_sibling_target, sibling_dist_target, _ = find_parallel_sibling(
                segment=target_geom_aligned,
                segment_id=target_seg_id,
                segment_name=effective_target_name,
                segment_class=target_class,
                spatial_index=target_sibling_context_full.spatial_index,
                segment_data=target_sibling_context_full.segment_data,
            )
        else:
            if target_sibling_context_full is None:
                logger.warning("target_sibling_context_full is None - sibling detection disabled")
            has_sibling_target, sibling_dist_target = False, MAX_DISTANCE_METERS

        # Core sibling detection
        has_parallel_sibling_ref = float(has_sibling_ref)

        # Derived: likely representation mismatch (XOR: one has sibling, other doesn't)
        likely_representation_mismatch = float(has_sibling_ref != has_sibling_target)

        # Corridor-aware offset ratio
        # Use sibling distance from whichever side is split (has sibling)
        # FIX: When no sibling detected, set to 0.0 instead of computing with MAX_DISTANCE
        # (which gave meaningless ~0.5 values that added noise to the ML model)
        if has_sibling_ref and not has_sibling_target:
            corridor_width = sibling_dist_ref
            half_corridor = corridor_width / 2.0
            offset_vs_half = abs(lateral_offset - half_corridor)
            offset_vs_half_corridor_ratio = offset_vs_half / (corridor_width + 1e-6)
        elif has_sibling_target and not has_sibling_ref:
            corridor_width = sibling_dist_target
            half_corridor = corridor_width / 2.0
            offset_vs_half = abs(lateral_offset - half_corridor)
            offset_vs_half_corridor_ratio = offset_vs_half / (corridor_width + 1e-6)
        elif has_sibling_ref and has_sibling_target:
            corridor_width = min(sibling_dist_ref, sibling_dist_target)
            half_corridor = corridor_width / 2.0
            offset_vs_half = abs(lateral_offset - half_corridor)
            offset_vs_half_corridor_ratio = offset_vs_half / (corridor_width + 1e-6)
        else:
            # No corridor detected - set to 0.0 as a clean signal
            offset_vs_half_corridor_ratio = 0.0

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
                "clustering_coef_ref": float("nan"),
                "clustering_coef_target": float("nan"),
                "clustering_coef_delta": float("nan"),
            }

    # Crossing angle features - detect segments transverse to nearby different-tier corridor
    # Both sides use ref_sibling_context (Overture spatial index) since target datasets
    # are often single-type (e.g., all footway) making target_sibling_context useless
    # for cross-tier detection.
    with timed_section("crossing_angle"):
        if precomputed_crossing_ref is not None:
            crossing_ref = precomputed_crossing_ref
        else:
            crossing_ref = _compute_crossing_angle(
                ref_geom_aligned,
                ref_class,
                ref_seg_id,
                ref_sibling_context_full,
            )
        if precomputed_crossing_target is not None:
            crossing_target = precomputed_crossing_target
        else:
            crossing_target = _compute_crossing_angle(
                target_geom_aligned,
                target_class,
                None,  # target not in ref spatial index, no self-exclusion needed
                ref_sibling_context_full,
            )

    # Interior connector sequence features
    _nan = float("nan")
    with timed_section("interior_connectors"):
        interior_feats: dict[str, float]
        if (
            alignment is not None
            and ref_graphlet_data is not None
            and ref_seg_id is not None
            and target_seg_id is not None
        ):
            _, ref_seg_to_connectors_ic, ref_node_features_ic, _ = ref_graphlet_data
            # Use Overture connectors projected onto target (same ID space as ref)
            target_conn_ic = target_overture_connectors or {}

            # Both sides use the same Overture connector graph node_features
            interior_feats = compute_interior_connector_features(
                ref_seg_id,
                target_seg_id,
                ref_seg_to_connectors_ic,
                target_conn_ic,
                ref_node_features_ic,
                ref_node_features_ic,  # Same node_features — shared Overture ID space
                alignment.overture_start_frac,
                alignment.overture_end_frac,
                alignment.dataset_start_frac,
                alignment.dataset_end_frac,
            )
        else:
            interior_feats = {
                "interior_junction_count_ref": _nan,
                "interior_junction_count_target": _nan,
                "interior_junction_count_delta": _nan,
                "interior_connector_jaccard": _nan,
                "interior_junction_position_sim": _nan,
            }

    # Shared anchor count (alignment endpoints landing on the same Overture connector)
    with timed_section("shared_anchor"):
        shared_anchor_feats: dict[str, float]
        if (
            alignment is not None
            and ref_graphlet_data is not None
            and ref_seg_id is not None
            and target_seg_id is not None
        ):
            _, ref_seg_to_connectors_sa, _, _ = ref_graphlet_data
            target_conn_sa = target_overture_connectors or {}
            # Derive full segment lengths from aligned geometry + coverage fracs.
            # Connectors are stored as fracs of the full segment, so we need
            # the full length to convert tolerance_m to fractional position.
            ref_cov = alignment.overture_end_frac - alignment.overture_start_frac
            target_cov = alignment.dataset_end_frac - alignment.dataset_start_frac
            ref_len = ref_geom_aligned.length / ref_cov if ref_cov > 0 else ref_geom_aligned.length
            target_len = (
                target_geom_aligned.length / target_cov
                if target_cov > 0
                else target_geom_aligned.length
            )
            shared_anchor_feats = compute_shared_anchor_features(
                ref_seg_id,
                target_seg_id,
                ref_seg_to_connectors_sa,
                target_conn_sa,
                alignment.overture_start_frac,
                alignment.overture_end_frac,
                alignment.dataset_start_frac,
                alignment.dataset_end_frac,
                ref_len,
                target_len,
            )
        else:
            shared_anchor_feats = {"shared_anchor_count": _nan}

    # Log timing summary periodically
    log_timing_summary_if_needed()

    return {
        # Per-pair geometric features (not batchable)
        "mean_hausdorff_distance_m": min(mean_hausdorff, MAX_DISTANCE_METERS),
        "hausdorff_p95_m": min(p95_hausdorff, MAX_DISTANCE_METERS),
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
        "min_endpoint_proximity_m": min(
            endpoint_features.get("min_endpoint_proximity_m", float("nan")),
            MAX_DISTANCE_METERS,
        ),
        "max_endpoint_proximity_m": min(
            endpoint_features.get("max_endpoint_proximity_m", float("nan")),
            MAX_DISTANCE_METERS,
        ),
        "shared_endpoint_count": endpoint_features.get("shared_endpoint_count", float("nan")),
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
            graphlet_features.get("graphlet_similarity", float("nan"))
            if graphlet_features
            else float("nan")
        ),
        "endpoint_degree_similarity": (
            graphlet_features.get("endpoint_degree_similarity", float("nan"))
            if graphlet_features
            else float("nan")
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
        # Parallel sibling features (5) - detect split vs centerline representation
        "has_parallel_sibling_ref": has_parallel_sibling_ref,
        "parallel_fraction_ref": parallel_fraction_ref,
        "offset_vs_half_corridor_ratio": offset_vs_half_corridor_ratio,
        "offset_over_expected_halfwidth": offset_over_expected_halfwidth,
        "likely_representation_mismatch": likely_representation_mismatch,
        # Shape/geometric features (new)
        "angle_histogram_similarity": angle_histogram_similarity,
        "edge_distance_rmse_m": min(edge_distance_rmse_m, MAX_DISTANCE_METERS),
        # Crossing angle features (4) - detect ACROSS-role segments (2 per side)
        "crossing_angle_min_ref": crossing_ref["crossing_angle_min"],
        "transverse_neighbor_fraction_ref": crossing_ref["transverse_neighbor_fraction"],
        "crossing_angle_min_target": crossing_target["crossing_angle_min"],
        "transverse_neighbor_fraction_target": crossing_target["transverse_neighbor_fraction"],
        # Interior connector sequence features (5)
        **interior_feats,
        # Shared anchor count (1)
        **shared_anchor_feats,
    }


_DEFAULT_CROSSING_ANGLE_FEATURES = {
    "crossing_angle_min": float("nan"),
    "crossing_angle_mean": float("nan"),
    "crossing_angle_std": float("nan"),
    "transverse_neighbor_fraction": float("nan"),
}

_CROSSING_ANGLE_SEARCH_RADIUS_M = 30.0


def _compute_crossing_angle(
    geom,
    road_class: str | None,
    seg_id: str | None,
    sibling_context: SiblingSearchContext | None,
) -> dict[str, float]:
    """Compute crossing angle features using the sibling search context.

    Queries nearby segments from the spatial index, filters to different
    traffic tiers, and delegates to compute_crossing_angle_features.

    Args:
        geom: Segment geometry (projected CRS, meters)
        road_class: Road class of the segment
        seg_id: Segment ID (to exclude self from results)
        sibling_context: Spatial index + metadata for the dataset

    Returns:
        Dict with crossing_angle_min, crossing_angle_mean,
        crossing_angle_std, transverse_neighbor_fraction
    """
    if sibling_context is None or geom is None or geom.is_empty:
        return _DEFAULT_CROSSING_ANGLE_FEATURES.copy()

    # Query spatial index for nearby segments
    buffer_geom = geom.buffer(_CROSSING_ANGLE_SEARCH_RADIUS_M)
    candidate_indices = sibling_context.spatial_index.query(buffer_geom)

    nearby_geoms: list = []
    nearby_classes: list[str | None] = []
    for idx in candidate_indices:
        sid, _, cls = sibling_context.segment_data[idx]
        # Exclude self
        if seg_id is not None and sid == seg_id:
            continue
        # Get geometry from spatial index
        nearby_geom = sibling_context.spatial_index.geometries[idx]
        nearby_geoms.append(nearby_geom)
        nearby_classes.append(cls)

    if not nearby_geoms:
        return _DEFAULT_CROSSING_ANGLE_FEATURES.copy()

    return compute_crossing_angle_features(
        candidate=geom,
        nearby_geometries=nearby_geoms,
        nearby_classes=nearby_classes,
        candidate_class=road_class,
    )


def _compute_intersection_overlap_features(
    ref_geom_full,
    target_geom_full,
    alignment: AlignmentResult | None,
) -> dict[str, float]:
    """Compute intersection overlap features for a candidate pair.

    These features encode the "overlap at intersection but doesn't continue"
    false positive pattern. They measure:
    1. How far the target continues past alignment boundary along ref heading
    2. Max heading divergence at alignment boundaries

    Args:
        ref_geom_full: Reference full geometry (LineString, projected CRS)
        target_geom_full: Target full geometry (LineString, projected CRS)
        alignment: Alignment result (None → defaults)

    Returns:
        Dict with post_node_continuation_m, endpoint_heading_divergence
    """
    from ._jit_helpers import (
        angle_diff_numba,
        compute_continuation_along_heading_numba,
        compute_heading_at_fraction_numba,
    )

    nan = float("nan")

    if alignment is None:
        return {
            "post_node_continuation_m": nan,
            "endpoint_heading_divergence": nan,
        }

    ref_start = alignment.overture_start_frac
    ref_end = alignment.overture_end_frac
    target_start = alignment.dataset_start_frac
    target_end = alignment.dataset_end_frac

    # Pre-compute coordinate arrays and per-segment lengths
    ref_coords = np.array(ref_geom_full.coords)
    target_coords = np.array(target_geom_full.coords)
    ref_length_full = ref_geom_full.length
    target_length_full = target_geom_full.length

    def _seg_lengths(coords):
        """Compute per-segment lengths from coordinate array."""
        diffs = np.diff(coords, axis=0)
        return np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)

    ref_seg_lens = _seg_lengths(ref_coords)
    target_seg_lens = _seg_lengths(target_coords)

    # Tolerance for "fully aligned" check
    FRAC_TOL = 0.01

    # ── 1. Post-node continuation ──────────────────────────────────
    # At each boundary where target has un-aligned remainder, measure
    # how far the remainder continues along the ref heading direction.
    continuation_values = []

    # Check start boundary (target_start > FRAC_TOL means remainder at start)
    has_start_remainder = target_start > FRAC_TOL
    has_end_remainder = target_end < (1.0 - FRAC_TOL)

    if not has_start_remainder and not has_end_remainder:
        # Target fully aligned — no remainder to measure
        post_node_continuation_m = nan
    else:
        if has_start_remainder:
            # Ref heading at the start boundary
            ref_heading_start = compute_heading_at_fraction_numba(
                ref_coords, ref_seg_lens, ref_length_full, ref_start
            )
            rad = np.radians(ref_heading_start)
            hdx, hdy = np.cos(rad), np.sin(rad)

            # Target remainder: from 0 to target_start (reversed — walk away from boundary)
            remainder_sub = create_subline(target_geom_full, 0.0, target_start)
            if remainder_sub is not None and remainder_sub.length > 0.5:
                rem_coords = np.array(remainder_sub.coords)
                # Reverse so we walk FROM boundary outward
                rem_coords = rem_coords[::-1].copy()
                cont = compute_continuation_along_heading_numba(rem_coords, hdx, hdy)
                continuation_values.append(cont)

        if has_end_remainder:
            # Ref heading at the end boundary
            ref_heading_end = compute_heading_at_fraction_numba(
                ref_coords, ref_seg_lens, ref_length_full, ref_end
            )
            rad = np.radians(ref_heading_end)
            hdx, hdy = np.cos(rad), np.sin(rad)

            # Target remainder: from target_end to 1.0
            remainder_sub = create_subline(target_geom_full, target_end, 1.0)
            if remainder_sub is not None and remainder_sub.length > 0.5:
                rem_coords = np.array(remainder_sub.coords)
                cont = compute_continuation_along_heading_numba(rem_coords, hdx, hdy)
                continuation_values.append(cont)

        if continuation_values:
            post_node_continuation_m = min(continuation_values)
        else:
            post_node_continuation_m = nan

    # ── 2. Endpoint heading divergence ─────────────────────────────
    # Max heading difference between ref and target at alignment boundaries
    divergence_values = []

    ref_heading_at_start = compute_heading_at_fraction_numba(
        ref_coords, ref_seg_lens, ref_length_full, ref_start
    )
    target_heading_at_start = compute_heading_at_fraction_numba(
        target_coords, target_seg_lens, target_length_full, target_start
    )
    divergence_values.append(angle_diff_numba(ref_heading_at_start, target_heading_at_start))

    ref_heading_at_end = compute_heading_at_fraction_numba(
        ref_coords, ref_seg_lens, ref_length_full, ref_end
    )
    target_heading_at_end = compute_heading_at_fraction_numba(
        target_coords, target_seg_lens, target_length_full, target_end
    )
    divergence_values.append(angle_diff_numba(ref_heading_at_end, target_heading_at_end))

    endpoint_heading_divergence = max(divergence_values)

    return {
        "post_node_continuation_m": post_node_continuation_m,
        "endpoint_heading_divergence": endpoint_heading_divergence,
    }


def assemble_feature_dict(
    geom_features: GeometricFeatures,
    aligned_length_m: float,
    non_geom: dict[str, float],
    intersection_overlap_feats: dict[str, float],
) -> dict[str, float]:
    """Single source of truth for merging geometric + non-geometric features.

    Applies MAX_DISTANCE_METERS clamping to distance features from the
    batchable geometric path (hausdorff). Distance features in non_geom
    are already clamped by _compute_non_geometric_features.

    Both compute_pair_features() and _compute_feature_chunk() (ml.py) call
    this to avoid divergent feature assembly logic.
    """
    return {
        "hausdorff_distance_m": min(geom_features.hausdorff_distance, MAX_DISTANCE_METERS),
        "buffer_iou_5m": geom_features.buffer_iou_5m,
        "buffer_iou_15m": geom_features.buffer_iou_15m,
        "heading_delta": geom_features.heading_delta,
        "aligned_length_m": aligned_length_m,
        **non_geom,
        **intersection_overlap_feats,
    }


def compute_pair_features(
    ref_geom_full,
    target_geom_full,
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
    ref_names_raw=None,
    target_names_raw=None,
    target_topo_connectors: dict[str, list[tuple[float, int]]] | None = None,
    target_topo_node_features: dict[int, int] | None = None,
    target_overture_connectors: dict[str, list[tuple[float, int]]] | None = None,
) -> dict[str, float]:
    """Compute all features for a single candidate pair.

    This is the AUTHORITATIVE source for feature computation. All consumers
    (ml.py scoring, backfill, labeling UI) MUST call this function to ensure
    consistency between training and inference.

    Args:
        ref_geom_full: Reference full geometry (LineString)
        target_geom_full: Target full geometry (LineString)
        ref_class: Reference road class
        target_class: Target road class
        ref_subclass: Reference road subclass (optional)
        target_subclass: Target road subclass (optional)
        endpoint_features: Pre-computed endpoint proximity features (optional)
        ref_topology: Pre-computed topology features for reference (required unless
            aligned topology path is active via graphlet_data + alignment + seg_ids)
        target_topology: Pre-computed topology features for target (required unless
            aligned topology path is active via graphlet_data + alignment + seg_ids)
        alignment: Pre-computed alignment result for extracting aligned portions (optional)
        graphlet_features: Pre-computed graphlet similarity features (optional)
        ref_graphlet_data: Graphlet data for reference (G, seg_to_connectors, node_features, use_connectors)
        target_graphlet_data: Graphlet data for target (G, seg_to_connectors, node_features, use_connectors)
        ref_seg_id: Reference segment ID (required for aligned topology when using graphlet_data)
        target_seg_id: Target segment ID (required for aligned topology when using graphlet_data)
        ref_sibling_context: Sibling search context for reference (built from
            full geometries of all segments)
        target_sibling_context: Sibling search context for target (built from
            full geometries of all segments)
        ref_names_raw: Raw Overture names dict with primary + common + rules.
            When provided, the best-matching name variant is selected before
            computing name similarity, improving scores for multilingual segments.
        target_names_raw: Raw target names dict (Overture format).
            When provided, bilateral variant resolution finds the best-matching
            pair across all ref and target name variants.

    Returns:
        Dictionary of feature name -> value. Keys match FEATURE_COLUMNS from config.py.
    """
    _current_phase = "init"
    try:
        # Determine geometries for similarity features
        # If alignment is provided, extract aligned portions for computing similarity features
        # (hausdorff, buffer_iou, etc.) on comparable portions only.
        # Topology/endpoint features use alignment-aware calculations via connectors.
        #
        # Optimization: Skip aligned portion extraction when coverage is >99.5%.
        # When alignment covers nearly the full geometry, extracting a portion
        # just creates a nearly-identical geometry with unnecessary overhead.
        # Note: This threshold must be very high (>99%) to avoid conflicting
        # with divergence detection (PR #81) which trims alignment at 95-99%
        # coverage — using full geometry at those levels re-introduces the
        # divergent portions that were deliberately trimmed.
        HIGH_COVERAGE_THRESHOLD = 0.995

        _current_phase = "subline_extraction"
        with timed_section("subline_extraction"):
            if alignment is not None:
                # Calculate coverage for each geometry
                ref_coverage = alignment.overture_end_frac - alignment.overture_start_frac
                target_coverage = alignment.dataset_end_frac - alignment.dataset_start_frac

                # Only extract aligned portion if coverage is below threshold
                if ref_coverage >= HIGH_COVERAGE_THRESHOLD:
                    ref_geom_aligned = ref_geom_full  # Use original (cacheable)
                else:
                    ref_subline = create_subline(
                        ref_geom_full, alignment.overture_start_frac, alignment.overture_end_frac
                    )
                    ref_geom_aligned = ref_subline if ref_subline else ref_geom_full

                if target_coverage >= HIGH_COVERAGE_THRESHOLD:
                    target_geom_aligned = target_geom_full  # Use original (cacheable)
                else:
                    target_subline = create_subline(
                        target_geom_full, alignment.dataset_start_frac, alignment.dataset_end_frac
                    )
                    target_geom_aligned = target_subline if target_subline else target_geom_full
            else:
                ref_geom_aligned = ref_geom_full
                target_geom_aligned = target_geom_full

        # Aligned length: absolute overlap length in meters (uses full geometry length)
        # Coverage features express overlap as fractions, but absolute length matters:
        # 5% coverage on 1km = 50m (plausible), 80% coverage on 12m = ~10m (intersection-only)
        ref_length_full = ref_geom_full.length
        if alignment is not None:
            aligned_length_m = ref_length_full * (
                alignment.overture_end_frac - alignment.overture_start_frac
            )
        else:
            aligned_length_m = 0.0  # No alignment → 0.0, consistent with coverage features

        # Intersection overlap features (continuation, divergence) - uses full geometries
        _current_phase = "intersection_overlap"
        with timed_section("intersection_overlap"):
            intersection_overlap_feats = _compute_intersection_overlap_features(
                ref_geom_full=ref_geom_full,
                target_geom_full=target_geom_full,
                alignment=alignment,
            )

        _current_phase = "coord_extraction"
        # Extract coords once for functions that accept optional coords parameter
        # This eliminates redundant np.array(line.coords) calls (~4.2 µs each)
        with timed_section("coord_extraction"):
            coords_aligned_ref = np.array(ref_geom_aligned.coords)
            coords_aligned_target = np.array(target_geom_aligned.coords)

        _current_phase = "geometric_features"
        # Compute geometric features on aligned portions (or full geom if no alignment)
        with timed_section("geometric_features"):
            geom_features = compute_geometric_features(
                ref_geom_aligned,
                target_geom_aligned,
            )

        _current_phase = "non_geometric_features"
        # Compute non-geometric features (semantic, topology, etc.)
        non_geom = _compute_non_geometric_features(
            ref_geom_aligned=ref_geom_aligned,
            target_geom_aligned=target_geom_aligned,
            coords_aligned_ref=coords_aligned_ref,
            coords_aligned_target=coords_aligned_target,
            ref_class=ref_class,
            target_class=target_class,
            ref_subclass=ref_subclass,
            target_subclass=target_subclass,
            endpoint_features=endpoint_features,
            ref_topology_full=ref_topology,
            target_topology_full=target_topology,
            alignment=alignment,
            graphlet_features=graphlet_features,
            ref_graphlet_data=ref_graphlet_data,
            target_graphlet_data=target_graphlet_data,
            ref_seg_id=ref_seg_id,
            target_seg_id=target_seg_id,
            geom_features=geom_features,
            ref_sibling_context_full=ref_sibling_context,
            target_sibling_context_full=target_sibling_context,
            ref_names_raw=ref_names_raw,
            target_names_raw=target_names_raw,
            target_topo_connectors=target_topo_connectors,
            target_topo_node_features=target_topo_node_features,
            target_overture_connectors=target_overture_connectors,
        )

        _current_phase = "merge_features"
        features = assemble_feature_dict(
            geom_features=geom_features,
            aligned_length_m=aligned_length_m,
            non_geom=non_geom,
            intersection_overlap_feats=intersection_overlap_feats,
        )

        # Embed per-pair timing data in the feature dict for main-process aggregation
        if is_profiling_enabled():
            stats = get_timing_stats()
            for name, total in stats.totals.items():
                features[f"_t_{name}"] = total
            stats.reset()

        return features

    except MissingContextError:
        raise  # Programmer error - must propagate
    except Exception as e:
        # Log at warning level with phase context for debugging
        logger.warning(
            f"Feature computation failed during '{_current_phase}': {type(e).__name__}: {e}"
        )
        # Return error values with metadata for tracking
        return _get_error_features(error=e, phase=f"compute_pair_features/{_current_phase}")


def _get_error_features(
    error: Exception | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    """Return default feature values for error cases.

    Args:
        error: Optional exception that caused the error
        phase: Optional phase where the error occurred (e.g., "compute_pair_features")

    Returns a dict with all features from FEATURE_COLUMNS set to NaN.
    XGBoost handles NaN natively (learns optimal split routing for missing values).
    Also includes error metadata fields (_error, _error_type, _error_phase) when
    error is provided.
    """
    _nan = float("nan")
    features = {col: _nan for col in ALL_FEATURE_COLUMNS}

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

    # Handle name column - extract primary from names struct
    names: list[str | None] = []
    if name_column in gdf.columns:
        for name_val in gdf[name_column]:
            if isinstance(name_val, dict):
                names.append(name_val.get("primary"))
            elif isinstance(name_val, str):
                names.append(name_val)
            else:
                names.append(None)
    else:
        names = [None] * len(gdf)

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
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
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
    aligned portion endpoints. For Overture data, uses explicit connectors.
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
        return {"graphlet_similarity": float("nan"), "endpoint_degree_similarity": float("nan")}

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
        return {"graphlet_similarity": float("nan"), "endpoint_degree_similarity": float("nan")}
