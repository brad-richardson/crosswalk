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

PROFILING_ENABLED = os.environ.get("MATCHER_PROFILE", "0") == "1"


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
    if not PROFILING_ENABLED:
        yield
        return

    t0 = time.perf_counter()
    yield
    get_timing_stats().record(name, time.perf_counter() - t0)


def log_timing_summary_if_needed(interval: int = 5000) -> None:
    """Log timing summary every N calls (worker-side).

    Call this after each compute_pair_features() invocation.
    """
    if not PROFILING_ENABLED:
        return

    count = increment_call_count()
    if count % interval == 0:
        stats = get_timing_stats()
        logger.info(f"Worker timing after {count} pairs:\n{stats.summary()}")


from ..config import DEFAULT_TOPOLOGY_FEATURES, FEATURE_COLUMNS, MAX_DISTANCE_METERS
from .alignment import AlignmentResult, compute_coverage_features, create_subline
from .geometric import (
    compute_geometric_features,
    compute_heading_consistency,
    compute_shape_complexity,
    compute_sinuosity,
    compute_vertex_density,
)
from .relational import compute_perpendicular_offset
from .semantic import (
    compute_class_similarity,
    compute_name_numeric_match,
    compute_name_similarity,
)
from .spatial_context import (
    build_connector_graph,
    compute_degree_match_score,
    compute_degree_signature_similarity,
    graphlet_similarity_with_alignment,
)

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
    ref_graphlet_data: tuple | None = None,
    target_graphlet_data: tuple | None = None,
    ref_seg_id: str | None = None,
    target_seg_id: str | None = None,
    precomputed_buffers: dict | None = None,
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
        precomputed_buffers: Pre-computed buffers for full geometries (optional).
            Only used when alignment coverage is high (>95%), otherwise subline buffers are computed.

    Returns:
        Dictionary of feature name -> value. Keys match FEATURE_COLUMNS from config.py.
    """
    try:
        # Determine geometries for similarity features
        # If alignment is provided, extract sublines for computing similarity features
        # (hausdorff, buffer_iou, etc.) on comparable portions only.
        # Topology/endpoint features still use full geometries.
        #
        # Optimization: Skip subline extraction when coverage is >95%.
        # When alignment covers nearly the full geometry, extracting a subline
        # just creates a nearly-identical geometry that defeats the buffer cache.
        # Using the original geometry allows cache hits across pairs.
        HIGH_COVERAGE_THRESHOLD = 0.95

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
        # Pass pre-computed buffers when using full geometries (high coverage)
        with timed_section("geometric_features"):
            # Determine if we can use pre-computed buffers (full geometry, not sublines)
            use_precomputed = (
                precomputed_buffers is not None
                and geom_for_similarity_ref is ref_geom
                and geom_for_similarity_target is target_geom
            )
            if use_precomputed:
                geom_features = compute_geometric_features(
                    geom_for_similarity_ref,
                    geom_for_similarity_target,
                    precomputed_buffers=precomputed_buffers,
                )
            else:
                geom_features = compute_geometric_features(
                    geom_for_similarity_ref, geom_for_similarity_target
                )

        # Compute semantic features
        with timed_section("name_similarity"):
            name_sim = compute_name_similarity(ref_name, target_name)

        with timed_section("class_similarity"):
            class_sim = compute_class_similarity(
                ref_class, target_class, ref_subclass, target_subclass
            )

        # Compute lateral offset on aligned sublines (not full geometries)
        # This prevents segments that extend beyond the overlap from inflating the offset
        with timed_section("perpendicular_offset"):
            lateral_offset, lateral_iqr, lateral_p95 = compute_perpendicular_offset(
                geom_for_similarity_target, geom_for_similarity_ref
            )

        # Compute sinuosity on aligned sublines (pass pre-extracted coords)
        with timed_section("sinuosity"):
            sinuosity_ref = compute_sinuosity(geom_for_similarity_ref, coords=coords_ref)
            sinuosity_target = compute_sinuosity(geom_for_similarity_target, coords=coords_target)
            sinuosity_delta = abs(sinuosity_ref - sinuosity_target)

        # Compute heading consistency on aligned sublines
        with timed_section("heading_consistency"):
            heading_consistency_ref = compute_heading_consistency(geom_for_similarity_ref)
            heading_consistency_target = compute_heading_consistency(geom_for_similarity_target)
            heading_consistency_delta = abs(heading_consistency_ref - heading_consistency_target)

        # Compute vertex density on aligned sublines (pass pre-extracted coords)
        with timed_section("vertex_density"):
            vertex_density_ref = compute_vertex_density(geom_for_similarity_ref, coords=coords_ref)
            vertex_density_target = compute_vertex_density(
                geom_for_similarity_target, coords=coords_target
            )
            # Ratio: min/max to get value in [0, 1]
            if vertex_density_ref > 0 and vertex_density_target > 0:
                vertex_density_ratio = min(vertex_density_ref, vertex_density_target) / max(
                    vertex_density_ref, vertex_density_target
                )
            else:
                vertex_density_ratio = 0.0

        # Compute min length using aligned subline lengths
        with timed_section("length_computation"):
            ref_length = geom_for_similarity_ref.length
            target_length = geom_for_similarity_target.length
            min_length_m = min(ref_length, target_length)

        # Compute shape complexity on aligned sublines (pass pre-extracted coords)
        with timed_section("shape_complexity"):
            shape_complexity_ref = compute_shape_complexity(
                geom_for_similarity_ref, coords=coords_ref
            )
            shape_complexity_target = compute_shape_complexity(
                geom_for_similarity_target, coords=coords_target
            )
            shape_complexity_delta = abs(shape_complexity_ref - shape_complexity_target)

        # Compute name numeric match for numbered routes
        with timed_section("name_numeric_match"):
            name_numeric_match = compute_name_numeric_match(ref_name, target_name)

        # Require endpoint features to be provided
        # These should be computed on aligned subline endpoints, not full geometry
        with timed_section("endpoint_features_lookup"):
            if endpoint_features is None:
                raise ValueError(
                    "endpoint_features is required - must be computed on aligned subline endpoints"
                )

        # Compute topology features - prefer alignment-aware when graphlet data is available
        # This is critical for partial overlaps where we need degrees at the aligned
        # subline endpoints, not at the full geometry endpoints.
        with timed_section("aligned_topology"):
            use_aligned_topology = (
                alignment is not None
                and ref_graphlet_data is not None
                and target_graphlet_data is not None
                and ref_seg_id is not None
                and target_seg_id is not None
            )

            if use_aligned_topology:
                # Compute alignment-aware topology from graphlet connector data
                from .spatial_context import compute_aligned_topology_features

                # Unpack graphlet data (G, seg_to_connectors, node_features, use_connectors)
                _, ref_seg_to_connectors, ref_node_features, _ = ref_graphlet_data
                _, target_seg_to_connectors, target_node_features, _ = target_graphlet_data

                # Compute aligned topology for reference
                ref_aligned_topo = compute_aligned_topology_features(
                    ref_seg_id,
                    ref_seg_to_connectors,
                    ref_node_features,
                    alignment.overture_start_frac,
                    alignment.overture_end_frac,
                )

                # Compute aligned topology for target
                target_aligned_topo = compute_aligned_topology_features(
                    target_seg_id,
                    target_seg_to_connectors,
                    target_node_features,
                    alignment.dataset_start_frac,
                    alignment.dataset_end_frac,
                )

                # Use aligned values
                from_degree_ref = ref_aligned_topo["from_degree"]
                to_degree_ref = ref_aligned_topo["to_degree"]
                from_degree_target = target_aligned_topo["from_degree"]
                to_degree_target = target_aligned_topo["to_degree"]
                ref_topology = ref_aligned_topo
                target_topology = target_aligned_topo
            else:
                # Fall back to pre-computed topology (for backward compatibility)
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
        with timed_section("degree_match"):
            degree_match = compute_degree_match_score(
                from_degree_ref, to_degree_ref, from_degree_target, to_degree_target
            )

        # Compute degree signature similarity
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

        # Compute coverage features from alignment
        with timed_section("coverage_features"):
            coverage_feats = compute_coverage_features(alignment)

        # Log timing summary periodically (when MATCHER_PROFILE=1)
        log_timing_summary_if_needed()

        return {
            # Geometric (distance features use _m suffix to indicate meters)
            "hausdorff_distance_m": geom_features.hausdorff_distance,
            "mean_hausdorff_distance_m": geom_features.mean_hausdorff_distance,
            "hausdorff_p95_m": geom_features.hausdorff_p95_distance,
            "buffer_iou_5m": geom_features.buffer_iou_5m,
            "buffer_iou_15m": geom_features.buffer_iou_15m,
            "heading_delta": geom_features.heading_delta,
            "length_ratio": geom_features.length_ratio,
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
            # Sinuosity features
            "sinuosity_ref": sinuosity_ref,
            "sinuosity_target": sinuosity_target,
            "sinuosity_delta": sinuosity_delta,
            # Heading consistency features
            "heading_consistency_ref": heading_consistency_ref,
            "heading_consistency_target": heading_consistency_target,
            "heading_consistency_delta": heading_consistency_delta,
            # Vertex density features
            "vertex_density_ref": vertex_density_ref,
            "vertex_density_target": vertex_density_target,
            "vertex_density_ratio": vertex_density_ratio,
            # Length features
            "min_length_m": min_length_m,
            # Shape complexity features
            "shape_complexity_ref": shape_complexity_ref,
            "shape_complexity_target": shape_complexity_target,
            "shape_complexity_delta": shape_complexity_delta,
            # Numeric route matching
            "name_numeric_match": name_numeric_match,
        }

    except Exception as e:
        # Log at debug level to avoid noise - errors are counted and reported in ml.py
        logger.debug(f"Feature computation failed: {e}")
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
    }


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
