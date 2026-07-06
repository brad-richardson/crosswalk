"""Shared pipeline setup for parallel feature computation.

Provides prepare_worker_data() which encapsulates the common setup logic
needed by both score_candidates() (live scoring) and compute_features_only()
(feature caching). This eliminates duplication that previously caused bugs
when new worker_data keys were added to one path but not the other.

Also provides compute_features_parallel() which encapsulates the common
ProcessPoolExecutor dispatch pattern shared by score_candidates() and
compute_features_only().
"""

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np
import shapely as shapely_mod

from ..config import (
    DEFAULT_SNAP_TOLERANCE_M,
    DEFAULT_TOPOLOGY_FEATURES,
    OVERTURE_ANCHOR_TOLERANCE_M,
    PHYSICAL_OVERLAP_FLOOR_M,
    PHYSICAL_OVERLAP_MIN_M,
    default_worker_count,
)
from ..features.alignment import compute_alignment_batch
from ..features.compute import precompute_graphlet_features
from ..features.relational import build_sibling_search_context
from ..features.spatial_context import (
    SpatialContextIndex,
    build_overture_connector_spatial_index,
    compute_aligned_endpoint_features_batch,
    compute_all_topology,
    find_overture_connectors_for_targets,
    sample_topology_batch,
)

logger = logging.getLogger(__name__)


def rebuild_connector_indices(
    worker_data: dict,
    reference_geoms_by_id: dict[str, Any],
    target_geoms_by_id: dict[str, Any],
    tolerance_m: float = OVERTURE_ANCHOR_TOLERANCE_M,
) -> None:
    """Rebuild all indices derived from graphlet data (in-place).

    Must be called after overriding worker_data["ref_graphlet_data"] or
    worker_data["target_graphlet_data"] to keep derived indices consistent.

    Rebuilds: target_overture_connectors.
    """
    ref_graphlet_data = worker_data["ref_graphlet_data"]
    _, ref_s2c, _, _ = ref_graphlet_data

    target_overture_connectors: dict[str, list[tuple[float, int]]] = {}

    # Build Overture connector spatial index and find connectors near targets
    connector_index = build_overture_connector_spatial_index(ref_s2c, reference_geoms_by_id)
    if connector_index is not None:
        target_overture_connectors = find_overture_connectors_for_targets(
            target_geoms_by_id, connector_index, tolerance_m=tolerance_m
        )

    worker_data["target_overture_connectors"] = target_overture_connectors


class WorkerDataResult(NamedTuple):
    """Result of preparing worker data for parallel feature computation."""

    worker_data: dict[str, Any]
    alignments: dict
    unique_ref_indices: set[int]
    unique_target_indices: set[int]
    candidates: list
    """Candidates after filtering (may be fewer than input if overlap filter is on)."""


def _extract_column_array(gdf: gpd.GeoDataFrame, column: str, fallback_len: int) -> np.ndarray:
    """Extract a column as numpy array, or return array of Nones if missing."""
    if column in gdf.columns:
        return gdf[column].to_numpy()
    return np.full(fallback_len, None, dtype=object)


def _extract_lr_column(gdf: gpd.GeoDataFrame, column: str) -> np.ndarray | None:
    """Extract a linear-referenced column as numpy array, or None if missing."""
    if column in gdf.columns:
        return gdf[column].to_numpy()
    return None


def prepare_worker_data(
    candidates: list,
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    ref_name_column: str = "names",
    target_name_column: str = "names",
    ref_class_column: str = "class",
    target_class_column: str = "class",
    ref_subclass_column: str = "subclass",
    target_subclass_column: str = "subclass",
    n_jobs: int = -1,
    ref_geoms: np.ndarray | None = None,
    target_geoms: np.ndarray | None = None,
    filter_physical_overlap: bool = True,
) -> WorkerDataResult:
    """Prepare worker_data dict for parallel feature computation.

    Encapsulates all shared setup: numpy extraction, topology, graphlets,
    alignments, sibling contexts, endpoint features, and worker_data assembly.

    Both score_candidates() and compute_features_only() delegate to this
    function, ensuring consistent worker_data structure.

    Args:
        candidates: List of CandidatePair objects (already filtered if needed)
        reference: Reference GeoDataFrame (MUST be in projected/metric CRS)
        target: Target GeoDataFrame (MUST be in projected/metric CRS)
        ref_id_column: Column name for reference IDs
        target_id_column: Column name for target IDs
        ref_name_column: Column name for reference names
        target_name_column: Column name for target names
        ref_class_column: Column name for reference road class
        target_class_column: Column name for target road class
        ref_subclass_column: Column name for reference subclass
        target_subclass_column: Column name for target subclass
        n_jobs: Number of parallel jobs for alignment (-1 for all cores)
        ref_geoms: Pre-extracted reference geometry array (avoids re-conversion
            if caller already materialized it, e.g., for overlap filtering)
        target_geoms: Pre-extracted target geometry array (same optimization)
        filter_physical_overlap: If True (default), remove candidates where
            the reference geometry within a PHYSICAL_OVERLAP_MIN_M buffer corridor
            around the target is shorter than an adaptive threshold based on
            the shorter segment's length (scales down for short segments).

    Returns:
        WorkerDataResult with worker_data dict, side-products, and filtered candidates
    """
    t_total = time.perf_counter()

    # --- Step 1: Extract numpy arrays from GeoDataFrames ---
    t0 = time.perf_counter()
    if ref_geoms is None:
        ref_geoms = reference.geometry.to_numpy()
    if target_geoms is None:
        target_geoms = target.geometry.to_numpy()

    ref_names = _extract_column_array(reference, ref_name_column, len(reference))
    target_names = _extract_column_array(target, target_name_column, len(target))
    ref_classes = _extract_column_array(reference, ref_class_column, len(reference))
    target_classes = _extract_column_array(target, target_class_column, len(target))
    ref_subclasses = _extract_column_array(reference, ref_subclass_column, len(reference))
    target_subclasses = _extract_column_array(target, target_subclass_column, len(target))

    # Linear-referenced attributes (None when column absent)
    ref_names_lr = _extract_lr_column(reference, "names_lr")
    target_names_lr = _extract_lr_column(target, "names_lr")
    ref_oneway_lr = _extract_lr_column(reference, "oneway_lr")
    target_oneway_lr = _extract_lr_column(target, "oneway_lr")
    ref_speed_limit_kph_lr = _extract_lr_column(reference, "speed_limit_kph_lr")
    target_speed_limit_kph_lr = _extract_lr_column(target, "speed_limit_kph_lr")

    ref_ids = reference[ref_id_column].to_numpy()
    target_ids = target[target_id_column].to_numpy()

    logger.debug(f"[TIMING] data_extraction: {time.perf_counter() - t0:.2f}s")

    # --- Step 1b: Filter by physical overlap (adaptive threshold) ---
    # Measures the length of the ref geometry falling within a buffer corridor
    # around the target. Threshold adapts to segment length so short segments
    # (sidewalks, footpaths) aren't penalized: max(1m, min(5m, 50% of shorter)).
    # Applied here so both ML scoring and labeling paths get the same filter.
    if filter_physical_overlap and candidates:
        t0 = time.perf_counter()
        ref_geom_arr = np.array([ref_geoms[c.ref_idx] for c in candidates], dtype=object)
        target_geom_arr = np.array([target_geoms[c.target_idx] for c in candidates], dtype=object)

        # Adaptive threshold: scale down for short segments so sidewalks/footpaths
        # aren't penalized by requiring 5m of overlap on a 3m segment.
        ref_lengths = shapely_mod.length(ref_geom_arr)
        target_lengths = shapely_mod.length(target_geom_arr)
        shorter_lengths = np.minimum(ref_lengths, target_lengths)
        thresholds = np.maximum(
            PHYSICAL_OVERLAP_FLOOR_M,
            np.minimum(PHYSICAL_OVERLAP_MIN_M, shorter_lengths * 0.5),
        )

        # Buffer corridor stays at fixed width for proximity check
        target_buffers = shapely_mod.buffer(target_geom_arr, PHYSICAL_OVERLAP_MIN_M)
        intersections = shapely_mod.intersection(ref_geom_arr, target_buffers)
        overlap_lengths = shapely_mod.length(intersections)

        mask = overlap_lengths >= thresholds
        filtered_candidates = [c for c, keep in zip(candidates, mask) if keep]

        n_filtered = len(candidates) - len(filtered_candidates)
        if n_filtered > 0:
            logger.info(
                f"Physical overlap filter: removed {n_filtered} candidates "
                f"(adaptive threshold {PHYSICAL_OVERLAP_FLOOR_M}-{PHYSICAL_OVERLAP_MIN_M}m), "
                f"{len(filtered_candidates)} remaining"
            )
        candidates = filtered_candidates
        logger.debug(f"[TIMING] physical_overlap_filter: {time.perf_counter() - t0:.2f}s")

        if not candidates:
            logger.info("All candidates filtered by physical overlap threshold")
            return WorkerDataResult(
                worker_data={},
                alignments={},
                unique_ref_indices=set(),
                unique_target_indices=set(),
                candidates=[],
            )

    # --- Step 2: Get unique indices, filter to candidate segments ---
    t0 = time.perf_counter()
    unique_target_indices = set(cand.target_idx for cand in candidates)
    unique_ref_indices = set(cand.ref_idx for cand in candidates)

    sorted_target_indices = sorted(unique_target_indices)
    target_candidates_only = target.iloc[sorted_target_indices].reset_index(drop=True)
    original_to_filtered = {orig: filt for filt, orig in enumerate(sorted_target_indices)}

    logger.info(
        f"Filtered target to {len(target_candidates_only)} candidate segments "
        f"(from {len(target)} total)"
    )
    logger.debug(f"[TIMING] segment_filtering: {time.perf_counter() - t0:.2f}s")

    # --- Step 3: Build spatial index for endpoint features ---
    logger.info("Building spatial index for endpoint features...")
    t0 = time.perf_counter()
    target_index = SpatialContextIndex()
    target_index.build_from_gdf(target_candidates_only, id_column=target_id_column)
    logger.debug(f"[TIMING] spatial_index_build: {time.perf_counter() - t0:.2f}s")

    # --- Step 4: Compute topology features ---
    unique_target_ids = {str(target_ids[idx]) for idx in unique_target_indices}
    unique_ref_ids = {str(ref_ids[idx]) for idx in unique_ref_indices}

    logger.info(
        f"Computing topology features for {len(unique_target_ids)} target "
        f"and {len(unique_ref_ids)} reference segments (batch)..."
    )

    ref_has_connectors = "connectors" in reference.columns

    t0 = time.perf_counter()
    # Target and ref topology are independent — compute in parallel threads.
    # Both use STRtree (C code, releases GIL), so threading gives real speedup.
    with ThreadPoolExecutor(max_workers=2) as topo_pool:
        # Target: always use geometry inference with spatial index for synthetic connectors
        target_topo_future = topo_pool.submit(
            compute_all_topology,
            target,
            id_column=target_id_column,
            tolerance_m=DEFAULT_SNAP_TOLERANCE_M,
            ids_to_compute=unique_target_ids,
            return_spatial_index=True,
        )
        ref_topo_future = topo_pool.submit(
            compute_all_topology,
            reference,
            id_column=ref_id_column,
            tolerance_m=DEFAULT_SNAP_TOLERANCE_M,
            ids_to_compute=unique_ref_ids,
            connectors_column="connectors" if ref_has_connectors else None,
        )
        target_topo_result = target_topo_future.result()
        ref_topology_by_id = ref_topo_future.result()

    target_topology_by_id, target_topo_spatial_index = target_topo_result

    # Map topology from segment IDs to DataFrame indices
    target_topology_full = {}
    for target_idx in unique_target_indices:
        seg_id = str(target_ids[target_idx])
        target_topology_full[target_idx] = target_topology_by_id.get(
            seg_id, DEFAULT_TOPOLOGY_FEATURES.copy()
        )

    ref_topology_full = {}
    for ref_idx in unique_ref_indices:
        seg_id = str(ref_ids[ref_idx])
        ref_topology_full[ref_idx] = ref_topology_by_id.get(
            seg_id, DEFAULT_TOPOLOGY_FEATURES.copy()
        )
    logger.debug(f"[TIMING] topology_computation: {time.perf_counter() - t0:.2f}s")

    # --- Step 4b: Compute synthetic topology connectors for target segments ---
    # Sample topology degrees at 50m intervals along each target candidate,
    # using the full-network spatial index. This produces synthetic connectors
    # in the same format as Overture connectors, enabling alignment-aware
    # topology for target segments. Uses batch query for single STRtree call.
    t0 = time.perf_counter()

    # Project target geometries if needed (spatial index is in projected CRS)
    work_target = target
    if target.crs is not None and target.crs.is_geographic:
        work_target = target.to_crs(target.estimate_utm_crs())

    batch_geoms = [work_target.geometry.iloc[idx] for idx in unique_target_indices]
    batch_ids = [str(target_ids[idx]) for idx in unique_target_indices]
    target_topo_connectors, target_topo_node_features = sample_topology_batch(
        batch_geoms, batch_ids, target_topo_spatial_index
    )

    logger.info(f"Sampled topology connectors for {len(target_topo_connectors)} target segments")
    logger.debug(f"[TIMING] target_topo_connectors: {time.perf_counter() - t0:.2f}s")

    # --- Step 4c: Build target geometry lookup (for later spatial connector matching) ---
    t0 = time.perf_counter()
    geoms_by_target_id = {
        str(target_ids[idx]): work_target.geometry.iloc[idx] for idx in unique_target_indices
    }
    logger.debug(f"[TIMING] target_geoms_by_id: {time.perf_counter() - t0:.2f}s")

    # --- Step 6: Compute graphlet features ---
    # IMPORTANT: Build the graphlet/clustering graphs on the FULL ref and target
    # networks, not the candidate-only subsets. Restricting the graph to segments
    # that happen to be candidates systematically deflates node degrees (a hub
    # collapses to degree 1 when its other spokes aren't candidates) and destroys
    # the triangles/squares/clustering these features are meant to measure. The
    # graph build is a once-per-dataset cost; on near-planar road networks it is
    # cheap relative to the per-pair feature computation. Measured worst case:
    # us_philadelphia_sidewalks target (204,760 segments, no connectors column,
    # so it takes the build_inferred_connector_graph STRtree path) builds in
    # ~16s; its 189K-segment Overture ref (explicit connectors) in ~1.7s.
    logger.info(
        f"Computing graphlet features on full networks ({len(reference)} reference "
        f"and {len(target)} target segments)..."
    )

    t0 = time.perf_counter()
    # Ref and target graphlets are independent — compute in parallel threads.
    with ThreadPoolExecutor(max_workers=2) as graphlet_pool:
        ref_graphlet_future = graphlet_pool.submit(
            precompute_graphlet_features,
            reference,
            id_column=ref_id_column,
            tolerance_m=DEFAULT_SNAP_TOLERANCE_M,
            connectors_column="connectors" if ref_has_connectors else None,
        )
        target_graphlet_future = graphlet_pool.submit(
            precompute_graphlet_features,
            target,
            id_column=target_id_column,
            tolerance_m=DEFAULT_SNAP_TOLERANCE_M,
        )
        ref_graphlet_data = ref_graphlet_future.result()
        target_graphlet_data = target_graphlet_future.result()
    logger.debug(f"[TIMING] graphlet_both: {time.perf_counter() - t0:.2f}s")

    # --- Step 6a/6b: Build connector indices (Overture spatial anchoring + reverse maps) ---
    # Use Overture connectors as spatial anchors: find which Overture connectors
    # are near each target segment. This puts both sides in the same ID space
    # for direct Jaccard comparison of shared junctions.
    t0 = time.perf_counter()
    geoms_by_ref_id = {
        str(ref_ids[idx]): reference.geometry.iloc[idx] for idx in unique_ref_indices
    }
    # Assemble a temporary worker_data with graphlet data for rebuild
    _wd_temp: dict[str, Any] = {"ref_graphlet_data": ref_graphlet_data}
    rebuild_connector_indices(_wd_temp, geoms_by_ref_id, geoms_by_target_id)
    target_overture_connectors = _wd_temp["target_overture_connectors"]
    if target_overture_connectors:
        logger.info(
            f"Matched Overture connectors to {len(target_overture_connectors)} target segments"
        )
    logger.debug(f"[TIMING] connector_indices: {time.perf_counter() - t0:.2f}s")

    # --- Step 7: Compute linestring alignments ---
    logger.info("Computing linestring alignments...")
    t0 = time.perf_counter()
    alignments = compute_alignment_batch(candidates, ref_geoms, target_geoms, n_jobs=n_jobs)
    logger.debug(f"[TIMING] alignment_batch: {time.perf_counter() - t0:.2f}s")

    # --- Step 8: Build sibling search contexts ---
    # IMPORTANT: Use ALL segments, not just candidates — a parallel sibling
    # might not have any candidate matches in the target dataset
    logger.info("Building sibling search contexts...")
    t0 = time.perf_counter()
    # Ref and target sibling contexts are independent — compute in parallel threads.
    # STRtree construction (C code) releases the GIL.
    with ThreadPoolExecutor(max_workers=2) as sibling_pool:
        ref_sibling_future = sibling_pool.submit(
            build_sibling_search_context,
            geometries=list(reference.geometry),
            segment_ids=[str(sid) for sid in reference[ref_id_column]],
            names=list(reference.get(ref_name_column, [None] * len(reference))),
            classes=list(reference.get(ref_class_column, [None] * len(reference))),
        )
        target_sibling_future = sibling_pool.submit(
            build_sibling_search_context,
            geometries=list(target.geometry),
            segment_ids=[str(sid) for sid in target[target_id_column]],
            names=list(target.get(target_name_column, [None] * len(target))),
            classes=list(target.get(target_class_column, [None] * len(target))),
        )
        ref_sibling_context = ref_sibling_future.result()
        target_sibling_context = target_sibling_future.result()
    logger.debug(f"[TIMING] sibling_context_both: {time.perf_counter() - t0:.2f}s")

    # --- Step 9: Compute aligned endpoint features ---
    _, target_seg_to_connectors_ep, _, _ = (
        target_graphlet_data if target_graphlet_data else (None, None, None, None)
    )
    aligned_endpoint_features = {}
    if alignments:
        logger.info(f"Computing aligned endpoint features for {len(alignments)} pairs...")
        t0 = time.perf_counter()
        aligned_endpoint_features = compute_aligned_endpoint_features_batch(
            alignments=alignments,
            target_geoms=target_geoms,
            target_ids=target_ids,
            target_index=target_index,
            original_to_filtered=original_to_filtered,
            seg_to_connectors=target_seg_to_connectors_ep,
        )
        logger.debug(f"[TIMING] aligned_endpoint_features: {time.perf_counter() - t0:.2f}s")
        logger.info(
            f"Computed aligned endpoint features for {len(aligned_endpoint_features)} pairs"
        )

    # --- Step 10: Assemble worker_data ---
    worker_data = {
        "ref_geoms_full": ref_geoms,
        "target_geoms_full": target_geoms,
        "ref_names": ref_names,
        "target_names": target_names,
        "ref_classes": ref_classes,
        "target_classes": target_classes,
        "ref_subclasses": ref_subclasses,
        "target_subclasses": target_subclasses,
        "ref_names_lr": ref_names_lr,
        "target_names_lr": target_names_lr,
        "ref_oneway_lr": ref_oneway_lr,
        "target_oneway_lr": target_oneway_lr,
        "ref_speed_limit_kph_lr": ref_speed_limit_kph_lr,
        "target_speed_limit_kph_lr": target_speed_limit_kph_lr,
        "ref_ids": ref_ids,
        "target_ids": target_ids,
        "aligned_endpoint_features": aligned_endpoint_features,
        "ref_topology_full": ref_topology_full,
        "target_topology_full": target_topology_full,
        "target_topo_connectors": target_topo_connectors,
        "target_topo_node_features": target_topo_node_features,
        "target_overture_connectors": target_overture_connectors,
        "ref_graphlet_data": ref_graphlet_data,
        "target_graphlet_data": target_graphlet_data,
        "ref_sibling_context_full": ref_sibling_context,
        "target_sibling_context_full": target_sibling_context,
        "alignments": alignments,
    }

    logger.info(f"Worker data prepared in {time.perf_counter() - t_total:.1f}s")

    return WorkerDataResult(
        worker_data=worker_data,
        alignments=alignments,
        unique_ref_indices=unique_ref_indices,
        unique_target_indices=unique_target_indices,
        candidates=candidates,
    )


class ParallelFeatureResult(NamedTuple):
    """Result of parallel feature computation."""

    features_list: list[dict | None]
    """Per-candidate feature dicts (None for rejected pairs), in candidate order."""
    wall_clock_seconds: float
    """Wall clock time for the parallel phase."""
    error_aggregator: Any
    """ErrorAggregator with cross-worker error stats."""


def compute_features_parallel(
    candidates: list,
    worker_data: dict,
    n_jobs: int = -1,
    sort_for_locality: bool = False,
) -> ParallelFeatureResult:
    """Run parallel feature computation on candidate pairs.

    Shared by MLMatcher.score_candidates() and data_loader.compute_features_only().

    Encapsulates: chunk splitting, ProcessPoolExecutor setup, _init_worker,
    _compute_feature_chunk mapping, progress logging, error aggregation,
    and result collection.

    Args:
        candidates: List of CandidatePair objects
        worker_data: Dict from prepare_worker_data()
        n_jobs: Number of parallel jobs (-1 for all cores)
        sort_for_locality: If True, sort work items by ref_idx for buffer cache
            locality (used by score_candidates). Results are reordered back to
            original candidate order.

    Returns:
        ParallelFeatureResult with features_list in original candidate order,
        wall clock time, and error aggregator.
    """
    from ..errors import ErrorAggregator
    from ..matching.ml import _compute_feature_chunk, _init_worker

    if n_jobs == -1:
        n_workers = default_worker_count()
    else:
        n_workers = max(1, n_jobs)

    n_candidates = len(candidates)
    logger.info(f"Computing features for {n_candidates} candidates using {n_workers} processes...")

    work_items = [(cand.ref_idx, cand.target_idx) for cand in candidates]

    # Optional spatial locality sorting for better buffer cache hit rates
    original_indices = None
    if sort_for_locality:
        work_items_with_idx = [
            (cand.ref_idx, cand.target_idx, i) for i, cand in enumerate(candidates)
        ]
        work_items_with_idx.sort(key=lambda x: (x[0], x[1]))
        work_items = [(item[0], item[1]) for item in work_items_with_idx]
        original_indices = [item[2] for item in work_items_with_idx]

    chunk_size = max(100, min(1000, n_candidates // (n_workers * 10)))
    chunks = [work_items[i : i + chunk_size] for i in range(0, len(work_items), chunk_size)]
    features_list: list[dict | None] = []

    total_errors = ErrorAggregator()

    logger.info(
        f"Starting parallel feature computation (chunk_size={chunk_size}, {len(chunks)} chunks)..."
    )
    t0 = time.perf_counter()

    # On Linux (fork), set worker_data as a module-level global before forking.
    # Child processes inherit the parent's memory via copy-on-write, avoiding
    # the O(N_workers * pickle_time) cost of serializing multi-GB worker_data
    # (STRtrees, geometry arrays, etc.) to each worker.
    use_fork_shortcut = multiprocessing.get_start_method(allow_none=True) == "fork"
    if use_fork_shortcut:
        from ..matching import ml as _ml_module

        _ml_module._worker_data = worker_data
        executor_kwargs: dict[str, Any] = {"max_workers": n_workers}
        logger.debug("Using fork shortcut: worker_data set via module global (COW)")
    else:
        executor_kwargs = {
            "max_workers": n_workers,
            "initializer": _init_worker,
            "initargs": (worker_data,),
        }

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        for chunk_results, chunk_errors in executor.map(_compute_feature_chunk, chunks):
            features_list.extend(chunk_results)
            total_errors.merge_serialized(chunk_errors)
            processed = len(features_list)
            pct = int(processed / len(work_items) * 100)
            logger.info(f"Feature computation: {processed:,}/{len(work_items):,} ({pct}%)")

    wall_clock = time.perf_counter() - t0

    # Reorder back to original candidate order if we sorted for locality
    if original_indices is not None:
        features_list_reordered: list[dict | None] = [None] * len(features_list)
        for sorted_idx, orig_idx in enumerate(original_indices):
            features_list_reordered[orig_idx] = features_list[sorted_idx]
        features_list = features_list_reordered

    return ParallelFeatureResult(
        features_list=features_list,
        wall_clock_seconds=wall_clock,
        error_aggregator=total_errors,
    )
