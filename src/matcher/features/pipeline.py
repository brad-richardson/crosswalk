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
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np

from ..config import DEFAULT_TOPOLOGY_FEATURES, default_worker_count
from ..features.alignment import compute_alignment_batch
from ..features.compute import precompute_graphlet_features
from ..features.relational import build_sibling_search_context
from ..features.spatial_context import (
    SpatialContextIndex,
    compute_aligned_endpoint_features_batch,
    compute_all_topology,
)

logger = logging.getLogger(__name__)


class WorkerDataResult(NamedTuple):
    """Result of preparing worker data for parallel feature computation."""

    worker_data: dict[str, Any]
    alignments: dict
    unique_ref_indices: set[int]
    unique_target_indices: set[int]


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

    Returns:
        WorkerDataResult with worker_data dict and side-products
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
    target_has_connectors = "connectors" in target.columns

    t0 = time.perf_counter()
    target_topology_by_id = compute_all_topology(
        target,
        id_column=target_id_column,
        tolerance_m=5.0,
        ids_to_compute=unique_target_ids,
        connectors_column="connectors" if target_has_connectors else None,
    )
    ref_topology_by_id = compute_all_topology(
        reference,
        id_column=ref_id_column,
        tolerance_m=5.0,
        ids_to_compute=unique_ref_ids,
        connectors_column="connectors" if ref_has_connectors else None,
    )

    # Map topology from segment IDs to DataFrame indices
    target_topology_features = {}
    for target_idx in unique_target_indices:
        seg_id = str(target_ids[target_idx])
        target_topology_features[target_idx] = target_topology_by_id.get(
            seg_id, DEFAULT_TOPOLOGY_FEATURES.copy()
        )

    ref_topology_features = {}
    for ref_idx in unique_ref_indices:
        seg_id = str(ref_ids[ref_idx])
        ref_topology_features[ref_idx] = ref_topology_by_id.get(
            seg_id, DEFAULT_TOPOLOGY_FEATURES.copy()
        )
    logger.debug(f"[TIMING] topology_computation: {time.perf_counter() - t0:.2f}s")

    # --- Step 5: Filter reference to candidate-only segments ---
    sorted_ref_indices = sorted(unique_ref_indices)
    ref_candidates_only = reference.iloc[sorted_ref_indices].reset_index(drop=True)

    # --- Step 6: Compute graphlet features ---
    logger.info(
        f"Computing graphlet features for {len(ref_candidates_only)} reference "
        f"and {len(target_candidates_only)} target segments..."
    )

    t0 = time.perf_counter()
    ref_graphlet_data = precompute_graphlet_features(
        ref_candidates_only,
        id_column=ref_id_column,
        tolerance_m=5.0,
        connectors_column="connectors" if ref_has_connectors else None,
    )
    logger.debug(f"[TIMING] graphlet_ref: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    target_graphlet_data = precompute_graphlet_features(
        target_candidates_only, id_column=target_id_column, tolerance_m=5.0
    )
    logger.debug(f"[TIMING] graphlet_target: {time.perf_counter() - t0:.2f}s")

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
    ref_sibling_context = build_sibling_search_context(
        geometries=list(reference.geometry),
        segment_ids=[str(sid) for sid in reference[ref_id_column]],
        names=list(reference.get(ref_name_column, [None] * len(reference))),
        classes=list(reference.get(ref_class_column, [None] * len(reference))),
    )
    logger.debug(f"[TIMING] sibling_context_ref: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    target_sibling_context = build_sibling_search_context(
        geometries=list(target.geometry),
        segment_ids=[str(sid) for sid in target[target_id_column]],
        names=list(target.get(target_name_column, [None] * len(target))),
        classes=list(target.get(target_class_column, [None] * len(target))),
    )
    logger.debug(f"[TIMING] sibling_context_target: {time.perf_counter() - t0:.2f}s")

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
        "ref_geoms": ref_geoms,
        "target_geoms": target_geoms,
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
        "ref_topology": ref_topology_features,
        "target_topology": target_topology_features,
        "ref_graphlet_data": ref_graphlet_data,
        "target_graphlet_data": target_graphlet_data,
        "ref_sibling_context": ref_sibling_context,
        "target_sibling_context": target_sibling_context,
        "alignments": alignments,
    }

    logger.info(f"Worker data prepared in {time.perf_counter() - t_total:.1f}s")

    return WorkerDataResult(
        worker_data=worker_data,
        alignments=alignments,
        unique_ref_indices=unique_ref_indices,
        unique_target_indices=unique_target_indices,
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

    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(worker_data,)
    ) as executor:
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
