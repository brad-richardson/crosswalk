"""Machine learning-based matcher using gradient boosted trees.

This module provides XGBoost-based matching trained on labeled data.
The model learns to classify road segment pairs as match/no_match
based on geometric and semantic features.

Training Data Format:
--------------------
Uses labels from Hive-partitioned CSVs in labels/ which contains:
- gers_id: Overture reference segment ID (GERS ID)
- target_id: Target segment identifier
- label: Human label (match, no_match, unsure)
- Feature columns: hausdorff_distance, buffer_iou, etc.

Model Architecture:
------------------
- XGBoost classifier with binary (match vs no_match) or multiclass output
- Features: Normalized geometric + semantic scores (same as rule-based)
- Handles class imbalance via scale_pos_weight or class_weight
"""

import time
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ..config import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    MAX_DISTANCE_METERS,
    METRIC_AVERAGE,
    SEMANTIC_FEATURES,
    default_worker_count,
)
from ..utils.crs import validate_projected_crs
from ..utils.linear_ref import extract_lr_value
from .types import MatchDecision, MatchResult

# Default XGBoost hyperparameters (F1-optimized via Optuna tuning).
# Updated by scripts/tune_model.py; used by MLMatcher.train() and scripts/ablation_study.py.
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 834,
    "max_depth": 9,
    "learning_rate": 0.022511981234250394,
    "min_child_weight": 9,
    "subsample": 0.7392909078558574,
    "colsample_bytree": 0.9537926227035262,
    "gamma": 0.38078555476202813,
    "reg_alpha": 0.19508694558590667,
    "reg_lambda": 4.990926294038142,
    "max_bin": 248,
    "tree_method": "hist",
}


class LRAttributes(NamedTuple):
    """Attributes extracted from linear-referenced data for a candidate pair."""

    ref_class: str | None
    target_class: str | None
    ref_subclass: str | None
    target_subclass: str | None
    # Road properties (data fetched but features parked - see RESEARCH_GRAVEYARD.md)
    ref_oneway: str | None
    target_oneway: str | None
    ref_speed_limit_kph: int | None
    target_speed_limit_kph: int | None


# Module-level globals for multiprocessing worker data
_worker_data = None


def _init_worker(data):
    """Initialize worker process with shared data."""
    global _worker_data
    _worker_data = data


def _extract_lr_attributes_for_pair(
    ref_idx: int,
    target_idx: int,
    alignment,
    worker_data: dict,
) -> LRAttributes:
    """Extract aligned attributes from LR data for a candidate pair.

    Uses alignment fractions to extract majority-covering values from
    linear-referenced attributes. Falls back to flat values if LR data
    is not available.

    Args:
        ref_idx: Reference segment index
        target_idx: Target segment index
        alignment: Alignment result (may be None)
        worker_data: Worker data dict with LR columns

    Returns:
        LRAttributes named tuple with ref/target class, subclass,
        oneway, and speed_limit_kph values.
    """
    # Get alignment fractions
    if alignment is not None:
        ref_start = alignment.overture_start_frac
        ref_end = alignment.overture_end_frac
        target_start = alignment.dataset_start_frac
        target_end = alignment.dataset_end_frac
    else:
        ref_start, ref_end = 0.0, 1.0
        target_start, target_end = 0.0, 1.0

    # Classes and subclasses - use flat values for now (LR support can be added)
    ref_class = worker_data["ref_classes"][ref_idx]
    target_class = worker_data["target_classes"][target_idx]
    ref_subclass = worker_data["ref_subclasses"][ref_idx]
    target_subclass = worker_data["target_subclasses"][target_idx]

    # Extract one-way direction from LR data
    ref_oneway = _extract_lr_value_from_column(
        worker_data.get("ref_oneway_lr"), ref_idx, ref_start, ref_end
    )
    target_oneway = _extract_lr_value_from_column(
        worker_data.get("target_oneway_lr"), target_idx, target_start, target_end
    )

    # Extract speed limit from LR data
    ref_speed_limit_kph = _extract_lr_value_from_column(
        worker_data.get("ref_speed_limit_kph_lr"), ref_idx, ref_start, ref_end
    )
    target_speed_limit_kph = _extract_lr_value_from_column(
        worker_data.get("target_speed_limit_kph_lr"), target_idx, target_start, target_end
    )

    return LRAttributes(
        ref_class=ref_class,
        target_class=target_class,
        ref_subclass=ref_subclass,
        target_subclass=target_subclass,
        ref_oneway=ref_oneway,
        target_oneway=target_oneway,
        ref_speed_limit_kph=ref_speed_limit_kph,
        target_speed_limit_kph=target_speed_limit_kph,
    )


def _extract_lr_value_from_column(lr_column, idx: int, start_frac: float, end_frac: float):
    """Extract a single value from an LR column array for a given index and alignment range."""
    if lr_column is None:
        return None
    lr_data = lr_column[idx]
    return extract_lr_value(lr_data, start_frac, end_frac)


def _compute_single_feature(args):
    """Compute features for a single candidate pair (worker function).

    Thin wrapper around _compute_feature_chunk for single-pair callers.

    Returns a dict of features, None if pair is rejected (missing aligned endpoint
    features), or a dict with error defaults if computation fails.
    """
    results, _errors = _compute_feature_chunk([args])
    return results[0] if results else None


def _compute_feature_chunk(chunk):
    """Process a chunk of pairs with 3-pass batch architecture.

    Pass 1: Collect geometry pairs, extract sublines, gather per-pair data
    Pass 2: Batch geometric computation via compute_geometric_features_batch()
    Pass 3: Per-pair non-batchable features + assembly

    This eliminates per-pair Python dispatch for all batchable Shapely operations
    (hausdorff, buffer, centroid, length, overlap, heading delta).

    Args:
        chunk: List of (ref_idx, target_idx) tuples

    Returns:
        Tuple of (results, error_summary_dict) where:
        - results: List of feature dicts (or None for rejected pairs)
        - error_summary_dict: Serializable dict with error counts and samples
    """
    from ..errors import ErrorAggregator, ErrorPhase, ErrorSeverity
    from ..features.alignment import create_subline
    from ..features.compute import (
        _compute_intersection_overlap_features,
        _compute_non_geometric_features,
        _get_error_features,
        assemble_feature_dict,
        compute_graphlet_similarity,
        get_timing_stats,
        is_profiling_enabled,
        timed_section,
    )
    from ..features.geometric import compute_geometric_features_batch
    from ..features.relational import compute_perpendicular_offset_batch

    HIGH_COVERAGE_THRESHOLD = 0.995
    error_tracker = ErrorAggregator()

    # ---- Pass 1: Collect geometry pairs and per-pair data ----
    # Store per-pair data for valid pairs only
    pair_data = []  # list of dicts with per-pair data
    # Track result positions: results[i] = None for rejected, or filled in pass 3
    results = [None] * len(chunk)
    valid_indices = []  # indices into results for valid pairs

    for chunk_idx, (ref_idx, target_idx) in enumerate(chunk):
        pair_key = (ref_idx, target_idx)

        try:
            with timed_section("worker_overhead"):
                ref_geom_full = _worker_data["ref_geoms_full"][ref_idx]
                target_geom_full = _worker_data["target_geoms_full"][target_idx]
                alignment = _worker_data.get("alignments", {}).get(pair_key)

                endpoint_features = _worker_data.get("aligned_endpoint_features", {}).get(pair_key)
                if endpoint_features is None:
                    # results[chunk_idx] stays None
                    continue

                ref_topology_full = _worker_data.get("ref_topology_full", {}).get(ref_idx)
                target_topology_full = _worker_data.get("target_topology_full", {}).get(target_idx)
                ref_graphlet_data = _worker_data.get("ref_graphlet_data")
                target_graphlet_data = _worker_data.get("target_graphlet_data")
                ref_seg_id = str(_worker_data["ref_ids"][ref_idx])
                target_seg_id = str(_worker_data["target_ids"][target_idx])

            with timed_section("graphlet_similarity"):
                graphlet_features = compute_graphlet_similarity(
                    ref_seg_id,
                    target_seg_id,
                    ref_graphlet_data,
                    target_graphlet_data,
                    alignment,
                )

            # Subline extraction (same logic as compute_pair_features)
            with timed_section("subline_extraction"):
                if alignment is not None:
                    ref_coverage = alignment.overture_end_frac - alignment.overture_start_frac
                    target_coverage = alignment.dataset_end_frac - alignment.dataset_start_frac

                    if ref_coverage >= HIGH_COVERAGE_THRESHOLD:
                        ref_geom_aligned = ref_geom_full
                    else:
                        ref_subline = create_subline(
                            ref_geom_full,
                            alignment.overture_start_frac,
                            alignment.overture_end_frac,
                        )
                        ref_geom_aligned = ref_subline if ref_subline else ref_geom_full

                    if target_coverage >= HIGH_COVERAGE_THRESHOLD:
                        target_geom_aligned = target_geom_full
                    else:
                        target_subline = create_subline(
                            target_geom_full,
                            alignment.dataset_start_frac,
                            alignment.dataset_end_frac,
                        )
                        target_geom_aligned = target_subline if target_subline else target_geom_full
                else:
                    ref_geom_aligned = ref_geom_full
                    target_geom_aligned = target_geom_full

            # Extract aligned attributes from LR data (using alignment fractions)
            lr_attrs = _extract_lr_attributes_for_pair(ref_idx, target_idx, alignment, _worker_data)

            pair_data.append(
                {
                    "chunk_idx": chunk_idx,
                    "ref_idx": ref_idx,
                    "target_idx": target_idx,
                    "ref_geom_aligned": ref_geom_aligned,
                    "target_geom_aligned": target_geom_aligned,
                    "ref_class": lr_attrs.ref_class,
                    "target_class": lr_attrs.target_class,
                    "ref_subclass": lr_attrs.ref_subclass,
                    "target_subclass": lr_attrs.target_subclass,
                    "ref_oneway": lr_attrs.ref_oneway,
                    "target_oneway": lr_attrs.target_oneway,
                    "ref_speed_limit_kph": lr_attrs.ref_speed_limit_kph,
                    "target_speed_limit_kph": lr_attrs.target_speed_limit_kph,
                    "endpoint_features": endpoint_features,
                    "ref_topology_full": ref_topology_full,
                    "target_topology_full": target_topology_full,
                    "alignment": alignment,
                    "graphlet_features": graphlet_features,
                    "ref_graphlet_data": ref_graphlet_data,
                    "target_graphlet_data": target_graphlet_data,
                    "ref_seg_id": ref_seg_id,
                    "target_seg_id": target_seg_id,
                    "ref_names_raw": (
                        _worker_data["ref_names"][ref_idx] if "ref_names" in _worker_data else None
                    ),
                    "target_names_raw": (
                        _worker_data["target_names"][target_idx]
                        if "target_names" in _worker_data
                        else None
                    ),
                }
            )
            valid_indices.append(chunk_idx)

        except Exception as e:
            error_tracker.add_simple(
                ErrorPhase.PAIR_FEATURES,
                e,
                ErrorSeverity.WARNING,
                ref_idx=ref_idx,
                target_idx=target_idx,
            )
            error_features = _get_error_features(error=e, phase=ErrorPhase.PAIR_FEATURES.value)
            if is_profiling_enabled():
                get_timing_stats().reset()
            results[chunk_idx] = error_features

    if not pair_data:
        return (results, error_tracker.to_serializable())

    # ---- Pass 2: Batch geometric computation ----
    arr_a = np.array([pd["ref_geom_aligned"] for pd in pair_data], dtype=object)
    arr_b = np.array([pd["target_geom_aligned"] for pd in pair_data], dtype=object)

    try:
        with timed_section("batch_geometric"):
            batch_result = compute_geometric_features_batch(arr_a, arr_b)
    except Exception as e:
        # Track batch failure - affects all pairs in chunk
        for pd_item in pair_data:
            error_tracker.add_simple(
                ErrorPhase.BATCH_GEOMETRIC,
                e,
                ErrorSeverity.CRITICAL,
                ref_idx=pd_item["ref_idx"],
                target_idx=pd_item["target_idx"],
            )
        error_features = _get_error_features(error=e, phase="batch_geometric")
        if is_profiling_enabled():
            get_timing_stats().reset()
        for pd_item in pair_data:
            results[pd_item["chunk_idx"]] = error_features
        return (results, error_tracker.to_serializable())

    # ---- Pass 2.5: Batch perpendicular offset ----
    # Batch line_interpolate_point + distance across all pairs in the chunk
    # instead of calling per-pair in _compute_non_geometric_features.
    # arr_b = targets, arr_a = anchors (refs) — matches single-pair call order.
    try:
        with timed_section("perpendicular_offset"):
            batch_mean_offsets, batch_iqr_offsets, batch_p95_offsets = (
                compute_perpendicular_offset_batch(arr_b, arr_a)
            )
    except Exception as e:
        # Track batch failure - affects all pairs in chunk
        for pd_item in pair_data:
            error_tracker.add_simple(
                ErrorPhase.PERPENDICULAR_OFFSET,
                e,
                ErrorSeverity.CRITICAL,
                ref_idx=pd_item["ref_idx"],
                target_idx=pd_item["target_idx"],
            )
        error_features = _get_error_features(error=e, phase="perpendicular_offset")
        if is_profiling_enabled():
            get_timing_stats().reset()
        for pd_item in pair_data:
            results[pd_item["chunk_idx"]] = error_features
        return (results, error_tracker.to_serializable())

    # ---- Pass 3: Per-pair non-batchable features + assembly ----
    for i, pd_item in enumerate(pair_data):
        chunk_idx = pd_item["chunk_idx"]
        try:
            with timed_section("coord_extraction"):
                coords_aligned_ref = np.array(pd_item["ref_geom_aligned"].coords)
                coords_aligned_target = np.array(pd_item["target_geom_aligned"].coords)

            # Build a GeometricFeatures stub with batch values for _compute_non_geometric_features
            from ..features.geometric import GeometricFeatures

            geom_features = GeometricFeatures(
                hausdorff_distance=float(batch_result.hausdorff_distances[i]),
                mean_hausdorff_distance=0.0,  # Filled by _compute_non_geometric_features
                hausdorff_p95_distance=0.0,  # Filled by _compute_non_geometric_features
                buffer_iou_5m=float(batch_result.buffer_iou_5m[i]),
                buffer_iou_15m=float(batch_result.buffer_iou_15m[i]),
                heading_delta=float(batch_result.heading_deltas[i]),
                length_ratio=float(batch_result.length_ratios[i]),
                projection_distance=0.0,  # Filled by _compute_non_geometric_features
                centroid_distance=float(batch_result.centroid_distances[i]),
                overlap_ratio=float(batch_result.overlap_ratios[i]),
                collinear_gap_ratio=0.0,  # Filled by _compute_non_geometric_features
            )

            non_geom = _compute_non_geometric_features(
                ref_geom_aligned=pd_item["ref_geom_aligned"],
                target_geom_aligned=pd_item["target_geom_aligned"],
                coords_aligned_ref=coords_aligned_ref,
                coords_aligned_target=coords_aligned_target,
                ref_class=pd_item["ref_class"],
                target_class=pd_item["target_class"],
                ref_subclass=pd_item["ref_subclass"],
                target_subclass=pd_item["target_subclass"],
                # Note: ref_oneway, target_oneway, ref_speed_limit_kph, target_speed_limit_kph
                # are extracted but not passed - features parked (see RESEARCH_GRAVEYARD.md)
                endpoint_features=pd_item["endpoint_features"],
                ref_topology_full=pd_item["ref_topology_full"],
                target_topology_full=pd_item["target_topology_full"],
                alignment=pd_item["alignment"],
                graphlet_features=pd_item["graphlet_features"],
                ref_graphlet_data=pd_item["ref_graphlet_data"],
                target_graphlet_data=pd_item["target_graphlet_data"],
                ref_seg_id=pd_item["ref_seg_id"],
                target_seg_id=pd_item["target_seg_id"],
                geom_features=geom_features,
                precomputed_lateral_offset=(
                    batch_mean_offsets[i],
                    batch_iqr_offsets[i],
                    batch_p95_offsets[i],
                ),
                ref_sibling_context_full=_worker_data.get("ref_sibling_context_full"),
                target_sibling_context_full=_worker_data.get("target_sibling_context_full"),
                ref_names_raw=pd_item.get("ref_names_raw"),
                target_names_raw=pd_item.get("target_names_raw"),
            )

            # Aligned length: absolute overlap length in meters (uses full geometry)
            alignment = pd_item["alignment"]
            ref_length_full = _worker_data["ref_geoms_full"][pd_item["ref_idx"]].length
            if alignment is not None:
                aligned_length_m = ref_length_full * (
                    alignment.overture_end_frac - alignment.overture_start_frac
                )
            else:
                aligned_length_m = 0.0

            # Intersection overlap features (continuation, divergence) - uses full geometries
            intersection_overlap_feats = _compute_intersection_overlap_features(
                ref_geom_full=_worker_data["ref_geoms_full"][pd_item["ref_idx"]],
                target_geom_full=_worker_data["target_geoms_full"][pd_item["target_idx"]],
                alignment=alignment,
            )

            # Use full geometries for length_ratio (not sublines)
            target_length_full = _worker_data["target_geoms_full"][pd_item["target_idx"]].length
            length_ratio = (
                min(ref_length_full, target_length_full) / max(ref_length_full, target_length_full)
                if max(ref_length_full, target_length_full) > 0
                else 0.0
            )

            # Assemble via shared function (single source of truth for clamping)
            features = assemble_feature_dict(
                geom_features=geom_features,
                length_ratio=length_ratio,
                aligned_length_m=aligned_length_m,
                non_geom=non_geom,
                intersection_overlap_feats=intersection_overlap_feats,
            )
            features["_error"] = None
            results[chunk_idx] = features

        except Exception as e:
            error_tracker.add_simple(
                ErrorPhase.PAIR_FEATURES,
                e,
                ErrorSeverity.WARNING,
                ref_idx=pd_item["ref_idx"],
                target_idx=pd_item["target_idx"],
            )
            error_features = _get_error_features(error=e, phase=ErrorPhase.PAIR_FEATURES.value)
            if is_profiling_enabled():
                get_timing_stats().reset()
            results[chunk_idx] = error_features

    # Distribute batch timing evenly across pairs for profiling aggregation
    if is_profiling_enabled() and pair_data:
        stats = get_timing_stats()
        n_pairs = len(pair_data)
        for name, total in stats.totals.items():
            per_pair = total / n_pairs
            for pd_item in pair_data:
                feat = results[pd_item["chunk_idx"]]
                if feat is not None:
                    feat[f"_t_{name}"] = per_pair
        stats.reset()

    return (results, error_tracker.to_serializable())


def select_model_for_dataset(
    target_gdf,
    full_model_path: str | None = None,
    geom_only_model_path: str | None = None,
    name_column: str = "names",
    min_name_coverage: float = 0.5,
) -> str:
    """Select model based on dataset attributes.

    Automatically chooses between full model (with semantic features) and
    geometry-only model based on the target dataset's name coverage.

    Args:
        target_gdf: Target GeoDataFrame
        full_model_path: Path to full model with semantic features
        geom_only_model_path: Path to geometry-only model
        name_column: Column name for segment names
        min_name_coverage: Minimum fraction of rows with non-null names to use full model

    Returns:
        Path to selected model
    """
    from ..config import settings

    # Use configured paths if not explicitly provided
    if full_model_path is None:
        full_model_path = str(settings.model_path)
    if geom_only_model_path is None:
        geom_only_model_path = str(settings.model_geom_only_path)

    # Check for name column variations
    has_names = name_column in target_gdf.columns or "name" in target_gdf.columns
    name_col = name_column if name_column in target_gdf.columns else "name"

    if has_names:
        # Calculate effective name coverage (non-null AND non-empty strings)
        non_empty_mask = target_gdf[name_col].notna() & (
            target_gdf[name_col].astype(str).str.strip() != ""
        )
        effective_coverage = non_empty_mask.mean()
    else:
        effective_coverage = 0.0

    logger.info(f"Target dataset name coverage: {effective_coverage:.1%}")

    # Check if geometry-only model exists
    geom_only_exists = Path(geom_only_model_path).exists()

    if effective_coverage >= min_name_coverage:
        logger.info(
            f"Using full model (name coverage {effective_coverage:.1%} >= {min_name_coverage:.0%})"
        )
        return full_model_path
    elif geom_only_exists:
        logger.info(
            f"Using geometry-only model (name coverage {effective_coverage:.1%} < {min_name_coverage:.0%})"
        )
        return geom_only_model_path
    else:
        logger.warning(
            f"Geometry-only model not found at {geom_only_model_path}, falling back to full model"
        )
        return full_model_path


def create_segment_groups(df: pd.DataFrame) -> pd.Series:
    """Create group IDs for segment-aware train/test splitting.

    Uses Union-Find to ensure pairs sharing any segment are in the same group.
    This prevents data leakage where the model sees a segment during training
    and then evaluates on the same segment.

    Args:
        df: DataFrame with 'gers_id' and 'target_id' columns

    Returns:
        Series of group IDs (one per row in df)

    Raises:
        ValueError: If gers_id or target_id columns contain null values
    """
    from collections import defaultdict

    # Validate no null values in segment ID columns
    if df["gers_id"].isna().any() or df["target_id"].isna().any():
        raise ValueError("gers_id and target_id columns must not contain null values")

    # Union-Find implementation
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        # Find root iteratively
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Map segments to pairs (using DataFrame index)
    segment_to_pairs = defaultdict(list)
    for idx in df.index:
        row = df.loc[idx]
        segment_to_pairs[row["gers_id"]].append(idx)
        segment_to_pairs[row["target_id"]].append(idx)

    # Union pairs that share a segment
    for _segment, pair_idxs in segment_to_pairs.items():
        for i in range(1, len(pair_idxs)):
            union(pair_idxs[0], pair_idxs[i])

    # Return group for each row
    return pd.Series([find(idx) for idx in df.index], index=df.index)


def segment_aware_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    return_groups: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, pd.Series]:
    """Split data ensuring no segment appears in both train and test sets.

    Uses Union-Find to group pairs that share segments, then splits by group.
    This prevents data leakage where the model trains on a segment and then
    evaluates on the same segment in a different pair.

    Args:
        df: DataFrame with 'gers_id' and 'target_id' columns
        test_size: Fraction of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility
        return_groups: If True, also return the computed segment groups

    Returns:
        Tuple of (train_indices, test_indices) as numpy arrays.
        If return_groups=True, returns (train_indices, test_indices, groups).

    Raises:
        ValueError: If test_size is not in range [0.0, 1.0]
    """
    # Validate test_size
    if not 0.0 <= test_size <= 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

    # Handle empty DataFrame
    if len(df) == 0:
        empty_idx = np.array([], dtype=int)
        if return_groups:
            return empty_idx, empty_idx, pd.Series([], dtype=int)
        return empty_idx, empty_idx

    # Handle test_size=0.0 (no split, all training)
    if test_size == 0.0:
        train_idx = np.arange(len(df))
        test_idx = np.array([], dtype=int)
        if return_groups:
            groups = create_segment_groups(df)
            return train_idx, test_idx, groups
        return train_idx, test_idx

    groups = create_segment_groups(df)
    n_groups = groups.nunique()

    # Need at least 2 groups to split
    if n_groups < 2:
        logger.warning(
            f"Only {n_groups} segment group(s) found - cannot split. "
            "All pairs are transitively connected. Placing all in training set."
        )
        train_idx = np.arange(len(df))
        test_idx = np.array([], dtype=int)
        if return_groups:
            return train_idx, test_idx, groups
        return train_idx, test_idx

    # Use GroupShuffleSplit to split by group
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    logger.info(
        f"Segment-aware split: {len(train_idx)} train, {len(test_idx)} test "
        f"across {n_groups} groups"
    )

    if return_groups:
        return train_idx, test_idx, groups
    return train_idx, test_idx


class MLMatcher:
    """Machine learning-based matcher using gradient boosted trees."""

    def __init__(self, model_path: str | None = None, auto_select: bool = False):
        """Initialize the ML matcher.

        Args:
            model_path: Path to trained model (optional)
            auto_select: If True, defer model loading until score_candidates is called
                        so that model can be selected based on target dataset
        """
        self.model = None
        self.model_path = model_path
        self.feature_names = FEATURE_COLUMNS.copy()
        self.label_encoder = {"match": 1, "no_match": 0}
        self.label_decoder = {1: "match", 0: "no_match"}
        self.is_binary = True  # Track if model is binary or multiclass
        self.feature_version = None
        self._auto_select = auto_select

        if model_path and not auto_select:
            self.load_model(model_path)

    def load_model(self, path: str) -> None:
        """Load a trained model from disk.

        Args:
            path: Path to model file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data.get("feature_names", FEATURE_COLUMNS.copy())
        self.label_encoder = data.get("label_encoder", self.label_encoder)
        self.label_decoder = data.get("label_decoder", self.label_decoder)
        self.is_binary = data.get("is_binary", True)
        self.feature_version = data.get("feature_version")
        if self.feature_version is None:
            logger.warning(
                f"Model {path} has no feature_version (pre-versioning model). "
                f"Current code uses FEATURE_VERSION={FEATURE_VERSION}."
            )
        elif self.feature_version != FEATURE_VERSION:
            logger.warning(
                f"Model feature_version={self.feature_version} does not match "
                f"current code FEATURE_VERSION={FEATURE_VERSION}. "
                "Consider retraining the model."
            )
        logger.info(f"Loaded model from {path}")

    def save_model(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "is_binary": self.is_binary,
            "feature_version": self.feature_version,
        }
        joblib.dump(data, path)
        logger.info(f"Saved model to {path}")

    def train(
        self,
        labels_dir: str = "labels",
        binary: bool = True,
        test_size: float = 0.2,
        exclude_semantic: bool = False,
        exclude_datasets: list[str] | None = None,
        exclude_features: list[str] | None = None,
        agent_weight: float = 0.0,
        min_agent_confidence: float = 0.0,
        max_centroid_distance_m: float = 500.0,
        max_hausdorff_m: float = 1000.0,
        **kwargs,
    ) -> dict[str, Any]:
        """Train the model on labeled data.

        Args:
            labels_dir: Path to Hive-partitioned labels directory
            binary: If True (default), train binary classifier (match vs non-match)
                   If False, train multiclass (not recommended)
            test_size: Fraction of data to hold out for testing
            exclude_semantic: If True, exclude semantic features (name_*, class_similarity)
                             for training a geometry-only model
            exclude_datasets: List of dataset names to exclude from training
                             (for leave-one-out cross-validation)
            exclude_features: List of feature names to exclude from training
                             (for feature importance analysis)
            agent_weight: Weight for agent labels in training (0.0 = ignore, 1.0 = equal to human).
                         When > 0, agent labels are included with this sample weight.
            min_agent_confidence: Minimum confidence threshold for including agent labels.
                                 Only agent labels with confidence >= this value are included.
            max_centroid_distance_m: Max centroid distance for valid training pairs (meters).
                                    Pairs exceeding this are dropped as corrupted.
            max_hausdorff_m: Max Hausdorff distance for valid training pairs (meters).
                            Pairs exceeding this are dropped as corrupted.
            **kwargs: Additional XGBoost parameters

        Returns:
            Dictionary of training metrics
        """
        try:
            import xgboost as xgb
        except ImportError as err:
            raise ImportError(
                "XGBoost is required for ML training. "
                "Install it with: pip install 'matcher[ml]' or pip install xgboost"
            ) from err

        from ..labeling.label_store import LabelStore

        self.is_binary = binary

        # Set feature columns based on exclude_semantic
        if exclude_semantic:
            self.feature_names = [f for f in FEATURE_COLUMNS if f not in SEMANTIC_FEATURES]
            logger.info(
                f"Training geometry-only model with {len(self.feature_names)} features (excluding semantic)"
            )
        else:
            self.feature_names = FEATURE_COLUMNS.copy()

        # Remove explicitly excluded features
        if exclude_features:
            # Validate that all excluded features actually exist
            invalid_features = [f for f in exclude_features if f not in FEATURE_COLUMNS]
            if invalid_features:
                raise ValueError(
                    f"Invalid feature names in exclude_features: {invalid_features}. "
                    f"Valid features are: {FEATURE_COLUMNS[:5]}... ({len(FEATURE_COLUMNS)} total)"
                )
            before_count = len(self.feature_names)
            self.feature_names = [f for f in self.feature_names if f not in exclude_features]
            excluded_count = before_count - len(self.feature_names)
            logger.info(
                f"Excluding {excluded_count} features: {exclude_features} "
                f"({len(self.feature_names)} features remaining)"
            )

        # Load all partitions using LabelStore
        df = LabelStore.load_all(Path(labels_dir))
        logger.info(
            f"Loaded {len(df)} labeled pairs from {df['dataset'].nunique() if 'dataset' in df.columns else 1} datasets"
        )

        # Check feature_version consistency across labels
        if "feature_version" in df.columns:
            version_counts = df["feature_version"].value_counts(dropna=False)
            n_versions = len(version_counts)
            if n_versions > 1:
                logger.warning(
                    f"Labels contain {n_versions} different feature versions: "
                    f"{version_counts.to_dict()}"
                )
            n_current = (df["feature_version"] == FEATURE_VERSION).sum()
            n_total = len(df)
            logger.info(
                f"Label feature versions: {n_current}/{n_total} match current "
                f"FEATURE_VERSION={FEATURE_VERSION}"
            )
        else:
            logger.warning(
                "Labels have no feature_version column (pre-versioning labels). "
                f"Current code uses FEATURE_VERSION={FEATURE_VERSION}."
            )

        # Set feature_version for this trained model
        self.feature_version = FEATURE_VERSION

        # Exclude specified datasets (for leave-one-out evaluation)
        if exclude_datasets:
            before_count = len(df)
            df = df[~df["dataset"].isin(exclude_datasets)].copy()
            excluded_count = before_count - len(df)
            logger.info(f"Excluded {excluded_count} labels from datasets: {exclude_datasets}")
            logger.info(f"Training on {len(df)} labels from {df['dataset'].nunique()} datasets")

        # Filter to only valid labels (exclude unsure, skip, and any unexpected values)
        valid_labels = {"match", "no_match"}
        invalid_mask = ~df["label"].isin(valid_labels)
        if invalid_mask.any():
            invalid_labels = df.loc[invalid_mask, "label"].value_counts().to_dict()
            logger.warning(f"Filtering out invalid labels: {invalid_labels}")
        df = df[df["label"].isin(valid_labels)].copy()
        logger.info(f"After filtering to valid labels: {len(df)} pairs")

        # Validate feature plausibility (drop corrupted/stale pairs)
        df = self._validate_training_pairs(
            df,
            max_centroid_distance_m=max_centroid_distance_m,
            max_hausdorff_m=max_hausdorff_m,
        )

        # Initialize sample weights (human labels get weight 1.0)
        sample_weights = np.ones(len(df), dtype=np.float32)
        n_human = len(df)

        # Load and merge agent labels if agent_weight > 0
        if agent_weight > 0:
            from ..labeling.feature_store import FeatureStore

            logger.info(
                f"Loading agent labels with weight={agent_weight}, min_confidence={min_agent_confidence}"
            )

            # Load agent labels from normalized format
            agent_dir = Path(labels_dir) / "agent"
            if agent_dir.exists():
                agent_labels = LabelStore.load_agent_labels(agent_dir)
            else:
                agent_labels = pd.DataFrame()

            if len(agent_labels) > 0:
                # Filter by minimum confidence
                if "confidence" in agent_labels.columns and min_agent_confidence > 0:
                    before_count = len(agent_labels)
                    agent_labels = agent_labels[
                        agent_labels["confidence"] >= min_agent_confidence
                    ].copy()
                    logger.info(
                        f"Filtered agent labels by confidence >= {min_agent_confidence}: "
                        f"{before_count} -> {len(agent_labels)}"
                    )

                # Filter to valid labels
                agent_labels = agent_labels[agent_labels["label"].isin(valid_labels)].copy()

                # Exclude same datasets if specified
                if exclude_datasets and "dataset" in agent_labels.columns:
                    agent_labels = agent_labels[
                        ~agent_labels["dataset"].isin(exclude_datasets)
                    ].copy()

                if len(agent_labels) > 0:
                    # Load features for agent labels
                    features_dir = Path(labels_dir) / "features"
                    if features_dir.exists():
                        all_features = FeatureStore.load_all(features_dir)

                        if len(all_features) > 0:
                            # Join agent labels with features
                            agent_with_features = agent_labels.merge(
                                all_features,
                                on=["gers_id", "target_id", "dataset"],
                                how="inner",
                            )

                            if len(agent_with_features) > 0:
                                logger.info(
                                    f"Merged {len(agent_with_features)} agent labels with features "
                                    f"({len(agent_labels) - len(agent_with_features)} missing features)"
                                )

                                # Validate agent labels too
                                agent_with_features = self._validate_training_pairs(
                                    agent_with_features,
                                    max_centroid_distance_m=max_centroid_distance_m,
                                    max_hausdorff_m=max_hausdorff_m,
                                )

                                # Create agent sample weights
                                agent_weights = np.full(
                                    len(agent_with_features), agent_weight, dtype=np.float32
                                )

                                # Combine human and agent labels
                                df = pd.concat([df, agent_with_features], ignore_index=True)
                                sample_weights = np.concatenate([sample_weights, agent_weights])

                                logger.info(
                                    f"Training data: {n_human} human (weight=1.0) + "
                                    f"{len(agent_with_features)} agent (weight={agent_weight})"
                                )
                            else:
                                logger.warning(
                                    "No agent labels have features - skipping agent labels"
                                )
                        else:
                            logger.warning(
                                "No features found in feature store - skipping agent labels"
                            )
                    else:
                        logger.warning(
                            f"Features directory not found: {features_dir} - skipping agent labels"
                        )
                else:
                    logger.info("No valid agent labels after filtering")
            else:
                logger.info("No agent labels found")

        # Extract features (without imputation - we'll do that after split)
        X, y = self._extract_features_and_labels(df, binary=binary)

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")

        # Segment-aware train/test split to prevent data leakage
        # Also get groups for reuse in cross-validation
        train_idx, test_idx, groups = segment_aware_split(
            df, test_size=test_size, random_state=42, return_groups=True
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        weights_train = sample_weights[train_idx]

        # Verify labels have all expected features before computing medians
        # This catches bugs where new features are added to FEATURE_COLUMNS but
        # the labels (created with older code) don't have them
        # Build expected features after all exclusions (semantic + explicit)
        expected_features = (
            [f for f in FEATURE_COLUMNS if f not in SEMANTIC_FEATURES]
            if exclude_semantic
            else FEATURE_COLUMNS  # No .copy() needed since we filter below anyway
        )
        if exclude_features:
            expected_features = [f for f in expected_features if f not in exclude_features]
        missing_in_labels = set(expected_features) - set(self.feature_names)
        if missing_in_labels:
            raise ValueError(
                f"Labels are missing {len(missing_in_labels)} expected features: {sorted(missing_in_labels)}. "
                f"This usually means labels were created with an older version. "
                f"Run backfill to add missing features, or retrain with updated labels."
            )

        # Cap infinite values (XGBoost handles NaN natively but not inf)
        X_train = self._cap_infinities(X_train)
        X_test = self._cap_infinities(X_test)

        # Handle class imbalance
        if binary:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            logger.info(
                f"Class balance: {n_pos} positive, {n_neg} negative, scale={scale_pos_weight:.2f}"
            )
        else:
            scale_pos_weight = None

        default_params = {
            **DEFAULT_XGB_PARAMS,
            "objective": "binary:logistic" if binary else "multi:softprob",
            "eval_metric": "logloss" if binary else "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        if scale_pos_weight and binary:
            # Prefer natural balance or calculated scale_pos_weight
            default_params["scale_pos_weight"] = scale_pos_weight
        if not binary:
            default_params["num_class"] = len(self.label_encoder)

        # Override with user params
        params = {**default_params, **kwargs}

        # Train
        logger.info(f"Training XGBoost with params: {params}")
        self.model = xgb.XGBClassifier(**params)

        # Prepare sample_weight for training (only use if we have mixed weights)
        use_sample_weight = agent_weight > 0 and len(weights_train) > 0
        fit_kwargs = {}
        if use_sample_weight:
            fit_kwargs["sample_weight"] = weights_train
            logger.info(
                f"Using sample weights (min={weights_train.min():.2f}, max={weights_train.max():.2f})"
            )

        # Only use eval_set if we have test data
        if len(X_test) > 0:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
                **fit_kwargs,
            )
        else:
            self.model.fit(X_train, y_train, verbose=False, **fit_kwargs)

        # Results dict
        target_names = ["no_match", "match"]
        results = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_)),
        }

        # Evaluate on test set if we have one
        if len(X_test) > 0:
            y_pred = self.model.predict(X_test)

            # Cross-validation score with segment-aware folding.
            # XGBoost handles NaN natively — no imputation needed.
            n_groups = groups.nunique()
            if n_groups >= 2:
                n_splits = min(5, n_groups)
                gkf = GroupKFold(n_splits=n_splits)
                cv_scores = []
                for cv_train_idx, cv_test_idx in gkf.split(X, y, groups):
                    X_cv_train, X_cv_test = X[cv_train_idx], X[cv_test_idx]
                    y_cv_train, y_cv_test = y[cv_train_idx], y[cv_test_idx]

                    # Train and score this fold
                    cv_model = xgb.XGBClassifier(**params)
                    cv_model.fit(X_cv_train, y_cv_train)
                    y_cv_pred = cv_model.predict(X_cv_test)
                    fold_f1 = f1_score(
                        y_cv_test,
                        y_cv_pred,
                        average=METRIC_AVERAGE,
                        zero_division=0,
                    )
                    cv_scores.append(fold_f1)
                cv_scores = np.array(cv_scores)
            else:
                # Cannot do CV with < 2 groups
                logger.warning("Skipping CV: fewer than 2 segment groups")
                cv_scores = np.array([np.nan])

            results.update(
                {
                    "test_accuracy": (y_pred == y_test).mean(),
                    "cv_f1_mean": cv_scores.mean(),
                    "cv_f1_std": cv_scores.std(),
                    "classification_report": classification_report(
                        y_test,
                        y_pred,
                        target_names=target_names,
                        output_dict=True,
                    ),
                    "confusion_matrix": sklearn_confusion_matrix(y_test, y_pred).tolist(),
                }
            )

            # Print summary with test metrics
            print("\n" + "=" * 50)
            print("TRAINING RESULTS")
            print("=" * 50)
            print(f"Training samples: {results['n_train']}")
            print(f"Test samples: {results['n_test']}")
            print(f"Test accuracy: {results['test_accuracy']:.3f}")
            print(f"CV F1 (5-fold): {results['cv_f1_mean']:.3f} ± {results['cv_f1_std']:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=target_names))
        else:
            # No test set - just print training info
            print("\n" + "=" * 50)
            print("TRAINING RESULTS (no test set)")
            print("=" * 50)
            print(f"Training samples: {results['n_train']}")

        print("\nFeature Importance (top 5):")
        importance = sorted(results["feature_importance"].items(), key=lambda x: -x[1])
        for feat, imp in importance[:5]:
            print(f"  {feat}: {imp:.3f}")

        return results

    def _extract_features_and_labels(
        self, df: pd.DataFrame, binary: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and labels from dataframe.

        Does NOT perform imputation - that should be done after train/test split.

        Args:
            df: Labels dataframe
            binary: If True, convert to binary labels (match=1, else=0)

        Returns:
            Tuple of (X features, y labels)
        """
        if len(df) == 0:
            raise ValueError("Cannot extract features from empty dataframe")

        # Check if features are in a nested dict or individual columns
        # Handle null/empty first row gracefully
        has_features_col = "features" in df.columns
        first_features = df["features"].iloc[0] if has_features_col else None
        if has_features_col and first_features and isinstance(first_features, dict):
            return self._extract_from_dict(df, binary)
        else:
            return self._extract_from_columns(df, binary)

    def _extract_from_columns(
        self, df: pd.DataFrame, binary: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features from individual columns."""
        # Use pre-set feature_names if already configured (e.g., by exclude_semantic)
        # Otherwise, build list from all available FEATURE_COLUMNS
        base_features = (
            self.feature_names
            if hasattr(self, "feature_names") and self.feature_names
            else FEATURE_COLUMNS
        )

        # Build list of actual feature columns present in the data
        actual_features = [feat for feat in base_features if feat in df.columns]
        self.feature_names = actual_features
        logger.info(f"Using features: {self.feature_names}")

        # Extract feature matrix (without imputation)
        X = df[self.feature_names].values.astype(np.float32)

        # Extract labels
        if binary:
            y = (df["label"] == "match").astype(int).values
        else:
            y = df["label"].map(self.label_encoder).values

        return X, y

    def _extract_from_dict(
        self, df: pd.DataFrame, binary: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features from nested dict column."""
        feature_rows = []

        for _, row in df.iterrows():
            features = row.get("features", {})
            if features and isinstance(features, dict):
                feat_row = [features.get(col, np.nan) for col in self.feature_names]
                feature_rows.append(feat_row)
            else:
                # Fill with NaNs for rows without features
                feature_rows.append([np.nan] * len(self.feature_names))

        X = np.array(feature_rows, dtype=np.float32)

        if binary:
            y = (df["label"] == "match").astype(int).values
        else:
            y = df["label"].map(self.label_encoder).values

        return X, y

    def _validate_training_pairs(
        self,
        df: pd.DataFrame,
        max_centroid_distance_m: float = 500.0,
        max_hausdorff_m: float = 1000.0,
    ) -> pd.DataFrame:
        """Validate training pairs and drop those with implausible features.

        Detects pairs with corrupted or stale geometry lookups by checking
        feature values against physically plausible thresholds.

        Args:
            df: Training DataFrame with feature columns
            max_centroid_distance_m: Max allowed centroid distance (default 500m)
            max_hausdorff_m: Max allowed Hausdorff distance (default 1000m)

        Returns:
            Filtered DataFrame with implausible pairs removed
        """
        n_before = len(df)
        drop_mask = pd.Series(False, index=df.index)

        # Check centroid distance
        if "centroid_distance_m" in df.columns:
            bad_centroid = df["centroid_distance_m"].notna() & (
                df["centroid_distance_m"] > max_centroid_distance_m
            )
            if bad_centroid.any():
                logger.warning(
                    f"Validation: {bad_centroid.sum()} pairs with "
                    f"centroid_distance_m > {max_centroid_distance_m}m"
                )
            drop_mask |= bad_centroid

        # Check Hausdorff distance
        if "hausdorff_distance_m" in df.columns:
            bad_hausdorff = df["hausdorff_distance_m"].notna() & (
                df["hausdorff_distance_m"] > max_hausdorff_m
            )
            if bad_hausdorff.any():
                logger.warning(
                    f"Validation: {bad_hausdorff.sum()} pairs with "
                    f"hausdorff_distance_m > {max_hausdorff_m}m"
                )
            drop_mask |= bad_hausdorff

        # Check all-NaN feature rows
        feature_cols = [c for c in self.feature_names if c in df.columns]
        if feature_cols:
            all_nan = df[feature_cols].isna().all(axis=1)
            if all_nan.any():
                logger.warning(f"Validation: {all_nan.sum()} pairs with all-NaN features")
            drop_mask |= all_nan

        if drop_mask.any():
            # Log per-dataset breakdown
            if "dataset" in df.columns:
                dropped_df = df[drop_mask]
                by_dataset = dropped_df.groupby("dataset").size()
                total_by_dataset = df.groupby("dataset").size()
                for dataset_name in by_dataset.index:
                    n_dropped = by_dataset[dataset_name]
                    n_total = total_by_dataset[dataset_name]
                    pct = n_dropped / n_total * 100
                    logger.info(
                        f"  {dataset_name}: dropped {n_dropped}/{n_total} pairs ({pct:.1f}%)"
                    )
                    if pct > 20:
                        logger.warning(
                            f"  {dataset_name}: >20% dropped — possible systematic data issue"
                        )

            n_dropped = drop_mask.sum()
            pct = n_dropped / n_before * 100
            logger.info(f"Validation: dropped {n_dropped}/{n_before} pairs ({pct:.1f}%)")
            df = df[~drop_mask].copy()
        else:
            logger.info("Validation: all training pairs passed plausibility checks")

        return df

    def _cap_infinities(self, X: np.ndarray) -> np.ndarray:
        """Cap infinite values at MAX_DISTANCE_METERS.

        XGBoost handles NaN natively but not inf.

        Args:
            X: Feature matrix with potential infinite values

        Returns:
            Feature matrix with infinities capped
        """
        X = X.copy()
        inf_mask = np.isinf(X)
        if inf_mask.any():
            X[inf_mask] = MAX_DISTANCE_METERS
        return X

    def predict(self, features: list[dict[str, float]]) -> np.ndarray:
        """Predict match probabilities.

        Args:
            features: List of feature dictionaries

        Returns:
            Array of match probabilities (0-1)
        """
        if self.model is None:
            raise ValueError("No model loaded - call train() or load_model() first")

        X = self._features_to_array(features)
        probs = self.model.predict_proba(X)

        # Find the index for 'match' class dynamically
        match_class = self.label_encoder.get("match", 1)
        class_indices = list(self.model.classes_)
        if match_class in class_indices:
            match_idx = class_indices.index(match_class)
        else:
            # Fallback for binary where classes are [0, 1]
            match_idx = 1

        return probs[:, match_idx]

    def predict_class(self, features: list[dict[str, float]]) -> list[str]:
        """Predict class labels.

        Args:
            features: List of feature dictionaries

        Returns:
            List of predicted labels
        """
        if self.model is None:
            raise ValueError("No model loaded")

        X = self._features_to_array(features)
        y_pred = self.model.predict(X)
        return [self.label_decoder.get(int(y), "unknown") for y in y_pred]

    def _features_to_array(self, features: list[dict[str, float]]) -> np.ndarray:
        """Convert feature dicts to numpy array.

        NaN values are preserved for XGBoost's native missing-value handling.
        Infinite values are capped at MAX_DISTANCE_METERS.
        """
        # pd.DataFrame(list-of-dicts) uses C-optimized path — orders of magnitude
        # faster than per-element Python dict lookups
        df = pd.DataFrame(features)
        # Reorder columns to match model's expected feature order; adds NaN
        # for any feature columns missing entirely from all dicts
        df = df.reindex(columns=self.feature_names)
        # Replace infinities with cap value; preserve NaN for XGBoost
        arr = df.to_numpy(dtype=np.float32)
        inf_mask = np.isinf(arr)
        if inf_mask.any():
            arr[inf_mask] = MAX_DISTANCE_METERS
        return arr

    def score_candidates(
        self,
        candidates: list,
        reference,
        target,
        ref_name_column: str = "names",
        target_name_column: str = "names",
        ref_class_column: str = "class",
        target_class_column: str = "class",
        ref_subclass_column: str = "subclass",
        target_subclass_column: str = "subclass",
        ref_id_column: str = "id",
        target_id_column: str = "id",
        n_jobs: int = -1,
    ) -> list[MatchResult]:
        """Score candidates using the ML model.

        Args:
            candidates: List of CandidatePair objects
            reference: Reference GeoDataFrame
            target: Target GeoDataFrame
            ref_name_column: Column name for reference names
            target_name_column: Column name for target names
            ref_class_column: Column name for reference class
            target_class_column: Column name for target class
            ref_subclass_column: Column name for reference subclass
            target_subclass_column: Column name for target subclass
            ref_id_column: Column name for reference IDs
            target_id_column: Column name for target IDs
            n_jobs: Number of parallel jobs (-1 for all cores)

        Returns:
            List of MatchResult objects
        """
        # Handle auto model selection
        if self.model is None and self._auto_select:
            from ..config import settings

            if settings.auto_select_model:
                # Auto-select model based on target dataset
                # select_model_for_dataset uses settings defaults when paths are None
                selected_model = select_model_for_dataset(
                    target,
                    full_model_path=self.model_path,
                    name_column=target_name_column,
                )
                self.load_model(selected_model)
            elif self.model_path:
                self.load_model(self.model_path)

        if self.model is None:
            raise ValueError(
                "No ML model loaded. Train a model first with 'matcher train' "
                "or provide a model path to MLMatcher(model_path=...)."
            )

        # Handle empty candidates list
        if not candidates:
            return []

        # Validate that data is in projected CRS (meters)
        # Projection should happen early in the pipeline (runner.py)
        # This validation catches misuse when score_candidates is called directly
        # validate_projected_crs raises for both None CRS and geographic CRS
        validate_projected_crs(reference, "reference")
        validate_projected_crs(target, "target")

        # Timing instrumentation for pipeline profiling (visible at DEBUG log level)
        timings = {}

        # Filter candidates by physical overlap (actual geometric intersection, no translation)
        # This removes collinear segments that barely touch at tips, which dominate labeling
        # time but are almost never real matches. 5m threshold gives 5.9:1 no_match:match ratio.
        from ..config import PHYSICAL_OVERLAP_MIN_M

        t0 = time.perf_counter()

        # Batch computation using vectorized shapely for performance
        import shapely as shapely_mod

        ref_geoms_arr = reference.geometry.to_numpy()
        target_geoms_arr = target.geometry.to_numpy()
        ref_geom_arr = np.array([ref_geoms_arr[c.ref_idx] for c in candidates], dtype=object)
        target_geom_arr = np.array(
            [target_geoms_arr[c.target_idx] for c in candidates], dtype=object
        )

        # Buffer targets and intersect with refs (vectorized)
        target_buffers = shapely_mod.buffer(target_geom_arr, PHYSICAL_OVERLAP_MIN_M)
        intersections = shapely_mod.intersection(ref_geom_arr, target_buffers)
        overlap_lengths = shapely_mod.length(intersections)

        # Filter candidates that meet threshold
        mask = overlap_lengths >= PHYSICAL_OVERLAP_MIN_M
        filtered_candidates = [c for c, keep in zip(candidates, mask) if keep]

        n_filtered = len(candidates) - len(filtered_candidates)
        if n_filtered > 0:
            logger.info(
                f"Physical overlap filter: removed {n_filtered} candidates "
                f"(<{PHYSICAL_OVERLAP_MIN_M}m overlap), {len(filtered_candidates)} remaining"
            )
        candidates = filtered_candidates
        timings["physical_overlap_filter"] = time.perf_counter() - t0
        logger.debug(f"[TIMING] physical_overlap_filter: {timings['physical_overlap_filter']:.2f}s")

        # Handle case where all candidates were filtered
        if not candidates:
            logger.info("All candidates filtered by physical overlap threshold")
            return []

        # Prepare worker data using shared pipeline setup
        # Pass pre-extracted geometry arrays to avoid redundant GeoSeries->ndarray conversion
        from ..features.pipeline import prepare_worker_data

        t0 = time.perf_counter()
        pipeline_result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column=ref_id_column,
            target_id_column=target_id_column,
            ref_name_column=ref_name_column,
            target_name_column=target_name_column,
            ref_class_column=ref_class_column,
            target_class_column=target_class_column,
            ref_subclass_column=ref_subclass_column,
            target_subclass_column=target_subclass_column,
            n_jobs=n_jobs,
            ref_geoms=ref_geoms_arr,
            target_geoms=target_geoms_arr,
        )
        worker_data = pipeline_result.worker_data
        alignments = pipeline_result.alignments
        timings["prepare_worker_data"] = time.perf_counter() - t0

        # Parallel feature computation using shared dispatch
        from ..features.pipeline import compute_features_parallel

        t0 = time.perf_counter()
        parallel_result = compute_features_parallel(
            candidates=candidates,
            worker_data=worker_data,
            n_jobs=n_jobs,
            sort_for_locality=True,
        )
        features_list = parallel_result.features_list
        total_errors = parallel_result.error_aggregator
        wall_clock = parallel_result.wall_clock_seconds
        timings["parallel_feature_computation"] = wall_clock
        logger.info(f"[TIMING] parallel_feature_computation: {wall_clock:.2f}s")

        # Aggregate per-feature timing from worker results (when profiling enabled)
        from ..features.compute import is_profiling_enabled

        if is_profiling_enabled():
            timing_agg: dict[str, float] = {}
            n_timed = 0
            for feat_dict in features_list:
                if feat_dict is None:
                    continue
                has_timing = False
                for k, v in list(feat_dict.items()):
                    if k.startswith("_t_"):
                        name = k[3:]  # strip "_t_" prefix
                        timing_agg[name] = timing_agg.get(name, 0.0) + v
                        has_timing = True
                if has_timing:
                    n_timed += 1

            if n_timed > 0:
                cpu_total = sum(timing_agg.values())
                throughput = n_timed / wall_clock if wall_clock > 0 else 0

                logger.info("[PROFILING] ===== Feature Computation Breakdown =====")
                logger.info(f"[PROFILING] {'Section':<35} {'Time (s)':>9} {'%':>7} {'us/pair':>9}")
                logger.info(f"[PROFILING] {'-' * 35} {'-' * 9} {'-' * 7} {'-' * 9}")
                for name, total in sorted(timing_agg.items(), key=lambda x: -x[1]):
                    pct = total / cpu_total * 100 if cpu_total > 0 else 0
                    us_per_pair = total / n_timed * 1e6
                    logger.info(
                        f"[PROFILING] {name:<35} {total:>9.2f} {pct:>6.1f}% {us_per_pair:>9.0f}"
                    )
                logger.info(f"[PROFILING] {'-' * 35} {'-' * 9} {'-' * 7} {'-' * 9}")
                logger.info(f"[PROFILING] {'SUM OF SECTIONS (cpu-time)':<35} {cpu_total:>9.2f}")
                logger.info(f"[PROFILING] {'WALL CLOCK':<35} {wall_clock:>9.2f}")
                n_workers_used = default_worker_count() if n_jobs == -1 else max(1, n_jobs)
                logger.info(
                    f"[PROFILING] Pairs: {n_timed:,} | Workers: {n_workers_used} "
                    f"| Throughput: {throughput:,.0f} pairs/s"
                )

            # Strip _t_* keys from feature dicts before ML prediction
            for feat_dict in features_list:
                if feat_dict is None:
                    continue
                timing_keys = [k for k in feat_dict if k.startswith("_t_")]
                for k in timing_keys:
                    del feat_dict[k]

        # Filter out rejected pairs (None results) and track valid candidates
        # Pairs are rejected when they don't have aligned endpoint features
        valid_pairs = [
            (cand, feat) for cand, feat in zip(candidates, features_list) if feat is not None
        ]
        rejected_count = len(candidates) - len(valid_pairs)
        if rejected_count > 0:
            rejection_rate = rejected_count / len(candidates)
            logger.info(
                f"Rejected {rejected_count} pairs without aligned endpoint features "
                f"({len(valid_pairs)} remaining, {rejection_rate:.1%} rejection rate)"
            )
            # Fail if too many pairs rejected - indicates data or configuration problem
            if rejection_rate > 0.01:
                raise ValueError(
                    f"Alignment rejection rate {rejection_rate:.1%} exceeds 1% threshold. "
                    f"{rejected_count} of {len(candidates)} pairs failed to align. "
                    "This may indicate CRS issues, bad geometry data, or a bug."
                )

        # Extract valid features and candidates
        valid_candidates = [p[0] for p in valid_pairs]
        valid_features = [p[1] for p in valid_pairs]

        # Check for degenerate geometry errors (aligned sublines becoming points/empty)
        # These pairs get error default features but may indicate data quality issues
        error_features_list = [f for f in valid_features if f.get("_error")]
        if error_features_list or total_errors.has_errors():
            error_rate = len(error_features_list) / len(valid_features) if valid_features else 0

            # Log detailed error breakdown from aggregated worker errors
            if total_errors.has_errors():
                summary = total_errors.summary()
                logger.warning(
                    f"Feature computation errors: {summary['total']} total across workers"
                )
                logger.warning(f"  Errors by phase: {summary['by_phase']}")
                logger.warning(f"  Errors by type: {summary['by_type']}")

                # Log sample errors (up to error_log_samples per type)
                for err_key, sample_info in list(summary["samples"].items())[
                    : settings.error_log_samples
                ]:
                    logger.warning(f"  Sample [{err_key}]: {sample_info['message'][:200]}")

            if error_features_list:
                logger.warning(
                    f"{len(error_features_list)} candidates had feature computation errors "
                    f"({error_rate:.1%} of valid pairs)"
                )

            # Fail if overall error rate exceeds 20% threshold
            if error_rate > 0.20:
                raise ValueError(
                    f"Feature computation error rate {error_rate:.1%} exceeds 20% threshold. "
                    f"{len(error_features_list)} of {len(valid_features)} pairs had errors "
                    "(likely degenerate aligned sublines). "
                    "This may indicate poor alignment coverage or bad geometry data."
                )

            # Check per-phase hard fail threshold
            if total_errors.has_errors():
                total_pairs = len(valid_features) if valid_features else 1
                for phase, count in total_errors.counts_by_phase.items():
                    phase_rate = count / total_pairs
                    if phase_rate > settings.error_hard_fail_threshold:
                        raise ValueError(
                            f"Error rate for phase '{phase}' ({phase_rate:.1%}) exceeds "
                            f"{settings.error_hard_fail_threshold:.0%} hard fail threshold. "
                            f"{count} of {total_pairs} pairs failed in this phase."
                        )

        # Batch prediction - use probability (confidence), not predicted class
        # This allows the downstream optimizer to use confidence threshold
        logger.info(f"Running XGBoost prediction on {len(valid_features):,} candidates...")
        t0 = time.perf_counter()
        probs = self.predict(valid_features)
        timings["xgboost_prediction"] = time.perf_counter() - t0
        logger.debug(f"[TIMING] xgboost_prediction: {timings['xgboost_prediction']:.2f}s")

        # Build results - use confidence-based decision, not class-based
        # This ensures high-confidence matches aren't filtered just because
        # the model's decision boundary puts them in "no_match" class
        logger.info(f"Building {len(valid_candidates):,} MatchResult objects...")
        t0 = time.perf_counter()
        results = []
        for i, cand in enumerate(valid_candidates):
            prob = probs[i]

            # Use confidence thresholds instead of class prediction
            # This makes the ML model behave more like a confidence scorer
            if prob >= settings.scoring_match_threshold:
                decision = MatchDecision.MATCH
            elif prob >= settings.scoring_review_threshold:
                decision = MatchDecision.REVIEW  # Low confidence but possible
            else:
                decision = MatchDecision.NO_MATCH

            # Get alignment for linear reference fields
            alignment = alignments.get((cand.ref_idx, cand.target_idx))

            results.append(
                MatchResult(
                    ref_id=cand.ref_id,
                    target_id=cand.target_id,
                    decision=decision,
                    confidence=prob,
                    score_breakdown={},  # ML doesn't have component scores
                    features=valid_features[i],
                    gers_start_frac=alignment.overture_start_frac if alignment else None,
                    gers_end_frac=alignment.overture_end_frac if alignment else None,
                    local_start_frac=alignment.dataset_start_frac if alignment else None,
                    local_end_frac=alignment.dataset_end_frac if alignment else None,
                )
            )

            # Progress logging every 100k
            if (i + 1) % 100000 == 0:
                logger.info(f"Built {i + 1:,}/{len(candidates):,} MatchResult objects...")

        timings["result_building"] = time.perf_counter() - t0
        logger.debug(f"[TIMING] result_building: {timings['result_building']:.2f}s")

        logger.info(f"Built {len(results):,} MatchResult objects")

        # Log timing summary table
        total = sum(timings.values())
        logger.debug("[TIMING] ===== Pipeline Stage Summary =====")
        logger.debug(f"[TIMING] {'Stage':<35} {'Time (s)':>10} {'%':>6}")
        logger.debug(f"[TIMING] {'-' * 35} {'-' * 10} {'-' * 6}")
        for stage, elapsed in timings.items():
            pct = elapsed / total * 100 if total > 0 else 0
            logger.debug(f"[TIMING] {stage:<35} {elapsed:>10.2f} {pct:>5.1f}%")
        logger.debug(f"[TIMING] {'-' * 35} {'-' * 10} {'-' * 6}")
        logger.debug(f"[TIMING] {'TOTAL':<35} {total:>10.2f} {'100.0':>6}%")
        logger.debug(f"[TIMING] candidates={len(candidates):,}")

        return results


def train_model(
    labels_dir: str = "labels",
    output_path: str = "data/models/matcher_model.joblib",
    binary: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function to train and save a model.

    Args:
        labels_dir: Path to Hive-partitioned labels directory
        output_path: Path to save trained model
        binary: Train binary (match/no_match) or multiclass
        **kwargs: XGBoost parameters

    Returns:
        Training results dict
    """
    matcher = MLMatcher()
    results = matcher.train(labels_dir=labels_dir, binary=binary, **kwargs)
    matcher.save_model(output_path)
    return results


def evaluate_by_dataset(
    model_path: str,
    labels_dir: str = "labels",
    binary: bool = True,
    show_by_dataset: bool = True,
    holdout: bool = True,
    holdout_pct: float = 0.2,
    seed: int = 42,
    filter_datasets: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate model performance broken down by dataset.

    Loads all label partitions and evaluates the model on each dataset separately.

    Args:
        model_path: Path to trained model
        labels_dir: Directory containing Hive-partitioned label CSVs
        binary: Evaluate as binary (match vs no_match)
        show_by_dataset: If True, show per-dataset metrics; if False, only show overall
        holdout: If True (default), use holdout set for unbiased evaluation
        holdout_pct: Fraction of data to hold out for testing (default 0.2 = 20%)
        seed: Random seed for holdout split (for reproducibility)
        filter_datasets: If provided, only evaluate on these datasets

    Returns:
        Dictionary mapping dataset name to metrics dict
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    from ..labeling.label_store import LabelStore

    # Load model
    matcher = MLMatcher(model_path)

    # Load all labels using LabelStore
    all_labels = LabelStore.load_all(Path(labels_dir))

    if len(all_labels) == 0:
        logger.warning(f"No labels found in {labels_dir}")
        return {}

    # Get unique datasets
    if "dataset" not in all_labels.columns:
        logger.warning("No 'dataset' column found - cannot evaluate by dataset")
        return {}

    # Filter to specific datasets if requested
    if filter_datasets:
        all_labels = all_labels[all_labels["dataset"].isin(filter_datasets)].copy()
        if len(all_labels) == 0:
            logger.warning(f"No labels found for datasets: {filter_datasets}")
            return {}
        logger.info(f"Filtered to {len(all_labels)} labels from: {filter_datasets}")

    # If holdout requested, split the data first using segment-aware splitting
    if holdout:
        valid_labels = {"match", "no_match"}
        eval_df = all_labels[all_labels["label"].isin(valid_labels)].copy()
        _, test_idx = segment_aware_split(eval_df, test_size=holdout_pct, random_state=seed)
        all_labels = eval_df.iloc[test_idx]
        print(
            f"\n[Holdout mode: evaluating on {len(all_labels)} samples ({holdout_pct * 100:.0f}% of data, seed={seed})]"
        )

    datasets = all_labels["dataset"].unique()

    results = {}
    all_y_true = []
    all_y_pred = []

    if show_by_dataset:
        print("\n" + "=" * 60)
        print("EVALUATION BY DATASET")
        print("=" * 60)

    for dataset_name in sorted(datasets):
        # Filter to this dataset
        df = all_labels[all_labels["dataset"] == dataset_name].copy()

        # Filter to valid labels
        valid_labels = {"match", "no_match"}
        df = df[df["label"].isin(valid_labels)].copy()

        if len(df) == 0:
            logger.warning(f"No valid labels for dataset {dataset_name}")
            continue

        # Extract features
        X, y = matcher._extract_features_and_labels(df, binary=binary)

        # Cap infinities (XGBoost handles NaN natively)
        X = matcher._cap_infinities(X)

        # Predict
        y_pred = matcher.model.predict(X)

        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average=METRIC_AVERAGE, zero_division=0)
        precision = precision_score(y, y_pred, average=METRIC_AVERAGE, zero_division=0)
        recall = recall_score(y, y_pred, average=METRIC_AVERAGE, zero_division=0)

        # Count labels
        n_match = (y == 1).sum() if binary else (df["label"] == "match").sum()
        n_no_match = (y == 0).sum() if binary else (df["label"] == "no_match").sum()

        results[dataset_name] = {
            "n_samples": len(df),
            "n_match": int(n_match),
            "n_no_match": int(n_no_match),
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

        # Accumulate for overall
        all_y_true.extend(y)
        all_y_pred.extend(y_pred)

        # Print summary (only if showing by dataset)
        if show_by_dataset:
            print(f"\n{dataset_name}:")
            print(f"  Samples: {len(df)} ({n_match} match, {n_no_match} no_match)")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  F1: {f1:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")

    # Overall metrics
    if all_y_true:
        overall_accuracy = accuracy_score(all_y_true, all_y_pred)
        overall_f1 = f1_score(all_y_true, all_y_pred, average=METRIC_AVERAGE, zero_division=0)

        if show_by_dataset:
            print("\n" + "-" * 60)
        else:
            print("\n" + "=" * 60)
        print("OVERALL:")
        print(f"  Total samples: {len(all_y_true)}")
        print(f"  Accuracy: {overall_accuracy:.3f}")
        print(f"  F1: {overall_f1:.3f}")

        results["_overall"] = {
            "n_samples": len(all_y_true),
            "accuracy": overall_accuracy,
            "f1": overall_f1,
        }

    return results
