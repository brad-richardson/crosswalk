"""Data loading and candidate preparation for labeling UI."""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString
from shapely.ops import transform

from ..blocking import generate_candidates
from ..config import FEATURE_COLUMNS, FEATURE_VERSION, default_worker_count, settings
from ..features.alignment import create_subline
from ..features.semantic import _extract_name_string
from ..filenames import feature_cache_path, scored_cache_path
from ..matching.ml import MLMatcher
from ..utils import ensure_projected_crs, filter_to_linestrings

logger = logging.getLogger(__name__)


_cached_matcher: MLMatcher | None = None
_matcher_loaded: bool = False


def get_cached_matcher() -> MLMatcher | None:
    """Get ML matcher, loading model if needed.

    Uses module-level caching to persist across calls.

    Returns:
        Loaded MLMatcher or None if model doesn't exist.
    """
    global _cached_matcher, _matcher_loaded
    if _matcher_loaded:
        return _cached_matcher

    model_path = settings.model_path
    if not model_path.exists():
        logger.warning(f"ML model not found at {model_path}")
        _matcher_loaded = True
        return None

    logger.info(f"Loading ML model from {model_path}...")
    matcher = MLMatcher(auto_select=True)
    matcher.load_model(str(model_path))
    logger.info("ML model loaded successfully")
    _cached_matcher = matcher
    _matcher_loaded = True
    return matcher


def _compute_score_breakdown_from_features(features: dict[str, float]) -> dict[str, float]:
    """Compute normalized score breakdown from raw ML features for UI display.

    The ML scorer doesn't produce component scores like the rule-based scorer,
    so we derive normalized 0-1 scores from raw features for the UI and cached
    score breakdowns.

    Args:
        features: Raw feature dict from ML scorer

    Returns:
        Dict of normalized scores derived from the raw ML feature values.
    """

    # Normalize distance features (lower distance = higher score)
    # Use 50m as "bad" threshold (score approaches 0)
    def norm_distance(dist_m: float, max_m: float = 50.0) -> float:
        if dist_m is None or dist_m < 0:
            return 0.0
        return max(0.0, 1.0 - dist_m / max_m)

    # Normalize heading delta (0-180 degrees, lower = better)
    def norm_heading(delta: float) -> float:
        if delta is None or delta < 0:
            return 0.0
        return max(0.0, 1.0 - delta / 90.0)  # 90 degrees = score 0

    # Normalize length ratio (1.0 = perfect, further from 1 = worse)
    def norm_length_ratio(ratio: float) -> float:
        if ratio is None or ratio <= 0:
            return 0.0
        if ratio > 1:
            ratio = 1.0 / ratio  # Normalize so ratio is always <= 1
        return ratio

    return {
        "hausdorff_norm": norm_distance(features.get("hausdorff_distance_m", 50)),
        "mean_hausdorff_norm": norm_distance(features.get("mean_hausdorff_distance_m", 50)),
        "buffer_iou": features.get("buffer_iou_5m", 0.0),
        "overlap_ratio": features.get("min_coverage", 0.0),
        "heading_norm": norm_heading(features.get("heading_delta", 90)),
        "length_ratio": norm_length_ratio(features.get("length_ratio", 0)),
        # projection_norm uses mean_hausdorff (they were equivalent features)
        "projection_norm": norm_distance(features.get("mean_hausdorff_distance_m", 50)),
        "name_similarity": features.get("name_jaro_winkler", 0.0),
        "class_similarity": features.get("class_similarity", 0.0),
    }


@dataclass
class CandidatePairView:
    """View model for a single candidate pair in the labeling UI."""

    ref_id: str
    target_id: str
    ref_geometry: LineString  # Full reference geometry (for context)
    target_geometry: LineString  # Full target geometry (for context)
    ref_name: str | None
    target_name: str | None
    ref_class: str | None
    target_class: str | None
    decision: str  # "match", "review", "no_match"
    confidence: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    ref_subclass: str | None = None
    target_subclass: str | None = None
    features: dict[str, float] = field(default_factory=dict)
    # Aligned/chopped geometries from the alignment algorithm
    # These represent the portions of each line that actually overlap
    ref_aligned_geometry: LineString | None = None
    target_aligned_geometry: LineString | None = None
    # Alignment fractions (0.0-1.0) showing where the overlap occurs on each line
    ref_start_frac: float = 0.0
    ref_end_frac: float = 1.0
    target_start_frac: float = 0.0
    target_end_frac: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for cache storage."""
        return {
            "ref_id": self.ref_id,
            "target_id": self.target_id,
            "ref_geometry_wkt": self.ref_geometry.wkt,
            "target_geometry_wkt": self.target_geometry.wkt,
            "ref_name": self.ref_name,
            "target_name": self.target_name,
            "ref_class": self.ref_class,
            "target_class": self.target_class,
            "ref_subclass": self.ref_subclass,
            "target_subclass": self.target_subclass,
            "decision": self.decision,
            "confidence": self.confidence,
            "score_breakdown_json": json.dumps(self.score_breakdown),
            "features_json": json.dumps(self.features),
            # Aligned geometries (may be None)
            "ref_aligned_geometry_wkt": self.ref_aligned_geometry.wkt
            if self.ref_aligned_geometry
            else None,
            "target_aligned_geometry_wkt": self.target_aligned_geometry.wkt
            if self.target_aligned_geometry
            else None,
            # Alignment fractions
            "ref_start_frac": self.ref_start_frac,
            "ref_end_frac": self.ref_end_frac,
            "target_start_frac": self.target_start_frac,
            "target_end_frac": self.target_end_frac,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidatePairView":
        """Deserialize from dictionary loaded from cache."""
        # Validate required fields to catch cache corruption early
        required_fields = [
            "ref_id",
            "target_id",
            "ref_geometry_wkt",
            "target_geometry_wkt",
            "decision",
            "confidence",
        ]
        missing = [key for key in required_fields if key not in data]
        if missing:
            raise ValueError(
                f"Cached CandidatePairView missing required fields {missing}. "
                f"Available keys: {list(data.keys())}"
            )

        # Parse aligned geometries (may be None in older caches)
        ref_aligned_wkt = data.get("ref_aligned_geometry_wkt")
        target_aligned_wkt = data.get("target_aligned_geometry_wkt")

        return cls(
            ref_id=data["ref_id"],
            target_id=data["target_id"],
            ref_geometry=wkt.loads(data["ref_geometry_wkt"]),
            target_geometry=wkt.loads(data["target_geometry_wkt"]),
            ref_name=data.get("ref_name"),
            target_name=data.get("target_name"),
            ref_class=data.get("ref_class"),
            target_class=data.get("target_class"),
            ref_subclass=data.get("ref_subclass"),
            target_subclass=data.get("target_subclass"),
            decision=data["decision"],
            confidence=data["confidence"],
            score_breakdown=json.loads(data.get("score_breakdown_json", "{}")),
            features=json.loads(data.get("features_json", "{}")),
            # Aligned geometries from cache
            ref_aligned_geometry=wkt.loads(ref_aligned_wkt) if ref_aligned_wkt else None,
            target_aligned_geometry=wkt.loads(target_aligned_wkt) if target_aligned_wkt else None,
            # Alignment fractions (default to full segment for backward compatibility)
            ref_start_frac=data.get("ref_start_frac", 0.0),
            ref_end_frac=data.get("ref_end_frac", 1.0),
            target_start_frac=data.get("target_start_frac", 0.0),
            target_end_frac=data.get("target_end_frac", 1.0),
        )


def get_cache_path(dataset_id: str) -> Path:
    """Get the cache file path for a dataset.

    Args:
        dataset_id: Unique identifier for the dataset (e.g., "boston_streets")

    Returns:
        Path to the cache parquet file
    """
    return scored_cache_path(dataset_id)


def get_cache_info(
    dataset_id: str,
    reference_path: Path | None = None,
    target_path: Path | None = None,
) -> dict[str, Any]:
    """Get information about the candidate cache for a dataset.

    Args:
        dataset_id: Unique identifier for the dataset
        reference_path: Path to reference data file (for freshness check)
        target_path: Path to target data file (for freshness check)

    Returns:
        Dictionary with cache info:
        - exists: Whether cache file exists
        - path: Path to cache file
        - created: Cache creation timestamp (or None)
        - age_hours: Hours since cache creation (or None)
        - is_fresh: Whether cache is newer than source files (or None)
        - candidate_count: Number of candidates in cache (or None)
    """
    cache_path = get_cache_path(dataset_id)
    info: dict[str, Any] = {
        "exists": cache_path.exists(),
        "path": cache_path,
        "created": None,
        "age_hours": None,
        "is_fresh": None,
        "candidate_count": None,
    }

    if not cache_path.exists():
        return info

    # Get cache modification time
    cache_mtime = cache_path.stat().st_mtime
    cache_datetime = datetime.fromtimestamp(cache_mtime)
    info["created"] = cache_datetime
    info["age_hours"] = (datetime.now() - cache_datetime).total_seconds() / 3600

    # Check freshness against source files
    if reference_path and target_path:
        source_mtimes = []
        if reference_path.exists():
            source_mtimes.append(reference_path.stat().st_mtime)
        if target_path.exists():
            source_mtimes.append(target_path.stat().st_mtime)
        if source_mtimes:
            info["is_fresh"] = cache_mtime > max(source_mtimes)

    # Get candidate count from metadata without loading full file
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(cache_path)
        info["candidate_count"] = metadata.num_rows
    except Exception as e:
        # Failed to read metadata; leave candidate_count as None
        logger.debug(f"Failed to read cache metadata from {cache_path}: {e}")

    return info


def load_cached_candidates(dataset_id: str) -> list[CandidatePairView] | None:
    """Load scored candidates from cache if available.

    Args:
        dataset_id: Unique identifier for the dataset

    Returns:
        List of CandidatePairView objects, or None if cache doesn't exist
    """
    cache_path = get_cache_path(dataset_id)
    if not cache_path.exists():
        logger.info(f"No cache found for dataset {dataset_id}")
        return None

    logger.info(f"Loading cached candidates from {cache_path}")
    try:
        df = pd.read_parquet(cache_path)
        # Use to_dict('records') instead of iterrows() for 10-20x faster loading
        candidates = [CandidatePairView.from_dict(row) for row in df.to_dict("records")]

        # Recompute score_breakdown from features for UI display
        # (cached data may have empty score_breakdown from ML scorer)
        for cand in candidates:
            if not cand.score_breakdown and cand.features:
                cand.score_breakdown = _compute_score_breakdown_from_features(cand.features)

        logger.info(f"Loaded {len(candidates)} candidates from cache")
        return candidates
    except Exception as e:
        logger.warning(f"Failed to load cache for {dataset_id}: {e}")
        return None


def save_candidates_to_cache(
    dataset_id: str,
    candidates: list[CandidatePairView],
) -> Path:
    """Save scored candidates to cache.

    Args:
        dataset_id: Unique identifier for the dataset
        candidates: List of CandidatePairView objects to cache

    Returns:
        Path to the saved cache file
    """
    cache_path = get_cache_path(dataset_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(candidates)} candidates to cache at {cache_path}")

    # Convert to DataFrame
    records = [c.to_dict() for c in candidates]
    df = pd.DataFrame(records)

    # Save as parquet
    df.to_parquet(cache_path, index=False, compression="zstd")

    logger.info(f"Cache saved successfully: {cache_path}")
    return cache_path


def delete_cache(dataset_id: str) -> bool:
    """Delete the cache for a dataset.

    Args:
        dataset_id: Unique identifier for the dataset

    Returns:
        True if cache was deleted, False if it didn't exist
    """
    cache_path = get_cache_path(dataset_id)
    if cache_path.exists():
        cache_path.unlink()
        logger.info(f"Deleted cache for {dataset_id}")
        return True
    return False


# ============================================================================
# FEATURE CACHE FUNCTIONS (versioned, without ML predictions)
# ============================================================================


def get_feature_cache_path(dataset_id: str) -> Path:
    """Get the versioned feature cache file path for a dataset.

    The feature cache stores computed features WITHOUT ML predictions,
    allowing fast re-scoring when the ML model changes.

    Args:
        dataset_id: Unique identifier for the dataset

    Returns:
        Path to the feature cache parquet file (versioned)
    """
    return feature_cache_path(dataset_id)


def get_feature_cache_info(
    dataset_id: str,
    reference_path: Path | None = None,
    target_path: Path | None = None,
) -> dict[str, Any]:
    """Get information about the feature cache for a dataset.

    Args:
        dataset_id: Unique identifier for the dataset
        reference_path: Path to reference data file (for freshness check)
        target_path: Path to target data file (for freshness check)

    Returns:
        Dictionary with cache info (same structure as get_cache_info)
    """
    cache_path = get_feature_cache_path(dataset_id)
    info: dict[str, Any] = {
        "exists": cache_path.exists(),
        "path": cache_path,
        "version": FEATURE_VERSION,
        "created": None,
        "age_hours": None,
        "is_fresh": None,
        "candidate_count": None,
    }

    if not cache_path.exists():
        return info

    # Get cache modification time
    cache_mtime = cache_path.stat().st_mtime
    cache_datetime = datetime.fromtimestamp(cache_mtime)
    info["created"] = cache_datetime
    info["age_hours"] = (datetime.now() - cache_datetime).total_seconds() / 3600

    # Check freshness against source files
    if reference_path and target_path:
        source_mtimes = []
        if reference_path.exists():
            source_mtimes.append(reference_path.stat().st_mtime)
        if target_path.exists():
            source_mtimes.append(target_path.stat().st_mtime)
        if source_mtimes:
            info["is_fresh"] = cache_mtime > max(source_mtimes)

    # Get candidate count from metadata without loading full file
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(cache_path)
        info["candidate_count"] = metadata.num_rows
    except Exception as e:
        logger.debug(f"Failed to read feature cache metadata from {cache_path}: {e}")

    return info


def load_feature_cache(dataset_id: str) -> pd.DataFrame | None:
    """Load the feature cache (without ML predictions) if available.

    The feature cache contains:
    - Identifiers: ref_id, target_id, ref_idx, target_idx
    - Alignment fractions: ref_start_frac, ref_end_frac, target_start_frac, target_end_frac
    - All computed features from FEATURE_COLUMNS

    Does NOT contain: decision, confidence, score_breakdown, geometries, names, classes.

    Args:
        dataset_id: Unique identifier for the dataset

    Returns:
        DataFrame with features, or None if cache doesn't exist
    """
    cache_path = get_feature_cache_path(dataset_id)
    if not cache_path.exists():
        logger.info(f"No feature cache found for dataset {dataset_id} (version {FEATURE_VERSION})")
        return None

    logger.info(f"Loading feature cache from {cache_path}")
    try:
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded {len(df)} cached features (version {FEATURE_VERSION})")
        return df
    except Exception as e:
        logger.warning(f"Failed to load feature cache for {dataset_id}: {e}")
        return None


def save_feature_cache(dataset_id: str, df: pd.DataFrame) -> Path:
    """Save the feature cache (without ML predictions).

    Args:
        dataset_id: Unique identifier for the dataset
        df: DataFrame with features (must have ref_id, target_id, features)

    Returns:
        Path to the saved cache file
    """
    cache_path = get_feature_cache_path(dataset_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(df)} features to cache at {cache_path}")
    df.to_parquet(cache_path, index=False, compression="zstd")
    logger.info(f"Feature cache saved successfully (version {FEATURE_VERSION})")
    return cache_path


def compute_features_only(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance_m: float = 50.0,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    ref_name_column: str = "names",
    target_name_column: str = "names",
    ref_class_column: str = "class",
    target_class_column: str = "class",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute features for all candidates WITHOUT ML scoring.

    This is the expensive part of the pipeline (~90% of compute time).
    Features are cached separately from ML predictions to allow fast re-scoring.

    Args:
        reference: Reference GeoDataFrame (Overture)
        target: Target GeoDataFrame (local data)
        buffer_distance_m: Search radius in meters
        ref_id_column: ID column in reference
        target_id_column: ID column in target
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        ref_class_column: Class column in reference
        target_class_column: Class column in target
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        DataFrame with columns:
        - ref_id, target_id, ref_idx, target_idx (identifiers)
        - ref_start_frac, ref_end_frac, target_start_frac, target_end_frac (alignment)
        - All features from FEATURE_COLUMNS
    """
    import numpy as np

    from ..config import DEFAULT_TOPOLOGY_FEATURES
    from ..features.alignment import compute_alignment_batch
    from ..features.compute import precompute_graphlet_features
    from ..features.spatial_context import (
        SpatialContextIndex,
        compute_aligned_endpoint_features_batch,
        compute_all_topology,
    )
    from ..matching.ml import _compute_feature_chunk, _init_worker

    # Filter to LineStrings only
    if len(reference) == 0 or len(target) == 0:
        logger.warning("No geometries in reference or target")
        return pd.DataFrame()

    # Project to metric CRS for accurate distances
    projection_result = ensure_projected_crs(reference, target)
    reference_proj = projection_result.reference
    target_proj = projection_result.target
    if projection_result.was_reprojected:
        logger.info(f"Projected to {projection_result.projected_crs} for meter-based computations")

    # Generate candidates (blocking step)
    t0 = time.perf_counter()
    candidates = generate_candidates(
        reference=reference_proj,
        target=target_proj,
        buffer_distance_m=buffer_distance_m,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )
    logger.info(f"Generated {len(candidates)} candidates in {time.perf_counter() - t0:.1f}s")

    if not candidates:
        return pd.DataFrame()

    # Pre-extract data into NumPy arrays
    ref_geoms = reference_proj.geometry.to_numpy()
    target_geoms = target_proj.geometry.to_numpy()
    ref_names = (
        reference[ref_name_column].to_numpy()
        if ref_name_column in reference.columns
        else np.full(len(reference), None, dtype=object)
    )
    target_names = (
        target[target_name_column].to_numpy()
        if target_name_column in target.columns
        else np.full(len(target), None, dtype=object)
    )
    ref_classes = (
        reference[ref_class_column].to_numpy()
        if ref_class_column in reference.columns
        else np.full(len(reference), None, dtype=object)
    )
    target_classes = (
        target[target_class_column].to_numpy()
        if target_class_column in target.columns
        else np.full(len(target), None, dtype=object)
    )
    ref_subclasses = reference.get("subclass", pd.Series([None] * len(reference))).to_numpy()
    target_subclasses = target.get("subclass", pd.Series([None] * len(target))).to_numpy()

    # Get unique indices from candidates
    unique_target_indices = set(cand.target_idx for cand in candidates)
    unique_ref_indices = set(cand.ref_idx for cand in candidates)

    # Pre-compute endpoint features
    # IMPORTANT: Use target_proj (projected CRS) not target (WGS84) to match the
    # CRS of target_geoms which are used as query points. Otherwise the spatial
    # index expects WGS84 inputs but receives UTM coordinates → no matches.
    sorted_target_indices = sorted(unique_target_indices)
    target_candidates_only = target_proj.iloc[sorted_target_indices].reset_index(drop=True)
    original_to_filtered = {orig: filt for filt, orig in enumerate(sorted_target_indices)}

    logger.info("Building spatial index for endpoint features...")
    target_index = SpatialContextIndex()
    target_index.build_from_gdf(target_candidates_only, id_column="id")

    # Pre-compute topology features
    target_ids = target["id"].to_numpy()
    ref_ids = reference["id"].to_numpy()
    unique_target_ids = {str(target_ids[idx]) for idx in unique_target_indices}
    unique_ref_ids = {str(ref_ids[idx]) for idx in unique_ref_indices}

    logger.info(
        f"Computing topology features for {len(unique_target_ids)} target and {len(unique_ref_ids)} reference segments..."
    )

    ref_has_connectors = "connectors" in reference.columns
    target_has_connectors = "connectors" in target.columns

    target_topology_by_id = compute_all_topology(
        target,
        id_column="id",
        tolerance_m=5.0,
        ids_to_compute=unique_target_ids,
        connectors_column="connectors" if target_has_connectors else None,
    )
    ref_topology_by_id = compute_all_topology(
        reference,
        id_column="id",
        tolerance_m=5.0,
        ids_to_compute=unique_ref_ids,
        connectors_column="connectors" if ref_has_connectors else None,
    )

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

    # Pre-compute graphlet features
    sorted_ref_indices = sorted(unique_ref_indices)
    ref_candidates_only = reference_proj.iloc[sorted_ref_indices].reset_index(drop=True)
    target_candidates_only_proj = target_proj.iloc[sorted_target_indices].reset_index(drop=True)

    logger.info(
        f"Computing graphlet features for {len(ref_candidates_only)} reference and {len(target_candidates_only_proj)} target segments..."
    )
    ref_graphlet_data = precompute_graphlet_features(
        ref_candidates_only,
        id_column="id",
        tolerance_m=5.0,
        connectors_column="connectors" if ref_has_connectors else None,
    )
    target_graphlet_data = precompute_graphlet_features(
        target_candidates_only_proj, id_column="id", tolerance_m=5.0
    )

    # Pre-compute linestring alignments
    logger.info("Computing linestring alignments...")
    alignments = compute_alignment_batch(candidates, ref_geoms, target_geoms, n_jobs=n_jobs)

    # Compute endpoint features using alignment fractions
    # This uses aligned subline endpoints instead of full segment endpoints
    # Extract connector data for snapping fractions to real network junctions
    _, target_seg_to_connectors_ep, _, _ = (
        target_graphlet_data if target_graphlet_data else (None, None, None, None)
    )
    aligned_endpoint_features = {}
    if alignments:
        logger.info(f"Computing aligned endpoint features for {len(alignments)} pairs...")
        aligned_endpoint_features = compute_aligned_endpoint_features_batch(
            alignments=alignments,
            target_geoms=target_geoms,
            target_ids=target_ids,
            target_index=target_index,
            original_to_filtered=original_to_filtered,
            seg_to_connectors=target_seg_to_connectors_ep,
        )

    # Build sibling search contexts for parallel sibling detection
    # Use ALL segments, not just candidates - a parallel sibling might not have candidate matches
    from ..features.relational import build_sibling_search_context

    logger.info("Building sibling search contexts...")
    ref_sibling_context = build_sibling_search_context(
        geometries=list(reference_proj.geometry),
        segment_ids=[str(sid) for sid in reference_proj[ref_id_column]],
        names=list(reference_proj.get(ref_name_column, [None] * len(reference_proj))),
        classes=list(reference_proj.get(ref_class_column, [None] * len(reference_proj))),
    )
    target_sibling_context = build_sibling_search_context(
        geometries=list(target_proj.geometry),
        segment_ids=[str(sid) for sid in target_proj[target_id_column]],
        names=list(target_proj.get(target_name_column, [None] * len(target_proj))),
        classes=list(target_proj.get(target_class_column, [None] * len(target_proj))),
    )

    # Determine number of workers
    if n_jobs == -1:
        n_workers = default_worker_count()
    else:
        n_workers = n_jobs

    n_candidates = len(candidates)
    logger.info(f"Computing features for {n_candidates} candidates using {n_workers} processes...")

    # Prepare worker data
    worker_data = {
        "ref_geoms": ref_geoms,
        "target_geoms": target_geoms,
        "ref_names": ref_names,
        "target_names": target_names,
        "ref_classes": ref_classes,
        "target_classes": target_classes,
        "ref_subclasses": ref_subclasses,
        "target_subclasses": target_subclasses,
        "ref_ids": ref_ids,
        "target_ids": target_ids,
        "aligned_endpoint_features": aligned_endpoint_features,
        "ref_topology": ref_topology_features,
        "target_topology": target_topology_features,
        "ref_graphlet_data": ref_graphlet_data,
        "target_graphlet_data": target_graphlet_data,
        "alignments": alignments,
        "ref_sibling_context": ref_sibling_context,
        "target_sibling_context": target_sibling_context,
    }

    work_items = [(cand.ref_idx, cand.target_idx) for cand in candidates]
    chunk_size = max(100, min(1000, n_candidates // (n_workers * 10)))
    features_list = []

    logger.info(f"Starting parallel feature computation (chunk_size={chunk_size})...")

    from concurrent.futures import ProcessPoolExecutor

    # Split work into chunks for batch geometric computation
    chunks = [work_items[i : i + chunk_size] for i in range(0, len(work_items), chunk_size)]

    # Log progress every 5% to avoid noise
    last_logged_pct = 0
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(worker_data,)
    ) as executor:
        for chunk_results, _chunk_errors in executor.map(_compute_feature_chunk, chunks):
            features_list.extend(chunk_results)
            processed = len(features_list)
            pct = int(processed / len(work_items) * 100)
            if pct >= last_logged_pct + 5:
                logger.info(f"Feature computation: {processed:,}/{len(work_items):,} ({pct}%)")
                last_logged_pct = pct

    # Filter out rejected pairs (None results) - pairs without aligned endpoint features
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

    # Build DataFrame with identifiers, alignment fractions, and features
    records = []
    for cand, features in valid_pairs:
        alignment = alignments.get((cand.ref_idx, cand.target_idx))

        record = {
            "ref_id": str(cand.ref_id),
            "target_id": str(cand.target_id),
            "ref_idx": cand.ref_idx,
            "target_idx": cand.target_idx,
            "ref_start_frac": alignment.overture_start_frac if alignment else 0.0,
            "ref_end_frac": alignment.overture_end_frac if alignment else 1.0,
            "target_start_frac": alignment.dataset_start_frac if alignment else 0.0,
            "target_end_frac": alignment.dataset_end_frac if alignment else 1.0,
        }

        # Add all feature columns
        for col in FEATURE_COLUMNS:
            record[col] = features.get(col, 0.0)

        records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Built feature DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df


def load_geodataframe(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoDataFrame from parquet or other formats."""
    if path.suffix == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)

    # Ensure CRS is set (default to WGS84 if missing)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Filter to LineString geometries only (drop MultiLineStrings)
    gdf = filter_to_linestrings(gdf, source_name=str(path.name))

    return gdf


def generate_scored_candidates(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance_m: float = 50.0,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    ref_name_column: str = "names",
    target_name_column: str = "names",
    ref_class_column: str = "class",
    target_class_column: str = "class",
) -> list[CandidatePairView]:
    """Generate and score candidates using the optimized ML pipeline.

    Uses MLMatcher.score_candidates() which provides:
    - Parallel feature computation via ProcessPoolExecutor
    - Pre-computed topology/endpoints/alignments
    - Batch XGBoost prediction

    Args:
        reference: Reference GeoDataFrame (Overture)
        target: Target GeoDataFrame (local data)
        buffer_distance_m: Search radius in meters
        ref_id_column: ID column in reference
        target_id_column: ID column in target
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        ref_class_column: Class column in reference
        target_class_column: Class column in target

    Returns:
        List of CandidatePairView objects sorted by decision priority, then confidence
    """
    # Data should already be filtered to LineStrings at load time
    if len(reference) == 0 or len(target) == 0:
        logger.warning("No geometries in reference or target")
        return []

    # Project to metric CRS for accurate distances
    projection_result = ensure_projected_crs(reference, target)
    reference_proj = projection_result.reference
    target_proj = projection_result.target
    if projection_result.was_reprojected:
        logger.info(f"Projected to {projection_result.projected_crs} for meter-based computations")

    # Create transformer for converting aligned geometries back to WGS84
    proj_to_wgs84 = None
    if projection_result.original_crs and projection_result.projected_crs:
        proj_to_wgs84 = Transformer.from_crs(
            projection_result.projected_crs, projection_result.original_crs, always_xy=True
        ).transform

    # Generate candidates (blocking step - already fast)
    t0 = time.perf_counter()
    candidates = generate_candidates(
        reference=reference_proj,
        target=target_proj,
        buffer_distance_m=buffer_distance_m,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )
    logger.info(f"Generated {len(candidates)} candidates in {time.perf_counter() - t0:.1f}s")

    if not candidates:
        return []

    # Use MLMatcher for optimized scoring (parallel feature computation + batch prediction)
    t0 = time.perf_counter()

    # Check if model exists
    model_path = settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"ML model not found at {model_path}. Run 'matcher train' to train the model."
        )

    matcher = MLMatcher(auto_select=True)

    # Score all candidates using ML pipeline (parallelized)
    match_results = matcher.score_candidates(
        candidates=candidates,
        reference=reference_proj,
        target=target_proj,
        ref_name_column=ref_name_column,
        target_name_column=target_name_column,
        ref_class_column=ref_class_column,
        target_class_column=target_class_column,
    )
    logger.info(f"ML scoring completed in {time.perf_counter() - t0:.1f}s")

    # Convert MatchResult objects to CandidatePairView objects
    t0 = time.perf_counter()

    # Build lookups for geometry and attribute access
    ref_lookup = reference.set_index(ref_id_column)
    target_lookup = target.set_index(target_id_column)
    ref_proj_lookup = reference_proj.set_index(ref_id_column)
    target_proj_lookup = target_proj.set_index(target_id_column)

    has_ref_name = ref_name_column in reference.columns
    has_target_name = target_name_column in target.columns
    has_ref_class = ref_class_column in reference.columns
    has_target_class = target_class_column in target.columns
    has_ref_subclass = "subclass" in reference.columns
    has_target_subclass = "subclass" in target.columns

    def get_row(lookup, id_val):
        """Get single row from lookup, handling duplicate IDs."""
        result = lookup.loc[[id_val]]
        if len(result) == 0:
            raise KeyError(f"ID {id_val} not found in lookup")
        return result.iloc[0]

    views = []
    total_results = len(match_results)
    logger.info(f"Converting {total_results:,} MatchResults to CandidatePairViews...")
    for i, result in enumerate(match_results):
        # Look up original geometries
        ref_row = get_row(ref_lookup, result.ref_id)
        target_row = get_row(target_lookup, result.target_id)

        # Extract names and classes
        ref_name = _extract_name_string(ref_row.get(ref_name_column)) if has_ref_name else None
        target_name = (
            _extract_name_string(target_row.get(target_name_column)) if has_target_name else None
        )
        ref_class = ref_row.get(ref_class_column) if has_ref_class else None
        target_class = target_row.get(target_class_column) if has_target_class else None
        ref_subclass = ref_row.get("subclass") if has_ref_subclass else None
        target_subclass = target_row.get("subclass") if has_target_subclass else None

        # Get alignment fractions from MatchResult
        ref_start_frac = result.gers_start_frac if result.gers_start_frac is not None else 0.0
        ref_end_frac = result.gers_end_frac if result.gers_end_frac is not None else 1.0
        target_start_frac = result.local_start_frac if result.local_start_frac is not None else 0.0
        target_end_frac = result.local_end_frac if result.local_end_frac is not None else 1.0

        # Create aligned geometries for map display
        ref_proj_row = get_row(ref_proj_lookup, result.ref_id)
        target_proj_row = get_row(target_proj_lookup, result.target_id)

        ref_aligned_proj = create_subline(ref_proj_row.geometry, ref_start_frac, ref_end_frac)
        target_aligned_proj = create_subline(
            target_proj_row.geometry, target_start_frac, target_end_frac
        )

        # Transform aligned geometries back to WGS84
        if proj_to_wgs84:
            ref_aligned = transform(proj_to_wgs84, ref_aligned_proj)
            target_aligned = transform(proj_to_wgs84, target_aligned_proj)
        else:
            ref_aligned = ref_aligned_proj
            target_aligned = target_aligned_proj

        # Convert decision enum to string
        decision = result.decision.value  # MatchDecision enum -> string

        # Compute score_breakdown from features for UI display
        # ML scorer doesn't provide component scores, so we derive them
        score_breakdown = _compute_score_breakdown_from_features(result.features)

        views.append(
            CandidatePairView(
                ref_id=str(result.ref_id),
                target_id=str(result.target_id),
                ref_geometry=ref_row.geometry,
                target_geometry=target_row.geometry,
                ref_name=ref_name,
                target_name=target_name,
                ref_class=ref_class,
                target_class=target_class,
                ref_subclass=ref_subclass,
                target_subclass=target_subclass,
                decision=decision,
                confidence=result.confidence,
                score_breakdown=score_breakdown,
                features=result.features,
                ref_aligned_geometry=ref_aligned,
                target_aligned_geometry=target_aligned,
                ref_start_frac=ref_start_frac,
                ref_end_frac=ref_end_frac,
                target_start_frac=target_start_frac,
                target_end_frac=target_end_frac,
            )
        )

        # Progress logging every 50k
        if (i + 1) % 50000 == 0:
            logger.info(f"View conversion: {i + 1:,}/{total_results:,}")

    logger.info(
        f"Built {len(views):,} CandidatePairView objects in {time.perf_counter() - t0:.1f}s"
    )

    # Sort: REVIEW first, then by confidence descending
    logger.info(f"Sorting {len(views):,} views...")

    def sort_key(v):
        decision_order = {"review": 0, "match": 1, "no_match": 2}
        return (decision_order.get(v.decision, 3), -v.confidence)

    views.sort(key=sort_key)
    logger.info("Sorting complete")

    return views


def generate_scored_candidates_with_cache(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    dataset_id: str,
    buffer_distance_m: float = 50.0,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    ref_name_column: str = "names",
    target_name_column: str = "names",
    ref_class_column: str = "class",
    target_class_column: str = "class",
    force_recompute: bool = False,
    n_jobs: int = -1,
) -> list[CandidatePairView]:
    """Generate and score candidates using feature cache when available.

    This is a two-stage caching flow:
    1. Check feature cache (versioned) - contains IDs + features only
    2. If miss: compute features -> save feature cache
    3. Run ML scoring on features (fast)
    4. Build CandidatePairView objects by joining with source GeoDataFrames

    Args:
        reference: Reference GeoDataFrame (Overture)
        target: Target GeoDataFrame (local data)
        dataset_id: Unique identifier for caching
        buffer_distance_m: Search radius in meters
        ref_id_column: ID column in reference
        target_id_column: ID column in target
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        ref_class_column: Class column in reference
        target_class_column: Class column in target
        force_recompute: If True, recompute features even if cached
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        List of CandidatePairView objects sorted by uncertainty (most uncertain first)
    """
    if len(reference) == 0 or len(target) == 0:
        logger.warning("No geometries in reference or target")
        return []

    logger.info(f"=== Loading dataset: {dataset_id} ===")
    logger.info(f"Reference: {len(reference):,} segments, Target: {len(target):,} segments")
    t_total = time.perf_counter()

    # Check for feature cache
    feature_df = None
    if not force_recompute:
        logger.info("Checking for feature cache...")
        feature_df = load_feature_cache(dataset_id)

    # Compute features if not cached
    if feature_df is None:
        logger.info(f"No feature cache found - computing features for {dataset_id}...")
        feature_df = compute_features_only(
            reference=reference,
            target=target,
            buffer_distance_m=buffer_distance_m,
            ref_id_column=ref_id_column,
            target_id_column=target_id_column,
            ref_name_column=ref_name_column,
            target_name_column=target_name_column,
            ref_class_column=ref_class_column,
            target_class_column=target_class_column,
            n_jobs=n_jobs,
        )

        if len(feature_df) == 0:
            return []

        # Save feature cache
        save_feature_cache(dataset_id, feature_df)
    else:
        logger.info(f"Feature cache hit: {len(feature_df):,} candidates")

    # Run ML scoring on features (filter to review band for labeling UI)
    views = build_views_from_feature_df(
        feature_df=feature_df,
        reference=reference,
        target=target,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
        ref_name_column=ref_name_column,
        target_name_column=target_name_column,
        ref_class_column=ref_class_column,
        target_class_column=target_class_column,
        filter_to_review_band=True,
    )

    logger.info(f"=== Dataset {dataset_id} loaded in {time.perf_counter() - t_total:.1f}s ===")
    return views


def build_views_from_feature_df(
    feature_df: pd.DataFrame,
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    ref_name_column: str = "names",
    target_name_column: str = "names",
    ref_class_column: str = "class",
    target_class_column: str = "class",
    filter_to_review_band: bool = False,
) -> list[CandidatePairView]:
    """Build CandidatePairView objects from a feature DataFrame.

    Runs ML prediction on features and joins with source GeoDataFrames
    to get geometries, names, and classes.

    Args:
        feature_df: DataFrame with ref_id, target_id, alignment fractions, and features
        reference: Reference GeoDataFrame for geometry/attribute lookup
        target: Target GeoDataFrame for geometry/attribute lookup
        ref_id_column: ID column in reference
        target_id_column: ID column in target
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        ref_class_column: Class column in reference
        target_class_column: Class column in target
        filter_to_review_band: If True, only build views for candidates
            in the review band. If False (default), build views for ALL candidates.

    Returns:
        List of CandidatePairView objects sorted by uncertainty (most uncertain first)
    """
    if len(feature_df) == 0:
        return []

    total = len(feature_df)
    logger.info(f"[1/5] Building views for {total:,} candidates from feature cache...")

    # Project to metric CRS for aligned geometry computation
    logger.info("[2/5] Projecting geometries to metric CRS...")
    t_proj = time.perf_counter()
    projection_result = ensure_projected_crs(reference, target)
    reference_proj = projection_result.reference
    target_proj = projection_result.target

    proj_to_wgs84 = None
    if projection_result.original_crs and projection_result.projected_crs:
        proj_to_wgs84 = Transformer.from_crs(
            projection_result.projected_crs, projection_result.original_crs, always_xy=True
        ).transform
    logger.info(f"[2/5] Projection completed in {time.perf_counter() - t_proj:.1f}s")

    # Convert features to list of dicts for ML prediction
    feature_cols = [col for col in feature_df.columns if col in FEATURE_COLUMNS]
    features_list = feature_df[feature_cols].to_dict("records")

    # Run ML prediction using cached matcher
    logger.info(f"[3/5] Running ML prediction on {total:,} candidates...")
    t0 = time.perf_counter()
    matcher = get_cached_matcher()
    if matcher is None:
        raise ValueError(
            "No ML model available. Train a model first with 'matcher train'. "
            "The labeling UI requires a trained model for scoring candidates."
        )
    probs = matcher.predict(features_list)
    logger.info(f"[3/5] ML prediction completed in {time.perf_counter() - t0:.1f}s")

    # Filter to review band BEFORE building views (huge speedup for large datasets)
    # Review band: review_threshold to match_threshold (no buffer beyond thresholds —
    # pairs above match_threshold are already auto-matched, pairs below review are no_match)
    import numpy as np

    probs_arr = np.array(probs)

    if filter_to_review_band:
        review_lower = settings.optimizer_review_threshold
        review_upper = settings.optimizer_match_threshold

        # Create mask for review band
        review_mask = (probs_arr >= review_lower) & (probs_arr <= review_upper)
        review_count = review_mask.sum()

        logger.info(
            f"[3.5/5] Filtering to review band ({review_lower:.2f}-{review_upper:.2f}): {review_count:,}/{total:,} candidates"
        )

        # Filter DataFrame and probs BEFORE materializing to dicts (speedup for large datasets)
        filtered_df = feature_df[review_mask].reset_index(drop=True)
        filtered_records = filtered_df.to_dict("records")
        filtered_probs = probs_arr[review_mask].tolist()
    else:
        logger.info(f"[3.5/5] Skipping review band filter — keeping all {total:,} candidates")
        filtered_records = feature_df.to_dict("records")
        filtered_probs = probs_arr.tolist()

    # Update total for progress reporting
    total = len(filtered_records)
    logger.info(
        f"[3.5/5] Will build views for {total:,} candidates (filtered from {len(feature_df):,})"
    )

    # Build fast dict lookups for geometry and attribute access (much faster than .loc)
    logger.info("[4/5] Building geometry lookup dictionaries...")
    t_lookup = time.perf_counter()

    # Convert to dicts for O(1) lookup - drop duplicates keeping first occurrence
    ref_records = (
        reference.drop_duplicates(subset=[ref_id_column], keep="first")
        .set_index(ref_id_column)
        .to_dict("index")
    )
    target_records = (
        target.drop_duplicates(subset=[target_id_column], keep="first")
        .set_index(target_id_column)
        .to_dict("index")
    )
    ref_proj_records = (
        reference_proj.drop_duplicates(subset=[ref_id_column], keep="first")
        .set_index(ref_id_column)
        .to_dict("index")
    )
    target_proj_records = (
        target_proj.drop_duplicates(subset=[target_id_column], keep="first")
        .set_index(target_id_column)
        .to_dict("index")
    )

    logger.info(f"[4/5] Built lookups in {time.perf_counter() - t_lookup:.1f}s")

    has_ref_name = ref_name_column in reference.columns
    has_target_name = target_name_column in target.columns
    has_ref_class = ref_class_column in reference.columns
    has_target_class = target_class_column in target.columns
    has_ref_subclass = "subclass" in reference.columns
    has_target_subclass = "subclass" in target.columns

    views = []
    skipped_count = 0
    logger.info(f"[5/5] Building {total:,} CandidatePairView objects with aligned geometries...")
    t_views = time.perf_counter()

    # Use filtered records and probs (already filtered to review band)
    for i, row in enumerate(filtered_records):
        if i > 0 and i % 10000 == 0:
            elapsed = time.perf_counter() - t_views
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            logger.info(
                f"[5/5] Progress: {i:,}/{total:,} ({100 * i / total:.0f}%) - {remaining:.0f}s remaining"
            )
        ref_id = row["ref_id"]
        target_id = row["target_id"]
        prob = filtered_probs[i]

        # Look up geometries using fast dict access
        ref_data = ref_records.get(ref_id)
        target_data = target_records.get(target_id)

        if ref_data is None or target_data is None:
            skipped_count += 1
            continue

        # Extract names and classes
        ref_name = _extract_name_string(ref_data.get(ref_name_column)) if has_ref_name else None
        target_name = (
            _extract_name_string(target_data.get(target_name_column)) if has_target_name else None
        )
        ref_class = ref_data.get(ref_class_column) if has_ref_class else None
        target_class = target_data.get(target_class_column) if has_target_class else None
        ref_subclass = ref_data.get("subclass") if has_ref_subclass else None
        target_subclass = target_data.get("subclass") if has_target_subclass else None

        # Get alignment fractions from cached data
        ref_start_frac = row.get("ref_start_frac", 0.0)
        ref_end_frac = row.get("ref_end_frac", 1.0)
        target_start_frac = row.get("target_start_frac", 0.0)
        target_end_frac = row.get("target_end_frac", 1.0)

        # Create aligned geometries
        ref_proj_data = ref_proj_records.get(ref_id)
        target_proj_data = target_proj_records.get(target_id)

        if ref_proj_data is not None and target_proj_data is not None:
            ref_aligned_proj = create_subline(
                ref_proj_data["geometry"], ref_start_frac, ref_end_frac
            )
            target_aligned_proj = create_subline(
                target_proj_data["geometry"], target_start_frac, target_end_frac
            )

            if proj_to_wgs84:
                ref_aligned = (
                    transform(proj_to_wgs84, ref_aligned_proj)
                    if ref_aligned_proj is not None
                    else None
                )
                target_aligned = (
                    transform(proj_to_wgs84, target_aligned_proj)
                    if target_aligned_proj is not None
                    else None
                )
            else:
                ref_aligned = ref_aligned_proj
                target_aligned = target_aligned_proj
        else:
            ref_aligned = None
            target_aligned = None

        # Determine decision from confidence using configurable thresholds
        if prob >= settings.optimizer_match_threshold:
            decision = "match"
        elif prob >= settings.optimizer_review_threshold:
            decision = "review"
        else:
            decision = "no_match"

        # Build features dict from row (row is now a dict, not Series)
        features = {col: row[col] for col in feature_cols if col in row}

        # Compute score_breakdown from features for UI display
        score_breakdown = _compute_score_breakdown_from_features(features)

        views.append(
            CandidatePairView(
                ref_id=str(ref_id),
                target_id=str(target_id),
                ref_geometry=ref_data["geometry"],
                target_geometry=target_data["geometry"],
                ref_name=ref_name,
                target_name=target_name,
                ref_class=ref_class,
                target_class=target_class,
                ref_subclass=ref_subclass,
                target_subclass=target_subclass,
                decision=decision,
                confidence=prob,
                score_breakdown=score_breakdown,
                features=features,
                ref_aligned_geometry=ref_aligned,
                target_aligned_geometry=target_aligned,
                ref_start_frac=ref_start_frac,
                ref_end_frac=ref_end_frac,
                target_start_frac=target_start_frac,
                target_end_frac=target_end_frac,
            )
        )

    logger.info(f"[5/5] Built {len(views):,} views in {time.perf_counter() - t_views:.1f}s")

    if skipped_count > 0:
        logger.warning(
            f"Skipped {skipped_count} candidates due to missing geometries in source data "
            "(feature cache may be stale)"
        )

    # Sort: REVIEW first, then by uncertainty (closest to band midpoint first)
    # This surfaces the most informative pairs — where the model is genuinely unsure
    logger.info("Sorting views by uncertainty (most uncertain first)...")
    band_midpoint = (settings.optimizer_review_threshold + settings.optimizer_match_threshold) / 2

    def sort_key(v):
        decision_order = {"review": 0, "match": 1, "no_match": 2}
        return (decision_order.get(v.decision, 3), abs(v.confidence - band_midpoint))

    views.sort(key=sort_key)
    logger.info(f"View building complete: {len(views):,} candidates ready")

    return views


def filter_candidates(
    views: list[CandidatePairView],
    decision_filter: str | None = None,
    labeled_pairs: set[tuple[str, str]] | None = None,
    show_labeled: bool = False,
    specific_pairs: list[tuple[str, str]] | None = None,
) -> list[CandidatePairView]:
    """Filter candidate views based on criteria.

    Args:
        views: List of candidate views
        decision_filter: Only show this decision type ("match", "review", "no_match")
        labeled_pairs: Set of already-labeled (ref_id, target_id) pairs
        show_labeled: If False, exclude already-labeled pairs
        specific_pairs: If provided, only show these specific (ref_id, target_id) pairs

    Returns:
        Filtered list of views
    """
    filtered = views

    # Filter to specific pairs if provided (e.g., for reviewing disagreements)
    if specific_pairs:
        specific_set = set(specific_pairs)
        filtered = [v for v in filtered if (v.ref_id, v.target_id) in specific_set]

    # Filter by decision
    if decision_filter:
        filtered = [v for v in filtered if v.decision == decision_filter]

    # Exclude already-labeled
    if labeled_pairs and not show_labeled:
        filtered = [v for v in filtered if (v.ref_id, v.target_id) not in labeled_pairs]

    return filtered


def filter_by_confidence_band(
    views: list[CandidatePairView],
    review_only: bool = True,
) -> list[CandidatePairView]:
    """Filter candidates to the review band or return all.

    The review band uses the production thresholds (review_threshold to
    match_threshold) without additional buffer, focusing labelers on pairs
    where the model is most uncertain.

    Args:
        views: List of candidate views
        review_only: If True, filter to review band. If False, return all.

    Returns:
        Filtered list of views within the confidence band
    """
    if not review_only:
        return views

    # Use production thresholds from settings (no buffer — stay within the review band)
    lower = settings.optimizer_review_threshold
    upper = settings.optimizer_match_threshold

    return [v for v in views if lower <= v.confidence <= upper]
