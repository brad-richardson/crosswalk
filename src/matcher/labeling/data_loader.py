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
from ..config import settings
from ..features.alignment import create_subline
from ..features.semantic import _extract_name_string
from ..matching.ml import MLMatcher
from ..utils import filter_to_linestrings

logger = logging.getLogger(__name__)

# Cache directory relative to project root (data/cache/labeling/)
# Path: src/matcher/labeling/data_loader.py -> parents[3] = project root
CACHE_DIR = Path(__file__).parents[3] / "data" / "cache" / "labeling"


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
        "projection_norm": norm_distance(features.get("projection_distance_m", 50)),
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
    return CACHE_DIR / f"{dataset_id}_candidates.parquet"


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
    df.to_parquet(cache_path, index=False)

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
        List of CandidatePairView objects sorted by confidence (REVIEW first)
    """
    # Data should already be filtered to LineStrings at load time
    if len(reference) == 0 or len(target) == 0:
        logger.warning("No geometries in reference or target")
        return []

    # Project to metric CRS for accurate distances
    original_crs = reference.crs
    projected_crs = None
    if reference.crs and reference.crs.is_geographic:
        centroid = reference.unary_union.centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        projected_crs = f"EPSG:326{utm_zone:02d}" if centroid.y >= 0 else f"EPSG:327{utm_zone:02d}"
        reference_proj = reference.to_crs(projected_crs)
        target_proj = target.to_crs(projected_crs)
    else:
        reference_proj = reference
        target_proj = target

    # Create transformer for converting aligned geometries back to WGS84
    proj_to_wgs84 = None
    if projected_crs and original_crs:
        proj_to_wgs84 = Transformer.from_crs(projected_crs, original_crs, always_xy=True).transform

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
    matcher = MLMatcher(auto_select=True)

    # Check if model exists, fall back to rules if not
    model_path = settings.model_path
    if not model_path.exists():
        logger.warning(f"ML model not found at {model_path}, falling back to rule-based scoring")
        return _generate_scored_candidates_rules(
            reference=reference,
            target=target,
            reference_proj=reference_proj,
            target_proj=target_proj,
            candidates=candidates,
            proj_to_wgs84=proj_to_wgs84,
            ref_id_column=ref_id_column,
            target_id_column=target_id_column,
            ref_name_column=ref_name_column,
            target_name_column=target_name_column,
            ref_class_column=ref_class_column,
            target_class_column=target_class_column,
        )

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

    logger.info(f"Built {len(views):,} CandidatePairView objects in {time.perf_counter() - t0:.1f}s")

    # Sort: REVIEW first, then by confidence descending
    logger.info(f"Sorting {len(views):,} views...")
    def sort_key(v):
        decision_order = {"review": 0, "match": 1, "no_match": 2}
        return (decision_order.get(v.decision, 3), -v.confidence)

    views.sort(key=sort_key)
    logger.info("Sorting complete")

    return views


def _generate_scored_candidates_rules(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    reference_proj: gpd.GeoDataFrame,
    target_proj: gpd.GeoDataFrame,
    candidates: list,
    proj_to_wgs84,
    ref_id_column: str,
    target_id_column: str,
    ref_name_column: str,
    target_name_column: str,
    ref_class_column: str,
    target_class_column: str,
) -> list[CandidatePairView]:
    """Fallback: generate scored candidates using rule-based scoring.

    Used when ML model is not available. This is the original sequential
    implementation, which is slower but doesn't require a trained model.
    """
    from ..features.alignment import linestring_alignment
    from ..features.compute import compute_pair_features, precompute_topology_and_endpoints
    from ..matching.rules import compute_match_score

    # Build lookups
    ref_lookup = reference.set_index(ref_id_column)
    target_lookup = target.set_index(target_id_column)
    ref_proj_lookup = reference_proj.set_index(ref_id_column)
    target_proj_lookup = target_proj.set_index(target_id_column)

    has_ref_name = ref_name_column in reference.columns
    has_target_name = target_name_column in target.columns
    has_ref_class = ref_class_column in reference.columns
    has_target_class = target_class_column in target.columns

    def get_row(lookup, id_val):
        result = lookup.loc[[id_val]]
        if len(result) == 0:
            raise KeyError(f"ID {id_val} not found in lookup")
        return result.iloc[0]

    # Pre-compute topology and endpoint features
    logger.info("Pre-computing topology and endpoint features...")
    unique_ref_indices = {cand.ref_idx for cand in candidates}
    unique_target_indices = {cand.target_idx for cand in candidates}

    target_endpoint_features, ref_topology_features, target_topology_features = (
        precompute_topology_and_endpoints(
            reference=reference_proj,
            target=target_proj,
            ref_indices=unique_ref_indices,
            target_indices=unique_target_indices,
            id_column=ref_id_column,
            tolerance_m=5.0,
        )
    )

    logger.info(f"Computing features for {len(candidates)} candidates (rule-based fallback)...")
    views = []

    for i, cand in enumerate(candidates):
        if i > 0 and i % 10000 == 0:
            logger.info(f"  Progress: {i}/{len(candidates)} ({100 * i / len(candidates):.0f}%)")

        ref_row = get_row(ref_lookup, cand.ref_id)
        target_row = get_row(target_lookup, cand.target_id)
        ref_proj_row = get_row(ref_proj_lookup, cand.ref_id)
        target_proj_row = get_row(target_proj_lookup, cand.target_id)

        ref_name = _extract_name_string(ref_row.get(ref_name_column)) if has_ref_name else None
        target_name = (
            _extract_name_string(target_row.get(target_name_column)) if has_target_name else None
        )
        ref_class = ref_row.get(ref_class_column) if has_ref_class else None
        target_class = target_row.get(target_class_column) if has_target_class else None

        alignment = linestring_alignment(ref_proj_row.geometry, target_proj_row.geometry)

        features = compute_pair_features(
            ref_geom=ref_proj_row.geometry,
            target_geom=target_proj_row.geometry,
            ref_name=ref_name,
            target_name=target_name,
            ref_class=ref_class,
            target_class=target_class,
            endpoint_features=target_endpoint_features.get(cand.target_idx),
            ref_topology=ref_topology_features.get(cand.ref_idx),
            target_topology=target_topology_features.get(cand.target_idx),
            alignment=alignment,
        )

        confidence, score_breakdown, _ = compute_match_score(
            ref_geom=ref_proj_row.geometry,
            target_geom=target_proj_row.geometry,
            ref_name=ref_name,
            target_name=target_name,
            ref_class=ref_class,
            target_class=target_class,
            precomputed_features=features,
        )

        if confidence >= 0.5:
            decision = "match"
        elif confidence >= 0.1:
            decision = "review"
        else:
            decision = "no_match"

        ref_aligned_proj = create_subline(
            ref_proj_row.geometry, alignment.overture_start_frac, alignment.overture_end_frac
        )
        target_aligned_proj = create_subline(
            target_proj_row.geometry, alignment.dataset_start_frac, alignment.dataset_end_frac
        )

        if proj_to_wgs84:
            ref_aligned = transform(proj_to_wgs84, ref_aligned_proj)
            target_aligned = transform(proj_to_wgs84, target_aligned_proj)
        else:
            ref_aligned = ref_aligned_proj
            target_aligned = target_aligned_proj

        views.append(
            CandidatePairView(
                ref_id=str(cand.ref_id),
                target_id=str(cand.target_id),
                ref_geometry=ref_row.geometry,
                target_geometry=target_row.geometry,
                ref_name=ref_name,
                target_name=target_name,
                ref_class=ref_class,
                target_class=target_class,
                decision=decision,
                confidence=confidence,
                score_breakdown=score_breakdown,
                features=features,
                ref_aligned_geometry=ref_aligned,
                target_aligned_geometry=target_aligned,
                ref_start_frac=alignment.overture_start_frac,
                ref_end_frac=alignment.overture_end_frac,
                target_start_frac=alignment.dataset_start_frac,
                target_end_frac=alignment.dataset_end_frac,
            )
        )

    # Sort: REVIEW first, then by confidence descending
    def sort_key(v):
        decision_order = {"review": 0, "match": 1, "no_match": 2}
        return (decision_order.get(v.decision, 3), -v.confidence)

    views.sort(key=sort_key)

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
    buffer: float = 0.05,
) -> list[CandidatePairView]:
    """Filter candidates to review band (near decision boundaries) or return all.

    The review band catches edge cases near the production thresholds by adding
    a small buffer on each side. This helps labelers focus on the most uncertain
    cases that would most benefit from human review.

    Args:
        views: List of candidate views
        review_only: If True, filter to review band. If False, return all.
        buffer: Buffer to add around thresholds (default 0.05)

    Returns:
        Filtered list of views within the confidence band
    """
    if not review_only:
        return views

    # Use production thresholds from settings
    lower = settings.review_threshold - buffer  # e.g., 0.50 - 0.05 = 0.45
    upper = settings.match_threshold + buffer  # e.g., 0.75 + 0.05 = 0.80

    return [v for v in views if lower <= v.confidence <= upper]
