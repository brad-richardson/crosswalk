"""Data loading and candidate preparation for labeling UI."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString

from ..blocking import generate_candidates
from ..features.compute import (
    compute_pair_features,
    precompute_topology_and_endpoints,
)
from ..features.semantic import _extract_name_string
from ..matching.rules import compute_match_score
from .subsegment import estimate_overlap_range

logger = logging.getLogger(__name__)

# Cache directory relative to project root (data/cache/labeling/)
# Path: src/matcher/labeling/data_loader.py -> parents[3] = project root
CACHE_DIR = Path(__file__).parents[3] / "data" / "cache" / "labeling"


@dataclass
class CandidatePairView:
    """View model for a single candidate pair in the labeling UI."""

    ref_id: str
    target_id: str
    ref_geometry: LineString
    target_geometry: LineString
    ref_name: str | None
    target_name: str | None
    ref_class: str | None
    target_class: str | None
    decision: str  # "match", "review", "no_match"
    confidence: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    # Sub-segment estimate - computed on-demand when viewing the pair
    # None means not yet computed, dict contains the estimated ranges
    estimated_subsegment: dict[str, float] | None = None

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
            # Subsegment estimate is always computed on-demand, not cached
            estimated_subsegment=None,
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
        candidates = [CandidatePairView.from_dict(row) for _, row in df.iterrows()]
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


def _filter_linestrings(gdf: gpd.GeoDataFrame, source_name: str) -> gpd.GeoDataFrame:
    """Filter GeoDataFrame to only LineString geometries.

    Args:
        gdf: Input GeoDataFrame
        source_name: Name for logging (e.g., "reference" or "target")

    Returns:
        Filtered GeoDataFrame with only LineString geometries
    """
    original_count = len(gdf)
    mask = gdf.geometry.apply(lambda g: isinstance(g, LineString))
    filtered = gdf[mask].copy()
    excluded_count = original_count - len(filtered)

    if excluded_count > 0:
        logger.warning(
            f"Excluded {excluded_count} non-LineString geometries from {source_name} "
            f"({excluded_count}/{original_count} features). "
            f"Sub-segment selection only supports LineString geometries."
        )

    return filtered


def load_geodataframe(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoDataFrame from parquet or other formats."""
    if path.suffix == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)

    # Ensure CRS is set (default to WGS84 if missing)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    return gdf


def generate_scored_candidates(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance: float = 50.0,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    ref_name_column: str = "names",
    target_name_column: str = "names",
    ref_class_column: str = "class",
    target_class_column: str = "class",
) -> list[CandidatePairView]:
    """Generate and score candidates, returning view models for UI.

    Args:
        reference: Reference GeoDataFrame (Overture)
        target: Target GeoDataFrame (local data)
        buffer_distance: Search radius in meters
        ref_id_column: ID column in reference
        target_id_column: ID column in target
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        ref_class_column: Class column in reference
        target_class_column: Class column in target

    Returns:
        List of CandidatePairView objects sorted by confidence (REVIEW first)
    """
    # Filter out non-LineString geometries (MultiLineString, etc.)
    # Sub-segment selection only supports LineString
    reference = _filter_linestrings(reference, "reference")
    target = _filter_linestrings(target, "target")

    if len(reference) == 0 or len(target) == 0:
        logger.warning("No valid LineString geometries after filtering")
        return []

    # Project to metric CRS for accurate distances
    if reference.crs and reference.crs.is_geographic:
        centroid = reference.unary_union.centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone:02d}" if centroid.y >= 0 else f"EPSG:327{utm_zone:02d}"
        reference_proj = reference.to_crs(utm_crs)
        target_proj = target.to_crs(utm_crs)
    else:
        reference_proj = reference
        target_proj = target

    # Generate candidates
    candidates = generate_candidates(
        reference=reference_proj,
        target=target_proj,
        buffer_distance=buffer_distance,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )

    if not candidates:
        return []

    # Build view models (use original WGS84 geometries for map display)
    # Create index lookups for O(1) access (avoid O(n) DataFrame filtering per candidate)
    # Note: If IDs are not unique, .loc returns DataFrame; we take first row
    ref_lookup = reference.set_index(ref_id_column)
    target_lookup = target.set_index(target_id_column)
    ref_proj_lookup = reference_proj.set_index(ref_id_column)
    target_proj_lookup = target_proj.set_index(target_id_column)

    # Pre-check column existence
    has_ref_name = ref_name_column in reference.columns
    has_target_name = target_name_column in target.columns
    has_ref_class = ref_class_column in reference.columns
    has_target_class = target_class_column in target.columns

    def get_row(lookup, id_val):
        """Get single row from lookup, handling duplicate IDs and single-column DataFrames."""
        # Use [[id_val]] to always get a DataFrame, then take first row as Series
        # This handles: duplicate IDs, single-column DataFrames (where .loc returns scalar)
        result = lookup.loc[[id_val]]
        if len(result) == 0:
            raise KeyError(f"ID {id_val} not found in lookup")
        return result.iloc[0]

    # Pre-compute topology and endpoint features for all candidate pairs
    # This is the same approach used by the ML pipeline for efficiency
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
            tolerance=5.0,
        )
    )

    logger.info(f"Computing features for {len(candidates)} candidates...")
    views = []
    for cand in candidates:
        # Get rows from indexed lookups
        ref_row = get_row(ref_lookup, cand.ref_id)
        target_row = get_row(target_lookup, cand.target_id)
        ref_proj_row = get_row(ref_proj_lookup, cand.ref_id)
        target_proj_row = get_row(target_proj_lookup, cand.target_id)

        # Extract names
        ref_name = _extract_name_string(ref_row.get(ref_name_column)) if has_ref_name else None
        target_name = (
            _extract_name_string(target_row.get(target_name_column)) if has_target_name else None
        )

        # Extract classes
        ref_class = ref_row.get(ref_class_column) if has_ref_class else None
        target_class = target_row.get(target_class_column) if has_target_class else None

        # Compute all features using shared module (uses projected geometries)
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
        )

        # Compute confidence and decision using rule-based scoring
        confidence, score_breakdown, _ = compute_match_score(
            ref_geom=ref_proj_row.geometry,
            target_geom=target_proj_row.geometry,
            ref_name=ref_name,
            target_name=target_name,
            ref_class=ref_class,
            target_class=target_class,
        )

        # Determine decision based on confidence
        if confidence >= 0.5:
            decision = "match"
        elif confidence >= 0.1:
            decision = "review"
        else:
            decision = "no_match"

        views.append(
            CandidatePairView(
                ref_id=str(cand.ref_id),
                target_id=str(cand.target_id),
                ref_geometry=ref_row.geometry,  # Use WGS84 for map display
                target_geometry=target_row.geometry,
                ref_name=ref_name,
                target_name=target_name,
                ref_class=ref_class,
                target_class=target_class,
                decision=decision,
                confidence=confidence,
                score_breakdown=score_breakdown,
                features=features,  # Now includes all features including topology
                # Defer subsegment estimation - computed on-demand when viewing
                estimated_subsegment=None,
            )
        )

    # Sort: REVIEW first, then by confidence descending
    def sort_key(v):
        decision_order = {"review": 0, "match": 1, "no_match": 2}
        return (decision_order.get(v.decision, 3), -v.confidence)

    views.sort(key=sort_key)

    return views


def get_subsegment_estimate(pair: CandidatePairView) -> dict[str, float]:
    """Get or compute subsegment estimate for a candidate pair.

    Computes on-demand if not already cached.

    Args:
        pair: The candidate pair

    Returns:
        Dict with ref_start_pct, ref_end_pct, target_start_pct, target_end_pct
    """
    if pair.estimated_subsegment is not None:
        return pair.estimated_subsegment

    # Compute and cache
    estimated = estimate_overlap_range(pair.ref_geometry, pair.target_geometry)
    pair.estimated_subsegment = estimated
    return estimated


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
