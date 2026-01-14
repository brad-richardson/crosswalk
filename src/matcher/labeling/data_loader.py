"""Data loading and candidate preparation for labeling UI."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import LineString

from ..blocking import generate_candidates
from ..features.semantic import _extract_name_string
from ..matching.rules import score_candidates
from .subsegment import estimate_overlap_range

logger = logging.getLogger(__name__)


@dataclass
class CandidatePairView:
    """View model for a single candidate pair in the labeling UI."""

    ref_id: str
    target_id: str
    ref_geometry: LineString
    target_geometry: LineString
    ref_name: Optional[str]
    target_name: Optional[str]
    ref_class: Optional[str]
    target_class: Optional[str]
    decision: str  # "match", "review", "no_match"
    confidence: float
    score_breakdown: dict[str, float]
    features: dict[str, float]
    # Sub-segment estimate - computed on-demand when viewing the pair
    # None means not yet computed, dict contains the estimated ranges
    estimated_subsegment: Optional[dict[str, float]] = None


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

    # Score candidates
    results = score_candidates(
        candidates=candidates,
        reference=reference_proj,
        target=target_proj,
        ref_name_column=ref_name_column,
        target_name_column=target_name_column,
        ref_class_column=ref_class_column,
        target_class_column=target_class_column,
    )

    # Build view models (use original WGS84 geometries for map display)
    # Create index lookups for O(1) access (avoid O(n) DataFrame filtering per candidate)
    ref_lookup = reference.set_index(ref_id_column)
    target_lookup = target.set_index(target_id_column)

    # Pre-check column existence
    has_ref_name = ref_name_column in reference.columns
    has_target_name = target_name_column in target.columns
    has_ref_class = ref_class_column in reference.columns
    has_target_class = target_class_column in target.columns

    views = []
    for result in results:
        # Get rows from indexed lookup (O(1) instead of O(n))
        ref_row = ref_lookup.loc[result.ref_id]
        target_row = target_lookup.loc[result.target_id]

        # Extract names
        ref_name = _extract_name_string(ref_row.get(ref_name_column)) if has_ref_name else None
        target_name = _extract_name_string(target_row.get(target_name_column)) if has_target_name else None

        # Extract classes
        ref_class = ref_row.get(ref_class_column) if has_ref_class else None
        target_class = target_row.get(target_class_column) if has_target_class else None

        # Skip sub-segment estimation during bulk loading - compute on-demand
        # when the pair is actually viewed in the UI
        # estimated = estimate_overlap_range(ref_row.geometry, target_row.geometry)

        views.append(CandidatePairView(
            ref_id=str(result.ref_id),
            target_id=str(result.target_id),
            ref_geometry=ref_row.geometry,
            target_geometry=target_row.geometry,
            ref_name=ref_name,
            target_name=target_name,
            ref_class=ref_class,
            target_class=target_class,
            decision=result.decision.value,
            confidence=result.confidence,
            score_breakdown=result.score_breakdown,
            features=result.features,
            # Defer subsegment estimation - computed on-demand when viewing
            estimated_subsegment=None,
        ))

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
    decision_filter: Optional[str] = None,
    labeled_pairs: Optional[set[tuple[str, str]]] = None,
    show_labeled: bool = False,
    specific_pairs: Optional[list[tuple[str, str]]] = None,
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
        filtered = [
            v for v in filtered
            if (v.ref_id, v.target_id) not in labeled_pairs
        ]

    return filtered
