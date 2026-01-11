"""Candidate generation via spatial indexing.

Uses STRtree for efficient spatial queries to find potential matches
without O(N*M) comparisons.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely import LineString
from shapely.strtree import STRtree

from ..config import settings


@dataclass
class CandidatePair:
    """A candidate match between reference and target edges."""

    ref_id: int
    ref_idx: int  # Index in reference GeoDataFrame
    target_id: int
    target_idx: int  # Index in target GeoDataFrame
    distance_estimate: float
    heading_diff: float
    length_ratio: float


def generate_candidates(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance: float = None,
    max_heading_diff: float = None,
    max_length_ratio: float = None,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
) -> list[CandidatePair]:
    """Generate candidate pairs using buffer-based spatial join.

    Args:
        reference: Reference edges (Overture) GeoDataFrame
        target: Target edges (local data) GeoDataFrame
        buffer_distance: Search radius in meters
        max_heading_diff: Maximum heading difference to consider (degrees)
        max_length_ratio: Maximum length ratio to consider
        ref_id_column: Column name for reference IDs
        target_id_column: Column name for target IDs

    Returns:
        List of CandidatePair objects
    """
    buffer_distance = buffer_distance or settings.buffer_distance
    max_heading_diff = max_heading_diff or settings.max_heading_diff
    max_length_ratio = max_length_ratio or settings.max_length_ratio

    logger.info(
        f"Generating candidates: {len(reference)} reference x {len(target)} target"
    )
    logger.info(f"  buffer_distance: {buffer_distance}m")
    logger.info(f"  max_heading_diff: {max_heading_diff}°")
    logger.info(f"  max_length_ratio: {max_length_ratio}")

    # Build spatial index on reference
    ref_tree = STRtree(reference.geometry.values)

    # Pre-compute headings and lengths
    ref_headings = reference.geometry.apply(_compute_overall_heading)
    target_headings = target.geometry.apply(_compute_overall_heading)

    ref_lengths = reference.geometry.length
    target_lengths = target.geometry.length

    candidates = []
    n_checked = 0
    n_passed_spatial = 0

    for target_idx in range(len(target)):
        target_row = target.iloc[target_idx]
        target_geom = target_row.geometry
        target_heading = target_headings.iloc[target_idx]
        target_length = target_lengths.iloc[target_idx]

        if target_id_column in target_row.index:
            target_id = target_row[target_id_column]
        else:
            target_id = target_idx

        # Buffer query
        buffered = target_geom.buffer(buffer_distance)
        candidate_indices = ref_tree.query(buffered)
        n_checked += len(candidate_indices)

        for ref_idx in candidate_indices:
            ref_row = reference.iloc[ref_idx]
            ref_heading = ref_headings.iloc[ref_idx]
            ref_length = ref_lengths.iloc[ref_idx]

            if ref_id_column in ref_row.index:
                ref_id = ref_row[ref_id_column]
            else:
                ref_id = ref_idx

            # Coarse filter: heading difference
            heading_diff = _angle_diff(target_heading, ref_heading)
            if heading_diff > max_heading_diff:
                continue

            # Coarse filter: length ratio
            length_ratio = (
                max(target_length, ref_length) / max(min(target_length, ref_length), 0.1)
            )
            if length_ratio > max_length_ratio:
                continue

            n_passed_spatial += 1

            # Estimate distance (centroid to centroid for speed)
            distance_estimate = target_geom.centroid.distance(ref_row.geometry.centroid)

            candidates.append(
                CandidatePair(
                    ref_id=ref_id,
                    ref_idx=ref_idx,
                    target_id=target_id,
                    target_idx=target_idx,
                    distance_estimate=distance_estimate,
                    heading_diff=heading_diff,
                    length_ratio=1.0 / length_ratio,  # Normalize to 0-1
                )
            )

    logger.info(f"  Checked {n_checked} spatial candidates")
    logger.info(f"  Generated {len(candidates)} candidates after filtering")

    return candidates


def generate_candidates_iter(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance: float = None,
    max_heading_diff: float = None,
    max_length_ratio: float = None,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
) -> Iterator[CandidatePair]:
    """Iterator version of generate_candidates for memory efficiency.

    Yields candidate pairs one at a time instead of building full list.
    """
    buffer_distance = buffer_distance or settings.buffer_distance
    max_heading_diff = max_heading_diff or settings.max_heading_diff
    max_length_ratio = max_length_ratio or settings.max_length_ratio

    # Build spatial index on reference
    ref_tree = STRtree(reference.geometry.values)

    # Pre-compute headings
    ref_headings = reference.geometry.apply(_compute_overall_heading)

    for target_idx in range(len(target)):
        target_row = target.iloc[target_idx]
        target_geom = target_row.geometry
        target_heading = _compute_overall_heading(target_geom)
        target_length = target_geom.length

        if target_id_column in target_row.index:
            target_id = target_row[target_id_column]
        else:
            target_id = target_idx

        # Buffer query
        buffered = target_geom.buffer(buffer_distance)
        candidate_indices = ref_tree.query(buffered)

        for ref_idx in candidate_indices:
            ref_row = reference.iloc[ref_idx]
            ref_heading = ref_headings.iloc[ref_idx]
            ref_length = ref_row.geometry.length

            if ref_id_column in ref_row.index:
                ref_id = ref_row[ref_id_column]
            else:
                ref_id = ref_idx

            # Coarse filters
            heading_diff = _angle_diff(target_heading, ref_heading)
            if heading_diff > max_heading_diff:
                continue

            length_ratio = (
                max(target_length, ref_length) / max(min(target_length, ref_length), 0.1)
            )
            if length_ratio > max_length_ratio:
                continue

            distance_estimate = target_geom.centroid.distance(ref_row.geometry.centroid)

            yield CandidatePair(
                ref_id=ref_id,
                ref_idx=ref_idx,
                target_id=target_id,
                target_idx=target_idx,
                distance_estimate=distance_estimate,
                heading_diff=heading_diff,
                length_ratio=1.0 / length_ratio,
            )


def _compute_overall_heading(geom: LineString) -> float:
    """Compute heading from first to last point in degrees (0-360)."""
    coords = np.array(geom.coords)
    if len(coords) < 2:
        return 0.0

    dx = coords[-1, 0] - coords[0, 0]
    dy = coords[-1, 1] - coords[0, 1]
    heading = np.degrees(np.arctan2(dy, dx))

    return (heading + 360) % 360


def _angle_diff(a: float, b: float) -> float:
    """Compute minimum angle difference in degrees (0-180).

    Handles the bidirectional nature of roads.
    """
    diff = abs(a - b)
    if diff > 180:
        diff = 360 - diff

    # Consider opposite direction
    opposite_diff = abs(180 - diff)

    return min(diff, opposite_diff)


def filter_candidates_by_name(
    candidates: list[CandidatePair],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_name_column: str = "name",
    target_name_column: str = "name",
    min_similarity: float = 0.5,
) -> list[CandidatePair]:
    """Filter candidates by name similarity.

    Useful as an additional filter when names are reliable.

    Args:
        candidates: List of candidate pairs
        reference: Reference GeoDataFrame
        target: Target GeoDataFrame
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        min_similarity: Minimum name similarity to keep candidate

    Returns:
        Filtered list of candidates
    """
    from ..features.semantic import compute_name_similarity

    filtered = []

    for cand in candidates:
        ref_name = reference.iloc[cand.ref_idx].get(ref_name_column)
        target_name = target.iloc[cand.target_idx].get(target_name_column)

        # If both have names, check similarity
        if ref_name and target_name:
            sim = compute_name_similarity(ref_name, target_name)
            if sim["token_sort_ratio"] >= min_similarity:
                filtered.append(cand)
        else:
            # Keep candidates where one or both names are missing
            filtered.append(cand)

    logger.info(f"Name filter: {len(candidates)} -> {len(filtered)} candidates")
    return filtered


def estimate_candidate_count(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance: float = None,
) -> int:
    """Estimate the number of candidates without generating them.

    Useful for progress bars and memory planning.
    """
    buffer_distance = buffer_distance or settings.buffer_distance

    # Sample-based estimation
    sample_size = min(100, len(target))
    sample_indices = np.random.choice(len(target), sample_size, replace=False)

    ref_tree = STRtree(reference.geometry.values)
    total_candidates = 0

    for idx in sample_indices:
        geom = target.iloc[idx].geometry
        buffered = geom.buffer(buffer_distance)
        candidates = ref_tree.query(buffered)
        total_candidates += len(candidates)

    # Extrapolate to full dataset
    estimated = int(total_candidates / sample_size * len(target))
    return estimated
