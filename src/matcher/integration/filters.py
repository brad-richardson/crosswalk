"""Optional pre-filters for integration pipeline.

Provides filtering logic to remove noise and detect potential duplicates
before integration.
"""

import geopandas as gpd
from loguru import logger
from shapely.strtree import STRtree

from ..config import settings


def _compute_buffer_iou(line_a, line_b, radius: float) -> float:
    """Compute Intersection over Union of buffered geometries."""
    buf_a = line_a.buffer(radius)
    buf_b = line_b.buffer(radius)

    intersection_area = buf_a.intersection(buf_b).area
    union_area = buf_a.union(buf_b).area

    return intersection_area / union_area if union_area > 0 else 0.0


def filter_short_segments(
    gdf: gpd.GeoDataFrame,
    min_length: float = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Filter out segments shorter than min_length meters.

    Short segments are often:
    - GPS noise artifacts
    - Incorrectly split segments
    - Parking lot fragments

    Args:
        gdf: GeoDataFrame with LineString geometries
        min_length: Minimum segment length in meters

    Returns:
        Tuple of (kept_segments, filtered_segments)
    """
    min_length = min_length or settings.min_segment_length

    if gdf is None or len(gdf) == 0:
        return gdf, gpd.GeoDataFrame()

    # Project to UTM for accurate length calculation if in geographic CRS
    working_gdf = gdf
    if gdf.crs is not None and gdf.crs.is_geographic:
        working_crs = gdf.estimate_utm_crs()
        working_gdf = gdf.to_crs(working_crs)
        logger.debug(f"Projected to {working_crs} for length calculation")

    # Compute lengths in projected CRS (meters)
    lengths = working_gdf.geometry.length

    # Filter
    keep_mask = lengths >= min_length
    kept = gdf[keep_mask].copy()  # Return in original CRS
    filtered = gdf[~keep_mask].copy()

    if len(filtered) > 0:
        logger.info(f"Filtered {len(filtered)} short segments (<{min_length}m)")

    return kept, filtered


def detect_near_duplicates(
    unmatched: gpd.GeoDataFrame,
    matched: gpd.GeoDataFrame,
    distance_tolerance: float = None,
    overlap_threshold: float = None,
    id_column: str = "local_id",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Detect near-duplicates that may be matching false negatives.

    A near-duplicate is an unmatched target that:
    1. Is within distance_tolerance of a matched target
    2. Has overlap_ratio >= overlap_threshold with matched target
    3. Could represent the same physical feature

    These get flagged for QA review rather than hard-filtered.

    Args:
        unmatched: Unmatched segments GeoDataFrame
        matched: Matched segments GeoDataFrame
        distance_tolerance: Distance tolerance for candidate detection (meters)
        overlap_threshold: Minimum overlap ratio to consider as near-duplicate
        id_column: ID column name

    Returns:
        Tuple of (clean_unmatched, potential_duplicates)
    """
    distance_tolerance = distance_tolerance or settings.near_duplicate_tolerance
    overlap_threshold = overlap_threshold or settings.near_duplicate_overlap

    if unmatched is None or len(unmatched) == 0:
        return unmatched, gpd.GeoDataFrame()

    if matched is None or len(matched) == 0:
        return unmatched, gpd.GeoDataFrame()

    logger.info(f"Detecting near-duplicates in {len(unmatched)} unmatched segments...")

    # Build spatial index of matched segments
    matched_geoms = matched.geometry.values
    tree = STRtree(matched_geoms)

    potential_duplicates = []
    clean_indices = []

    for idx, row in unmatched.iterrows():
        geom = row.geometry
        _original_id = row.get(id_column, idx)  # noqa: F841 - reserved for debugging

        # Find nearby matched segments
        buffered = geom.buffer(distance_tolerance)
        candidate_indices = tree.query(buffered)

        is_duplicate = False
        best_overlap = 0.0
        best_matched_id = None

        for matched_idx in candidate_indices:
            matched_geom = matched_geoms[matched_idx]

            # Compute overlap
            iou = _compute_buffer_iou(geom, matched_geom, distance_tolerance)

            if iou > best_overlap:
                best_overlap = iou
                best_matched_id = matched.iloc[matched_idx].get(id_column, matched_idx)

            if iou >= overlap_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            row_dict = row.to_dict()
            row_dict["duplicate_of_id"] = best_matched_id
            row_dict["duplicate_overlap"] = best_overlap
            potential_duplicates.append(row_dict)
        else:
            clean_indices.append(idx)

    # Build result DataFrames
    clean = unmatched.loc[clean_indices].copy() if clean_indices else gpd.GeoDataFrame()

    if potential_duplicates:
        duplicates = gpd.GeoDataFrame(potential_duplicates, crs=unmatched.crs)
        logger.info(f"Found {len(duplicates)} potential near-duplicates")
    else:
        duplicates = gpd.GeoDataFrame(crs=unmatched.crs)

    return clean, duplicates


def filter_by_road_class(
    gdf: gpd.GeoDataFrame,
    exclude_classes: list[str] | None = None,
    road_class_column: str = "road_class",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Filter segments by road class.

    Can be used to exclude certain road types like service roads,
    driveways, parking lots, etc.

    Args:
        gdf: GeoDataFrame with road segments
        exclude_classes: List of road classes to exclude
        road_class_column: Column name for road class

    Returns:
        Tuple of (kept_segments, filtered_segments)
    """
    if gdf is None or len(gdf) == 0:
        return gdf, gpd.GeoDataFrame()

    if exclude_classes is None or len(exclude_classes) == 0:
        return gdf, gpd.GeoDataFrame()

    if road_class_column not in gdf.columns:
        logger.warning(f"Road class column '{road_class_column}' not found, skipping filter")
        return gdf, gpd.GeoDataFrame()

    # Filter
    exclude_set = set(c.lower() for c in exclude_classes)
    keep_mask = ~gdf[road_class_column].str.lower().isin(exclude_set)

    kept = gdf[keep_mask].copy()
    filtered = gdf[~keep_mask].copy()

    if len(filtered) > 0:
        logger.info(f"Filtered {len(filtered)} segments by road class")

    return kept, filtered
