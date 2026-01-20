"""Holdout creation for validation experiments.

Functions to create reduced reference datasets by dropping segments
based on various strategies (random, bbox, source, class).
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import box


def extract_record_ids(
    sources_column: pd.Series,
    dataset_filter: str = "OpenStreetMap",
) -> pd.Series:
    """Extract record_ids from Overture sources column for a given dataset.

    Handles both single-source and multi-source (merged) segments.
    Returns a Series of sets, where each set contains all record_ids
    from the specified dataset for that row.

    Args:
        sources_column: Series containing sources arrays
        dataset_filter: Dataset name to filter by (e.g., "OpenStreetMap", "TomTom")

    Returns:
        Series of sets containing record_ids for each row
    """

    def extract_from_row(sources):
        if sources is None:
            return set()
        if not isinstance(sources, (list, np.ndarray)):
            return set()

        record_ids = set()
        for source in sources:
            if isinstance(source, dict):
                ds = source.get("dataset", "")
                record_id = source.get("record_id", "")
                if ds == dataset_filter and record_id:
                    # Normalize record_id: strip version suffix if present (e.g., w123@5 -> w123)
                    base_id = record_id.split("@")[0] if "@" in str(record_id) else str(record_id)
                    record_ids.add(base_id)
        return record_ids

    return sources_column.apply(extract_from_row)


def extract_all_record_ids(
    gdf: gpd.GeoDataFrame,
    dataset_filter: str = "OpenStreetMap",
) -> set[str]:
    """Extract all unique record_ids from a GeoDataFrame.

    Args:
        gdf: GeoDataFrame with 'sources' column
        dataset_filter: Dataset name to filter by

    Returns:
        Set of all unique record_ids
    """
    if "sources" not in gdf.columns:
        return set()

    record_ids_series = extract_record_ids(gdf["sources"], dataset_filter)
    all_ids = set()
    for ids in record_ids_series:
        all_ids.update(ids)
    return all_ids


def has_source(sources, dataset_name: str) -> bool:
    """Check if sources array contains a specific dataset.

    Args:
        sources: Sources array from Overture data
        dataset_name: Dataset name to check for

    Returns:
        True if dataset is present in sources
    """
    if sources is None:
        return False
    if not isinstance(sources, (list, np.ndarray)):
        return False

    for source in sources:
        if isinstance(source, dict):
            if source.get("dataset") == dataset_name:
                return True
    return False


def create_holdout(
    overture: gpd.GeoDataFrame,
    segments_to_drop: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, set[str]]:
    """Create reduced reference by dropping specified segments.

    Args:
        overture: Full Overture reference dataset
        segments_to_drop: Subset of segments to remove

    Returns:
        reduced_reference: Overture without dropped segments
        dropped_record_ids: All record_ids from dropped segments
    """
    # Get indices to drop
    drop_indices = set(segments_to_drop.index)

    # Extract all record_ids from dropped segments
    dropped_record_ids = extract_all_record_ids(segments_to_drop, "OpenStreetMap")

    # Also extract TomTom record_ids if present
    tomtom_ids = extract_all_record_ids(segments_to_drop, "TomTom")
    dropped_record_ids.update(tomtom_ids)

    # Create reduced reference
    reduced = overture.drop(index=drop_indices)

    logger.info(f"Created holdout: dropped {len(segments_to_drop)} segments")
    logger.info(f"  Dropped record_ids: {len(dropped_record_ids)}")
    logger.info(f"  Remaining segments: {len(reduced)}")

    return reduced, dropped_record_ids


def drop_random_osm(
    overture: gpd.GeoDataFrame,
    fraction: float,
    seed: int = 42,
) -> tuple[gpd.GeoDataFrame, set[str]]:
    """Drop random fraction of OSM-sourced segments.

    Args:
        overture: Full Overture reference dataset
        fraction: Fraction of OSM segments to drop (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        reduced_reference: Overture without dropped segments
        dropped_record_ids: All record_ids from dropped segments

    Raises:
        ValueError: If fraction is not in range [0.0, 1.0]
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be between 0.0 and 1.0, got {fraction}")

    if "sources" not in overture.columns:
        raise ValueError("Overture data must have 'sources' column")

    # Filter to OSM-sourced segments
    osm_mask = overture["sources"].apply(lambda s: has_source(s, "OpenStreetMap"))
    osm_segments = overture[osm_mask]

    logger.info(f"Found {len(osm_segments)} OSM-sourced segments out of {len(overture)} total")

    # Sample random fraction
    n_to_drop = int(len(osm_segments) * fraction)

    # Handle edge case where n_to_drop is 0
    if n_to_drop == 0:
        logger.warning("No segments to drop (fraction too small or no OSM segments)")
        return overture.copy(), set()

    np.random.seed(seed)
    drop_indices = np.random.choice(osm_segments.index, size=n_to_drop, replace=False)

    segments_to_drop = overture.loc[drop_indices]
    return create_holdout(overture, segments_to_drop)


def drop_by_bbox(
    overture: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
) -> tuple[gpd.GeoDataFrame, set[str]]:
    """Drop segments within a bounding box.

    Args:
        overture: Full Overture reference dataset
        bbox: Bounding box (xmin, ymin, xmax, ymax) in same CRS as data

    Returns:
        reduced_reference: Overture without dropped segments
        dropped_record_ids: All record_ids from dropped segments
    """
    xmin, ymin, xmax, ymax = bbox
    bbox_geom = box(xmin, ymin, xmax, ymax)

    # Find segments intersecting bbox
    intersects_mask = overture.geometry.intersects(bbox_geom)
    segments_to_drop = overture[intersects_mask]

    logger.info(f"Found {len(segments_to_drop)} segments in bbox {bbox}")

    return create_holdout(overture, segments_to_drop)


def drop_by_source(
    overture: gpd.GeoDataFrame,
    source_dataset: str,
) -> tuple[gpd.GeoDataFrame, set[str]]:
    """Drop all segments from a specific source dataset.

    Args:
        overture: Full Overture reference dataset
        source_dataset: Dataset name to drop (e.g., "TomTom")

    Returns:
        reduced_reference: Overture without dropped segments
        dropped_record_ids: All record_ids from dropped segments
    """
    if "sources" not in overture.columns:
        raise ValueError("Overture data must have 'sources' column")

    # Filter to segments with the specified source
    source_mask = overture["sources"].apply(lambda s: has_source(s, source_dataset))
    segments_to_drop = overture[source_mask]

    logger.info(f"Found {len(segments_to_drop)} segments from {source_dataset}")

    return create_holdout(overture, segments_to_drop)


def drop_by_class(
    overture: gpd.GeoDataFrame,
    road_class: str,
    source_dataset: str | None = None,
) -> tuple[gpd.GeoDataFrame, set[str]]:
    """Drop segments of a specific road class.

    Args:
        overture: Full Overture reference dataset
        road_class: Road class to drop (e.g., "residential")
        source_dataset: Optional - only drop from this source

    Returns:
        reduced_reference: Overture without dropped segments
        dropped_record_ids: All record_ids from dropped segments
    """
    if "class" not in overture.columns:
        raise ValueError("Overture data must have 'class' column")

    # Filter by class
    class_mask = overture["class"] == road_class

    # Optionally filter by source
    if source_dataset and "sources" in overture.columns:
        source_mask = overture["sources"].apply(lambda s: has_source(s, source_dataset))
        class_mask = class_mask & source_mask

    segments_to_drop = overture[class_mask]

    logger.info(f"Found {len(segments_to_drop)} segments of class '{road_class}'")

    return create_holdout(overture, segments_to_drop)
