"""Geometry utility functions."""

import geopandas as gpd
from loguru import logger


def filter_to_linestrings(
    gdf: gpd.GeoDataFrame,
    source_name: str = "dataset",
) -> gpd.GeoDataFrame:
    """Filter GeoDataFrame to only LineString geometries, dropping MultiLineStrings.

    MultiLineString geometries are filtered out at ingest time as they typically
    represent bad data or edge cases that complicate downstream processing.
    A warning is logged when MultiLineStrings are dropped.

    Args:
        gdf: Input GeoDataFrame with geometry column
        source_name: Name of the data source for logging (e.g., "reference", "target")

    Returns:
        GeoDataFrame with only LineString geometries
    """
    if gdf.empty:
        return gdf

    original_count = len(gdf)

    # Count different geometry types
    geom_types = gdf.geometry.geom_type
    multilinestring_count = (geom_types == "MultiLineString").sum()
    linestring_count = (geom_types == "LineString").sum()
    other_count = original_count - multilinestring_count - linestring_count

    # Filter to only LineString geometries
    mask = geom_types == "LineString"
    filtered = gdf[mask].copy()

    # Log warnings for filtered geometries
    if multilinestring_count > 0:
        logger.warning(
            f"Filtered {multilinestring_count} MultiLineString geometries from {source_name} "
            f"({multilinestring_count}/{original_count} features). "
            f"MultiLineStrings are not supported and likely represent bad data."
        )

    if other_count > 0:
        logger.warning(
            f"Filtered {other_count} non-LineString geometries from {source_name} "
            f"({other_count}/{original_count} features). "
            f"Only LineString geometries are supported."
        )

    if len(filtered) == 0 and original_count > 0:
        logger.error(
            f"All {original_count} geometries were filtered from {source_name}. "
            f"No LineString geometries found."
        )

    return filtered
