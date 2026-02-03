"""Shared DataFrame utility functions.

Provides common operations for working with GeoDataFrames.
"""

import geopandas as gpd
from loguru import logger


def find_id_column(
    gdf: gpd.GeoDataFrame, raise_on_missing: bool = True, fallback: bool = False
) -> str | None:
    """Find the ID column in a GeoDataFrame.

    Searches for common ID column names in order of preference:
    - Standard names: id, ID, edge_id
    - ArcGIS names: OBJECTID, FID, ObjectID, objectid, fid

    Args:
        gdf: Input GeoDataFrame
        raise_on_missing: If True and no ID column found (after fallback if enabled),
                         raise ValueError. If False, return None instead.
        fallback: If True, fall back to first non-geometry column with a warning
                 when no standard ID column is found. If False, do not fall back.

    Returns:
        Name of the ID column, or None if not found and raise_on_missing=False

    Raises:
        ValueError: If no ID column found and raise_on_missing=True

    Note:
        When fallback=True (default), the function will use the first column
        as a last resort. This may not be unique, so a warning is logged.
        Use fallback=False when you need strict ID column detection.
    """
    # Check standard ID column names first
    for col in ["id", "ID", "edge_id"]:
        if col in gdf.columns:
            return col

    # Check ArcGIS-style ID columns
    for col in ["OBJECTID", "FID", "ObjectID", "objectid", "fid"]:
        if col in gdf.columns:
            return col

    # Check if index has a name that's a column
    if gdf.index.name and gdf.index.name in gdf.columns:
        return gdf.index.name

    # Fall back to first non-geometry column with a warning (if enabled)
    if fallback:
        non_geom_cols = [c for c in gdf.columns if c != "geometry"]
        if non_geom_cols:
            fallback_col = non_geom_cols[0]
            logger.warning(
                f"No standard ID column found (id, ID, edge_id, OBJECTID, FID, etc.). "
                f"Falling back to '{fallback_col}' which may not be unique."
            )
            return fallback_col

    if raise_on_missing:
        raise ValueError(
            f"Could not find ID column in GeoDataFrame. "
            f"Expected one of ['id', 'ID', 'edge_id', 'OBJECTID', 'FID']. "
            f"Available columns: {list(gdf.columns)}"
        )

    return None
