"""Shared DataFrame utility functions.

Provides common operations for working with GeoDataFrames.
"""

import geopandas as gpd


def find_id_column(gdf: gpd.GeoDataFrame, raise_on_missing: bool = True) -> str | None:
    """Find the ID column in a GeoDataFrame.

    Searches for common ID column names in order of preference:
    - Standard names: id, ID, edge_id
    - ArcGIS names: OBJECTID, FID, ObjectID, objectid, fid

    Args:
        gdf: Input GeoDataFrame
        raise_on_missing: If True and no ID column found, raise ValueError.
                         If False, return None instead.

    Returns:
        Name of the ID column, or None if not found and raise_on_missing=False

    Raises:
        ValueError: If no ID column found and raise_on_missing=True
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

    if raise_on_missing:
        raise ValueError(
            f"Could not find ID column in GeoDataFrame. "
            f"Expected one of ['id', 'ID', 'edge_id', 'OBJECTID', 'FID']. "
            f"Available columns: {list(gdf.columns)}"
        )

    return None
