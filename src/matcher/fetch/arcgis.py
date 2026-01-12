"""Fetch data from ArcGIS REST API FeatureServer/MapServer endpoints.

Provides a reusable utility for fetching geospatial features from any
ArcGIS REST API and converting them to GeoParquet with Overture-compatible
schema.
"""

from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger


def fetch_arcgis_layer(
    url: str,
    output_path: Path,
    id_prefix: str,
    name_column: Optional[str] = None,
    class_column: Optional[str] = None,
    class_mapping: Optional[dict] = None,
    level_column: Optional[str] = None,
    source_name: str = "ArcGIS",
    page_size: int = 2000,
) -> Path:
    """Fetch features from ArcGIS REST API and save as GeoParquet.

    Works with both FeatureServer and MapServer endpoints. Handles pagination,
    coordinate reprojection to WGS84, and Overture schema transformation.

    Args:
        url: ArcGIS REST API layer URL (e.g., .../FeatureServer/0)
        output_path: Path for output GeoParquet file
        id_prefix: Prefix for generated IDs (e.g., "boston_streets")
        name_column: Column name for feature names
        class_column: Column name for road classification
        class_mapping: Dict mapping source class values to standard classes
        level_column: Column name for z-level/layer
        source_name: Name for the data source in sources array
        page_size: Number of features per API request

    Returns:
        Path to the output GeoParquet file
    """
    logger.info(f"Fetching ArcGIS layer: {url}")

    # Fetch all features with pagination
    features = _fetch_all_features(url, page_size)

    if not features:
        logger.warning(f"No features returned from {url}")
        return output_path

    logger.info(f"Fetched {len(features)} features")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

    # Transform to Overture schema
    gdf = _transform_to_overture_schema(
        gdf,
        id_prefix=id_prefix,
        name_column=name_column,
        class_column=class_column,
        class_mapping=class_mapping,
        level_column=level_column,
        source_name=source_name,
    )

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)

    logger.info(f"Saved {len(gdf)} features to {output_path}")
    return output_path


def _fetch_all_features(url: str, page_size: int) -> list[dict]:
    """Fetch all features from ArcGIS REST API with pagination.

    Args:
        url: ArcGIS REST API layer URL
        page_size: Number of features per request

    Returns:
        List of GeoJSON features
    """
    # First, get layer metadata to determine the correct ID field for ordering
    id_field = _get_layer_id_field(url)

    all_features = []
    offset = 0

    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "returnGeometry": "true",
            "outSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        # Add orderByFields for deterministic pagination if we have an ID field
        if id_field:
            params["orderByFields"] = id_field

        response = requests.get(f"{url}/query", params=params, timeout=120)
        response.raise_for_status()

        data = response.json()

        # Check for error response
        if "error" in data:
            raise RuntimeError(f"ArcGIS API error: {data['error']}")

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        logger.debug(f"Fetched {len(all_features)} features so far...")

        # Check if there are more results
        # For GeoJSON format, exceededTransferLimit is in properties
        props = data.get("properties", {})
        exceeded = props.get("exceededTransferLimit", False) or data.get(
            "exceededTransferLimit", False
        )
        if not exceeded:
            break

        # Increment by actual returned count (server may cap below page_size)
        offset += len(features)

    return all_features


def _transform_to_overture_schema(
    gdf: gpd.GeoDataFrame,
    id_prefix: str,
    name_column: Optional[str],
    class_column: Optional[str],
    class_mapping: Optional[dict],
    level_column: Optional[str],
    source_name: str,
) -> gpd.GeoDataFrame:
    """Transform ArcGIS data to match osm_segments.parquet schema.

    Args:
        gdf: Input GeoDataFrame from ArcGIS
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for road classification
        class_mapping: Dict mapping source values to standard classes
        level_column: Column name for z-level
        source_name: Name for the data source

    Returns:
        GeoDataFrame with Overture-compatible schema
    """
    if len(gdf) == 0:
        return gdf

    # Find the ID column (OBJECTID, FID, or similar)
    id_col = _find_id_column(gdf)

    # Store original columns for source_tags (before we modify anything)
    original_cols = [c for c in gdf.columns if c != "geometry"]
    source_tags_data = gdf[original_cols].to_dict(orient="records")

    # Build result data dict
    data = {}

    # ID: prefix + objectid
    data["id"] = gdf[id_col].apply(lambda x: f"{id_prefix}_{x}").values

    # Names struct
    if name_column and name_column in gdf.columns:
        data["names"] = gdf[name_column].apply(
            lambda x: {"primary": str(x)} if pd.notna(x) and x else None
        ).values
    else:
        data["names"] = [None] * len(gdf)

    # Class with mapping
    if class_column and class_column in gdf.columns:
        if class_mapping:
            data["class"] = (
                gdf[class_column].map(class_mapping).fillna("unclassified").values
            )
        else:
            data["class"] = gdf[class_column].fillna("unclassified").astype(str).values
    else:
        data["class"] = ["unclassified"] * len(gdf)

    # Subtype (constant)
    data["subtype"] = ["road"] * len(gdf)

    # Sources array
    data["sources"] = gdf[id_col].apply(
        lambda x: [{"dataset": source_name, "record_id": str(x)}]
    ).values

    # Road flags (empty - no bridge/tunnel info in these sources)
    data["road_flags"] = [[] for _ in range(len(gdf))]

    # Level rules
    if level_column and level_column in gdf.columns:
        data["level_rules"] = gdf[level_column].apply(_build_level_rules).values
    else:
        data["level_rules"] = [[] for _ in range(len(gdf))]

    # Source tags (all original columns as dict)
    data["source_tags"] = source_tags_data

    # Create GeoDataFrame with geometry
    result = gpd.GeoDataFrame(data, geometry=gdf.geometry.values, crs=gdf.crs)

    return result


def _get_layer_id_field(url: str) -> Optional[str]:
    """Get the object ID field name from layer metadata.

    Args:
        url: ArcGIS REST API layer URL

    Returns:
        Object ID field name, or None if not found
    """
    try:
        response = requests.get(url, params={"f": "json"}, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Try objectIdFieldName first (standard for FeatureServer)
        if "objectIdFieldName" in data:
            return data["objectIdFieldName"]

        # Fall back to looking for OID type field in fields array
        for field in data.get("fields", []):
            if field.get("type") == "esriFieldTypeOID":
                return field.get("name")

        return None
    except Exception as e:
        logger.warning(f"Could not determine ID field from layer metadata: {e}")
        return None


def _find_id_column(gdf: gpd.GeoDataFrame) -> str:
    """Find the ID column in a GeoDataFrame.

    Args:
        gdf: Input GeoDataFrame

    Returns:
        Name of the ID column

    Raises:
        ValueError: If no ID column found
    """
    candidates = ["OBJECTID", "FID", "ObjectID", "objectid", "fid", "ID", "id"]
    for col in candidates:
        if col in gdf.columns:
            return col

    # Fall back to first column (may not be unique)
    non_geom_cols = [c for c in gdf.columns if c != "geometry"]
    if non_geom_cols:
        fallback_col = non_geom_cols[0]
        logger.warning(
            f"No standard ID column found (OBJECTID, FID, etc.). "
            f"Falling back to '{fallback_col}' which may not be unique."
        )
        return fallback_col

    raise ValueError("Could not find ID column in GeoDataFrame")


def _build_level_rules(value: Any) -> list:
    """Build level_rules array from z-level value.

    Args:
        value: Z-level value from source data

    Returns:
        List of level rule dicts, or empty list
    """
    if pd.isna(value):
        return []
    try:
        level = int(value)
        if level == 0:
            return []  # Ground level is omitted
        return [{"value": level}]
    except (ValueError, TypeError):
        return []
