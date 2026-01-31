"""Fetch data from ArcGIS REST API FeatureServer/MapServer endpoints.

Provides a reusable utility for fetching geospatial features from any
ArcGIS REST API and converting them to GeoParquet with Overture-compatible
schema.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger

from ..config import DATA_VERSION, SCHEMA_VERSION, TRANSFORM_VERSION
from ..utils.linear_ref import create_trivial_lr
from .metadata import FetchMetadata, save_metadata


def fetch_arcgis_layer(
    url: str,
    output_path: Path,
    id_prefix: str,
    name_column: str | None = None,
    class_column: str | None = None,
    class_mapping: dict | None = None,
    subclass_column: str | None = None,
    subclass_mapping: dict | None = None,
    level_column: str | None = None,
    bridge_column: str | None = None,
    tunnel_column: str | None = None,
    status_column: str | None = None,
    status_mapping: dict | None = None,
    source_name: str = "ArcGIS",
    page_size: int = 5000,
    where_clause: str = "1=1",
    bbox: tuple[float, float, float, float] | None = None,
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
        subclass_column: Column name for subclass (e.g., sidewalk vs crosswalk)
        subclass_mapping: Dict mapping source values to subclass values
        level_column: Column name for z-level/layer
        bridge_column: Column name for bridge indicator (truthy value = bridge)
        tunnel_column: Column name for tunnel indicator (truthy value = tunnel)
        status_column: Column name for lifecycle status (proposed/construction)
        status_mapping: Dict mapping source status values to standard status values
        source_name: Name for the data source in sources array
        page_size: Number of features per API request
        where_clause: SQL WHERE clause to filter features (default: "1=1" for all)
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) for server-side filtering

    Returns:
        Path to the output GeoParquet file
    """
    logger.info(f"Fetching ArcGIS layer: {url}")
    if where_clause != "1=1":
        logger.info(f"Filtering with: {where_clause}")
    if bbox:
        logger.info(f"Server-side bbox filter: {bbox}")

    # Fetch all features with pagination
    features = _fetch_all_features(url, page_size, where_clause, bbox)

    if not features:
        logger.warning(f"No features returned from {url}")
        return output_path

    logger.info(f"Fetched {len(features)} features")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

    # Find source ID column before transformation (for metadata tracking)
    source_id_col = _find_id_column(gdf) if len(gdf) > 0 else None

    # Transform to Overture schema
    gdf = _transform_to_overture_schema(
        gdf,
        id_prefix=id_prefix,
        name_column=name_column,
        class_column=class_column,
        class_mapping=class_mapping,
        subclass_column=subclass_column,
        subclass_mapping=subclass_mapping,
        level_column=level_column,
        bridge_column=bridge_column,
        tunnel_column=tunnel_column,
        status_column=status_column,
        status_mapping=status_mapping,
        source_name=source_name,
    )

    # Deduplicate by ID (ArcGIS pagination can return duplicates if data changes during fetch)
    if len(gdf) > 0 and "id" in gdf.columns:
        n_before = len(gdf)
        gdf = gdf.drop_duplicates(subset=["id"], keep="first")
        n_dropped = n_before - len(gdf)
        if n_dropped > 0:
            logger.warning(f"Dropped {n_dropped} duplicate IDs from ArcGIS fetch")

    # Save to parquet with bbox metadata for DuckDB spatial predicate pushdown
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, write_covering_bbox=True)

    # Calculate bounding box from the data
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    data_bbox = (bounds[0], bounds[1], bounds[2], bounds[3]) if len(gdf) > 0 else None

    # Save fetch metadata
    filters_dict: dict[str, Any] = {}
    if where_clause != "1=1":
        filters_dict["where_clause"] = where_clause
    if class_mapping:
        filters_dict["class_mapping_applied"] = True

    # Filter out None from geometry types (null geometries)
    geom_types = (
        [g for g in gdf.geometry.geom_type.unique() if g is not None] if len(gdf) > 0 else []
    )

    metadata = FetchMetadata(
        source="arcgis",
        source_url=url,
        bbox=data_bbox,
        feature_count=len(gdf),
        geometry_types=geom_types,
        filters=filters_dict if filters_dict else {},
        notes=f"Fetched from {source_name}",
        # Version tracking
        transform_version=TRANSFORM_VERSION,
        schema_version=SCHEMA_VERSION,
        data_version=DATA_VERSION,
        # ID column tracking
        id_column=source_id_col,
        id_prefix=id_prefix,
    )
    save_metadata(output_path, metadata)

    logger.info(f"Saved {len(gdf)} features to {output_path}")
    return output_path


def _fetch_all_features(
    url: str,
    page_size: int,
    where_clause: str = "1=1",
    bbox: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """Fetch all features from ArcGIS REST API with pagination.

    Uses adaptive page sizing - if server returns fewer features than requested,
    adjusts subsequent requests to match the server's cap for efficiency.

    Args:
        url: ArcGIS REST API layer URL
        page_size: Number of features per request (initial, may be reduced)
        where_clause: SQL WHERE clause to filter features
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) for server-side spatial filtering

    Returns:
        List of GeoJSON features
    """
    # First, get layer metadata to determine the correct ID field for ordering
    id_field = _get_layer_id_field(url)

    all_features = []
    offset = 0
    effective_page_size = page_size
    server_cap_detected = False

    while True:
        params = {
            "where": where_clause,
            "outFields": "*",
            "f": "geojson",
            "returnGeometry": "true",
            "outSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": effective_page_size,
        }
        # Add server-side bbox filtering if provided
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            params["geometry"] = f"{xmin},{ymin},{xmax},{ymax}"
            params["geometryType"] = "esriGeometryEnvelope"
            params["spatialRel"] = "esriSpatialRelIntersects"
            params["inSR"] = "4326"
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

        # Check if there are more results
        # For GeoJSON format, exceededTransferLimit is in properties
        props = data.get("properties", {})
        exceeded = props.get("exceededTransferLimit", False) or data.get(
            "exceededTransferLimit", False
        )

        # Adaptive page sizing: detect if server caps below our requested size
        if exceeded and len(features) < effective_page_size and not server_cap_detected:
            server_cap_detected = True
            old_page_size = effective_page_size
            effective_page_size = len(features)
            logger.info(
                f"Server caps page size at {effective_page_size} (requested {old_page_size}), "
                f"adapting for subsequent requests"
            )

        logger.debug(f"Fetched {len(all_features)} features so far...")

        if not exceeded:
            break

        # Increment by actual returned count (server may cap below page_size)
        offset += len(features)

    return all_features


def _transform_to_overture_schema(
    gdf: gpd.GeoDataFrame,
    id_prefix: str,
    name_column: str | None,
    class_column: str | None,
    class_mapping: dict | None,
    subclass_column: str | None,
    subclass_mapping: dict | None,
    level_column: str | None,
    bridge_column: str | None,
    tunnel_column: str | None,
    status_column: str | None,
    status_mapping: dict | None,
    source_name: str,
) -> gpd.GeoDataFrame:
    """Transform ArcGIS data to match osm_segments.parquet schema.

    Args:
        gdf: Input GeoDataFrame from ArcGIS
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for road classification
        class_mapping: Dict mapping source values to standard classes
        subclass_column: Column name for subclass
        subclass_mapping: Dict mapping source values to subclass values
        level_column: Column name for z-level
        bridge_column: Column name for bridge indicator
        tunnel_column: Column name for tunnel indicator
        status_column: Column name for lifecycle status
        status_mapping: Dict mapping source status values to standard values
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
        data["names"] = (
            gdf[name_column]
            .apply(lambda x: {"primary": str(x)} if pd.notna(x) and x else None)
            .values
        )
    else:
        data["names"] = [None] * len(gdf)

    # Class with mapping
    if class_column and class_column in gdf.columns:
        if class_mapping:
            data["class"] = gdf[class_column].map(class_mapping).fillna("unknown").values
        else:
            data["class"] = gdf[class_column].fillna("unknown").astype(str).values
    else:
        data["class"] = ["unknown"] * len(gdf)

    # Subtype (constant)
    data["subtype"] = ["road"] * len(gdf)

    # Subclass with mapping (e.g., sidewalk vs crosswalk)
    if subclass_column and subclass_column in gdf.columns:
        if subclass_mapping:
            data["subclass"] = gdf[subclass_column].map(subclass_mapping).values
        else:
            data["subclass"] = gdf[subclass_column].astype(str).values
    else:
        data["subclass"] = [None] * len(gdf)

    # Sources array
    data["sources"] = (
        gdf[id_col].apply(lambda x: [{"dataset": source_name, "record_id": str(x)}]).values
    )

    # Road flags (bridge/tunnel indicators)
    data["road_flags"] = _build_road_flags(gdf, bridge_column, tunnel_column)

    # Level rules
    if level_column and level_column in gdf.columns:
        data["level_rules"] = gdf[level_column].apply(_build_level_rules).values
    else:
        data["level_rules"] = [[] for _ in range(len(gdf))]

    # Status (lifecycle: proposed, construction, etc.)
    if status_column and status_column in gdf.columns:
        if status_mapping:
            data["status"] = gdf[status_column].map(status_mapping).values
        else:
            data["status"] = gdf[status_column].astype(str).values
    else:
        data["status"] = [None] * len(gdf)

    # Source tags (all original columns as dict)
    data["source_tags"] = source_tags_data

    # Create GeoDataFrame with geometry
    result = gpd.GeoDataFrame(data, geometry=gdf.geometry.values, crs=gdf.crs)

    # Add trivial linear-referenced columns
    result = add_trivial_lr_columns(result)

    return result


def _get_layer_id_field(url: str) -> str | None:
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


def _build_road_flags(
    gdf: gpd.GeoDataFrame,
    bridge_column: str | None,
    tunnel_column: str | None,
) -> list[list[str]]:
    """Build road_flags arrays from bridge/tunnel columns.

    Args:
        gdf: Input GeoDataFrame
        bridge_column: Column name for bridge indicator
        tunnel_column: Column name for tunnel indicator

    Returns:
        List of road flag lists (one per row)
    """
    result = []

    for idx in range(len(gdf)):
        flags = []

        # Check bridge
        if bridge_column and bridge_column in gdf.columns:
            val = gdf.iloc[idx][bridge_column]
            if _is_truthy(val):
                flags.append("is_bridge")

        # Check tunnel
        if tunnel_column and tunnel_column in gdf.columns:
            val = gdf.iloc[idx][tunnel_column]
            if _is_truthy(val):
                flags.append("is_tunnel")

        result.append(flags)

    return result


def _is_truthy(value: Any) -> bool:
    """Check if a value indicates True/Yes/1.

    Handles various representations from different data sources:
    - Boolean True/False
    - Numeric 1/0
    - String "Y"/"N", "Yes"/"No", "True"/"False", "1"/"0"

    Args:
        value: Value to check

    Returns:
        True if value indicates a truthy state
    """
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value == 1 or value > 0

    if isinstance(value, str):
        return value.upper() in ("Y", "YES", "TRUE", "1", "T")

    return bool(value)


def add_trivial_lr_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add trivial linear-referenced columns for target-side data.

    Target-side data typically doesn't have linear-referenced attributes,
    so we create trivial LR columns with a single range [0.0, 1.0, value]
    for each attribute.

    This enables the same feature computation code to work with both
    Overture (which has LR attributes) and target data.

    Args:
        gdf: GeoDataFrame with flat attribute columns

    Returns:
        GeoDataFrame with added *_lr columns
    """

    # Get name from names struct or flat name column
    def get_name(row):
        names = row.get("names")
        if isinstance(names, dict):
            return names.get("primary")
        return row.get("name")

    # Names LR - extract primary from names struct
    gdf["names_lr"] = gdf.apply(
        lambda row: create_trivial_lr(get_name(row)).to_dict_list(),
        axis=1,
    )

    # Subclass LR
    if "subclass" in gdf.columns:
        gdf["subclass_lr"] = gdf["subclass"].apply(lambda x: create_trivial_lr(x).to_dict_list())
    else:
        gdf["subclass_lr"] = [[{"start": 0.0, "end": 1.0, "value": None}] for _ in range(len(gdf))]

    # Level LR - extract from level_rules if present, otherwise use 0
    def get_level(row):
        level_rules = row.get("level_rules")
        if isinstance(level_rules, list) and len(level_rules) > 0:
            first = level_rules[0]
            if isinstance(first, dict):
                return first.get("value", 0)
        return 0

    gdf["level_lr"] = gdf.apply(
        lambda row: create_trivial_lr(get_level(row)).to_dict_list(),
        axis=1,
    )

    # Road flags LR - extract from road_flags if present
    def get_flags(row):
        road_flags = row.get("road_flags")
        if isinstance(road_flags, list):
            return sorted(road_flags)
        return []

    gdf["road_flags_lr"] = gdf.apply(
        lambda row: create_trivial_lr(get_flags(row)).to_dict_list(),
        axis=1,
    )

    return gdf
