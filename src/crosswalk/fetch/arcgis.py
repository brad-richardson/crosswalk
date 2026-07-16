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
from ..utils.dataframe import find_id_column
from ..utils.geometry import convert_polygons_to_centerlines
from .metadata import FetchMetadata, save_metadata
from .normalize import (
    default_class_for_type,
    map_column,
    normalize_oneway_value,
    normalize_speed_to_kph,
    resolve_column,
)
from .physical_tags import (
    add_trivial_lr_columns,
    build_level_rules,
    build_road_flags,
)


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
    bridge_values: list[str | int | float] | None = None,
    tunnel_values: list[str | int | float] | None = None,
    status_column: str | None = None,
    status_mapping: dict | None = None,
    source_name: str = "ArcGIS",
    page_size: int = 5000,
    where_clause: str = "1=1",
    bbox: tuple[float, float, float, float] | None = None,
    exclude: dict[str, list[str]] | None = None,
    oneway_column: str | None = None,
    speed_limit_column: str | None = None,
    speed_limit_unit: str = "kph",
    id_column: str | None = None,
    polygon_to_centerline: bool = False,
    dataset_type: str | None = None,
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
        oneway_column: Column name for one-way direction
        speed_limit_column: Column name for speed limit
        speed_limit_unit: Unit of speed limit values ("kph" or "mph")
        polygon_to_centerline: Convert polygon geometries to centerline LineStrings
        dataset_type: Dataset type (e.g., "sidewalk", "bike") for default class

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
    source_id_col = find_id_column(gdf, raise_on_missing=False) if len(gdf) > 0 else None

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
        bridge_values=bridge_values,
        tunnel_values=tunnel_values,
        status_column=status_column,
        status_mapping=status_mapping,
        source_name=source_name,
        exclude=exclude,
        oneway_column=oneway_column,
        speed_limit_column=speed_limit_column,
        speed_limit_unit=speed_limit_unit,
        id_column=id_column,
        polygon_to_centerline=polygon_to_centerline,
        dataset_type=dataset_type,
    )

    # Deduplicate by ID (ArcGIS pagination can return duplicates if data changes during fetch)
    if len(gdf) > 0 and "id" in gdf.columns:
        n_before = len(gdf)
        gdf = gdf.drop_duplicates(subset=["id"], keep="first")
        n_dropped = n_before - len(gdf)
        if n_dropped > 0:
            logger.info(
                f"{source_name}: {n_dropped} duplicate IDs removed "
                f"(pagination artifacts, kept first occurrence)"
            )

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
        notes=f"Fetched from {source_name}"
        + (" (polygon-to-centerline conversion applied)" if polygon_to_centerline else ""),
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
    exclude: dict[str, list[str]] | None = None,
    oneway_column: str | None = None,
    speed_limit_column: str | None = None,
    speed_limit_unit: str = "kph",
    id_column: str | None = None,
    polygon_to_centerline: bool = False,
    dataset_type: str | None = None,
    bridge_values: list[str | int | float] | None = None,
    tunnel_values: list[str | int | float] | None = None,
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
        oneway_column: Column name for one-way direction
        speed_limit_column: Column name for speed limit
        speed_limit_unit: Unit of speed limit values ("kph" or "mph")
        polygon_to_centerline: Convert polygon geometries to centerline LineStrings

    Returns:
        GeoDataFrame with Overture-compatible schema
    """
    if len(gdf) == 0:
        return gdf

    # Convert polygons to centerlines if enabled (before schema transform)
    if polygon_to_centerline:
        gdf = convert_polygons_to_centerlines(gdf, source_name=source_name)
        if len(gdf) == 0:
            return gdf

    # Resolve configured column names case-insensitively
    name_column = resolve_column(gdf, name_column)
    class_column = resolve_column(gdf, class_column)
    subclass_column = resolve_column(gdf, subclass_column)
    level_column = resolve_column(gdf, level_column)
    bridge_column = resolve_column(gdf, bridge_column)
    tunnel_column = resolve_column(gdf, tunnel_column)
    status_column = resolve_column(gdf, status_column)
    oneway_column = resolve_column(gdf, oneway_column)
    speed_limit_column = resolve_column(gdf, speed_limit_column)

    # Apply exclude filter if configured
    if exclude:
        for column, values in exclude.items():
            resolved = resolve_column(gdf, column)
            if resolved:
                before_count = len(gdf)
                gdf = gdf[~gdf[resolved].isin(values)]
                excluded = before_count - len(gdf)
                if excluded > 0:
                    logger.info(f"Excluded {excluded} features where {resolved} in {values}")
        if len(gdf) == 0:
            return gdf

    # Find ID column - MUST be specified in config to ensure stable IDs
    # Sequential or auto-detected IDs are NOT stable across data refreshes
    # and break label linkage when data is re-fetched
    if not id_column:
        raise ValueError(
            f"id_column must be specified in fetch config for {source_name}. "
            f"Available columns: {list(gdf.columns)}. "
            "Choose a stable upstream ID column (e.g., OBJECTID, FID)."
        )

    # Resolve id_column case-insensitively
    id_col = resolve_column(gdf, id_column)
    if not id_col:
        raise ValueError(
            f"Configured id_column '{id_column}' not found in data for {source_name}. "
            f"Available columns: {list(gdf.columns)}"
        )

    # Store original columns for source_tags (before we modify anything)
    original_cols = [c for c in gdf.columns if c != "geometry"]
    source_tags_data = gdf[original_cols].to_dict(orient="records")

    # Build result data dict
    data = {}

    # ID: prefix + objectid + spatial suffix (H3 hex for disambiguation)
    from ..utils.spatial_id import compute_spatial_suffix

    suffixes = gdf.geometry.apply(compute_spatial_suffix)
    data["id"] = [f"{id_prefix}_{uid}_{sfx}" for uid, sfx in zip(gdf[id_col], suffixes)]

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
            data["class"] = map_column(gdf[class_column], class_mapping, fallback="unknown")
        else:
            data["class"] = gdf[class_column].fillna("unknown").astype(str).values
    else:
        data["class"] = [default_class_for_type(dataset_type)] * len(gdf)

    # Subtype (constant)
    data["subtype"] = ["road"] * len(gdf)

    # Subclass with mapping
    if subclass_column and subclass_column in gdf.columns:
        if subclass_mapping:
            data["subclass"] = map_column(gdf[subclass_column], subclass_mapping)
        else:
            data["subclass"] = gdf[subclass_column].astype(str).values
    else:
        data["subclass"] = [None] * len(gdf)

    # Sources array
    data["sources"] = (
        gdf[id_col].apply(lambda x: [{"dataset": source_name, "record_id": str(x)}]).values
    )

    # Road flags (bridge/tunnel indicators)
    data["road_flags"] = build_road_flags(
        gdf,
        bridge_column,
        tunnel_column,
        bridge_values=bridge_values,
        tunnel_values=tunnel_values,
    )

    # Level rules
    if level_column and level_column in gdf.columns:
        data["level_rules"] = gdf[level_column].apply(build_level_rules).values
    else:
        data["level_rules"] = [[] for _ in range(len(gdf))]

    # Status (lifecycle: proposed, construction, etc.)
    if status_column and status_column in gdf.columns:
        if status_mapping:
            data["status"] = map_column(gdf[status_column], status_mapping)
        else:
            data["status"] = gdf[status_column].astype(str).values
    else:
        data["status"] = [None] * len(gdf)

    # One-way direction - normalize to standard format
    if oneway_column and oneway_column in gdf.columns:
        data["oneway"] = gdf[oneway_column].apply(normalize_oneway_value).values
    else:
        data["oneway"] = [None] * len(gdf)

    # Speed limit - normalize to kph
    if speed_limit_column and speed_limit_column in gdf.columns:
        data["speed_limit_kph"] = (
            gdf[speed_limit_column]
            .apply(lambda x: normalize_speed_to_kph(x, speed_limit_unit))
            .values
        )
    else:
        data["speed_limit_kph"] = [None] * len(gdf)

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
