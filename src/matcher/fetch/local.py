"""Load local road datasets from various formats."""

from pathlib import Path

import geopandas as gpd
from loguru import logger

from ..utils import filter_to_linestrings


def load_local_roads(
    path: Path,
    id_column: str | None = None,
    name_column: str | None = None,
    class_column: str | None = None,
    bridge_column: str | None = None,
    tunnel_column: str | None = None,
    layer_column: str | None = None,
) -> gpd.GeoDataFrame:
    """Load local road data from various formats.

    Supports: GeoParquet, Shapefile, GeoJSON, GeoPackage, FileGDB

    Args:
        path: Path to the data file
        id_column: Column name for feature IDs (auto-detected if None)
        name_column: Column name for road names
        class_column: Column name for road class/type
        bridge_column: Column name for bridge flag
        tunnel_column: Column name for tunnel flag
        layer_column: Column name for z-level/layer

    Returns:
        GeoDataFrame with normalized schema
    """
    logger.info(f"Loading local roads from {path}")

    # Load based on file extension
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        gdf = gpd.read_parquet(path)
    elif suffix in [".shp", ".geojson", ".json", ".gpkg"]:
        gdf = gpd.read_file(path)
    elif suffix == ".gdb":
        # FileGDB - need to specify layer
        import fiona

        layers = fiona.listlayers(path)
        logger.info(f"GDB layers: {layers}")
        # Try to find a roads layer
        road_layer = None
        for layer in layers:
            if "road" in layer.lower() or "street" in layer.lower():
                road_layer = layer
                break
        if road_layer is None:
            road_layer = layers[0]
        logger.info(f"Loading layer: {road_layer}")
        gdf = gpd.read_file(path, layer=road_layer)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info(f"Loaded {len(gdf)} features")
    logger.info(f"Columns: {list(gdf.columns)}")

    # Filter to LineString geometries only (drop MultiLineStrings)
    gdf = filter_to_linestrings(gdf, source_name=str(path.name))

    # Normalize schema
    gdf = _normalize_local_schema(
        gdf,
        id_column=id_column,
        name_column=name_column,
        class_column=class_column,
        bridge_column=bridge_column,
        tunnel_column=tunnel_column,
        layer_column=layer_column,
    )

    return gdf


def _normalize_local_schema(
    gdf: gpd.GeoDataFrame,
    id_column: str | None = None,
    name_column: str | None = None,
    class_column: str | None = None,
    bridge_column: str | None = None,
    tunnel_column: str | None = None,
    layer_column: str | None = None,
) -> gpd.GeoDataFrame:
    """Normalize local data schema to standard format."""
    result = gdf.copy()

    # Auto-detect or use provided column mappings
    columns = {c.lower(): c for c in gdf.columns}

    # ID column
    if id_column:
        result["local_id"] = result[id_column]
    elif "id" in columns:
        result["local_id"] = result[columns["id"]]
    elif "objectid" in columns:
        result["local_id"] = result[columns["objectid"]]
    elif "fid" in columns:
        result["local_id"] = result[columns["fid"]]
    else:
        result["local_id"] = range(len(result))

    # Name column
    name_candidates = ["name", "street", "streetname", "st_name", "road_name", "fullname"]
    if name_column:
        result["name"] = result[name_column]
    else:
        for candidate in name_candidates:
            if candidate in columns:
                result["name"] = result[columns[candidate]]
                break
        if "name" not in result.columns:
            result["name"] = None

    # Road class column
    class_candidates = ["class", "road_class", "highway", "funcclass", "roadclass", "type"]
    if class_column:
        result["road_class"] = result[class_column]
    else:
        for candidate in class_candidates:
            if candidate in columns:
                result["road_class"] = result[columns[candidate]]
                break
        if "road_class" not in result.columns:
            result["road_class"] = "unclassified"

    # Bridge column
    if bridge_column and bridge_column in gdf.columns:
        result["is_bridge"] = result[bridge_column].apply(
            lambda x: x is not None and str(x).lower() not in ["no", "0", "false", ""]
        )
    elif "bridge" in columns:
        result["is_bridge"] = result[columns["bridge"]].apply(
            lambda x: x is not None and str(x).lower() not in ["no", "0", "false", ""]
        )
    else:
        result["is_bridge"] = False

    # Tunnel column
    if tunnel_column and tunnel_column in gdf.columns:
        result["is_tunnel"] = result[tunnel_column].apply(
            lambda x: x is not None and str(x).lower() not in ["no", "0", "false", ""]
        )
    elif "tunnel" in columns:
        result["is_tunnel"] = result[columns["tunnel"]].apply(
            lambda x: x is not None and str(x).lower() not in ["no", "0", "false", ""]
        )
    else:
        result["is_tunnel"] = False

    # Layer/level column
    if layer_column and layer_column in gdf.columns:
        result["layer"] = result[layer_column].apply(_parse_layer)
    elif "layer" in columns:
        result["layer"] = result[columns["layer"]].apply(_parse_layer)
    elif "level" in columns:
        result["layer"] = result[columns["level"]].apply(_parse_layer)
    else:
        result["layer"] = 0

    return result


def _parse_layer(value) -> int:
    """Parse layer/level value to integer."""
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
