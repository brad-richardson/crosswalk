"""Fetch target/local road data from municipal GIS portals worldwide.

Downloads road centerlines, sidewalks, and trail data from various GIS portals
and converts them to GeoParquet with Overture-compatible schema.

Supports multiple fetch types based on source.type in YAML config:
- arcgis: ArcGIS FeatureServer/MapServer (default)
- download: File downloads (Shapefile, GeoPackage)
- ogc_features: OGC API Features (modern REST API)
- wfs: Web Feature Service
- manual: Requires manual download (skipped)

Usage:
    from crosswalk.fetch.target import fetch_dataset, list_datasets

    # Fetch specific dataset
    fetch_dataset("us_boston_streets")

    # Fetch all datasets for a prefix
    fetch_datasets_by_prefix("us_boston")

    # List available datasets
    list_datasets()
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
import shapely
from loguru import logger

from ..datasets.schema import FetchConfig, get_dataset_config, list_dataset_configs
from ..filenames import target_filename
from ..utils.geometry import convert_polygons_to_centerlines, flatten_to_linestring
from ..utils.linear_ref import create_trivial_lr
from .arcgis import fetch_arcgis_layer
from .normalize import (
    default_class_for_type,
    map_column,
    normalize_oneway_value,
    normalize_speed_to_kph,
    resolve_column,
)

# Default output directory
DEFAULT_DATA_DIR = Path("data/raw")


# Null markers found in source datasets. HK Transport Department uses
# U+2013 (EN DASH) + U+FF19 (full-width 9) as their null sentinel "–９９".
_SENTINEL_VALUES = {"-99", "", "\u2013\uff19\uff19"}


def _is_valid_name(value) -> bool:
    """Check if a name value is valid (not None, NaN, or sentinel)."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    s = str(value).strip()
    return s not in _SENTINEL_VALUES


def _build_multilingual_names(
    name_columns: dict[str, str],
    source_tags: list[dict],
) -> list[dict | None]:
    """Build Overture-compatible multilingual names structs from source columns.

    Args:
        gdf: Input GeoDataFrame
        name_columns: Map of language code -> source column name.
            Keys with 'alias_' prefix become alternate variant rules.
        source_tags: List of source tag dicts (one per row)

    Returns:
        List of names dicts (Overture format) or None per row
    """
    results = []
    # Separate primary/common columns from alias columns
    primary_cols = {k: v for k, v in name_columns.items() if not k.startswith("alias_")}
    alias_cols = {k: v for k, v in name_columns.items() if k.startswith("alias_")}

    for tags in source_tags:
        common = {}
        rules = []

        # Build common names from primary columns
        for lang, col_name in primary_cols.items():
            val = tags.get(col_name)
            if _is_valid_name(val):
                common[lang] = str(val).strip()

        # Build alternate variant rules from alias columns
        for key, col_name in alias_cols.items():
            val = tags.get(col_name)
            if _is_valid_name(val):
                # Extract language from alias_XX key
                lang = key.removeprefix("alias_")
                rules.append(
                    {
                        "value": str(val).strip(),
                        "variant": "alternate",
                        "language": lang,
                    }
                )

        if not common and not rules:
            results.append(None)
            continue

        # Use first common value as primary, falling back to first rule value
        primary = next(iter(common.values()), None)
        if primary is None and rules:
            primary = rules[0]["value"]

        names_struct = {"primary": primary}
        if common:
            names_struct["common"] = common
        if rules:
            names_struct["rules"] = rules

        results.append(names_struct)

    return results


def _transform_download_data(
    gdf: gpd.GeoDataFrame,
    fetch_config: FetchConfig,
    source_name: str,
    dataset_type: str | None = None,
    default_subclass: str | None = None,
) -> gpd.GeoDataFrame:
    """Transform downloaded data to Overture-compatible schema.

    Args:
        gdf: Input GeoDataFrame
        fetch_config: FetchConfig with column mappings and transform settings
        source_name: Name for the data source (used as id_prefix fallback)
        dataset_type: Dataset type (e.g., "sidewalk", "bike") for default class
        default_subclass: Default subclass if no column provided
    """
    if len(gdf) == 0:
        return gdf

    # Read settings from fetch_config, with fallbacks
    id_prefix = fetch_config.id_prefix if fetch_config.id_prefix else source_name
    id_column = fetch_config.id_column
    name_column = fetch_config.name_column
    class_column = fetch_config.class_column
    class_mapping = fetch_config.class_mapping
    subclass_column = fetch_config.subclass_column
    subclass_mapping = fetch_config.subclass_mapping
    exclude = fetch_config.exclude
    oneway_column = fetch_config.oneway_column
    speed_limit_column = fetch_config.speed_limit_column
    speed_limit_unit = fetch_config.speed_limit_unit
    polygon_to_centerline = fetch_config.polygon_to_centerline

    # Convert polygons to centerlines if enabled (before any other processing)
    if polygon_to_centerline:
        gdf = convert_polygons_to_centerlines(gdf, source_name=source_name)
        if len(gdf) == 0:
            return gdf

    # Resolve configured column names case-insensitively
    name_column = resolve_column(gdf, name_column)
    class_column = resolve_column(gdf, class_column)
    subclass_column = resolve_column(gdf, subclass_column)
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

    # Flatten MultiLineStrings to LineStrings (merge contiguous parts, else
    # longest disjoint part) rather than dropping multi-part data.
    multi_mask = gdf.geometry.geom_type == "MultiLineString"
    if multi_mask.any():
        gdf.geometry = gdf.geometry.apply(flatten_to_linestring)
        n_flattened = int((multi_mask & (gdf.geometry.geom_type == "LineString")).sum())
        logger.info(f"Flattened {n_flattened} MultiLineStrings to LineStrings")

    # Filter to LineStrings only (drop remaining non-line geoms: Points, etc.)
    linestring_mask = gdf.geometry.geom_type == "LineString"
    if not linestring_mask.all():
        n_filtered = (~linestring_mask).sum()
        logger.warning(f"Filtering {n_filtered} non-LineString geometries from {source_name}")
        gdf = gdf[linestring_mask].copy()

    # Strip Z coordinates if present (force 2D)
    if gdf.geometry.has_z.any():
        n_3d = gdf.geometry.has_z.sum()
        logger.info(f"Stripping Z coordinates from {n_3d} geometries")
        gdf.geometry = shapely.force_2d(gdf.geometry)

    # Find ID column - MUST be specified in config to ensure stable IDs
    # Sequential or auto-detected IDs are NOT stable across data refreshes
    # and break label linkage when data is re-fetched
    if not id_column:
        raise ValueError(
            f"id_column must be specified in fetch config for {source_name}. "
            f"Available columns: {list(gdf.columns)}. "
            "Choose a stable upstream ID column (e.g., OBJECTID, FID, CENTLNID)."
        )

    # Resolve id_column case-insensitively
    id_col = resolve_column(gdf, id_column)
    if not id_col:
        raise ValueError(
            f"Configured id_column '{id_column}' not found in data for {source_name}. "
            f"Available columns: {list(gdf.columns)}"
        )

    # Store original columns
    original_cols = [c for c in gdf.columns if c != "geometry"]
    source_tags_data = gdf[original_cols].to_dict(orient="records")

    # Compute spatial suffix for ID disambiguation (H3 hex)
    from ..utils.spatial_id import compute_spatial_suffix

    suffixes = gdf.geometry.apply(compute_spatial_suffix)

    # Build result
    data: dict[str, Any] = {
        "id": [f"{id_prefix}_{uid}_{sfx}" for uid, sfx in zip(gdf[id_col], suffixes)],
        "subtype": ["road"] * len(gdf),
        "sources": gdf[id_col]
        .apply(lambda x: [{"dataset": source_name, "record_id": str(x)}])
        .values,
        "road_flags": [[] for _ in range(len(gdf))],
        "level_rules": [[] for _ in range(len(gdf))],
        "source_tags": source_tags_data,
    }

    # Names
    name_columns_config = fetch_config.name_columns
    if name_columns_config:
        # Build multilingual names struct from configured column mapping
        data["names"] = _build_multilingual_names(name_columns_config, source_tags_data)
    elif name_column and name_column in gdf.columns:
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

    # Subclass with mapping
    if subclass_column and subclass_column in gdf.columns:
        if subclass_mapping:
            data["subclass"] = map_column(gdf[subclass_column], subclass_mapping)
        else:
            data["subclass"] = gdf[subclass_column].astype(str).values
    elif default_subclass is not None:
        data["subclass"] = [default_subclass] * len(gdf)
    else:
        data["subclass"] = [None] * len(gdf)

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

    result = gpd.GeoDataFrame(data, geometry=gdf.geometry.values, crs=gdf.crs)

    # Deduplicate by composite ID (same upstream ID + same H3 cell = true duplicate)
    if len(result) > 0 and "id" in result.columns:
        n_before = len(result)
        result = result.drop_duplicates(subset=["id"], keep="first")
        n_dropped = n_before - len(result)
        if n_dropped > 0:
            logger.info(f"{source_name}: {n_dropped} duplicate IDs removed (kept first occurrence)")

    # Add trivial linear-referenced columns
    result = _add_trivial_lr_columns(result)

    return result


def _add_trivial_lr_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add trivial linear-referenced columns for target-side data.

    Target-side data typically doesn't have linear-referenced attributes,
    so we create trivial LR columns with a single range [0.0, 1.0, value]
    for each attribute.

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
        gdf["subclass_lr"] = [[{"between": [0.0, 1.0], "value": None}] for _ in range(len(gdf))]

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

    # One-way LR - extract from oneway flat column
    if "oneway" in gdf.columns:
        gdf["oneway_lr"] = gdf["oneway"].apply(lambda x: create_trivial_lr(x).to_dict_list())
    else:
        gdf["oneway_lr"] = [[{"between": [0.0, 1.0], "value": None}] for _ in range(len(gdf))]

    # Speed limit LR - extract from speed_limit_kph flat column
    if "speed_limit_kph" in gdf.columns:
        gdf["speed_limit_kph_lr"] = gdf["speed_limit_kph"].apply(
            lambda x: create_trivial_lr(x).to_dict_list()
        )
    else:
        gdf["speed_limit_kph_lr"] = [
            [{"between": [0.0, 1.0], "value": None}] for _ in range(len(gdf))
        ]

    return gdf


def fetch_ogc_features(
    url: str,
    output_path: Path,
    fetch_config: FetchConfig,
    source_name: str = "OGC API Features",
    bbox: tuple | None = None,
    limit_per_page: int = 5000,
    dataset_type: str | None = None,
) -> Path:
    """Fetch geospatial data from an OGC API Features endpoint.

    OGC API Features is a modern REST API standard for geospatial data that
    returns GeoJSON. Supports pagination and bbox filtering.

    Args:
        url: OGC API Features items endpoint URL
        output_path: Path for output GeoParquet file
        fetch_config: FetchConfig with column mappings and transform settings
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter
        limit_per_page: Number of features per API request
        dataset_type: Dataset type for default class assignment

    Returns:
        Path to the output GeoParquet file
    """
    logger.info(f"Fetching OGC API Features: {url}")

    all_features = []
    start_index = 0

    # Build base params
    params: dict[str, Any] = {"f": "json", "limit": limit_per_page}
    if bbox:
        params["bbox"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    while True:
        params["startIndex"] = start_index

        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        features = data.get("features", [])

        if not features:
            break

        all_features.extend(features)
        logger.debug(f"Fetched {len(all_features)} features so far...")

        # Check for more results
        # numberMatched tells us total, numberReturned tells us this batch
        number_matched = data.get("numberMatched", 0)
        if len(all_features) >= number_matched or len(features) < limit_per_page:
            break

        start_index += len(features)

    if not all_features:
        logger.warning(f"No features returned from {url}")
        return output_path

    logger.info(f"Fetched {len(all_features)} total features")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")

    # Transform to Overture schema
    gdf = _transform_download_data(
        gdf,
        fetch_config=fetch_config,
        source_name=source_name,
        dataset_type=dataset_type,
    )

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, write_covering_bbox=True)

    logger.success(f"Saved {len(gdf)} features to {output_path}")
    return output_path


def fetch_wfs(
    url: str,
    type_name: str,
    output_path: Path,
    fetch_config: FetchConfig,
    source_name: str = "WFS",
    bbox: tuple | None = None,
    max_features: int = 300000,
    dataset_type: str | None = None,
) -> Path:
    """Fetch geospatial data from a WFS (Web Feature Service) endpoint.

    Args:
        url: WFS service base URL
        type_name: Layer/type name to fetch (e.g., 'geoportal:segmento_logradouro')
        output_path: Path for output GeoParquet file
        fetch_config: FetchConfig with column mappings and transform settings
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter
        max_features: Maximum features to fetch (pagination limit)
        dataset_type: Dataset type for default class assignment

    Returns:
        Path to the output GeoParquet file
    """
    logger.info(f"Fetching WFS: {url} / {type_name}")

    all_features = []
    start_index = 0
    page_size = 5000

    while True:
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": type_name,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": page_size,
            "startIndex": start_index,
        }
        # Note: bbox filtering disabled due to GeoServer coordinate order issues
        # The full dataset will be downloaded and filtered locally if needed

        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        features = data.get("features", [])

        if not features:
            break

        all_features.extend(features)
        logger.debug(f"Fetched {len(all_features)} features so far...")

        # Check if we got all features or hit the limit
        if len(features) < page_size or len(all_features) >= max_features:
            break

        start_index += len(features)

    if not all_features:
        logger.warning(f"No features returned from {url}")
        return output_path

    logger.info(f"Fetched {len(all_features)} total features")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")

    # Transform to Overture schema
    gdf = _transform_download_data(
        gdf,
        fetch_config=fetch_config,
        source_name=source_name,
        dataset_type=dataset_type,
    )

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, write_covering_bbox=True)

    logger.success(f"Saved {len(gdf)} features to {output_path}")
    return output_path


def fetch_lta_geospatial(
    layer_id: str,
    output_path: Path,
    fetch_config: FetchConfig,
    api_key: str,
    source_name: str = "LTA DataMall",
    bbox: tuple | None = None,
    bbox_filter: bool = False,
    dataset_type: str | None = None,
) -> Path:
    """Fetch geospatial data from Singapore LTA DataMall API.

    The LTA API returns a presigned S3 URL for downloading shapefiles.
    API endpoint: https://datamall2.mytransport.sg/ltaodataservice/GeospatialWholeIsland

    Args:
        layer_id: Geospatial layer ID (e.g., 'RoadSectionLine', 'Footpath')
        output_path: Path for output GeoParquet file
        fetch_config: FetchConfig with column mappings and transform settings
        api_key: LTA DataMall API key
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter
        bbox_filter: Whether to apply bbox filter after loading
        dataset_type: Dataset type for default class assignment

    Returns:
        Path to the output GeoParquet file
    """
    api_url = "https://datamall2.mytransport.sg/ltaodataservice/GeospatialWholeIsland"

    logger.info(f"Fetching LTA geospatial layer: {layer_id}")

    # Get presigned download URL from API
    resp = requests.get(
        api_url,
        params={"ID": layer_id},
        headers={"AccountKey": api_key, "Accept": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()

    data = resp.json()
    if "value" not in data or not data["value"]:
        raise ValueError(f"No download link returned for layer {layer_id}")

    download_url = data["value"][0].get("Link")
    if not download_url:
        raise ValueError(f"Empty download link for layer {layer_id}")

    logger.info("Got presigned S3 URL, downloading shapefile...")

    # Download the actual file using the presigned URL
    return fetch_download(
        url=download_url,
        output_path=output_path,
        file_format="shp",
        fetch_config=fetch_config,
        source_name=source_name,
        bbox=bbox,
        bbox_filter=bbox_filter,
        dataset_type=dataset_type,
    )


def _get_download_cache_dir() -> Path:
    """Get or create download cache directory."""
    cache_dir = Path.home() / ".cache" / "matcher" / "downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cached_download(url: str, cache_ttl_hours: int = 168) -> Path | None:
    """Check if cached download exists and is valid.

    Args:
        url: URL that was downloaded
        cache_ttl_hours: Hours before cache expires (default: 168 = 7 days)

    Returns:
        Path to cached file if valid, None otherwise
    """
    import hashlib
    from datetime import UTC, datetime, timedelta

    # Use URL hash as cache key
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    cache_dir = _get_download_cache_dir()

    # Look for any cached file with this hash prefix
    cached_files = list(cache_dir.glob(f"{url_hash}_*"))
    if not cached_files:
        return None

    cached_file = cached_files[0]
    mtime = datetime.fromtimestamp(cached_file.stat().st_mtime, tz=UTC)
    if datetime.now(UTC) - mtime < timedelta(hours=cache_ttl_hours):
        logger.info(f"Using cached download: {cached_file}")
        return cached_file
    else:
        logger.info(f"Cache expired for {url}, re-downloading...")
        cached_file.unlink()
        return None


def _load_ms_roads_tsv(
    path: Path,
    country_filter: str | None = None,
) -> gpd.GeoDataFrame:
    """Load Microsoft Road Detections TSV file.

    Format: country_code<TAB>GeoJSON_Feature per line.

    Args:
        path: Path to the TSV file
        country_filter: If set, only include rows matching this country code prefix

    Returns:
        GeoDataFrame in EPSG:4326
    """
    import json

    from shapely.geometry import shape

    features = []
    total = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            country_code, geojson_str = parts

            if country_filter and country_code != country_filter:
                continue

            total += 1
            try:
                feature = json.loads(geojson_str)
                geom = shape(feature["geometry"])
                props = feature.get("properties", {})
                row = {"geometry": geom}
                row.update(props)
                features.append(row)
            except (json.JSONDecodeError, KeyError):
                continue

    logger.info(f"Parsed {len(features)} features from TSV (filtered from {total} matching rows)")

    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")


# Extension patterns for each file format (used for ZIP extraction)
_FORMAT_EXTENSIONS: dict[str, list[str]] = {
    "shp": ["*.shp"],
    "gpkg": ["*.gpkg"],
    "gml": ["*.gml"],
    "geojson": ["*.geojson", "*.json"],
    "ms_roads_tsv": ["*.tsv"],
}


def _find_data_file_in_zip(
    extract_path: Path,
    file_format: str,
    file_pattern: str | None = None,
) -> Path:
    """Find the data file within an extracted ZIP archive.

    Args:
        extract_path: Root directory of extracted ZIP contents
        file_format: Expected format (shp, gpkg, gml, geojson, gdb, ms_roads_tsv)
        file_pattern: Optional glob pattern to select a specific file (matched
            against path relative to extract_path, e.g. "UTF-8/*.geojson")

    Returns:
        Path to the data file to load
    """
    import fnmatch

    # GDB is a directory, not a file
    if file_format == "gdb":
        found_dirs = [d for d in extract_path.rglob("*.gdb") if d.is_dir()]
        if not found_dirs:
            raise ValueError("No .gdb folder found in zip")
        return found_dirs[0]

    # Look up extensions for this format
    extensions = _FORMAT_EXTENSIONS.get(file_format)
    if extensions is None:
        raise ValueError(f"Unsupported file format: {file_format}")

    found_files: list[Path] = []
    for ext in extensions:
        found_files.extend(extract_path.rglob(ext))

    # ms_roads_tsv fallback: try extensionless files
    if not found_files and file_format == "ms_roads_tsv":
        found_files = [f for f in extract_path.rglob("*") if f.is_file() and not f.suffix]

    if not found_files:
        raise ValueError(f"No {file_format} file found in zip")

    # Filter by file_pattern if specified (match against relative path)
    if file_pattern and len(found_files) > 1:
        filtered = [
            f
            for f in found_files
            if fnmatch.fnmatch(str(f.relative_to(extract_path)), file_pattern)
        ]
        if filtered:
            found_files = filtered

    return found_files[0]


# Preferred layer names for multi-layer formats (gdb)
_GDB_PREFERRED_LAYERS = [
    "CENTERLINE",
    "centerline",
    "road_link",
    "RoadLink",
    "roads",
    "road",
]


def _load_geodata_file(
    data_file: Path,
    file_format: str,
    encoding: str | None = None,
) -> gpd.GeoDataFrame:
    """Load a geodata file using geopandas with format-specific options.

    Handles GML driver selection and GDB layer auto-selection.
    """
    read_kwargs: dict[str, Any] = {}
    if encoding:
        read_kwargs["encoding"] = encoding
        logger.info(f"Using encoding: {encoding}")
    if file_format == "gml":
        read_kwargs["driver"] = "GML"
    elif file_format == "gdb":
        import pyogrio

        layers = pyogrio.list_layers(data_file)
        layer_names = [layer_info[0] for layer_info in layers]
        logger.debug(f"Available layers: {layer_names}")
        for preferred in _GDB_PREFERRED_LAYERS:
            if preferred in layer_names:
                read_kwargs["layer"] = preferred
                logger.info(f"Selected layer: {preferred}")
                break

    return gpd.read_file(data_file, **read_kwargs)


def fetch_download(
    url: str,
    output_path: Path,
    file_format: str,
    fetch_config: FetchConfig,
    source_name: str = "Download",
    bbox: tuple | None = None,
    bbox_filter: bool = False,
    api_key: str | None = None,
    api_key_header: str | None = None,
    cache_download: bool = False,
    cache_ttl_hours: int = 168,
    dataset_type: str | None = None,
    file_pattern: str | None = None,
    where_clause: str | None = None,
) -> Path:
    """Download and process geospatial file (Shapefile, GeoPackage, GML, TSV).

    Args:
        url: URL to download from
        output_path: Path for output GeoParquet file
        file_format: Format of downloaded file (shp, gpkg, gml, geojson, ms_roads_tsv)
        fetch_config: FetchConfig with column mappings and transform settings
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter
        bbox_filter: Whether to apply bbox filter after loading
        api_key: Optional API key for authenticated downloads
        api_key_header: Header name for API key (e.g., "AccountKey")
        cache_download: If True, cache the downloaded file for future use
        cache_ttl_hours: Hours before cache expires (default: 168 = 7 days)
        dataset_type: Dataset type for default class assignment
        file_pattern: Glob pattern to select specific file within ZIP archive
        where_clause: Filter clause (e.g., country code for ms_roads_tsv format)

    Returns:
        Path to the output GeoParquet file
    """
    import hashlib

    # Read encoding and source_crs from fetch_config
    encoding = fetch_config.encoding
    source_crs = fetch_config.source_crs

    logger.info(f"Downloading {file_format} from: {url}")

    # Check cache first if caching is enabled
    cached_file = None
    if cache_download:
        cached_file = _get_cached_download(url, cache_ttl_hours)

    headers = {}
    if api_key and api_key_header:
        headers[api_key_header] = api_key

    try:
        # If we have a cached file, use it directly
        if cached_file:
            data_file = cached_file
            is_zip = cached_file.suffix == ".zip"
            tmpdir_context = None
        else:
            # Download the file
            resp = requests.get(url, headers=headers, timeout=600, stream=True)
            resp.raise_for_status()

            # Get content length for progress reporting
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            last_progress_mb = 0
            progress_interval_mb = 100  # Report every 100 MB

            # Determine if it's a zip file
            content_type = resp.headers.get("content-type", "")
            is_zip = "zip" in content_type or url.endswith(".zip")

            # Determine where to save (cache or temp)
            if cache_download:
                url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
                ext = ".zip" if is_zip else f".{file_format}"
                cache_path = _get_download_cache_dir() / f"{url_hash}_{Path(url).stem}{ext}"
                download_path = cache_path
            else:
                tmpdir_context = tempfile.TemporaryDirectory()
                tmpdir_path = Path(tmpdir_context.name)
                ext = ".zip" if is_zip else f".{file_format}"
                download_path = tmpdir_path / f"download{ext}"

            # Download with progress reporting
            with open(download_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    current_mb = downloaded // (1024 * 1024)
                    if current_mb >= last_progress_mb + progress_interval_mb:
                        last_progress_mb = current_mb
                        if total_size:
                            pct = 100 * downloaded / total_size
                            logger.info(
                                f"Download progress: {current_mb} MB / "
                                f"{total_size // (1024 * 1024)} MB ({pct:.0f}%)"
                            )
                        else:
                            logger.info(f"Download progress: {current_mb} MB")

            logger.info(f"Downloaded {download_path.stat().st_size / 1024 / 1024:.1f} MB")
            if cache_download:
                logger.info(f"Cached to: {download_path}")

            data_file = download_path

        # Process the downloaded file
        if is_zip:
            # Need a temp directory for extraction
            extract_dir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(data_file, "r") as zf:
                    zf.extractall(extract_dir)
                extract_path = Path(extract_dir)

                # Find the data file within the extracted ZIP
                data_file = _find_data_file_in_zip(extract_path, file_format, file_pattern)
            finally:
                # Clean up extract directory after we're done (handled after geopandas load)
                pass

        logger.info(f"Loading from: {data_file}")

        # Load data based on format
        if file_format == "ms_roads_tsv":
            gdf = _load_ms_roads_tsv(data_file, country_filter=where_clause)
            logger.info(f"Loaded {len(gdf)} features from MS Roads TSV")
        else:
            gdf = _load_geodata_file(data_file, file_format, encoding)
            logger.info(f"Loaded {len(gdf)} features")

        # Set source CRS if provided (overrides file metadata)
        if source_crs:
            logger.info(f"Setting source CRS from config: {source_crs}")
            gdf = gdf.set_crs(source_crs, allow_override=True)

        # Reproject to WGS84 if needed
        if gdf.crs and gdf.crs != "EPSG:4326":
            logger.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")

        # Apply bbox filter if requested
        if bbox_filter and bbox and len(gdf) > 0:
            xmin, ymin, xmax, ymax = bbox
            original_count = len(gdf)
            gdf = gdf.cx[xmin:xmax, ymin:ymax]
            logger.info(f"Filtered from {original_count} to {len(gdf)} features within bbox")

        if len(gdf) == 0:
            logger.warning("No features after filtering")
            return output_path

        # Compute geometry hash IDs if configured
        if fetch_config.id_column == "_geom_hash" and "_geom_hash" not in gdf.columns:
            import hashlib

            gdf["_geom_hash"] = gdf.geometry.apply(
                lambda g: hashlib.md5(shapely.to_wkt(g, rounding_precision=7).encode()).hexdigest()[
                    :12
                ]
            )
            logger.info(f"Computed geometry hash IDs for {len(gdf)} features")

        # Transform to Overture schema
        gdf = _transform_download_data(
            gdf,
            fetch_config=fetch_config,
            source_name=source_name,
            dataset_type=dataset_type,
        )

        # Save to parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(output_path)
        logger.success(f"Saved {len(gdf)} features to {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Failed to download: {e}")
        raise


def _get_os_cache_dir() -> Path:
    """Get or create OS Data Hub cache directory."""
    cache_dir = Path.home() / ".cache" / "matcher" / "os_downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cached_os_data(product_id: str, cache_ttl_hours: int = 168) -> Path | None:
    """Check if cached OS data exists and is valid.

    Args:
        product_id: OS product ID
        cache_ttl_hours: Cache TTL in hours (default: 7 days)

    Returns:
        Path to cached gpkg file if valid, None otherwise
    """
    from datetime import UTC, datetime, timedelta

    cache_dir = _get_os_cache_dir()
    cached_gpkg = cache_dir / f"{product_id}.gpkg"

    if cached_gpkg.exists():
        mtime = datetime.fromtimestamp(cached_gpkg.stat().st_mtime, tz=UTC)
        if datetime.now(UTC) - mtime < timedelta(hours=cache_ttl_hours):
            logger.info(f"Using cached OS data: {cached_gpkg}")
            return cached_gpkg
        else:
            logger.info(f"Cache expired for {product_id}, re-downloading...")
            cached_gpkg.unlink()

    return None


def fetch_os_downloads(
    product_id: str,
    output_path: Path,
    fetch_config: FetchConfig,
    source_name: str = "OS Data Hub",
    bbox: tuple | None = None,
    api_key: str | None = None,
    cache_ttl_hours: int = 168,
    dataset_type: str | None = None,
) -> Path:
    """Fetch geospatial data from Ordnance Survey Data Hub Downloads API.

    Uses the osdatahub package for access to OS OpenData products like
    OpenRoads, OpenMap Local, etc. No API key required for open data.
    Downloads are cached for 7 days by default.

    Args:
        product_id: OS Data Hub product ID (e.g., "OpenRoads", "OpenMapLocal")
        output_path: Path for output GeoParquet file
        fetch_config: FetchConfig with column mappings and transform settings
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter after download
        api_key: Not used for OpenData (kept for API compatibility)
        cache_ttl_hours: Cache TTL in hours (default: 168 = 7 days)
        dataset_type: Dataset type for default class assignment

    Returns:
        Path to the output GeoParquet file
    """
    try:
        from osdatahub import OpenDataDownload
    except ImportError as err:
        raise ImportError(
            "osdatahub package required for OS Data Hub downloads. "
            'Install with: pip install "crosswalk-py[os]"'
        ) from err

    logger.info(f"Fetching OS Data Hub product: {product_id}")
    if bbox:
        logger.info(f"Will filter to bbox after download: {bbox}")

    # Check cache first
    cached_gpkg = _get_cached_os_data(product_id, cache_ttl_hours)

    if cached_gpkg:
        data_file = cached_gpkg
    else:
        # Download to cache directory
        cache_dir = _get_os_cache_dir()
        download = OpenDataDownload(product_id)

        # Get available downloads to find GeoPackage format
        available = download.product_list(file_format="GeoPackage")
        if not available:
            available = download.product_list()

        if not available:
            raise ValueError(f"No downloads available for product: {product_id}")

        logger.info(f"Found {len(available)} download option(s)")

        # Download to temp dir then extract to cache
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Download the files (returns list of zip file paths)
            downloaded_files = download.download(
                output_dir=str(tmpdir_path),
                file_format="GeoPackage",
                download_multiple=False,
                overwrite=True,
            )

            if not downloaded_files:
                downloaded_files = download.download(
                    output_dir=str(tmpdir_path),
                    download_multiple=False,
                    overwrite=True,
                )

            logger.info(f"Downloaded {len(downloaded_files)} file(s): {downloaded_files}")

            # The download returns zip files - extract them
            for downloaded_file in downloaded_files:
                downloaded_path = Path(downloaded_file)
                if downloaded_path.suffix == ".zip":
                    logger.info(f"Extracting {downloaded_path.name}...")
                    with zipfile.ZipFile(downloaded_path, "r") as zf:
                        zf.extractall(tmpdir_path / "extracted")

            # Find the gpkg file
            gpkg_files = list(tmpdir_path.rglob("*.gpkg"))
            if not gpkg_files:
                # Also check extracted subdirectory
                gpkg_files = list((tmpdir_path / "extracted").rglob("*.gpkg"))

            if not gpkg_files:
                # Fall back to shapefile
                shp_files = list(tmpdir_path.rglob("*.shp"))
                if not shp_files:
                    shp_files = list((tmpdir_path / "extracted").rglob("*.shp"))
                if shp_files:
                    data_file = shp_files[0]
                    # Copy to cache
                    cached_file = cache_dir / f"{product_id}.shp"
                    shutil.copy(data_file, cached_file)
                    # Copy associated files (.dbf, .shx, .prj)
                    for ext in [".dbf", ".shx", ".prj", ".cpg"]:
                        src = data_file.with_suffix(ext)
                        if src.exists():
                            shutil.copy(src, cached_file.with_suffix(ext))
                    data_file = cached_file
                else:
                    raise ValueError("No .gpkg or .shp found in downloaded data")
            else:
                # Copy gpkg to cache
                cached_gpkg = cache_dir / f"{product_id}.gpkg"
                shutil.copy(gpkg_files[0], cached_gpkg)
                data_file = cached_gpkg
                logger.info(f"Cached to: {cached_gpkg}")

    logger.info(f"Loading from: {data_file}")

    # For GeoPackages with multiple layers, select the road links layer
    layer = None
    if data_file.suffix == ".gpkg":
        import pyogrio

        layers = pyogrio.list_layers(data_file)
        layer_names = [layer_info[0] for layer_info in layers]
        logger.debug(f"Available layers: {layer_names}")

        # Prefer road_link layer for OS OpenRoads (contains actual road segments)
        for preferred in ["road_link", "RoadLink", "roads", "road"]:
            if preferred in layer_names:
                layer = preferred
                break

        if layer:
            logger.info(f"Selected layer: {layer}")

    gdf = gpd.read_file(data_file, layer=layer)
    logger.info(f"Loaded {len(gdf)} features")

    # Reproject to WGS84 if needed (OS data is typically in EPSG:27700)
    if gdf.crs and gdf.crs != "EPSG:4326":
        logger.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    # Apply bbox filter after download (if specified)
    if bbox and len(gdf) > 0:
        xmin, ymin, xmax, ymax = bbox
        original_count = len(gdf)
        gdf = gdf.cx[xmin:xmax, ymin:ymax]
        logger.info(f"Filtered from {original_count} to {len(gdf)} features within bbox")

    if len(gdf) == 0:
        logger.warning("No features to save")
        return output_path

    # Transform to Overture schema
    gdf = _transform_download_data(
        gdf,
        fetch_config=fetch_config,
        source_name=source_name,
        dataset_type=dataset_type,
    )

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)
    logger.success(f"Saved {len(gdf)} features to {output_path}")

    return output_path


def fetch_dataset(
    dataset_name: str,
    output_dir: Path | None = None,
    page_size: int | None = None,
    force: bool = False,
    skip_quality_check: bool = False,
) -> Path | None:
    """Fetch a single dataset based on its YAML configuration.

    Args:
        dataset_name: Name of the dataset (matches YAML filename)
        output_dir: Directory for output files (default: data/raw)
        page_size: Override page size for ArcGIS fetches (default: use arcgis module default)
        force: If False (default), skip if output file already exists
        skip_quality_check: If True, skip quality regression check against saved fingerprint

    Returns:
        Path to output file, or None if fetch failed or requires manual download
    """
    if output_dir is None:
        output_dir = DEFAULT_DATA_DIR

    config = get_dataset_config(dataset_name)
    if config is None:
        logger.error(f"No config found for {dataset_name}")
        return None

    output_path = output_dir / target_filename(dataset_name)

    # Skip if file already exists (unless force=True)
    if not force and output_path.exists():
        logger.info(
            f"Skipping {dataset_name}: {output_path.name} already exists (use --force to re-fetch)"
        )
        return output_path

    logger.info(f"Fetching dataset: {dataset_name}")
    if config.description:
        logger.info(f"  Description: {config.description}")

    source_type = config.source.type if config.source else "unknown"
    url = config.source.url if config.source else None

    result_path: Path | None = None

    try:
        # Handle manual download datasets first
        if source_type == "manual":
            logger.warning(f"Dataset {dataset_name} requires manual download")
            portal_url = config.source.portal_url if config.source else "data portal"
            logger.info(f"  Portal: {portal_url}")
            if config.notes:
                logger.info(f"  Notes: {config.notes}")
            logger.info(
                f"  After downloading, place the file in {output_dir} and convert to parquet."
            )
            return None

        # Build FetchConfig - use config.fetch or create a default one
        fetch_config = config.fetch or FetchConfig(id_prefix=dataset_name)
        # If id_prefix is not set, default to dataset_name
        if not fetch_config.id_prefix:
            fetch_config = fetch_config.model_copy(update={"id_prefix": dataset_name})
        source_name = config.display_name or dataset_name

        # Extract bbox from fetch_config for server-side filtering
        bbox = fetch_config.bbox

        # Handle os_downloads before URL check (it uses product_id, not url)
        if source_type == "os_downloads":
            product_id = config.source.product_id if config.source else None
            if not product_id:
                logger.error("os_downloads source requires product_id in source config")
                return None

            api_key = None
            api_key_env_var = config.source.api_key_env_var if config.source else "OS_API_KEY"
            if api_key_env_var:
                api_key = os.environ.get(api_key_env_var)

            result_path = fetch_os_downloads(
                product_id=product_id,
                output_path=output_path,
                fetch_config=fetch_config,
                source_name=source_name,
                bbox=bbox,
                api_key=api_key,
                dataset_type=config.type,
            )

        elif url is None:
            logger.error(f"No URL in config for {dataset_name}")
            return None

        elif source_type == "download":
            # Get file format from source config
            file_format = config.source.file_format if config.source else "shp"

            # Check for API key requirement
            api_key = None
            api_key_header = config.source.api_key_header if config.source else None
            api_key_env_var = config.source.api_key_env_var if config.source else None

            if api_key_env_var:
                api_key = os.environ.get(api_key_env_var)
                if not api_key:
                    logger.warning(f"Skipping {dataset_name}: API key required ({api_key_env_var})")
                    if "singapore" in dataset_name.lower():
                        logger.info("Register at https://datamall.lta.gov.sg to get an API key")
                    return None  # Gracefully skip instead of attempting fetch

            # Special handling for Singapore LTA DataMall
            # The static download URLs are blocked; must use their API instead
            if api_key_env_var == "LTA_API_KEY" and api_key:
                # Extract layer ID from URL (e.g., RoadSectionLine.zip -> RoadSectionLine)
                layer_id = url.split("/")[-1].replace(".zip", "").split("_")[0]
                logger.info(f"Using LTA DataMall API for layer: {layer_id}")
                result_path = fetch_lta_geospatial(
                    layer_id=layer_id,
                    output_path=output_path,
                    fetch_config=fetch_config,
                    api_key=api_key,
                    source_name=source_name,
                    bbox=bbox,
                    bbox_filter=bool(bbox),
                    dataset_type=config.type,
                )
            else:
                # Check if caching is enabled for this download
                cache_download = config.source.cache_download if config.source else False
                cache_ttl_hours = config.source.cache_ttl_hours if config.source else 168

                # Extract file_pattern and where_clause from source config
                src_file_pattern = config.source.file_pattern if config.source else None
                src_where_clause = config.source.where_clause if config.source else None

                result_path = fetch_download(
                    url=url,
                    output_path=output_path,
                    file_format=file_format,
                    fetch_config=fetch_config,
                    source_name=source_name,
                    bbox=bbox,
                    bbox_filter=bool(bbox),
                    api_key=api_key,
                    api_key_header=api_key_header,
                    cache_download=cache_download,
                    cache_ttl_hours=cache_ttl_hours,
                    dataset_type=config.type,
                    file_pattern=src_file_pattern,
                    where_clause=src_where_clause,
                )

        elif source_type == "ogc_features":
            # OGC API Features (modern REST API for geospatial data)
            result_path = fetch_ogc_features(
                url=url,
                output_path=output_path,
                fetch_config=fetch_config,
                source_name=source_name,
                bbox=bbox,
                dataset_type=config.type,
            )

        elif source_type == "wfs":
            # WFS (Web Feature Service)
            # URL should be base WFS URL, type_name from where_clause field
            type_name = config.source.where_clause if config.source else None
            if not type_name:
                logger.error("WFS source requires type_name in where_clause field")
                return None
            result_path = fetch_wfs(
                url=url,
                type_name=type_name,
                output_path=output_path,
                fetch_config=fetch_config,
                source_name=source_name,
                bbox=bbox,
                dataset_type=config.type,
            )

        else:
            # Default: ArcGIS FeatureServer/MapServer fetch
            # ArcGIS still uses individual params (separate module, not part of this refactor)
            arcgis_kwargs: dict[str, Any] = {
                "url": url,
                "output_path": output_path,
                "id_prefix": fetch_config.id_prefix,
                "name_column": fetch_config.name_column,
                "class_column": fetch_config.class_column,
                "class_mapping": fetch_config.class_mapping,
                "subclass_column": fetch_config.subclass_column,
                "subclass_mapping": fetch_config.subclass_mapping,
                "source_name": source_name,
            }
            if page_size is not None:
                arcgis_kwargs["page_size"] = page_size
            # Pass bbox for server-side filtering (reduces bandwidth for large datasets)
            if bbox:
                arcgis_kwargs["bbox"] = bbox
            # Pass exclude filter
            if fetch_config.exclude:
                arcgis_kwargs["exclude"] = fetch_config.exclude
            # Pass oneway/speed columns
            if fetch_config.oneway_column:
                arcgis_kwargs["oneway_column"] = fetch_config.oneway_column
            if fetch_config.speed_limit_column:
                arcgis_kwargs["speed_limit_column"] = fetch_config.speed_limit_column
                arcgis_kwargs["speed_limit_unit"] = fetch_config.speed_limit_unit
            # Pass id_column for stable IDs
            if fetch_config.id_column:
                arcgis_kwargs["id_column"] = fetch_config.id_column
            if fetch_config.polygon_to_centerline:
                arcgis_kwargs["polygon_to_centerline"] = fetch_config.polygon_to_centerline
            if config.type:
                arcgis_kwargs["dataset_type"] = config.type

            result_path = fetch_arcgis_layer(**arcgis_kwargs)

    except Exception as e:
        logger.error(f"Failed to fetch {dataset_name}: {e}")
        return None

    # Quality regression check and fingerprint auto-update
    if result_path is not None and result_path.exists() and not skip_quality_check:
        from .exceptions import QualityRegressionError

        try:
            fetched_gdf = gpd.read_parquet(result_path)
        except Exception:
            logger.warning(f"Could not read {result_path} for quality check, skipping")
            return result_path

        # Check for regression if fingerprint exists
        if config.quality_fingerprint is not None:
            from ..quality.regression import check_quality_regression

            violations = check_quality_regression(
                fetched_gdf, config.quality_fingerprint, dataset_name
            )
            if violations:
                # Rename output to .suspect so it's not used accidentally
                suspect_path = result_path.with_suffix(".parquet.suspect")
                result_path.rename(suspect_path)
                raise QualityRegressionError(
                    f"Quality regression detected for {dataset_name}: "
                    f"{len(violations)} violation(s). "
                    f"Output renamed to {suspect_path.name}. "
                    f"Use --skip-quality-check to override.\n"
                    + "\n".join(f"  - {v.message}" for v in violations)
                )

        # Auto-update fingerprint after successful fetch (passed checks or first fetch)
        from ..datasets.schema import update_quality_fingerprint
        from ..quality.regression import compute_quick_fingerprint

        new_fp = compute_quick_fingerprint(fetched_gdf)
        update_quality_fingerprint(dataset_name, new_fp)
        if config.quality_fingerprint is None:
            logger.info(f"Created quality fingerprint for {dataset_name}")
        else:
            logger.debug(f"Updated quality fingerprint for {dataset_name}")

    return result_path


def _fetch_dataset_worker(
    args: tuple[str, Path | None, int | None, bool, bool],
) -> tuple[str, Path | None]:
    """Worker function for parallel dataset fetching."""
    name, output_dir, page_size, force, skip_quality_check = args
    try:
        result = fetch_dataset(name, output_dir, page_size, force, skip_quality_check)
        return (name, result)
    except Exception as e:
        logger.error(f"Failed to fetch {name}: {e}")
        return (name, None)


def fetch_datasets_by_prefix(
    prefix: str,
    output_dir: Path | None = None,
    page_size: int | None = None,
    force: bool = False,
    max_workers: int = 4,
    skip_quality_check: bool = False,
) -> dict[str, Path | None]:
    """Fetch all datasets matching a prefix.

    Args:
        prefix: Prefix to match (e.g., "us_boston")
        output_dir: Directory for output files (default: data/raw)
        page_size: Override page size for ArcGIS fetches
        force: If False (default), skip datasets whose files already exist
        max_workers: Maximum number of parallel downloads (default: 4)
        skip_quality_check: If True, skip quality regression checks

    Returns:
        Dict mapping dataset names to output paths (None if failed)
    """
    all_datasets = list_dataset_configs()
    matching = [d for d in all_datasets if d.startswith(prefix)]

    if not matching:
        logger.error(f"No datasets found matching prefix: {prefix}")
        return {}

    logger.info(f"Found {len(matching)} datasets matching prefix '{prefix}'")
    logger.info(f"Fetching with {max_workers} parallel workers")

    tasks = [(name, output_dir, page_size, force, skip_quality_check) for name in sorted(matching)]

    return _parallel_fetch_with_progress(tasks, max_workers, len(matching))


def _parallel_fetch_with_progress(
    tasks: list[tuple],
    max_workers: int,
    total: int,
    progress_interval: int = 30,
) -> dict[str, Path | None]:
    """Execute parallel fetches with periodic progress reporting.

    Args:
        tasks: List of (name, output_dir, page_size, force, skip_quality_check) tuples
        max_workers: Maximum number of parallel workers
        total: Total number of tasks
        progress_interval: Seconds between progress reports (default: 30)

    Returns:
        Dict mapping dataset names to output paths (None if failed)
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, Path | None] = {}
    completed_count = 0
    failed_count = 0
    in_progress: set[str] = set()
    lock = threading.Lock()
    stop_event = threading.Event()

    def progress_reporter():
        """Report progress every N seconds."""
        while not stop_event.wait(progress_interval):
            with lock:
                pending = total - completed_count - len(in_progress)
                success = completed_count - failed_count
                logger.info(
                    f"Progress: {completed_count}/{total} complete "
                    f"({success} success, {failed_count} failed), "
                    f"{len(in_progress)} in progress, {pending} pending"
                )
                if in_progress:
                    logger.info(f"  In progress: {', '.join(sorted(in_progress))}")

    # Start progress reporter thread
    progress_thread = threading.Thread(target=progress_reporter, daemon=True)
    progress_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for task in tasks:
                name = task[0]
                with lock:
                    in_progress.add(name)
                futures[executor.submit(_fetch_dataset_worker, task)] = name

            for future in as_completed(futures):
                name, result = future.result()
                with lock:
                    in_progress.discard(name)
                    completed_count += 1
                    if result is None:
                        failed_count += 1
                results[name] = result
    finally:
        stop_event.set()
        progress_thread.join(timeout=1)

    # Final summary
    success = completed_count - failed_count
    logger.info(f"Completed: {success}/{total} successful, {failed_count} failed")

    return results


def fetch_all_datasets(
    output_dir: Path | None = None,
    page_size: int | None = None,
    force: bool = False,
    max_workers: int = 4,
    skip_quality_check: bool = False,
) -> dict[str, Path | None]:
    """Fetch all available datasets.

    Args:
        output_dir: Directory for output files (default: data/raw)
        page_size: Override page size for ArcGIS fetches
        force: If False (default), skip datasets whose files already exist
        max_workers: Maximum number of parallel downloads (default: 4)
        skip_quality_check: If True, skip quality regression checks

    Returns:
        Dict mapping dataset names to output paths (None if failed)
    """
    all_datasets = list_dataset_configs()
    logger.info(f"Fetching {len(all_datasets)} datasets with {max_workers} parallel workers")

    tasks = [
        (name, output_dir, page_size, force, skip_quality_check) for name in sorted(all_datasets)
    ]

    return _parallel_fetch_with_progress(tasks, max_workers, len(all_datasets))


def list_datasets(prefix: str | None = None) -> list[dict[str, Any]]:
    """List available datasets with their metadata.

    Args:
        prefix: Optional prefix to filter datasets (e.g., "us_boston")

    Returns:
        List of dataset info dicts with keys: name, type, description, source_type, api_key_required
    """
    datasets = list_dataset_configs()
    if prefix:
        datasets = [d for d in datasets if d.startswith(prefix)]
    datasets.sort()

    result = []
    for name in datasets:
        config = get_dataset_config(name)
        if config:
            source_type = config.source.type if config.source else "unknown"
            api_key_required = bool(config.source and config.source.api_key_env_var)
            result.append(
                {
                    "name": name,
                    "type": config.type,
                    "description": config.description,
                    "source_type": source_type,
                    "api_key_required": api_key_required,
                }
            )

    return result


def print_datasets(prefix: str | None = None) -> None:
    """Print available datasets in a formatted table.

    Args:
        prefix: Optional prefix to filter datasets
    """
    datasets = list_datasets(prefix)

    print("\nAvailable Datasets:")
    print("=" * 60)

    # Group by country prefix
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ds in datasets:
        name = ds["name"]
        parts = name.split("_", 1)
        if len(parts) > 1 and len(parts[0]) == 2:
            country = parts[0].upper()
        else:
            country = "Other"
        grouped.setdefault(country, []).append(ds)

    for country, items in sorted(grouped.items()):
        print(f"\n{country}:")
        for ds in sorted(items, key=lambda x: x["name"]):
            api_note = " (API key required)" if ds["api_key_required"] else ""
            print(f"  - {ds['name']} [{ds['source_type']}]{api_note}")
            if ds["description"]:
                print(f"    {ds['description']}")

    print("\n" + "=" * 60)
    print("\nUsage examples:")
    print("  crosswalk fetch target us_boston_streets")
    print("  crosswalk fetch target --prefix us_boston")
    print()
