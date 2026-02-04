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
    from matcher.fetch.target import fetch_dataset, list_datasets

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

from ..datasets.schema import get_dataset_config, list_dataset_configs
from ..filenames import target_filename
from ..utils.linear_ref import create_trivial_lr
from .arcgis import fetch_arcgis_layer
from .normalize import normalize_oneway_value, normalize_speed_to_kph

# Default output directory
DEFAULT_DATA_DIR = Path("data/raw")


def _transform_download_data(
    gdf: gpd.GeoDataFrame,
    id_prefix: str,
    name_column: str | None,
    class_column: str | None,
    class_mapping: dict | None,
    source_name: str,
    subclass_column: str | None = None,
    subclass_mapping: dict | None = None,
    default_subclass: str | None = None,
    exclude: dict[str, list[str]] | None = None,
    oneway_column: str | None = None,
    speed_limit_column: str | None = None,
    speed_limit_unit: str = "kph",
    id_column: str | None = None,
) -> gpd.GeoDataFrame:
    """Transform downloaded data to Overture-compatible schema.

    Args:
        gdf: Input GeoDataFrame
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for classification
        class_mapping: Dict mapping source values to standard classes
        source_name: Name for the data source
        subclass_column: Column name for subclass (optional)
        subclass_mapping: Dict mapping source values to subclass values
        default_subclass: Default subclass if no column provided
        id_column: Column to use as stable ID (auto-detected if None)
    """
    if len(gdf) == 0:
        return gdf

    # Apply exclude filter if configured
    if exclude:
        for column, values in exclude.items():
            if column in gdf.columns:
                before_count = len(gdf)
                gdf = gdf[~gdf[column].isin(values)]
                excluded = before_count - len(gdf)
                if excluded > 0:
                    logger.info(f"Excluded {excluded} features where {column} in {values}")
        if len(gdf) == 0:
            return gdf

    # Convert single-part MultiLineStrings to LineStrings
    multi_mask = gdf.geometry.geom_type == "MultiLineString"
    if multi_mask.any():

        def to_linestring(geom):
            if geom.geom_type == "MultiLineString" and len(geom.geoms) == 1:
                return geom.geoms[0]
            return geom

        gdf.geometry = gdf.geometry.apply(to_linestring)
        n_converted = (gdf.geometry.geom_type == "LineString").sum() - (~multi_mask).sum()
        logger.info(f"Converted {n_converted} single-part MultiLineStrings to LineStrings")

    # Filter to LineStrings only (drop remaining MultiLineStrings, Points, etc.)
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

    if id_column not in gdf.columns:
        raise ValueError(
            f"Configured id_column '{id_column}' not found in data for {source_name}. "
            f"Available columns: {list(gdf.columns)}"
        )

    id_col = id_column

    # Store original columns
    original_cols = [c for c in gdf.columns if c != "geometry"]
    source_tags_data = gdf[original_cols].to_dict(orient="records")

    # Build result
    data: dict[str, Any] = {
        "id": gdf[id_col].apply(lambda x: f"{id_prefix}_{x}").values,
        "subtype": ["road"] * len(gdf),
        "sources": gdf[id_col]
        .apply(lambda x: [{"dataset": source_name, "record_id": str(x)}])
        .values,
        "road_flags": [[] for _ in range(len(gdf))],
        "level_rules": [[] for _ in range(len(gdf))],
        "source_tags": source_tags_data,
    }

    # Names
    if name_column and name_column in gdf.columns:
        data["names"] = (
            gdf[name_column]
            .apply(lambda x: {"primary": str(x)} if pd.notna(x) and x else None)
            .values
        )
    else:
        data["names"] = [None] * len(gdf)

    # Class (case-insensitive mapping)
    if class_column and class_column in gdf.columns:
        if class_mapping:
            # Normalize mapping keys to lowercase for case-insensitive matching
            lower_mapping = {str(k).lower(): v for k, v in class_mapping.items()}
            data["class"] = (
                gdf[class_column]
                .fillna("")
                .astype(str)
                .str.lower()
                .map(lower_mapping)
                .fillna("unknown")
                .values
            )
        else:
            data["class"] = gdf[class_column].fillna("unknown").astype(str).values
    else:
        data["class"] = ["footway"] * len(gdf)  # Default for sidewalks

    # Subclass - use explicit column/mapping if provided, otherwise use default or None
    # (case-insensitive mapping)
    if subclass_column and subclass_column in gdf.columns:
        if subclass_mapping:
            # Normalize mapping keys to lowercase for case-insensitive matching
            lower_subclass_mapping = {str(k).lower(): v for k, v in subclass_mapping.items()}
            data["subclass"] = (
                gdf[subclass_column]
                .fillna("")
                .astype(str)
                .str.lower()
                .map(lower_subclass_mapping)
                .values
            )
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
    id_prefix: str,
    name_column: str | None = None,
    class_column: str | None = None,
    class_mapping: dict | None = None,
    source_name: str = "OGC API Features",
    bbox: tuple | None = None,
    limit_per_page: int = 5000,
    exclude: dict[str, list[str]] | None = None,
    id_column: str | None = None,
) -> Path:
    """Fetch geospatial data from an OGC API Features endpoint.

    OGC API Features is a modern REST API standard for geospatial data that
    returns GeoJSON. Supports pagination and bbox filtering.

    Args:
        url: OGC API Features items endpoint URL
        output_path: Path for output GeoParquet file
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for classification
        class_mapping: Dict mapping source values to standard classes
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter
        limit_per_page: Number of features per API request

    Returns:
        Path to the output GeoParquet file
    """
    logger.info(f"Fetching OGC API Features: {url}")

    all_features = []
    offset = 0

    # Build base params
    params: dict[str, Any] = {"f": "json", "limit": limit_per_page}
    if bbox:
        params["bbox"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    while True:
        params["offset"] = offset

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

        offset += len(features)

    if not all_features:
        logger.warning(f"No features returned from {url}")
        return output_path

    logger.info(f"Fetched {len(all_features)} total features")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")

    # Transform to Overture schema
    gdf = _transform_download_data(
        gdf,
        id_prefix=id_prefix,
        name_column=name_column,
        class_column=class_column,
        class_mapping=class_mapping,
        source_name=source_name,
        exclude=exclude,
        id_column=id_column,
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
    id_prefix: str,
    name_column: str | None = None,
    class_column: str | None = None,
    class_mapping: dict | None = None,
    source_name: str = "WFS",
    bbox: tuple | None = None,
    max_features: int = 300000,
    exclude: dict[str, list[str]] | None = None,
    id_column: str | None = None,
) -> Path:
    """Fetch geospatial data from a WFS (Web Feature Service) endpoint.

    Args:
        url: WFS service base URL
        type_name: Layer/type name to fetch (e.g., 'geoportal:segmento_logradouro')
        output_path: Path for output GeoParquet file
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for classification
        class_mapping: Dict mapping source values to standard classes
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter
        max_features: Maximum features to fetch (pagination limit)

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
        id_prefix=id_prefix,
        name_column=name_column,
        class_column=class_column,
        class_mapping=class_mapping,
        source_name=source_name,
        exclude=exclude,
        id_column=id_column,
    )

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, write_covering_bbox=True)

    logger.success(f"Saved {len(gdf)} features to {output_path}")
    return output_path


def fetch_lta_geospatial(
    layer_id: str,
    output_path: Path,
    id_prefix: str,
    api_key: str,
    name_column: str | None = None,
    class_column: str | None = None,
    class_mapping: dict | None = None,
    source_name: str = "LTA DataMall",
    bbox: tuple | None = None,
    bbox_filter: bool = False,
    id_column: str | None = None,
) -> Path:
    """Fetch geospatial data from Singapore LTA DataMall API.

    The LTA API returns a presigned S3 URL for downloading shapefiles.
    API endpoint: https://datamall2.mytransport.sg/ltaodataservice/GeospatialWholeIsland

    Args:
        layer_id: Geospatial layer ID (e.g., 'RoadSectionLine', 'Footpath')
        output_path: Path for output GeoParquet file
        id_prefix: Prefix for generated IDs
        api_key: LTA DataMall API key
        name_column: Column name for feature names
        class_column: Column name for classification
        class_mapping: Dict mapping source values to standard classes
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter
        bbox_filter: Whether to apply bbox filter after loading

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
        id_prefix=id_prefix,
        name_column=name_column,
        class_column=class_column,
        class_mapping=class_mapping,
        source_name=source_name,
        bbox=bbox,
        bbox_filter=bbox_filter,
        id_column=id_column,
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


def fetch_download(
    url: str,
    output_path: Path,
    file_format: str,
    id_prefix: str,
    name_column: str | None = None,
    class_column: str | None = None,
    class_mapping: dict | None = None,
    source_name: str = "Download",
    bbox: tuple | None = None,
    bbox_filter: bool = False,
    api_key: str | None = None,
    api_key_header: str | None = None,
    encoding: str | None = None,
    source_crs: str | None = None,
    cache_download: bool = False,
    cache_ttl_hours: int = 168,
    exclude: dict[str, list[str]] | None = None,
    oneway_column: str | None = None,
    speed_limit_column: str | None = None,
    speed_limit_unit: str = "kph",
    id_column: str | None = None,
) -> Path:
    """Download and process geospatial file (Shapefile, GeoPackage, GML).

    Args:
        url: URL to download from
        output_path: Path for output GeoParquet file
        file_format: Format of downloaded file (shp, gpkg, gml, geojson)
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for classification
        class_mapping: Dict mapping source values to standard classes
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter
        bbox_filter: Whether to apply bbox filter after loading
        api_key: Optional API key for authenticated downloads
        api_key_header: Header name for API key (e.g., "AccountKey")
        encoding: File encoding for non-UTF8 files (e.g., "EUC-KR")
        source_crs: Source CRS if different from data file (e.g., "EPSG:5179")
        cache_download: If True, cache the downloaded file for future use
        cache_ttl_hours: Hours before cache expires (default: 168 = 7 days)
        oneway_column: Column name for one-way direction
        speed_limit_column: Column name for speed limit
        speed_limit_unit: Unit of speed limit values ("kph" or "mph")

    Returns:
        Path to the output GeoParquet file
    """
    import hashlib

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

                # Find the data file
                if file_format == "shp":
                    found_files = list(extract_path.rglob("*.shp"))
                    if not found_files:
                        raise ValueError("No .shp file found in zip")
                    data_file = found_files[0]
                elif file_format == "gpkg":
                    found_files = list(extract_path.rglob("*.gpkg"))
                    if not found_files:
                        raise ValueError("No .gpkg file found in zip")
                    data_file = found_files[0]
                elif file_format == "gml":
                    found_files = list(extract_path.rglob("*.gml"))
                    if not found_files:
                        raise ValueError("No .gml file found in zip")
                    data_file = found_files[0]
                elif file_format == "geojson":
                    found_files = list(extract_path.rglob("*.geojson")) + list(
                        extract_path.rglob("*.json")
                    )
                    if not found_files:
                        raise ValueError("No .geojson file found in zip")
                    data_file = found_files[0]
                elif file_format == "gdb":
                    found_dirs = [d for d in extract_path.rglob("*.gdb") if d.is_dir()]
                    if not found_dirs:
                        raise ValueError("No .gdb folder found in zip")
                    data_file = found_dirs[0]
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")

                logger.info(f"Loading from: {data_file}")
            finally:
                # Clean up extract directory after we're done (handled after geopandas load)
                pass

        logger.info(f"Loading from: {data_file}")

        # Load with geopandas
        read_kwargs: dict[str, Any] = {}
        if encoding:
            read_kwargs["encoding"] = encoding
            logger.info(f"Using encoding: {encoding}")
        if file_format == "gml":
            read_kwargs["driver"] = "GML"

        # For multi-layer formats (gdb, gpkg), select the road centerline layer
        if file_format == "gdb":
            import pyogrio

            layers = pyogrio.list_layers(data_file)
            layer_names = [layer_info[0] for layer_info in layers]
            logger.debug(f"Available layers: {layer_names}")

            # Prefer centerline/road layers
            for preferred in [
                "CENTERLINE",
                "centerline",
                "road_link",
                "RoadLink",
                "roads",
                "road",
            ]:
                if preferred in layer_names:
                    read_kwargs["layer"] = preferred
                    logger.info(f"Selected layer: {preferred}")
                    break

        gdf = gpd.read_file(data_file, **read_kwargs)
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

        # Transform to Overture schema
        gdf = _transform_download_data(
            gdf,
            id_prefix=id_prefix,
            name_column=name_column,
            class_column=class_column,
            class_mapping=class_mapping,
            source_name=source_name,
            exclude=exclude,
            oneway_column=oneway_column,
            speed_limit_column=speed_limit_column,
            speed_limit_unit=speed_limit_unit,
            id_column=id_column,
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
    id_prefix: str,
    name_column: str | None = None,
    class_column: str | None = None,
    class_mapping: dict | None = None,
    source_name: str = "OS Data Hub",
    bbox: tuple | None = None,
    api_key: str | None = None,
    cache_ttl_hours: int = 168,
    exclude: dict[str, list[str]] | None = None,
    id_column: str | None = None,
) -> Path:
    """Fetch geospatial data from Ordnance Survey Data Hub Downloads API.

    Uses the osdatahub package for access to OS OpenData products like
    OpenRoads, OpenMap Local, etc. No API key required for open data.
    Downloads are cached for 7 days by default.

    Args:
        product_id: OS Data Hub product ID (e.g., "OpenRoads", "OpenMapLocal")
        output_path: Path for output GeoParquet file
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for classification
        class_mapping: Dict mapping source values to standard classes
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter after download
        api_key: Not used for OpenData (kept for API compatibility)
        cache_ttl_hours: Cache TTL in hours (default: 168 = 7 days)

    Returns:
        Path to the output GeoParquet file
    """
    try:
        from osdatahub import OpenDataDownload
    except ImportError as err:
        raise ImportError(
            "osdatahub package required for OS Data Hub downloads. "
            'Install with: pip install "matcher[os]"'
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
        id_prefix=id_prefix,
        name_column=name_column,
        class_column=class_column,
        class_mapping=class_mapping,
        source_name=source_name,
        exclude=exclude,
        id_column=id_column,
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
) -> Path | None:
    """Fetch a single dataset based on its YAML configuration.

    Args:
        dataset_name: Name of the dataset (matches YAML filename)
        output_dir: Directory for output files (default: data/raw)
        page_size: Override page size for ArcGIS fetches (default: use arcgis module default)
        force: If False (default), skip if output file already exists

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

        # Common parameters from config
        fetch_config = config.fetch
        id_prefix = fetch_config.id_prefix if fetch_config else dataset_name
        id_column = fetch_config.id_column if fetch_config else None
        name_column = fetch_config.name_column if fetch_config else None
        class_column = fetch_config.class_column if fetch_config else None
        class_mapping = fetch_config.class_mapping if fetch_config else None
        subclass_column = fetch_config.subclass_column if fetch_config else None
        subclass_mapping = fetch_config.subclass_mapping if fetch_config else None
        bbox = fetch_config.bbox if fetch_config else None
        encoding = fetch_config.encoding if fetch_config else None
        source_crs = fetch_config.source_crs if fetch_config else None
        exclude = fetch_config.exclude if fetch_config else None
        oneway_column = fetch_config.oneway_column if fetch_config else None
        speed_limit_column = fetch_config.speed_limit_column if fetch_config else None
        speed_limit_unit = fetch_config.speed_limit_unit if fetch_config else "kph"

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

            return fetch_os_downloads(
                product_id=product_id,
                output_path=output_path,
                id_prefix=id_prefix,
                name_column=name_column,
                class_column=class_column,
                class_mapping=class_mapping,
                source_name=config.display_name or dataset_name,
                bbox=bbox,
                api_key=api_key,
                exclude=exclude,
                id_column=id_column,
            )

        if url is None:
            logger.error(f"No URL in config for {dataset_name}")
            return None

        if source_type == "download":
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
                return fetch_lta_geospatial(
                    layer_id=layer_id,
                    output_path=output_path,
                    id_prefix=id_prefix,
                    api_key=api_key,
                    name_column=name_column,
                    class_column=class_column,
                    class_mapping=class_mapping,
                    source_name=config.display_name or dataset_name,
                    bbox=bbox,
                    bbox_filter=bool(bbox),
                    id_column=id_column,
                )

            # Check if caching is enabled for this download
            cache_download = config.source.cache_download if config.source else False
            cache_ttl_hours = config.source.cache_ttl_hours if config.source else 168

            return fetch_download(
                url=url,
                output_path=output_path,
                file_format=file_format,
                id_prefix=id_prefix,
                name_column=name_column,
                class_column=class_column,
                class_mapping=class_mapping,
                source_name=config.display_name or dataset_name,
                bbox=bbox,
                bbox_filter=bool(bbox),
                api_key=api_key,
                api_key_header=api_key_header,
                encoding=encoding,
                source_crs=source_crs,
                cache_download=cache_download,
                cache_ttl_hours=cache_ttl_hours,
                exclude=exclude,
                oneway_column=oneway_column,
                speed_limit_column=speed_limit_column,
                speed_limit_unit=speed_limit_unit,
                id_column=id_column,
            )

        elif source_type == "ogc_features":
            # OGC API Features (modern REST API for geospatial data)
            return fetch_ogc_features(
                url=url,
                output_path=output_path,
                id_prefix=id_prefix,
                name_column=name_column,
                class_column=class_column,
                class_mapping=class_mapping,
                source_name=config.display_name or dataset_name,
                bbox=bbox,
                exclude=exclude,
                id_column=id_column,
            )

        elif source_type == "wfs":
            # WFS (Web Feature Service)
            # URL should be base WFS URL, type_name from where_clause field
            type_name = config.source.where_clause if config.source else None
            if not type_name:
                logger.error("WFS source requires type_name in where_clause field")
                return None
            return fetch_wfs(
                url=url,
                type_name=type_name,
                output_path=output_path,
                id_prefix=id_prefix,
                name_column=name_column,
                class_column=class_column,
                class_mapping=class_mapping,
                source_name=config.display_name or dataset_name,
                bbox=bbox,
                exclude=exclude,
                id_column=id_column,
            )

        else:
            # Default: ArcGIS FeatureServer/MapServer fetch
            # Build kwargs, only pass page_size if specified
            arcgis_kwargs: dict[str, Any] = {
                "url": url,
                "output_path": output_path,
                "id_prefix": id_prefix,
                "name_column": name_column,
                "class_column": class_column,
                "class_mapping": class_mapping,
                "subclass_column": subclass_column,
                "subclass_mapping": subclass_mapping,
                "source_name": config.display_name or dataset_name,
            }
            if page_size is not None:
                arcgis_kwargs["page_size"] = page_size
            # Pass bbox for server-side filtering (reduces bandwidth for large datasets)
            if bbox:
                arcgis_kwargs["bbox"] = bbox
            # Pass exclude filter
            if exclude:
                arcgis_kwargs["exclude"] = exclude
            # Pass oneway/speed columns
            if oneway_column:
                arcgis_kwargs["oneway_column"] = oneway_column
            if speed_limit_column:
                arcgis_kwargs["speed_limit_column"] = speed_limit_column
                arcgis_kwargs["speed_limit_unit"] = speed_limit_unit
            # Pass id_column for stable IDs
            if id_column:
                arcgis_kwargs["id_column"] = id_column

            return fetch_arcgis_layer(**arcgis_kwargs)

    except Exception as e:
        logger.error(f"Failed to fetch {dataset_name}: {e}")
        return None


def _fetch_dataset_worker(
    args: tuple[str, Path | None, int | None, bool],
) -> tuple[str, Path | None]:
    """Worker function for parallel dataset fetching."""
    name, output_dir, page_size, force = args
    try:
        result = fetch_dataset(name, output_dir, page_size, force)
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
) -> dict[str, Path | None]:
    """Fetch all datasets matching a prefix.

    Args:
        prefix: Prefix to match (e.g., "us_boston")
        output_dir: Directory for output files (default: data/raw)
        page_size: Override page size for ArcGIS fetches
        force: If False (default), skip datasets whose files already exist
        max_workers: Maximum number of parallel downloads (default: 4)

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

    tasks = [(name, output_dir, page_size, force) for name in sorted(matching)]

    return _parallel_fetch_with_progress(tasks, max_workers, len(matching))


def _parallel_fetch_with_progress(
    tasks: list[tuple],
    max_workers: int,
    total: int,
    progress_interval: int = 30,
) -> dict[str, Path | None]:
    """Execute parallel fetches with periodic progress reporting.

    Args:
        tasks: List of (name, output_dir, page_size, force) tuples
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
) -> dict[str, Path | None]:
    """Fetch all available datasets.

    Args:
        output_dir: Directory for output files (default: data/raw)
        page_size: Override page size for ArcGIS fetches
        force: If False (default), skip datasets whose files already exist
        max_workers: Maximum number of parallel downloads (default: 4)

    Returns:
        Dict mapping dataset names to output paths (None if failed)
    """
    all_datasets = list_dataset_configs()
    logger.info(f"Fetching {len(all_datasets)} datasets with {max_workers} parallel workers")

    tasks = [(name, output_dir, page_size, force) for name in sorted(all_datasets)]

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
    print("  matcher fetch target us_boston_streets")
    print("  matcher fetch target --prefix us_boston")
    print()
