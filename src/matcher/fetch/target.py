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
from .arcgis import fetch_arcgis_layer

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
    """
    if len(gdf) == 0:
        return gdf

    # Filter to LineStrings only (drop MultiLineStrings, Points, etc.)
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

    # Find ID column
    id_col = None
    for col in ["OBJECTID", "FID", "ObjectID", "objectid", "fid", "ID", "id"]:
        if col in gdf.columns:
            id_col = col
            break

    if id_col is None:
        # Generate sequential IDs
        gdf["_generated_id"] = range(len(gdf))
        id_col = "_generated_id"

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

    # Class
    if class_column and class_column in gdf.columns:
        if class_mapping:
            data["class"] = gdf[class_column].map(class_mapping).fillna("unclassified").values
        else:
            data["class"] = gdf[class_column].fillna("unclassified").astype(str).values
    else:
        data["class"] = ["footway"] * len(gdf)  # Default for sidewalks

    # Subclass - use explicit column/mapping if provided, otherwise use default or None
    if subclass_column and subclass_column in gdf.columns:
        if subclass_mapping:
            data["subclass"] = gdf[subclass_column].map(subclass_mapping).values
        else:
            data["subclass"] = gdf[subclass_column].astype(str).values
    elif default_subclass is not None:
        data["subclass"] = [default_subclass] * len(gdf)
    else:
        data["subclass"] = [None] * len(gdf)

    return gpd.GeoDataFrame(data, geometry=gdf.geometry.values, crs=gdf.crs)


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
    )


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
) -> Path:
    """Download and process geospatial file (Shapefile, GeoPackage).

    Args:
        url: URL to download from
        output_path: Path for output GeoParquet file
        file_format: Format of downloaded file (shp, gpkg)
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for classification
        class_mapping: Dict mapping source values to standard classes
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter
        bbox_filter: Whether to apply bbox filter after loading
        api_key: Optional API key for authenticated downloads
        api_key_header: Header name for API key (e.g., "AccountKey")

    Returns:
        Path to the output GeoParquet file
    """
    logger.info(f"Downloading {file_format} from: {url}")

    headers = {}
    if api_key and api_key_header:
        headers[api_key_header] = api_key

    try:
        resp = requests.get(url, headers=headers, timeout=600, stream=True)
        resp.raise_for_status()

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Check if it's a zip file
            content_type = resp.headers.get("content-type", "")
            if "zip" in content_type or url.endswith(".zip"):
                zip_path = tmpdir_path / "download.zip"
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"Downloaded {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

                # Extract zip
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmpdir_path)

                # Find the data file
                if file_format == "shp":
                    shp_files = list(tmpdir_path.rglob("*.shp"))
                    if not shp_files:
                        raise ValueError("No .shp file found in zip")
                    data_file = shp_files[0]
                elif file_format == "gpkg":
                    gpkg_files = list(tmpdir_path.rglob("*.gpkg"))
                    if not gpkg_files:
                        raise ValueError("No .gpkg file found in zip")
                    data_file = gpkg_files[0]
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")

                logger.info(f"Loading from: {data_file}")
            else:
                # Direct file download
                data_file = tmpdir_path / f"data.{file_format}"
                with open(data_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Load with geopandas
            gdf = gpd.read_file(data_file)
            logger.info(f"Loaded {len(gdf)} features")

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
            )

            # Save to parquet
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_parquet(output_path)
            logger.success(f"Saved {len(gdf)} features to {output_path}")

            return output_path

    except Exception as e:
        logger.error(f"Failed to download: {e}")
        raise


def fetch_dataset(
    dataset_name: str,
    output_dir: Path | None = None,
    page_size: int | None = None,
) -> Path | None:
    """Fetch a single dataset based on its YAML configuration.

    Args:
        dataset_name: Name of the dataset (matches YAML filename)
        output_dir: Directory for output files (default: data/raw)
        page_size: Override page size for ArcGIS fetches (default: use arcgis module default)

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

        if url is None:
            logger.error(f"No URL in config for {dataset_name}")
            return None

        # Common parameters from config
        fetch_config = config.fetch
        id_prefix = fetch_config.id_prefix if fetch_config else dataset_name
        name_column = fetch_config.name_column if fetch_config else None
        class_column = fetch_config.class_column if fetch_config else None
        class_mapping = fetch_config.class_mapping if fetch_config else None
        subclass_column = fetch_config.subclass_column if fetch_config else None
        subclass_mapping = fetch_config.subclass_mapping if fetch_config else None
        bbox = fetch_config.bbox if fetch_config else None

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
                )

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

            return fetch_arcgis_layer(**arcgis_kwargs)

    except Exception as e:
        logger.error(f"Failed to fetch {dataset_name}: {e}")
        return None


def fetch_datasets_by_prefix(
    prefix: str,
    output_dir: Path | None = None,
    page_size: int | None = None,
) -> dict[str, Path | None]:
    """Fetch all datasets matching a prefix.

    Args:
        prefix: Prefix to match (e.g., "us_boston")
        output_dir: Directory for output files (default: data/raw)
        page_size: Override page size for ArcGIS fetches

    Returns:
        Dict mapping dataset names to output paths (None if failed)
    """
    all_datasets = list_dataset_configs()
    matching = [d for d in all_datasets if d.startswith(prefix)]

    if not matching:
        logger.error(f"No datasets found matching prefix: {prefix}")
        return {}

    logger.info(f"Found {len(matching)} datasets matching prefix '{prefix}'")
    results: dict[str, Path | None] = {}

    for name in sorted(matching):
        logger.info(f"\n{'=' * 60}")
        results[name] = fetch_dataset(name, output_dir, page_size)

    return results


def fetch_all_datasets(
    output_dir: Path | None = None,
    page_size: int | None = None,
) -> dict[str, Path | None]:
    """Fetch all available datasets.

    Args:
        output_dir: Directory for output files (default: data/raw)
        page_size: Override page size for ArcGIS fetches

    Returns:
        Dict mapping dataset names to output paths (None if failed)
    """
    all_datasets = list_dataset_configs()
    results: dict[str, Path | None] = {}

    for name in sorted(all_datasets):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Fetching {name}...")
        logger.info(f"{'=' * 60}")
        try:
            results[name] = fetch_dataset(name, output_dir, page_size)
        except Exception as e:
            logger.error(f"Failed to fetch {name}: {e}")
            results[name] = None

    return results


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
