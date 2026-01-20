#!/usr/bin/env python
"""Fetch road and sidewalk data from municipal GIS portals worldwide.

Downloads road centerlines, sidewalks, and trail data from various GIS portals
and converts them to GeoParquet with Overture-compatible schema.

Supports multiple fetch types:
- ArcGIS FeatureServer/MapServer (default)
- ArcGIS Hub portals
- WFS (Web Feature Service)
- GeoJSON direct download
- File downloads (Shapefile, GeoPackage)

Usage:
    # Fetch all datasets for a city
    python scripts/fetch_new_cities.py --city bogota

    # Fetch specific dataset
    python scripts/fetch_new_cities.py --dataset bogota_roads

    # List available cities and datasets
    python scripts/fetch_new_cities.py --list

Output files will be saved to data/raw/<dataset_name>.parquet
"""

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))  # scripts directory for dataset_configs
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # src directory for matcher

from dataset_configs import ALL_DATASETS

from matcher.fetch.arcgis import fetch_arcgis_layer

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def fetch_from_hub_portal(
    portal_url: str,
    output_path: Path,
    id_prefix: str,
    name_column: str | None = None,
    class_column: str | None = None,
    class_mapping: dict | None = None,
    source_name: str = "ArcGIS Hub",
    subclass_column: str | None = None,
    subclass_mapping: dict | None = None,
    default_subclass: str | None = None,
) -> Path:
    """Fetch data from ArcGIS Hub/Open Data portal.

    Downloads GeoJSON directly from the Hub API and converts to GeoParquet.

    Args:
        portal_url: URL to the Hub dataset page (e.g., .../datasets/sidewalk-1)
        output_path: Path for output GeoParquet file
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for classification
        class_mapping: Dict mapping source values to standard classes
        source_name: Name for the data source
        subclass_column: Column name for subclass (optional)
        subclass_mapping: Dict mapping source values to subclass values
        default_subclass: Default subclass if no column provided

    Returns:
        Path to the output GeoParquet file
    """
    logger.info(f"Fetching from ArcGIS Hub: {portal_url}")

    # Extract dataset ID from portal URL
    # Format: https://xxx.opendata.arcgis.com/datasets/DATASET_ID
    dataset_id = portal_url.rstrip("/").split("/")[-1]

    # Try to get the GeoJSON download URL from Hub API
    # Hub datasets typically have a GeoService endpoint we can query
    hub_domain = portal_url.split("/datasets/")[0]

    # First, try to get dataset metadata to find the service URL
    api_url = f"{hub_domain}/api/v3/datasets/{dataset_id}"

    try:
        resp = requests.get(api_url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            # Look for the service URL in the response
            if "data" in data and "attributes" in data["data"]:
                attrs = data["data"]["attributes"]
                if "url" in attrs:
                    service_url = attrs["url"]
                    logger.info(f"Found service URL: {service_url}")
                    # Use the standard ArcGIS fetch
                    return fetch_arcgis_layer(
                        url=service_url,
                        output_path=output_path,
                        id_prefix=id_prefix,
                        name_column=name_column,
                        class_column=class_column,
                        class_mapping=class_mapping,
                        subclass_column=subclass_column,
                        subclass_mapping=subclass_mapping,
                        source_name=source_name,
                    )
    except Exception as e:
        logger.warning(f"Could not get service URL from Hub API: {e}")

    # Fallback: Try direct GeoJSON download
    # Hub provides GeoJSON at: /datasets/{id}/downloads/data.geojson
    geojson_url = f"{hub_domain}/datasets/{dataset_id}/downloads/data.geojson"
    logger.info(f"Trying direct GeoJSON download: {geojson_url}")

    try:
        resp = requests.get(geojson_url, timeout=300)  # Large file, longer timeout
        resp.raise_for_status()
        geojson_data = resp.json()

        if "features" not in geojson_data:
            raise ValueError("No features in GeoJSON response")

        logger.info(f"Downloaded {len(geojson_data['features'])} features")

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"], crs="EPSG:4326")

        # Apply basic schema transformation
        gdf = _transform_hub_data(
            gdf,
            id_prefix=id_prefix,
            name_column=name_column,
            class_column=class_column,
            class_mapping=class_mapping,
            source_name=source_name,
            subclass_column=subclass_column,
            subclass_mapping=subclass_mapping,
            default_subclass=default_subclass,
        )

        # Save to parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(output_path)
        logger.success(f"Saved {len(gdf)} features to {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Failed to download from Hub: {e}")
        logger.info(
            "Please download manually from the portal and place in data/raw/. "
            f"Portal URL: {portal_url}"
        )
        raise


def _transform_hub_data(
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
    """Transform Hub download data to Overture-compatible schema.

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
    import pandas as pd

    if len(gdf) == 0:
        return gdf

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
    data = {
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


def fetch_geojson(
    url: str,
    output_path: Path,
    id_prefix: str,
    name_column: str | None = None,
    class_column: str | None = None,
    class_mapping: dict | None = None,
    source_name: str = "GeoJSON",
    bbox: tuple | None = None,
) -> Path:
    """Fetch data from a GeoJSON URL and save as GeoParquet.

    Args:
        url: URL to fetch GeoJSON from
        output_path: Path for output GeoParquet file
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for classification
        class_mapping: Dict mapping source values to standard classes
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter

    Returns:
        Path to the output GeoParquet file
    """
    logger.info(f"Fetching GeoJSON from: {url}")

    try:
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()

        # Try to parse as GeoJSON
        geojson_data = resp.json()

        if "features" not in geojson_data:
            raise ValueError("No features in GeoJSON response")

        logger.info(f"Downloaded {len(geojson_data['features'])} features")

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"], crs="EPSG:4326")

        # Apply bbox filter if provided
        if bbox and len(gdf) > 0:
            xmin, ymin, xmax, ymax = bbox
            gdf = gdf.cx[xmin:xmax, ymin:ymax]
            logger.info(f"Filtered to {len(gdf)} features within bbox")

        # Transform to Overture schema
        gdf = _transform_hub_data(
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
        logger.error(f"Failed to fetch GeoJSON: {e}")
        raise


def fetch_wfs(
    url: str,
    output_path: Path,
    typename: str,
    id_prefix: str,
    name_column: str | None = None,
    class_column: str | None = None,
    class_mapping: dict | None = None,
    source_name: str = "WFS",
    bbox: tuple | None = None,
) -> Path:
    """Fetch data from a WFS endpoint and save as GeoParquet.

    Args:
        url: WFS service URL
        output_path: Path for output GeoParquet file
        typename: WFS typename (layer) to fetch
        id_prefix: Prefix for generated IDs
        name_column: Column name for feature names
        class_column: Column name for classification
        class_mapping: Dict mapping source values to standard classes
        source_name: Name for the data source
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to filter

    Returns:
        Path to the output GeoParquet file
    """
    logger.info(f"Fetching WFS layer {typename} from: {url}")

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": typename,
        "outputFormat": "application/json",
    }

    # Add bbox filter if provided
    # bbox convention is (xmin, ymin, xmax, ymax) in lon/lat order
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        params["bbox"] = f"{xmin},{ymin},{xmax},{ymax},EPSG:4326"

    try:
        resp = requests.get(url, params=params, timeout=600)
        resp.raise_for_status()

        geojson_data = resp.json()

        if "features" not in geojson_data:
            raise ValueError("No features in WFS response")

        logger.info(f"Downloaded {len(geojson_data['features'])} features")

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"], crs="EPSG:4326")

        if len(gdf) == 0:
            logger.warning("No features returned from WFS")
            return output_path

        # Transform to Overture schema
        gdf = _transform_hub_data(
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
        logger.error(f"Failed to fetch WFS: {e}")
        raise


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
            gdf = _transform_hub_data(
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


def fetch_dataset(dataset_config: dict, output_dir: Path) -> Path | None:
    """Fetch a single dataset based on its configuration.

    Supports multiple fetch types:
    - ArcGIS FeatureServer/MapServer (default)
    - ArcGIS Hub portals (portal_url)
    - GeoJSON direct download (fetch_type: geojson)
    - WFS (fetch_type: wfs)
    - File downloads (fetch_type: download)
    - Manual download (fetch_type: manual)

    Args:
        dataset_config: Dataset configuration dict
        output_dir: Directory for output files

    Returns:
        Path to output file, or None if fetch failed
    """
    name = dataset_config["name"]
    output_path = output_dir / f"{name}.parquet"

    logger.info(f"Fetching dataset: {name}")
    if "description" in dataset_config:
        logger.info(f"  Description: {dataset_config['description']}")

    fetch_type = dataset_config.get("fetch_type", "arcgis")

    try:
        # Handle manual download datasets first
        if fetch_type == "manual":
            logger.warning(f"Dataset {name} requires manual download")
            portal_url = dataset_config.get("portal_url", "data portal")
            notes = dataset_config.get("notes", "")
            logger.info(f"  Portal: {portal_url}")
            if notes:
                logger.info(f"  Notes: {notes}")
            logger.info(
                f"  After downloading, place the file in {output_dir} and convert to parquet."
            )
            return None

        # Check if this is a Hub portal dataset (no direct URL)
        if dataset_config.get("url") is None and dataset_config.get("portal_url"):
            return fetch_from_hub_portal(
                portal_url=dataset_config["portal_url"],
                output_path=output_path,
                id_prefix=dataset_config["id_prefix"],
                name_column=dataset_config.get("name_column"),
                class_column=dataset_config.get("class_column"),
                class_mapping=dataset_config.get("class_mapping"),
                subclass_column=dataset_config.get("subclass_column"),
                subclass_mapping=dataset_config.get("subclass_mapping"),
                default_subclass=dataset_config.get("default_subclass"),
                source_name=dataset_config.get("source_name", "ArcGIS Hub"),
            )

        elif fetch_type == "geojson":
            return fetch_geojson(
                url=dataset_config["url"],
                output_path=output_path,
                id_prefix=dataset_config["id_prefix"],
                name_column=dataset_config.get("name_column"),
                class_column=dataset_config.get("class_column"),
                class_mapping=dataset_config.get("class_mapping"),
                source_name=dataset_config.get("source_name", "GeoJSON"),
                bbox=dataset_config.get("bbox"),
            )

        elif fetch_type == "wfs":
            return fetch_wfs(
                url=dataset_config["url"],
                output_path=output_path,
                typename=dataset_config["wfs_typename"],
                id_prefix=dataset_config["id_prefix"],
                name_column=dataset_config.get("name_column"),
                class_column=dataset_config.get("class_column"),
                class_mapping=dataset_config.get("class_mapping"),
                source_name=dataset_config.get("source_name", "WFS"),
                bbox=dataset_config.get("bbox"),
            )

        elif fetch_type == "download":
            # Check for API key requirement
            api_key = None
            api_key_header = dataset_config.get("api_key_header")
            if dataset_config.get("api_key_required"):
                # Use custom env var name if provided, otherwise generate from dataset name
                env_var = (
                    dataset_config.get("api_key_env_var")
                    or f"{name.upper().replace('_', '')}_API_KEY"
                )
                api_key = os.environ.get(env_var)
                if not api_key:
                    logger.warning(
                        f"API key required but not found. Set {env_var} environment variable."
                    )
                    if "singapore" in name.lower():
                        logger.info("For Singapore LTA: Register at https://datamall.lta.gov.sg")

            return fetch_download(
                url=dataset_config["url"],
                output_path=output_path,
                file_format=dataset_config["file_format"],
                id_prefix=dataset_config["id_prefix"],
                name_column=dataset_config.get("name_column"),
                class_column=dataset_config.get("class_column"),
                class_mapping=dataset_config.get("class_mapping"),
                source_name=dataset_config.get("source_name", "Download"),
                bbox=dataset_config.get("bbox"),
                bbox_filter=dataset_config.get("bbox_filter", False),
                api_key=api_key,
                api_key_header=api_key_header,
            )

        else:
            # Default: ArcGIS FeatureServer/MapServer fetch
            return fetch_arcgis_layer(
                url=dataset_config["url"],
                output_path=output_path,
                id_prefix=dataset_config["id_prefix"],
                name_column=dataset_config.get("name_column"),
                class_column=dataset_config.get("class_column"),
                class_mapping=dataset_config.get("class_mapping"),
                subclass_column=dataset_config.get("subclass_column"),
                subclass_mapping=dataset_config.get("subclass_mapping"),
                source_name=dataset_config.get("source_name", "ArcGIS"),
            )

    except Exception as e:
        logger.error(f"Failed to fetch {name}: {e}")
        return None


def fetch_city(city_name: str, output_dir: Path) -> list[Path]:
    """Fetch all datasets for a city.

    Args:
        city_name: City identifier (fort_collins, frisco, salt_lake_city)
        output_dir: Directory for output files

    Returns:
        List of paths to output files
    """
    if city_name not in ALL_DATASETS:
        available = ", ".join(ALL_DATASETS.keys())
        raise ValueError(f"Unknown city: {city_name}. Available: {available}")

    datasets = ALL_DATASETS[city_name]
    results = []

    for dataset in datasets:
        result = fetch_dataset(dataset, output_dir)
        if result:
            results.append(result)

    return results


def list_datasets():
    """List all available cities and datasets."""
    print("\nAvailable Cities and Datasets:")
    print("=" * 60)

    for city, datasets in ALL_DATASETS.items():
        print(f"\n{city.replace('_', ' ').title()}:")
        for ds in datasets:
            # Use explicit fetch_type first, only infer "hub" if not set
            explicit_fetch_type = ds.get("fetch_type")
            if explicit_fetch_type:
                fetch_type = explicit_fetch_type
            elif ds.get("portal_url"):
                fetch_type = "hub"
            else:
                fetch_type = "arcgis"
            api_note = " (API key required)" if ds.get("api_key_required") else ""
            print(f"  - {ds['name']} [{fetch_type}]{api_note}")
            if ds.get("description"):
                print(f"    {ds['description']}")

    print("\n" + "=" * 60)
    print("\nUsage examples:")
    print("  python scripts/fetch_new_cities.py --city bogota")
    print("  python scripts/fetch_new_cities.py --dataset bogota_roads")
    print("  python scripts/fetch_new_cities.py --city cape_town")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch road and sidewalk data from municipal GIS portals worldwide"
    )
    parser.add_argument(
        "--city",
        choices=list(ALL_DATASETS.keys()) + ["all"],
        help="City to fetch data for",
    )
    parser.add_argument(
        "--dataset",
        help="Specific dataset name to fetch (e.g., bogota_roads)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory (default: data/raw/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available cities and datasets",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        # Fetch specific dataset
        all_configs = []
        for datasets in ALL_DATASETS.values():
            all_configs.extend(datasets)

        config = next((d for d in all_configs if d["name"] == args.dataset), None)
        if config is None:
            available = [d["name"] for d in all_configs]
            logger.error(f"Unknown dataset: {args.dataset}")
            logger.info(f"Available datasets: {', '.join(sorted(available))}")
            sys.exit(1)

        fetch_dataset(config, args.output_dir)

    elif args.city and args.city != "all":
        # Fetch specific city
        fetch_city(args.city, args.output_dir)

    elif args.city == "all":
        # Fetch all cities
        for city in ALL_DATASETS:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Fetching {city.replace('_', ' ').title()}...")
            logger.info(f"{'=' * 60}")
            try:
                fetch_city(city, args.output_dir)
            except Exception as e:
                logger.error(f"Failed to fetch {city}: {e}")

    else:
        # No arguments - show help
        parser.print_help()
        print("\nTo see available datasets, use: --list")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
