#!/usr/bin/env python
"""Fetch road and sidewalk data from Fort Collins, Frisco, and Salt Lake City.

Downloads road centerlines, sidewalks, and trail data from municipal GIS portals
and converts them to GeoParquet with Overture-compatible schema.

Usage:
    # Fetch all datasets
    python scripts/fetch_new_cities.py

    # Fetch specific city
    python scripts/fetch_new_cities.py --city fort_collins
    python scripts/fetch_new_cities.py --city frisco
    python scripts/fetch_new_cities.py --city salt_lake_city

    # Fetch specific dataset
    python scripts/fetch_new_cities.py --dataset fort_collins_sidewalks

Output files will be saved to data/raw/<dataset_name>.parquet
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import requests
from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_configs import (
    ALL_DATASETS,
)

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
) -> gpd.GeoDataFrame:
    """Transform Hub download data to Overture-compatible schema."""
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

    # Subclass
    data["subclass"] = ["sidewalk"] * len(gdf)  # Default for sidewalk datasets

    return gpd.GeoDataFrame(data, geometry=gdf.geometry.values, crs=gdf.crs)


def fetch_dataset(dataset_config: dict, output_dir: Path) -> Path | None:
    """Fetch a single dataset based on its configuration.

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

    try:
        # Check if this is a Hub portal dataset (no direct URL)
        if dataset_config.get("url") is None and dataset_config.get("portal_url"):
            return fetch_from_hub_portal(
                portal_url=dataset_config["portal_url"],
                output_path=output_path,
                id_prefix=dataset_config["id_prefix"],
                name_column=dataset_config.get("name_column"),
                class_column=dataset_config.get("class_column"),
                class_mapping=dataset_config.get("class_mapping"),
                source_name=dataset_config.get("source_name", "ArcGIS Hub"),
            )
        else:
            # Standard ArcGIS FeatureServer/MapServer fetch
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch road and sidewalk data from municipal GIS portals"
    )
    parser.add_argument(
        "--city",
        choices=["fort_collins", "frisco", "salt_lake_city", "all"],
        help="City to fetch data for (default: all)",
    )
    parser.add_argument(
        "--dataset",
        help="Specific dataset name to fetch (e.g., fort_collins_sidewalks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory (default: data/raw/)",
    )

    args = parser.parse_args()

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
            logger.info(f"Available datasets: {', '.join(available)}")
            sys.exit(1)

        fetch_dataset(config, args.output_dir)

    elif args.city and args.city != "all":
        # Fetch specific city
        fetch_city(args.city, args.output_dir)

    else:
        # Fetch all cities
        cities = ["fort_collins", "frisco", "salt_lake_city"]
        for city in cities:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Fetching {city.replace('_', ' ').title()}...")
            logger.info(f"{'=' * 60}")
            try:
                fetch_city(city, args.output_dir)
            except Exception as e:
                logger.error(f"Failed to fetch {city}: {e}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
