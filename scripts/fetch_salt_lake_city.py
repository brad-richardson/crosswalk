#!/usr/bin/env python
"""Fetch Salt Lake City sidewalk data from ArcGIS Hub.

Downloads sidewalk centerlines from Salt Lake City's Open Data portal
and converts them to GeoParquet with Overture-compatible schema.

The script attempts to:
1. Discover the FeatureServer URL from the Hub API
2. Fall back to direct GeoJSON download if API discovery fails
3. Provide manual download instructions as last resort

Usage:
    python scripts/fetch_salt_lake_city.py

Output files will be saved to:
    - data/raw/salt_lake_city_sidewalks.parquet
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matcher.fetch.arcgis import fetch_arcgis_layer

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Salt Lake City sidewalk dataset
PORTAL_URL = "https://gis-slcgov.opendata.arcgis.com/datasets/sidewalk-1"
DATASET_ID = "sidewalk-1"
HUB_DOMAIN = "https://gis-slcgov.opendata.arcgis.com"


def discover_service_url() -> str | None:
    """Try to discover the FeatureServer URL from Hub API.

    Returns:
        FeatureServer URL if found, None otherwise
    """
    logger.info("Attempting to discover FeatureServer URL from Hub API...")

    # Try the Hub API v3
    api_url = f"{HUB_DOMAIN}/api/v3/datasets/{DATASET_ID}"

    try:
        resp = requests.get(api_url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data and "attributes" in data["data"]:
                attrs = data["data"]["attributes"]
                if "url" in attrs:
                    url = attrs["url"]
                    logger.info(f"Found service URL: {url}")
                    return url
    except Exception as e:
        logger.debug(f"Hub API v3 failed: {e}")

    # Try alternate API endpoints
    alt_urls = [
        f"{HUB_DOMAIN}/api/search/v1/datasets/{DATASET_ID}",
        f"{HUB_DOMAIN}/datasets/{DATASET_ID}.json",
    ]

    for url in alt_urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # Look for service URL in various locations
                for key in ["url", "serviceUrl", "service_url"]:
                    if key in data:
                        logger.info(f"Found service URL: {data[key]}")
                        return data[key]
        except Exception as e:
            logger.debug(f"Alternate API {url} failed: {e}")

    return None


def fetch_from_geojson() -> gpd.GeoDataFrame | None:
    """Try to download GeoJSON directly from Hub.

    Returns:
        GeoDataFrame if successful, None otherwise
    """
    logger.info("Attempting direct GeoJSON download...")

    # Hub provides GeoJSON at various endpoints
    geojson_urls = [
        f"{HUB_DOMAIN}/datasets/{DATASET_ID}/downloads/data.geojson",
        f"{HUB_DOMAIN}/api/download/v1/items/{DATASET_ID}/geojson",
    ]

    for url in geojson_urls:
        try:
            logger.info(f"Trying: {url}")
            resp = requests.get(url, timeout=300)  # Large file
            if resp.status_code == 200:
                data = resp.json()
                if "features" in data and len(data["features"]) > 0:
                    logger.info(f"Downloaded {len(data['features'])} features")
                    return gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
        except Exception as e:
            logger.debug(f"GeoJSON download failed for {url}: {e}")

    return None


def transform_sidewalk_data(
    gdf: gpd.GeoDataFrame,
    id_prefix: str = "slc_sidewalk",
    source_name: str = "Salt Lake City Sidewalks",
) -> gpd.GeoDataFrame:
    """Transform downloaded data to Overture-compatible schema.

    Args:
        gdf: Input GeoDataFrame
        id_prefix: Prefix for generated IDs
        source_name: Name for the data source

    Returns:
        Transformed GeoDataFrame
    """
    if len(gdf) == 0:
        return gdf

    logger.info("Transforming to Overture-compatible schema...")
    logger.info(f"Input columns: {list(gdf.columns)}")

    # Find ID column
    id_col = None
    for col in ["OBJECTID", "FID", "ObjectID", "objectid", "fid", "ID", "id"]:
        if col in gdf.columns:
            id_col = col
            break

    if id_col is None:
        gdf["_generated_id"] = range(len(gdf))
        id_col = "_generated_id"

    # Store original columns
    original_cols = [c for c in gdf.columns if c != "geometry"]
    source_tags_data = gdf[original_cols].to_dict(orient="records")

    # Look for classification column
    class_col = None
    for col in ["TYPE", "Type", "type", "CLASS", "Class", "class", "SIDEWALK_TYPE"]:
        if col in gdf.columns:
            class_col = col
            break

    # Look for name column
    name_col = None
    for col in ["NAME", "Name", "name", "STREET", "Street", "street"]:
        if col in gdf.columns:
            name_col = col
            break

    # Build result
    data = {
        "id": gdf[id_col].apply(lambda x: f"{id_prefix}_{x}").values,
        "subtype": ["road"] * len(gdf),
        "subclass": ["sidewalk"] * len(gdf),
        "sources": gdf[id_col]
        .apply(lambda x: [{"dataset": source_name, "record_id": str(x)}])
        .values,
        "road_flags": [[] for _ in range(len(gdf))],
        "level_rules": [[] for _ in range(len(gdf))],
        "source_tags": source_tags_data,
    }

    # Names
    if name_col:
        data["names"] = (
            gdf[name_col].apply(lambda x: {"primary": str(x)} if pd.notna(x) and x else None).values
        )
    else:
        data["names"] = [None] * len(gdf)

    # Class
    if class_col:
        # Log unique values for debugging
        logger.info(f"Class column '{class_col}' unique values: {gdf[class_col].unique()[:10]}")
        data["class"] = ["footway"] * len(gdf)  # All sidewalks are footway
    else:
        data["class"] = ["footway"] * len(gdf)

    return gpd.GeoDataFrame(data, geometry=gdf.geometry.values, crs=gdf.crs)


def main():
    """Fetch Salt Lake City sidewalk data."""
    logger.info("Fetching Salt Lake City sidewalk data...")
    logger.info(f"Portal URL: {PORTAL_URL}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "salt_lake_city_sidewalks.parquet"

    # Strategy 1: Discover FeatureServer URL and use standard fetch
    service_url = discover_service_url()
    if service_url:
        try:
            fetch_arcgis_layer(
                url=service_url,
                output_path=output_path,
                id_prefix="slc_sidewalk",
                name_column=None,
                class_column=None,
                class_mapping=None,
                source_name="Salt Lake City Sidewalks",
            )
            logger.success(f"Saved to {output_path}")
            return
        except Exception as e:
            logger.warning(f"FeatureServer fetch failed: {e}")

    # Strategy 2: Direct GeoJSON download
    gdf = fetch_from_geojson()
    if gdf is not None and len(gdf) > 0:
        gdf = transform_sidewalk_data(gdf)
        gdf.to_parquet(output_path)
        logger.success(f"Saved {len(gdf)} features to {output_path}")

        # Print summary
        logger.info("Class distribution:")
        for cls, count in gdf["class"].value_counts().items():
            logger.info(f"  {cls}: {count}")
        return

    # Strategy 3: Check for manually downloaded file
    manual_files = [
        DATA_DIR / "salt_lake_city_sidewalks.geojson",
        DATA_DIR / "Sidewalk.geojson",
        DATA_DIR / "sidewalk.geojson",
        DATA_DIR / "Sidewalk.shp",
        DATA_DIR / "sidewalk.shp",
    ]

    for manual_file in manual_files:
        if manual_file.exists():
            logger.info(f"Found manually downloaded file: {manual_file}")
            gdf = gpd.read_file(manual_file)
            gdf = transform_sidewalk_data(gdf)
            gdf.to_parquet(output_path)
            logger.success(f"Saved {len(gdf)} features to {output_path}")
            return

    # Strategy 4: Provide manual download instructions
    logger.warning("Automatic download failed. Please download manually:")
    logger.info(f"1. Visit: {PORTAL_URL}")
    logger.info("2. Click 'Download' and select 'Shapefile' or 'GeoJSON'")
    logger.info(f"3. Save to: {DATA_DIR}/")
    logger.info("4. Re-run this script to convert to parquet")
    logger.info("")
    logger.info("Alternatively, you can skip Salt Lake City for now and proceed with")
    logger.info("Fort Collins and Frisco datasets which downloaded successfully.")


if __name__ == "__main__":
    main()
