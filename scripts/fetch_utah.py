#!/usr/bin/env python
"""Fetch Utah Salt Lake County road data from UGRC.

Downloads road centerlines with CARTOCODE classification from Utah's SGID
and converts them to GeoParquet with Overture-compatible schema.

Usage:
    python scripts/fetch_utah.py

Output files will be saved to data/raw/utah_roads.parquet
"""

from pathlib import Path

import geopandas as gpd
import requests
from loguru import logger

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Utah SGID Roads FeatureServer
UTAH_ROADS_URL = "https://services1.arcgis.com/99lidPhWCzftIe9K/ArcGIS/rest/services/UtahRoads/FeatureServer/0/query"

# Salt Lake County FIPS code
SALT_LAKE_COUNTY = "49035"

# CARTOCODE to Overture class mapping
# Based on Utah's cartographic codes
CARTOCODE_MAPPING = {
    "1": "motorway",  # Interstate
    "2": "trunk",  # US Highway
    "3": "trunk",  # US Highway
    "4": "primary",  # Parkway
    "5": "primary",  # State Route (major)
    "6": "secondary",  # State Route (minor)
    "7": "motorway_link",  # Ramps
    "8": "secondary",  # Major street
    "9": "tertiary",  # Minor road
    "10": "tertiary",  # Secondary street
    "11": "residential",  # Local street
    "12": "track",  # Unpaved/dirt road
    "14": "service",  # Private road
    "15": "service",  # Private road
    "16": "path",  # Trail/path
    "17": "unclassified",  # Other
}


def fetch_utah_roads(output_path: Path, batch_size: int = 2000) -> gpd.GeoDataFrame:
    """Fetch Salt Lake County roads from Utah SGID.

    Args:
        output_path: Path for output parquet file
        batch_size: Number of features per request (max 2000)

    Returns:
        GeoDataFrame with Overture-compatible schema
    """
    logger.info("Fetching Salt Lake County roads from Utah SGID...")

    # Get total count first
    count_params = {
        "where": f"COUNTY_L = '{SALT_LAKE_COUNTY}'",
        "returnCountOnly": "true",
        "f": "json",
    }
    resp = requests.get(UTAH_ROADS_URL, params=count_params)
    resp.raise_for_status()
    total_count = resp.json()["count"]
    logger.info(f"Total Salt Lake County roads: {total_count}")

    # Fetch in batches
    all_features = []
    offset = 0

    while offset < total_count:
        logger.info(f"Fetching batch {offset // batch_size + 1} ({offset}/{total_count})...")

        params = {
            "where": f"COUNTY_L = '{SALT_LAKE_COUNTY}'",
            "outFields": "FULLNAME,CARTOCODE,DOT_FCLASS,DOT_CLASS,SPEED_LMT,DOT_AADT,ONEWAY",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": batch_size,
        }

        resp = requests.get(UTAH_ROADS_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        if "features" in data:
            all_features.extend(data["features"])
            logger.info(f"  Retrieved {len(data['features'])} features")

        offset += batch_size

    logger.info(f"Total features retrieved: {len(all_features)}")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")

    # Transform to Overture-compatible schema
    logger.info("Transforming to Overture-compatible schema...")

    # Generate unique IDs
    gdf["id"] = [f"utah_roads_{i}" for i in range(len(gdf))]

    # Map CARTOCODE to Overture class
    gdf["class"] = gdf["CARTOCODE"].map(CARTOCODE_MAPPING).fillna("unclassified")

    # Create names dict (Overture format)
    gdf["names"] = gdf["FULLNAME"].apply(lambda n: {"primary": n} if n and n.strip() else None)

    # Store original attributes in source_tags
    gdf["source_tags"] = gdf.apply(
        lambda row: {
            "CARTOCODE": row["CARTOCODE"],
            "DOT_FCLASS": row.get("DOT_FCLASS"),
            "DOT_CLASS": row.get("DOT_CLASS"),
            "SPEED_LMT": row.get("SPEED_LMT"),
            "DOT_AADT": row.get("DOT_AADT"),
            "ONEWAY": row.get("ONEWAY"),
        },
        axis=1,
    )

    # Select final columns
    final_columns = ["id", "geometry", "names", "class", "source_tags"]
    gdf = gdf[final_columns]

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)
    logger.success(f"Saved {len(gdf)} roads to {output_path}")

    # Print class distribution
    logger.info("Class distribution:")
    for cls, count in gdf["class"].value_counts().items():
        logger.info(f"  {cls}: {count}")

    # Count roads with names
    has_names = gdf["names"].notna().sum()
    logger.info(f"Roads with names: {has_names} ({has_names / len(gdf) * 100:.1f}%)")

    return gdf


def main():
    """Fetch Salt Lake County roads."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "utah_roads.parquet"
    fetch_utah_roads(output_path)


if __name__ == "__main__":
    main()
