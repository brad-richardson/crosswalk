#!/usr/bin/env python
"""Fetch Fresno County road data from Caltrans Functional Classification.

Downloads road centerlines with FHWA functional classification from Caltrans
and converts them to GeoParquet with Overture-compatible schema.

Usage:
    python scripts/fetch_fresno.py

Output files will be saved to data/raw/fresno_roads.parquet
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Caltrans Functional Classification FeatureServer
CALTRANS_URL = "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/CHhighway/CRS_Functional_Classification/FeatureServer/0/query"

# FHWA Functional Classification to Overture class mapping
# Based on match analysis showing F_System 2 maps to Overture "motorway" (95.1% agreement)
F_SYSTEM_MAPPING = {
    1: "motorway",      # Interstate
    2: "motorway",      # Principal Arterial - Freeways/Expressways (maps to motorway in Overture)
    3: "primary",       # Principal Arterial - Other
    4: "secondary",     # Minor Arterial
    5: "tertiary",      # Major Collector
    6: "tertiary",      # Minor Collector
    7: "residential",   # Local
}


def fetch_fresno_roads(output_path: Path, batch_size: int = 2000) -> gpd.GeoDataFrame:
    """Fetch Fresno County roads from Caltrans.

    Args:
        output_path: Path for output parquet file
        batch_size: Number of features per request (max 2000)

    Returns:
        GeoDataFrame with Overture-compatible schema
    """
    logger.info("Fetching Fresno County roads from Caltrans...")

    # Query parameters
    base_params = {
        "where": "County_label = 'FRESNO'",
        "outFields": "*",
        "f": "geojson",
        "outSR": "4326",
    }

    # Get total count first
    count_params = {**base_params, "returnCountOnly": "true", "f": "json"}
    resp = requests.get(CALTRANS_URL, params=count_params)
    resp.raise_for_status()
    total_count = resp.json()["count"]
    logger.info(f"Total Fresno County roads: {total_count}")

    # Fetch in batches
    all_features = []
    offset = 0

    while offset < total_count:
        logger.info(f"Fetching batch {offset // batch_size + 1} ({offset}/{total_count})...")

        params = {
            **base_params,
            "resultOffset": offset,
            "resultRecordCount": batch_size,
        }

        resp = requests.get(CALTRANS_URL, params=params)
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
    gdf["id"] = [f"fresno_roads_{i}" for i in range(len(gdf))]

    # Map functional classification to Overture class
    gdf["class"] = gdf["F_System"].map(F_SYSTEM_MAPPING).fillna("unclassified")

    # Extract road name from RouteID (e.g., "SHS_041._P" -> "SR 41")
    route_type_prefixes = {"IS": "I-", "US": "US ", "SHS": "SR "}

    def parse_route_name(route_id):
        if pd.isna(route_id):
            return None
        # Parse patterns like "SHS_041._P", "IS_005._P", "US_099._P"
        parts = str(route_id).split("_")
        if len(parts) >= 2:
            route_type = parts[0]
            route_num = parts[1].lstrip("0")
            prefix = route_type_prefixes.get(route_type)
            if prefix:
                return f"{prefix}{route_num}"
        return None

    gdf["primary_name"] = gdf["RouteID"].apply(parse_route_name)

    # Create names dict (Overture format)
    gdf["names"] = gdf["primary_name"].apply(
        lambda n: {"primary": n} if n else None
    )

    # Store original attributes in source_tags
    gdf["source_tags"] = gdf.apply(
        lambda row: {
            "F_System": row["F_System"],
            "RouteID": row["RouteID"],
            "County_label": row["County_label"],
            "Caltrans_District": row.get("Caltrans_District"),
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

    return gdf


def main():
    """Fetch Fresno County roads."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "fresno_roads.parquet"
    fetch_fresno_roads(output_path)


if __name__ == "__main__":
    main()
