#!/usr/bin/env python
"""Fetch Boston-area datasets from ArcGIS REST APIs.

Downloads municipal road, sidewalk, and bike network data from Boston's
open data portals and converts them to GeoParquet with Overture-compatible schema.

Usage:
    python scripts/fetch_boston.py

Output files will be saved to data/raw/boston_*.parquet
"""

from pathlib import Path

from loguru import logger

from matcher.fetch import fetch_arcgis_layer

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Class mappings for standardization

# MassDOT functional classification codes
STREET_CLASS_MAPPING = {
    1: "motorway",       # Interstate
    2: "primary",        # Principal arterial
    3: "secondary",      # Minor arterial
    4: "residential",    # Collector/local streets
    5: "tertiary",       # Other
    6: "service",        # Service roads
}

# Sidewalk type codes -> Overture "footway" class
SIDEWALK_CLASS_MAPPING = {
    "SWALK-CL": "footway",
    "CWALK-CL": "footway",       # Crosswalks are pedestrian paths
    "CWALK-CL-UM": "footway",    # Unmarked crosswalks
    "PWALK-CL": "footway",       # Private walkway
    "Sidewalk centerline": "footway",
    "Crosswalk centerline": "footway",
    "Privatewalk centerline": "footway",
}

# Bike facility type codes -> Overture classes
BIKE_CLASS_MAPPING = {
    # Dedicated bike infrastructure -> cycleway
    "BL": "cycleway",           # Bike lane
    "BL-PEAKBUS": "cycleway",   # Bike lane (peak bus hours)
    "BFBL": "cycleway",         # Buffered bike lane
    "BLSL": "cycleway",         # Bike lane + shared lane (has dedicated component)
    "CFBL": "cycleway",         # Contraflow bike lane
    "SBL": "cycleway",          # Separated bike lane
    "SBLBL": "cycleway",        # Separated + bike lane
    "SBLSL": "cycleway",        # Separated + shared lane (has dedicated component)
    "CFSBL": "cycleway",        # Contraflow separated bike lane
    "CFBS": "cycleway",         # Contraflow bike street
    # Shared use paths -> path
    "SUP": "path",              # Shared use path
    "SUPN": "path",             # Natural surface shared use path
    "SUPM": "path",             # Minor shared use path
    # Shared lane markings (not dedicated) -> unknown
    "SLM": "unknown",           # Shared lane markings (sharrows)
    "SLMTC": "unknown",         # Shared lane traffic calmed
    # Other infrastructure types
    "PED": "pedestrian",        # Pedestrianized street
    "WALK": "footway",          # Walkway
}

# Dataset configurations
BOSTON_DATASETS = [
    {
        "name": "boston_streets",
        "url": "https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/City_of_Boston_Managed_Streets/FeatureServer/0",
        "id_prefix": "boston_streets",
        "name_column": "STREETNAME",
        "class_column": "CLASS",
        "class_mapping": STREET_CLASS_MAPPING,
        "level_column": None,
        "source_name": "Boston Managed Streets",
    },
    {
        "name": "boston_sidewalks",
        "url": "https://gisportal.boston.gov/arcgis/rest/services/Infrastructure/OpenData/MapServer/5",
        "id_prefix": "boston_sidewalk",
        "name_column": None,  # Sidewalks unnamed
        "class_column": "TYPE",
        "class_mapping": SIDEWALK_CLASS_MAPPING,
        "level_column": None,
        "source_name": "Boston Sidewalk Centerlines",
    },
    {
        "name": "boston_bike_network",
        "url": "https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/Boston_Bicycle_Network_2024/FeatureServer/0",
        "id_prefix": "boston_bike",
        "name_column": "STREET_NAM",
        "class_column": "ExisFacil",
        "class_mapping": BIKE_CLASS_MAPPING,
        "level_column": None,
        "source_name": "Boston Bicycle Network 2024",
    },
]


def main():
    """Fetch all Boston datasets."""
    logger.info("Fetching Boston datasets...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in BOSTON_DATASETS:
        name = dataset.pop("name")
        output_path = DATA_DIR / f"{name}.parquet"

        logger.info(f"Fetching {name}...")
        try:
            fetch_arcgis_layer(output_path=output_path, **dataset)
            logger.success(f"Saved {name} to {output_path}")
        except Exception as e:
            logger.error(f"Failed to fetch {name}: {e}")
            # Re-add name for next iteration
            dataset["name"] = name
            continue

        # Re-add name for next iteration
        dataset["name"] = name

    logger.info("Done fetching Boston datasets!")


if __name__ == "__main__":
    main()
