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
# Based on F_CLASS cross-reference: CLASS 3/4/5 were mismapped
STREET_CLASS_MAPPING = {
    1: "motorway",       # Interstate (not in Boston data)
    2: "primary",        # Principal arterial
    3: "primary",        # Also Principal arterial (79% are F_CLASS 3)
    4: "tertiary",       # Major Collector (65% are F_CLASS 5)
    5: "residential",    # Local/Minor Collector (83% F_CLASS 0/unknown)
    6: "service",        # Service roads (not in Boston data)
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

# Sidewalk type codes -> subclass (to match Overture schema)
SIDEWALK_SUBCLASS_MAPPING = {
    "SWALK-CL": "sidewalk",
    "CWALK-CL": "crosswalk",
    "CWALK-CL-UM": "crosswalk",  # Unmarked crosswalks
    "PWALK-CL": "sidewalk",      # Private walkway -> sidewalk
    "Sidewalk centerline": "sidewalk",
    "Crosswalk centerline": "crosswalk",
    "Privatewalk centerline": "sidewalk",
}

# Bike facility type codes -> Overture classes
# Key distinction: facilities on road surface vs physically separated
BIKE_CLASS_MAPPING = {
    # Physically separated infrastructure -> cycleway
    "SBL": "cycleway",          # Separated bike lane (raised/curbed)
    "SBLBL": "cycleway",        # Separated + bike lane
    "SBLSL": "cycleway",        # Separated + shared lane
    "CFSBL": "cycleway",        # Contraflow separated bike lane
    # On-road painted facilities -> unknown (same surface as road)
    # These may match to road segments, not cycleways
    "BL": "unknown",            # Bike lane (painted on road)
    "BL-PEAKBUS": "unknown",    # Bike lane (peak bus hours)
    "BFBL": "unknown",          # Buffered bike lane (paint only)
    "BLSL": "unknown",          # Bike lane + shared lane
    "CFBL": "unknown",          # Contraflow bike lane (painted)
    "CFBS": "unknown",          # Contraflow bike street
    # Shared use paths -> path (separate from road)
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
        "subclass_column": "TYPE",
        "subclass_mapping": SIDEWALK_SUBCLASS_MAPPING,
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
