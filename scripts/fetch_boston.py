#!/usr/bin/env python
"""Fetch Boston-area datasets from ArcGIS REST APIs.

Downloads municipal road, sidewalk, and bike network data from Boston's
open data portals and converts them to GeoParquet with Overture-compatible schema.

Usage:
    python scripts/fetch_boston.py

Output files will be saved to data/raw/us_boston_*.parquet
"""

from pathlib import Path

from loguru import logger

from matcher.datasets.schema import get_dataset_config
from matcher.fetch.arcgis import fetch_arcgis_layer

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Boston datasets to fetch (using new country-prefixed names)
BOSTON_DATASET_NAMES = [
    "us_boston_streets",
    "us_boston_sidewalks",
    "us_boston_bike_network",
]


def main():
    """Fetch all Boston datasets."""
    logger.info("Fetching Boston datasets...")
    logger.info("  - Streets: Managed roads with MassDOT functional classification")
    logger.info("  - Sidewalks: Centerlines including crosswalks")
    logger.info("  - Bike Network: 2024 bicycle facilities")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in BOSTON_DATASET_NAMES:
        config = get_dataset_config(dataset_name)
        if config is None:
            logger.warning(f"No config found for {dataset_name}, skipping")
            continue

        output_path = DATA_DIR / f"{dataset_name}.parquet"

        logger.info(f"\nFetching {dataset_name}...")
        if config.description:
            logger.info(f"  {config.description}")

        try:
            # Extract parameters from config
            url = config.source.url if config.source else None
            if not url:
                logger.warning(f"No URL in config for {dataset_name}, skipping")
                continue

            fetch_params = {
                "url": url,
                "output_path": output_path,
                "id_prefix": config.fetch.id_prefix if config.fetch else dataset_name,
                "name_column": config.fetch.name_column if config.fetch else None,
                "class_column": config.fetch.class_column if config.fetch else None,
                "class_mapping": config.fetch.class_mapping if config.fetch else None,
                "subclass_column": config.fetch.subclass_column if config.fetch else None,
                "subclass_mapping": config.fetch.subclass_mapping if config.fetch else None,
            }

            fetch_arcgis_layer(**fetch_params)
            logger.success(f"Saved {dataset_name} to {output_path}")
        except Exception as e:
            logger.error(f"Failed to fetch {dataset_name}: {e}")

    logger.info("\nDone fetching Boston datasets!")


if __name__ == "__main__":
    main()
