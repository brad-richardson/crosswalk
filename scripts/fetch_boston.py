#!/usr/bin/env python
"""Fetch Boston-area datasets from ArcGIS REST APIs.

Downloads municipal road, sidewalk, and bike network data from Boston's
open data portals and converts them to GeoParquet with Overture-compatible schema.

Usage:
    python scripts/fetch_boston.py

Output files will be saved to data/raw/boston_*.parquet
"""

import sys
from pathlib import Path

from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))  # scripts directory for dataset_configs
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # src directory for matcher

from dataset_configs import BOSTON_DATASETS

from matcher.fetch.arcgis import fetch_arcgis_layer

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def main():
    """Fetch all Boston datasets."""
    logger.info("Fetching Boston datasets...")
    logger.info("  - Streets: Managed roads with MassDOT functional classification")
    logger.info("  - Sidewalks: Centerlines including crosswalks")
    logger.info("  - Bike Network: 2024 bicycle facilities")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in BOSTON_DATASETS:
        name = dataset.pop("name")
        description = dataset.pop("description", None)
        output_path = DATA_DIR / f"{name}.parquet"

        logger.info(f"\nFetching {name}...")
        if description:
            logger.info(f"  {description}")

        try:
            fetch_arcgis_layer(output_path=output_path, **dataset)
            logger.success(f"Saved {name} to {output_path}")
        except Exception as e:
            logger.error(f"Failed to fetch {name}: {e}")

        # Restore popped keys for next iteration
        dataset["name"] = name
        if description:
            dataset["description"] = description

    logger.info("\nDone fetching Boston datasets!")


if __name__ == "__main__":
    main()
