#!/usr/bin/env python
"""Fetch Fort Collins sidewalk and street data.

Downloads sidewalk inventory and street centerlines from Fort Collins GIS
and converts them to GeoParquet with Overture-compatible schema.

Usage:
    python scripts/fetch_fort_collins.py

Output files will be saved to:
    - data/raw/fort_collins_sidewalks.parquet
    - data/raw/fort_collins_streets.parquet
"""

import sys
from pathlib import Path

from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))  # scripts directory for dataset_configs
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # src directory for matcher

from dataset_configs import FORT_COLLINS_DATASETS

from matcher.fetch.arcgis import fetch_arcgis_layer

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def main():
    """Fetch all Fort Collins datasets."""
    logger.info("Fetching Fort Collins datasets...")
    logger.info("  - Sidewalks: 540 miles of inventory with attachment type")
    logger.info("  - Streets: Centerlines with 10 road type classifications")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in FORT_COLLINS_DATASETS:
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

    logger.info("\nDone fetching Fort Collins datasets!")


if __name__ == "__main__":
    main()
