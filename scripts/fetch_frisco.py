#!/usr/bin/env python
"""Fetch Frisco, Texas trail and road data.

Downloads trail network and road centerlines from Frisco GIS
and converts them to GeoParquet with Overture-compatible schema.

Usage:
    python scripts/fetch_frisco.py

Output files will be saved to:
    - data/raw/frisco_trails.parquet
    - data/raw/frisco_roads.parquet
"""

import sys
from pathlib import Path

from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_configs import FRISCO_DATASETS

from matcher.fetch.arcgis import fetch_arcgis_layer

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def main():
    """Fetch all Frisco datasets."""
    logger.info("Fetching Frisco, TX datasets...")
    logger.info("  - Trails: Pedestrian paths, bike routes, walking loops")
    logger.info("  - Roads: Centerlines with 9 classifications and lifecycle status")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in FRISCO_DATASETS:
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

    logger.info("\nDone fetching Frisco datasets!")


if __name__ == "__main__":
    main()
