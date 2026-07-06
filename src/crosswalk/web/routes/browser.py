"""Dataset browser routes for the crosswalk web UI.

Provides a map-based page for browsing raw fetched features by dataset.
"""

import logging
from pathlib import Path

import geopandas as gpd
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from ...datasets.schema import get_dataset_config, list_dataset_configs
from ...filenames import find_target_file
from ..jinja import templates

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/browser")

# Project root for data directory
PROJECT_ROOT = Path(__file__).parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

MAX_FEATURES = 10_000


@router.get("")
async def browser_page(request: Request, dataset: str | None = None):
    """Dataset browser page with map."""
    datasets = sorted(list_dataset_configs())
    return templates.TemplateResponse(
        request,
        "browser/page.html",
        {
            "mode": "browser",
            "datasets": datasets,
            "dataset": dataset,
        },
    )


@router.get("/features")
async def browser_features(dataset: str = Query(...)):
    """Return GeoJSON FeatureCollection for a dataset's target parquet file."""
    # Validate dataset against known configs to prevent path traversal
    known_datasets = list_dataset_configs()
    if dataset not in known_datasets:
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown dataset: {dataset}"},
        )

    # Find the target file
    target_path = find_target_file(DATA_DIR, dataset)
    if target_path is None or not target_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"No data file found for {dataset}"},
        )

    try:
        gdf = gpd.read_parquet(target_path)
    except Exception:
        logger.exception("Failed to read %s", target_path)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to read data"},
        )

    total_count = len(gdf)

    # Select subset of properties for browser display
    keep_cols = ["id", "names", "class", "subclass", "geometry"]
    available = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[available].copy()

    # Extract primary name from names struct for display
    if "names" in gdf.columns:
        gdf["name"] = gdf["names"].apply(
            lambda x: x.get("primary") if isinstance(x, dict) else None
        )
        gdf = gdf.drop(columns=["names"])

    # Cap features for browser performance
    truncated = total_count > MAX_FEATURES
    if truncated:
        gdf = gdf.head(MAX_FEATURES)

    # Get geometry type summary
    geom_types = gdf.geometry.geom_type.value_counts().to_dict()

    # Get dataset config for metadata
    config = get_dataset_config(dataset)
    display_name = config.display_name if config else dataset

    # Convert to GeoJSON
    geojson = gdf.to_crs("EPSG:4326").__geo_interface__

    # Add metadata to the response
    geojson["metadata"] = {
        "dataset": dataset,
        "display_name": display_name,
        "total_count": total_count,
        "returned_count": len(gdf),
        "truncated": truncated,
        "geometry_types": geom_types,
    }

    return JSONResponse(content=geojson)
