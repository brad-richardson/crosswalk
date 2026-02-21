"""Dynamic MVT vector tile routes for the context layer.

Serves Mapbox Vector Tiles for target dataset roads, allowing MapLibre GL
to handle viewport-based loading natively instead of loading all features as GeoJSON.
"""

import logging
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import mercantile
from fastapi import APIRouter
from fastapi.responses import Response

from ...datasets.schema import list_dataset_configs
from ...filenames import find_target_file
from ..services import _context_cache_path, build_context_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context")

PROJECT_ROOT = Path(__file__).parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# In-memory cache of loaded context GeoDataFrames
_context_gdfs: dict[str, gpd.GeoDataFrame] = {}

EMPTY_TILE = b""


def _load_context_gdf(dataset: str) -> gpd.GeoDataFrame | None:
    """Load context cache GeoDataFrame, with in-memory caching."""
    if dataset in _context_gdfs:
        return _context_gdfs[dataset]

    cache_path = _context_cache_path(dataset)
    if not cache_path.exists():
        # Try to build from raw target file
        target_path = find_target_file(DATA_DIR, dataset)
        if target_path is None or not target_path.exists():
            return None
        try:
            target_gdf = gpd.read_parquet(target_path)
            build_context_cache(dataset, target_gdf)
        except Exception:
            logger.exception("Failed to build context cache for %s", dataset)
            return None

    try:
        gdf = gpd.read_parquet(cache_path)
        # Ensure WGS84
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        # Build spatial index eagerly
        gdf.sindex  # noqa: B018
        _context_gdfs[dataset] = gdf
        return gdf
    except Exception:
        logger.exception("Failed to load context cache for %s", dataset)
        return None


@lru_cache(maxsize=512)
def _generate_tile(dataset: str, z: int, x: int, y: int) -> bytes:
    """Generate an MVT tile for the given dataset and tile coordinates."""
    import mapbox_vector_tile

    gdf = _load_context_gdf(dataset)
    if gdf is None or gdf.empty:
        return EMPTY_TILE

    # Get tile bounds in WGS84
    tile_bounds = mercantile.bounds(x, y, z)
    west, south, east, north = (
        tile_bounds.west,
        tile_bounds.south,
        tile_bounds.east,
        tile_bounds.north,
    )

    # Spatial filter using GeoDataFrame's cx indexer
    tile_gdf = gdf.cx[west:east, south:north]

    if tile_gdf.empty:
        return EMPTY_TILE

    # Build feature list for MVT encoding
    features = []
    for _, row in tile_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        props = {}
        if "name" in tile_gdf.columns and row.get("name") is not None:
            props["name"] = str(row["name"])
        if "class" in tile_gdf.columns and row.get("class") is not None:
            props["class"] = str(row["class"])

        features.append(
            {
                "geometry": geom.__geo_interface__,
                "properties": props,
            }
        )

    if not features:
        return EMPTY_TILE

    layer = {
        "name": "context",
        "features": features,
    }

    return mapbox_vector_tile.encode(
        [layer],
        quantize_bounds=(west, south, east, north),
    )


@router.get("/tiles/{dataset}/{z}/{x}/{y}.pbf")
async def context_tile(dataset: str, z: int, x: int, y: int):
    """Serve a dynamic MVT vector tile for the context layer.

    Returns a Mapbox Vector Tile (protobuf) containing target dataset roads
    for the requested tile coordinates.
    """
    # Validate dataset against known configs to prevent path traversal
    known_datasets = list_dataset_configs()
    if dataset not in known_datasets:
        return Response(content=EMPTY_TILE, media_type="application/x-protobuf")

    tile_bytes = _generate_tile(dataset, z, x, y)

    return Response(
        content=tile_bytes,
        media_type="application/x-protobuf",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*",
        },
    )
