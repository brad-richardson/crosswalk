"""Fetch Overture building footprint polygons for screen tests.

Uses Overture Maps buildings theme to get building footprints that roads
should not pass through.
"""

import geopandas as gpd
from loguru import logger
from overturemaps.core import geodataframe, get_latest_release
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def fetch_overture_buildings(
    bbox: tuple[float, float, float, float],
    release: str | None = None,
    min_area_m2: float = 20.0,
) -> gpd.GeoDataFrame:
    """Fetch building footprint polygons from Overture Maps.

    Args:
        bbox: Bounding box as (xmin, ymin, xmax, ymax) in EPSG:4326
        release: Overture release version (None = latest)
        min_area_m2: Minimum building area in square meters to include

    Returns:
        GeoDataFrame with building footprint polygons in EPSG:4326
    """
    if release is None:
        release = get_latest_release()
        logger.debug(f"Using latest Overture release: {release}")

    logger.info(f"Fetching Overture buildings for bbox: {bbox}")

    gdf = geodataframe("building", bbox=bbox, release=release)

    if len(gdf) == 0:
        logger.info("No buildings found in bbox")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Filter to polygon geometries only
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    if len(gdf) == 0:
        logger.info("No polygon buildings found")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Filter by minimum area
    if min_area_m2 > 0:
        gdf_metric = gdf.to_crs(gdf.estimate_utm_crs())
        gdf["area_m2"] = gdf_metric.geometry.area
        initial_count = len(gdf)
        gdf = gdf[gdf["area_m2"] >= min_area_m2]
        filtered_count = initial_count - len(gdf)
        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} small buildings (< {min_area_m2} m2)")

    logger.info(f"Fetched {len(gdf)} building footprints")
    return gdf


def get_building_union(gdf: gpd.GeoDataFrame) -> Polygon | MultiPolygon | None:
    """Get union of all building footprint geometries for efficient intersection testing."""
    if len(gdf) == 0:
        return None

    valid_geoms = [g.buffer(0) if not g.is_valid else g for g in gdf.geometry if g is not None]
    if not valid_geoms:
        return None

    return unary_union(valid_geoms)
