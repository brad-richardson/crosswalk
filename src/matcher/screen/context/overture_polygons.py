"""Fetch Overture polygon context data for screen tests.

Provides a unified interface for fetching polygon data from various Overture themes
(buildings, water, land_use) for road screening purposes.
"""

import geopandas as gpd
from loguru import logger
from overturemaps.core import geodataframe, get_latest_release
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def fetch_overture_polygons(
    theme: str,
    bbox: tuple[float, float, float, float],
    release: str | None = None,
    min_area_m2: float = 0.0,
    subtypes: set[str] | None = None,
) -> gpd.GeoDataFrame:
    """Fetch polygon data from Overture Maps for a given theme.

    Args:
        theme: Overture theme name ("building", "water", "land_use")
        bbox: Bounding box as (xmin, ymin, xmax, ymax) in EPSG:4326
        release: Overture release version (None = latest)
        min_area_m2: Minimum polygon area in square meters to include
        subtypes: For land_use theme, filter to these subtypes only

    Returns:
        GeoDataFrame with polygon geometries in EPSG:4326
    """
    if release is None:
        release = get_latest_release()
        logger.debug(f"Using latest Overture release: {release}")

    logger.info(f"Fetching Overture {theme} for bbox: {bbox}")

    gdf = geodataframe(theme, bbox=bbox, release=release)

    if len(gdf) == 0:
        logger.info(f"No {theme} found in bbox")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Filter to requested subtypes (for land_use theme)
    if subtypes is not None and "subtype" in gdf.columns:
        mask = gdf["subtype"].str.lower().isin({s.lower() for s in subtypes})
        gdf = gdf.loc[mask].copy()
        if len(gdf) == 0:
            logger.info(f"No {theme} matching subtypes: {subtypes}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Filter to polygon geometries only
    mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf.loc[mask].copy()

    if len(gdf) == 0:
        logger.info(f"No polygon {theme} found")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Filter by minimum area
    if min_area_m2 > 0:
        gdf_metric = gdf.to_crs(gdf.estimate_utm_crs())
        gdf["area_m2"] = gdf_metric.geometry.area
        initial_count = len(gdf)
        gdf = gdf[gdf["area_m2"] >= min_area_m2]
        filtered_count = initial_count - len(gdf)
        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} small {theme} (< {min_area_m2} m2)")

    logger.info(f"Fetched {len(gdf)} {theme} polygons")
    return gdf


def get_polygon_union(gdf: gpd.GeoDataFrame) -> Polygon | MultiPolygon | None:
    """Get union of all polygon geometries for efficient intersection testing.

    Args:
        gdf: GeoDataFrame with polygon geometries

    Returns:
        Union of all valid geometries, or None if empty
    """
    if len(gdf) == 0:
        return None

    valid_geoms = [
        g.buffer(0) if not g.is_valid else g
        for g in gdf.geometry
        if g is not None and not g.is_empty
    ]
    if not valid_geoms:
        return None

    result = unary_union(valid_geoms)
    return None if result.is_empty else result


# Convenience functions with sensible defaults
from ..constants import (
    MIN_BUILDING_AREA_M2,
    MIN_LANDCOVER_AREA_M2,
    MIN_WATER_AREA_M2,
    RESTRICTED_LANDCOVER_SUBTYPES,
)


def fetch_overture_buildings(
    bbox: tuple[float, float, float, float],
    release: str | None = None,
    min_area_m2: float = MIN_BUILDING_AREA_M2,
) -> gpd.GeoDataFrame:
    """Fetch building footprint polygons from Overture Maps."""
    return fetch_overture_polygons("building", bbox, release, min_area_m2)


def fetch_overture_water(
    bbox: tuple[float, float, float, float],
    release: str | None = None,
    min_area_m2: float = MIN_WATER_AREA_M2,
) -> gpd.GeoDataFrame:
    """Fetch water body polygons from Overture Maps."""
    return fetch_overture_polygons("water", bbox, release, min_area_m2)


def fetch_overture_landcover(
    bbox: tuple[float, float, float, float],
    release: str | None = None,
    subtypes: set[str] | None = None,
    min_area_m2: float = MIN_LANDCOVER_AREA_M2,
) -> gpd.GeoDataFrame:
    """Fetch landcover polygons from Overture Maps."""
    if subtypes is None:
        subtypes = RESTRICTED_LANDCOVER_SUBTYPES
    return fetch_overture_polygons("land_use", bbox, release, min_area_m2, subtypes)


# Alias for backwards compatibility
get_building_union = get_polygon_union
get_water_union = get_polygon_union
get_landcover_union = get_polygon_union
