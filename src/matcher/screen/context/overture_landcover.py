"""Fetch Overture landcover polygons for screen tests.

Uses Overture Maps land_use theme to get wetlands, sports fields, and other
landcover types that roads should not pass through.
"""

import geopandas as gpd
from loguru import logger
from overturemaps.core import geodataframe, get_latest_release
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# Landcover subtypes that roads should never cross
RESTRICTED_SUBTYPES = {
    # Wetlands - similar to water bodies
    "wetland",
    "marsh",
    "swamp",
    "bog",
    # Sports surfaces - no roads through playing fields
    "pitch",
    "sports_centre",
    "stadium",
    "track",
    "golf_course",  # fairways/greens, not cart paths
}


def fetch_overture_landcover(
    bbox: tuple[float, float, float, float],
    release: str | None = None,
    subtypes: set[str] | None = None,
    min_area_m2: float = 50.0,
) -> gpd.GeoDataFrame:
    """Fetch landcover polygons from Overture Maps.

    Args:
        bbox: Bounding box as (xmin, ymin, xmax, ymax) in EPSG:4326
        release: Overture release version (None = latest)
        subtypes: Landcover subtypes to include (None = RESTRICTED_SUBTYPES)
        min_area_m2: Minimum area in square meters to include

    Returns:
        GeoDataFrame with landcover polygons in EPSG:4326
    """
    if release is None:
        release = get_latest_release()
        logger.debug(f"Using latest Overture release: {release}")

    if subtypes is None:
        subtypes = RESTRICTED_SUBTYPES

    logger.info(f"Fetching Overture landcover for bbox: {bbox}")

    gdf = geodataframe("land_use", bbox=bbox, release=release)

    if len(gdf) == 0:
        logger.info("No landcover found in bbox")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Filter to requested subtypes
    if "subtype" in gdf.columns:
        mask = gdf["subtype"].str.lower().isin({s.lower() for s in subtypes})
        gdf = gdf.loc[mask].copy()

    if len(gdf) == 0:
        logger.info(f"No landcover matching subtypes: {subtypes}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Filter to polygon geometries only
    mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf.loc[mask].copy()

    if len(gdf) == 0:
        logger.info("No polygon landcover found")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Filter by minimum area
    if min_area_m2 > 0:
        gdf_metric = gdf.to_crs(gdf.estimate_utm_crs())
        gdf["area_m2"] = gdf_metric.geometry.area
        initial_count = len(gdf)
        gdf = gdf[gdf["area_m2"] >= min_area_m2]
        filtered_count = initial_count - len(gdf)
        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} small landcover areas (< {min_area_m2} m2)")

    logger.info(f"Fetched {len(gdf)} landcover polygons")
    return gdf


def get_landcover_union(gdf: gpd.GeoDataFrame) -> Polygon | MultiPolygon | None:
    """Get union of all landcover geometries for efficient intersection testing."""
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
