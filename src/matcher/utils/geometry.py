"""Geometry utility functions."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from loguru import logger
from shapely.ops import transform


def geometry_length_meters(geom) -> float:
    """Return the length of a WGS84 (lon/lat) geometry in meters.

    Projects the geometry to its local UTM zone (chosen from the centroid) for an
    accurate metric length. Returns 0.0 for empty/None geometries and falls back
    to a rough degrees-to-meters approximation if projection fails.

    Args:
        geom: A shapely geometry in EPSG:4326 (lon/lat degrees).
    """
    if geom is None or getattr(geom, "is_empty", True):
        return 0.0
    try:
        centroid = geom.centroid
        lon, lat = centroid.x, centroid.y
        # Clamp: lon=180 would otherwise compute zone 61 (valid zones are 1-60).
        utm_zone = min(60, max(1, int((lon + 180) / 6) + 1))
        # WGS84 / UTM EPSG codes: 326xx northern hemisphere, 327xx southern.
        epsg = (32600 if lat >= 0 else 32700) + utm_zone
        transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        return transform(transformer.transform, geom).length
    except Exception:
        # Fallback: 1 degree ~= 111 km at the equator.
        return geom.length * 111000.0


def filter_to_linestrings(
    gdf: gpd.GeoDataFrame,
    source_name: str = "dataset",
) -> gpd.GeoDataFrame:
    """Filter GeoDataFrame to only LineString geometries, dropping MultiLineStrings.

    MultiLineString geometries are filtered out at ingest time as they typically
    represent bad data or edge cases that complicate downstream processing.
    A warning is logged when MultiLineStrings are dropped.

    Args:
        gdf: Input GeoDataFrame with geometry column
        source_name: Name of the data source for logging (e.g., "reference", "target")

    Returns:
        GeoDataFrame with only LineString geometries
    """
    if gdf.empty:
        return gdf

    original_count = len(gdf)

    # Count null/empty geometries explicitly
    null_mask = gdf.geometry.isna() | gdf.geometry.is_empty
    null_count = null_mask.sum()

    # Count different geometry types (excluding nulls)
    geom_types = gdf.geometry.geom_type
    multilinestring_count = (geom_types == "MultiLineString").sum()
    linestring_count = (geom_types == "LineString").sum()
    other_count = original_count - multilinestring_count - linestring_count - null_count

    # Filter to only LineString geometries (this also excludes nulls)
    mask = geom_types == "LineString"
    filtered = gdf[mask].copy()

    # Log warnings for filtered geometries
    if null_count > 0:
        logger.warning(
            f"Filtered {null_count} null/empty geometries from {source_name} "
            f"({null_count}/{original_count} features)."
        )

    if multilinestring_count > 0:
        logger.warning(
            f"Filtered {multilinestring_count} MultiLineString geometries from {source_name} "
            f"({multilinestring_count}/{original_count} features). "
            f"MultiLineStrings are not supported and likely represent bad data."
        )

    if other_count > 0:
        logger.warning(
            f"Filtered {other_count} non-LineString geometries from {source_name} "
            f"({other_count}/{original_count} features). "
            f"Only LineString geometries are supported."
        )

    if len(filtered) == 0 and original_count > 0:
        logger.error(
            f"All {original_count} geometries were filtered from {source_name}. "
            f"No LineString geometries found."
        )

    return filtered


def _remove_tiny_holes(polygon: shapely.Polygon, min_hole_area: float = 1e-10) -> shapely.Polygon:
    """Remove interior holes smaller than a threshold from a polygon.

    Tiny holes create junk branches in centerline extraction.

    Args:
        polygon: Input polygon (may have holes)
        min_hole_area: Minimum hole area to keep (in CRS units squared)

    Returns:
        Polygon with tiny holes removed
    """
    if not polygon.interiors:
        return polygon

    kept = [ring for ring in polygon.interiors if shapely.Polygon(ring).area >= min_hole_area]
    return shapely.Polygon(polygon.exterior, kept)


def convert_polygons_to_centerlines(
    gdf: gpd.GeoDataFrame,
    source_name: str = "dataset",
) -> gpd.GeoDataFrame:
    """Convert polygon geometries to centerline LineStrings, preserving existing LineStrings.

    Uses pygeoops.centerline() (medial axis via Voronoi diagrams) to extract
    centerlines from polygon features, typically road/path features served as
    polygons rather than LineStrings.

    Args:
        gdf: Input GeoDataFrame with mixed geometry types
        source_name: Name of the data source for logging

    Returns:
        GeoDataFrame with polygons converted to LineStrings.
        Existing LineStrings are passed through unchanged.
        Failed conversions are dropped with a warning.
    """
    if gdf.empty:
        return gdf

    try:
        import pygeoops
    except ImportError as err:
        raise ImportError(
            "pygeoops package required for polygon-to-centerline conversion. "
            'Install with: pip install "matcher[ml]"'
        ) from err

    geom_types = gdf.geometry.geom_type

    # Separate LineStrings, MultiLineStrings, and Polygons/MultiPolygons
    line_mask = geom_types == "LineString"
    multiline_mask = geom_types == "MultiLineString"
    poly_mask = geom_types.isin(["Polygon", "MultiPolygon"])

    lines_gdf = gdf[line_mask].copy()
    polys_gdf = gdf[poly_mask].copy()

    # Explode MultiLineStrings into individual LineStrings (unsupported downstream)
    multilines_gdf = gdf[multiline_mask]
    if not multilines_gdf.empty:
        logger.warning(
            f"Exploding {len(multilines_gdf)} MultiLineString geometries in {source_name}"
        )
        multilines_gdf = multilines_gdf.explode(index_parts=False)
        lines_gdf = gpd.GeoDataFrame(
            pd.concat([lines_gdf, multilines_gdf], ignore_index=True), crs=gdf.crs
        )

    # Anything else (Point, etc.) is dropped
    other_count = len(gdf) - line_mask.sum() - multiline_mask.sum() - poly_mask.sum()
    if other_count > 0:
        logger.warning(f"Dropping {other_count} non-line/polygon geometries from {source_name}")

    if polys_gdf.empty:
        logger.info(f"No polygon geometries found in {source_name}, nothing to convert")
        return lines_gdf

    logger.info(
        f"Converting {len(polys_gdf)} polygon(s) to centerlines in {source_name} "
        f"(passing through {len(lines_gdf)} existing line geometries)"
    )

    # Explode MultiPolygons into individual Polygons
    polys_gdf = polys_gdf.explode(index_parts=False)

    # Preprocess: make_valid + remove tiny holes
    polys_gdf.geometry = shapely.make_valid(polys_gdf.geometry)
    polys_gdf.geometry = polys_gdf.geometry.apply(
        lambda g: _remove_tiny_holes(g) if g.geom_type == "Polygon" else g
    )

    # Filter out any geometries that became non-polygons after make_valid
    valid_poly_mask = polys_gdf.geometry.geom_type == "Polygon"
    if not valid_poly_mask.all():
        n_invalid = (~valid_poly_mask).sum()
        logger.warning(f"{n_invalid} geometries became non-polygon after make_valid, dropping")
        polys_gdf = polys_gdf[valid_poly_mask].copy()

    if polys_gdf.empty:
        return lines_gdf

    # Extract centerlines using pygeoops
    centerlines = pygeoops.centerline(np.array(polys_gdf.geometry))

    # Replace geometry column with centerlines
    polys_gdf = polys_gdf.copy()
    polys_gdf.geometry = centerlines

    # Drop rows where centerline extraction failed (None results)
    null_mask = polys_gdf.geometry.isna()
    n_failed = null_mask.sum()
    if n_failed > 0:
        logger.warning(
            f"Centerline extraction failed for {n_failed}/{len(polys_gdf)} polygons "
            f"in {source_name}, dropping these features"
        )
        polys_gdf = polys_gdf[~null_mask].copy()

    if polys_gdf.empty:
        return lines_gdf

    # Explode any MultiLineString results to individual LineStrings
    polys_gdf = polys_gdf.explode(index_parts=False)

    # Recombine with passthrough LineStrings
    result = gpd.GeoDataFrame(
        pd.concat([lines_gdf, polys_gdf], ignore_index=True),
        crs=gdf.crs,
    )

    logger.info(
        f"Centerline conversion complete: {len(result)} features "
        f"({len(lines_gdf)} passthrough + {len(polys_gdf)} converted)"
    )

    return result
