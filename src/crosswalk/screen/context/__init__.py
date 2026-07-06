"""Context data fetchers for screen tests."""

from .overture_polygons import (
    fetch_overture_buildings,
    fetch_overture_landcover,
    fetch_overture_polygons,
    fetch_overture_water,
    get_building_union,
    get_landcover_union,
    get_polygon_union,
    get_water_union,
)

__all__ = [
    # Generic polygon functions
    "fetch_overture_polygons",
    "get_polygon_union",
    # Backwards-compatible aliases
    "fetch_overture_water",
    "get_water_union",
    "fetch_overture_buildings",
    "get_building_union",
    "fetch_overture_landcover",
    "get_landcover_union",
]
