"""Landcover screen test.

Detects matches where the target road passes through restricted landcover
types like wetlands or sports fields.
"""

import geopandas as gpd
from loguru import logger
from shapely.geometry import MultiPolygon, Polygon

from ..base import (
    MatchContext,
    ScreenOutcome,
    ScreenResult,
    ScreenTest,
    register_test,
)
from ..context.overture_landcover import fetch_overture_landcover, get_landcover_union
from .travel_mode import get_travel_mode

# Buffer distances by travel mode (meters)
# Wetlands need buffers like water; sports fields less so
LANDCOVER_BUFFER_M = {
    "vehicle": 3.0,
    "bike": 1.5,
    "pedestrian": 0.5,
}


@register_test
class LandcoverTest(ScreenTest):
    """Test if a match's target geometry passes through restricted landcover.

    Checks for:
    - Wetlands (marsh, swamp, bog) - roads shouldn't pass through
    - Sports fields (pitch, track, stadium) - no roads on playing surfaces

    Buffers based on road type like other tests.
    """

    name = "landcover"

    def __init__(self, buffer_overrides: dict[str, float] | None = None) -> None:
        """Initialize the landcover test.

        Args:
            buffer_overrides: Optional override buffer distances by mode
        """
        self.buffers = {**LANDCOVER_BUFFER_M, **(buffer_overrides or {})}
        self.landcover_gdf: gpd.GeoDataFrame | None = None
        self.landcover_union: Polygon | MultiPolygon | None = None
        self._metric_crs: str | None = None

    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Fetch restricted landcover polygons for the bounding box."""
        self.landcover_gdf = fetch_overture_landcover(bbox)

        if len(self.landcover_gdf) > 0:
            self.landcover_union = get_landcover_union(self.landcover_gdf)
            self._metric_crs = str(self.landcover_gdf.estimate_utm_crs())
        else:
            self.landcover_union = None
            self._metric_crs = None

        logger.info(f"LandcoverTest prepared with {len(self.landcover_gdf)} restricted areas")

    def test_match(self, ctx: MatchContext) -> ScreenResult:
        """Test if the target geometry intersects buffered restricted landcover."""
        if self.landcover_union is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="No restricted landcover in area",
            )

        if self._metric_crs is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="Could not determine CRS for buffering",
            )

        # Get buffer distance based on road class
        mode = get_travel_mode(ctx.road_class)
        buffer_m = self.buffers[mode]

        # Buffer landcover in metric CRS
        landcover_series = gpd.GeoSeries([self.landcover_union], crs="EPSG:4326")
        landcover_metric = landcover_series.to_crs(self._metric_crs)
        buffered_metric = landcover_metric.buffer(buffer_m)
        buffered = buffered_metric.to_crs("EPSG:4326").iloc[0]

        # Check intersection
        target_geom = ctx.target_geom
        if not target_geom.intersects(buffered):
            return ScreenResult(outcome=ScreenOutcome.PASS, test_name=self.name)

        intersection = target_geom.intersection(buffered)
        if intersection.is_empty:
            return ScreenResult(outcome=ScreenOutcome.PASS, test_name=self.name)

        return ScreenResult(
            outcome=ScreenOutcome.FAIL,
            test_name=self.name,
            reason=f"Road intersects restricted landcover buffer ({buffer_m}m for {mode})",
            details={
                "road_class": ctx.road_class,
                "travel_mode": mode,
                "buffer_m": buffer_m,
            },
        )
