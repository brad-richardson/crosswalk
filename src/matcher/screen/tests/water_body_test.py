"""Water body screen test.

Detects matches where the target road intersects water bodies (lakes, rivers, etc.)
beyond what could reasonably be a bridge.
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
from ..context.overture_water import fetch_overture_water, get_water_union
from .travel_mode import get_travel_mode

# Buffer distances by travel mode (meters)
# Water needs larger buffers - roads shouldn't be right at water's edge
WATER_BUFFER_M = {
    "vehicle": 5.0,
    "bike": 2.0,
    "pedestrian": 1.0,
}


@register_test
class WaterBodyTest(ScreenTest):
    """Test if a match's target geometry intersects water bodies.

    Buffers water bodies based on road type - vehicle roads need more
    clearance than pedestrian paths.
    """

    name = "water_body"

    def __init__(self, buffer_overrides: dict[str, float] | None = None) -> None:
        """Initialize the water body test.

        Args:
            buffer_overrides: Optional override buffer distances by mode
        """
        self.buffers = {**WATER_BUFFER_M, **(buffer_overrides or {})}
        self.water_gdf: gpd.GeoDataFrame | None = None
        self.water_union: Polygon | MultiPolygon | None = None
        self._metric_crs: str | None = None

    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Fetch water body polygons for the bounding box."""
        self.water_gdf = fetch_overture_water(bbox)

        if len(self.water_gdf) > 0:
            self.water_union = get_water_union(self.water_gdf)
            self._metric_crs = str(self.water_gdf.estimate_utm_crs())
        else:
            self.water_union = None
            self._metric_crs = None

        logger.info(f"WaterBodyTest prepared with {len(self.water_gdf)} water bodies")

    def test_match(self, ctx: MatchContext) -> ScreenResult:
        """Test if the target geometry intersects buffered water bodies."""
        if self.water_union is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="No water bodies in area",
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

        # Buffer water in metric CRS
        water_series = gpd.GeoSeries([self.water_union], crs="EPSG:4326")
        water_metric = water_series.to_crs(self._metric_crs)
        buffered_metric = water_metric.buffer(buffer_m)
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
            reason=f"Road intersects water body buffer ({buffer_m}m for {mode})",
            details={
                "road_class": ctx.road_class,
                "travel_mode": mode,
                "buffer_m": buffer_m,
            },
        )
