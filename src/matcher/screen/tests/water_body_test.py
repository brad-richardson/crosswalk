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
from ..constants import WATER_BUFFER_M
from ..context.overture_water import fetch_overture_water, get_water_union
from .travel_mode import get_travel_mode


@register_test
class WaterBodyTest(ScreenTest):
    """Test if a match's target geometry intersects water bodies.

    Buffers water bodies based on road type - vehicle roads need more
    clearance than pedestrian paths. Buffers are pre-computed in prepare()
    for efficiency.
    """

    name = "water_body"

    def __init__(self, buffer_overrides: dict[str, float] | None = None) -> None:
        self.buffers = {**WATER_BUFFER_M, **(buffer_overrides or {})}
        self.water_gdf: gpd.GeoDataFrame | None = None
        self.water_union: Polygon | MultiPolygon | None = None
        # Pre-computed buffered geometries by travel mode (in EPSG:4326)
        self._buffered: dict[str, Polygon | MultiPolygon] = {}

    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Fetch water body polygons and pre-compute buffered versions."""
        self.water_gdf = fetch_overture_water(bbox)
        self._buffered = {}

        if len(self.water_gdf) == 0:
            self.water_union = None
            logger.info("WaterBodyTest prepared with 0 water bodies")
            return

        self.water_union = get_water_union(self.water_gdf)
        if self.water_union is None:
            logger.info("WaterBodyTest prepared with 0 valid water bodies")
            return

        # Pre-compute buffered geometries for each travel mode
        metric_crs = str(self.water_gdf.estimate_utm_crs())
        water_series = gpd.GeoSeries([self.water_union], crs="EPSG:4326")
        water_metric = water_series.to_crs(metric_crs)

        for mode, buffer_m in self.buffers.items():
            buffered_metric = water_metric.buffer(buffer_m)
            self._buffered[mode] = buffered_metric.to_crs("EPSG:4326").iloc[0]

        logger.info(f"WaterBodyTest prepared with {len(self.water_gdf)} water bodies")

    def test_match(self, ctx: MatchContext) -> ScreenResult:
        """Test if the target geometry intersects buffered water bodies."""
        if self.water_union is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="No water bodies in area",
            )

        mode = get_travel_mode(ctx.road_class)
        buffered = self._buffered.get(mode)
        if buffered is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason=f"No buffer computed for mode: {mode}",
            )

        buffer_m = self.buffers.get(mode, self.buffers["vehicle"])

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
