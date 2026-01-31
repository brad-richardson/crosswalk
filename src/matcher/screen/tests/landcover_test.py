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
from ..constants import LANDCOVER_BUFFER_M
from ..context.overture_landcover import fetch_overture_landcover, get_landcover_union
from .travel_mode import get_travel_mode


@register_test
class LandcoverTest(ScreenTest):
    """Test if a match's target geometry passes through restricted landcover.

    Checks for:
    - Wetlands (marsh, swamp, bog) - roads shouldn't pass through
    - Sports fields (pitch, track, stadium) - no roads on playing surfaces

    Buffers are pre-computed in prepare() for efficiency.
    """

    name = "landcover"

    def __init__(self, buffer_overrides: dict[str, float] | None = None) -> None:
        self.buffers = {**LANDCOVER_BUFFER_M, **(buffer_overrides or {})}
        self.landcover_gdf: gpd.GeoDataFrame | None = None
        self.landcover_union: Polygon | MultiPolygon | None = None
        # Pre-computed buffered geometries by travel mode (in EPSG:4326)
        self._buffered: dict[str, Polygon | MultiPolygon] = {}

    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Fetch restricted landcover polygons and pre-compute buffered versions."""
        self.landcover_gdf = fetch_overture_landcover(bbox)
        self._buffered = {}

        if len(self.landcover_gdf) == 0:
            self.landcover_union = None
            logger.info("LandcoverTest prepared with 0 restricted areas")
            return

        self.landcover_union = get_landcover_union(self.landcover_gdf)
        if self.landcover_union is None:
            logger.info("LandcoverTest prepared with 0 valid restricted areas")
            return

        # Pre-compute buffered geometries for each travel mode
        metric_crs = str(self.landcover_gdf.estimate_utm_crs())
        landcover_series = gpd.GeoSeries([self.landcover_union], crs="EPSG:4326")
        landcover_metric = landcover_series.to_crs(metric_crs)

        for mode, buffer_m in self.buffers.items():
            buffered_metric = landcover_metric.buffer(buffer_m)
            self._buffered[mode] = buffered_metric.to_crs("EPSG:4326").iloc[0]

        logger.info(f"LandcoverTest prepared with {len(self.landcover_gdf)} restricted areas")

    def test_match(self, ctx: MatchContext) -> ScreenResult:
        """Test if the target geometry intersects buffered restricted landcover."""
        if self.landcover_union is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="No restricted landcover in area",
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
            reason=f"Road intersects restricted landcover buffer ({buffer_m}m for {mode})",
            details={
                "road_class": ctx.road_class,
                "travel_mode": mode,
                "buffer_m": buffer_m,
            },
        )
