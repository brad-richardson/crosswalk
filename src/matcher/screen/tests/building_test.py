"""Building footprint screen test.

Detects target segments that pass through or too close to building footprints.
"""

import geopandas as gpd
from loguru import logger
from shapely.geometry import MultiPolygon, Polygon

from ..base import (
    CandidateContext,
    ScreenOutcome,
    ScreenResult,
    ScreenTest,
    register_test,
)
from ..constants import BUILDING_BUFFER_M
from ..context.overture_buildings import fetch_overture_buildings, get_building_union
from .travel_mode import get_travel_mode


@register_test
class BuildingTest(ScreenTest):
    """Test if a candidate segment passes through buildings.

    Buffers building footprints based on road type - vehicle roads need
    more clearance than pedestrian paths. Buffers are pre-computed in
    prepare() for efficiency.
    """

    name = "building"

    def __init__(self, buffer_overrides: dict[str, float] | None = None) -> None:
        self.buffers = {**BUILDING_BUFFER_M, **(buffer_overrides or {})}
        self.building_gdf: gpd.GeoDataFrame | None = None
        self.building_union: Polygon | MultiPolygon | None = None
        # Pre-computed buffered geometries by travel mode (in EPSG:4326)
        self._buffered: dict[str, Polygon | MultiPolygon] = {}

    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Fetch building footprints and pre-compute buffered versions."""
        self.building_gdf = fetch_overture_buildings(bbox)
        self._buffered = {}

        if len(self.building_gdf) == 0:
            self.building_union = None
            logger.info("BuildingTest prepared with 0 buildings")
            return

        self.building_union = get_building_union(self.building_gdf)
        if self.building_union is None:
            logger.info("BuildingTest prepared with 0 valid buildings")
            return

        # Pre-compute buffered geometries for each travel mode
        metric_crs = str(self.building_gdf.estimate_utm_crs())
        building_series = gpd.GeoSeries([self.building_union], crs="EPSG:4326")
        building_metric = building_series.to_crs(metric_crs)

        for mode, buffer_m in self.buffers.items():
            buffered_metric = building_metric.buffer(buffer_m)
            self._buffered[mode] = buffered_metric.to_crs("EPSG:4326").iloc[0]

        logger.info(f"BuildingTest prepared with {len(self.building_gdf)} buildings")

    def test_candidate(self, ctx: CandidateContext) -> ScreenResult:
        """Test if the candidate geometry intersects buffered buildings."""
        if self.building_union is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="No buildings in area",
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
            reason=f"Road intersects building buffer ({buffer_m}m for {mode})",
            details={
                "road_class": ctx.road_class,
                "travel_mode": mode,
                "buffer_m": buffer_m,
            },
        )
