"""Building footprint screen test.

Detects matches where the target road passes through or too close to
building footprints.
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
from ..constants import BUILDING_BUFFER_M
from ..context.overture_buildings import fetch_overture_buildings, get_building_union
from .travel_mode import get_travel_mode


@register_test
class BuildingTest(ScreenTest):
    """Test if a match's target geometry passes through buildings.

    Buffers building footprints based on road type - vehicle roads need
    more clearance than pedestrian paths.
    """

    name = "building"

    def __init__(self, buffer_overrides: dict[str, float] | None = None) -> None:
        """Initialize the building test.

        Args:
            buffer_overrides: Optional override buffer distances by mode
        """
        self.buffers = {**BUILDING_BUFFER_M, **(buffer_overrides or {})}
        self.building_gdf: gpd.GeoDataFrame | None = None
        self.building_union: Polygon | MultiPolygon | None = None
        self._metric_crs: str | None = None

    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Fetch building footprints for the bounding box."""
        self.building_gdf = fetch_overture_buildings(bbox)

        if len(self.building_gdf) > 0:
            self.building_union = get_building_union(self.building_gdf)
            self._metric_crs = str(self.building_gdf.estimate_utm_crs())
        else:
            self.building_union = None
            self._metric_crs = None

        logger.info(f"BuildingTest prepared with {len(self.building_gdf)} buildings")

    def test_match(self, ctx: MatchContext) -> ScreenResult:
        """Test if the target geometry intersects buffered buildings."""
        if self.building_union is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="No buildings in area",
            )

        if self._metric_crs is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="Could not determine CRS for buffering",
            )

        # Get buffer distance based on road class
        mode = get_travel_mode(ctx.road_class)
        buffer_m = self.buffers.get(mode, self.buffers["vehicle"])

        # Buffer buildings in metric CRS
        building_series = gpd.GeoSeries([self.building_union], crs="EPSG:4326")
        building_metric = building_series.to_crs(self._metric_crs)
        buffered_metric = building_metric.buffer(buffer_m)
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
            reason=f"Road intersects building buffer ({buffer_m}m for {mode})",
            details={
                "road_class": ctx.road_class,
                "travel_mode": mode,
                "buffer_m": buffer_m,
            },
        )
