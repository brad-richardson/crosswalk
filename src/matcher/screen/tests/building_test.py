"""Building footprint screen test.

Detects matches where the target road passes through building footprints.
Roads should not typically pass through buildings (except for covered passages,
parking structures, etc.).
"""

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, MultiPolygon, Polygon

from ..base import (
    MatchContext,
    ScreenOutcome,
    ScreenResult,
    ScreenTest,
    register_test,
)
from ..context.overture_buildings import fetch_overture_buildings, get_building_union

# Building intersection thresholds (more lenient than water)
RELATIVE_BUILDING_THRESHOLD = 0.15  # Allow up to 15% of road length through buildings
ABSOLUTE_BUILDING_THRESHOLD_M = 50.0  # Allow up to 50m through buildings

# Warning thresholds
RELATIVE_WARN_THRESHOLD = 0.05  # Warn if > 5% but < 15%
ABSOLUTE_WARN_THRESHOLD_M = 20.0  # Warn if > 20m but < 50m


@register_test
class BuildingTest(ScreenTest):
    """Test if a match's target geometry passes through buildings.

    Uses dual threshold similar to water body test:
    - Relative: Allow up to 15% of road length through buildings
    - Absolute: Allow up to 50m through buildings

    More lenient than water because covered passages, parking structures,
    and imprecise footprints exist.
    """

    name = "building"

    def __init__(
        self,
        relative_threshold: float = RELATIVE_BUILDING_THRESHOLD,
        absolute_threshold_m: float = ABSOLUTE_BUILDING_THRESHOLD_M,
        relative_warn_threshold: float = RELATIVE_WARN_THRESHOLD,
        absolute_warn_threshold_m: float = ABSOLUTE_WARN_THRESHOLD_M,
    ) -> None:
        self.relative_threshold = relative_threshold
        self.absolute_threshold_m = absolute_threshold_m
        self.relative_warn_threshold = relative_warn_threshold
        self.absolute_warn_threshold_m = absolute_warn_threshold_m
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
        """Test if the target geometry passes through buildings."""
        if self.building_union is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="No buildings in area",
            )

        target_geom = ctx.target_geom
        if not target_geom.intersects(self.building_union):
            return ScreenResult(outcome=ScreenOutcome.PASS, test_name=self.name)

        intersection = target_geom.intersection(self.building_union)
        if intersection.is_empty:
            return ScreenResult(outcome=ScreenOutcome.PASS, test_name=self.name)

        intersection_length_m = self._calculate_length_m(intersection)
        target_length_m = self._calculate_length_m(target_geom)

        if target_length_m <= 0:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="Zero-length geometry",
            )

        relative_intersection = intersection_length_m / target_length_m

        details = {
            "intersection_length_m": round(intersection_length_m, 2),
            "target_length_m": round(target_length_m, 2),
            "relative_intersection": round(relative_intersection, 4),
        }

        passes_relative = relative_intersection <= self.relative_threshold
        passes_absolute = intersection_length_m <= self.absolute_threshold_m

        if passes_relative or passes_absolute:
            warns_relative = relative_intersection > self.relative_warn_threshold
            warns_absolute = intersection_length_m > self.absolute_warn_threshold_m

            if warns_relative and warns_absolute:
                return ScreenResult(
                    outcome=ScreenOutcome.WARN,
                    test_name=self.name,
                    reason=f"Road passes through buildings: {intersection_length_m:.1f}m "
                    f"({relative_intersection:.1%} of road length)",
                    details=details,
                )

            return ScreenResult(
                outcome=ScreenOutcome.PASS,
                test_name=self.name,
                details=details,
            )

        return ScreenResult(
            outcome=ScreenOutcome.FAIL,
            test_name=self.name,
            reason=f"Road significantly passes through buildings: {intersection_length_m:.1f}m "
            f"({relative_intersection:.1%} of road length). "
            f"Exceeds both {self.relative_threshold:.0%} relative and "
            f"{self.absolute_threshold_m:.0f}m absolute thresholds.",
            details=details,
        )

    def _calculate_length_m(self, geom: LineString) -> float:
        """Calculate geometry length in meters."""
        if self._metric_crs is None:
            return geom.length * 111000

        geom_series = gpd.GeoSeries([geom], crs="EPSG:4326")
        geom_metric = geom_series.to_crs(self._metric_crs)
        return geom_metric.length.iloc[0]
