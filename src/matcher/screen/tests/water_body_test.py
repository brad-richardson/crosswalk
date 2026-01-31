"""Water body screen test.

Detects matches where the target road intersects water bodies (lakes, rivers, etc.)
beyond what could reasonably be a bridge. Uses dual threshold to allow legitimate
bridges while catching impossible matches.
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
from ..context.overture_water import fetch_overture_water, get_water_union

# Bridge thresholds - road passes if it satisfies EITHER threshold
RELATIVE_WATER_THRESHOLD = 0.10  # Allow up to 10% of road length in water
ABSOLUTE_WATER_THRESHOLD_M = 200.0  # Allow up to 200m in water (for long roads)

# Warning threshold for suspicious but not definitive cases
RELATIVE_WARN_THRESHOLD = 0.05  # Warn if > 5% but < 10% in water
ABSOLUTE_WARN_THRESHOLD_M = 100.0  # Warn if > 100m but < 200m in water


@register_test
class WaterBodyTest(ScreenTest):
    """Test if a match's target geometry intersects water bodies.

    Uses dual threshold for bridge handling:
    - Relative: Allow up to 10% of road length to intersect water
    - Absolute: Allow up to 200m of intersection (for long roads)

    A road passes if it satisfies EITHER threshold (whichever is more permissive).
    """

    name = "water_body"

    def __init__(
        self,
        relative_threshold: float = RELATIVE_WATER_THRESHOLD,
        absolute_threshold_m: float = ABSOLUTE_WATER_THRESHOLD_M,
        relative_warn_threshold: float = RELATIVE_WARN_THRESHOLD,
        absolute_warn_threshold_m: float = ABSOLUTE_WARN_THRESHOLD_M,
    ) -> None:
        self.relative_threshold = relative_threshold
        self.absolute_threshold_m = absolute_threshold_m
        self.relative_warn_threshold = relative_warn_threshold
        self.absolute_warn_threshold_m = absolute_warn_threshold_m
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
        """Test if the target geometry intersects water bodies."""
        if self.water_union is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="No water bodies in area",
            )

        target_geom = ctx.target_geom
        if not target_geom.intersects(self.water_union):
            return ScreenResult(outcome=ScreenOutcome.PASS, test_name=self.name)

        intersection = target_geom.intersection(self.water_union)
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
                    reason=f"Road intersects water: {intersection_length_m:.1f}m "
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
            reason=f"Road significantly intersects water: {intersection_length_m:.1f}m "
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
