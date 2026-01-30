"""Water body falsification test.

Detects matches where the target road intersects water bodies (lakes, rivers, etc.)
beyond what could reasonably be a bridge. Uses dual threshold to allow legitimate
bridges while catching impossible matches.
"""

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, MultiPolygon, Polygon

from ..base import (
    FalsificationOutcome,
    FalsificationResult,
    FalsificationTest,
    MatchContext,
    register_test,
)
from ..context.overture_water import fetch_overture_water, get_water_union

# Bridge thresholds - road passes if it satisfies EITHER threshold
# This allows legitimate bridges while catching impossible matches
RELATIVE_WATER_THRESHOLD = 0.10  # Allow up to 10% of road length in water
ABSOLUTE_WATER_THRESHOLD_M = 200.0  # Allow up to 200m in water (for long roads)

# Warning threshold for suspicious but not definitive cases
RELATIVE_WARN_THRESHOLD = 0.05  # Warn if > 5% but < 10% in water
ABSOLUTE_WARN_THRESHOLD_M = 100.0  # Warn if > 100m but < 200m in water


@register_test
class WaterBodyTest(FalsificationTest):
    """Test if a match's target geometry intersects water bodies.

    Uses dual threshold for bridge handling:
    - Relative: Allow up to 10% of road length to intersect water
    - Absolute: Allow up to 200m of intersection (for long roads)

    A road passes if it satisfies EITHER threshold (whichever is more permissive).
    This accommodates legitimate bridges while catching roads that are clearly
    impossible (e.g., a 500m road entirely in a lake).
    """

    name = "water_body"

    def __init__(
        self,
        relative_threshold: float = RELATIVE_WATER_THRESHOLD,
        absolute_threshold_m: float = ABSOLUTE_WATER_THRESHOLD_M,
        relative_warn_threshold: float = RELATIVE_WARN_THRESHOLD,
        absolute_warn_threshold_m: float = ABSOLUTE_WARN_THRESHOLD_M,
    ) -> None:
        """Initialize the water body test.

        Args:
            relative_threshold: Maximum fraction of road length allowed in water
            absolute_threshold_m: Maximum meters of road allowed in water
            relative_warn_threshold: Fraction threshold for WARN outcome
            absolute_warn_threshold_m: Meters threshold for WARN outcome
        """
        self.relative_threshold = relative_threshold
        self.absolute_threshold_m = absolute_threshold_m
        self.relative_warn_threshold = relative_warn_threshold
        self.absolute_warn_threshold_m = absolute_warn_threshold_m
        self.water_gdf: gpd.GeoDataFrame | None = None
        self.water_union: Polygon | MultiPolygon | None = None
        self._metric_crs: str | None = None

    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Fetch water body polygons for the bounding box.

        Args:
            bbox: Bounding box as (xmin, ymin, xmax, ymax) in EPSG:4326
        """
        self.water_gdf = fetch_overture_water(bbox)

        if len(self.water_gdf) > 0:
            self.water_union = get_water_union(self.water_gdf)
            # Estimate UTM CRS for metric calculations
            self._metric_crs = str(self.water_gdf.estimate_utm_crs())
        else:
            self.water_union = None
            self._metric_crs = None

        logger.info(f"WaterBodyTest prepared with {len(self.water_gdf)} water bodies")

    def test_match(self, ctx: MatchContext) -> FalsificationResult:
        """Test if the target geometry intersects water bodies.

        Uses the target geometry since that's what we're adding to the network.
        The reference (Overture) road is assumed to be authoritative.

        Args:
            ctx: Match context with geometries

        Returns:
            FalsificationResult with PASS, FAIL, WARN, or SKIP
        """
        # Skip if no water data
        if self.water_union is None:
            return FalsificationResult(
                outcome=FalsificationOutcome.SKIP,
                test_name=self.name,
                reason="No water bodies in area",
            )

        # Check if target intersects water
        target_geom = ctx.target_geom
        if not target_geom.intersects(self.water_union):
            return FalsificationResult(
                outcome=FalsificationOutcome.PASS,
                test_name=self.name,
            )

        # Calculate intersection length in meters
        intersection = target_geom.intersection(self.water_union)
        if intersection.is_empty:
            return FalsificationResult(
                outcome=FalsificationOutcome.PASS,
                test_name=self.name,
            )

        # Project to metric CRS for length calculation
        intersection_length_m = self._calculate_length_m(intersection)
        target_length_m = self._calculate_length_m(target_geom)

        if target_length_m <= 0:
            return FalsificationResult(
                outcome=FalsificationOutcome.SKIP,
                test_name=self.name,
                reason="Zero-length geometry",
            )

        # Calculate relative intersection
        relative_intersection = intersection_length_m / target_length_m

        details = {
            "intersection_length_m": round(intersection_length_m, 2),
            "target_length_m": round(target_length_m, 2),
            "relative_intersection": round(relative_intersection, 4),
        }

        # Apply dual threshold: PASS if either threshold is satisfied
        passes_relative = relative_intersection <= self.relative_threshold
        passes_absolute = intersection_length_m <= self.absolute_threshold_m

        if passes_relative or passes_absolute:
            # Check for warning condition
            warns_relative = relative_intersection > self.relative_warn_threshold
            warns_absolute = intersection_length_m > self.absolute_warn_threshold_m

            if warns_relative and warns_absolute:
                return FalsificationResult(
                    outcome=FalsificationOutcome.WARN,
                    test_name=self.name,
                    reason=f"Road intersects water: {intersection_length_m:.1f}m "
                    f"({relative_intersection:.1%} of road length)",
                    details=details,
                )

            return FalsificationResult(
                outcome=FalsificationOutcome.PASS,
                test_name=self.name,
                details=details,
            )

        # Fails both thresholds
        return FalsificationResult(
            outcome=FalsificationOutcome.FAIL,
            test_name=self.name,
            reason=f"Road significantly intersects water: {intersection_length_m:.1f}m "
            f"({relative_intersection:.1%} of road length). "
            f"Exceeds both {self.relative_threshold:.0%} relative and "
            f"{self.absolute_threshold_m:.0f}m absolute thresholds.",
            details=details,
        )

    def _calculate_length_m(self, geom: LineString) -> float:
        """Calculate geometry length in meters.

        Args:
            geom: Geometry in EPSG:4326

        Returns:
            Length in meters
        """
        if self._metric_crs is None:
            # Fallback: approximate using latitude
            # 1 degree ~ 111km at equator
            return geom.length * 111000

        # Project to metric CRS for accurate measurement
        geom_series = gpd.GeoSeries([geom], crs="EPSG:4326")
        geom_metric = geom_series.to_crs(self._metric_crs)
        return geom_metric.length.iloc[0]
