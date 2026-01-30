"""Building footprint falsification test.

Detects matches where the target road passes through building footprints.
Roads should not typically pass through buildings (except for covered passages,
parking structures, etc.).
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
from ..context.overture_buildings import fetch_overture_buildings, get_building_union

# Building intersection thresholds
# More lenient than water since covered passages, parking garages, etc. exist
RELATIVE_BUILDING_THRESHOLD = 0.15  # Allow up to 15% of road length through buildings
ABSOLUTE_BUILDING_THRESHOLD_M = 50.0  # Allow up to 50m through buildings

# Warning thresholds
RELATIVE_WARN_THRESHOLD = 0.05  # Warn if > 5% but < 15%
ABSOLUTE_WARN_THRESHOLD_M = 20.0  # Warn if > 20m but < 50m


@register_test
class BuildingTest(FalsificationTest):
    """Test if a match's target geometry passes through buildings.

    Uses dual threshold similar to water body test:
    - Relative: Allow up to 15% of road length through buildings
    - Absolute: Allow up to 50m through buildings

    More lenient than water because:
    - Covered passages and arcades exist
    - Parking structures often have roads through them
    - Building footprints may be imprecise
    - Some roads legitimately go under/over buildings
    """

    name = "building"

    def __init__(
        self,
        relative_threshold: float = RELATIVE_BUILDING_THRESHOLD,
        absolute_threshold_m: float = ABSOLUTE_BUILDING_THRESHOLD_M,
        relative_warn_threshold: float = RELATIVE_WARN_THRESHOLD,
        absolute_warn_threshold_m: float = ABSOLUTE_WARN_THRESHOLD_M,
    ) -> None:
        """Initialize the building test.

        Args:
            relative_threshold: Maximum fraction of road length allowed in buildings
            absolute_threshold_m: Maximum meters of road allowed in buildings
            relative_warn_threshold: Fraction threshold for WARN outcome
            absolute_warn_threshold_m: Meters threshold for WARN outcome
        """
        self.relative_threshold = relative_threshold
        self.absolute_threshold_m = absolute_threshold_m
        self.relative_warn_threshold = relative_warn_threshold
        self.absolute_warn_threshold_m = absolute_warn_threshold_m
        self.building_gdf: gpd.GeoDataFrame | None = None
        self.building_union: Polygon | MultiPolygon | None = None
        self._metric_crs: str | None = None

    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Fetch building footprints for the bounding box.

        Args:
            bbox: Bounding box as (xmin, ymin, xmax, ymax) in EPSG:4326
        """
        self.building_gdf = fetch_overture_buildings(bbox)

        if len(self.building_gdf) > 0:
            self.building_union = get_building_union(self.building_gdf)
            # Estimate UTM CRS for metric calculations
            self._metric_crs = str(self.building_gdf.estimate_utm_crs())
        else:
            self.building_union = None
            self._metric_crs = None

        logger.info(f"BuildingTest prepared with {len(self.building_gdf)} buildings")

    def test_match(self, ctx: MatchContext) -> FalsificationResult:
        """Test if the target geometry passes through buildings.

        Args:
            ctx: Match context with geometries

        Returns:
            FalsificationResult with PASS, FAIL, WARN, or SKIP
        """
        # Skip if no building data
        if self.building_union is None:
            return FalsificationResult(
                outcome=FalsificationOutcome.SKIP,
                test_name=self.name,
                reason="No buildings in area",
            )

        # Check if target intersects buildings
        target_geom = ctx.target_geom
        if not target_geom.intersects(self.building_union):
            return FalsificationResult(
                outcome=FalsificationOutcome.PASS,
                test_name=self.name,
            )

        # Calculate intersection length in meters
        intersection = target_geom.intersection(self.building_union)
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
                    reason=f"Road passes through buildings: {intersection_length_m:.1f}m "
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
            reason=f"Road significantly passes through buildings: {intersection_length_m:.1f}m "
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
            return geom.length * 111000

        # Project to metric CRS for accurate measurement
        geom_series = gpd.GeoSeries([geom], crs="EPSG:4326")
        geom_metric = geom_series.to_crs(self._metric_crs)
        return geom_metric.length.iloc[0]
