"""Base class for polygon buffer screen tests.

Provides shared functionality for tests that check if road segments intersect
buffered polygon geometries (buildings, water bodies, landcover, etc.).
"""

from collections.abc import Callable

import geopandas as gpd
from loguru import logger
from shapely.geometry import MultiPolygon, Polygon

from ..base import (
    CandidateContext,
    ScreenOutcome,
    ScreenResult,
    ScreenTest,
)
from ..context import get_polygon_union
from .travel_mode import get_travel_mode


class PolygonBufferTest(ScreenTest):
    """Base class for polygon buffer intersection tests.

    Subclasses must define:
    - name: str - Test identifier
    - feature_type: str - Human-readable name for the feature (e.g., "building", "water body")
    - default_buffers: dict[str, float] - Buffer distances by travel mode
    - fetch_func: Callable - Function to fetch polygon data

    Optionally:
    - min_area_m2: float - Minimum polygon area (from constants)
    """

    name: str = ""
    feature_type: str = ""
    default_buffers: dict[str, float] = {}
    fetch_func: Callable[..., gpd.GeoDataFrame] | None = None

    def __init__(self, buffer_overrides: dict[str, float] | None = None) -> None:
        self.buffers = {**self.default_buffers, **(buffer_overrides or {})}
        self.polygon_gdf: gpd.GeoDataFrame | None = None
        self.polygon_union: Polygon | MultiPolygon | None = None
        # Pre-computed buffered geometries by travel mode (in EPSG:4326)
        self._buffered: dict[str, Polygon | MultiPolygon] = {}

    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Fetch polygons and pre-compute buffered versions."""
        if self.fetch_func is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define fetch_func")

        self.polygon_gdf = self.fetch_func(bbox)
        self._buffered = {}

        if len(self.polygon_gdf) == 0:
            self.polygon_union = None
            logger.info(f"{self.__class__.__name__} prepared with 0 {self.feature_type}s")
            return

        self.polygon_union = get_polygon_union(self.polygon_gdf)
        if self.polygon_union is None:
            logger.info(f"{self.__class__.__name__} prepared with 0 valid {self.feature_type}s")
            return

        # Pre-compute buffered geometries for each travel mode
        metric_crs = str(self.polygon_gdf.estimate_utm_crs())
        polygon_series = gpd.GeoSeries([self.polygon_union], crs="EPSG:4326")
        polygon_metric = polygon_series.to_crs(metric_crs)

        for mode, buffer_m in self.buffers.items():
            buffered_metric = polygon_metric.buffer(buffer_m)
            self._buffered[mode] = buffered_metric.to_crs("EPSG:4326").iloc[0]

        logger.info(
            f"{self.__class__.__name__} prepared with {len(self.polygon_gdf)} {self.feature_type}s"
        )

    def test_candidate(self, ctx: CandidateContext) -> ScreenResult:
        """Test if the candidate geometry intersects buffered polygons."""
        if self.polygon_union is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason=f"No {self.feature_type}s in area",
            )

        mode = get_travel_mode(ctx.road_class)
        buffered = self._buffered.get(mode)
        if buffered is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason=f"No buffer computed for mode: {mode}",
            )

        buffer_m = self.buffers.get(mode, self.buffers.get("vehicle", 0))

        target_geom = ctx.target_geom
        if not target_geom.intersects(buffered):
            return ScreenResult(outcome=ScreenOutcome.PASS, test_name=self.name)

        intersection = target_geom.intersection(buffered)
        if intersection.is_empty:
            return ScreenResult(outcome=ScreenOutcome.PASS, test_name=self.name)

        return ScreenResult(
            outcome=ScreenOutcome.FAIL,
            test_name=self.name,
            reason=f"Road intersects {self.feature_type} buffer ({buffer_m}m for {mode})",
            details={
                "road_class": ctx.road_class,
                "travel_mode": mode,
                "buffer_m": buffer_m,
            },
        )
