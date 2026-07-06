"""Fringe detection screen test.

Detects target segments that are outside the reference network coverage area.
These "fringe" segments are at the boundary where reference data may be incomplete,
and should not be integrated as they could be false positives.
"""

import geopandas as gpd
from loguru import logger
from shapely import MultiPoint, Point, concave_hull
from shapely.geometry import MultiPolygon, Polygon

from ..base import (
    CandidateContext,
    ScreenOutcome,
    ScreenResult,
    ScreenTest,
    register_test,
)
from ..constants import FRINGE_BUFFER_M, FRINGE_HULL_RATIO, FRINGE_MIN_INSIDE_LENGTH_M


def compute_reference_coverage(
    reference_edges: gpd.GeoDataFrame,
    buffer_distance_m: float = FRINGE_BUFFER_M,
    hull_ratio: float = FRINGE_HULL_RATIO,
) -> Polygon | MultiPolygon | None:
    """Compute a coverage polygon from the reference network.

    Creates a concave hull around the reference network and buffers it
    to define the area where target segments are considered valid.
    Segments outside this area are likely "fringe" data at the boundary
    of the reference coverage.

    Args:
        reference_edges: GeoDataFrame of reference network edges
        buffer_distance_m: Buffer distance (meters) to expand the hull
        hull_ratio: Concave hull ratio (0=convex, 1=very tight). Default 0.3.

    Returns:
        Shapely Polygon representing the coverage area, or None if no coverage
    """
    if len(reference_edges) == 0:
        return None

    # Ensure we're working in a metric CRS for accurate buffering
    working_gdf = reference_edges
    original_crs = reference_edges.crs
    metric_crs = None

    if original_crs is not None and original_crs.is_geographic:
        metric_crs = reference_edges.estimate_utm_crs()
        working_gdf = reference_edges.to_crs(metric_crs)

    # Extract all coordinates from the reference network (LineStrings only)
    all_coords = []
    for geom in working_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        all_coords.extend(list(geom.coords))

    if len(all_coords) < 3:
        return None

    # Create concave hull
    points = MultiPoint([Point(c) for c in all_coords])
    try:
        hull = concave_hull(points, ratio=hull_ratio)
    except (ValueError, RuntimeError) as e:
        # Fall back to convex hull if concave hull fails
        logger.debug(f"Concave hull failed, using convex hull: {e}")
        hull = points.convex_hull

    if hull is None or hull.is_empty:
        return None

    # Buffer the hull to account for gaps at the edges
    coverage = hull.buffer(buffer_distance_m)

    # Convert back to original CRS if we transformed
    if metric_crs is not None and original_crs is not None:
        coverage_series = gpd.GeoSeries([coverage], crs=metric_crs)
        coverage = coverage_series.to_crs(original_crs).iloc[0]

    return coverage


def compute_inside_length(
    geom,
    coverage_polygon: Polygon | MultiPolygon,
) -> float:
    """Compute how much of a geometry is inside the coverage polygon.

    Args:
        geom: LineString geometry to check
        coverage_polygon: Coverage polygon

    Returns:
        Length in the geometry's CRS units that is inside coverage
    """
    if geom is None or geom.is_empty:
        return 0.0

    try:
        inside_portion = geom.intersection(coverage_polygon)
        if inside_portion.is_empty:
            return 0.0
        return inside_portion.length
    except (ValueError, RuntimeError) as e:
        logger.debug(f"Intersection failed for geometry: {e}")
        return 0.0


@register_test
class FringeTest(ScreenTest):
    """Test if a candidate segment is within the reference network coverage area.

    Segments outside the buffered concave hull of the reference network are
    considered "fringe" segments at the boundary where reference data may be
    incomplete. These should not be integrated as they're likely false positives.

    Unlike other screen tests that fetch external context, this test requires
    the reference network to be provided via set_reference().
    """

    name = "fringe"

    def __init__(
        self,
        buffer_distance_m: float = FRINGE_BUFFER_M,
        hull_ratio: float = FRINGE_HULL_RATIO,
        min_inside_length_m: float = FRINGE_MIN_INSIDE_LENGTH_M,
    ) -> None:
        """Initialize the fringe test.

        Args:
            buffer_distance_m: Buffer distance (meters) around reference coverage
            hull_ratio: Concave hull ratio (0=convex, 1=very tight)
            min_inside_length_m: Minimum length (meters) inside coverage to pass
        """
        self.buffer_distance_m = buffer_distance_m
        self.hull_ratio = hull_ratio
        self.min_inside_length_m = min_inside_length_m
        self.coverage_polygon: Polygon | MultiPolygon | None = None
        self._reference_set = False

    def set_reference(self, reference_edges: gpd.GeoDataFrame) -> None:
        """Set the reference network for coverage computation.

        This must be called before prepare() or test_candidate().

        Args:
            reference_edges: GeoDataFrame of reference network edges
        """
        self.coverage_polygon = compute_reference_coverage(
            reference_edges,
            buffer_distance_m=self.buffer_distance_m,
            hull_ratio=self.hull_ratio,
        )
        self._reference_set = True

        if self.coverage_polygon is not None:
            logger.info(
                f"FringeTest: computed coverage polygon from {len(reference_edges)} reference edges"
            )
        else:
            logger.warning("FringeTest: could not compute coverage polygon")

    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Prepare the test.

        For FringeTest, set_reference() should be called before this.
        The bbox is not used since we compute coverage from reference.
        """
        if not self._reference_set:
            logger.warning(
                "FringeTest.prepare() called without set_reference(). "
                "Call set_reference() first with reference network."
            )

    def test_candidate(self, ctx: CandidateContext) -> ScreenResult:
        """Test if the candidate is within reference coverage.

        Args:
            ctx: Candidate context with geometry

        Returns:
            ScreenResult - PASS if inside coverage, FAIL if outside
        """
        if self.coverage_polygon is None:
            return ScreenResult(
                outcome=ScreenOutcome.SKIP,
                test_name=self.name,
                reason="No reference coverage polygon computed",
            )

        inside_length = compute_inside_length(ctx.target_geom, self.coverage_polygon)

        if inside_length >= self.min_inside_length_m:
            return ScreenResult(
                outcome=ScreenOutcome.PASS,
                test_name=self.name,
            )

        return ScreenResult(
            outcome=ScreenOutcome.FAIL,
            test_name=self.name,
            reason="Segment outside reference coverage area",
            details={
                "inside_length_m": round(inside_length, 2),
                "min_required_m": self.min_inside_length_m,
                "buffer_distance_m": self.buffer_distance_m,
            },
        )


def filter_fringe_segments(
    target_edges: gpd.GeoDataFrame,
    reference_edges: gpd.GeoDataFrame,
    buffer_distance_m: float = FRINGE_BUFFER_M,
    hull_ratio: float = FRINGE_HULL_RATIO,
    min_inside_length_m: float = FRINGE_MIN_INSIDE_LENGTH_M,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Filter target segments that are outside the reference coverage area.

    This is a convenience function for batch filtering. For individual
    segment testing, use the FringeTest class.

    Args:
        target_edges: GeoDataFrame of target edges to filter
        reference_edges: GeoDataFrame of reference network edges
        buffer_distance_m: Buffer distance (meters) around reference coverage
        hull_ratio: Concave hull ratio (0=convex, 1=very tight)
        min_inside_length_m: Minimum length (meters) inside coverage to be valid

    Returns:
        Tuple of:
        - valid_targets: Targets with sufficient coverage
        - fringe_targets: Targets outside coverage (marked with screen_result)
    """
    if len(target_edges) == 0:
        return target_edges, gpd.GeoDataFrame(columns=target_edges.columns, crs=target_edges.crs)

    # Compute coverage polygon
    coverage_polygon = compute_reference_coverage(
        reference_edges,
        buffer_distance_m=buffer_distance_m,
        hull_ratio=hull_ratio,
    )

    if coverage_polygon is None:
        logger.warning("Could not compute reference coverage - accepting all targets")
        return target_edges, gpd.GeoDataFrame(columns=target_edges.columns, crs=target_edges.crs)

    # Compute inside length for each target
    inside_lengths = []
    for geom in target_edges.geometry:
        inside_lengths.append(compute_inside_length(geom, coverage_polygon))

    target_edges = target_edges.copy()
    target_edges["_inside_coverage_length"] = inside_lengths

    # Split into valid and fringe
    valid_mask = target_edges["_inside_coverage_length"] >= min_inside_length_m
    valid_targets = target_edges[valid_mask].drop(columns=["_inside_coverage_length"])
    fringe_targets = target_edges[~valid_mask].copy()

    # Mark fringe segments
    if len(fringe_targets) > 0:
        fringe_targets["screen_result"] = "fringe"
        fringe_targets["unmatched_reason"] = "outside_reference_coverage"
        fringe_targets = fringe_targets.drop(columns=["_inside_coverage_length"])

    logger.info(
        f"Fringe filter: {len(valid_targets)} valid, {len(fringe_targets)} fringe "
        f"(buffer={buffer_distance_m}m, min_inside={min_inside_length_m}m)"
    )

    return valid_targets, fringe_targets
