"""Shared spatial query utilities.

Provides reusable STRtree-based spatial query functions used across
blocking, integration, and feature computation modules.
"""

import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree


class SpatialIndex:
    """Lightweight STRtree wrapper for common spatial query patterns.

    Builds once, queries many times. Avoids rebuilding the tree
    for repeated queries against the same geometry set.

    Example:
        idx = SpatialIndex(reference_gdf.geometry.values)
        nearby = idx.query_nearby(target_geom, buffer_distance=10.0)
        # nearby is an array of indices into the original geometry array
    """

    def __init__(self, geometries: np.ndarray):
        """Build spatial index from geometry array.

        Args:
            geometries: Array of shapely geometries (e.g., gdf.geometry.values)
        """
        self.geometries = geometries
        self.tree = STRtree(geometries)

    def query_nearby(self, geom: BaseGeometry, buffer_distance: float) -> np.ndarray:
        """Find geometries within buffer_distance of geom.

        Args:
            geom: Query geometry
            buffer_distance: Buffer radius (in CRS units - meters if projected)

        Returns:
            Array of indices into self.geometries
        """
        search_area = geom.buffer(buffer_distance)
        return self.tree.query(search_area)

    def nearest(self, geom: BaseGeometry) -> int:
        """Find the nearest geometry to geom.

        Args:
            geom: Query geometry

        Returns:
            Index of nearest geometry in self.geometries
        """
        return self.tree.nearest(geom)

    def query_nearby_union(
        self,
        geom: BaseGeometry,
        buffer_distance: float,
        query_margin: float | None = None,
    ) -> BaseGeometry | None:
        """Find nearby geometries, union them, and return the merged result.

        Common pattern: find nearby reference edges, merge them into a single
        geometry for coverage/difference calculations.

        Args:
            geom: Query geometry
            buffer_distance: Buffer to apply to the unioned result
            query_margin: Search radius around geom (defaults to 2x buffer_distance).
                Controls how far to search for candidate geometries.

        Returns:
            Buffered union of nearby geometries, or None if no nearby geometries
        """
        if query_margin is None:
            query_margin = buffer_distance * 2

        nearby_idx = self.query_nearby(geom, query_margin)

        if len(nearby_idx) == 0:
            return None

        nearby_geoms = unary_union([self.geometries[i] for i in nearby_idx])
        return nearby_geoms.buffer(buffer_distance)

    def compute_net_new(
        self,
        geom: BaseGeometry,
        coverage_buffer_m: float,
        query_margin: float | None = None,
    ) -> BaseGeometry | None:
        """Compute the portion of geom not covered by nearby indexed geometries.

        This is the "net new coverage" calculation: given a target segment,
        subtract the buffered area of nearby reference segments to find
        what's genuinely new.

        Args:
            geom: Target geometry to compute net-new for
            coverage_buffer_m: Buffer around reference geometries defining "coverage"
            query_margin: Search radius (defaults to 2x coverage_buffer_m)

        Returns:
            Net-new geometry (portion of geom outside coverage), or None if fully covered
        """
        coverage = self.query_nearby_union(geom, coverage_buffer_m, query_margin)

        if coverage is None:
            return geom  # No nearby reference = entirely net-new

        net_new = geom.difference(coverage)
        return None if net_new.is_empty else net_new
