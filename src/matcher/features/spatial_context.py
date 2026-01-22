"""Spatial context indexing for relational feature computation.

This module provides spatial indexes and utilities for:
1. Finding anchor roads for parallel infrastructure (sidewalks, bike lanes)
2. Inferring endpoint connectivity from proximity
3. Supporting context propagation across nearby segments
4. Building road network graphs from spaghetti geometry for graphlet analysis

These features work without requiring explicit topology in the target data,
making them suitable for raw "spaghetti" line datasets.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely import LineString, Point
from shapely.strtree import STRtree

if TYPE_CHECKING:
    import networkx as nx

from .relational import (
    compute_parallel_alignment,
    compute_perpendicular_offset,
    compute_side_of_street,
)


@dataclass
class AnchorMatch:
    """Result of matching a segment to its anchor road."""

    anchor_idx: int
    """Index of the anchor road in the roads GeoDataFrame."""

    anchor_id: str | None
    """ID of the anchor road (if available)."""

    perpendicular_offset: float
    """Mean perpendicular distance to anchor (meters)."""

    offset_iqr: float
    """Interquartile range of perpendicular offset (meters)."""

    offset_p95: float
    """95th percentile of perpendicular offset (meters)."""

    parallel_alignment: float
    """How parallel the segment is to anchor (0-1)."""

    side_of_street: str
    """Which side: 'left', 'right', or 'unknown'."""

    side_confidence: float
    """Confidence in side determination (0-1)."""

    score: float
    """Overall match score (0-1, higher is better)."""


class AnchorRoadMatcher:
    """Match sidewalks/bike lanes to their anchor road centerlines.

    Uses spatial indexing to efficiently find candidate anchor roads,
    then scores them based on perpendicular offset and parallel alignment.

    Example:
        >>> roads_gdf = gpd.read_parquet("roads.parquet")
        >>> matcher = AnchorRoadMatcher(roads_gdf, max_offset=30.0)
        >>> sidewalk_geom = sidewalks_gdf.iloc[0].geometry
        >>> match = matcher.find_anchor_road(sidewalk_geom)
        >>> print(f"Anchor road: {match.anchor_id}, side: {match.side_of_street}")
    """

    def __init__(
        self,
        roads_gdf: gpd.GeoDataFrame,
        max_offset: float = 30.0,
        min_alignment: float = 0.7,
        id_column: str = "id",
    ):
        """Initialize the anchor road matcher.

        Args:
            roads_gdf: GeoDataFrame of road centerlines
            max_offset: Maximum perpendicular distance to consider (meters)
            min_alignment: Minimum parallel alignment score (0-1)
            id_column: Column name for road IDs
        """
        self.roads_gdf = roads_gdf
        self.max_offset = max_offset
        self.min_alignment = min_alignment
        self.id_column = id_column

        # Build spatial index
        self.roads_tree = STRtree(roads_gdf.geometry.values)

        # Cache road IDs
        if id_column in roads_gdf.columns:
            self.road_ids = roads_gdf[id_column].values
        else:
            self.road_ids = np.arange(len(roads_gdf))

        logger.debug(f"Built AnchorRoadMatcher with {len(roads_gdf)} roads")

    def find_anchor_road(
        self,
        target_geom: LineString,
    ) -> AnchorMatch | None:
        """Find the most likely anchor road for a target segment.

        Args:
            target_geom: Target geometry (sidewalk, bike lane)

        Returns:
            AnchorMatch if a suitable anchor is found, None otherwise
        """
        if target_geom.is_empty:
            return None

        # Query candidate roads within max_offset
        buffered = target_geom.buffer(self.max_offset)
        candidate_indices = self.roads_tree.query(buffered)

        if len(candidate_indices) == 0:
            return None

        # Score each candidate
        best_match: AnchorMatch | None = None
        best_score = -1.0

        for road_idx in candidate_indices:
            road_geom = self.roads_gdf.iloc[road_idx].geometry

            # Compute features (now returns mean, iqr, p95)
            offset, offset_iqr, offset_p95 = compute_perpendicular_offset(target_geom, road_geom)
            alignment = compute_parallel_alignment(target_geom, road_geom)
            side, side_conf = compute_side_of_street(target_geom, road_geom)

            # Skip if offset too large or not parallel enough
            if offset > self.max_offset or alignment < self.min_alignment:
                continue

            # Score: prefer low offset and high alignment
            # Normalize offset to 0-1 (lower is better)
            offset_score = max(0, 1 - offset / self.max_offset)
            score = 0.4 * offset_score + 0.4 * alignment + 0.2 * side_conf

            if score > best_score:
                best_score = score
                best_match = AnchorMatch(
                    anchor_idx=road_idx,
                    anchor_id=self.road_ids[road_idx],
                    perpendicular_offset=offset,
                    offset_iqr=offset_iqr,
                    offset_p95=offset_p95,
                    parallel_alignment=alignment,
                    side_of_street=side,
                    side_confidence=side_conf,
                    score=score,
                )

        return best_match

    def find_anchor_roads_batch(
        self,
        targets_gdf: gpd.GeoDataFrame,
    ) -> list[AnchorMatch | None]:
        """Find anchor roads for multiple targets efficiently.

        Args:
            targets_gdf: GeoDataFrame of target geometries

        Returns:
            List of AnchorMatch (or None) for each target
        """
        results = []
        for idx in range(len(targets_gdf)):
            geom = targets_gdf.iloc[idx].geometry
            match = self.find_anchor_road(geom)
            results.append(match)

        n_matched = sum(1 for m in results if m is not None)
        logger.info(f"Found anchors for {n_matched}/{len(targets_gdf)} targets")

        return results


@dataclass
class SpatialContextIndex:
    """Spatial indexes for endpoint proximity and connectivity inference.

    Builds R-tree indexes from a GeoDataFrame to efficiently query:
    - Nearby endpoints (for connectivity inference)
    - Nearby segments (for context propagation)

    Example:
        >>> ctx = SpatialContextIndex()
        >>> ctx.build_from_gdf(sidewalks_gdf, id_column="id")
        >>> nearby = ctx.query_nearby_endpoints(Point(0, 0), radius=5.0)
    """

    endpoint_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape (N, 2) with all endpoint coordinates."""

    endpoint_to_segment: dict[int, list[int]] = field(default_factory=dict)
    """Map from endpoint index to list of segment indices that share it."""

    segment_endpoints: dict[int, tuple[int, int]] = field(default_factory=dict)
    """Map from segment index to (start_endpoint_idx, end_endpoint_idx)."""

    segment_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of segment IDs corresponding to indices."""

    _endpoint_tree: STRtree | None = field(default=None, repr=False)
    _segment_tree: STRtree | None = field(default=None, repr=False)
    _geometries: gpd.GeoSeries | None = field(default=None, repr=False)

    def build_from_gdf(
        self,
        gdf: gpd.GeoDataFrame,
        id_column: str = "id",
        snap_tolerance_m: float = 1.0,
    ) -> None:
        """Build spatial indexes from a GeoDataFrame.

        Args:
            gdf: GeoDataFrame with LineString geometries
            id_column: Column name for segment IDs
            snap_tolerance_m: Distance within which endpoints are considered the same (meters)
        """
        logger.debug(f"Building SpatialContextIndex from {len(gdf)} segments")

        # Store geometries and IDs
        self._geometries = gdf.geometry
        if id_column in gdf.columns:
            self.segment_ids = gdf[id_column].values
        else:
            self.segment_ids = np.arange(len(gdf))

        # Build segment tree
        self._segment_tree = STRtree(gdf.geometry.values)

        # Project to local CRS if in geographic coordinates (for accurate distance)
        work_gdf = gdf
        if gdf.crs is not None and gdf.crs.is_geographic:
            centroid = gdf.geometry.union_all().centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            hemisphere = "north" if centroid.y >= 0 else "south"
            epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
            work_gdf = gdf.to_crs(epsg=epsg)

        # Extract all endpoints using vectorized shapely operations
        geometries = work_gdf.geometry.values

        # Get start and end points vectorized (all geometries should be LineString after filtering)
        import shapely

        # Fast vectorized path for LineStrings
        # Filter out empty/null geometries
        valid_mask = ~shapely.is_empty(geometries) & ~shapely.is_missing(geometries)
        valid_indices = np.where(valid_mask)[0]
        valid_geoms = geometries[valid_mask]

        if len(valid_geoms) == 0:
            logger.warning("No endpoints found in GeoDataFrame")
            return

        # Get first and last points vectorized
        start_points = shapely.get_point(valid_geoms, 0)
        end_points = shapely.get_point(valid_geoms, -1)

        # Filter out geometries where start or end point is null/empty
        # (can happen for degenerate LineStrings with single point, etc.)
        points_valid = ~shapely.is_missing(start_points) & ~shapely.is_missing(end_points)
        if not points_valid.all():
            n_filtered = (~points_valid).sum()
            logger.debug(f"Filtered {n_filtered} geometries with invalid start/end points")
            start_points = start_points[points_valid]
            end_points = end_points[points_valid]
            valid_indices = valid_indices[points_valid]

        # Extract coordinates
        start_coords = shapely.get_coordinates(start_points)
        end_coords = shapely.get_coordinates(end_points)

        # Interleave start and end coordinates
        n_valid = len(valid_indices)  # Use actual filtered count
        all_endpoints = np.empty((n_valid * 2, 2), dtype=np.float64)
        all_endpoints[0::2] = start_coords
        all_endpoints[1::2] = end_coords

        self.endpoint_coords = all_endpoints
        self.segment_endpoints = {
            seg_idx: (i * 2, i * 2 + 1) for i, seg_idx in enumerate(valid_indices)
        }

        # Cluster nearby endpoints (snap tolerance)
        self._cluster_endpoints(snap_tolerance_m)

        # Build endpoint tree
        endpoint_points = [Point(c) for c in self.endpoint_coords]
        self._endpoint_tree = STRtree(endpoint_points)

        logger.debug(f"Built index with {len(self.endpoint_coords)} endpoints")

    def _cluster_endpoints(self, tolerance: float) -> None:
        """Cluster nearby endpoints and build endpoint->segment mapping.

        Uses Union-Find for O(N log N) complexity instead of O(N × M²).

        Args:
            tolerance: Distance within which endpoints are considered the same
        """
        n_endpoints = len(self.endpoint_coords)
        if n_endpoints == 0:
            self.endpoint_to_segment = {}
            return

        # Step 1: Build initial endpoint -> segments mapping
        endpoint_segments: dict[int, list[int]] = {}
        for seg_idx, (start_ep, end_ep) in self.segment_endpoints.items():
            if start_ep not in endpoint_segments:
                endpoint_segments[start_ep] = []
            endpoint_segments[start_ep].append(seg_idx)

            if end_ep not in endpoint_segments:
                endpoint_segments[end_ep] = []
            endpoint_segments[end_ep].append(seg_idx)

        # Step 2: Use Union-Find to cluster nearby endpoints
        if tolerance > 0:
            # Use tile-based spatial hashing for O(N × k) clustering
            uf = _cluster_endpoints_fast(self.endpoint_coords, tolerance)

            # Step 3: Build cluster -> segments mapping
            cluster_segments: dict[int, set[int]] = {}
            for ep_idx in range(n_endpoints):
                root = uf.find(ep_idx)
                if root not in cluster_segments:
                    cluster_segments[root] = set()
                if ep_idx in endpoint_segments:
                    cluster_segments[root].update(endpoint_segments[ep_idx])

            # Step 4: For each endpoint, its segment list is the cluster's segments
            self.endpoint_to_segment = {}
            for ep_idx in range(n_endpoints):
                root = uf.find(ep_idx)
                self.endpoint_to_segment[ep_idx] = list(cluster_segments[root])
        else:
            # No clustering, just use direct mapping
            self.endpoint_to_segment = {k: list(v) for k, v in endpoint_segments.items()}

    def query_nearby_endpoints(
        self,
        point: Point,
        radius: float,
    ) -> list[tuple[int, float]]:
        """Find endpoints within radius of a point.

        Args:
            point: Query point
            radius: Search radius (meters)

        Returns:
            List of (endpoint_idx, distance) tuples sorted by distance
        """
        if self._endpoint_tree is None:
            return []

        buffered = point.buffer(radius)
        candidate_indices = self._endpoint_tree.query(buffered)

        results = []
        # Handle 3D coordinates by taking only x, y (first 2 dimensions)
        point_coords = np.array(point.coords[0])[:2]

        for ep_idx in candidate_indices:
            ep_coords = self.endpoint_coords[ep_idx]
            dist = np.linalg.norm(ep_coords - point_coords)
            if dist <= radius:
                results.append((ep_idx, dist))

        return sorted(results, key=lambda x: x[1])

    def query_nearby_segments(
        self,
        geom: LineString | Point,
        radius: float,
    ) -> list[int]:
        """Find segment indices within radius of a geometry.

        Args:
            geom: Query geometry
            radius: Search radius (meters)

        Returns:
            List of segment indices
        """
        if self._segment_tree is None:
            return []

        buffered = geom.buffer(radius)
        return list(self._segment_tree.query(buffered))

    def infer_connectivity(
        self,
        segment_idx: int,
        tolerance_m: float = 5.0,
    ) -> list[int]:
        """Infer segments connected to this one via endpoint proximity.

        Args:
            segment_idx: Index of segment to check
            tolerance_m: Distance threshold for connectivity (meters)

        Returns:
            List of connected segment indices
        """
        if segment_idx not in self.segment_endpoints:
            return []

        start_ep_idx, end_ep_idx = self.segment_endpoints[segment_idx]
        connected = set()

        # Get segments sharing start endpoint
        if start_ep_idx in self.endpoint_to_segment:
            connected.update(self.endpoint_to_segment[start_ep_idx])

        # Get segments sharing end endpoint
        if end_ep_idx in self.endpoint_to_segment:
            connected.update(self.endpoint_to_segment[end_ep_idx])

        # Also query nearby endpoints within tolerance
        if self._endpoint_tree is not None:
            start_point = Point(self.endpoint_coords[start_ep_idx])
            end_point = Point(self.endpoint_coords[end_ep_idx])

            for nearby_ep, _dist in self.query_nearby_endpoints(start_point, tolerance_m):
                if nearby_ep in self.endpoint_to_segment:
                    connected.update(self.endpoint_to_segment[nearby_ep])

            for nearby_ep, _dist in self.query_nearby_endpoints(end_point, tolerance_m):
                if nearby_ep in self.endpoint_to_segment:
                    connected.update(self.endpoint_to_segment[nearby_ep])

        # Remove self
        connected.discard(segment_idx)

        return list(connected)

    def get_segment_geometry(self, segment_idx: int) -> LineString | None:
        """Get geometry for a segment index."""
        if self._geometries is None or segment_idx >= len(self._geometries):
            return None
        return self._geometries.iloc[segment_idx]


def infer_endpoint_degree(
    geom: LineString,
    context: SpatialContextIndex,
    tolerance_m: float = 5.0,
) -> tuple[int, int]:
    """Infer degree at each endpoint based on nearby segment count.

    For spaghetti data without explicit topology, we infer connectivity
    by counting segments with endpoints within tolerance distance.

    Args:
        geom: Segment geometry
        context: SpatialContextIndex with all segments
        tolerance_m: Distance threshold for connectivity (meters)

    Returns:
        Tuple of (start_degree, end_degree) where degree = count of segments
        with endpoints within tolerance. Minimum degree is 1 (self).
    """
    if geom.is_empty or context.endpoint_coords.size == 0:
        return (1, 1)

    # Extract endpoints
    coords = np.array(geom.coords)
    start_point = Point(coords[0])
    end_point = Point(coords[-1])

    # Query nearby endpoints at start
    # Note: start_segments will include the segment itself since its own endpoint
    # is within tolerance of itself. This is intentional - degree includes self,
    # so an isolated segment has degree 1 (counting only itself).
    start_nearby = context.query_nearby_endpoints(start_point, tolerance_m)
    start_segments = set()
    for ep_idx, _dist in start_nearby:
        if ep_idx in context.endpoint_to_segment:
            start_segments.update(context.endpoint_to_segment[ep_idx])
    start_degree = max(1, len(start_segments))

    # Query nearby endpoints at end
    end_nearby = context.query_nearby_endpoints(end_point, tolerance_m)
    end_segments = set()
    for ep_idx, _dist in end_nearby:
        if ep_idx in context.endpoint_to_segment:
            end_segments.update(context.endpoint_to_segment[ep_idx])
    end_degree = max(1, len(end_segments))

    return (start_degree, end_degree)


def compute_endpoint_features(
    target_geom: LineString,
    context: SpatialContextIndex,
    exclude_segment_idx: int | None = None,
    tolerance_m: float = 5.0,
) -> dict[str, float]:
    """Compute endpoint connectivity features for a target segment.

    Uses direction-invariant min/max proximities to avoid sensitivity
    to line digitization direction.

    Args:
        target_geom: Target geometry
        context: SpatialContextIndex with other segments
        exclude_segment_idx: Segment index to exclude (self)
        tolerance_m: Distance threshold for "shared" endpoints (meters)

    Returns:
        Dictionary with:
        - min_endpoint_proximity_m: Minimum of start/end proximities (meters)
        - max_endpoint_proximity_m: Maximum of start/end proximities (meters)
        - shared_endpoint_count: Number of segments with shared endpoints
    """
    if target_geom.is_empty or context.endpoint_coords.size == 0:
        return {
            "min_endpoint_proximity_m": float("inf"),
            "max_endpoint_proximity_m": float("inf"),
            "shared_endpoint_count": 0,
        }

    coords = np.array(target_geom.coords)
    start_point = Point(coords[0])
    end_point = Point(coords[-1])

    # Query nearby endpoints
    start_nearby = context.query_nearby_endpoints(start_point, tolerance_m * 2)
    end_nearby = context.query_nearby_endpoints(end_point, tolerance_m * 2)

    # Filter out endpoints from excluded segment
    if exclude_segment_idx is not None and exclude_segment_idx in context.segment_endpoints:
        excluded_eps = set(context.segment_endpoints[exclude_segment_idx])
        start_nearby = [(ep, d) for ep, d in start_nearby if ep not in excluded_eps]
        end_nearby = [(ep, d) for ep, d in end_nearby if ep not in excluded_eps]

    # Get minimum distances for start and end
    start_proximity = start_nearby[0][1] if start_nearby else float("inf")
    end_proximity = end_nearby[0][1] if end_nearby else float("inf")

    # Direction-invariant: use min/max instead of start/end
    min_proximity = min(start_proximity, end_proximity)
    max_proximity = max(start_proximity, end_proximity)

    # Count shared endpoints (within tolerance)
    shared_segments = set()
    for ep_idx, dist in start_nearby:
        if dist <= tolerance_m and ep_idx in context.endpoint_to_segment:
            shared_segments.update(context.endpoint_to_segment[ep_idx])

    for ep_idx, dist in end_nearby:
        if dist <= tolerance_m and ep_idx in context.endpoint_to_segment:
            shared_segments.update(context.endpoint_to_segment[ep_idx])

    # Remove excluded segment
    if exclude_segment_idx is not None:
        shared_segments.discard(exclude_segment_idx)

    return {
        "min_endpoint_proximity_m": min_proximity,
        "max_endpoint_proximity_m": max_proximity,
        "shared_endpoint_count": len(shared_segments),
    }


def compute_topology_features(
    geom: LineString,
    context: SpatialContextIndex,
    tolerance_m: float = 5.0,
) -> dict[str, float]:
    """Compute topology features for a segment based on inferred connectivity.

    These features capture local network structure without requiring explicit
    topology in the source data. Useful for matching spaghetti line data.

    Args:
        geom: Segment geometry
        context: SpatialContextIndex with all segments
        tolerance_m: Distance threshold for connectivity inference (meters)

    Returns:
        Dictionary with topology features:
        - from_degree: Number of segments connected at start
        - to_degree: Number of segments connected at end
        - is_dead_end: True if either endpoint has degree 1
        - is_intersection: True if either endpoint has degree > 2
        - degree_signature: Tuple of sorted neighbor degrees
    """
    if geom.is_empty or context.endpoint_coords.size == 0:
        return {
            "from_degree": 1,
            "to_degree": 1,
            "is_dead_end": True,
            "is_intersection": False,
            "degree_signature": (1,),
        }

    from_degree, to_degree = infer_endpoint_degree(geom, context, tolerance_m)

    return {
        "from_degree": from_degree,
        "to_degree": to_degree,
        "is_dead_end": min(from_degree, to_degree) == 1,
        "is_intersection": max(from_degree, to_degree) > 2,
        "degree_signature": tuple(sorted([from_degree, to_degree])),
    }


def compute_degree_match_score(
    ref_from_degree: int,
    ref_to_degree: int,
    target_from_degree: int,
    target_to_degree: int,
) -> float:
    """Compute how well endpoint degrees match between reference and target.

    This compares the degrees at the two endpoints of a reference segment
    with those of a target segment, allowing for the possibility that the
    segment direction is reversed (i.e. start/end swapped).

    The score is based on the minimum total absolute difference between
    endpoint degrees across both possible alignments:

        diff_same = |ref_from - target_from| + |ref_to - target_to|
        diff_swap = |ref_from - target_to| + |ref_to - target_from|
        min_diff = min(diff_same, diff_swap)

    This raw difference is then normalized by the sum of all four degrees:

        max_possible = ref_from + ref_to + target_from + target_to
        score = 1.0 - (min_diff / max_possible)

    so that:
    - The score is always in the range [0, 1].
    - Identical degrees give score 1.0 (min_diff = 0).
    - Larger total degree allows a larger absolute difference for the same
      score, since the normalization is relative to the total degree mass.
    - When all degrees are zero, the segments are treated as maximally
      similar and the score is defined as 1.0.

    Args:
        ref_from_degree: Reference segment's start degree.
        ref_to_degree: Reference segment's end degree.
        target_from_degree: Target segment's start degree.
        target_to_degree: Target segment's end degree.

    Returns:
        Similarity score between 0 and 1, where higher values indicate
        more similar local endpoint topology.
    """
    # Try both orderings (endpoints might be reversed)
    diff_same = abs(ref_from_degree - target_from_degree) + abs(ref_to_degree - target_to_degree)
    diff_swap = abs(ref_from_degree - target_to_degree) + abs(ref_to_degree - target_from_degree)
    min_diff = min(diff_same, diff_swap)

    # Normalize by the sum of all endpoint degrees to keep the score in [0, 1]
    max_possible = ref_from_degree + ref_to_degree + target_from_degree + target_to_degree
    if max_possible == 0:
        # If all degrees are zero, treat the segments as maximally similar
        return 1.0

    return 1.0 - (min_diff / max_possible)


class UnionFind:
    """Disjoint set data structure for efficient endpoint clustering.

    Union-Find provides near-constant time operations for:
    - find(): Determine which set an element belongs to
    - union(): Merge two sets together

    This enables O(N log N) total complexity for clustering nearby endpoints,
    compared to O(N²) for naive pairwise approaches.
    """

    def __init__(self, n: int):
        """Initialize Union-Find with n elements.

        Args:
            n: Number of elements (0 to n-1)
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find the root/representative of the set containing x.

        Uses path compression for efficiency.

        Args:
            x: Element index

        Returns:
            Root index of the set containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing x and y.

        Uses union by rank for efficiency.

        Args:
            x: First element index
            y: Second element index
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def _cluster_endpoints_fast(
    endpoint_coords: np.ndarray,
    tolerance: float,
) -> UnionFind:
    """Cluster endpoints using scipy's cKDTree for fast neighbor queries.

    Uses cKDTree.query_pairs() which efficiently finds all pairs of points
    within a given distance using a single optimized call.

    Args:
        endpoint_coords: Nx2 array of endpoint coordinates
        tolerance: Distance within which endpoints are considered same

    Returns:
        UnionFind structure with clustered endpoints
    """
    from scipy.spatial import cKDTree

    n_endpoints = len(endpoint_coords)
    if n_endpoints == 0:
        return UnionFind(0)

    uf = UnionFind(n_endpoints)

    # Build KD-tree and find all pairs within tolerance
    tree = cKDTree(endpoint_coords)
    pairs = tree.query_pairs(r=tolerance, output_type="ndarray")

    # Union all nearby pairs
    for i, j in pairs:
        uf.union(i, j)

    return uf


def compute_all_topology(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    tolerance_m: float = 5.0,
    ids_to_compute: set[str] | None = None,
) -> dict[str, dict]:
    """Compute topology features for all segments using Union-Find clustering.

    This is an O(N log N) batch computation that's much faster than per-segment
    queries for large datasets. Uses Union-Find to cluster nearby endpoints
    without materializing an O(N²) adjacency matrix.

    Algorithm:
        1. Extract endpoints from all segments              O(N)
        2. Build STRtree from endpoints                     O(N log N)
        3. For each endpoint, query nearby within tolerance O(N × log N)
        4. Union nearby endpoints into clusters             O(N × α(N)) ≈ O(N)
        5. Degree = number of unique segments per cluster   O(N)

        Total: O(N log N) time, O(N) space

    Note:
        If the input data is in a geographic CRS (EPSG:4326), it will be
        automatically projected to a local UTM zone for accurate distance
        calculations. The tolerance is always interpreted as meters.

    Args:
        gdf: GeoDataFrame with LineString geometries
        id_column: Column name for segment IDs
        tolerance_m: Distance within which endpoints are considered connected (meters)
        ids_to_compute: If provided, only return topology for these IDs

    Returns:
        Dict mapping segment_id -> topology features dict with:
        - from_degree: Number of segments connected at start
        - to_degree: Number of segments connected at end
        - is_dead_end: True if either endpoint has degree 1
        - is_intersection: True if either endpoint has degree > 2
        - degree_signature: Tuple of sorted [from_degree, to_degree]
    """
    if gdf.empty:
        return {}

    t_start = time.perf_counter()
    n_total = len(gdf)
    logger.info(f"[topology] Starting compute_all_topology for {n_total} segments")

    # Project to local CRS if in geographic coordinates (EPSG:4326)
    # This ensures tolerance is interpreted as meters
    work_gdf = gdf
    if gdf.crs is not None and gdf.crs.is_geographic:
        # Estimate UTM zone from centroid
        t0 = time.perf_counter()
        centroid = gdf.geometry.union_all().centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        work_gdf = gdf.to_crs(epsg=epsg)
        logger.debug(f"[topology] Projected to EPSG:{epsg} in {time.perf_counter() - t0:.2f}s")

    # Step 1: Extract endpoints from all geometries using vectorized shapely
    t0 = time.perf_counter()
    import shapely

    geometries = work_gdf.geometry.values
    segment_ids_arr = work_gdf[id_column].astype(str).values

    # Fast vectorized path for LineStrings (MultiLineStrings filtered at ingest)
    valid_mask = ~shapely.is_empty(geometries) & ~shapely.is_missing(geometries)
    valid_geoms = geometries[valid_mask]
    valid_seg_ids = segment_ids_arr[valid_mask]

    if len(valid_geoms) == 0:
        return {}

    # Get first and last points vectorized
    start_points = shapely.get_point(valid_geoms, 0)
    end_points = shapely.get_point(valid_geoms, -1)

    # Filter out geometries where start or end point is null/empty
    # (can happen for degenerate LineStrings with single point, etc.)
    points_valid = ~shapely.is_missing(start_points) & ~shapely.is_missing(end_points)
    if not points_valid.all():
        n_filtered = (~points_valid).sum()
        logger.debug(f"[topology] Filtered {n_filtered} geometries with invalid start/end points")
        start_points = start_points[points_valid]
        end_points = end_points[points_valid]
        valid_seg_ids = valid_seg_ids[points_valid]

    # Extract coordinates
    start_coords = shapely.get_coordinates(start_points)
    end_coords = shapely.get_coordinates(end_points)

    # Interleave into endpoint arrays
    n_valid = len(start_points)  # Use actual filtered count
    n_endpoints = n_valid * 2

    endpoint_coords = np.empty((n_endpoints, 2), dtype=np.float64)
    endpoint_coords[0::2] = start_coords
    endpoint_coords[1::2] = end_coords

    # Build segment ID and is_start arrays
    endpoint_segment_ids = np.repeat(valid_seg_ids, 2)
    endpoint_is_start = np.tile([True, False], n_valid)
    logger.debug(
        f"[topology] Step 1: Extracted {n_endpoints} endpoints in {time.perf_counter() - t0:.2f}s"
    )

    # Step 2-4: Cluster endpoints using tile-based spatial hashing
    t0 = time.perf_counter()
    endpoint_coords_arr = np.array(endpoint_coords)
    uf = _cluster_endpoints_fast(endpoint_coords_arr, tolerance_m)
    logger.debug(f"[topology] Step 2-4: Tile-based clustering in {time.perf_counter() - t0:.2f}s")

    # Step 5: Build cluster -> set of segment_ids mapping
    cluster_segments: dict[int, set[str]] = {}
    for ep_idx in range(n_endpoints):
        root = uf.find(ep_idx)
        if root not in cluster_segments:
            cluster_segments[root] = set()
        cluster_segments[root].add(endpoint_segment_ids[ep_idx])

    logger.debug(
        f"[topology] Step 5: Built {len(cluster_segments)} clusters in {time.perf_counter() - t0:.2f}s"
    )

    # Step 6: Compute degrees for each segment
    t0 = time.perf_counter()
    # For each segment, find the cluster its start and end belong to
    segment_from_degree: dict[str, int] = {}
    segment_to_degree: dict[str, int] = {}

    for ep_idx in range(n_endpoints):
        seg_id = endpoint_segment_ids[ep_idx]
        is_start = endpoint_is_start[ep_idx]
        root = uf.find(ep_idx)
        degree = len(cluster_segments[root])

        if is_start:
            segment_from_degree[seg_id] = degree
        else:
            segment_to_degree[seg_id] = degree

    # Build final topology dict
    topology = {}
    all_segment_ids = set(endpoint_segment_ids)

    for seg_id in all_segment_ids:
        # Skip if not in requested set
        if ids_to_compute is not None and seg_id not in ids_to_compute:
            continue

        from_degree = segment_from_degree.get(seg_id, 1)
        to_degree = segment_to_degree.get(seg_id, 1)

        topology[seg_id] = {
            "from_degree": from_degree,
            "to_degree": to_degree,
            "is_dead_end": min(from_degree, to_degree) == 1,
            "is_intersection": max(from_degree, to_degree) > 2,
            "degree_signature": tuple(sorted([from_degree, to_degree])),
        }

    # Fill in any missing segments from ids_to_compute with defaults
    if ids_to_compute is not None:
        for seg_id in ids_to_compute:
            if seg_id not in topology:
                topology[seg_id] = {
                    "from_degree": 1,
                    "to_degree": 1,
                    "is_dead_end": True,
                    "is_intersection": False,
                    "degree_signature": (1, 1),
                }

    logger.debug(f"[topology] Step 6: Computed degrees in {time.perf_counter() - t0:.2f}s")
    logger.info(
        f"[topology] Complete: {len(topology)} segments in {time.perf_counter() - t_start:.2f}s total"
    )
    return topology


def compute_degree_signature_similarity(
    sig_a: tuple[int, ...],
    sig_b: tuple[int, ...],
) -> float:
    """Compute Jaccard similarity between degree signatures.

    Degree signatures are sorted tuples of endpoint degrees. Similar local
    topology should have similar signatures.

    Args:
        sig_a: First degree signature
        sig_b: Second degree signature

    Returns:
        Similarity score between 0 and 1
    """
    from collections import Counter

    if not sig_a or not sig_b:
        return 0.0

    counter_a = Counter(sig_a)
    counter_b = Counter(sig_b)

    # Jaccard similarity on multisets
    intersection = sum((counter_a & counter_b).values())
    union = sum((counter_a | counter_b).values())

    return intersection / union if union > 0 else 0.0


def build_inferred_graph(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    tolerance_m: float = 5.0,
) -> tuple["nx.Graph", dict[str, int], dict[str, int]]:
    """Build NetworkX graph from spaghetti geometry using endpoint clustering.

    Uses Union-Find clustering to infer nodes from endpoint proximity.
    Each endpoint cluster becomes a node in the graph, and each segment
    becomes an edge connecting its endpoint clusters.

    This is essential for computing graphlet features on road networks
    where explicit topology is not available.

    Args:
        gdf: GeoDataFrame with LineString geometries
        id_column: Column name for segment IDs
        tolerance_m: Distance within which endpoints are considered connected (meters)

    Returns:
        G: NetworkX graph where nodes=endpoint clusters, edges=segments
        seg_to_start_node: Maps segment ID -> start node cluster ID
        seg_to_end_node: Maps segment ID -> end node cluster ID
    """
    import networkx as nx

    if gdf.empty:
        return nx.Graph(), {}, {}

    t_start = time.perf_counter()
    logger.info(f"[graphlet] Building inferred graph from {len(gdf)} segments")

    # Project to local CRS if in geographic coordinates
    work_gdf = gdf
    if gdf.crs is not None and gdf.crs.is_geographic:
        centroid = gdf.geometry.union_all().centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        work_gdf = gdf.to_crs(epsg=epsg)
        logger.debug(f"[graphlet] Projected to EPSG:{epsg}")

    # Step 1: Extract endpoints using vectorized shapely (LineStrings only, filtered at ingest)
    import shapely

    geometries = work_gdf.geometry.values
    segment_ids_arr = work_gdf[id_column].astype(str).values

    # Fast vectorized path for LineStrings
    valid_mask = ~shapely.is_empty(geometries) & ~shapely.is_missing(geometries)
    valid_geoms = geometries[valid_mask]
    valid_seg_ids = segment_ids_arr[valid_mask]

    if len(valid_geoms) == 0:
        return nx.Graph(), {}, {}

    start_points = shapely.get_point(valid_geoms, 0)
    end_points = shapely.get_point(valid_geoms, -1)
    start_coords = shapely.get_coordinates(start_points)
    end_coords = shapely.get_coordinates(end_points)

    n_valid = len(valid_geoms)
    n_endpoints = n_valid * 2
    endpoint_coords = np.empty((n_endpoints, 2), dtype=np.float64)
    endpoint_coords[0::2] = start_coords
    endpoint_coords[1::2] = end_coords
    endpoint_segment_ids = np.repeat(valid_seg_ids, 2)
    endpoint_is_start = np.tile([True, False], n_valid)

    logger.debug(f"[graphlet] Extracted {n_endpoints} endpoints")

    # Step 2: Cluster endpoints using tile-based spatial hashing
    endpoint_coords_arr = np.array(endpoint_coords)
    uf = _cluster_endpoints_fast(endpoint_coords_arr, tolerance_m)

    # Step 3: Build graph
    G = nx.Graph()
    seg_to_start_node: dict[str, int] = {}
    seg_to_end_node: dict[str, int] = {}

    for ep_idx in range(n_endpoints):
        seg_id = endpoint_segment_ids[ep_idx]
        is_start = endpoint_is_start[ep_idx]
        node_id = uf.find(ep_idx)

        G.add_node(node_id)
        if is_start:
            seg_to_start_node[seg_id] = node_id
        else:
            seg_to_end_node[seg_id] = node_id

    # Add edges (segments connect their endpoint clusters)
    for seg_id in seg_to_start_node:
        start_node = seg_to_start_node[seg_id]
        end_node = seg_to_end_node.get(seg_id)
        if end_node is not None and start_node != end_node:
            G.add_edge(start_node, end_node, segment_id=seg_id)

    logger.info(
        f"[graphlet] Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"in {time.perf_counter() - t_start:.2f}s"
    )

    return G, seg_to_start_node, seg_to_end_node


# Threshold for using numba-accelerated graphlet computation
# Below this, pure Python is fast enough and avoids JIT compilation overhead
_NUMBA_THRESHOLD_NODES = 500


def _build_csr_from_graph(G: "nx.Graph") -> tuple[np.ndarray, np.ndarray, list, dict]:
    """Convert NetworkX graph to CSR format for numba processing.

    Returns:
        indptr: CSR row pointers
        indices: CSR column indices (sorted per row)
        node_list: List of original node IDs
        node_to_idx: Mapping from node ID to integer index
    """
    from scipy import sparse

    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    rows, cols = [], []
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        rows.extend([i, j])
        cols.extend([j, i])

    if rows:
        A = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, n))
        # Ensure indices are sorted (required for merge-based intersection)
        A.sort_indices()
        return A.indptr.astype(np.int64), A.indices.astype(np.int64), node_list, node_to_idx
    else:
        # Empty graph
        return np.zeros(n + 1, dtype=np.int64), np.array([], dtype=np.int64), node_list, node_to_idx


def _get_numba_graphlet_functions():
    """Lazy-load numba-accelerated graphlet functions.

    Returns None if numba is not available.
    """
    try:
        from numba import njit
    except ImportError:
        return None, None

    @njit
    def count_squares_numba(n_nodes, indptr, indices):
        """Count 4-cycles (squares) through each node using CSR adjacency.

        For each node v, count squares v-n1-c-n2-v where n1,n2 are neighbors
        of v and c is a common neighbor of n1,n2 (c != v).

        Uses merge-based intersection of sorted neighbor lists for efficiency.
        """
        result = np.zeros(n_nodes, dtype=np.int64)

        for v in range(n_nodes):
            start_v = indptr[v]
            end_v = indptr[v + 1]
            neighbors_v = indices[start_v:end_v]
            n_neighbors = len(neighbors_v)

            if n_neighbors < 2:
                continue

            square_count = 0
            for i in range(n_neighbors):
                n1 = neighbors_v[i]
                start_n1 = indptr[n1]
                end_n1 = indptr[n1 + 1]
                neighbors_n1 = indices[start_n1:end_n1]

                for j in range(i + 1, n_neighbors):
                    n2 = neighbors_v[j]
                    start_n2 = indptr[n2]
                    end_n2 = indptr[n2 + 1]
                    neighbors_n2 = indices[start_n2:end_n2]

                    # Count common neighbors of n1 and n2, excluding v
                    # Both arrays are sorted (CSR property after sort_indices)
                    p1, p2 = 0, 0
                    common_count = 0
                    while p1 < len(neighbors_n1) and p2 < len(neighbors_n2):
                        if neighbors_n1[p1] == neighbors_n2[p2]:
                            if neighbors_n1[p1] != v:
                                common_count += 1
                            p1 += 1
                            p2 += 1
                        elif neighbors_n1[p1] < neighbors_n2[p2]:
                            p1 += 1
                        else:
                            p2 += 1

                    square_count += common_count

            result[v] = square_count

        return result

    @njit
    def count_two_hop_numba(n_nodes, indptr, indices):
        """Count two-hop neighbors for each node.

        Two-hop neighbors are nodes reachable in exactly 2 hops,
        excluding direct neighbors and the node itself.
        """
        result = np.zeros(n_nodes, dtype=np.int64)

        for v in range(n_nodes):
            start_v = indptr[v]
            end_v = indptr[v + 1]
            neighbors_v = indices[start_v:end_v]

            # Use a seen array to track unique two-hop neighbors
            seen = np.zeros(n_nodes, dtype=np.int8)

            # Mark v and its direct neighbors
            seen[v] = 1
            for ni in neighbors_v:
                seen[ni] = 1

            # Count unique two-hop neighbors
            two_hop_count = 0
            for ni in neighbors_v:
                start_ni = indptr[ni]
                end_ni = indptr[ni + 1]
                for k in range(start_ni, end_ni):
                    nj = indices[k]
                    if seen[nj] == 0:
                        seen[nj] = 1
                        two_hop_count += 1

            result[v] = two_hop_count

        return result

    return count_squares_numba, count_two_hop_numba


def compute_road_graphlet_features(G: "nx.Graph") -> dict[int, np.ndarray]:
    """Compute simplified graphlet features optimized for road networks.

    For large graphs (>500 nodes), uses numba-accelerated computation for
    square counting and two-hop neighbor counting, providing ~10-90x speedup.
    Falls back to pure Python for smaller graphs or if numba is unavailable.

    Returns a 6-dimensional feature vector per node:
    - degree: Number of edges at node (1-4 typical for roads)
    - triangles: Count of 3-cycles through node (rare in roads)
    - squares: Count of 4-cycles through node (common in grid cities)
    - clustering: Local clustering coefficient (low for roads)
    - two_hop_count: Nodes reachable in 2 hops (distinguishes grid vs tree)
    - is_articulation: Whether removal disconnects graph (bridge intersections)

    Args:
        G: NetworkX graph from build_inferred_graph()

    Returns:
        Dictionary mapping node_id -> 6-dimensional numpy array of features
    """
    import networkx as nx

    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return {}

    t_start = time.perf_counter()
    features: dict[int, np.ndarray] = {}

    # Pre-compute graph-wide properties using NetworkX
    triangles = nx.triangles(G)
    clustering = nx.clustering(G)

    # Articulation points only make sense for connected graphs
    try:
        if nx.is_connected(G):
            articulation_points = set(nx.articulation_points(G))
        else:
            # For disconnected graphs, compute articulation points per component
            articulation_points = set()
            for component in nx.connected_components(G):
                subgraph = G.subgraph(component)
                if subgraph.number_of_nodes() > 2:
                    articulation_points.update(nx.articulation_points(subgraph))
    except nx.NetworkXError:
        articulation_points = set()

    # Try numba-accelerated path for large graphs
    use_numba = False
    if n_nodes >= _NUMBA_THRESHOLD_NODES:
        count_squares_numba, count_two_hop_numba = _get_numba_graphlet_functions()
        if count_squares_numba is not None:
            use_numba = True

    if use_numba:
        # Convert graph to CSR format for numba
        indptr, indices, node_list, node_to_idx = _build_csr_from_graph(G)

        # Compute squares and two-hop counts using numba (vectorized over all nodes)
        squares_arr = count_squares_numba(n_nodes, indptr, indices)
        two_hop_arr = count_two_hop_numba(n_nodes, indptr, indices)

        # Build feature dictionary
        for i, node in enumerate(node_list):
            features[node] = np.array(
                [
                    G.degree(node),
                    triangles.get(node, 0),
                    squares_arr[i],
                    clustering.get(node, 0.0),
                    two_hop_arr[i],
                    1.0 if node in articulation_points else 0.0,
                ]
            )

        logger.debug(
            f"[graphlet] Computed features for {len(features)} nodes in "
            f"{time.perf_counter() - t_start:.2f}s (numba-accelerated)"
        )
    else:
        # Pure Python fallback for small graphs or when numba is unavailable
        for node in G.nodes():
            degree = G.degree(node)
            neighbors = set(G.neighbors(node))

            # Count 4-cycles (squares) through this node
            # A square exists if two neighbors share a common neighbor (not this node)
            square_count = 0
            neighbor_list = list(neighbors)
            for i, n1 in enumerate(neighbor_list):
                for n2 in neighbor_list[i + 1 :]:
                    # Check if n1 and n2 share a neighbor other than node
                    common = set(G.neighbors(n1)) & set(G.neighbors(n2)) - {node}
                    square_count += len(common)

            # Two-hop neighbors (excluding direct neighbors and self)
            two_hop = set()
            for neighbor in neighbors:
                two_hop.update(G.neighbors(neighbor))
            two_hop -= neighbors
            two_hop.discard(node)

            features[node] = np.array(
                [
                    degree,
                    triangles.get(node, 0),
                    square_count,
                    clustering.get(node, 0.0),
                    len(two_hop),
                    1.0 if node in articulation_points else 0.0,
                ]
            )

        logger.debug(
            f"[graphlet] Computed features for {len(features)} nodes in "
            f"{time.perf_counter() - t_start:.2f}s"
        )

    return features


def build_connector_graph(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    connectors_column: str = "connectors",
    tolerance_m: float = 5.0,
) -> tuple["nx.Graph", dict[str, list[tuple[float, int]]], dict[int, np.ndarray]]:
    """Build NetworkX graph from Overture segments using explicit connector data.

    Unlike build_inferred_graph() which clusters endpoints, this function uses
    explicit connector positions from Overture data. Each connector becomes a
    node, and segments connect their connectors.

    This enables alignment-aware graphlet similarity: we can lookup the nearest
    connector to any position along a segment.

    Args:
        gdf: GeoDataFrame with Overture segments containing connectors column
        id_column: Column name for segment IDs
        connectors_column: Column name for connectors array (each element has 'at' and 'connector_id')
        tolerance_m: Distance for clustering connectors at same physical location (meters)

    Returns:
        G: NetworkX graph where nodes=connectors, edges=segment connections
        seg_to_connectors: Maps segment ID -> list of (at_position, node_id) sorted by position
        node_features: Dict mapping node_id -> graphlet feature vector
    """
    import networkx as nx

    if gdf.empty:
        return nx.Graph(), {}, {}

    t_start = time.perf_counter()
    logger.info(f"[graphlet-connector] Building connector graph from {len(gdf)} segments")

    # Check if connectors column exists
    if connectors_column not in gdf.columns:
        logger.info(
            f"[graphlet-connector] No '{connectors_column}' column found, "
            "inferring connectivity from spatial proximity"
        )
        # Use the new inference function that detects mid-segment crossings
        return build_inferred_connector_graph(gdf, id_column, tolerance_m)

    # Build mapping from connector_id -> node_id
    # First pass: collect all unique connector IDs
    connector_to_node: dict[str, int] = {}
    node_counter = 0

    segment_ids_arr = gdf[id_column].astype(str).values

    for connectors in gdf[connectors_column].values:
        if connectors is None:
            continue
        for conn in connectors:
            if isinstance(conn, dict):
                conn_id = conn.get("connector_id")
                if conn_id and conn_id not in connector_to_node:
                    connector_to_node[conn_id] = node_counter
                    node_counter += 1

    logger.debug(f"[graphlet-connector] Found {node_counter} unique connectors")

    # Build graph and segment->connectors mapping
    G = nx.Graph()
    seg_to_connectors: dict[str, list[tuple[float, int]]] = {}

    for seg_idx, connectors in enumerate(gdf[connectors_column].values):
        seg_id = segment_ids_arr[seg_idx]
        if connectors is None or len(connectors) == 0:
            seg_to_connectors[seg_id] = []
            continue

        # Extract connector positions for this segment
        segment_connectors = []
        for conn in connectors:
            if isinstance(conn, dict):
                at_pos = conn.get("at", 0.0)
                conn_id = conn.get("connector_id")
                if conn_id and conn_id in connector_to_node:
                    node_id = connector_to_node[conn_id]
                    G.add_node(node_id)
                    segment_connectors.append((at_pos, node_id))

        # Sort by position along segment
        segment_connectors.sort(key=lambda x: x[0])
        seg_to_connectors[seg_id] = segment_connectors

        # Add edges between consecutive connectors on this segment
        for i in range(len(segment_connectors) - 1):
            _, node_a = segment_connectors[i]
            _, node_b = segment_connectors[i + 1]
            if node_a != node_b:
                G.add_edge(node_a, node_b, segment_id=seg_id)

    # Compute graphlet features for all nodes
    t0 = time.perf_counter()
    node_features = compute_road_graphlet_features(G)
    logger.debug(f"[graphlet-connector] Computed features in {time.perf_counter() - t0:.2f}s")

    logger.info(
        f"[graphlet-connector] Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"in {time.perf_counter() - t_start:.2f}s"
    )

    return G, seg_to_connectors, node_features


def _build_inferred_graph_with_features(
    gdf: gpd.GeoDataFrame,
    id_column: str,
    tolerance_m: float,
) -> tuple["nx.Graph", dict[str, int], dict[str, int], dict[int, np.ndarray]]:
    """Helper: build inferred graph and compute features in one step."""
    G, seg_to_start, seg_to_end = build_inferred_graph(gdf, id_column, tolerance_m)
    node_features = compute_road_graphlet_features(G)
    return G, seg_to_start, seg_to_end, node_features


def build_inferred_connector_graph(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    tolerance_m: float = 5.0,
) -> tuple["nx.Graph", dict[str, list[tuple[float, int]]], dict[int, np.ndarray]]:
    """Build connector graph by inferring connectivity from spatial proximity.

    Unlike build_inferred_graph() which only uses endpoints, this function
    detects mid-segment crossings where two segments pass close to each other.
    This creates "virtual connectors" at those positions, enabling alignment-aware
    graphlet comparison for spaghetti geometry.

    Algorithm:
    1. Extract all segment endpoints
    2. Find segment pairs that are spatially close (using STRtree)
    3. For each close pair, find where they're closest and create virtual connectors
    4. Cluster all connection points (endpoints + virtual) using Union-Find
    5. Build graph with all connectors

    Args:
        gdf: GeoDataFrame with LineString geometries
        id_column: Column name for segment IDs
        tolerance_m: Distance within which segments are considered connected (meters)

    Returns:
        G: NetworkX graph where nodes=connection points, edges=segment portions
        seg_to_connectors: Maps segment ID -> list of (at_position, node_id) sorted by position
        node_features: Dict mapping node_id -> graphlet feature vector
    """
    import networkx as nx
    from shapely import STRtree
    from shapely.ops import nearest_points

    if gdf.empty:
        return nx.Graph(), {}, {}

    t_start = time.perf_counter()
    logger.info(f"[graphlet-infer] Inferring connector graph from {len(gdf)} segments")

    # Project to local CRS if in geographic coordinates
    work_gdf = gdf
    if gdf.crs is not None and gdf.crs.is_geographic:
        centroid = gdf.geometry.union_all().centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        work_gdf = gdf.to_crs(epsg=epsg)
        logger.debug(f"[graphlet-infer] Projected to EPSG:{epsg}")

    geometries = work_gdf.geometry.values
    segment_ids = work_gdf[id_column].astype(str).values

    # Collect all connection points: (seg_id, seg_idx, at_position, x, y)
    connection_points = []

    # Step 1: Add endpoints for all segments (LineStrings only, MultiLineStrings filtered at ingest)
    for seg_idx, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        seg_id = segment_ids[seg_idx]

        coords = list(geom.coords)
        start_coords = coords[0]
        end_coords = coords[-1]

        # Start point
        connection_points.append((seg_id, seg_idx, 0.0, start_coords[0], start_coords[1]))
        # End point
        connection_points.append((seg_id, seg_idx, 1.0, end_coords[0], end_coords[1]))

    logger.debug(f"[graphlet-infer] Added {len(connection_points)} endpoints")

    # Step 2: Find mid-segment crossings using spatial index
    tree = STRtree(geometries)
    mid_crossings_added = 0

    for seg_idx, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        seg_id = segment_ids[seg_idx]

        # Query for nearby segments using buffered geometry
        buffered = geom.buffer(tolerance_m)
        nearby_indices = tree.query(buffered)

        for other_idx in nearby_indices:
            if other_idx <= seg_idx:  # Avoid duplicates and self
                continue

            other_geom = geometries[other_idx]
            if other_geom is None or other_geom.is_empty:
                continue

            # Check actual distance between segments
            distance = geom.distance(other_geom)
            if distance > tolerance_m:
                continue

            # Find closest points between the two segments
            p1, p2 = nearest_points(geom, other_geom)

            # Calculate position along each segment (normalized 0-1)
            frac1 = geom.project(p1, normalized=True)
            frac2 = other_geom.project(p2, normalized=True)

            # Only add if NOT at endpoints (those are already covered)
            # Use small threshold to avoid duplicates with endpoints
            if 0.02 < frac1 < 0.98:
                connection_points.append((seg_id, seg_idx, frac1, p1.x, p1.y))
                mid_crossings_added += 1

            if 0.02 < frac2 < 0.98:
                other_seg_id = segment_ids[other_idx]
                connection_points.append((other_seg_id, other_idx, frac2, p2.x, p2.y))
                mid_crossings_added += 1

    logger.debug(f"[graphlet-infer] Added {mid_crossings_added} mid-segment crossings")

    # Step 3: Cluster all connection points using Union-Find
    coords = np.array([(p[3], p[4]) for p in connection_points])
    uf = _cluster_endpoints_fast(coords, tolerance_m)

    # Step 4: Build seg_to_connectors mapping
    # Group connection points by segment
    from collections import defaultdict

    seg_connectors_raw: dict[str, list[tuple[float, int]]] = defaultdict(list)

    for i, (seg_id, _seg_idx, frac, _x, _y) in enumerate(connection_points):
        node_id = uf.find(i)
        seg_connectors_raw[seg_id].append((frac, node_id))

    # Deduplicate and sort connectors for each segment
    seg_to_connectors: dict[str, list[tuple[float, int]]] = {}

    for seg_id, connectors in seg_connectors_raw.items():
        # Sort by position
        connectors.sort(key=lambda x: x[0])

        # Deduplicate: keep only one connector per cluster at similar positions
        unique = []
        for frac, node_id in connectors:
            # Check if we already have this node or a very close position
            if not unique:
                unique.append((frac, node_id))
            else:
                last_frac, last_node = unique[-1]
                # If same node or very close position, skip
                if node_id == last_node or abs(frac - last_frac) < 0.01:
                    continue
                unique.append((frac, node_id))

        seg_to_connectors[seg_id] = unique

    # Step 5: Build graph
    G = nx.Graph()

    # Add all unique nodes
    all_nodes = set()
    for connectors in seg_to_connectors.values():
        for _, node_id in connectors:
            all_nodes.add(node_id)

    for node_id in all_nodes:
        G.add_node(node_id)

    # Add edges between consecutive connectors on each segment
    for seg_id, connectors in seg_to_connectors.items():
        for i in range(len(connectors) - 1):
            _, node_a = connectors[i]
            _, node_b = connectors[i + 1]
            if node_a != node_b:
                G.add_edge(node_a, node_b, segment_id=seg_id)

    # Compute graphlet features
    t0 = time.perf_counter()
    node_features = compute_road_graphlet_features(G)
    logger.debug(f"[graphlet-infer] Computed features in {time.perf_counter() - t0:.2f}s")

    logger.info(
        f"[graphlet-infer] Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"({mid_crossings_added} mid-segment crossings) in {time.perf_counter() - t_start:.2f}s"
    )

    return G, seg_to_connectors, node_features


def find_nearest_connector(
    seg_connectors: list[tuple[float, int]],
    position: float,
) -> int | None:
    """Find the connector node nearest to a given position along a segment.

    Args:
        seg_connectors: List of (at_position, node_id) tuples, sorted by position
        position: Linear position along segment (0.0 to 1.0)

    Returns:
        Node ID of nearest connector, or None if no connectors
    """
    if not seg_connectors:
        return None

    # Binary search for nearest connector
    best_node = None
    best_dist = float("inf")

    for at_pos, node_id in seg_connectors:
        dist = abs(at_pos - position)
        if dist < best_dist:
            best_dist = dist
            best_node = node_id

    return best_node


def get_alignment_connectors(
    seg_id: str,
    seg_to_connectors: dict[str, list[tuple[float, int]]],
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> tuple[int | None, int | None]:
    """Get connector nodes at the endpoints of an aligned subline.

    For partial matches (when alignment trims the segment), this finds the
    connectors nearest to where the alignment starts and ends.

    Args:
        seg_id: Segment ID
        seg_to_connectors: Mapping from segment ID to connector list
        start_frac: Start position of alignment (0.0 to 1.0)
        end_frac: End position of alignment (0.0 to 1.0)

    Returns:
        Tuple of (start_node, end_node) - nearest connectors to alignment endpoints
    """
    connectors = seg_to_connectors.get(seg_id, [])
    if not connectors:
        return None, None

    start_node = find_nearest_connector(connectors, start_frac)
    end_node = find_nearest_connector(connectors, end_frac)

    return start_node, end_node


def graphlet_similarity_with_alignment(
    ref_seg_id: str,
    target_seg_id: str,
    ref_features: dict[int, np.ndarray],
    target_features: dict[int, np.ndarray],
    ref_seg_to_connectors: dict[str, list[tuple[float, int]]],
    target_seg_to_connectors: dict[str, list[tuple[float, int]]],
    ref_start_frac: float = 0.0,
    ref_end_frac: float = 1.0,
    target_start_frac: float = 0.0,
    target_end_frac: float = 1.0,
) -> dict[str, float]:
    """Compare graphlet features at aligned subline endpoints using connector positions.

    This function accounts for alignment by finding the nearest connectors to
    the aligned portion endpoints, rather than always using segment endpoints.

    For example, if alignment indicates the match covers ref[0.3:0.8], we find
    the nearest connectors to positions 0.3 and 0.8, then compare their graphlet
    features to the corresponding target positions.

    Args:
        ref_seg_id: Reference segment ID
        target_seg_id: Target segment ID
        ref_features: Graphlet features for reference graph nodes
        target_features: Graphlet features for target graph nodes
        ref_seg_to_connectors: Maps ref segment ID -> [(at, node_id), ...] sorted by at
        target_seg_to_connectors: Maps target segment ID -> [(at, node_id), ...] sorted by at
        ref_start_frac: Start position of alignment on reference (0.0 to 1.0)
        ref_end_frac: End position of alignment on reference (0.0 to 1.0)
        target_start_frac: Start position of alignment on target (0.0 to 1.0)
        target_end_frac: End position of alignment on target (0.0 to 1.0)

    Returns:
        Dictionary with:
        - graphlet_similarity: Overall endpoint feature similarity (best orientation)
        - endpoint_degree_similarity: Degree match at endpoints (most discriminative)
    """
    # Find nearest connectors to alignment positions
    ref_start_node, ref_end_node = get_alignment_connectors(
        ref_seg_id, ref_seg_to_connectors, ref_start_frac, ref_end_frac
    )
    target_start_node, target_end_node = get_alignment_connectors(
        target_seg_id, target_seg_to_connectors, target_start_frac, target_end_frac
    )

    # Default feature vector: [degree=1, triangles=0, squares=0, clustering=0, two_hop=0, is_articulation=0]
    default = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ref_start_f = (
        ref_features.get(ref_start_node, default) if ref_start_node is not None else default
    )
    ref_end_f = ref_features.get(ref_end_node, default) if ref_end_node is not None else default
    target_start_f = (
        target_features.get(target_start_node, default)
        if target_start_node is not None
        else default
    )
    target_end_f = (
        target_features.get(target_end_node, default) if target_end_node is not None else default
    )

    def feature_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute normalized similarity between two feature vectors."""
        diff = np.abs(a - b)
        norms = np.array([10.0, 10.0, 50.0, 1.0, 50.0, 1.0])
        normalized = 1.0 - np.clip(diff / norms, 0, 1)
        return float(normalized.mean())

    # Try both orientations
    fwd = (
        feature_similarity(ref_start_f, target_start_f)
        + feature_similarity(ref_end_f, target_end_f)
    ) / 2
    rev = (
        feature_similarity(ref_start_f, target_end_f)
        + feature_similarity(ref_end_f, target_start_f)
    ) / 2

    # Also compare degree specifically (most discriminative for roads)
    degree_fwd = (
        1.0
        - abs(ref_start_f[0] - target_start_f[0]) / 10.0
        + 1.0
        - abs(ref_end_f[0] - target_end_f[0]) / 10.0
    ) / 2.0
    degree_fwd = max(0.0, min(1.0, degree_fwd))
    degree_rev = (
        1.0
        - abs(ref_start_f[0] - target_end_f[0]) / 10.0
        + 1.0
        - abs(ref_end_f[0] - target_start_f[0]) / 10.0
    ) / 2.0
    degree_rev = max(0.0, min(1.0, degree_rev))

    return {
        "graphlet_similarity": max(fwd, rev),
        "endpoint_degree_similarity": max(degree_fwd, degree_rev),
    }


def graphlet_segment_similarity(
    ref_seg_id: str,
    target_seg_id: str,
    ref_features: dict[int, np.ndarray],
    target_features: dict[int, np.ndarray],
    ref_seg_to_nodes: tuple[dict[str, int], dict[str, int]],
    target_seg_to_nodes: tuple[dict[str, int], dict[str, int]],
) -> dict[str, float]:
    """Compare graphlet features at segment endpoints.

    Computes similarity between the graphlet features at the endpoints
    of a reference segment and a target segment. Handles segment orientation
    by trying both forward and reverse alignments.

    NOTE: For alignment-aware comparison using connector positions, use
    graphlet_similarity_with_alignment() instead.

    Args:
        ref_seg_id: Reference segment ID
        target_seg_id: Target segment ID
        ref_features: Graphlet features for reference graph nodes
        target_features: Graphlet features for target graph nodes
        ref_seg_to_nodes: Tuple of (start_node_map, end_node_map) for reference
        target_seg_to_nodes: Tuple of (start_node_map, end_node_map) for target

    Returns:
        Dictionary with:
        - graphlet_similarity: Overall endpoint feature similarity (best orientation)
        - endpoint_degree_similarity: Degree match at endpoints (most discriminative)
    """
    # Get node IDs for segment endpoints
    ref_start = ref_seg_to_nodes[0].get(ref_seg_id)
    ref_end = ref_seg_to_nodes[1].get(ref_seg_id)
    target_start = target_seg_to_nodes[0].get(target_seg_id)
    target_end = target_seg_to_nodes[1].get(target_seg_id)

    # Default feature vector: [degree=1, triangles=0, squares=0, clustering=0, two_hop=0, is_articulation=0]
    default = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ref_start_f = ref_features.get(ref_start, default) if ref_start is not None else default
    ref_end_f = ref_features.get(ref_end, default) if ref_end is not None else default
    target_start_f = (
        target_features.get(target_start, default) if target_start is not None else default
    )
    target_end_f = target_features.get(target_end, default) if target_end is not None else default

    def feature_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute normalized similarity between two feature vectors."""
        # Per-feature comparison with appropriate normalization
        # [degree, triangles, squares, clustering, two_hop, is_articulation]
        diff = np.abs(a - b)
        # Normalize each feature to [0, 1] range:
        # - degree: by 10 (typical road intersections have degree 1-4)
        # - triangles: by 10 (rare in roads, small values significant)
        # - squares: by 50 (more common in grid cities)
        # - clustering: already 0-1
        # - two_hop: by 50 (varies by network density)
        # - is_articulation: already 0/1
        norms = np.array([10.0, 10.0, 50.0, 1.0, 50.0, 1.0])
        normalized = 1.0 - np.clip(diff / norms, 0, 1)
        return float(normalized.mean())

    # Try both orientations
    fwd = (
        feature_similarity(ref_start_f, target_start_f)
        + feature_similarity(ref_end_f, target_end_f)
    ) / 2
    rev = (
        feature_similarity(ref_start_f, target_end_f)
        + feature_similarity(ref_end_f, target_start_f)
    ) / 2

    # Also compare degree specifically (most discriminative for roads)
    # Average degree match at both endpoints for consistency with graphlet_similarity
    degree_fwd = (
        1.0
        - abs(ref_start_f[0] - target_start_f[0]) / 10.0
        + 1.0
        - abs(ref_end_f[0] - target_end_f[0]) / 10.0
    ) / 2.0
    degree_fwd = max(0.0, min(1.0, degree_fwd))  # Clamp to [0, 1]
    degree_rev = (
        1.0
        - abs(ref_start_f[0] - target_end_f[0]) / 10.0
        + 1.0
        - abs(ref_end_f[0] - target_start_f[0]) / 10.0
    ) / 2.0
    degree_rev = max(0.0, min(1.0, degree_rev))

    return {
        "graphlet_similarity": max(fwd, rev),
        "endpoint_degree_similarity": max(degree_fwd, degree_rev),
    }
