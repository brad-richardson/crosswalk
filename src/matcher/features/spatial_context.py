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
from shapely import LineString, MultiLineString, Point
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

    offset_consistency: float
    """Standard deviation of perpendicular offset (meters)."""

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
        target_geom: LineString | MultiLineString,
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

            # Compute features
            offset, offset_std = compute_perpendicular_offset(target_geom, road_geom)
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
                    offset_consistency=offset_std,
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
        snap_tolerance: float = 1.0,
    ) -> None:
        """Build spatial indexes from a GeoDataFrame.

        Args:
            gdf: GeoDataFrame with LineString geometries
            id_column: Column name for segment IDs
            snap_tolerance: Distance within which endpoints are considered the same
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

        # Extract all endpoints
        all_endpoints = []
        segment_to_endpoints = {}

        for seg_idx in range(len(gdf)):
            geom = gdf.iloc[seg_idx].geometry
            if geom is None or geom.is_empty:
                continue

            # Handle MultiLineString by getting endpoints of first/last component
            if geom.geom_type == "MultiLineString":
                if len(geom.geoms) == 0:
                    continue
                start = np.array(geom.geoms[0].coords[0])
                end = np.array(geom.geoms[-1].coords[-1])
            else:
                coords = np.array(geom.coords)
                start = coords[0]
                end = coords[-1]

            all_endpoints.append(start)
            all_endpoints.append(end)

            start_ep_idx = len(all_endpoints) - 2
            end_ep_idx = len(all_endpoints) - 1

            segment_to_endpoints[seg_idx] = (start_ep_idx, end_ep_idx)

        if not all_endpoints:
            logger.warning("No endpoints found in GeoDataFrame")
            return

        self.endpoint_coords = np.array(all_endpoints)
        self.segment_endpoints = segment_to_endpoints

        # Cluster nearby endpoints (snap tolerance)
        self._cluster_endpoints(snap_tolerance)

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
            # Build STRtree for spatial queries
            endpoint_points = [Point(c) for c in self.endpoint_coords]
            tree = STRtree(endpoint_points)

            # Initialize Union-Find
            uf = UnionFind(n_endpoints)

            # Union nearby endpoints using dwithin predicate (faster than buffer)
            for ep_idx in range(n_endpoints):
                point = endpoint_points[ep_idx]
                nearby = tree.query(point, predicate="dwithin", distance=tolerance)

                for other_idx in nearby:
                    if ep_idx != other_idx:
                        uf.union(ep_idx, other_idx)

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
        point_coords = np.array(point.coords[0])

        for ep_idx in candidate_indices:
            ep_coords = self.endpoint_coords[ep_idx]
            dist = np.linalg.norm(ep_coords - point_coords)
            if dist <= radius:
                results.append((ep_idx, dist))

        return sorted(results, key=lambda x: x[1])

    def query_nearby_segments(
        self,
        geom: LineString | MultiLineString | Point,
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
        tolerance: float = 5.0,
    ) -> list[int]:
        """Infer segments connected to this one via endpoint proximity.

        Args:
            segment_idx: Index of segment to check
            tolerance: Distance threshold for connectivity

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

            for nearby_ep, _dist in self.query_nearby_endpoints(start_point, tolerance):
                if nearby_ep in self.endpoint_to_segment:
                    connected.update(self.endpoint_to_segment[nearby_ep])

            for nearby_ep, _dist in self.query_nearby_endpoints(end_point, tolerance):
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
    geom: LineString | MultiLineString,
    context: SpatialContextIndex,
    tolerance: float = 5.0,
) -> tuple[int, int]:
    """Infer degree at each endpoint based on nearby segment count.

    For spaghetti data without explicit topology, we infer connectivity
    by counting segments with endpoints within tolerance distance.

    Args:
        geom: Segment geometry
        context: SpatialContextIndex with all segments
        tolerance: Distance threshold for connectivity (meters)

    Returns:
        Tuple of (start_degree, end_degree) where degree = count of segments
        with endpoints within tolerance. Minimum degree is 1 (self).
    """
    if geom.is_empty or context.endpoint_coords.size == 0:
        return (1, 1)

    # Extract endpoints
    if geom.geom_type == "MultiLineString":
        if len(geom.geoms) == 0:
            return (1, 1)
        start_point = Point(geom.geoms[0].coords[0])
        end_point = Point(geom.geoms[-1].coords[-1])
    else:
        coords = np.array(geom.coords)
        start_point = Point(coords[0])
        end_point = Point(coords[-1])

    # Query nearby endpoints at start
    # Note: start_segments will include the segment itself since its own endpoint
    # is within tolerance of itself. This is intentional - degree includes self,
    # so an isolated segment has degree 1 (counting only itself).
    start_nearby = context.query_nearby_endpoints(start_point, tolerance)
    start_segments = set()
    for ep_idx, _dist in start_nearby:
        if ep_idx in context.endpoint_to_segment:
            start_segments.update(context.endpoint_to_segment[ep_idx])
    start_degree = max(1, len(start_segments))

    # Query nearby endpoints at end
    end_nearby = context.query_nearby_endpoints(end_point, tolerance)
    end_segments = set()
    for ep_idx, _dist in end_nearby:
        if ep_idx in context.endpoint_to_segment:
            end_segments.update(context.endpoint_to_segment[ep_idx])
    end_degree = max(1, len(end_segments))

    return (start_degree, end_degree)


def compute_endpoint_features(
    target_geom: LineString | MultiLineString,
    context: SpatialContextIndex,
    exclude_segment_idx: int | None = None,
    tolerance: float = 5.0,
) -> dict[str, float]:
    """Compute endpoint connectivity features for a target segment.

    Args:
        target_geom: Target geometry
        context: SpatialContextIndex with other segments
        exclude_segment_idx: Segment index to exclude (self)
        tolerance: Distance threshold for "shared" endpoints

    Returns:
        Dictionary with:
        - start_endpoint_proximity: Distance to nearest other endpoint from start
        - end_endpoint_proximity: Distance to nearest other endpoint from end
        - shared_endpoint_count: Number of segments with shared endpoints
    """
    if target_geom.is_empty or context.endpoint_coords.size == 0:
        return {
            "start_endpoint_proximity": float("inf"),
            "end_endpoint_proximity": float("inf"),
            "shared_endpoint_count": 0,
        }

    # Handle both LineString and MultiLineString
    if target_geom.geom_type == "MultiLineString":
        if len(target_geom.geoms) == 0:
            return {
                "start_endpoint_proximity": float("inf"),
                "end_endpoint_proximity": float("inf"),
                "shared_endpoint_count": 0,
            }
        start_point = Point(target_geom.geoms[0].coords[0])
        end_point = Point(target_geom.geoms[-1].coords[-1])
    else:
        coords = np.array(target_geom.coords)
        start_point = Point(coords[0])
        end_point = Point(coords[-1])

    # Query nearby endpoints
    start_nearby = context.query_nearby_endpoints(start_point, tolerance * 2)
    end_nearby = context.query_nearby_endpoints(end_point, tolerance * 2)

    # Filter out endpoints from excluded segment
    if exclude_segment_idx is not None and exclude_segment_idx in context.segment_endpoints:
        excluded_eps = set(context.segment_endpoints[exclude_segment_idx])
        start_nearby = [(ep, d) for ep, d in start_nearby if ep not in excluded_eps]
        end_nearby = [(ep, d) for ep, d in end_nearby if ep not in excluded_eps]

    # Get minimum distances
    start_proximity = start_nearby[0][1] if start_nearby else float("inf")
    end_proximity = end_nearby[0][1] if end_nearby else float("inf")

    # Count shared endpoints (within tolerance)
    shared_segments = set()
    for ep_idx, dist in start_nearby:
        if dist <= tolerance and ep_idx in context.endpoint_to_segment:
            shared_segments.update(context.endpoint_to_segment[ep_idx])

    for ep_idx, dist in end_nearby:
        if dist <= tolerance and ep_idx in context.endpoint_to_segment:
            shared_segments.update(context.endpoint_to_segment[ep_idx])

    # Remove excluded segment
    if exclude_segment_idx is not None:
        shared_segments.discard(exclude_segment_idx)

    return {
        "start_endpoint_proximity": start_proximity,
        "end_endpoint_proximity": end_proximity,
        "shared_endpoint_count": len(shared_segments),
    }


def compute_topology_features(
    geom: LineString | MultiLineString,
    context: SpatialContextIndex,
    tolerance: float = 5.0,
) -> dict[str, float]:
    """Compute topology features for a segment based on inferred connectivity.

    These features capture local network structure without requiring explicit
    topology in the source data. Useful for matching spaghetti line data.

    Args:
        geom: Segment geometry
        context: SpatialContextIndex with all segments
        tolerance: Distance threshold for connectivity inference

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

    from_degree, to_degree = infer_endpoint_degree(geom, context, tolerance)

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


def compute_all_topology(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    tolerance: float = 5.0,
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
        tolerance: Distance within which endpoints are considered connected (meters)
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

    # Step 1: Extract endpoints from all geometries
    # Each entry: (endpoint_coords, segment_id, is_start)
    t0 = time.perf_counter()
    endpoint_coords = []
    endpoint_segment_ids = []
    endpoint_is_start = []

    for _, row in work_gdf.iterrows():
        seg_id = str(row[id_column])
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        # Handle both LineString and MultiLineString
        if geom.geom_type == "MultiLineString":
            if len(geom.geoms) == 0:
                continue
            start_coords = geom.geoms[0].coords[0]
            end_coords = geom.geoms[-1].coords[-1]
        else:
            coords = list(geom.coords)
            start_coords = coords[0]
            end_coords = coords[-1]

        # Add start endpoint
        endpoint_coords.append(start_coords)
        endpoint_segment_ids.append(seg_id)
        endpoint_is_start.append(True)

        # Add end endpoint
        endpoint_coords.append(end_coords)
        endpoint_segment_ids.append(seg_id)
        endpoint_is_start.append(False)

    if not endpoint_coords:
        return {}

    n_endpoints = len(endpoint_coords)
    logger.debug(
        f"[topology] Step 1: Extracted {n_endpoints} endpoints in {time.perf_counter() - t0:.2f}s"
    )

    # Step 2: Build STRtree from endpoint points
    t0 = time.perf_counter()
    endpoint_points = [Point(c) for c in endpoint_coords]
    tree = STRtree(endpoint_points)
    logger.debug(f"[topology] Step 2: Built STRtree in {time.perf_counter() - t0:.2f}s")

    # Step 3 & 4: For each endpoint, query nearby and union into clusters
    t0 = time.perf_counter()
    uf = UnionFind(n_endpoints)

    for i, point in enumerate(endpoint_points):
        # Query all endpoints within tolerance using dwithin predicate (faster than buffer)
        nearby_indices = tree.query(point, predicate="dwithin", distance=tolerance)

        # Union this endpoint with all nearby endpoints
        for j in nearby_indices:
            if i != j:
                uf.union(i, j)

    logger.debug(f"[topology] Step 3-4: Union-Find clustering in {time.perf_counter() - t0:.2f}s")

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
    tolerance: float = 5.0,
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
        tolerance: Distance within which endpoints are considered connected (meters)

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

    # Step 1: Extract endpoints
    endpoint_coords = []
    endpoint_segment_ids = []
    endpoint_is_start = []

    for _, row in work_gdf.iterrows():
        seg_id = str(row[id_column])
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "MultiLineString":
            if len(geom.geoms) == 0:
                continue
            start_coords = geom.geoms[0].coords[0]
            end_coords = geom.geoms[-1].coords[-1]
        else:
            coords = list(geom.coords)
            start_coords = coords[0]
            end_coords = coords[-1]

        endpoint_coords.append(start_coords)
        endpoint_segment_ids.append(seg_id)
        endpoint_is_start.append(True)

        endpoint_coords.append(end_coords)
        endpoint_segment_ids.append(seg_id)
        endpoint_is_start.append(False)

    if not endpoint_coords:
        return nx.Graph(), {}, {}

    n_endpoints = len(endpoint_coords)
    logger.debug(f"[graphlet] Extracted {n_endpoints} endpoints")

    # Step 2: Build STRtree and cluster with Union-Find
    endpoint_points = [Point(c) for c in endpoint_coords]
    tree = STRtree(endpoint_points)
    uf = UnionFind(n_endpoints)

    for i, point in enumerate(endpoint_points):
        nearby = tree.query(point, predicate="dwithin", distance=tolerance)
        for j in nearby:
            if i != j:
                uf.union(i, j)

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


def compute_road_graphlet_features(G: "nx.Graph") -> dict[int, np.ndarray]:
    """Compute simplified graphlet features optimized for road networks.

    Uses pure NetworkX - no external dependencies.
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

    if G.number_of_nodes() == 0:
        return {}

    t_start = time.perf_counter()
    features: dict[int, np.ndarray] = {}

    # Pre-compute graph-wide properties
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
        f"[graphlet] Computed features for {len(features)} nodes in {time.perf_counter() - t_start:.2f}s"
    )

    return features


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
