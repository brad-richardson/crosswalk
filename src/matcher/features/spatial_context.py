"""Spatial context indexing for relational feature computation.

This module provides spatial indexes and utilities for:
1. Finding anchor roads for parallel infrastructure (sidewalks, bike lanes)
2. Inferring endpoint connectivity from proximity
3. Supporting context propagation across nearby segments

These features work without requiring explicit topology in the target data,
making them suitable for raw "spaghetti" line datasets.
"""

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely import LineString, MultiLineString, Point
from shapely.strtree import STRtree

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

        Args:
            tolerance: Distance within which endpoints are considered the same
        """
        # Simple approach: for each endpoint, find segments that share it
        self.endpoint_to_segment = {}

        for seg_idx, (start_ep, end_ep) in self.segment_endpoints.items():
            # Add to start endpoint's segment list
            if start_ep not in self.endpoint_to_segment:
                self.endpoint_to_segment[start_ep] = []
            self.endpoint_to_segment[start_ep].append(seg_idx)

            # Add to end endpoint's segment list
            if end_ep not in self.endpoint_to_segment:
                self.endpoint_to_segment[end_ep] = []
            self.endpoint_to_segment[end_ep].append(seg_idx)

        # Now cluster: for nearby endpoints, merge their segment lists
        if tolerance > 0 and len(self.endpoint_coords) > 0:
            tree = STRtree([Point(c) for c in self.endpoint_coords])

            for ep_idx in range(len(self.endpoint_coords)):
                point = Point(self.endpoint_coords[ep_idx])
                nearby = tree.query(point.buffer(tolerance))

                # Merge segment lists from nearby endpoints
                merged_segments = set()
                for other_idx in nearby:
                    if other_idx in self.endpoint_to_segment:
                        merged_segments.update(self.endpoint_to_segment[other_idx])

                self.endpoint_to_segment[ep_idx] = list(merged_segments)

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

    Handles the ambiguity that endpoints might be swapped (reversed direction).
    Higher score means more similar local topology.

    Args:
        ref_from_degree: Reference segment's start degree
        ref_to_degree: Reference segment's end degree
        target_from_degree: Target segment's start degree
        target_to_degree: Target segment's end degree

    Returns:
        Similarity score between 0 and 1
    """
    # Try both orderings (endpoints might be reversed)
    diff_same = abs(ref_from_degree - target_from_degree) + abs(ref_to_degree - target_to_degree)
    diff_swap = abs(ref_from_degree - target_to_degree) + abs(ref_to_degree - target_from_degree)
    min_diff = min(diff_same, diff_swap)

    # Normalize: max possible diff is the sum of all degrees
    max_possible = ref_from_degree + ref_to_degree + target_from_degree + target_to_degree
    if max_possible == 0:
        return 1.0

    return 1.0 - (min_diff / max_possible)


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
