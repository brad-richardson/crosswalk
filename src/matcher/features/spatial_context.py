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
    from pyproj import Transformer

    from matcher.topology.sparse_graph import SparseGraph

from matcher.config import DEFAULT_SNAP_TOLERANCE_M, MAX_DISTANCE_METERS

from ._jit_helpers import query_nearby_endpoints_numba
from .relational import (
    compute_parallel_alignment,
    compute_perpendicular_offset,
)


@dataclass
class TopologySpatialIndex:
    """Spatial index of topology cluster centroids for querying degree at arbitrary positions.

    Built during compute_all_topology() from the Union-Find clustering of all
    segment endpoints. Enables sampling topology degrees at positions along
    segments (e.g., for target-side aligned topology via synthetic connectors).

    Attributes:
        centroids: Nx2 array of cluster centroid coordinates (in projected CRS)
        degrees: 1D array of degree per cluster (same order as centroids)
        tree: STRtree built from centroid Point geometries for spatial queries
    """

    centroids: np.ndarray
    degrees: np.ndarray
    tree: "STRtree"


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
        >>> print(f"Anchor road: {match.anchor_id}, offset: {match.perpendicular_offset:.1f}m")
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

            # Skip if offset too large or not parallel enough
            if offset > self.max_offset or alignment < self.min_alignment:
                continue

            # Score: prefer low offset and high alignment
            # Normalize offset to 0-1 (lower is better)
            offset_score = max(0, 1 - offset / self.max_offset)
            score = 0.5 * offset_score + 0.5 * alignment

            if score > best_score:
                best_score = score
                best_match = AnchorMatch(
                    anchor_idx=road_idx,
                    anchor_id=self.road_ids[road_idx],
                    perpendicular_offset=offset,
                    offset_iqr=offset_iqr,
                    offset_p95=offset_p95,
                    parallel_alignment=alignment,
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
    """Array of shape (N, 2) with all endpoint coordinates (in projected CRS if geographic)."""

    endpoint_to_segment: dict[int, list[int]] = field(default_factory=dict)
    """Map from endpoint index to list of segment indices that share it."""

    segment_endpoints: dict[int, tuple[int, int]] = field(default_factory=dict)
    """Map from segment index to (start_endpoint_idx, end_endpoint_idx)."""

    segment_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of segment IDs corresponding to indices."""

    _endpoint_tree: STRtree | None = field(default=None, repr=False)
    _segment_tree: STRtree | None = field(default=None, repr=False)
    _geometries: gpd.GeoSeries | None = field(default=None, repr=False)
    _transformer: "Transformer | None" = field(default=None, repr=False)
    """Transformer to project query points from source CRS to endpoint CRS."""
    _kdtree: object = field(default=None, repr=False)
    """Cached scipy cKDTree for batch radius queries."""

    @property
    def kdtree(self):
        """Lazily build and cache a scipy cKDTree from endpoint coordinates."""
        if self._kdtree is None and self.endpoint_coords.size > 0:
            from scipy.spatial import cKDTree

            self._kdtree = cKDTree(self.endpoint_coords)
        return self._kdtree

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
        self._transformer = None
        if gdf.crs is not None and gdf.crs.is_geographic:
            from pyproj import Transformer

            # Use geopandas' estimate_utm_crs() for consistent UTM zone selection
            utm_crs = gdf.estimate_utm_crs()
            work_gdf = gdf.to_crs(utm_crs)
            # Store transformer to project query points
            self._transformer = Transformer.from_crs(gdf.crs, utm_crs, always_xy=True)

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
        already_projected: bool = False,
    ) -> list[tuple[int, float]]:
        """Find endpoints within radius of a point.

        Args:
            point: Query point (in source CRS, typically WGS84)
            radius: Search radius (meters)
            already_projected: If True, skip projection (point already in index CRS)

        Returns:
            List of (endpoint_idx, distance) tuples sorted by distance
        """
        if self._endpoint_tree is None:
            return []

        # Project query point if we have a transformer (geographic -> projected)
        # Skip projection if caller indicates point is already in index CRS
        query_point = point
        if self._transformer is not None and not already_projected:
            x, y = point.coords[0][:2]
            px, py = self._transformer.transform(x, y)
            query_point = Point(px, py)

        buffered = query_point.buffer(radius)
        candidate_indices = self._endpoint_tree.query(buffered)

        if len(candidate_indices) == 0:
            return []

        # Handle 3D coordinates by taking only x, y (first 2 dimensions)
        point_coords = np.array(query_point.coords[0])[:2]

        # Convert to numpy array for JIT function (STRtree returns numpy array)
        candidate_indices_arr = np.asarray(candidate_indices, dtype=np.int64)

        # Use JIT-compiled distance filtering (5-10x faster for large candidate sets)
        result_indices, result_dists = query_nearby_endpoints_numba(
            self.endpoint_coords, candidate_indices_arr, point_coords, radius
        )

        # Convert to list of tuples and sort by distance
        results = [
            (int(result_indices[i]), float(result_dists[i])) for i in range(len(result_indices))
        ]
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
        tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
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

            # Points from endpoint_coords are already projected
            for nearby_ep, _dist in self.query_nearby_endpoints(
                start_point, tolerance_m, already_projected=True
            ):
                if nearby_ep in self.endpoint_to_segment:
                    connected.update(self.endpoint_to_segment[nearby_ep])

            for nearby_ep, _dist in self.query_nearby_endpoints(
                end_point, tolerance_m, already_projected=True
            ):
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
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
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
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
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
        # Use MAX_DISTANCE_METERS instead of float("inf") for consistency
        # with _get_error_features() and ml.py fallback defaults
        return {
            "min_endpoint_proximity_m": MAX_DISTANCE_METERS,
            "max_endpoint_proximity_m": MAX_DISTANCE_METERS,
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


def compute_aligned_endpoint_features(
    geom: LineString,
    context: SpatialContextIndex,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    exclude_segment_idx: int | None = None,
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    seg_id: str | None = None,
    seg_to_connectors: dict[str, list[tuple[float, int]]] | None = None,
) -> dict[str, float]:
    """Compute endpoint proximity at aligned portion endpoints.

    Instead of using the full geometry endpoints, this function uses the
    coordinates at the alignment boundaries (start_frac, end_frac). This
    is critical for partial overlaps where the segments only partially match.

    For example, if a 438m segment only overlaps with a 186m segment at
    [0%, 43%], we want to measure endpoint proximity at the 43% position,
    not at the 100% position (the actual segment end).

    Delegates to compute_aligned_endpoint_features_batch with a single pair.

    Args:
        geom: Full segment geometry
        context: SpatialContextIndex with all segments
        start_frac: Start of aligned region (0.0 to 1.0)
        end_frac: End of aligned region (0.0 to 1.0)
        exclude_segment_idx: Segment index to exclude (self)
        tolerance_m: Distance threshold for "shared" endpoints (meters)
        seg_id: Segment ID for connector lookup
        seg_to_connectors: Optional connector data for snapping fractions

    Returns:
        Dictionary with:
        - min_endpoint_proximity_m: Minimum of start/end proximities (meters)
        - max_endpoint_proximity_m: Maximum of start/end proximities (meters)
        - shared_endpoint_count: Number of segments with shared endpoints
    """
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class _SingleAlignment:
        dataset_start_frac: float
        dataset_end_frac: float

    default_result = {
        "min_endpoint_proximity_m": MAX_DISTANCE_METERS,
        "max_endpoint_proximity_m": MAX_DISTANCE_METERS,
        "shared_endpoint_count": 0,
    }

    if geom is None or geom.is_empty:
        return default_result

    alignments = {(0, 0): _SingleAlignment(start_frac, end_frac)}
    target_geoms = np.array([geom])
    target_ids = np.array([seg_id or ""])
    original_to_filtered = {0: exclude_segment_idx} if exclude_segment_idx is not None else {}

    result = compute_aligned_endpoint_features_batch(
        alignments=alignments,
        target_geoms=target_geoms,
        target_ids=target_ids,
        target_index=context,
        original_to_filtered=original_to_filtered,
        seg_to_connectors=seg_to_connectors,
    )

    return result.get((0, 0), default_result)


def compute_aligned_endpoint_features_batch(
    alignments: dict,
    target_geoms: "np.ndarray",
    target_ids: "np.ndarray",
    target_index: "SpatialContextIndex",
    original_to_filtered: dict,
    seg_to_connectors: dict | None = None,
) -> dict[tuple[int, int], dict[str, float]]:
    """Batch-compute aligned endpoint features for all alignment pairs.

    Shared implementation used by both the ML scoring path and the labeling
    data loader path to avoid code duplication.

    Uses vectorized Shapely interpolation and scipy cKDTree batch queries
    instead of per-pair spatial index lookups for significantly better
    performance on large candidate sets.

    Args:
        alignments: Dict mapping (ref_idx, target_idx) -> AlignmentResult
        target_geoms: Array of target geometries indexed by position
        target_ids: Array of target IDs indexed by position
        target_index: SpatialContextIndex built from target segments
        original_to_filtered: Dict mapping original index -> filtered index
        seg_to_connectors: Optional connector data for snapping fractions

    Returns:
        Dict mapping (ref_idx, target_idx) -> endpoint feature dict
    """
    if not alignments:
        return {}

    import shapely

    tolerance_m = DEFAULT_SNAP_TOLERANCE_M
    radius = tolerance_m * 2  # Same search radius as compute_aligned_endpoint_features

    # 1. Extract alignment data into arrays
    keys = list(alignments.keys())
    n = len(keys)
    target_indices = np.empty(n, dtype=np.intp)
    start_fracs = np.empty(n, dtype=np.float64)
    end_fracs = np.empty(n, dtype=np.float64)

    for i, key in enumerate(keys):
        target_indices[i] = key[1]
        alignment = alignments[key]
        start_fracs[i] = alignment.dataset_start_frac
        end_fracs[i] = alignment.dataset_end_frac

    # Clamp fractions to [0.0, 1.0]
    np.clip(start_fracs, 0.0, 1.0, out=start_fracs)
    np.clip(end_fracs, 0.0, 1.0, out=end_fracs)

    # Connector snapping (per-pair but cheap — just float comparisons)
    if seg_to_connectors is not None:
        for i in range(n):
            sid = str(target_ids[target_indices[i]])
            connectors = seg_to_connectors.get(sid)
            if connectors:
                snapped = find_nearest_connector_position(connectors, start_fracs[i])
                if snapped is not None:
                    start_fracs[i] = snapped
                snapped = find_nearest_connector_position(connectors, end_fracs[i])
                if snapped is not None:
                    end_fracs[i] = snapped

    # 2. Filter to valid geometries
    geoms = target_geoms[target_indices]
    valid_mask = np.array(
        [g is not None and not shapely.is_missing(g) and not g.is_empty for g in geoms]
    )

    if not valid_mask.any():
        return {}

    valid_indices = np.where(valid_mask)[0]
    valid_geoms = geoms[valid_mask]
    valid_start_fracs = start_fracs[valid_mask]
    valid_end_fracs = end_fracs[valid_mask]

    # 3. Vectorized interpolation (single C call for all pairs)
    start_points = shapely.line_interpolate_point(valid_geoms, valid_start_fracs, normalized=True)
    end_points = shapely.line_interpolate_point(valid_geoms, valid_end_fracs, normalized=True)

    start_coords = shapely.get_coordinates(start_points)
    end_coords = shapely.get_coordinates(end_points)

    # 4. Get cKDTree from endpoint coordinates for batch queries (cached on index)
    endpoint_coords = target_index.endpoint_coords
    if endpoint_coords.size == 0:
        # No endpoints: return defaults for all valid pairs
        default = {
            "min_endpoint_proximity_m": float(MAX_DISTANCE_METERS),
            "max_endpoint_proximity_m": float(MAX_DISTANCE_METERS),
            "shared_endpoint_count": 0,
        }
        return {keys[valid_indices[j]]: default.copy() for j in range(len(valid_indices))}

    tree = target_index.kdtree

    # 5. Batch radius queries (single C call each, replaces 2N individual STRtree queries)
    start_neighbors_list = tree.query_ball_point(start_coords, r=radius)
    end_neighbors_list = tree.query_ball_point(end_coords, r=radius)

    # 6. Process results — compute distances and shared counts per pair
    endpoint_to_segment = target_index.endpoint_to_segment
    segment_endpoints = target_index.segment_endpoints

    result = {}
    for j in range(len(valid_indices)):
        i = valid_indices[j]  # Index into original keys array
        key = keys[i]
        target_idx = target_indices[i]
        filtered_idx = original_to_filtered.get(int(target_idx))

        # Get excluded endpoints for this segment
        excluded_eps = None
        if filtered_idx is not None and filtered_idx in segment_endpoints:
            excluded_eps = segment_endpoints[filtered_idx]

        s_x, s_y = start_coords[j, 0], start_coords[j, 1]
        e_x, e_y = end_coords[j, 0], end_coords[j, 1]

        start_min_dist = float("inf")
        end_min_dist = float("inf")
        shared_segments = set()

        # Process start-point neighbors
        for ep_idx in start_neighbors_list[j]:
            if excluded_eps is not None and (
                ep_idx == excluded_eps[0] or ep_idx == excluded_eps[1]
            ):
                continue
            dx = endpoint_coords[ep_idx, 0] - s_x
            dy = endpoint_coords[ep_idx, 1] - s_y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < start_min_dist:
                start_min_dist = dist
            if dist <= tolerance_m and ep_idx in endpoint_to_segment:
                shared_segments.update(endpoint_to_segment[ep_idx])

        # Process end-point neighbors
        for ep_idx in end_neighbors_list[j]:
            if excluded_eps is not None and (
                ep_idx == excluded_eps[0] or ep_idx == excluded_eps[1]
            ):
                continue
            dx = endpoint_coords[ep_idx, 0] - e_x
            dy = endpoint_coords[ep_idx, 1] - e_y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < end_min_dist:
                end_min_dist = dist
            if dist <= tolerance_m and ep_idx in endpoint_to_segment:
                shared_segments.update(endpoint_to_segment[ep_idx])

        if filtered_idx is not None:
            shared_segments.discard(filtered_idx)

        result[key] = {
            "min_endpoint_proximity_m": min(start_min_dist, end_min_dist),
            "max_endpoint_proximity_m": max(start_min_dist, end_min_dist),
            "shared_endpoint_count": len(shared_segments),
        }

    return result


def compute_topology_features(
    geom: LineString,
    context: SpatialContextIndex,
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
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


def compute_all_topology_explicit(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    connectors_column: str = "connectors",
) -> dict[str, dict] | None:
    """Compute topology from explicit connector data (Overture/OSM style).

    This function uses explicit connector IDs rather than inferring connectivity
    from geometry proximity. It computes endpoint degrees by counting how many
    segments share each connector.

    The degree at each endpoint is the number of segments that reference that
    connector at any position (not just endpoints). This means:
    - A segment's start connector shared by 2 other segments → from_degree=3
    - A segment's end connector used only by that segment → to_degree=1

    Args:
        gdf: GeoDataFrame with LineString geometries and connectors column
        id_column: Column name for segment IDs
        connectors_column: Column name for connectors array
            (each element should have 'at' and 'connector_id' keys)

    Returns:
        Dict mapping segment_id -> topology features dict, or None if
        connectors are not available (signals caller should fall back to
        geometry-based inference).

    Example connector format:
        [
            {"at": 0.0, "connector_id": "conn_a"},  # start
            {"at": 0.5, "connector_id": "conn_b"},  # mid-segment
            {"at": 1.0, "connector_id": "conn_c"},  # end
        ]
    """
    # Check if connectors column exists and has data
    if connectors_column not in gdf.columns:
        return None

    if gdf[connectors_column].isna().all():
        return None

    t_start = time.perf_counter()
    logger.info(f"[topology-explicit] Computing topology from {connectors_column} column")

    # Step 1: Build connector_id -> set of segment_ids mapping
    # This counts how many segments reference each connector
    connector_segments: dict[str, set[str]] = {}
    segment_ids = gdf[id_column].astype(str).values

    for seg_idx, connectors in enumerate(gdf[connectors_column].values):
        if connectors is None:
            continue

        seg_id = segment_ids[seg_idx]
        for conn in connectors:
            if isinstance(conn, dict):
                conn_id = conn.get("connector_id")
                if conn_id:
                    if conn_id not in connector_segments:
                        connector_segments[conn_id] = set()
                    connector_segments[conn_id].add(seg_id)

    logger.debug(f"[topology-explicit] Found {len(connector_segments)} unique connectors")

    # Step 2: For each segment, compute degrees from ALL connectors (not just endpoints)
    # This properly handles mid-segment intersections
    topology = {}

    for seg_idx, connectors in enumerate(gdf[connectors_column].values):
        seg_id = segment_ids[seg_idx]

        if connectors is None or len(connectors) == 0:
            # No connector data - use default (isolated segment)
            topology[seg_id] = {
                "from_degree": 1,
                "to_degree": 1,
                "is_dead_end": True,
                "is_intersection": False,
                "degree_signature": (1, 1),
            }
            continue

        # Collect degrees for ALL connectors, tracking endpoint vs midpoint
        from_degree = 1  # Default for start
        to_degree = 1  # Default for end
        all_degrees = []

        for conn in connectors:
            if isinstance(conn, dict):
                at_pos = conn.get("at", -1)
                conn_id = conn.get("connector_id")
                if conn_id:
                    degree = len(connector_segments.get(conn_id, set()))
                    degree = max(1, degree)  # Ensure minimum of 1
                    all_degrees.append(degree)

                    # Track endpoint degrees specifically
                    # Use small epsilon for floating point comparison
                    if at_pos <= 0.001:  # Start (at ~0.0)
                        from_degree = degree
                    elif at_pos >= 0.999:  # End (at ~1.0)
                        to_degree = degree

        # Use max degree across ALL connectors for intersection detection
        # (mid-segment junctions count as intersections)
        max_degree = max(all_degrees) if all_degrees else 1

        # Dead-end is based on ENDPOINT degrees only, not mid-segment connectors
        # A segment is a dead-end if either endpoint has degree 1
        topology[seg_id] = {
            "from_degree": from_degree,
            "to_degree": to_degree,
            "is_dead_end": from_degree == 1 or to_degree == 1,
            "is_intersection": max_degree > 2,  # True if ANY connector has degree > 2
            "degree_signature": tuple(sorted([from_degree, to_degree])),
        }

    logger.info(
        f"[topology-explicit] Computed topology for {len(topology)} segments "
        f"in {time.perf_counter() - t_start:.2f}s"
    )
    return topology


def compute_all_topology(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    ids_to_compute: set[str] | None = None,
    connectors_column: str | None = None,
    return_spatial_index: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], "TopologySpatialIndex"]:
    """Compute topology features for all segments.

    When connectors_column is provided and the data contains explicit connector
    information (Overture/OSM style), topology is computed from connector sharing.
    Otherwise, falls back to Union-Find clustering based on endpoint proximity.

    The explicit connector approach is preferred when available because:
    - It uses authoritative topology from the source data
    - It's O(N) time complexity vs O(N log N) for geometry inference
    - It handles complex intersections more accurately

    Geometry-based inference algorithm (fallback):
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
        connectors_column: Column name for explicit connector data. If provided and
            data is available, uses explicit topology instead of geometry inference.
        return_spatial_index: If True, also return a TopologySpatialIndex for querying
            degree at arbitrary positions (used for synthetic connector sampling).
            Only supported for geometry-inferred topology (not explicit connectors).

    Returns:
        If return_spatial_index is False:
            Dict mapping segment_id -> topology features dict with:
            - from_degree: Number of segments connected at start
            - to_degree: Number of segments connected at end
            - is_dead_end: True if either endpoint has degree 1
            - is_intersection: True if either endpoint has degree > 2
            - degree_signature: Tuple of sorted [from_degree, to_degree]
        If return_spatial_index is True:
            Tuple of (topology_dict, TopologySpatialIndex)
    """
    # Explicit connectors and spatial index are mutually exclusive —
    # the spatial index is built from geometry-inferred clustering, not connectors.
    if connectors_column is not None and return_spatial_index:
        raise ValueError(
            "return_spatial_index=True is not supported with connectors_column. "
            "The spatial index requires geometry-inferred clustering."
        )

    # Try explicit connector-based topology first (if available)
    if connectors_column is not None:
        explicit_result = compute_all_topology_explicit(gdf, id_column, connectors_column)
        if explicit_result is not None:
            # Filter to requested IDs if specified
            if ids_to_compute is not None:
                explicit_result = {k: v for k, v in explicit_result.items() if k in ids_to_compute}
                # Fill in any missing IDs with defaults
                for seg_id in ids_to_compute:
                    if seg_id not in explicit_result:
                        explicit_result[seg_id] = {
                            "from_degree": 1,
                            "to_degree": 1,
                            "is_dead_end": True,
                            "is_intersection": False,
                            "degree_signature": (1, 1),
                        }
            return explicit_result
        # Fall through to geometry inference if explicit failed
        logger.debug(
            f"[topology] Explicit connectors not available in '{connectors_column}', "
            "falling back to geometry inference"
        )
    if gdf.empty:
        return {}

    t_start = time.perf_counter()
    n_total = len(gdf)
    logger.info(f"[topology] Starting compute_all_topology for {n_total} segments")

    # Project to local CRS if in geographic coordinates (EPSG:4326)
    # This ensures tolerance is interpreted as meters
    work_gdf = gdf
    if gdf.crs is not None and gdf.crs.is_geographic:
        # Use geopandas' estimate_utm_crs() for consistent UTM zone selection
        t0 = time.perf_counter()
        utm_crs = gdf.estimate_utm_crs()
        work_gdf = gdf.to_crs(utm_crs)
        logger.debug(f"[topology] Projected to {utm_crs} in {time.perf_counter() - t0:.2f}s")

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

    # Step 5b: Build spatial index of cluster centroids (if requested)
    topology_spatial_index = None
    if return_spatial_index:
        t0_si = time.perf_counter()
        # Pre-build cluster membership in O(N) — avoids O(N*C) nested scan
        from collections import defaultdict

        import shapely

        cluster_members: dict[int, list[int]] = defaultdict(list)
        for ep_idx in range(n_endpoints):
            cluster_members[uf.find(ep_idx)].append(ep_idx)

        # Compute centroid and degree for each cluster
        cluster_ids = sorted(cluster_segments.keys())
        n_clusters = len(cluster_ids)
        cluster_centroids = np.empty((n_clusters, 2), dtype=np.float64)
        cluster_degrees = np.empty(n_clusters, dtype=np.int32)

        for ci, root in enumerate(cluster_ids):
            member_coords = endpoint_coords_arr[cluster_members[root]]
            cluster_centroids[ci] = member_coords.mean(axis=0)
            cluster_degrees[ci] = len(cluster_segments[root])

        # Build STRtree from centroid points
        centroid_points = shapely.points(cluster_centroids[:, 0], cluster_centroids[:, 1])
        tree = STRtree(centroid_points)
        topology_spatial_index = TopologySpatialIndex(
            centroids=cluster_centroids,
            degrees=cluster_degrees,
            tree=tree,
        )
        logger.debug(
            f"[topology] Step 5b: Built spatial index with {n_clusters} clusters "
            f"in {time.perf_counter() - t0_si:.2f}s"
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
    if return_spatial_index:
        return topology, topology_spatial_index
    return topology


def sample_topology_along_segment(
    geom,
    topology_index: TopologySpatialIndex,
    sample_interval_m: float = 50.0,
    tolerance_m: float = 10.0,
) -> tuple[list[tuple[float, int]], dict[int, int]]:
    """Sample topology degrees along a single segment. Thin wrapper around sample_topology_batch()."""
    seg_to_conn, node_features = sample_topology_batch(
        [geom], ["_single"], topology_index, sample_interval_m, tolerance_m
    )
    return seg_to_conn.get("_single", []), node_features


def sample_topology_batch(
    geoms: list,
    seg_ids: list[str],
    topology_index: TopologySpatialIndex,
    sample_interval_m: float = 50.0,
    tolerance_m: float = 10.0,
) -> tuple[dict[str, list[tuple[float, int]]], dict[int, int]]:
    """Batch-sample topology degrees for multiple segments in one spatial query.

    More efficient than calling sample_topology_along_segment() per-segment
    because all sample points are batched into a single STRtree.query_nearest().

    Args:
        geoms: List of LineString geometries (projected CRS)
        seg_ids: Corresponding segment IDs
        topology_index: Spatial index from compute_all_topology()
        sample_interval_m: Distance between sample points in meters
        tolerance_m: Max distance to match a cluster centroid

    Returns:
        Tuple of (seg_to_connectors, node_features) where:
        - seg_to_connectors: {seg_id: [(frac, node_id), ...]}
        - node_features: {node_id: degree}
    """
    import shapely

    if not geoms:
        return {}, {}

    # Phase 1: generate all sample fractions and points per segment
    all_fracs = []  # flat array of fractions
    all_points = []  # flat array of shapely points
    seg_offsets = []  # (seg_idx, start, end) into flat arrays

    offset = 0
    for seg_idx, geom in enumerate(geoms):
        if geom is None or geom.is_empty or geom.length <= 0:
            seg_offsets.append((seg_idx, offset, offset))
            continue

        length = geom.length
        if length > sample_interval_m:
            n_interior = int(length / sample_interval_m)
            interior = np.arange(1, n_interior + 1) * sample_interval_m / length
            interior = interior[interior < 1.0]
            fracs = np.concatenate(([0.0], interior, [1.0]))
        else:
            fracs = np.array([0.0, 1.0])

        points = shapely.line_interpolate_point(geom, fracs * length)
        all_fracs.append(fracs)
        all_points.append(points)
        seg_offsets.append((seg_idx, offset, offset + len(fracs)))
        offset += len(fracs)

    if offset == 0:
        return {seg_id: [] for seg_id in seg_ids}, {}

    # Phase 2: single batch spatial query
    all_points_arr = np.concatenate(all_points)
    all_fracs_arr = np.concatenate(all_fracs)

    nearest_indices, nearest_dists = topology_index.tree.query_nearest(
        all_points_arr, max_distance=tolerance_m, return_distance=True
    )

    # Deduplicate: keep closest match per input point
    best_tree_idx = np.full(offset, -1, dtype=np.intp)
    best_dist = np.full(offset, np.inf)

    if nearest_indices.shape[1] > 0:
        input_idxs = nearest_indices[0]
        tree_idxs = nearest_indices[1]
        for k in range(len(input_idxs)):
            inp = input_idxs[k]
            if nearest_dists[k] < best_dist[inp]:
                best_dist[inp] = nearest_dists[k]
                best_tree_idx[inp] = tree_idxs[k]

    # Phase 3: split results back per segment
    seg_to_connectors: dict[str, list[tuple[float, int]]] = {}
    node_features: dict[int, int] = {}

    for seg_idx, start, end in seg_offsets:
        sid = seg_ids[seg_idx]
        if start == end:
            seg_to_connectors[sid] = []
            continue

        connectors = []
        for i in range(start, end):
            tid = int(best_tree_idx[i])
            if tid >= 0:
                connectors.append((float(all_fracs_arr[i]), tid))
                node_features[tid] = int(topology_index.degrees[tid])
            else:
                vid = -(i + 1)
                connectors.append((float(all_fracs_arr[i]), vid))
                node_features[vid] = 1
        seg_to_connectors[sid] = connectors

    return seg_to_connectors, node_features


def build_overture_connector_spatial_index(
    ref_seg_to_connectors: dict[str, list[tuple[float, int]]],
    ref_geoms_by_id: dict[str, "LineString"],
) -> tuple[np.ndarray, np.ndarray, object] | None:
    """Build spatial index of Overture connector positions.

    Computes physical positions of all Overture connectors by interpolating
    along ref segment geometries, then builds an STRtree for efficient
    spatial queries. This enables using Overture connectors as spatial
    anchors for both ref and target segments.

    Args:
        ref_seg_to_connectors: From build_connector_graph — {seg_id: [(frac, node_id), ...]}
        ref_geoms_by_id: Ref segment geometries — {seg_id: LineString}

    Returns:
        (node_ids_array, points_array, STRtree) or None if no connectors.
    """
    import shapely as shp

    # Compute positions for all unique connectors
    connector_positions: dict[int, tuple[float, float]] = {}
    for seg_id, connectors in ref_seg_to_connectors.items():
        geom = ref_geoms_by_id.get(seg_id)
        if geom is None or geom.is_empty:
            continue
        for frac, node_id in connectors:
            if node_id not in connector_positions:
                pt = geom.interpolate(frac, normalized=True)
                connector_positions[node_id] = (pt.x, pt.y)

    if not connector_positions:
        return None

    node_ids = np.array(list(connector_positions.keys()), dtype=np.int64)
    coords = np.array(list(connector_positions.values()), dtype=np.float64)
    points = shp.points(coords)
    tree = shp.STRtree(points)

    return node_ids, points, tree


def find_overture_connectors_for_targets(
    target_geoms_by_id: dict[str, "LineString"],
    connector_index: tuple[np.ndarray, np.ndarray, object],
    tolerance_m: float = 5.0,
) -> dict[str, list[tuple[float, int]]]:
    """Find Overture connectors near each target segment via spatial query.

    Projects Overture connector positions onto target segments, returning
    results in the same ID space as ref_seg_to_connectors. This enables
    direct Jaccard comparison of connector sets between ref and target.

    Args:
        target_geoms_by_id: Target segment geometries — {seg_id: LineString}
        connector_index: From build_overture_connector_spatial_index()
        tolerance_m: Max distance for a connector to be considered "near" a segment

    Returns:
        {target_seg_id: [(frac, overture_node_id), ...]} sorted by frac.
        Uses the same node IDs as ref_seg_to_connectors.
    """
    import shapely as shp

    node_ids, points, tree = connector_index
    result: dict[str, list[tuple[float, int]]] = {}

    for target_id, target_geom in target_geoms_by_id.items():
        if target_geom is None or target_geom.is_empty:
            result[target_id] = []
            continue

        # Query tree for connectors within tolerance (dwithin avoids buffer overhead)
        nearby_indices = tree.query(target_geom, predicate="dwithin", distance=tolerance_m)

        if len(nearby_indices) == 0:
            result[target_id] = []
            continue

        # Project each nearby connector onto the target geometry
        connectors = []
        for idx in nearby_indices:
            frac = float(shp.line_locate_point(target_geom, points[idx]) / target_geom.length)
            frac = max(0.0, min(1.0, frac))
            connectors.append((frac, int(node_ids[idx])))

        connectors.sort(key=lambda x: x[0])
        result[target_id] = connectors

    return result


def _connectors_near_endpoint(
    connectors: list[tuple[float, int]],
    endpoint_frac: float,
    frac_tolerance: float,
) -> set[int]:
    """Return node IDs of all connectors within frac_tolerance of endpoint_frac."""
    return {node_id for frac, node_id in connectors if abs(frac - endpoint_frac) <= frac_tolerance}


def compute_shared_anchor_features(
    ref_seg_id: str,
    target_seg_id: str,
    ref_seg_to_connectors: dict[str, list[tuple[float, int]]],
    target_overture_connectors: dict[str, list[tuple[float, int]]],
    ref_start_frac: float,
    ref_end_frac: float,
    target_start_frac: float,
    target_end_frac: float,
    ref_length_m: float,
    target_length_m: float,
    tolerance_m: float = 5.0,
) -> dict[str, float]:
    """Count alignment endpoints where ref and target share an Overture connector.

    At each alignment endpoint, collects ALL connectors within tolerance_m
    (converted to fractional position using segment length), then checks
    set intersection. This avoids picking a single "nearest" winner that
    might not match across sides.

    Args:
        ref_seg_id: Reference segment ID
        target_seg_id: Target segment ID
        ref_seg_to_connectors: Ref connector mapping {seg_id: [(frac, node_id), ...]}
        target_overture_connectors: Target connector mapping {seg_id: [(frac, node_id), ...]}
        ref_start_frac: Alignment start fraction on ref
        ref_end_frac: Alignment end fraction on ref
        target_start_frac: Alignment start fraction on target
        target_end_frac: Alignment end fraction on target
        ref_length_m: Reference segment length in meters
        target_length_m: Target segment length in meters
        tolerance_m: Max distance along segment to consider a connector "at" an endpoint

    Returns:
        {"shared_anchor_count": count} where count is 0, 1, or 2
    """
    ref_connectors = ref_seg_to_connectors.get(ref_seg_id, [])
    target_connectors = target_overture_connectors.get(target_seg_id, [])

    if not ref_connectors or not target_connectors:
        return {"shared_anchor_count": 0.0}

    # Convert absolute meter tolerance to fractional position per segment
    ref_frac_tol = tolerance_m / ref_length_m if ref_length_m > 0 else 0.0
    target_frac_tol = tolerance_m / target_length_m if target_length_m > 0 else 0.0

    count = 0

    # Start endpoint: collect all connectors within tolerance, check intersection
    ref_start_nodes = _connectors_near_endpoint(ref_connectors, ref_start_frac, ref_frac_tol)
    target_start_nodes = _connectors_near_endpoint(
        target_connectors, target_start_frac, target_frac_tol
    )
    if ref_start_nodes & target_start_nodes:
        count += 1

    # End endpoint: same logic
    ref_end_nodes = _connectors_near_endpoint(ref_connectors, ref_end_frac, ref_frac_tol)
    target_end_nodes = _connectors_near_endpoint(
        target_connectors, target_end_frac, target_frac_tol
    )
    if ref_end_nodes & target_end_nodes:
        count += 1

    return {"shared_anchor_count": float(count)}


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
        return float("nan")

    counter_a = Counter(sig_a)
    counter_b = Counter(sig_b)

    # Jaccard similarity on multisets
    intersection = sum((counter_a & counter_b).values())
    union = sum((counter_a | counter_b).values())

    return intersection / union if union > 0 else 0.0


def build_inferred_graph(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
) -> tuple["SparseGraph", dict[str, int], dict[str, int]]:
    """Build sparse graph from spaghetti geometry using endpoint clustering.

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
        G: SparseGraph where nodes=endpoint clusters, edges=segments
        seg_to_start_node: Maps segment ID -> start node cluster ID
        seg_to_end_node: Maps segment ID -> end node cluster ID
    """
    from scipy.sparse import csr_matrix

    from matcher.topology.sparse_graph import SparseGraph, build_graph_from_edges

    if gdf.empty:
        empty_graph = SparseGraph(
            adjacency=csr_matrix((0, 0), dtype=np.int32),
            node_ids=[],
            node_to_idx={},
        )
        return empty_graph, {}, {}

    t_start = time.perf_counter()
    logger.info(f"[graphlet] Building inferred graph from {len(gdf)} segments")

    # Project to local CRS if in geographic coordinates
    work_gdf = gdf
    if gdf.crs is not None and gdf.crs.is_geographic:
        # Use geopandas' estimate_utm_crs() for consistent UTM zone selection
        utm_crs = gdf.estimate_utm_crs()
        work_gdf = gdf.to_crs(utm_crs)
        logger.debug(f"[graphlet] Projected to {utm_crs}")

    # Step 1: Extract endpoints using vectorized shapely (LineStrings only, filtered at ingest)
    import shapely

    geometries = work_gdf.geometry.values
    segment_ids_arr = work_gdf[id_column].astype(str).values

    # Fast vectorized path for LineStrings
    valid_mask = ~shapely.is_empty(geometries) & ~shapely.is_missing(geometries)
    valid_geoms = geometries[valid_mask]
    valid_seg_ids = segment_ids_arr[valid_mask]

    if len(valid_geoms) == 0:
        empty_graph = SparseGraph(
            adjacency=csr_matrix((0, 0), dtype=np.int32),
            node_ids=[],
            node_to_idx={},
        )
        return empty_graph, {}, {}

    start_points = shapely.get_point(valid_geoms, 0)
    end_points = shapely.get_point(valid_geoms, -1)

    # Filter out geometries where get_point returned None (e.g., degenerate LineStrings)
    valid_points_mask = ~shapely.is_missing(start_points) & ~shapely.is_missing(end_points)
    if not valid_points_mask.all():
        n_filtered = (~valid_points_mask).sum()
        logger.debug(f"[graphlet] Filtered {n_filtered} geometries with invalid start/end points")
        start_points = start_points[valid_points_mask]
        end_points = end_points[valid_points_mask]
        valid_geoms = valid_geoms[valid_points_mask]
        valid_seg_ids = valid_seg_ids[valid_points_mask]

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

    # Step 3: Build node mappings
    seg_to_start_node: dict[str, int] = {}
    seg_to_end_node: dict[str, int] = {}
    all_nodes: set[int] = set()

    for ep_idx in range(n_endpoints):
        seg_id = endpoint_segment_ids[ep_idx]
        is_start = endpoint_is_start[ep_idx]
        node_id = uf.find(ep_idx)

        all_nodes.add(node_id)
        if is_start:
            seg_to_start_node[seg_id] = node_id
        else:
            seg_to_end_node[seg_id] = node_id

    # Collect edges (segments connect their endpoint clusters)
    edges: list[tuple[int, int]] = []
    for seg_id in seg_to_start_node:
        start_node = seg_to_start_node[seg_id]
        end_node = seg_to_end_node.get(seg_id)
        if end_node is not None and start_node != end_node:
            edges.append((start_node, end_node))

    # Build SparseGraph
    G = build_graph_from_edges(edges)

    logger.info(
        f"[graphlet] Built graph: {G.n_nodes} nodes, {G.n_edges} edges "
        f"in {time.perf_counter() - t_start:.2f}s"
    )

    return G, seg_to_start_node, seg_to_end_node


# Cached numba functions (compiled on first use)
_NUMBA_GRAPHLET_FUNCS: tuple | None = None


def _build_csr_from_graph(G: "SparseGraph") -> tuple[np.ndarray, np.ndarray, list, dict]:
    """Extract CSR arrays from SparseGraph for numba processing.

    Returns:
        indptr: CSR row pointers
        indices: CSR column indices (sorted per row)
        node_list: List of original node IDs
        node_to_idx: Mapping from node ID to integer index
    """
    # SparseGraph already stores CSR format
    indptr = G.adjacency.indptr.astype(np.int64)
    indices = G.adjacency.indices.astype(np.int64)
    return indptr, indices, G.node_ids, G.node_to_idx


def _get_numba_graphlet_functions():
    """Get numba-accelerated graphlet functions (cached after first call).

    Functions are compiled on first call and cached to disk via numba's cache=True.

    Returns:
        Tuple of (count_squares_numba, count_two_hop_numba) functions
    """
    global _NUMBA_GRAPHLET_FUNCS
    if _NUMBA_GRAPHLET_FUNCS is not None:
        return _NUMBA_GRAPHLET_FUNCS

    from numba import njit

    @njit(cache=True)
    def count_squares_numba(n_nodes, indptr, indices):
        """Count 4-cycles (squares) through each node using CSR adjacency.

        For each node v, count squares v-n1-c-n2-v where n1,n2 are neighbors
        of v and c is a common neighbor of n1,n2 (c != v).

        Uses merge-based intersection of sorted neighbor lists for efficiency.

        Args:
            n_nodes: Number of nodes in the graph
            indptr: CSR row pointers array (length n_nodes + 1)
            indices: CSR column indices array (sorted per row)

        Returns:
            Array of square counts, one per node
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

    @njit(cache=True)
    def count_two_hop_numba(n_nodes, indptr, indices):
        """Count two-hop neighbors for each node.

        Two-hop neighbors are nodes reachable in exactly 2 hops,
        excluding direct neighbors and the node itself.

        Args:
            n_nodes: Number of nodes in the graph
            indptr: CSR row pointers array
            indices: CSR column indices array

        Returns:
            Array of two-hop neighbor counts, one per node
        """
        result = np.zeros(n_nodes, dtype=np.int64)

        # Allocate seen array once and reuse (memory efficiency)
        seen = np.zeros(n_nodes, dtype=np.int8)

        for v in range(n_nodes):
            start_v = indptr[v]
            end_v = indptr[v + 1]
            neighbors_v = indices[start_v:end_v]

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

            # Reset seen array for next iteration (only reset marked entries)
            seen[v] = 0
            for ni in neighbors_v:
                seen[ni] = 0
                start_ni = indptr[ni]
                end_ni = indptr[ni + 1]
                for k in range(start_ni, end_ni):
                    seen[indices[k]] = 0

        return result

    # Cache the functions
    _NUMBA_GRAPHLET_FUNCS = (count_squares_numba, count_two_hop_numba)
    return _NUMBA_GRAPHLET_FUNCS


def compute_road_graphlet_features(
    G: "SparseGraph",
    degrees_only: bool = False,
) -> dict[int, np.ndarray] | dict[int, int]:
    """Compute simplified graphlet features optimized for road networks.

    Uses numba-accelerated computation for square counting and two-hop neighbor
    counting, providing ~10-90x speedup over pure Python.

    When degrees_only=True, returns only degree values for minimal memory usage.
    This is sufficient for the most discriminative feature (endpoint_degree_similarity)
    and reduces memory by ~90% compared to full 6-feature vectors.

    Returns a 6-dimensional feature vector per node (or just degree if degrees_only):
    - degree: Number of edges at node (1-4 typical for roads)
    - triangles: Count of 3-cycles through node (rare in roads)
    - squares: Count of 4-cycles through node (common in grid cities)
    - clustering: Local clustering coefficient (low for roads)
    - two_hop_count: Nodes reachable in 2 hops (distinguishes grid vs tree)
    - is_articulation: Whether removal disconnects graph (bridge intersections)

    Args:
        G: SparseGraph from build_inferred_graph()
        degrees_only: If True, return dict of node_id -> degree (int) only

    Returns:
        If degrees_only=False: Dictionary mapping node_id -> 6-dimensional numpy array
        If degrees_only=True: Dictionary mapping node_id -> degree (int)
    """
    from matcher.topology.sparse_graph import (
        compute_clustering,
        compute_triangles,
        find_articulation_points,
    )

    n_nodes = G.n_nodes
    if n_nodes == 0:
        return {}

    # Fast path: only compute degrees for memory efficiency
    if degrees_only:
        t_start = time.perf_counter()
        degrees = G.degrees()
        logger.debug(
            f"[graphlet] Computed degrees for {len(degrees)} nodes in "
            f"{time.perf_counter() - t_start:.2f}s (degrees_only mode)"
        )
        return degrees

    t_start = time.perf_counter()
    features: dict[int, np.ndarray] = {}

    # Pre-compute graph-wide properties using sparse_graph functions
    triangles = compute_triangles(G)
    clustering = compute_clustering(G)

    # Find articulation points (handles disconnected graphs internally)
    articulation_points = find_articulation_points(G)

    # Get numba-accelerated functions (cached and warmed up)
    count_squares_numba, count_two_hop_numba = _get_numba_graphlet_functions()

    # Get CSR arrays from SparseGraph
    indptr, indices, node_list, _ = _build_csr_from_graph(G)

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
                clustering.get(node, float("nan")),
                two_hop_arr[i],
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
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    degrees_only: bool = False,
) -> tuple[
    "SparseGraph | None", dict[str, list[tuple[float, int]]], dict[int, np.ndarray] | dict[int, int]
]:
    """Build sparse graph from Overture segments using explicit connector data.

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
        degrees_only: If True, only return node degrees for memory efficiency

    Returns:
        G: SparseGraph (None if degrees_only) where nodes=connectors, edges=segment connections
        seg_to_connectors: Maps segment ID -> list of (at_position, node_id) sorted by position
        node_features: Dict mapping node_id -> degree (int) if degrees_only else feature vector
    """
    from scipy.sparse import csr_matrix

    from matcher.topology.sparse_graph import SparseGraph, build_graph_from_edges

    if gdf.empty:
        empty_graph = SparseGraph(
            adjacency=csr_matrix((0, 0), dtype=np.int32),
            node_ids=[],
            node_to_idx={},
        )
        return empty_graph, {}, {}

    t_start = time.perf_counter()
    logger.info(f"[graphlet-connector] Building connector graph from {len(gdf)} segments")

    # Check if connectors column exists
    if connectors_column not in gdf.columns:
        logger.info(
            f"[graphlet-connector] No '{connectors_column}' column found, "
            "inferring connectivity from spatial proximity"
        )
        # Use the new inference function that detects mid-segment crossings
        return build_inferred_connector_graph(gdf, id_column, tolerance_m, degrees_only)

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

    # Build segment->connectors mapping and collect edges
    seg_to_connectors: dict[str, list[tuple[float, int]]] = {}
    edges: list[tuple[int, int]] = []
    all_nodes: set[int] = set()

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
                    all_nodes.add(node_id)
                    segment_connectors.append((at_pos, node_id))

        # Sort by position along segment
        segment_connectors.sort(key=lambda x: x[0])
        seg_to_connectors[seg_id] = segment_connectors

        # Collect edges between consecutive connectors on this segment
        for i in range(len(segment_connectors) - 1):
            _, node_a = segment_connectors[i]
            _, node_b = segment_connectors[i + 1]
            if node_a != node_b:
                edges.append((node_a, node_b))

    # Build SparseGraph
    G = build_graph_from_edges(edges)

    # Compute graphlet features for all nodes (or just degrees for memory efficiency)
    t0 = time.perf_counter()
    n_nodes = G.n_nodes
    n_edges = G.n_edges
    node_features = compute_road_graphlet_features(G, degrees_only=degrees_only)
    logger.debug(
        f"[graphlet-connector] Computed {'degrees' if degrees_only else 'features'} "
        f"in {time.perf_counter() - t0:.2f}s"
    )

    # Discard graph if using degrees_only mode for memory efficiency
    if degrees_only:
        G = None

    logger.info(
        f"[graphlet-connector] Built graph: {n_nodes} nodes, {n_edges} edges "
        f"in {time.perf_counter() - t_start:.2f}s"
    )

    return G, seg_to_connectors, node_features


def _build_inferred_graph_with_features(
    gdf: gpd.GeoDataFrame,
    id_column: str,
    tolerance_m: float,
) -> tuple["SparseGraph", dict[str, int], dict[str, int], dict[int, np.ndarray]]:
    """Helper: build inferred graph and compute features in one step."""
    G, seg_to_start, seg_to_end = build_inferred_graph(gdf, id_column, tolerance_m)
    node_features = compute_road_graphlet_features(G)
    return G, seg_to_start, seg_to_end, node_features


def build_inferred_connector_graph(
    gdf: gpd.GeoDataFrame,
    id_column: str = "id",
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    degrees_only: bool = False,
) -> tuple[
    "SparseGraph | None", dict[str, list[tuple[float, int]]], dict[int, np.ndarray] | dict[int, int]
]:
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
        degrees_only: If True, only return node degrees for memory efficiency

    Returns:
        G: SparseGraph (None if degrees_only) where nodes=connection points, edges=segment portions
        seg_to_connectors: Maps segment ID -> list of (at_position, node_id) sorted by position
        node_features: Dict mapping node_id -> degree (int) if degrees_only else feature vector
    """
    from scipy.sparse import csr_matrix
    from shapely import STRtree
    from shapely.ops import nearest_points

    from matcher.topology.sparse_graph import SparseGraph, build_graph_from_edges

    if gdf.empty:
        empty_graph = SparseGraph(
            adjacency=csr_matrix((0, 0), dtype=np.int32),
            node_ids=[],
            node_to_idx={},
        )
        return empty_graph, {}, {}

    t_start = time.perf_counter()
    logger.info(f"[graphlet-infer] Inferring connector graph from {len(gdf)} segments")

    # Project to local CRS if in geographic coordinates
    work_gdf = gdf
    if gdf.crs is not None and gdf.crs.is_geographic:
        # Use geopandas' estimate_utm_crs() for consistent UTM zone selection
        utm_crs = gdf.estimate_utm_crs()
        work_gdf = gdf.to_crs(utm_crs)
        logger.debug(f"[graphlet-infer] Projected to {utm_crs}")

    geometries = work_gdf.geometry.values
    segment_ids = work_gdf[id_column].astype(str).values

    # Collect all connection points: (seg_id, seg_idx, at_position, x, y)
    connection_points = []

    # Step 1: Add endpoints for all segments (LineStrings only, MultiLineStrings skipped)
    from shapely import LineString

    for seg_idx, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        # Skip MultiLineStrings - they don't have direct coords
        if not isinstance(geom, LineString):
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
        if not isinstance(geom, LineString):
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
            if not isinstance(other_geom, LineString):
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

    # Step 5: Build graph - collect edges
    edges: list[tuple[int, int]] = []

    # Collect edges between consecutive connectors on each segment
    for _seg_id, connectors in seg_to_connectors.items():
        for i in range(len(connectors) - 1):
            _, node_a = connectors[i]
            _, node_b = connectors[i + 1]
            if node_a != node_b:
                edges.append((node_a, node_b))

    # Build SparseGraph
    G = build_graph_from_edges(edges)

    # Compute graphlet features (or just degrees for memory efficiency)
    t0 = time.perf_counter()
    n_nodes = G.n_nodes
    n_edges = G.n_edges
    node_features = compute_road_graphlet_features(G, degrees_only=degrees_only)
    logger.debug(
        f"[graphlet-infer] Computed {'degrees' if degrees_only else 'features'} "
        f"in {time.perf_counter() - t0:.2f}s"
    )

    logger.info(
        f"[graphlet-infer] Built graph: {n_nodes} nodes, {n_edges} edges "
        f"({mid_crossings_added} mid-segment crossings) in {time.perf_counter() - t_start:.2f}s"
    )

    # Discard graph if using degrees_only mode for memory efficiency
    if degrees_only:
        G = None

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


def find_nearest_connector_position(
    seg_connectors: list[tuple[float, int]],
    position: float,
) -> float | None:
    """Find the position of the connector nearest to a given position along a segment.

    Like find_nearest_connector() but returns the connector's at_position
    instead of the node_id. Used to snap alignment fractions to real
    network junctions for endpoint proximity computation.

    Args:
        seg_connectors: List of (at_position, node_id) tuples, sorted by position
        position: Linear position along segment (0.0 to 1.0)

    Returns:
        at_position of nearest connector, or None if no connectors
    """
    if not seg_connectors:
        return None

    best_pos = None
    best_dist = float("inf")

    for at_pos, _node_id in seg_connectors:
        dist = abs(at_pos - position)
        if dist < best_dist:
            best_dist = dist
            best_pos = at_pos

    return best_pos


def compute_aligned_topology_at_position(
    seg_to_connectors: dict[str, list[tuple[float, int]]],
    node_features: dict[int, int],
    seg_id: str,
    position: float,
) -> int:
    """Get topology degree at a specific position along a segment.

    Uses the connector graph to find the nearest connector to the
    given position, then returns its degree.

    This function enables alignment-aware topology computation. For partial
    overlaps, instead of using full geometry endpoints, we use the degrees
    at the aligned portion endpoints (where the segments actually overlap).

    Args:
        seg_to_connectors: Maps segment ID -> list of (at_position, node_id) tuples
        node_features: Maps node_id -> degree (or feature vector, degree is first element)
        seg_id: Segment ID to lookup
        position: Linear position along segment (0.0 to 1.0)

    Returns:
        Degree at the nearest connector to the position.
        Returns 1 (dead end default) if no connectors found.
    """
    connectors = seg_to_connectors.get(seg_id, [])
    if not connectors:
        return 1  # Dead end default

    node_id = find_nearest_connector(connectors, position)
    if node_id is None:
        return 1

    # Handle both int (degrees only) and array (full feature vector) formats
    feat = node_features.get(node_id)
    if feat is None:
        return 1
    if isinstance(feat, (int, np.integer)):
        return int(feat)
    # Array format - degree is first element
    return int(feat[0]) if len(feat) > 0 else 1


def compute_aligned_topology_features(
    seg_id: str,
    seg_to_connectors: dict[str, list[tuple[float, int]]],
    node_features: dict[int, int],
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> dict[str, int | float | bool | tuple]:
    """Compute topology features at aligned portion endpoints.

    This is the alignment-aware version of compute_topology_features().
    Instead of using the full geometry endpoints, it uses the degrees
    at the aligned portion endpoints.

    For partial overlaps (e.g., only 43% of the segment overlaps), this
    gives the correct topology at the boundaries of the actual overlap,
    not at the full segment endpoints.

    Args:
        seg_id: Segment ID
        seg_to_connectors: Maps segment ID -> list of (at_position, node_id) tuples
        node_features: Maps node_id -> degree (int or feature vector)
        start_frac: Start of aligned region (0.0 to 1.0)
        end_frac: End of aligned region (0.0 to 1.0)

    Returns:
        Dictionary with topology features:
        - from_degree: Degree at alignment start
        - to_degree: Degree at alignment end
        - is_dead_end: True if either endpoint has degree 1
        - is_intersection: True if either endpoint has degree > 2
        - degree_signature: Tuple of sorted [from_degree, to_degree]
    """
    from_degree = compute_aligned_topology_at_position(
        seg_to_connectors, node_features, seg_id, start_frac
    )
    to_degree = compute_aligned_topology_at_position(
        seg_to_connectors, node_features, seg_id, end_frac
    )

    return {
        "from_degree": from_degree,
        "to_degree": to_degree,
        "is_dead_end": min(from_degree, to_degree) == 1,
        "is_intersection": max(from_degree, to_degree) > 2,
        "degree_signature": tuple(sorted([from_degree, to_degree])),
    }


def get_alignment_connectors(
    seg_id: str,
    seg_to_connectors: dict[str, list[tuple[float, int]]],
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> tuple[int | None, int | None]:
    """Get connector nodes at the endpoints of an aligned portion.

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
    ref_features: dict[int, np.ndarray] | dict[int, int],
    target_features: dict[int, np.ndarray] | dict[int, int],
    ref_seg_to_connectors: dict[str, list[tuple[float, int]]],
    target_seg_to_connectors: dict[str, list[tuple[float, int]]],
    ref_start_frac: float = 0.0,
    ref_end_frac: float = 1.0,
    target_start_frac: float = 0.0,
    target_end_frac: float = 1.0,
) -> dict[str, float]:
    """Compare graphlet features at aligned portion endpoints using connector positions.

    This function accounts for alignment by finding the nearest connectors to
    the aligned portion endpoints, rather than always using segment endpoints.

    For example, if alignment indicates the match covers ref[0.3:0.8], we find
    the nearest connectors to positions 0.3 and 0.8, then compare their graphlet
    features to the corresponding target positions.

    Supports both full feature vectors (6-element arrays) and degrees-only mode
    (int values) for memory efficiency. In degrees-only mode, graphlet_similarity
    is set equal to endpoint_degree_similarity.

    Args:
        ref_seg_id: Reference segment ID
        target_seg_id: Target segment ID
        ref_features: Node features - either Dict[node_id, array] or Dict[node_id, int] (degrees only)
        target_features: Node features - either Dict[node_id, array] or Dict[node_id, int] (degrees only)
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

    # Helper to extract degree from feature (handles both int and array)
    def get_degree(features: dict, node: int | None, default: int = 1) -> int:
        if node is None:
            return default
        feat = features.get(node)
        if feat is None:
            return default
        if isinstance(feat, (int, np.integer)):
            return int(feat)
        # Array format - degree is first element
        return int(feat[0]) if len(feat) > 0 else default

    # Get degrees for all endpoints
    ref_start_deg = get_degree(ref_features, ref_start_node)
    ref_end_deg = get_degree(ref_features, ref_end_node)
    target_start_deg = get_degree(target_features, target_start_node)
    target_end_deg = get_degree(target_features, target_end_node)

    # Compute degree similarity (works for both formats)
    degree_fwd = (
        1.0
        - abs(ref_start_deg - target_start_deg) / 10.0
        + 1.0
        - abs(ref_end_deg - target_end_deg) / 10.0
    ) / 2.0
    degree_fwd = max(0.0, min(1.0, degree_fwd))
    degree_rev = (
        1.0
        - abs(ref_start_deg - target_end_deg) / 10.0
        + 1.0
        - abs(ref_end_deg - target_start_deg) / 10.0
    ) / 2.0
    degree_rev = max(0.0, min(1.0, degree_rev))

    # Check if we have full feature vectors or just degrees
    sample_feat = next(iter(ref_features.values()), None) if ref_features else None
    if sample_feat is not None and isinstance(sample_feat, np.ndarray):
        # Full feature mode - compute graphlet_similarity from all features
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
            target_features.get(target_end_node, default)
            if target_end_node is not None
            else default
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

        return {
            "graphlet_similarity": max(fwd, rev),
            "endpoint_degree_similarity": max(degree_fwd, degree_rev),
        }
    else:
        # Degrees-only mode - use degree similarity for both metrics
        # This is the most discriminative feature for roads anyway
        return {
            "graphlet_similarity": max(degree_fwd, degree_rev),
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


def compute_clustering_coefficient_features(
    ref_seg_id: str,
    target_seg_id: str,
    ref_features: dict[int, np.ndarray] | dict[int, int],
    target_features: dict[int, np.ndarray] | dict[int, int],
    ref_seg_to_connectors: dict[str, list[tuple[float, int]]],
    target_seg_to_connectors: dict[str, list[tuple[float, int]]],
    ref_start_frac: float = 0.0,
    ref_end_frac: float = 1.0,
    target_start_frac: float = 0.0,
    target_end_frac: float = 1.0,
) -> dict[str, float]:
    """Extract clustering coefficient features at aligned endpoints.

    The clustering coefficient measures how interconnected a node's neighbors are.
    For road networks:
    - Low clustering (~0) is typical for most intersections (cars can't make U-turns)
    - Higher clustering appears in complex interchanges or grid networks

    The clustering coefficient is extracted from the precomputed graphlet feature
    vectors (index 3 in the 6-element vector).

    Args:
        ref_seg_id: Reference segment ID
        target_seg_id: Target segment ID
        ref_features: Node features - Dict[node_id, array] (full) or Dict[node_id, int] (degrees only)
        target_features: Node features
        ref_seg_to_connectors: Maps ref segment ID -> [(at, node_id), ...] sorted by at
        target_seg_to_connectors: Maps target segment ID -> [(at, node_id), ...] sorted by at
        ref_start_frac: Start position of alignment on reference (0.0 to 1.0)
        ref_end_frac: End position of alignment on reference (0.0 to 1.0)
        target_start_frac: Start position of alignment on target (0.0 to 1.0)
        target_end_frac: End position of alignment on target (0.0 to 1.0)

    Returns:
        Dictionary with:
        - clustering_coef_ref: Average clustering coefficient at ref endpoints
        - clustering_coef_target: Average clustering coefficient at target endpoints
        - clustering_coef_delta: Absolute difference between ref and target clustering
    """
    _nan = float("nan")
    default_result = {
        "clustering_coef_ref": _nan,
        "clustering_coef_target": _nan,
        "clustering_coef_delta": _nan,
    }

    # Check if we have full feature vectors (clustering is index 3)
    sample_feat = next(iter(ref_features.values()), None) if ref_features else None
    if sample_feat is None or not isinstance(sample_feat, np.ndarray) or len(sample_feat) < 4:
        # Degrees-only mode or insufficient features - return NaN
        return default_result

    def get_clustering(features: dict, connectors: list, frac: float) -> float:
        """Get clustering coefficient at position along segment."""
        if not connectors:
            return _nan
        node_id = find_nearest_connector(connectors, frac)
        if node_id is None:
            return _nan
        feat = features.get(node_id)
        if feat is None or not isinstance(feat, np.ndarray) or len(feat) < 4:
            return _nan
        return float(feat[3])  # Index 3 is clustering coefficient

    # Get connectors for both segments
    ref_connectors = ref_seg_to_connectors.get(ref_seg_id, [])
    target_connectors = target_seg_to_connectors.get(target_seg_id, [])

    # Get clustering at each alignment endpoint
    ref_start_clust = get_clustering(ref_features, ref_connectors, ref_start_frac)
    ref_end_clust = get_clustering(ref_features, ref_connectors, ref_end_frac)
    target_start_clust = get_clustering(target_features, target_connectors, target_start_frac)
    target_end_clust = get_clustering(target_features, target_connectors, target_end_frac)

    # Average across endpoints
    clustering_coef_ref = (ref_start_clust + ref_end_clust) / 2.0
    clustering_coef_target = (target_start_clust + target_end_clust) / 2.0
    clustering_coef_delta = abs(clustering_coef_ref - clustering_coef_target)

    return {
        "clustering_coef_ref": clustering_coef_ref,
        "clustering_coef_target": clustering_coef_target,
        "clustering_coef_delta": clustering_coef_delta,
    }


def compute_interior_connector_features(
    ref_seg_id: str,
    target_seg_id: str,
    ref_seg_to_connectors: dict[str, list[tuple[float, int]]],
    target_seg_to_connectors: dict[str, list[tuple[float, int]]],
    ref_node_features: dict[int, int | np.ndarray],
    target_node_features: dict[int, int | np.ndarray],
    ref_start_frac: float,
    ref_end_frac: float,
    target_start_frac: float,
    target_end_frac: float,
) -> dict[str, float]:
    """Compare interior connector sequences along the aligned portion.

    Interior connectors are those strictly between the alignment endpoints
    (exclusive of start/end). This captures whether two segments have similar
    junction patterns along their aligned overlap — a strong indicator of
    structural correspondence.

    For the target side, expects Overture connectors projected onto the
    target segment (from find_overture_connectors_for_targets), ensuring
    both sides use the same connector ID space for direct Jaccard comparison.

    Args:
        ref_seg_id: Reference segment ID
        target_seg_id: Target segment ID
        ref_seg_to_connectors: Ref connector map {seg_id: [(frac, node_id), ...]}
        target_seg_to_connectors: Target connector map (junction-filtered)
        ref_node_features: Ref node features {node_id: degree or feature vector}
        target_node_features: Target node features {node_id: degree or feature vector}
        ref_start_frac: Start of alignment on ref (0-1)
        ref_end_frac: End of alignment on ref (0-1)
        target_start_frac: Start of alignment on target (0-1)
        target_end_frac: End of alignment on target (0-1)

    Returns:
        Dict with interior_junction_count_ref, interior_junction_count_target,
        interior_junction_count_delta, interior_connector_jaccard,
        interior_junction_position_sim
    """
    _nan = float("nan")
    ENDPOINT_TOL = 0.01  # Tolerance for "at endpoint" exclusion

    def _get_degree(node_features: dict, node_id: int) -> int:
        """Extract degree from node_features, handling both int and ndarray values."""
        val = node_features.get(node_id, 1)
        if isinstance(val, np.ndarray):
            return int(val[0])  # degree is index 0 of feature vector
        return int(val)

    def _get_interior_connectors(
        seg_id: str,
        seg_to_connectors: dict[str, list[tuple[float, int]]],
        node_features: dict,
        start_frac: float,
        end_frac: float,
    ) -> list[tuple[float, int]]:
        """Get connectors strictly inside the aligned range."""
        connectors = seg_to_connectors.get(seg_id, [])
        interior = []
        for frac, node_id in connectors:
            if (start_frac + ENDPOINT_TOL) < frac < (end_frac - ENDPOINT_TOL):
                degree = _get_degree(node_features, node_id)
                if degree >= 2:  # Only count junctions
                    interior.append((frac, node_id))
        return interior

    ref_interior = _get_interior_connectors(
        ref_seg_id, ref_seg_to_connectors, ref_node_features, ref_start_frac, ref_end_frac
    )
    target_interior = _get_interior_connectors(
        target_seg_id,
        target_seg_to_connectors,
        target_node_features,
        target_start_frac,
        target_end_frac,
    )

    count_ref = len(ref_interior)
    count_target = len(target_interior)
    count_delta = abs(count_ref - count_target)

    # Connector ID Jaccard: both sides use Overture connector node IDs
    # (same ID space), so set intersection measures shared physical junctions
    ref_ids = set(nid for _, nid in ref_interior)
    target_ids = set(nid for _, nid in target_interior)
    if not ref_ids and not target_ids:
        connector_jaccard = 1.0  # Both empty = perfect match
    elif not ref_ids or not target_ids:
        connector_jaccard = 0.0  # One has junctions, other doesn't
    else:
        connector_jaccard = len(ref_ids & target_ids) / len(ref_ids | target_ids)

    # Position similarity: rescale interior positions to [0,1] within alignment,
    # then match by nearest position
    if count_ref == 0 and count_target == 0:
        position_sim = 1.0
    elif count_ref == 0 or count_target == 0:
        position_sim = 0.0
    else:
        ref_span = ref_end_frac - ref_start_frac
        target_span = target_end_frac - target_start_frac

        ref_positions = sorted(
            (frac - ref_start_frac) / ref_span if ref_span > 0 else 0.5 for frac, _ in ref_interior
        )
        target_positions = sorted(
            (frac - target_start_frac) / target_span if target_span > 0 else 0.5
            for frac, _ in target_interior
        )

        # Greedy nearest matching: pair each position in the shorter list
        # with its nearest in the longer list
        if len(ref_positions) <= len(target_positions):
            shorter, longer = ref_positions, target_positions
        else:
            shorter, longer = target_positions, ref_positions

        total_diff = 0.0
        used = set()
        for pos in shorter:
            best_diff = float("inf")
            best_idx = 0
            for j, lpos in enumerate(longer):
                if j not in used:
                    d = abs(pos - lpos)
                    if d < best_diff:
                        best_diff = d
                        best_idx = j
            total_diff += best_diff
            used.add(best_idx)

        # Penalize unmatched positions in the longer list
        n_unmatched = len(longer) - len(shorter)
        # Unmatched positions contribute 0.5 each (midrange penalty)
        total_diff += n_unmatched * 0.5
        max_count = max(len(shorter), len(longer))
        mean_diff = total_diff / max_count if max_count > 0 else 0.0
        position_sim = max(0.0, 1.0 - mean_diff)

    return {
        "interior_junction_count_ref": float(count_ref),
        "interior_junction_count_target": float(count_target),
        "interior_junction_count_delta": float(count_delta),
        "interior_connector_jaccard": connector_jaccard,
        "interior_junction_position_sim": position_sim,
    }
