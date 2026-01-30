"""Island detection for post-integration analysis.

Detects disconnected components in integrated road networks using NetworkX
graph analysis. Classifies islands by severity to guide review/repair.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from loguru import logger
from shapely.geometry import Point


class IslandSeverity(Enum):
    """Severity classification for disconnected components."""

    CRITICAL = "critical"  # Single isolated segment - likely error
    WARNING = "warning"  # Small cluster far from main network - review
    INFO = "info"  # Small cluster near main network - probably OK


@dataclass
class Island:
    """A disconnected component in the network."""

    component_id: int
    edge_ids: list[str | int]
    severity: IslandSeverity
    edge_count: int
    total_length_m: float
    centroid: Point
    min_distance_to_main_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IslandDetectionResult:
    """Result of island detection analysis."""

    total_edges: int
    total_components: int
    main_component_size: int
    main_component_ratio: float
    islands: list[Island]

    # Severity counts
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_edges": self.total_edges,
            "total_components": self.total_components,
            "main_component_size": self.main_component_size,
            "main_component_ratio": round(self.main_component_ratio, 4),
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "islands": [
                {
                    "component_id": i.component_id,
                    "edge_count": i.edge_count,
                    "severity": i.severity.value,
                    "total_length_m": round(i.total_length_m, 2),
                    "min_distance_to_main_m": (
                        round(i.min_distance_to_main_m, 2)
                        if i.min_distance_to_main_m is not None
                        else None
                    ),
                }
                for i in self.islands
            ],
        }


def detect_islands(
    edges_gdf: gpd.GeoDataFrame,
    snap_tolerance_m: float = 5.0,
    single_segment_is_critical: bool = True,
    small_cluster_threshold: int = 5,
    far_distance_m: float = 100.0,
    id_column: str | None = None,
) -> IslandDetectionResult:
    """Detect disconnected components (islands) in a road network.

    Builds a graph from edge geometries using endpoint proximity, then
    finds connected components and classifies them by severity.

    Args:
        edges_gdf: GeoDataFrame with road edges (LineString geometries)
        snap_tolerance_m: Distance in meters to consider endpoints connected
        single_segment_is_critical: If True, single isolated segments are CRITICAL
        small_cluster_threshold: Clusters with <= this many edges are "small"
        far_distance_m: Distance in meters to consider "far from main network"
        id_column: Column to use as edge ID (auto-detected if None)

    Returns:
        IslandDetectionResult with component analysis
    """
    if len(edges_gdf) == 0:
        return IslandDetectionResult(
            total_edges=0,
            total_components=0,
            main_component_size=0,
            main_component_ratio=1.0,
            islands=[],
        )

    # Ensure metric CRS for distance calculations
    if edges_gdf.crs is None:
        edges_gdf = edges_gdf.set_crs("EPSG:4326")

    # Work in metric CRS
    metric_crs = edges_gdf.estimate_utm_crs()
    edges_metric = edges_gdf.to_crs(metric_crs)

    # Determine ID column
    id_col = id_column or _get_id_column(edges_gdf)

    # If no ID column exists, create one from index
    if id_col == "__idx__":
        edges_metric = edges_metric.reset_index(drop=True)
        edges_metric["__idx__"] = edges_metric.index

    # Build graph from edge geometries
    logger.info(f"Building graph from {len(edges_metric)} edges")
    G = _build_graph(edges_metric, id_col, snap_tolerance_m)

    # Find connected components
    components = list(nx.connected_components(G))
    logger.info(f"Found {len(components)} connected components")

    if len(components) == 0:
        return IslandDetectionResult(
            total_edges=len(edges_gdf),
            total_components=0,
            main_component_size=0,
            main_component_ratio=0.0,
            islands=[],
        )

    # Find main (largest) component
    main_component = max(components, key=len)
    main_edges = set(main_component)
    main_component_ratio = len(main_edges) / len(edges_gdf)

    # Analyze islands (non-main components)
    islands: list[Island] = []
    critical_count = 0
    warning_count = 0
    info_count = 0

    # Get main component centroid for distance calculations
    main_mask = edges_metric[id_col].isin(main_edges)
    if main_mask.any():
        main_union = edges_metric[main_mask].union_all()
    else:
        main_union = None

    for comp_id, component in enumerate(components):
        if component == main_component:
            continue  # Skip main component

        edge_ids = list(component)
        edge_count = len(edge_ids)

        # Get component edges
        comp_mask = edges_metric[id_col].isin(edge_ids)
        comp_edges = edges_metric[comp_mask]

        # Calculate total length
        total_length_m = comp_edges.geometry.length.sum()

        # Calculate centroid
        comp_union = comp_edges.union_all()
        centroid = comp_union.centroid

        # Calculate distance to main component
        if main_union is not None:
            min_distance_to_main_m = comp_union.distance(main_union)
        else:
            min_distance_to_main_m = None

        # Classify severity
        severity = _classify_severity(
            edge_count=edge_count,
            min_distance_m=min_distance_to_main_m,
            single_segment_is_critical=single_segment_is_critical,
            small_cluster_threshold=small_cluster_threshold,
            far_distance_m=far_distance_m,
        )

        if severity == IslandSeverity.CRITICAL:
            critical_count += 1
        elif severity == IslandSeverity.WARNING:
            warning_count += 1
        else:
            info_count += 1

        islands.append(
            Island(
                component_id=comp_id,
                edge_ids=edge_ids,
                severity=severity,
                edge_count=edge_count,
                total_length_m=total_length_m,
                centroid=centroid,
                min_distance_to_main_m=min_distance_to_main_m,
            )
        )

    # Sort islands by severity (critical first) then by edge count (smallest first)
    severity_order = {IslandSeverity.CRITICAL: 0, IslandSeverity.WARNING: 1, IslandSeverity.INFO: 2}
    islands.sort(key=lambda i: (severity_order[i.severity], i.edge_count))

    logger.info(
        f"Island analysis: {critical_count} critical, {warning_count} warning, {info_count} info"
    )

    return IslandDetectionResult(
        total_edges=len(edges_gdf),
        total_components=len(components),
        main_component_size=len(main_edges),
        main_component_ratio=main_component_ratio,
        islands=islands,
        critical_count=critical_count,
        warning_count=warning_count,
        info_count=info_count,
    )


def _build_graph(
    edges_gdf: gpd.GeoDataFrame,
    id_col: str,
    snap_tolerance_m: float,
) -> nx.Graph:
    """Build a graph from edge geometries using endpoint proximity.

    Creates a graph where edges are nodes and connections exist when
    edge endpoints are within snap_tolerance_m of each other.

    Args:
        edges_gdf: GeoDataFrame with edges in metric CRS
        id_col: Column containing edge IDs
        snap_tolerance_m: Distance tolerance for endpoint connections

    Returns:
        NetworkX graph with edge IDs as nodes
    """
    G = nx.Graph()

    # Collect all endpoints
    endpoints: list[tuple[float, float, str | int]] = []

    for _, row in edges_gdf.iterrows():
        geom = row.geometry
        edge_id = row[id_col]
        G.add_node(edge_id)

        if geom is None or geom.is_empty:
            continue

        # Get start and end points
        coords = list(geom.coords)
        if len(coords) >= 2:
            start = coords[0]
            end = coords[-1]
            endpoints.append((start[0], start[1], edge_id))
            endpoints.append((end[0], end[1], edge_id))

    # Build spatial index for fast endpoint matching
    # Use numpy for vectorized distance calculation
    if len(endpoints) == 0:
        return G

    points = np.array([(x, y) for x, y, _ in endpoints])
    edge_ids = [eid for _, _, eid in endpoints]

    # For each endpoint, find other endpoints within tolerance
    # This is O(n^2) but with numpy it's fast enough for most networks
    # For very large networks, consider using scipy.spatial.KDTree
    for i in range(len(points)):
        dists = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))
        nearby = np.where(dists <= snap_tolerance_m)[0]

        for j in nearby:
            # Connect different edges that share endpoints
            if edge_ids[i] != edge_ids[j]:
                G.add_edge(edge_ids[i], edge_ids[j])

    return G


def _classify_severity(
    edge_count: int,
    min_distance_m: float | None,
    single_segment_is_critical: bool,
    small_cluster_threshold: int,
    far_distance_m: float,
) -> IslandSeverity:
    """Classify island severity based on size and distance.

    Args:
        edge_count: Number of edges in the island
        min_distance_m: Distance to main network (None if unknown)
        single_segment_is_critical: Single segments are CRITICAL
        small_cluster_threshold: Max edges for "small" cluster
        far_distance_m: Distance threshold for "far from main"

    Returns:
        IslandSeverity classification
    """
    # Single isolated segment
    if edge_count == 1 and single_segment_is_critical:
        return IslandSeverity.CRITICAL

    # Small cluster far from main network
    if edge_count <= small_cluster_threshold:
        if min_distance_m is not None and min_distance_m > far_distance_m:
            return IslandSeverity.WARNING

    # Default: INFO (probably OK)
    return IslandSeverity.INFO


def _get_id_column(gdf: gpd.GeoDataFrame) -> str:
    """Determine the ID column for a GeoDataFrame."""
    for col in ["id", "ID", "edge_id"]:
        if col in gdf.columns:
            return col
    # If no ID column found, use the index by resetting it to a column
    return "__idx__"
