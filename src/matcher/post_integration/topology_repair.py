"""Topology repair for post-integration analysis.

Provides tools for repairing common topology issues in integrated road networks:
- Snap endpoints: Fix undershoots/overshoots within tolerance
- Remove islands: Delete critical-severity disconnected components
"""

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.geometry import LineString, Point
from shapely.ops import snap

from .island_detector import IslandSeverity, detect_islands


@dataclass
class RepairResult:
    """Result of topology repair operations."""

    original_edge_count: int
    final_edge_count: int
    edges_snapped: int
    edges_removed: int
    islands_removed: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_edge_count": self.original_edge_count,
            "final_edge_count": self.final_edge_count,
            "edges_snapped": self.edges_snapped,
            "edges_removed": self.edges_removed,
            "islands_removed": self.islands_removed,
        }


def repair_topology(
    edges_gdf: gpd.GeoDataFrame,
    snap_tolerance_m: float = 5.0,
    remove_critical_islands: bool = True,
    id_column: str | None = None,
) -> tuple[gpd.GeoDataFrame, RepairResult]:
    """Repair topology issues in a road network.

    Performs the following repairs:
    1. Snap endpoints that are close but not connected
    2. Remove critical-severity islands (single isolated segments)

    Args:
        edges_gdf: GeoDataFrame with road edges
        snap_tolerance_m: Distance tolerance for endpoint snapping
        remove_critical_islands: If True, remove single isolated segments
        id_column: Column to use as edge ID (auto-detected if None)

    Returns:
        Tuple of (repaired_gdf, repair_result)
    """
    if len(edges_gdf) == 0:
        return edges_gdf.copy(), RepairResult(
            original_edge_count=0,
            final_edge_count=0,
            edges_snapped=0,
            edges_removed=0,
            islands_removed=0,
        )

    original_count = len(edges_gdf)
    logger.info(f"Starting topology repair on {original_count} edges")

    # Work with a copy
    repaired_gdf = edges_gdf.copy()

    # Determine ID column
    id_col = id_column or _get_id_column(repaired_gdf)

    # Step 1: Snap endpoints
    repaired_gdf, edges_snapped = snap_endpoints(repaired_gdf, snap_tolerance_m, id_col)

    # Step 2: Remove critical islands
    edges_removed = 0
    islands_removed = 0

    if remove_critical_islands:
        # Detect islands
        island_result = detect_islands(
            repaired_gdf,
            snap_tolerance_m=snap_tolerance_m,
            single_segment_is_critical=True,
            id_column=id_col,
        )

        # Collect critical island edge IDs
        critical_edge_ids: set[str | int] = set()
        for island in island_result.islands:
            if island.severity == IslandSeverity.CRITICAL:
                critical_edge_ids.update(island.edge_ids)
                islands_removed += 1

        if critical_edge_ids:
            mask = ~repaired_gdf[id_col].isin(critical_edge_ids)
            edges_removed = len(critical_edge_ids)
            repaired_gdf = repaired_gdf[mask].copy()
            logger.info(f"Removed {edges_removed} edges from {islands_removed} critical islands")

    final_count = len(repaired_gdf)

    logger.info(
        f"Topology repair complete: {edges_snapped} snapped, "
        f"{edges_removed} removed, {final_count} edges remaining"
    )

    return repaired_gdf, RepairResult(
        original_edge_count=original_count,
        final_edge_count=final_count,
        edges_snapped=edges_snapped,
        edges_removed=edges_removed,
        islands_removed=islands_removed,
    )


def snap_endpoints(
    edges_gdf: gpd.GeoDataFrame,
    tolerance_m: float = 5.0,
    id_column: str | None = None,
) -> tuple[gpd.GeoDataFrame, int]:
    """Snap edge endpoints that are close but not connected.

    For each edge endpoint, if there's another endpoint within tolerance
    that it's not already connected to, snap to the nearest one.

    Args:
        edges_gdf: GeoDataFrame with road edges
        tolerance_m: Distance tolerance for snapping
        id_column: Column to use as edge ID

    Returns:
        Tuple of (snapped_gdf, edges_snapped_count)
    """
    if len(edges_gdf) == 0:
        return edges_gdf.copy(), 0

    # Ensure metric CRS
    original_crs = edges_gdf.crs
    if original_crs is None:
        edges_gdf = edges_gdf.set_crs("EPSG:4326")
        original_crs = edges_gdf.crs

    metric_crs = edges_gdf.estimate_utm_crs()
    edges_metric = edges_gdf.to_crs(metric_crs)

    id_col = id_column or _get_id_column(edges_gdf)

    # Collect all endpoints
    endpoints: list[dict[str, Any]] = []
    for idx, row in edges_metric.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        coords = list(geom.coords)
        if len(coords) >= 2:
            endpoints.append(
                {
                    "idx": idx,
                    "edge_id": row[id_col],
                    "point": Point(coords[0]),
                    "is_start": True,
                }
            )
            endpoints.append(
                {
                    "idx": idx,
                    "edge_id": row[id_col],
                    "point": Point(coords[-1]),
                    "is_start": False,
                }
            )

    if len(endpoints) < 2:
        return edges_gdf.to_crs(original_crs), 0

    # Build endpoint array for fast distance calculation
    points_array = np.array([[ep["point"].x, ep["point"].y] for ep in endpoints])

    # Find endpoints that need snapping
    snapped_edges: set[Any] = set()
    snap_targets: dict[Any, Point] = {}  # idx -> target point

    for i, ep in enumerate(endpoints):
        # Calculate distances to all other endpoints
        dists = np.sqrt(np.sum((points_array - points_array[i]) ** 2, axis=1))

        # Find endpoints within tolerance (excluding self and same-edge endpoints)
        for j, d in enumerate(dists):
            if i >= j:  # Skip self and already-processed pairs
                continue
            if d > tolerance_m or d == 0:
                continue
            if endpoints[j]["edge_id"] == ep["edge_id"]:
                continue

            # Check if these endpoints are already connected (same coordinates)
            if d < 0.001:  # Already connected
                continue

            # Snap the endpoint of the edge that has more endpoints to snap
            # (or arbitrary if equal)
            snapped_edges.add(ep["idx"])
            snap_targets[ep["idx"]] = endpoints[j]["point"]
            break

    if not snapped_edges:
        return edges_gdf.to_crs(original_crs), 0

    # Apply snapping
    def snap_geometry(row: Any) -> LineString:
        idx = row.name
        geom = row.geometry

        if idx not in snap_targets or geom is None:
            return geom

        target = snap_targets[idx]
        # Use shapely snap function
        snapped = snap(geom, target, tolerance_m)
        return snapped

    edges_metric["geometry"] = edges_metric.apply(snap_geometry, axis=1)

    # Convert back to original CRS
    result_gdf = edges_metric.to_crs(original_crs)

    return result_gdf, len(snapped_edges)


def _get_id_column(gdf: gpd.GeoDataFrame) -> str:
    """Determine the ID column for a GeoDataFrame."""
    for col in ["id", "ID", "edge_id"]:
        if col in gdf.columns:
            return col
    return gdf.index.name or "index"
