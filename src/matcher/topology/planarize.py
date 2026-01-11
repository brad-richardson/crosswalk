"""Topology reconstruction (planarization) for spaghetti road data.

Converts non-topological LineStrings into a planar graph where
lines only intersect at explicit node vertices.

Algorithm:
1. Explode MultiLineStrings to simple LineStrings
2. Filter intersections by z-level (bridge/tunnel awareness)
3. Find all geometric intersections via spatial self-join
4. Collect intersection points as new nodes
5. Split lines at intersection points
6. Snap undershoots/overshoots to nearby edges
7. Cluster nearby endpoints within tolerance
8. Build node/edge tables with connectivity
"""

from typing import NamedTuple, Optional

import geopandas as gpd
import numpy as np
from loguru import logger
from pyproj import CRS
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from shapely import LineString, Point, get_coordinates
from shapely.ops import linemerge, split, snap
from shapely.strtree import STRtree

from ..config import settings


def _parse_bool(value) -> bool:
    """Parse various boolean representations from data sources.

    Handles OSM-style values like "yes", "no", "true", "false",
    as well as Python booleans and numeric 0/1 values.

    Args:
        value: Value to parse (bool, int, float, str, or None)

    Returns:
        Boolean interpretation of the value
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # Only treat 1 as True, 0 as False; log unexpected values
        if value == 1:
            return True
        if value == 0:
            return False
        logger.debug(f"Unexpected numeric boolean value: {value}, treating as False")
        return False
    if isinstance(value, str):
        normalized = value.lower().strip()
        if normalized in ("yes", "true", "1"):
            return True
        if normalized in ("no", "false", "0", ""):
            return False
        logger.debug(f"Unexpected string boolean value: '{value}', treating as False")
        return False
    logger.debug(f"Unexpected boolean value type: {type(value).__name__}, treating as False")
    return False


def _ensure_projected_crs(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, Optional[CRS]]:
    """Ensure GeoDataFrame is in a projected CRS for metric operations.

    If the input is in a geographic CRS (e.g., EPSG:4326), it will be
    auto-projected to the appropriate UTM zone based on the centroid.

    Args:
        gdf: Input GeoDataFrame

    Returns:
        Tuple of (projected GeoDataFrame, original CRS or None)
    """
    original_crs = gdf.crs

    if original_crs is None:
        logger.warning("No CRS set on input data, assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
        original_crs = gdf.crs

    if original_crs.is_geographic:
        # Auto-detect UTM zone from centroid
        centroid = gdf.union_all().centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "N" if centroid.y >= 0 else "S"
        epsg_code = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone
        utm_crs = f"EPSG:{epsg_code}"

        logger.info(f"Auto-projecting from {original_crs} to {utm_crs} for metric operations")
        gdf = gdf.to_crs(utm_crs)

    return gdf, original_crs


class PlanarizedNetwork(NamedTuple):
    """Result of planarization."""

    nodes: gpd.GeoDataFrame  # node_id, geometry (Point)
    edges: gpd.GeoDataFrame  # edge_id, from_node, to_node, geometry (LineString), attrs


def planarize(
    lines: gpd.GeoDataFrame,
    snap_tolerance: float = None,
    node_cluster_tolerance: float = None,
    respect_z_levels: bool = True,
    id_column: str = "local_id",
) -> PlanarizedNetwork:
    """Planarize a spaghetti road network.

    Args:
        lines: GeoDataFrame with LineString geometries
        snap_tolerance: Distance to snap undershoots/overshoots (meters)
        node_cluster_tolerance: Tolerance for clustering nearby nodes (meters)
        respect_z_levels: Whether to respect bridge/tunnel z-levels
        id_column: Column name for original feature IDs

    Returns:
        PlanarizedNetwork with nodes and edges GeoDataFrames
    """
    snap_tolerance = snap_tolerance or settings.snap_tolerance
    node_cluster_tolerance = node_cluster_tolerance or settings.node_cluster_tolerance

    logger.info(f"Planarizing {len(lines)} features")
    logger.info(f"  snap_tolerance: {snap_tolerance}m")
    logger.info(f"  node_cluster_tolerance: {node_cluster_tolerance}m")
    logger.info(f"  respect_z_levels: {respect_z_levels}")

    # Ensure projected CRS for metric operations
    lines, original_crs = _ensure_projected_crs(lines)
    working_crs = lines.crs

    # Ensure we have an ID column
    if id_column not in lines.columns:
        lines = lines.copy()
        lines[id_column] = range(len(lines))

    # Step 1: Explode MultiLineStrings to simple LineStrings
    logger.info("Step 1: Exploding MultiLineStrings...")
    lines = lines.explode(index_parts=False).reset_index(drop=True)
    lines = lines[lines.geometry.type == "LineString"]
    logger.info(f"  After explode: {len(lines)} LineStrings")

    # Step 2: Find all intersections (respecting z-levels)
    logger.info("Step 2: Finding intersections...")
    intersection_points = _find_intersections(lines, respect_z_levels)
    logger.info(f"  Found {len(intersection_points)} intersection points")

    # Step 3: Collect all endpoints
    logger.info("Step 3: Collecting endpoints...")
    endpoints = _collect_endpoints(lines)
    logger.info(f"  Found {len(endpoints)} endpoints")

    # Step 4: Combine all candidate nodes
    all_node_points = endpoints + intersection_points

    # Step 5: Cluster nearby nodes
    logger.info("Step 4: Clustering nearby nodes...")
    clustered_nodes = _cluster_nodes(all_node_points, node_cluster_tolerance)
    logger.info(f"  After clustering: {len(clustered_nodes)} unique nodes")

    # Step 6: Split lines at intersection nodes
    logger.info("Step 5: Splitting lines at nodes...")
    split_edges = _split_lines_at_nodes(lines, clustered_nodes, id_column)
    logger.info(f"  After splitting: {len(split_edges)} edges")

    # Step 7: Snap undershoots/overshoots
    logger.info("Step 6: Snapping undershoots/overshoots...")
    split_edges, clustered_nodes = _snap_undershoots(
        split_edges, clustered_nodes, snap_tolerance
    )
    logger.info(f"  After snapping: {len(clustered_nodes)} nodes, {len(split_edges)} edges")

    # Step 8: Build node and edge GeoDataFrames with connectivity
    logger.info("Step 7: Building node/edge tables...")
    nodes_gdf, edges_gdf = _build_node_edge_tables(split_edges, clustered_nodes, working_crs)

    # Reproject back to original CRS if it was geographic
    if original_crs and original_crs.is_geographic:
        logger.info(f"Reprojecting output back to {original_crs}")
        nodes_gdf = nodes_gdf.to_crs(original_crs)
        edges_gdf = edges_gdf.to_crs(original_crs)

    logger.info(f"Planarization complete: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")
    return PlanarizedNetwork(nodes=nodes_gdf, edges=edges_gdf)


def _get_z_level(row) -> int:
    """Extract z-level from row attributes.

    Handles OSM-style values like bridge="yes", tunnel="no", layer=1.
    """
    # Check for explicit level/layer
    level = 0
    for col in ["level", "layer"]:
        if col in row.index and row[col] is not None:
            try:
                level = int(row[col])
                break
            except (ValueError, TypeError):
                pass

    # Bridge/tunnel flags override (using proper boolean parsing)
    is_bridge = _parse_bool(row.get("is_bridge")) or _parse_bool(row.get("bridge"))
    is_tunnel = _parse_bool(row.get("is_tunnel")) or _parse_bool(row.get("tunnel"))

    if is_bridge and level == 0:
        level = 1
    if is_tunnel and level == 0:
        level = -1

    return level


def should_intersect(row_a, row_b, respect_z_levels: bool) -> bool:
    """Check if two edges should form an intersection based on z-level."""
    if not respect_z_levels:
        return True

    level_a = _get_z_level(row_a)
    level_b = _get_z_level(row_b)

    return level_a == level_b


def _find_intersections(
    lines: gpd.GeoDataFrame, respect_z_levels: bool
) -> list[Point]:
    """Find all intersection points between lines (respecting z-levels).

    Detects both crossing intersections (X-junctions) and T-junctions
    where one line's endpoint touches another line.
    """
    tree = STRtree(lines.geometry.values)
    intersection_points = []

    for idx, row in lines.iterrows():
        geom = row.geometry
        candidates = tree.query(geom)

        for other_idx in candidates:
            if other_idx <= idx:
                continue  # Skip self and already-processed pairs

            other_row = lines.iloc[other_idx]
            other_geom = other_row.geometry

            # Check z-level compatibility
            if not should_intersect(row, other_row, respect_z_levels):
                continue

            # Check for crossing (interior intersection - X-junction)
            if geom.crosses(other_geom):
                isect = geom.intersection(other_geom)
                _collect_points(isect, intersection_points)

            # Check for T-junction (endpoint touches interior of other line)
            elif geom.touches(other_geom):
                touch_point = geom.intersection(other_geom)
                _collect_points(touch_point, intersection_points)

    return intersection_points


def _collect_points(geom, points_list: list[Point]) -> None:
    """Extract Point geometries from intersection result."""
    if isinstance(geom, Point):
        points_list.append(geom)
    elif hasattr(geom, "geoms"):
        # MultiPoint or GeometryCollection
        for part in geom.geoms:
            if isinstance(part, Point):
                points_list.append(part)


def _collect_endpoints(lines: gpd.GeoDataFrame) -> list[Point]:
    """Collect all endpoints from LineStrings."""
    endpoints = []
    for geom in lines.geometry:
        coords = get_coordinates(geom)
        if len(coords) >= 2:
            endpoints.append(Point(coords[0]))
            endpoints.append(Point(coords[-1]))
    return endpoints


def _cluster_nodes(points: list[Point], tolerance: float) -> list[Point]:
    """Cluster nearby points and return centroids."""
    if len(points) < 2:
        return points

    coords = np.array([[p.x, p.y] for p in points])

    # Handle duplicate coordinates
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

    if len(unique_coords) < 2:
        return [Point(unique_coords[0])]

    # Hierarchical clustering
    distances = pdist(unique_coords)
    linkage_matrix = linkage(distances, method="single")
    cluster_labels = fcluster(linkage_matrix, t=tolerance, criterion="distance")

    # Map back to original points
    full_labels = cluster_labels[inverse]

    # Compute cluster centroids
    unique_labels = np.unique(full_labels)
    centroids = []
    for label in unique_labels:
        mask = full_labels == label
        centroid = coords[mask].mean(axis=0)
        centroids.append(Point(centroid))

    return centroids


def _split_lines_at_nodes(
    lines: gpd.GeoDataFrame, nodes: list[Point], id_column: str
) -> list[dict]:
    """Split lines at node points."""
    from shapely import unary_union

    # Create a MultiPoint from all nodes for splitting
    if not nodes:
        # No nodes - return original lines as edges
        edges = []
        for idx, row in lines.iterrows():
            edges.append({
                "geometry": row.geometry,
                "original_id": row[id_column],
                **{k: row[k] for k in row.index if k not in ["geometry", id_column]},
            })
        return edges

    splitter = unary_union(nodes)

    edges = []
    for idx, row in lines.iterrows():
        geom = row.geometry
        original_attrs = {k: row[k] for k in row.index if k not in ["geometry", id_column]}

        try:
            # Snap geometry to splitter points first
            snapped_geom = snap(geom, splitter, tolerance=0.1)
            split_result = split(snapped_geom, splitter)

            for part in split_result.geoms:
                if isinstance(part, LineString) and part.length > 0.01:
                    edges.append({
                        "geometry": part,
                        "original_id": row[id_column],
                        **original_attrs,
                    })
        except Exception as e:
            # If split fails, keep original
            logger.warning(f"Split failed for line {idx}: {e}")
            edges.append({
                "geometry": geom,
                "original_id": row[id_column],
                **original_attrs,
            })

    return edges


def _snap_undershoots(
    edges: list[dict], nodes: list[Point], snap_tolerance: float
) -> tuple[list[dict], list[Point]]:
    """Snap dangling endpoints to nearby edges."""
    if not edges:
        return edges, nodes

    # Build tree of all edges
    edge_geoms = [e["geometry"] for e in edges]
    edge_tree = STRtree(edge_geoms)

    # Find dangling endpoints (appear only once)
    endpoint_counts = {}
    for edge in edges:
        coords = get_coordinates(edge["geometry"])
        start = (round(coords[0][0], 6), round(coords[0][1], 6))
        end = (round(coords[-1][0], 6), round(coords[-1][1], 6))
        endpoint_counts[start] = endpoint_counts.get(start, 0) + 1
        endpoint_counts[end] = endpoint_counts.get(end, 0) + 1

    # Endpoints with count = 1 are dangling
    dangling = {pt for pt, count in endpoint_counts.items() if count == 1}

    new_nodes = list(nodes)
    modified_edges = []

    for edge in edges:
        geom = edge["geometry"]
        coords = get_coordinates(geom)
        start = (round(coords[0][0], 6), round(coords[0][1], 6))
        end = (round(coords[-1][0], 6), round(coords[-1][1], 6))

        modified = False
        new_coords = list(coords)

        # Check if start is dangling
        if start in dangling:
            start_pt = Point(coords[0])
            snap_pt = _find_snap_point(start_pt, edge_geoms, edge_tree, snap_tolerance, geom)
            if snap_pt is not None:
                new_coords[0] = [snap_pt.x, snap_pt.y]
                new_nodes.append(snap_pt)
                modified = True

        # Check if end is dangling
        if end in dangling:
            end_pt = Point(coords[-1])
            snap_pt = _find_snap_point(end_pt, edge_geoms, edge_tree, snap_tolerance, geom)
            if snap_pt is not None:
                new_coords[-1] = [snap_pt.x, snap_pt.y]
                new_nodes.append(snap_pt)
                modified = True

        if modified:
            new_geom = LineString(new_coords)
            if new_geom.length > 0.01:
                edge = dict(edge)
                edge["geometry"] = new_geom
        modified_edges.append(edge)

    # Re-cluster nodes after adding snap points
    new_nodes = _cluster_nodes(new_nodes, snap_tolerance * 0.5)

    return modified_edges, new_nodes


def _find_snap_point(
    point: Point,
    edge_geoms: list[LineString],
    edge_tree: STRtree,
    tolerance: float,
    exclude_geom: LineString,
) -> Optional[Point]:
    """Find a point to snap to on nearby edges."""
    # Search for nearby edges
    search_buffer = point.buffer(tolerance)
    candidates = edge_tree.query(search_buffer)

    best_dist = tolerance
    best_point = None

    for idx in candidates:
        edge_geom = edge_geoms[idx]

        # Don't snap to self
        if edge_geom.equals(exclude_geom):
            continue

        # Find nearest point on edge
        dist = edge_geom.distance(point)
        if dist < best_dist and dist > 0.01:  # Don't snap if already touching
            # Project point onto line
            nearest_pt = edge_geom.interpolate(edge_geom.project(point))
            best_dist = dist
            best_point = nearest_pt

    return best_point


def _build_node_edge_tables(
    edges: list[dict], nodes: list[Point], crs: CRS
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build node and edge GeoDataFrames with connectivity.

    Args:
        edges: List of edge dictionaries with geometry and attributes
        nodes: List of node Point geometries
        crs: CRS to set on output GeoDataFrames

    Returns:
        Tuple of (nodes_gdf, edges_gdf) with proper CRS set
    """
    # Create node lookup tree
    node_tree = STRtree(nodes)

    # Build nodes GeoDataFrame with explicit CRS
    nodes_gdf = gpd.GeoDataFrame(
        {"node_id": range(len(nodes))},
        geometry=nodes,
        crs=crs,
    )

    # Assign from_node and to_node to each edge
    for edge in edges:
        geom = edge["geometry"]
        coords = get_coordinates(geom)

        start = Point(coords[0])
        end = Point(coords[-1])

        # Find nearest node
        start_idx = node_tree.nearest(start)
        end_idx = node_tree.nearest(end)

        edge["from_node"] = start_idx
        edge["to_node"] = end_idx

    # Build edges GeoDataFrame with explicit CRS
    edges_gdf = gpd.GeoDataFrame(edges, crs=crs)
    edges_gdf["edge_id"] = range(len(edges_gdf))

    return nodes_gdf, edges_gdf
