"""Orphan detection for integrated networks.

Identifies disconnected components (orphans) that are not connected
to the reference network, flagging them for QA review.
"""

from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from shapely import Point
from shapely.strtree import STRtree

from ..topology.graph import build_graph, find_connected_components
from ..topology.planarize import PlanarizedNetwork
from .provenance import ComponentStatus, EdgeSource


def detect_orphans_by_proximity(
    combined_gdf: gpd.GeoDataFrame,
    connection_tolerance: float = 75.0,  # Match buffer_distance from matching pipeline
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Identify orphan segments based on endpoint proximity to reference network.

    A segment is considered connected if at least one of its endpoints is within
    connection_tolerance of any reference segment. Target segments not connected
    to the reference network are flagged as orphans.

    This is a simpler approach that doesn't require planarization - it just checks
    if target endpoints are near the reference network.

    Args:
        combined_gdf: Combined GeoDataFrame with _source column
        connection_tolerance: Distance in meters to consider "connected"

    Returns:
        Tuple of:
        - main_edges: Reference edges + connected target edges
        - orphan_edges: Target edges not connected to reference
        - stats: Statistics
    """
    logger.info("Detecting orphans by endpoint proximity...")
    logger.info(f"  Connection tolerance: {connection_tolerance}m")

    # Work in a projected CRS for accurate distance calculations
    original_crs = combined_gdf.crs
    working_crs = original_crs
    if original_crs is not None and original_crs.is_geographic:
        working_crs = combined_gdf.estimate_utm_crs()
        logger.info(f"  Using projected CRS for distance calculations: {working_crs}")
        combined_gdf = combined_gdf.to_crs(working_crs)

    # Separate reference and target segments
    ref_mask = combined_gdf["_source"] == EdgeSource.REFERENCE.value
    reference_edges = combined_gdf[ref_mask].copy()
    target_edges = combined_gdf[~ref_mask].copy()

    logger.info(f"  Reference edges: {len(reference_edges)}")
    logger.info(f"  Target edges: {len(target_edges)}")

    if len(target_edges) == 0:
        # No targets, everything is main
        main_edges = reference_edges.copy()
        main_edges["component_status"] = ComponentStatus.MAIN.value
        main_edges["is_connected"] = True
        orphan_edges = gpd.GeoDataFrame(columns=main_edges.columns, crs=combined_gdf.crs)
        stats = {
            "total_segments": len(combined_gdf),
            "reference_edges": len(reference_edges),
            "connected_target_edges": 0,
            "orphan_edges": 0,
        }
        return main_edges, orphan_edges, stats

    # Build spatial index of reference segments
    ref_geoms = reference_edges.geometry.values
    ref_tree = STRtree(ref_geoms)

    # Check each target segment's endpoints for proximity to reference
    connected_mask = []
    min_distances = []

    for idx, row in target_edges.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            connected_mask.append(False)
            min_distances.append(np.nan)
            continue

        # Get endpoints (handle both LineString and MultiLineString)
        try:
            if geom.geom_type == "MultiLineString":
                # For MultiLineString, get first point of first part and last point of last part
                first_part = geom.geoms[0]
                last_part = geom.geoms[-1]
                start_pt = Point(first_part.coords[0])
                end_pt = Point(last_part.coords[-1])
            else:
                coords = list(geom.coords)
                start_pt = Point(coords[0])
                end_pt = Point(coords[-1])
        except Exception:
            connected_mask.append(False)
            min_distances.append(np.nan)
            continue

        # Find nearest reference segment to each endpoint
        start_nearest_idx = ref_tree.nearest(start_pt)
        end_nearest_idx = ref_tree.nearest(end_pt)

        start_dist = start_pt.distance(ref_geoms[start_nearest_idx])
        end_dist = end_pt.distance(ref_geoms[end_nearest_idx])

        min_dist = min(start_dist, end_dist)
        min_distances.append(min_dist)

        # Connected if either endpoint is within tolerance
        is_connected = min_dist <= connection_tolerance
        connected_mask.append(is_connected)

    target_edges["is_connected"] = connected_mask
    target_edges["nearest_ref_distance"] = min_distances

    # Split into connected and orphan
    connected_targets = target_edges[target_edges["is_connected"]].copy()
    orphan_targets = target_edges[~target_edges["is_connected"]].copy()

    logger.info(f"  Connected target edges: {len(connected_targets)}")
    logger.info(f"  Orphan target edges: {len(orphan_targets)}")

    # Build main edges (reference + connected targets)
    reference_edges["component_status"] = ComponentStatus.MAIN.value
    reference_edges["is_connected"] = True
    reference_edges["nearest_ref_distance"] = 0.0

    connected_targets["component_status"] = ComponentStatus.MAIN.value

    main_edges = gpd.GeoDataFrame(
        pd.concat([reference_edges, connected_targets], ignore_index=True),
        crs=combined_gdf.crs,
    )

    # Mark orphans
    orphan_targets["component_status"] = ComponentStatus.ORPHAN.value

    # Add QA priority to orphans
    if len(orphan_targets) > 0:
        orphan_targets = _add_orphan_qa_priority(orphan_targets, main_edges)

    stats = {
        "total_segments": len(combined_gdf),
        "reference_edges": len(reference_edges),
        "connected_target_edges": len(connected_targets),
        "orphan_edges": len(orphan_targets),
    }

    # Convert back to original CRS if needed
    if original_crs is not None and working_crs != original_crs:
        logger.info(f"  Converting output back to original CRS: {original_crs}")
        main_edges = main_edges.to_crs(original_crs)
        if len(orphan_targets) > 0:
            orphan_targets = orphan_targets.to_crs(original_crs)

    logger.info(f"Orphan detection complete: {stats}")

    return main_edges, orphan_targets, stats


def detect_orphan_components(
    network: PlanarizedNetwork,
    edges_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Identify orphan components not connected to reference network.

    An orphan component is a connected subgraph that contains no reference edges.
    These are flagged for QA review as they may be:
    - Legitimate new infrastructure
    - Data errors or noise
    - Incorrectly unmatched segments

    Args:
        network: PlanarizedNetwork with nodes and edges
        edges_gdf: Edges GeoDataFrame with provenance columns (_source)

    Returns:
        Tuple of:
        - main_edges: Edges in the main (reference-connected) component
        - orphan_edges: Edges in orphan components (for QA)
        - stats: Statistics about components
    """
    logger.info("Detecting orphan components...")

    # Build graph
    G = build_graph(network)

    # Find connected components
    components = find_connected_components(G)
    logger.info(f"Found {len(components)} connected components")

    # Identify which edges are from reference
    reference_edge_ids = set()
    for idx, row in edges_gdf.iterrows():
        if row.get("_source") == EdgeSource.REFERENCE.value:
            reference_edge_ids.add(row["edge_id"])

    logger.info(f"Reference edges: {len(reference_edge_ids)}")

    # Map edges to their components
    edge_to_component = _map_edges_to_components(network.edges, components, G)

    # Identify main components (containing reference edges)
    main_component_ids = set()
    orphan_component_ids = set()

    for comp_id, node_set in enumerate(components):
        # Get edges in this component
        comp_edges = [
            eid for eid, cid in edge_to_component.items()
            if cid == comp_id
        ]

        # Check if any edge is from reference
        has_reference = any(eid in reference_edge_ids for eid in comp_edges)

        if has_reference:
            main_component_ids.add(comp_id)
        else:
            orphan_component_ids.add(comp_id)

    logger.info(
        f"Main components: {len(main_component_ids)}, "
        f"Orphan components: {len(orphan_component_ids)}"
    )

    # Annotate edges with component info
    annotated_edges = _annotate_edges_with_components(
        edges_gdf,
        edge_to_component,
        main_component_ids,
        orphan_component_ids,
        components,
    )

    # Separate main and orphan edges
    main_mask = annotated_edges["component_status"] == ComponentStatus.MAIN.value
    main_edges = annotated_edges[main_mask].copy()
    orphan_edges = annotated_edges[~main_mask].copy()

    # Add QA priority to orphan edges
    if len(orphan_edges) > 0:
        orphan_edges = _add_orphan_qa_priority(orphan_edges, main_edges)

    # Calculate stats
    stats = {
        "total_components": len(components),
        "main_components": len(main_component_ids),
        "orphan_components": len(orphan_component_ids),
        "main_edges": len(main_edges),
        "orphan_edges": len(orphan_edges),
    }

    logger.info(f"Component detection complete: {stats}")

    return main_edges, orphan_edges, stats


def _map_edges_to_components(
    edges: gpd.GeoDataFrame,
    components: list[set[int]],
    G: nx.Graph,
) -> dict[int, int]:
    """Map edge IDs to their component index."""
    # Build node to component mapping
    node_to_component = {}
    for comp_id, node_set in enumerate(components):
        for node in node_set:
            node_to_component[node] = comp_id

    # Map edges to components via their nodes
    edge_to_component = {}
    for _, row in edges.iterrows():
        edge_id = row["edge_id"]
        from_node = row["from_node"]
        to_node = row["to_node"]

        # Edge belongs to same component as its nodes
        comp_id = node_to_component.get(from_node)
        if comp_id is None:
            comp_id = node_to_component.get(to_node)

        if comp_id is not None:
            edge_to_component[edge_id] = comp_id

    return edge_to_component


def _annotate_edges_with_components(
    edges_gdf: gpd.GeoDataFrame,
    edge_to_component: dict[int, int],
    main_component_ids: set[int],
    orphan_component_ids: set[int],
    components: list[set[int]],
) -> gpd.GeoDataFrame:
    """Add component annotations to edges GeoDataFrame."""
    annotated = edges_gdf.copy()

    # Compute component sizes (number of edges in each)
    component_sizes = {}
    for edge_id, comp_id in edge_to_component.items():
        component_sizes[comp_id] = component_sizes.get(comp_id, 0) + 1

    # Add columns
    annotated["component_id"] = annotated["edge_id"].map(edge_to_component)
    annotated["component_status"] = annotated["component_id"].apply(
        lambda cid: ComponentStatus.MAIN.value if cid in main_component_ids
        else ComponentStatus.ORPHAN.value
    )
    annotated["component_size"] = annotated["component_id"].map(component_sizes)

    return annotated


def _add_orphan_qa_priority(
    orphan_edges: gpd.GeoDataFrame,
    main_edges: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add QA priority and nearest main distance to orphan edges."""
    orphan_edges = orphan_edges.copy()

    # Compute nearest distance to main network
    if len(main_edges) > 0:
        main_geoms = main_edges.geometry.values
        tree = STRtree(main_geoms)

        distances = []
        for geom in orphan_edges.geometry:
            nearest_idx = tree.nearest(geom)
            dist = geom.distance(main_geoms[nearest_idx])
            distances.append(dist)

        orphan_edges["nearest_main_distance"] = distances
    else:
        orphan_edges["nearest_main_distance"] = np.nan

    # Compute QA priority based on length and characteristics
    def compute_priority(row):
        length = row.geometry.length if hasattr(row.geometry, "length") else 0

        # Longer segments are higher priority
        if length > 50:
            return "high"
        elif length > 10:
            return "medium"
        else:
            return "low"

    orphan_edges["qa_priority"] = orphan_edges.apply(compute_priority, axis=1)

    return orphan_edges


def annotate_nodes_with_components(
    nodes_gdf: gpd.GeoDataFrame,
    components: list[set[int]],
    main_component_ids: set[int],
) -> gpd.GeoDataFrame:
    """Add component annotations to nodes GeoDataFrame."""
    annotated = nodes_gdf.copy()

    # Build node to component mapping
    node_to_component = {}
    for comp_id, node_set in enumerate(components):
        for node in node_set:
            node_to_component[node] = comp_id

    # Add columns
    annotated["component_id"] = annotated["node_id"].map(node_to_component)
    annotated["component_status"] = annotated["component_id"].apply(
        lambda cid: ComponentStatus.MAIN.value if cid in main_component_ids
        else ComponentStatus.ORPHAN.value
    )

    return annotated
