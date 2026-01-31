"""Orphan detection for integrated networks.

Identifies disconnected components (orphans) that are not connected
to the reference network, flagging them for QA review.

This module focuses purely on connectivity analysis. Pre-integration
screening (fringe detection, water/building intersection) is handled
by the screen module.
"""

from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from shapely import Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

from ..topology.graph import build_graph, find_connected_components
from ..topology.planarize import PlanarizedNetwork
from .provenance import ComponentStatus, EdgeSource


def _get_endpoints(geom) -> list[Point]:
    """Extract start and end points from a LineString geometry.

    Args:
        geom: LineString geometry (MultiLineStrings filtered at ingest)

    Returns:
        List of Point objects for the endpoints
    """
    if geom is None or geom.is_empty:
        return []

    try:
        coords = list(geom.coords)
        return [Point(coords[0]), Point(coords[-1])]
    except Exception:
        return []


def propagate_transitive_connectivity(
    connected_targets: gpd.GeoDataFrame,
    orphan_targets: gpd.GeoDataFrame,
    connection_tolerance_m: float,
    max_hops: int = 2,
    debug: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Propagate connectivity through transitive connections.

    Segments not directly connected to reference may be connected via other
    target segments. This performs iterative wave propagation to find all
    transitively connected segments.

    Args:
        connected_targets: GeoDataFrame of targets directly connected to reference (hop 0)
        orphan_targets: GeoDataFrame of targets not connected to reference
        connection_tolerance_m: Distance in meters to consider segments connected
        max_hops: Maximum number of hops from reference (default 2)
        debug: Enable debug logging

    Returns:
        Tuple of:
        - Updated connected_targets with transitively connected segments
        - Updated orphan_targets with remaining disconnected segments
    """
    if len(orphan_targets) == 0 or len(connected_targets) == 0 or max_hops < 1:
        return connected_targets, orphan_targets

    # Initialize hop tracking
    if "_connectivity_hop" not in connected_targets.columns:
        connected_targets = connected_targets.copy()
        connected_targets["_connectivity_hop"] = 0

    orphan_targets = orphan_targets.copy()
    all_connected = [connected_targets]
    remaining_orphans = orphan_targets

    for hop in range(1, max_hops + 1):
        if len(remaining_orphans) == 0:
            break

        # For each hop, check against ALL connected so far (not just frontier)
        # This ensures we don't miss connections to earlier hops
        all_connected_so_far = gpd.GeoDataFrame(
            pd.concat(all_connected, ignore_index=True),
            crs=connected_targets.crs,
        )

        if len(all_connected_so_far) == 0:
            break

        # Build STRtree from all connected geometries
        connected_geoms = all_connected_so_far.geometry.values
        connected_tree = STRtree(connected_geoms)

        # Check each remaining orphan for connection to connected segments
        newly_connected_indices = []
        closest_distances = []  # For debug logging
        for idx, row in remaining_orphans.iterrows():
            endpoints = _get_endpoints(row.geometry)
            if not endpoints:
                if debug:
                    logger.debug(f"    Orphan {idx}: no endpoints extracted")
                continue

            # Check if any endpoint is near the connected segments
            min_dist = float("inf")
            for pt in endpoints:
                nearest_idx = connected_tree.nearest(pt)
                dist = pt.distance(connected_geoms[nearest_idx])
                min_dist = min(min_dist, dist)
                if dist <= connection_tolerance_m:
                    newly_connected_indices.append(idx)
                    break

            closest_distances.append((idx, min_dist))

        if debug and closest_distances:
            # Log the closest distances for remaining orphans
            sorted_dists = sorted(closest_distances, key=lambda x: x[1])[:5]
            logger.debug(f"    Hop {hop} closest orphan distances: {sorted_dists}")

        if not newly_connected_indices:
            # No new connections at this hop level
            if debug:
                logger.debug(
                    f"    Hop {hop}: no connections within tolerance {connection_tolerance_m}m"
                )
            break

        # Move newly connected from orphans to connected
        newly_connected = remaining_orphans.loc[newly_connected_indices].copy()
        newly_connected["_connectivity_hop"] = hop
        newly_connected["is_connected"] = True

        all_connected.append(newly_connected)
        remaining_orphans = remaining_orphans.drop(newly_connected_indices)

        logger.info(f"  Hop {hop}: {len(newly_connected)} segments transitively connected")

    # Combine all connected segments
    if len(all_connected) > 1:
        connected_targets = gpd.GeoDataFrame(
            pd.concat(all_connected, ignore_index=True),
            crs=connected_targets.crs,
        )

    return connected_targets, remaining_orphans


def detect_orphans_by_proximity(
    combined_gdf: gpd.GeoDataFrame,
    connection_tolerance_m: float = 3.0,
    min_merge_length_m: float = 20.0,
    net_new_buffer_m: float = 5.0,
    max_hops: int = 2,
    transitive_tolerance_m: float | None = None,
    debug_connectivity: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame | None, dict[str, Any]]:
    """Identify orphan segments based on endpoint proximity to reference network.

    A segment is considered connected if at least one of its endpoints is within
    connection_tolerance_m of any reference segment. Target segments not connected
    to the reference network are flagged as orphans.

    Transitive connectivity: If trail A connects to a road, and trail B connects
    to trail A (but not the road), trail B is also considered connected via
    transitive connectivity (up to max_hops).

    Additionally, connected segments shorter than min_merge_length_m are treated
    as orphans since they don't add meaningful new coverage.

    Note: Pre-integration screening (fringe detection, water/building checks) should
    be performed before calling this function using the screen module.

    Args:
        combined_gdf: Combined GeoDataFrame with _source column
        connection_tolerance_m: Distance in meters to consider "connected"
        min_merge_length_m: Minimum segment length (meters) to merge into network.
            Connected segments shorter than this are treated as orphans.
        net_new_buffer_m: Buffer distance (meters) around reference for net-new calculation.
            Segments within this buffer are considered "covered" by reference.
        max_hops: Maximum transitive connectivity hops from reference (default 2).
            0 = only direct connections, 1 = direct + 1 hop, 2 = direct + 2 hops.
        transitive_tolerance_m: Tolerance (meters) for transitive connections between
            target segments. Defaults to 2x connection_tolerance_m since trails often
            don't share exact endpoints. Set to connection_tolerance_m for strict mode.
        debug_connectivity: Enable debug logging for transitive connectivity analysis.

    Returns:
        Tuple of:
        - main_edges: Reference edges + connected target edges meeting length requirement
        - orphan_edges: Target edges not connected or too short
        - net_new_edges: GeoDataFrame with net-new geometry portions (for visualization)
        - stats: Statistics including connectivity hop breakdown
    """
    # Set default transitive tolerance to 2x connection tolerance
    if transitive_tolerance_m is None:
        transitive_tolerance_m = connection_tolerance_m * 2

    logger.info("Detecting orphans by endpoint proximity...")
    logger.info(f"  Connection tolerance: {connection_tolerance_m}m")
    logger.info(f"  Transitive tolerance: {transitive_tolerance_m}m")
    logger.info(f"  Min merge length: {min_merge_length_m}m")
    logger.info(f"  Net-new buffer: {net_new_buffer_m}m")
    logger.info(f"  Max transitive hops: {max_hops}")

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
        return main_edges, orphan_edges, None, stats

    # Build spatial index of reference segments
    ref_geoms = reference_edges.geometry.values
    ref_tree = STRtree(ref_geoms)

    # Check each target segment's endpoints for proximity to reference
    connected_mask = []
    min_distances = []

    for _idx, row in target_edges.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            connected_mask.append(False)
            min_distances.append(np.nan)
            continue

        # Get endpoints (LineStrings only, MultiLineStrings filtered at ingest)
        try:
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
        is_connected = min_dist <= connection_tolerance_m
        connected_mask.append(is_connected)

    target_edges["is_connected"] = connected_mask
    target_edges["nearest_ref_distance"] = min_distances

    # Split into directly connected and initially orphan
    connected_targets = target_edges[target_edges["is_connected"]].copy()
    orphan_targets = target_edges[~target_edges["is_connected"]].copy()

    # Initialize connectivity hop for directly connected targets (hop 0)
    connected_targets["_connectivity_hop"] = 0

    logger.info(f"  Directly connected target edges (hop 0): {len(connected_targets)}")
    logger.info(f"  Initially orphan target edges: {len(orphan_targets)}")

    # Propagate transitive connectivity
    if max_hops > 0 and len(orphan_targets) > 0 and len(connected_targets) > 0:
        logger.info(f"  Propagating transitive connectivity (max {max_hops} hops)...")
        connected_targets, orphan_targets = propagate_transitive_connectivity(
            connected_targets=connected_targets,
            orphan_targets=orphan_targets,
            connection_tolerance_m=transitive_tolerance_m,
            max_hops=max_hops,
            debug=debug_connectivity,
        )

    logger.info(f"  Connected target edges (after transitive): {len(connected_targets)}")
    logger.info(f"  Orphan target edges (disconnected): {len(orphan_targets)}")

    # Filter connected targets by minimum NET NEW length
    net_new_edges = None
    if min_merge_length_m > 0 and len(connected_targets) > 0:
        # Create a buffer around reference network to define "existing coverage"
        logger.info(f"  Computing net-new lengths (coverage buffer: {net_new_buffer_m}m)...")

        ref_union = unary_union(reference_edges.geometry.values)
        ref_buffered = ref_union.buffer(net_new_buffer_m)

        # Compute net-new length and geometry for each connected target
        net_new_lengths = []
        net_new_geoms = []
        for geom in connected_targets.geometry:
            if geom is None or geom.is_empty:
                net_new_lengths.append(0.0)
                net_new_geoms.append(None)
                continue

            # Subtract the reference coverage from the target segment
            net_new_geom = geom.difference(ref_buffered)

            if net_new_geom.is_empty:
                net_new_lengths.append(0.0)
                net_new_geoms.append(None)
            else:
                net_new_lengths.append(net_new_geom.length)
                net_new_geoms.append(net_new_geom)

        connected_targets["_net_new_length_m"] = net_new_lengths
        connected_targets["_total_length_m"] = connected_targets.geometry.length
        connected_targets["_net_new_geometry"] = net_new_geoms

        long_enough = connected_targets["_net_new_length_m"] >= min_merge_length_m
        too_short = connected_targets[~long_enough].copy()

        if len(too_short) > 0:
            too_short["unmatched_reason"] = "insufficient_net_new_length"
            orphan_targets = gpd.GeoDataFrame(
                pd.concat([orphan_targets, too_short], ignore_index=True),
                crs=combined_gdf.crs,
            )
            avg_net_new = too_short["_net_new_length_m"].mean()
            logger.info(
                f"  Insufficient net-new length (<{min_merge_length_m}m): {len(too_short)} "
                f"(avg net-new: {avg_net_new:.1f}m)"
            )

        connected_targets = connected_targets[long_enough].copy()

        # Build net-new edges GeoDataFrame for visualization
        if len(connected_targets) > 0:
            net_new_records = []
            for _, row in connected_targets.iterrows():
                full_geom = row.geometry
                if full_geom is not None and not full_geom.is_empty:
                    net_new_records.append(
                        {
                            "geometry": full_geom,
                            "_original_id": row.get("_original_id"),
                            "_source_dataset": row.get("_source_dataset"),
                            "_net_new_length_m": row.get("_net_new_length_m"),
                            "_total_length_m": row.get("_total_length_m"),
                            "_connectivity_hop": row.get("_connectivity_hop", 0),
                        }
                    )
            if net_new_records:
                net_new_edges = gpd.GeoDataFrame(net_new_records, crs=combined_gdf.crs)
                logger.info(f"  Net-new edges for visualization: {len(net_new_edges)}")

    logger.info(f"  Connected target edges (after length filter): {len(connected_targets)}")
    logger.info(f"  Orphan edges (disconnected/too short): {len(orphan_targets)}")

    # Build main edges (reference + connected targets)
    reference_edges["component_status"] = ComponentStatus.MAIN.value
    reference_edges["is_connected"] = True
    reference_edges["nearest_ref_distance"] = 0.0

    connected_targets["component_status"] = ComponentStatus.MAIN.value

    main_edges = gpd.GeoDataFrame(
        pd.concat([reference_edges, connected_targets], ignore_index=True),
        crs=combined_gdf.crs,
    )

    # Drop internal columns that can't be serialized
    drop_cols = ["_net_new_geometry"]
    for col in drop_cols:
        if col in main_edges.columns:
            main_edges = main_edges.drop(columns=[col])

    # Mark orphans
    orphan_targets["component_status"] = ComponentStatus.ORPHAN.value

    # Set unmatched_reason for truly disconnected segments (no reason yet)
    if "unmatched_reason" in orphan_targets.columns:
        no_reason_mask = orphan_targets["unmatched_reason"].isna()
        if no_reason_mask.any():
            orphan_targets.loc[no_reason_mask, "unmatched_reason"] = "not_connected_to_network"
    else:
        orphan_targets["unmatched_reason"] = "not_connected_to_network"

    # Drop internal columns from orphans too
    for col in drop_cols:
        if col in orphan_targets.columns:
            orphan_targets = orphan_targets.drop(columns=[col])

    # Add QA priority to orphans
    if len(orphan_targets) > 0:
        orphan_targets = _add_orphan_qa_priority(orphan_targets, main_edges)

    # Count how many were filtered for being too short
    too_short_count = (
        len(orphan_targets[orphan_targets.get("unmatched_reason") == "insufficient_net_new_length"])
        if "unmatched_reason" in orphan_targets.columns
        else 0
    )

    # Count connectivity by hop level
    hop_counts = {}
    if "_connectivity_hop" in connected_targets.columns:
        for hop in range(max_hops + 1):
            hop_counts[f"hop_{hop}_connected"] = int(
                (connected_targets["_connectivity_hop"] == hop).sum()
            )

    stats = {
        "total_segments": len(combined_gdf),
        "reference_edges": len(reference_edges),
        "connected_target_edges": len(connected_targets),
        "orphan_edges": len(orphan_targets),
        "too_short_to_merge": too_short_count,
        **hop_counts,
    }

    # Convert back to original CRS if needed
    if original_crs is not None and working_crs != original_crs:
        logger.info(f"  Converting output back to original CRS: {original_crs}")
        main_edges = main_edges.to_crs(original_crs)
        if len(orphan_targets) > 0:
            orphan_targets = orphan_targets.to_crs(original_crs)
        if net_new_edges is not None and len(net_new_edges) > 0:
            net_new_edges = net_new_edges.to_crs(original_crs)

    logger.info(f"Orphan detection complete: {stats}")

    return main_edges, orphan_targets, net_new_edges, stats


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
    for _idx, row in edges_gdf.iterrows():
        if row.get("_source") == EdgeSource.REFERENCE.value:
            reference_edge_ids.add(row["edge_id"])

    logger.info(f"Reference edges: {len(reference_edge_ids)}")

    # Map edges to their components
    edge_to_component = _map_edges_to_components(network.edges, components, G)

    # Identify main components (containing reference edges)
    main_component_ids = set()
    orphan_component_ids = set()

    for comp_id, _node_set in enumerate(components):
        # Get edges in this component
        comp_edges = [eid for eid, cid in edge_to_component.items() if cid == comp_id]

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
    for _edge_id, comp_id in edge_to_component.items():
        component_sizes[comp_id] = component_sizes.get(comp_id, 0) + 1

    # Add columns
    annotated["component_id"] = annotated["edge_id"].map(edge_to_component)
    annotated["component_status"] = annotated["component_id"].apply(
        lambda cid: ComponentStatus.MAIN.value
        if cid in main_component_ids
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
        lambda cid: ComponentStatus.MAIN.value
        if cid in main_component_ids
        else ComponentStatus.ORPHAN.value
    )

    return annotated
