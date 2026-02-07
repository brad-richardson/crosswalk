"""Orphan detection for integrated networks.

Identifies disconnected components (orphans) that are not connected
to the reference network, flagging them for QA review.

This module focuses purely on connectivity analysis. Pre-integration
screening (fringe detection, water/building intersection) is handled
by the screen module.
"""

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely as shp
from loguru import logger
from scipy.spatial import cKDTree
from shapely import Point, points
from shapely.ops import substring
from shapely.strtree import STRtree

from ..spatial import SpatialIndex
from ..topology.graph import build_graph, find_connected_components
from ..topology.planarize import PlanarizedNetwork
from ..topology.sparse_graph import SparseGraph
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

        # Vectorized endpoint proximity check for remaining orphans
        orphan_geoms = remaining_orphans.geometry.values
        n_orphans = len(orphan_geoms)

        # Extract start/end coords
        o_start = np.empty((n_orphans, 2))
        o_end = np.empty((n_orphans, 2))
        o_valid = np.ones(n_orphans, dtype=bool)
        for i, geom in enumerate(orphan_geoms):
            if geom is None or geom.is_empty:
                o_valid[i] = False
                o_start[i] = [0, 0]
                o_end[i] = [0, 0]
            else:
                try:
                    coords = geom.coords
                    o_start[i] = coords[0][:2]
                    o_end[i] = coords[-1][:2]
                except Exception:
                    o_valid[i] = False
                    o_start[i] = [0, 0]
                    o_end[i] = [0, 0]

        start_pts = points(o_start)
        end_pts = points(o_end)
        valid_idx = np.where(o_valid)[0]

        if len(valid_idx) > 0:
            # Batch nearest queries
            s_result = connected_tree.query_nearest(start_pts[o_valid], all_matches=False)
            e_result = connected_tree.query_nearest(end_pts[o_valid], all_matches=False)

            s_dists = shp.distance(start_pts[o_valid], connected_geoms[s_result[1]])
            e_dists = shp.distance(end_pts[o_valid], connected_geoms[e_result[1]])
            min_dists = np.minimum(s_dists, e_dists)

            connected_in_valid = min_dists <= connection_tolerance_m
            newly_connected_indices = list(remaining_orphans.index[valid_idx[connected_in_valid]])
        else:
            min_dists = np.array([])
            newly_connected_indices = []

        if debug and len(valid_idx) > 0:
            closest_pairs = sorted(zip(valid_idx, min_dists), key=lambda x: x[1])[:5]
            logger.debug(f"    Hop {hop} closest orphan distances: {closest_pairs}")

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


def _check_bridges_components(
    candidates: gpd.GeoDataFrame,
    reference_edges: gpd.GeoDataFrame,
    main_edges: gpd.GeoDataFrame,
    tolerance_m: float,
    min_overlap_m: float = 10.0,
) -> pd.Series:
    """Check which candidates bridge disconnected reference components.

    A segment qualifies as a bridge if:
    1. Its start half overlaps >= min_overlap_m with reference segment A
    2. Its end half overlaps >= min_overlap_m with reference segment B
    3. A != B
    4. A and B are in different connected components of the main network

    Args:
        candidates: Filtered segments to check
        reference_edges: Reference-only segments (for overlap check)
        main_edges: Full main network (reference + connected targets, for components)
        tolerance_m: Buffer for overlap computation
        min_overlap_m: Minimum overlap at each end to qualify

    Returns:
        Boolean Series aligned to candidates index: True for bridge segments
    """
    if len(candidates) == 0 or len(reference_edges) == 0:
        return pd.Series(dtype=bool)

    # Step 1: Build spatial index of reference geometries
    ref_geoms = reference_edges.geometry.values
    ref_tree = STRtree(ref_geoms)

    # Step 2: Build connected components of the main network using Union-Find
    # Extract endpoints from all main edges, cluster within tolerance
    main_geoms = main_edges.geometry.values
    n_main = len(main_geoms)

    # Vectorized endpoint extraction
    start_coords = []
    end_coords = []
    seg_indices = []
    for i, geom in enumerate(main_geoms):
        if geom is None or geom.is_empty:
            continue
        try:
            coords = geom.coords
            start_coords.append(coords[0][:2])
            end_coords.append(coords[-1][:2])
            seg_indices.append(i)
        except Exception:
            continue

    if not seg_indices:
        return pd.Series(False, index=candidates.index)

    # Stack start + end into single array for cKDTree
    seg_indices_arr = np.array(seg_indices)
    all_endpoints = np.vstack([np.array(start_coords), np.array(end_coords)])
    all_seg_indices = np.concatenate([seg_indices_arr, seg_indices_arr])

    # Union-Find on segment indices
    parent = list(range(n_main))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Cluster endpoints within tolerance using cKDTree
    tree = cKDTree(all_endpoints)
    pairs = tree.query_pairs(tolerance_m)

    for i, j in pairs:
        union(all_seg_indices[i], all_seg_indices[j])

    # Step 3: Map reference segments to their component labels
    # Reference edges are concatenated first into main_edges, so direct indexing works
    ref_source_mask = main_edges["_source"] == EdgeSource.REFERENCE.value
    ref_main_indices = np.where(ref_source_mask.values)[0]

    # Direct array mapping — ref positional index -> main edge positional index
    ref_to_main = ref_main_indices  # array indexing: ref_to_main[ref_pos] = main_pos

    # Step 4: For each candidate, check bridge condition
    results = []
    for _idx, row in candidates.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.length < 1e-6:
            results.append(False)
            continue

        # Split at midpoint
        mid = geom.length / 2.0
        start_half = substring(geom, 0, mid)
        end_half = substring(geom, mid, geom.length)

        # Find best overlapping reference for each half
        def best_ref_overlap(half_geom):
            """Return (ref_positional_index, overlap_length) for best match."""
            if half_geom is None or half_geom.is_empty:
                return None, 0.0

            # Query nearby reference segments
            buffered = half_geom.buffer(tolerance_m)
            nearby_indices = ref_tree.query(buffered)

            best_idx = None
            best_overlap = 0.0
            for ri in nearby_indices:
                ref_buf = ref_geoms[ri].buffer(tolerance_m)
                intersection = half_geom.intersection(ref_buf)
                overlap = intersection.length
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = ri
            return best_idx, best_overlap

        start_ref, start_overlap = best_ref_overlap(start_half)
        end_ref, end_overlap = best_ref_overlap(end_half)

        # Check all conditions
        if (
            start_ref is not None
            and end_ref is not None
            and start_overlap >= min_overlap_m
            and end_overlap >= min_overlap_m
            and start_ref != end_ref
        ):
            # Check if they're in different components
            if (
                start_ref < len(ref_to_main)
                and end_ref < len(ref_to_main)
                and find(ref_to_main[start_ref]) != find(ref_to_main[end_ref])
            ):
                results.append(True)
                continue

        results.append(False)

    return pd.Series(results, index=candidates.index)


def detect_orphans_by_proximity(
    combined_gdf: gpd.GeoDataFrame,
    connection_tolerance_m: float = 3.0,
    min_merge_length_m: float = 20.0,
    net_new_buffer_m: float = 5.0,
    matched_net_new_buffer_m: float = 15.0,
    max_hops: int = 2,
    transitive_tolerance_m: float | None = None,
    debug_connectivity: bool = False,
    enable_connectivity_gating: bool = True,
    min_bridge_overlap_m: float = 10.0,
) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame | None,
    dict[str, Any],
]:
    """Identify orphan segments based on endpoint proximity to reference network.

    A segment is considered connected if at least one of its endpoints is within
    connection_tolerance_m of any reference segment. Target segments not connected
    to the reference network are flagged as disconnected.

    Transitive connectivity: If trail A connects to a road, and trail B connects
    to trail A (but not the road), trail B is also considered connected via
    transitive connectivity (up to max_hops).

    Connected segments with less than min_merge_length_m of net-new coverage are
    separated into filtered_edges — they are connected but don't add meaningful
    new coverage.

    Note: Pre-integration screening (fringe detection, water/building checks) should
    be performed before calling this function using the screen module.

    Args:
        combined_gdf: Combined GeoDataFrame with _source column
        connection_tolerance_m: Distance in meters to consider "connected"
        min_merge_length_m: Minimum segment length (meters) to merge into network.
            Connected segments shorter than this are filtered out.
        net_new_buffer_m: Buffer distance (meters) around reference for net-new calculation
            of unmatched (target_new) segments. Tight buffer to detect genuine new coverage.
        matched_net_new_buffer_m: Buffer distance (meters) for matched (target_matched)
            segments. Wider buffer since geometry differences are digitization noise, not
            real new coverage.
        max_hops: Maximum transitive connectivity hops from reference (default 2).
            0 = only direct connections, 1 = direct + 1 hop, 2 = direct + 2 hops.
        transitive_tolerance_m: Tolerance (meters) for transitive connections between
            target segments. Defaults to 2x connection_tolerance_m since trails often
            don't share exact endpoints. Set to connection_tolerance_m for strict mode.
        debug_connectivity: Enable debug logging for transitive connectivity analysis.
        enable_connectivity_gating: Check if filtered segments bridge disconnected
            reference components. Segments that bridge different components are promoted
            back to main despite insufficient net-new coverage. Default True.
        min_bridge_overlap_m: Minimum overlap (meters) at each end of a candidate
            bridge segment with its reference segment to qualify. Default 10.0.

    Returns:
        Tuple of:
        - main_edges: Reference edges + connected target edges meeting length requirement
        - disconnected_edges: Target edges not connected to reference network
        - filtered_edges: Connected target edges with insufficient net-new coverage
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
    logger.info(f"  Net-new buffer (unmatched): {net_new_buffer_m}m")
    logger.info(f"  Net-new buffer (matched): {matched_net_new_buffer_m}m")
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
        empty = gpd.GeoDataFrame(columns=main_edges.columns, crs=combined_gdf.crs)
        stats = {
            "total_segments": len(combined_gdf),
            "reference_edges": len(reference_edges),
            "connected_target_edges": 0,
            "disconnected_edges": 0,
            "filtered_edges": 0,
        }
        return main_edges, empty, empty.copy(), None, stats

    # Build spatial index of reference segments
    ref_geoms = reference_edges.geometry.values
    ref_tree = STRtree(ref_geoms)

    # Vectorized endpoint proximity check using shapely batch operations
    target_geom_arr = target_edges.geometry.values
    n_targets = len(target_geom_arr)

    # Extract start/end coordinates vectorized via get_coordinates
    # For LineStrings, first coord = start, last coord = end
    start_coords = np.empty((n_targets, 2))
    end_coords = np.empty((n_targets, 2))
    valid_mask = np.ones(n_targets, dtype=bool)

    for i, geom in enumerate(target_geom_arr):
        if geom is None or geom.is_empty:
            valid_mask[i] = False
            start_coords[i] = [0, 0]
            end_coords[i] = [0, 0]
        else:
            try:
                coords = geom.coords
                start_coords[i] = coords[0][:2]
                end_coords[i] = coords[-1][:2]
            except Exception:
                valid_mask[i] = False
                start_coords[i] = [0, 0]
                end_coords[i] = [0, 0]

    # Create point arrays for batch STRtree query
    start_pts = points(start_coords)
    end_pts = points(end_coords)

    # Batch nearest queries — returns (input_idx, tree_idx) arrays
    start_nearest = ref_tree.query_nearest(start_pts[valid_mask], all_matches=False)
    end_nearest = ref_tree.query_nearest(end_pts[valid_mask], all_matches=False)

    # Compute distances vectorized
    min_distances = np.full(n_targets, np.nan)
    valid_indices = np.where(valid_mask)[0]

    # start_nearest[0] = input indices (into valid subset), start_nearest[1] = tree indices
    start_ref_geoms = ref_geoms[start_nearest[1]]
    end_ref_geoms = ref_geoms[end_nearest[1]]

    start_dists = shp.distance(start_pts[valid_mask], start_ref_geoms)
    end_dists = shp.distance(end_pts[valid_mask], end_ref_geoms)

    valid_min_dists = np.minimum(start_dists, end_dists)
    min_distances[valid_indices] = valid_min_dists

    connected_mask = min_distances <= connection_tolerance_m
    connected_mask[~valid_mask] = False

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
    filtered_targets = gpd.GeoDataFrame(columns=connected_targets.columns, crs=combined_gdf.crs)
    if min_merge_length_m > 0 and len(connected_targets) > 0:
        logger.info(
            f"  Computing net-new lengths (unmatched buffer: {net_new_buffer_m}m, "
            f"matched buffer: {matched_net_new_buffer_m}m)..."
        )

        ref_index = SpatialIndex(reference_edges.geometry.values)
        ref_geoms = reference_edges.geometry.values
        ref_tree = ref_index.tree

        ct_geoms = connected_targets.geometry.values
        ct_sources = (
            connected_targets["_source"].values
            if "_source" in connected_targets.columns
            else np.full(len(connected_targets), "")
        )
        is_matched = ct_sources == EdgeSource.TARGET_MATCHED.value

        net_new_lengths = np.zeros(len(connected_targets))
        net_new_geoms = np.empty(len(connected_targets), dtype=object)

        # Vectorized fast-path: batch nearest-ref containment check
        # Skip targets fully inside their nearest ref's buffer (net-new = 0)
        valid_mask = np.array([g is not None and not g.is_empty for g in ct_geoms], dtype=bool)
        valid_indices = np.where(valid_mask)[0]
        valid_geoms = ct_geoms[valid_indices]

        if len(valid_geoms) > 0:
            # Batch nearest-neighbor query (vectorized C)
            _input_idxs, nearest_ref_idxs = ref_tree.query_nearest(valid_geoms, all_matches=False)

            # Per-target buffer distance based on matched/unmatched
            buffer_distances = np.where(
                is_matched[valid_indices], matched_net_new_buffer_m, net_new_buffer_m
            )

            # Vectorized buffer + containment (vectorized C, no Python loop)
            nearest_buffered = shp.buffer(ref_geoms[nearest_ref_idxs], buffer_distances)
            contained = shp.contains(nearest_buffered, valid_geoms)

            n_contained = int(contained.sum())
            n_valid = len(valid_geoms)
            logger.info(
                f"  Net-new fast path: {n_contained}/{n_valid} fully covered by nearest ref"
            )

            # Full computation only for targets NOT fully covered
            needs_full = valid_indices[~contained]
            logger.info(f"  Computing net-new for {len(needs_full)} remaining targets...")
            for count, i in enumerate(needs_full):
                if count > 0 and count % 2000 == 0:
                    logger.info(f"    Net-new progress: {count}/{len(needs_full)}")
                geom = ct_geoms[i]
                buffer = matched_net_new_buffer_m if is_matched[i] else net_new_buffer_m
                net_new_geom = ref_index.compute_net_new(geom, buffer)
                if net_new_geom is not None:
                    net_new_lengths[i] = net_new_geom.length
                    net_new_geoms[i] = net_new_geom

        connected_targets["_net_new_length_m"] = net_new_lengths
        connected_targets["_total_length_m"] = connected_targets.geometry.length
        connected_targets["_net_new_geometry"] = net_new_geoms

        long_enough = connected_targets["_net_new_length_m"] >= min_merge_length_m
        too_short = connected_targets[~long_enough].copy()

        if len(too_short) > 0:
            too_short["unmatched_reason"] = "insufficient_net_new_length"
            filtered_targets = too_short
            avg_net_new = too_short["_net_new_length_m"].mean()
            logger.info(
                f"  Insufficient net-new length (<{min_merge_length_m}m): {len(too_short)} "
                f"(avg net-new: {avg_net_new:.1f}m)"
            )

        connected_targets = connected_targets[long_enough].copy()

        # Connectivity gating: rescue filtered segments that bridge components
        if enable_connectivity_gating and len(filtered_targets) > 0:
            main_so_far = gpd.GeoDataFrame(
                pd.concat([reference_edges, connected_targets], ignore_index=True),
                crs=combined_gdf.crs,
            )
            bridges = _check_bridges_components(
                candidates=filtered_targets,
                reference_edges=reference_edges,
                main_edges=main_so_far,
                tolerance_m=connection_tolerance_m,
                min_overlap_m=min_bridge_overlap_m,
            )
            if len(bridges) > 0:
                promoted = filtered_targets[bridges].copy()
                if len(promoted) > 0:
                    promoted["_connectivity_role"] = "bridge"
                    connected_targets = gpd.GeoDataFrame(
                        pd.concat([connected_targets, promoted], ignore_index=True),
                        crs=combined_gdf.crs,
                    )
                    filtered_targets = filtered_targets[~bridges].copy()
                    logger.info(f"  Connectivity gating: promoted {len(promoted)} bridge segments")

        # Build net-new edges GeoDataFrame for visualization
        # Use the computed _net_new_geometry (subline) not the full segment
        if len(connected_targets) > 0 and "_net_new_geometry" in connected_targets.columns:
            has_net_new = connected_targets["_net_new_geometry"].apply(
                lambda g: g is not None and not g.is_empty
            )
            if has_net_new.any():
                nn_subset = connected_targets[has_net_new]
                net_new_edges = gpd.GeoDataFrame(
                    {
                        "geometry": nn_subset["_net_new_geometry"].values,
                        "_original_id": nn_subset["_original_id"].values
                        if "_original_id" in nn_subset.columns
                        else None,
                        "_source_dataset": nn_subset["_source_dataset"].values
                        if "_source_dataset" in nn_subset.columns
                        else None,
                        "_net_new_length_m": nn_subset["_net_new_length_m"].values
                        if "_net_new_length_m" in nn_subset.columns
                        else None,
                        "_total_length_m": nn_subset["_total_length_m"].values
                        if "_total_length_m" in nn_subset.columns
                        else None,
                        "_connectivity_hop": nn_subset["_connectivity_hop"].values
                        if "_connectivity_hop" in nn_subset.columns
                        else 0,
                    },
                    crs=combined_gdf.crs,
                )
                logger.info(f"  Net-new edges for visualization: {len(net_new_edges)}")

    logger.info(f"  Connected target edges (after length filter): {len(connected_targets)}")
    logger.info(f"  Disconnected edges: {len(orphan_targets)}")
    logger.info(f"  Filtered edges (insufficient net-new): {len(filtered_targets)}")

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

    # Mark disconnected segments
    orphan_targets["component_status"] = ComponentStatus.DISCONNECTED.value
    orphan_targets["unmatched_reason"] = "not_connected_to_network"

    # Mark filtered segments
    if len(filtered_targets) > 0:
        filtered_targets["component_status"] = ComponentStatus.FILTERED.value
        # unmatched_reason already set to "insufficient_net_new_length" above

    # Drop internal columns from both
    for col in drop_cols:
        if col in orphan_targets.columns:
            orphan_targets = orphan_targets.drop(columns=[col])
        if col in filtered_targets.columns:
            filtered_targets = filtered_targets.drop(columns=[col])

    # Add QA priority to both
    if len(orphan_targets) > 0:
        orphan_targets = _add_orphan_qa_priority(orphan_targets, main_edges)
    if len(filtered_targets) > 0:
        filtered_targets = _add_orphan_qa_priority(filtered_targets, main_edges)

    # Count connectivity by hop level
    hop_counts = {}
    if "_connectivity_hop" in connected_targets.columns:
        for hop in range(max_hops + 1):
            hop_counts[f"hop_{hop}_connected"] = int(
                (connected_targets["_connectivity_hop"] == hop).sum()
            )

    # Count bridge promotions
    bridge_promoted = 0
    if "_connectivity_role" in connected_targets.columns:
        bridge_promoted = int((connected_targets["_connectivity_role"] == "bridge").sum())

    stats = {
        "total_segments": len(combined_gdf),
        "reference_edges": len(reference_edges),
        "connected_target_edges": len(connected_targets),
        "disconnected_edges": len(orphan_targets),
        "filtered_edges": len(filtered_targets),
        "bridge_promoted": bridge_promoted,
        **hop_counts,
    }

    # Convert back to original CRS if needed
    if original_crs is not None and working_crs != original_crs:
        logger.info(f"  Converting output back to original CRS: {original_crs}")
        main_edges = main_edges.to_crs(original_crs)
        if len(orphan_targets) > 0:
            orphan_targets = orphan_targets.to_crs(original_crs)
        if len(filtered_targets) > 0:
            filtered_targets = filtered_targets.to_crs(original_crs)
        if net_new_edges is not None and len(net_new_edges) > 0:
            net_new_edges = net_new_edges.to_crs(original_crs)

    logger.info(f"Orphan detection complete: {stats}")

    return main_edges, orphan_targets, filtered_targets, net_new_edges, stats


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
    graph: SparseGraph,
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
        lambda cid: (
            ComponentStatus.MAIN.value
            if cid in main_component_ids
            else ComponentStatus.DISCONNECTED.value
        )
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
        lambda cid: (
            ComponentStatus.MAIN.value
            if cid in main_component_ids
            else ComponentStatus.DISCONNECTED.value
        )
    )

    return annotated
