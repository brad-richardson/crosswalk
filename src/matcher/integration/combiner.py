"""Network combiner for integration pipeline.

Combines reference network with multiple target datasets,
tracking provenance and handling overlap conflicts with priority-based resolution.
"""

import geopandas as gpd
from loguru import logger
from shapely import LineString
from shapely.strtree import STRtree

from ..config import settings
from .provenance import DroppedSegment, EdgeSource, TargetInput


def _compute_buffer_iou(line_a, line_b, radius: float) -> float:
    """Compute Intersection over Union of buffered geometries."""
    buf_a = line_a.buffer(radius)
    buf_b = line_b.buffer(radius)

    intersection_area = buf_a.intersection(buf_b).area
    union_area = buf_a.union(buf_b).area

    return intersection_area / union_area if union_area > 0 else 0.0


def combine_networks(
    reference: gpd.GeoDataFrame,
    target_inputs: list[TargetInput],
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
    overlap_iou_threshold: float = None,
    overlap_buffer_m: float = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Combine reference and target networks with provenance tracking.

    Creates a unified GeoDataFrame where each segment has provenance columns
    indicating its origin. Handles multi-dataset conflict resolution using
    priority-based overlap detection.

    Args:
        reference: Reference network GeoDataFrame (Overture)
        target_inputs: List of TargetInput objects, ordered by priority
        ref_id_column: ID column in reference
        target_id_column: ID column in targets
        overlap_iou_threshold: IoU threshold for detecting overlaps
        overlap_buffer_m: Buffer distance for overlap detection (meters)

    Returns:
        Tuple of:
        - combined_gdf: Combined network with provenance columns
        - dropped_overlaps_gdf: Segments dropped due to priority conflicts
    """
    overlap_iou_threshold = overlap_iou_threshold or settings.overlap_iou_threshold
    overlap_buffer_m = overlap_buffer_m or settings.overlap_buffer_m

    logger.info(f"Combining reference with {len(target_inputs)} target datasets")
    logger.info(f"  overlap_iou_threshold: {overlap_iou_threshold}")
    logger.info(f"  overlap_buffer_m: {overlap_buffer_m}m")

    # Store original CRS for final output
    original_crs = reference.crs

    # Determine working CRS (metric) for overlap detection
    # If reference is geographic (e.g., WGS84), project to UTM for processing
    working_crs = original_crs
    if original_crs is not None and original_crs.is_geographic:
        working_crs = reference.estimate_utm_crs()
        logger.info(f"Using projected CRS for processing: {working_crs}")
        reference = reference.to_crs(working_crs)

    # Align all targets to working CRS
    for target_input in target_inputs:
        if (
            target_input.matched is not None
            and len(target_input.matched) > 0
            and target_input.matched.crs != working_crs
        ):
            logger.info(f"Reprojecting '{target_input.name}' matched to working CRS")
            target_input.matched = target_input.matched.to_crs(working_crs)
        if (
            target_input.unmatched is not None
            and len(target_input.unmatched) > 0
            and target_input.unmatched.crs != working_crs
        ):
            logger.info(f"Reprojecting '{target_input.name}' unmatched to working CRS")
            target_input.unmatched = target_input.unmatched.to_crs(working_crs)

    all_segments = []
    dropped_segments: list[DroppedSegment] = []

    # Step 1: Add all reference segments (priority 0, never dropped)
    logger.info(f"Adding {len(reference)} reference segments (priority 0)")
    ref_segments = _add_reference_segments(reference, ref_id_column)
    all_segments.extend(ref_segments)

    # Build spatial index for overlap detection
    included_geoms = [s["geometry"] for s in all_segments]
    included_tree = STRtree(included_geoms) if included_geoms else None

    # Step 2: Process targets in priority order
    for target_input in sorted(target_inputs, key=lambda t: t.priority):
        logger.info(
            f"Processing target '{target_input.name}' "
            f"(priority {target_input.priority}): "
            f"{len(target_input.matched)} matched, {len(target_input.unmatched)} unmatched"
        )

        # Build match lookup
        match_lookup = _build_match_lookup(target_input.match_results)

        # Add matched segments
        matched_added, matched_dropped = _add_target_segments(
            target_input.matched,
            target_input.name,
            target_input.priority,
            EdgeSource.TARGET_MATCHED,
            target_id_column,
            match_lookup,
            included_geoms,
            included_tree,
            overlap_iou_threshold,
            overlap_buffer_m,
        )
        all_segments.extend(matched_added)
        dropped_segments.extend(matched_dropped)

        # Update spatial index
        if matched_added:
            included_geoms.extend([s["geometry"] for s in matched_added])
            included_tree = STRtree(included_geoms)

        # Add unmatched segments
        unmatched_added, unmatched_dropped = _add_target_segments(
            target_input.unmatched,
            target_input.name,
            target_input.priority,
            EdgeSource.TARGET_UNMATCHED,
            target_id_column,
            {},  # No match info for unmatched
            included_geoms,
            included_tree,
            overlap_iou_threshold,
            overlap_buffer_m,
        )
        all_segments.extend(unmatched_added)
        dropped_segments.extend(unmatched_dropped)

        # Update spatial index
        if unmatched_added:
            included_geoms.extend([s["geometry"] for s in unmatched_added])
            included_tree = STRtree(included_geoms)

        logger.info(
            f"  Added {len(matched_added)} matched, {len(unmatched_added)} unmatched; "
            f"dropped {len(matched_dropped) + len(unmatched_dropped)} overlaps"
        )

    # Build combined GeoDataFrame (in working CRS)
    combined_gdf = gpd.GeoDataFrame(all_segments, crs=working_crs)
    logger.info(f"Combined network: {len(combined_gdf)} segments")

    # Build dropped overlaps GeoDataFrame (in working CRS)
    dropped_gdf = _build_dropped_gdf(dropped_segments, working_crs)
    logger.info(f"Dropped overlaps: {len(dropped_gdf)} segments")

    # Convert back to original CRS if we used a different working CRS
    if original_crs is not None and working_crs != original_crs:
        logger.info(f"Converting output to original CRS: {original_crs}")
        combined_gdf = combined_gdf.to_crs(original_crs)
        if len(dropped_gdf) > 0:
            dropped_gdf = dropped_gdf.to_crs(original_crs)

    return combined_gdf, dropped_gdf


def _add_reference_segments(
    reference: gpd.GeoDataFrame,
    ref_id_column: str,
) -> list[dict]:
    """Add reference segments with provenance columns."""
    segments = []

    for idx, row in reference.iterrows():
        segment = {
            "geometry": row.geometry,
            "_source": EdgeSource.REFERENCE.value,
            "_original_id": str(row.get(ref_id_column, idx)),
            "_source_dataset": "overture",
            "_priority": 0,
            "_match_ref_id": None,
            "_match_confidence": None,
        }

        # Copy original attributes (excluding geometry and ID)
        for col in reference.columns:
            if col not in ["geometry", ref_id_column] and not col.startswith("_"):
                segment[col] = row[col]

        segments.append(segment)

    return segments


def _add_target_segments(
    target: gpd.GeoDataFrame,
    dataset_name: str,
    priority: int,
    source_type: EdgeSource,
    id_column: str,
    match_lookup: dict,
    existing_geoms: list,
    existing_tree: STRtree | None,
    overlap_iou_threshold: float,
    overlap_buffer_m: float,
) -> tuple[list[dict], list[DroppedSegment]]:
    """Add target segments, detecting and handling overlaps."""
    added = []
    dropped = []

    if target is None or len(target) == 0:
        return added, dropped

    for idx, row in target.iterrows():
        geom = row.geometry
        original_id = str(row.get(id_column, idx))

        # Check for overlap with existing segments
        if existing_tree is not None:
            overlapping_idx, overlap_iou = _find_overlapping_segment(
                geom,
                existing_geoms,
                existing_tree,
                overlap_buffer_m,
                overlap_iou_threshold,
            )

            if overlapping_idx is not None:
                # Overlap detected - drop this segment
                dropped.append(
                    DroppedSegment(
                        original_id=original_id,
                        source_dataset=dataset_name,
                        source_type=source_type,
                        geometry=geom,
                        dropped_reason="overlap_lower_priority",
                        overlapping_edge_id=overlapping_idx,
                        overlap_iou=overlap_iou,
                        priority=priority,
                    )
                )
                continue

        # No overlap - add segment
        match_info = match_lookup.get(original_id, {})

        segment = {
            "geometry": geom,
            "_source": source_type.value,
            "_original_id": original_id,
            "_source_dataset": dataset_name,
            "_priority": priority,
            "_match_ref_id": match_info.get("gers_id"),
            "_match_confidence": match_info.get("confidence"),
        }

        # Copy original attributes
        for col in target.columns:
            if col not in ["geometry", id_column] and not col.startswith("_"):
                segment[col] = row[col]

        added.append(segment)

    return added, dropped


def _find_overlapping_segment(
    geom: LineString,
    existing_geoms: list,
    tree: STRtree,
    buffer_m: float,
    iou_threshold: float,
) -> tuple[int | None, float | None]:
    """Find an overlapping segment in existing geometries.

    Returns:
        Tuple of (overlapping_index, overlap_iou) or (None, None) if no overlap
    """
    # Query spatial index for nearby segments
    buffered = geom.buffer(buffer_m)
    candidate_indices = tree.query(buffered)

    best_iou = 0.0
    best_idx = None

    for idx in candidate_indices:
        existing_geom = existing_geoms[idx]

        # Compute buffer IoU
        iou = _compute_buffer_iou(geom, existing_geom, buffer_m)

        if iou > best_iou:
            best_iou = iou
            best_idx = idx

    if best_iou >= iou_threshold:
        return best_idx, best_iou

    return None, None


def _build_match_lookup(match_results: list) -> dict:
    """Build lookup from target_id to match info."""
    lookup = {}
    for result in match_results:
        target_id = (
            str(result.target_id)
            if hasattr(result, "target_id")
            else str(result.get("local_id", ""))
        )
        gers_id = (
            str(result.ref_id) if hasattr(result, "ref_id") else str(result.get("gers_id", ""))
        )
        confidence = (
            result.confidence if hasattr(result, "confidence") else result.get("confidence", 0.0)
        )

        lookup[target_id] = {
            "gers_id": gers_id,
            "confidence": confidence,
        }
    return lookup


def _build_dropped_gdf(
    dropped_segments: list[DroppedSegment],
    crs,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame from dropped segments."""
    if not dropped_segments:
        return gpd.GeoDataFrame(
            columns=[
                "geometry",
                "original_id",
                "source_dataset",
                "source_type",
                "dropped_reason",
                "overlapping_edge_id",
                "overlap_iou",
                "priority",
            ],
            crs=crs,
        )

    records = []
    for ds in dropped_segments:
        records.append(
            {
                "geometry": ds.geometry,
                "original_id": ds.original_id,
                "source_dataset": ds.source_dataset,
                "source_type": ds.source_type.value,
                "dropped_reason": ds.dropped_reason,
                "overlapping_edge_id": ds.overlapping_edge_id,
                "overlap_iou": ds.overlap_iou,
                "priority": ds.priority,
            }
        )

    return gpd.GeoDataFrame(records, crs=crs)


def separate_matched_unmatched(
    target: gpd.GeoDataFrame,
    match_results: list,
    target_id_column: str = "local_id",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Separate target into matched and unmatched segments.

    Only high-confidence MATCH decisions are considered matched. REVIEW decisions
    (low confidence, needs human review) are treated as unmatched to prevent
    incorrect conflation from flowing into integration output.

    Args:
        target: Target GeoDataFrame
        match_results: List of match results (MatchResult or dict from bridge file)
        target_id_column: ID column in target

    Returns:
        Tuple of (matched_segments, unmatched_segments)
    """
    from ..matching.rules import MatchDecision

    # Build set of matched IDs - only include high-confidence MATCH decisions
    matched_ids = set()
    for result in match_results:
        if hasattr(result, "target_id"):
            # MatchResult object - check decision
            if result.decision == MatchDecision.MATCH:
                matched_ids.add(str(result.target_id))
        elif isinstance(result, dict):
            # Dict from bridge file - check match_decision column
            decision = result.get("match_decision", "match")
            if decision == "match":
                matched_ids.add(str(result.get("local_id", "")))

    # Split target
    if target_id_column in target.columns:
        target_ids = target[target_id_column].astype(str)
        matched_mask = target_ids.isin(matched_ids)
    else:
        matched_mask = target.index.astype(str).isin(matched_ids)

    matched = target[matched_mask].copy()
    unmatched = target[~matched_mask].copy()

    logger.info(f"Separated target: {len(matched)} matched, {len(unmatched)} unmatched")

    return matched, unmatched
