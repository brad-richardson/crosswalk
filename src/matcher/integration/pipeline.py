"""Integration pipeline orchestration.

Main entry point for running the network integration pipeline.

Pipeline phases:
1. Pre-screening: Filter targets using screen module (fringe, water, buildings)
2. Integration: Combine reference + targets, detect connectivity
3. Post-integration: Island detection, GPS drift analysis (via post_integration module)
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
from loguru import logger

from ..config import settings
from ..resolution.bridge import load_bridge_file
from ..screen.constants import FRINGE_BUFFER_M, FRINGE_MIN_INSIDE_LENGTH_M
from ..screen.tests.fringe_test import filter_fringe_segments
from .combiner import combine_networks, separate_matched_unmatched
from .filters import detect_near_duplicates, filter_short_segments
from .orphan_detector import detect_orphans_by_proximity
from .output import write_integration_outputs
from .provenance import (
    IntegrationResult,
    IntegrationStatistics,
    TargetInput,
)


@dataclass
class TargetConfig:
    """Configuration for a target dataset to integrate."""

    name: str
    bridge_path: Path
    unmatched_path: Path
    priority: int
    target_path: Path | None = None  # Full target for separating matched/unmatched


def run_integration_pipeline(
    reference_path: Path,
    target_configs: list[TargetConfig],
    output_dir: Path,
    overlap_iou_threshold: float = None,
    min_segment_length_m: float = None,
    filter_near_duplicates_flag: bool = True,
    connection_tolerance_m: float = 3.0,
    min_merge_length_m: float = 20.0,
    net_new_buffer_m: float = 5.0,
    max_hops: int = 2,
    fringe_buffer_m: float = FRINGE_BUFFER_M,
    enable_fringe_screening: bool = True,
    transitive_tolerance_m: float | None = None,
    debug_connectivity: bool = False,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
) -> IntegrationResult:
    """Run the full integration pipeline.

    Pipeline phases:
    1. Load: Load reference network and target datasets
    2. Pre-screen: Apply filters (min length, near-duplicates, fringe detection)
    3. Combine: Merge reference + targets with provenance
    4. Connectivity: Detect orphans by endpoint proximity
    5. Output: Write results

    Pre-screening (fringe detection) is handled by the screen module before
    combining. This separates "should this segment be considered?" from
    "is this segment connected to the network?".

    Args:
        reference_path: Path to reference network (Overture segments parquet)
        target_configs: List of TargetConfig for each target dataset
        output_dir: Directory for output files
        overlap_iou_threshold: IoU threshold for overlap detection
        min_segment_length_m: Filter segments shorter than this (meters)
        filter_near_duplicates_flag: Whether to detect and filter near-duplicates
        connection_tolerance_m: Distance to consider a segment "connected" to reference (meters).
            Default 3m requires actual physical connection to infrastructure.
        min_merge_length_m: Minimum net-new length (meters) to merge a segment.
            Connected segments with less than this much new coverage are orphaned.
        net_new_buffer_m: Buffer distance (meters) around reference for net-new calculation.
            Segments within this buffer are considered "covered" by reference.
        max_hops: Maximum transitive connectivity hops from reference (default 2).
            Segments connected via other target segments are included up to this depth.
        fringe_buffer_m: Buffer distance (meters) around reference coverage for fringe
            screening. Segments outside this area are filtered before combining.
        enable_fringe_screening: Whether to pre-screen fringe segments. Default True.
        transitive_tolerance_m: Tolerance (meters) for transitive connections between
            target segments. Defaults to 2x connection_tolerance_m since trails often
            don't share exact endpoints. Set to connection_tolerance_m for strict mode.
        debug_connectivity: Enable debug logging for transitive connectivity analysis.
        ref_id_column: ID column in reference
        target_id_column: ID column in targets

    Returns:
        IntegrationResult with edges, orphans, and statistics
    """
    overlap_iou_threshold = overlap_iou_threshold or settings.overlap_iou_threshold
    min_segment_length_m = min_segment_length_m or settings.min_segment_length_m

    logger.info("=" * 60)
    logger.info("Starting integration pipeline")
    logger.info("=" * 60)
    logger.info(f"Reference: {reference_path}")
    logger.info(f"Targets: {len(target_configs)}")
    logger.info(f"Output: {output_dir}")

    # Initialize statistics
    stats = IntegrationStatistics()

    # Step 1: Load reference network
    logger.info("Step 1: Loading reference network...")
    reference = gpd.read_parquet(reference_path)
    logger.info(f"  Loaded {len(reference)} reference segments")
    stats.reference_edges = len(reference)

    # Step 2: Load and prepare target datasets
    logger.info("Step 2: Loading target datasets...")
    target_inputs = []

    for config in sorted(target_configs, key=lambda c: c.priority):
        logger.info(f"  Loading '{config.name}' (priority {config.priority})...")

        # Load bridge file (match results)
        match_results = load_bridge_file(config.bridge_path)
        logger.info(f"    Match results: {len(match_results)}")

        # Load unmatched segments
        unmatched = gpd.read_parquet(config.unmatched_path)
        logger.info(f"    Unmatched segments: {len(unmatched)}")

        # If we have full target, separate matched/unmatched
        if config.target_path is not None:
            target = gpd.read_parquet(config.target_path)
            matched, unmatched = separate_matched_unmatched(target, match_results, target_id_column)
        else:
            # Assume unmatched_path contains unmatched segments
            # For matched, we need the full target - create empty if not provided
            matched = gpd.GeoDataFrame()

        # Apply length filter
        if min_segment_length_m > 0:
            unmatched, filtered_short = filter_short_segments(unmatched, min_segment_length_m)
            if len(filtered_short) > 0:
                logger.info(f"    Filtered {len(filtered_short)} short unmatched segments")

        # Detect near-duplicates
        if filter_near_duplicates_flag and len(matched) > 0 and len(unmatched) > 0:
            unmatched, duplicates = detect_near_duplicates(
                unmatched, matched, target_id_column=target_id_column
            )
            if len(duplicates) > 0:
                logger.info(f"    Detected {len(duplicates)} potential near-duplicates")

        # Pre-screen fringe segments (outside reference coverage)
        if enable_fringe_screening and len(unmatched) > 0 and len(reference) > 0:
            unmatched, fringe_segments = filter_fringe_segments(
                target_edges=unmatched,
                reference_edges=reference,
                buffer_distance_m=fringe_buffer_m,
                min_inside_length_m=FRINGE_MIN_INSIDE_LENGTH_M,
            )
            if len(fringe_segments) > 0:
                logger.info(f"    Pre-screened {len(fringe_segments)} fringe segments")

        target_inputs.append(
            TargetInput(
                name=config.name,
                matched=matched,
                unmatched=unmatched,
                match_results=match_results,
                priority=config.priority,
            )
        )

        stats.datasets_integrated.append(config.name)
        stats.target_edges_matched += len(matched)
        stats.target_edges_unmatched += len(unmatched)

    # Step 3: Combine networks with provenance
    logger.info("Step 3: Combining networks...")
    combined_gdf, dropped_overlaps = combine_networks(
        reference=reference,
        target_inputs=target_inputs,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
        overlap_iou_threshold=overlap_iou_threshold,
    )
    stats.dropped_overlaps = len(dropped_overlaps)
    stats.total_edges = len(combined_gdf)

    # Step 4: Detect orphans by endpoint proximity (connectivity analysis)
    # Note: Fringe detection is now done in pre-screening step above
    logger.info("Step 4: Detecting orphans by endpoint proximity...")
    main_edges, orphan_edges, net_new_edges, orphan_stats = detect_orphans_by_proximity(
        combined_gdf,
        connection_tolerance_m=connection_tolerance_m,
        min_merge_length_m=min_merge_length_m,
        net_new_buffer_m=net_new_buffer_m,
        max_hops=max_hops,
        transitive_tolerance_m=transitive_tolerance_m,
        debug_connectivity=debug_connectivity,
    )
    stats.main_component_edges = len(main_edges)
    stats.orphan_edges = len(orphan_edges)
    stats.orphan_components = len(orphan_edges)  # Each orphan is its own "component"

    # Step 5: Build result
    logger.info("Step 5: Building result...")
    result = IntegrationResult(
        nodes=gpd.GeoDataFrame(),  # No nodes without planarization
        edges=main_edges,
        orphan_edges=orphan_edges,
        dropped_overlaps=dropped_overlaps,
        net_new_edges=net_new_edges,
        statistics=stats,
        created_at=datetime.now(UTC),
    )

    # Step 6: Write outputs
    logger.info("Step 6: Writing outputs...")
    write_integration_outputs(result, output_dir)

    logger.info("=" * 60)
    logger.info("Integration pipeline complete!")
    logger.info(f"  Reference edges: {stats.reference_edges}")
    logger.info(f"  Target edges (matched): {stats.target_edges_matched}")
    logger.info(f"  Target edges (unmatched): {stats.target_edges_unmatched}")
    logger.info(f"  Dropped overlaps: {stats.dropped_overlaps}")
    logger.info(f"  Total edges: {stats.total_edges}")
    logger.info(f"  Main (connected) edges: {stats.main_component_edges}")
    logger.info(f"  Orphan edges: {stats.orphan_edges}")
    logger.info("=" * 60)

    # Log detailed layer summary with lengths
    logger.info("")
    logger.info("Layer Summary (segment count / total length):")

    def _format_length(length_m: float) -> str:
        if length_m >= 1000:
            return f"{length_m / 1000:.1f} km"
        return f"{length_m:.0f} m"

    def _layer_stats(gdf: gpd.GeoDataFrame, name: str, source_filter: str | None = None):
        if gdf is None or len(gdf) == 0:
            logger.info(f"  {name}: 0 segments / 0 m")
            return
        if source_filter and "_source" in gdf.columns:
            gdf = gdf[gdf["_source"] == source_filter]
        if len(gdf) == 0:
            logger.info(f"  {name}: 0 segments / 0 m")
            return
        # Project to UTM for accurate length calculation
        working_gdf = gdf
        if gdf.crs and gdf.crs.is_geographic:
            working_gdf = gdf.to_crs(gdf.estimate_utm_crs())
        total_length = working_gdf.geometry.length.sum()
        logger.info(f"  {name}: {len(gdf)} segments / {_format_length(total_length)}")

    _layer_stats(main_edges, "Reference", "reference")
    _layer_stats(main_edges, "Matched (target)", "target_matched")
    _layer_stats(main_edges, "To Merge (connected)", "target_new")
    _layer_stats(net_new_edges, "Net New Coverage")
    _layer_stats(orphan_edges, "Orphan")
    logger.info("")

    return result


def run_integration_from_config(config_path: Path, output_dir: Path) -> IntegrationResult:
    """Run integration pipeline from YAML config file.

    Config file format:
    ```yaml
    reference: data/raw/overture.parquet
    targets:
      - name: boston_streets
        bridge: data/boston_streets/bridge.parquet
        unmatched: data/boston_streets/unmatched.parquet
        target: data/raw/boston_streets.parquet  # optional
        priority: 1
      - name: boston_bikes
        bridge: data/boston_bikes/bridge.parquet
        unmatched: data/boston_bikes/unmatched.parquet
        priority: 2
    overlap_threshold: 0.8
    min_segment_length: 3.0
    ```

    Args:
        config_path: Path to YAML config file
        output_dir: Directory for output files

    Returns:
        IntegrationResult
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    reference_path = Path(config["reference"])

    target_configs = []
    for t in config["targets"]:
        target_configs.append(
            TargetConfig(
                name=t["name"],
                bridge_path=Path(t["bridge"]),
                unmatched_path=Path(t["unmatched"]),
                priority=t["priority"],
                target_path=Path(t["target"]) if "target" in t else None,
            )
        )

    return run_integration_pipeline(
        reference_path=reference_path,
        target_configs=target_configs,
        output_dir=output_dir,
        overlap_iou_threshold=config.get("overlap_threshold"),
        min_segment_length=config.get("min_segment_length"),
    )
