"""Pipeline orchestration - runs the full matching pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import geopandas as gpd
from loguru import logger

from ..blocking import generate_candidates
from ..matching import MatchDecision, optimize_with_one_to_many
from ..matching.rules import score_candidates
from ..resolution import generate_bridge_file, generate_unmatched_report


class PipelineError(Exception):
    """Error during pipeline execution."""

    pass


@dataclass
class PipelineResult:
    """Result of running the matching pipeline."""

    n_reference: int
    n_target: int
    n_candidates: int
    n_matched: int
    n_review: int
    n_unmatched: int
    bridge_file: Path
    unmatched_file: Optional[Path]


def run_pipeline(
    reference_path: Path,
    target_path: Path,
    output_path: Path,
    method: str = "rule",
    buffer_distance: float = 75.0,
    max_heading_diff: float = 90.0,  # Relaxed for aggressive matching
    max_length_ratio: float = 20.0,  # Relaxed for aggressive matching
    min_confidence: float = 0.1,  # Lower = more aggressive matching
    progress_callback: Optional[Callable[[int], None]] = None,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
    ref_name_column: str = "name",
    target_name_column: str = "name",
    ref_class_column: str = "class",
    target_class_column: str = "road_class",
) -> PipelineResult:
    """Run the full matching pipeline.

    Args:
        reference_path: Path to reference GeoParquet (Overture)
        target_path: Path to target GeoParquet (local data)
        output_path: Path for output bridge file
        method: Matching method ("rule" or "xgboost")
        buffer_distance: Candidate search radius in meters
        progress_callback: Optional callback for progress updates
        ref_id_column: ID column in reference
        target_id_column: ID column in target
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        ref_class_column: Class column in reference
        target_class_column: Class column in target

    Returns:
        PipelineResult with statistics
    """
    logger.info("=" * 60)
    logger.info("Starting matching pipeline")
    logger.info("=" * 60)

    # Validate input files exist
    if not reference_path.exists():
        raise PipelineError(f"Reference file not found: {reference_path}")
    if not target_path.exists():
        raise PipelineError(f"Target file not found: {target_path}")

    # Step 1: Load data
    logger.info("Step 1: Loading data...")
    try:
        reference = gpd.read_parquet(reference_path)
    except Exception as e:
        raise PipelineError(f"Failed to read reference file {reference_path}: {e}")

    try:
        target = gpd.read_parquet(target_path)
    except Exception as e:
        raise PipelineError(f"Failed to read target file {target_path}: {e}")

    # Validate geometry columns
    if reference.geometry.isna().any():
        n_null = reference.geometry.isna().sum()
        logger.warning(f"Reference has {n_null} null geometries - these will be skipped")
        reference = reference[~reference.geometry.isna()]

    if target.geometry.isna().any():
        n_null = target.geometry.isna().sum()
        logger.warning(f"Target has {n_null} null geometries - these will be skipped")
        target = target[~target.geometry.isna()]

    if len(reference) == 0:
        raise PipelineError("Reference dataset is empty after removing null geometries")
    if len(target) == 0:
        raise PipelineError("Target dataset is empty after removing null geometries")

    logger.info(f"  Reference: {len(reference)} features from {reference_path}")
    logger.info(f"  Target: {len(target)} features from {target_path}")

    if progress_callback:
        progress_callback(10)

    # Ensure both are in the same CRS
    if reference.crs != target.crs:
        logger.info(f"  Reprojecting target from {target.crs} to {reference.crs}")
        target = target.to_crs(reference.crs)

    # Convert to projected CRS if needed for accurate distance calculations
    if reference.crs and reference.crs.is_geographic:
        # Estimate UTM zone from centroid
        centroid = reference.unary_union.centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone:02d}" if centroid.y >= 0 else f"EPSG:327{utm_zone:02d}"
        logger.info(f"  Reprojecting to {utm_crs} for metric calculations")
        reference = reference.to_crs(utm_crs)
        target = target.to_crs(utm_crs)

    if progress_callback:
        progress_callback(20)

    # Step 2: Generate candidates
    logger.info("Step 2: Generating candidates...")
    candidates = generate_candidates(
        reference=reference,
        target=target,
        buffer_distance=buffer_distance,
        max_heading_diff=max_heading_diff,
        max_length_ratio=max_length_ratio,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )

    logger.info(f"  Generated {len(candidates)} candidates")

    if progress_callback:
        progress_callback(40)

    if not candidates:
        logger.warning("No candidates found! Check data alignment and buffer distance.")

        # Still write empty output files for consistency
        generate_bridge_file(
            matches=[],
            output_path=output_path,
            match_method=method,
        )

        unmatched_path = output_path.parent / "unmatched.parquet"
        generate_unmatched_report(
            target=target,
            matched_ids=set(),
            output_path=unmatched_path,
            id_column=target_id_column,
        )

        return PipelineResult(
            n_reference=len(reference),
            n_target=len(target),
            n_candidates=0,
            n_matched=0,
            n_review=0,
            n_unmatched=len(target),
            bridge_file=output_path,
            unmatched_file=unmatched_path,
        )

    # Step 3: Score candidates
    logger.info("Step 3: Scoring candidates...")

    if method == "rule":
        results = score_candidates(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_name_column=ref_name_column,
            target_name_column=target_name_column,
            ref_class_column=ref_class_column,
            target_class_column=target_class_column,
        )
    elif method == "xgboost":
        from ..matching.ml import MLMatcher

        # Load trained model from default location (combined model for better generalization)
        model_path = "data/models/matcher_model_combined.joblib"
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"ML model not found at {model_path}. "
                "Run 'matcher train --combined' to train the model on labeled data from data/labels/."
            )
        matcher = MLMatcher(model_path=model_path)
        results = matcher.score_candidates(
            candidates,
            reference,
            target,
            ref_name_column=ref_name_column,
            target_name_column=target_name_column,
            ref_class_column=ref_class_column,
            target_class_column=target_class_column,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if progress_callback:
        progress_callback(70)

    # Step 4: Optimize matches with 1:N support (resolve conflicts)
    # 1:N matching allows multiple contiguous target segments to match the same reference
    # This handles different segmentation schemes between datasets
    logger.info(f"Step 4: Optimizing matches with 1:N support (min_confidence={min_confidence})...")
    optimized = optimize_with_one_to_many(
        results,
        target,
        min_confidence=min_confidence,
        # Contiguity tolerance: segments within 5m of each other are considered
        # connected for 1:N matching. This is tighter than buffer_distance (75m)
        # because we want to be confident segments are actually adjacent, not just nearby.
        contiguity_tolerance=5.0,
        target_id_column=target_id_column,
    )

    if progress_callback:
        progress_callback(85)

    # Step 5: Generate output files
    logger.info("Step 5: Generating output files...")

    # Bridge file
    generate_bridge_file(
        matches=optimized,
        output_path=output_path,
        match_method=method,
    )

    # Unmatched report
    matched_target_ids = {m.target_id for m in optimized if m.decision != MatchDecision.NO_MATCH}
    unmatched_path = output_path.parent / "unmatched.parquet"
    generate_unmatched_report(
        target=target,
        matched_ids=matched_target_ids,
        output_path=unmatched_path,
        id_column=target_id_column,
    )

    if progress_callback:
        progress_callback(100)

    # Compute statistics
    n_matched = sum(1 for m in optimized if m.decision == MatchDecision.MATCH)
    n_review = sum(1 for m in optimized if m.decision == MatchDecision.REVIEW)
    n_unmatched = len(target) - len(matched_target_ids)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Matched: {n_matched}")
    logger.info(f"  Review: {n_review}")
    logger.info(f"  Unmatched: {n_unmatched}")
    logger.info("=" * 60)

    return PipelineResult(
        n_reference=len(reference),
        n_target=len(target),
        n_candidates=len(candidates),
        n_matched=n_matched,
        n_review=n_review,
        n_unmatched=n_unmatched,
        bridge_file=output_path,
        unmatched_file=unmatched_path,
    )


def run_topology_pipeline(
    input_path: Path,
    output_dir: Path,
    snap_tolerance: float = 2.0,
    respect_z_levels: bool = True,
) -> dict[str, Any]:
    """Run the topology reconstruction pipeline.

    Args:
        input_path: Path to input GeoParquet/GeoJSON
        output_dir: Directory for output files
        snap_tolerance: Snap tolerance in meters
        respect_z_levels: Whether to respect bridge/tunnel z-levels

    Returns:
        Dictionary with statistics
    """
    from ..topology import build_graph, compute_topology_features, planarize

    logger.info("=" * 60)
    logger.info("Starting topology reconstruction pipeline")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading {input_path}...")
    if input_path.suffix == ".parquet":
        gdf = gpd.read_parquet(input_path)
    else:
        gdf = gpd.read_file(input_path)

    logger.info(f"  Loaded {len(gdf)} features")

    # Planarize
    logger.info("Planarizing...")
    network = planarize(
        gdf,
        snap_tolerance=snap_tolerance,
        respect_z_levels=respect_z_levels,
    )

    # Build graph
    logger.info("Building graph...")
    G = build_graph(network)

    # Compute topology features
    features = compute_topology_features(G)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = output_dir / "nodes.parquet"
    edges_path = output_dir / "edges.parquet"

    network.nodes.to_parquet(nodes_path)
    network.edges.to_parquet(edges_path)

    logger.info(f"Saved nodes to {nodes_path}")
    logger.info(f"Saved edges to {edges_path}")

    logger.info("=" * 60)
    logger.info("Topology reconstruction complete!")
    logger.info(f"  Nodes: {features['n_nodes']}")
    logger.info(f"  Edges: {features['n_edges']}")
    logger.info(f"  Components: {features['n_components']}")
    logger.info("=" * 60)

    return {
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        **features,
    }
