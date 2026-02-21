"""Pipeline orchestration - runs the full matching pipeline."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from loguru import logger

from ..blocking import generate_candidates
from ..config import CLASS_COLUMN, DATA_VERSION, DEFAULT_SNAP_TOLERANCE_M, NAMES_COLUMN, settings
from ..filenames import extract_version_from_filename, groups_sidecar_path
from ..matching import MatchDecision, optimize_matches_with_grouping
from ..matching.graph_consistency import validate_graph_consistency
from ..matching.optimizer import compute_group_id, find_match_components
from ..matching.types import MatchType
from ..resolution import generate_bridge_file, generate_unmatched_report
from ..utils import ensure_projected_crs
from ..utils.crs import ProjectionResult
from ..utils.geometry import filter_to_linestrings


class PipelineError(Exception):
    """Error during pipeline execution."""

    pass


def score_candidates_from_geodataframes(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_distance_m: float | None = None,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    ref_name_column: str = NAMES_COLUMN,
    target_name_column: str = NAMES_COLUMN,
    ref_class_column: str = CLASS_COLUMN,
    target_class_column: str = CLASS_COLUMN,
    n_jobs: int = -1,
    model_path: str | None = None,
    auto_select: bool = False,
) -> tuple[list, ProjectionResult]:
    """Project, block, and score candidates from GeoDataFrames.

    Shared by run_pipeline() and labeling UI's generate_scored_candidates().
    Handles projection to metric CRS, candidate generation (blocking), and
    ML scoring.

    Args:
        reference: Reference GeoDataFrame (Overture)
        target: Target GeoDataFrame (local data)
        buffer_distance_m: Candidate search radius in meters (None = settings default)
        ref_id_column: ID column in reference
        target_id_column: ID column in target
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        ref_class_column: Class column in reference
        target_class_column: Class column in target
        n_jobs: Number of parallel jobs (-1 for all cores)
        model_path: Explicit model path (if None, uses settings.model_path)
        auto_select: If True, auto-select model based on target dataset

    Returns:
        Tuple of (match_results, projection_result) where:
        - match_results: List of MatchResult objects
        - projection_result: ProjectionResult with CRS info
    """
    from ..matching.ml import MLMatcher

    # Project to metric CRS for accurate distances
    projection_result = ensure_projected_crs(reference, target)
    reference_proj = projection_result.reference
    target_proj = projection_result.target
    if projection_result.was_reprojected:
        logger.info(f"Projected to {projection_result.projected_crs} for meter-based computations")

    # Generate candidates (blocking step)
    candidates = generate_candidates(
        reference=reference_proj,
        target=target_proj,
        buffer_distance_m=buffer_distance_m,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )
    logger.info(f"Generated {len(candidates)} candidates")

    if not candidates:
        return [], projection_result

    # Score candidates using ML
    if model_path:
        matcher = MLMatcher(model_path=model_path)
    elif auto_select:
        matcher = MLMatcher(auto_select=True)
    else:
        from ..config import settings as _settings

        _model_path = _settings.model_path
        if not _model_path.exists():
            raise FileNotFoundError(
                f"ML model not found at {_model_path}. "
                "Run 'matcher train' to train the model on labeled data."
            )
        matcher = MLMatcher(model_path=str(_model_path))

    results = matcher.score_candidates(
        candidates,
        reference_proj,
        target_proj,
        ref_name_column=ref_name_column,
        target_name_column=target_name_column,
        ref_class_column=ref_class_column,
        target_class_column=target_class_column,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
        n_jobs=n_jobs,
    )

    return results, projection_result


def validate_data_version(file_path: Path, file_type: str = "data") -> None:
    """Validate that a data file's version matches the current code.

    This is backward-compatible with legacy/unversioned files: if no version
    suffix can be extracted from the filename, a warning is logged and the
    function returns without raising an error.

    Args:
        file_path: Path to the data file
        file_type: Description of the file type for error messages

    Raises:
        PipelineError: If a version suffix is present but does not match
            the expected version.
    """
    file_version = extract_version_from_filename(file_path)
    expected = DATA_VERSION.lstrip("v")  # '1.0'

    if file_version is None:
        # No version suffix - could be legacy file or different naming scheme
        # Log a warning but don't fail (backward compatibility during migration)
        logger.warning(
            f"{file_type} file {file_path.name} has no version suffix. "
            f"Expected format: <name>_{DATA_VERSION}.parquet. "
            f"Re-fetch data with: matcher fetch --for-dataset <name> -d <source>"
        )
        return

    if file_version != expected:
        raise PipelineError(
            f"Version mismatch for {file_path.name}:\n"
            f"  File version: v{file_version}\n"
            f"  Expected: {DATA_VERSION}\n"
            f"Re-fetch data to update."
        )


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
    unmatched_file: Path | None

    # Screen test results (if run)
    n_screen_failed: int | None = None
    n_screen_warned: int | None = None


# Coordinate precision for GeoJSON output in groups sidecar.
# 7 decimal places in WGS84 gives ~1.1cm accuracy.
GEOJSON_COORD_PRECISION = 7

# Precision for alignment fraction values (0-1 linear reference along a segment).
# 7 decimal places gives sub-mm precision on typical road segments.
ALIGNMENT_FRAC_PRECISION = 7


def _is_nan(val) -> bool:
    """Check if a value is NaN (works for float, numpy, pandas NA)."""
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def _extract_name_string(name) -> str:
    """Extract a human-readable name string from various name formats.

    Handles Overture-style name dicts (with 'primary' key) and plain strings.
    """
    if name is None:
        return ""
    if isinstance(name, str):
        return name
    if isinstance(name, dict):
        for key in ("primary", "common", "name", "value"):
            if key in name and name[key] and isinstance(name[key], str):
                return name[key]
        for v in name.values():
            if isinstance(v, str) and v:
                return v
    return ""


def _export_groups_sidecar(
    results: list,
    optimized: list,
    output_path: Path,
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    min_confidence: float,
    ref_id_column: str = "id",
    target_id_column: str = "id",
) -> Path | None:
    """Export a groups sidecar JSON alongside the bridge file.

    For each non-1:1 connected component, serializes the group's edges,
    optimizer assignment, and geometries (WGS84 GeoJSON) for downstream
    stitching review.

    Args:
        results: All raw MatchResult objects (pre-optimization)
        optimized: Optimized MatchResult objects (post-optimization)
        output_path: Path to bridge file (sidecar written alongside)
        reference: Reference GeoDataFrame (WGS84 / original CRS)
        target: Target GeoDataFrame (WGS84 / original CRS)
        min_confidence: Minimum confidence used during optimization
        ref_id_column: Reference ID column name
        target_id_column: Target ID column name

    Returns:
        Path to sidecar file, or None if no groups to export
    """
    import json

    from shapely import to_geojson

    # Re-derive components from raw results
    components = find_match_components(results, min_confidence)

    # Build optimizer assignment lookup from optimized results
    optimizer_edges: dict[str, list[dict]] = defaultdict(list)
    for r in optimized:
        gid = r.features.get("group_id")
        if not gid:
            continue
        optimizer_edges[gid].append(
            {
                "ref_id": str(r.ref_id),
                "target_id": str(r.target_id),
                "confidence": round(float(r.confidence), 4),
            }
        )

    # Build geometry lookups (column must exist; caller passes ref/target_id_column)
    ref_geom_lookup = dict(zip(reference[ref_id_column], reference.geometry))
    tgt_geom_lookup = dict(zip(target[target_id_column], target.geometry))

    # Build name/class lookups for stitching review display
    from ..config import CLASS_COLUMN, NAMES_COLUMN

    ref_name_lookup = (
        dict(zip(reference[ref_id_column], reference[NAMES_COLUMN]))
        if NAMES_COLUMN in reference.columns
        else {}
    )
    tgt_name_lookup = (
        dict(zip(target[target_id_column], target[NAMES_COLUMN]))
        if NAMES_COLUMN in target.columns
        else {}
    )
    ref_class_lookup = (
        dict(zip(reference[ref_id_column], reference[CLASS_COLUMN]))
        if CLASS_COLUMN in reference.columns
        else {}
    )
    tgt_class_lookup = (
        dict(zip(target[target_id_column], target[CLASS_COLUMN]))
        if CLASS_COLUMN in target.columns
        else {}
    )

    groups = []
    for component in components:
        ref_ids = set(r.ref_id for r in component)
        target_ids = set(r.target_id for r in component)

        # Skip 1:1 components
        if len(ref_ids) == 1 and len(target_ids) == 1:
            continue

        group_id = compute_group_id(ref_ids, target_ids)

        # Classify match type
        if len(ref_ids) == 1:
            match_type = MatchType.ONE_TO_N
        elif len(target_ids) == 1:
            match_type = MatchType.N_TO_ONE
        else:
            match_type = MatchType.M_TO_N

        # Serialize edges (cast numpy scalars to Python float for JSON)
        edges = []
        for r in component:
            edge = {
                "ref_id": str(r.ref_id),
                "target_id": str(r.target_id),
                "confidence": round(float(r.confidence), 4),
            }
            if r.gers_start_frac is not None:
                edge["gers_start_frac"] = round(float(r.gers_start_frac), ALIGNMENT_FRAC_PRECISION)
                edge["gers_end_frac"] = round(float(r.gers_end_frac), ALIGNMENT_FRAC_PRECISION)
            if r.local_start_frac is not None:
                edge["local_start_frac"] = round(
                    float(r.local_start_frac), ALIGNMENT_FRAC_PRECISION
                )
                edge["local_end_frac"] = round(float(r.local_end_frac), ALIGNMENT_FRAC_PRECISION)
            edges.append(edge)

        # Serialize geometries as GeoJSON with coordinate rounding
        def _round_coords(coords):
            """Recursively round coordinates to GEOJSON_COORD_PRECISION."""
            if isinstance(coords[0], (list, tuple)):
                return [_round_coords(c) for c in coords]
            return [round(v, GEOJSON_COORD_PRECISION) for v in coords]

        def _geom_to_geojson(geom) -> dict | None:
            if geom is None or geom.is_empty:
                return None
            gj = json.loads(to_geojson(geom))
            gj["coordinates"] = _round_coords(gj["coordinates"])
            return gj

        ref_geometries = {}
        for rid in sorted(str(r) for r in ref_ids):
            geom = ref_geom_lookup.get(rid) or ref_geom_lookup.get(
                int(rid) if rid.isdigit() else rid
            )
            if geom is not None:
                gj = _geom_to_geojson(geom)
                if gj:
                    ref_geometries[str(rid)] = gj

        target_geometries = {}
        for tid in sorted(str(t) for t in target_ids):
            geom = tgt_geom_lookup.get(tid) or tgt_geom_lookup.get(
                int(tid) if tid.isdigit() else tid
            )
            if geom is not None:
                gj = _geom_to_geojson(geom)
                if gj:
                    target_geometries[str(tid)] = gj

        # Collect names and classes for each segment in the group
        ref_names = {}
        ref_classes = {}
        for rid in sorted(str(r) for r in ref_ids):
            name = ref_name_lookup.get(rid)
            if name is not None:
                ref_names[rid] = _extract_name_string(name)
            cls = ref_class_lookup.get(rid)
            if cls is not None:
                ref_classes[rid] = str(cls) if not _is_nan(cls) else ""

        target_names = {}
        target_classes = {}
        for tid in sorted(str(t) for t in target_ids):
            name = tgt_name_lookup.get(tid)
            if name is not None:
                target_names[tid] = _extract_name_string(name)
            cls = tgt_class_lookup.get(tid)
            if cls is not None:
                target_classes[tid] = str(cls) if not _is_nan(cls) else ""

        groups.append(
            {
                "group_id": group_id,
                "match_type": match_type.value,
                "ref_ids": sorted(str(r) for r in ref_ids),
                "target_ids": sorted(str(t) for t in target_ids),
                "edges": edges,
                "optimizer_assignment": optimizer_edges.get(group_id, []),
                "ref_geometries": ref_geometries,
                "target_geometries": target_geometries,
                "ref_names": ref_names,
                "target_names": target_names,
                "ref_classes": ref_classes,
                "target_classes": target_classes,
            }
        )

    if not groups:
        # Remove stale sidecar from a previous run so batch generation
        # doesn't pick up outdated group data.
        stale = groups_sidecar_path(output_path)
        if stale.exists():
            stale.unlink()
            logger.info(f"Removed stale groups sidecar: {stale}")
        return None

    sidecar_path = groups_sidecar_path(output_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    sidecar = {
        "n_groups": len(groups),
        "groups": groups,
    }

    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    logger.info(f"Exported {len(groups)} match groups to {sidecar_path}")
    return sidecar_path


def run_pipeline(
    reference_path: Path,
    target_path: Path,
    output_path: Path,
    method: str = "xgboost",
    buffer_distance_m: float = 75.0,
    min_confidence: float = 0.1,  # Lower = more aggressive matching
    progress_callback: Callable[[int], None] | None = None,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    ref_name_column: str = NAMES_COLUMN,
    target_name_column: str = NAMES_COLUMN,
    ref_class_column: str = CLASS_COLUMN,
    target_class_column: str = CLASS_COLUMN,
    n_jobs: int = -1,
    run_screen: bool = False,
    screen_tests: list[str] | None = None,
) -> PipelineResult:
    """Run the full matching pipeline.

    Args:
        reference_path: Path to reference GeoParquet (Overture)
        target_path: Path to target GeoParquet (local data)
        output_path: Path for output bridge file
        method: Matching method (only "xgboost" supported)
        buffer_distance_m: Candidate search radius in meters
        progress_callback: Optional callback for progress updates
        ref_id_column: ID column in reference
        target_id_column: ID column in target
        ref_name_column: Name column in reference
        target_name_column: Name column in target
        ref_class_column: Class column in reference
        target_class_column: Class column in target
        run_screen: Whether to run screen tests after matching
        screen_tests: Specific screen tests to run (None = all)

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
        raise PipelineError(f"Failed to read reference file {reference_path}: {e}") from e

    try:
        target = gpd.read_parquet(target_path)
    except Exception as e:
        raise PipelineError(f"Failed to read target file {target_path}: {e}") from e

    # Validate geometry columns
    if reference.geometry.isna().any():
        n_null = reference.geometry.isna().sum()
        logger.warning(f"Reference has {n_null} null geometries - these will be skipped")
        reference = reference[~reference.geometry.isna()]

    if target.geometry.isna().any():
        n_null = target.geometry.isna().sum()
        logger.warning(f"Target has {n_null} null geometries - these will be skipped")
        target = target[~target.geometry.isna()]

    # Filter to LineString geometries only (drop MultiLineStrings)
    reference = filter_to_linestrings(reference, source_name="reference")
    target = filter_to_linestrings(target, source_name="target")

    if len(reference) == 0:
        raise PipelineError(
            "Reference dataset is empty after filtering (null geometries and non-LineStrings removed)"
        )
    if len(target) == 0:
        raise PipelineError(
            "Target dataset is empty after filtering (null geometries and non-LineStrings removed)"
        )

    logger.info(f"  Reference: {len(reference)} features from {reference_path}")
    logger.info(f"  Target: {len(target)} features from {target_path}")

    if progress_callback:
        progress_callback(10)

    # Steps 2-3: Generate candidates and score using shared function
    logger.info("Steps 2-3: Generating candidates and scoring...")

    if method == "rule":
        raise ValueError(
            "Rule-based matching has been removed. Use method='xgboost' instead. "
            "Train a model first with 'matcher train'."
        )
    elif method != "xgboost":
        raise ValueError(f"Unknown method: {method}")

    results, projection_result = score_candidates_from_geodataframes(
        reference=reference,
        target=target,
        buffer_distance_m=buffer_distance_m,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
        ref_name_column=ref_name_column,
        target_name_column=target_name_column,
        ref_class_column=ref_class_column,
        target_class_column=target_class_column,
        n_jobs=n_jobs,
    )
    # Ensure WGS84 GeoDataFrames for sidecar export (web map needs EPSG:4326)
    if reference.crs and not reference.crs.equals("EPSG:4326"):
        reference_wgs84 = reference.to_crs("EPSG:4326")
    else:
        reference_wgs84 = reference
    if target.crs and not target.crs.equals("EPSG:4326"):
        target_wgs84 = target.to_crs("EPSG:4326")
    else:
        target_wgs84 = target
    # Update reference/target to projected versions for downstream use
    reference = projection_result.reference
    target = projection_result.target

    logger.info(f"  Generated and scored {len(results)} candidates")

    if not results:
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

    if progress_callback:
        progress_callback(70)

    # Step 4: Optimize matches with M:N grouping (resolve conflicts)
    # Grouping allows multiple contiguous segments and supports 1:1, 1:N, N:1, and M:N match types
    # This handles different segmentation schemes and overlapping relationships between datasets
    logger.info(
        f"Step 4: Optimizing matches with M:N grouping (min_confidence={min_confidence})..."
    )
    optimized = optimize_matches_with_grouping(
        results,
        reference=reference,
        target=target,
        min_confidence=min_confidence,
        contiguity_tolerance=DEFAULT_SNAP_TOLERANCE_M,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )

    # Step 4b: Graph consistency validation — demote topologically
    # inconsistent matches from MATCH → REVIEW.
    logger.info("Step 4b: Validating graph consistency...")
    optimized = validate_graph_consistency(
        optimized,
        reference=reference,
        target=target,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
        snap_tolerance=DEFAULT_SNAP_TOLERANCE_M,
    )

    # Export groups sidecar for stitching review (using WGS84 geometries)
    _export_groups_sidecar(
        results=results,
        optimized=optimized,
        output_path=output_path,
        reference=reference_wgs84,
        target=target_wgs84,
        min_confidence=min_confidence,
        ref_id_column=ref_id_column,
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
        bridge_min_confidence=settings.bridge_min_confidence,
    )

    # Step 5.5: Optional screen tests (placeholder - not yet implemented)
    n_screen_failed = None
    n_screen_warned = None

    if run_screen:
        logger.warning("Screen tests on bridge files not yet implemented, skipping...")

    # Unmatched report
    # Only MATCH decisions count as matched. REVIEW decisions are low-confidence
    # and should appear in unmatched.parquet so they can be labeled/reviewed.
    matched_target_ids = {m.target_id for m in optimized if m.decision == MatchDecision.MATCH}
    review_target_ids = {m.target_id for m in optimized if m.decision == MatchDecision.REVIEW}
    unmatched_path = output_path.parent / "unmatched.parquet"
    generate_unmatched_report(
        target=target,
        matched_ids=matched_target_ids,
        output_path=unmatched_path,
        id_column=target_id_column,
        review_ids=review_target_ids,
    )

    if progress_callback:
        progress_callback(100)

    # Compute statistics - counts should be mutually exclusive and sum to n_target
    n_matched = len(matched_target_ids)
    n_review = len(review_target_ids)
    n_unmatched = len(target) - n_matched - n_review

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Matched: {n_matched}")
    logger.info(f"  Review: {n_review}")
    logger.info(f"  Unmatched: {n_unmatched}")
    if n_screen_failed is not None:
        logger.info(f"  Screen failed: {n_screen_failed}")
        logger.info(f"  Screen warned: {n_screen_warned}")
    logger.info("=" * 60)

    return PipelineResult(
        n_reference=len(reference),
        n_target=len(target),
        n_candidates=len(results),
        n_matched=n_matched,
        n_review=n_review,
        n_unmatched=n_unmatched,
        bridge_file=output_path,
        unmatched_file=unmatched_path,
        n_screen_failed=n_screen_failed,
        n_screen_warned=n_screen_warned,
    )


def run_topology_pipeline(
    input_path: Path,
    output_dir: Path,
    snap_tolerance_m: float = 2.0,
    respect_z_levels: bool = True,
) -> dict[str, Any]:
    """Run the topology reconstruction pipeline.

    Args:
        input_path: Path to input GeoParquet/GeoJSON
        output_dir: Directory for output files
        snap_tolerance_m: Snap tolerance in meters
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
        snap_tolerance_m=snap_tolerance_m,
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
