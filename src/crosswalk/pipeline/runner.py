"""Pipeline orchestration - runs the full matching pipeline."""

import hashlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from loguru import logger

from ..blocking import generate_candidates
from ..config import (
    CLASS_COLUMN,
    DATA_VERSION,
    DEFAULT_SNAP_TOLERANCE_M,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    NAMES_COLUMN,
    settings,
)
from ..fetch.overture import backfill_overture_physical_lr
from ..fetch.target import backfill_physical_lr_from_source_tags
from ..filenames import candidates_sidecar_path, extract_version_from_filename, groups_sidecar_path
from ..matching import MatchDecision, optimize_matches_with_grouping
from ..matching.optimizer import compute_sliver_candidate_edges
from ..matching.types import MatchResult, MatchType
from ..resolution import generate_bridge_file, generate_unmatched_report
from ..utils import ensure_projected_crs
from ..utils.crs import ProjectionResult
from ..utils.geometry import filter_to_linestrings
from ..utils.physical import clip_physical_attributes, physical_attributes


class PipelineError(Exception):
    """Error during pipeline execution."""

    pass


def _partition_target_decision_ids(
    matches: list[MatchResult],
) -> tuple[set[Any], set[Any]]:
    """Return mutually exclusive MATCH and REVIEW target-id sets.

    Optimizer decisions are edge-level, so one target can legitimately have a
    MATCH edge and a separate REVIEW edge. At target-summary level MATCH takes
    precedence: that target is published and must not also inflate the review
    count or reduce the unmatched count.
    """
    matched = {match.target_id for match in matches if match.decision == MatchDecision.MATCH}
    review = {
        match.target_id for match in matches if match.decision == MatchDecision.REVIEW
    } - matched
    return matched, review


def _default_model_path() -> Path:
    """Return the configured production model path.

    ``settings.model_path`` defaults to the artifact bundled with the package.
    ``MATCHER_MODEL_PATH`` is an explicit process-wide override; callers may also
    pass a per-run path to :func:`_resolve_model_path`.
    """
    return Path(settings.model_path)


def _resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve and validate the one model artifact used for a production run.

    There is deliberately no missing-local fallback: the configured default is
    already the bundled model, so a different ``MATCHER_MODEL_PATH`` or explicit
    argument is an opt-in whose mistakes must fail loudly instead of silently
    reverting to another artifact.
    """
    active = _default_model_path() if model_path is None else Path(model_path).expanduser()
    if not active.is_file():
        raise FileNotFoundError(
            f"Active ML model not found: {active}. Production defaults to the bundled "
            "artifact; remove an invalid MATCHER_MODEL_PATH override or pass a valid "
            "--model-path."
        )
    return active


def _is_bundled_model_path(model_path: Path) -> bool:
    """Whether ``model_path`` names the package's lockstep-tested artifact."""
    from ..config import bundled_model_path

    return model_path.resolve() == bundled_model_path().resolve()


def _calibration_active(model_path: str | Path | None = None) -> bool:
    """Whether the pipeline's active model applies isotonic calibration.

    Inspects the exact resolved model used by scoring. Short-circuits without
    loading it when calibration is globally disabled
    (``MLMatcher.calibration_active`` can never be True then).
    """
    # Short-circuit when calibration is globally disabled: skip the model load
    # (and its I/O) entirely — the answer is unconditionally False.
    if not settings.enable_calibration:
        return False

    from ..matching.ml import MLMatcher

    active_model_path = _resolve_model_path(model_path)
    try:
        return MLMatcher(
            model_path=str(active_model_path), allow_version_mismatch=True
        ).calibration_active
    except Exception as exc:  # pragma: no cover - defensive; scorer surfaces load errors
        logger.warning(f"Could not determine calibration state: {exc}")
        return False


def _effective_glue_min_confidence(model_path: str | Path | None = None) -> float:
    """Select the grouping-only glue prune for the pipeline's active model.

    The prune (``optimizer_glue_min_confidence``) was validated by #267 against
    RAW XGBoost scores at 0.5. When the loaded model applies isotonic calibration
    (``MLMatcher.calibration_active``), ``MatchResult.confidence`` is a calibrated
    P(match), so the equivalent operating point is the calibrated image of raw
    0.5 (``settings.optimizer_glue_min_confidence``, ~0.575). An uncalibrated
    model keeps the raw-0.5 point (``settings.optimizer_glue_min_confidence_raw``)
    so it never silently over-prunes. ``model_path`` is the same resolved
    artifact that scored the candidates.
    """
    if _calibration_active(model_path):
        return settings.optimizer_glue_min_confidence
    return settings.optimizer_glue_min_confidence_raw


def _effective_prune_threshold(
    dataset_key: str | None,
    model_path: str | Path | None = None,
) -> float:
    """Resolve the confidence-drop prune floor for this run (0 = disabled).

    The prune is PER-DATASET OPT-IN via an allowlist: it applies ONLY to datasets
    with an explicit, validated threshold in ``settings.resolver_prune_overrides``,
    and only while ``settings.resolver_prune_enabled`` (the master switch) is True.
    A dataset absent from the allowlist is NOT pruned — the floor is
    dataset-dependent and the Boston-tuned 0.96 over-prunes never-tuned /
    sidewalk-like sets (#284 sweep), so applying it by default silently
    over-prunes. An allowlist value <= 0 keeps a listed dataset explicitly
    disabled. Returns 0.0 when the prune is off, which
    ``apply_confidence_drop_prune`` treats as a no-op (selections byte-identical to
    the pre-prune pipeline).

    KEYED ON DATASET IDENTITY ONLY (#348). ``dataset_key`` is the dataset name the
    caller already knows — ``crosswalk stitch``'s dataset argument or the factory
    pair name. It is NEVER derived from the output path: the previous
    filename-based resolution (bridge stem minus ``_bridge``, with ``before_`` /
    ``after_`` prefix stripping) silently skipped pruning for any other output
    name (e.g. ``after4_us_boston_streets_bridge.parquet``), changing match counts
    mid-measurement. ``dataset_key=None`` means the run has no dataset identity
    (raw ``--reference``/``--target`` path mode without a dataset name) — the
    prune is off and a log line says so.

    Every branch logs its decision loudly so a run's prune state is never silent:
    enabled (dataset @ threshold), not allowlisted, explicitly disabled, no
    dataset identity, master switch off, or calibration guard.

    Calibration guard: every allowlist operating point was tuned and validated ONLY
    on CALIBRATED ``MatchResult.confidence`` (#272/#284). Unlike the glue prune, NO
    raw-score operating point was validated. Applying the calibrated floor to raw
    XGBoost scores would silently over-prune — raw match scores seldom clear 0.9x,
    so nearly every non-top M:N/1:N/N:1 edge would be dropped, collapsing multi-edge
    groups to their single best edge. So, mirroring the glue-prune guarantee, the
    prune is skipped when the active model applies no calibration.
    """
    if not settings.resolver_prune_enabled:
        logger.info(
            "Resolver confidence-drop prune OFF: master switch "
            "settings.resolver_prune_enabled is False."
        )
        return 0.0

    if dataset_key is None:
        # No dataset identity for this run (raw --reference/--target path mode
        # without a dataset name). The allowlist is keyed by dataset name, so
        # there is nothing to resolve — the prune is off. Deriving an identity
        # from the output filename is exactly the #348 footgun, so we never do.
        logger.info(
            "Resolver confidence-drop prune OFF: no dataset identity for this run "
            "(raw path mode). The allowlist (settings.resolver_prune_overrides) is "
            "keyed by dataset name; pass a dataset name to 'crosswalk stitch' to "
            "apply a tuned prune."
        )
        return 0.0

    allowlist = settings.resolver_prune_overrides or {}
    if dataset_key not in allowlist:
        # Not in the validated allowlist: opt-in only, so the prune is off. One
        # info line makes the skip visible (vs. silently over-pruning by default).
        logger.info(
            f"Resolver confidence-drop prune OFF for dataset '{dataset_key}': "
            "not in the validated allowlist (settings.resolver_prune_overrides). "
            "Tune it via the #284 sweep recipe before enabling."
        )
        return 0.0

    threshold = max(0.0, float(allowlist[dataset_key]))
    if threshold <= 0:
        # Explicitly disabled for this dataset (allowlist value <= 0).
        logger.info(
            f"Resolver confidence-drop prune OFF for dataset '{dataset_key}': "
            "explicitly disabled (allowlist value <= 0)."
        )
        return 0.0

    if not _calibration_active(model_path):
        logger.warning(
            "Resolver confidence-drop prune skipped: its operating points are "
            "calibrated-only, but the active model applies no calibration "
            f"(enable_calibration={settings.enable_calibration}). No raw-score "
            "floor is validated, so pruning here would silently over-prune."
        )
        return 0.0

    logger.info(
        f"Resolver confidence-drop prune ON for dataset '{dataset_key}': "
        f"allowlisted @ threshold {threshold} (calibrated confidence floor)."
    )
    return threshold


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
    model_path: str | Path | None = None,
    auto_select: bool = False,
    allow_version_mismatch: bool = False,
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
        model_path: Explicit production model path. If None, uses
            ``settings.model_path`` (bundled by default; ``MATCHER_MODEL_PATH``
            is the explicit environment override).
        auto_select: If True, auto-select model based on target dataset

    Returns:
        Tuple of (match_results, projection_result) where:
        - match_results: List of MatchResult objects
        - projection_result: ProjectionResult with CRS info
    """
    from ..matching.ml import MLMatcher

    # The labeling UI's auto-select flow intentionally resolves its local full /
    # geometry-only models later from dataset name coverage. Every production
    # path resolves its artifact exactly once here and fails before doing work if
    # an explicit override is missing.
    active_model_path = (
        None if auto_select and model_path is None else _resolve_model_path(model_path)
    )

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
    if active_model_path is None:
        matcher = MLMatcher(auto_select=True)
    else:
        bundled = _is_bundled_model_path(active_model_path)
        source = "bundled" if bundled else "explicit override"
        logger.info(f"Using {source} production model: {active_model_path}")
        matcher = MLMatcher(
            model_path=str(active_model_path),
            # The bundled artifact's feature-version lockstep is enforced by CI.
            allow_version_mismatch=allow_version_mismatch or bundled,
        )

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
            f"Re-fetch data with: crosswalk fetch --for-dataset <name> -d <source>"
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

CANDIDATE_SIDECAR_SCHEMA_VERSION = "1.1"


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


def _compute_candidate_graph_by_group(
    results: list,
    optimized: list,
    assignment_by_gid: dict[str, list],
    min_confidence: float,
    sliver_edges: set[tuple[Any, Any]],
    glue_min_confidence: float | None,
    pruned_gid_by_pair: dict[tuple[str, str], str] | None = None,
) -> dict[str, list[dict]]:
    """Attribute the FULL pre-selection candidate graph to sidecar groups.

    Learned-resolver flip condition #1 (docs/SCALING_ROADMAP.md): persist every
    candidate edge the optimizer saw — not just the selected assignment — so
    under-selection is learnable. Recomputes the optimizer's connected
    components from the SAME inputs it used (``find_match_components`` is
    deterministic), deduplicates to the highest-confidence result per
    (ref, target) pair, and attributes every component edge to EXACTLY ONE
    sidecar group.

    Attribution rule (components are enumerated pre-decomposition, so edges are
    attributed to the post-decomposition groups; ties break on the
    lexicographically smallest ``group_id`` for determinism):

    0. A confidence-pruned pair belongs to its snapshotted PRE-prune group.
    1. Else, a pair in some group's optimizer assignment belongs to THAT group.
    2. Else, a group containing BOTH endpoints (in-product alternative).
    3. Else, a group containing the ref endpoint.
    4. Else, a group containing the target endpoint.
    5. Else (neither endpoint was selected into any group — e.g. both resolved
       as 1:1 leftovers), the smallest group_id among groups touching the
       component, so no candidate in a grouped component is ever lost.

    Components that produced NO sidecar group (pure-1:1 components) have no
    group to attach to and are not represented — by construction the sidecar
    only carries non-1:1 groups. An assignment pair absent from every component
    (e.g. greedy expansion recovered an unattached weak edge) is added back to
    its own group so ``candidate_edges`` is always a superset of the group's
    assignment.

    Per edge: ``ref_id``, ``target_id``, ``confidence`` (raw ML score, rounded
    like every other sidecar confidence), ``selected`` (True iff the pair is in
    the OWNING group's assignment), ``pruned: true`` for an owned confidence-
    pruned pair, and ``selected_elsewhere: true`` only when a non-selected edge
    was selected by the optimizer somewhere else (another group or a 1:1 match)
    — so a resolver never learns a genuinely-selected pair as an optimizer drop.

    Returns:
        Mapping of ``group_id`` -> candidate edge dicts, each list sorted by
        (ref_id, target_id) for deterministic output.
    """
    from ..matching.optimizer import _match_result_rank, find_match_components

    pruned_gid_by_pair = pruned_gid_by_pair or {}
    pruned_pairs_to_restore = set(pruned_gid_by_pair)
    gid_refs = {gid: {str(r.ref_id) for r in assign} for gid, assign in assignment_by_gid.items()}
    gid_tgts = {
        gid: {str(r.target_id) for r in assign} for gid, assign in assignment_by_gid.items()
    }
    pair_to_assignment_gid: dict[tuple[str, str], str] = {}
    for gid, assign in assignment_by_gid.items():
        for r in assign:
            pair_to_assignment_gid.setdefault((str(r.ref_id), str(r.target_id)), gid)

    # Every pair the optimizer selected ANYWHERE (groups + 1:1), post-prune.
    selected_global = {(str(r.ref_id), str(r.target_id)) for r in optimized}
    selected_pruned = sorted(pruned_pairs_to_restore & selected_global)
    if selected_pruned:
        raise ValueError(
            f"Candidate graph received pairs marked both selected and pruned: {selected_pruned[:5]}"
        )

    # Best floor-passing scored result for every pair. Component reconstruction
    # below intentionally omits some disconnected/glue-pruned edges; the global
    # lookup lets rule 0 restore an edge the optimizer actually selected before
    # the confidence-drop prune, even when both endpoints left the surviving
    # group and the edge no longer touches a reconstructed component group.
    best_result_by_pair: dict[tuple[str, str], Any] = {}
    for r in results:
        if _is_nan(r.confidence) or r.confidence < min_confidence:
            continue
        pair = (str(r.ref_id), str(r.target_id))
        prev = best_result_by_pair.get(pair)
        if prev is None or _match_result_rank(r) < _match_result_rank(prev):
            best_result_by_pair[pair] = r

    missing_owners = sorted(
        (pair, owner)
        for pair, owner in pruned_gid_by_pair.items()
        if owner not in assignment_by_gid
    )
    if missing_owners:
        raise ValueError(
            "Candidate graph cannot restore pruned pairs whose pre-prune group "
            f"did not survive: {missing_owners[:5]}"
        )
    missing_results = sorted(pruned_pairs_to_restore - best_result_by_pair.keys())
    if missing_results:
        raise ValueError(
            "Candidate graph cannot restore pruned pairs without a floor-passing "
            f"scored result: {missing_results[:5]}"
        )

    # Recompute the optimizer's components. Deterministic given the same
    # inputs (results, floor, sliver set, glue prune) the optimizer used.
    components = find_match_components(
        results,
        min_confidence,
        sliver_edges=sliver_edges,
        glue_min_confidence=glue_min_confidence,
    )

    # Best result per pair within each component + node -> component index.
    node_comp: dict[tuple[str, str], int] = {}
    comp_best: list[dict[tuple[str, str], Any]] = []
    for ci, comp in enumerate(components):
        best: dict[tuple[str, str], Any] = {}
        for r in comp:
            pair = (str(r.ref_id), str(r.target_id))
            prev = best.get(pair)
            if prev is None or _match_result_rank(r) < _match_result_rank(prev):
                best[pair] = r
        comp_best.append(best)
        for rid, tid in best:
            node_comp[("ref", rid)] = ci
            node_comp[("target", tid)] = ci

    # Groups touching each component. A group's nodes normally live in one
    # component, but greedy expansion can stitch nodes across components, so
    # membership is collected per node rather than assumed.
    comp_gids: dict[int, set[str]] = defaultdict(set)
    for gid in assignment_by_gid:
        for rid in gid_refs[gid]:
            ci = node_comp.get(("ref", rid))
            if ci is not None:
                comp_gids[ci].add(gid)
        for tid in gid_tgts[gid]:
            ci = node_comp.get(("target", tid))
            if ci is not None:
                comp_gids[ci].add(gid)

    def _edge_dict(
        pair: tuple[str, str], confidence: float, selected: bool, *, pruned: bool = False
    ) -> dict:
        edge = {
            "ref_id": pair[0],
            "target_id": pair[1],
            "confidence": round(float(confidence), 4),
            "selected": selected,
        }
        if pruned:
            edge["pruned"] = True
        elif not selected and pair in selected_global:
            edge["selected_elsewhere"] = True
        return edge

    by_gid: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for ci, gids in comp_gids.items():
        gids_sorted = sorted(gids)
        for pair, r in comp_best[ci].items():
            # Authoritative rule 0 restores these directly below. Skipping them
            # here avoids both temporary foreign attribution and an O(P*G)
            # cleanup pass on datasets with many pruned pairs/groups.
            if pair in pruned_pairs_to_restore:
                continue
            rid, tid = pair
            owner = pair_to_assignment_gid.get(pair)
            if owner is None:
                owner = next(
                    (g for g in gids_sorted if rid in gid_refs[g] and tid in gid_tgts[g]), None
                )
            if owner is None:
                owner = next((g for g in gids_sorted if rid in gid_refs[g]), None)
            if owner is None:
                owner = next((g for g in gids_sorted if tid in gid_tgts[g]), None)
            if owner is None:
                owner = gids_sorted[0]
            selected = pair_to_assignment_gid.get(pair) == owner
            by_gid[owner][pair] = _edge_dict(pair, r.confidence, selected)

    # Completeness backstop: every assignment pair appears in its own group's
    # candidate list even when it was in no component (see docstring).
    for gid, assign in assignment_by_gid.items():
        for r in assign:
            pair = (str(r.ref_id), str(r.target_id))
            if pair not in by_gid[gid]:
                by_gid[gid][pair] = _edge_dict(pair, r.confidence, True)

    # Rule 0 is authoritative and runs last. The normal attribution pass skipped
    # these prevalidated pairs, so each is restored exactly once in O(P), under
    # its pre-prune owner, including fully detached pendant edges.
    for pair, owner in pruned_gid_by_pair.items():
        result = best_result_by_pair[pair]
        by_gid[owner][pair] = _edge_dict(pair, result.confidence, False, pruned=True)

    return {gid: [pairs[p] for p in sorted(pairs)] for gid, pairs in by_gid.items()}


def _signed_lateral_offset_m(ref_geom, target_geom) -> float:
    """Mean target offset signed left (+) / right (-) of ref orientation.

    The existing ``lateral_offset_m`` feature is an unsigned distance.  This
    companion value samples the target, projects each sample to the reference,
    and signs its distance with the local reference tangent.  It deliberately
    preserves the reference geometry's stored orientation; a future group
    decoder can normalize corridor direction before measuring sign consistency.
    """
    if (
        ref_geom is None
        or target_geom is None
        or ref_geom.is_empty
        or target_geom.is_empty
        or ref_geom.length <= 0
        or target_geom.length <= 0
    ):
        return float("nan")

    signed: list[float] = []
    tangent_step = max(float(ref_geom.length) * 1e-6, 1e-4)
    for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
        sample = target_geom.interpolate(fraction, normalized=True)
        along = float(ref_geom.project(sample))
        nearest = ref_geom.interpolate(along)
        before = ref_geom.interpolate(max(0.0, along - tangent_step))
        after = ref_geom.interpolate(min(float(ref_geom.length), along + tangent_step))
        tx = float(after.x - before.x)
        ty = float(after.y - before.y)
        vx = float(sample.x - nearest.x)
        vy = float(sample.y - nearest.y)
        cross = tx * vy - ty * vx
        tangent_length = (tx * tx + ty * ty) ** 0.5
        signed.append(cross / tangent_length if tangent_length > 0 else 0.0)
    return float(sum(signed) / len(signed))


def _active_model_hash(model_path: str | Path | None = None) -> str:
    """SHA-256 of the exact model used by this stitch scoring path."""
    path = _resolve_model_path(model_path)
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Non-feature columns of the typed candidates sidecar parquet, in row order.
# The full parquet column set is this tuple + config.FEATURE_COLUMNS. This MUST
# stay in lockstep with the ``row`` dict built in ``_export_candidates_sidecar``
# below — enforced by a unit test that runs the real writer and compares emitted
# columns (tests/unit/test_groups_sidecar_candidate_graph.py::
# test_candidate_parquet_columns_match_declared_constant). The resolver training
# table derives its known/expected join set from this constant
# (resolver/extract.py::EXPECTED_CANDIDATE_JOIN_COLUMNS), so a column added to
# the writer without updating this tuple fails the lockstep test rather than
# silently accreting into (or falsely warning on) the resolver table.
CANDIDATE_SIDECAR_BASE_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "group_id",
    "ref_id",
    "target_id",
    "ref_idx",
    "target_idx",
    "selected",
    "selected_elsewhere",
    "pruned",
    "is_sliver",
    "decision",
    "optimizer_decision",
    "decision_reason",
    "confidence",
    "gers_start_frac",
    "gers_end_frac",
    "local_start_frac",
    "local_end_frac",
    "degree_ref",
    "degree_tgt",
    "candidate_graph_bridge",
    "is_bridge",
    "biconnected_block",
    "corridor_ref",
    "corridor_tgt",
    "match_type",
    "n_edges",
    "n_candidate_edges",
    "n_corridors",
    "n_assignment_components",
    "largest_biconnected_block",
    "oversized_group",
    "ref_class",
    "target_class",
    "ref_length_m",
    "target_length_m",
    "lateral_offset_signed_m",
    "feature_version",
    "model_hash",
    "calibration_active",
    "schema_version",
)


def _export_candidates_sidecar(
    *,
    groups: list[dict],
    candidate_graph_by_gid: dict[str, list[dict]],
    best_by_pair: dict[tuple[str, str], Any],
    optimized_by_pair: dict[tuple[str, str], Any],
    output_path: Path,
    reference_proj: gpd.GeoDataFrame,
    target_proj: gpd.GeoDataFrame,
    ref_id_column: str,
    target_id_column: str,
    sliver_pairs_str: set[tuple[str, str]],
    pruned_gid_by_pair: dict[tuple[str, str], str],
    dataset_id: str | None,
    model_path: str | Path | None = None,
) -> Path | None:
    """Persist one typed parquet row for every stage-1 resolver candidate."""
    path = candidates_sidecar_path(output_path)
    if not settings.stitch_persist_candidates or not groups:
        if path.exists():
            path.unlink()
            logger.info(f"Removed stale candidates sidecar: {path}")
        return None

    from ..features.alignment import create_subline
    from ..matching.optimizer import compute_group_structure, optimizer_review_reason

    ref_geoms = reference_proj.geometry.to_numpy()
    target_geoms = target_proj.geometry.to_numpy()
    ref_classes = (
        reference_proj[CLASS_COLUMN].to_numpy() if CLASS_COLUMN in reference_proj.columns else None
    )
    target_classes = (
        target_proj[CLASS_COLUMN].to_numpy() if CLASS_COLUMN in target_proj.columns else None
    )
    ref_geom_by_id = {
        str(segment_id): geom
        for segment_id, geom in zip(reference_proj[ref_id_column], reference_proj.geometry)
    }
    target_geom_by_id = {
        str(segment_id): geom
        for segment_id, geom in zip(target_proj[target_id_column], target_proj.geometry)
    }
    ref_class_by_id = (
        {
            str(segment_id): value
            for segment_id, value in zip(
                reference_proj[ref_id_column], reference_proj[CLASS_COLUMN]
            )
        }
        if ref_classes is not None
        else {}
    )
    target_class_by_id = (
        {
            str(segment_id): value
            for segment_id, value in zip(target_proj[target_id_column], target_proj[CLASS_COLUMN])
        }
        if target_classes is not None
        else {}
    )

    def _at(values, idx):
        if values is None or idx is None or idx < 0 or idx >= len(values):
            return None
        return values[idx]

    def _class_value(value) -> str | None:
        return None if value is None or _is_nan(value) else str(value)

    def _aligned(geom, start, end):
        if geom is None or start is None or end is None or _is_nan(start) or _is_nan(end):
            return geom
        subline = create_subline(geom, float(start), float(end))
        return subline if subline is not None else geom

    group_by_gid = {str(group["group_id"]): group for group in groups}
    active_model_path = _resolve_model_path(model_path)
    model_hash = _active_model_hash(active_model_path)
    calibration_active = _calibration_active(active_model_path)
    rows: list[dict[str, Any]] = []
    for gid in sorted(candidate_graph_by_gid):
        group = group_by_gid.get(gid)
        if group is None:
            continue
        candidates = candidate_graph_by_gid[gid]
        pairs = [(str(edge["ref_id"]), str(edge["target_id"])) for edge in candidates]
        assignment_pairs = {
            (str(edge["ref_id"]), str(edge["target_id"]))
            for edge in group.get("optimizer_assignment", [])
        }
        all_ref_ids = sorted({pair[0] for pair in pairs})
        all_target_ids = sorted({pair[1] for pair in pairs})
        structure, _ = compute_group_structure(
            edges=pairs,
            ref_ids=all_ref_ids,
            target_ids=all_target_ids,
            assignment_pairs=assignment_pairs,
            sliver_pairs=sliver_pairs_str,
            ref_geoms=ref_geom_by_id,
            target_geoms=target_geom_by_id,
            tolerance=DEFAULT_SNAP_TOLERANCE_M,
            corridor_aware=settings.optimizer_corridor_aware,
            max_turn_deg=settings.optimizer_corridor_max_turn_deg,
        )

        for candidate, pair in zip(candidates, pairs):
            result = best_by_pair.get(pair)
            if result is None:
                logger.warning(f"Candidate sidecar skipped missing scored pair {gid}: {pair}")
                continue
            ref_idx = getattr(result, "ref_idx", None)
            target_idx = getattr(result, "target_idx", None)
            ref_geom = _at(ref_geoms, ref_idx)
            if ref_geom is None:
                ref_geom = ref_geom_by_id.get(pair[0])
            target_geom = _at(target_geoms, target_idx)
            if target_geom is None:
                target_geom = target_geom_by_id.get(pair[1])
            ref_class = _at(ref_classes, ref_idx)
            if ref_class is None:
                ref_class = ref_class_by_id.get(pair[0])
            target_class = _at(target_classes, target_idx)
            if target_class is None:
                target_class = target_class_by_id.get(pair[1])
            aligned_ref = _aligned(ref_geom, result.gers_start_frac, result.gers_end_frac)
            aligned_target = _aligned(target_geom, result.local_start_frac, result.local_end_frac)

            pruned = pruned_gid_by_pair.get(pair) == gid
            selected = bool(candidate.get("selected", False))
            selected_elsewhere = bool(candidate.get("selected_elsewhere", False))
            features = getattr(result, "features", {}) or {}
            if pruned:
                optimizer_decision = "pruned"
                decision_reason = "confidence_drop_prune"
            elif selected:
                optimized_result = optimized_by_pair.get(pair)
                review_reason = (
                    optimizer_review_reason(optimized_result)
                    if optimized_result is not None
                    else None
                )
                optimizer_decision = "review" if review_reason else "selected"
                decision_reason = review_reason or "optimizer_assignment"
            elif selected_elsewhere:
                optimizer_decision = "selected_elsewhere"
                decision_reason = "assigned_to_other_group"
            else:
                optimizer_decision = "rejected"
                decision_reason = "optimizer_rejected"

            edge_structure = structure.get(pair, {})
            candidate_graph_bridge = bool(edge_structure.get("is_bridge", False))
            row: dict[str, Any] = {
                "dataset_id": dataset_id or "",
                "group_id": gid,
                "ref_id": pair[0],
                "target_id": pair[1],
                "ref_idx": ref_idx,
                "target_idx": target_idx,
                "selected": selected,
                "selected_elsewhere": selected_elsewhere,
                "pruned": pruned,
                "is_sliver": bool(edge_structure.get("is_sliver", pair in sliver_pairs_str)),
                "decision": result.decision.value,
                "optimizer_decision": optimizer_decision,
                "decision_reason": decision_reason,
                "confidence": float(result.confidence),
                "gers_start_frac": result.gers_start_frac,
                "gers_end_frac": result.gers_end_frac,
                "local_start_frac": result.local_start_frac,
                "local_end_frac": result.local_end_frac,
                "degree_ref": edge_structure.get("degree_ref"),
                "degree_tgt": edge_structure.get("degree_tgt"),
                "candidate_graph_bridge": candidate_graph_bridge,
                # Resolver feature compatibility: the trained feature schema still
                # calls this graph-theory value ``is_bridge``. New human-facing
                # evidence uses the explicit name above.
                "is_bridge": candidate_graph_bridge,
                "biconnected_block": edge_structure.get("biconnected_block"),
                "corridor_ref": edge_structure.get("corridor_ref"),
                "corridor_tgt": edge_structure.get("corridor_tgt"),
                "match_type": group.get("match_type"),
                "n_edges": group.get("n_edges"),
                "n_candidate_edges": len(candidates),
                "n_corridors": group.get("n_corridors"),
                "n_assignment_components": group.get("n_assignment_components"),
                "largest_biconnected_block": group.get("largest_biconnected_block"),
                "oversized_group": bool(group.get("oversized_group", False)),
                "ref_class": _class_value(ref_class),
                "target_class": _class_value(target_class),
                "ref_length_m": float(ref_geom.length) if ref_geom is not None else float("nan"),
                "target_length_m": (
                    float(target_geom.length) if target_geom is not None else float("nan")
                ),
                "lateral_offset_signed_m": _signed_lateral_offset_m(aligned_ref, aligned_target),
                "feature_version": FEATURE_VERSION,
                "model_hash": model_hash,
                "calibration_active": calibration_active,
                "schema_version": CANDIDATE_SIDECAR_SCHEMA_VERSION,
            }
            for feature_name in FEATURE_COLUMNS:
                try:
                    row[feature_name] = float(features.get(feature_name, float("nan")))
                except (TypeError, ValueError):
                    row[feature_name] = float("nan")
            rows.append(row)

    if not rows:
        if path.exists():
            path.unlink()
        return None

    frame = pd.DataFrame(rows).sort_values(["group_id", "ref_id", "target_id"])
    string_columns = [
        "dataset_id",
        "group_id",
        "ref_id",
        "target_id",
        "decision",
        "optimizer_decision",
        "decision_reason",
        "match_type",
        "ref_class",
        "target_class",
        "feature_version",
        "model_hash",
        "schema_version",
    ]
    nullable_int_columns = [
        "ref_idx",
        "target_idx",
        "degree_ref",
        "degree_tgt",
        "biconnected_block",
        "corridor_ref",
        "corridor_tgt",
        "n_edges",
        "n_candidate_edges",
        "n_corridors",
        "n_assignment_components",
        "largest_biconnected_block",
    ]
    bool_columns = [
        "selected",
        "selected_elsewhere",
        "pruned",
        "is_sliver",
        "candidate_graph_bridge",
        "is_bridge",
        "oversized_group",
        "calibration_active",
    ]
    float_columns = [
        "confidence",
        "gers_start_frac",
        "gers_end_frac",
        "local_start_frac",
        "local_end_frac",
        "ref_length_m",
        "target_length_m",
        "lateral_offset_signed_m",
        *FEATURE_COLUMNS,
    ]
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    for column in nullable_int_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in bool_columns:
        frame[column] = frame[column].astype(bool)
    for column in float_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(path)
    logger.info(
        f"Exported {len(frame)} typed resolver candidates "
        f"({len(FEATURE_COLUMNS)} pair features) to {path}"
    )
    return path


def _export_groups_sidecar(
    results: list,
    optimized: list,
    output_path: Path,
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    min_confidence: float,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    sliver_edges: set[tuple[Any, Any]] | None = None,
    reference_proj: gpd.GeoDataFrame | None = None,
    target_proj: gpd.GeoDataFrame | None = None,
    pruned_pairs: set[tuple[Any, Any]] | None = None,
    pruned_group_ids: dict[tuple[Any, Any], Any] | None = None,
    glue_min_confidence: float | None = None,
    dataset_id: str | None = None,
    model_path: str | Path | None = None,
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
        sliver_edges: Junction-sliver candidate pairs to exclude from
            component adjacency (must be the SAME set the optimizer used so
            sidecar groups match optimizer grouping). When None, it is
            recomputed here from the provided GeoDataFrames.
        glue_min_confidence: The grouping-only glue prune the optimizer ran
            with. Used ONLY to reproduce the optimizer's components for the
            full candidate-graph persistence (``candidate_edges``); pass the
            same value given to ``optimize_matches_with_grouping`` so the
            recomputed components match exactly. None (legacy callers) builds
            components with no glue prune.

    Returns:
        Path to sidecar file, or None if no groups to export
    """
    import json

    from shapely import to_geojson

    from ..matching.optimizer import (
        _geom_lookup,
        _match_result_rank,
        _name_lookup,
        compute_group_structure,
        group_is_structurally_simple,
        optimizer_review_reason,
    )

    # Structure/corridor analysis needs metric geometry; fall back to the
    # display GeoDataFrames only if projected ones were not supplied.
    if reference_proj is None:
        reference_proj = reference
    if target_proj is None:
        target_proj = target

    if sliver_edges is None:
        sliver_edges = compute_sliver_candidate_edges(
            results, reference, target, ref_id_column, target_id_column
        )
    sliver_pairs_str = {(str(r), str(t)) for r, t in sliver_edges}

    # The sidecar mirrors the optimizer's DECOMPOSED grouping: after
    # corridor-aware M:N splitting, each optimizer sub-group is a per-corridor
    # unit (not the raw over-merged component). Group the optimizer assignment
    # by its sub-group ``group_id``; 1:1 matches carry no group_id and are
    # excluded. Each sub-group becomes one sidecar group, with candidate edges
    # restricted to that sub-group's ref x target id product (so cross-corridor
    # alternatives, which belong to a different sub-group, are not shown here).
    assignment_by_gid: dict[str, list] = defaultdict(list)
    optimized_by_pair: dict[tuple[str, str], Any] = {}
    for r in optimized:
        pair = (str(r.ref_id), str(r.target_id))
        previous = optimized_by_pair.get(pair)
        if previous is None or _match_result_rank(r) < _match_result_rank(previous):
            optimized_by_pair[pair] = r
        gid = r.features.get("group_id")
        if gid:
            assignment_by_gid[str(gid)].append(r)

    # Highest-confidence raw candidate per (ref,target) pair, indexed by ref AND
    # by target for fast per-group candidate collection. Keyed by string ids to
    # match output.
    best_by_pair: dict[tuple[str, str], Any] = {}
    for r in results:
        # NaN comparisons are always False, so `r.confidence < min_confidence`
        # alone would let a NaN-confidence candidate pass the floor and flow
        # into the sidecar, where json.dump serializes it as a bare `NaN`
        # token -- invalid strict JSON for non-Python consumers. Treat NaN as
        # failing the floor (excluded), matching the `_is_nan` guard already
        # used elsewhere in this module for other NaN-prone fields.
        if _is_nan(r.confidence) or r.confidence < min_confidence:
            continue
        pair = (str(r.ref_id), str(r.target_id))
        prev = best_by_pair.get(pair)
        if prev is None or _match_result_rank(r) < _match_result_rank(prev):
            best_by_pair[pair] = r
    cands_by_ref: dict[str, list] = defaultdict(list)
    cands_by_tgt: dict[str, list] = defaultdict(list)
    for (rid, tid), r in best_by_pair.items():
        cands_by_ref[rid].append(r)
        cands_by_tgt[tid].append(r)

    # Every (ref,target) pair the optimizer SELECTED into any group (post-prune).
    # A rejected candidate must never collide with a pair that is selected in
    # some other group, so this global set gates the rejected-edge collection.
    pruned_pairs = pruned_pairs or set()
    pruned_pairs_str = {(str(r), str(t)) for r, t in pruned_pairs}
    # Str-keyed pruned-pair -> original (pre-prune) group_id. Lets each group
    # claim exactly its own pruned edges, so ``n_pruned`` stays exact even for
    # pendant edges whose both endpoints left the surviving group (they are in
    # neither this group's in-product `edges` nor its incident rejected set).
    pruned_gid_by_pair: dict[tuple[str, str], str] = {
        (str(r), str(t)): str(gid)
        for (r, t), gid in (pruned_group_ids or {}).items()
        if gid is not None
    }
    # Group attribution is the stronger form of the SAME prune provenance, not
    # an independent source of truth. Production snapshots both together; fail
    # closed on partial/mismatched metadata so JSON/parquet flags and n_pruned
    # cannot disagree. ``None`` retains the legacy unattributed fallback.
    if pruned_group_ids is not None and set(pruned_gid_by_pair) != pruned_pairs_str:
        missing_attribution = sorted(pruned_pairs_str - pruned_gid_by_pair.keys())
        unexpected_attribution = sorted(pruned_gid_by_pair.keys() - pruned_pairs_str)
        raise ValueError(
            "Pruned pair/group attribution mismatch: "
            f"missing={missing_attribution[:5]}, unexpected={unexpected_attribution[:5]}"
        )

    # Index ownership once so group serialization is O(P + G), not O(P*G).
    pruned_pairs_by_gid: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for pair, gid in pruned_gid_by_pair.items():
        pruned_pairs_by_gid[gid].add(pair)
    all_selected_pairs: set[tuple[str, str]] = set()
    for _gid, _assign in assignment_by_gid.items():
        for r in _assign:
            all_selected_pairs.add((str(r.ref_id), str(r.target_id)))

    persist_rejected = settings.stitch_persist_rejected_edges
    rejected_cap = settings.stitch_rejected_edges_max_per_group

    # Full candidate-graph persistence (learned-resolver flip condition #1):
    # every floor-passing candidate pair in each group's component, attributed
    # to exactly one group with a ``selected`` flag. Purely additive sibling key
    # (``candidate_edges``); no existing key or consumer is affected. See
    # ``_compute_candidate_graph_by_group`` for the attribution rule.
    candidate_graph_by_gid: dict[str, list[dict]] = {}
    if (
        settings.stitch_persist_candidate_graph or settings.stitch_persist_candidates
    ) and assignment_by_gid:
        candidate_graph_by_gid = _compute_candidate_graph_by_group(
            results=results,
            optimized=optimized,
            assignment_by_gid=assignment_by_gid,
            min_confidence=min_confidence,
            sliver_edges=sliver_edges,
            glue_min_confidence=glue_min_confidence,
            pruned_gid_by_pair=pruned_gid_by_pair,
        )
        for candidate_edges in candidate_graph_by_gid.values():
            for edge in candidate_edges:
                if not edge.get("selected"):
                    continue
                pair = (str(edge["ref_id"]), str(edge["target_id"]))
                optimized_result = optimized_by_pair.get(pair)
                if optimized_result is None:
                    continue
                edge["decision"] = optimized_result.decision.value
                review_reason = optimizer_review_reason(optimized_result)
                if review_reason:
                    edge["review_reason"] = review_reason

    # Projected (metric) geometry + name lookups for corridor/structure analysis
    # (contiguity tolerance is in meters, so WGS84 geoms cannot be used here).
    ref_geoms_proj = {str(k): v for k, v in _geom_lookup(reference_proj, ref_id_column).items()}
    tgt_geoms_proj = {str(k): v for k, v in _geom_lookup(target_proj, target_id_column).items()}
    ref_names_proj = {str(k): v for k, v in _name_lookup(reference_proj, ref_id_column).items()}
    tgt_names_proj = {str(k): v for k, v in _name_lookup(target_proj, target_id_column).items()}

    # Build id-keyed geometry lookups as a FALLBACK only. dict(zip(...)) silently
    # keeps the last row when an id repeats, so it cannot distinguish reference
    # rows that share a GERS id (an Overture segment split into multiple edges).
    # The primary path below resolves geometry by the scored positional index
    # (MatchResult.ref_idx / target_idx) so co-id edges keep their real geometry.
    ref_geom_lookup = dict(zip(reference[ref_id_column], reference.geometry))
    tgt_geom_lookup = dict(zip(target[target_id_column], target.geometry))
    ref_geoms_by_pos = reference.geometry.to_numpy()
    tgt_geoms_by_pos = target.geometry.to_numpy()

    def _geom_at_pos(geoms_by_pos, idx):
        """Return the geometry at positional index ``idx``, or None if invalid."""
        if idx is None:
            return None
        if 0 <= idx < len(geoms_by_pos):
            return geoms_by_pos[idx]
        return None

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

    def _physical_at_pos(frame: gpd.GeoDataFrame, idx: int | None) -> dict[str, Any]:
        """Physical LR block for one scored row; positional to preserve duplicate ids."""
        if idx is None or idx < 0 or idx >= len(frame):
            return {}
        row = frame.iloc[idx]
        return physical_attributes(
            row.get("level_lr") if "level_lr" in frame.columns else None,
            row.get("road_flags_lr") if "road_flags_lr" in frame.columns else None,
        )

    # Serialize geometries as GeoJSON with coordinate rounding (defined once).
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

    def _lookup_with_int_fallback(lookup, sid):
        val = lookup.get(sid)
        if val is None and sid.isdigit():
            val = lookup.get(int(sid))
        return val

    # Set (per group) of pruned pairs OWNED by the group currently being
    # serialized — i.e. pairs whose pre-prune ``group_id`` is this group. A pruned
    # pair is marked/counted in EXACTLY its owner group; in any foreign group where
    # a surviving endpoint makes it an incident rejected candidate it is a normal
    # (non-pruned) alternative, so ``n_pruned`` never double-counts across the
    # corridor sub-groups a shared node spans. ``_serialize_edge`` reads this via
    # closure; it is re-bound at the top of each group iteration below.
    group_owned_pruned: set[tuple[str, str]] = set()

    # Per-group optimizer REVIEW provenance, keyed by selected assignment pair.
    # Re-bound at the top of each group iteration; ``_serialize_edge`` reads it
    # via closure like ``group_owned_pruned``.
    group_review_reasons: dict[tuple[str, str], str] = {}
    group_assignment_decisions: dict[tuple[str, str], str] = {}

    def _serialize_edge(pair: tuple[str, str], r: Any, struct: dict | None) -> dict:
        """Serialize one candidate edge (selected or rejected) to a sidecar dict.

        Casts numpy scalars to JSON floats, attaches alignment fractions and the
        per-edge structure block, and marks ``pruned`` when the pair was dropped
        by the confidence-drop prune AND this group owns it (Task B). ``selected``
        comes from ``struct`` (True iff the pair is in the group's assignment).
        Sets ``review_reason`` when the optimizer retained this selected pair as
        REVIEW. The field is additive and absent from non-review edges.
        """
        edge = {
            "ref_id": pair[0],
            "target_id": pair[1],
            "confidence": round(float(r.confidence), 4),
        }
        if r.gers_start_frac is not None:
            edge["gers_start_frac"] = round(float(r.gers_start_frac), ALIGNMENT_FRAC_PRECISION)
            edge["gers_end_frac"] = round(float(r.gers_end_frac), ALIGNMENT_FRAC_PRECISION)
        if r.local_start_frac is not None:
            edge["local_start_frac"] = round(float(r.local_start_frac), ALIGNMENT_FRAC_PRECISION)
            edge["local_end_frac"] = round(float(r.local_end_frac), ALIGNMENT_FRAC_PRECISION)
        if struct:
            edge.update(struct)
        # ``compute_group_structure`` historically called the graph-theory cut
        # edge bit ``is_bridge``. Rename it at the evidence boundary so it can
        # never be mistaken for a physical bridge; readers retain a legacy
        # fallback for old sidecars.
        if "is_bridge" in edge:
            edge["candidate_graph_bridge"] = bool(edge.pop("is_bridge"))

        ref_physical = clip_physical_attributes(
            _physical_at_pos(reference, getattr(r, "ref_idx", None)),
            getattr(r, "gers_start_frac", None),
            getattr(r, "gers_end_frac", None),
        )
        target_physical = clip_physical_attributes(
            _physical_at_pos(target, getattr(r, "target_idx", None)),
            getattr(r, "local_start_frac", None),
            getattr(r, "local_end_frac", None),
        )
        if ref_physical:
            edge["ref_physical"] = ref_physical
        if target_physical:
            edge["target_physical"] = target_physical
        if pair in group_owned_pruned:
            edge["pruned"] = True
        if pair in group_assignment_decisions:
            edge["decision"] = group_assignment_decisions[pair]
        if pair in group_review_reasons:
            edge["review_reason"] = group_review_reasons[pair]
        return edge

    groups = []
    for group_id, assign in assignment_by_gid.items():
        # Pruned pairs owned by THIS group. With group attribution present
        # (production), a pruned pair belongs to its pre-prune group_id only, so it
        # is marked/counted exactly once even when its surviving endpoint anchors a
        # different corridor sub-group. Without attribution (e.g. legacy callers
        # that pass ``pruned_pairs`` but not ``pruned_group_ids``), fall back to
        # global membership so a single-group prune is still recorded.
        if pruned_gid_by_pair:
            group_owned_pruned = pruned_pairs_by_gid.get(str(group_id), set())
        else:
            group_owned_pruned = pruned_pairs_str
        group_review_reasons = {
            (str(r.ref_id), str(r.target_id)): reason
            for r in assign
            if (reason := optimizer_review_reason(r)) is not None
        }
        group_assignment_decisions = {
            (str(r.ref_id), str(r.target_id)): r.decision.value for r in assign
        }
        ref_ids = sorted({str(r.ref_id) for r in assign})
        target_ids = sorted({str(r.target_id) for r in assign})
        tgt_set = set(target_ids)
        assignment_pairs = {(str(r.ref_id), str(r.target_id)) for r in assign}

        # Candidate edges: highest-confidence raw pair for every (ref,target)
        # within this sub-group's id product. Includes the selected assignment
        # plus any in-corridor sliver/weak alternatives; cross-corridor
        # candidates belong to a different sub-group and are excluded.
        cand_by_pair: dict[tuple[str, str], Any] = {}
        for rid in ref_ids:
            for r in cands_by_ref.get(rid, []):
                tid = str(r.target_id)
                if tid in tgt_set:
                    cand_by_pair[(rid, tid)] = r
        for r in assign:  # defensive: ensure every selected edge is present
            cand_by_pair.setdefault((str(r.ref_id), str(r.target_id)), r)

        # Classify match type from the resolved sub-group.
        if len(ref_ids) == 1:
            match_type = MatchType.ONE_TO_N
        elif len(target_ids) == 1:
            match_type = MatchType.N_TO_ONE
        else:
            match_type = MatchType.M_TO_N

        # Resolve each edge's geometry by its scored positional index so that
        # rows sharing an id don't collapse to one geometry (id-keyed lookup is
        # a fallback only). Pick the smallest index deterministically.
        ref_idx_by_id: dict[str, int] = {}
        tgt_idx_by_id: dict[str, int] = {}
        for r in cand_by_pair.values():
            rid = str(r.ref_id)
            ridx = getattr(r, "ref_idx", None)
            if ridx is not None and ridx < ref_idx_by_id.get(rid, ridx + 1):
                ref_idx_by_id[rid] = ridx
            tid = str(r.target_id)
            tidx = getattr(r, "target_idx", None)
            if tidx is not None and tidx < tgt_idx_by_id.get(tid, tidx + 1):
                tgt_idx_by_id[tid] = tidx
        ref_geom_by_id = {
            rid: geom
            for rid, idx in ref_idx_by_id.items()
            if (geom := _geom_at_pos(ref_geoms_by_pos, idx)) is not None
        }
        tgt_geom_by_id = {
            tid: geom
            for tid, idx in tgt_idx_by_id.items()
            if (geom := _geom_at_pos(tgt_geoms_by_pos, idx)) is not None
        }

        # Candidate-graph structure features (degree, bridge, biconnected block,
        # corridor ids, selected, sliver) + per-group counts. Purely structural,
        # derived from data already here; feeds the resolver + oversized flag.
        per_edge, per_group = compute_group_structure(
            edges=list(cand_by_pair.keys()),
            ref_ids=ref_ids,
            target_ids=target_ids,
            assignment_pairs=assignment_pairs,
            sliver_pairs=sliver_pairs_str,
            ref_geoms=ref_geoms_proj,
            target_geoms=tgt_geoms_proj,
            tolerance=DEFAULT_SNAP_TOLERANCE_M,
            corridor_aware=settings.optimizer_corridor_aware,
            max_turn_deg=settings.optimizer_corridor_max_turn_deg,
            ref_name_lookup=ref_names_proj,
            target_name_lookup=tgt_names_proj,
        )

        # Serialize the in-product candidate edges (selected assignment + any
        # in-corridor sliver/weak alternatives) with their per-edge structure.
        # NOTE: this `edges` list and its per-edge fields are UNCHANGED from the
        # pre-M2 sidecar (the rejected candidates below go in a sibling list), so
        # every existing consumer — and the stitch gate — is byte-invariant.
        edges = [
            _serialize_edge(pair, cand_by_pair[pair], per_edge.get(pair))
            for pair in sorted(cand_by_pair.keys())
        ]

        # Rejected candidate edges (M2): every OTHER raw candidate the optimizer
        # saw for this group's nodes — a group ref matched to an out-of-group
        # target, or vice versa — that it did NOT select anywhere. These are the
        # under-selection negatives the resolver needs and the "extra plausible
        # edges" the review UI previously discarded. Persisted in a SEPARATE list
        # so `edges` stays byte-identical. Bounded per group (highest-confidence
        # kept) with truncation recorded.
        rejected_edges: list[dict] = []
        n_rejected_total = 0
        rejected_truncated = False
        if persist_rejected:
            rej_best: dict[tuple[str, str], Any] = {}
            # Candidates incident to a group node (ref->out-of-group target, or
            # target->out-of-group ref) that are neither an in-product candidate
            # nor selected in any other group. Keep the highest-confidence per pair.
            _incident = [
                (rid, str(r.target_id), r) for rid in ref_ids for r in cands_by_ref.get(rid, [])
            ]
            _incident += [
                (str(r.ref_id), tid, r) for tid in target_ids for r in cands_by_tgt.get(tid, [])
            ]
            for _r0, _t0, r in _incident:
                pair = (_r0, _t0)
                if pair in cand_by_pair or pair in all_selected_pairs:
                    continue
                prev = rej_best.get(pair)
                if prev is None or _match_result_rank(r) < _match_result_rank(prev):
                    rej_best[pair] = r

            n_rejected_total = len(rej_best)
            # Highest-confidence first; cap per group to bound sidecar growth.
            rej_ranked = sorted(rej_best.items(), key=lambda kv: (-kv[1].confidence, kv[0]))
            # Pruned edges OWNED by this group are EXEMPT from the truncation cap:
            # dropping one would make ``n_pruned`` undercount (the prune's effect
            # must stay auditable). Only owned pruned pairs are exempt — a globally
            # pruned pair that is merely a foreign group's incident alternative is
            # not marked pruned here, so it is capped like any other candidate. The
            # cap bounds the non-owned-pruned remainder; ``rejected_truncated``
            # reflects whether any of those was dropped.
            if rejected_cap >= 0:
                pruned_rej = [kv for kv in rej_ranked if kv[0] in group_owned_pruned]
                other_rej = [kv for kv in rej_ranked if kv[0] not in group_owned_pruned]
                if len(other_rej) > rejected_cap:
                    other_rej = other_rej[:rejected_cap]
                    rejected_truncated = True
                rej_ranked = sorted(
                    pruned_rej + other_rej, key=lambda kv: (-kv[1].confidence, kv[0])
                )

            if rej_ranked:
                # Structure for rejected edges is computed on the AUGMENTED
                # candidate graph (group nodes + the rejected edges' foreign
                # endpoints) so degree/bridge/corridor are well-defined. Computed
                # separately from `edges` so the in-product edges' structure is
                # never perturbed.
                rej_pairs = [p for p, _ in rej_ranked]
                aug_ref_ids = sorted({p[0] for p in rej_pairs} | set(ref_ids))
                aug_tgt_ids = sorted({p[1] for p in rej_pairs} | set(target_ids))
                aug_edges = list(cand_by_pair.keys()) + rej_pairs
                rej_struct, _ = compute_group_structure(
                    edges=aug_edges,
                    ref_ids=aug_ref_ids,
                    target_ids=aug_tgt_ids,
                    assignment_pairs=assignment_pairs,
                    sliver_pairs=sliver_pairs_str,
                    ref_geoms=ref_geoms_proj,
                    target_geoms=tgt_geoms_proj,
                    tolerance=DEFAULT_SNAP_TOLERANCE_M,
                    corridor_aware=settings.optimizer_corridor_aware,
                    max_turn_deg=settings.optimizer_corridor_max_turn_deg,
                    ref_name_lookup=ref_names_proj,
                    target_name_lookup=tgt_names_proj,
                )
                rejected_edges = [
                    _serialize_edge(pair, r, rej_struct.get(pair)) for pair, r in rej_ranked
                ]

            # Pendant pruned edges (M2 auditability): a pruned edge whose BOTH
            # endpoints left this group post-prune is neither an in-product `edges`
            # pair nor incident to a surviving group node, so it is collected by
            # nothing above and would silently drop from the sidecar — making
            # `n_pruned` undercount. Recover each such edge, attributed to this
            # group by its pre-prune group_id, as an explicit pruned record.
            # Serialized with a minimal struct (its endpoints are not group nodes,
            # so degree/bridge/corridor are undefined). Counted in n_rejected_total
            # so ``n_rejected_edges <= n_rejected_total`` holds.
            if pruned_gid_by_pair:
                serialized_pruned = {
                    (e["ref_id"], e["target_id"]) for e in edges + rejected_edges if e.get("pruned")
                }
                for pair in sorted(group_owned_pruned - serialized_pruned):
                    r = best_by_pair.get(pair)
                    if r is None:
                        continue
                    rejected_edges.append(_serialize_edge(pair, r, {"selected": False}))
                    n_rejected_total += 1

        optimizer_assignment = []
        for r in assign:
            assignment_edge = {
                "ref_id": str(r.ref_id),
                "target_id": str(r.target_id),
                "confidence": round(float(r.confidence), 4),
                "decision": r.decision.value,
            }
            review_reason = optimizer_review_reason(r)
            if review_reason:
                assignment_edge["review_reason"] = review_reason
            optimizer_assignment.append(assignment_edge)

        ref_geometries = {}
        for rid in ref_ids:
            geom = ref_geom_by_id.get(rid)
            if geom is None:
                geom = ref_geom_lookup.get(rid) or ref_geom_lookup.get(
                    int(rid) if rid.isdigit() else rid
                )
            if geom is not None:
                gj = _geom_to_geojson(geom)
                if gj:
                    ref_geometries[rid] = gj

        target_geometries = {}
        for tid in target_ids:
            geom = tgt_geom_by_id.get(tid)
            if geom is None:
                geom = tgt_geom_lookup.get(tid) or tgt_geom_lookup.get(
                    int(tid) if tid.isdigit() else tid
                )
            if geom is not None:
                gj = _geom_to_geojson(geom)
                if gj:
                    target_geometries[tid] = gj

        # Collect names and classes for each segment in the group
        ref_names = {}
        ref_classes = {}
        for rid in ref_ids:
            name = _lookup_with_int_fallback(ref_name_lookup, rid)
            if name is not None:
                ref_names[rid] = _extract_name_string(name)
            cls = _lookup_with_int_fallback(ref_class_lookup, rid)
            if cls is not None:
                ref_classes[rid] = str(cls) if not _is_nan(cls) else ""

        target_names = {}
        target_classes = {}
        for tid in target_ids:
            name = _lookup_with_int_fallback(tgt_name_lookup, tid)
            if name is not None:
                target_names[tid] = _extract_name_string(name)
            cls = _lookup_with_int_fallback(tgt_class_lookup, tid)
            if cls is not None:
                target_classes[tid] = str(cls) if not _is_nan(cls) else ""

        # Segment-wide LR rules accompany the aligned, clipped copies on each
        # edge. This preserves attribution changes along a segment for audit/UI
        # while the edge block answers the actual aligned-range question.
        ref_physical = {
            rid: physical
            for rid, idx in ref_idx_by_id.items()
            if (physical := _physical_at_pos(reference, idx))
        }
        target_physical = {
            tid: physical
            for tid, idx in tgt_idx_by_id.items()
            if (physical := _physical_at_pos(target, idx))
        }

        oversized = not group_is_structurally_simple(
            per_group["n_corridors"],
            per_group["n_assignment_components"],
            per_group["n_edges"],
            settings.stitch_export_max_assignment_components,
            settings.stitch_export_soft_max_edges,
            settings.stitch_export_backstop_max_edges,
        )

        groups.append(
            {
                "group_id": group_id,
                "match_type": match_type.value,
                "ref_ids": ref_ids,
                "target_ids": target_ids,
                "edges": edges,
                "optimizer_assignment": optimizer_assignment,
                "ref_geometries": ref_geometries,
                "target_geometries": target_geometries,
                "ref_names": ref_names,
                "target_names": target_names,
                "ref_classes": ref_classes,
                "target_classes": target_classes,
                "ref_physical": ref_physical,
                "target_physical": target_physical,
                "n_edges": per_group["n_edges"],
                "n_corridors": per_group["n_corridors"],
                "n_assignment_components": per_group["n_assignment_components"],
                "largest_biconnected_block": per_group["largest_biconnected_block"],
                "oversized_group": oversized,
                # Full candidate graph (learned-resolver flip condition #1):
                # EVERY floor-passing candidate pair in this group's component,
                # attributed to exactly one group, with the optimizer's decision
                # (`selected`), plus authoritative pre-prune ownership (`pruned`)
                # for confidence-pruned pairs. Uncapped, minimal uniform schema;
                # empty when settings.stitch_persist_candidate_graph is off.
                "candidate_edges": (
                    candidate_graph_by_gid.get(group_id, [])
                    if settings.stitch_persist_candidate_graph
                    else []
                ),
                "n_candidate_edges": (
                    len(candidate_graph_by_gid.get(group_id, []))
                    if settings.stitch_persist_candidate_graph
                    else 0
                ),
                # M2 candidate-graph persistence: non-selected candidates the
                # optimizer saw for this group's nodes (sibling to `edges`).
                "rejected_edges": rejected_edges,
                "n_rejected_edges": len(rejected_edges),
                "n_rejected_total": n_rejected_total,
                "rejected_truncated": rejected_truncated,
                # Confidence-drop prune effect (Task B); 0 when the flag is OFF.
                # When group attribution is present (production), count is
                # AUTHORITATIVE from the owned pruned set — exact regardless of
                # ``stitch_persist_rejected_edges`` and independent of whether each
                # pruned edge was serialized. Equals the serialized-pruned count
                # when persistence is on (owned edges are all recorded, cap-exempt).
                # Legacy callers without attribution fall back to the serialized
                # count (a single-group prune is still recorded).
                "n_pruned": (
                    len(group_owned_pruned)
                    if pruned_gid_by_pair
                    else sum(1 for e in edges + rejected_edges if e.get("pruned"))
                ),
            }
        )

    if not groups:
        # Remove stale sidecar from a previous run so batch generation
        # doesn't pick up outdated group data.
        stale = groups_sidecar_path(output_path)
        if stale.exists():
            stale.unlink()
            logger.info(f"Removed stale groups sidecar: {stale}")
        stale_candidates = candidates_sidecar_path(output_path)
        if stale_candidates.exists():
            stale_candidates.unlink()
            logger.info(f"Removed stale candidates sidecar: {stale_candidates}")
        return None

    sidecar_path = groups_sidecar_path(output_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    sidecar = {
        "n_groups": len(groups),
        "groups": groups,
    }

    _export_candidates_sidecar(
        groups=groups,
        candidate_graph_by_gid=candidate_graph_by_gid,
        best_by_pair=best_by_pair,
        optimized_by_pair=optimized_by_pair,
        output_path=output_path,
        reference_proj=reference_proj,
        target_proj=target_proj,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
        sliver_pairs_str=sliver_pairs_str,
        pruned_gid_by_pair=pruned_gid_by_pair,
        dataset_id=dataset_id,
        model_path=model_path,
    )
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    logger.info(f"Exported {len(groups)} match groups to {sidecar_path}")
    return sidecar_path


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return ``gdf`` reprojected to EPSG:4326, or unchanged if already there.

    Row order is preserved, so positional indices (``MatchResult.ref_idx`` /
    ``target_idx``) remain valid against the returned frame.
    """
    if gdf.crs and not gdf.crs.equals("EPSG:4326"):
        return gdf.to_crs("EPSG:4326")
    return gdf


def load_and_filter_inputs(
    reference_path: Path,
    target_path: Path,
    dataset_id: str | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load reference/target parquets and apply the pipeline's Step-1 filtering.

    Factored out of :func:`run_pipeline` so the bridge-table factory can reproduce
    the exact same reference/target row ordering when re-optimizing from cached
    scores — positional ``ref_idx`` / ``target_idx`` in cached ``MatchResult``
    objects index into these frames, so the filtering (null-geometry drop +
    LineString-only) must match byte-for-byte.

    When ``dataset_id`` is supplied, newly configured physical target fields are
    normalized from retained ``source_tags`` before filtering. This is row-order
    preserving and lets existing snapshots feed bridge/tunnel/layer LR evidence.

    Raises:
        PipelineError: If a file is missing or empty after filtering.
    """
    if not reference_path.exists():
        raise PipelineError(f"Reference file not found: {reference_path}")
    if not target_path.exists():
        raise PipelineError(f"Target file not found: {target_path}")

    logger.info("Step 1: Loading data...")
    try:
        reference = gpd.read_parquet(reference_path)
    except Exception as e:
        raise PipelineError(f"Failed to read reference file {reference_path}: {e}") from e

    # Existing Overture snapshots retain current top-level road_flags and
    # level_rules even when an older transform wrote stale derived LR columns.
    reference = backfill_overture_physical_lr(reference)

    try:
        target = gpd.read_parquet(target_path)
    except Exception as e:
        raise PipelineError(f"Failed to read target file {target_path}: {e}") from e

    if dataset_id:
        from ..datasets.schema import get_dataset_config

        dataset_config = get_dataset_config(dataset_id)
        if dataset_config is not None and dataset_config.fetch is not None:
            target = backfill_physical_lr_from_source_tags(target, dataset_config.fetch)

    if reference.geometry.isna().any():
        n_null = reference.geometry.isna().sum()
        logger.warning(f"Reference has {n_null} null geometries - these will be skipped")
        reference = reference[~reference.geometry.isna()]

    if target.geometry.isna().any():
        n_null = target.geometry.isna().sum()
        logger.warning(f"Target has {n_null} null geometries - these will be skipped")
        target = target[~target.geometry.isna()]

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
    return reference, target


def optimize_and_export(
    results: list,
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    reference_wgs84: gpd.GeoDataFrame,
    target_wgs84: gpd.GeoDataFrame,
    output_path: Path,
    method: str = "xgboost",
    min_confidence: float = 0.1,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    progress_callback: Callable[[int], None] | None = None,
    prune_dataset_key: str | None = None,
    model_path: str | Path | None = None,
) -> PipelineResult:
    """Optimize scored candidates, prune, export sidecar, and write outputs.

    This is the post-scoring half of :func:`run_pipeline` (score propagation →
    M:N optimizer → confidence-drop prune → groups sidecar → bridge/unmatched
    files). Factored out so the factory can drive it from CACHED ``MatchResult``
    objects (``crosswalk factory reoptimize``, ~2 s vs a full re-score).
    ``run_pipeline`` calls this on the freshly-scored results, so the normal path
    is behavior-identical.

    Args:
        results: Raw ``MatchResult`` objects from scoring.
        reference: Reference GeoDataFrame in the projected (metric) CRS.
        target: Target GeoDataFrame in the projected (metric) CRS.
        reference_wgs84: Reference GeoDataFrame in EPSG:4326 (sidecar geometry).
        target_wgs84: Target GeoDataFrame in EPSG:4326 (sidecar geometry).
        output_path: Path for the output bridge file.
        method: Matching method (recorded in the bridge file).
        min_confidence: Minimum confidence used during optimization.
        ref_id_column: Reference ID column name.
        target_id_column: Target ID column name.
        progress_callback: Optional progress callback.
        prune_dataset_key: Dataset identity for resolver-prune allowlist
            resolution (see :func:`_effective_prune_threshold`) — the dataset
            name the caller already knows (``crosswalk stitch`` dataset argument,
            factory pair name). NEVER derived from ``output_path`` (#348). None
            means the run has no dataset identity, so the prune is off (logged).
        model_path: Exact model that produced ``results``. Defaults to the active
            production setting (bundled unless explicitly overridden) and is
            reused for calibration-dependent thresholds and sidecar provenance.

    Returns:
        PipelineResult with statistics.
    """
    active_model_path = _resolve_model_path(model_path)
    logger.info(f"  Generated and scored {len(results)} candidates")

    if not results:
        logger.warning("No candidates found! Check data alignment and buffer distance.")

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

    # Step 3.5 (EXPERIMENTAL, opt-in): structure-aware score propagation.
    # Runs after per-pair scoring and before the optimizer. Adjusts confidences
    # using network topology so confident, structurally-consistent corridors
    # reinforce each other and competing non-adjacent alternatives are dampened.
    # Gated behind settings.enable_score_propagation (default off => no-op, and
    # the results list is left byte-identical to the pre-propagation pipeline).
    if settings.enable_score_propagation:
        from ..matching.score_propagation import propagate_scores

        logger.info("Step 3.5: Applying structure-aware score propagation...")
        results, _prop_stats = propagate_scores(
            results,
            reference=reference,
            target=target,
            ref_id_column=ref_id_column,
            target_id_column=target_id_column,
        )

    # Step 4: Optimize matches with M:N grouping (resolve conflicts)
    # Grouping allows multiple contiguous segments and supports 1:1, 1:N, N:1, and M:N match types
    # This handles different segmentation schemes and overlapping relationships between datasets
    glue_min_confidence = _effective_glue_min_confidence(active_model_path)
    logger.info(
        f"Step 4: Optimizing matches with M:N grouping "
        f"(min_confidence={min_confidence}, glue_min_confidence={glue_min_confidence})..."
    )
    optimized = optimize_matches_with_grouping(
        results,
        reference=reference,
        target=target,
        min_confidence=min_confidence,
        glue_min_confidence=glue_min_confidence,
        contiguity_tolerance=DEFAULT_SNAP_TOLERANCE_M,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
    )

    # Step 4.5 (flag-gated, default OFF): confidence-drop prune of group edges
    # (M2 / resolver Phase 1). Drops selected group edges below an absolute
    # confidence floor — the one-parameter filter the #272 eval validated. When
    # disabled the optimizer selections are byte-identical to the pre-prune
    # pipeline; the pruned pairs are recorded in the sidecar so the prune's
    # effect is auditable and gate-measurable.
    pruned_pairs: set[tuple[Any, Any]] = set()
    # Group attribution for pruned pairs, snapshotted from the PRE-prune
    # assignment (the prune drops these results, losing their group_id). Lets the
    # sidecar attribute every pruned edge to its group so ``n_pruned`` is exact
    # even for pendant edges whose both endpoints leave the surviving group.
    pruned_group_ids: dict[tuple[Any, Any], Any] = {}
    prune_threshold = _effective_prune_threshold(prune_dataset_key, active_model_path)
    if prune_threshold > 0:
        from ..matching.optimizer import apply_confidence_drop_prune

        n_before = len(optimized)
        gid_by_pair = {(r.ref_id, r.target_id): r.features.get("group_id") for r in optimized}
        optimized, pruned_pairs = apply_confidence_drop_prune(optimized, prune_threshold)
        pruned_group_ids = {pair: gid_by_pair.get(pair) for pair in pruned_pairs}
        logger.info(
            f"Step 4.5: Resolver confidence-drop prune "
            f"(min_confidence={prune_threshold}): dropped "
            f"{len(pruned_pairs)} group edges ({n_before} -> {len(optimized)})"
        )

    # Export groups sidecar for stitching review (using WGS84 geometries).
    # The sliver set is computed from the PROJECTED data (identical to what the
    # optimizer classified internally) so sidecar grouping matches optimizer
    # grouping exactly, independent of any WGS84 length re-measurement.
    sliver_edges = compute_sliver_candidate_edges(
        results, reference, target, ref_id_column, target_id_column
    )
    _export_groups_sidecar(
        results=results,
        optimized=optimized,
        output_path=output_path,
        reference=reference_wgs84,
        target=target_wgs84,
        min_confidence=min_confidence,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
        sliver_edges=sliver_edges,
        reference_proj=reference,
        target_proj=target,
        pruned_pairs=pruned_pairs,
        pruned_group_ids=pruned_group_ids,
        glue_min_confidence=glue_min_confidence,
        dataset_id=prune_dataset_key,
        model_path=active_model_path,
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

    # Unmatched report
    # Only MATCH decisions count as matched. REVIEW decisions are low-confidence
    # and should appear in unmatched.parquet so they can be labeled/reviewed.
    matched_target_ids, review_target_ids = _partition_target_decision_ids(optimized)
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
    )


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
    allow_version_mismatch: bool = False,
    prune_dataset_key: str | None = None,
    model_path: str | Path | None = None,
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
        prune_dataset_key: Dataset identity for the resolver-prune allowlist
            (see :func:`_effective_prune_threshold`). NEVER derived from
            ``output_path`` (#348). None = no dataset identity, prune off.
        model_path: Explicit production model artifact. None uses
            ``settings.model_path`` (bundled by default; ``MATCHER_MODEL_PATH``
            is the environment override).

    Returns:
        PipelineResult with statistics
    """
    active_model_path = _resolve_model_path(model_path)
    logger.info("=" * 60)
    logger.info("Starting matching pipeline")
    logger.info("=" * 60)

    # Step 1: Load + filter inputs (shared with the factory reoptimize path).
    load_kwargs = {"dataset_id": prune_dataset_key} if prune_dataset_key else {}
    reference, target = load_and_filter_inputs(reference_path, target_path, **load_kwargs)

    if progress_callback:
        progress_callback(10)

    # Steps 2-3: Generate candidates and score using shared function
    logger.info("Steps 2-3: Generating candidates and scoring...")

    if method == "rule":
        raise ValueError(
            "Rule-based matching has been removed. Use method='xgboost' instead. "
            "Train a model first with 'crosswalk train'."
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
        model_path=active_model_path,
        allow_version_mismatch=allow_version_mismatch,
    )
    # Ensure WGS84 GeoDataFrames for sidecar export (web map needs EPSG:4326).
    # Computed from the pre-projection (filtered) frames; row order is preserved.
    reference_wgs84 = _to_wgs84(reference)
    target_wgs84 = _to_wgs84(target)
    # Use projected versions for downstream (optimizer/structure) computations.
    reference = projection_result.reference
    target = projection_result.target

    if run_screen:
        logger.warning("Screen tests on bridge files not yet implemented, skipping...")

    return optimize_and_export(
        results=results,
        reference=reference,
        target=target,
        reference_wgs84=reference_wgs84,
        target_wgs84=target_wgs84,
        output_path=output_path,
        method=method,
        min_confidence=min_confidence,
        ref_id_column=ref_id_column,
        target_id_column=target_id_column,
        progress_callback=progress_callback,
        prune_dataset_key=prune_dataset_key,
        model_path=active_model_path,
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
