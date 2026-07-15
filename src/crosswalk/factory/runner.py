"""Per-dataset factory jobs + parallel batch orchestration.

``run_dataset`` runs the full stitch pipeline for one dataset via the shared
pipeline seams (``load_and_filter_inputs`` → ``score_candidates_from_geodataframes``
→ ``optimize_and_export``), caches the scored candidates, and writes the versioned
output layout + manifest. ``reoptimize_dataset`` reuses the cache to re-run only
the optimize/export half. ``run_batch`` fans datasets out across worker processes
with per-dataset failure isolation and structured per-dataset logs.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ..config import settings
from ..filenames import candidates_sidecar_path, groups_sidecar_path
from . import manifest as manifest_mod
from .discovery import DatasetPair, resolve_release
from .manifest import Manifest
from .scored_cache import (
    SCORED_CACHE_SCHEMA_VERSION,
    read_scored_cache,
    write_scored_cache,
)

SCORED_CACHE_FILENAME = "scored_candidates.parquet"
BRIDGE_FILENAME = "bridge.parquet"
GROUPS_FILENAME = "groups.json"
CANDIDATES_FILENAME = "candidates.parquet"
UNMATCHED_FILENAME = "unmatched.parquet"
DATASET_LOG_FILENAME = "run.log"


@dataclass(frozen=True)
class FactoryPaths:
    """Resolves the versioned factory output layout under a root directory."""

    root: Path

    def release_dir(self, release: str) -> Path:
        return self.root / f"release={release}"

    def dataset_dir(self, release: str, name: str) -> Path:
        return self.release_dir(release) / f"dataset={name}"

    def bridge(self, release: str, name: str) -> Path:
        return self.dataset_dir(release, name) / BRIDGE_FILENAME

    def groups(self, release: str, name: str) -> Path:
        return self.dataset_dir(release, name) / GROUPS_FILENAME

    def candidates(self, release: str, name: str) -> Path:
        return self.dataset_dir(release, name) / CANDIDATES_FILENAME

    def manifest(self, release: str, name: str) -> Path:
        return self.dataset_dir(release, name) / manifest_mod.MANIFEST_FILENAME

    def scored_cache(self, release: str, name: str) -> Path:
        return self.dataset_dir(release, name) / SCORED_CACHE_FILENAME


def build_keys(
    pair: DatasetPair,
    buffer_distance_m: float,
    method: str = "xgboost",
    min_confidence: float = 0.1,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute staleness keys + provenance blocks for a pair.

    Returns a dict with ``inputs``, ``model``, ``snapshot``, ``score_key``,
    ``optimize_key``, ``full_key``. ``method`` participates in ``score_key``;
    ``min_confidence`` (an optimizer argument, not a settings field) participates
    in ``optimize_key`` — so a future CLI flag for either cannot silently skip.
    """
    ref_fp = manifest_mod.file_fingerprint(pair.reference_path)
    target_fp = manifest_mod.file_fingerprint(pair.target_path)
    connectors_fp = (
        manifest_mod.file_fingerprint(pair.connectors_path) if pair.has_connectors else None
    )
    model_fp = manifest_mod.model_fingerprint(model_path)
    snapshot = manifest_mod.settings_snapshot()

    scoring_settings = {
        key: snapshot[key]
        for key in (
            "enable_calibration",
            "scoring_match_threshold",
            "scoring_review_threshold",
        )
    }
    score_key = manifest_mod.compute_score_key(
        ref_fp,
        target_fp,
        model_fp,
        buffer_distance_m,
        method=method,
        scoring_settings=scoring_settings,
    )
    optimize_key = manifest_mod.compute_optimize_key(snapshot, min_confidence=min_confidence)
    full_key = manifest_mod.compute_full_key(score_key, optimize_key)
    return {
        "inputs": {"reference": ref_fp, "target": target_fp, "connectors": connectors_fp},
        "model": model_fp,
        "snapshot": snapshot,
        "score_key": score_key,
        "optimize_key": optimize_key,
        "full_key": full_key,
    }


def is_up_to_date(manifest_path: Path, bridge_path: Path, full_key: str) -> bool:
    """Whether an existing output is current (safe to skip in ``factory run``).

    True iff the manifest exists, its ``full_key`` matches, and the bridge file
    is present. A killed run leaves no manifest (written last), so it re-runs.
    """
    if not manifest_path.exists() or not bridge_path.exists():
        return False
    try:
        m = Manifest.read(manifest_path)
    except Exception:
        return False
    if m.full_key != full_key:
        return False
    return not (
        settings.stitch_persist_candidates
        and m.groups.get("n_groups", 0) > 0
        and not (bridge_path.parent / CANDIDATES_FILENAME).exists()
    )


def _normalize_outputs(dataset_dir: Path, bridge_path: Path) -> None:
    """Rename the pipeline's groups/candidates sidecars to factory names.

    ``optimize_and_export`` writes the sidecar at ``groups_sidecar_path(bridge)``
    and typed candidates at ``candidates_sidecar_path(bridge)``. Normalize those
    to ``groups.json`` and ``candidates.parquet``; unmatched already matches.
    """
    sidecar = groups_sidecar_path(bridge_path)
    groups_target = dataset_dir / GROUPS_FILENAME
    if sidecar.exists():
        sidecar.replace(groups_target)
    elif groups_target.exists():
        # Previous run produced groups; this run produced none — clear stale.
        groups_target.unlink()

    candidates = candidates_sidecar_path(bridge_path)
    candidates_target = dataset_dir / CANDIDATES_FILENAME
    if candidates.exists():
        candidates.replace(candidates_target)
    elif candidates_target.exists():
        candidates_target.unlink()


def run_dataset(
    pair: DatasetPair,
    release: str,
    paths: FactoryPaths,
    buffer_distance_m: float = 75.0,
    method: str = "xgboost",
    min_confidence: float = 0.1,
    n_jobs: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline for one dataset and write its versioned output.

    Designed to run in a worker process: catches its own errors and returns a
    summary dict (never raises), so one dataset's failure cannot abort a batch.
    Writes the manifest LAST so a killed run re-runs cleanly.
    """
    from ..pipeline import (
        load_and_filter_inputs,
        optimize_and_export,
        score_candidates_from_geodataframes,
    )
    from ..pipeline.runner import _resolve_model_path, _to_wgs84

    dataset_dir = paths.dataset_dir(release, pair.name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    bridge_path = paths.bridge(release, pair.name)
    manifest_path = paths.manifest(release, pair.name)
    cache_path = paths.scored_cache(release, pair.name)

    try:
        active_model_path = _resolve_model_path()
    except (FileNotFoundError, OSError) as exc:
        return {
            "dataset": pair.name,
            "release": release,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    keys = build_keys(
        pair,
        buffer_distance_m,
        method=method,
        min_confidence=min_confidence,
        model_path=active_model_path,
    )

    if not force and is_up_to_date(manifest_path, bridge_path, keys["full_key"]):
        logger.info(f"[{pair.name}] up-to-date (full_key match) — skipping")
        return {"dataset": pair.name, "release": release, "status": "skipped"}

    # Per-dataset structured log sink (added for this process's run).
    log_sink = logger.add(
        dataset_dir / DATASET_LOG_FILENAME,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
        enqueue=False,
    )
    t0 = time.perf_counter()
    try:
        logger.info(f"[{pair.name}] start (release={release}, buffer={buffer_distance_m}m)")
        reference, target = load_and_filter_inputs(
            pair.reference_path,
            pair.target_path,
            dataset_id=pair.name,
        )

        t_score0 = time.perf_counter()
        results, projection_result = score_candidates_from_geodataframes(
            reference=reference,
            target=target,
            buffer_distance_m=buffer_distance_m,
            n_jobs=n_jobs,
            model_path=active_model_path,
        )
        reference_wgs84 = _to_wgs84(reference)
        target_wgs84 = _to_wgs84(target)
        reference = projection_result.reference
        target = projection_result.target
        score_wall = time.perf_counter() - t_score0

        # Cache scored candidates BEFORE optimize so reoptimize survives even a
        # later optimize failure (keyed by the manifest's score_key).
        n_cached = write_scored_cache(results, cache_path)

        t_opt0 = time.perf_counter()
        pipeline_result = optimize_and_export(
            results=results,
            reference=reference,
            target=target,
            reference_wgs84=reference_wgs84,
            target_wgs84=target_wgs84,
            output_path=bridge_path,
            method=method,
            min_confidence=min_confidence,
            prune_dataset_key=pair.name,
            model_path=active_model_path,
        )
        optimize_wall = time.perf_counter() - t_opt0

        _normalize_outputs(dataset_dir, bridge_path)
        wall = time.perf_counter() - t0

        m = _build_manifest(
            pair,
            release,
            keys,
            buffer_distance_m,
            method,
            pipeline_result,
            paths.groups(release, pair.name),
            cache_path,
            n_cached,
            score_wall,
            optimize_wall,
            wall,
        )
        m.write(manifest_path)

        logger.info(
            f"[{pair.name}] done in {wall:.1f}s "
            f"(matched={pipeline_result.n_matched}/{pipeline_result.n_target})"
        )
        return {
            "dataset": pair.name,
            "release": release,
            "status": "done",
            "wall_s": round(wall, 1),
            "score_wall_s": round(score_wall, 1),
            "n_target": pipeline_result.n_target,
            "n_matched": pipeline_result.n_matched,
            "n_review": pipeline_result.n_review,
            "n_unmatched": pipeline_result.n_unmatched,
            "n_groups": m.groups.get("n_groups", 0),
            "n_oversized": m.groups.get("n_oversized", 0),
        }
    except Exception as exc:
        logger.exception(f"[{pair.name}] FAILED: {exc}")
        return {
            "dataset": pair.name,
            "release": release,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        logger.remove(log_sink)


def reoptimize_dataset(
    pair: DatasetPair,
    release: str,
    paths: FactoryPaths,
    buffer_distance_m: float = 75.0,
    method: str = "xgboost",
    min_confidence: float = 0.1,
) -> dict[str, Any]:
    """Re-run only optimize/export from the cached scored candidates (~2 s).

    Requires a manifest whose ``score_key`` still matches the current inputs +
    model + FEATURE_VERSION (else the cache is stale and a full ``run`` is needed).
    Optimizer/prune/export settings may have changed — that is the point.
    """
    from ..pipeline import load_and_filter_inputs, optimize_and_export
    from ..pipeline.runner import _resolve_model_path, _to_wgs84
    from ..utils import ensure_projected_crs

    dataset_dir = paths.dataset_dir(release, pair.name)
    bridge_path = paths.bridge(release, pair.name)
    manifest_path = paths.manifest(release, pair.name)
    cache_path = paths.scored_cache(release, pair.name)

    if not cache_path.exists():
        return {
            "dataset": pair.name,
            "release": release,
            "status": "failed",
            "error": "no scored cache; run 'crosswalk factory run' first",
        }
    if not manifest_path.exists():
        return {
            "dataset": pair.name,
            "release": release,
            "status": "failed",
            "error": "no manifest; run 'crosswalk factory run' first",
        }

    try:
        active_model_path = _resolve_model_path()
    except (FileNotFoundError, OSError) as exc:
        return {
            "dataset": pair.name,
            "release": release,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    keys = build_keys(
        pair,
        buffer_distance_m,
        method=method,
        min_confidence=min_confidence,
        model_path=active_model_path,
    )
    prev = Manifest.read(manifest_path)
    if prev.score_key != keys["score_key"]:
        return {
            "dataset": pair.name,
            "release": release,
            "status": "failed",
            "error": "scored cache is stale (inputs/model/FEATURE_VERSION changed); "
            "run 'crosswalk factory run --force'",
        }
    # Belt-and-braces: score_key already folds in SCORED_CACHE_SCHEMA_VERSION, but
    # a manifest hand-edited or produced by a future writer could pass the key
    # check with a different recorded layout version — refuse to misread it.
    prev_cache_schema = (prev.scored_cache or {}).get("schema_version")
    if prev_cache_schema is not None and prev_cache_schema != SCORED_CACHE_SCHEMA_VERSION:
        return {
            "dataset": pair.name,
            "release": release,
            "status": "failed",
            "error": f"scored cache schema v{prev_cache_schema} != reader "
            f"v{SCORED_CACHE_SCHEMA_VERSION}; run 'crosswalk factory run --force'",
        }

    log_sink = logger.add(
        dataset_dir / DATASET_LOG_FILENAME,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
        enqueue=False,
    )
    t0 = time.perf_counter()
    try:
        logger.info(f"[{pair.name}] reoptimize from cache")
        reference, target = load_and_filter_inputs(
            pair.reference_path,
            pair.target_path,
            dataset_id=pair.name,
        )
        projection_result = ensure_projected_crs(reference, target)
        reference_wgs84 = _to_wgs84(reference)
        target_wgs84 = _to_wgs84(target)
        reference = projection_result.reference
        target = projection_result.target

        results = read_scored_cache(cache_path)

        t_opt0 = time.perf_counter()
        pipeline_result = optimize_and_export(
            results=results,
            reference=reference,
            target=target,
            reference_wgs84=reference_wgs84,
            target_wgs84=target_wgs84,
            output_path=bridge_path,
            method=method,
            min_confidence=min_confidence,
            prune_dataset_key=pair.name,
            model_path=active_model_path,
        )
        optimize_wall = time.perf_counter() - t_opt0
        _normalize_outputs(dataset_dir, bridge_path)
        wall = time.perf_counter() - t0

        m = _build_manifest(
            pair,
            release,
            keys,
            buffer_distance_m,
            method,
            pipeline_result,
            paths.groups(release, pair.name),
            cache_path,
            len(results),
            prev.score_wall_s,  # carried over from the run that produced the cache
            optimize_wall,
            wall,
        )
        m.write(manifest_path)
        logger.info(f"[{pair.name}] reoptimized in {wall:.1f}s")
        return {
            "dataset": pair.name,
            "release": release,
            "status": "done",
            "wall_s": round(wall, 1),
            "n_target": pipeline_result.n_target,
            "n_matched": pipeline_result.n_matched,
            "n_review": pipeline_result.n_review,
            "n_unmatched": pipeline_result.n_unmatched,
            "n_groups": m.groups.get("n_groups", 0),
            "n_oversized": m.groups.get("n_oversized", 0),
        }
    except Exception as exc:
        logger.exception(f"[{pair.name}] reoptimize FAILED: {exc}")
        return {
            "dataset": pair.name,
            "release": release,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        logger.remove(log_sink)


def _build_manifest(
    pair: DatasetPair,
    release: str,
    keys: dict[str, Any],
    buffer_distance_m: float,
    method: str,
    pipeline_result,
    groups_path: Path,
    cache_path: Path,
    n_cached: int,
    score_wall: float | None,
    optimize_wall: float | None,
    wall: float,
) -> Manifest:
    return Manifest(
        dataset=pair.name,
        release=release,
        created_at=Manifest.now_iso(),
        buffer_distance_m=buffer_distance_m,
        method=method,
        inputs=keys["inputs"],
        model=keys["model"],
        settings_snapshot=keys["snapshot"],
        score_key=keys["score_key"],
        optimize_key=keys["optimize_key"],
        full_key=keys["full_key"],
        score_wall_s=round(score_wall, 1) if score_wall is not None else None,
        optimize_wall_s=round(optimize_wall, 1) if optimize_wall is not None else None,
        wall_s=round(wall, 1),
        n_reference=pipeline_result.n_reference,
        n_target=pipeline_result.n_target,
        n_candidates=pipeline_result.n_candidates,
        n_matched=pipeline_result.n_matched,
        n_review=pipeline_result.n_review,
        n_unmatched=pipeline_result.n_unmatched,
        groups=manifest_mod.compute_group_stats(groups_path),
        scored_cache={
            "path": cache_path.name,
            "n_results": n_cached,
            "schema_version": SCORED_CACHE_SCHEMA_VERSION,
        },
    )


def run_batch(
    pairs: list[DatasetPair],
    paths: FactoryPaths,
    release_override: str | None = None,
    workers: int = 1,
    buffer_distance_m: float = 75.0,
    method: str = "xgboost",
    min_confidence: float = 0.1,
    n_jobs: int = 1,
    force: bool = False,
    reoptimize: bool = False,
) -> list[dict[str, Any]]:
    """Run ``pairs`` across ``workers`` processes with failure isolation.

    Returns one summary dict per dataset (status done/skipped/failed). Datasets
    whose release cannot be resolved are reported as failed and skipped, never
    aborting the batch.
    """
    # Resolve releases up front so a missing release is a per-dataset failure.
    jobs: list[tuple[DatasetPair, str]] = []
    summaries: list[dict[str, Any]] = []
    for pair in pairs:
        try:
            release = resolve_release(pair, release_override)
        except ValueError as exc:
            summaries.append(
                {"dataset": pair.name, "release": None, "status": "failed", "error": str(exc)}
            )
            continue
        jobs.append((pair, release))

    fn = _reoptimize_job if reoptimize else _run_job
    if workers <= 1:
        for pair, release in jobs:
            summaries.append(
                fn(pair, release, paths, buffer_distance_m, method, min_confidence, n_jobs, force)
            )
        return summaries

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                fn, pair, release, paths, buffer_distance_m, method, min_confidence, n_jobs, force
            ): pair
            for pair, release in jobs
        }
        for fut in as_completed(futs):
            pair = futs[fut]
            try:
                summaries.append(fut.result())
            except Exception as exc:  # worker crash (e.g. OOM) — isolate it
                summaries.append(
                    {
                        "dataset": pair.name,
                        "release": None,
                        "status": "failed",
                        "error": f"worker crashed: {type(exc).__name__}: {exc}",
                    }
                )
    return summaries


def _run_job(pair, release, paths, buffer, method, min_confidence, n_jobs, force):
    """Top-level (picklable) wrapper for a full-run worker task."""
    return run_dataset(
        pair,
        release,
        paths,
        buffer_distance_m=buffer,
        method=method,
        min_confidence=min_confidence,
        n_jobs=n_jobs,
        force=force,
    )


def _reoptimize_job(pair, release, paths, buffer, method, min_confidence, n_jobs, force):
    """Top-level (picklable) wrapper for a reoptimize worker task."""
    return reoptimize_dataset(
        pair,
        release,
        paths,
        buffer_distance_m=buffer,
        method=method,
        min_confidence=min_confidence,
    )
