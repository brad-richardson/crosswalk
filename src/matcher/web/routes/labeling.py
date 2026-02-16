"""Labeling mode routes for the matcher web UI."""

import contextlib
import json
import logging
from threading import Thread

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from shapely.geometry import mapping

from ..jinja import templates
from ..services import (
    CONFIG_FILE,
    get_unlabeled_candidates,
    is_dataset_cached,
    list_datasets,
    load_candidates,
    loading_errors,
    loading_lock,
    loading_tasks,
    record_label,
    undo_last_label,
)
from ..utils import round_geom

logger = logging.getLogger(__name__)

VALID_LABELS = {"match", "no_match", "unsure"}

router = APIRouter()

# Module-level cache for loaded candidates per dataset
_candidate_cache: dict[str, list] = {}


def _start_background_load(dataset_id: str) -> None:
    """Start loading candidates in a background thread.

    If a load is already in progress for this dataset, does nothing.
    On success, stores result in _candidate_cache.
    On error, stores error message in loading_errors.
    """
    with loading_lock:
        if dataset_id in loading_tasks:
            return  # Already running

        def _do_load():
            try:
                result = load_candidates(dataset_id)
                with loading_lock:
                    _candidate_cache[dataset_id] = result
            except Exception:
                logger.exception("Background load failed for dataset %s", dataset_id)
                with loading_lock:
                    loading_errors[dataset_id] = "Feature computation failed. Check server logs."
            finally:
                with loading_lock:
                    loading_tasks.pop(dataset_id, None)

        thread = Thread(target=_do_load, daemon=True, name=f"load-{dataset_id}")
        loading_errors.pop(dataset_id, None)  # Clear any previous error
        loading_tasks[dataset_id] = thread
        thread.start()


def _get_candidates(dataset_id: str) -> list:
    """Get candidates for a dataset, using module-level cache.

    Args:
        dataset_id: Dataset identifier

    Returns:
        List of CandidatePairView objects.
    """
    if dataset_id not in _candidate_cache:
        _candidate_cache[dataset_id] = load_candidates(dataset_id)
    return _candidate_cache[dataset_id]


def _pair_to_geojson(pair) -> str:
    """Convert a CandidatePairView's geometries to a GeoJSON JSON string.

    Returns a JSON string with keys:
    - reference: aligned reference geometry (or full if no aligned)
    - target: aligned target geometry (or full if no aligned)
    - reference_full: full reference geometry
    - target_full: full target geometry

    Args:
        pair: CandidatePairView instance

    Returns:
        JSON string suitable for embedding in an HTML data attribute.
    """
    result = {
        "reference_full": round_geom(mapping(pair.ref_geometry)),
        "target_full": round_geom(mapping(pair.target_geometry)),
    }

    # Use aligned geometries if available, otherwise fall back to full
    if pair.ref_aligned_geometry is not None:
        result["reference"] = round_geom(mapping(pair.ref_aligned_geometry))
    else:
        result["reference"] = result["reference_full"]

    if pair.target_aligned_geometry is not None:
        result["target"] = round_geom(mapping(pair.target_aligned_geometry))
    else:
        result["target"] = result["target_full"]

    return json.dumps(result)


def _render_pair(request, dataset, datasets, index=0):
    """Render a normal pair view for a dataset that is loaded in _candidate_cache."""
    is_htmx = request.headers.get("HX-Request") == "true"

    all_candidates = _candidate_cache[dataset]
    unlabeled = get_unlabeled_candidates(dataset, all_candidates)

    if unlabeled:
        index = max(0, min(index, len(unlabeled) - 1))

    pair = None
    geojson = "{}"
    if unlabeled and 0 <= index < len(unlabeled):
        pair = unlabeled[index]
        geojson = _pair_to_geojson(pair)

    total_candidates = len(all_candidates)
    labeled_count = total_candidates - len(unlabeled)

    context = {
        "mode": "labeling",
        "datasets": datasets,
        "dataset": dataset,
        "pair": pair,
        "geojson": geojson,
        "pair_index": index,
        "total_pairs": len(unlabeled),
        "labeled_count": labeled_count,
        "total_candidates": total_candidates,
    }

    if is_htmx:
        return templates.TemplateResponse(request, "labeling/pair.html", context)
    return templates.TemplateResponse(request, "labeling/page.html", context)


@router.get("/")
async def index():
    """Redirect root to the dashboard."""
    return RedirectResponse(url="/dashboard", status_code=307)


@router.get("/datasets")
async def datasets_endpoint():
    """Return available datasets as JSON."""
    return JSONResponse(content=list_datasets())


@router.get("/labeling")
async def labeling(
    request: Request,
    dataset: str | None = None,
    index: int = 0,
):
    """Render the labeling page or pair fragment.

    Flow when a dataset is selected:
    1. Already in _candidate_cache → render pair as normal (fast path)
    2. Background load in progress → render loading template with polling
    3. Previous load error → render error state
    4. Cache file exists on disk → start background load (fast), render loading
    5. No cache file → render "not cached" template with Compute button
    """
    datasets = list_datasets()
    is_htmx = request.headers.get("HX-Request") == "true"

    # No dataset selected
    if not dataset:
        context = {
            "mode": "labeling",
            "datasets": datasets,
            "dataset": None,
            "pair": None,
            "geojson": "{}",
            "pair_index": 0,
            "total_pairs": 0,
            "labeled_count": 0,
            "total_candidates": 0,
        }
        if is_htmx:
            return templates.TemplateResponse(request, "labeling/pair.html", context)
        return templates.TemplateResponse(request, "labeling/page.html", context)

    # Reject unknown dataset IDs before any cache/filesystem interaction
    if dataset not in datasets:
        return HTMLResponse(status_code=404, content="Unknown dataset")

    # 1. Already loaded in memory cache → render pair
    if dataset in _candidate_cache:
        return _render_pair(request, dataset, datasets, index)

    # Shared defaults for templates that extend page.html but have no pair data
    base_context = {
        "mode": "labeling",
        "datasets": datasets,
        "dataset": dataset,
        "pair": None,
        "geojson": "{}",
        "pair_index": 0,
        "total_pairs": 0,
        "labeled_count": 0,
        "total_candidates": 0,
    }

    # 2. Background load in progress → show loading spinner
    if dataset in loading_tasks:
        if is_htmx:
            return templates.TemplateResponse(request, "labeling/loading.html", base_context)
        return templates.TemplateResponse(request, "labeling/page_loading.html", base_context)

    # 3. Previous load error → show error
    if dataset in loading_errors:
        error = loading_errors.pop(dataset)
        context = {**base_context, "error": error}
        if is_htmx:
            return templates.TemplateResponse(request, "labeling/not_cached.html", context)
        return templates.TemplateResponse(request, "labeling/page_not_cached.html", context)

    # 4. Cache file exists on disk → start background load (will be fast)
    if is_dataset_cached(dataset):
        _start_background_load(dataset)
        if is_htmx:
            return templates.TemplateResponse(request, "labeling/loading.html", base_context)
        return templates.TemplateResponse(request, "labeling/page_loading.html", base_context)

    # 5. No cache → prompt user to explicitly start computation
    if is_htmx:
        return templates.TemplateResponse(request, "labeling/not_cached.html", base_context)
    return templates.TemplateResponse(request, "labeling/page_not_cached.html", base_context)


@router.post("/labeling/compute")
async def compute_candidates(request: Request, dataset: str = Form(...)):
    """Start background feature computation for a dataset."""
    datasets = list_datasets()
    if dataset not in datasets:
        return HTMLResponse(status_code=404, content="Unknown dataset")
    if dataset not in loading_tasks:
        _start_background_load(dataset)
    context = {"mode": "labeling", "datasets": datasets, "dataset": dataset}
    return templates.TemplateResponse(request, "labeling/loading.html", context)


@router.get("/labeling/status")
async def loading_status(request: Request, dataset: str):
    """Polled by HTMX to check loading progress."""
    datasets = list_datasets()
    if dataset not in datasets:
        return HTMLResponse(status_code=404, content="Unknown dataset")

    # Done — return real pair content
    if dataset in _candidate_cache:
        return _render_pair(request, dataset, datasets)

    # Failed — show error with retry option
    if dataset in loading_errors:
        error = loading_errors.pop(dataset)
        context = {
            "mode": "labeling",
            "datasets": datasets,
            "dataset": dataset,
            "error": error,
        }
        return templates.TemplateResponse(request, "labeling/not_cached.html", context)

    # Still loading — return loading template (HTMX keeps polling)
    context = {"mode": "labeling", "datasets": datasets, "dataset": dataset}
    return templates.TemplateResponse(request, "labeling/loading.html", context)


@router.post("/labeling/label")
async def label_pair(
    request: Request,
    dataset: str = Form(...),
    index: int = Form(0),
    label: str = Form(...),
):
    """Record a label and return the next pair fragment.

    Args:
        request: FastAPI request
        dataset: Dataset ID
        index: Current index in unlabeled candidates
        label: Label value (match, no_match, unsure)
    """
    if label not in VALID_LABELS:
        return HTMLResponse(status_code=400, content="Invalid label value")

    if dataset not in list_datasets():
        return HTMLResponse(status_code=404, content="Unknown dataset")

    # Get the current candidates
    all_candidates = _get_candidates(dataset)
    unlabeled = get_unlabeled_candidates(dataset, all_candidates)

    # Record the label for the current pair
    if unlabeled and 0 <= index < len(unlabeled):
        pair = unlabeled[index]
        record_label(dataset, pair, label)

    # Re-filter after labeling (the pair we just labeled is now excluded)
    unlabeled = get_unlabeled_candidates(dataset, all_candidates)

    # Stay at the same index (the next pair slides into the current position)
    # but clamp to valid range
    next_index = min(index, max(0, len(unlabeled) - 1))

    next_pair = None
    geojson = "{}"
    if unlabeled and 0 <= next_index < len(unlabeled):
        next_pair = unlabeled[next_index]
        geojson = _pair_to_geojson(next_pair)

    context = {
        "mode": "labeling",
        "datasets": list_datasets(),
        "dataset": dataset,
        "pair": next_pair,
        "geojson": geojson,
        "pair_index": next_index,
        "total_pairs": len(unlabeled),
        "labeled_count": len(all_candidates) - len(unlabeled),
        "total_candidates": len(all_candidates),
    }

    return templates.TemplateResponse(request, "labeling/pair.html", context)


@router.post("/labeling/undo")
async def undo_label(
    request: Request,
    dataset: str = Form(...),
):
    """Undo the last label and return updated pair fragment.

    Args:
        request: FastAPI request
        dataset: Dataset ID
    """
    if dataset not in list_datasets():
        return HTMLResponse(status_code=404, content="Unknown dataset")

    undo_last_label(dataset)

    # Re-filter candidates
    all_candidates = _get_candidates(dataset)
    unlabeled = get_unlabeled_candidates(dataset, all_candidates)

    pair = None
    geojson = "{}"
    pair_index = 0
    if unlabeled:
        pair = unlabeled[0]
        geojson = _pair_to_geojson(pair)

    context = {
        "mode": "labeling",
        "datasets": list_datasets(),
        "dataset": dataset,
        "pair": pair,
        "geojson": geojson,
        "pair_index": pair_index,
        "total_pairs": len(unlabeled),
        "labeled_count": len(all_candidates) - len(unlabeled),
        "total_candidates": len(all_candidates),
    }

    return templates.TemplateResponse(request, "labeling/pair.html", context)


@router.get("/labeling/features")
async def labeling_features(request: Request, dataset: str, index: int = 0):
    """Return features HTML for a specific pair (lazy-loaded by the features drawer)."""
    if dataset not in _candidate_cache:
        return HTMLResponse("<p>No features available.</p>")

    all_candidates = _candidate_cache[dataset]
    unlabeled = get_unlabeled_candidates(dataset, all_candidates)

    pair = None
    if unlabeled and 0 <= index < len(unlabeled):
        pair = unlabeled[index]

    return templates.TemplateResponse(request, "labeling/features.html", {"pair": pair})


@router.post("/labeling/refresh")
async def refresh_candidates(
    dataset: str = Form(...),
):
    """Clear cached candidates and redirect to reload the dataset."""
    # Don't clear if a background task is already running
    if dataset not in loading_tasks:
        _candidate_cache.pop(dataset, None)
    return RedirectResponse(
        url=f"/labeling?dataset={dataset}",
        status_code=303,
    )


@router.post("/settings/labeler")
async def set_labeler_name(name: str = Form(...)):
    """Save labeler name to config file."""
    config = {}
    if CONFIG_FILE.exists():
        with contextlib.suppress(Exception):
            config = json.loads(CONFIG_FILE.read_text())
    config["labeler_name"] = name
    CONFIG_FILE.write_text(json.dumps(config))
    return HTMLResponse(status_code=204)
