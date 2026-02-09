"""Batch Label mode routes for the matcher web UI."""

import json
import logging
from pathlib import Path
from threading import Thread

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from shapely.geometry import mapping

from ..services import (
    delete_batch_manifest,
    generate_batch,
    get_unlabeled_candidates,
    has_batch,
    is_dataset_cached,
    list_datasets,
    load_batch,
    load_batch_manifest,
    loading_errors,
    loading_lock,
    loading_tasks,
    record_label,
    undo_last_label,
)

logger = logging.getLogger(__name__)

VALID_LABELS = {"match", "no_match", "unsure"}

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

# Module-level cache for loaded batch candidates per dataset
_batch_cache: dict[str, list] = {}


def _pair_to_geojson(pair) -> str:
    """Convert a CandidatePairView's geometries to a GeoJSON JSON string."""
    result = {
        "reference_full": mapping(pair.ref_geometry),
        "target_full": mapping(pair.target_geometry),
    }

    if pair.ref_aligned_geometry is not None:
        result["reference"] = mapping(pair.ref_aligned_geometry)
    else:
        result["reference"] = mapping(pair.ref_geometry)

    if pair.target_aligned_geometry is not None:
        result["target"] = mapping(pair.target_aligned_geometry)
    else:
        result["target"] = mapping(pair.target_geometry)

    return json.dumps(result)


def _start_background_generate(dataset_id: str) -> None:
    """Start batch generation in a background thread."""
    with loading_lock:
        if dataset_id in loading_tasks:
            return

        def _do_generate():
            try:
                result = generate_batch(dataset_id)
                with loading_lock:
                    _batch_cache[dataset_id] = result
            except Exception:
                logger.exception("Batch generation failed for dataset %s", dataset_id)
                with loading_lock:
                    loading_errors[dataset_id] = "Batch generation failed. Check server logs."
            finally:
                with loading_lock:
                    loading_tasks.pop(dataset_id, None)

        thread = Thread(target=_do_generate, daemon=True, name=f"batch-{dataset_id}")
        loading_errors.pop(dataset_id, None)
        loading_tasks[dataset_id] = thread
        thread.start()


def _start_background_load(dataset_id: str) -> None:
    """Start loading an existing batch in a background thread."""
    with loading_lock:
        if dataset_id in loading_tasks:
            return

        def _do_load():
            try:
                result = load_batch(dataset_id)
                with loading_lock:
                    _batch_cache[dataset_id] = result
            except Exception:
                logger.exception("Batch load failed for dataset %s", dataset_id)
                with loading_lock:
                    loading_errors[dataset_id] = "Failed to load batch. Check server logs."
            finally:
                with loading_lock:
                    loading_tasks.pop(dataset_id, None)

        thread = Thread(target=_do_load, daemon=True, name=f"batch-load-{dataset_id}")
        loading_errors.pop(dataset_id, None)
        loading_tasks[dataset_id] = thread
        thread.start()


def _get_batch_progress(dataset_id: str, all_candidates: list) -> tuple[int, int]:
    """Get batch progress: (labeled_count, total_in_batch).

    Returns the number of pairs in this batch that have been labeled,
    and the total number of pairs in the batch.
    """
    manifest = load_batch_manifest(dataset_id)
    batch_total = manifest["n_total"] if manifest else len(all_candidates)
    unlabeled = get_unlabeled_candidates(dataset_id, all_candidates)
    batch_labeled = len(all_candidates) - len(unlabeled)
    return batch_labeled, batch_total


def _render_pair(request, dataset, datasets, index=0):
    """Render a normal pair view for a batch that is loaded."""
    is_htmx = request.headers.get("HX-Request") == "true"

    all_candidates = _batch_cache[dataset]
    unlabeled = get_unlabeled_candidates(dataset, all_candidates)
    batch_labeled, batch_total = _get_batch_progress(dataset, all_candidates)

    # All labeled in batch
    if not unlabeled:
        context = {
            "mode": "batch",
            "datasets": datasets,
            "dataset": dataset,
            "batch_labeled": batch_labeled,
            "batch_total": batch_total,
        }
        if is_htmx:
            return templates.TemplateResponse(request, "batch/complete.html", context)
        context.update(
            {
                "pair": None,
                "geojson": "{}",
                "pair_index": 0,
                "total_pairs": 0,
                "labeled_count": batch_labeled,
                "total_candidates": batch_total,
                "url_prefix": "/batch",
            }
        )
        return templates.TemplateResponse(request, "batch/page.html", context)

    if unlabeled:
        index = max(0, min(index, len(unlabeled) - 1))

    pair = None
    geojson = "{}"
    if unlabeled and 0 <= index < len(unlabeled):
        pair = unlabeled[index]
        geojson = _pair_to_geojson(pair)

    context = {
        "mode": "batch",
        "datasets": datasets,
        "dataset": dataset,
        "pair": pair,
        "geojson": geojson,
        "pair_index": index,
        "total_pairs": len(unlabeled),
        "labeled_count": batch_labeled,
        "total_candidates": batch_total,
        "batch_labeled": batch_labeled,
        "batch_total": batch_total,
        "url_prefix": "/batch",
    }

    if is_htmx:
        return templates.TemplateResponse(request, "labeling/pair.html", context)
    return templates.TemplateResponse(request, "batch/page.html", context)


@router.get("/batch")
async def batch_page(
    request: Request,
    dataset: str | None = None,
    index: int = 0,
):
    """Render the batch labeling page.

    Flow when a dataset is selected:
    1. Already in _batch_cache -> render pair
    2. Background load in progress -> render loading template
    3. Previous load error -> render error state
    4. Batch file exists on disk -> start background load, render loading
    5. Feature cache exists but no batch -> show "Generate Batch" button
    6. No feature cache -> show "Features not computed" message
    """
    datasets = list_datasets()
    is_htmx = request.headers.get("HX-Request") == "true"

    # No dataset selected
    if not dataset:
        context = {
            "mode": "batch",
            "datasets": datasets,
            "dataset": None,
            "pair": None,
            "geojson": "{}",
            "pair_index": 0,
            "total_pairs": 0,
            "labeled_count": 0,
            "total_candidates": 0,
            "batch_labeled": 0,
            "batch_total": 0,
            "url_prefix": "/batch",
        }
        if is_htmx:
            return templates.TemplateResponse(request, "labeling/pair.html", context)
        return templates.TemplateResponse(request, "batch/page.html", context)

    if dataset not in datasets:
        return HTMLResponse(status_code=404, content="Unknown dataset")

    # 1. Already loaded in memory cache -> render pair
    if dataset in _batch_cache:
        return _render_pair(request, dataset, datasets, index)

    # Shared defaults for non-pair templates
    base_context = {
        "mode": "batch",
        "datasets": datasets,
        "dataset": dataset,
        "pair": None,
        "geojson": "{}",
        "pair_index": 0,
        "total_pairs": 0,
        "labeled_count": 0,
        "total_candidates": 0,
        "batch_labeled": 0,
        "batch_total": 0,
        "url_prefix": "/batch",
    }

    # 2. Background load in progress -> show loading spinner
    if dataset in loading_tasks:
        if is_htmx:
            return templates.TemplateResponse(request, "batch/loading.html", base_context)
        return templates.TemplateResponse(request, "batch/page_loading.html", base_context)

    # 3. Previous load error -> show error
    if dataset in loading_errors:
        error = loading_errors.pop(dataset)
        context = {**base_context, "error": error}
        if is_htmx:
            return templates.TemplateResponse(request, "batch/empty.html", context)
        return templates.TemplateResponse(request, "batch/page_empty.html", context)

    # 4. Batch file exists on disk -> start background load
    if has_batch(dataset):
        _start_background_load(dataset)
        if is_htmx:
            return templates.TemplateResponse(request, "batch/loading.html", base_context)
        return templates.TemplateResponse(request, "batch/page_loading.html", base_context)

    # 5. Feature cache exists but no batch -> show Generate button
    if is_dataset_cached(dataset):
        if is_htmx:
            return templates.TemplateResponse(request, "batch/empty.html", base_context)
        return templates.TemplateResponse(request, "batch/page_empty.html", base_context)

    # 6. No feature cache -> show message
    context = {
        **base_context,
        "error": (
            "Feature cache not found. Use Label Creation mode to compute "
            "features first, then return here to generate a batch."
        ),
    }
    if is_htmx:
        return templates.TemplateResponse(request, "batch/empty.html", context)
    return templates.TemplateResponse(request, "batch/page_empty.html", context)


@router.post("/batch/generate")
async def generate_batch_endpoint(request: Request, dataset: str = Form(...)):
    """Start batch generation in a background thread."""
    datasets = list_datasets()
    if dataset not in datasets:
        return HTMLResponse(status_code=404, content="Unknown dataset")

    if dataset not in loading_tasks:
        _start_background_generate(dataset)

    context = {"mode": "batch", "datasets": datasets, "dataset": dataset}
    return templates.TemplateResponse(request, "batch/loading.html", context)


@router.get("/batch/status")
async def batch_status(request: Request, dataset: str):
    """Polled by HTMX to check batch generation progress."""
    datasets = list_datasets()
    if dataset not in datasets:
        return HTMLResponse(status_code=404, content="Unknown dataset")

    # Done -> return real pair content
    if dataset in _batch_cache:
        return _render_pair(request, dataset, datasets)

    # Failed -> show error with retry option
    if dataset in loading_errors:
        error = loading_errors.pop(dataset)
        context = {
            "mode": "batch",
            "datasets": datasets,
            "dataset": dataset,
            "error": error,
        }
        return templates.TemplateResponse(request, "batch/empty.html", context)

    # Still loading -> return loading template
    context = {"mode": "batch", "datasets": datasets, "dataset": dataset}
    return templates.TemplateResponse(request, "batch/loading.html", context)


@router.post("/batch/label")
async def label_batch_pair(
    request: Request,
    dataset: str = Form(...),
    index: int = Form(0),
    label: str = Form(...),
):
    """Record a label for a batch pair and return the next pair."""
    if label not in VALID_LABELS:
        return HTMLResponse(status_code=400, content="Invalid label value")

    if dataset not in list_datasets():
        return HTMLResponse(status_code=404, content="Unknown dataset")

    # Guard: batch must be loaded before accepting labels
    if dataset not in _batch_cache:
        return HTMLResponse(status_code=409, content="Batch not loaded yet")

    all_candidates = _batch_cache[dataset]
    unlabeled = get_unlabeled_candidates(dataset, all_candidates)

    # Record the label
    if unlabeled and 0 <= index < len(unlabeled):
        pair = unlabeled[index]
        record_label(dataset, pair, label)

    # Re-filter after labeling
    unlabeled = get_unlabeled_candidates(dataset, all_candidates)
    batch_labeled, batch_total = _get_batch_progress(dataset, all_candidates)
    datasets = list_datasets()

    # All labeled -> show complete
    if not unlabeled:
        context = {
            "mode": "batch",
            "datasets": datasets,
            "dataset": dataset,
            "batch_labeled": batch_labeled,
            "batch_total": batch_total,
        }
        return templates.TemplateResponse(request, "batch/complete.html", context)

    next_index = min(index, max(0, len(unlabeled) - 1))

    next_pair = None
    geojson = "{}"
    if unlabeled and 0 <= next_index < len(unlabeled):
        next_pair = unlabeled[next_index]
        geojson = _pair_to_geojson(next_pair)

    context = {
        "mode": "batch",
        "datasets": datasets,
        "dataset": dataset,
        "pair": next_pair,
        "geojson": geojson,
        "pair_index": next_index,
        "total_pairs": len(unlabeled),
        "labeled_count": batch_labeled,
        "total_candidates": batch_total,
        "batch_labeled": batch_labeled,
        "batch_total": batch_total,
        "url_prefix": "/batch",
    }

    return templates.TemplateResponse(request, "labeling/pair.html", context)


@router.post("/batch/undo")
async def undo_batch_label(
    request: Request,
    dataset: str = Form(...),
):
    """Undo the last label and return updated pair fragment."""
    if dataset not in list_datasets():
        return HTMLResponse(status_code=404, content="Unknown dataset")

    undo_last_label(dataset)

    all_candidates = _batch_cache.get(dataset, [])
    unlabeled = get_unlabeled_candidates(dataset, all_candidates)
    batch_labeled, batch_total = _get_batch_progress(dataset, all_candidates)
    datasets = list_datasets()

    pair = None
    geojson = "{}"
    pair_index = 0
    if unlabeled:
        pair = unlabeled[0]
        geojson = _pair_to_geojson(pair)

    context = {
        "mode": "batch",
        "datasets": datasets,
        "dataset": dataset,
        "pair": pair,
        "geojson": geojson,
        "pair_index": pair_index,
        "total_pairs": len(unlabeled),
        "labeled_count": batch_labeled,
        "total_candidates": batch_total,
        "batch_labeled": batch_labeled,
        "batch_total": batch_total,
        "url_prefix": "/batch",
    }

    return templates.TemplateResponse(request, "labeling/pair.html", context)


@router.post("/batch/regenerate")
async def regenerate_batch(request: Request, dataset: str = Form(...)):
    """Clear current batch and generate a new one."""
    datasets = list_datasets()
    if dataset not in datasets:
        return HTMLResponse(status_code=404, content="Unknown dataset")

    # Clear existing batch
    _batch_cache.pop(dataset, None)
    delete_batch_manifest(dataset)

    # Start new generation
    if dataset not in loading_tasks:
        _start_background_generate(dataset)

    context = {"mode": "batch", "datasets": datasets, "dataset": dataset}
    return templates.TemplateResponse(request, "batch/loading.html", context)
