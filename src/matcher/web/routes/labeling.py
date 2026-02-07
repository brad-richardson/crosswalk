"""Labeling mode routes for the matcher web UI."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from shapely.geometry import mapping

from ..services import (
    get_unlabeled_candidates,
    list_datasets,
    load_candidates,
    record_label,
    undo_last_label,
)

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

# Module-level cache for loaded candidates per dataset
_candidate_cache: dict[str, list] = {}


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
    - gers: aligned reference geometry (or full if no aligned)
    - local: aligned target geometry (or full if no aligned)
    - gers_full: full reference geometry
    - local_full: full target geometry

    Args:
        pair: CandidatePairView instance

    Returns:
        JSON string suitable for embedding in an HTML data attribute.
    """
    result = {
        "gers_full": mapping(pair.ref_geometry),
        "local_full": mapping(pair.target_geometry),
    }

    # Use aligned geometries if available, otherwise fall back to full
    if pair.ref_aligned_geometry is not None:
        result["gers"] = mapping(pair.ref_aligned_geometry)
    else:
        result["gers"] = mapping(pair.ref_geometry)

    if pair.target_aligned_geometry is not None:
        result["local"] = mapping(pair.target_aligned_geometry)
    else:
        result["local"] = mapping(pair.target_geometry)

    return json.dumps(result)


@router.get("/")
async def index():
    """Redirect root to the labeling page."""
    return RedirectResponse(url="/labeling", status_code=307)


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

    Args:
        request: FastAPI request
        dataset: Optional dataset ID to load
        index: Index into the unlabeled candidates list
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
        }
        if is_htmx:
            return templates.TemplateResponse(request, "labeling/pair.html", context)
        return templates.TemplateResponse(request, "labeling/page.html", context)

    # Load candidates and filter to unlabeled
    try:
        all_candidates = _get_candidates(dataset)
        unlabeled = get_unlabeled_candidates(dataset, all_candidates)
    except Exception:
        logger.exception("Failed to load candidates for dataset %s", dataset)
        all_candidates = []
        unlabeled = []

    # Get the current pair
    pair = None
    geojson = "{}"
    if unlabeled and 0 <= index < len(unlabeled):
        pair = unlabeled[index]
        geojson = _pair_to_geojson(pair)

    context = {
        "mode": "labeling",
        "datasets": datasets,
        "dataset": dataset,
        "pair": pair,
        "geojson": geojson,
        "pair_index": index,
        "total_pairs": len(unlabeled),
    }

    if is_htmx:
        return templates.TemplateResponse(request, "labeling/pair.html", context)
    return templates.TemplateResponse(request, "labeling/page.html", context)


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
    }

    return templates.TemplateResponse(request, "labeling/pair.html", context)
