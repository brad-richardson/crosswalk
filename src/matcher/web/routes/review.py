"""Label review routes for the matcher web UI."""

import logging
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..services import (
    delete_review_label,
    get_labels_for_review,
    list_datasets,
    update_review_label,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/review")
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("")
async def review_page(
    request: Request,
    dataset: str | None = None,
    filter: str = "all",
    page: int = 0,
):
    """Render the label review page.

    Args:
        request: FastAPI request
        dataset: Optional dataset ID to load
        filter: Filter by label type (all, match, no_match, unsure)
        page: Zero-based page number
    """
    datasets = list_datasets()
    labels = []
    total = 0
    page_size = 50

    if dataset:
        try:
            labels, total = get_labels_for_review(
                dataset, filter_type=filter, page=page, page_size=page_size
            )
        except Exception:
            logger.exception("Failed to load labels for review: %s", dataset)

    has_more = (page + 1) * page_size < total

    context = {
        "mode": "review",
        "datasets": datasets,
        "dataset": dataset,
        "filter": filter,
        "labels": labels,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": has_more,
    }
    return templates.TemplateResponse(request, "review/page.html", context)


@router.put("/{gers_id}/{target_id}")
async def update_label(
    request: Request,
    gers_id: str,
    target_id: str,
    dataset: str = Form(...),
    label: str = Form(...),
):
    """Update a label and return updated card fragment.

    Args:
        request: FastAPI request
        gers_id: Overture reference segment ID
        target_id: Target segment ID
        dataset: Dataset ID
        label: New label value (match, no_match, unsure)
    """
    success = update_review_label(dataset, gers_id, target_id, label)

    if not success:
        logger.warning("Label not found for update: %s / %s", gers_id, target_id)

    # Re-fetch the single label to get updated data
    labels, _ = get_labels_for_review(dataset, page=0, page_size=10000)
    updated = None
    for lbl in labels:
        if str(lbl.get("gers_id")) == gers_id and str(lbl.get("target_id")) == target_id:
            updated = lbl
            break

    if updated is None:
        # Fallback: construct a minimal label dict
        updated = {
            "gers_id": gers_id,
            "target_id": target_id,
            "label": label,
            "labeler": "",
            "ref_name_raw": "",
            "target_name_raw": "",
            "original_confidence": 0,
        }

    context = {
        "label": updated,
        "dataset": dataset,
    }
    return templates.TemplateResponse(request, "review/card.html", context)


@router.delete("/{gers_id}/{target_id}")
async def delete_label(
    request: Request,
    gers_id: str,
    target_id: str,
    dataset: str = Form(...),
):
    """Delete a label and return empty response (HTMX removes the element).

    Args:
        request: FastAPI request
        gers_id: Overture reference segment ID
        target_id: Target segment ID
        dataset: Dataset ID
    """
    success = delete_review_label(dataset, gers_id, target_id)

    if not success:
        logger.warning("Label not found for delete: %s / %s", gers_id, target_id)

    return HTMLResponse("")
