"""Labeling mode routes for the matcher web UI."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/")
async def index():
    """Redirect root to the labeling page."""
    return RedirectResponse(url="/labeling", status_code=307)


@router.get("/labeling")
async def labeling(request: Request):
    """Render the labeling page."""
    return templates.TemplateResponse(
        request,
        "base.html",
        {
            "mode": "labeling",
            "datasets": [],
            "dataset": None,
        },
    )
