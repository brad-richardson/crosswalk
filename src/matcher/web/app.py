"""FastAPI application factory for the matcher web UI."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes.labeling import router as labeling_router
from .routes.qa import router as qa_router
from .routes.review import router as review_router

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Matcher Web UI", docs_url=None, redoc_url=None)

    # Mount static files
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include routers
    app.include_router(labeling_router)
    app.include_router(qa_router)
    app.include_router(review_router)

    return app
