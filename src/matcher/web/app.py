"""FastAPI application factory for the matcher web UI."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .routes.labeling import router as labeling_router
from .routes.qa import router as qa_router
from .routes.review import router as review_router

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"


class LabelerNameMiddleware(BaseHTTPMiddleware):
    """Inject the configured labeler name into every request's state."""

    async def dispatch(self, request, call_next):
        from .services import get_labeler_name

        request.state.labeler_name = get_labeler_name()
        return await call_next(request)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Matcher Web UI", docs_url=None, redoc_url=None)

    # Middleware (runs before routers)
    app.add_middleware(LabelerNameMiddleware)

    # Mount static files
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include routers
    app.include_router(labeling_router)
    app.include_router(qa_router)
    app.include_router(review_router)

    return app
