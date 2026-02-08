"""FastAPI application factory for the matcher web UI."""

import atexit
import logging
import multiprocessing
import os
import signal
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .routes.browser import router as browser_router
from .routes.dashboard import router as dashboard_router
from .routes.labeling import router as labeling_router
from .routes.qa import router as qa_router
from .routes.review import router as review_router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"


def _kill_child_processes() -> None:
    """Terminate all multiprocessing child processes."""
    children = multiprocessing.active_children()
    if not children:
        return
    logger.info("Shutting down %d child processes", len(children))
    for child in children:
        child.terminate()
    for child in children:
        child.join(timeout=5)
    for child in multiprocessing.active_children():
        logger.warning("Force killing child process %d", child.pid)
        child.kill()


class LabelerNameMiddleware(BaseHTTPMiddleware):
    """Inject the configured labeler name into every request's state."""

    async def dispatch(self, request, call_next):
        from .services import get_labeler_name

        request.state.labeler_name = get_labeler_name()
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage application lifespan — clean up child processes on shutdown."""
    # Register cleanup for SIGINT/SIGTERM (Ctrl+C and kill)
    # Uvicorn may not run lifespan teardown on abrupt exit, so these are backup.
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _signal_handler(signum: int, frame: object) -> None:
        _kill_child_processes()
        # Re-raise with the original handler so uvicorn can shut down normally
        original = original_sigint if signum == signal.SIGINT else original_sigterm
        if callable(original):
            original(signum, frame)
        else:
            # Restore default handler and re-raise so the process exits cleanly
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_kill_child_processes)

    yield

    _kill_child_processes()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Matcher Web UI", docs_url=None, redoc_url=None, lifespan=lifespan)

    # Middleware (runs before routers)
    app.add_middleware(LabelerNameMiddleware)

    # Mount static files
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include routers
    app.include_router(browser_router)
    app.include_router(dashboard_router)
    app.include_router(labeling_router)
    app.include_router(qa_router)
    app.include_router(review_router)

    return app
