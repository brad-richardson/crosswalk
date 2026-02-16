"""Dashboard routes for the matcher web UI."""

import json
import logging
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from ..jinja import templates
from ..services import (
    get_cv_trends,
    get_dataset_detail,
    get_dataset_metrics,
    get_overall_metrics,
    list_datasets,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard")

# Project root for CLI commands
PROJECT_ROOT = Path(__file__).parents[4]

VALID_FETCH_TYPES = {"target", "reference", "all"}
VALID_CACHE_TYPES = {"features", "candidates", "integration", "all"}
MAX_COMPLETED_TASKS = 50

# Background task state
_tasks: dict[str, dict] = {}
_task_lock = threading.Lock()


def _run_subprocess(task_id: str, cmd: list[str], description: str) -> None:
    """Run a CLI command in a daemon thread, capture output."""
    try:
        with _task_lock:
            _tasks[task_id]["status"] = "running"
            _tasks[task_id]["message"] = f"Running: {description}"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=600,
        )

        with _task_lock:
            if result.returncode == 0:
                _tasks[task_id]["status"] = "completed"
                _tasks[task_id]["message"] = f"Completed: {description}"
                _tasks[task_id]["output"] = result.stdout[-2000:] if result.stdout else ""
            else:
                _tasks[task_id]["status"] = "failed"
                _tasks[task_id]["message"] = f"Failed: {description}"
                _tasks[task_id]["output"] = result.stderr[-2000:] if result.stderr else ""
    except subprocess.TimeoutExpired:
        with _task_lock:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["message"] = f"Timed out: {description}"
            _tasks[task_id]["output"] = "Command timed out after 10 minutes"
    except Exception as e:
        logger.exception("Task %s failed", task_id)
        with _task_lock:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["message"] = f"Error: {description}"
            _tasks[task_id]["output"] = str(e)


def _prune_tasks() -> None:
    """Remove oldest completed/failed tasks beyond MAX_COMPLETED_TASKS. Caller holds _task_lock."""
    done = [tid for tid, t in _tasks.items() if t["status"] in ("completed", "failed")]
    if len(done) > MAX_COMPLETED_TASKS:
        for tid in done[: len(done) - MAX_COMPLETED_TASKS]:
            del _tasks[tid]


def _start_task(cmd: list[str], description: str) -> str:
    """Start a background task and return its ID."""
    task_id = str(uuid.uuid4())[:8]
    with _task_lock:
        _tasks[task_id] = {
            "status": "starting",
            "message": f"Starting: {description}",
            "output": "",
            "description": description,
        }
        _prune_tasks()
    thread = threading.Thread(
        target=_run_subprocess,
        args=(task_id, cmd, description),
        daemon=True,
        name=f"task-{task_id}",
    )
    thread.start()
    return task_id


# --- Page routes ---


@router.get("")
async def dashboard_page(request: Request):
    """Main dashboard page with metrics, dataset table, and trends chart."""
    overall = get_overall_metrics()
    datasets = get_dataset_metrics()
    trends = get_cv_trends()
    return templates.TemplateResponse(
        request,
        "dashboard/page.html",
        {
            "mode": "dashboard",
            "overall": overall,
            "datasets": datasets,
            "trends_json": json.dumps(trends),
        },
    )


@router.get("/admin")
async def admin_page(request: Request):
    """Admin actions page."""
    with _task_lock:
        active_tasks = [
            {"id": tid, **tdata}
            for tid, tdata in _tasks.items()
            if tdata["status"] in ("starting", "running")
        ]
        recent_tasks = [
            {"id": tid, **tdata}
            for tid, tdata in sorted(_tasks.items(), key=lambda x: x[0], reverse=True)
            if tdata["status"] in ("completed", "failed")
        ][:10]
    return templates.TemplateResponse(
        request,
        "dashboard/admin.html",
        {
            "mode": "dashboard",
            "active_tasks": active_tasks,
            "recent_tasks": recent_tasks,
        },
    )


@router.get("/{dataset}")
async def dataset_detail_page(request: Request, dataset: str):
    """Dataset detail page."""
    detail = get_dataset_detail(dataset)
    if detail is None:
        return HTMLResponse(status_code=404, content="Unknown dataset")
    return templates.TemplateResponse(
        request,
        "dashboard/dataset.html",
        {
            "mode": "dashboard",
            "detail": detail,
            "dataset": dataset,
        },
    )


# --- Task endpoints ---


@router.post("/task/fetch")
async def task_fetch(
    request: Request,
    dataset: str = Form(...),
    fetch_type: str = Form("target"),
):
    """Start a data fetch task."""
    if fetch_type not in VALID_FETCH_TYPES:
        return HTMLResponse(status_code=400, content="Invalid fetch type")
    if dataset not in list_datasets():
        return HTMLResponse(status_code=404, content="Unknown dataset")
    cmd = ["uv", "run", "matcher", "data", "fetch", fetch_type, dataset]
    description = f"Fetch {fetch_type} data for {dataset}"
    task_id = _start_task(cmd, description)
    return _task_status_response(request, task_id)


@router.post("/task/features")
async def task_features(
    request: Request,
    dataset: str = Form(...),
    force: bool = Form(False),
):
    """Start feature regeneration."""
    if dataset not in list_datasets():
        return HTMLResponse(status_code=404, content="Unknown dataset")
    cmd = ["uv", "run", "matcher", "ml", "features", dataset]
    if force:
        cmd.append("--force")
    description = f"Regenerate features for {dataset}"
    task_id = _start_task(cmd, description)
    return _task_status_response(request, task_id)


@router.post("/task/train")
async def task_train(request: Request):
    """Start model training."""
    cmd = ["uv", "run", "matcher", "train"]
    task_id = _start_task(cmd, "Train model")
    return _task_status_response(request, task_id)


@router.post("/task/eval")
async def task_eval(request: Request):
    """Start CV evaluation."""
    cmd = ["uv", "run", "matcher", "ml", "eval"]
    task_id = _start_task(cmd, "Run CV evaluation")
    return _task_status_response(request, task_id)


@router.post("/task/backfill")
async def task_backfill(
    request: Request,
    force: bool = Form(False),
):
    """Start feature backfill."""
    cmd = ["uv", "run", "matcher", "labels", "backfill"]
    if not force:
        cmd.append("--missing-only")
    task_id = _start_task(cmd, "Backfill features")
    return _task_status_response(request, task_id)


@router.post("/task/clear-cache")
async def task_clear_cache(
    request: Request,
    cache_type: str = Form("all"),
):
    """Clear caches directly (not via subprocess)."""
    if cache_type not in VALID_CACHE_TYPES:
        return HTMLResponse(status_code=400, content="Invalid cache type")

    from ...filenames import INTEGRATION_CACHE_DIR, LABELING_CACHE_DIR

    task_id = str(uuid.uuid4())[:8]
    cleared = []
    errors = []

    def _clear_files(label: str, directory: Path, pattern: str) -> None:
        """Remove files matching a glob pattern from a directory."""
        if not directory.exists():
            return
        files = list(directory.glob(pattern))
        if not files:
            return
        try:
            for f in files:
                f.unlink()
            cleared.append(f"{label} ({len(files)} files)")
        except OSError as e:
            logger.exception("Failed to clear %s cache", label)
            errors.append(f"{label}: {e}")

    def _clear_dir(label: str, directory: Path) -> None:
        """Remove an entire directory tree."""
        if not directory.exists():
            return
        try:
            shutil.rmtree(directory)
            cleared.append(label)
        except OSError as e:
            logger.exception("Failed to clear %s cache", label)
            errors.append(f"{label}: {e}")

    if cache_type in ("features", "all"):
        _clear_files("features", LABELING_CACHE_DIR, "*_features_*.parquet")

    if cache_type in ("candidates", "all"):
        _clear_files("candidates", LABELING_CACHE_DIR, "*_candidates*.parquet")

    if cache_type in ("integration", "all"):
        _clear_dir("integration", INTEGRATION_CACHE_DIR)

    status = "failed" if errors else "completed"
    parts = []
    if cleared:
        parts.append(f"Cleared: {', '.join(cleared)}")
    if errors:
        parts.append(f"Errors: {'; '.join(errors)}")
    message = ". ".join(parts) if parts else "No caches found"

    with _task_lock:
        _tasks[task_id] = {
            "status": status,
            "message": message,
            "output": "",
            "description": f"Clear {cache_type} caches",
        }

    return _task_status_response(request, task_id)


@router.get("/task/{task_id}")
async def task_status(request: Request, task_id: str):
    """Poll task status (HTMX)."""
    return _task_status_response(request, task_id)


def _task_status_response(request: Request, task_id: str):
    """Render task status fragment."""
    with _task_lock:
        task = _tasks.get(task_id)
    if task is None:
        return HTMLResponse(content="<p>Unknown task</p>")
    return templates.TemplateResponse(
        request,
        "dashboard/task_status.html",
        {"task_id": task_id, "task": task},
    )
