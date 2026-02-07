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
from fastapi.templating import Jinja2Templates

from ..services import (
    get_cv_trends,
    get_dataset_detail,
    get_dataset_metrics,
    get_overall_metrics,
    list_datasets,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard")
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

# Project root for CLI commands
PROJECT_ROOT = Path(__file__).parents[4]

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
    task_id = str(uuid.uuid4())[:8]
    cleared = []

    from ..services import PROJECT_ROOT as PROJ

    if cache_type in ("features", "all"):
        cache_dir = PROJ / "cache" / "features"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cleared.append("features")

    if cache_type in ("candidates", "all"):
        cache_dir = PROJ / "cache" / "candidates"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cleared.append("candidates")

    if cache_type in ("integration", "all"):
        cache_dir = PROJ / "cache" / "integration"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cleared.append("integration")

    with _task_lock:
        _tasks[task_id] = {
            "status": "completed",
            "message": f"Cleared caches: {', '.join(cleared) if cleared else 'none found'}",
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
