"""Stitching Review mode routes for the matcher web UI."""

import json
import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from ..jinja import templates
from ..services import (
    get_unreviewed_stitch_groups,
    list_datasets,
    load_stitch_batch,
    record_stitching_label,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_group_geojson(group: dict, option_index: int = 0) -> str:
    """Build GeoJSON FeatureCollection for a group's current option.

    Each feature has properties:
    - _role: "ref" or "target"
    - _id: segment ID
    - _assignment_color: hex color for the assignment group
    - _assigned: boolean
    """
    # Color palette for assignment groups
    COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#00BCD4"]

    alternatives = group.get("alternatives", [])
    if not alternatives:
        return json.dumps({"type": "FeatureCollection", "features": []})

    # Clamp option_index
    option_index = max(0, min(option_index, len(alternatives) - 1))
    option = alternatives[option_index]

    # Build ref_id -> color mapping from edges
    ref_to_color: dict[str, str] = {}
    assigned_targets: set[str] = set()
    for edge in option.get("edges", []):
        rid = edge["ref_id"]
        if rid not in ref_to_color:
            ref_to_color[rid] = COLORS[len(ref_to_color) % len(COLORS)]
        assigned_targets.add(edge["target_id"])

    # Build target -> color mapping
    target_to_color: dict[str, str] = {}
    for edge in option.get("edges", []):
        target_to_color[edge["target_id"]] = ref_to_color[edge["ref_id"]]

    features = []

    # Add ref geometries
    for rid, geom in group.get("ref_geometries", {}).items():
        color = ref_to_color.get(rid, "#999999")
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "_role": "ref",
                    "_id": rid,
                    "_assignment_color": color,
                    "_assigned": rid in ref_to_color,
                },
            }
        )

    # Add target geometries
    for tid, geom in group.get("target_geometries", {}).items():
        is_assigned = tid in assigned_targets
        color = target_to_color.get(tid, "#999999")
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "_role": "target",
                    "_id": tid,
                    "_assignment_color": color,
                    "_assigned": is_assigned,
                },
            }
        )

    return json.dumps({"type": "FeatureCollection", "features": features})


@router.get("/stitching-review", response_class=HTMLResponse)
async def stitching_review(request: Request, dataset: str = ""):
    """Main stitching review page."""
    datasets = list_datasets()

    if not dataset:
        return templates.TemplateResponse(
            "stitching/page.html",
            {
                "request": request,
                "mode": "stitching",
                "datasets": datasets,
                "dataset": "",
                "group": None,
                "group_index": 0,
                "total_groups": 0,
                "no_groups": True,
            },
        )

    try:
        batch = load_stitch_batch(dataset)
    except Exception:
        logger.exception("Failed to load stitch batch for %s", dataset)
        batch = None

    if not batch:
        return templates.TemplateResponse(
            "stitching/page.html",
            {
                "request": request,
                "mode": "stitching",
                "datasets": datasets,
                "dataset": dataset,
                "group": None,
                "group_index": 0,
                "total_groups": 0,
                "no_groups": True,
            },
        )

    groups = get_unreviewed_stitch_groups(dataset, batch.get("groups", []))
    total = len(groups)

    if total == 0:
        return templates.TemplateResponse(
            "stitching/page.html",
            {
                "request": request,
                "mode": "stitching",
                "datasets": datasets,
                "dataset": dataset,
                "group": None,
                "group_index": 0,
                "total_groups": 0,
                "no_groups": True,
                "all_reviewed": True,
            },
        )

    group = groups[0]
    geojson = _build_group_geojson(group, 0)

    return templates.TemplateResponse(
        "stitching/page.html",
        {
            "request": request,
            "mode": "stitching",
            "datasets": datasets,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": 0,
            "total_groups": total,
            "option_index": 0,
            "no_groups": False,
        },
    )


@router.get("/stitching-review/group", response_class=HTMLResponse)
async def stitching_group(
    request: Request,
    dataset: str = "",
    group_index: int = 0,
    option_index: int = 0,
):
    """HTMX fragment: renders group card + map data for a specific option."""
    try:
        batch = load_stitch_batch(dataset)
    except Exception:
        return HTMLResponse("<div>Error loading batch</div>")

    if not batch:
        return templates.TemplateResponse(
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset},
        )

    groups = get_unreviewed_stitch_groups(dataset, batch.get("groups", []))
    total = len(groups)

    if group_index >= total:
        return templates.TemplateResponse(
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    group = groups[group_index]
    geojson = _build_group_geojson(group, option_index)

    return templates.TemplateResponse(
        "stitching/group.html",
        {
            "request": request,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": group_index,
            "total_groups": total,
            "option_index": option_index,
        },
    )


@router.post("/stitching-review/select", response_class=HTMLResponse)
async def stitching_select(
    request: Request,
    dataset: str = Form(...),
    group_id: str = Form(...),
    group_index: int = Form(0),
    option_index: int = Form(0),
):
    """Records selection, returns next group via HTMX swap."""
    try:
        batch = load_stitch_batch(dataset)
    except Exception:
        return HTMLResponse("<div>Error loading batch</div>")

    if not batch:
        return HTMLResponse("<div>No batch found</div>")

    # Find the group
    all_groups = batch.get("groups", [])
    group = None
    for g in all_groups:
        if g.get("group_id") == group_id:
            group = g
            break

    if group:
        alternatives = group.get("alternatives", [])
        option_index = max(0, min(option_index, len(alternatives) - 1))

        if alternatives:
            option = alternatives[option_index]
            selected_edges = [
                {"ref_id": e["ref_id"], "target_id": e["target_id"]}
                for e in option.get("edges", [])
            ]
        else:
            selected_edges = []

        record_stitching_label(
            dataset_id=dataset,
            group_id=group_id,
            selected_option_index=option_index,
            selected_edges=selected_edges,
            match_type=group.get("match_type", ""),
            num_refs=len(group.get("ref_ids", [])),
            num_targets=len(group.get("target_ids", [])),
        )

    # Load next group
    groups = get_unreviewed_stitch_groups(dataset, all_groups)
    total = len(groups)

    if total == 0:
        return templates.TemplateResponse(
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    group = groups[0]
    geojson = _build_group_geojson(group, 0)

    return templates.TemplateResponse(
        "stitching/group.html",
        {
            "request": request,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": 0,
            "total_groups": total,
            "option_index": 0,
        },
    )


@router.post("/stitching-review/skip", response_class=HTMLResponse)
async def stitching_skip(
    request: Request,
    dataset: str = Form(...),
    group_id: str = Form(""),
):
    """Skips current group, loads next unreviewed group after it."""
    try:
        batch = load_stitch_batch(dataset)
    except Exception:
        return HTMLResponse("<div>Error loading batch</div>")

    if not batch:
        return templates.TemplateResponse(
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset},
        )

    groups = get_unreviewed_stitch_groups(dataset, batch.get("groups", []))
    total = len(groups)

    if total == 0:
        return templates.TemplateResponse(
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    # Find the current group by ID and advance past it
    next_index = 0
    if group_id:
        for i, g in enumerate(groups):
            if g.get("group_id") == group_id:
                next_index = (i + 1) % total
                break

    group = groups[next_index]
    geojson = _build_group_geojson(group, 0)

    return templates.TemplateResponse(
        "stitching/group.html",
        {
            "request": request,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": next_index,
            "total_groups": total,
            "option_index": 0,
        },
    )
