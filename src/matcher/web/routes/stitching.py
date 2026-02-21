"""Stitching Review mode routes for the matcher web UI."""

import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from shapely.geometry import LineString, mapping, shape
from shapely.ops import substring

from ..jinja import templates
from ..services import (
    get_unreviewed_stitch_groups,
    list_datasets,
    load_stitch_batch,
    record_stitching_label,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_dataset(dataset: str) -> bool:
    """Check dataset exists in the known dataset list (prevents path traversal)."""
    if not dataset:
        return False
    return dataset in list_datasets()


def _extract_subline_geojson(full_geojson: dict, start_frac: float, end_frac: float) -> dict | None:
    """Extract an aligned sub-segment from a GeoJSON geometry using alignment fracs."""
    try:
        geom = shape(full_geojson)
    except Exception:
        return None
    if not isinstance(geom, LineString) or geom.is_empty or geom.length == 0:
        return None
    start_frac = max(0.0, min(1.0, start_frac))
    end_frac = max(0.0, min(1.0, end_frac))
    if abs(end_frac - start_frac) < 1e-6:
        return None
    sub = substring(geom, start_frac * geom.length, end_frac * geom.length)
    if sub.is_empty:
        return None
    return mapping(sub)


def _build_group_geojson(group: dict, option_index: int = 0) -> dict:
    """Build GeoJSON FeatureCollection for a group's current option.

    Emits two tiers of features (matching the labeling UI pattern):
    - Full geometries: thin, faded — show full extent of each segment
    - Aligned sub-segments: thick, bright, colored by assignment group

    Feature _role values:
    - "ref-full" / "target-full": full segment geometries
    - "ref-aligned" / "target-aligned": aligned sub-segments from edges
    """
    # Color palette for assignment groups
    COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#00BCD4"]

    alternatives = group.get("alternatives", [])
    if not alternatives:
        return {"type": "FeatureCollection", "features": []}

    # Clamp option_index
    option_index = max(0, min(option_index, len(alternatives) - 1))
    option = alternatives[option_index]

    # Build ref_id -> color mapping from edges
    ref_to_color: dict[str, str] = {}
    for edge in option.get("edges", []):
        rid = edge["ref_id"]
        if rid not in ref_to_color:
            ref_to_color[rid] = COLORS[len(ref_to_color) % len(COLORS)]

    # Build target -> color mapping
    target_to_color: dict[str, str] = {}
    for edge in option.get("edges", []):
        target_to_color[edge["target_id"]] = ref_to_color[edge["ref_id"]]

    ref_geoms = group.get("ref_geometries", {})
    target_geoms = group.get("target_geometries", {})
    features = []

    # Tier 1: Full geometries (thin, faded)
    for rid, geom in ref_geoms.items():
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"_role": "ref-full", "_id": rid},
            }
        )
    for tid, geom in target_geoms.items():
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"_role": "target-full", "_id": tid},
            }
        )

    # Tier 2: Aligned sub-segments from edges (thick, bright, assignment-colored)
    for edge in option.get("edges", []):
        rid = edge["ref_id"]
        tid = edge["target_id"]
        color = ref_to_color.get(rid, "#999999")

        # Ref-side aligned sub-segment
        ref_geom = ref_geoms.get(rid)
        if ref_geom:
            gers_start = edge.get("gers_start_frac")
            gers_end = edge.get("gers_end_frac")
            if gers_start is not None and gers_end is not None:
                aligned = _extract_subline_geojson(ref_geom, gers_start, gers_end)
            else:
                aligned = ref_geom  # no alignment data → show full
            if aligned:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": aligned,
                        "properties": {
                            "_role": "ref-aligned",
                            "_id": rid,
                            "_assignment_color": color,
                        },
                    }
                )

        # Target-side aligned sub-segment
        target_geom = target_geoms.get(tid)
        if target_geom:
            local_start = edge.get("local_start_frac")
            local_end = edge.get("local_end_frac")
            if local_start is not None and local_end is not None:
                aligned = _extract_subline_geojson(target_geom, local_start, local_end)
            else:
                aligned = target_geom  # no alignment data → show full
            if aligned:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": aligned,
                        "properties": {
                            "_role": "target-aligned",
                            "_id": tid,
                            "_assignment_color": color,
                        },
                    }
                )

    return {"type": "FeatureCollection", "features": features}


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

    if not _validate_dataset(dataset):
        return HTMLResponse("Unknown dataset", status_code=404)

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
    if not _validate_dataset(dataset):
        return HTMLResponse("Unknown dataset", status_code=404)
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
    n_alts = len(group.get("alternatives", []))
    if n_alts > 0:
        option_index = max(0, min(option_index, n_alts - 1))
    else:
        option_index = 0
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
    if not _validate_dataset(dataset):
        return HTMLResponse("Unknown dataset", status_code=404)
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
    if not _validate_dataset(dataset):
        return HTMLResponse("Unknown dataset", status_code=404)
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
