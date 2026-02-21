"""Stitching Review mode routes for the matcher web UI."""

import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from shapely.geometry import LineString, mapping, shape
from shapely.ops import substring

from ...matching.alternatives import _shorten_id
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


def _build_group_geojson(group: dict) -> dict:
    """Build GeoJSON FeatureCollection using ALL edges in the group.

    Emits two tiers of features:
    - Full geometries: thin, faded — show full extent of each segment
    - Aligned sub-segments: thick, bright — from all edges (no gaps)

    Feature _role values:
    - "ref-full" / "target-full": full segment geometries
    - "ref-aligned" / "target-aligned": aligned sub-segments from edges
    """
    ref_geoms = group.get("ref_geometries", {})
    target_geoms = group.get("target_geometries", {})
    edges = group.get("edges", [])
    ref_id_list = group.get("ref_ids", list(ref_geoms.keys()))
    target_id_list = group.get("target_ids", list(target_geoms.keys()))

    if not ref_geoms and not target_geoms:
        return {"type": "FeatureCollection", "features": []}

    # Build label index maps: segment ID -> "R1", "T2", etc.
    ref_labels = {rid: f"R{i + 1}" for i, rid in enumerate(ref_id_list)}
    target_labels = {tid: f"T{i + 1}" for i, tid in enumerate(target_id_list)}

    features = []

    # Tier 1: Full geometries (thin, faded)
    for rid, geom in ref_geoms.items():
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"_role": "ref-full", "_id": rid, "_label": ref_labels.get(rid, "")},
            }
        )
    for tid, geom in target_geoms.items():
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "_role": "target-full",
                    "_id": tid,
                    "_label": target_labels.get(tid, ""),
                },
            }
        )

    # Tier 2: Aligned sub-segments from ALL edges (thick, bright)
    for edge in edges:
        rid = edge["ref_id"]
        tid = edge["target_id"]

        # Ref-side aligned sub-segment
        ref_geom = ref_geoms.get(rid)
        if ref_geom:
            gers_start = edge.get("gers_start_frac")
            gers_end = edge.get("gers_end_frac")
            if gers_start is not None and gers_end is not None:
                aligned = _extract_subline_geojson(ref_geom, gers_start, gers_end)
            else:
                aligned = ref_geom
            if aligned:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": aligned,
                        "properties": {"_role": "ref-aligned", "_id": rid},
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
                aligned = target_geom
            if aligned:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": aligned,
                        "properties": {"_role": "target-aligned", "_id": tid},
                    }
                )

    return {"type": "FeatureCollection", "features": features}


def _build_group_context(group: dict) -> dict:
    """Build extra template context for a group.

    Returns class/name summaries and per-segment detail lists for the
    expandable detail view.
    """
    ref_id_list = group.get("ref_ids", [])
    target_id_list = group.get("target_ids", [])

    if not ref_id_list and not target_id_list:
        return {}

    ref_names = group.get("ref_names", {})
    target_names = group.get("target_names", {})
    ref_classes = group.get("ref_classes", {})
    target_classes = group.get("target_classes", {})

    ref_class_vals = [ref_classes.get(rid, "") for rid in ref_id_list]
    ref_name_vals = [ref_names.get(rid, "") for rid in ref_id_list]
    target_class_vals = [target_classes.get(tid, "") for tid in target_id_list]
    target_name_vals = [target_names.get(tid, "") for tid in target_id_list]

    # Compute overlap-weighted average confidence from all edges.
    # Weight each edge by the average of its ref and target aligned fractions
    # so tiny junction slivers (2-3% overlap) don't drag the score down.
    edges = group.get("edges", [])
    if edges:
        weighted_sum = 0.0
        weight_total = 0.0
        for e in edges:
            ref_frac = abs(e.get("gers_end_frac", 1) - e.get("gers_start_frac", 0))
            tgt_frac = abs(e.get("local_end_frac", 1) - e.get("local_start_frac", 0))
            w = (ref_frac + tgt_frac) / 2
            weighted_sum += e.get("confidence", 0) * w
            weight_total += w
        avg_conf = weighted_sum / weight_total if weight_total > 0 else 0.0
    else:
        avg_conf = 0.0

    def _join(vals: list[str]) -> str:
        filtered = [v for v in vals if v]
        return " + ".join(filtered) if filtered else "\u2014"

    class_summary = f"{_join(ref_class_vals)} \u2192 {_join(target_class_vals)}"
    name_summary = f"{_join(ref_name_vals)} \u2192 {_join(target_name_vals)}"

    # Build details for expanded view
    ref_details = [
        {"id": rid, "cls": ref_classes.get(rid, ""), "name": ref_names.get(rid, "")}
        for rid in ref_id_list
    ]
    target_details = [
        {"id": tid, "cls": target_classes.get(tid, ""), "name": target_names.get(tid, "")}
        for tid in target_id_list
    ]

    id_summary = (
        " + ".join(_shorten_id(r) for r in ref_id_list)
        + " \u2192 "
        + " + ".join(_shorten_id(t) for t in target_id_list)
    )

    return {
        "id_summary": id_summary,
        "class_summary": class_summary,
        "name_summary": name_summary,
        "avg_confidence": avg_conf,
        "ref_details": ref_details,
        "target_details": target_details,
    }


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

    all_groups = batch.get("groups", [])
    batch_total = len(all_groups)
    groups = get_unreviewed_stitch_groups(dataset, all_groups)
    reviewed_count = batch_total - len(groups)

    if not groups:
        return templates.TemplateResponse(
            "stitching/page.html",
            {
                "request": request,
                "mode": "stitching",
                "datasets": datasets,
                "dataset": dataset,
                "group": None,
                "group_index": 0,
                "total_groups": batch_total,
                "no_groups": True,
                "all_reviewed": True,
            },
        )

    group = groups[0]
    geojson = _build_group_geojson(group)
    group_ctx = _build_group_context(group)

    return templates.TemplateResponse(
        "stitching/page.html",
        {
            "request": request,
            "mode": "stitching",
            "datasets": datasets,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": reviewed_count,
            "total_groups": batch_total,
            "no_groups": False,
            **group_ctx,
        },
    )


@router.get("/stitching-review/group", response_class=HTMLResponse)
async def stitching_group(
    request: Request,
    dataset: str = "",
    group_id: str = "",
    group_index: int = 0,
):
    """HTMX fragment: renders group card + map data."""
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

    all_groups = batch.get("groups", [])

    # Find group by ID, fall back to index
    group = None
    if group_id:
        for g in all_groups:
            if g.get("group_id") == group_id:
                group = g
                break
    if not group and 0 <= group_index < len(all_groups):
        group = all_groups[group_index]

    if not group:
        return templates.TemplateResponse(
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    geojson = _build_group_geojson(group)
    group_ctx = _build_group_context(group)

    return templates.TemplateResponse(
        "stitching/group.html",
        {
            "request": request,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": group_index,
            "total_groups": len(all_groups),
            **group_ctx,
        },
    )


@router.post("/stitching-review/select", response_class=HTMLResponse)
async def stitching_select(
    request: Request,
    dataset: str = Form(...),
    group_id: str = Form(...),
    group_index: int = Form(0),
    included_refs: str = Form(""),
    included_targets: str = Form(""),
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
        # Build selected edges from the included ref/target IDs
        ref_set = set(r for r in included_refs.split(",") if r)
        target_set = set(t for t in included_targets.split(",") if t)
        selected_edges = [
            {"ref_id": e["ref_id"], "target_id": e["target_id"]}
            for e in group.get("edges", [])
            if e["ref_id"] in ref_set and e["target_id"] in target_set
        ]

        record_stitching_label(
            dataset_id=dataset,
            group_id=group_id,
            selected_edges=selected_edges,
            match_type=group.get("match_type", ""),
            num_refs=len(ref_set),
            num_targets=len(target_set),
        )

    # Load next group
    batch_total = len(all_groups)
    groups = get_unreviewed_stitch_groups(dataset, all_groups)
    reviewed_count = batch_total - len(groups)

    if not groups:
        return templates.TemplateResponse(
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    group = groups[0]
    geojson = _build_group_geojson(group)
    group_ctx = _build_group_context(group)

    return templates.TemplateResponse(
        "stitching/group.html",
        {
            "request": request,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": reviewed_count,
            "total_groups": batch_total,
            **group_ctx,
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

    all_groups = batch.get("groups", [])
    batch_total = len(all_groups)
    groups = get_unreviewed_stitch_groups(dataset, all_groups)
    reviewed_count = batch_total - len(groups)

    if not groups:
        return templates.TemplateResponse(
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    # Find the current group by ID and advance past it
    next_index = 0
    if group_id:
        for i, g in enumerate(groups):
            if g.get("group_id") == group_id:
                next_index = (i + 1) % len(groups)
                break

    group = groups[next_index]
    geojson = _build_group_geojson(group)
    group_ctx = _build_group_context(group)

    return templates.TemplateResponse(
        "stitching/group.html",
        {
            "request": request,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": reviewed_count + next_index,
            "total_groups": batch_total,
            **group_ctx,
        },
    )
