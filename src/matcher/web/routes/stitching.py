"""Stitching Review mode routes for the matcher web UI."""

import json
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

    # Tier 3: Context segments — same role as group segments so they get
    # identical solid styling when activated.  They start hidden via
    # hiddenSegments in the JS and appear as normal edges when toggled on.
    n_ref = len(ref_id_list)
    for i, (rid, geom) in enumerate(group.get("context_ref_geometries", {}).items()):
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "_role": "ref-full",
                    "_id": rid,
                    "_label": f"R{n_ref + i + 1}",
                },
            }
        )
    n_target = len(target_id_list)
    for i, (tid, geom) in enumerate(group.get("context_target_geometries", {}).items()):
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "_role": "target-full",
                    "_id": tid,
                    "_label": f"T{n_target + i + 1}",
                },
            }
        )

    # Envelope polygon
    envelope = group.get("envelope")
    if envelope:
        features.append(
            {
                "type": "Feature",
                "geometry": envelope,
                "properties": {"_role": "envelope"},
            }
        )

    return {"type": "FeatureCollection", "features": features}


def _build_stitch_options(group: dict) -> dict:
    """Build the assignment option picker + optimizer pre-seed for a group.

    The optimizer's own proposed assignment and the pre-computed top-K
    alternatives are turned into one-click options ("verify, don't construct").

    Returns a context dict with:
    - options: list of option dicts (optimizer first, then alternatives,
      deduplicated by exact edge set). Each option carries its EXACT edge set
      so the client can submit it verbatim (see the edge-set fidelity note in
      the submit endpoint).
    - preseed_active_refs / preseed_active_targets: segment IDs that should be
      active (pill selected) on load, derived from the optimizer assignment.
      Both are None when no optimizer assignment is present — the template then
      falls back to all-active, exactly matching the pre-change behavior.
    """
    group_edges = group.get("edges", []) or []
    group_edge_set = {(e["ref_id"], e["target_id"]) for e in group_edges}
    optimizer = group.get("optimizer_assignment") or []
    alternatives = group.get("alternatives") or []

    def _valid_edges(edges: list[dict]) -> list[dict]:
        """Keep only edges that exist in the group, deduplicated; strip to id pairs."""
        out = []
        seen = set()
        for e in edges:
            key = (e.get("ref_id"), e.get("target_id"))
            if key in group_edge_set and key not in seen:
                seen.add(key)
                out.append({"ref_id": key[0], "target_id": key[1]})
        return out

    def _edge_key(edges: list[dict]) -> frozenset:
        return frozenset((e["ref_id"], e["target_id"]) for e in edges)

    def _confidences(raw_edges: list[dict]) -> tuple[float, float]:
        confs = [
            e.get("confidence", 0.0)
            for e in raw_edges
            if (e.get("ref_id"), e.get("target_id")) in group_edge_set
        ]
        total = round(sum(confs), 4)
        mean = round(total / len(confs), 4) if confs else 0.0
        return total, mean

    def _make_option(key: str, label: str, is_optimizer: bool, raw_edges: list[dict]) -> dict:
        edges = _valid_edges(raw_edges)
        total, mean = _confidences(raw_edges)
        return {
            "key": key,
            "label": label,
            "is_optimizer": is_optimizer,
            "edges": edges,
            "edge_count": len(edges),
            "total_confidence": total,
            "mean_confidence": mean,
            "active_refs": sorted({e["ref_id"] for e in edges}),
            "active_targets": sorted({e["target_id"] for e in edges}),
        }

    options: list[dict] = []
    seen: set[frozenset] = set()

    # Optimizer's assignment always comes first (when present).
    if optimizer:
        opt = _make_option("optimizer", "Optimizer", True, optimizer)
        if opt["edges"]:
            options.append(opt)
            seen.add(_edge_key(opt["edges"]))

    # Alternatives, deduplicated against the optimizer's answer and each other.
    alt_num = 0
    for alt in alternatives:
        edges = _valid_edges(alt.get("edges", []))
        if not edges:
            continue
        key = _edge_key(edges)
        if key in seen:
            continue
        seen.add(key)
        alt_num += 1
        opt = _make_option(f"alt{alt_num}", f"Alt {alt_num}", False, alt.get("edges", []))
        # Prefer the alternative's own precomputed total when available.
        if "total_confidence" in alt:
            opt["total_confidence"] = round(alt["total_confidence"], 4)
            opt["mean_confidence"] = (
                round(opt["total_confidence"] / opt["edge_count"], 4) if opt["edge_count"] else 0.0
            )
        options.append(opt)

    # Pre-seed pill active-state from the optimizer assignment. Only a
    # non-empty assignment drives the pre-seed; an absent/empty assignment
    # (old batch format, or a group the optimizer dropped entirely) leaves
    # preseed as None so the template keeps every group pill active.
    preseed_refs = None
    preseed_targets = None
    preseed_inactive_ids: list[str] = []
    preseed_valid = _valid_edges(optimizer)
    if preseed_valid:
        preseed_refs = sorted({e["ref_id"] for e in preseed_valid})
        preseed_targets = sorted({e["target_id"] for e in preseed_valid})
        # Group segments the optimizer left out start hidden on the map so the
        # map matches the pre-seeded pill state.
        active_ids = set(preseed_refs) | set(preseed_targets)
        for sid in group.get("ref_ids", []) + group.get("target_ids", []):
            if sid not in active_ids:
                preseed_inactive_ids.append(sid)

    # The client submits this exact edge set verbatim when the pre-seeded
    # option is chosen without manual edits.
    preseed_edges = options[0]["edges"] if (options and options[0]["is_optimizer"]) else []

    return {
        "options": options,
        "preseed_active_refs": preseed_refs,
        "preseed_active_targets": preseed_targets,
        "preseed_inactive_ids": preseed_inactive_ids,
        "preseed_edges": preseed_edges,
        "has_preseed": bool(preseed_refs) or bool(preseed_targets),
    }


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

    def _join(vals: list[str], *, max_items: int = 6) -> str:
        filtered = [v for v in vals if v]
        if not filtered:
            return "\u2014"
        if len(filtered) <= max_items:
            return " + ".join(filtered)
        return " + ".join(filtered[:max_items]) + f" + \u2026({len(filtered)} total)"

    def _join_dedup(vals: list[str]) -> str:
        """Join with deduplication and counts for repetitive values."""
        filtered = [v for v in vals if v]
        if not filtered:
            return "\u2014"
        from collections import Counter

        counts = Counter(filtered)
        if len(counts) == 1:
            val, n = next(iter(counts.items()))
            return f"{val} \u00d7{n}" if n > 1 else val
        parts = [f"{val} \u00d7{n}" if n > 1 else val for val, n in counts.most_common(4)]
        if len(counts) > 4:
            parts.append(f"\u2026({len(counts)} unique)")
        return ", ".join(parts)

    class_summary = f"{_join_dedup(ref_class_vals)} \u2192 {_join_dedup(target_class_vals)}"
    name_summary = f"{_join_dedup(ref_name_vals)} \u2192 {_join_dedup(target_name_vals)}"

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
        _join([_shorten_id(r) for r in ref_id_list])
        + " \u2192 "
        + _join([_shorten_id(t) for t in target_id_list])
    )

    # Context segment details (from spatial fill-in)
    context_ref_names = group.get("context_ref_names", {})
    context_ref_classes = group.get("context_ref_classes", {})
    context_target_names = group.get("context_target_names", {})
    context_target_classes = group.get("context_target_classes", {})

    context_ref_details = [
        {"id": rid, "cls": context_ref_classes.get(rid, ""), "name": context_ref_names.get(rid, "")}
        for rid in group.get("context_ref_ids", [])
    ]
    context_target_details = [
        {
            "id": tid,
            "cls": context_target_classes.get(tid, ""),
            "name": context_target_names.get(tid, ""),
        }
        for tid in group.get("context_target_ids", [])
    ]

    return {
        "id_summary": id_summary,
        "class_summary": class_summary,
        "name_summary": name_summary,
        "avg_confidence": avg_conf,
        "ref_details": ref_details,
        "target_details": target_details,
        "context_ref_details": context_ref_details,
        "context_target_details": context_target_details,
        **_build_stitch_options(group),
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


def _parse_explicit_edges(raw: str, group: dict) -> list[dict] | None:
    """Parse and validate an explicit selected_edges payload.

    An option in the picker IS an exact edge set. For M:N groups the
    cross-product of an option's endpoints can contain MORE edges than the
    option itself, so when the client submits an unmodified option it sends the
    option's exact edge set here and we store it verbatim.

    Returns:
        - None when no explicit payload was sent (caller uses cross-product).
        - A validated list of {ref_id, target_id} dicts otherwise.

    Raises:
        ValueError if the payload is malformed or references an edge that does
        not exist in the group (guards against a stale/forged client submit).
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as e:
        raise ValueError("selected_edges is not valid JSON") from e
    if not isinstance(parsed, list):
        raise ValueError("selected_edges must be a list")

    group_edge_set = {(e["ref_id"], e["target_id"]) for e in group.get("edges", [])}
    cleaned = []
    for e in parsed:
        if not isinstance(e, dict) or "ref_id" not in e or "target_id" not in e:
            raise ValueError("selected_edges contains a malformed edge")
        key = (e["ref_id"], e["target_id"])
        if key not in group_edge_set:
            raise ValueError(f"selected_edges contains a non-group edge: {key}")
        cleaned.append({"ref_id": e["ref_id"], "target_id": e["target_id"]})
    return cleaned


@router.post("/stitching-review/select", response_class=HTMLResponse)
async def stitching_select(
    request: Request,
    dataset: str = Form(...),
    group_id: str = Form(...),
    group_index: int = Form(0),
    included_refs: str = Form(""),
    included_targets: str = Form(""),
    selected_edges: str = Form(""),
):
    """Records selection, returns next group via HTMX swap.

    Edge-set fidelity: when the client submits an unmodified option it includes
    an explicit ``selected_edges`` payload (the option's exact edge set), which
    is stored verbatim. Any manual pill toggle after picking an option clears
    that payload on the client, so we fall back to reconstructing edges as the
    cross-product of the active ref/target pills — exactly the original path.
    """
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
        try:
            explicit_edges = _parse_explicit_edges(selected_edges, group)
        except ValueError as e:
            # Detail can embed client-supplied ids — log it, never reflect it
            logger.warning(f"Rejected selected_edges for group {group_id}: {e}")
            return HTMLResponse("Invalid selected_edges", status_code=400)

        if explicit_edges is not None:
            # Exact option edge set — store verbatim.
            final_edges = explicit_edges
            num_refs = len({e["ref_id"] for e in final_edges})
            num_targets = len({e["target_id"] for e in final_edges})
        else:
            # Manual mode: cross-product of the active ref/target pills.
            ref_set = set(r for r in included_refs.split(",") if r)
            target_set = set(t for t in included_targets.split(",") if t)
            final_edges = [
                {"ref_id": e["ref_id"], "target_id": e["target_id"]}
                for e in group.get("edges", [])
                if e["ref_id"] in ref_set and e["target_id"] in target_set
            ]
            num_refs = len(ref_set)
            num_targets = len(target_set)

        record_stitching_label(
            dataset_id=dataset,
            group_id=group_id,
            selected_edges=final_edges,
            match_type=group.get("match_type", ""),
            num_refs=num_refs,
            num_targets=num_targets,
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
