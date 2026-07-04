"""Stitching Review mode routes for the matcher web UI."""

import json
import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from shapely.geometry import LineString, mapping, shape
from shapely.ops import substring

from ...matching.alternatives import _shorten_id
from ...matching.stitch_options import build_stitch_options as _build_stitch_options
from ..jinja import templates
from ..services import (
    get_unreviewed_stitch_groups,
    list_datasets,
    load_stitch_batch,
    record_stitching_label,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Sliver-edge classification
# ---------------------------------------------------------------------------
#
# Junction "slivers" are candidate edges where two segments barely overlap —
# typically where a road end clips the side of another road at an intersection
# (e.g. 0.2 m of a 162 m road). When a human submits a cross-product manual
# selection, these slivers get silently reconstructed into the stored label,
# polluting the training data with near-zero-overlap pairs.
#
# Classification rule: an edge is a sliver when its LARGER alignment span — the
# max of the ref coverage fraction (gers_start_frac..gers_end_frac) and the
# target coverage fraction (local_start_frac..local_end_frac) — falls below
# SLIVER_SPAN_THRESHOLD. Using the max (not min) is deliberate: legitimate
# asymmetric matches (a 10 m local segment against a 1 km ref) have one tiny
# span but the other near 1.0, so their max stays high. A true sliver fails to
# substantially cover EITHER segment, so its max span is small.
#
# Real-data justification for the 0.10 threshold: audited sliver examples had a
# max span <= 0.075 (min span <= 0.028), while substantive edges had a max span
# >= 0.10. The threshold sits in that empty band and, with a strict `<`, never
# excludes an observed substantive edge (0.10 is not < 0.10) while comfortably
# catching every observed sliver (0.075 < 0.10). Edges missing alignment fracs
# default to a span of 1.0 and are therefore treated as substantive (we never
# drop an edge we cannot measure).
SLIVER_SPAN_THRESHOLD = 0.10


def _edge_spans(edge: dict) -> tuple[float, float]:
    """Return (ref_span, tgt_span) alignment fractions for an edge.

    Missing fracs default to a full [0, 1] span (1.0) so an unmeasurable edge is
    never mistaken for a sliver.
    """
    ref_span = abs(float(edge.get("gers_end_frac", 1.0)) - float(edge.get("gers_start_frac", 0.0)))
    tgt_span = abs(
        float(edge.get("local_end_frac", 1.0)) - float(edge.get("local_start_frac", 0.0))
    )
    return ref_span, tgt_span


def is_sliver_edge(edge: dict) -> bool:
    """Classify an edge as a junction sliver (see SLIVER_SPAN_THRESHOLD notes)."""
    ref_span, tgt_span = _edge_spans(edge)
    return max(ref_span, tgt_span) < SLIVER_SPAN_THRESHOLD


def _annotate_sliver_flags(edges: list[dict]) -> list[dict]:
    """Return a copy of each edge with an ``is_sliver`` boolean added.

    Used to expose the classification to the client so the review UI can show a
    live "N slivers excluded" indicator and respect the exclusion in its live
    confidence/summary computations.
    """
    out = []
    for e in edges or []:
        ce = dict(e)
        ce["is_sliver"] = is_sliver_edge(e)
        out.append(ce)
    return out


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

    # Edges enriched with a per-edge sliver flag for the client-side UI. The map
    # + panel use these to render coverage gaps and a live sliver-exclusion count.
    client_edges = _annotate_sliver_flags(edges)
    sliver_count = sum(1 for e in client_edges if e["is_sliver"])

    return {
        "id_summary": id_summary,
        "class_summary": class_summary,
        "name_summary": name_summary,
        "avg_confidence": avg_conf,
        "client_edges": client_edges,
        "sliver_count": sliver_count,
        "ref_details": ref_details,
        "target_details": target_details,
        "context_ref_details": context_ref_details,
        "context_target_details": context_target_details,
        **_build_stitch_options(group),
    }


@router.get("/stitching-review", response_class=HTMLResponse)
async def stitching_review(request: Request, dataset: str = "", group_id: str = ""):
    """Main stitching review page.

    With ``group_id``, deep-links a specific group (reviewed or not) as a full
    page — used by audit sheets and shared links. The bare
    ``/stitching-review/group`` endpoint is an HTMX fragment and renders
    without styles/map when opened directly.
    """
    datasets = list_datasets()

    if not dataset:
        return templates.TemplateResponse(
            request,
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
            request,
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

    if group_id:
        # Deep link: render the requested group (even if already reviewed)
        # inside the full page so styles/map/JS load.
        deep_index = next(
            (i for i, g in enumerate(all_groups) if g.get("group_id") == group_id), None
        )
        if deep_index is None:
            logger.warning(f"Deep-link group not found in {dataset} batch: {group_id!r}")
            return HTMLResponse("Group not found in batch", status_code=404)
        deep_group = all_groups[deep_index]
        geojson = _build_group_geojson(deep_group)
        group_ctx = _build_group_context(deep_group)
        return templates.TemplateResponse(
            request,
            "stitching/page.html",
            {
                "request": request,
                "mode": "stitching",
                "datasets": datasets,
                "dataset": dataset,
                "group": deep_group,
                "group_geojson": geojson,
                "group_index": deep_index,
                "total_groups": batch_total,
                "no_groups": False,
                **group_ctx,
            },
        )

    if not groups:
        return templates.TemplateResponse(
            request,
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
        request,
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
            request,
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
            request,
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    geojson = _build_group_geojson(group)
    group_ctx = _build_group_context(group)

    return templates.TemplateResponse(
        request,
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
    if not cleaned:
        # A real option always has >= 1 edge, and manual mode clears the field
        # entirely. Treat an explicit empty list as "no payload" so the
        # manual-mode inconsistency guard applies instead of silently storing
        # an empty label.
        return None
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
    exclude_slivers: str = Form(""),
):
    """Records selection, returns next group via HTMX swap.

    Edge-set fidelity: when the client submits an unmodified option it includes
    an explicit ``selected_edges`` payload (the option's exact edge set), which
    is stored verbatim. Any manual pill toggle after picking an option clears
    that payload on the client, so we fall back to reconstructing edges as the
    cross-product of the active ref/target pills — exactly the original path.

    Sliver exclusion: the cross-product fallback can silently pull in junction
    "sliver" edges (near-zero overlap). When the ``exclude_slivers`` form field
    is truthy, those flagged edges are dropped from the reconstructed set. The
    field defaults to FALSE when absent, preserving behaviour for old clients
    and existing tests; the current UI always sends it explicitly. The explicit
    option path is unaffected — an option is a curated exact edge set.
    """
    exclude_sliver_edges = exclude_slivers.strip().lower() in {"true", "1", "on", "yes"}
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
            matched = [
                e
                for e in group.get("edges", [])
                if e["ref_id"] in ref_set and e["target_id"] in target_set
            ]

            # Guard against silently recording an empty (label-corrupting)
            # selection. Distinguish intent by the active-pill fields:
            #   - Both empty  -> deliberate reject-all; store [] normally.
            #   - Non-empty but zero group edges matched -> inconsistent
            #     submission (e.g. a client-side regression that drops the pill
            #     IDs, or active pills that share no edge). Refuse rather than
            #     corrupt the label. Logged without reflecting client input.
            #
            # Checked BEFORE sliver exclusion so an all-sliver selection is not
            # misread as an inconsistent submit — it is a legitimate (if empty)
            # result once the slivers are dropped.
            if group.get("edges") and not matched and (ref_set or target_set):
                logger.warning(
                    "Rejected inconsistent stitching submit for group %s: "
                    "%d refs / %d targets claimed but 0 group edges matched",
                    group_id,
                    len(ref_set),
                    len(target_set),
                )
                return HTMLResponse("Inconsistent selection", status_code=400)

            if exclude_sliver_edges:
                matched = [e for e in matched if not is_sliver_edge(e)]

            final_edges = [{"ref_id": e["ref_id"], "target_id": e["target_id"]} for e in matched]
            # Recompute counts from the final (post-exclusion) edge set so they
            # never claim segments that dropped out with their sliver edges.
            num_refs = len({e["ref_id"] for e in final_edges})
            num_targets = len({e["target_id"] for e in final_edges})

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
            request,
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    group = groups[0]
    geojson = _build_group_geojson(group)
    group_ctx = _build_group_context(group)

    return templates.TemplateResponse(
        request,
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
            request,
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset},
        )

    all_groups = batch.get("groups", [])
    batch_total = len(all_groups)
    groups = get_unreviewed_stitch_groups(dataset, all_groups)
    reviewed_count = batch_total - len(groups)

    if not groups:
        return templates.TemplateResponse(
            request,
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
        request,
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
