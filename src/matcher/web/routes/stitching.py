"""Stitching Review mode routes for the matcher web UI."""

import json
import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from shapely.geometry import LineString, mapping, shape
from shapely.ops import substring, unary_union

from ...filenames import PROJECT_ROOT, bridge_filename, groups_sidecar_path
from ...matching.alternatives import _shorten_id
from ...matching.sliver import (
    annotate_group_sliver_flags,
    edge_is_sliver,
    group_segment_lengths_m,
)
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

# Junction "sliver" classification is centralized in matcher.config
# (SLIVER_SPAN_THRESHOLD / SLIVER_ABS_OVERLAP_M / is_sliver_edge) and applied to
# edge/group dicts via matcher.matching.sliver. See those modules for the hybrid
# fraction + absolute-meters rule and its rationale. The prior fraction-only test
# here misclassified long-ref junction overlaps (e.g. 9% of a 2 km ref = 180 m of
# real road) as slivers; the shared hybrid rule keeps those as substantive edges.


def _validate_dataset(dataset: str) -> bool:
    """Check dataset exists in the known dataset list (prevents path traversal)."""
    if not dataset:
        return False
    return dataset in list_datasets()


# Display-only cap on how many spatial-context segments are presented per side
# (pills, map geometries, and the group-context-ids JSON blob). PR #262 expanded
# the context clip envelope to a group's full bounds, which is correct for the
# group itself but balloons the *context* layer for large groups (e.g. Boston
# cbc5cae8: 51x45 group carrying 4,470 context refs / 615 context targets). That
# swamps the UI with thousands of grey pills and a heavy Leaflet layer. We cap
# the PRESENTATION to the N context segments nearest the group; this never
# touches group data, the cached JSON, or recorded labels. Context pills are not
# selectable as group edges — the manual submit cross-products only against
# group.edges (which never reference context ids), so a context pill only toggles
# map visibility. Median per-group context_ref is ~91 (< cap), so typical groups
# render unchanged; only oversized groups get bounded.
CONTEXT_DISPLAY_CAP = 150


# Module-level cache of the segment-id -> owning-group_id map built from a
# dataset's groups sidecar. Spatial-context pills in the review UI can belong to
# a NEIGHBORING group (post corridor-decomposition), and reviewers need to know
# which group a continuation will be reviewed in. The sidecar is large (tens of
# MB), so it is loaded at most once per (dataset, sidecar-mtime) and reused
# across requests. Never a per-pill file read. Keyed by dataset; the stored
# mtime invalidates the entry when the sidecar is regenerated.
_MEMBERSHIP_CACHE: dict[str, tuple[float, dict[str, str]]] = {}


def _load_group_membership(dataset: str) -> dict[str, str]:
    """Map each ref/target segment id to its owning ``group_id``.

    Built from the dataset's groups sidecar JSON (alongside the bridge file) and
    module-cached per (dataset, sidecar-mtime). Used only to annotate context
    pills with the neighboring group a segment belongs to. Best-effort: any
    missing/unreadable/malformed sidecar yields an empty map (no hint), never an
    error — the hint is purely informational.
    """
    if not dataset:
        return {}
    try:
        bridge_path = PROJECT_ROOT / "data" / "output" / bridge_filename(dataset)
        sidecar = groups_sidecar_path(bridge_path)
        mtime = sidecar.stat().st_mtime
    except OSError:
        return {}

    cached = _MEMBERSHIP_CACHE.get(dataset)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        with open(sidecar) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}

    membership: dict[str, str] = {}
    for g in data.get("groups", []):
        gid = g.get("group_id")
        if not gid:
            continue
        for sid in g.get("ref_ids", []):
            membership.setdefault(sid, gid)
        for sid in g.get("target_ids", []):
            membership.setdefault(sid, gid)

    _MEMBERSHIP_CACHE[dataset] = (mtime, membership)
    return membership


def _cap_context_ids(group: dict, cap: int = CONTEXT_DISPLAY_CAP) -> dict:
    """Select the ``cap`` context segments per side nearest the group.

    Returns a dict with the capped, order-stable id lists plus the true totals so
    the template can surface truncation honestly. "Nearest" = shapely distance
    from each context segment geometry to the union of the group's own segment
    geometries (cheap: single-page render, not the feature hot path).

    Display-only: group ref/target ids, edges, pills, and the cached batch JSON
    are untouched. When a side is at/under the cap its ids are returned unchanged.
    """
    ctx_ref_ids = list(group.get("context_ref_ids", []))
    ctx_target_ids = list(group.get("context_target_ids", []))
    ref_total = len(ctx_ref_ids)
    target_total = len(ctx_target_ids)

    if ref_total <= cap and target_total <= cap:
        return {
            "ref_ids": ctx_ref_ids,
            "target_ids": ctx_target_ids,
            "ref_total": ref_total,
            "target_total": target_total,
        }

    # Anchor: union of whichever of the group's own geometries parse. If none
    # parse (no usable anchor), fall back to the original prefix order (ids[:cap]).
    anchor = None
    group_geoms = []
    for geom in list(group.get("ref_geometries", {}).values()) + list(
        group.get("target_geometries", {}).values()
    ):
        try:
            shp = shape(geom)
            if not shp.is_empty:
                group_geoms.append(shp)
        except Exception:
            continue
    if group_geoms:
        try:
            anchor = unary_union(group_geoms)
        except Exception:
            anchor = None

    def _nearest(ids: list[str], geoms: dict, side_total: int) -> list[str]:
        if side_total <= cap:
            return ids
        if anchor is None:
            return ids[:cap]
        scored = []
        for idx, sid in enumerate(ids):
            geom = geoms.get(sid)
            try:
                dist = shape(geom).distance(anchor) if geom is not None else float("inf")
            except Exception:
                dist = float("inf")
            # idx as tiebreaker keeps order stable for equal distances
            scored.append((dist, idx, sid))
        scored.sort()
        return [sid for _, _, sid in scored[:cap]]

    return {
        "ref_ids": _nearest(ctx_ref_ids, group.get("context_ref_geometries", {}), ref_total),
        "target_ids": _nearest(
            ctx_target_ids, group.get("context_target_geometries", {}), target_total
        ),
        "ref_total": ref_total,
        "target_total": target_total,
    }


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
    # Capped to the nearest N per side (display-only; see _cap_context_ids).
    # Iterating the capped id lists (not the geometry dicts) keeps the map,
    # pills, and context-ids JSON on one consistent ordering + label scheme.
    capped = _cap_context_ids(group)
    ctx_ref_geoms = group.get("context_ref_geometries", {})
    ctx_target_geoms = group.get("context_target_geometries", {})
    n_ref = len(ref_id_list)
    for i, rid in enumerate(capped["ref_ids"]):
        geom = ctx_ref_geoms.get(rid)
        if geom is None:
            continue
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
    for i, tid in enumerate(capped["target_ids"]):
        geom = ctx_target_geoms.get(tid)
        if geom is None:
            continue
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


def _build_group_context(group: dict, dataset: str = "") -> dict:
    """Build extra template context for a group.

    Returns class/name summaries and per-segment detail lists for the
    expandable detail view. ``dataset`` (optional) enables the context-segment
    membership hint: each context detail is annotated with the neighboring
    group it belongs to (from the groups sidecar), so reviewers know where a
    corridor continuation will be reviewed rather than assuming a context pill
    is part of this group.
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

    # Cap the presented context to the nearest N per side (display-only; the
    # cached batch, group edges, and recorded labels are untouched). The map
    # (_build_group_geojson) and the context-ids JSON blob use the same capped
    # ordering so pills, geometries, and labels stay consistent.
    capped = _cap_context_ids(group)
    capped_ref_ids = capped["ref_ids"]
    capped_target_ids = capped["target_ids"]

    # Membership hint: which neighboring group each context segment belongs to.
    # Best-effort, cached; skip self-references (a context id should never point
    # back at the current group, but guard anyway). Only touch the (potentially
    # tens-of-MB) sidecar when this group actually has context pills to annotate.
    membership = _load_group_membership(dataset) if (capped_ref_ids or capped_target_ids) else {}
    current_gid = group.get("group_id")

    def _member_group(sid: str) -> str | None:
        gid = membership.get(sid)
        return gid if gid and gid != current_gid else None

    context_ref_details = [
        {
            "id": rid,
            "cls": context_ref_classes.get(rid, ""),
            "name": context_ref_names.get(rid, ""),
            "member_group": _member_group(rid),
        }
        for rid in capped_ref_ids
    ]
    context_target_details = [
        {
            "id": tid,
            "cls": context_target_classes.get(tid, ""),
            "name": context_target_names.get(tid, ""),
            "member_group": _member_group(tid),
        }
        for tid in capped_target_ids
    ]
    context_capped = capped["ref_total"] > len(capped_ref_ids) or capped["target_total"] > len(
        capped_target_ids
    )

    # Edges enriched with a per-edge sliver flag for the client-side UI. The map
    # + panel use these to render coverage gaps and a live sliver-exclusion count.
    # The hybrid rule needs metric segment lengths, so the whole group (with its
    # geometries) is passed through, not just the raw edge list.
    client_edges, sliver_count = annotate_group_sliver_flags(group)

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
        "context_ref_total": capped["ref_total"],
        "context_target_total": capped["target_total"],
        "context_capped": context_capped,
        # Combined capped id list for the map's initial hidden-context set (JSON
        # blob in group.html). Keeps the DOM/map bounded for large groups.
        "context_ids": list(capped_ref_ids) + list(capped_target_ids),
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
        group_ctx = _build_group_context(deep_group, dataset=dataset)
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
    group_ctx = _build_group_context(group, dataset=dataset)

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
    group_ctx = _build_group_context(group, dataset=dataset)

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
                ref_lens, tgt_lens = group_segment_lengths_m(group)
                matched = [e for e in matched if not edge_is_sliver(e, ref_lens, tgt_lens)]

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
    group_ctx = _build_group_context(group, dataset=dataset)

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
    group_ctx = _build_group_context(group, dataset=dataset)

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
