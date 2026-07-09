"""Stitching Review mode routes for the crosswalk web UI."""

import json
import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from shapely.geometry import LineString, mapping, shape
from shapely.ops import substring, unary_union

from ...filenames import (
    PROJECT_ROOT,
    STITCH_ALL_QUEUE,
    bridge_filename,
    groups_sidecar_path,
    stitch_batch_path,
)
from ...labeling.stitching_store import LABEL_SEMANTICS_PAIR, LABEL_SEMANTICS_SET
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

# Junction "sliver" classification is centralized in crosswalk.config
# (SLIVER_SPAN_THRESHOLD / SLIVER_ABS_OVERLAP_M / is_sliver_edge) and applied to
# edge/group dicts via crosswalk.matching.sliver. See those modules for the hybrid
# fraction + absolute-meters rule and its rationale. The prior fraction-only test
# here misclassified long-ref junction overlaps (e.g. 9% of a 2 km ref = 180 m of
# real road) as slivers; the shared hybrid rule keeps those as substantive edges.


def _validate_dataset(dataset: str) -> bool:
    """Check dataset exists in the known dataset list (prevents path traversal).

    The synthetic ``__all__`` combined queue is accepted too: it is not a real
    dataset but a curated cache file whose id is a fixed sentinel (no user input
    reaches a path here), so it cannot be a traversal vector.
    """
    if not dataset:
        return False
    if dataset == STITCH_ALL_QUEUE:
        return True
    return dataset in list_datasets()


def _group_dataset(group: dict, page_dataset: str) -> str:
    """Resolve the dataset that OWNS a group.

    For a normal per-dataset queue this is just the page dataset. For the
    combined ``__all__`` queue each group carries its own ``dataset_id``, which
    is the partition its label must be written to and the sidecar its
    spatial-context membership is resolved against.
    """
    return group.get("dataset_id") or page_dataset


def _find_group(all_groups: list[dict], group_id: str, group_dataset: str = "") -> dict | None:
    """Find a group by id, disambiguated by owning dataset when provided.

    ``group_id`` is only unique WITHIN a dataset (a 32-bit content hash of the
    group's segment ids), so in the combined ``__all__`` queue two datasets can
    in principle carry the same id. When the caller knows the owning dataset
    (the forms submit it as ``group_dataset``), match on BOTH so a label can
    never resolve to a same-id group in the wrong dataset. Falls back to
    id-only match for the per-dataset queues (which submit no owning dataset).
    """
    for g in all_groups:
        if g.get("group_id") != group_id:
            continue
        if group_dataset and (g.get("dataset_id") or "") != group_dataset:
            continue
        return g
    return None


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


def _group_candidate_edges(group: dict) -> list[dict]:
    """Union of a group's selected ``edges`` and its ``rejected_edges``.

    Deduplicated by (ref_id, target_id), selected edges taking precedence on a
    clash. Both lists live in the post-#282/#284 sidecar; ``rejected_edges`` are
    the non-selected candidates over the group's own segments (the same
    structural layer, carrying ``is_sliver``). Order-stable: selected first.

    This is the candidate set a reviewer can legitimately construct a pair from
    — a pair whose only shared edge was REJECTED by the optimizer must still be
    recordable rather than silently dropped (same bug class as the #270
    context-pill trap).
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for e in (group.get("edges") or []) + (group.get("rejected_edges") or []):
        key = (e.get("ref_id"), e.get("target_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _annotate_candidate_edges(group: dict) -> list[dict]:
    """Candidate-edge union annotated with a per-edge ``is_sliver`` flag.

    Used as the client ``#group-edges`` payload in de-anchored mode so the live
    confidence / coverage-gap overlay can reason about rejected candidates the
    reviewer may pair up. Sliver classification uses the group's own segment
    lengths (hybrid rule), so it is computed fresh rather than trusting any
    baked-in flag.
    """
    ref_lens, tgt_lens = group_segment_lengths_m(group)
    out: list[dict] = []
    for e in _group_candidate_edges(group):
        ce = dict(e)
        ce["is_sliver"] = edge_is_sliver(e, ref_lens, tgt_lens)
        out.append(ce)
    return out


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


def _build_group_geojson(group: dict, deanchored: bool = False) -> dict:
    """Build GeoJSON FeatureCollection using ALL edges in the group.

    Emits two tiers of features:
    - Full geometries: thin, faded — show full extent of each segment
    - Aligned sub-segments: thick, bright — from all edges (no gaps)

    Feature _role values:
    - "ref-full" / "target-full": full segment geometries
    - "ref-aligned" / "target-aligned": aligned sub-segments from edges

    In de-anchored mode the aligned tier is built from the FULL candidate set
    (selected + rejected) so that activating any candidate pair reveals its
    overlap identically — the map must never signal which edges the optimizer
    chose. The aligned features are still filtered client-side by the active
    segments, so a blank-slate group shows no aligned overlays until the
    reviewer starts selecting.
    """
    ref_geoms = group.get("ref_geometries", {})
    target_geoms = group.get("target_geometries", {})
    edges = _group_candidate_edges(group) if deanchored else group.get("edges", [])
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


def _render_group(group: dict, dataset: str, deanchored: bool) -> tuple[dict, dict]:
    """Build the (geojson, template-context) pair for a group render.

    Centralizes the de-anchored overrides so every render path (initial page,
    deep link, HTMX fragment, post-submit/skip next-group) behaves identically:

    - ``deanchored`` is threaded into the template for the mode switch, the
      collapsed proposals, and the blank confidence readout.
    - In de-anchored mode the optimizer pre-seed is stripped to a blank slate
      (no active pills, no pre-picked option, every group segment starts hidden
      so nothing signals the proposal) and the client edge payload is widened to
      the full candidate union so live confidence can reason about rejected
      candidates the reviewer pairs up.
    """
    geojson = _build_group_geojson(group, deanchored=deanchored)
    ctx = _build_group_context(group, dataset=dataset)
    ctx["deanchored"] = deanchored
    if deanchored:
        annotated = _annotate_candidate_edges(group)
        ctx["client_edges"] = annotated
        # Keep the server-rendered sliver count consistent with the widened
        # candidate payload (the live indicator recomputes client-side anyway).
        ctx["sliver_count"] = sum(1 for e in annotated if e["is_sliver"])
        ctx["preseed_active_refs"] = []
        ctx["preseed_active_targets"] = []
        ctx["preseed_edges"] = []
        ctx["preseed_inactive_ids"] = list(group.get("ref_ids", [])) + list(
            group.get("target_ids", [])
        )
    return geojson, ctx


@router.get("/stitching-review", response_class=HTMLResponse)
async def stitching_review(
    request: Request,
    dataset: str = "",
    group_id: str = "",
    group_dataset: str = "",
    deanchored: bool = False,
):
    """Main stitching review page.

    With ``group_id``, deep-links a specific group (reviewed or not) as a full
    page — used by audit sheets and shared links. The bare
    ``/stitching-review/group`` endpoint is an HTMX fragment and renders
    without styles/map when opened directly.
    """
    # Surface the combined cross-dataset queue at the top of the switcher, but
    # only once it has actually been generated (crosswalk data stitch-batch-all).
    datasets = list_datasets()
    if stitch_batch_path(STITCH_ALL_QUEUE).exists():
        datasets = [STITCH_ALL_QUEUE, *datasets]

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
    # Position-based navigation walks ONLY the unreviewed queue (recomputed per
    # request), so a reload lands on the first unreviewed group and paging never
    # re-serves a completed one. The "N of M" counter is expressed relative to
    # that queue (position within unreviewed / total unreviewed) so it stays
    # coherent with the list actually being paged. Deep links by group_id are the
    # exception: they address the FULL batch (they intentionally target reviewed
    # groups for re-adjudication) and report batch position / batch total.
    groups = get_unreviewed_stitch_groups(dataset, all_groups)

    if group_id:
        # Deep link: render the requested group (even if already reviewed)
        # inside the full page so styles/map/JS load. Match owning dataset too so
        # a shared group_id in the combined queue resolves the right occurrence.
        deep_index = next(
            (
                i
                for i, g in enumerate(all_groups)
                if g.get("group_id") == group_id
                and (not group_dataset or (g.get("dataset_id") or "") == group_dataset)
            ),
            None,
        )
        if deep_index is None:
            logger.warning(f"Deep-link group not found in {dataset} batch: {group_id!r}")
            return HTMLResponse("Group not found in batch", status_code=404)
        deep_group = all_groups[deep_index]
        geojson, group_ctx = _render_group(
            deep_group, _group_dataset(deep_group, dataset), deanchored
        )
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
                "total_groups": 0,
                "no_groups": True,
                "all_reviewed": True,
            },
        )

    group = groups[0]
    geojson, group_ctx = _render_group(group, _group_dataset(group, dataset), deanchored)

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
            "group_index": 0,
            "total_groups": len(groups),
            "no_groups": False,
            **group_ctx,
        },
    )


@router.get("/stitching-review/group", response_class=HTMLResponse)
async def stitching_group(
    request: Request,
    dataset: str = "",
    group_id: str = "",
    group_dataset: str = "",
    group_index: int = 0,
    deanchored: bool = False,
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
    batch_total = len(all_groups)

    # Resolve the group + its display position. Two navigation modes:
    #   - Deep link by group_id: exact match over the FULL batch (may be an
    #     already-reviewed group — re-adjudication flows rely on this). Reports
    #     batch position / batch total.
    #   - Position-based paging (no/unknown group_id): index into the UNREVIEWED
    #     queue only, so next/skip never revisit a completed group. Reports the
    #     position within that queue / total unreviewed.
    group = None
    display_index = group_index
    display_total = batch_total
    if group_id:
        for i, g in enumerate(all_groups):
            if g.get("group_id") == group_id and (
                not group_dataset or (g.get("dataset_id") or "") == group_dataset
            ):
                group, display_index, display_total = g, i, batch_total
                break
    if group is None:
        groups = get_unreviewed_stitch_groups(dataset, all_groups)
        display_total = len(groups)
        if 0 <= group_index < len(groups):
            group = groups[group_index]
            display_index = group_index

    if not group:
        return templates.TemplateResponse(
            request,
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    geojson, group_ctx = _render_group(group, _group_dataset(group, dataset), deanchored)

    return templates.TemplateResponse(
        request,
        "stitching/group.html",
        {
            "request": request,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": display_index,
            "total_groups": display_total,
            **group_ctx,
        },
    )


def _parse_explicit_edges(raw: str, group: dict) -> list[dict] | None:
    """Parse and validate an explicit selected_edges payload.

    An option in the picker IS an exact edge set. For M:N groups the
    cross-product of an option's endpoints can contain MORE edges than the
    option itself, so when the client submits an unmodified option it sends the
    option's exact edge set here and we store it verbatim.

    The submit-time pair-confirmation panel also routes through this path: a
    manual / de-anchored selection the reviewer confirmed pair-by-pair is sent
    as an explicit list so unticked pairs are excluded exactly. Those pairs are
    drawn from the group's CANDIDATE union (selected edges ∪ rejected
    candidates), so validation is against that union — matching the manual
    cross-product path, which already lets a reviewer pair segments whose only
    shared edge was rejected by the optimizer (the #270 silent-drop bug class).
    Context pills are never in the candidate union, so they remain unrecordable.

    Returns:
        - None when no explicit payload was sent (caller uses cross-product).
        - A validated list of {ref_id, target_id} dicts otherwise.

    Raises:
        ValueError if the payload is malformed or references an edge that does
        not exist in the group's candidate union (guards a stale/forged submit).
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as e:
        raise ValueError("selected_edges is not valid JSON") from e
    if not isinstance(parsed, list):
        raise ValueError("selected_edges must be a list")

    group_edge_set = {(e["ref_id"], e["target_id"]) for e in _group_candidate_edges(group)}
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
    group_dataset: str = Form(""),
    group_index: int = Form(0),
    included_refs: str = Form(""),
    included_targets: str = Form(""),
    selected_edges: str = Form(""),
    exclude_slivers: str = Form(""),
    deanchored: bool = Form(False),
    confirm_reject_all: str = Form(""),
):
    """Records selection, returns next group via HTMX swap.

    Two storage paths by intent (see docs/ARCHITECTURE.md "Stitching labels"):

    * EXPLICIT option ratification — the client submits an unmodified option's
      exact edge set as ``selected_edges``. This endorses a specific listed edge
      set, so it is stored verbatim with PAIR semantics.
    * MANUAL / de-anchored membership — no ``selected_edges`` payload. The active
      ref/target pills are the reviewer's asserted group MEMBERSHIP, not a
      pair-level adjudication, so a SET label is recorded (membership in
      ref_ids/target_ids, empty selected_edges). Reject-all (both pill sets
      empty) keeps the historical empty-edge PAIR encoding.

    ``exclude_slivers`` is retained for client compatibility but no longer
    affects storage (set labels store no edges); it still drives the client-side
    confidence display and confirm-panel sliver badges.
    """
    if not _validate_dataset(dataset):
        return HTMLResponse("Unknown dataset", status_code=404)
    try:
        batch = load_stitch_batch(dataset)
    except Exception:
        return HTMLResponse("<div>Error loading batch</div>")

    if not batch:
        return HTMLResponse("<div>No batch found</div>")

    # Find the group (disambiguated by owning dataset for the combined queue).
    all_groups = batch.get("groups", [])
    group = _find_group(all_groups, group_id, group_dataset)

    if group:
        # Resolve the owning dataset ONCE and refuse to write to the synthetic
        # __all__ partition: a group that reached here without a dataset stamp
        # in the combined queue would otherwise corrupt a labels/.../dataset=
        # __all__/ partition no consumer reads. Never fires for a stamped group.
        owner_dataset = _group_dataset(group, dataset)
        if owner_dataset == STITCH_ALL_QUEUE or not _validate_dataset(owner_dataset):
            logger.error(
                "Refusing stitching label for group %s: unresolved owning dataset %r",
                group_id,
                owner_dataset,
            )
            return HTMLResponse("Unresolved group dataset", status_code=400)
        try:
            explicit_edges = _parse_explicit_edges(selected_edges, group)
        except ValueError as e:
            # Detail can embed client-supplied ids — log it, never reflect it
            logger.warning(f"Rejected selected_edges for group {group_id}: {e}")
            return HTMLResponse("Invalid selected_edges", status_code=400)

        # Storage variables resolved by the two paths below.
        label_semantics = LABEL_SEMANTICS_PAIR
        ref_members: list[str] | None = None
        target_members: list[str] | None = None
        if explicit_edges is not None:
            # Exact option edge set (option-card ratification, incl. stale) —
            # this endorses a SPECIFIC listed edge set, so it keeps PAIR
            # semantics and is stored verbatim.
            final_edges = explicit_edges
            num_refs = len({e["ref_id"] for e in final_edges})
            num_targets = len({e["target_id"] for e in final_edges})
        else:
            # Manual / de-anchored mode: the reviewer asserted only group
            # MEMBERSHIP (these refs and these targets form one matched group)
            # via the active pills — he never adjudicated individual pairings.
            # So this records a SET label (membership in ref_ids/target_ids,
            # empty selected_edges), NOT the ref×target cross-product. The
            # candidate union is still computed to run the inconsistency guard
            # (and drives the client's zero-support warning).
            candidate_edges = _group_candidate_edges(group)
            ref_set = set(r for r in included_refs.split(",") if r)
            target_set = set(t for t in included_targets.split(",") if t)
            matched = [
                e
                for e in candidate_edges
                if e["ref_id"] in ref_set and e["target_id"] in target_set
            ]

            # Guard against silently recording an empty (label-corrupting)
            # selection. Distinguish intent by the active-pill fields:
            #   - Both empty  -> deliberate reject-all; store [] normally.
            #   - Non-empty but zero candidate edges matched -> inconsistent
            #     submission (e.g. a client-side regression that drops the pill
            #     IDs, or active pills that share no edge). Refuse rather than
            #     corrupt the label. Logged without reflecting client input.
            #     The union is counted here too, so a pair whose only edge is
            #     rejected is treated as a real match, not an inconsistency.
            #
            # Checked BEFORE sliver exclusion so an all-sliver selection is not
            # misread as an inconsistent submit — it is a legitimate (if empty)
            # result once the slivers are dropped.
            if candidate_edges and not matched and (ref_set or target_set):
                logger.warning(
                    "Rejected inconsistent stitching submit for group %s: "
                    "%d refs / %d targets claimed but 0 candidate edges matched",
                    group_id,
                    len(ref_set),
                    len(target_set),
                )
                return HTMLResponse("Inconsistent selection", status_code=400)

            # De-anchored empty-submit guard: in normal mode "both pill fields
            # empty" is a deliberate deselection of the pre-seed, but in
            # de-anchored mode it is the UNTOUCHED default (blank slate), so a
            # misclick on "Select This" would silently record a reject-all label
            # into the exact eval slice this mode exists to keep clean. Require
            # an explicit confirmation flag (the client shows a confirm dialog
            # and sets it) before storing an empty de-anchored selection.
            if (
                deanchored
                and not ref_set
                and not target_set
                and confirm_reject_all.strip().lower() not in {"true", "1", "on", "yes"}
            ):
                logger.warning(
                    "Refused unconfirmed empty de-anchored submit for group %s",
                    group_id,
                )
                return HTMLResponse(
                    "Empty de-anchored selection requires confirmation", status_code=400
                )

            if ref_set or target_set:
                # SET label: store membership only. selected_edges stays empty;
                # sliver exclusion no longer applies (no pairs are stored). The
                # membership is exactly the active pills the reviewer chose.
                label_semantics = LABEL_SEMANTICS_SET
                ref_members = sorted(ref_set)
                target_members = sorted(target_set)
                final_edges = []
                num_refs = len(ref_set)
                num_targets = len(target_set)
            else:
                # Reject-all (both pill sets empty): no membership to overstate,
                # so keep the historical PAIR reject-all encoding (empty edges).
                final_edges = []
                num_refs = 0
                num_targets = 0

        record_stitching_label(
            # Route to the group's OWNING dataset partition — for the combined
            # __all__ queue this is the group's own dataset_id, not "__all__".
            dataset_id=owner_dataset,
            group_id=group_id,
            selected_edges=final_edges,
            match_type=group.get("match_type", ""),
            num_refs=num_refs,
            num_targets=num_targets,
            label_semantics=label_semantics,
            ref_ids=ref_members,
            target_ids=target_members,
            # Stamp de-anchored reviews so the eval can slice an unbiased set of
            # labels elicited without the optimizer's pre-seeded proposal. No new
            # CSV column — reuses the existing session_id provenance field.
            session_id="deanchored_v1" if deanchored else None,
        )

    # Save-and-advance: recompute the unreviewed queue AFTER recording, so the
    # just-labeled group has dropped out. Serving groups[0] (the earliest
    # remaining unreviewed group) can neither repeat the group we just recorded
    # nor skip an unreviewed one. Counter is queue-relative (first of N remaining).
    groups = get_unreviewed_stitch_groups(dataset, all_groups)

    if not groups:
        return templates.TemplateResponse(
            request,
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    group = groups[0]
    geojson, group_ctx = _render_group(group, _group_dataset(group, dataset), deanchored)

    return templates.TemplateResponse(
        request,
        "stitching/group.html",
        {
            "request": request,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": 0,
            "total_groups": len(groups),
            **group_ctx,
        },
    )


@router.post("/stitching-review/skip", response_class=HTMLResponse)
async def stitching_skip(
    request: Request,
    dataset: str = Form(...),
    group_id: str = Form(""),
    group_dataset: str = Form(""),
    deanchored: bool = Form(False),
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
    # Skip walks ONLY the unreviewed queue (recomputed per request) so it never
    # lands on a completed group. Counter is queue-relative (position within the
    # unreviewed list / total unreviewed).
    groups = get_unreviewed_stitch_groups(dataset, all_groups)

    if not groups:
        return templates.TemplateResponse(
            request,
            "stitching/no_groups.html",
            {"request": request, "dataset": dataset, "all_reviewed": True},
        )

    # Find the current group within the unreviewed queue and advance past it
    # (wrapping so skipping the last returns to the first still-unreviewed).
    # Match on owning dataset too so a shared group_id in the combined queue
    # advances from the right occurrence.
    next_index = 0
    if group_id:
        for i, g in enumerate(groups):
            if g.get("group_id") == group_id and (
                not group_dataset or (g.get("dataset_id") or "") == group_dataset
            ):
                next_index = (i + 1) % len(groups)
                break

    group = groups[next_index]
    geojson, group_ctx = _render_group(group, _group_dataset(group, dataset), deanchored)

    return templates.TemplateResponse(
        request,
        "stitching/group.html",
        {
            "request": request,
            "dataset": dataset,
            "group": group,
            "group_geojson": geojson,
            "group_index": next_index,
            "total_groups": len(groups),
            **group_ctx,
        },
    )
