"""Evidence-pack generation for agent stitching-group labeling.

For each M:N group in a stitch batch, writes a self-contained evidence pack:

    {group_dir}/
        overview.png       - full group + spatial context (geometry-only)
        option_A.png        - option A's edges highlighted
        option_B.png        - ...
        metadata.yaml       - option table + per-segment names/classes
        prompt.txt          - rubric + option table + required JSON output
        evidence.json       - exact displayed menu + content hashes

The agent (any provider) picks one option letter, or NONE if no option is
correct. Options are the deduplicated optimizer assignment + top-K
alternatives, from the shared :func:`build_stitch_options` so humans and agents
see the identical option set ("verify, don't construct"). One panel-only
exception: on monster groups (thousands of distinct candidate edges) the
near-duplicate perturbation options are pruned to a small, maximally-distinct
subset (:func:`prune_options_for_panel`) before rendering — the web review UI
keeps the full set.
"""

from __future__ import annotations

import math
import string
from pathlib import Path

import yaml
from loguru import logger
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import shape as shape_from_geojson
from shapely.ops import nearest_points

from ..config import SLIVER_ABS_OVERLAP_M, SLIVER_SPAN_THRESHOLD, settings
from ..matching.sliver import (
    edge_overlap_m,
    edge_sliver_tag,
    group_segment_lengths_m,
)
from ..matching.stitch_options import build_stitch_options
from ..utils.physical import summarize_physical
from .image_renderer import (
    BACKGROUND_COLOR,
    MIN_IMAGE_SIZE,
    REFERENCE_COLOR,
    TARGET_COLOR,
    _calculate_size_from_bbox,
    _draw_dashed_linestring,
    _draw_linestring,
    _expand_bbox,
    _geo_to_pixel,
    _make_bbox_square,
    _to_linestring,
)
from .matching_rubric import (
    CANONICAL_RULES_DOC,
    MATCH_IDENTITY_RUBRIC,
    STITCH_ASSIGNMENT_RUBRIC,
)
from .stitch_provenance import build_evidence_record, safe_group_id, write_evidence_manifest

# Styling
CONTEXT_COLOR = (190, 190, 190)  # light gray context roads
FADED_REF_COLOR = (170, 200, 235)  # de-emphasized (not-in-option) reference
FADED_TARGET_COLOR = (240, 190, 190)  # de-emphasized (not-in-option) target
LABEL_COLOR = (30, 30, 30)
OPTION_LINE_WIDTH = 4
GROUP_LINE_WIDTH = 3
CONTEXT_WIDTH = 1

# Stitch groups can span whole chains (1-2km) now that group geometry is never
# clipped to a 500m box. The generic 512px render cap would squash a long chain
# to ~4m/px; allow a larger canvas so large groups stay legible. Small groups
# are unaffected (size scales with extent, clamped to this max).
STITCH_MAX_IMAGE_SIZE = 1280

# Junction zoom crops for SLIVER/BORDERLINE edges. Panels repeatedly asked for a
# close-up at the contested junction (55 feedback items). Each crop is a small
# fixed-size close-up centred on the edge's aligned-overlap midpoint. Capped per
# pack so the pack stays in the measured ~500-620 KB / ~11-PNG envelope: a 256px
# crop is ~5-20 KB, so <= 6 crops adds at most ~120 KB on the hardest groups.
ZOOM_BOX_M = 60.0  # crop side length in meters (~30 m radius around the junction)
ZOOM_IMAGE_SIZE = 256
MAX_ZOOM_CROPS = 6

# Per-edge #267 structural fields surfaced in metadata/prompt. Kept in sync with
# crosswalk.matching.stitch_options._STRUCT_KEYS (the pass-through source).
_STRUCT_KEYS = (
    "degree_ref",
    "degree_tgt",
    "candidate_graph_bridge",
    "biconnected_block",
    "corridor_ref",
    "corridor_tgt",
    "selected",
    "decision",
    "review_reason",
    "optimizer_decision",
    "decision_reason",
    "pruned",
    "selected_elsewhere",
    "ref_physical",
    "target_physical",
)

# Group-level #267 structural summary fields (surfaced compactly, missing omitted).
_GROUP_STRUCT_KEYS = (
    "n_corridors",
    "n_assignment_components",
    "largest_biconnected_block",
    "oversized_group",
)


def _iter_lines(geojson_map: dict) -> list[tuple[str, LineString]]:
    """Convert a {id: geojson} map to a list of (id, LineString)."""
    out: list[tuple[str, LineString]] = []
    for sid, gj in (geojson_map or {}).items():
        try:
            geom = shape_from_geojson(gj)
        except Exception:
            continue
        line = _to_linestring(geom)
        if line is not None and not line.is_empty:
            out.append((str(sid), line))
    return out


def _group_bbox(
    group: dict, include_context: bool = True, padding_ratio: float = 0.15
) -> tuple[float, float, float, float] | None:
    """Compute a square bbox covering the group (and optional context)."""
    lines: list[LineString] = []
    lines += [ln for _, ln in _iter_lines(group.get("ref_geometries", {}))]
    lines += [ln for _, ln in _iter_lines(group.get("target_geometries", {}))]
    if include_context:
        lines += [ln for _, ln in _iter_lines(group.get("context_ref_geometries", {}))]
        lines += [ln for _, ln in _iter_lines(group.get("context_target_geometries", {}))]
    if not lines:
        return None
    union = MultiLineString(lines) if len(lines) > 1 else lines[0]
    bbox = _expand_bbox(union.bounds, padding_ratio)
    return _make_bbox_square(bbox)


def _draw_label(draw: ImageDraw.ImageDraw, line: LineString, bbox, size, text: str) -> None:
    """Draw a small text label near a line's midpoint."""
    if not text:
        return
    try:
        mid = line.interpolate(0.5, normalized=True)
        px, py = _geo_to_pixel(mid.x, mid.y, bbox, size)
        draw.text((px + 3, py - 6), text, fill=LABEL_COLOR)
    except Exception:
        pass


def _base_image(size) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", size, BACKGROUND_COLOR)
    return img, ImageDraw.Draw(img)


def _draw_context(draw, group, bbox, size) -> None:
    for _, ln in _iter_lines(group.get("context_ref_geometries", {})):
        _draw_dashed_linestring(draw, ln, bbox, size, CONTEXT_COLOR, CONTEXT_WIDTH, (4, 4))
    for _, ln in _iter_lines(group.get("context_target_geometries", {})):
        _draw_dashed_linestring(draw, ln, bbox, size, CONTEXT_COLOR, CONTEXT_WIDTH, (4, 4))


def _seg_labels(group: dict) -> tuple[dict[str, str], dict[str, str]]:
    ref_ids = group.get("ref_ids", list(group.get("ref_geometries", {}).keys()))
    target_ids = group.get("target_ids", list(group.get("target_geometries", {}).keys()))
    ref_labels = {rid: f"R{i + 1}" for i, rid in enumerate(ref_ids)}
    target_labels = {tid: f"T{i + 1}" for i, tid in enumerate(target_ids)}
    return ref_labels, target_labels


def render_group_overview(group: dict, size: tuple[int, int] | None = None) -> Image.Image:
    """Render the whole group (all candidate segments) + spatial context.

    Reference segments blue, target segments red, context light-gray dashed.
    Each group segment is annotated with its R#/T# label.
    """
    bbox = _group_bbox(group, include_context=True)
    if bbox is None:
        return Image.new("RGB", (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), BACKGROUND_COLOR)
    if size is None:
        size = _calculate_size_from_bbox(bbox, max_size=STITCH_MAX_IMAGE_SIZE)

    ref_labels, target_labels = _seg_labels(group)
    img, draw = _base_image(size)
    _draw_context(draw, group, bbox, size)

    for sid, ln in _iter_lines(group.get("ref_geometries", {})):
        _draw_linestring(
            draw,
            ln,
            bbox,
            size,
            REFERENCE_COLOR,
            GROUP_LINE_WIDTH,
            decoration="circle",
            decoration_spacing=30,
        )
        _draw_label(draw, ln, bbox, size, ref_labels.get(sid, ""))
    for sid, ln in _iter_lines(group.get("target_geometries", {})):
        _draw_linestring(
            draw,
            ln,
            bbox,
            size,
            TARGET_COLOR,
            GROUP_LINE_WIDTH,
            decoration="circle",
            decoration_spacing=42,
        )
        _draw_label(draw, ln, bbox, size, target_labels.get(sid, ""))
    return img


def render_option(group: dict, option: dict, size: tuple[int, int] | None = None) -> Image.Image:
    """Render one option: its edges' segments highlighted, others de-emphasized.

    Segments participating in the option are drawn bright/solid (blue ref, red
    target); other group segments are drawn faded; context is gray dashed.
    """
    bbox = _group_bbox(group, include_context=True)
    if bbox is None:
        return Image.new("RGB", (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), BACKGROUND_COLOR)
    if size is None:
        size = _calculate_size_from_bbox(bbox, max_size=STITCH_MAX_IMAGE_SIZE)

    active_refs = set(option.get("active_refs", []))
    active_targets = set(option.get("active_targets", []))
    ref_labels, target_labels = _seg_labels(group)

    img, draw = _base_image(size)
    _draw_context(draw, group, bbox, size)

    # Faded (not-in-option) group segments first, so active ones draw on top.
    for sid, ln in _iter_lines(group.get("ref_geometries", {})):
        if sid not in active_refs:
            _draw_dashed_linestring(draw, ln, bbox, size, FADED_REF_COLOR, 2, (8, 6))
    for sid, ln in _iter_lines(group.get("target_geometries", {})):
        if sid not in active_targets:
            _draw_dashed_linestring(draw, ln, bbox, size, FADED_TARGET_COLOR, 2, (8, 6))

    for sid, ln in _iter_lines(group.get("ref_geometries", {})):
        if sid in active_refs:
            _draw_linestring(
                draw,
                ln,
                bbox,
                size,
                REFERENCE_COLOR,
                OPTION_LINE_WIDTH,
                decoration="circle",
                decoration_spacing=30,
            )
            _draw_label(draw, ln, bbox, size, ref_labels.get(sid, ""))
    for sid, ln in _iter_lines(group.get("target_geometries", {})):
        if sid in active_targets:
            _draw_linestring(
                draw,
                ln,
                bbox,
                size,
                TARGET_COLOR,
                OPTION_LINE_WIDTH,
                decoration="circle",
                decoration_spacing=42,
            )
            _draw_label(draw, ln, bbox, size, target_labels.get(sid, ""))
    return img


def _edge_align_fracs(edge: dict) -> dict:
    """Extract alignment fractions (ref/target aligned span) from an edge."""
    out = {}
    if "gers_start_frac" in edge and "gers_end_frac" in edge:
        out["ref_aligned_frac"] = round(abs(edge["gers_end_frac"] - edge["gers_start_frac"]), 3)
    if "local_start_frac" in edge and "local_end_frac" in edge:
        out["target_aligned_frac"] = round(
            abs(edge["local_end_frac"] - edge["local_start_frac"]), 3
        )
    return out


def prune_options_for_panel(
    options_ctx: dict,
    group: dict,
    *,
    max_options: int | None = None,
    min_distinct_edges_trigger: int | None = None,
) -> dict | None:
    """Prune a monster group's option set to a small, maximally-distinct subset.

    On large M:N groups the greedy-perturbation alternatives are ~20
    near-duplicate variations of one assignment (each differing by only a few
    edges), which wastes panel attention and prompt bytes. This prunes the
    options IN PLACE (evidence-pack path only — the web review UI keeps the full
    one-click set) to at most ``max_options`` picked for max-min edge-set
    diversity, then re-letters the survivors A, B, C... so metadata and prompt
    stay consistent (vote parsing is driven by metadata letters).

    No-op (returns ``None``, mutates nothing) when the group's options span at
    most ``min_distinct_edges_trigger`` distinct candidate edges or there are at
    most ``max_options`` options — small groups keep byte-identical packs.

    Always kept (never pruned, even beyond ``max_options``):
      * the optimizer's proposed option (``is_optimizer``), and
      * the whole-group seed options from ``generate_top_k_alternatives``
        (the seeds' ``is_seed`` flag does not survive ``build_stitch_options``,
        so they are re-identified by edge set: the full candidate set, and the
        optimizer-selected set from the group edges' ``selected`` flags).

    Remaining slots are filled greedily with the option whose edge set has the
    maximum minimum symmetric difference to every already-kept option
    (max-min diversity), tie-broken by higher ``total_confidence`` then original
    option order, so the result is deterministic.

    If the pruned-away options contained the panel's "right answer", NONE
    remains a valid vote and routes the group to human review.

    Returns a provenance dict ``{n_before, n_after, dropped_keys}`` (recorded in
    metadata as ``options_pruned``) when pruning ran, else ``None``.
    """
    if max_options is None:
        max_options = settings.stitch_panel_max_options
    if min_distinct_edges_trigger is None:
        min_distinct_edges_trigger = settings.stitch_panel_prune_min_distinct_edges

    options = options_ctx["options"]
    if len(options) <= max_options:
        return None
    keys = [frozenset((e["ref_id"], e["target_id"]) for e in o["edges"]) for o in options]
    distinct_edges: set[tuple] = set().union(*keys) if keys else set()
    if len(distinct_edges) <= min_distinct_edges_trigger:
        return None

    # Protected options: optimizer proposal + whole-group seeds (re-identified
    # by edge set; ``selected`` flags are absent on older sidecars, in which
    # case only the full-set seed is identifiable).
    group_edges = group.get("edges", []) or []
    full_set = frozenset((e["ref_id"], e["target_id"]) for e in group_edges)
    selected_set = frozenset(
        (e["ref_id"], e["target_id"]) for e in group_edges if e.get("selected")
    )
    kept: list[int] = []
    for i, opt in enumerate(options):
        if opt["is_optimizer"] or keys[i] == full_set or (selected_set and keys[i] == selected_set):
            kept.append(i)

    remaining = [i for i in range(len(options)) if i not in kept]
    if not kept and remaining:
        # No protected option at all: seed the greedy walk with the
        # highest-confidence option (deterministic tie-break by original order).
        start = max(remaining, key=lambda i: (options[i]["total_confidence"], -i))
        kept.append(start)
        remaining.remove(start)

    # Greedy max-min diversity fill.
    while len(kept) < max_options and remaining:
        best = max(
            remaining,
            key=lambda i: (
                min(len(keys[i] ^ keys[j]) for j in kept),
                options[i]["total_confidence"],
                -i,
            ),
        )
        kept.append(best)
        remaining.remove(best)

    kept.sort()  # survivors keep their original relative order
    dropped_keys = [options[i]["key"] for i in range(len(options)) if i not in kept]
    survivors = [options[i] for i in kept]

    # Re-letter A, B, C... (same scheme as build_stitch_options) and refresh the
    # optimizer letter so metadata, prompt, and vote parsing agree.
    letters = list(string.ascii_uppercase)
    optimizer_letter = None
    for i, opt in enumerate(survivors):
        letter = letters[i] if i < len(letters) else f"O{i}"
        opt["letter"] = letter
        if opt["is_optimizer"]:
            optimizer_letter = letter

    options_ctx["options"] = survivors
    options_ctx["optimizer_letter"] = optimizer_letter
    return {
        "n_before": len(keys),
        "n_after": len(survivors),
        "dropped_keys": dropped_keys,
    }


def build_metadata(group: dict, options_ctx: dict, *, evidence: dict | None = None) -> dict:
    """Build the metadata dict describing the group and its options."""
    ref_ids = group.get("ref_ids", list(group.get("ref_geometries", {}).keys()))
    target_ids = group.get("target_ids", list(group.get("target_geometries", {}).keys()))
    ref_labels = {rid: f"R{i + 1}" for i, rid in enumerate(ref_ids)}
    target_labels = {tid: f"T{i + 1}" for i, tid in enumerate(target_ids)}
    ref_names = group.get("ref_names", {})
    target_names = group.get("target_names", {})
    ref_classes = group.get("ref_classes", {})
    target_classes = group.get("target_classes", {})
    ref_physical = group.get("ref_physical", {})
    target_physical = group.get("target_physical", {})

    # Per-edge junction-sliver flags (hybrid fraction + absolute-meters rule).
    # Slivers are ANNOTATED here, never silently dropped: an option may legitimately
    # exclude one, and the agent should be able to see which edges are artifacts.
    ref_lens, tgt_lens = group_segment_lengths_m(group)

    options_meta = []
    for opt in options_ctx["options"]:
        edges_meta = []
        sliver_edge_count = 0
        borderline_edge_count = 0
        for e in opt["edges"]:
            rid, tid = e["ref_id"], e["target_id"]
            tag = edge_sliver_tag(e, ref_lens, tgt_lens)
            is_sliver = tag == "SLIVER"
            sliver_edge_count += int(is_sliver)
            borderline_edge_count += int(tag == "BORDERLINE")
            row = {
                "edge": f"{ref_labels.get(rid, rid)}->{target_labels.get(tid, tid)}",
                "ref": ref_labels.get(rid, rid),
                "target": target_labels.get(tid, tid),
                "ref_id": str(rid),
                "target_id": str(tid),
                "confidence": round(float(e.get("confidence", 0.0)), 3),
                "is_sliver": is_sliver,
            }
            row.update(_edge_align_fracs(e))
            # Absolute overlap in meters (same arithmetic as the sliver rule's
            # absolute gate). Omit when unmeasurable (+inf from a missing length).
            overlap = edge_overlap_m(e, ref_lens, tgt_lens)
            if math.isfinite(overlap):
                row["overlap_m"] = round(overlap, 1)
            if tag is not None:
                row["tag"] = tag
            # Pass through #267 structural fields present on the enriched edge.
            for sk in _STRUCT_KEYS:
                if sk in e:
                    row[sk] = e[sk]
            edges_meta.append(row)
        options_meta.append(
            {
                "letter": opt["letter"],
                "is_optimizer": opt["is_optimizer"],
                "edge_count": opt["edge_count"],
                "sliver_edge_count": sliver_edge_count,
                "borderline_edge_count": borderline_edge_count,
                "total_confidence": round(float(opt["total_confidence"]), 3),
                "mean_confidence": round(float(opt["mean_confidence"]), 3),
                "edges": edges_meta,
            }
        )

    # Group-level #267 structural summary (present keys only; degrades gracefully).
    group_structure = {k: group[k] for k in _GROUP_STRUCT_KEYS if k in group}

    # Decomposition provenance (#367 Mode B): a sub-problem pack records its
    # parent group so votes.csv/consensus.csv rows (keyed by the sub-problem
    # group_id) trace back to the decomposed group without external state.
    decomposition_meta = {}
    if group.get("parent_group_id"):
        decomposition_meta["parent_group_id"] = group["parent_group_id"]
        if group.get("n_subproblems"):
            decomposition_meta["n_subproblems"] = group["n_subproblems"]

    metadata = {
        "group_id": group.get("group_id"),
        **decomposition_meta,
        "match_type": group.get("match_type"),
        "n_ref_segments": len(ref_ids),
        "n_target_segments": len(target_ids),
        # Audit: full vs rendered edge counts. Post-fix these are always equal
        # (group data is never clipped); a divergence flags a clipping regression.
        "n_edges_full": group.get("n_edges_full", len(group.get("edges", []))),
        "n_edges_rendered": group.get("n_edges_rendered", len(group.get("edges", []))),
        "context_clipped": bool(group.get("context_clipped", False)),
        "optimizer_letter": options_ctx.get("optimizer_letter"),
        "structure": group_structure,
        "segments": {
            "reference": [
                {
                    "label": ref_labels[rid],
                    "id": rid,
                    "name": ref_names.get(rid, ""),
                    "class": ref_classes.get(rid, ""),
                    "physical": ref_physical.get(rid, {}),
                }
                for rid in ref_ids
            ],
            "target": [
                {
                    "label": target_labels[tid],
                    "id": tid,
                    "name": target_names.get(tid, ""),
                    "class": target_classes.get(tid, ""),
                    "physical": target_physical.get(tid, {}),
                }
                for tid in target_ids
            ],
        },
        "options": options_meta,
    }
    if evidence is not None:
        metadata["evidence"] = evidence
    return metadata


def _edge_struct_str(e: dict) -> str:
    """Compact per-edge graph + aligned physical evidence line.

    Only includes fields present on the edge (older sidecars omit some/all).
    """
    parts: list[str] = []
    dr, dt = e.get("degree_ref"), e.get("degree_tgt")
    if dr is not None or dt is not None:
        parts.append(f"deg R{dr if dr is not None else '?'}/T{dt if dt is not None else '?'}")
    if e.get("candidate_graph_bridge") or (
        "candidate_graph_bridge" not in e and e.get("is_bridge")
    ):
        parts.append("candidate-graph cut edge")
    cr, ct = e.get("corridor_ref"), e.get("corridor_tgt")
    if cr is not None or ct is not None:
        parts.append(f"corr R{cr if cr is not None else '?'}/T{ct if ct is not None else '?'}")
    ref_physical = summarize_physical(e.get("ref_physical"))
    target_physical = summarize_physical(e.get("target_physical"))
    if ref_physical:
        parts.append(f"R physical: {ref_physical}")
    if target_physical:
        parts.append(f"T physical: {target_physical}")
    return ", ".join(parts)


def _group_struct_str(structure: dict | None) -> str:
    """One-line group-level structural summary; empty when no fields present."""
    if not structure:
        return ""
    parts: list[str] = []
    if "n_corridors" in structure:
        parts.append(f"{structure['n_corridors']} corridors")
    if "n_assignment_components" in structure:
        parts.append(f"{structure['n_assignment_components']} assignment-components")
    if "largest_biconnected_block" in structure:
        parts.append(f"largest biconnected block {structure['largest_biconnected_block']} edges")
    if structure.get("oversized_group"):
        parts.append("OVERSIZED")
    return ("Group structure: " + ", ".join(parts)) if parts else ""


def _edge_junction_point(edge: dict, ref_line: LineString | None, tgt_line: LineString | None):
    """Best-effort junction point for an edge: midpoint of the aligned overlap.

    Prefers the ref-side aligned subline midpoint, then the target-side, then the
    midpoint between the two lines' closest points, then a line midpoint.
    """
    rs, re_ = edge.get("gers_start_frac"), edge.get("gers_end_frac")
    if ref_line is not None and rs is not None and re_ is not None:
        try:
            return ref_line.interpolate((float(rs) + float(re_)) / 2.0, normalized=True)
        except Exception:
            pass
    ts, te = edge.get("local_start_frac"), edge.get("local_end_frac")
    if tgt_line is not None and ts is not None and te is not None:
        try:
            return tgt_line.interpolate((float(ts) + float(te)) / 2.0, normalized=True)
        except Exception:
            pass
    if ref_line is not None and tgt_line is not None:
        try:
            p1, p2 = nearest_points(ref_line, tgt_line)
            return LineString([p1, p2]).interpolate(0.5, normalized=True)
        except Exception:
            pass
    for ln in (ref_line, tgt_line):
        if ln is not None:
            try:
                return ln.interpolate(0.5, normalized=True)
            except Exception:
                pass
    return None


def render_junction_zoom(
    group: dict,
    edge: dict,
    ref_line: LineString | None,
    tgt_line: LineString | None,
    ref_labels: dict[str, str],
    target_labels: dict[str, str],
) -> Image.Image | None:
    """Render a small close-up crop centred on a flagged edge's junction.

    Both edge segments are drawn bright/solid (blue ref, red target); other group
    segments are faded and nearby context roads gray-dashed, so the panel can see
    exactly how the two segments meet at the contested junction. Returns None when
    no junction point can be located.
    """
    center = _edge_junction_point(edge, ref_line, tgt_line)
    if center is None:
        return None
    lat = center.y
    half = ZOOM_BOX_M / 2.0
    half_lat = half / 111000.0
    half_lon = half / (111000.0 * max(math.cos(math.radians(lat)), 1e-6))
    bbox = _make_bbox_square(
        (center.x - half_lon, lat - half_lat, center.x + half_lon, lat + half_lat)
    )
    size = (ZOOM_IMAGE_SIZE, ZOOM_IMAGE_SIZE)

    img, draw = _base_image(size)
    _draw_context(draw, group, bbox, size)

    # Faded other group segments first, so the edge's two segments draw on top.
    active = {str(edge.get("ref_id")), str(edge.get("target_id"))}
    for sid, ln in _iter_lines(group.get("ref_geometries", {})):
        if sid not in active:
            _draw_dashed_linestring(draw, ln, bbox, size, FADED_REF_COLOR, 1, (6, 5))
    for sid, ln in _iter_lines(group.get("target_geometries", {})):
        if sid not in active:
            _draw_dashed_linestring(draw, ln, bbox, size, FADED_TARGET_COLOR, 1, (6, 5))

    if ref_line is not None:
        _draw_linestring(
            draw,
            ref_line,
            bbox,
            size,
            REFERENCE_COLOR,
            OPTION_LINE_WIDTH,
            decoration="circle",
            decoration_spacing=24,
        )
        _draw_label(draw, ref_line, bbox, size, ref_labels.get(edge.get("ref_id"), ""))
    if tgt_line is not None:
        _draw_linestring(
            draw,
            tgt_line,
            bbox,
            size,
            TARGET_COLOR,
            OPTION_LINE_WIDTH,
            decoration="circle",
            decoration_spacing=30,
        )
        _draw_label(draw, tgt_line, bbox, size, target_labels.get(edge.get("target_id"), ""))

    # Mark the junction center itself.
    px, py = _geo_to_pixel(center.x, center.y, bbox, size)
    draw.ellipse([(px - 4, py - 4), (px + 4, py + 4)], outline=LABEL_COLOR)
    return img


def _render_junction_zooms(
    group: dict, group_dir: Path, allowed_keys: set[tuple] | None = None
) -> dict[str, str]:
    """Render junction crops for SLIVER/BORDERLINE edges; return {edge_label: file}.

    Dedupes to one crop per (ref, target) pair, prioritizes SLIVER over BORDERLINE
    then smallest overlap, and caps at ``MAX_ZOOM_CROPS`` to bound pack size.
    When ``allowed_keys`` is given, only edges in that set are considered — the
    caller passes the union of option edge sets, so crops are never rendered for
    edges no option displays (which would be unreferenced files in the pack).
    """
    ref_lens, tgt_lens = group_segment_lengths_m(group)
    ref_labels, target_labels = _seg_labels(group)
    ref_lines = dict(_iter_lines(group.get("ref_geometries", {})))
    tgt_lines = dict(_iter_lines(group.get("target_geometries", {})))

    flagged: list[tuple[int, float, dict, tuple]] = []
    seen: set[tuple] = set()
    for e in group.get("edges", []) or []:
        key = (e.get("ref_id"), e.get("target_id"))
        if key in seen:
            continue
        seen.add(key)
        if allowed_keys is not None and key not in allowed_keys:
            continue
        tag = edge_sliver_tag(e, ref_lens, tgt_lens)
        if tag is None:
            continue
        priority = 0 if tag == "SLIVER" else 1
        flagged.append((priority, edge_overlap_m(e, ref_lens, tgt_lens), e, key))

    flagged.sort(key=lambda x: (x[0], x[1]))

    out: dict[str, str] = {}
    for _prio, _ov, e, (rid, tid) in flagged[:MAX_ZOOM_CROPS]:
        rl = ref_lines.get(str(rid))
        tl = tgt_lines.get(str(tid))
        img = render_junction_zoom(group, e, rl, tl, ref_labels, target_labels)
        if img is None:
            continue
        rlab = ref_labels.get(rid, str(rid))
        tlab = target_labels.get(tid, str(tid))
        fname = f"zoom_{rlab}_{tlab}.png"
        img.save(group_dir / fname)
        out[f"{rlab}->{tlab}"] = fname
    return out


def build_prompt(group_dir: Path, metadata: dict, options_ctx: dict) -> str:
    """Build the option-picker prompt referencing absolute image paths."""
    group_dir = group_dir.resolve()
    letters = [o["letter"] for o in options_ctx["options"]]
    choices = "|".join(letters + ["NONE"])
    opt_letter = metadata.get("optimizer_letter")

    lines: list[str] = []
    lines.append(
        "You are curating ground truth for a road-network conflation optimizer.\n"
        "A 'stitching group' is a cluster of candidate matches between REFERENCE\n"
        "transportation segments (blue, labeled R1, R2, ...) and TARGET segments (red,\n"
        "labeled T1, T2, ...). Your job: determine the exact final accepted edge set, then\n"
        "pick the ONE assignment option that contains all and only those edges."
    )
    lines.append("")
    lines.append("HOW TO READ THE IMAGES:")
    lines.append(f"- overview: {group_dir / 'overview.png'}")
    lines.append(
        "  Shows every candidate segment in the group (blue=reference, red=target) plus\n"
        "  light-gray dashed nearby roads for spatial context. Labels R#/T# identify segments."
    )
    lines.append("- One image PER OPTION below. In an option image, the segments included in")
    lines.append("  that option are drawn bright/solid; excluded group segments are faded/dashed.")
    lines.append(
        "  Use these images to judge the exact edge set; do not choose the closest picture."
    )
    lines.append("- Some edges carry a 'junction zoom' image: a close-up centred on where")
    lines.append("  those two segments meet, so you can see the actual overlap at the junction.")
    lines.append("- Every candidate edge is described ONCE in the EDGES legend below, each given a")
    lines.append("  short id (e1, e2, ...) alongside its R#->T# endpoints. Each OPTION then lists")
    lines.append("  only the short ids of the edges it selects — look up each id in the legend for")
    lines.append("  its R#->T# endpoints and full detail. (An option's edge set is exactly the")
    lines.append("  edges whose ids it lists.)")
    lines.append("")
    lines.append("GUIDANCE:")
    lines.append(f"Canonical source: {CANONICAL_RULES_DOC}")
    lines.append("")
    lines.append(MATCH_IDENTITY_RUBRIC)
    lines.append("")
    lines.append(STITCH_ASSIGNMENT_RUBRIC)
    lines.append("")
    lines.append("EVIDENCE-SPECIFIC NOTES:")
    lines.append("- A SLIVER tag is a geometry-derived warning, not an identity verdict. It means")
    lines.append(
        "  both aligned spans cover less than "
        f"{SLIVER_SPAN_THRESHOLD:.0%} of their segments and the absolute overlap is under "
        f"{SLIVER_ABS_OVERLAP_M:g}m."
    )
    lines.append("  Exclude an endpoint clip or different continuation, but do not reject a")
    lines.append("  same-direction, same-role, corridor-supported edge solely because of the tag.")
    lines.append("- 'overlap~Xm' on an edge is the absolute length the two segments physically")
    lines.append("  share (aligned span x segment length). Small overlap is evidence to inspect,")
    lines.append("  not proof that the segments merely touch or that the edge is wrong.")
    lines.append("- An edge tagged BORDERLINE covers only a small fraction of BOTH its segments")
    lines.append("  but does not meet the strict SLIVER rule. BORDERLINE is display-only and")
    lines.append("  deliberately neutral; judge identity and resolution from the full context.")
    lines.append("- Edges may carry neutral structural context from the road graph (these are")
    lines.append("  facts, not verdicts, and favor neither including nor excluding an edge):")
    lines.append("  'deg R#/T#' is how many road segments meet at that edge's ref/target endpoint")
    lines.append("  (a high degree is a busy junction; degree ~2 is a simple continuation);")
    lines.append("  'candidate-graph cut edge' means removing that candidate would split the")
    lines.append("  bipartite candidate graph. It is graph theory, NOT a claim that either road")
    lines.append("  is a physical bridge. 'corr R#' compares reference segments with references;")
    lines.append("  'corr T#' compares targets with targets. R0 and T0 are independent labels and")
    lines.append("  do not assert cross-side identity. Use corridor context only after judging")
    lines.append("  whether the two segments represent the same traveled way.")
    lines.append("- 'R physical' / 'T physical' reports bridge, tunnel, and vertical layer rules")
    lines.append("  clipped to that edge's own aligned fractions. Segment details retain the full")
    lines.append("  linear-referenced rules. Missing physical evidence means unknown, not ground;")
    lines.append("  road flags are positive observations, so an absent flag is not proof that the")
    lines.append("  provider surveyed that attribute.")
    lines.append("- The optimizer's own proposed option is labeled below; it is often but not")
    lines.append(
        "  always correct. Judge from the geometry, not from which one is the optimizer's."
    )
    lines.append("")
    lines.append(
        f"GROUP {metadata['group_id']}  (match_type={metadata['match_type']}, "
        f"{metadata['n_ref_segments']} ref x {metadata['n_target_segments']} target)"
    )
    if metadata.get("parent_group_id"):
        lines.append(
            f"This group is one sub-problem of a larger group ({metadata['parent_group_id']}); "
            "some nearby gray context roads belong to sibling sub-problems. Judge ONLY "
            "the labeled segments and edges shown here."
        )
    if opt_letter:
        lines.append(f"Optimizer's proposed option: {opt_letter}")
    struct_summary = _group_struct_str(metadata.get("structure"))
    if struct_summary:
        lines.append(struct_summary)
    lines.append("")
    # EDGES legend: describe every DISTINCT candidate edge exactly once, each given a
    # short id (e1, e2, ... in first-seen order) alongside its R#->T# endpoints.
    # Options below reference edges by short id only, so an edge that appears in many
    # overlapping options is no longer re-printed in longhand each time (that
    # duplication made large-group prompts several times bigger than necessary: the
    # worst Boston group re-printed 4,183 distinct edges as ~17k descriptor lines).
    lines.append(
        "EDGES (each candidate edge described once, keyed by a short id; "
        "options reference these ids):"
    )
    edge_ids: dict[str, str] = {}
    for opt in metadata["options"]:
        for e in opt["edges"]:
            label = e["edge"]
            if label in edge_ids:
                continue
            eid = f"e{len(edge_ids) + 1}"
            edge_ids[label] = eid
            extra = []
            if "ref_aligned_frac" in e:
                extra.append(f"ref_aln={e['ref_aligned_frac']}")
            if "target_aligned_frac" in e:
                extra.append(f"tgt_aln={e['target_aligned_frac']}")
            if e.get("overlap_m") is not None:
                extra.append(f"overlap~{e['overlap_m']}m")
            etag = e.get("tag")
            if etag == "SLIVER":
                extra.append("SLIVER(low-span/low-absolute-overlap warning)")
            elif etag == "BORDERLINE":
                # Fraction-based display band: on long segments the absolute
                # overlap~Xm printed alongside can still be large.
                extra.append("BORDERLINE(low span fraction, display-only)")
            struct_s = _edge_struct_str(e)
            if struct_s:
                extra.append(f"[{struct_s}]")
            extra_s = ("  " + " ".join(extra)) if extra else ""
            lines.append(f"  {eid}: {label}  conf={e['confidence']}{extra_s}")
            if e.get("zoom"):
                lines.append(f"      junction zoom: {group_dir / e['zoom']}")
    lines.append("")
    lines.append("OPTIONS:")
    for opt in metadata["options"]:
        tag = " (optimizer)" if opt["is_optimizer"] else ""
        img_path = group_dir / f"option_{opt['letter']}.png"
        lines.append(
            f"  Option {opt['letter']}{tag}: {opt['edge_count']} edges, "
            f"total_conf={opt['total_confidence']}, mean_conf={opt['mean_confidence']}"
        )
        lines.append(f"    image: {img_path}")
        edge_refs = [edge_ids[e["edge"]] for e in opt["edges"]]
        if edge_refs:
            lines.append(f"    edges: {', '.join(edge_refs)}")
        else:
            lines.append("    edges: (none)")
    lines.append("")
    lines.append("SEGMENTS (name / class / segment-wide physical evidence):")
    for s in metadata["segments"]["reference"]:
        physical = summarize_physical(s.get("physical")) or "unknown"
        lines.append(
            f"  {s['label']}: name='{s['name']}' class='{s['class']}' physical='{physical}'"
        )
    for s in metadata["segments"]["target"]:
        physical = summarize_physical(s.get("physical")) or "unknown"
        lines.append(
            f"  {s['label']}: name='{s['name']}' class='{s['class']}' physical='{physical}'"
        )
    lines.append("")
    lines.append("Look at overview.png first, then each option image. Then respond with ONLY a")
    lines.append("single JSON object (no prose, no markdown fence) of the form:")
    lines.append(f'  {{"choice": "<{choices}>", "confidence": 0.0-1.0, "reasoning": "..."}}')
    lines.append('"choice" MUST be exactly one of the option letters above, or "NONE".')
    return "\n".join(lines)


def generate_group_evidence(
    group: dict,
    group_dir: Path,
    *,
    source_artifacts: dict | None = None,
    batch_generation_source: dict | None = None,
) -> dict | None:
    """Generate the full evidence pack for one group. Returns the metadata dict.

    Returns None if the group has no options.
    """
    safe_group_id(group.get("group_id"))
    group_dir = Path(group_dir)

    # Clear the files a voter can consume before deciding whether the refreshed
    # group still has a menu.  Otherwise options -> no-options regeneration
    # leaves an old prompt/images behind and the runner can vote stale evidence.
    if group_dir.exists():
        for pattern in (
            "overview.png",
            "option_*.png",
            "zoom_*.png",
            "metadata.yaml",
            "prompt.txt",
            "evidence.json",
        ):
            for old in group_dir.glob(pattern):
                if old.is_file():
                    old.unlink()

    options_ctx = build_stitch_options(group)
    if not options_ctx["options"]:
        logger.warning(f"Group {group.get('group_id')}: no options, skipping")
        return None

    # Panel-only diversity pruning for monster groups (no-op below thresholds;
    # the web review UI, which shares build_stitch_options, is unaffected).
    prune_info = prune_options_for_panel(options_ctx, group)
    if prune_info is not None:
        logger.info(
            f"Group {group.get('group_id')}: pruned options "
            f"{prune_info['n_before']} -> {prune_info['n_after']} (diversity)"
        )

    evidence = build_evidence_record(
        group,
        options_ctx,
        source_artifacts=source_artifacts,
        batch_generation_source=batch_generation_source,
        options_pruned=prune_info,
    )

    # Regeneration in the same directory must never leave a stale option or
    # junction crop that a glob-based provider attachment could still see.
    group_dir.mkdir(parents=True, exist_ok=True)

    overview = render_group_overview(group)
    overview.save(group_dir / "overview.png")

    for opt in options_ctx["options"]:
        img = render_option(group, opt)
        img.save(group_dir / f"option_{opt['letter']}.png")

    metadata = build_metadata(group, options_ctx, evidence=evidence)
    if prune_info is not None:
        metadata["options_pruned"] = prune_info

    # Junction zoom crops for SLIVER/BORDERLINE edges, and annotate the metadata
    # edge rows (all options that contain the edge) with their crop filename so
    # both the prompt and metadata.yaml reference them. Restricted to edges that
    # appear in at least one option so every crop is actually referenced.
    option_keys = {
        (e["ref_id"], e["target_id"]) for opt in options_ctx["options"] for e in opt["edges"]
    }
    zoom_files = _render_junction_zooms(group, group_dir, allowed_keys=option_keys)
    if zoom_files:
        metadata["zoom_crops"] = sorted(zoom_files.values())
        for opt_meta in metadata["options"]:
            for e in opt_meta["edges"]:
                fname = zoom_files.get(e["edge"])
                if fname:
                    e["zoom"] = fname

    (group_dir / "metadata.yaml").write_text(
        yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    )

    prompt = build_prompt(group_dir, metadata, options_ctx)
    (group_dir / "prompt.txt").write_text(prompt)
    write_evidence_manifest(group_dir, evidence)

    return metadata


def generate_stitch_evidence(
    batch: dict,
    output_dir: Path,
    group_ids: list[str] | None = None,
) -> list[str]:
    """Generate evidence packs for groups in a batch.

    Args:
        batch: Loaded stitch batch dict (must have 'groups').
        output_dir: Batch evidence dir (packs written under {output_dir}/{group_id}/).
        group_ids: If provided, only generate for these group_ids.

    Returns:
        List of group_ids for which an evidence pack was generated.
    """
    output_dir = Path(output_dir)
    groups = batch.get("groups", [])
    wanted = set(group_ids) if group_ids else None

    generated: list[str] = []
    for group in groups:
        gid = group.get("group_id")
        if wanted is not None and gid not in wanted:
            continue
        meta = generate_group_evidence(
            group,
            output_dir / str(gid),
            source_artifacts=batch.get("source_artifacts"),
            batch_generation_source=batch.get("batch_generation_source"),
        )
        if meta is not None:
            generated.append(gid)
    return generated


def missing_evidence_packs(
    packable: list[dict], generated: list[str]
) -> tuple[list[tuple[str, str]], list[str]]:
    """Which packable groups did NOT get an evidence pack, split by kind.

    :func:`generate_group_evidence` silently returns ``None`` (no pack) when a
    group yields no options, so a group can be requested yet never packed. For a
    decomposed sub-problem (#367 Mode B) that is not benign: a missing sub-problem
    pack is never voted, and the recomposition contract then blocks the parent's
    whole-group label PERMANENTLY (``subproblems_unvoted``) with no visible cause.

    Args:
        packable: the group dicts an evidence pack was requested for.
        generated: the group_ids :func:`generate_stitch_evidence` actually packed.

    Returns ``(missing_subproblems, missing_other)`` where ``missing_subproblems``
    is ``[(subproblem_id, parent_group_id), ...]`` (packable groups carrying a
    ``parent_group_id``) and ``missing_other`` is the remaining missing group_ids.
    """
    gen = {str(g) for g in generated}
    missing_subs: list[tuple[str, str]] = []
    missing_other: list[str] = []
    for g in packable:
        gid = str(g.get("group_id"))
        if gid in gen:
            continue
        parent = g.get("parent_group_id")
        if parent:
            missing_subs.append((gid, str(parent)))
        else:
            missing_other.append(gid)
    return missing_subs, missing_other
