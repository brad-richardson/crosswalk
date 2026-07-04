"""Evidence-pack generation for agent stitching-group labeling.

For each M:N group in a stitch batch, writes a self-contained evidence pack:

    {group_dir}/
        overview.png       - full group + spatial context (geometry-only)
        option_A.png        - option A's edges highlighted
        option_B.png        - ...
        metadata.yaml       - option table + per-segment names/classes
        prompt.txt          - rubric + option table + required JSON output

The agent (any provider) picks one option letter, or NONE if no option is
correct. Options are the deduplicated optimizer assignment + top-K
alternatives, from the shared :func:`build_stitch_options` so humans and agents
see the identical option set ("verify, don't construct").
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import shape as shape_from_geojson

from ..matching.sliver import edge_is_sliver as _edge_is_sliver
from ..matching.sliver import group_segment_lengths_m
from ..matching.stitch_options import build_stitch_options
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


def build_metadata(group: dict, options_ctx: dict) -> dict:
    """Build the metadata dict describing the group and its options."""
    ref_ids = group.get("ref_ids", list(group.get("ref_geometries", {}).keys()))
    target_ids = group.get("target_ids", list(group.get("target_geometries", {}).keys()))
    ref_labels = {rid: f"R{i + 1}" for i, rid in enumerate(ref_ids)}
    target_labels = {tid: f"T{i + 1}" for i, tid in enumerate(target_ids)}
    ref_names = group.get("ref_names", {})
    target_names = group.get("target_names", {})
    ref_classes = group.get("ref_classes", {})
    target_classes = group.get("target_classes", {})

    # Per-edge junction-sliver flags (hybrid fraction + absolute-meters rule).
    # Slivers are ANNOTATED here, never silently dropped: an option may legitimately
    # exclude one, and the agent should be able to see which edges are artifacts.
    ref_lens, tgt_lens = group_segment_lengths_m(group)

    options_meta = []
    for opt in options_ctx["options"]:
        edges_meta = []
        sliver_edge_count = 0
        for e in opt["edges"]:
            rid, tid = e["ref_id"], e["target_id"]
            is_sliver = _edge_is_sliver(e, ref_lens, tgt_lens)
            sliver_edge_count += int(is_sliver)
            row = {
                "edge": f"{ref_labels.get(rid, rid)}->{target_labels.get(tid, tid)}",
                "ref": ref_labels.get(rid, rid),
                "target": target_labels.get(tid, tid),
                "confidence": round(float(e.get("confidence", 0.0)), 3),
                "is_sliver": is_sliver,
            }
            row.update(_edge_align_fracs(e))
            edges_meta.append(row)
        options_meta.append(
            {
                "letter": opt["letter"],
                "is_optimizer": opt["is_optimizer"],
                "edge_count": opt["edge_count"],
                "sliver_edge_count": sliver_edge_count,
                "total_confidence": round(float(opt["total_confidence"]), 3),
                "mean_confidence": round(float(opt["mean_confidence"]), 3),
                "edges": edges_meta,
            }
        )

    return {
        "group_id": group.get("group_id"),
        "match_type": group.get("match_type"),
        "n_ref_segments": len(ref_ids),
        "n_target_segments": len(target_ids),
        # Audit: full vs rendered edge counts. Post-fix these are always equal
        # (group data is never clipped); a divergence flags a clipping regression.
        "n_edges_full": group.get("n_edges_full", len(group.get("edges", []))),
        "n_edges_rendered": group.get("n_edges_rendered", len(group.get("edges", []))),
        "context_clipped": bool(group.get("context_clipped", False)),
        "optimizer_letter": options_ctx.get("optimizer_letter"),
        "segments": {
            "reference": [
                {
                    "label": ref_labels[rid],
                    "id": rid,
                    "name": ref_names.get(rid, ""),
                    "class": ref_classes.get(rid, ""),
                }
                for rid in ref_ids
            ],
            "target": [
                {
                    "label": target_labels[tid],
                    "id": tid,
                    "name": target_names.get(tid, ""),
                    "class": target_classes.get(tid, ""),
                }
                for tid in target_ids
            ],
        },
        "options": options_meta,
    }


def build_prompt(group_dir: Path, metadata: dict, options_ctx: dict) -> str:
    """Build the option-picker prompt referencing absolute image paths."""
    group_dir = group_dir.resolve()
    letters = [o["letter"] for o in options_ctx["options"]]
    choices = "|".join(letters + ["NONE"])
    opt_letter = metadata.get("optimizer_letter")

    lines: list[str] = []
    lines.append(
        "You are curating ground truth for a road-network conflation optimizer.\n"
        "A 'stitching group' is a cluster of candidate matches between REFERENCE road\n"
        "segments (blue, labeled R1, R2, ...) and TARGET road segments (red, labeled\n"
        "T1, T2, ...). Your job: pick the ONE assignment option whose highlighted edges\n"
        "best represent the true same-physical-road correspondences in the group."
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
    lines.append("  Pick the option whose bright segments overlap/follow the SAME physical roads.")
    lines.append("")
    lines.append("GUIDANCE:")
    lines.append("- A correct edge R#->T# means the reference and target segment are the same")
    lines.append("  physical traveled way (overlapping geometry, same path). Small offsets ok.")
    lines.append("- Parallel-but-separate roads, opposite carriageways, and perpendicular")
    lines.append("  crossings are NOT correct edges even if they touch at a junction.")
    lines.append("- An edge tagged SLIVER below is a junction artifact: the two segments share")
    lines.append("  almost no physical overlap (a road end merely clips another at a corner).")
    lines.append("  Prefer an option that excludes it; it is almost never a correct edge.")
    lines.append("- A pedestrian-class segment (footway/sidewalk/path) is a DIFFERENT physical")
    lines.append("  feature than a road-class segment (residential/primary/service/...), even")
    lines.append("  when it runs right alongside one. Never match a footway/sidewalk/path to a")
    lines.append("  road class just because they are parallel or nearby. If an option's only")
    lines.append("  advantage is that it adds such a cross-mode edge, prefer the option without")
    lines.append("  it, or NONE.")
    lines.append("- The optimizer's own proposed option is labeled below; it is often but not")
    lines.append(
        "  always correct. Judge from the geometry, not from which one is the optimizer's."
    )
    lines.append("- Choose NONE only if NO option is a good representation (e.g. the correct")
    lines.append("  assignment would need edges no option contains, or all options are wrong).")
    lines.append("")
    lines.append(
        f"GROUP {metadata['group_id']}  (match_type={metadata['match_type']}, "
        f"{metadata['n_ref_segments']} ref x {metadata['n_target_segments']} target)"
    )
    if opt_letter:
        lines.append(f"Optimizer's proposed option: {opt_letter}")
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
        for e in opt["edges"]:
            extra = []
            if "ref_aligned_frac" in e:
                extra.append(f"ref_aln={e['ref_aligned_frac']}")
            if "target_aligned_frac" in e:
                extra.append(f"tgt_aln={e['target_aligned_frac']}")
            if e.get("is_sliver"):
                extra.append("SLIVER(junction artifact, ~0 overlap)")
            extra_s = ("  " + " ".join(extra)) if extra else ""
            lines.append(f"      {e['edge']}  conf={e['confidence']}{extra_s}")
    lines.append("")
    lines.append("SEGMENTS (name / class):")
    for s in metadata["segments"]["reference"]:
        lines.append(f"  {s['label']}: name='{s['name']}' class='{s['class']}'")
    for s in metadata["segments"]["target"]:
        lines.append(f"  {s['label']}: name='{s['name']}' class='{s['class']}'")
    lines.append("")
    lines.append("Look at overview.png first, then each option image. Then respond with ONLY a")
    lines.append("single JSON object (no prose, no markdown fence) of the form:")
    lines.append(f'  {{"choice": "<{choices}>", "confidence": 0.0-1.0, "reasoning": "..."}}')
    lines.append('"choice" MUST be exactly one of the option letters above, or "NONE".')
    return "\n".join(lines)


def generate_group_evidence(group: dict, group_dir: Path) -> dict | None:
    """Generate the full evidence pack for one group. Returns the metadata dict.

    Returns None if the group has no options.
    """
    options_ctx = build_stitch_options(group)
    if not options_ctx["options"]:
        logger.warning(f"Group {group.get('group_id')}: no options, skipping")
        return None

    group_dir.mkdir(parents=True, exist_ok=True)

    overview = render_group_overview(group)
    overview.save(group_dir / "overview.png")

    for opt in options_ctx["options"]:
        img = render_option(group, opt)
        img.save(group_dir / f"option_{opt['letter']}.png")

    metadata = build_metadata(group, options_ctx)
    (group_dir / "metadata.yaml").write_text(
        yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    )

    prompt = build_prompt(group_dir, metadata, options_ctx)
    (group_dir / "prompt.txt").write_text(prompt)

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
        meta = generate_group_evidence(group, output_dir / str(gid))
        if meta is not None:
            generated.append(gid)
    return generated
