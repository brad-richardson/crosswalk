#!/usr/bin/env python
"""Reflect a human's M:N stitching-review choices back as annotated PNGs.

For each curated row in ``labels/stitching/dataset=<ds>/data.csv`` this renders:

  * an **overview** PNG per group — selected refs thick blue, selected targets
    thick orange, everything else pale; green-dotted connectors for pairs the
    labeler ADDED vs the optimizer, red-dashed connectors for optimizer pairs
    the labeler EXCLUDED; the title carries your-label-vs-optimizer counts.
  * a **zoom** PNG (one panel per differing pair, only when the label differs
    from the optimizer) — ~50 m bbox around the pair, faint group context, a
    green (added) / red (excluded) halo on the diff pair, and a panel title with
    the confidence for that exact pair plus ref/target ids and street names.

Data sources (with the prototypes' fallback logic):

  * group geometries / names / classes: ``data/cache/stitch/<ds>_batch.json``
  * optimizer selection: ``data/output/<ds>_groups.json`` ``edges[selected]``
    when the group is present there, otherwise the cache group's
    ``optimizer_assignment`` (an *old-grouping* queue item — flagged in titles
    and the summary table).
  * per-pair confidence: looked up across cache ``edges`` -> sidecar ``edges``
    -> sidecar ``rejected_edges`` -> cache ``optimizer_assignment`` (first
    non-null wins).

A terminal summary table is always printed so the script is useful without
opening the images.

Example::

    uv run python scripts/render_review_diffs.py \
        --data-root /Users/you/dev/matcher \
        --dataset us_boston_streets --labeler brad \
        --since 2026-07-05T14:40 -o /tmp/review_renders
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------------------------------------------------------------------------
# Visual language (preserved verbatim from the approved prototypes)
# ---------------------------------------------------------------------------
REF_SEL = "#2b7bd3"  # ref (Overture) segment in the label
REF_UNSEL = "#b8cbe4"  # group ref segment not in the label
TGT_SEL = "#e2611f"  # target (local) segment in the label
TGT_UNSEL = "#f0c4ab"  # group target segment not in the label
ADDED_COLOR = "#1a9c3e"  # pair added vs optimizer (green)
REMOVED_COLOR = "#d81f3d"  # optimizer pair excluded by labeler (red)

METERS_PER_DEG = 111_320.0  # rough lat->m; matches prototype constant


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------
def coerce_coords(geom: dict | None) -> list[list[list[float]]]:
    """Return a list of coordinate rings for a GeoJSON LineString/MultiLineString.

    Always returns a list of line coordinate-sequences so callers can iterate
    uniformly over single- and multi-part geometries. Unknown/None geometries
    yield ``[]``.
    """
    if not geom or "type" not in geom:
        return []
    gtype = geom["type"]
    coords = geom.get("coordinates") or []
    if gtype == "LineString":
        return [list(coords)] if coords else []
    if gtype == "MultiLineString":
        return [list(part) for part in coords if part]
    return []


def geom_points(geom: dict | None) -> list[list[float]]:
    """Flatten every coordinate of a (Multi)LineString to a single point list."""
    return [pt for part in coerce_coords(geom) for pt in part]


def geom_midpoint(geom: dict | None) -> list[float] | None:
    """Midpoint of the first part of a geometry (connector anchor)."""
    parts = coerce_coords(geom)
    if not parts or not parts[0]:
        return None
    part = parts[0]
    return part[len(part) // 2]


# ---------------------------------------------------------------------------
# Pure diff / lookup logic
# ---------------------------------------------------------------------------
def parse_selected_edges(raw: str | None) -> set[tuple[str, str]]:
    """Parse the ``selected_edges`` CSV cell into a set of (ref_id, target_id)."""
    if not raw:
        return set()
    parsed = ast.literal_eval(raw)
    return {(e["ref_id"], e["target_id"]) for e in parsed}


def edge_pairs(edges: list[dict] | None, selected_only: bool = False) -> set[tuple[str, str]]:
    """Collect (ref_id, target_id) tuples from an edge list."""
    out: set[tuple[str, str]] = set()
    for e in edges or []:
        if selected_only and not e.get("selected"):
            continue
        out.add((e["ref_id"], e["target_id"]))
    return out


def resolve_optimizer(
    cache_group: dict | None, sidecar_group: dict | None
) -> tuple[set[tuple[str, str]], bool]:
    """Return the optimizer's selected pairs and whether they came from the sidecar.

    Prefers the sidecar group's ``edges[selected]``; falls back to the cache
    group's ``optimizer_assignment`` (old-grouping queue item). The bool is
    ``True`` when the sidecar supplied the set.
    """
    if sidecar_group is not None:
        return edge_pairs(sidecar_group.get("edges"), selected_only=True), True
    if cache_group is not None:
        return edge_pairs(cache_group.get("optimizer_assignment")), False
    return set(), False


def compute_diff(
    label_pairs: set[tuple[str, str]], opt_pairs: set[tuple[str, str]]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return (added, removed): pairs the label adds vs / drops from the optimizer."""
    return label_pairs - opt_pairs, opt_pairs - label_pairs


def build_confidence_lookup(
    cache_group: dict | None, sidecar_group: dict | None
) -> dict[tuple[str, str], float]:
    """Map (ref_id, target_id) -> confidence with prototype precedence.

    Precedence (first non-null confidence wins): cache ``edges`` ->
    sidecar ``edges`` -> sidecar ``rejected_edges`` -> cache
    ``optimizer_assignment``.
    """
    m: dict[tuple[str, str], float] = {}
    sources = [
        (cache_group or {}).get("edges", []),
        (sidecar_group or {}).get("edges", []),
        (sidecar_group or {}).get("rejected_edges", []),
        (cache_group or {}).get("optimizer_assignment", []),
    ]
    for src in sources:
        for e in src or []:
            key = (e.get("ref_id"), e.get("target_id"))
            conf = e.get("confidence")
            if key not in m and conf is not None:
                m[key] = conf
    return m


def candidate_universe(
    cache_group: dict | None, sidecar_group: dict | None
) -> set[tuple[str, str]]:
    """All (ref_id, target_id) pairs the reviewer could have chosen from.

    Union of cache ``edges``, sidecar ``edges`` and sidecar ``rejected_edges`` —
    every candidate pair surfaced in the group, selected or not.
    """
    return (
        edge_pairs((cache_group or {}).get("edges"))
        | edge_pairs((sidecar_group or {}).get("edges"))
        | edge_pairs((sidecar_group or {}).get("rejected_edges"))
    )


def crossproduct_within_universe(
    label_pairs: set[tuple[str, str]], universe: set[tuple[str, str]]
) -> set[tuple[str, str]]:
    """The (label refs × label targets) grid, restricted to candidate pairs.

    In manual / de-anchored labelling the submit records the full cross-product
    of the active ref-pills and target-pills intersected with the candidate
    universe — so this is what a "select-all-pills" submit would have stored.
    """
    refs = {a for a, _ in label_pairs}
    tgts = {b for _, b in label_pairs}
    return {(r, t) for r in refs for t in tgts if (r, t) in universe}


def is_crossproduct_artifact(
    label_pairs: set[tuple[str, str]],
    opt_pairs: set[tuple[str, str]],
    universe: set[tuple[str, str]],
) -> bool:
    """Flag labels whose extra pairs are likely cross-product artifacts.

    True when the stored pair set is *exactly* the ref×target cross-product
    within the candidate universe AND it adds pairs beyond the optimizer — i.e.
    the reviewer's pill selection over-expanded into pairs they may never have
    consciously chosen. Pure exclusions (added set empty) never flag.
    """
    if not label_pairs:
        return False
    if not (label_pairs - opt_pairs):  # no pairs beyond the optimizer
        return False
    return label_pairs == crossproduct_within_universe(label_pairs, universe)


# ---------------------------------------------------------------------------
# Group-id indexing (ids are matched on their leading 8 hex chars)
# ---------------------------------------------------------------------------
def _gid_key(gid: str) -> str:
    return gid[:8]


def index_groups(groups: list[dict]) -> dict[str, dict]:
    """Index group dicts by their 8-char group-id key.

    Emits nothing on collision but keeps the first seen (callers may warn).
    """
    idx: dict[str, dict] = {}
    for g in groups:
        idx.setdefault(_gid_key(g["group_id"]), g)
    return idx


def gid_matches(gid: str, wanted: list[str] | None) -> bool:
    """Whether ``gid`` matches any of the (possibly prefix) --group-id filters."""
    if not wanted:
        return True
    key = _gid_key(gid)
    return any(key == _gid_key(w) for w in wanted)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cache_groups(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return index_groups(data.get("groups", []))


def load_sidecar_groups(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    groups = data["groups"] if isinstance(data, dict) else data
    return index_groups(groups)


def load_label_rows(
    csv_path: Path,
    labeler: str,
    since: str | None,
    session: str | None,
    group_ids: list[str] | None,
) -> list[dict]:
    """Load & filter stitching label rows (labeler / since / session / group)."""
    rows: list[dict] = []
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if labeler and r.get("labeler") != labeler:
                continue
            if since and (r.get("labeled_at") or "") <= since:
                continue
            if session and (r.get("session_id") or "") != session:
                continue
            if not gid_matches(r["group_id"], group_ids):
                continue
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Review record
# ---------------------------------------------------------------------------
@dataclass
class ReviewDiff:
    gid: str
    match_type: str
    session_id: str
    label_pairs: set[tuple[str, str]]
    opt_pairs: set[tuple[str, str]]
    added: set[tuple[str, str]] = field(default_factory=set)
    removed: set[tuple[str, str]] = field(default_factory=set)
    from_sidecar: bool = False
    crossproduct_artifact: bool = False

    @property
    def is_reject_all(self) -> bool:
        return len(self.label_pairs) == 0

    @property
    def has_diff(self) -> bool:
        return bool(self.added or self.removed)


def build_review(row: dict, cache_group: dict | None, sidecar_group: dict | None) -> ReviewDiff:
    label_pairs = parse_selected_edges(row.get("selected_edges"))
    opt_pairs, from_sidecar = resolve_optimizer(cache_group, sidecar_group)
    added, removed = compute_diff(label_pairs, opt_pairs)
    universe = candidate_universe(cache_group, sidecar_group)
    return ReviewDiff(
        gid=_gid_key(row["group_id"]),
        match_type=row.get("match_type", ""),
        session_id=row.get("session_id") or "",
        label_pairs=label_pairs,
        opt_pairs=opt_pairs,
        added=added,
        removed=removed,
        from_sidecar=from_sidecar,
        crossproduct_artifact=is_crossproduct_artifact(label_pairs, opt_pairs, universe),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _draw_geom(ax, geom: dict | None, **kw) -> None:
    for part in coerce_coords(geom):
        ax.plot([c[0] for c in part], [c[1] for c in part], **kw)


def _group_names(group: dict) -> list[str]:
    names = set(group.get("ref_names", {}).values()) | set(group.get("target_names", {}).values())
    return sorted(n for n in names if n)


def render_overview(review: ReviewDiff, group: dict, out_path: Path) -> Path:
    """Render the per-group overview PNG. Returns the output path."""
    sel_refs = {a for a, _ in review.label_pairs}
    sel_tgts = {b for _, b in review.label_pairs}

    fig, ax = plt.subplots(figsize=(11, 11))
    lat0 = None
    for rid, geom in group.get("ref_geometries", {}).items():
        pts = geom_points(geom)
        if lat0 is None and pts:
            lat0 = pts[0][1]
        insel = rid in sel_refs
        _draw_geom(
            ax,
            geom,
            color=REF_SEL if insel else REF_UNSEL,
            lw=4.5 if insel else 1.8,
            solid_capstyle="round",
            zorder=3 if insel else 1,
        )
    for tid, geom in group.get("target_geometries", {}).items():
        pts = geom_points(geom)
        if lat0 is None and pts:
            lat0 = pts[0][1]
        insel = tid in sel_tgts
        _draw_geom(
            ax,
            geom,
            color=TGT_SEL if insel else TGT_UNSEL,
            lw=2.8 if insel else 1.2,
            solid_capstyle="round",
            zorder=4 if insel else 2,
        )

    def connect(pairs, color, style):
        for a, b in pairs:
            ma = geom_midpoint(group.get("ref_geometries", {}).get(a))
            mb = geom_midpoint(group.get("target_geometries", {}).get(b))
            if ma is None or mb is None:
                continue
            ax.plot([ma[0], mb[0]], [ma[1], mb[1]], color=color, ls=style, lw=1.6, zorder=6)

    connect(review.added, ADDED_COLOR, ":")
    connect(review.removed, REMOVED_COLOR, "--")

    if lat0 is not None:
        ax.set_aspect(1 / math.cos(math.radians(lat0)))
    ax.set_xticks([])
    ax.set_yticks([])

    names = _group_names(group)
    old_grp = "  (old-grouping queue item)" if not review.from_sidecar else ""
    if review.is_reject_all:
        title = f"{review.gid} | {review.match_type} | reject-all (no pairs kept)"
        sub = f"optimizer had {len(review.opt_pairs)} pairs{old_grp}"
    else:
        title = (
            f"{review.gid} | {review.match_type} | your label "
            f"{len(review.label_pairs)} pairs vs optimizer {len(review.opt_pairs)}"
        )
        sub = f"added {len(review.added)} / removed {len(review.removed)}{old_grp}"
    parts = [title, sub]
    if review.crossproduct_artifact:
        parts.append(
            f"⚠ pair-level intent not expressed — {len(review.added)} pairs beyond "
            "optimizer may be cross-product artifacts"
        )
    parts.append(", ".join(names[:3]))
    ax.set_title("\n".join(parts), fontsize=11)

    handles = [
        Line2D([], [], color=REF_SEL, lw=4, label="ref (Overture) - selected"),
        Line2D([], [], color=TGT_SEL, lw=3, label="target (local) - selected"),
        Line2D([], [], color=REF_UNSEL, lw=2, label="group segment, not in your label"),
    ]
    if review.added:
        handles.append(
            Line2D(
                [],
                [],
                color=ADDED_COLOR,
                ls=":",
                label=f"pair you ADDED vs optimizer ({len(review.added)})",
            )
        )
    if review.removed:
        handles.append(
            Line2D(
                [],
                [],
                color=REMOVED_COLOR,
                ls="--",
                label=f"optimizer pair you EXCLUDED ({len(review.removed)})",
            )
        )
    ax.legend(handles=handles, loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def render_zoom(
    review: ReviewDiff, group: dict, sidecar_group: dict | None, out_path: Path
) -> Path | None:
    """Render the per-diff-pair zoom panels. Returns path, or None if no diffs."""
    diffs = [("ADDED", p) for p in sorted(review.added)] + [
        ("EXCLUDED", p) for p in sorted(review.removed)
    ]
    if not diffs:
        return None
    confs = build_confidence_lookup(group, sidecar_group)

    n = len(diffs)
    cols = min(n, 3)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 7 * rows_n), squeeze=False)
    for i, (kind, (a, b)) in enumerate(diffs):
        ax = axes[i // cols][i % cols]
        ga = group.get("ref_geometries", {}).get(a)
        gb = group.get("target_geometries", {}).get(b)
        pts = geom_points(ga) + geom_points(gb)
        if not pts:
            ax.axis("off")
            continue
        lat0 = pts[0][1]
        margin_m = 50.0
        my = margin_m / METERS_PER_DEG
        mx = margin_m / METERS_PER_DEG / math.cos(math.radians(lat0))
        x0 = min(p[0] for p in pts) - mx
        x1 = max(p[0] for p in pts) + mx
        y0 = min(p[1] for p in pts) - my
        y1 = max(p[1] for p in pts) + my

        # faint context: every other group segment
        for rid, geom in group.get("ref_geometries", {}).items():
            if rid != a:
                _draw_geom(ax, geom, color=REF_UNSEL, lw=2.2, zorder=1)
        for tid, geom in group.get("target_geometries", {}).items():
            if tid != b:
                _draw_geom(ax, geom, color=TGT_UNSEL, lw=1.6, zorder=2)

        col = ADDED_COLOR if kind == "ADDED" else REMOVED_COLOR
        if ga:
            _draw_geom(ax, ga, color=REF_SEL, lw=6, zorder=3, solid_capstyle="round")
        if gb:
            _draw_geom(ax, gb, color=TGT_SEL, lw=3.4, zorder=4, solid_capstyle="round")
        # halo marking the diff pair
        if ga:
            _draw_geom(ax, ga, color=col, lw=11, alpha=0.25, zorder=2.5)
        if gb:
            _draw_geom(ax, gb, color=col, lw=8, alpha=0.25, zorder=2.5)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(1 / math.cos(math.radians(lat0)))
        ax.set_xticks([])
        ax.set_yticks([])
        rn = group.get("ref_names", {}).get(a, "")
        tn = group.get("target_names", {}).get(b, "")
        c = confs.get((a, b))
        conf_str = f"conf {c:.2f}" if c is not None else "conf n/a"
        tgt_tail = "_".join(b.split("_")[-2:])
        artifact_note = (
            "\ncross-product artifact?"
            if (kind == "ADDED" and review.crossproduct_artifact)
            else ""
        )
        ax.set_title(
            f"{kind}: {conf_str}{artifact_note}\nref {a[:8]} {rn}\n→ tgt …{tgt_tail} {tn}",
            fontsize=10,
            color=col,
        )
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].axis("off")

    add_txt = "your additions" if review.added else ""
    amp = " & " if (review.added and review.removed) else ""
    excl_txt = "your exclusions" if review.removed else ""
    fig.suptitle(
        f"group {review.gid} — {add_txt}{amp}{excl_txt} vs optimizer (halo = the diff pair)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def format_summary(reviews: list[ReviewDiff]) -> str:
    headers = ["group", "match_type", "label", "opt", "added", "removed", "session", "artifact"]
    rows = []
    n_flagged = 0
    for rv in reviews:
        session = rv.session_id or "-"
        if not rv.from_sidecar:
            session = f"{session} (old-grp)"
        artifact = "-"
        if rv.crossproduct_artifact:
            n_flagged += 1
            artifact = f"xprod? +{len(rv.added)}"
        rows.append(
            [
                rv.gid,
                rv.match_type or "-",
                str(len(rv.label_pairs)),
                str(len(rv.opt_pairs)),
                str(len(rv.added)),
                str(len(rv.removed)),
                session,
                artifact,
            ]
        )
    if not rows:
        return "(no matching label rows)"
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)) for r in rows)
    out = "\n".join([line, sep, body])
    if n_flagged:
        out += (
            f"\n\n⚠ {n_flagged} label(s) flagged 'xprod?': stored pairs exactly equal the "
            "ref×target cross-product within the candidate universe and add pairs beyond the "
            "optimizer — likely manual/de-anchored cross-product artifacts, not deliberate picks."
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True, help="Dataset id, e.g. us_boston_streets")
    p.add_argument("--labeler", default="brad", help="Filter to this labeler (default: brad)")
    p.add_argument(
        "--since",
        default=None,
        help="Only rows with labeled_at > this ISO timestamp (default: all)",
    )
    p.add_argument(
        "--group-id",
        action="append",
        dest="group_ids",
        help="Restrict to these group ids (repeatable; 8-char prefix ok)",
    )
    p.add_argument("--session", default=None, help="Filter to this session_id (e.g. deanchored_v1)")
    p.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output dir (default: <data-root>/data/output/review_renders/<dataset>/)",
    )
    p.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Repo root holding data/ and labels/ (default: this checkout)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    ds = args.dataset

    cache_path = data_root / "data" / "cache" / "stitch" / f"{ds}_batch.json"
    sidecar_path = data_root / "data" / "output" / f"{ds}_groups.json"
    labels_path = data_root / "labels" / "stitching" / f"dataset={ds}" / "data.csv"

    if not cache_path.exists():
        print(f"ERROR: cache not found: {cache_path}")
        return 1
    if not labels_path.exists():
        print(f"ERROR: stitching labels not found: {labels_path}")
        return 1

    cache_groups = load_cache_groups(cache_path)
    sidecar_groups = load_sidecar_groups(sidecar_path)
    rows = load_label_rows(labels_path, args.labeler, args.since, args.session, args.group_ids)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_root / "data" / "output" / "review_renders" / ds
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews: list[ReviewDiff] = []
    n_overview = n_zoom = 0
    for row in sorted(rows, key=lambda r: r["group_id"]):
        key = _gid_key(row["group_id"])
        cache_group = cache_groups.get(key)
        sidecar_group = sidecar_groups.get(key)
        if cache_group is None:
            print(f"WARN: group {key} absent from cache and sidecar; skipping")
            continue
        review = build_review(row, cache_group, sidecar_group)
        reviews.append(review)

        render_overview(review, cache_group, out_dir / f"review_{key}.png")
        n_overview += 1
        if review.has_diff:
            render_zoom(review, cache_group, sidecar_group, out_dir / f"zoom_{key}.png")
            n_zoom += 1

    print(format_summary(reviews))
    print(f"\nRendered {n_overview} overview + {n_zoom} zoom PNG(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
