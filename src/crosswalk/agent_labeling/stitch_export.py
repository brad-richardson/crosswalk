"""Export unanimous LLM-panel consensus into human-equivalent stitching labels.

The 3-provider stitching panel (see :mod:`stitch_runner`) votes on M:N group edge
selections and writes a ``consensus.csv`` per batch. This module promotes the
subset of those verdicts that are safe to treat as durable labels -- only
*unanimous* auto-accept groups -- into ``labels/stitching`` alongside the human
labels, tagged with the labeler ``panel_unanimous_v3`` so their provenance stays
visible (v1 tagged the earlier sonnet/gpt-5.4/Gemini-Flash-Low panel; v2 the
Opus 4.8/gpt-5.5/Gemini-3.5-Flash panel on pre-enrichment packs; the tag is
bumped whenever the panel composition OR its pack inputs change).

Gates (applied in order; the first failing gate decides the group and is
reported):

  a. routing == ``auto_accept`` (unanimous, non-NONE) -- everything else is not a
     candidate at all.
  b. candidate edge count <= ``max_edges`` (default 20) -- huge groups stay for
     human review.
  c. class-consistency gate -- reuses the panel runner's cross-mode rule
     (:func:`stitch_runner.has_cross_mode_edge`) on the chosen edge set
     (pedestrian / vehicular / bike; any two different modes are cross-mode).
  d. sliver canonicalization -- drops junction-sliver edges (shared definition in
     :func:`crosswalk.matching.sliver`); if that empties the set the group is
     skipped.
  e. human precedence -- a group already covered by a *human* label (by exact
     group_id or by edge-overlap, reusing :func:`stitch_eval.map_human_labels_to_groups`)
     is left untouched.

Writing is idempotent: rows are upserted by ``group_id`` under the
``panel_unanimous_v3`` labeler, so re-running never duplicates and always
refreshes to the latest consensus. Previously exported panel rows are excluded
from the human-precedence check (they are not human), so re-runs stay accurate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import settings
from ..labeling.stitching_store import StitchingLabelStore
from ..matching.optimizer import group_is_structurally_simple
from ..matching.sliver import annotate_group_sliver_flags
from .stitch_eval import (
    _is_set_label,
    _load_group_metadata,
    _map_set_labels_to_groups,
    map_human_labels_to_groups,
)
from .stitch_runner import _edge_classes_for, _segment_class_maps, has_cross_mode_edge

# Bumped v1 -> v2 when the panel composition changed (Opus 4.8 / gpt-5.5 /
# Gemini 3.5 Flash Medium); v2 -> v3 when the evidence-pack inputs changed
# (#302 enrichment: per-edge overlap meters, BORDERLINE tags, junction zoom
# crops -- votes are not comparable across pack versions, see
# research/panel_enriched_ab.md). Existing v1/v2 labels stay untouched; future
# waves are tagged v3. Any labeler with the PANEL_LABELER_PREFIX is a panel
# (non-human) label and is excluded from the human-precedence check below.
PANEL_LABELER = "panel_unanimous_v3"
PANEL_LABELER_PREFIX = "panel_"

# Per-group outcome reasons (stable strings for reporting/tests).
REASON_EXPORTED = "exported"
REASON_OVER_MAX = "over_max_edges"
REASON_STRUCTURAL_TANGLE = "structural_tangle"
REASON_CLASS_MISMATCH = "class_mismatch"
REASON_EMPTIED_BY_SLIVER = "emptied_by_sliver"
REASON_HUMAN_PRECEDENCE = "human_precedence"


@dataclass
class GroupExport:
    """Outcome of running the export gates on a single consensus group."""

    group_id: str
    source_batch: str
    exported: bool
    reason: str
    match_type: str = ""
    n_edges_raw: int = 0
    n_slivers_dropped: int = 0
    n_edges_final: int = 0
    mean_confidence: float = 0.0
    selected_edges: list[dict] = field(default_factory=list)
    human_group_id: str = ""


@dataclass
class ExportReport:
    """Full plan of what would be / was written, plus tallies for reporting."""

    dataset: str
    n_total_groups: int
    n_auto_accept: int
    groups: list[GroupExport]

    @property
    def exported(self) -> list[GroupExport]:
        return [g for g in self.groups if g.exported]

    @property
    def skipped(self) -> list[GroupExport]:
        return [g for g in self.groups if not g.exported]

    def skipped_by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for g in self.skipped:
            out[g.reason] = out.get(g.reason, 0) + 1
        return out

    def total_slivers_dropped(self) -> int:
        return sum(g.n_slivers_dropped for g in self.groups)


def _parse_edge_set_pairs(raw) -> list[tuple[str, str]]:
    """Parse a consensus ``edge_set`` JSON string ``[[ref,tgt],...]`` to pairs."""
    if raw is None or isinstance(raw, float):
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [(str(a), str(b)) for a, b in data]


def _merge_consensus(batch_dirs: list[Path]) -> dict[str, tuple[Path, dict]]:
    """Merge each batch's ``consensus.csv``; later batch dirs supersede earlier.

    Returns ``{group_id: (batch_dir, consensus_row_dict)}``.
    """
    merged: dict[str, tuple[Path, dict]] = {}
    for bd in batch_dirs:
        cpath = Path(bd) / "consensus.csv"
        if not cpath.exists():
            raise FileNotFoundError(f"No consensus.csv in batch dir: {bd}")
        df = pd.read_csv(cpath, dtype={"group_id": str})
        for _, row in df.iterrows():
            merged[str(row["group_id"])] = (Path(bd), row.to_dict())
    return merged


def _load_batch_groups(batch_dir: Path) -> dict[str, dict]:
    """Load ``batch.json`` groups keyed by group_id (geometries + edges)."""
    batch_path = Path(batch_dir) / "batch.json"
    if not batch_path.exists():
        return {}
    try:
        batch = json.loads(batch_path.read_text())
    except (ValueError, OSError):
        return {}
    return {str(g.get("group_id")): g for g in batch.get("groups", [])}


def _group_sliver_pairs(group: dict) -> set[tuple[str, str]]:
    """Return the set of ``(ref_id, target_id)`` pairs the group classifies as slivers."""
    if not group:
        return set()
    annotated, _ = annotate_group_sliver_flags(group)
    return {(str(e["ref_id"]), str(e["target_id"])) for e in annotated if e.get("is_sliver")}


def _meta_from_group(grp: dict) -> dict | None:
    """Synthesize a ``metadata.yaml``-shaped dict from a ``batch.json`` group.

    Fallback for packs that carry ``batch.json`` (ids + classes) but no per-group
    ``metadata.yaml``, so the class-consistency gate and the human edge-overlap
    mapping can still run instead of silently degrading. Returns ``None`` when the
    group has no usable segment ids.
    """
    if not grp:
        return None
    ref_classes = grp.get("ref_classes") or {}
    tgt_classes = grp.get("target_classes") or {}
    ref_ids = grp.get("ref_ids") or sorted({str(e["ref_id"]) for e in grp.get("edges", [])})
    tgt_ids = grp.get("target_ids") or sorted({str(e["target_id"]) for e in grp.get("edges", [])})
    if not ref_ids and not tgt_ids:
        return None
    return {
        "match_type": grp.get("match_type", ""),
        "segments": {
            "reference": [
                {"label": f"R{i + 1}", "id": str(r), "class": ref_classes.get(str(r), "") or ""}
                for i, r in enumerate(ref_ids)
            ],
            "target": [
                {"label": f"T{i + 1}", "id": str(t), "class": tgt_classes.get(str(t), "") or ""}
                for i, t in enumerate(tgt_ids)
            ],
        },
    }


def plan_exports(
    batch_dirs: list[Path],
    dataset: str,
    labels_dir: Path,
    max_edges: int = 20,
    max_assignment_components: int | None = None,
    soft_max_edges: int | None = None,
    backstop_max_edges: int | None = None,
) -> ExportReport:
    """Run the export gates over merged consensus and return a full plan.

    Pure w.r.t. the label store: reads human labels but writes nothing. Call
    :func:`write_exports` with the returned report to persist.

    The size gate is *structural*, not a flat edge count: a group auto-exports
    when it is a single corridor-pair OR has few assignment-components within a
    soft edge budget (so a clean 30-edge single corridor is exportable, while a
    small two-highway tangle is blocked). A hard backstop ceiling blocks
    anything larger regardless — a defence against a structure-detection bug,
    not the primary gate. When a group lacks structure fields (an older
    batch.json predating the structure sidecar) the gate falls back to the flat
    ``max_edges`` cap on the selected edge set.
    """
    if max_assignment_components is None:
        max_assignment_components = settings.stitch_export_max_assignment_components
    if soft_max_edges is None:
        soft_max_edges = settings.stitch_export_soft_max_edges
    if backstop_max_edges is None:
        backstop_max_edges = settings.stitch_export_backstop_max_edges
    merged = _merge_consensus(batch_dirs)

    # Load batch.json groups (geometries/edges) once per distinct batch dir.
    # batch.json is the ONLY source of geometries (sliver gate) and candidate
    # edges (overlap precedence); warn loudly if a batch dir lacks it, since
    # those gates then degrade rather than fail.
    batch_groups: dict[Path, dict[str, dict]] = {}
    for bd in {bd for bd, _ in merged.values()}:
        if not (bd / "batch.json").exists():
            logger.warning(
                f"No batch.json in {bd}: sliver canonicalization and edge-overlap "
                "precedence cannot run for its groups (gates degrade)."
            )
        batch_groups[bd] = _load_batch_groups(bd)

    # Human labels for precedence. Exclude our own previously-exported panel rows
    # (they are not human) so re-runs stay idempotent and accurately reported.
    store = StitchingLabelStore(dataset, labels_dir=labels_dir)
    human_df = store.load(dataset)
    if not human_df.empty and "labeler" in human_df.columns:
        # Exclude ALL panel labelers (v1, v2, ...) — none are human, so they must
        # not confer human precedence. Matching the prefix keeps re-runs idempotent
        # across a labeler-tag bump. astype("string") + na=False: a missing
        # labeler (hand-edited/legacy CSV, possibly an all-NaN float column) is
        # not a panel label — keep the row instead of raising.
        labeler = human_df["labeler"].astype("string")
        human_df = human_df[~labeler.str.startswith(PANEL_LABELER_PREFIX, na=False)]
    human_gids = set(human_df["group_id"].astype(str)) if not human_df.empty else set()

    # Metadata + candidate edges for auto-accept groups (for the class gate and
    # the human edge-overlap mapping, reusing the eval module's approach).
    candidate_metas: dict[str, dict] = {}
    candidate_edges: dict[str, frozenset] = {}
    for gid, (bd, row) in merged.items():
        if str(row.get("routing")) != "auto_accept":
            continue
        grp = batch_groups.get(bd, {}).get(gid)
        # Prefer per-group metadata.yaml; fall back to batch.json (ids + classes)
        # so the class gate and overlap mapping still run when it is absent.
        gdir = bd / gid
        if (gdir / "metadata.yaml").exists():
            candidate_metas[gid] = _load_group_metadata(gdir)
        else:
            meta = _meta_from_group(grp) if grp is not None else None
            if meta is not None:
                candidate_metas[gid] = meta
        if grp is not None:
            candidate_edges[gid] = frozenset(
                (str(e["ref_id"]), str(e["target_id"])) for e in grp.get("edges", [])
            )

    overlap_map: dict[str, str] = {}
    if not human_df.empty:
        overlap_map = map_human_labels_to_groups(human_df, candidate_metas, candidate_edges)

    # SET-semantics human labels carry no edges, so the edge-overlap mapping above
    # cannot see them. A set label still means "a human reviewed this group", so
    # it MUST confer precedence: map set rows to candidate groups by MEMBERSHIP
    # overlap and merge the result (an edge-overlap match wins on a clash — it is
    # the stronger signal). Precedence is about the human having reviewed the
    # group, not about pair-level detail.
    if not human_df.empty and "label_semantics" in human_df.columns:
        set_rows = human_df[human_df.apply(_is_set_label, axis=1)]
        if not set_rows.empty:
            group_members = {
                gid: frozenset(r for r, _ in edges) | frozenset(t for _, t in edges)
                for gid, edges in candidate_edges.items()
            }
            # {human_gid: panel_gid} -> invert to {panel_gid: human_gid}, not
            # clobbering an existing (edge-overlap) mapping for that panel group.
            for hgid, pgid in _map_set_labels_to_groups(set_rows, group_members).items():
                overlap_map.setdefault(pgid, hgid)

    groups: list[GroupExport] = []
    n_auto = 0
    for gid, (bd, row) in sorted(merged.items()):
        # Gate (a): only unanimous auto-accept rows are candidates.
        if str(row.get("routing")) != "auto_accept":
            continue
        n_auto += 1
        groups.append(
            _gate_group(
                gid=gid,
                bd=bd,
                row=row,
                grp=batch_groups.get(bd, {}).get(gid, {}),
                meta=candidate_metas.get(gid),
                human_gids=human_gids,
                overlap_map=overlap_map,
                max_edges=max_edges,
                max_assignment_components=max_assignment_components,
                soft_max_edges=soft_max_edges,
                backstop_max_edges=backstop_max_edges,
            )
        )

    return ExportReport(
        dataset=dataset,
        n_total_groups=len(merged),
        n_auto_accept=n_auto,
        groups=groups,
    )


def _gate_group(
    gid: str,
    bd: Path,
    row: dict,
    grp: dict,
    meta: dict | None,
    human_gids: set[str],
    overlap_map: dict[str, str],
    max_edges: int,
    max_assignment_components: int,
    soft_max_edges: int,
    backstop_max_edges: int,
) -> GroupExport:
    """Apply gates (b)-(e) to one auto-accept group and return its outcome.

    The first failing gate decides the group. Gate (a) (auto_accept) is applied
    by the caller before this is invoked.
    """
    match_type = str(grp.get("match_type") or (meta.get("match_type") if meta else "") or "")
    edges_pairs = _parse_edge_set_pairs(row.get("edge_set"))
    n_raw = len(edges_pairs)
    try:
        mean_conf = float(row.get("mean_confidence") or 0.0)
    except (ValueError, TypeError):
        mean_conf = 0.0

    def _mk(reason: str, **kw) -> GroupExport:
        return GroupExport(
            group_id=gid,
            source_batch=bd.name,
            exported=(reason == REASON_EXPORTED),
            reason=reason,
            match_type=match_type,
            n_edges_raw=n_raw,
            mean_confidence=mean_conf,
            **kw,
        )

    # Gate (b): size gate. Prefer the structural gate when the group carries
    # structure fields (single corridor / few assignment-components within a
    # soft budget, under a hard backstop). Fall back to the flat edge cap on the
    # selected edge set for older batch.json packs without structure fields.
    n_group_edges = grp.get("n_edges")
    n_corridors = grp.get("n_corridors")
    n_assign = grp.get("n_assignment_components")
    if n_group_edges is None:
        n_group_edges = len(grp.get("edges", [])) or n_raw
    if n_corridors is not None and n_assign is not None:
        simple = group_is_structurally_simple(
            int(n_corridors),
            int(n_assign),
            int(n_group_edges),
            max_assignment_components,
            soft_max_edges,
            backstop_max_edges,
        )
        if not simple:
            reason = (
                REASON_OVER_MAX
                if int(n_group_edges) > backstop_max_edges
                else REASON_STRUCTURAL_TANGLE
            )
            return _mk(reason, n_edges_final=n_raw)
    elif n_raw > max_edges:
        return _mk(REASON_OVER_MAX, n_edges_final=n_raw)

    # Gate (c): class-consistency (any cross-mode pair: pedestrian/vehicular/bike).
    if meta is not None:
        ref_c, tgt_c = _segment_class_maps(meta)
        edge_classes = _edge_classes_for(frozenset(edges_pairs), ref_c, tgt_c)
        if has_cross_mode_edge(edge_classes):
            return _mk(REASON_CLASS_MISMATCH, n_edges_final=n_raw)

    # Gate (d): sliver canonicalization.
    sliver_pairs = _group_sliver_pairs(grp)
    final_pairs = [p for p in edges_pairs if p not in sliver_pairs]
    n_slivers = n_raw - len(final_pairs)
    if not final_pairs:
        return _mk(REASON_EMPTIED_BY_SLIVER, n_slivers_dropped=n_slivers, n_edges_final=0)

    # Gate (e): human precedence (exact group_id or edge-overlap).
    if gid in human_gids or gid in overlap_map:
        return _mk(
            REASON_HUMAN_PRECEDENCE,
            n_slivers_dropped=n_slivers,
            n_edges_final=len(final_pairs),
            human_group_id=(gid if gid in human_gids else overlap_map[gid]),
        )

    selected = [{"ref_id": r, "target_id": t} for r, t in final_pairs]
    return _mk(
        REASON_EXPORTED,
        n_slivers_dropped=n_slivers,
        n_edges_final=len(final_pairs),
        selected_edges=selected,
    )


def write_exports(
    report: ExportReport,
    dataset: str,
    labels_dir: Path,
) -> int:
    """Persist the report's exported groups as ``panel_unanimous_v3`` labels.

    Upserts by ``group_id`` (the store replaces an existing row for the same
    group_id), so this is idempotent. The source batch name is recorded in the
    ``session_id`` field for provenance. Returns the number of rows written.
    """
    store = StitchingLabelStore(dataset, labels_dir=labels_dir)
    written = 0
    for g in report.exported:
        ref_ids = {e["ref_id"] for e in g.selected_edges}
        tgt_ids = {e["target_id"] for e in g.selected_edges}
        store.add(
            group_id=g.group_id,
            selected_edges=g.selected_edges,
            match_type=g.match_type,
            num_refs=len(ref_ids),
            num_targets=len(tgt_ids),
            labeler=PANEL_LABELER,
            session_id=g.source_batch,
        )
        written += 1
    return written


def write_vote_provenance(
    batch_dirs: list[Path],
    dataset: str,
    votes_dir: Path = Path("labels/votes"),
) -> tuple[int, int]:
    """Snapshot raw panel ballots + consensus into a git-tracked location.

    The panel writes ``votes.csv`` (every raw ballot) and ``consensus.csv`` per
    batch, but those live under the batch dir in the git-ignored ``data/`` tree,
    so the audit trail behind every exported label is never committed. This
    copies them into ``labels/votes/dataset=<dataset>/`` — which *is* tracked —
    tagging each row with a ``source_batch`` column for cross-batch traceability.

    **Accumulates** like the label store: the existing archived files are read
    back and merged with the current batches, so exporting batches across
    *separate* invocations (``stitch-export -b b1`` then later ``-b b2``) never
    drops earlier ballots — matching ``write_exports``, which only upserts the
    labels for the groups in each run. Idempotent via dedup (votes by
    ``source_batch``/``group_id``/``provider``, consensus by
    ``source_batch``/``group_id``; ``keep="last"`` lets a re-run refresh a batch).

    ``source_batch`` is the batch dir *basename*, so duplicate basenames in one
    call would collapse distinct ballots — we refuse that rather than lose data.
    Field-level best-effort: an empty or malformed batch CSV is skipped, not
    fatal. Returns ``(n_vote_rows, n_consensus_rows)``.
    """
    out_dir = Path(votes_dir) / f"dataset={dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [Path(bd).name for bd in batch_dirs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"batch dirs have duplicate basenames {dupes}; the source_batch tag "
            "would collapse them and lose ballots — pass uniquely-named batch dirs."
        )

    def _read_csv(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, dtype={"group_id": str})
        except pd.errors.EmptyDataError:
            return None

    def _collect(filename: str, dedupe_on: list[str]) -> int:
        frames = []
        # Existing archive first (already carries source_batch); current batches
        # append after it, so keep="last" lets a re-run supersede a stale batch.
        existing = _read_csv(out_dir / filename)
        if existing is not None:
            frames.append(existing)
        for bd in batch_dirs:
            df = _read_csv(Path(bd) / filename)
            if df is None:
                continue
            if "source_batch" in df.columns:  # a re-archived tree; re-tag cleanly
                df = df.drop(columns="source_batch")
            df.insert(0, "source_batch", Path(bd).name)
            frames.append(df)
        if not frames:
            return 0
        merged = pd.concat(frames, ignore_index=True)
        keys = [c for c in dedupe_on if c in merged.columns]
        if keys:
            merged = merged.drop_duplicates(subset=keys, keep="last")
        merged.to_csv(out_dir / filename, index=False)
        return len(merged)

    n_votes = _collect("votes.csv", ["source_batch", "group_id", "provider"])
    n_consensus = _collect("consensus.csv", ["source_batch", "group_id"])
    return n_votes, n_consensus
