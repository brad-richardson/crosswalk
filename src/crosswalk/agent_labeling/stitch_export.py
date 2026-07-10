"""Export unanimous LLM-panel consensus into human-equivalent stitching labels.

The 3-provider stitching panel (see :mod:`stitch_runner`) votes on M:N group edge
selections and writes a ``consensus.csv`` per batch. This module promotes the
subset of those verdicts that are safe to treat as durable labels into
``labels/stitching`` alongside the human labels, tagged with a ``panel_*``
labeler so their provenance stays visible (v1 tagged the earlier
sonnet/gpt-5.4/Gemini-Flash-Low panel; v2 the Opus 4.8/gpt-5.5/Gemini-3.5-Flash
panel on pre-enrichment packs; v3 that composition on #302-enriched packs; v4
the 2026-07-09 bless — Opus 4.8 / gpt-5.6-sol / Kimi K2.6. The tag is bumped
whenever the panel composition OR its pack inputs change, and each batch is
stamped with ITS OWN era's tag — see :data:`STANDARD_PANEL_VOTERS`).

Two verdict classes are promoted:

  * **Unanimous accept** (routing == ``auto_accept``) -> a normal pair label with
    the panel's chosen edge set, tagged ``panel_unanimous_v4`` (v3-era batches:
    ``panel_unanimous_v3``).
  * **Unanimous NONE** (all panelists voted "none of the options fit"; routed to
    ``human_review`` with route_reason ``unanimous_none``) -> an EMPTY-SET pair
    label (``selected_edges == []``), tagged ``panel_unanimous_none_v4``. This is
    the reject-all ground truth the learned group resolver needs to train/eval on
    rejects (see ``research/learned_optimizer_design.md`` §2.4a / milestone L1):
    the cross-mode defect (a cycleway wrongly grouped with a parallel road) has
    exactly this shape -- the correct answer is "select nothing". Empty-set export
    is on by default; pass ``export_empty_set=False`` (CLI ``--no-empty-set``) to
    skip it.

A third class covers DECOMPOSED groups (#367 Mode B, ``stitch-batch
--decompose``): an over-backstop group split into panel-sized sub-problems is
recomposed here — a whole-group label (labeler ``panel_unanimous_decomposed_v4``,
the union of the sub-selections) is minted ONLY when every sub-problem in the
batch.json roster resolved as a unanimous accept; any failed or unvoted
sub-problem blocks the group (``subproblem_failed`` / ``subproblems_unvoted``),
as does a sub-verdict set whose contributing batch dirs resolve to different
panel eras (``subproblem_era_mixed`` — a mixed-composition union must not be
stamped under a single era). Sub-problem consensus rows are consumed by that
recomposition and never export individually. See
:mod:`crosswalk.matching.group_decomposition`.

The empty-set label uses the SAME on-disk representation as a human reject-all
review (PAIR semantics, ``selected_edges == "[]"``, ``num_refs/num_targets == 0``),
so it round-trips through every consumer that already handles reject-all human
labels -- ``stitch_eval.recover_labeled_groups`` (``empty`` bucket),
``recover_empty_reject_all`` / ``mbench.map_labels_to_groups`` (verbatim group_id
recovery), the ``edge_prf`` empty-vs-empty perfect score, and
``render_review_diffs`` (``is_reject_all``).

Gates (applied in order; the first failing gate decides the group and is
reported):

  a. routing gate -- an ``auto_accept`` row takes the accept path; a
     ``unanimous_none`` row takes the empty-set path (when ``export_empty_set``);
     everything else is not a candidate at all.
  b. size gate -- accept groups: huge/tangled groups stay for human review
     (structural gate, or flat ``max_edges`` fallback). Empty-set groups apply
     ONLY the hard backstop ceiling on the group's candidate size (see
     :func:`_gate_empty_group`): the corridor/assignment-tangle sub-gate targets
     *selection greediness*, which is irrelevant to an empty selection, and it
     would wrongly block the 2-corridor cross-mode reject this path exists to
     capture.
  c. class-consistency gate -- reuses the panel runner's cross-mode rule
     (:func:`stitch_runner.has_cross_mode_edge`) on the chosen edge set
     (pedestrian / vehicular / bike; any two different modes are cross-mode).
     Vacuous on an empty set (no edges to gate), so it is skipped on the
     empty-set path -- correctly, since the whole point is to record cross-mode
     rejects.
  d. sliver canonicalization -- drops junction-sliver edges (shared definition in
     :func:`crosswalk.matching.sliver`); if that empties the set the accept group
     is skipped. Skipped on the empty-set path (the emptiness is intentional, not
     a sliver artifact).
  e. human precedence -- a group already covered by a *human* label (by exact
     group_id or by edge-overlap, reusing :func:`stitch_eval.map_human_labels_to_groups`)
     is left untouched. Applies to both paths.

Writing is idempotent: rows are upserted by ``group_id`` under the appropriate
``panel_*`` labeler, so re-running never duplicates and always refreshes to the
latest consensus. Previously exported panel rows (any ``panel_`` prefix) are
excluded from the human-precedence check (they are not human), so re-runs stay
accurate.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import settings
from ..labeling.stitching_store import StitchingLabelStore
from ..matching.group_decomposition import (
    STATUS_FAILED,
    STATUS_UNVOTED,
    Recomposition,
    recompose_subproblem_verdicts,
)
from ..matching.optimizer import group_is_structurally_simple
from ..matching.sliver import annotate_group_sliver_flags
from .panel_routing import REASON_UNANIMOUS_NONE, _int_or_none, derive_route_reason
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
# research/panel_enriched_ab.md); v3 -> v4 when the composition changed again
# (2026-07-09 bless: agy/Gemini replaced by opencode/Kimi K2.6, codex bumped
# gpt-5.5 -> gpt-5.6-sol; validated in #397). Existing v1/v2/v3 labels stay
# untouched — the v3 constants below remain the write-time tags for v3-era
# batches (see :data:`STANDARD_PANEL_VOTERS` era scoping) — and new default-
# panel waves are tagged v4. Any labeler with the PANEL_LABELER_PREFIX is a
# panel (non-human) label and is excluded from the human-precedence check below.
PANEL_LABELER_V3 = "panel_unanimous_v3"
PANEL_NONE_LABELER_V3 = "panel_unanimous_none_v3"
PANEL_DECOMPOSED_LABELER_V3 = "panel_unanimous_decomposed_v3"

PANEL_LABELER = "panel_unanimous_v4"
PANEL_LABELER_PREFIX = "panel_"

# Distinct labeler for unanimous-NONE (reject-all / empty-set) verdicts. Kept
# UNDER the ``panel_`` prefix so every consumer that buckets labels by that
# prefix -- the human-precedence filter here, ``mbench.eval.stitch_metrics``
# (``_labeler_class`` -> "panel"), ``xprod``/``cli.data`` -- still classes it as
# a non-human panel label. A *separate* tag (rather than reusing PANEL_LABELER)
# keeps reject-all ground truth sliceable on its own in per-labeler eval: it is
# semantically a different verdict (select nothing vs select a subset) and the
# cross-mode acceptance test reports rejects as their own table
# (research/learned_optimizer_design.md §6.3). Version suffix tracks PANEL_LABELER
# (same panel composition / pack inputs).
PANEL_NONE_LABELER = "panel_unanimous_none_v4"

# Distinct labeler for a RECOMPOSED whole-group label (#367 Mode B): the union
# of unanimous per-sub-problem verdicts from a decomposed over-backstop group.
# No single panel saw the whole group, so this is a different labeling process
# than PANEL_LABELER and must stay sliceable on its own in per-labeler eval.
# Kept under the ``panel_`` prefix (non-human); version suffix tracks
# PANEL_LABELER (same panel composition / pack inputs per sub-problem).
PANEL_DECOMPOSED_LABELER = "panel_unanimous_decomposed_v4"

#: Blessed (provider, model) voter compositions, keyed by labeler era. The gate
#: keys on the PAIR, not the provider name alone: opencode has driven Gemini
#: Flash (no-agy quota-outage waves), Qwen3-VL (v3-candidate), and Kimi K2.6
#: (the blessed v4 voter), and a provider-name-only set cannot tell them apart.
PANEL_VOTERS_V3 = frozenset(
    {
        ("claude", "claude-opus-4-8"),
        ("codex", "gpt-5.5"),
        ("agy", "Gemini 3.5 Flash (Medium)"),
    }
)
PANEL_VOTERS_V4 = frozenset(
    {
        ("claude", "claude-opus-4-8"),
        ("codex", "gpt-5.6-sol"),
        ("opencode", "openrouter/moonshotai/kimi-k2.6"),
    }
)

#: Era -> blessed voter set. A batch matching an era's set exactly is STANDARD
#: for that era: it passes the export gate and its labels are stamped with that
#: era's labeler tags (v3-era batches keep minting ``*_v3`` labels on
#: re-export — the committed v3 history is never retroactively flagged as
#: nonstandard, nor silently re-stamped ``*_v4``).
STANDARD_PANEL_VOTERS: dict[str, frozenset[tuple[str, str]]] = {
    "v3": PANEL_VOTERS_V3,
    "v4": PANEL_VOTERS_V4,
}

#: STAMPING-ONLY historical compositions -> labeler era. These compositions
#: were NEVER a blessed default — they still fail the export gate and require
#: ``--allow-nonstandard-panel``, exactly as when they were first exported —
#: but they DID mint committed labels under a specific era via an explicit
#: operator decision, so era resolution must keep attributing them to that
#: era: a re-export must re-stamp the SAME tag, never drift to the current one.
#:
#: * 2026-07-07 8-dataset wave (commit 80dbe1f, #356): the Gemini third voter
#:   ran via the opencode transport instead of agy (Brad-approved swap during
#:   the agy quota outage) — same model, different provider string. Six
#:   datasets' committed ``*_v3`` labels trace to this composition (pairs
#:   verified against ``labels/votes/dataset=*/votes.csv``).
HISTORICAL_ERA_VOTERS: dict[frozenset[tuple[str, str]], str] = {
    frozenset(
        {
            ("claude", "claude-opus-4-8"),
            ("codex", "gpt-5.5"),
            ("opencode", "openrouter/google/gemini-3.5-flash"),
        }
    ): "v3",
}

#: The (provider, model) composition of the CURRENT default panel — must stay
#: in lockstep with ``stitch_runner.DEFAULT_PANEL`` (asserted in
#: tests/unit/test_stitch_export.py, so a panel change without a provenance
#: decision here fails CI). Replaces the provider-name-only
#: ``DEFAULT_PANEL_PROVIDERS``.
DEFAULT_PANEL_VOTERS = PANEL_VOTERS_V4


def _batch_voters(batch_dir: Path) -> set[tuple[str, str]] | None:
    """Read a batch's ``votes.csv`` into its set of (provider, model) voters.

    Returns ``None`` when ``votes.csv`` is missing, unreadable, or has no rows
    (the CLI already hard-requires ``consensus.csv``; provenance for such
    batches is best-effort). A missing ``model`` column or a blank/NaN model
    cell yields pairs with ``model == ""`` — deliberately KEPT rather than
    skipped, so incomplete provenance reads as a composition mismatch (flagged
    by :func:`nonstandard_panel_batches`), never as the blessed panel and never
    as a crash.
    """
    votes_path = Path(batch_dir) / "votes.csv"
    if not votes_path.exists():
        return None
    try:
        df = pd.read_csv(votes_path)
        providers = df["provider"].fillna("").astype(str).str.strip()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError):
        return None
    if "model" in df.columns:
        models = df["model"].fillna("").astype(str).str.strip()
    else:
        models = pd.Series([""] * len(df), index=df.index, dtype=str)
    voters = set(zip(providers, models, strict=True))
    return voters or None


def batch_panel_era(batch_dir: Path) -> str | None:
    """Return the labeler era ("v3"/"v4") a batch's voter composition belongs to.

    Resolution order: the blessed sets (:data:`STANDARD_PANEL_VOTERS`), then the
    stamping-only historical map (:data:`HISTORICAL_ERA_VOTERS`) — compositions
    that minted committed labels under an era via an explicit operator decision
    but were never a blessed default. The historical map affects STAMPING only;
    the export gate (:func:`nonstandard_panel_batches`) still flags those
    batches, exactly as it did when they were first exported.

    ``None`` means the batch is attributable to NO known era: an unknown
    composition, or no readable ``votes.csv``. Such batches are refused at
    write time unless the operator declares an era explicitly
    (``plan_exports(stamp_era=...)`` / CLI ``--stamp-era``); they never
    silently default to the current era.
    """
    voters = _batch_voters(batch_dir)
    if voters is None:
        return None
    for era, blessed in STANDARD_PANEL_VOTERS.items():
        if voters == blessed:
            return era
    return HISTORICAL_ERA_VOTERS.get(frozenset(voters))


def nonstandard_panel_batches(
    batch_dirs: list[Path],
    expected: frozenset[tuple[str, str]] | None = None,
) -> dict[str, set[tuple[str, str]]]:
    """Return ``{batch_name: voter_pairs}`` for batches with a nonstandard panel.

    Reads each batch's ``votes.csv`` (provider, model) pairs and flags any batch
    whose voter set is not a blessed composition. With the default
    ``expected=None``, a batch passes when it exactly matches ANY era's blessed
    set (:data:`STANDARD_PANEL_VOTERS`): historical v3 batches (claude+codex+agy
    with their v3 models) stay standard instead of being retroactively flagged,
    and v4 batches validate against the v4 set. Pass an explicit ``expected``
    frozenset of (provider, model) pairs to pin a single composition.

    Keying on the pair (not the provider name) is the point of this gate: a
    batch voted by opencode/Kimi (blessed) is now distinguishable from
    opencode/Gemini or opencode/Qwen (not blessed). A vote row with a
    blank/missing model reads as ``(provider, "")`` and therefore flags — it is
    never treated as standard and never a crash. Batches with a
    missing/unreadable ``votes.csv`` are skipped (the CLI already hard-requires
    ``consensus.csv``; provenance for such batches is best-effort).
    """
    accepted = (
        [frozenset(expected)] if expected is not None else list(STANDARD_PANEL_VOTERS.values())
    )
    offending: dict[str, set[tuple[str, str]]] = {}
    for bd in batch_dirs:
        voters = _batch_voters(bd)
        if voters is None:
            continue
        if not any(voters == blessed for blessed in accepted):
            offending[Path(bd).name] = voters
    return offending


# Per-group outcome reasons (stable strings for reporting/tests).
REASON_EXPORTED = "exported"
REASON_OVER_MAX = "over_max_edges"
REASON_STRUCTURAL_TANGLE = "structural_tangle"
REASON_CLASS_MISMATCH = "class_mismatch"
REASON_EMPTIED_BY_SLIVER = "emptied_by_sliver"
REASON_HUMAN_PRECEDENCE = "human_precedence"
# Decomposed-group (recomposition) outcomes: a sub-problem the panel could not
# unanimously accept, or one never voted (including size-gated irreducible
# blocks), blocks the whole-group label.
REASON_SUBPROBLEM_FAILED = "subproblem_failed"
REASON_SUBPROBLEMS_UNVOTED = "subproblems_unvoted"
# The batch dirs contributing a parent's consumed sub-problem verdicts resolve
# to DIFFERENT panel eras (e.g. a v3-era wave completed by post-bless v4
# re-votes): the union label would mix compositions under a single era tag, so
# the group is blocked rather than stamped.
REASON_SUBPROBLEM_ERA_MIXED = "subproblem_era_mixed"


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
    # True for a unanimous-NONE (reject-all) export: selected_edges is empty and
    # the row is stamped PANEL_NONE_LABELER. False for a normal accept export.
    is_empty_set: bool = False
    # True for a recomposed decomposed-group outcome (#367 Mode B): the group's
    # verdict is the union of per-sub-problem panel votes; an export is stamped
    # PANEL_DECOMPOSED_LABELER.
    from_decomposition: bool = False
    n_subproblems: int = 0
    n_subproblems_resolved: int = 0
    # Labeler era of the SOURCE BATCH ("v3"/"v4", from batch_panel_era or an
    # explicit stamp_era), or "" when the batch matches no known composition.
    # write_exports stamps each era's own labeler tags and REFUSES to write an
    # exported group with era "" — an unknown composition never silently mints
    # the current era's provenance (declare one via plan_exports(stamp_era=...)
    # / CLI --stamp-era). Recomposed groups resolve era across every batch dir
    # contributing a consumed sub-problem verdict.
    panel_era: str = ""


@dataclass
class ExportReport:
    """Full plan of what would be / was written, plus tallies for reporting."""

    dataset: str
    n_total_groups: int
    n_auto_accept: int
    groups: list[GroupExport]
    # Count of unanimous-NONE candidate groups seen (the empty-set path's analog
    # of ``n_auto_accept``). Zero when ``export_empty_set`` is off.
    n_unanimous_none: int = 0
    # Decomposition (#367 Mode B): consensus rows consumed as sub-problem
    # verdicts (never exported individually), and parents with a roster.
    n_subproblem_rows: int = 0
    n_decomposed_parents: int = 0

    @property
    def exported(self) -> list[GroupExport]:
        return [g for g in self.groups if g.exported]

    @property
    def exported_empty(self) -> list[GroupExport]:
        """Exported reject-all (empty-set) groups."""
        return [g for g in self.groups if g.exported and g.is_empty_set]

    @property
    def exported_nonempty(self) -> list[GroupExport]:
        """Exported normal (non-empty accept) groups."""
        return [g for g in self.groups if g.exported and not g.is_empty_set]

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


def _is_unanimous_none(row: dict) -> bool:
    """True when a consensus row is a unanimous-NONE (reject-all) verdict.

    Reuses the shared route-reason derivation so it matches both freshly-stamped
    rows (``route_reason == "unanimous_none"``) and historical waves that predate
    the stamp (derived from ``consensus == "unanimous"`` + ``choice == "NONE"``).
    A unanimous-NONE row routes to ``human_review`` (it is never ``auto_accept``),
    so it is disjoint from the accept path.

    Defense-in-depth quorum check (this path mints reject ground truth): the
    derivation trusts the ``consensus`` column verbatim, but "unanimous" is only
    meaningful with a full quorum (``compute_consensus`` requires >= 3 agreeing
    valid votes). A hand-edited or pre-quorum-rule historical row claiming
    ``consensus=unanimous`` with ``n_valid < 3`` must not be exported, so:

    * ``n_valid`` present -> require ``n_valid >= 3`` (contradicting evidence
      blocks the export even when a ``route_reason`` stamp is present);
    * ``n_valid`` missing/unparseable -> conservatively require the explicit
      ``route_reason`` stamp (written only by ``compute_consensus``, which
      enforces the quorum) rather than deriving from consensus/choice alone.
    """
    if derive_route_reason(row) != REASON_UNANIMOUS_NONE:
        return False
    n_valid = _int_or_none(row.get("n_valid"))
    if n_valid is not None:
        return n_valid >= 3
    # No n_valid evidence: trust only the compute_consensus stamp (accepting the
    # legacy "unanimous_NONE" spelling normalized by derive_route_reason).
    stamp = str(row.get("route_reason") or "").strip()
    return stamp in (REASON_UNANIMOUS_NONE, "unanimous_NONE")


def plan_exports(
    batch_dirs: list[Path],
    dataset: str,
    labels_dir: Path,
    max_edges: int = 20,
    max_assignment_components: int | None = None,
    soft_max_edges: int | None = None,
    backstop_max_edges: int | None = None,
    export_empty_set: bool = True,
    stamp_era: str | None = None,
) -> ExportReport:
    """Run the export gates over merged consensus and return a full plan.

    Pure w.r.t. the label store: reads human labels but writes nothing. Call
    :func:`write_exports` with the returned report to persist.

    ``stamp_era`` ("v3"/"v4") is a FILL-IN for era-less batches only: each
    batch is first resolved via :func:`batch_panel_era`, and ``stamp_era``
    applies solely to batches whose composition resolves to NO era (unknown
    compositions, which :func:`write_exports` otherwise refuses). A batch that
    genuinely resolves to an era always keeps it — a mixed run of one
    era-less and one blessed-v4 batch with ``stamp_era="v3"`` stamps v3 only
    on the era-less one, never re-stamping the v4 batch. It should accompany
    ``--allow-nonstandard-panel``-style explicit provenance decisions only.
    ``None`` (default) leaves era-less batches unresolved.

    The size gate is *structural*, not a flat edge count: a group auto-exports
    when it is a single corridor-pair OR has few assignment-components within a
    soft edge budget (so a clean 30-edge single corridor is exportable, while a
    small two-highway tangle is blocked). A hard backstop ceiling blocks
    anything larger regardless — a defence against a structure-detection bug,
    not the primary gate. When a group lacks structure fields (an older
    batch.json predating the structure sidecar) the gate falls back to the flat
    ``max_edges`` cap on the selected edge set PLUS the hard backstop on the
    group's candidate count, so the backstop invariant (no over-backstop group
    ever auto-exports) holds on the legacy path too.

    When ``export_empty_set`` (default), unanimous-NONE groups (all panelists
    voted "none of the options fit") additionally become EMPTY-SET candidates:
    reject-all pair labels with ``selected_edges == []`` (see
    :func:`_gate_empty_group`). Set it False to plan the accept path only.
    """
    if max_assignment_components is None:
        max_assignment_components = settings.stitch_export_max_assignment_components
    if soft_max_edges is None:
        soft_max_edges = settings.stitch_export_soft_max_edges
    if backstop_max_edges is None:
        backstop_max_edges = settings.stitch_export_backstop_max_edges
    if stamp_era is not None and stamp_era not in LABELERS_BY_ERA:
        raise ValueError(f"stamp_era must be one of {sorted(LABELERS_BY_ERA)}, got {stamp_era!r}")

    # Era stamping (and vote-provenance archival) keys batches by BASENAME, so
    # duplicate basenames would mis-attribute one dir's era/rows to another
    # (last-dir-wins). Refuse up front rather than plan on ambiguous identity.
    names = [Path(bd).name for bd in batch_dirs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"batch dirs have duplicate basenames {dupes}; era stamping and "
            "vote-provenance archival key on the basename and would collapse "
            "them — pass uniquely-named batch dirs."
        )
    merged = _merge_consensus(batch_dirs)

    # Load batch.json groups (geometries/edges) once per distinct batch dir,
    # preserving the caller's precedence order (later supersedes earlier).
    # batch.json is the ONLY source of geometries (sliver gate) and candidate
    # edges (overlap precedence); warn loudly if a batch dir lacks it, since
    # those gates then degrade rather than fail.
    batch_groups: dict[Path, dict[str, dict]] = {}
    for bd in dict.fromkeys(Path(b) for b in batch_dirs):
        if not (bd / "batch.json").exists():
            logger.warning(
                f"No batch.json in {bd}: sliver canonicalization and edge-overlap "
                "precedence cannot run for its groups (gates degrade)."
            )
        batch_groups[bd] = _load_batch_groups(bd)

    # Decomposition rosters (#367 Mode B): a parent entry written by
    # ``stitch-batch --decompose`` carries the FULL sub-problem roster
    # (including size-gated oversized sub-problems that were never packed).
    # Later batch dirs supersede earlier ones per parent, mirroring
    # ``_merge_consensus`` precedence. Every roster sub-problem id maps to its
    # parent so its consensus rows are consumed by recomposition below and are
    # never exported individually (a sub-problem id is not a sidecar group).
    rosters: dict[str, tuple[Path, dict]] = {}
    for bd in batch_groups:
        for gid, grp in batch_groups[bd].items():
            if grp.get("decomposed_parent") and grp.get("subproblem_ids"):
                rosters[gid] = (bd, grp)
    sub_to_parent: dict[str, str] = {
        str(sid): parent_gid
        for parent_gid, (_bd, grp) in rosters.items()
        for sid in grp["subproblem_ids"]
    }
    # Defense in depth: ANY batch group carrying ``parent_group_id`` is a
    # sub-problem, even when a newer decomposition's roster no longer lists it
    # (e.g. the budget changed between waves). Such an orphaned sub row is
    # consumed (skipped) rather than falling through to the normal loop, where
    # its auto_accept could mint a label under a non-sidecar sub-problem id.
    for bd in batch_groups:
        for gid, grp in batch_groups[bd].items():
            if grp.get("parent_group_id"):
                sub_to_parent.setdefault(str(gid), str(grp["parent_group_id"]))

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

    # Metadata + candidate edges for candidate groups (for the class gate and
    # the human edge-overlap mapping, reusing the eval module's approach).
    # Includes unanimous-NONE groups when empty-set export is on, so their
    # human-precedence edge-overlap check has the group's candidate edges.
    candidate_metas: dict[str, dict] = {}
    candidate_edges: dict[str, frozenset] = {}
    for gid, (bd, row) in merged.items():
        if gid in sub_to_parent:
            continue  # sub-problem rows are recomposition inputs, not candidates
        is_accept = str(row.get("routing")) == "auto_accept"
        is_none = export_empty_set and _is_unanimous_none(row)
        if not (is_accept or is_none):
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

    # Decomposed parents are recomposition candidates: include their metadata
    # (class gate) and candidate edges (human edge-overlap / set-membership
    # precedence) exactly like directly-voted candidate groups. Parents carry
    # no evidence pack, so batch.json is the metadata source.
    for parent_gid, (bd, grp) in sorted(rosters.items()):
        gdir = bd / parent_gid
        if (gdir / "metadata.yaml").exists():
            candidate_metas[parent_gid] = _load_group_metadata(gdir)
        else:
            meta = _meta_from_group(grp)
            if meta is not None:
                candidate_metas[parent_gid] = meta
        candidate_edges[parent_gid] = frozenset(
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
    n_none = 0
    n_sub_rows = 0
    for gid, (bd, row) in sorted(merged.items()):
        # Sub-problem rows (#367 Mode B) are consumed by the recomposition path
        # below — a sub-problem id is not a sidecar group and must never mint a
        # label of its own (not even an empty-set one).
        if gid in sub_to_parent:
            n_sub_rows += 1
            continue
        # Gate (a): route each candidate row to its path. auto_accept -> accept
        # gates; unanimous-NONE -> empty-set gates (when enabled). Everything
        # else is not a candidate at all.
        if str(row.get("routing")) == "auto_accept":
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
        elif export_empty_set and _is_unanimous_none(row):
            n_none += 1
            groups.append(
                _gate_empty_group(
                    gid=gid,
                    bd=bd,
                    row=row,
                    grp=batch_groups.get(bd, {}).get(gid, {}),
                    meta=candidate_metas.get(gid),
                    human_gids=human_gids,
                    overlap_map=overlap_map,
                    max_edges=max_edges,
                    backstop_max_edges=backstop_max_edges,
                )
            )

    # Per-dir labeler era ("v3"/"v4"; "" for unattributable). stamp_era is a
    # FILL-IN for dirs that resolve to no era — a genuinely-resolved batch
    # always keeps its own era (resolution first), so an operator passing
    # --stamp-era v3 for one era-less batch can never re-stamp a blessed-v4
    # batch in the same run. Resolved once; used for both the direct groups
    # (by source batch) and the recomposition era-mix check below.
    era_by_dir: dict[Path, str] = {
        bd: (batch_panel_era(bd) or stamp_era or "") for bd in batch_groups
    }

    # Recomposition (#367 Mode B): one outcome per decomposed parent, from its
    # sub-problem verdicts. Conservative all-or-nothing rule — see
    # :func:`_gate_recomposed_group`. When an OLD wave also voted the parent
    # directly, its (size-gated) outcome above coexists in the report; a later
    # recomposed export for the same group_id supersedes it at write time
    # (``write_exports`` upserts by group_id in report order).
    for parent_gid, (bd, grp) in sorted(rosters.items()):
        roster_ids = [str(s) for s in grp["subproblem_ids"]]
        verdicts: dict[str, tuple[str, list[tuple[str, str]]]] = {}
        sub_confs: list[float] = []
        for sid in roster_ids:
            if sid not in merged:
                continue
            srow = merged[sid][1]
            verdicts[sid] = (
                str(srow.get("routing", "")),
                _parse_edge_set_pairs(srow.get("edge_set")),
            )
            if str(srow.get("routing")) == "auto_accept":
                with suppress(ValueError, TypeError):
                    sub_confs.append(float(srow.get("mean_confidence") or 0.0))
        rec = recompose_subproblem_verdicts(parent_gid, roster_ids, verdicts)
        # Weakest-link confidence: the minimum accepted sub-panel mean, so a
        # reviewer sees the least-certain sub-decision.
        min_conf = min(sub_confs) if sub_confs else 0.0

        # Era of the recomposed label = the era of the batch dirs whose
        # consensus rows the recomposition CONSUMED (merged precedence), not
        # the roster dir's: a v3-era wave whose failed sub-problems were
        # re-voted post-bless in a v4 batch dir would otherwise stamp a
        # mixed-composition union under a single era. On a mismatch the group
        # is blocked (subproblem_era_mixed) — never stamped.
        sub_dirs = {merged[sid][0] for sid in roster_ids if sid in merged}
        sub_eras = {era_by_dir.get(d, "") for d in (sub_dirs or {bd})}
        if len(sub_eras) > 1:
            groups.append(
                GroupExport(
                    group_id=parent_gid,
                    source_batch=bd.name,
                    exported=False,
                    reason=REASON_SUBPROBLEM_ERA_MIXED,
                    match_type=str(grp.get("match_type") or ""),
                    n_edges_raw=len(rec.union_edges),
                    mean_confidence=min_conf,
                    from_decomposition=True,
                    n_subproblems=rec.n_subproblems,
                    n_subproblems_resolved=rec.n_resolved,
                )
            )
            continue
        ge = _gate_recomposed_group(
            gid=parent_gid,
            bd=bd,
            grp=grp,
            rec=rec,
            meta=candidate_metas.get(parent_gid),
            human_gids=human_gids,
            overlap_map=overlap_map,
            mean_confidence=min_conf,
        )
        ge.panel_era = next(iter(sub_eras))
        groups.append(ge)

    # Stamp every DIRECT outcome with its source batch's era so write_exports
    # can tag each batch with its own labeler generation. Recomposed groups
    # were already stamped above from their CONTRIBUTING dirs (which may differ
    # from the roster dir this loop would key on) — leave them untouched.
    era_by_name = {bd.name: era for bd, era in era_by_dir.items()}
    for g in groups:
        if not g.from_decomposition:
            g.panel_era = era_by_name.get(g.source_batch, "")

    return ExportReport(
        dataset=dataset,
        n_total_groups=len(merged),
        n_auto_accept=n_auto,
        groups=groups,
        n_unanimous_none=n_none,
        n_subproblem_rows=n_sub_rows,
        n_decomposed_parents=len(rosters),
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
    # selected edge set for older batch.json packs without structure fields —
    # ALSO enforcing the hard backstop on the group's CANDIDATE count there,
    # mirroring _gate_empty_group. Without it, a legacy over-backstop group
    # with a small selected set (e.g. calib0709's 0cbcf706: 45 candidates, 18
    # selected) slipped past the flat cap and minted a label, resurrecting the
    # size-routing void the consensus/read-time size gates close.
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
    elif n_raw > max_edges or int(n_group_edges) > backstop_max_edges:
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


def _gate_empty_group(
    gid: str,
    bd: Path,
    row: dict,
    grp: dict,
    meta: dict | None,
    human_gids: set[str],
    overlap_map: dict[str, str],
    max_edges: int,
    backstop_max_edges: int,
) -> GroupExport:
    """Gate a unanimous-NONE group into an EMPTY-SET (reject-all) export.

    Gate (a) (unanimous-NONE) is applied by the caller. The remaining gates are
    tailored to an empty selection:

      * **Size** — apply ONLY the hard backstop ceiling on the group's *candidate*
        edge count (``n_edges`` / ``len(edges)``, since the selected set is empty
        by definition). The corridor/assignment-tangle sub-gate that
        :func:`_gate_group` uses guards against a too-greedy *selection*; an empty
        selection has no such risk, and that sub-gate would wrongly block the
        2-corridor cross-mode reject (parallel road + cycleway) this path exists
        to capture. A genuine monster still routes to a human (``over_max_edges``).
        With no structure fields, fall back to the flat ``max_edges`` cap.
      * **Class** — skipped. ``has_cross_mode_edge`` on an empty edge set is
        vacuously False; a cross-mode reject is exactly what we want to record.
      * **Sliver** — skipped. The emptiness is the intended verdict, not a sliver
        artifact, so it must not be reclassified as ``emptied_by_sliver``.
      * **Human precedence** — same as the accept path (exact group_id or
        edge-overlap). A prior human reject-all on this group matches by group_id.
    """
    match_type = str(grp.get("match_type") or (meta.get("match_type") if meta else "") or "")
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
            n_edges_raw=0,  # a reject-all selects nothing
            mean_confidence=mean_conf,
            is_empty_set=True,
            **kw,
        )

    # Size gate: hard backstop ceiling on the group's candidate size only.
    n_group_edges = grp.get("n_edges")
    if n_group_edges is None:
        n_group_edges = len(grp.get("edges", []))
    ceiling = backstop_max_edges if grp.get("n_edges") is not None else max_edges
    if int(n_group_edges) > ceiling:
        return _mk(REASON_OVER_MAX, n_edges_final=0)

    # Human precedence (exact group_id or edge-overlap): never overwrite a human.
    if gid in human_gids or gid in overlap_map:
        return _mk(
            REASON_HUMAN_PRECEDENCE,
            n_edges_final=0,
            human_group_id=(gid if gid in human_gids else overlap_map[gid]),
        )

    # Export the empty set (reject-all).
    return _mk(REASON_EXPORTED, n_edges_final=0, selected_edges=[])


def _gate_recomposed_group(
    gid: str,
    bd: Path,
    grp: dict,
    rec: Recomposition,
    meta: dict | None,
    human_gids: set[str],
    overlap_map: dict[str, str],
    mean_confidence: float,
) -> GroupExport:
    """Gate a decomposed group's recomposed verdict into a whole-group export.

    Conservative all-or-nothing rule (#367 Mode B): a label is minted ONLY when
    every sub-problem in the roster resolved as a unanimous panel accept — any
    failed sub-problem (``subproblem_failed``) or unvoted/size-gated one
    (``subproblems_unvoted``) blocks the group, and the failing sub-problems
    stay routed to human review. Mixed human+panel recomposition (a human
    sub-verdict completing a partially-panel-resolved group) is deferred.

    The remaining gates mirror :func:`_gate_group` on the UNION selection, with
    one deliberate exception — there is NO size gate: each sub-decision was
    within the panel envelope by construction, and the union spans the whole
    over-backstop parent, which is exactly what decomposition exists to label.
    Class-consistency, sliver canonicalization, and human precedence apply to
    the union / parent group unchanged.
    """
    match_type = str(grp.get("match_type") or (meta.get("match_type") if meta else "") or "")
    n_raw = len(rec.union_edges)

    def _mk(reason: str, **kw) -> GroupExport:
        return GroupExport(
            group_id=gid,
            source_batch=bd.name,
            exported=(reason == REASON_EXPORTED),
            reason=reason,
            match_type=match_type,
            n_edges_raw=n_raw,
            mean_confidence=mean_confidence,
            from_decomposition=True,
            n_subproblems=rec.n_subproblems,
            n_subproblems_resolved=rec.n_resolved,
            **kw,
        )

    if rec.status == STATUS_FAILED:
        return _mk(REASON_SUBPROBLEM_FAILED, n_edges_final=n_raw)
    if rec.status == STATUS_UNVOTED:
        return _mk(REASON_SUBPROBLEMS_UNVOTED, n_edges_final=n_raw)

    # Class-consistency gate on the union (same rule as _gate_group).
    if meta is not None:
        ref_c, tgt_c = _segment_class_maps(meta)
        edge_classes = _edge_classes_for(frozenset(rec.union_edges), ref_c, tgt_c)
        if has_cross_mode_edge(edge_classes):
            return _mk(REASON_CLASS_MISMATCH, n_edges_final=n_raw)

    # Sliver canonicalization on the union.
    sliver_pairs = _group_sliver_pairs(grp)
    final_pairs = [p for p in rec.union_edges if p not in sliver_pairs]
    n_slivers = n_raw - len(final_pairs)
    if not final_pairs:
        return _mk(REASON_EMPTIED_BY_SLIVER, n_slivers_dropped=n_slivers, n_edges_final=0)

    # Human precedence (exact group_id or edge-overlap): never overwrite a human.
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


#: Era -> (accept, reject-all, decomposed) labeler tags for write_exports.
#: v3-era batches keep minting v3-tagged labels on (re-)export; v4 batches mint
#: the current tags. There is deliberately NO fallback entry: an era-less group
#: ("" — unknown composition, or no readable votes.csv) makes write_exports
#: refuse rather than silently mint the current era's provenance.
LABELERS_BY_ERA: dict[str, tuple[str, str, str]] = {
    "v3": (PANEL_LABELER_V3, PANEL_NONE_LABELER_V3, PANEL_DECOMPOSED_LABELER_V3),
    "v4": (PANEL_LABELER, PANEL_NONE_LABELER, PANEL_DECOMPOSED_LABELER),
}


def write_exports(
    report: ExportReport,
    dataset: str,
    labels_dir: Path,
) -> int:
    """Persist the report's exported groups as ``panel_*`` stitching labels.

    Accept groups are stamped ``panel_unanimous_v4`` with their chosen edge set;
    reject-all (empty-set) groups are stamped ``panel_unanimous_none_v4`` with
    ``selected_edges == []`` (PAIR semantics, num_refs/num_targets == 0 — the same
    on-disk shape as a human reject-all); recomposed decomposed-group verdicts
    (#367 Mode B) are stamped ``panel_unanimous_decomposed_v4`` with the union of
    their sub-problem selections. Groups whose source batch is a v3-era panel
    (``panel_era == "v3"``, see :func:`batch_panel_era`) are stamped with the v3
    variants instead — re-exporting committed v3 history never rewrites its
    provenance to v4 — and a group with NO resolvable era raises ``ValueError``
    (declare one explicitly via ``plan_exports(stamp_era=...)`` / CLI
    ``--stamp-era``) instead of silently minting current-era provenance.
    Upserts by ``group_id`` (the store replaces an
    existing row for the same group_id), so this is idempotent. The source batch
    name is recorded in the ``session_id`` field for provenance.
    Returns the number of rows written.
    """
    era_less = sorted(
        {g.source_batch for g in report.exported if g.panel_era not in LABELERS_BY_ERA}
    )
    if era_less:
        raise ValueError(
            f"batches {era_less} match no known panel era (blessed or historical) — "
            f"refusing to stamp panel_* labelers on unknown provenance. Declare the "
            f"era explicitly with plan_exports(stamp_era=...) / CLI --stamp-era "
            f"{{{', '.join(sorted(LABELERS_BY_ERA))}}}."
        )
    store = StitchingLabelStore(dataset, labels_dir=labels_dir)
    written = 0
    for g in report.exported:
        accept_tag, none_tag, decomposed_tag = LABELERS_BY_ERA[g.panel_era]
        if g.is_empty_set:
            labeler = none_tag
        elif g.from_decomposition:
            labeler = decomposed_tag
        else:
            labeler = accept_tag
        ref_ids = {e["ref_id"] for e in g.selected_edges}
        tgt_ids = {e["target_id"] for e in g.selected_edges}
        store.add(
            group_id=g.group_id,
            selected_edges=g.selected_edges,
            match_type=g.match_type,
            num_refs=len(ref_ids),
            num_targets=len(tgt_ids),
            labeler=labeler,
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
    labels for the groups in each run. A re-archived batch REPLACES all of its
    previous rows wholesale (keyed by ``source_batch``, only when the batch
    contributes a readable file in this call): per-row upsert would leave stale
    ballots from a different panel composition lingering under the same batch
    name (e.g. archived agy rows surviving a v4-panel re-run — a 4-voter
    chimera batch). Rows are still deduped within a run as defense in depth.

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
            df = pd.read_csv(path, dtype={"group_id": str})
        except pd.errors.EmptyDataError:
            return None
        # A header-only (zero-row) file is as empty as a 0-byte one: it must
        # not mark a batch as "contributing" in _collect's wholesale
        # replacement, or it would delete the batch's archived ballots with no
        # replacement rows.
        return None if df.empty else df

    def _collect(filename: str, dedupe_on: list[str]) -> int:
        # Current batches first: only a batch that contributes a READABLE file
        # in this call supersedes its archived rows — a listed batch whose CSV
        # is missing/empty keeps its archive untouched (never delete ballots
        # without a replacement).
        new_frames: list[pd.DataFrame] = []
        contributing: set[str] = set()
        for bd in batch_dirs:
            df = _read_csv(Path(bd) / filename)
            if df is None:
                continue
            if "source_batch" in df.columns:  # a re-archived tree; re-tag cleanly
                df = df.drop(columns="source_batch")
            df.insert(0, "source_batch", Path(bd).name)
            contributing.add(Path(bd).name)
            new_frames.append(df)
        frames = []
        existing = _read_csv(out_dir / filename)
        if existing is not None:
            # Wholesale replacement per re-archived batch: drop ALL previously
            # archived rows for the batches refreshed in this call, so stale
            # ballots from an earlier panel composition (e.g. agy rows under a
            # batch re-run with the v4 panel) can never linger beside the new
            # rows as a chimera composition.
            if "source_batch" in existing.columns and contributing:
                existing = existing[~existing["source_batch"].astype(str).isin(contributing)]
            if not existing.empty:
                frames.append(existing)
        frames.extend(new_frames)
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
