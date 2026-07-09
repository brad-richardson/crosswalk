"""Rekey drifted stitching labels onto the CURRENT sidecar grouping (#374/#375).

Stitching labels are keyed by ``group_id`` = hash of the exact ref/target id
set, so any re-grouping (re-optimize / re-prune / re-segment) detaches the
label from the current sidecar. This module builds a **rekey plan** on top of
``recover_labeled_groups`` (the same edge/membership-overlap recovery the
resolver's ``extract.build_edge_table`` uses — the mapping contract the rekey
must match) and applies only the provably-safe subset.

Bucket semantics (the #375 agreed semantics):

* ``unchanged`` — the label's ``group_id`` still exists in the current sidecar;
  nothing to do.
* ``clean`` (1:1) — a drifted pair label whose selected edges all land in
  exactly ONE current group, no other label lands on that group, and no
  existing label row already occupies that ``group_id``. Blind rekey is safe;
  ``--apply`` executes exactly this bucket, nothing else.
* ``merge_union`` / ``merge_conflict`` (N->1) — the optimizer *merged* old
  components, so >=2 old labels land on one current group. A naive
  ``group_id = current_id`` rewrite would collapse them last-write-wins (the
  Boston 49-labels-into-8-groups hazard, #375), so these are NEVER
  auto-applied. ``union``: the contributing labels have disjoint segment
  scopes, so the union of their edge assertions is a coherent reconciliation
  *proposal* (reported for review). ``conflict``: the scopes overlap — two
  labels adjudicated the same segment with independent edge sets, where an
  edge absent from one label is a rejection the union would silently
  overrule — a human must reconcile.
* ``split`` (1->N) — the label's edges span multiple current groups. The
  human's boundary judgment is the contested thing; review only.
* ``set_review`` — SET-semantics labels assert group MEMBERSHIP, not edges,
  so remapping is a membership-overlap heuristic. Reported with a
  membership-based summary; never auto-applied.
* ``collision_refused`` — the mapped-to ``group_id`` already carries a label
  row (or the store itself already holds duplicate rows for one group_id).
  Rekeying onto it would create duplicate ``group_id`` rows, which silently
  corrupts resolver extraction (``resolver/extract.py`` builds
  ``human_by = {group_id: row}`` last-row-wins while
  ``recover_labeled_groups`` maps per-row) and double-counts eval. REFUSED —
  the CLI exits nonzero unless ``--allow-partial``.
* ``empty_unrecoverable`` / ``lost`` / ``set_lost`` — nothing left to remap
  onto (reject-all labels recover only on a verbatim group_id match; see
  ``recover_empty_reject_all``).

Audit trail: applied moves are appended to a **sidecar join table**
``labels/stitching/dataset={id}/rekey_log.csv`` (the #369
``rekey_seattle_target.py`` pattern) rather than a ``prior_group_id`` label
column: ``StitchingLabelStore._ensure_schema`` deliberately drops unknown
columns on round-trip, #374's planned schema additions do not include
operation provenance, and a single column cannot represent the old->new
*chain* across repeated rekeys (#375 requires the command to be repeatable).
The log is append-only, so the full lineage survives any number of runs.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from crosswalk.agent_labeling.stitch_eval import recover_labeled_groups
from crosswalk.labeling.stitching_store import StitchingLabelStore

REKEY_LOG_FILENAME = "rekey_log.csv"
REKEY_LOG_COLUMNS = [
    "rekeyed_at",
    "dataset_id",
    "old_group_id",
    "new_group_id",
    "labeler",
    "labeled_at",
    "sidecar",
]


def _edge_set(selected_edges_raw) -> frozenset[tuple[str, str]]:
    """Parse a label's ``selected_edges`` JSON into an edge frozenset.

    Local copy of the tiny parser (same choice as ``resolver/extract.py``)
    rather than importing the private ``stitch_eval._human_edge_set``.
    """
    try:
        edges = json.loads(selected_edges_raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset((str(e["ref_id"]), str(e["target_id"])) for e in edges)


def _id_list(raw) -> frozenset[str]:
    """Parse a JSON id array (``ref_ids`` / ``target_ids``) into a string set."""
    if raw is None or isinstance(raw, float) or not str(raw).strip():
        return frozenset()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset(str(x) for x in data)


@dataclass(frozen=True)
class RekeyMove:
    """A safe 1:1 rekey: one drifted label onto one unoccupied current group."""

    old_group_id: str
    new_group_id: str
    labeler: str = ""
    labeled_at: str = ""


@dataclass(frozen=True)
class MergeCase:
    """N->1: several old labels land on one merged current group (review)."""

    new_group_id: str
    old_group_ids: tuple[str, ...]
    union_edges: tuple[tuple[str, str], ...] = ()
    shared_segments: tuple[str, ...] = ()  # non-empty iff conflict


@dataclass(frozen=True)
class SplitCase:
    """1->N: a label's edges span multiple current groups (review)."""

    old_group_id: str
    best_group_id: str
    n_edges_in_best: int
    n_edges_total: int


@dataclass(frozen=True)
class SetCase:
    """SET-semantics label recovered by membership overlap (review)."""

    old_group_id: str
    dominant_group_id: str
    n_members_in_dominant: int
    n_members_total: int
    n_groups_spanned: int


@dataclass(frozen=True)
class Collision:
    """A rekey that would create duplicate group_id rows (refused)."""

    new_group_id: str
    old_group_ids: tuple[str, ...]
    reason: str


@dataclass
class RekeyPlan:
    """Full classification of every stitching label against the current sidecar."""

    dataset_id: str
    n_labels: int = 0
    n_current_groups: int = 0
    unchanged: list[str] = field(default_factory=list)
    clean: list[RekeyMove] = field(default_factory=list)
    merge_union: list[MergeCase] = field(default_factory=list)
    merge_conflict: list[MergeCase] = field(default_factory=list)
    split: list[SplitCase] = field(default_factory=list)
    set_review: list[SetCase] = field(default_factory=list)
    collision_refused: list[Collision] = field(default_factory=list)
    empty_unrecoverable: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    set_lost: list[str] = field(default_factory=list)

    @property
    def mapping(self) -> dict[str, str]:
        """The clean-bucket old->new mapping — the ONLY auto-applied moves."""
        return {m.old_group_id: m.new_group_id for m in self.clean}

    @property
    def has_refusals(self) -> bool:
        return bool(self.collision_refused)

    def counts(self) -> dict[str, int]:
        return {
            "unchanged": len(self.unchanged),
            "clean": len(self.clean),
            "merge_union": len(self.merge_union),
            "merge_conflict": len(self.merge_conflict),
            "split": len(self.split),
            "set_review": len(self.set_review),
            "collision_refused": len(self.collision_refused),
            "empty_unrecoverable": len(self.empty_unrecoverable),
            "lost": len(self.lost),
            "set_lost": len(self.set_lost),
        }


def _label_scope(row: pd.Series) -> frozenset[str]:
    """All segment ids a label adjudicated (edge endpoints + set membership).

    Used for the merge union/conflict decision: two labels whose scopes share
    a segment both made an assertion about that segment, so their union is not
    a safe reconciliation.
    """
    scope: set[str] = set()
    for r, t in _edge_set(row.get("selected_edges")):
        scope.add(r)
        scope.add(t)
    scope |= _id_list(row.get("ref_ids"))
    scope |= _id_list(row.get("target_ids"))
    return frozenset(scope)


def build_rekey_plan(groups: list[dict], labels_df: pd.DataFrame, dataset_id: str) -> RekeyPlan:
    """Classify every stitching label into a rekey-plan bucket.

    Args:
        groups: current sidecar groups (``*_groups.json`` ``groups`` list).
        labels_df: the dataset's stitching labels (``StitchingLabelStore.load``).
        dataset_id: dataset identifier (report provenance only).

    Returns:
        A :class:`RekeyPlan`. Only ``plan.clean`` is safe to apply; the
        ``plan.mapping`` property exposes it as an old->new dict.
    """
    current_gids = {str(g["group_id"]) for g in groups}
    plan = RekeyPlan(
        dataset_id=dataset_id, n_labels=len(labels_df), n_current_groups=len(current_gids)
    )
    if labels_df.empty:
        return plan

    label_gids = [str(g) for g in labels_df["group_id"]]
    label_gid_set = set(label_gids)
    rows_by_gid = {str(r["group_id"]): r for _, r in labels_df.iterrows()}

    # A store that already holds duplicate rows for one group_id is corrupt
    # for resolver extraction (human_by last-row-wins); surface it as a
    # refusal so the exit code trips before anyone rekeys on top of it.
    dup_counts = pd.Series(label_gids).value_counts()
    for gid in sorted(dup_counts[dup_counts > 1].index):
        plan.collision_refused.append(
            Collision(
                new_group_id=str(gid),
                old_group_ids=(str(gid),),
                reason=f"store already holds {int(dup_counts[gid])} label rows for this group_id",
            )
        )

    plan.unchanged = sorted(label_gid_set & current_gids)

    rec = recover_labeled_groups(groups, labels_df)

    # --- pair labels with edges: clean / split (drifted only) ---------------
    drifted_clean = [(h, s) for h, s in rec["clean"] if h not in current_gids]
    by_target: dict[str, list[str]] = defaultdict(list)
    for h, s in drifted_clean:
        by_target[s].append(h)

    for new_gid in sorted(by_target):
        old_gids = sorted(by_target[new_gid])
        if new_gid in label_gid_set:
            # An existing label row (typically an ``unchanged`` one) already
            # occupies the target group_id: rekeying onto it would duplicate.
            plan.collision_refused.append(
                Collision(
                    new_group_id=new_gid,
                    old_group_ids=tuple(old_gids),
                    reason="target group_id already has a label row",
                )
            )
            continue
        if len(old_gids) > 1:
            scopes = {h: _label_scope(rows_by_gid[h]) for h in old_gids}
            shared: set[str] = set()
            for i, a in enumerate(old_gids):
                for b in old_gids[i + 1 :]:
                    shared |= scopes[a] & scopes[b]
            union_edges = tuple(
                sorted(
                    set().union(
                        *(_edge_set(rows_by_gid[h].get("selected_edges")) for h in old_gids)
                    )
                )
            )
            case = MergeCase(
                new_group_id=new_gid,
                old_group_ids=tuple(old_gids),
                union_edges=union_edges,
                shared_segments=tuple(sorted(shared)),
            )
            (plan.merge_conflict if shared else plan.merge_union).append(case)
            continue
        row = rows_by_gid[old_gids[0]]
        plan.clean.append(
            RekeyMove(
                old_group_id=old_gids[0],
                new_group_id=new_gid,
                labeler=str(row.get("labeler") or ""),
                labeled_at=str(row.get("labeled_at") or ""),
            )
        )

    plan.split = [
        SplitCase(old_group_id=h, best_group_id=s, n_edges_in_best=n, n_edges_total=tot)
        for h, s, n, tot in rec["split"]
        if h not in current_gids
    ]

    # --- SET-semantics labels: membership-based review report ---------------
    seg_groups: dict[str, set[str]] = defaultdict(set)
    for g in groups:
        gid = str(g["group_id"])
        for e in g.get("edges", []):
            seg_groups[str(e["ref_id"])].add(gid)
            seg_groups[str(e["target_id"])].add(gid)
    for h, dom in rec.get("set", []):
        if h in current_gids:
            continue  # already counted as unchanged
        row = rows_by_gid[h]
        members = _id_list(row.get("ref_ids")) | _id_list(row.get("target_ids"))
        spanned: set[str] = set()
        n_in_dom = 0
        for m in members:
            gids = seg_groups.get(m, set())
            spanned |= gids
            if dom in gids:
                n_in_dom += 1
        plan.set_review.append(
            SetCase(
                old_group_id=h,
                dominant_group_id=dom,
                n_members_in_dominant=n_in_dom,
                n_members_total=len(members),
                n_groups_spanned=len(spanned),
            )
        )

    # --- unrecoverable -------------------------------------------------------
    # Reject-all (empty-edge) labels survive only on a verbatim group_id match
    # (same rule as recover_empty_reject_all / resolver extract).
    plan.empty_unrecoverable = sorted(h for h in rec["empty"] if h not in current_gids)
    plan.lost = sorted(rec["lost"])
    plan.set_lost = sorted(rec.get("set_lost", []))
    return plan


def append_rekey_log(
    partition_path: Path,
    moves: list[RekeyMove],
    dataset_id: str,
    sidecar: str,
    rekeyed_at: str | None = None,
) -> Path:
    """Append applied moves to the dataset's ``rekey_log.csv`` join table.

    Append-only: repeated rekeys accumulate rows, so the full old->new lineage
    chain survives (a ``prior_group_id`` label column could only hold the last
    hop, and ``_ensure_schema`` would drop it anyway). Returns the log path.
    """
    partition_path = Path(partition_path)
    partition_path.mkdir(parents=True, exist_ok=True)
    log_path = partition_path / REKEY_LOG_FILENAME
    stamp = rekeyed_at or datetime.now(UTC).isoformat()
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(REKEY_LOG_COLUMNS)
        for m in moves:
            writer.writerow(
                [
                    stamp,
                    dataset_id,
                    m.old_group_id,
                    m.new_group_id,
                    m.labeler,
                    m.labeled_at,
                    sidecar,
                ]
            )
    return log_path


def read_rekey_log(partition_path: Path) -> pd.DataFrame:
    """Read a dataset's rekey join table (empty frame when none exists)."""
    log_path = Path(partition_path) / REKEY_LOG_FILENAME
    if not log_path.exists():
        return pd.DataFrame(columns=REKEY_LOG_COLUMNS)
    return pd.read_csv(log_path, dtype=str)


def apply_clean_rekey(store: StitchingLabelStore, plan: RekeyPlan, sidecar: str) -> int:
    """Apply the plan's clean bucket through the store API and write the audit log.

    Everything else in the plan (merges, splits, sets, collisions) is review
    material and is deliberately NOT touched. Returns rows rekeyed.
    """
    mapping = plan.mapping
    if not mapping:
        return 0
    n = store.rekey_group_ids(mapping)  # collision guards live in the store
    append_rekey_log(store.partition_path, plan.clean, plan.dataset_id, sidecar)
    return n
