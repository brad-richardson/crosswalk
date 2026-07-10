"""Drift-aware prior-label coverage of CURRENT stitch groups (review-queue filter).

Stitch group_ids are content hashes of the exact ref/target id sets, so any
re-grouping (re-optimize / re-prune / re-segment) mints NEW ids for the same
physical geometry. The review queue's "already reviewed" filter used to match
on exact group_id only, so a relabeled group whose membership churned re-queued
as if it had never been seen. Motivating case (Bogotá, 2026-07): group
``3c3e6853`` (64x191) was reviewed de-anchored and saved as a set label keeping
55 refs x 153 targets; a regenerated sidecar re-minted the same monster as
``8e32a935`` (70x239) and the queue served it again as brand new.

This module classifies every CURRENT group against the dataset's stitching
labels using the SAME drift mapping the eval side uses
(:func:`crosswalk.agent_labeling.stitch_eval.recover_labeled_groups` — pair
labels by selected-edge overlap, set labels by membership overlap, both with
the #354 deterministic lexicographic tie-break). Reusing that mapper — the
choice `stitch_rekey` already made — keeps the queue filter and the eval/rekey
mapping contracts from ever diverging.

Coverage semantics for a current group G and a label L that maps to it:

* **Exact id** — L's ``group_id`` still exists verbatim in the current groups.
  The reviewer adjudicated exactly this membership (the id IS the membership
  hash), so G is reviewed regardless of what subset the label kept. Excluded
  from the queue — identical to the pre-drift-aware behavior, so id-stable
  paths (e.g. re-keyed labels) are unaffected.
* **Fully covered** (drifted id, ``G.refs ⊆ L.kept_refs`` AND
  ``G.targets ⊆ L.kept_targets``) — every current member was affirmatively
  kept by the prior review; re-review would be a mechanical re-approve.
  Treated as reviewed → excluded.
* **Partially covered** (maps, but G has members outside L's kept universes) —
  INCLUDED in the queue with delta metadata (prior label provenance,
  covered/total counts per side, new-member ids) so the review UI can show a
  banner, prefill ``kept ∩ current``, and highlight new-since-label members.
  New membership means the prior decisions may legitimately flip.
* **No mapping** — included, exactly as before.

A label's KEPT membership:

* ``set`` labels (``label_semantics=set``): the stored ``ref_ids`` /
  ``target_ids`` (removals are NOT recorded).
* ``pair`` labels: the union of the ``selected_edges`` endpoints plus
  ``ref_ids`` / ``target_ids`` when present.
* Reject-all pair labels (empty edges, no membership) keep their historical
  exact-id-only semantics: they never drift-map (nothing to overlap on), same
  as :func:`recover_empty_reject_all`.

Conservative merge rule: when SEVERAL drifted labels map onto one current
group (an optimizer merge), the group is only excluded if a SINGLE label fully
covers it — the union of two partial reviews never auto-settles a merged group
(the same posture as ``stitch_rekey``'s refusal to auto-apply merges). The
best-covering label (ties broken on the lexicographically smallest prior
group_id) supplies the delta metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from crosswalk.agent_labeling.stitch_eval import recover_labeled_groups

# Key under which delta metadata is attached to queued batch-JSON group entries.
PRIOR_LABEL_KEY = "prior_label"


def _edge_set(selected_edges_raw) -> frozenset[tuple[str, str]]:
    """Parse a label's ``selected_edges`` JSON into an edge frozenset.

    Local copy of the tiny parser (same choice as ``stitch_rekey`` /
    ``resolver/extract.py``) rather than importing the private
    ``stitch_eval._human_edge_set``.
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


def _kept_membership(row) -> tuple[frozenset[str], frozenset[str]]:
    """A label row's KEPT membership as ``(kept_refs, kept_targets)`` id sets.

    Set labels store it in ``ref_ids``/``target_ids``; pair labels' adjudicated
    membership is the union of their edge endpoints plus ``ref_ids``/``target_ids``
    when present. Empty on both sides for a reject-all pair label.
    """
    edges = _edge_set(row.get("selected_edges"))
    kept_refs = frozenset(r for r, _ in edges) | _id_list(row.get("ref_ids"))
    kept_targets = frozenset(t for _, t in edges) | _id_list(row.get("target_ids"))
    return kept_refs, kept_targets


def _group_membership(group: dict) -> tuple[frozenset[str], frozenset[str]]:
    """A current group's membership as ``(refs, targets)`` string-id sets.

    Prefers the sidecar's authoritative ``ref_ids``/``target_ids`` fields,
    falling back to the edge endpoints for minimal group dicts.
    """
    refs = group.get("ref_ids")
    targets = group.get("target_ids")
    if refs is None or targets is None:
        edges = group.get("edges", []) or []
        if refs is None:
            refs = [e["ref_id"] for e in edges]
        if targets is None:
            targets = [e["target_id"] for e in edges]
    return frozenset(str(r) for r in refs), frozenset(str(t) for t in targets)


@dataclass(frozen=True)
class PriorLabelCoverage:
    """How one prior stitching label covers one CURRENT group's membership."""

    group_id: str  # current group id
    prior_group_id: str  # the label row's stored group_id
    labeler: str
    labeled_at: str
    label_semantics: str
    exact_id: bool  # label's group_id == current group_id (verbatim survival)
    covered_ref_ids: tuple[str, ...]  # current refs ∩ label's kept refs
    new_ref_ids: tuple[str, ...]  # current refs outside the kept universe
    covered_target_ids: tuple[str, ...]
    new_target_ids: tuple[str, ...]

    @property
    def n_total_refs(self) -> int:
        return len(self.covered_ref_ids) + len(self.new_ref_ids)

    @property
    def n_total_targets(self) -> int:
        return len(self.covered_target_ids) + len(self.new_target_ids)

    @property
    def fully_covered(self) -> bool:
        """True when the group is settled ground truth (reviewed → exclude).

        Exact-id survival counts as fully covered by fiat: the id IS the
        membership hash, so the reviewer adjudicated exactly this group even if
        the label kept only a subset (removals are simply not re-litigated —
        identical to the pre-drift-aware exact-id filter).
        """
        return self.exact_id or (not self.new_ref_ids and not self.new_target_ids)

    def to_batch_dict(self) -> dict:
        """JSON-serializable delta metadata for a queued batch group entry."""
        return {
            "prior_group_id": self.prior_group_id,
            "labeler": self.labeler,
            "labeled_at": self.labeled_at,
            "label_semantics": self.label_semantics,
            "n_covered_refs": len(self.covered_ref_ids),
            "n_total_refs": self.n_total_refs,
            "n_covered_targets": len(self.covered_target_ids),
            "n_total_targets": self.n_total_targets,
            "covered_ref_ids": list(self.covered_ref_ids),
            "new_ref_ids": list(self.new_ref_ids),
            "covered_target_ids": list(self.covered_target_ids),
            "new_target_ids": list(self.new_target_ids),
        }


def _coverage_for(group: dict, gid: str, row, prior_gid: str, exact: bool) -> PriorLabelCoverage:
    """Build the coverage record of ``group`` under label ``row``."""
    refs, targets = _group_membership(group)
    if exact:
        # Exact-id survival: reviewed by fiat (see PriorLabelCoverage.fully_covered).
        covered_refs, new_refs = refs, frozenset()
        covered_targets, new_targets = targets, frozenset()
    else:
        kept_refs, kept_targets = _kept_membership(row)
        covered_refs, new_refs = refs & kept_refs, refs - kept_refs
        covered_targets, new_targets = targets & kept_targets, targets - kept_targets
    return PriorLabelCoverage(
        group_id=gid,
        prior_group_id=prior_gid,
        labeler=str(row.get("labeler") or ""),
        labeled_at=str(row.get("labeled_at") or ""),
        label_semantics=str(row.get("label_semantics") or "pair"),
        exact_id=exact,
        covered_ref_ids=tuple(sorted(covered_refs)),
        new_ref_ids=tuple(sorted(new_refs)),
        covered_target_ids=tuple(sorted(covered_targets)),
        new_target_ids=tuple(sorted(new_targets)),
    )


def compute_prior_coverage(
    groups: list[dict], labels_df: pd.DataFrame | None
) -> dict[str, PriorLabelCoverage]:
    """Map each CURRENT group_id to its best prior-label coverage, if any.

    Args:
        groups: current group dicts (sidecar groups or batch-queue entries);
            each needs ``group_id`` plus ``ref_ids``/``target_ids`` (or
            ``edges`` to derive them) — the same shape ``recover_labeled_groups``
            consumes.
        labels_df: the dataset's stitching labels
            (``StitchingLabelStore.load(dataset)`` schema). ``None``/empty means
            no coverage anywhere.

    Returns:
        ``{current_group_id: PriorLabelCoverage}`` for every group at least one
        label maps to. Groups with no entry are unreviewed (queue as-is);
        entries with ``fully_covered`` are settled (exclude); the rest are
        partial (queue with :meth:`PriorLabelCoverage.to_batch_dict` metadata).
    """
    if labels_df is None or len(labels_df) == 0 or not groups:
        return {}

    groups_by_gid = {str(g.get("group_id")): g for g in groups}
    current_gids = set(groups_by_gid)

    coverage: dict[str, PriorLabelCoverage] = {}

    # --- exact-id survivors: pinned to their own group, reviewed by fiat -----
    label_gids = labels_df["group_id"].astype(str)
    for _, row in labels_df[label_gids.isin(current_gids)].iterrows():
        gid = str(row["group_id"])
        coverage[gid] = _coverage_for(groups_by_gid[gid], gid, row, gid, exact=True)

    # --- drifted labels: the eval-side drift mapping (recover_labeled_groups) --
    drifted_df = labels_df[~label_gids.isin(current_gids)]
    if len(drifted_df) == 0:
        return coverage

    rec = recover_labeled_groups(groups, drifted_df)
    mapped_pairs: list[tuple[str, str]] = list(rec["clean"])
    mapped_pairs += [(h, s) for h, s, _, _ in rec["split"]]
    mapped_pairs += list(rec.get("set", []))
    # rec["empty"] / rec["lost"] / rec["set_lost"]: no mapping -> no coverage.

    rows_by_gid = {str(r["group_id"]): r for _, r in drifted_df.iterrows()}
    candidates: dict[str, list[PriorLabelCoverage]] = {}
    for hgid, sidecar_gid in mapped_pairs:
        if sidecar_gid in coverage:
            continue  # an exact-id label already settles this group
        group = groups_by_gid.get(sidecar_gid)
        row = rows_by_gid.get(str(hgid))
        if group is None or row is None:
            continue
        candidates.setdefault(sidecar_gid, []).append(
            _coverage_for(group, sidecar_gid, row, str(hgid), exact=False)
        )

    for gid, entries in candidates.items():
        # Best-covering label wins; ties break on the lexicographically smallest
        # prior group_id (the #354 determinism convention). A merged group is
        # only excluded when ONE label fully covers it — partial unions stay
        # queued (see module docstring).
        entries.sort(
            key=lambda c: (
                -(len(c.covered_ref_ids) + len(c.covered_target_ids)),
                c.prior_group_id,
            )
        )
        coverage[gid] = entries[0]

    return coverage


def fully_covered_group_ids(coverage: dict[str, PriorLabelCoverage]) -> set[str]:
    """Group ids settled by a prior label (exact-id or drift-mapped full cover)."""
    return {gid for gid, c in coverage.items() if c.fully_covered}
