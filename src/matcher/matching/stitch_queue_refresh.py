"""Refresh a stitching review-queue cache against the current groups sidecar.

Motivation (the "stale proposal" bug class)
--------------------------------------------
The review-queue cache (``data/cache/stitch/{ds}_batch.json``) is a *snapshot*
of the groups sidecar taken at batch-generation time. When the model is
retrained or the optimizer re-prunes, the sidecar (``data/output/{ds}_groups.json``)
moves on: a group's ``optimizer_assignment`` (the selected edge set), per-edge
``confidence`` values, and the enumerated ``alternatives`` all change. But a
queue that was rebuilt by *preserving* still-unreviewed entries verbatim keeps
the OLD vintage of those fields. The review UI pre-seeds and renders its option
cards straight from the cache, so reviewers are shown — and ratify — stale
proposals, and the eval then diffs their labels against the current sidecar and
reports phantom added/removed pairs.

This module provides:

* :func:`optimizer_pair_set` / :func:`selected_pair_set` — the two canonical
  (ref_id, target_id) sets a queue entry and a sidecar group expose.
* :func:`check_queue_optimizer_parity` — the maintenance check the cache
  rebuild path calls: for every queue entry whose ``group_id`` is present in the
  sidecar, the queue's ``optimizer_assignment`` MUST equal the sidecar's
  selected edge set. Any drift is a stale proposal.
* :func:`plan_queue_refresh` — classify each queue entry as *refreshable*
  (group still exists in the sidecar → rebuild from the authoritative sidecar
  group) or *stale-grouping* (group no longer exists → cannot refresh; the UI
  flags it so ratifying it is an informed act). Order is preserved; the queue is
  never reshaped (the reviewer may be mid-review).

The heavy per-entry rebuild (alternatives + review tier/score + spatial context)
lives in the CLI (``matcher data stitch-refresh-queue``) because spatial context
requires the raw parquet; this module holds the pure, testable core.
"""

from __future__ import annotations

# Marker key stamped on queue entries whose group no longer exists in the
# current sidecar (grouping changed since the batch was generated). The review
# UI renders a visible "stale proposal" notice on these so ratifying them is an
# informed act. Refreshed entries carry this set to False.
STALE_GROUPING_KEY = "stale_grouping"


def _pair_set(edges) -> set[tuple[str, str]]:
    return {(e.get("ref_id"), e.get("target_id")) for e in (edges or [])}


def optimizer_pair_set(group: dict) -> set[tuple[str, str]]:
    """(ref_id, target_id) pairs the group proposes as the selected assignment.

    Uses ``optimizer_assignment`` — the field the review UI pre-seeds and the
    option-card "auto" pick reads.
    """
    return _pair_set(group.get("optimizer_assignment"))


def selected_pair_set(group: dict) -> set[tuple[str, str]]:
    """(ref_id, target_id) pairs currently SELECTED by the group.

    Prefers ``edges`` flagged ``selected`` (the sidecar's authoritative marking).
    Falls back to ``optimizer_assignment`` when no edge carries a ``selected``
    flag (e.g. a queue entry whose edges predate the flag).
    """
    edges = group.get("edges") or []
    selected = {(e.get("ref_id"), e.get("target_id")) for e in edges if e.get("selected")}
    if selected:
        return selected
    if any("selected" in e for e in edges):
        # Edges DO carry the flag but none are selected → a genuine empty
        # selection (reject-all). Do not fall back to optimizer_assignment.
        return set()
    return optimizer_pair_set(group)


def check_queue_optimizer_parity(
    queue_groups: list[dict],
    sidecar_by_id: dict[str, dict],
) -> list[dict]:
    """Return queue entries whose proposal drifted from the current sidecar.

    For every queue entry whose ``group_id`` is present in ``sidecar_by_id``, the
    queue's ``optimizer_assignment`` pair-set MUST equal the sidecar's selected
    pair-set. Entries whose group is absent from the sidecar (old-grouping) are
    NOT checkable and are skipped.

    An empty list means the queue is in parity with the sidecar. Each mismatch
    dict carries ``group_id``, ``queue_only`` (proposed by the queue but no
    longer selected) and ``sidecar_only`` (now selected but missing from the
    queue proposal).
    """
    mismatches: list[dict] = []
    for entry in queue_groups:
        gid = entry.get("group_id")
        sidecar = sidecar_by_id.get(gid)
        if sidecar is None:
            continue
        queue_opt = optimizer_pair_set(entry)
        sidecar_sel = selected_pair_set(sidecar)
        if queue_opt != sidecar_sel:
            mismatches.append(
                {
                    "group_id": gid,
                    "queue_only": sorted(queue_opt - sidecar_sel),
                    "sidecar_only": sorted(sidecar_sel - queue_opt),
                }
            )
    return mismatches


def plan_queue_refresh(
    queue_groups: list[dict],
    sidecar_by_id: dict[str, dict],
) -> tuple[list[str], list[str]]:
    """Split queue entries into (refreshable_ids, stale_grouping_ids).

    Order-preserving classification only — no mutation. A queue entry is
    refreshable when its ``group_id`` still exists in the sidecar (rebuild from
    the authoritative sidecar group); otherwise it is stale-grouping (the group
    was split/merged away by re-grouping and cannot be refreshed).
    """
    refreshable: list[str] = []
    stale: list[str] = []
    for entry in queue_groups:
        gid = entry.get("group_id")
        if gid in sidecar_by_id:
            refreshable.append(gid)
        else:
            stale.append(gid)
    return refreshable, stale
