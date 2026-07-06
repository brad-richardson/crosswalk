"""Cross-product artifact detection for stitching labels (shared #289 logic).

Manual / de-anchored stitching submits historically recorded the full
ref×target cross-product of the active pills (intersected with the candidate
universe) as individual pair assertions, even though the reviewer only asserted
group MEMBERSHIP, not each pairing. This module is the single source of truth for
detecting those cross-product artifacts. It is imported by:

  * ``scripts/render_review_diffs.py`` (flags them in the review renders), and
  * ``crosswalk data stitch-reinterpret-sets`` (converts them to SET labels).

Pure functions only — no I/O, no matplotlib — so both callers share exactly the
same signature and it is unit-testable in isolation.
"""

from __future__ import annotations

import ast

Pair = tuple[str, str]


def parse_selected_edges(raw: str | None) -> set[Pair]:
    """Parse a ``selected_edges`` CSV cell into a set of (ref_id, target_id).

    The cell is a JSON/py-literal list of ``{"ref_id":..,"target_id":..}`` dicts.
    ``ast.literal_eval`` handles both single- and double-quoted encodings.

    NaN-safe and malformed-safe: a blank CSV cell reads back as a float NaN
    (which is truthy), and a hand-edited cell may not parse at all — both return
    an empty set rather than aborting a whole reinterpretation/render run on one
    bad row.
    """
    if raw is None or isinstance(raw, float) or not str(raw).strip():
        return set()
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return set()
    if not isinstance(parsed, list):
        return set()
    return {(e["ref_id"], e["target_id"]) for e in parsed}


def edge_pairs(edges: list[dict] | None, selected_only: bool = False) -> set[Pair]:
    """Collect (ref_id, target_id) tuples from an edge list."""
    out: set[Pair] = set()
    for e in edges or []:
        if selected_only and not e.get("selected"):
            continue
        out.add((e["ref_id"], e["target_id"]))
    return out


def resolve_optimizer(
    cache_group: dict | None, sidecar_group: dict | None
) -> tuple[set[Pair], bool]:
    """Return the optimizer's selected pairs and whether they came from the sidecar.

    Prefers the sidecar group's ``edges[selected]``; falls back to the cache
    group's ``optimizer_assignment`` (old-grouping queue item). The bool is
    ``True`` when the sidecar supplied the set.

    Pre-flag sidecars: mirrors ``stitch_queue_refresh.selected_pair_set`` — when
    the sidecar's edges carry NO ``selected`` key at all (a pack predating the
    flag), the sidecar cannot express the optimizer set, so fall back to the
    cache rather than returning an empty set (which would make every full-grid
    label look like it "adds pairs beyond the optimizer" and over-flag
    ratifications as artifacts). Edges that DO carry the flag but select nothing
    are a genuine reject-all and are returned as empty.
    """
    if sidecar_group is not None:
        edges = sidecar_group.get("edges") or []
        selected = edge_pairs(edges, selected_only=True)
        if selected or any("selected" in e for e in edges):
            return selected, True
    if cache_group is not None:
        return edge_pairs(cache_group.get("optimizer_assignment")), False
    return set(), False


def compute_diff(label_pairs: set[Pair], opt_pairs: set[Pair]) -> tuple[set[Pair], set[Pair]]:
    """Return (added, removed): pairs the label adds vs / drops from the optimizer."""
    return label_pairs - opt_pairs, opt_pairs - label_pairs


def candidate_universe(cache_group: dict | None, sidecar_group: dict | None) -> set[Pair]:
    """All (ref_id, target_id) pairs the reviewer could have chosen from.

    Union of cache ``edges``, sidecar ``edges`` and sidecar ``rejected_edges`` —
    every candidate pair surfaced in the group, selected or not.
    """
    return (
        edge_pairs((cache_group or {}).get("edges"))
        | edge_pairs((sidecar_group or {}).get("edges"))
        | edge_pairs((sidecar_group or {}).get("rejected_edges"))
    )


def crossproduct_within_universe(label_pairs: set[Pair], universe: set[Pair]) -> set[Pair]:
    """The (label refs × label targets) grid, restricted to candidate pairs.

    In manual / de-anchored labelling the submit recorded the full cross-product
    of the active ref-pills and target-pills intersected with the candidate
    universe — so this is what a "select-all-pills" submit would have stored.
    """
    refs = {a for a, _ in label_pairs}
    tgts = {b for _, b in label_pairs}
    return {(r, t) for r in refs for t in tgts if (r, t) in universe}


def is_crossproduct_artifact(
    label_pairs: set[Pair],
    opt_pairs: set[Pair],
    universe: set[Pair],
) -> bool:
    """Flag labels whose extra pairs are likely cross-product artifacts.

    True when the stored pair set is *exactly* the ref×target cross-product
    within the candidate universe AND it adds pairs beyond the optimizer — i.e.
    the reviewer's pill selection over-expanded into pairs they may never have
    consciously chosen. Pure exclusions (added set empty) never flag.

    Requires a genuine grid — at least two refs *and* two targets. A 1:1 or
    1:N (single ref or single target) selection has no meaningful cross-product,
    so a deliberately-added single pair or a legitimate fan is never flagged.
    """
    if not label_pairs:
        return False
    refs = {a for a, _ in label_pairs}
    tgts = {b for _, b in label_pairs}
    if len(refs) < 2 or len(tgts) < 2:  # no ref×target grid to over-expand into
        return False
    if not (label_pairs - opt_pairs):  # no pairs beyond the optimizer
        return False
    return label_pairs == crossproduct_within_universe(label_pairs, universe)


def reinterpret_row_to_set(
    row,
    cache_group: dict | None,
    sidecar_group: dict | None,
) -> tuple[list[str], list[str]] | None:
    """Decide whether a stitching label row should become a SET label.

    Shared decision used by ``crosswalk data stitch-reinterpret-sets`` and its
    tests. Returns ``(sorted_ref_ids, sorted_target_ids)`` when the row is a
    cross-product artifact that should be converted, else ``None``.

    Never converts (returns None) for:
      * ``panel_*`` labeler rows (agent labels — provenance-protected),
      * rows already ``label_semantics=set`` (idempotent re-runs),
      * rows whose group appears in NEITHER the stitch cache NOR the groups
        sidecar (no source can establish the candidate universe). Either source
        alone suffices: the universe is cache ``edges`` ∪ sidecar ``edges`` ∪
        sidecar ``rejected_edges``, and the optimizer selection prefers the
        sidecar — geometries (cache-only) are not needed here, unlike in the
        render script,
      * sidecar-only rows whose group has a truncated ``rejected_edges`` list
        (incomplete universe — conservative skip),
      * rows that do not match the cross-product signature (explicit
        ratifications, deliberate single/fan picks, pure exclusions).

    ``row`` is any mapping with ``labeler``, ``label_semantics``, and
    ``selected_edges`` keys (a pandas Series or a dict).
    """
    lab = str(row.get("labeler") or "")
    if lab.startswith("panel"):
        return None
    if str(row.get("label_semantics") or "pair") == "set":
        return None
    if cache_group is None and sidecar_group is None:
        return None
    # Sidecar-only universe with a TRUNCATED rejected_edges list: the candidate
    # universe is provably incomplete (pairs past the per-group cap are absent),
    # so ``label_pairs == grid ∩ universe`` could hold spuriously and convert a
    # deliberate partial-grid label. The cache (review-queue) universe carries
    # the label-time edges, so it anchors the check; without it, be conservative
    # and leave the row untouched.
    if (
        cache_group is None
        and sidecar_group is not None
        and bool(sidecar_group.get("rejected_truncated"))
    ):
        return None
    label_pairs = parse_selected_edges(row.get("selected_edges"))
    opt_pairs, _ = resolve_optimizer(cache_group, sidecar_group)
    universe = candidate_universe(cache_group, sidecar_group)
    if not is_crossproduct_artifact(label_pairs, opt_pairs, universe):
        return None
    return sorted({a for a, _ in label_pairs}), sorted({b for _, b in label_pairs})
