"""Consensus-desired-edges option-menu seed.

The v8 stitch panel prompt contract (#451) asks a seat that votes ``NONE`` with
``none_reason == "no_exact_option"`` to record, in the ``desired_edges`` column
of ``votes.csv``, the exact ``R#/T#`` edge set it WANTED but that no displayed
option expressed. The v8 analysis found 38 residual ``no_exact_option`` NONEs
where the correct set simply was not on the menu — and in several groups
MULTIPLE seats independently emitted the SAME desired set (after mapping the
``R#/T#`` display labels through the batch's label map to source ids), e.g.
``bdbdf792`` (all three seats), ``00e8e9fd`` (codex+muse), ``fb8f359f``
(claude+codex).

A set that >= 2 INDEPENDENT seats wanted, but the menu never offered, is a
strong signal the menu should have included it. This module turns that signal
into an OPTION-MENU SEED: it returns the consensus-desired edge sets (mapped to
ids, non-empty, subset of the candidate universe, and different from every
offered option) so they can be injected as ``is_seed`` options on the next pack
build — a sibling to the exact-pair option seeds (#450) via
``generate_top_k_alternatives(seed_edge_sets=...)``. This closes an
expressibility gap that pure menu enumeration (cap-4 minus-edge / max-20) cannot
reach on 14-19-segment groups.

The ``R#/T# -> id`` mapping mirrors ``stitch_diagnostic._label_maps`` (the map
the diagnostic/analysts used): a ``{"reference": {"R1": id, ...}, "target":
{"T1": id, ...}}`` dict, authoritatively the pack's ``metadata.yaml`` segments.
:func:`label_map_from_group` builds the equivalent map from a ``batch.json``
group's ordered ``ref_ids``/``target_ids`` (mirroring
``stitch_export._meta_from_group``) when no ``metadata.yaml`` is at hand.

This module is pure (no I/O) and never raises on malformed ballot content — a
bad ``desired_edges`` cell contributes nothing rather than failing the build.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping

# The ``none_reason`` enum value a ballot carries when the voter wanted a set no
# option expressed. Kept in sync with ``stitch_runner.NONE_REASONS``.
NO_EXACT_OPTION = "no_exact_option"

EdgeKey = tuple[str, str]


def _text(value: object) -> str:
    """Coerce a cell (str / None / NaN float) to a stripped string ("" for null)."""
    if value is None:
        return ""
    # pandas NaN is a float that is not equal to itself.
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def parse_desired_edges(cell: object) -> list[tuple[str, str]]:
    """Parse a ``votes.csv`` ``desired_edges`` cell into ``(R#, T#)`` label pairs.

    Accepts the canonical JSON list-of-pairs the runner writes
    (``[["R1","T4"], ...]``), an already-decoded list, or a list of
    ``{"ref_id","target_id"}`` dicts. Never raises: an empty/malformed cell
    yields ``[]``.
    """
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        try:
            data: object = json.loads(s)
        except (ValueError, TypeError):
            return []
    else:
        data = cell
    if not isinstance(data, list):
        return []
    out: list[tuple[str, str]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            r, t = _text(item[0]), _text(item[1])
        elif isinstance(item, Mapping):
            r, t = _text(item.get("ref_id")), _text(item.get("target_id"))
        else:
            continue
        if r and t:
            out.append((r, t))
    return out


def parse_seed_edges_map(obj: object) -> dict[str, list[frozenset[EdgeKey]]]:
    """Parse a ``{group_id: [edge_set, ...]}`` seed spec into id-space frozensets.

    This is the file format for the ``crosswalk agent stitch-batch
    --seed-edges-file`` option: a JSON object mapping each ``group_id`` to a LIST
    of edge sets to inject as option-menu seeds (``generate_top_k_alternatives``'s
    ``seed_edge_sets``). Each edge set is a list of ``[ref_id, target_id]`` SOURCE
    id pairs (or ``{"ref_id","target_id"}`` dicts) — already-resolved ids, NOT
    ``R#/T#`` display labels, so the file is drift-safe (source ids survive
    group-id drift; ``R#/T#`` positions do not — see
    :func:`label_map_from_group`). Supplying already-mapped ids is exactly what a
    future automated resolver built on :func:`consensus_seed_edge_sets_for_group`
    would hand the pack build; this file lets an operator feed the same seam by
    hand (e.g. a consensus-desired set from a prior ``no_exact_option`` panel).

    A group's value MUST be a list of edge sets, so a single desired set is
    ``[[[ref, tgt], ...]]``. Mirrors :func:`parse_desired_edges` for robustness:
    a malformed group value or a malformed/empty edge set contributes nothing
    rather than raising, per-group sets are deduped preserving order, and a group
    left with no valid set is omitted entirely.
    """
    out: dict[str, list[frozenset[EdgeKey]]] = {}
    if not isinstance(obj, Mapping):
        return out
    for gid, sets in obj.items():
        key = _text(gid)
        if not key or not isinstance(sets, list):
            continue
        parsed: list[frozenset[EdgeKey]] = []
        seen: set[frozenset[EdgeKey]] = set()
        for one in sets:
            pairs = parse_desired_edges(one)
            if not pairs:
                continue
            fs = frozenset((str(r), str(t)) for r, t in pairs)
            if fs and fs not in seen:
                seen.add(fs)
                parsed.append(fs)
        if parsed:
            out[key] = parsed
    return out


def map_desired_to_ids(
    desired: Iterable[tuple[str, str]],
    label_map: Mapping[str, Mapping[str, str]],
) -> frozenset[EdgeKey] | None:
    """Map ``(R#, T#)`` display labels to ``(ref_id, target_id)`` source ids.

    ``label_map`` is ``{"reference": {"R1": id, ...}, "target": {"T1": id, ...}}``
    (``stitch_diagnostic._label_maps`` shape). Returns ``None`` — treated as
    degenerate/unmappable and skipped by the detector — when ``desired`` is empty
    or ANY label has no id in the map (an all-or-nothing rule: a partially mapped
    set would misrepresent what the seat wanted).
    """
    ref_map = label_map.get("reference", {}) or {}
    tgt_map = label_map.get("target", {}) or {}
    mapped: set[EdgeKey] = set()
    any_edge = False
    for r, t in desired:
        any_edge = True
        rid = ref_map.get(r)
        tid = tgt_map.get(t)
        if rid is None or tid is None:
            return None
        mapped.add((str(rid), str(tid)))
    if not any_edge or not mapped:
        return None
    return frozenset(mapped)


def _normalize_sets(sets: Iterable[Iterable[EdgeKey]]) -> set[frozenset[EdgeKey]]:
    return {frozenset((str(r), str(t)) for r, t in s) for s in sets}


def _iter_ballots(ballots: object) -> list[Mapping]:
    """Normalize ballots (list of dict-likes or a pandas DataFrame) to dict rows."""
    to_dict = getattr(ballots, "to_dict", None)
    if callable(to_dict) and hasattr(ballots, "columns"):
        return list(ballots.to_dict("records"))
    return list(ballots)  # type: ignore[arg-type]


def _seat_of(row: Mapping) -> str:
    """The independent-seat identity of a ballot (``provider``, else ``model``)."""
    seat = _text(row.get("provider"))
    return seat or _text(row.get("model"))


def consensus_desired_edge_sets(
    ballots: object,
    label_map: Mapping[str, Mapping[str, str]],
    offered_option_sets: Iterable[Iterable[EdgeKey]],
    candidate_edges: Iterable[EdgeKey] | None = None,
    min_seats: int = 2,
) -> list[frozenset[EdgeKey]]:
    """Consensus-desired edge sets for one group's archived ballots.

    A set qualifies when, after mapping ``R#/T#`` -> ids, it is:

    * wanted by ``>= min_seats`` DISTINCT seats (``provider``), each via a
      ``none_reason == "no_exact_option"`` ballot (a seat is counted once per
      set no matter how many attempts it cast);
    * non-empty and fully mappable (``map_desired_to_ids`` returned a set);
    * a subset of ``candidate_edges`` when supplied (a set referencing an edge
      outside the current candidate universe cannot be built as an option, so it
      is skipped as degenerate); and
    * different from EVERY offered option (a set the menu already expresses is
      not a gap).

    Returns a deterministic list (fewest edges first, then sorted pairs). Pure;
    never raises on malformed ballot content.
    """
    offered = _normalize_sets(offered_option_sets)
    candidate: set[EdgeKey] | None = (
        {(str(r), str(t)) for r, t in candidate_edges} if candidate_edges is not None else None
    )

    seats_by_set: dict[frozenset[EdgeKey], set[str]] = defaultdict(set)
    for row in _iter_ballots(ballots):
        if _text(row.get("none_reason")) != NO_EXACT_OPTION:
            continue
        desired = parse_desired_edges(row.get("desired_edges"))
        mapped = map_desired_to_ids(desired, label_map)
        if mapped is None:
            continue
        if candidate is not None and not (mapped <= candidate):
            continue
        seat = _seat_of(row)
        if not seat:
            continue
        seats_by_set[mapped].add(seat)

    consensus = [
        es for es, seats in seats_by_set.items() if len(seats) >= min_seats and es not in offered
    ]
    consensus.sort(key=lambda es: (len(es), sorted(es)))
    return consensus


def label_map_from_group(group: Mapping) -> dict[str, dict[str, str]]:
    """Build the ``R#/T# -> id`` label map for a group, matching the pack labeler.

    Mirrors ``stitch_evidence._seg_labels`` EXACTLY — that is the function that
    stamped the ``R#/T#`` labels the seat actually saw, so any other ordering
    would map desired sets to the WRONG source ids silently. The label order is:
    ``group["ref_ids"]`` / ``group["target_ids"]`` when present, else the
    **insertion order of the geometry dicts** (``ref_geometries`` /
    ``target_geometries`` keys) — NOT a sorted order. If neither ids nor
    geometries are present the corresponding side maps to ``{}`` and
    :func:`map_desired_to_ids` will treat the desired set as unmappable (returns
    ``None``) rather than guessing.

    !!! IMPORTANT — historical label maps !!!
    Call this ONLY with the group whose pack a ballot was voted against. When the
    future CLI wiring resolves an ARCHIVED ballot, it must map that ballot's
    ``R#/T#`` through the ballot's OWN originating batch's ``batch.json`` /
    ``metadata.yaml`` label map (authoritatively ``stitch_diagnostic._label_maps``),
    NEVER through this helper on the current (drifted) group: source ids survive
    group-id drift but ``R#/T#`` label positions do NOT, so cross-mapping a
    historical ballot through a re-labeled current group silently corrupts the
    edge set. The seam is deliberately id-space and validated against the current
    candidate universe precisely to keep this property enforceable.
    """
    ref_ids = group.get("ref_ids")
    if ref_ids is None:
        ref_ids = list(group.get("ref_geometries", {}).keys())
    tgt_ids = group.get("target_ids")
    if tgt_ids is None:
        tgt_ids = list(group.get("target_geometries", {}).keys())
    return {
        "reference": {f"R{i + 1}": str(r) for i, r in enumerate(ref_ids)},
        "target": {f"T{i + 1}": str(t) for i, t in enumerate(tgt_ids)},
    }


def consensus_seed_edge_sets_for_group(
    group: Mapping,
    ballots: object,
    label_map: Mapping[str, Mapping[str, str]],
    k: int = 8,
    min_seats: int = 2,
) -> list[frozenset[EdgeKey]]:
    """Consensus-desired seed sets for a sidecar/``batch.json`` group.

    Builds the group's CURRENT offered option menu (the same
    ``generate_top_k_alternatives`` + ``build_stitch_options`` path the pack and
    review UI use, so the exact-pair seeds are included in ``offered`` and never
    re-seeded), derives the candidate universe from the group's edges, and
    returns the detector's consensus sets — ready to hand back to
    ``generate_top_k_alternatives(seed_edge_sets=..., max_total_options=...)``
    (pass ``settings.stitch_panel_max_options`` so injection respects the
    max-menu bound; the generator additionally caps the count at
    ``MAX_INJECTED_SEEDS``).

    !!! IMPORTANT — ``label_map`` must be the ballots' OWN pack map !!!
    ``label_map`` maps ``R#/T#`` for the ballots supplied. Pass the map of the
    batch the ballots were voted against — for archived ballots that is their
    originating ``batch.json`` / ``metadata.yaml`` map, resolved per source
    batch, NOT :func:`label_map_from_group` on the current drifted group (see
    that function's warning). The detector returns id-space sets validated
    against this group's current candidate universe, so a stale/cross-mapped
    label map cannot smuggle in an edge that does not exist here — but it CAN
    still map to the wrong (yet extant) edge, so supplying the correct map is the
    caller's responsibility.
    """
    from ..matching.alternatives import generate_top_k_alternatives
    from ..matching.stitch_options import build_stitch_options

    candidate_edges = {(str(e["ref_id"]), str(e["target_id"])) for e in group.get("edges", [])}
    g = dict(group)
    g["alternatives"] = generate_top_k_alternatives(
        group.get("edges", []),
        ref_geoms=group.get("ref_geometries", {}),
        target_geoms=group.get("target_geometries", {}),
        k=k,
    )
    ctx = build_stitch_options(g)
    offered = [[(e["ref_id"], e["target_id"]) for e in opt["edges"]] for opt in ctx["options"]]
    return consensus_desired_edge_sets(
        ballots,
        label_map,
        offered,
        candidate_edges=candidate_edges,
        min_seats=min_seats,
    )
