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
    """Build the ``R#/T# -> id`` label map from a ``batch.json`` group.

    Mirrors ``stitch_export._meta_from_group`` / the ``stitch_evidence`` labeling
    scheme: ``R{i+1} -> ref_ids[i]``, ``T{i+1} -> target_ids[i]`` in the group's
    stored id order (falling back to the sorted ids seen on ``edges`` when the
    explicit id lists are absent). The pack's ``metadata.yaml`` segments remain
    the authoritative source when available (see
    ``stitch_diagnostic._label_maps``); this is the convenience path for callers
    holding only a ``batch.json`` group.
    """
    ref_ids = group.get("ref_ids") or sorted({str(e["ref_id"]) for e in group.get("edges", [])})
    tgt_ids = group.get("target_ids") or sorted(
        {str(e["target_id"]) for e in group.get("edges", [])}
    )
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
    ``generate_top_k_alternatives(seed_edge_sets=...)``.
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
