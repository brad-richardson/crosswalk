"""Junction-sliver classification helpers over edge/group data structures.

This module is the shared, geometry-aware layer on top of the pure numeric
classifier :func:`matcher.config.is_sliver_edge`. All consumers (the stitching
review UI, the agent-labeling evaluation, and the optimizer's component graph)
route through here so humans, agents, and the optimizer agree on exactly which
edges are junction slivers.

The hybrid rule and its rationale (and residual limitation) live in ``config.py``
next to ``SLIVER_SPAN_THRESHOLD`` / ``SLIVER_ABS_OVERLAP_M``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shapely.geometry import shape

from ..config import (
    SLIVER_BORDERLINE_SPAN_THRESHOLD,
    is_sliver_edge,
    sliver_overlap_m,
)
from ..utils.geometry import geometry_length_meters

if TYPE_CHECKING:
    from .types import MatchResult


def edge_span_fracs(edge: dict) -> tuple[float, float]:
    """Return (ref_span, tgt_span) alignment fractions for an edge dict.

    Missing fracs default to a full [0, 1] span (1.0) so an unmeasurable edge is
    never mistaken for a sliver.
    """

    def _frac(key: str, default: float) -> float:
        v = edge.get(key)
        return default if v is None else float(v)

    ref_span = abs(_frac("gers_end_frac", 1.0) - _frac("gers_start_frac", 0.0))
    tgt_span = abs(_frac("local_end_frac", 1.0) - _frac("local_start_frac", 0.0))
    return ref_span, tgt_span


def group_segment_lengths_m(group: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Compute metric lengths (meters) for every ref/target segment in a group.

    Lengths are derived once from the group's stored GeoJSON geometries (WGS84)
    and keyed by segment id, so per-edge sliver classification is O(1) afterward.
    Segments without a usable geometry are omitted (their length reads as unknown
    → treated as +inf → never a sliver).
    """
    ref_lens: dict[str, float] = {}
    for sid, gj in (group.get("ref_geometries") or {}).items():
        length = _geojson_length_m(gj)
        if length is not None:
            ref_lens[str(sid)] = length

    tgt_lens: dict[str, float] = {}
    for sid, gj in (group.get("target_geometries") or {}).items():
        length = _geojson_length_m(gj)
        if length is not None:
            tgt_lens[str(sid)] = length

    return ref_lens, tgt_lens


def _geojson_length_m(gj: dict | None) -> float | None:
    if not gj:
        return None
    try:
        geom = shape(gj)
    except Exception:
        return None
    if geom.is_empty:
        return None
    return geometry_length_meters(geom)


def edge_is_sliver(
    edge: dict,
    ref_lens: dict[str, float] | None = None,
    tgt_lens: dict[str, float] | None = None,
) -> bool:
    """Classify an edge dict as a junction sliver using the hybrid rule.

    ``ref_lens`` / ``tgt_lens`` map segment id -> length in meters (from
    :func:`group_segment_lengths_m`). When lengths are unavailable the classifier
    falls back to the fraction-only behaviour for that side (length +inf), which
    can only make an edge LESS likely to be a sliver — never more.
    """
    ref_span, tgt_span = edge_span_fracs(edge)
    ref_len = (ref_lens or {}).get(str(edge.get("ref_id")))
    tgt_len = (tgt_lens or {}).get(str(edge.get("target_id")))
    return is_sliver_edge(ref_span, tgt_span, ref_len, tgt_len)


def edge_overlap_m(
    edge: dict,
    ref_lens: dict[str, float] | None = None,
    tgt_lens: dict[str, float] | None = None,
) -> float:
    """Absolute aligned-overlap (meters) for an edge dict.

    Thin geometry-aware wrapper over :func:`matcher.config.sliver_overlap_m` — the
    single definition the sliver rule's absolute gate uses. Returns +inf when the
    relevant segment length is unknown (same convention as the classifier).
    """
    ref_span, tgt_span = edge_span_fracs(edge)
    ref_len = (ref_lens or {}).get(str(edge.get("ref_id")))
    tgt_len = (tgt_lens or {}).get(str(edge.get("target_id")))
    return sliver_overlap_m(ref_span, tgt_span, ref_len, tgt_len)


def edge_is_borderline(
    edge: dict,
    ref_lens: dict[str, float] | None = None,
    tgt_lens: dict[str, float] | None = None,
) -> bool:
    """DISPLAY-ONLY near-sliver classification for an edge dict.

    Returns True when the edge is NOT a strict junction sliver
    (:func:`edge_is_sliver`) yet its larger coverage fraction is still below
    ``SLIVER_BORDERLINE_SPAN_THRESHOLD``. This surfaces the junction-kiss edges
    the hybrid rule leaves untagged — those failing the sliver test only on the
    5 m absolute floor (a tiny span that maps to >= 5 m on a long urban segment),
    plus those sitting just above the span threshold. It never overlaps the
    SLIVER tag (a sliver is never also borderline) and is not consumed by the
    optimizer or any label gate. See ``config.SLIVER_BORDERLINE_SPAN_THRESHOLD``.
    """
    if edge_is_sliver(edge, ref_lens, tgt_lens):
        return False
    ref_span, tgt_span = edge_span_fracs(edge)
    return max(ref_span, tgt_span) < SLIVER_BORDERLINE_SPAN_THRESHOLD


def edge_sliver_tag(
    edge: dict,
    ref_lens: dict[str, float] | None = None,
    tgt_lens: dict[str, float] | None = None,
) -> str | None:
    """Return the display tag for an edge: ``"SLIVER"``, ``"BORDERLINE"``, or None.

    SLIVER takes precedence (the validated hybrid definition, #244); BORDERLINE
    is the display-only near-sliver band. Both are pack-display concerns only.
    """
    if edge_is_sliver(edge, ref_lens, tgt_lens):
        return "SLIVER"
    if edge_is_borderline(edge, ref_lens, tgt_lens):
        return "BORDERLINE"
    return None


def annotate_group_sliver_flags(group: dict) -> tuple[list[dict], int]:
    """Return (edges-with-``is_sliver``-flag, sliver_count) for a group.

    Each returned edge is a shallow copy of the group edge with an added
    ``is_sliver`` boolean. Segment lengths are computed once from the group's
    geometries so the hybrid (fraction + absolute-meters) rule is applied.
    """
    edges = group.get("edges") or []
    ref_lens, tgt_lens = group_segment_lengths_m(group)
    out: list[dict] = []
    sliver_count = 0
    for e in edges:
        ce = dict(e)
        flag = edge_is_sliver(e, ref_lens, tgt_lens)
        ce["is_sliver"] = flag
        sliver_count += int(flag)
        out.append(ce)
    return out, sliver_count


def sliver_edges_for_match_results(
    results: list[MatchResult],
    ref_geoms: dict[Any, Any],
    target_geoms: dict[Any, Any],
    metric: bool = True,
) -> set[tuple[Any, Any]]:
    """Classify MatchResult candidate edges with the hybrid sliver rule.

    Used by the optimizer's component graph (and the groups-sidecar export) so
    grouping agrees with the UI/eval definition of a junction sliver.

    Args:
        results: Candidate MatchResults (alignment fracs read from the result).
        ref_geoms: id -> shapely geometry lookup for reference segments.
        target_geoms: id -> shapely geometry lookup for target segments.
        metric: True when the geometries are in a projected CRS (``.length`` is
            meters). False for WGS84 lon/lat input, in which case lengths are
            measured via :func:`geometry_length_meters`.

    Returns:
        Set of ``(ref_id, target_id)`` pairs classified as junction slivers.
        Results with missing fracs or missing geometries are never slivers
        (the config-level defaults apply).
    """

    def _length(geom) -> float | None:
        if geom is None or getattr(geom, "is_empty", True):
            return None
        return geom.length if metric else geometry_length_meters(geom)

    # Deduplicate to the highest-confidence result per pair, mirroring the
    # component builder's duplicate handling, so classification is based on the
    # edge instance the optimizer would actually keep.
    best_by_pair: dict[tuple[Any, Any], MatchResult] = {}
    for r in results:
        key = (r.ref_id, r.target_id)
        if key not in best_by_pair or r.confidence > best_by_pair[key].confidence:
            best_by_pair[key] = r

    ref_len_cache: dict[Any, float | None] = {}
    tgt_len_cache: dict[Any, float | None] = {}
    slivers: set[tuple[Any, Any]] = set()

    for key, r in best_by_pair.items():
        ref_span = None
        if r.gers_start_frac is not None and r.gers_end_frac is not None:
            ref_span = abs(r.gers_end_frac - r.gers_start_frac)
        tgt_span = None
        if r.local_start_frac is not None and r.local_end_frac is not None:
            tgt_span = abs(r.local_end_frac - r.local_start_frac)

        if r.ref_id not in ref_len_cache:
            ref_len_cache[r.ref_id] = _length(ref_geoms.get(r.ref_id))
        if r.target_id not in tgt_len_cache:
            tgt_len_cache[r.target_id] = _length(target_geoms.get(r.target_id))

        if is_sliver_edge(ref_span, tgt_span, ref_len_cache[r.ref_id], tgt_len_cache[r.target_id]):
            slivers.add(key)

    return slivers
