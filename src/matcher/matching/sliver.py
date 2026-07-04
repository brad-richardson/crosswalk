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

from shapely.geometry import shape

from ..config import is_sliver_edge
from ..utils.geometry import geometry_length_meters


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
