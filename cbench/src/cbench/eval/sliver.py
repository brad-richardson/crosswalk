"""Standalone junction-sliver classification for cbench.

cbench is a standalone package and does NOT depend on matcher. This module
replicates the minimal sliver rule from ``matcher.config.is_sliver_edge`` (the
pure numeric classifier) and the group-level geometry handling from
``matcher.matching.sliver``, using only pure Python (no shapely / pyproj) so
cbench keeps its light dependency footprint.

A parity test on the matcher side
(``tests/unit/test_cbench_sliver_parity.py``) asserts this classifier matches
``matcher.config.is_sliver_edge`` across a grid of representative inputs so the
two definitions cannot drift silently.

A junction "sliver" is a candidate edge where two segments barely overlap —
typically where a road end clips the side of another road at an intersection.
The HYBRID rule (both conditions must hold):

  1. FRACTION test: ``max(ref_span_frac, tgt_span_frac) < SLIVER_SPAN_THRESHOLD``
  2. ABSOLUTE test: ``max(ref_span_frac*ref_len_m, tgt_span_frac*tgt_len_m)
                       < SLIVER_ABS_OVERLAP_M``

Edges with missing/NaN fractions default to a full [0, 1] span (1.0) and
missing/NaN lengths default to +inf meters, so an unmeasurable edge is NEVER
classified as a sliver.
"""

from __future__ import annotations

import math

# Keep these in sync with matcher.config (enforced by the parity test).
SLIVER_SPAN_THRESHOLD = 0.10  # fraction of segment length (dimensionless)
SLIVER_ABS_OVERLAP_M = 5.0  # absolute overlap floor (meters)

_EARTH_RADIUS_M = 6371008.8  # mean Earth radius (IUGG)


def _sliver_frac(value: float | None, default: float) -> float:
    """Normalize an alignment span fraction, defaulting missing/NaN to ``default``."""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    return abs(v)


def _sliver_len(value: float | None) -> float:
    """Normalize a segment length (meters), defaulting missing/NaN/<=0 to +inf."""
    if value is None:
        return math.inf
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.inf
    if math.isnan(v) or v <= 0:
        return math.inf
    return v


def is_sliver_edge(
    ref_span_frac: float | None,
    tgt_span_frac: float | None,
    ref_len_m: float | None = None,
    tgt_len_m: float | None = None,
) -> bool:
    """Classify a candidate edge as a junction sliver using the hybrid rule.

    Mirror of ``matcher.config.is_sliver_edge`` (verified by the parity test).
    """
    rf = _sliver_frac(ref_span_frac, 1.0)
    tf = _sliver_frac(tgt_span_frac, 1.0)
    rl = _sliver_len(ref_len_m)
    tl = _sliver_len(tgt_len_m)

    frac_test = max(rf, tf) < SLIVER_SPAN_THRESHOLD
    abs_overlap = max(rf * rl, tf * tl)
    abs_test = abs_overlap < SLIVER_ABS_OVERLAP_M
    return frac_test and abs_test


def _geojson_length_m(gj: dict | None) -> float | None:
    """Length in meters of a GeoJSON LineString (WGS84 lon/lat), or None.

    Uses a local equirectangular approximation (scale longitude by cos(lat)),
    which is within ~0.3% of a proper UTM projection for the short segments seen
    in road data — far below the sliver thresholds — and needs no shapely/pyproj.
    """
    if not gj:
        return None
    coords = gj.get("coordinates")
    if not coords or len(coords) < 2:
        return None
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
        mean_lat = math.radians((lat1 + lat2) / 2.0)
        dx = math.radians(lon2 - lon1) * math.cos(mean_lat) * _EARTH_RADIUS_M
        dy = math.radians(lat2 - lat1) * _EARTH_RADIUS_M
        total += math.hypot(dx, dy)
    return total


def _edge_span_fracs(edge: dict) -> tuple[float, float]:
    """Return (ref_span, tgt_span) alignment fractions for a group edge dict.

    Missing fracs default to a full [0, 1] span (1.0) so an unmeasurable edge is
    never mistaken for a sliver.
    """

    def _frac(key: str, default: float) -> float:
        v = edge.get(key)
        return default if v is None else float(v)

    ref_span = abs(_frac("gers_end_frac", 1.0) - _frac("gers_start_frac", 0.0))
    tgt_span = abs(_frac("local_end_frac", 1.0) - _frac("local_start_frac", 0.0))
    return ref_span, tgt_span


def group_sliver_edges(group: dict) -> frozenset[tuple[str, str]]:
    """Return the set of ``(ref_id, target_id)`` edges in a group that are slivers.

    Segment lengths are computed once from the group's stored GeoJSON geometries
    (``ref_geometries`` / ``target_geometries``, WGS84). Edges whose geometry is
    missing fall back to the fraction-only behaviour (length +inf), which can
    only make an edge LESS likely to be a sliver. Empty when the group carries no
    geometries (nothing classifiable -> filtered == raw).
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

    slivers: set[tuple[str, str]] = set()
    for e in group.get("edges") or []:
        ref_span, tgt_span = _edge_span_fracs(e)
        ref_id = str(e.get("ref_id"))
        tgt_id = str(e.get("target_id"))
        if is_sliver_edge(ref_span, tgt_span, ref_lens.get(ref_id), tgt_lens.get(tgt_id)):
            slivers.add((ref_id, tgt_id))
    return frozenset(slivers)
