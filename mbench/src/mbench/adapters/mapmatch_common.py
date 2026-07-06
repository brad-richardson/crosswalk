"""Shared helpers for map-matching baselines (Valhalla Meili, GraphHopper).

Both adapters follow the same formulation — densify each local target segment
into a synthetic GPS trace, map-match it onto the Overture-derived routable graph
(built once by ``convert/pbf.py``), then aggregate the matched reference edges per
target with an overlap-length filter that stands in for a first-class no-match
abstention. Only the *engine* (Valhalla vs GraphHopper) differs; the trace
building and edge aggregation are identical, so they live here.
"""

from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
import shapely


def densify_lonlat(geom, metric_crs, densify_m: float) -> list[tuple[float, float]]:
    """Return a densified lon/lat vertex sequence for a (Multi)LineString.

    Densification is done in a metric CRS so spacing is truly ~``densify_m``
    meters, then reprojected back to lon/lat. Returns the longest constituent
    part for a MultiLineString. Input geometry is assumed to be EPSG:4326.
    """
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "MultiLineString":
        parts = [g for g in geom.geoms if not g.is_empty]
        if not parts:
            return []
        geom = max(parts, key=lambda g: g.length)
    if geom.geom_type != "LineString":
        return []
    gs = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(metric_crs)
    dens = gs.segmentize(densify_m).to_crs("EPSG:4326").iloc[0]
    coords = shapely.get_coordinates(dens)
    return [(float(x), float(y)) for x, y in coords]


def aggregate_edges(
    edge_lengths_m: Iterable[tuple[object, float]],
    id_map: dict[str, str],
    target_len_m: float,
    min_frac: float,
    min_m: float,
) -> list[tuple[str, float]]:
    """Aggregate matched edges into ``(gers_id, confidence)`` with overlap filtering.

    ``edge_lengths_m`` is an iterable of ``(way_id, matched_length_meters)`` — one
    entry per matched edge, where a routing engine may split a single reference way
    into several edges (all sharing the same ``way_id``). Matched lengths are summed
    per GERS id (a MultiLineString reference maps several way_ids to one GERS id).
    A GERS id is kept if its matched length is >= ``min_frac`` of the target length
    OR >= ``min_m`` meters — dropping spuriously-touched edges (e.g. a snap that
    clips one node of a crossing street) without penalizing legitimate short
    slivers. Confidence is the matched-length fraction of the target, capped at 1.0.
    """
    per_gers: dict[str, float] = {}
    for way_id, length_m in edge_lengths_m:
        if way_id is None:
            continue
        gers = id_map.get(str(way_id))
        if gers is None:
            continue
        per_gers[gers] = per_gers.get(gers, 0.0) + float(length_m)

    out: list[tuple[str, float]] = []
    denom = target_len_m if target_len_m > 0 else 1.0
    for gers, matched_m in per_gers.items():
        frac = matched_m / denom
        if frac >= min_frac or matched_m >= min_m:
            out.append((gers, min(1.0, frac)))
    return out
