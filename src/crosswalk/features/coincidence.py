"""Geometry-derived same-side coincidence context for stitching.

Near-coincident alternatives make pairwise geometry non-identifying: two roads
on one provider side can occupy effectively the same centerline while differing
in network role or vertical representation. This module measures that ambiguity
without inventing a physical layer value. It is intentionally experimental and
is not part of the production pairwise feature contract yet.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pyproj
import shapely
from shapely.ops import transform


@dataclass(frozen=True)
class CoincidentAlternativeResult:
    """Coverage and role ambiguity induced by same-side alternatives."""

    covered_fraction: float
    covered_length_m: float
    max_symmetric_fraction: float
    alternative_count: int
    has_role_conflict: bool
    alternative_ids: tuple[str, ...] = ()


@lru_cache(maxsize=64)
def _cached_transformer(source_crs: str, target_crs: str) -> pyproj.Transformer:
    """Reuse CRS pipelines while scanning thousands of groups in one dataset."""
    return pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)


def compute_coincident_alternatives(
    segment,
    alternatives: Iterable[tuple[Any, Any] | tuple[Any, Any, Any]],
    *,
    segment_role: Any = None,
    tolerance_m: float = 3.0,
    qualifying_fraction: float = 0.8,
    min_coincident_length_m: float = 20.0,
) -> CoincidentAlternativeResult:
    """Measure how much a segment is geometrically ambiguous on its own side.

    Each alternative is ``(geometry, role)``. An alternative qualifies when at
    least one line is mostly covered by the other's tolerance corridor. The
    symmetric rule catches a short surface road that lies entirely over one
    portion of a much longer tunnel/reference line, as in Geneva.
    """
    if segment is None or segment.is_empty or segment.length <= 0:
        return CoincidentAlternativeResult(0.0, 0.0, 0.0, 0, False)

    qualifying_buffers = []
    max_symmetric_fraction = 0.0
    alternative_count = 0
    has_role_conflict = False
    alternative_ids: list[str] = []
    for alternative in alternatives:
        geometry, role = alternative[:2]
        alternative_id = str(alternative[2]) if len(alternative) > 2 else None
        if geometry is None or geometry.is_empty or geometry.length <= 0:
            continue
        segment_overlap = segment.intersection(geometry.buffer(tolerance_m)).length
        alternative_overlap = geometry.intersection(segment.buffer(tolerance_m)).length
        segment_fraction = min(max(segment_overlap / segment.length, 0.0), 1.0)
        alternative_fraction = min(max(alternative_overlap / geometry.length, 0.0), 1.0)
        symmetric_fraction = max(segment_fraction, alternative_fraction)
        max_symmetric_fraction = max(max_symmetric_fraction, symmetric_fraction)
        coincident_length = min(segment_overlap, alternative_overlap)
        if (
            symmetric_fraction < qualifying_fraction
            or coincident_length < min_coincident_length_m
        ):
            continue
        alternative_count += 1
        qualifying_buffers.append(geometry.buffer(tolerance_m))
        if alternative_id is not None:
            alternative_ids.append(alternative_id)
        if segment_role is not None and role is not None and role != segment_role:
            has_role_conflict = True

    if not qualifying_buffers:
        covered_length = 0.0
        covered_fraction = 0.0
    else:
        covered_length = segment.intersection(shapely.union_all(qualifying_buffers)).length
        covered_fraction = min(max(covered_length / segment.length, 0.0), 1.0)
    return CoincidentAlternativeResult(
        covered_fraction=covered_fraction,
        covered_length_m=covered_length,
        max_symmetric_fraction=max_symmetric_fraction,
        alternative_count=alternative_count,
        has_role_conflict=has_role_conflict,
        alternative_ids=tuple(alternative_ids),
    )


def compute_same_side_coincidence_context(
    geometries: dict[str, Any],
    *,
    roles: dict[str, Any] | None = None,
    labels: dict[str, str] | None = None,
    source_crs: str = "EPSG:4326",
    tolerance_m: float = 3.0,
    qualifying_fraction: float = 0.8,
    min_coincident_length_m: float = 20.0,
) -> dict[str, CoincidentAlternativeResult]:
    """Measure coincident alternatives for every segment on one provider side.

    Stitch sidecars store WGS84 GeoJSON, while the primitive above deliberately
    operates in metric coordinates. This adapter projects the small group into
    its local UTM zone and returns only segments with a qualifying alternative.
    Labels are carried into ``alternative_ids`` so evidence packs can name the
    exact R#/T# alternatives instead of exposing a context-free scalar.
    """
    roles = roles or {}
    labels = labels or {}
    valid = {
        str(segment_id): geometry
        for segment_id, geometry in geometries.items()
        if geometry is not None and not geometry.is_empty and geometry.length > 0
    }
    if len(valid) < 2:
        return {}

    union = shapely.union_all(list(valid.values()))
    centroid = union.centroid
    to_wgs84 = _cached_transformer(source_crs, "EPSG:4326")
    centroid_lon, centroid_lat = to_wgs84.transform(centroid.x, centroid.y)
    zone = min(60, max(1, int((centroid_lon + 180) / 6) + 1))
    epsg = (32600 if centroid_lat >= 0 else 32700) + zone
    transformer = _cached_transformer(source_crs, f"EPSG:{epsg}")
    projected = {
        segment_id: transform(transformer.transform, geometry)
        for segment_id, geometry in valid.items()
    }

    results: dict[str, CoincidentAlternativeResult] = {}
    for segment_id, geometry in projected.items():
        alternatives = [
            (
                other_geometry,
                roles.get(other_id),
                labels.get(other_id, other_id),
            )
            for other_id, other_geometry in projected.items()
            if other_id != segment_id
        ]
        result = compute_coincident_alternatives(
            geometry,
            alternatives,
            segment_role=roles.get(segment_id),
            tolerance_m=tolerance_m,
            qualifying_fraction=qualifying_fraction,
            min_coincident_length_m=min_coincident_length_m,
        )
        if result.alternative_count:
            results[segment_id] = result
    return results
