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
from typing import Any

import shapely


@dataclass(frozen=True)
class CoincidentAlternativeResult:
    """Coverage and role ambiguity induced by same-side alternatives."""

    covered_fraction: float
    covered_length_m: float
    max_symmetric_fraction: float
    alternative_count: int
    has_role_conflict: bool


def compute_coincident_alternatives(
    segment,
    alternatives: Iterable[tuple[Any, Any]],
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
    for geometry, role in alternatives:
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
    )
