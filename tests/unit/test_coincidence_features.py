from __future__ import annotations

import pytest
from shapely import LineString

from crosswalk.features.coincidence import compute_coincident_alternatives


def test_short_role_conflicting_line_over_long_line_is_ambiguous() -> None:
    tunnel = LineString([(0, 0), (100, 0)])
    surface = LineString([(20, 1), (80, 1)])

    result = compute_coincident_alternatives(
        tunnel,
        [(surface, "surface-cycle-corridor")],
        segment_role="covered-trench",
    )

    assert result.alternative_count == 1
    assert result.max_symmetric_fraction == pytest.approx(1.0)
    assert result.covered_fraction > 0.6
    assert result.covered_length_m > 60.0
    assert result.has_role_conflict is True


def test_perpendicular_crossing_does_not_create_coincident_alternative() -> None:
    horizontal = LineString([(0, 0), (100, 0)])
    vertical = LineString([(50, -50), (50, 50)])

    result = compute_coincident_alternatives(
        horizontal,
        [(vertical, "other")],
        segment_role="road",
    )

    assert result.alternative_count == 0
    assert result.covered_fraction == 0.0


def test_union_coverage_combines_sequential_coincident_alternatives() -> None:
    long_line = LineString([(0, 0), (100, 0)])
    alternatives = [
        (LineString([(0, 1), (50, 1)]), "surface"),
        (LineString([(50, 1), (100, 1)]), "surface"),
    ]

    result = compute_coincident_alternatives(
        long_line,
        alternatives,
        segment_role="tunnel",
    )

    assert result.alternative_count == 2
    assert result.covered_fraction == pytest.approx(1.0)
    assert result.has_role_conflict is True


def test_short_endpoint_stub_does_not_create_layer_ambiguity() -> None:
    corridor = LineString([(0, 0), (100, 0)])
    stub = LineString([(0, 0), (5, 0)])

    result = compute_coincident_alternatives(
        corridor,
        [(stub, "stub")],
        segment_role="corridor",
    )

    assert result.max_symmetric_fraction == pytest.approx(1.0)
    assert result.alternative_count == 0
    assert result.covered_length_m == 0.0
