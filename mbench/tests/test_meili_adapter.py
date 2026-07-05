"""Tests for the Meili adapter's pure logic (no Docker/Valhalla required).

Covers the edge-aggregation / overlap filter and trace densification — the parts
that decide match quality — without standing up a routing service.
"""

import geopandas as gpd
import pytest
from shapely.geometry import LineString

pytest.importorskip("shapely")

from mbench.adapters.meili import (  # noqa: E402
    _aggregate_edges,
    _densify_lonlat,
    _trace_request_payload,
)


def _resp(edges):
    """Build a minimal trace_attributes response with (way_id, length_km) edges."""
    return {"edges": [{"way_id": w, "length": km} for w, km in edges]}


ID_MAP = {"1": "gers-a", "2": "gers-b", "3": "gers-c"}


class TestAggregateEdges:
    def test_sums_length_per_gers_id(self):
        # way 1 appears twice (Valhalla splits a way at nodes) -> summed length.
        resp = _resp([(1, 0.030), (1, 0.020), (2, 0.040)])
        out = dict(_aggregate_edges(resp, ID_MAP, target_len_m=100.0, min_frac=0.1, min_m=8.0))
        assert set(out) == {"gers-a", "gers-b"}
        # gers-a matched 50m of a 100m target -> confidence 0.5.
        assert out["gers-a"] == pytest.approx(0.5)
        assert out["gers-b"] == pytest.approx(0.4)

    def test_drops_spurious_short_edge(self):
        # A 2m clip of a crossing street: below both min_frac (0.1*100=10m) and min_m (8m).
        resp = _resp([(1, 0.050), (3, 0.002)])
        out = dict(_aggregate_edges(resp, ID_MAP, target_len_m=100.0, min_frac=0.1, min_m=8.0))
        assert "gers-c" not in out
        assert "gers-a" in out

    def test_keeps_short_sliver_above_absolute_floor(self):
        # 9m matched on a 500m target: below min_frac (0.1*500=50m) but above min_m (8m).
        resp = _resp([(3, 0.009)])
        out = dict(_aggregate_edges(resp, ID_MAP, target_len_m=500.0, min_frac=0.1, min_m=8.0))
        assert "gers-c" in out

    def test_confidence_capped_at_one(self):
        # Matched length exceeds target length (overshoot at ends) -> capped.
        resp = _resp([(1, 0.150)])
        out = dict(_aggregate_edges(resp, ID_MAP, target_len_m=100.0, min_frac=0.1, min_m=8.0))
        assert out["gers-a"] == 1.0

    def test_unknown_way_id_ignored(self):
        resp = _resp([(999, 0.050), (1, 0.050)])
        out = dict(_aggregate_edges(resp, ID_MAP, target_len_m=100.0, min_frac=0.1, min_m=8.0))
        assert set(out) == {"gers-a"}

    def test_empty_response(self):
        assert _aggregate_edges(None, ID_MAP, 100.0, 0.1, 8.0) == []
        assert _aggregate_edges({}, ID_MAP, 100.0, 0.1, 8.0) == []


class TestDensify:
    def test_densifies_long_segment(self):
        # ~1 km segment (0.01 deg lon at 42N ~ 825m) densified to 10m -> many points.
        line = LineString([(-71.06, 42.36), (-71.05, 42.36)])
        crs = gpd.GeoSeries([line], crs="EPSG:4326").estimate_utm_crs()
        pts = _densify_lonlat(line, crs, densify_m=10.0)
        assert len(pts) > 50
        assert all(len(p) == 2 for p in pts)

    def test_empty_geometry_returns_empty(self):
        crs = "EPSG:32619"
        assert _densify_lonlat(None, crs, 10.0) == []


class TestTracePayload:
    def test_units_pinned_to_kilometers(self):
        # _aggregate_edges converts edge.length km -> m; the request must pin
        # units=kilometers so a Valhalla default change can't silently corrupt it.
        payload = _trace_request_payload(
            [(-71.06, 42.36), (-71.05, 42.36)],
            costing="pedestrian",
            search_radius=25.0,
            gps_acc=10.0,
        )
        assert payload["units"] == "kilometers"
