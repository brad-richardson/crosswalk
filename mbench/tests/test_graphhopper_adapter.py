"""Tests for the GraphHopper adapter's pure logic (no JVM/jbang required).

Covers trace building, the bridged-edge filter, the density-based matched-length
estimate, and the id_map -> GERS aggregation — the parts that decide match quality
— without standing up the JVM. An optional end-to-end test runs the real jbang
runner on a tiny synthetic graph and skips cleanly when jbang is absent (so CI
without Java stays green).
"""

import shutil

import geopandas as gpd
import pytest
from shapely.geometry import LineString

pytest.importorskip("shapely")
pytest.importorskip("osmium")

from mbench.adapters.graphhopper import (  # noqa: E402
    GraphHopperAdapter,
    _require_jbang,
    build_traces_tsv,
    parse_matches_tsv,
)

ID_MAP = {"1": "gers-a", "2": "gers-b", "3": "gers-c"}


def _write(tmp_path, lines):
    p = tmp_path / "gh_matches.tsv"
    p.write_text("\n".join(lines) + "\n")
    return p


class TestParseMatches:
    def test_maps_way_id_to_gers_and_filters(self, tmp_path):
        # matched span = n_states * densify, capped at full edge length. way 1: 6 pts
        # @100m -> 600m capped at edge 50m -> 50m -> conf 0.5; way 2: 5 pts -> capped
        # at 40m -> 0.4.
        p = _write(tmp_path, ["t1\t1,50.0,6;2,40.0,5"])
        df = parse_matches_tsv(p, ID_MAP, {"t1": 100.0}, 0.1, 8.0, densify_m=100.0)
        got = dict(zip(df["ref_id"], df["confidence"]))
        assert set(got) == {"gers-a", "gers-b"}
        assert got["gers-a"] == pytest.approx(0.5)
        assert got["gers-b"] == pytest.approx(0.4)

    def test_drops_bridged_edge_zero_states(self, tmp_path):
        # way 3 was routed through but no trace point snapped onto it (n_states=0):
        # a GraphHopper "bridged" parallel/connecting edge. The density estimate
        # gives it 0 m matched (0 states * spacing) -> filtered out.
        p = _write(tmp_path, ["t1\t1,50.0,6;3,50.0,0"])
        df = parse_matches_tsv(p, ID_MAP, {"t1": 100.0}, 0.1, 8.0, densify_m=100.0)
        assert set(df["ref_id"]) == {"gers-a"}

    def test_density_caps_matched_length(self, tmp_path):
        # A 200m parallel edge clipped by a single 5m-spaced observation is credited
        # ~5m (1 * densify), NOT its full 200m — below both min_frac (50m on a 500m
        # target) and min_m (8m) -> dropped.
        p = _write(tmp_path, ["t1\t3,200.0,1"])
        df = parse_matches_tsv(p, ID_MAP, {"t1": 500.0}, 0.1, 8.0, densify_m=5.0)
        assert len(df) == 0

    def test_full_length_when_many_observations(self, tmp_path):
        # Same edge but 10 observations -> ~100m matched (10*10, > min_m 8m) -> kept.
        p = _write(tmp_path, ["t1\t3,200.0,10"])
        df = parse_matches_tsv(p, ID_MAP, {"t1": 500.0}, 0.1, 8.0, densify_m=10.0)
        assert set(df["ref_id"]) == {"gers-c"}

    def test_unknown_way_id_ignored(self, tmp_path):
        p = _write(tmp_path, ["t1\t999,50.0,6;1,50.0,6"])
        df = parse_matches_tsv(p, ID_MAP, {"t1": 100.0}, 0.1, 8.0, densify_m=100.0)
        assert set(df["ref_id"]) == {"gers-a"}

    def test_missing_output_file(self, tmp_path):
        df = parse_matches_tsv(tmp_path / "nope.tsv", ID_MAP, {}, 0.1, 8.0, densify_m=10.0)
        assert len(df) == 0


class TestBuildTraces:
    def test_writes_densified_trace_and_lengths(self, tmp_path):
        tgt = gpd.GeoDataFrame(
            [{"id": "t1", "geometry": LineString([(-71.06, 42.36), (-71.05, 42.36)])}],
            crs="EPSG:4326",
        )
        out = tmp_path / "traces.tsv"
        ids, lens = build_traces_tsv(tgt, out, densify_m=10.0)
        assert ids == ["t1"]
        assert lens["t1"] > 700  # ~825m at 42N
        line = out.read_text().strip()
        tid, coords = line.split("\t")
        assert tid == "t1"
        # densified to 10m over ~825m -> many points
        assert len(coords.split(";")) > 50


class TestJbangGuard:
    def test_require_jbang_raises_clear_message(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _: None)
        with pytest.raises(RuntimeError, match="jbang"):
            _require_jbang()


# --- optional end-to-end (needs jbang + a JVM); skips cleanly in CI without Java ---

jbang_missing = shutil.which("jbang") is None


@pytest.mark.skipif(jbang_missing, reason="jbang not installed; JVM map-matching unavailable")
def test_end_to_end_tiny_graph(tmp_path):
    # A connected chain of residential ways; a target running along it should match
    # the collinear ways (not a perpendicular branch).
    rows = [
        {
            "id": "gers-h0",
            "class": "residential",
            "geometry": LineString([(-71.070, 42.360), (-71.065, 42.360)]),
        },
        {
            "id": "gers-h1",
            "class": "residential",
            "geometry": LineString([(-71.065, 42.360), (-71.060, 42.360)]),
        },
        {
            "id": "gers-branch",
            "class": "footway",
            "geometry": LineString([(-71.065, 42.360), (-71.065, 42.365)]),
        },
    ]
    ref = tmp_path / "ref.parquet"
    gpd.GeoDataFrame(rows, crs="EPSG:4326").to_parquet(ref)
    tgt = tmp_path / "tgt.parquet"
    gpd.GeoDataFrame(
        [{"id": "t1", "geometry": LineString([(-71.069, 42.36003), (-71.061, 42.36003)])}],
        crs="EPSG:4326",
    ).to_parquet(tgt)

    out_dir = tmp_path / "out"
    adapter = GraphHopperAdapter()
    out_path = adapter.run(ref, tgt, out_dir, graph_cache_dir=str(tmp_path / "cache"))
    result = adapter.parse_output(out_path)
    matched = set(result.matches["ref_id"])
    # Perfect-recall formulation: the two collinear ways are recovered.
    assert {"gers-h0", "gers-h1"}.issubset(matched)
    # The perpendicular footway branch is not on the trace.
    assert "gers-branch" not in matched
