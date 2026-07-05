"""Tests for the Overture -> OSM PBF routable-graph converter (convert/pbf.py)."""

import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString

pytest.importorskip("osmium")
import osmium  # noqa: E402

from cbench.convert.pbf import convert_overture_to_pbf  # noqa: E402


def _gdf(rows):
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


class _Reader(osmium.SimpleHandler):
    """Collect nodes and ways from a written PBF for assertions."""

    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.ways = []

    def node(self, n):
        self.nodes[n.id] = (n.location.lon, n.location.lat)

    def way(self, w):
        self.ways.append((w.id, [nd.ref for nd in w.nodes], {t.k: t.v for t in w.tags}))


def _read(pbf_path):
    r = _Reader()
    r.apply_file(str(pbf_path))
    return r


def test_shared_coordinate_collapses_to_one_node(tmp_path):
    """Two segments meeting at the same coordinate share exactly one OSM node."""
    gdf = _gdf(
        [
            {
                "id": "gers-a",
                "class": "residential",
                "geometry": LineString([(-71.07, 42.36), (-71.06, 42.36)]),
            },
            {
                "id": "gers-b",
                "class": "primary",
                "geometry": LineString([(-71.06, 42.36), (-71.05, 42.36)]),
            },
        ]
    )
    ref = tmp_path / "ref.parquet"
    gdf.to_parquet(ref)
    pbf = tmp_path / "g.osm.pbf"
    idmap = tmp_path / "id.json"

    meta = convert_overture_to_pbf(ref, pbf, idmap)
    # 3 distinct coordinates -> 3 nodes (the shared -71.06,42.36 counts once).
    assert meta["n_nodes"] == 3
    assert meta["n_ways"] == 2

    r = _read(pbf)
    assert len(r.nodes) == 3
    # The two ways must reference a common node id at the join.
    w0_nodes = set(r.ways[0][1])
    w1_nodes = set(r.ways[1][1])
    assert len(w0_nodes & w1_nodes) == 1


def test_way_id_maps_to_gers_and_highway_tag(tmp_path):
    gdf = _gdf(
        [
            {
                "id": "gers-a",
                "class": "footway",
                "geometry": LineString([(-71.07, 42.36), (-71.06, 42.36)]),
            }
        ]
    )
    ref = tmp_path / "ref.parquet"
    gdf.to_parquet(ref)
    pbf = tmp_path / "g.osm.pbf"
    idmap = tmp_path / "id.json"
    convert_overture_to_pbf(ref, pbf, idmap)

    id_map = json.loads(idmap.read_text())
    r = _read(pbf)
    way_id, _, tags = r.ways[0]
    # sidewalk/footway class maps to highway=footway; way_id resolves to the GERS id.
    assert tags["highway"] == "footway"
    assert id_map[str(way_id)] == "gers-a"


def test_missing_class_defaults_to_routable_highway(tmp_path):
    gdf = _gdf(
        [
            {
                "id": "gers-a",
                "class": None,
                "geometry": LineString([(-71.07, 42.36), (-71.06, 42.36)]),
            }
        ]
    )
    ref = tmp_path / "ref.parquet"
    gdf.to_parquet(ref)
    pbf = tmp_path / "g.osm.pbf"
    idmap = tmp_path / "id.json"
    convert_overture_to_pbf(ref, pbf, idmap)
    r = _read(pbf)
    assert r.ways[0][2]["highway"] == "residential"


def test_multilinestring_yields_multiple_ways_same_gers(tmp_path):
    gdf = _gdf(
        [
            {
                "id": "gers-multi",
                "class": "residential",
                "geometry": MultiLineString(
                    [
                        [(-71.07, 42.36), (-71.06, 42.36)],
                        [(-71.05, 42.36), (-71.04, 42.36)],
                    ]
                ),
            }
        ]
    )
    ref = tmp_path / "ref.parquet"
    gdf.to_parquet(ref)
    pbf = tmp_path / "g.osm.pbf"
    idmap = tmp_path / "id.json"
    meta = convert_overture_to_pbf(ref, pbf, idmap)
    assert meta["n_ways"] == 2
    id_map = json.loads(idmap.read_text())
    # Both ways map to the same GERS id.
    assert set(id_map.values()) == {"gers-multi"}
    assert len(id_map) == 2
