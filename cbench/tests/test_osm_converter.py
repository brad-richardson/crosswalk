"""Tests for OSM converter topology via connector-based node sharing."""

import geopandas as gpd
from shapely.geometry import LineString, Point

from cbench.convert.osm import OSMConverter


def _make_segments_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    """Build a segments GeoDataFrame from a list of dicts with geometry and connectors."""
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _make_connectors_gdf(connectors: list[dict]) -> gpd.GeoDataFrame:
    """Build a connectors GeoDataFrame from id/geometry pairs."""
    return gpd.GeoDataFrame(connectors, crs="EPSG:4326")


class TestSharedConnectorNodes:
    """Two segments sharing a connector at their endpoints produce shared OSM node IDs."""

    def test_shared_connector_nodes(self):
        shared_pt = Point(-71.06, 42.36)
        connectors_gdf = _make_connectors_gdf([{"id": "conn-shared", "geometry": shared_pt}])

        seg1 = {
            "id": "seg-1",
            "geometry": LineString([(-71.07, 42.36), (-71.06, 42.36)]),
            "connectors": [
                {"connector_id": "conn-shared", "at": 1.0},
            ],
        }
        seg2 = {
            "id": "seg-2",
            "geometry": LineString([(-71.06, 42.36), (-71.05, 42.36)]),
            "connectors": [
                {"connector_id": "conn-shared", "at": 0.0},
            ],
        }
        gdf = _make_segments_gdf([seg1, seg2])

        converter = OSMConverter(connectors_gdf=connectors_gdf)
        osm = converter.convert(gdf)

        ways = osm.findall("way")
        assert len(ways) == 2

        way1_refs = [nd.get("ref") for nd in ways[0].findall("nd")]
        way2_refs = [nd.get("ref") for nd in ways[1].findall("nd")]

        # Last node of seg1 should equal first node of seg2 (shared connector)
        assert way1_refs[-1] == way2_refs[0]

        # Non-connector endpoints should be unique
        assert way1_refs[0] != way2_refs[-1]
        assert way1_refs[0] != way1_refs[-1]


class TestNoConnectorsUniqueNodes:
    """Without connectors, every vertex gets a unique node (no fabricated topology)."""

    def test_no_connectors_unique_nodes(self):
        # Two segments sharing an endpoint coordinate but no connectors
        seg1 = {
            "id": "seg-1",
            "geometry": LineString([(-71.07, 42.36), (-71.06, 42.36)]),
        }
        seg2 = {
            "id": "seg-2",
            "geometry": LineString([(-71.06, 42.36), (-71.05, 42.36)]),
        }
        gdf = _make_segments_gdf([seg1, seg2])

        converter = OSMConverter()  # No connectors
        osm = converter.convert(gdf)

        ways = osm.findall("way")
        assert len(ways) == 2

        way1_refs = [nd.get("ref") for nd in ways[0].findall("nd")]
        way2_refs = [nd.get("ref") for nd in ways[1].findall("nd")]

        # All node IDs should be unique — no fabricated topology
        all_refs = way1_refs + way2_refs
        assert len(set(all_refs)) == len(all_refs)


class TestIntermediateConnectorMatching:
    """A connector at a mid-segment position with non-uniform vertex spacing."""

    def test_intermediate_connector_matching(self):
        # Non-uniform vertex spacing: vertices at 0.0, 0.1, 0.5, 1.0 (lon)
        # Connector sits at lon=-71.05 which is at vertex index 2
        # With index-fraction matching, vertex 2 would be at fraction 2/3=0.667
        # but the connector's "at" linear ref might differ. With geometry proximity
        # matching, it finds the nearest vertex regardless.
        conn_pt = Point(-71.05, 42.36)
        connectors_gdf = _make_connectors_gdf([{"id": "conn-mid", "geometry": conn_pt}])

        seg = {
            "id": "seg-1",
            "geometry": LineString(
                [
                    (-71.10, 42.36),  # vertex 0
                    (-71.09, 42.36),  # vertex 1 (close to start)
                    (-71.05, 42.36),  # vertex 2 (big gap — connector is here)
                    (-71.00, 42.36),  # vertex 3
                ]
            ),
            "connectors": [
                {"connector_id": "conn-mid", "at": 0.5},  # "at" value doesn't matter
            ],
        }
        gdf = _make_segments_gdf([seg])

        converter = OSMConverter(connectors_gdf=connectors_gdf)
        osm = converter.convert(gdf)

        ways = osm.findall("way")
        assert len(ways) == 1

        nd_refs = [nd.get("ref") for nd in ways[0].findall("nd")]
        assert len(nd_refs) == 4

        # Vertex 2 should use the connector node ID
        connector_node_id = str(converter.connector_node_map["conn-mid"])
        assert nd_refs[2] == connector_node_id

        # Other vertices should NOT use connector node ID
        assert nd_refs[0] != connector_node_id
        assert nd_refs[1] != connector_node_id
        assert nd_refs[3] != connector_node_id


class TestConnectorWithoutMatchingVertex:
    """A connector whose Point is far from all vertices is skipped."""

    def test_connector_without_matching_vertex(self):
        # Connector is far from any vertex
        far_pt = Point(-70.00, 41.00)
        connectors_gdf = _make_connectors_gdf([{"id": "conn-far", "geometry": far_pt}])

        seg = {
            "id": "seg-1",
            "geometry": LineString([(-71.07, 42.36), (-71.06, 42.36)]),
            "connectors": [
                {"connector_id": "conn-far", "at": 0.5},
            ],
        }
        gdf = _make_segments_gdf([seg])

        converter = OSMConverter(connectors_gdf=connectors_gdf)
        osm = converter.convert(gdf)

        ways = osm.findall("way")
        assert len(ways) == 1

        nd_refs = [nd.get("ref") for nd in ways[0].findall("nd")]

        # No connector node should be created — all nodes are unique
        assert "conn-far" not in converter.connector_node_map
        assert len(set(nd_refs)) == len(nd_refs)


class TestConnectorNodeUsesConnectorGeometry:
    """The shared node's lat/lon comes from the connector Point, not the segment vertex."""

    def test_connector_node_uses_connector_geometry(self):
        # Connector point is very slightly offset from the segment vertex
        # (within tolerance of 1e-6 degrees, but not identical)
        conn_pt = Point(-71.0600005, 42.3600005)
        connectors_gdf = _make_connectors_gdf([{"id": "conn-precise", "geometry": conn_pt}])

        seg = {
            "id": "seg-1",
            "geometry": LineString([(-71.07, 42.36), (-71.06, 42.36)]),
            "connectors": [
                {"connector_id": "conn-precise", "at": 1.0},
            ],
        }
        gdf = _make_segments_gdf([seg])

        converter = OSMConverter(connectors_gdf=connectors_gdf)
        osm = converter.convert(gdf)

        # Find the connector node in the OSM output
        connector_node_id = str(converter.connector_node_map["conn-precise"])
        nodes = osm.findall("node")

        connector_node = None
        for node in nodes:
            if node.get("id") == connector_node_id:
                connector_node = node
                break

        assert connector_node is not None

        # Node coordinates should come from the connector point, not the vertex
        assert float(connector_node.get("lon")) == conn_pt.x
        assert float(connector_node.get("lat")) == conn_pt.y
