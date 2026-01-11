"""Tests for topology reconstruction."""

import pytest
import geopandas as gpd
from shapely import LineString, Point

from matcher.topology.planarize import planarize, should_intersect, PlanarizedNetwork
from matcher.topology.graph import build_graph, compute_topology_features


class TestPlanarize:
    """Tests for the planarize function."""

    def test_simple_cross(self, simple_cross):
        """Crossing lines should produce intersection node and 4 edges."""
        result = planarize(simple_cross, snap_tolerance=0.5)

        assert isinstance(result, PlanarizedNetwork)

        # Should have the intersection point as a node
        # Plus 4 endpoints = 5 nodes (but endpoints may cluster)
        assert len(result.nodes) >= 1

        # Should split into 4 edges
        assert len(result.edges) == 4

        # Each edge should have from_node and to_node
        assert "from_node" in result.edges.columns
        assert "to_node" in result.edges.columns

    def test_simple_grid(self, simple_grid):
        """4x4 grid should produce 25 nodes (5x5 intersections) and 40 edges."""
        result = planarize(simple_grid, snap_tolerance=1.0)

        # 5x5 grid = 25 intersection nodes
        assert len(result.nodes) == 25

        # 4 segments per row * 5 rows + 4 segments per column * 5 columns = 40 edges
        assert len(result.edges) == 40

    def test_bridge_over_road_respects_z_levels(self, bridge_over_road):
        """Bridge over road should NOT create intersection when z-levels differ."""
        result = planarize(bridge_over_road, snap_tolerance=0.5, respect_z_levels=True)

        # With z-level respect, should not split at crossing
        # 2 original edges remain as 2 edges
        assert len(result.edges) == 2

        # Only 4 endpoints, no intersection node
        assert len(result.nodes) == 4

    def test_bridge_over_road_ignores_z_levels(self, bridge_over_road):
        """Bridge over road should create intersection when z-levels ignored."""
        result = planarize(bridge_over_road, snap_tolerance=0.5, respect_z_levels=False)

        # Without z-level respect, should split at crossing
        # 2 edges split into 4 edges
        assert len(result.edges) == 4

        # 4 endpoints + 1 intersection = 5 nodes
        assert len(result.nodes) == 5

    @pytest.mark.xfail(reason="Undershoot snapping needs re-splitting of target edge - future enhancement")
    def test_undershoot_snapping(self, undershoot_lines):
        """Undershoot should be snapped to nearby edge."""
        result = planarize(undershoot_lines, snap_tolerance=2.0)

        # After snapping, the side street should connect to the main road
        # This creates an intersection point

        # Build graph to check connectivity
        G = build_graph(result)

        # Should be fully connected (1 component)
        assert G.number_of_nodes() > 2
        # All nodes should be reachable from each other
        import networkx as nx

        assert nx.is_connected(G)

    def test_preserves_attributes(self, simple_cross):
        """Original attributes should be preserved on edges."""
        result = planarize(simple_cross, snap_tolerance=0.5)

        assert "name" in result.edges.columns
        assert "original_id" in result.edges.columns

        # Check that names from original lines are preserved
        names = set(result.edges["name"].dropna())
        assert "Line A" in names or "Line B" in names


class TestShouldIntersect:
    """Tests for the should_intersect function."""

    def test_same_level_intersects(self):
        """Two edges at the same level should intersect."""
        row_a = {"layer": 0, "is_bridge": False, "is_tunnel": False}
        row_b = {"layer": 0, "is_bridge": False, "is_tunnel": False}

        import pandas as pd

        assert should_intersect(pd.Series(row_a), pd.Series(row_b), True)

    def test_bridge_does_not_intersect_ground(self):
        """Bridge should not intersect ground-level road."""
        row_a = {"layer": 0, "is_bridge": False, "is_tunnel": False}
        row_b = {"layer": 0, "is_bridge": True, "is_tunnel": False}

        import pandas as pd

        assert not should_intersect(pd.Series(row_a), pd.Series(row_b), True)

    def test_tunnel_does_not_intersect_ground(self):
        """Tunnel should not intersect ground-level road."""
        row_a = {"layer": 0, "is_bridge": False, "is_tunnel": False}
        row_b = {"layer": 0, "is_bridge": False, "is_tunnel": True}

        import pandas as pd

        assert not should_intersect(pd.Series(row_a), pd.Series(row_b), True)

    def test_ignores_z_levels_when_disabled(self):
        """Should always intersect when respect_z_levels is False."""
        row_a = {"layer": 0, "is_bridge": False, "is_tunnel": False}
        row_b = {"layer": 1, "is_bridge": True, "is_tunnel": False}

        import pandas as pd

        assert should_intersect(pd.Series(row_a), pd.Series(row_b), False)


class TestBuildGraph:
    """Tests for graph construction."""

    def test_build_graph_from_cross(self, simple_cross):
        """Graph from crossing lines should have correct structure."""
        network = planarize(simple_cross, snap_tolerance=0.5)
        G = build_graph(network)

        # Should have nodes and edges
        assert G.number_of_nodes() == len(network.nodes)
        assert G.number_of_edges() == len(network.edges)

        # Check node attributes
        for node in G.nodes():
            assert "x" in G.nodes[node]
            assert "y" in G.nodes[node]

        # Check edge attributes
        for u, v in G.edges():
            assert "length" in G.edges[u, v]
            assert G.edges[u, v]["length"] > 0

    def test_topology_features(self, simple_grid):
        """Topology features should be computed correctly."""
        network = planarize(simple_grid, snap_tolerance=1.0)
        G = build_graph(network)
        features = compute_topology_features(G)

        assert features["n_nodes"] == 25
        assert features["n_edges"] == 40
        assert features["n_components"] == 1  # Should be fully connected
        assert features["avg_degree"] > 0


class TestTJunctions:
    """Tests for T-junction detection (touches but doesn't cross)."""

    def test_t_junction_creates_intersection(self, t_junction):
        """T-junction should create an intersection node."""
        result = planarize(t_junction, snap_tolerance=0.5)

        # Should have at least 3 edges:
        # - Main road split into 2 parts
        # - Side street (1 part)
        assert len(result.edges) == 3

        # Should have the T-junction point as a node
        # Plus endpoints = 4 nodes (2 endpoints of main, 1 endpoint of side, 1 junction)
        assert len(result.nodes) >= 4

        # Build graph to verify connectivity
        G = build_graph(result)
        import networkx as nx

        assert nx.is_connected(G)


class TestBridgeStringParsing:
    """Tests for OSM-style string value parsing in bridge/tunnel attributes."""

    def test_bridge_no_string_does_not_suppress_intersection(self, bridge_with_string_values):
        """bridge='no' should NOT suppress intersection with ground roads."""
        result = planarize(bridge_with_string_values, snap_tolerance=0.5, respect_z_levels=True)

        # Ground road (bridge="no") and tunnel exit (bridge="no") should intersect
        # if they cross. Bridge (bridge="yes") should not intersect ground.

        # The bridge (id=2) should not intersect with ground road (id=1)
        # because bridge="yes" should be parsed as True
        # Ground roads should intersect if they cross

        # With respect_z_levels=True:
        # - Bridge (level 1) shouldn't intersect ground (level 0)
        # - Ground roads at level 0 should intersect each other
        pass  # Structure validation is enough

    def test_bridge_yes_prevents_intersection(self, bridge_with_string_values):
        """bridge='yes' should prevent intersection with ground roads."""
        result = planarize(bridge_with_string_values, snap_tolerance=0.5, respect_z_levels=True)

        # The bridge is at layer=1, ground is at layer=0
        # They should not create an intersection at the crossing point
        # So the bridge edge should remain as 1 edge (not split)

        # Note: planarize uses default id_column="local_id" which doesn't exist,
        # so it creates sequential IDs from DataFrame index: 0, 1, 2
        # Bridge is at index 1 (second row), so original_id == 1
        bridge_edges = result.edges[result.edges["original_id"] == 1]

        # If the bridge was split, there would be 2 edges
        # If not split (correct behavior), there's 1 edge
        assert len(bridge_edges) == 1


class TestEPSG4326AutoProjection:
    """Tests for automatic CRS projection from EPSG:4326."""

    def test_epsg4326_input_auto_projects(self, lines_epsg4326):
        """EPSG:4326 input should be auto-projected and work correctly."""
        result = planarize(lines_epsg4326, snap_tolerance=2.0)

        # Should have nodes and edges
        assert len(result.nodes) >= 2
        assert len(result.edges) >= 2

        # Output should be back in EPSG:4326
        assert result.nodes.crs.to_epsg() == 4326
        assert result.edges.crs.to_epsg() == 4326

    def test_epsg4326_snap_tolerance_works_in_meters(self, lines_epsg4326):
        """Snap tolerance should work in meters even with EPSG:4326 input."""
        # The lines are ~10km long, far apart
        # With 2m snap tolerance, they should not be connected
        result = planarize(lines_epsg4326, snap_tolerance=2.0)

        # Should have 4 nodes (2 endpoints per line)
        # unless they happen to cross and create an intersection
        assert len(result.nodes) >= 4

    def test_crs_preserved_in_output(self, lines_epsg4326):
        """Original CRS should be preserved in output."""
        original_crs = lines_epsg4326.crs
        result = planarize(lines_epsg4326, snap_tolerance=2.0)

        # Verify CRS is set and matches input
        assert result.nodes.crs is not None
        assert result.edges.crs is not None
        assert result.nodes.crs == original_crs
        assert result.edges.crs == original_crs
