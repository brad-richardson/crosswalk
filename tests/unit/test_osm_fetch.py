"""Tests for OSM PBF fetching modules."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from matcher.fetch.osm import (
    _build_level_rules,
    _build_road_flags,
    _filter_connectors_for_roads,
    _filter_fully_inside,
    _get_level_from_rules,
    _has_flag,
    _node_ids_to_connectors,
    _normalize_road_class,
    _transform_connectors_schema,
    _transform_to_overture_schema,
)
from matcher.fetch.osm_download import (
    find_best_region,
    get_pbf_url,
)
from matcher.fetch.osm_pbf import HIGHWAY_VALUES, RoadHandler
from matcher.fetch.overture import BoundingBox


class TestBuildRoadFlags:
    """Tests for _build_road_flags function."""

    def test_no_flags_returns_empty(self):
        result = _build_road_flags({}, "residential")
        assert result == []

    def test_bridge_yes_adds_flag(self):
        result = _build_road_flags({"bridge": "yes"}, "residential")
        assert len(result) == 1
        assert "is_bridge" in result[0]["values"]

    def test_bridge_viaduct_adds_flag(self):
        """Viaduct is a valid bridge value."""
        result = _build_road_flags({"bridge": "viaduct"}, "residential")
        assert len(result) == 1
        assert "is_bridge" in result[0]["values"]

    def test_bridge_invalid_value_no_flag(self):
        """Invalid bridge values should not set the flag."""
        result = _build_road_flags({"bridge": "maybe"}, "residential")
        assert result == []

    def test_bridge_no_does_not_add_flag(self):
        result = _build_road_flags({"bridge": "no"}, "residential")
        assert result == []

    def test_tunnel_yes_adds_flag(self):
        result = _build_road_flags({"tunnel": "yes"}, "residential")
        assert len(result) == 1
        assert "is_tunnel" in result[0]["values"]

    def test_tunnel_building_passage_adds_flag(self):
        result = _build_road_flags({"tunnel": "building_passage"}, "residential")
        assert len(result) == 1
        assert "is_tunnel" in result[0]["values"]

    def test_tunnel_no_does_not_add_flag(self):
        result = _build_road_flags({"tunnel": "no"}, "residential")
        assert result == []

    def test_covered_yes_adds_flag(self):
        result = _build_road_flags({"covered": "yes"}, "residential")
        assert len(result) == 1
        assert "is_covered" in result[0]["values"]

    def test_link_road_adds_is_link_flag(self):
        result = _build_road_flags({}, "motorway_link")
        assert len(result) == 1
        assert "is_link" in result[0]["values"]

    def test_construction_yes_adds_flag(self):
        result = _build_road_flags({"construction": "yes"}, "residential")
        assert len(result) == 1
        assert "is_under_construction" in result[0]["values"]

    def test_construction_other_value_no_flag(self):
        """Only 'yes' should set the construction flag."""
        result = _build_road_flags({"construction": "primary"}, "residential")
        assert result == []

    def test_abandoned_adds_flag(self):
        result = _build_road_flags({"abandoned": "yes"}, "residential")
        assert len(result) == 1
        assert "is_abandoned" in result[0]["values"]

    def test_multiple_flags(self):
        """Test that multiple flags can be set at once."""
        result = _build_road_flags(
            {"bridge": "yes", "construction": "yes"},
            "motorway_link",
        )
        assert len(result) == 1
        values = result[0]["values"]
        assert "is_bridge" in values
        assert "is_under_construction" in values
        assert "is_link" in values

    def test_none_tags_returns_empty(self):
        result = _build_road_flags(None, "residential")
        assert result == []


class TestBuildLevelRules:
    """Tests for _build_level_rules function."""

    def test_no_layer_returns_empty(self):
        result = _build_level_rules({})
        assert result == []

    def test_layer_zero_returns_empty(self):
        """Ground level (0) is omitted."""
        result = _build_level_rules({"layer": "0"})
        assert result == []

    def test_positive_layer(self):
        result = _build_level_rules({"layer": "1"})
        assert result == [{"value": 1}]

    def test_negative_layer(self):
        result = _build_level_rules({"layer": "-1"})
        assert result == [{"value": -1}]

    def test_invalid_layer_returns_empty(self):
        result = _build_level_rules({"layer": "invalid"})
        assert result == []

    def test_none_tags_returns_empty(self):
        result = _build_level_rules(None)
        assert result == []


class TestHasFlag:
    """Tests for _has_flag function."""

    def test_none_returns_false(self):
        assert _has_flag(None, "is_bridge") is False

    def test_empty_list_returns_false(self):
        assert _has_flag([], "is_bridge") is False

    def test_flag_present(self):
        road_flags = [{"values": ["is_bridge", "is_link"]}]
        assert _has_flag(road_flags, "is_bridge") is True
        assert _has_flag(road_flags, "is_link") is True

    def test_flag_not_present(self):
        road_flags = [{"values": ["is_bridge"]}]
        assert _has_flag(road_flags, "is_tunnel") is False

    def test_handles_numpy_array(self):
        """Test that numpy arrays in values are handled correctly."""
        import numpy as np

        road_flags = [{"values": np.array(["is_bridge", "is_link"])}]
        assert _has_flag(road_flags, "is_bridge") is True


class TestGetLevelFromRules:
    """Tests for _get_level_from_rules function."""

    def test_none_returns_zero(self):
        assert _get_level_from_rules(None) == 0

    def test_empty_returns_zero(self):
        assert _get_level_from_rules([]) == 0

    def test_extracts_value(self):
        assert _get_level_from_rules([{"value": 1}]) == 1
        assert _get_level_from_rules([{"value": -2}]) == -2


class TestNormalizeRoadClass:
    """Tests for _normalize_road_class function."""

    def test_none_returns_unclassified(self):
        assert _normalize_road_class(None) == "unclassified"

    def test_link_roads_normalized(self):
        assert _normalize_road_class("motorway_link") == "motorway"
        assert _normalize_road_class("primary_link") == "primary"

    def test_living_street_to_residential(self):
        assert _normalize_road_class("living_street") == "residential"

    def test_path_variants(self):
        assert _normalize_road_class("footway") == "path"
        assert _normalize_road_class("cycleway") == "path"
        assert _normalize_road_class("pedestrian") == "path"

    def test_unknown_returns_unclassified(self):
        assert _normalize_road_class("unknown_type") == "unclassified"

    def test_case_insensitive(self):
        assert _normalize_road_class("MOTORWAY") == "motorway"
        assert _normalize_road_class("Primary") == "primary"


class TestTransformToOvertureSchema:
    """Tests for _transform_to_overture_schema function."""

    def test_empty_geodataframe(self):
        gdf = gpd.GeoDataFrame(columns=["id", "geometry", "tags", "name"])
        result = _transform_to_overture_schema(gdf)
        assert len(result) == 0

    def test_basic_transformation(self):
        gdf = gpd.GeoDataFrame(
            {
                "id": ["w123@1"],
                "geometry": [LineString([(0, 0), (1, 1)])],
                "tags": [{"highway": "residential", "bridge": "yes"}],
                "name": ["Main Street"],
                "node_ids": [[1, 2]],
            },
            crs="EPSG:4326",
        )
        result = _transform_to_overture_schema(gdf)

        assert "class" in result.columns
        assert result.iloc[0]["class"] == "residential"

        assert "names" in result.columns
        assert result.iloc[0]["names"] == {"primary": "Main Street"}

        assert "subtype" in result.columns
        assert result.iloc[0]["subtype"] == "road"

        assert "sources" in result.columns
        sources = result.iloc[0]["sources"]
        assert sources[0]["dataset"] == "OpenStreetMap"
        assert sources[0]["record_id"] == "w123@1"

        assert "road_flags" in result.columns
        assert _has_flag(result.iloc[0]["road_flags"], "is_bridge")

        # Original columns should be removed
        assert "tags" not in result.columns
        assert "node_ids" not in result.columns

    def test_names_none_when_no_name(self):
        gdf = gpd.GeoDataFrame(
            {
                "id": ["w123@1"],
                "geometry": [LineString([(0, 0), (1, 1)])],
                "tags": [{"highway": "residential"}],
                "name": [None],
                "node_ids": [[1, 2]],
            },
            crs="EPSG:4326",
        )
        result = _transform_to_overture_schema(gdf)
        assert result.iloc[0]["names"] is None


class TestTransformConnectorsSchema:
    """Tests for _transform_connectors_schema function."""

    def test_empty_geodataframe(self):
        gdf = gpd.GeoDataFrame(columns=["id", "geometry"])
        result = _transform_connectors_schema(gdf)
        assert len(result) == 0

    def test_adds_sources(self):
        gdf = gpd.GeoDataFrame(
            {
                "id": ["n456@1"],
                "geometry": [Point(0, 0)],
            },
            crs="EPSG:4326",
        )
        result = _transform_connectors_schema(gdf)

        assert "sources" in result.columns
        sources = result.iloc[0]["sources"]
        assert sources[0]["dataset"] == "OpenStreetMap"
        # record_id uses normalized ID (without @version)
        assert sources[0]["record_id"] == "n456"


class TestFindBestRegion:
    """Tests for find_best_region function."""

    @pytest.fixture
    def mock_index(self):
        """Create a mock Geofabrik index with nested regions."""
        return {
            "features": [
                {
                    "properties": {"name": "North America", "urls": {"pbf": "http://na.pbf"}},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-180, 0], [-180, 90], [0, 90], [0, 0], [-180, 0]]],
                    },
                },
                {
                    "properties": {"name": "United States", "urls": {"pbf": "http://us.pbf"}},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-130, 24], [-130, 50], [-60, 50], [-60, 24], [-130, 24]]],
                    },
                },
                {
                    "properties": {"name": "Oregon", "urls": {"pbf": "http://or.pbf"}},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-125, 42], [-125, 46], [-116, 46], [-116, 42], [-125, 42]]
                        ],
                    },
                },
            ]
        }

    def test_finds_smallest_containing_region(self, mock_index):
        bbox = BoundingBox(xmin=-123, ymin=44, xmax=-122, ymax=45)
        result = find_best_region(bbox, mock_index)
        assert result["properties"]["name"] == "Oregon"

    def test_raises_when_no_region_contains_bbox(self, mock_index):
        bbox = BoundingBox(xmin=10, ymin=50, xmax=11, ymax=51)  # Europe
        with pytest.raises(ValueError, match="No Geofabrik region contains bbox"):
            find_best_region(bbox, mock_index)


class TestGetPbfUrl:
    """Tests for get_pbf_url function."""

    def test_extracts_url(self):
        region = {"properties": {"urls": {"pbf": "http://example.com/data.osm.pbf"}}}
        assert get_pbf_url(region) == "http://example.com/data.osm.pbf"

    def test_raises_when_no_url(self):
        region = {"properties": {"urls": {}}}
        with pytest.raises(ValueError, match="No PBF URL found"):
            get_pbf_url(region)


class TestHighwayValues:
    """Tests for HIGHWAY_VALUES constant."""

    def test_includes_major_roads(self):
        for road_type in ["motorway", "trunk", "primary", "secondary", "tertiary"]:
            assert road_type in HIGHWAY_VALUES

    def test_includes_link_roads(self):
        for road_type in ["motorway_link", "trunk_link", "primary_link"]:
            assert road_type in HIGHWAY_VALUES

    def test_includes_minor_roads(self):
        for road_type in ["residential", "service", "unclassified"]:
            assert road_type in HIGHWAY_VALUES

    def test_includes_paths(self):
        for road_type in ["footway", "path", "cycleway", "steps"]:
            assert road_type in HIGHWAY_VALUES

    def test_excludes_non_road_types(self):
        assert "bus_stop" not in HIGHWAY_VALUES
        assert "traffic_signals" not in HIGHWAY_VALUES


class TestRoadHandler:
    """Tests for RoadHandler class."""

    def test_initializes_empty(self):
        handler = RoadHandler()
        assert handler.roads == []
        assert len(handler.node_refs) == 0
        assert len(handler.node_locations) == 0

    def test_tracks_node_refs(self):
        """Test that node references are counted correctly."""
        handler = RoadHandler()
        # Simulate node refs from multiple ways
        handler.node_refs[1] += 1
        handler.node_refs[2] += 1
        handler.node_refs[1] += 1  # Node 1 is shared

        assert handler.node_refs[1] == 2
        assert handler.node_refs[2] == 1


class TestFilterFullyInside:
    """Tests for _filter_fully_inside function."""

    def test_empty_geodataframe(self):
        """Empty input returns empty output."""
        gdf = gpd.GeoDataFrame(columns=["id", "geometry"], crs="EPSG:4326")
        bbox = BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1)
        result = _filter_fully_inside(gdf, bbox)
        assert len(result) == 0

    def test_fully_inside_road_kept(self):
        """Road completely inside bbox is kept."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["w1"],
                "geometry": [LineString([(0.2, 0.2), (0.8, 0.8)])],
            },
            crs="EPSG:4326",
        )
        bbox = BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1)
        result = _filter_fully_inside(gdf, bbox)
        assert len(result) == 1

    def test_road_extending_outside_filtered(self):
        """Road extending outside bbox is filtered out."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["w1"],
                "geometry": [LineString([(0.2, 0.2), (1.5, 1.5)])],  # Extends past bbox
            },
            crs="EPSG:4326",
        )
        bbox = BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1)
        result = _filter_fully_inside(gdf, bbox)
        assert len(result) == 0

    def test_road_touching_boundary_kept(self):
        """Road with vertex exactly on bbox boundary is kept.

        Shapely's within() for LineStrings considers a feature "within" a polygon
        if all points are inside or on the boundary. A line ending at the corner
        is still fully contained.
        """
        gdf = gpd.GeoDataFrame(
            {
                "id": ["w1"],
                "geometry": [LineString([(0.5, 0.5), (1.0, 1.0)])],  # Ends on boundary
            },
            crs="EPSG:4326",
        )
        bbox = BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1)
        result = _filter_fully_inside(gdf, bbox)
        # within() includes boundary points, so line touching corner is kept
        assert len(result) == 1

    def test_mixed_roads_partial_kept(self):
        """Mix of inside and outside roads filters correctly."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["w1", "w2", "w3"],
                "geometry": [
                    LineString([(0.2, 0.2), (0.8, 0.8)]),  # Inside
                    LineString([(0.5, 0.5), (1.5, 1.5)]),  # Extends outside
                    LineString([(0.1, 0.1), (0.5, 0.5)]),  # Inside
                ],
            },
            crs="EPSG:4326",
        )
        bbox = BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1)
        result = _filter_fully_inside(gdf, bbox)
        assert len(result) == 2
        assert set(result["id"]) == {"w1", "w3"}

    def test_returns_copy(self):
        """Result is a copy, not a view of original."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["w1"],
                "geometry": [LineString([(0.2, 0.2), (0.8, 0.8)])],
            },
            crs="EPSG:4326",
        )
        bbox = BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1)
        result = _filter_fully_inside(gdf, bbox)
        # Modifying result should not affect original
        result["id"] = ["modified"]
        assert gdf.iloc[0]["id"] == "w1"


class TestFilterConnectorsForRoads:
    """Tests for _filter_connectors_for_roads function."""

    def test_empty_connectors(self):
        """Empty connectors input returns empty output."""
        connectors = gpd.GeoDataFrame(columns=["id", "geometry"], crs="EPSG:4326")
        roads = gpd.GeoDataFrame(
            {
                "id": ["w1"],
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            crs="EPSG:4326",
        )
        result = _filter_connectors_for_roads(connectors, roads)
        assert len(result) == 0

    def test_empty_roads(self):
        """Empty roads input returns empty output."""
        connectors = gpd.GeoDataFrame(
            {
                "id": ["n1"],
                "geometry": [Point(0, 0)],
            },
            crs="EPSG:4326",
        )
        roads = gpd.GeoDataFrame(columns=["id", "geometry"], crs="EPSG:4326")
        result = _filter_connectors_for_roads(connectors, roads)
        assert len(result) == 0

    def test_connector_at_road_endpoint_kept(self):
        """Connector at road endpoint is kept."""
        connectors = gpd.GeoDataFrame(
            {
                "id": ["n1", "n2"],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs="EPSG:4326",
        )
        roads = gpd.GeoDataFrame(
            {
                "id": ["w1"],
                "geometry": [LineString([(0, 0), (0.5, 0.5), (1, 1)])],
            },
            crs="EPSG:4326",
        )
        result = _filter_connectors_for_roads(connectors, roads)
        assert len(result) == 2
        assert set(result["id"]) == {"n1", "n2"}

    def test_connector_not_at_endpoint_filtered(self):
        """Connector not at any road endpoint is filtered out."""
        connectors = gpd.GeoDataFrame(
            {
                "id": ["n1", "n2", "n3"],
                "geometry": [Point(0, 0), Point(0.5, 0.5), Point(5, 5)],  # n3 is far away
            },
            crs="EPSG:4326",
        )
        roads = gpd.GeoDataFrame(
            {
                "id": ["w1"],
                "geometry": [LineString([(0, 0), (0.5, 0.5), (1, 1)])],
            },
            crs="EPSG:4326",
        )
        result = _filter_connectors_for_roads(connectors, roads)
        # Road endpoints are (0,0) and (1,1). Intermediate vertex (0.5,0.5) is NOT an endpoint.
        # n1 at (0,0) = start endpoint -> kept
        # n2 at (0.5,0.5) = intermediate vertex -> filtered out
        # n3 at (5,5) = distant point -> filtered out
        assert len(result) == 1
        assert result.iloc[0]["id"] == "n1"

    def test_shared_endpoint_multiple_roads(self):
        """Connector at shared endpoint of multiple roads is kept."""
        connectors = gpd.GeoDataFrame(
            {
                "id": ["n1"],
                "geometry": [Point(0.5, 0.5)],
            },
            crs="EPSG:4326",
        )
        roads = gpd.GeoDataFrame(
            {
                "id": ["w1", "w2"],
                "geometry": [
                    LineString([(0, 0), (0.5, 0.5)]),
                    LineString([(0.5, 0.5), (1, 1)]),
                ],
            },
            crs="EPSG:4326",
        )
        result = _filter_connectors_for_roads(connectors, roads)
        assert len(result) == 1

    def test_handles_null_geometry(self):
        """Roads with null geometry don't cause errors."""
        connectors = gpd.GeoDataFrame(
            {
                "id": ["n1"],
                "geometry": [Point(0, 0)],
            },
            crs="EPSG:4326",
        )
        roads = gpd.GeoDataFrame(
            {
                "id": ["w1", "w2"],
                "geometry": [None, LineString([(0, 0), (1, 1)])],
            },
            crs="EPSG:4326",
        )
        result = _filter_connectors_for_roads(connectors, roads)
        assert len(result) == 1

    def test_returns_copy(self):
        """Result is a copy, not a view of original."""
        connectors = gpd.GeoDataFrame(
            {
                "id": ["n1"],
                "geometry": [Point(0, 0)],
            },
            crs="EPSG:4326",
        )
        roads = gpd.GeoDataFrame(
            {
                "id": ["w1"],
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            crs="EPSG:4326",
        )
        result = _filter_connectors_for_roads(connectors, roads)
        result["id"] = ["modified"]
        assert connectors.iloc[0]["id"] == "n1"


class TestNodeIdsToConnectors:
    """Tests for _node_ids_to_connectors function."""

    def test_none_returns_none(self):
        """None input returns None."""
        geom = LineString([(0, 0), (1, 1)])
        assert _node_ids_to_connectors(None, geom) is None

    def test_empty_list_returns_none(self):
        """Empty list returns None."""
        geom = LineString([(0, 0), (1, 1)])
        assert _node_ids_to_connectors([], geom) is None

    def test_single_node_returns_none(self):
        """Single node (degenerate linestring) returns None."""
        geom = LineString([(0, 0), (1, 1)])
        assert _node_ids_to_connectors([123], geom) is None

    def test_two_nodes_returns_two_connectors(self):
        """Two nodes returns connectors at 0.0 and 1.0."""
        geom = LineString([(0, 0), (1, 1)])
        result = _node_ids_to_connectors([100, 200], geom)

        assert result is not None
        assert len(result) == 2

        # First connector at position 0.0
        assert result[0]["at"] == 0.0
        assert result[0]["connector_id"] == "n100"

        # Second connector at position 1.0
        assert result[1]["at"] == 1.0
        assert result[1]["connector_id"] == "n200"

    def test_multiple_nodes_includes_all(self):
        """All nodes are included with geodetic position."""
        # 4 evenly spaced points along latitude line
        geom = LineString([(0, 0), (0, 1), (0, 2), (0, 3)])
        result = _node_ids_to_connectors([100, 150, 175, 200], geom)

        assert result is not None
        assert len(result) == 4  # All nodes included

        # First connector (start node)
        assert result[0]["at"] == 0.0
        assert result[0]["connector_id"] == "n100"

        # Middle connectors (at ~0.33 and ~0.67 for evenly spaced)
        assert 0.3 < result[1]["at"] < 0.4
        assert result[1]["connector_id"] == "n150"
        assert 0.6 < result[2]["at"] < 0.7
        assert result[2]["connector_id"] == "n175"

        # Last connector (end node)
        assert result[3]["at"] == 1.0
        assert result[3]["connector_id"] == "n200"

    def test_connector_id_format(self):
        """Connector IDs use 'n' prefix for OSM node IDs."""
        geom = LineString([(0, 0), (1, 1)])
        result = _node_ids_to_connectors([61341696, 99999999], geom)

        assert result[0]["connector_id"] == "n61341696"
        assert result[1]["connector_id"] == "n99999999"

    def test_handles_numpy_array(self):
        """Should handle numpy arrays (common from pyosmium parsing)."""
        import numpy as np

        geom = LineString([(0, 0), (0, 1), (0, 2)])
        node_ids = np.array([100, 150, 200])
        result = _node_ids_to_connectors(node_ids, geom)

        assert result is not None
        assert len(result) == 3  # All nodes included
        assert result[0]["connector_id"] == "n100"
        assert result[1]["connector_id"] == "n150"
        assert result[2]["connector_id"] == "n200"

    def test_mismatched_node_count_falls_back_to_endpoints(self):
        """If node count doesn't match coordinate count, fall back to endpoints."""
        # 3 coordinates but only 2 node_ids
        geom = LineString([(0, 0), (0, 1), (0, 2)])
        result = _node_ids_to_connectors([100, 200], geom)

        # Should fall back to endpoints only
        assert result is not None
        assert len(result) == 2
        assert result[0]["at"] == 0.0
        assert result[1]["at"] == 1.0


class TestTransformToOvertureSchemaWithConnectors:
    """Tests for _transform_to_overture_schema preserving connectors."""

    def test_creates_connectors_from_node_ids(self):
        """Transformation creates connectors column from node_ids."""
        # Geometry with 3 coordinates to match 3 node_ids
        gdf = gpd.GeoDataFrame(
            {
                "id": ["w123@1"],
                "geometry": [LineString([(0, 0), (0.5, 0.5), (1, 1)])],
                "tags": [{"highway": "residential"}],
                "name": ["Main Street"],
                "node_ids": [[100, 150, 200]],  # Three nodes in way
            },
            crs="EPSG:4326",
        )
        result = _transform_to_overture_schema(gdf)

        # Should have connectors column
        assert "connectors" in result.columns

        # Should have three connectors (all nodes now included)
        connectors = result.iloc[0]["connectors"]
        assert connectors is not None
        assert len(connectors) == 3
        assert connectors[0]["at"] == 0.0
        assert connectors[0]["connector_id"] == "n100"
        # Middle connector at ~0.5
        assert 0.4 < connectors[1]["at"] < 0.6
        assert connectors[1]["connector_id"] == "n150"
        assert connectors[2]["at"] == 1.0
        assert connectors[2]["connector_id"] == "n200"

    def test_handles_missing_node_ids(self):
        """Transformation handles missing node_ids column gracefully."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["w123@1"],
                "geometry": [LineString([(0, 0), (1, 1)])],
                "tags": [{"highway": "residential"}],
                "name": ["Main Street"],
                # No node_ids column
            },
            crs="EPSG:4326",
        )
        result = _transform_to_overture_schema(gdf)

        # Should still have connectors column (with None values)
        assert "connectors" in result.columns
        assert result.iloc[0]["connectors"] is None

    def test_handles_none_node_ids(self):
        """Transformation handles None node_ids gracefully."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["w123@1"],
                "geometry": [LineString([(0, 0), (1, 1)])],
                "tags": [{"highway": "residential"}],
                "name": ["Main Street"],
                "node_ids": [None],
            },
            crs="EPSG:4326",
        )
        result = _transform_to_overture_schema(gdf)

        assert "connectors" in result.columns
        assert result.iloc[0]["connectors"] is None


class TestTransformConnectorsSchemaWithNormalization:
    """Tests for _transform_connectors_schema ID normalization."""

    def test_strips_version_from_id(self):
        """Connector IDs should have @version suffix stripped."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["n61341696@5", "n12345678@10"],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs="EPSG:4326",
        )
        result = _transform_connectors_schema(gdf)

        # IDs should be normalized (no @version)
        assert result.iloc[0]["id"] == "n61341696"
        assert result.iloc[1]["id"] == "n12345678"

    def test_handles_id_without_version(self):
        """IDs without @version should remain unchanged."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["n61341696", "n12345678"],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs="EPSG:4326",
        )
        result = _transform_connectors_schema(gdf)

        assert result.iloc[0]["id"] == "n61341696"
        assert result.iloc[1]["id"] == "n12345678"

    def test_normalized_ids_match_segment_connectors(self):
        """Connector file IDs should match segment connector_id references.

        This is critical for 1:1 mapping between segments and connectors:
        - Segment connectors reference: n{node_id}
        - Connector file IDs: n{node_id} (after stripping @version)
        """
        # Simulate OSM connector data
        connectors_gdf = gpd.GeoDataFrame(
            {
                "id": ["n100@3", "n200@5"],  # Raw OSM format with version
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs="EPSG:4326",
        )
        connectors_result = _transform_connectors_schema(connectors_gdf)

        # Simulate OSM segment with node_ids converted to connectors
        # Need geometry with 2 coords to match 2 node_ids
        segment_geometry = LineString([(0, 0), (1, 1)])
        segment_connectors = _node_ids_to_connectors([100, 200], segment_geometry)

        # Verify 1:1 mapping
        segment_conn_ids = {c["connector_id"] for c in segment_connectors}
        file_conn_ids = set(connectors_result["id"].values)

        assert segment_conn_ids == file_conn_ids
