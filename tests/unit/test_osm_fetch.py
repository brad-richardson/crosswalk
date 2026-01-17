"""Tests for OSM PBF fetching modules."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from matcher.fetch.osm import (
    _build_level_rules,
    _build_road_flags,
    _get_level_from_rules,
    _has_flag,
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
        assert sources[0]["record_id"] == "n456@1"


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
