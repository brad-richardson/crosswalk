"""Tests for the integration QA routes in the matcher web UI."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString

from matcher.web.app import create_app


def _make_edges_gdf():
    """Create a test GeoDataFrame mimicking integration edges."""
    return gpd.GeoDataFrame(
        {
            "edge_id": [1, 2, 3],
            "original_id": ["orig_1", "orig_2", "orig_3"],
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
                LineString([(2, 2), (3, 3)]),
            ],
            "road_class": ["residential", "primary", "residential"],
            "length_m": [100.0, 200.0, 150.0],
            "_source": ["reference", "target_matched", "target_new"],
        },
        crs="EPSG:4326",
    )


def _make_net_new_gdf():
    """Create a test GeoDataFrame for net new edges."""
    return gpd.GeoDataFrame(
        {
            "edge_id": [10, 11],
            "original_id": ["new_1", "new_2"],
            "geometry": [
                LineString([(5, 5), (6, 6)]),
                LineString([(6, 6), (7, 7)]),
            ],
            "road_class": ["residential", "tertiary"],
            "length_m": [120.0, 80.0],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def mock_qa_services():
    """Mock the service functions used by QA routes."""
    with (
        patch("matcher.web.routes.qa.list_datasets") as mock_list,
        patch("matcher.web.routes.qa.load_qa_edges") as mock_load,
        patch("matcher.web.routes.qa.record_qa_decision") as mock_record,
        patch("matcher.web.routes.qa.integration_cache_dir") as mock_cache_dir,
    ):
        mock_list.return_value = ["dataset_a", "dataset_b"]
        mock_load.return_value = {
            "edges": _make_edges_gdf(),
            "net_new_edges": _make_net_new_gdf(),
            "disconnected_edges": None,
            "filtered_edges": None,
            "bridge_edges": None,
        }
        # Mock cache dir to return a non-existent path by default
        mock_cache_dir.return_value = MagicMock()
        mock_cache_dir.return_value.__truediv__ = MagicMock(
            return_value=MagicMock(exists=MagicMock(return_value=False))
        )

        yield {
            "list_datasets": mock_list,
            "load_qa_edges": mock_load,
            "record_qa_decision": mock_record,
            "integration_cache_dir": mock_cache_dir,
        }


@pytest.fixture
def client(mock_qa_services):
    """Create a test client with mocked services."""
    app = create_app()
    return TestClient(app)


class TestQAPageRoute:
    """Tests for GET /qa."""

    def test_qa_page_no_dataset(self, client):
        """GET /qa with no dataset returns 200."""
        response = client.get("/qa")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_qa_page_with_dataset(self, client):
        """GET /qa?dataset=dataset_a returns 200 with edge data."""
        response = client.get("/qa?dataset=dataset_a")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # Should contain edge GeoJSON data
        assert "edgeGeojson" in response.text

    def test_qa_page_contains_map(self, client):
        """GET /qa should include the map container."""
        response = client.get("/qa")
        assert '<div id="map">' in response.text

    def test_qa_page_contains_edge_detail_div(self, client):
        """GET /qa should include the edge-detail container."""
        response = client.get("/qa?dataset=dataset_a")
        assert 'id="edge-detail"' in response.text

    def test_qa_page_contains_qa_map_js(self, client):
        """GET /qa with dataset should include qa-map.js script."""
        response = client.get("/qa?dataset=dataset_a")
        assert "qa-map.js" in response.text

    def test_qa_page_with_type_param(self, client):
        """GET /qa?dataset=dataset_a&type=orphan passes type to template."""
        response = client.get("/qa?dataset=dataset_a&type=orphan")
        assert response.status_code == 200
        assert "Orphan" in response.text


class TestEdgeDetailRoute:
    """Tests for GET /qa/edge/{edge_id}."""

    def test_edge_detail_returns_200(self, client):
        """GET /qa/edge/2 returns 200 with edge info."""
        response = client.get("/qa/edge/2?dataset=dataset_a&type=merged")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_edge_detail_shows_edge_id(self, client):
        """GET /qa/edge/2 should display the edge ID."""
        response = client.get("/qa/edge/2?dataset=dataset_a&type=merged")
        assert "Edge 2" in response.text

    def test_edge_detail_shows_road_class(self, client):
        """GET /qa/edge/2 should display the road class."""
        response = client.get("/qa/edge/2?dataset=dataset_a&type=merged")
        assert "primary" in response.text

    def test_edge_detail_has_accept_reject_buttons(self, client):
        """GET /qa/edge/2 should have accept and reject buttons."""
        response = client.get("/qa/edge/2?dataset=dataset_a&type=merged")
        assert "Accept" in response.text
        assert "Reject" in response.text

    def test_edge_detail_not_found(self, client):
        """GET /qa/edge/999 returns empty div when edge not found."""
        response = client.get("/qa/edge/999?dataset=dataset_a&type=merged")
        assert response.status_code == 200
        # Should return empty div for not-found edge
        assert "Edge 999" not in response.text

    def test_edge_detail_has_notes_field(self, client):
        """GET /qa/edge/2 should have a notes textarea."""
        response = client.get("/qa/edge/2?dataset=dataset_a&type=merged")
        assert "qa-note" in response.text


class TestDecisionRoute:
    """Tests for POST /qa/decision."""

    def test_decision_returns_200(self, client, mock_qa_services):
        """POST /qa/decision returns 200."""
        response = client.post(
            "/qa/decision",
            data={
                "edge_id": 2,
                "original_id": "orig_2",
                "dataset": "dataset_a",
                "edge_type": "merged",
                "decision": "correct",
                "reason": "accepted",
                "note": "",
            },
        )
        assert response.status_code == 200
        mock_qa_services["record_qa_decision"].assert_called_once()

    def test_decision_shows_success_message(self, client, mock_qa_services):
        """POST /qa/decision should show a success message."""
        response = client.post(
            "/qa/decision",
            data={
                "edge_id": 2,
                "original_id": "orig_2",
                "dataset": "dataset_a",
                "edge_type": "merged",
                "decision": "correct",
                "reason": "accepted",
                "note": "",
            },
        )
        assert "Recorded" in response.text
        assert "correct" in response.text

    def test_decision_reject(self, client, mock_qa_services):
        """POST /qa/decision with reject returns success."""
        response = client.post(
            "/qa/decision",
            data={
                "edge_id": 2,
                "original_id": "orig_2",
                "dataset": "dataset_a",
                "edge_type": "orphan",
                "decision": "incorrect",
                "reason": "rejected",
                "note": "looks wrong",
            },
        )
        assert response.status_code == 200
        mock_qa_services["record_qa_decision"].assert_called_once()


class TestPipelineRoute:
    """Tests for GET /qa/pipeline/{dataset}."""

    def test_pipeline_not_run(self, client, mock_qa_services):
        """GET /qa/pipeline/dataset_a returns not_run when no output."""
        # Default mock returns exists=False
        response = client.get("/qa/pipeline/dataset_a")
        assert response.status_code == 200
        assert "Not run yet" in response.text

    def test_pipeline_ready(self, client, mock_qa_services):
        """GET /qa/pipeline/dataset_a returns ready when output exists."""
        # Make the path mock return exists=True
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_cache = MagicMock()
        mock_cache.__truediv__ = MagicMock(return_value=mock_path)
        mock_qa_services["integration_cache_dir"].return_value = mock_cache

        response = client.get("/qa/pipeline/dataset_a")
        assert response.status_code == 200
        assert "Ready" in response.text


class TestEdgeGeoJSON:
    """Tests for edge GeoJSON conversion."""

    def test_geojson_contains_features(self, client):
        """QA page GeoJSON should contain features from edge data."""
        response = client.get("/qa?dataset=dataset_a")
        # Should have reference and non-reference features
        assert "reference" in response.text
        assert "non_reference" in response.text

    def test_geojson_contains_net_new(self, client):
        """QA page GeoJSON should contain net_new features."""
        response = client.get("/qa?dataset=dataset_a")
        assert "net_new" in response.text

    def test_geojson_contains_colors(self, client):
        """QA page GeoJSON should contain color codes."""
        response = client.get("/qa?dataset=dataset_a")
        # Reference blue
        assert "#2196F3" in response.text
        # Net new cyan
        assert "#00bcd4" in response.text
