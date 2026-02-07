"""Tests for the labeling routes in the matcher web UI."""

from unittest.mock import patch

import pytest

pytest.importorskip(
    "fastapi", reason="fastapi not installed (install with: pip install -e '.[web]')"
)

from fastapi.testclient import TestClient  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from matcher.labeling.data_loader import CandidatePairView  # noqa: E402
from matcher.web.app import create_app  # noqa: E402


def _make_pair(ref_id="ref_001", target_id="target_001", confidence=0.65):
    """Create a test CandidatePairView with minimal data."""
    return CandidatePairView(
        ref_id=ref_id,
        target_id=target_id,
        ref_geometry=LineString([(0, 0), (1, 1)]),
        target_geometry=LineString([(0, 0.001), (1, 1.001)]),
        ref_name="Main Street",
        target_name="Main St",
        ref_class="residential",
        target_class="residential",
        decision="review",
        confidence=confidence,
        features={"name_jaro_winkler": 0.85, "hausdorff_distance_m": 5.2},
        ref_aligned_geometry=LineString([(0, 0), (0.5, 0.5)]),
        target_aligned_geometry=LineString([(0, 0.001), (0.5, 0.501)]),
    )


@pytest.fixture
def mock_services():
    """Mock the service functions used by labeling routes."""
    with (
        patch("matcher.web.routes.labeling.list_datasets") as mock_list,
        patch("matcher.web.routes.labeling._get_candidates") as mock_get_cands,
        patch("matcher.web.routes.labeling.get_unlabeled_candidates") as mock_unlabeled,
        patch("matcher.web.routes.labeling.record_label") as mock_record,
        patch("matcher.web.routes.labeling.undo_last_label") as mock_undo,
    ):
        mock_list.return_value = ["dataset_a", "dataset_b"]
        pairs = [_make_pair(), _make_pair("ref_002", "target_002", 0.55)]
        mock_get_cands.return_value = pairs
        mock_unlabeled.return_value = pairs

        yield {
            "list_datasets": mock_list,
            "get_candidates": mock_get_cands,
            "get_unlabeled": mock_unlabeled,
            "record_label": mock_record,
            "undo_last_label": mock_undo,
        }


@pytest.fixture
def client(mock_services):
    """Create a test client with mocked services."""
    # Clear the candidate cache before each test
    from matcher.web.routes.labeling import _candidate_cache

    _candidate_cache.clear()

    app = create_app()
    return TestClient(app)


class TestLabelingPageRoute:
    """Tests for GET /labeling."""

    def test_labeling_page_no_dataset(self, client):
        """GET /labeling with no dataset returns 200 with page template."""
        response = client.get("/labeling")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Select a dataset" in response.text

    def test_labeling_page_with_dataset(self, client):
        """GET /labeling?dataset=dataset_a returns 200 with pair card."""
        response = client.get("/labeling?dataset=dataset_a")
        assert response.status_code == 200
        assert "Main Street" in response.text
        assert "Main St" in response.text

    def test_labeling_page_contains_map(self, client):
        """GET /labeling should include the map container."""
        response = client.get("/labeling")
        assert '<div id="map">' in response.text

    def test_labeling_page_contains_decision_buttons(self, client):
        """GET /labeling with dataset should contain decision buttons."""
        response = client.get("/labeling?dataset=dataset_a")
        assert "btn-match" in response.text
        assert "btn-no-match" in response.text

    def test_labeling_page_contains_datasets_in_picker(self, client):
        """GET /labeling should list datasets in the picker."""
        response = client.get("/labeling?dataset=dataset_a")
        assert "dataset_a" in response.text
        assert "dataset_b" in response.text

    def test_labeling_htmx_returns_fragment(self, client):
        """GET /labeling with HX-Request header returns pair fragment only."""
        response = client.get(
            "/labeling?dataset=dataset_a",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        # Fragment should NOT contain the full page elements
        assert "<!DOCTYPE html>" not in response.text
        # But should contain the pair card
        assert "Main Street" in response.text

    def test_labeling_htmx_no_dataset_returns_fragment(self, client):
        """GET /labeling with HX-Request but no dataset returns select message."""
        response = client.get(
            "/labeling",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "Select a dataset" in response.text

    def test_labeling_page_with_index(self, client):
        """GET /labeling?dataset=dataset_a&index=1 returns second pair."""
        response = client.get("/labeling?dataset=dataset_a&index=1")
        assert response.status_code == 200
        assert "ref_002" in response.text

    def test_labeling_page_geojson_in_attribute(self, client):
        """GET /labeling with dataset should embed geojson in data attribute."""
        response = client.get("/labeling?dataset=dataset_a")
        assert "data-geometry" in response.text
        # Should contain GeoJSON coordinate data
        assert "coordinates" in response.text

    def test_labeling_page_shows_confidence(self, client):
        """GET /labeling with dataset should show confidence percentage."""
        response = client.get("/labeling?dataset=dataset_a")
        assert "65%" in response.text

    def test_labeling_page_shows_progress(self, client):
        """GET /labeling with dataset should show pair progress."""
        response = client.get("/labeling?dataset=dataset_a")
        assert "Pair 1 of 2" in response.text


class TestLabelRoute:
    """Tests for POST /labeling/label."""

    def test_label_match_returns_200(self, client, mock_services):
        """POST /labeling/label with match label returns 200."""
        response = client.post(
            "/labeling/label",
            data={"dataset": "dataset_a", "index": 0, "label": "match"},
        )
        assert response.status_code == 200
        mock_services["record_label"].assert_called_once()

    def test_label_no_match_returns_200(self, client, mock_services):
        """POST /labeling/label with no_match label returns 200."""
        response = client.post(
            "/labeling/label",
            data={"dataset": "dataset_a", "index": 0, "label": "no_match"},
        )
        assert response.status_code == 200
        mock_services["record_label"].assert_called_once()

    def test_label_returns_next_pair(self, client, mock_services):
        """POST /labeling/label should return the next pair fragment."""
        response = client.post(
            "/labeling/label",
            data={"dataset": "dataset_a", "index": 0, "label": "match"},
        )
        assert response.status_code == 200
        # Should return HTML fragment with pair content
        assert "text/html" in response.headers["content-type"]


class TestUndoRoute:
    """Tests for POST /labeling/undo."""

    def test_undo_returns_200(self, client, mock_services):
        """POST /labeling/undo returns 200."""
        response = client.post(
            "/labeling/undo",
            data={"dataset": "dataset_a"},
        )
        assert response.status_code == 200
        mock_services["undo_last_label"].assert_called_once_with("dataset_a")

    def test_undo_returns_pair_fragment(self, client, mock_services):
        """POST /labeling/undo should return updated pair fragment."""
        response = client.post(
            "/labeling/undo",
            data={"dataset": "dataset_a"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Main Street" in response.text


class TestRootRedirect:
    """Tests for GET /."""

    def test_root_redirects(self, client):
        """GET / should redirect to /labeling."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/labeling"


class TestDatasetsEndpoint:
    """Tests for GET /datasets."""

    def test_datasets_endpoint_returns_json(self, client):
        """GET /datasets returns JSON list of dataset IDs."""
        response = client.get("/datasets")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert isinstance(data, list)
        assert data == ["dataset_a", "dataset_b"]
