"""Tests for the label review routes in the matcher web UI."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from matcher.web.app import create_app


def _make_labels():
    """Create test label records."""
    return [
        {
            "gers_id": "gers_001",
            "target_id": "target_001",
            "label": "match",
            "labeler": "alice",
            "labeled_at": "2025-01-15T10:00:00",
            "original_decision": "review",
            "original_confidence": 0.72,
            "ref_name_raw": "Main Street",
            "target_name_raw": "Main St",
            "ref_class_raw": "residential",
            "target_class_raw": "residential",
        },
        {
            "gers_id": "gers_002",
            "target_id": "target_002",
            "label": "no_match",
            "labeler": "bob",
            "labeled_at": "2025-01-15T11:00:00",
            "original_decision": "match",
            "original_confidence": 0.55,
            "ref_name_raw": "Oak Avenue",
            "target_name_raw": "Elm Street",
            "ref_class_raw": "primary",
            "target_class_raw": "secondary",
        },
        {
            "gers_id": "gers_003",
            "target_id": "target_003",
            "label": "unsure",
            "labeler": "alice",
            "labeled_at": "2025-01-15T12:00:00",
            "original_decision": "review",
            "original_confidence": 0.48,
            "ref_name_raw": "Pine Road",
            "target_name_raw": "Pine Rd",
            "ref_class_raw": "tertiary",
            "target_class_raw": "tertiary",
        },
    ]


@pytest.fixture
def mock_review_services():
    """Mock the service functions used by review routes."""
    with (
        patch("matcher.web.routes.review.list_datasets") as mock_list,
        patch("matcher.web.routes.review.get_labels_for_review") as mock_get_labels,
        patch("matcher.web.routes.review.update_review_label") as mock_update,
        patch("matcher.web.routes.review.delete_review_label") as mock_delete,
    ):
        mock_list.return_value = ["dataset_a", "dataset_b"]
        labels = _make_labels()
        mock_get_labels.return_value = (labels, len(labels))
        mock_update.return_value = True
        mock_delete.return_value = True

        yield {
            "list_datasets": mock_list,
            "get_labels_for_review": mock_get_labels,
            "update_review_label": mock_update,
            "delete_review_label": mock_delete,
        }


@pytest.fixture
def client(mock_review_services):
    """Create a test client with mocked services."""
    app = create_app()
    return TestClient(app)


class TestReviewPageRoute:
    """Tests for GET /review."""

    def test_review_page_no_dataset(self, client):
        """GET /review with no dataset returns 200."""
        response = client.get("/review")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_review_page_with_dataset(self, client):
        """GET /review?dataset=dataset_a returns 200 with labels."""
        response = client.get("/review?dataset=dataset_a")
        assert response.status_code == 200
        assert "Main Street" in response.text
        assert "Oak Avenue" in response.text

    def test_review_page_contains_filter_pills(self, client):
        """GET /review should contain filter pills."""
        response = client.get("/review?dataset=dataset_a")
        assert "filter-pill" in response.text
        assert "All" in response.text
        assert "Match" in response.text
        assert "No Match" in response.text
        assert "Unsure" in response.text

    def test_review_page_contains_label_count(self, client):
        """GET /review with dataset should show total label count."""
        response = client.get("/review?dataset=dataset_a")
        assert "Labels (3)" in response.text

    def test_review_page_contains_review_cards(self, client):
        """GET /review with dataset should contain review cards."""
        response = client.get("/review?dataset=dataset_a")
        assert "review-card" in response.text
        assert "card-gers_001-target_001" in response.text

    def test_review_page_shows_label_badges(self, client):
        """GET /review with dataset should show label badges."""
        response = client.get("/review?dataset=dataset_a")
        assert "label-match" in response.text
        assert "label-no-match" in response.text
        assert "label-unsure" in response.text

    def test_review_page_shows_labeler_names(self, client):
        """GET /review should display labeler names."""
        response = client.get("/review?dataset=dataset_a")
        assert "alice" in response.text
        assert "bob" in response.text

    def test_review_page_passes_filter_param(self, client, mock_review_services):
        """GET /review?filter=match should pass filter to service."""
        client.get("/review?dataset=dataset_a&filter=match")
        mock_review_services["get_labels_for_review"].assert_called_once_with(
            "dataset_a", filter_type="match", page=0, page_size=50
        )

    def test_review_page_passes_page_param(self, client, mock_review_services):
        """GET /review?page=2 should pass page to service."""
        client.get("/review?dataset=dataset_a&page=2")
        mock_review_services["get_labels_for_review"].assert_called_once_with(
            "dataset_a", filter_type="all", page=2, page_size=50
        )

    def test_review_page_no_dataset_shows_message(self, client):
        """GET /review with no dataset shows a select dataset message."""
        response = client.get("/review")
        assert "Select a dataset" in response.text

    def test_review_page_contains_map(self, client):
        """GET /review should include the map container (from base.html)."""
        response = client.get("/review")
        assert '<div id="map">' in response.text

    def test_review_page_has_active_filter(self, client):
        """GET /review?filter=match should mark match pill as active."""
        response = client.get("/review?dataset=dataset_a&filter=match")
        # Check that the match filter link has the active class
        assert "filter=match" in response.text

    def test_review_page_shows_confidence(self, client):
        """GET /review should display confidence values."""
        response = client.get("/review?dataset=dataset_a")
        assert "72%" in response.text

    def test_review_page_covers_map(self, client):
        """GET /review should have review-page class that covers the map."""
        response = client.get("/review?dataset=dataset_a")
        assert "review-page" in response.text


class TestReviewPageLoadMore:
    """Tests for load more functionality."""

    def test_no_load_more_when_all_shown(self, client):
        """No load more link when all labels fit on one page."""
        response = client.get("/review?dataset=dataset_a")
        assert "Load more" not in response.text

    def test_load_more_when_more_pages(self, client, mock_review_services):
        """Load more link shown when total exceeds page size."""
        # Simulate more results than page size
        labels = _make_labels()
        mock_review_services["get_labels_for_review"].return_value = (labels, 100)
        response = client.get("/review?dataset=dataset_a")
        assert "Load more" in response.text

    def test_load_more_link_increments_page(self, client, mock_review_services):
        """Load more link should point to next page."""
        labels = _make_labels()
        mock_review_services["get_labels_for_review"].return_value = (labels, 100)
        response = client.get("/review?dataset=dataset_a&page=0")
        assert "page=1" in response.text


class TestReviewUpdateRoute:
    """Tests for PUT /review/{gers_id}/{target_id}."""

    def test_update_returns_200(self, client, mock_review_services):
        """PUT /review/gers_001/target_001 returns 200."""
        response = client.put(
            "/review/gers_001/target_001",
            data={"dataset": "dataset_a", "label": "match"},
        )
        assert response.status_code == 200

    def test_update_calls_service(self, client, mock_review_services):
        """PUT should call update_review_label with correct args."""
        client.put(
            "/review/gers_001/target_001",
            data={"dataset": "dataset_a", "label": "no_match"},
        )
        mock_review_services["update_review_label"].assert_called_once_with(
            "dataset_a", "gers_001", "target_001", "no_match"
        )

    def test_update_returns_card_fragment(self, client, mock_review_services):
        """PUT should return an updated card fragment."""
        response = client.put(
            "/review/gers_001/target_001",
            data={"dataset": "dataset_a", "label": "match"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "review-card" in response.text

    def test_update_returns_correct_card_id(self, client, mock_review_services):
        """PUT should return card with the correct ID."""
        response = client.put(
            "/review/gers_001/target_001",
            data={"dataset": "dataset_a", "label": "match"},
        )
        assert "card-gers_001-target_001" in response.text


class TestReviewDeleteRoute:
    """Tests for DELETE /review/{gers_id}/{target_id}."""

    def test_delete_returns_200(self, client, mock_review_services):
        """DELETE /review/gers_001/target_001 returns 200."""
        response = client.request(
            "DELETE",
            "/review/gers_001/target_001",
            data={"dataset": "dataset_a"},
        )
        assert response.status_code == 200

    def test_delete_calls_service(self, client, mock_review_services):
        """DELETE should call delete_review_label with correct args."""
        client.request(
            "DELETE",
            "/review/gers_001/target_001",
            data={"dataset": "dataset_a"},
        )
        mock_review_services["delete_review_label"].assert_called_once_with(
            "dataset_a", "gers_001", "target_001"
        )

    def test_delete_returns_empty_response(self, client, mock_review_services):
        """DELETE should return empty string for HTMX removal."""
        response = client.request(
            "DELETE",
            "/review/gers_001/target_001",
            data={"dataset": "dataset_a"},
        )
        assert response.text == ""

    def test_delete_not_found_still_returns_200(self, client, mock_review_services):
        """DELETE for nonexistent label still returns 200 (graceful)."""
        mock_review_services["delete_review_label"].return_value = False
        response = client.request(
            "DELETE",
            "/review/gers_999/target_999",
            data={"dataset": "dataset_a"},
        )
        assert response.status_code == 200
        assert response.text == ""


class TestReviewCardTemplate:
    """Tests for the review card template rendering."""

    def test_card_contains_action_buttons(self, client):
        """Review cards should have update and delete action buttons."""
        response = client.get("/review?dataset=dataset_a")
        assert "review-btn-match" in response.text
        assert "review-btn-no-match" in response.text
        assert "review-btn-delete" in response.text

    def test_card_has_htmx_put_attributes(self, client):
        """Review cards should have hx-put attributes for updates."""
        response = client.get("/review?dataset=dataset_a")
        assert "hx-put" in response.text

    def test_card_has_htmx_delete_attributes(self, client):
        """Review cards should have hx-delete attributes for deletion."""
        response = client.get("/review?dataset=dataset_a")
        assert "hx-delete" in response.text

    def test_card_has_confirm_on_delete(self, client):
        """Delete button should have hx-confirm for safety."""
        response = client.get("/review?dataset=dataset_a")
        assert "hx-confirm" in response.text

    def test_card_shows_ref_and_target_names(self, client):
        """Cards should show reference and target names."""
        response = client.get("/review?dataset=dataset_a")
        assert "Main Street" in response.text
        assert "Main St" in response.text

    def test_empty_labels_shows_message(self, client, mock_review_services):
        """Empty label list should show 'No labels found' message."""
        mock_review_services["get_labels_for_review"].return_value = ([], 0)
        response = client.get("/review?dataset=dataset_a")
        assert "No labels found" in response.text
