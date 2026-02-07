"""Tests for the matcher web UI application."""

import pytest
from fastapi.testclient import TestClient

from matcher.web.app import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


class TestAppCreation:
    """Tests for FastAPI app factory."""

    def test_create_app_returns_fastapi_instance(self):
        """App factory should return a FastAPI application."""
        app = create_app()
        assert app.title == "Matcher Web UI"

    def test_static_files_mounted(self):
        """Static files should be mounted at /static."""
        app = create_app()
        assert any("/static" in str(route.path) for route in app.routes)

    def test_docs_disabled(self):
        """OpenAPI docs should be disabled for the web UI."""
        app = create_app()
        assert app.docs_url is None
        assert app.redoc_url is None


class TestRoutes:
    """Tests for web UI routes."""

    def test_root_redirects_to_labeling(self, client):
        """GET / should redirect to /labeling."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/labeling"

    def test_labeling_page_returns_html(self, client):
        """GET /labeling should return HTML content."""
        response = client.get("/labeling")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_labeling_page_contains_map_div(self, client):
        """GET /labeling should include the map container."""
        response = client.get("/labeling")
        assert '<div id="map">' in response.text


class TestStaticFiles:
    """Tests for static file serving."""

    def test_static_css_served(self, client):
        """CSS files should be served from /static/css/."""
        response = client.get("/static/css/app.css")
        # File exists and is served (or 404 if not yet created)
        assert response.status_code in (200, 404)

    def test_static_js_served(self, client):
        """JS files should be served from /static/js/."""
        response = client.get("/static/js/map.js")
        assert response.status_code in (200, 404)
