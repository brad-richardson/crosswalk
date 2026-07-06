"""Tests for the crosswalk web UI application."""

import pytest

pytest.importorskip(
    "fastapi", reason="fastapi not installed (install with: pip install -e '.[web]')"
)

from fastapi.testclient import TestClient  # noqa: E402

from crosswalk.web.app import create_app  # noqa: E402


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

    def test_root_redirects_to_dashboard(self, client):
        """GET / should redirect to /dashboard."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/dashboard"

    def test_labeling_page_returns_html(self, client):
        """GET /labeling should return HTML content."""
        response = client.get("/labeling")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_labeling_page_contains_map_div(self, client):
        """GET /labeling should include the map container."""
        response = client.get("/labeling")
        assert '<div id="map">' in response.text


class TestBaseTemplate:
    """Tests for the base HTML template content."""

    def test_maplibre_css_included(self, client):
        """Template should include MapLibre GL CSS from CDN."""
        response = client.get("/labeling")
        assert "maplibre-gl.css" in response.text

    def test_maplibre_js_included(self, client):
        """Template should include MapLibre GL JS from CDN."""
        response = client.get("/labeling")
        assert "maplibre-gl.js" in response.text

    def test_htmx_included(self, client):
        """Template should include HTMX from CDN."""
        response = client.get("/labeling")
        assert "htmx" in response.text
        assert "unpkg.com/htmx.org@2.0" in response.text

    def test_app_css_linked(self, client):
        """Template should link to app.css."""
        response = client.get("/labeling")
        assert "/static/css/app.css" in response.text

    def test_map_js_linked(self, client):
        """Template should link to map.js."""
        response = client.get("/labeling")
        assert "/static/js/map.js" in response.text

    def test_menu_toggle_present(self, client):
        """Template should include the hamburger menu toggle button."""
        response = client.get("/labeling")
        assert 'id="menu-toggle"' in response.text

    def test_menu_has_mode_links(self, client):
        """Template should include navigation links for modes."""
        response = client.get("/labeling")
        assert "/labeling" in response.text
        assert "/review" in response.text
        assert "/qa" in response.text

    def test_dataset_picker_present(self, client):
        """Template should include the dataset picker select."""
        response = client.get("/labeling")
        assert 'id="dataset-picker"' in response.text

    def test_overlay_content_present(self, client):
        """Template should include the overlay content container."""
        response = client.get("/labeling")
        assert 'id="overlay-content"' in response.text

    def test_viewport_meta_tag(self, client):
        """Template should include a viewport meta tag for mobile."""
        response = client.get("/labeling")
        assert "viewport" in response.text
        assert "width=device-width" in response.text


class TestStaticFiles:
    """Tests for static file serving."""

    def test_static_css_served(self, client):
        """CSS file should be served successfully."""
        response = client.get("/static/css/app.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

    def test_static_js_served(self, client):
        """JS file should be served successfully."""
        response = client.get("/static/js/map.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]

    def test_css_contains_map_styles(self, client):
        """CSS should contain map-first fullscreen styles."""
        response = client.get("/static/css/app.css")
        assert "#map" in response.text
        assert "position: fixed" in response.text
        assert "inset: 0" in response.text

    def test_css_contains_mobile_breakpoint(self, client):
        """CSS should contain mobile responsive breakpoint."""
        response = client.get("/static/css/app.css")
        assert "768px" in response.text

    def test_js_creates_maplibre_map(self, client):
        """JS should initialize a MapLibre GL map."""
        response = client.get("/static/js/map.js")
        assert "maplibregl.Map" in response.text
        assert "42.36" in response.text
        assert "-71.06" in response.text

    def test_js_handles_htmx_afterswap(self, client):
        """JS should listen for htmx:afterSwap events."""
        response = client.get("/static/js/map.js")
        assert "htmx:afterSwap" in response.text
        assert "data-geometry" in response.text
