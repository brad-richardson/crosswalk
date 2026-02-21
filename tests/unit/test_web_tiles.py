"""Tests for the MVT vector tile endpoint."""

from unittest.mock import patch

import pytest

pytest.importorskip(
    "fastapi", reason="fastapi not installed (install with: pip install -e '.[web]')"
)

from fastapi.testclient import TestClient  # noqa: E402

from matcher.web.app import create_app  # noqa: E402


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


class TestContextTileEndpoint:
    """Tests for /context/tiles/{dataset}/{z}/{x}/{y}.pbf."""

    def test_unknown_dataset_returns_empty_protobuf(self, client):
        """Unknown dataset should return empty protobuf, not 404."""
        response = client.get("/context/tiles/nonexistent_dataset/14/4952/6064.pbf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-protobuf"
        assert response.content == b""

    @patch("matcher.web.routes.tiles._generate_tile")
    @patch("matcher.web.routes.tiles.list_dataset_configs")
    def test_known_dataset_returns_protobuf(self, mock_configs, mock_gen, client):
        """Known dataset should return protobuf content type."""
        mock_configs.return_value = {"test_dataset"}
        mock_gen.return_value = b"\x1a\x03foo"  # dummy protobuf bytes
        response = client.get("/context/tiles/test_dataset/14/4952/6064.pbf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-protobuf"
        assert response.content == b"\x1a\x03foo"
        mock_gen.assert_called_once_with("test_dataset", 14, 4952, 6064)

    def test_cache_control_header(self, client):
        """Tile responses should have cache-control header."""
        response = client.get("/context/tiles/nonexistent_dataset/14/4952/6064.pbf")
        assert "max-age=3600" in response.headers.get("cache-control", "")
