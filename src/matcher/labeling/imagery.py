"""Editor Layer Index integration for dynamic satellite imagery.

This module provides access to satellite/aerial imagery from the OSM Editor Layer Index,
which aggregates imagery sources from various providers including ESRI, Mapbox, and
regional orthoimagery services.

See: https://github.com/osmlab/editor-layer-index
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

# Cache directory for ELI data
CACHE_DIR = Path.home() / ".cache" / "matcher"
ELI_CACHE_FILE = CACHE_DIR / "editor-layer-index.json"
ELI_URL = "https://osmlab.github.io/editor-layer-index/imagery.json"

# Environment variable for Mapbox API key
MAPBOX_API_KEY_ENV = "MAPBOX_ACCESS_TOKEN"


@dataclass
class ImageryLayer:
    """A satellite/aerial imagery layer."""

    id: str
    name: str
    url: str
    max_zoom: int
    max_native_zoom: int | None
    attribution: str
    attribution_url: str | None
    subdomains: list[str] | None
    tms: bool  # True if y-axis is flipped


def get_mapbox_api_key() -> str | None:
    """Get Mapbox API key from environment variable."""
    return os.environ.get(MAPBOX_API_KEY_ENV)


def _convert_eli_url(url: str) -> tuple[str, list[str] | None, bool]:
    """Convert ELI URL template to Leaflet/Folium format.

    Args:
        url: ELI URL template with {zoom}, {x}, {y}, {switch:...}, etc.

    Returns:
        Tuple of (converted_url, subdomains, tms_y_flip)
    """
    # Handle {switch:a,b,c} -> {s} with subdomains list
    switch_match = re.search(r"\{switch:([^}]+)\}", url)
    subdomains = None
    if switch_match:
        subdomains = switch_match.group(1).split(",")
        url = re.sub(r"\{switch:[^}]+\}", "{s}", url)

    # Normalize zoom placeholder
    url = url.replace("{zoom}", "{z}")

    # Handle TMS y-flip {-y}
    tms = "{-y}" in url
    url = url.replace("{-y}", "{y}")

    return url, subdomains, tms


def _substitute_api_key(url: str, api_key: str | None) -> str | None:
    """Substitute API key placeholder in URL.

    Returns None if API key is required but not provided.
    """
    if "{apikey}" in url or "access_token={" in url:
        if not api_key:
            return None
        url = url.replace("{apikey}", api_key)
        # Handle Mapbox-style access_token
        url = re.sub(r"access_token=\{[^}]+\}", f"access_token={api_key}", url)
    return url


def load_eli_data(force_refresh: bool = False) -> list[dict]:
    """Load Editor Layer Index data, using cache if available.

    Args:
        force_refresh: If True, download fresh data even if cache exists.

    Returns:
        List of layer dictionaries from ELI.
    """
    # Check cache first
    if not force_refresh and ELI_CACHE_FILE.exists():
        try:
            with open(ELI_CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # Cache corrupted, re-download

    # Download fresh data
    try:
        with urlopen(ELI_URL, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        # Cache for future use
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(ELI_CACHE_FILE, "w") as f:
            json.dump(data, f)

        return data
    except Exception:
        # If download fails, try to use stale cache
        if ELI_CACHE_FILE.exists():
            with open(ELI_CACHE_FILE) as f:
                return json.load(f)
        raise


def get_global_satellite_layers(include_mapbox: bool = True) -> list[ImageryLayer]:
    """Get global satellite/aerial imagery layers.

    Args:
        include_mapbox: If True, include Mapbox if API key is available.

    Returns:
        List of ImageryLayer objects for global satellite imagery.
    """
    mapbox_key = get_mapbox_api_key()

    try:
        eli_data = load_eli_data()
    except Exception:
        # Fallback to hardcoded layers if ELI unavailable
        return _get_fallback_layers(mapbox_key if include_mapbox else None)

    layers = []
    target_ids = ["EsriWorldImagery", "EsriWorldImageryClarity", "OpenAerialMapMosaic"]
    if include_mapbox and mapbox_key:
        target_ids.append("Mapbox")

    for item in eli_data:
        if item.get("id") not in target_ids:
            continue
        if item.get("type") not in ("tms", "xyz"):
            continue

        url = item.get("url", "")
        url = _substitute_api_key(url, mapbox_key)
        if url is None:
            continue  # API key required but not available

        converted_url, subdomains, tms = _convert_eli_url(url)

        # Get max zoom from extent
        extent = item.get("extent", {})
        max_zoom = extent.get("max_zoom", 19)

        # Attribution
        attr = item.get("attribution", {})
        attr_text = attr.get("text", item.get("name", ""))
        attr_url = attr.get("url")

        layers.append(
            ImageryLayer(
                id=item["id"],
                name=item.get("name", item["id"]),
                url=converted_url,
                max_zoom=max_zoom,
                max_native_zoom=_get_native_zoom(item["id"], max_zoom),
                attribution=attr_text,
                attribution_url=attr_url,
                subdomains=subdomains,
                tms=tms,
            )
        )

    # Sort so Mapbox is first (if available), then ESRI, then others
    priority = {"Mapbox": 0, "EsriWorldImagery": 1, "EsriWorldImageryClarity": 2}
    layers.sort(key=lambda x: priority.get(x.id, 99))

    return layers if layers else _get_fallback_layers(mapbox_key if include_mapbox else None)


def _get_native_zoom(layer_id: str, max_zoom: int) -> int | None:
    """Get the actual native zoom level for a layer.

    Some providers advertise higher max_zoom than they actually have data for.
    """
    # Based on empirical testing
    native_zoom_overrides = {
        "EsriWorldImagery": 19,
        "EsriWorldImageryClarity": 19,
        "Mapbox": None,  # Mapbox actually has data at high zooms
        "OpenAerialMapMosaic": None,  # Varies by location
    }
    return native_zoom_overrides.get(layer_id, max_zoom)


def _get_fallback_layers(mapbox_key: str | None = None) -> list[ImageryLayer]:
    """Get hardcoded fallback layers if ELI is unavailable."""
    layers = [
        ImageryLayer(
            id="EsriWorldImagery",
            name="Esri World Imagery",
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            max_zoom=22,
            max_native_zoom=19,
            attribution="Esri",
            attribution_url="https://wiki.openstreetmap.org/wiki/Esri",
            subdomains=None,
            tms=False,
        ),
    ]

    if mapbox_key:
        layers.insert(
            0,
            ImageryLayer(
                id="Mapbox",
                name="Mapbox Satellite",
                url=f"https://a.tiles.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}.jpg?access_token={mapbox_key}",
                max_zoom=22,
                max_native_zoom=None,
                attribution="Mapbox",
                attribution_url="https://www.mapbox.com/about/maps",
                subdomains=None,
                tms=False,
            ),
        )

    return layers


def get_best_satellite_layer() -> ImageryLayer | None:
    """Get the best available satellite layer.

    Returns Mapbox if API key is available, otherwise ESRI.
    """
    layers = get_global_satellite_layers(include_mapbox=True)
    return layers[0] if layers else None
