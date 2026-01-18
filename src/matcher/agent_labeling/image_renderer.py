"""Image rendering for agent labeling pipeline.

Generates satellite imagery with geometry overlays and clean geometry-only
visualizations for AI agent labeling tasks.
"""

import io
import math
from typing import Any

import mercantile
import requests
from loguru import logger
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString

# Esri World Imagery (free, no API key required)
ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Colors for rendering (RGB)
REFERENCE_COLOR = (33, 150, 243)  # Blue #2196F3
TARGET_COLOR = (244, 67, 54)  # Red #F44336
BACKGROUND_COLOR = (255, 255, 255)  # White

# Line widths
OVERLAY_LINE_WIDTH = 4
GEOMETRY_LINE_WIDTH = 3

# Default image size
DEFAULT_IMAGE_SIZE = (512, 512)


def _to_linestring(geom: Any) -> LineString | None:
    """Convert geometry to LineString."""
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        if geom.is_empty or len(geom.geoms) == 0:
            return None
        # Return longest component
        return max(geom.geoms, key=lambda g: g.length)
    return None


def _expand_bbox(
    bbox: tuple[float, float, float, float], padding_ratio: float = 0.2
) -> tuple[float, float, float, float]:
    """Expand bounding box by padding ratio."""
    minx, miny, maxx, maxy = bbox
    width = maxx - minx
    height = maxy - miny
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    return (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)


def _get_combined_bbox(
    ref_geom: LineString, target_geom: LineString, padding_ratio: float = 0.2
) -> tuple[float, float, float, float]:
    """Get combined bounding box of both geometries with padding."""
    combined = ref_geom.union(target_geom)
    bbox = combined.bounds  # (minx, miny, maxx, maxy)
    return _expand_bbox(bbox, padding_ratio)


def _choose_zoom(bbox: tuple[float, float, float, float], target_size: int = 512) -> int:
    """Choose appropriate zoom level for bounding box.

    Aims to have the bbox fill most of the image while not requiring
    too many tiles.
    """
    minx, miny, maxx, maxy = bbox
    bbox_width = maxx - minx
    bbox_height = maxy - miny

    # Start from high zoom and work down
    for zoom in range(19, 10, -1):
        # At this zoom, how many pixels would the bbox span?
        # Each tile is 256x256 pixels
        # At zoom z, there are 2^z tiles spanning 360 degrees longitude
        tile_span_lon = 360.0 / (2**zoom)
        tile_span_lat = 170.1022 / (2**zoom)  # Approximate, varies with latitude

        bbox_pixels_x = (bbox_width / tile_span_lon) * 256
        bbox_pixels_y = (bbox_height / tile_span_lat) * 256

        # If bbox fits in target size with some margin, use this zoom
        if bbox_pixels_x < target_size * 0.9 and bbox_pixels_y < target_size * 0.9:
            return zoom

    return 14  # Default fallback


def fetch_satellite_tile(
    bbox: tuple[float, float, float, float],
    zoom: int | None = None,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Image.Image | None:
    """Fetch satellite imagery for bounding box.

    Uses Esri World Imagery tiles (free, no API key required).

    Args:
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        zoom: Zoom level (auto-calculated if not provided)
        size: Output image size (width, height)

    Returns:
        PIL Image or None if fetch fails
    """
    if zoom is None:
        zoom = _choose_zoom(bbox, min(size))

    minx, miny, maxx, maxy = bbox

    # Get tiles that cover the bbox
    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zoom))

    if not tiles:
        logger.warning(f"No tiles found for bbox {bbox} at zoom {zoom}")
        return None

    if len(tiles) > 16:
        logger.warning(f"Too many tiles ({len(tiles)}), reducing zoom")
        return fetch_satellite_tile(bbox, zoom - 1, size)

    # Calculate tile bounds
    min_tile_x = min(t.x for t in tiles)
    max_tile_x = max(t.x for t in tiles)
    min_tile_y = min(t.y for t in tiles)
    max_tile_y = max(t.y for t in tiles)

    # Create composite image
    tile_width = max_tile_x - min_tile_x + 1
    tile_height = max_tile_y - min_tile_y + 1
    composite = Image.new("RGB", (tile_width * 256, tile_height * 256))

    # Fetch and composite tiles
    session = requests.Session()
    for tile in tiles:
        try:
            url = ESRI_TILE_URL.format(z=tile.z, x=tile.x, y=tile.y)
            response = session.get(url, timeout=10)
            response.raise_for_status()

            tile_img = Image.open(io.BytesIO(response.content))
            x_offset = (tile.x - min_tile_x) * 256
            y_offset = (tile.y - min_tile_y) * 256
            composite.paste(tile_img, (x_offset, y_offset))

        except Exception as e:
            logger.warning(f"Failed to fetch tile {tile}: {e}")
            # Fill with gray for failed tiles
            x_offset = (tile.x - min_tile_x) * 256
            y_offset = (tile.y - min_tile_y) * 256
            gray = Image.new("RGB", (256, 256), (128, 128, 128))
            composite.paste(gray, (x_offset, y_offset))

    # Crop to exact bbox bounds
    # Get pixel coordinates for bbox corners
    top_left_tile = mercantile.Tile(min_tile_x, min_tile_y, zoom)
    tl_bounds = mercantile.bounds(top_left_tile)

    # Pixels per degree at this zoom level
    tile_lon_span = tl_bounds.east - tl_bounds.west
    tile_lat_span = tl_bounds.north - tl_bounds.south
    px_per_lon = 256 / tile_lon_span
    px_per_lat = 256 / tile_lat_span

    # Calculate crop box
    left_px = int((minx - tl_bounds.west) * px_per_lon)
    top_px = int((tl_bounds.north - maxy) * px_per_lat)
    right_px = int((maxx - tl_bounds.west) * px_per_lon)
    bottom_px = int((tl_bounds.north - miny) * px_per_lat)

    # Ensure bounds are valid
    left_px = max(0, left_px)
    top_px = max(0, top_px)
    right_px = min(composite.width, right_px)
    bottom_px = min(composite.height, bottom_px)

    if right_px <= left_px or bottom_px <= top_px:
        logger.warning("Invalid crop bounds, returning full composite")
        return composite.resize(size, Image.Resampling.LANCZOS)

    cropped = composite.crop((left_px, top_px, right_px, bottom_px))

    # Resize to target size
    return cropped.resize(size, Image.Resampling.LANCZOS)


def _geo_to_pixel(
    lon: float,
    lat: float,
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[int, int]:
    """Convert geographic coordinates to pixel coordinates."""
    minx, miny, maxx, maxy = bbox
    width, height = size

    # Normalize to 0-1
    norm_x = (lon - minx) / (maxx - minx) if maxx != minx else 0.5
    norm_y = (lat - miny) / (maxy - miny) if maxy != miny else 0.5

    # Convert to pixels (note: y is inverted)
    px = int(norm_x * width)
    py = int((1 - norm_y) * height)

    return px, py


def _draw_linestring(
    draw: ImageDraw.ImageDraw,
    line: LineString,
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
    dashed: bool = False,
):
    """Draw a LineString on an image."""
    if line is None or line.is_empty:
        return

    coords = list(line.coords)
    if len(coords) < 2:
        return

    # Convert to pixel coordinates
    pixel_coords = [_geo_to_pixel(lon, lat, bbox, size) for lon, lat in coords]

    if dashed:
        # Draw dashed line
        dash_length = 10
        gap_length = 5
        for i in range(len(pixel_coords) - 1):
            x1, y1 = pixel_coords[i]
            x2, y2 = pixel_coords[i + 1]

            # Calculate segment length
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx * dx + dy * dy)

            if length == 0:
                continue

            # Normalize direction
            dx /= length
            dy /= length

            # Draw dashes
            pos = 0
            while pos < length:
                dash_end = min(pos + dash_length, length)
                sx = int(x1 + dx * pos)
                sy = int(y1 + dy * pos)
                ex = int(x1 + dx * dash_end)
                ey = int(y1 + dy * dash_end)
                draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
                pos += dash_length + gap_length
    else:
        # Draw solid line
        draw.line(pixel_coords, fill=color, width=width)


def render_with_overlay(
    satellite: Image.Image,
    ref_geom: LineString | MultiLineString,
    target_geom: LineString | MultiLineString,
    bbox: tuple[float, float, float, float],
) -> Image.Image:
    """Render geometries overlaid on satellite imagery.

    Args:
        satellite: Background satellite image
        ref_geom: Reference geometry (drawn in blue, solid)
        target_geom: Target geometry (drawn in red, dashed)
        bbox: Bounding box of the image

    Returns:
        PIL Image with geometry overlay
    """
    # Convert to LineString
    ref_line = _to_linestring(ref_geom)
    target_line = _to_linestring(target_geom)

    # Create a copy to draw on
    result = satellite.copy()
    draw = ImageDraw.Draw(result)

    size = satellite.size

    # Draw target first (underneath), then reference
    if target_line:
        _draw_linestring(
            draw, target_line, bbox, size, TARGET_COLOR, OVERLAY_LINE_WIDTH, dashed=True
        )
    if ref_line:
        _draw_linestring(
            draw, ref_line, bbox, size, REFERENCE_COLOR, OVERLAY_LINE_WIDTH, dashed=False
        )

    return result


def render_geometry_only(
    ref_geom: LineString | MultiLineString,
    target_geom: LineString | MultiLineString,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    padding_ratio: float = 0.1,
) -> Image.Image:
    """Render clean two-tone geometry visualization.

    Args:
        ref_geom: Reference geometry (blue)
        target_geom: Target geometry (red)
        size: Output image size
        padding_ratio: Padding around geometries

    Returns:
        PIL Image with geometry on white background
    """
    # Convert to LineString
    ref_line = _to_linestring(ref_geom)
    target_line = _to_linestring(target_geom)

    # Get combined bbox with padding
    if ref_line and target_line:
        bbox = _get_combined_bbox(ref_line, target_line, padding_ratio)
    elif ref_line:
        bbox = _expand_bbox(ref_line.bounds, padding_ratio)
    elif target_line:
        bbox = _expand_bbox(target_line.bounds, padding_ratio)
    else:
        # No valid geometries
        return Image.new("RGB", size, BACKGROUND_COLOR)

    # Create white background
    result = Image.new("RGB", size, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(result)

    # Draw target first (underneath), then reference
    if target_line:
        _draw_linestring(
            draw, target_line, bbox, size, TARGET_COLOR, GEOMETRY_LINE_WIDTH, dashed=True
        )
    if ref_line:
        _draw_linestring(
            draw, ref_line, bbox, size, REFERENCE_COLOR, GEOMETRY_LINE_WIDTH, dashed=False
        )

    return result


def render_candidate_images(
    ref_geom: LineString | MultiLineString,
    target_geom: LineString | MultiLineString,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    padding_ratio: float = 0.2,
    fetch_satellite: bool = True,
) -> tuple[Image.Image | None, Image.Image]:
    """Render both satellite overlay and geometry-only images for a candidate.

    Args:
        ref_geom: Reference geometry
        target_geom: Target geometry
        size: Output image size
        padding_ratio: Padding around geometries
        fetch_satellite: Whether to fetch satellite imagery

    Returns:
        Tuple of (satellite_with_overlay, geometry_only).
        satellite_with_overlay may be None if fetch fails.
    """
    # Convert to LineString
    ref_line = _to_linestring(ref_geom)
    target_line = _to_linestring(target_geom)

    if not ref_line or not target_line:
        logger.warning("Invalid geometries, returning empty images")
        empty = Image.new("RGB", size, BACKGROUND_COLOR)
        return None, empty

    # Get combined bbox
    bbox = _get_combined_bbox(ref_line, target_line, padding_ratio)

    # Render geometry-only
    geometry_img = render_geometry_only(ref_geom, target_geom, size, padding_ratio)

    # Fetch and render satellite overlay
    satellite_img = None
    if fetch_satellite:
        satellite = fetch_satellite_tile(bbox, size=size)
        if satellite:
            satellite_img = render_with_overlay(satellite, ref_geom, target_geom, bbox)

    return satellite_img, geometry_img
