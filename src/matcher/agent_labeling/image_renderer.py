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
from shapely.geometry import LineString, MultiLineString, Polygon, box

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

# Dynamic image sizing constants
MIN_IMAGE_SIZE = 128
MAX_IMAGE_SIZE = 512
TARGET_METERS_PER_PIXEL = 0.5  # Geometry resolution in meters per pixel
MIN_REFERENCE_VISIBLE_M = 25  # Minimum meters of reference geometry to show
SATELLITE_SIZE_MULTIPLIER = 2  # Satellite images are 2x geometry size


def _calculate_size_from_bbox(
    bbox: tuple[float, float, float, float],
    min_size: int = MIN_IMAGE_SIZE,
    max_size: int = MAX_IMAGE_SIZE,
) -> tuple[int, int]:
    """Calculate image size based on bbox extent in meters.

    Size is clamped to [min_size, max_size] and rounded to nearest 64.
    """
    minx, miny, maxx, maxy = bbox

    # Convert degrees to meters (approximate)
    lat_center = (miny + maxy) / 2
    lon_span_m = (maxx - minx) * 111000 * math.cos(math.radians(lat_center))
    lat_span_m = (maxy - miny) * 111000

    max_span = max(lon_span_m, lat_span_m)
    ideal_size = int(max_span / TARGET_METERS_PER_PIXEL)

    # Clamp to [min_size, max_size] and round to nearest 64
    clamped = max(min_size, min(max_size, ideal_size))
    rounded = ((clamped + 63) // 64) * 64

    return (rounded, rounded)


def calculate_image_size(
    target_geom: LineString | MultiLineString,
    min_size: int = MIN_IMAGE_SIZE,
    max_size: int = MAX_IMAGE_SIZE,
    padding_ratio: float = 0.3,
) -> tuple[int, int]:
    """Calculate appropriate image size based on target geometry extent.

    Uses target geometry only (not union with reference) to determine size.
    Size is clamped to [128, 512] and rounded to nearest 64.

    Args:
        target_geom: Target geometry to size around
        min_size: Minimum image dimension
        max_size: Maximum image dimension
        padding_ratio: Padding around geometry as ratio of extent

    Returns:
        Tuple of (width, height) in pixels
    """
    line = _to_linestring(target_geom)
    if not line:
        return (min_size, min_size)

    bbox = _expand_bbox(line.bounds, padding_ratio)
    return _calculate_size_from_bbox(bbox, min_size, max_size)


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


def _make_bbox_square(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Expand bbox to be square (in degrees) to avoid image stretching.

    Expands the smaller dimension to match the larger one, centered on the original.
    """
    minx, miny, maxx, maxy = bbox
    width = maxx - minx
    height = maxy - miny

    if width > height:
        # Expand height
        diff = (width - height) / 2
        miny -= diff
        maxy += diff
    elif height > width:
        # Expand width
        diff = (height - width) / 2
        minx -= diff
        maxx += diff

    return (minx, miny, maxx, maxy)


def _bbox_to_polygon(bbox: tuple[float, float, float, float]) -> Polygon:
    """Convert bbox to a Shapely polygon for intersection calculations."""
    minx, miny, maxx, maxy = bbox
    return box(minx, miny, maxx, maxy)


def _geometry_length_in_bbox_meters(
    geom: LineString, bbox: tuple[float, float, float, float]
) -> float:
    """Calculate the length in meters of a geometry that falls within a bbox."""
    if geom is None or geom.is_empty:
        return 0.0

    bbox_poly = _bbox_to_polygon(bbox)
    clipped = geom.intersection(bbox_poly)

    if clipped.is_empty:
        return 0.0

    # Convert clipped geometry length from degrees to meters (approximate)
    # Use the bbox center latitude for the conversion
    minx, miny, maxx, maxy = bbox
    lat_center = (miny + maxy) / 2

    # For a LineString, length is in degrees - convert to meters
    # At the equator, 1 degree ≈ 111km. Adjust for latitude.
    meters_per_degree = 111000 * math.cos(math.radians(lat_center))

    return clipped.length * meters_per_degree


def _expand_bbox_for_reference(
    target_bbox: tuple[float, float, float, float],
    ref_line: LineString,
    min_visible_m: float = MIN_REFERENCE_VISIBLE_M,
) -> tuple[float, float, float, float]:
    """Expand bbox to ensure minimum reference geometry is visible.

    Starts with target-based bbox and expands toward reference geometry
    until at least min_visible_m meters of reference are within the bbox.

    Args:
        target_bbox: Initial bbox based on target geometry
        ref_line: Reference geometry LineString
        min_visible_m: Minimum meters of reference to make visible

    Returns:
        Expanded bbox that includes sufficient reference geometry
    """
    if ref_line is None or ref_line.is_empty:
        return target_bbox

    # Check current visibility
    current_visible = _geometry_length_in_bbox_meters(ref_line, target_bbox)
    if current_visible >= min_visible_m:
        return target_bbox

    # Need to expand - calculate how much of reference we need to include
    # Strategy: progressively expand bbox toward reference until we have enough
    ref_bounds = ref_line.bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = target_bbox

    # Iteratively expand bbox toward reference (max 10 iterations)
    for _ in range(10):
        # Expand each edge toward reference bounds if reference extends beyond
        expansion_factor = 0.2  # Expand 20% toward reference each iteration

        if ref_bounds[0] < minx:  # Reference extends left
            minx = minx - (minx - ref_bounds[0]) * expansion_factor
        if ref_bounds[1] < miny:  # Reference extends down
            miny = miny - (miny - ref_bounds[1]) * expansion_factor
        if ref_bounds[2] > maxx:  # Reference extends right
            maxx = maxx + (ref_bounds[2] - maxx) * expansion_factor
        if ref_bounds[3] > maxy:  # Reference extends up
            maxy = maxy + (ref_bounds[3] - maxy) * expansion_factor

        expanded_bbox = (minx, miny, maxx, maxy)
        visible = _geometry_length_in_bbox_meters(ref_line, expanded_bbox)

        if visible >= min_visible_m:
            return expanded_bbox

    # If we still don't have enough after iterations, use union of both bounds
    # This ensures reference is fully visible as a fallback
    return (
        min(minx, ref_bounds[0]),
        min(miny, ref_bounds[1]),
        max(maxx, ref_bounds[2]),
        max(maxy, ref_bounds[3]),
    )


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


def _draw_decoration(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    decoration: str,
    color: tuple[int, int, int],
    size: int = 6,
):
    """Draw a decoration marker at a point.

    Args:
        draw: PIL ImageDraw object
        x, y: Center coordinates
        decoration: Type of decoration ('diamond', 'circle', 'square', 'triangle')
        color: RGB color tuple
        size: Size of the decoration in pixels
    """
    half = size // 2

    if decoration == "diamond":
        # Diamond shape
        points = [(x, y - half), (x + half, y), (x, y + half), (x - half, y)]
        draw.polygon(points, fill=color, outline=color)
    elif decoration == "circle":
        # Circle/dot
        draw.ellipse([(x - half, y - half), (x + half, y + half)], fill=color, outline=color)
    elif decoration == "square":
        # Square
        draw.rectangle([(x - half, y - half), (x + half, y + half)], fill=color, outline=color)
    elif decoration == "triangle":
        # Triangle pointing up
        points = [(x, y - half), (x + half, y + half), (x - half, y + half)]
        draw.polygon(points, fill=color, outline=color)


def _draw_linestring(
    draw: ImageDraw.ImageDraw,
    line: LineString,
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
    decoration: str | None = None,
    decoration_spacing: int = 30,
):
    """Draw a LineString on an image with optional decorations.

    Args:
        draw: PIL ImageDraw object
        line: Shapely LineString geometry
        bbox: Bounding box for coordinate transformation
        size: Image size (width, height)
        color: RGB color tuple
        width: Line width in pixels
        decoration: Type of decoration ('diamond', 'circle', 'square', 'triangle', or None)
        decoration_spacing: Pixels between decorations
    """
    if line is None or line.is_empty:
        return

    coords = list(line.coords)
    if len(coords) < 2:
        return

    # Convert to pixel coordinates
    pixel_coords = [_geo_to_pixel(lon, lat, bbox, size) for lon, lat in coords]

    # Draw the main line (solid)
    draw.line(pixel_coords, fill=color, width=width)

    # Draw decorations along the line if specified
    if decoration:
        # Calculate total line length and place decorations at intervals
        total_dist = 0
        next_decoration_at = decoration_spacing // 2  # Start offset from beginning

        for i in range(len(pixel_coords) - 1):
            x1, y1 = pixel_coords[i]
            x2, y2 = pixel_coords[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            segment_length = math.sqrt(dx * dx + dy * dy)

            if segment_length == 0:
                continue

            # Normalize direction
            ndx = dx / segment_length
            ndy = dy / segment_length

            # Place decorations along this segment
            segment_start = total_dist
            segment_end = total_dist + segment_length

            while next_decoration_at < segment_end:
                # Calculate position along this segment
                pos_in_segment = next_decoration_at - segment_start
                px = int(x1 + ndx * pos_in_segment)
                py = int(y1 + ndy * pos_in_segment)

                _draw_decoration(draw, px, py, decoration, color, size=width + 4)
                next_decoration_at += decoration_spacing

            total_dist += segment_length


def render_with_overlay(
    satellite: Image.Image,
    ref_geom: LineString | MultiLineString,
    target_geom: LineString | MultiLineString,
    bbox: tuple[float, float, float, float],
) -> Image.Image:
    """Render geometries overlaid on satellite imagery.

    Args:
        satellite: Background satellite image
        ref_geom: Reference geometry (drawn in blue, dashed)
        target_geom: Target geometry (drawn in red, solid)
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

    # Draw reference first (underneath, diamonds), then target (circles)
    # Different decorations and spacing make both visible even when overlapping
    if ref_line:
        _draw_linestring(
            draw,
            ref_line,
            bbox,
            size,
            REFERENCE_COLOR,
            OVERLAY_LINE_WIDTH,
            decoration="diamond",
            decoration_spacing=25,
        )
    if target_line:
        _draw_linestring(
            draw,
            target_line,
            bbox,
            size,
            TARGET_COLOR,
            OVERLAY_LINE_WIDTH,
            decoration="circle",
            decoration_spacing=40,
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

    # Draw reference first (underneath, diamonds), then target (circles)
    if ref_line:
        _draw_linestring(
            draw,
            ref_line,
            bbox,
            size,
            REFERENCE_COLOR,
            GEOMETRY_LINE_WIDTH,
            decoration="diamond",
            decoration_spacing=25,
        )
    if target_line:
        _draw_linestring(
            draw,
            target_line,
            bbox,
            size,
            TARGET_COLOR,
            GEOMETRY_LINE_WIDTH,
            decoration="circle",
            decoration_spacing=40,
        )

    return result


def render_candidate_images(
    ref_geom: LineString | MultiLineString,
    target_geom: LineString | MultiLineString,
    size: tuple[int, int] | None = None,
    padding_ratio: float = 0.3,
    fetch_satellite: bool = True,
) -> tuple[Image.Image | None, Image.Image, dict[str, tuple[int, int]]]:
    """Render both satellite overlay and geometry-only images for a candidate.

    When size is None (default), calculates dynamic size based on the final bbox.
    The bbox starts from target geometry with padding, then expands to ensure
    at least MIN_REFERENCE_VISIBLE_M meters of reference geometry are visible.

    Args:
        ref_geom: Reference geometry
        target_geom: Target geometry
        size: Output image size, or None for dynamic sizing
        padding_ratio: Padding around target geometry as ratio of extent
        fetch_satellite: Whether to fetch satellite imagery

    Returns:
        Tuple of (satellite_with_overlay, geometry_only, sizes_dict).
        satellite_with_overlay may be None if fetch fails.
        sizes_dict contains {"geometry": (w, h), "satellite": (w, h)} sizes.
    """
    # Convert to LineString
    ref_line = _to_linestring(ref_geom)
    target_line = _to_linestring(target_geom)

    if not ref_line or not target_line:
        logger.warning("Invalid geometries, returning empty images")
        if size is None:
            size = (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE)
        empty = Image.new("RGB", size, BACKGROUND_COLOR)
        return None, empty, {"geometry": size, "satellite": size}

    # Get bbox based on TARGET geometry with padding
    target_bbox = _expand_bbox(target_line.bounds, padding_ratio)

    # Expand bbox to ensure minimum reference visibility
    bbox = _expand_bbox_for_reference(target_bbox, ref_line, MIN_REFERENCE_VISIBLE_M)

    # Make bbox square to avoid image stretching
    bbox = _make_bbox_square(bbox)

    # Calculate dynamic size based on FINAL bbox (after reference expansion)
    if size is None:
        size = _calculate_size_from_bbox(bbox)

    # Satellite size is 2x geometry size (for better detail)
    sat_dim = min(size[0] * SATELLITE_SIZE_MULTIPLIER, 1024)  # Cap at 1024
    satellite_size = (sat_dim, sat_dim)

    # Render geometry-only (using target-based bbox)
    geometry_img = _render_geometry_with_bbox(ref_geom, target_geom, bbox, size)

    # Fetch and render satellite overlay at higher resolution
    satellite_img = None
    if fetch_satellite:
        satellite = fetch_satellite_tile(bbox, size=satellite_size)
        if satellite:
            satellite_img = render_with_overlay(satellite, ref_geom, target_geom, bbox)

    return satellite_img, geometry_img, {"geometry": size, "satellite": satellite_size}


def _render_geometry_with_bbox(
    ref_geom: LineString | MultiLineString,
    target_geom: LineString | MultiLineString,
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
) -> Image.Image:
    """Render geometry-only image with explicit bbox.

    Internal helper for render_candidate_images() to use target-based bbox.
    """
    ref_line = _to_linestring(ref_geom)
    target_line = _to_linestring(target_geom)

    # Create white background
    result = Image.new("RGB", size, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(result)

    # Draw reference first (underneath, diamonds), then target (circles)
    if ref_line:
        _draw_linestring(
            draw,
            ref_line,
            bbox,
            size,
            REFERENCE_COLOR,
            GEOMETRY_LINE_WIDTH,
            decoration="diamond",
            decoration_spacing=25,
        )
    if target_line:
        _draw_linestring(
            draw,
            target_line,
            bbox,
            size,
            TARGET_COLOR,
            GEOMETRY_LINE_WIDTH,
            decoration="circle",
            decoration_spacing=40,
        )

    return result
