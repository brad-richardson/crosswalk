"""Map visualization component using folium."""

import folium
from shapely.geometry import LineString, Point, mapping

from ..config import ALIGNMENT_FULL_TOLERANCE
from .data_loader import CandidatePairView

# Colors for visualization - aligned portions are bright, full geometries are faded
REFERENCE_COLOR = "#2196F3"  # Blue (aligned/matched portion)
REFERENCE_FADED_COLOR = "#90CAF9"  # Light blue (full geometry context)
TARGET_COLOR = "#F44336"  # Red (aligned/matched portion)
TARGET_FADED_COLOR = "#FFCDD2"  # Light red (full geometry context)
REFERENCE_WEIGHT = 5
TARGET_WEIGHT = 4

# Dot marker percentages along line
DOT_PERCENTAGES = [0.0, 0.5, 1.0]


TILE_LAYERS = {
    "Light": "cartodbpositron",
    "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "OpenStreetMap": "openstreetmap",
}


def _add_tile_layer(m: folium.Map, layer_name: str = "Light") -> None:
    """Add a single tile layer to the map."""
    tiles = TILE_LAYERS.get(layer_name, "cartodbpositron")
    if layer_name == "Satellite":
        folium.TileLayer(
            tiles=tiles,
            attr="Esri",
            max_zoom=21,
            name=layer_name,
        ).add_to(m)
    else:
        folium.TileLayer(tiles=tiles, max_zoom=21).add_to(m)


def _add_line_markers(
    m: folium.Map,
    geometry: LineString,
    color: str,
    percentages: list[float] = DOT_PERCENTAGES,
) -> None:
    """Add dot markers at specified percentages along a line.

    Args:
        m: Folium map to add markers to
        geometry: LineString geometry
        color: Color for the markers
        percentages: List of percentages (0.0-1.0) where to place markers
    """
    if not isinstance(geometry, LineString) or geometry.is_empty:
        return

    for pct in percentages:
        # Interpolate point along line
        point = geometry.interpolate(pct, normalized=True)
        label = f"{int(pct * 100)}%"

        folium.CircleMarker(
            location=[point.y, point.x],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=2,
            popup=label,
        ).add_to(m)


def create_comparison_map(
    pair: CandidatePairView,
    tile_layer: str = "Light",
) -> folium.Map:
    """Create a folium map showing reference and target geometries with alignment.

    Shows both the full geometries (faded, for context) and the aligned/matched
    portions (bright, solid) to help labelers understand which parts of each
    segment correspond to each other.

    Args:
        pair: The candidate pair to display
        tile_layer: Map tile layer name

    Returns:
        Configured folium Map object
    """
    # Calculate bounds for auto-zoom
    ref_bounds = pair.ref_geometry.bounds
    target_bounds = pair.target_geometry.bounds

    minx = min(ref_bounds[0], target_bounds[0])
    miny = min(ref_bounds[1], target_bounds[1])
    maxx = max(ref_bounds[2], target_bounds[2])
    maxy = max(ref_bounds[3], target_bounds[3])

    # Add padding (10%)
    padx = (maxx - minx) * 0.1
    pady = (maxy - miny) * 0.1

    # Create map centered on the bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        tiles=None,
        zoom_start=16,
        max_zoom=21,
    )

    # Add selected tile layer
    _add_tile_layer(m, tile_layer)

    # Fit bounds
    m.fit_bounds(
        [
            [miny - pady, minx - padx],
            [maxy + pady, maxx + padx],
        ]
    )

    # Check if we have aligned geometries (non-trivial alignment)
    has_alignment = (
        pair.ref_aligned_geometry is not None
        and pair.target_aligned_geometry is not None
        and not pair.ref_aligned_geometry.is_empty
        and not pair.target_aligned_geometry.is_empty
    )

    # Check if alignment is partial (not the full segment)
    # Use centralized tolerance for consistency with label metadata
    tol = ALIGNMENT_FULL_TOLERANCE
    is_partial_alignment = has_alignment and (
        pair.ref_start_frac > tol
        or pair.ref_end_frac < (1.0 - tol)
        or pair.target_start_frac > tol
        or pair.target_end_frac < (1.0 - tol)
    )

    if has_alignment and is_partial_alignment:
        # Show full geometries faded (for context)
        ref_full_popup = _create_popup(
            "Reference - Full (context)",
            pair.ref_id,
            pair.ref_name,
            pair.ref_class,
        )
        _add_geometry_layer(
            m,
            pair.ref_geometry,
            color=REFERENCE_FADED_COLOR,
            weight=REFERENCE_WEIGHT - 2,
            dash_array="5, 5",
            popup=ref_full_popup,
            opacity=0.5,
        )

        target_full_popup = _create_popup(
            "Target - Full (context)",
            pair.target_id,
            pair.target_name,
            pair.target_class,
        )
        _add_geometry_layer(
            m,
            pair.target_geometry,
            color=TARGET_FADED_COLOR,
            weight=TARGET_WEIGHT - 2,
            dash_array="5, 5",
            popup=target_full_popup,
            opacity=0.5,
        )

        # Show aligned portions bright (the actual match)
        ref_pct = f"{pair.ref_start_frac * 100:.0f}%-{pair.ref_end_frac * 100:.0f}%"
        ref_aligned_popup = _create_popup(
            f"Reference - Aligned ({ref_pct})",
            pair.ref_id,
            pair.ref_name,
            pair.ref_class,
        )
        _add_geometry_layer(
            m,
            pair.ref_aligned_geometry,
            color=REFERENCE_COLOR,
            weight=REFERENCE_WEIGHT,
            dash_array=None,
            popup=ref_aligned_popup,
        )

        target_pct = f"{pair.target_start_frac * 100:.0f}%-{pair.target_end_frac * 100:.0f}%"
        target_aligned_popup = _create_popup(
            f"Target - Aligned ({target_pct})",
            pair.target_id,
            pair.target_name,
            pair.target_class,
        )
        _add_geometry_layer(
            m,
            pair.target_aligned_geometry,
            color=TARGET_COLOR,
            weight=TARGET_WEIGHT,
            dash_array=None,
            popup=target_aligned_popup,
        )

        # Add dot markers at 0%, 50%, 100% for aligned portions
        _add_line_markers(m, pair.ref_aligned_geometry, REFERENCE_COLOR)
        _add_line_markers(m, pair.target_aligned_geometry, TARGET_COLOR)

        # Add alignment legend
        _add_alignment_legend(m)
    else:
        # No partial alignment - show full geometries normally
        ref_popup = _create_popup(
            "Reference (Overture)",
            pair.ref_id,
            pair.ref_name,
            pair.ref_class,
        )
        _add_geometry_layer(
            m,
            pair.ref_geometry,
            color=REFERENCE_COLOR,
            weight=REFERENCE_WEIGHT,
            dash_array=None,
            popup=ref_popup,
        )

        target_popup = _create_popup(
            "Target (Local)",
            pair.target_id,
            pair.target_name,
            pair.target_class,
        )
        _add_geometry_layer(
            m,
            pair.target_geometry,
            color=TARGET_COLOR,
            weight=TARGET_WEIGHT,
            dash_array="10, 5",
            popup=target_popup,
        )

        # Add dot markers at 0%, 50%, 100% for full geometries
        _add_line_markers(m, pair.ref_geometry, REFERENCE_COLOR)
        _add_line_markers(m, pair.target_geometry, TARGET_COLOR)

        # Add simple legend
        _add_legend(m)

    return m


def _add_geometry_layer(
    m: folium.Map,
    geometry,
    color: str,
    weight: int,
    dash_array: str | None,
    popup: str,
    opacity: float = 0.9,
) -> None:
    """Add a geometry to the map."""
    style = {
        "color": color,
        "weight": weight,
        "opacity": opacity,
    }
    if dash_array:
        style["dashArray"] = dash_array

    if isinstance(geometry, LineString):
        geojson = mapping(geometry)
        folium.GeoJson(
            geojson,
            style_function=lambda x, s=style: s,
            popup=popup,
        ).add_to(m)


def _create_popup(
    label: str,
    segment_id: str,
    name: str | None,
    road_class: str | None,
) -> str:
    """Create HTML popup content."""
    lines = [f"<b>{label}</b>"]
    lines.append(f"ID: {segment_id}")
    if name:
        lines.append(f"Name: {name}")
    if road_class:
        lines.append(f"Class: {road_class}")
    return "<br>".join(lines)


def _add_legend(m: folium.Map) -> None:
    """Add a legend to the map."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 10px;
        z-index: 1000;
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
        font-family: Arial, sans-serif;
        font-size: 12px;
    ">
        <div style="margin-bottom: 5px;">
            <span style="
                display: inline-block;
                width: 30px;
                height: 3px;
                background: #2196F3;
                vertical-align: middle;
            "></span>
            <span style="vertical-align: middle; margin-left: 5px;">Reference (Overture)</span>
        </div>
        <div>
            <span style="
                display: inline-block;
                width: 30px;
                height: 3px;
                background: #F44336;
                border-style: dashed;
                vertical-align: middle;
            "></span>
            <span style="vertical-align: middle; margin-left: 5px;">Target (Local)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def _add_alignment_legend(m: folium.Map) -> None:
    """Add a legend showing aligned vs full geometry styles."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 10px;
        z-index: 1000;
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
        font-family: Arial, sans-serif;
        font-size: 12px;
    ">
        <div style="font-weight: bold; margin-bottom: 5px;">Alignment View</div>
        <div style="margin-bottom: 3px;">
            <span style="display: inline-block; width: 30px; height: 4px; background: #2196F3; vertical-align: middle;"></span>
            <span style="vertical-align: middle; margin-left: 5px;">Reference (aligned)</span>
        </div>
        <div style="margin-bottom: 3px;">
            <span style="display: inline-block; width: 30px; height: 2px; background: #90CAF9; border-style: dashed; vertical-align: middle;"></span>
            <span style="vertical-align: middle; margin-left: 5px;">Reference (full)</span>
        </div>
        <div style="margin-bottom: 3px;">
            <span style="display: inline-block; width: 30px; height: 4px; background: #F44336; vertical-align: middle;"></span>
            <span style="vertical-align: middle; margin-left: 5px;">Target (aligned)</span>
        </div>
        <div>
            <span style="display: inline-block; width: 30px; height: 2px; background: #FFCDD2; border-style: dashed; vertical-align: middle;"></span>
            <span style="vertical-align: middle; margin-left: 5px;">Target (full)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def create_multi_reference_map(
    target_geometry,
    target_name: str | None,
    related_candidates: list,
    selected_refs: set,
    height: int = 500,
    tile_layer: str = "Light",
) -> folium.Map:
    """Create a map showing one target with multiple reference candidates.

    Args:
        target_geometry: The target geometry (local segment)
        target_name: Name of the target segment
        related_candidates: List of CandidatePairView objects for this target
        selected_refs: Set of ref_ids that are currently selected
        height: Map height in pixels

    Returns:
        Configured folium Map object
    """
    # Calculate bounds from all geometries
    all_bounds = [target_geometry.bounds]
    for cand in related_candidates:
        all_bounds.append(cand.ref_geometry.bounds)

    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)

    # Add padding
    padx = (maxx - minx) * 0.1
    pady = (maxy - miny) * 0.1

    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        tiles=None,
        zoom_start=16,
        max_zoom=21,
    )

    # Add selected tile layer
    _add_tile_layer(m, tile_layer)

    m.fit_bounds(
        [
            [miny - pady, minx - padx],
            [maxy + pady, maxx + padx],
        ]
    )

    # Add target geometry (red, thick, solid)
    target_popup = _create_popup("Target (Local)", "target", target_name, None)
    _add_geometry_layer(
        m,
        target_geometry,
        color=TARGET_COLOR,
        weight=6,
        dash_array=None,
        popup=target_popup,
    )

    # Add reference geometries
    # Selected = green, unselected = blue/light
    for cand in related_candidates:
        is_selected = cand.ref_id in selected_refs
        color = "#4CAF50" if is_selected else "#90CAF9"  # Green if selected, light blue otherwise
        weight = 5 if is_selected else 3
        opacity = 0.9 if is_selected else 0.5

        popup = _create_popup(
            f"Reference {'(SELECTED)' if is_selected else ''}",
            cand.ref_id,
            cand.ref_name,
            cand.ref_class,
        )
        popup += f"<br>Confidence: {cand.confidence:.0%}"

        geojson = mapping(cand.ref_geometry)
        folium.GeoJson(
            geojson,
            style_function=lambda x, c=color, w=weight, o=opacity: {
                "color": c,
                "weight": w,
                "opacity": o,
            },
            popup=popup,
        ).add_to(m)

    # Add legend for 1:N mode
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 10px;
        z-index: 1000;
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
        font-family: Arial, sans-serif;
        font-size: 12px;
    ">
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block; width: 30px; height: 3px; background: #4CAF50; vertical-align: middle;"></span>
            <span style="vertical-align: middle; margin-left: 5px;">Selected Reference</span>
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block; width: 30px; height: 3px; background: #90CAF9; vertical-align: middle;"></span>
            <span style="vertical-align: middle; margin-left: 5px;">Unselected Reference</span>
        </div>
        <div>
            <span style="display: inline-block; width: 30px; height: 3px; background: #F44336; vertical-align: middle;"></span>
            <span style="vertical-align: middle; margin-left: 5px;">Target (Local)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
