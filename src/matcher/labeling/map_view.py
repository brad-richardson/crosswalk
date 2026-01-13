"""Map visualization component using folium."""

from typing import Optional

import folium
from shapely.geometry import LineString, MultiLineString, mapping

from .data_loader import CandidatePairView


# Colors for visualization
REFERENCE_COLOR = "#2196F3"  # Blue
TARGET_COLOR = "#F44336"  # Red
REFERENCE_WEIGHT = 5
TARGET_WEIGHT = 4


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
        folium.TileLayer(tiles=tiles, max_zoom=20).add_to(m)


def create_comparison_map(
    pair: CandidatePairView,
    height: int = 500,
    tile_layer: str = "Light",
) -> folium.Map:
    """Create a folium map showing reference and target geometries.

    Args:
        pair: The candidate pair to display
        height: Map height in pixels

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
    )

    # Add selected tile layer
    _add_tile_layer(m, tile_layer)

    # Fit bounds
    m.fit_bounds([
        [miny - pady, minx - padx],
        [maxy + pady, maxx + padx],
    ])

    # Add reference geometry (blue, solid)
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

    # Add target geometry (red, dashed)
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

    # Add legend
    _add_legend(m)

    return m


def _add_geometry_layer(
    m: folium.Map,
    geometry,
    color: str,
    weight: int,
    dash_array: Optional[str],
    popup: str,
) -> None:
    """Add a geometry to the map."""
    style = {
        "color": color,
        "weight": weight,
        "opacity": 0.9,
    }
    if dash_array:
        style["dashArray"] = dash_array

    if isinstance(geometry, (LineString, MultiLineString)):
        geojson = mapping(geometry)
        folium.GeoJson(
            geojson,
            style_function=lambda x, s=style: s,
            popup=popup,
        ).add_to(m)


def _create_popup(
    label: str,
    segment_id: str,
    name: Optional[str],
    road_class: Optional[str],
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


def create_multi_reference_map(
    target_geometry,
    target_name: Optional[str],
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
    )

    # Add selected tile layer
    _add_tile_layer(m, tile_layer)

    m.fit_bounds([
        [miny - pady, minx - padx],
        [maxy + pady, maxx + padx],
    ])

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
