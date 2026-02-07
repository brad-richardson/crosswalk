"""Map rendering for integration QA app."""

import folium
import geopandas as gpd
from shapely.geometry import LineString

# Color scheme for different edge sources
SOURCE_COLORS = {
    "reference": "#3388ff",  # Blue
    "target_matched": "#28a745",  # Green
    "target_new": "#fd7e14",  # Orange - connected but unmatched
    "disconnected": "#dc3545",  # Red - truly disconnected
    "filtered": "#6c757d",  # Gray - connected but filtered
    "net_new": "#00ffff",  # Cyan - net new coverage portions
}

PRIORITY_COLORS = {
    "high": "#dc3545",  # Red
    "medium": "#fd7e14",  # Orange
    "low": "#ffc107",  # Yellow
}

# Selected edge highlight color
SELECTED_COLOR = "#ff00ff"  # Magenta
SELECTED_WEIGHT = 5  # Visible but not too thick

# Layer names for map display
LAYER_NAMES = {
    "reference": "Reference (Overture)",
    "target_matched": "Matched",
    "target_new": "To Merge (Connected)",
    "disconnected": "Disconnected",
    "filtered": "Filtered (Short Net-New)",
    "net_new": "Net New Coverage",
}


def _add_circle_markers_along_line(
    layer: folium.FeatureGroup,
    geom: LineString,
    color: str,
    spacing_m: float = 20.0,
    radius: int = 4,
) -> None:
    """Add circle markers along a LineString at regular intervals.

    Args:
        layer: Folium FeatureGroup to add markers to
        geom: LineString geometry (in WGS84)
        color: Marker color
        spacing_m: Spacing between markers in meters (approximate)
        radius: Marker radius in pixels
    """
    if geom is None or geom.is_empty:
        return

    # Convert spacing from meters to approximate degrees
    # At typical latitudes, 1 degree ~ 111km, so spacing_m meters ~ spacing_m/111000 degrees
    spacing_deg = spacing_m / 111000.0

    # Get total length in degrees (approximate)
    total_length = geom.length
    if total_length == 0:
        return

    # Place markers at intervals
    num_markers = max(2, int(total_length / spacing_deg))
    for i in range(num_markers + 1):
        fraction = i / num_markers
        point = geom.interpolate(fraction, normalized=True)
        folium.CircleMarker(
            location=[point.y, point.x],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=1.0,
            weight=1,
        ).add_to(layer)


def create_base_map(
    center_lat: float = 42.36,
    center_lon: float = -71.06,
    zoom: int = 14,
    default_tiles: str = "satellite",
) -> folium.Map:
    """Create base folium map with multiple tile layer options.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        zoom: Initial zoom level
        default_tiles: Default tile layer ("satellite" or "light")
    """
    # Create map without default tiles (we'll add them manually)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
    )

    # Add satellite imagery (Esri World Imagery)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        max_zoom=20,
        show=(default_tiles == "satellite"),
    ).add_to(m)

    # Add light basemap (CartoDB Positron)
    folium.TileLayer(
        tiles="cartodbpositron",
        name="Light",
        max_zoom=20,
        show=(default_tiles == "light"),
    ).add_to(m)

    # Add OpenStreetMap
    folium.TileLayer(
        tiles="openstreetmap",
        name="OpenStreetMap",
        max_zoom=19,
        show=(default_tiles == "osm"),
    ).add_to(m)

    return m


def add_edges_layer(
    m: folium.Map,
    edges: gpd.GeoDataFrame,
    layer_name: str,
    color: str,
    weight: float = 2,
    opacity: float = 0.8,
    show: bool = True,
    add_markers: bool = False,
    marker_spacing: float = 30.0,
) -> folium.Map:
    """Add edges layer to map.

    Args:
        m: Folium map
        edges: GeoDataFrame of edges
        layer_name: Name for the layer
        color: Line color
        weight: Line weight
        opacity: Line opacity
        show: Whether layer is visible by default
        add_markers: Whether to add circle markers along lines
        marker_spacing: Spacing between markers in meters (if add_markers=True)
    """
    if edges is None or len(edges) == 0:
        return m

    # Ensure WGS84 for Folium
    if edges.crs and edges.crs.to_epsg() != 4326:
        edges = edges.to_crs("EPSG:4326")

    layer = folium.FeatureGroup(name=layer_name, show=show)

    for _, row in edges.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # Build popup content
        popup_html = f"<b>Edge ID:</b> {row.get('edge_id', row.get('_original_id', 'N/A'))}<br>"
        popup_html += f"<b>Source:</b> {row.get('_source', 'N/A')}<br>"
        popup_html += f"<b>Original ID:</b> {row.get('_original_id', 'N/A')}<br>"
        popup_html += f"<b>Dataset:</b> {row.get('_source_dataset', 'N/A')}<br>"

        if "_match_ref_id" in row and row["_match_ref_id"]:
            popup_html += f"<b>Match Ref:</b> {row['_match_ref_id']}<br>"
        if "_match_confidence" in row and row["_match_confidence"]:
            popup_html += f"<b>Confidence:</b> {row['_match_confidence']:.2f}<br>"
        if "component_size" in row:
            popup_html += f"<b>Component Size:</b> {row['component_size']}<br>"

        # Add geometry
        if geom.geom_type == "LineString":
            coords = [[p[1], p[0]] for p in geom.coords]
            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                opacity=opacity,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(layer)

            # Add circle markers along line if requested
            if add_markers:
                _add_circle_markers_along_line(layer, geom, color, marker_spacing, radius=3)

    layer.add_to(m)
    return m


def add_disconnected_layers(
    m: folium.Map,
    disconnected_edges: gpd.GeoDataFrame,
    by_priority: bool = True,
) -> folium.Map:
    """Add disconnected edges to map, optionally grouped by priority.

    Disconnected edges get circle markers to distinguish them from reference edges.
    """
    if disconnected_edges is None or len(disconnected_edges) == 0:
        return m

    # Ensure WGS84 for Folium
    if disconnected_edges.crs and disconnected_edges.crs.to_epsg() != 4326:
        disconnected_edges = disconnected_edges.to_crs("EPSG:4326")

    if by_priority and "qa_priority" in disconnected_edges.columns:
        for priority in ["high", "medium", "low"]:
            priority_edges = disconnected_edges[disconnected_edges["qa_priority"] == priority]
            if len(priority_edges) > 0:
                add_edges_layer(
                    m,
                    priority_edges,
                    f"Disconnected ({priority.title()})",
                    PRIORITY_COLORS[priority],
                    weight=3,
                    add_markers=True,
                    marker_spacing=30.0,
                )
    else:
        add_edges_layer(
            m,
            disconnected_edges,
            LAYER_NAMES["disconnected"],
            SOURCE_COLORS["disconnected"],
            weight=3,
            add_markers=True,
            marker_spacing=30.0,
        )

    return m


def highlight_edge(
    m: folium.Map,
    edge: gpd.GeoSeries,
    color: str = SELECTED_COLOR,
    weight: float = SELECTED_WEIGHT,
) -> folium.Map:
    """Highlight a specific edge on the map with thick line and markers.

    The highlighted edge is drawn on a separate layer that's added last
    to ensure it appears on top of other layers.
    """
    geom = edge.geometry
    if geom is None:
        return m

    # Ensure WGS84
    if hasattr(edge, "crs") and edge.crs and edge.crs.to_epsg() != 4326:
        edge = edge.to_crs("EPSG:4326")
        geom = edge.geometry

    # Create a dedicated layer for the highlighted edge (added last = on top)
    highlight_layer = folium.FeatureGroup(name="Selected Edge", show=True)

    if geom.geom_type == "LineString":
        coords = [[p[1], p[0]] for p in geom.coords]

        # Draw a white outline first for contrast
        folium.PolyLine(
            coords,
            color="white",
            weight=weight + 4,
            opacity=1.0,
        ).add_to(highlight_layer)

        # Draw the colored line on top
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=1.0,
        ).add_to(highlight_layer)

        # Add circle markers along the line for extra visibility
        _add_circle_markers_along_line(highlight_layer, geom, color, spacing_m=20.0, radius=4)

        # Add start and end markers
        start = geom.coords[0]
        end = geom.coords[-1]
        folium.CircleMarker(
            location=[start[1], start[0]],
            radius=7,
            color="white",
            fill=True,
            fillColor=color,
            fillOpacity=1.0,
            weight=2,
            popup="Start",
        ).add_to(highlight_layer)
        folium.CircleMarker(
            location=[end[1], end[0]],
            radius=7,
            color="white",
            fill=True,
            fillColor=color,
            fillOpacity=1.0,
            weight=2,
            popup="End",
        ).add_to(highlight_layer)

    highlight_layer.add_to(m)
    return m


def fit_bounds(m: folium.Map, gdf: gpd.GeoDataFrame) -> folium.Map:
    """Fit map bounds to GeoDataFrame extent."""
    if gdf is None or len(gdf) == 0:
        return m

    # Ensure WGS84
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    bounds = gdf.total_bounds
    m.fit_bounds(
        [
            [bounds[1], bounds[0]],  # SW corner
            [bounds[3], bounds[2]],  # NE corner
        ]
    )
    return m


def create_integration_map(
    edges: gpd.GeoDataFrame,
    net_new_edges: gpd.GeoDataFrame | None = None,
    selected_edge_id: int | None = None,
    focus_on_selected: bool = True,
    context_radius: float = 500.0,  # meters around selected edge
    disconnected_edges: gpd.GeoDataFrame | None = None,
    filtered_edges: gpd.GeoDataFrame | None = None,
) -> folium.Map:
    """Create full integration QA map with all layers."""
    # Default center
    center_lat, center_lon = 42.36, -71.06
    zoom = 14

    # Find selected edge for centering
    selected_edge = None
    if selected_edge_id is not None:

        def find_edge(gdf, edge_id):
            if gdf is None or len(gdf) == 0:
                return None
            id_col = "edge_id" if "edge_id" in gdf.columns else "_original_id"
            if id_col in gdf.columns and edge_id in gdf[id_col].values:
                return gdf[gdf[id_col] == edge_id].iloc[0]
            return None

        selected_edge = find_edge(disconnected_edges, selected_edge_id)
        if selected_edge is None:
            selected_edge = find_edge(filtered_edges, selected_edge_id)
        if selected_edge is None:
            selected_edge = find_edge(edges, selected_edge_id)

    # Center on selected edge if found
    if selected_edge is not None and selected_edge.geometry is not None:
        geom = selected_edge.geometry
        centroid = geom.centroid
        center_lat, center_lon = centroid.y, centroid.x
        zoom = 17  # Zoom in on selected edge
    elif edges is not None and len(edges) > 0:
        # Fallback: center on non-reference edges if any
        non_ref = edges[edges["_source"] != "reference"] if "_source" in edges.columns else edges
        if len(non_ref) > 0:
            if non_ref.crs and non_ref.crs.to_epsg() != 4326:
                non_ref = non_ref.to_crs("EPSG:4326")
            centroid = non_ref.union_all().centroid
            center_lat, center_lon = centroid.y, centroid.x

    m = create_base_map(center_lat, center_lon, zoom)

    # Add layers by source type (reference first, then targets, selected last)
    if edges is not None and len(edges) > 0 and "_source" in edges.columns:
        # Ensure WGS84 for filtering
        working_edges = edges
        if working_edges.crs and working_edges.crs.to_epsg() != 4326:
            working_edges = working_edges.to_crs("EPSG:4326")

        # Filter reference edges to only those near the view center (for performance)
        ref_edges = working_edges[working_edges["_source"] == "reference"]
        if len(ref_edges) > 1000 and selected_edge is not None:
            # Only show reference edges within ~0.01 degrees (~1km) of selected
            from shapely.geometry import box

            cx, cy = center_lon, center_lat
            bbox = box(cx - 0.01, cy - 0.01, cx + 0.01, cy + 0.01)
            ref_edges = ref_edges[ref_edges.geometry.intersects(bbox)]
        # Reference edges: no markers (too many, would clutter)
        add_edges_layer(m, ref_edges, LAYER_NAMES["reference"], SOURCE_COLORS["reference"])

        # Matched target edges: add markers to distinguish from reference
        matched_edges = working_edges[working_edges["_source"] == "target_matched"]
        add_edges_layer(
            m,
            matched_edges,
            LAYER_NAMES["target_matched"],
            SOURCE_COLORS["target_matched"],
            weight=2,
            add_markers=True,
            marker_spacing=40.0,
        )

        # Unmatched but connected target edges (to be merged into network)
        new_edges = working_edges[working_edges["_source"] == "target_new"]
        add_edges_layer(
            m,
            new_edges,
            LAYER_NAMES["target_new"],
            SOURCE_COLORS["target_new"],
            weight=3,
            add_markers=True,
            marker_spacing=35.0,
        )

    # Add net-new coverage layer (shows just the new portions)
    if net_new_edges is not None and len(net_new_edges) > 0:
        add_edges_layer(
            m,
            net_new_edges,
            LAYER_NAMES["net_new"],
            SOURCE_COLORS["net_new"],
            weight=4,
            opacity=0.9,
            add_markers=True,
            marker_spacing=15.0,
        )

    # Add disconnected layers (with markers, priority-colored)
    add_disconnected_layers(m, disconnected_edges)

    # Add filtered layer (single gray layer)
    if filtered_edges is not None and len(filtered_edges) > 0:
        add_edges_layer(
            m,
            filtered_edges,
            LAYER_NAMES["filtered"],
            SOURCE_COLORS["filtered"],
            weight=2,
            add_markers=True,
            marker_spacing=30.0,
        )

    # Highlight selected edge LAST so it's on top
    if selected_edge is not None:
        highlight_edge(m, selected_edge)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m
