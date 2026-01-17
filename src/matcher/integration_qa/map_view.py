"""Map rendering for integration QA app."""

from typing import Optional

import folium
import geopandas as gpd
from folium.plugins import Draw


# Color scheme for different edge sources
SOURCE_COLORS = {
    "reference": "#3388ff",  # Blue
    "target_matched": "#28a745",  # Green
    "target_new": "#fd7e14",  # Orange
    "orphan": "#dc3545",  # Red
}

PRIORITY_COLORS = {
    "high": "#dc3545",  # Red
    "medium": "#fd7e14",  # Orange
    "low": "#ffc107",  # Yellow
}


def create_base_map(
    center_lat: float = 42.36,
    center_lon: float = -71.06,
    zoom: int = 14,
) -> folium.Map:
    """Create base folium map."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="cartodbpositron",
    )
    return m


def add_edges_layer(
    m: folium.Map,
    edges: gpd.GeoDataFrame,
    layer_name: str,
    color: str,
    weight: float = 2,
    opacity: float = 0.8,
    show: bool = True,
) -> folium.Map:
    """Add edges layer to map."""
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

    layer.add_to(m)
    return m


def add_orphan_layers(
    m: folium.Map,
    orphan_edges: gpd.GeoDataFrame,
    by_priority: bool = True,
) -> folium.Map:
    """Add orphan edges to map, optionally grouped by priority."""
    if orphan_edges is None or len(orphan_edges) == 0:
        return m

    # Ensure WGS84 for Folium
    if orphan_edges.crs and orphan_edges.crs.to_epsg() != 4326:
        orphan_edges = orphan_edges.to_crs("EPSG:4326")

    if by_priority and "qa_priority" in orphan_edges.columns:
        for priority in ["high", "medium", "low"]:
            priority_edges = orphan_edges[orphan_edges["qa_priority"] == priority]
            if len(priority_edges) > 0:
                add_edges_layer(
                    m,
                    priority_edges,
                    f"Orphans ({priority})",
                    PRIORITY_COLORS[priority],
                    weight=3,
                )
    else:
        add_edges_layer(
            m,
            orphan_edges,
            "Orphans",
            SOURCE_COLORS["orphan"],
            weight=3,
        )

    return m


def highlight_edge(
    m: folium.Map,
    edge: gpd.GeoSeries,
    color: str = "#ff00ff",
    weight: float = 5,
) -> folium.Map:
    """Highlight a specific edge on the map."""
    geom = edge.geometry
    if geom is None:
        return m

    # Ensure WGS84
    if hasattr(edge, "crs") and edge.crs and edge.crs.to_epsg() != 4326:
        edge = edge.to_crs("EPSG:4326")
        geom = edge.geometry

    if geom.geom_type == "LineString":
        coords = [[p[1], p[0]] for p in geom.coords]
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=1.0,
        ).add_to(m)

    return m


def fit_bounds(m: folium.Map, gdf: gpd.GeoDataFrame) -> folium.Map:
    """Fit map bounds to GeoDataFrame extent."""
    if gdf is None or len(gdf) == 0:
        return m

    # Ensure WGS84
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    bounds = gdf.total_bounds
    m.fit_bounds([
        [bounds[1], bounds[0]],  # SW corner
        [bounds[3], bounds[2]],  # NE corner
    ])
    return m


def create_integration_map(
    edges: gpd.GeoDataFrame,
    orphan_edges: gpd.GeoDataFrame,
    selected_edge_id: Optional[int] = None,
    focus_on_selected: bool = True,
    context_radius: float = 500.0,  # meters around selected edge
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

        selected_edge = find_edge(orphan_edges, selected_edge_id)
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

    # Add layers by source type
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
        add_edges_layer(m, ref_edges, "Reference (Overture)", SOURCE_COLORS["reference"])

        # Matched target edges
        matched_edges = working_edges[working_edges["_source"] == "target_matched"]
        add_edges_layer(m, matched_edges, "Target (Matched)", SOURCE_COLORS["target_matched"])

        # Unmatched target edges (in main network) - always show all
        new_edges = working_edges[working_edges["_source"] == "target_new"]
        add_edges_layer(m, new_edges, "Target (New)", SOURCE_COLORS["target_new"], weight=4)

    # Add orphan layers
    add_orphan_layers(m, orphan_edges)

    # Highlight selected edge (already found above)
    if selected_edge is not None:
        highlight_edge(m, selected_edge)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m
