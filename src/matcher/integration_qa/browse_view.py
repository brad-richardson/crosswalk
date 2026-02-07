"""Browse map view for integration QA with heatmap overlay."""

import logging

import folium
import geopandas as gpd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from matcher.integration_qa.map_view import (
    PRIORITY_COLORS,
    SOURCE_COLORS,
    add_edges_layer,
    fit_bounds,
)

logger = logging.getLogger(__name__)

# Maximum features to render per layer before sampling
MAX_BROWSE_FEATURES = 2000

# Basemap name to tile argument mapping
_BASEMAP_TILES = {
    "Light": "light",
    "Satellite": "satellite",
    "OpenStreetMap": "osm",
}


def render_browse_view(
    edges: gpd.GeoDataFrame,
    net_new_edges: gpd.GeoDataFrame | None = None,
    basemap: str = "Light",
    disconnected_edges: gpd.GeoDataFrame | None = None,
    filtered_edges: gpd.GeoDataFrame | None = None,
    bridge_edges: gpd.GeoDataFrame | None = None,
) -> None:
    """Render the browse map view with all layers toggled via Folium LayerControl.

    All layers are always added to the map. Toggling is handled client-side
    by Folium's LayerControl, avoiding server round-trips.

    Args:
        edges: All edges GeoDataFrame (with _source column)
        net_new_edges: Net-new coverage edges GeoDataFrame
        basemap: Default basemap name ("Light", "Satellite", "OpenStreetMap")
        disconnected_edges: Truly disconnected edges GeoDataFrame
        filtered_edges: Connected but filtered edges GeoDataFrame
    """
    # Show sampling notice if disconnected count exceeds browse limit
    if disconnected_edges is not None and len(disconnected_edges) > MAX_BROWSE_FEATURES:
        st.caption(
            f"Showing {MAX_BROWSE_FEATURES:,} of {len(disconnected_edges):,} disconnected edges "
            f"(sampled for performance). Use Edge Review tab for full data."
        )

    m = create_browse_map(
        edges=edges,
        disconnected_edges=disconnected_edges,
        filtered_edges=filtered_edges,
        net_new_edges=net_new_edges,
        bridge_edges=bridge_edges,
        basemap=basemap,
    )

    st_folium(m, height=650, returned_objects=[], key="browse_map", use_container_width=True)


def create_browse_map(
    edges: gpd.GeoDataFrame,
    net_new_edges: gpd.GeoDataFrame | None = None,
    basemap: str = "Light",
    disconnected_edges: gpd.GeoDataFrame | None = None,
    filtered_edges: gpd.GeoDataFrame | None = None,
    bridge_edges: gpd.GeoDataFrame | None = None,
) -> folium.Map:
    """Create the browse map with heatmap and all layers.

    All layers are added to the map and can be toggled client-side
    via Folium's built-in LayerControl (no Streamlit round-trip).

    Args:
        edges: All edges GeoDataFrame (with _source column)
        net_new_edges: Net-new coverage edges GeoDataFrame
        basemap: Default basemap name
        disconnected_edges: Truly disconnected edges GeoDataFrame
        filtered_edges: Connected but filtered edges GeoDataFrame
    """
    # Create map without default tiles
    m = folium.Map(location=[0, 0], zoom_start=2, tiles=None)

    # Add basemap tiles
    _add_basemap_tiles(m, basemap)

    # Prepare working copies in WGS84
    working_edges = edges
    if working_edges is not None and len(working_edges) > 0:
        if working_edges.crs and working_edges.crs.to_epsg() != 4326:
            working_edges = working_edges.to_crs("EPSG:4326")

    # Unmatched (target_new) edges - orange
    if working_edges is not None and len(working_edges) > 0:
        if "_source" in working_edges.columns:
            unmatched = working_edges[working_edges["_source"] == "target_new"]
            if len(unmatched) > 0:
                add_edges_layer(
                    m,
                    unmatched,
                    "Unmatched (Connected)",
                    SOURCE_COLORS["target_new"],
                    weight=3,
                )

    # Net-new coverage - cyan
    if net_new_edges is not None and len(net_new_edges) > 0:
        add_edges_layer(
            m,
            net_new_edges,
            "Net New Coverage",
            SOURCE_COLORS["net_new"],
            weight=4,
            opacity=0.9,
        )

    # Bridge edges — two layers: dashed full geometry + solid subline
    if bridge_edges is not None and len(bridge_edges) > 0:
        browse_bridges = bridge_edges
        if bridge_edges.crs and bridge_edges.crs.to_epsg() != 4326:
            browse_bridges = bridge_edges.to_crs("EPSG:4326")

        # Full geometry context (dashed, faded)
        if "_full_geometry" in browse_bridges.columns:
            full_gdf = browse_bridges.set_geometry("_full_geometry")
            add_edges_layer(
                m,
                full_gdf,
                "Bridges (Full)",
                SOURCE_COLORS["bridge"],
                weight=2,
                opacity=0.4,
                dash_array="8 4",
                add_markers=False,
            )

        # Subline (solid, prominent)
        add_edges_layer(
            m,
            browse_bridges,
            "Bridges (Subline)",
            SOURCE_COLORS["bridge"],
            weight=4,
            add_markers=False,
        )

    # Disconnected edges — no per-edge markers in browse mode (too many objects),
    # and sample when dataset is very large to keep the map responsive
    if disconnected_edges is not None and len(disconnected_edges) > 0:
        browse_disconnected = disconnected_edges
        if disconnected_edges.crs and disconnected_edges.crs.to_epsg() != 4326:
            browse_disconnected = disconnected_edges.to_crs("EPSG:4326")
        sampled = len(browse_disconnected) > MAX_BROWSE_FEATURES
        if sampled:
            logger.info(
                f"Sampling disconnected for browse map: {len(browse_disconnected)} -> {MAX_BROWSE_FEATURES}"
            )
            browse_disconnected = browse_disconnected.sample(MAX_BROWSE_FEATURES, random_state=42)

        # Render by priority if available, but without circle markers
        if "qa_priority" in browse_disconnected.columns:
            for priority in ["high", "medium", "low"]:
                priority_edges = browse_disconnected[browse_disconnected["qa_priority"] == priority]
                if len(priority_edges) > 0:
                    add_edges_layer(
                        m,
                        priority_edges,
                        f"Disconnected ({priority.title()})",
                        PRIORITY_COLORS[priority],
                        weight=2,
                        add_markers=False,
                    )
        else:
            add_edges_layer(
                m,
                browse_disconnected,
                "Disconnected",
                SOURCE_COLORS["disconnected"],
                weight=2,
                add_markers=False,
            )

    # Filtered edges are not shown in browse mode (too many features, use Edge Review)

    # Heatmap overlay from edge centroids (uses all non-reference data)
    heatmap_points = _compute_heatmap_points(working_edges, net_new_edges, disconnected_edges)
    if heatmap_points:
        HeatMap(
            heatmap_points,
            radius=15,
            blur=10,
            max_zoom=16,
            name="Density Heatmap",
        ).add_to(m)

    # Fit bounds to all non-reference data
    bounds_gdf = _get_non_reference_bounds(working_edges, disconnected_edges, net_new_edges)
    if bounds_gdf is not None and len(bounds_gdf) > 0:
        fit_bounds(m, bounds_gdf)

    # Layer control for client-side toggling (no server round-trip)
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def _compute_heatmap_points(
    edges: gpd.GeoDataFrame | None,
    net_new_edges: gpd.GeoDataFrame | None,
    disconnected_edges: gpd.GeoDataFrame | None = None,
) -> list[list[float]]:
    """Compute heatmap points from edge centroids.

    Uses target_new edges, net_new_edges, and disconnected edges as heat sources,
    since these represent where new/unintegrated data exists.

    Returns:
        List of [lat, lon] coordinate pairs.
    """
    points = []

    def _add_centroids(gdf: gpd.GeoDataFrame) -> None:
        working = gdf
        if working.crs and working.crs.to_epsg() != 4326:
            working = working.to_crs("EPSG:4326")
        for geom in working.geometry:
            if geom is not None and not geom.is_empty:
                c = geom.centroid
                points.append([c.y, c.x])

    # Centroids from unmatched (target_new) edges
    if edges is not None and len(edges) > 0 and "_source" in edges.columns:
        unmatched = edges[edges["_source"] == "target_new"]
        if len(unmatched) > 0:
            _add_centroids(unmatched)

    # Centroids from net-new edges
    if net_new_edges is not None and len(net_new_edges) > 0:
        _add_centroids(net_new_edges)

    # Centroids from disconnected edges
    if disconnected_edges is not None and len(disconnected_edges) > 0:
        _add_centroids(disconnected_edges)

    return points


def _add_basemap_tiles(m: folium.Map, default_basemap: str) -> None:
    """Add satellite, light, and OSM tile layers to the map.

    Args:
        m: Folium map
        default_basemap: Which basemap to show by default
    """
    default_key = _BASEMAP_TILES.get(default_basemap, "light")

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        max_zoom=20,
        show=(default_key == "satellite"),
    ).add_to(m)

    folium.TileLayer(
        tiles="cartodbpositron",
        name="Light",
        max_zoom=20,
        show=(default_key == "light"),
    ).add_to(m)

    folium.TileLayer(
        tiles="openstreetmap",
        name="OpenStreetMap",
        max_zoom=19,
        show=(default_key == "osm"),
    ).add_to(m)


def _get_non_reference_bounds(
    edges: gpd.GeoDataFrame | None,
    disconnected_edges: gpd.GeoDataFrame | None,
    net_new_edges: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame | None:
    """Combine non-reference geodata for bounds calculation.

    Returns:
        A GeoDataFrame with all non-reference geometries, or None.
    """
    parts = []

    if edges is not None and len(edges) > 0 and "_source" in edges.columns:
        non_ref = edges[edges["_source"] != "reference"]
        if len(non_ref) > 0:
            if non_ref.crs and non_ref.crs.to_epsg() != 4326:
                non_ref = non_ref.to_crs("EPSG:4326")
            parts.append(non_ref)

    if disconnected_edges is not None and len(disconnected_edges) > 0:
        disc = disconnected_edges
        if disc.crs and disc.crs.to_epsg() != 4326:
            disc = disc.to_crs("EPSG:4326")
        parts.append(disc)

    if net_new_edges is not None and len(net_new_edges) > 0:
        nn = net_new_edges
        if nn.crs and nn.crs.to_epsg() != 4326:
            nn = nn.to_crs("EPSG:4326")
        parts.append(nn)

    if not parts:
        return None

    import pandas as pd

    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
