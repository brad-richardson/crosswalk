"""Parse OSM PBF files using pyosmium.

This module extracts road features from OSM PBF files, filtering by highway tag
and reconstructing geometries from node coordinates. Also extracts connectors
(intersection nodes) where multiple roads meet.
"""

from collections import Counter
from pathlib import Path

import geopandas as gpd
import osmium
from loguru import logger
from shapely.geometry import LineString, Point

# Highway values to include (aligned with Overture Maps classes)
# See: https://docs.overturemaps.org/schema/reference/transportation/segment
HIGHWAY_VALUES = {
    # Major roads
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    # Link roads (connect major roads, e.g., on-ramps)
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    # Minor roads
    "residential",
    "living_street",
    "unclassified",
    "service",
    # Paths
    "pedestrian",
    "footway",
    "steps",
    "path",
    "track",
    "cycleway",
    "bridleway",
    # Unknown (OSM "road" maps to Overture "unknown")
    "road",
    # Lifecycle states
    "construction",
    "proposed",
    "abandoned",
}


class RoadHandler(osmium.SimpleHandler):
    """Handler that collects road ways with their geometries, tags, and node references."""

    def __init__(self):
        super().__init__()
        self.roads = []
        self.node_refs = Counter()  # Count how many ways reference each node
        self.node_locations = {}  # Store node locations for connector extraction
        self.node_versions = {}  # Store node versions (populated in second pass)
        self._invalid_count = 0

    def way(self, w):
        """Process a way element."""
        highway = w.tags.get("highway")
        if highway not in HIGHWAY_VALUES:
            return

        # Build geometry from node locations and track node references
        try:
            coords = []
            node_ids = []
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.lon, n.lat))
                    node_ids.append(n.ref)
                    self.node_locations[n.ref] = (n.lon, n.lat)
        except osmium.InvalidLocationError:
            self._invalid_count += 1
            return

        # Need at least 2 points for a LineString
        if len(coords) < 2:
            return

        # Count node references (endpoints and shared nodes become connectors)
        for node_id in node_ids:
            self.node_refs[node_id] += 1

        # Extract relevant tags as dict
        tags = {
            "highway": highway,
            "name": w.tags.get("name"),
            "bridge": w.tags.get("bridge"),
            "tunnel": w.tags.get("tunnel"),
            "layer": w.tags.get("layer"),
            "oneway": w.tags.get("oneway"),
            "lanes": w.tags.get("lanes"),
            "surface": w.tags.get("surface"),
            "maxspeed": w.tags.get("maxspeed"),
            "ref": w.tags.get("ref"),
            "access": w.tags.get("access"),
            "service": w.tags.get("service"),
            "construction": w.tags.get("construction"),
            "covered": w.tags.get("covered"),
            "indoor": w.tags.get("indoor"),
            "abandoned": w.tags.get("abandoned"),
            "disused": w.tags.get("disused"),
        }
        # Remove None values
        tags = {k: v for k, v in tags.items() if v is not None}

        self.roads.append(
            {
                "id": f"w{w.id}@{w.version}",
                "geometry": LineString(coords),
                "tags": tags,
                "name": w.tags.get("name"),
                "node_ids": node_ids,  # Track node IDs for connector linking
            }
        )

    def node(self, n):
        """Collect version info for all nodes (nodes are processed before ways)."""
        # Store all node versions - we'll filter later for ones we need
        # This is memory-efficient enough for bbox extracts
        self.node_versions[n.id] = n.version


def parse_pbf(pbf_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Parse roads and connectors from a PBF file.

    Uses pyosmium with location caching for efficient geometry reconstruction.
    Connectors are nodes that are referenced by multiple ways (intersections)
    or are at the start/end of ways (dead ends, cul-de-sacs).

    Args:
        pbf_path: Path to the OSM PBF file

    Returns:
        Tuple of (roads_gdf, connectors_gdf)
        - roads_gdf: GeoDataFrame with columns: id, geometry, tags, name, node_ids
        - connectors_gdf: GeoDataFrame with columns: id, geometry
    """
    logger.info(f"Parsing roads from {pbf_path}...")

    handler = RoadHandler()

    # Use locations=True to enable node coordinate caching
    # idx='flex_mem' is a good balance of speed and memory usage
    handler.apply_file(
        str(pbf_path),
        locations=True,
        idx="flex_mem",
    )

    if handler._invalid_count > 0:
        logger.warning(f"Skipped {handler._invalid_count} ways with invalid node locations")

    logger.info(f"Parsed {len(handler.roads)} road segments")

    # Build roads GeoDataFrame
    if not handler.roads:
        logger.warning("No roads found in PBF file")
        roads_gdf = gpd.GeoDataFrame(
            columns=["id", "geometry", "tags", "name", "node_ids"],
            crs="EPSG:4326",
        )
    else:
        roads_gdf = gpd.GeoDataFrame(handler.roads, crs="EPSG:4326")

    # Extract connectors (nodes referenced by 2+ ways, or at endpoints)
    # Endpoints of ways are also connectors (dead ends, or shared with other ways)
    connector_node_ids = set()

    # Add all nodes that are referenced by multiple ways
    for node_id, count in handler.node_refs.items():
        if count >= 2:
            connector_node_ids.add(node_id)

    # Add endpoints of all ways (first and last node)
    for road in handler.roads:
        node_ids = road.get("node_ids", [])
        if node_ids:
            connector_node_ids.add(node_ids[0])  # Start node
            connector_node_ids.add(node_ids[-1])  # End node

    # Build connectors GeoDataFrame
    connectors = []
    for node_id in connector_node_ids:
        if node_id in handler.node_locations:
            lon, lat = handler.node_locations[node_id]
            version = handler.node_versions.get(node_id, 1)
            connectors.append(
                {
                    "id": f"n{node_id}@{version}",
                    "geometry": Point(lon, lat),
                }
            )

    logger.info(f"Found {len(connectors)} connectors (intersections/endpoints)")

    if not connectors:
        connectors_gdf = gpd.GeoDataFrame(
            columns=["id", "geometry"],
            crs="EPSG:4326",
        )
    else:
        connectors_gdf = gpd.GeoDataFrame(connectors, crs="EPSG:4326")

    return roads_gdf, connectors_gdf
