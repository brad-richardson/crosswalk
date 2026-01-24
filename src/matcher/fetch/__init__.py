"""Data fetching module for Overture, OSM, local, and ArcGIS sources."""

# Re-export version constants and filename utilities for convenience
from ..config import DATA_VERSION, SCHEMA_VERSION, TRANSFORM_VERSION
from ..filenames import (
    extract_version_from_filename,
    find_osm_segments,
    find_overture_segments,
    find_target_file,
    osm_connectors_filename,
    osm_segments_filename,
    overture_connectors_filename,
    overture_segments_filename,
    target_filename,
)
from .arcgis import fetch_arcgis_layer
from .local import load_local_roads
from .metadata import FetchMetadata, load_metadata, save_metadata
from .osm import DEFAULT_OSM_BUFFER_M, fetch_osm_data, fetch_osm_segments, load_osm_roads
from .osm_download import download_and_extract
from .osm_pbf import parse_pbf
from .overture import (
    DEFAULT_OVERTURE_BUFFER_M,
    BoundingBox,
    fetch_overture_connectors,
    fetch_overture_segments,
    load_overture_segments,
)

__all__ = [
    "BoundingBox",
    "DEFAULT_OVERTURE_BUFFER_M",
    "DEFAULT_OSM_BUFFER_M",
    "fetch_overture_segments",
    "fetch_overture_connectors",
    "load_overture_segments",
    "fetch_osm_data",
    "fetch_osm_segments",
    "load_osm_roads",
    "download_and_extract",
    "parse_pbf",
    "load_local_roads",
    "fetch_arcgis_layer",
    "FetchMetadata",
    "load_metadata",
    "save_metadata",
    # Version constants
    "DATA_VERSION",
    "SCHEMA_VERSION",
    "TRANSFORM_VERSION",
    # Filename utilities
    "target_filename",
    "overture_segments_filename",
    "overture_connectors_filename",
    "osm_segments_filename",
    "osm_connectors_filename",
    "extract_version_from_filename",
    "find_overture_segments",
    "find_osm_segments",
    "find_target_file",
]
