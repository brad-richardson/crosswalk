"""Download and extract OSM PBF files from Geofabrik.

This module handles:
1. Fetching the Geofabrik region index
2. Finding the smallest region containing a bounding box
3. Downloading regional PBF files with caching
4. Extracting bbox areas using osmium CLI
"""

import hashlib
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests
from loguru import logger
from shapely.geometry import box, shape

from ..config import settings
from .overture import BoundingBox

GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"


def get_geofabrik_index(cache_dir: Path | None = None) -> dict:
    """Fetch and cache the Geofabrik region index.

    The index is cached for 24 hours to avoid repeated downloads.

    Args:
        cache_dir: Directory for caching (default from settings)

    Returns:
        Geofabrik index JSON as dict
    """
    cache_dir = cache_dir or settings.pbf_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "geofabrik_index.json"

    # Check if cached index is fresh
    if index_path.exists():
        mtime = datetime.fromtimestamp(index_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(hours=settings.pbf_cache_ttl_hours):
            logger.debug("Using cached Geofabrik index")
            import json

            return json.loads(index_path.read_text())

    logger.info("Downloading Geofabrik index...")
    response = requests.get(GEOFABRIK_INDEX_URL, timeout=30)
    response.raise_for_status()
    index_data = response.json()

    # Cache the index
    import json

    index_path.write_text(json.dumps(index_data))
    logger.debug(f"Cached Geofabrik index to {index_path}")
    return index_data


def find_best_region(bbox: BoundingBox, index: dict) -> dict:
    """Find the smallest Geofabrik region that fully contains the bbox.

    Args:
        bbox: Bounding box to find region for
        index: Geofabrik index JSON

    Returns:
        Region feature dict from the index

    Raises:
        ValueError: If no region contains the bbox
    """
    bbox_geom = box(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)

    # Build list of (region, geometry) tuples for regions that contain the bbox
    regions_with_geom = []
    for feature in index.get("features", []):
        geom = feature.get("geometry")
        if geom:
            try:
                region_geom = shape(geom)
                if region_geom.contains(bbox_geom):
                    regions_with_geom.append((feature, region_geom))
            except Exception:
                continue

    if not regions_with_geom:
        raise ValueError(f"No Geofabrik region contains bbox: {bbox}")

    # Find smallest containing region (by area)
    regions_with_geom.sort(key=lambda x: x[1].area)
    best_region = regions_with_geom[0][0]

    region_name = best_region.get("properties", {}).get("name", "unknown")
    logger.info(f"Selected Geofabrik region: {region_name}")
    return best_region


def get_pbf_url(region: dict) -> str:
    """Extract the PBF download URL from a region feature.

    Args:
        region: Region feature dict from Geofabrik index

    Returns:
        URL for the PBF file

    Raises:
        ValueError: If no PBF URL found
    """
    urls = region.get("properties", {}).get("urls", {})
    pbf_url = urls.get("pbf")
    if not pbf_url:
        raise ValueError(f"No PBF URL found for region: {region}")
    return pbf_url


def download_pbf(
    url: str,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download a PBF file, using cache if available.

    Args:
        url: URL to download from
        cache_dir: Directory for caching (default from settings)
        force: Force re-download even if cached

    Returns:
        Path to the downloaded file
    """
    cache_dir = cache_dir or settings.pbf_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use filename from URL for cache key
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    cache_path = cache_dir / filename

    # Check cache freshness
    if not force and cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(hours=settings.pbf_cache_ttl_hours):
            size_mb = cache_path.stat().st_size / 1e6
            logger.info(f"Using cached PBF: {cache_path} ({size_mb:.1f} MB)")
            return cache_path

    logger.info(f"Downloading PBF from {url}...")
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    # Download with progress logging
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    last_pct = 0

    with open(cache_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = int((downloaded / total_size) * 100)
                if pct >= last_pct + 10:  # Log every 10%
                    logger.info(f"Downloaded {pct}%...")
                    last_pct = pct

    size_mb = downloaded / 1e6
    logger.info(f"Downloaded PBF to {cache_path} ({size_mb:.1f} MB)")
    return cache_path


def _check_osmium_cli() -> bool:
    """Check if osmium CLI tool is available."""
    try:
        result = subprocess.run(
            ["osmium", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def extract_bbox(
    input_pbf: Path,
    bbox: BoundingBox,
    output_pbf: Path,
) -> Path:
    """Extract a bounding box from a PBF.

    Tries osmium CLI first (faster), falls back to pyosmium if CLI unavailable.

    Args:
        input_pbf: Path to input PBF file
        bbox: Bounding box to extract
        output_pbf: Path for output PBF file

    Returns:
        Path to the extracted PBF file
    """
    output_pbf.parent.mkdir(parents=True, exist_ok=True)

    if _check_osmium_cli():
        return _extract_bbox_cli(input_pbf, bbox, output_pbf)
    else:
        logger.info("osmium CLI not found, using pyosmium fallback")
        return _extract_bbox_pyosmium(input_pbf, bbox, output_pbf)


def _extract_bbox_cli(
    input_pbf: Path,
    bbox: BoundingBox,
    output_pbf: Path,
) -> Path:
    """Extract bbox using osmium CLI (faster)."""
    # Format: west,south,east,north (lon,lat,lon,lat)
    bbox_str = f"{bbox.xmin},{bbox.ymin},{bbox.xmax},{bbox.ymax}"

    cmd = [
        "osmium",
        "extract",
        "-b",
        bbox_str,
        "-o",
        str(output_pbf),
        "--overwrite",
        str(input_pbf),
    ]

    logger.info(f"Extracting bbox {bbox_str} from {input_pbf.name} using osmium CLI...")
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.debug(result.stdout)

    size_kb = output_pbf.stat().st_size / 1e3
    logger.info(f"Extracted bbox to {output_pbf} ({size_kb:.1f} KB)")
    return output_pbf


def _extract_bbox_pyosmium(
    input_pbf: Path,
    bbox: BoundingBox,
    output_pbf: Path,
) -> Path:
    """Extract bbox using pyosmium and save directly to parquet.

    Note: This skips the PBF intermediate step since writing a valid PBF
    with osmium.SimpleWriter requires careful handling of node references.
    Instead, we extract directly to parquet files.
    """
    import osmium
    import geopandas as gpd
    from collections import Counter
    from shapely.geometry import LineString, Point

    logger.info(f"Extracting bbox from {input_pbf.name} using pyosmium...")
    logger.warning("pyosmium extraction is slower than osmium CLI for large files")

    # Highway values to include
    HIGHWAY_VALUES = {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
        "residential",
        "unclassified",
        "service",
        "living_street",
        "road",
        "footway",
        "path",
        "cycleway",
        "steps",
        "pedestrian",
        "bridleway",
        "track",
        "construction",
        "proposed",
        "abandoned",
    }

    class DirectExtractHandler(osmium.SimpleHandler):
        """Handler that extracts roads directly from PBF."""

        def __init__(self, bbox: BoundingBox):
            super().__init__()
            self.bbox = bbox
            self.roads = []
            self.node_refs = Counter()
            self.node_locations = {}
            self.node_versions = {}
            self._invalid_count = 0

        def node(self, n):
            """Store node versions for connector IDs."""
            self.node_versions[n.id] = n.version

        def way(self, w):
            """Extract highway ways that intersect the bbox."""
            highway = w.tags.get("highway")
            if highway not in HIGHWAY_VALUES:
                return

            try:
                coords = []
                node_ids = []
                has_node_in_bbox = False

                for n in w.nodes:
                    if n.location.valid():
                        lon, lat = n.location.lon, n.location.lat
                        coords.append((lon, lat))
                        node_ids.append(n.ref)
                        self.node_locations[n.ref] = (lon, lat)
                        if (
                            self.bbox.xmin <= lon <= self.bbox.xmax
                            and self.bbox.ymin <= lat <= self.bbox.ymax
                        ):
                            has_node_in_bbox = True
            except osmium.InvalidLocationError:
                self._invalid_count += 1
                return

            if not has_node_in_bbox or len(coords) < 2:
                return

            # Count node references for connector extraction
            for node_id in node_ids:
                self.node_refs[node_id] += 1

            # Extract tags
            tags = {
                "highway": highway,
                "name": w.tags.get("name"),
                "bridge": w.tags.get("bridge"),
                "tunnel": w.tags.get("tunnel"),
                "layer": w.tags.get("layer"),
                "oneway": w.tags.get("oneway"),
            }
            tags = {k: v for k, v in tags.items() if v is not None}

            self.roads.append(
                {
                    "id": f"w{w.id}@{w.version}",
                    "geometry": LineString(coords),
                    "tags": tags,
                    "name": w.tags.get("name"),
                    "node_ids": node_ids,
                }
            )

    handler = DirectExtractHandler(bbox)
    handler.apply_file(str(input_pbf), locations=True, idx="flex_mem")

    if handler._invalid_count > 0:
        logger.warning(f"Skipped {handler._invalid_count} ways with invalid locations")

    logger.info(f"Extracted {len(handler.roads)} road segments in bbox")

    # Build connectors
    connector_node_ids = set()
    for node_id, count in handler.node_refs.items():
        if count >= 2:
            connector_node_ids.add(node_id)
    for road in handler.roads:
        node_ids = road.get("node_ids", [])
        if node_ids:
            connector_node_ids.add(node_ids[0])
            connector_node_ids.add(node_ids[-1])

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

    # Save directly to parquet (bypassing PBF)
    output_dir = output_pbf.parent
    roads_path = output_dir / "osm_roads_raw.parquet"
    connectors_path = output_dir / "osm_connectors_raw.parquet"

    if handler.roads:
        roads_gdf = gpd.GeoDataFrame(handler.roads, crs="EPSG:4326")
        roads_gdf.to_parquet(roads_path)
    else:
        roads_gdf = gpd.GeoDataFrame(
            columns=["id", "geometry", "tags", "name", "node_ids"], crs="EPSG:4326"
        )
        roads_gdf.to_parquet(roads_path)

    if connectors:
        connectors_gdf = gpd.GeoDataFrame(connectors, crs="EPSG:4326")
        connectors_gdf.to_parquet(connectors_path)
    else:
        connectors_gdf = gpd.GeoDataFrame(columns=["id", "geometry"], crs="EPSG:4326")
        connectors_gdf.to_parquet(connectors_path)

    # Return a marker path - the caller will check for parquet files
    marker = output_dir / ".pyosmium_extracted"
    marker.touch()
    return marker


def download_and_extract(
    bbox: BoundingBox,
    output_dir: Path,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Full workflow: find region, download, and extract bbox.

    This is the main entry point for downloading OSM data for a bounding box.

    Args:
        bbox: Bounding box in WGS84 coordinates
        output_dir: Directory for extracted PBF output
        cache_dir: Directory for caching regional PBFs (default from settings)
        force: Force re-download even if cached

    Returns:
        Path to the extracted bbox PBF file
    """
    cache_dir = cache_dir or settings.pbf_cache_dir

    # Step 1: Get Geofabrik index
    index = get_geofabrik_index(cache_dir)

    # Step 2: Find best region
    region = find_best_region(bbox, index)

    # Step 3: Download regional PBF
    pbf_url = get_pbf_url(region)
    regional_pbf = download_pbf(pbf_url, cache_dir, force)

    # Step 4: Extract bbox
    # Create unique filename based on bbox
    bbox_hash = hashlib.md5(
        f"{bbox.xmin},{bbox.ymin},{bbox.xmax},{bbox.ymax}".encode()
    ).hexdigest()[:16]
    output_pbf = output_dir / f"osm_extract_{bbox_hash}.osm.pbf"

    return extract_bbox(regional_pbf, bbox, output_pbf)
