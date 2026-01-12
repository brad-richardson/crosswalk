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


def extract_bbox(
    input_pbf: Path,
    bbox: BoundingBox,
    output_pbf: Path,
) -> Path:
    """Extract a bounding box from a PBF using osmium CLI.

    Args:
        input_pbf: Path to input PBF file
        bbox: Bounding box to extract
        output_pbf: Path for output PBF file

    Returns:
        Path to the extracted PBF file

    Raises:
        RuntimeError: If osmium-tool is not installed or extraction fails
    """
    output_pbf.parent.mkdir(parents=True, exist_ok=True)

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

    logger.info(f"Extracting bbox {bbox_str} from {input_pbf.name}...")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"osmium extract failed: {e.stderr}")
        raise RuntimeError(f"osmium extract failed: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "osmium-tool not found. Install with: brew install osmium-tool (macOS) "
            "or apt install osmium-tool (Ubuntu)"
        )

    size_kb = output_pbf.stat().st_size / 1e3
    logger.info(f"Extracted bbox to {output_pbf} ({size_kb:.1f} KB)")
    return output_pbf


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
    ).hexdigest()[:8]
    output_pbf = output_dir / f"osm_extract_{bbox_hash}.osm.pbf"

    return extract_bbox(regional_pbf, bbox, output_pbf)
