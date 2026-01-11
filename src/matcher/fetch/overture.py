"""Fetch Overture Transportation data for a bounding box using DuckDB.

Uses DuckDB's S3 httpfs extension to query Overture GeoParquet files
directly and download only the data within the specified AOI.
"""

from pathlib import Path
from typing import Optional

import duckdb
from loguru import logger
from pydantic import BaseModel

from ..config import settings


class BoundingBox(BaseModel):
    """Bounding box in EPSG:4326 (WGS84)."""

    minx: float
    miny: float
    maxx: float
    maxy: float

    def to_wkt(self) -> str:
        """Convert to WKT polygon."""
        return (
            f"POLYGON(({self.minx} {self.miny}, {self.maxx} {self.miny}, "
            f"{self.maxx} {self.maxy}, {self.minx} {self.maxy}, {self.minx} {self.miny}))"
        )


def _get_overture_base_url(release: Optional[str] = None) -> str:
    """Get the S3 base URL for Overture data."""
    release = release or settings.overture_release
    return f"s3://overturemaps-us-west-2/release/{release}"


def _setup_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Set up DuckDB with spatial and httpfs extensions."""
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(f"SET s3_region='{settings.overture_s3_region}';")
    return conn


def fetch_overture_segments(
    bbox: BoundingBox,
    output_path: Path,
    release: Optional[str] = None,
) -> Path:
    """Download Overture road segments for a bounding box.

    Args:
        bbox: Bounding box in WGS84 coordinates
        output_path: Path for output GeoParquet file
        release: Overture release version (default from settings)

    Returns:
        Path to the output file
    """
    logger.info(f"Fetching Overture segments for bbox: {bbox}")

    base_url = _get_overture_base_url(release)
    conn = _setup_duckdb_connection()

    # Query segments within bbox
    # Using bbox columns for efficient filtering before geometry check
    query = f"""
    COPY (
        SELECT
            id,
            geometry,
            names.primary AS name,
            class,
            subclass,
            connectors,
            road,
            sources,
            level
        FROM read_parquet('{base_url}/theme=transportation/type=segment/*', filename=true, hive_partitioning=true)
        WHERE
            bbox.xmin <= {bbox.maxx}
            AND bbox.xmax >= {bbox.minx}
            AND bbox.ymin <= {bbox.maxy}
            AND bbox.ymax >= {bbox.miny}
            AND subtype = 'road'
    ) TO '{output_path}' (FORMAT PARQUET);
    """

    logger.debug(f"Executing query: {query}")
    conn.execute(query)
    conn.close()

    logger.info(f"Saved Overture segments to {output_path}")
    return output_path


def fetch_overture_connectors(
    bbox: BoundingBox,
    output_path: Path,
    release: Optional[str] = None,
) -> Path:
    """Download Overture connectors (intersections) for a bounding box.

    Args:
        bbox: Bounding box in WGS84 coordinates
        output_path: Path for output GeoParquet file
        release: Overture release version (default from settings)

    Returns:
        Path to the output file
    """
    logger.info(f"Fetching Overture connectors for bbox: {bbox}")

    base_url = _get_overture_base_url(release)
    conn = _setup_duckdb_connection()

    query = f"""
    COPY (
        SELECT
            id,
            geometry,
            connectors,
            sources
        FROM read_parquet('{base_url}/theme=transportation/type=connector/*', filename=true, hive_partitioning=true)
        WHERE
            bbox.xmin <= {bbox.maxx}
            AND bbox.xmax >= {bbox.minx}
            AND bbox.ymin <= {bbox.maxy}
            AND bbox.ymax >= {bbox.miny}
    ) TO '{output_path}' (FORMAT PARQUET);
    """

    logger.debug(f"Executing query: {query}")
    conn.execute(query)
    conn.close()

    logger.info(f"Saved Overture connectors to {output_path}")
    return output_path


def load_overture_segments(path: Path):
    """Load Overture segments from a GeoParquet file.

    Args:
        path: Path to GeoParquet file

    Returns:
        GeoDataFrame with Overture segments
    """
    import geopandas as gpd

    logger.info(f"Loading Overture segments from {path}")
    gdf = gpd.read_parquet(path)

    # Extract bridge/tunnel info from road struct if present
    if "road" in gdf.columns:
        # road is a struct with flags including is_bridge, is_tunnel
        try:
            gdf["is_bridge"] = gdf["road"].apply(
                lambda x: x.get("flags", {}).get("is_bridge", False) if x else False
            )
            gdf["is_tunnel"] = gdf["road"].apply(
                lambda x: x.get("flags", {}).get("is_tunnel", False) if x else False
            )
        except (AttributeError, TypeError):
            gdf["is_bridge"] = False
            gdf["is_tunnel"] = False

    logger.info(f"Loaded {len(gdf)} Overture segments")
    return gdf
