"""CRS (Coordinate Reference System) utilities for consistent projection handling.

This module provides utilities for standardizing CRS handling across the pipeline.
All distance-based computations require projected CRS (meters), not geographic CRS (degrees).

Key design decisions:
- Project once at pipeline entry, not in each module
- Use UTM for projection (automatic zone detection)
- Store original CRS for restoring output if needed

Usage:
    from matcher.utils.crs import ensure_projected_crs

    # At pipeline entry
    ref_proj, target_proj, original_crs = ensure_projected_crs(reference, target)

    # All downstream operations use projected data
    candidates = generate_candidates(ref_proj, target_proj, ...)
    results = score_candidates(candidates, ref_proj, target_proj, ...)
"""

from typing import NamedTuple

import geopandas as gpd
from loguru import logger
from pyproj import CRS


class ProjectionResult(NamedTuple):
    """Result of projecting GeoDataFrames to a metric CRS."""

    reference: gpd.GeoDataFrame
    """Reference GeoDataFrame in projected CRS (meters)."""

    target: gpd.GeoDataFrame
    """Target GeoDataFrame in projected CRS (meters)."""

    original_crs: CRS | None
    """Original CRS before projection (for restoring output if needed)."""

    projected_crs: CRS | None
    """The projected CRS used (UTM or original if already projected)."""

    was_reprojected: bool
    """True if data was reprojected, False if already in projected CRS."""


def validate_projected_crs(gdf: gpd.GeoDataFrame, name: str = "GeoDataFrame") -> None:
    """Validate that a GeoDataFrame has a projected (non-geographic) CRS.

    Raises ValueError if the CRS is geographic (lat/lon degrees) because
    distance computations will be in degrees, not meters.

    Args:
        gdf: GeoDataFrame to validate
        name: Name for error messages (e.g., "reference", "target")

    Raises:
        ValueError: If CRS is geographic or not set
    """
    if gdf.crs is None:
        raise ValueError(
            f"{name} has no CRS set. Cannot compute accurate distances. "
            f"Set CRS with gdf.set_crs() or project with ensure_projected_crs()."
        )

    if gdf.crs.is_geographic:
        raise ValueError(
            f"{name} has geographic CRS ({gdf.crs}). Distance computations require "
            f"a projected CRS (meters). Use ensure_projected_crs() to project first."
        )


def ensure_projected_crs(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
) -> ProjectionResult:
    """Ensure both GeoDataFrames are in a projected CRS suitable for distance computations.

    If data is already in a projected CRS, returns as-is.
    If data is in a geographic CRS (WGS84), projects to UTM based on data extent.

    This function handles the common case where reference and target need to be
    in the same projected CRS for accurate distance-based operations.

    Args:
        reference: Reference GeoDataFrame (e.g., Overture segments)
        target: Target GeoDataFrame (e.g., local road data)

    Returns:
        ProjectionResult containing:
        - reference: Reference in projected CRS
        - target: Target in projected CRS (same CRS as reference)
        - original_crs: The original CRS before projection (or None if no reprojection)
        - projected_crs: The projected CRS used
        - was_reprojected: True if data was reprojected

    Example:
        >>> result = ensure_projected_crs(reference, target)
        >>> # Use projected data for computations
        >>> candidates = generate_candidates(result.reference, result.target)
        >>> # Later, restore original CRS if needed
        >>> if result.original_crs:
        ...     output = output.to_crs(result.original_crs)
    """
    # Store original CRS (from reference, which is typically the "master" CRS)
    original_crs = reference.crs

    # Ensure target matches reference CRS first
    if target.crs != reference.crs:
        logger.debug(f"Aligning target CRS from {target.crs} to {reference.crs}")
        target = target.to_crs(reference.crs)

    # Check if already projected
    if reference.crs is not None and not reference.crs.is_geographic:
        logger.debug(f"Data already in projected CRS: {reference.crs}")
        return ProjectionResult(
            reference=reference,
            target=target,
            original_crs=None,  # No reprojection needed
            projected_crs=reference.crs,
            was_reprojected=False,
        )

    # Need to project to UTM
    utm_crs = reference.estimate_utm_crs()
    logger.info(f"Projecting to {utm_crs} for meter-based computations")

    reference_proj = reference.to_crs(utm_crs)
    target_proj = target.to_crs(utm_crs)

    return ProjectionResult(
        reference=reference_proj,
        target=target_proj,
        original_crs=original_crs,
        projected_crs=utm_crs,
        was_reprojected=True,
    )


def ensure_single_projected_crs(
    gdf: gpd.GeoDataFrame,
    name: str = "GeoDataFrame",
) -> tuple[gpd.GeoDataFrame, CRS | None]:
    """Ensure a single GeoDataFrame is in a projected CRS.

    Simpler version of ensure_projected_crs for cases where only one
    GeoDataFrame needs to be projected.

    Args:
        gdf: GeoDataFrame to project
        name: Name for logging

    Returns:
        Tuple of (projected_gdf, original_crs)
        original_crs is None if no reprojection was needed
    """
    if gdf.crs is not None and not gdf.crs.is_geographic:
        return gdf, None

    original_crs = gdf.crs
    utm_crs = gdf.estimate_utm_crs()
    logger.debug(f"Projecting {name} to {utm_crs}")
    return gdf.to_crs(utm_crs), original_crs
