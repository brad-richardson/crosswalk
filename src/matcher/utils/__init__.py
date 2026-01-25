"""Shared utilities."""

from .crs import (
    ProjectionResult,
    ensure_projected_crs,
    ensure_single_projected_crs,
    validate_projected_crs,
)
from .geometry import filter_to_linestrings

__all__ = [
    "filter_to_linestrings",
    "ProjectionResult",
    "ensure_projected_crs",
    "ensure_single_projected_crs",
    "validate_projected_crs",
]
