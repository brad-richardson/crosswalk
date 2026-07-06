"""Shared utilities."""

from .crs import (
    ProjectionResult,
    ensure_projected_crs,
    ensure_single_projected_crs,
    validate_projected_crs,
)
from .geometry import filter_to_linestrings
from .linear_ref import (
    AttributeRange,
    LinearReferencedAttribute,
    coverage_for_value,
    create_trivial_lr,
    extract_aligned_attributes,
    extract_majority,
    normalize_ranges,
)

__all__ = [
    "AttributeRange",
    "LinearReferencedAttribute",
    "coverage_for_value",
    "create_trivial_lr",
    "extract_aligned_attributes",
    "extract_majority",
    "filter_to_linestrings",
    "normalize_ranges",
    "ProjectionResult",
    "ensure_projected_crs",
    "ensure_single_projected_crs",
    "validate_projected_crs",
]
