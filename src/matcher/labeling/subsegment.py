"""Sub-segment extraction and linear referencing utilities.

This module provides functions for extracting portions of LineStrings
using percentage-based linear referencing, and for estimating which
portions of two lines overlap.

NOTE: These functions only support LineString geometries. MultiLineString
and other geometry types will raise TypeError. Filter out non-LineString
geometries before using sub-segment features.
"""

from shapely.geometry import LineString, Point
from shapely.ops import substring


def extract_subsegment(line: LineString, start_pct: float, end_pct: float) -> LineString:
    """Extract a portion of a line using percentage-based linear referencing.

    Args:
        line: The input LineString (MultiLineString not supported)
        start_pct: Start position as fraction (0.0 to 1.0)
        end_pct: End position as fraction (0.0 to 1.0)

    Returns:
        A new LineString representing the extracted portion.
        Returns original line if start_pct >= end_pct or range is [0, 1].

    Raises:
        TypeError: If line is not a LineString
        ValueError: If line is empty or zero-length
    """
    # Only support LineString - raise for anything else
    if not isinstance(line, LineString):
        raise TypeError(
            f"line must be LineString, got {type(line).__name__}. "
            "Filter out non-LineString geometries before using sub-segment features."
        )
    if line.is_empty:
        raise ValueError("line must not be empty")
    if line.length == 0:
        raise ValueError("line must have non-zero length")

    # Handle edge cases
    if start_pct >= end_pct or (start_pct <= 0.0 and end_pct >= 1.0):
        return line

    # Clamp to valid range
    start_pct = max(0.0, min(1.0, start_pct))
    end_pct = max(0.0, min(1.0, end_pct))

    # Convert percentages to distances
    total_length = line.length
    start_dist = total_length * start_pct
    end_dist = total_length * end_pct

    # Use shapely's substring to extract the portion
    return substring(line, start_dist, end_dist)


def pct_to_distance(line: LineString, pct: float) -> float:
    """Convert percentage along line to actual distance.

    Args:
        line: The LineString
        pct: Position as fraction (0.0 to 1.0)

    Returns:
        Distance in the line's coordinate units
    """
    return line.length * max(0.0, min(1.0, pct))


def distance_to_pct(line: LineString, dist: float) -> float:
    """Convert distance along line to percentage.

    Args:
        line: The LineString
        dist: Distance in the line's coordinate units

    Returns:
        Position as fraction (0.0 to 1.0)
    """
    if line.length <= 0:
        return 0.0
    return max(0.0, min(1.0, dist / line.length))


def get_point_pct(line: LineString, point: Point) -> float:
    """Get the percentage along the line where a point projects.

    Args:
        line: The LineString to project onto
        point: The point to project

    Returns:
        Position as fraction (0.0 to 1.0) where the point
        projects onto the line
    """
    dist = line.project(point)
    return distance_to_pct(line, dist)


def get_point_at_pct(line: LineString, pct: float) -> Point:
    """Get the point at a given percentage along the line.

    Args:
        line: The LineString
        pct: Position as fraction (0.0 to 1.0)

    Returns:
        Point at that position along the line
    """
    dist = pct_to_distance(line, pct)
    return line.interpolate(dist)


def estimate_overlap_range(ref_line: LineString, target_line: LineString) -> dict[str, float]:
    """Estimate which portions of each line overlap.

    Uses projection of endpoints to estimate the overlapping ranges.
    This provides a starting point for manual adjustment in the UI.

    Args:
        ref_line: The reference LineString (MultiLineString not supported)
        target_line: The target LineString (MultiLineString not supported)

    Returns:
        Dictionary with keys:
        - ref_start_pct: Where overlap starts on reference (0.0-1.0)
        - ref_end_pct: Where overlap ends on reference (0.0-1.0)
        - target_start_pct: Where overlap starts on target (0.0-1.0)
        - target_end_pct: Where overlap ends on target (0.0-1.0)

    Raises:
        TypeError: If inputs are not LineString geometries
        ValueError: If inputs are empty or zero-length
    """
    # Only support LineString - raise for anything else
    if not isinstance(ref_line, LineString):
        raise TypeError(
            f"ref_line must be LineString, got {type(ref_line).__name__}. "
            "Filter out non-LineString geometries before using sub-segment features."
        )
    if not isinstance(target_line, LineString):
        raise TypeError(
            f"target_line must be LineString, got {type(target_line).__name__}. "
            "Filter out non-LineString geometries before using sub-segment features."
        )
    if ref_line.is_empty or target_line.is_empty:
        raise ValueError("Input geometries must not be empty")
    if ref_line.length == 0 or target_line.length == 0:
        raise ValueError("Input geometries must have non-zero length")

    # Project target endpoints onto reference line
    target_start_pt = Point(target_line.coords[0])
    target_end_pt = Point(target_line.coords[-1])
    ref_at_target_start = ref_line.project(target_start_pt, normalized=True)
    ref_at_target_end = ref_line.project(target_end_pt, normalized=True)

    # Project reference endpoints onto target line
    ref_start_pt = Point(ref_line.coords[0])
    ref_end_pt = Point(ref_line.coords[-1])
    target_at_ref_start = target_line.project(ref_start_pt, normalized=True)
    target_at_ref_end = target_line.project(ref_end_pt, normalized=True)

    # Ensure start < end by taking min/max
    ref_start = min(ref_at_target_start, ref_at_target_end)
    ref_end = max(ref_at_target_start, ref_at_target_end)
    target_start = min(target_at_ref_start, target_at_ref_end)
    target_end = max(target_at_ref_start, target_at_ref_end)

    # Clamp to [0, 1]
    return {
        "ref_start_pct": max(0.0, ref_start),
        "ref_end_pct": min(1.0, ref_end),
        "target_start_pct": max(0.0, target_start),
        "target_end_pct": min(1.0, target_end),
    }


def compute_subsegment_length(line: LineString, start_pct: float, end_pct: float) -> float:
    """Compute the length of a subsegment in the line's units.

    Args:
        line: The LineString
        start_pct: Start position as fraction (0.0 to 1.0)
        end_pct: End position as fraction (0.0 to 1.0)

    Returns:
        Length of the subsegment
    """
    start_pct = max(0.0, min(1.0, start_pct))
    end_pct = max(0.0, min(1.0, end_pct))
    return line.length * max(0.0, end_pct - start_pct)


def is_subsegment_selection(
    ref_start: float,
    ref_end: float,
    target_start: float,
    target_end: float,
    tolerance: float = 0.001,
) -> bool:
    """Check if the selection represents a sub-segment (not whole segment).

    Args:
        ref_start: Reference start percentage
        ref_end: Reference end percentage
        target_start: Target start percentage
        target_end: Target end percentage
        tolerance: Tolerance for floating point comparison

    Returns:
        True if this is a sub-segment selection (not 0-100% for both)
    """
    ref_is_full = abs(ref_start) < tolerance and abs(ref_end - 1.0) < tolerance
    target_is_full = abs(target_start) < tolerance and abs(target_end - 1.0) < tolerance
    return not (ref_is_full and target_is_full)
