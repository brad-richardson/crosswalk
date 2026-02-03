"""Linear-referenced attribute handling for road segments.

This module provides data structures and functions for handling attributes that vary
along the length of a road segment (linear referencing). Overture segments can have
attributes like names, subclass, level, and road_flags that change at different
positions along the segment.

Key concepts:
- AttributeRange: A single range [start, end) with a value
- LinearReferencedAttribute: A normalized set of disjoint ranges covering [0.0, 1.0]
- Normalization: Converting overlapping/gapped rules into disjoint ranges
- Majority extraction: Getting the value with longest coverage in a subrange
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class AttributeRange:
    """A single range with an associated value.

    Represents an attribute value that applies over a portion of a segment.
    Ranges use half-open intervals [start, end) where start and end are
    fractions from 0.0 to 1.0 representing positions along the segment.
    """

    start: float  # Start position (0.0-1.0), inclusive
    end: float  # End position (0.0-1.0), exclusive
    value: Any  # The attribute value for this range

    def __post_init__(self) -> None:
        """Validate range bounds."""
        if not (0.0 <= self.start <= 1.0):
            raise ValueError(f"start must be in [0.0, 1.0], got {self.start}")
        if not (0.0 <= self.end <= 1.0):
            raise ValueError(f"end must be in [0.0, 1.0], got {self.end}")
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")

    @property
    def length(self) -> float:
        """Return the length of this range as a fraction."""
        return self.end - self.start

    def overlaps(self, other: AttributeRange) -> bool:
        """Check if this range overlaps with another."""
        return self.start < other.end and other.start < self.end

    def intersection(self, start: float, end: float) -> float:
        """Return the length of intersection with a query range.

        Args:
            start: Start of query range (0.0-1.0)
            end: End of query range (0.0-1.0)

        Returns:
            Length of the intersection as a fraction
        """
        overlap_start = max(self.start, start)
        overlap_end = min(self.end, end)
        return max(0.0, overlap_end - overlap_start)


@dataclass
class LinearReferencedAttribute:
    """A normalized linear-referenced attribute covering [0.0, 1.0].

    Represents an attribute that may vary along a segment's length.
    The ranges are guaranteed to be:
    - Disjoint (non-overlapping)
    - Sorted by start position
    - Covering the entire [0.0, 1.0] range (gaps filled with default_value)
    """

    ranges: list[AttributeRange] = field(default_factory=list)
    default_value: Any = None

    def is_uniform(self) -> bool:
        """Check if this attribute has a single uniform value."""
        return len(self.ranges) == 1

    def get_value_at(self, position: float) -> Any:
        """Get the attribute value at a specific position.

        Args:
            position: Position along the segment (0.0-1.0)

        Returns:
            The attribute value at that position
        """
        for attr_range in self.ranges:
            if attr_range.start <= position < attr_range.end:
                return attr_range.value
            # Handle the special case of position == 1.0
            if position == 1.0 and attr_range.end == 1.0:
                return attr_range.value
        return self.default_value

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert to a JSON-serializable list of dicts.

        Returns:
            List of dicts with 'between' and 'value' keys, matching Overture schema
        """
        return [{"between": [r.start, r.end], "value": r.value} for r in self.ranges]

    @classmethod
    def from_dict_list(
        cls, data: list[dict[str, Any]], default_value: Any = None
    ) -> LinearReferencedAttribute:
        """Create from a JSON-deserialized list of dicts.

        Args:
            data: List of dicts with 'between' and 'value' keys
            default_value: Default value for gaps

        Returns:
            LinearReferencedAttribute instance
        """
        ranges = []
        for d in data:
            start, end = d["between"]
            ranges.append(AttributeRange(start=start, end=end, value=d["value"]))
        return cls(ranges=ranges, default_value=default_value)


def normalize_ranges(
    rules: list[tuple[float, float, Any, int]],
    default_value: Any,
) -> LinearReferencedAttribute:
    """Normalize a set of potentially overlapping rules into disjoint ranges.

    Algorithm:
    1. Collect all boundary points (starts and ends of rules, plus 0.0 and 1.0)
    2. For each interval between boundaries, determine the winning value
    3. Merge adjacent ranges with the same value

    When multiple rules apply to the same interval, the rule with the lowest
    priority value wins (priority 0 beats priority 1).

    Args:
        rules: List of (start, end, value, priority) tuples where:
            - start: Start position (0.0-1.0)
            - end: End position (0.0-1.0)
            - value: The attribute value
            - priority: Lower priority wins (0 beats 1)
        default_value: Value to use for ranges not covered by any rule

    Returns:
        A LinearReferencedAttribute with normalized disjoint ranges
    """
    if not rules:
        # No rules - entire segment has default value
        return LinearReferencedAttribute(
            ranges=[AttributeRange(start=0.0, end=1.0, value=default_value)],
            default_value=default_value,
        )

    # Step 1: Collect all boundary points
    boundaries: set[float] = {0.0, 1.0}
    for start, end, _, _ in rules:
        boundaries.add(start)
        boundaries.add(end)

    sorted_boundaries = sorted(boundaries)

    # Step 2: For each interval, determine the winner
    intervals: list[AttributeRange] = []
    for i in range(len(sorted_boundaries) - 1):
        interval_start = sorted_boundaries[i]
        interval_end = sorted_boundaries[i + 1]

        # Find all rules that apply to this interval
        applicable: list[tuple[Any, int]] = []
        for start, end, value, priority in rules:
            if start <= interval_start and interval_end <= end:
                applicable.append((value, priority))

        if applicable:
            # Sort by priority (lowest wins), then by order added (stable sort)
            applicable.sort(key=lambda x: x[1])
            winner_value = applicable[0][0]
        else:
            winner_value = default_value

        intervals.append(AttributeRange(start=interval_start, end=interval_end, value=winner_value))

    # Step 3: Merge adjacent ranges with the same value
    merged: list[AttributeRange] = []
    for interval in intervals:
        if merged and merged[-1].value == interval.value:
            # Extend the previous range
            merged[-1] = AttributeRange(
                start=merged[-1].start, end=interval.end, value=interval.value
            )
        else:
            merged.append(interval)

    return LinearReferencedAttribute(ranges=merged, default_value=default_value)


def extract_majority(
    lr_attr: LinearReferencedAttribute,
    start_frac: float,
    end_frac: float,
) -> Any:
    """Extract the majority-covering value for a subrange.

    Returns the attribute value that has the longest coverage within
    the specified query range. In case of a tie, returns the first
    value encountered (stable by range order).

    Args:
        lr_attr: The linear-referenced attribute
        start_frac: Start of query range (0.0-1.0)
        end_frac: End of query range (0.0-1.0)

    Returns:
        The value with the longest coverage in the query range
    """
    if not lr_attr.ranges:
        return lr_attr.default_value

    # Clamp to valid range
    start_frac = max(0.0, min(1.0, start_frac))
    end_frac = max(0.0, min(1.0, end_frac))

    if start_frac >= end_frac:
        return lr_attr.default_value

    # Calculate coverage for each unique value
    # Use a list to preserve order for tie-breaking
    value_coverage: list[tuple[Any, float]] = []
    # Map hashable values to their index in value_coverage
    seen_hashable: dict[Any, int] = {}

    for attr_range in lr_attr.ranges:
        coverage = attr_range.intersection(start_frac, end_frac)
        if coverage > 0:
            value = attr_range.value
            # Try to use value-based grouping (hashable types)
            try:
                hash(value)
                is_hashable = True
            except TypeError:
                is_hashable = False

            if is_hashable:
                # Hashable: use the value itself as the key
                if value in seen_hashable:
                    idx = seen_hashable[value]
                    old_value, old_coverage = value_coverage[idx]
                    value_coverage[idx] = (old_value, old_coverage + coverage)
                else:
                    seen_hashable[value] = len(value_coverage)
                    value_coverage.append((value, coverage))
            else:
                # Unhashable: linear search for equality
                found_idx = None
                for i, (existing_value, _) in enumerate(value_coverage):
                    if existing_value == value:
                        found_idx = i
                        break
                if found_idx is not None:
                    old_value, old_coverage = value_coverage[found_idx]
                    value_coverage[found_idx] = (old_value, old_coverage + coverage)
                else:
                    value_coverage.append((value, coverage))

    if not value_coverage:
        return lr_attr.default_value

    # Find the value with maximum coverage (first encountered wins ties)
    max_coverage = -1.0
    winner = lr_attr.default_value
    for value, coverage in value_coverage:
        if coverage > max_coverage:
            max_coverage = coverage
            winner = value

    return winner


def coverage_for_value(
    lr_attr: LinearReferencedAttribute,
    start_frac: float,
    end_frac: float,
    target_value: Any,
) -> float:
    """Calculate the coverage of a specific value in a subrange.

    Args:
        lr_attr: The linear-referenced attribute
        start_frac: Start of query range (0.0-1.0)
        end_frac: End of query range (0.0-1.0)
        target_value: The value to calculate coverage for

    Returns:
        The coverage length as a fraction of the query range
    """
    if not lr_attr.ranges:
        return 0.0

    # Clamp to valid range
    start_frac = max(0.0, min(1.0, start_frac))
    end_frac = max(0.0, min(1.0, end_frac))

    if start_frac >= end_frac:
        return 0.0

    total_coverage = 0.0
    for attr_range in lr_attr.ranges:
        if attr_range.value == target_value:
            total_coverage += attr_range.intersection(start_frac, end_frac)

    return total_coverage


def extract_aligned_attributes(
    lr_data: dict[str, LinearReferencedAttribute],
    start_frac: float,
    end_frac: float,
) -> dict[str, Any]:
    """Extract majority-covering attributes for an aligned portion.

    This is the main entry point for getting attribute values during
    feature computation. Given a set of linear-referenced attributes
    and an alignment range, returns the majority-covering value for
    each attribute.

    Args:
        lr_data: Dict mapping attribute names to LinearReferencedAttribute
            e.g., {"name": ..., "subclass": ..., "level": ..., "road_flags": ...}
        start_frac: From alignment.overture_start_frac (0.0-1.0)
        end_frac: From alignment.overture_end_frac (0.0-1.0)

    Returns:
        Dict mapping attribute names to their majority-covering values
        e.g., {"name": "Oak St", "subclass": "residential", "level": 0, "road_flags": []}
    """
    result: dict[str, Any] = {}
    for attr_name, lr_attr in lr_data.items():
        result[attr_name] = extract_majority(lr_attr, start_frac, end_frac)
    return result


def create_trivial_lr(value: Any) -> LinearReferencedAttribute:
    """Create a trivial linear-referenced attribute with a single uniform value.

    This is used for data sources that don't have linear-referenced attributes,
    creating a single range covering [0.0, 1.0] with the given value.

    Args:
        value: The attribute value

    Returns:
        A LinearReferencedAttribute with a single range
    """
    return LinearReferencedAttribute(
        ranges=[AttributeRange(start=0.0, end=1.0, value=value)],
        default_value=value,
    )
