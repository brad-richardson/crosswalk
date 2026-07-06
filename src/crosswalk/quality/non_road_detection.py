"""Non-road feature detection for filtering plazas, barriers, and other non-road geometry.

Some datasets include road-like features that aren't actual roads:
- Plazas and squares (closed-loop outlines)
- Barriers and fences
- Parking lot markings
- Stairs and escalators

This module provides detection strategies to identify these features.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from ..config import DEFAULT_SNAP_TOLERANCE_M


@dataclass
class NonRoadDetectionReport:
    """Report from non-road feature detection."""

    total_features: int = 0
    closed_loops: int = 0
    high_compactness: int = 0
    type_code_matches: int = 0
    total_non_road: int = 0

    # Examples of detected features
    examples: list[dict] = field(default_factory=list)

    # Per-type breakdown
    type_code_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_features": self.total_features,
                "closed_loops": self.closed_loops,
                "high_compactness": self.high_compactness,
                "type_code_matches": self.type_code_matches,
                "total_non_road": self.total_non_road,
                "non_road_percentage": (
                    self.total_non_road / self.total_features * 100
                    if self.total_features > 0
                    else 0
                ),
            },
            "type_code_counts": self.type_code_counts,
            "examples": self.examples,
        }


def is_closed_loop(geometry: LineString, tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M) -> bool:
    """Detect if linestring forms a closed polygon-like shape.

    Plazas, squares, and roundabouts often form closed loops.
    Real road segments rarely start and end at the same point.

    Args:
        geometry: LineString geometry to check
        tolerance_m: Distance threshold to consider endpoints as matching (meters)

    Returns:
        True if the geometry forms a closed loop
    """
    if geometry is None or geometry.is_empty:
        return False

    if not isinstance(geometry, LineString):
        return False

    if len(geometry.coords) < 3:
        return False

    start = Point(geometry.coords[0])
    end = Point(geometry.coords[-1])

    return start.distance(end) < tolerance_m


def compute_compactness_ratio(geometry: LineString) -> float:
    """Compute compactness ratio for a closed linestring.

    High compactness suggests plaza/square outline, not road.

    Compactness = 4*pi * area / perimeter^2
    - A circle has compactness = 1 (maximum)
    - A square has compactness = pi/4 ~= 0.785
    - Roads have low compactness (long thin shapes)

    Args:
        geometry: LineString geometry (should be closed)

    Returns:
        Compactness ratio (0-1), or 0.0 if not a closed loop
    """
    if not is_closed_loop(geometry):
        return 0.0

    try:
        # Create polygon from linestring coords
        polygon = Polygon(geometry.coords)
        if not polygon.is_valid or polygon.area <= 0:
            return 0.0

        perimeter = geometry.length
        if perimeter <= 0:
            return 0.0

        compactness = 4 * math.pi * polygon.area / (perimeter**2)
        return min(compactness, 1.0)  # Cap at 1.0
    except Exception:
        return 0.0


def compute_length_to_area_ratio(geometry: LineString) -> float:
    """Compute length-to-enclosed-area ratio for closed features.

    Real roads: high length, no enclosed area (ratio = infinity/high)
    Plazas: perimeter length encloses significant area (ratio = low)

    Args:
        geometry: LineString geometry (should be closed)

    Returns:
        Length / sqrt(area) ratio, or 0.0 if not closed
    """
    if not is_closed_loop(geometry):
        return 0.0

    try:
        polygon = Polygon(geometry.coords)
        if not polygon.is_valid or polygon.area <= 0:
            return 0.0

        # Use length / sqrt(area) for scale-invariant ratio
        return geometry.length / math.sqrt(polygon.area)
    except Exception:
        return 0.0


def detect_non_road_features(
    gdf: gpd.GeoDataFrame,
    type_code_column: str | None = None,
    non_road_type_codes: set[str] | None = None,
    check_closed_loops: bool = True,
    compactness_threshold: float = 0.3,
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
) -> pd.Series:
    """Flag features that appear to be non-roads.

    Args:
        gdf: GeoDataFrame with road features
        type_code_column: Column containing type codes (e.g., source_tags.cd_tipo_logradouro)
        non_road_type_codes: Set of type codes to treat as non-road (e.g., {'PC', 'PQ', 'ES'})
        check_closed_loops: Enable geometry-based closed loop detection
        compactness_threshold: Compactness ratio above which to flag as non-road
        tolerance_m: Tolerance for closed loop detection (meters)

    Returns:
        Boolean Series: True = likely non-road
    """
    flags = pd.Series(False, index=gdf.index)

    if check_closed_loops:
        # Check for closed loops with high compactness
        is_closed = gdf.geometry.apply(lambda g: is_closed_loop(g, tolerance_m))
        compactness = gdf.geometry.apply(compute_compactness_ratio)
        flags |= is_closed & (compactness > compactness_threshold)

    # Dataset-specific type codes
    if type_code_column is not None and non_road_type_codes:
        # Handle nested column names like "source_tags.cd_tipo_logradouro"
        if "." in type_code_column:
            parts = type_code_column.split(".", 1)
            parent_col, key = parts

            if parent_col in gdf.columns:
                type_codes = gdf[parent_col].apply(
                    lambda x: x.get(key) if isinstance(x, dict) else None
                )
            else:
                type_codes = pd.Series(None, index=gdf.index)
        elif type_code_column in gdf.columns:
            type_codes = gdf[type_code_column]
        else:
            type_codes = pd.Series(None, index=gdf.index)

        # Convert codes to uppercase for comparison
        type_codes_upper = type_codes.fillna("").astype(str).str.upper()
        non_road_codes_upper = {code.upper() for code in non_road_type_codes}

        flags |= type_codes_upper.isin(non_road_codes_upper)

    return flags


def analyze_non_road_features(
    gdf: gpd.GeoDataFrame,
    type_code_column: str | None = None,
    non_road_type_codes: set[str] | None = None,
    check_closed_loops: bool = True,
    compactness_threshold: float = 0.3,
    tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    max_examples: int = 20,
    id_column: str = "id",
    name_column: str | None = "names",
) -> NonRoadDetectionReport:
    """Analyze a GeoDataFrame for non-road features and generate a report.

    Args:
        gdf: GeoDataFrame with road features
        type_code_column: Column containing type codes
        non_road_type_codes: Set of type codes to treat as non-road
        check_closed_loops: Enable geometry-based closed loop detection
        compactness_threshold: Compactness ratio above which to flag as non-road
        tolerance_m: Tolerance for closed loop detection (meters)
        max_examples: Maximum number of examples to include
        id_column: Column name for feature IDs
        name_column: Column name for feature names

    Returns:
        NonRoadDetectionReport with analysis results
    """
    report = NonRoadDetectionReport(total_features=len(gdf))

    if len(gdf) == 0:
        return report

    # Detect closed loops
    if check_closed_loops:
        is_closed = gdf.geometry.apply(lambda g: is_closed_loop(g, tolerance_m))
        compactness = gdf.geometry.apply(compute_compactness_ratio)
        high_compactness = is_closed & (compactness > compactness_threshold)

        report.closed_loops = int(is_closed.sum())
        report.high_compactness = int(high_compactness.sum())

    # Detect by type codes
    type_code_flags = pd.Series(False, index=gdf.index)
    if type_code_column is not None and non_road_type_codes:
        # Handle nested column names
        if "." in type_code_column:
            parts = type_code_column.split(".", 1)
            parent_col, key = parts

            if parent_col in gdf.columns:
                type_codes = gdf[parent_col].apply(
                    lambda x: x.get(key) if isinstance(x, dict) else None
                )
            else:
                type_codes = pd.Series(None, index=gdf.index)
        elif type_code_column in gdf.columns:
            type_codes = gdf[type_code_column]
        else:
            type_codes = pd.Series(None, index=gdf.index)

        # Count type codes
        for code in type_codes.dropna().unique():
            code_str = str(code).upper()
            count = int((type_codes.astype(str).str.upper() == code_str).sum())
            if count > 0:
                report.type_code_counts[code_str] = count

        # Flag non-road type codes
        type_codes_upper = type_codes.fillna("").astype(str).str.upper()
        non_road_codes_upper = {code.upper() for code in non_road_type_codes}
        type_code_flags = type_codes_upper.isin(non_road_codes_upper)
        report.type_code_matches = int(type_code_flags.sum())

    # Combine all flags
    all_flags = detect_non_road_features(
        gdf,
        type_code_column=type_code_column,
        non_road_type_codes=non_road_type_codes,
        check_closed_loops=check_closed_loops,
        compactness_threshold=compactness_threshold,
        tolerance_m=tolerance_m,
    )
    report.total_non_road = int(all_flags.sum())

    # Collect examples
    flagged_indices = gdf.index[all_flags].tolist()[:max_examples]
    for idx in flagged_indices:
        row = gdf.loc[idx]
        geom = row.geometry

        example = {
            "id": str(row[id_column]) if id_column in gdf.columns else str(idx),
            "is_closed": bool(is_closed_loop(geom, tolerance_m)) if check_closed_loops else None,
            "compactness": (
                round(compute_compactness_ratio(geom), 3) if check_closed_loops else None
            ),
            "length_m": round(geom.length, 1) if geom else None,
        }

        if name_column and name_column in gdf.columns:
            example["name"] = row[name_column]

        if type_code_column:
            if "." in type_code_column:
                parts = type_code_column.split(".", 1)
                parent_col, key = parts
                if parent_col in gdf.columns:
                    val = row[parent_col]
                    example["type_code"] = val.get(key) if isinstance(val, dict) else None
            elif type_code_column in gdf.columns:
                example["type_code"] = row[type_code_column]

        report.examples.append(example)

    return report


def format_non_road_report(report: NonRoadDetectionReport) -> str:
    """Format non-road detection report for console output."""
    lines = []
    lines.append(f"=== Non-Road Feature Detection ({report.total_features:,} features) ===")
    lines.append("")

    pct = report.total_non_road / report.total_features * 100 if report.total_features > 0 else 0
    lines.append(f"Total non-road features detected: {report.total_non_road:,} ({pct:.1f}%)")
    lines.append("")

    lines.append("Detection breakdown:")
    lines.append(f"  Closed loops: {report.closed_loops:,}")
    lines.append(f"  High compactness (>0.3): {report.high_compactness:,}")
    lines.append(f"  Type code matches: {report.type_code_matches:,}")
    lines.append("")

    if report.type_code_counts:
        lines.append("Type code distribution:")
        sorted_codes = sorted(report.type_code_counts.items(), key=lambda x: -x[1])
        for code, count in sorted_codes[:15]:
            lines.append(f"  {code}: {count:,}")
        lines.append("")

    if report.examples:
        lines.append(f"Example non-road features (first {len(report.examples)}):")
        for ex in report.examples[:10]:
            name = ex.get("name", "unnamed")
            if isinstance(name, dict):
                name = name.get("primary", "unnamed")
            lines.append(
                f"  - {ex['id']}: {name} "
                f"(closed={ex.get('is_closed')}, compactness={ex.get('compactness')}, "
                f"type={ex.get('type_code')})"
            )

    return "\n".join(lines)


# Common non-road type codes by region/dataset
# Add to this as we discover more
KNOWN_NON_ROAD_TYPE_CODES = {
    # Brazil / São Paulo
    "br_sao_paulo": {
        "PC",  # Praça (Plaza/Square)
        "PQ",  # Parque (Park)
        "ES",  # Escada (Stairs)
        "ESC",  # Escadaria (Stairway)
        "LG",  # Largo (Small square)
        "BC",  # Beco (Alley - sometimes non-vehicular)
        "CM",  # Caminho (Path - pedestrian)
        "TN",  # Túnel (Tunnel - often separate from road)
    },
}
