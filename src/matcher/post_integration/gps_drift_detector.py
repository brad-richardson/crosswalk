"""GPS drift pattern detection for post-integration analysis.

Detects common GPS recording artifacts in road geometries:
- Zigzag patterns: High vertex density with alternating angles
- Spikes: Single vertices far from the regression line
- Small loops: U-turn artifacts from track recording
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.geometry import LineString


class DriftPattern(Enum):
    """Types of GPS drift patterns."""

    ZIGZAG = "zigzag"  # High vertex density with alternating angles
    SPIKE = "spike"  # Single vertex far from regression line
    SMALL_LOOP = "small_loop"  # U-turn artifact


class DriftSeverity(Enum):
    """Severity of detected drift pattern."""

    MINOR = "minor"  # Can be simplified automatically
    SEVERE = "severe"  # Should be dropped or manually reviewed


@dataclass
class DriftDetection:
    """A detected GPS drift pattern in a geometry."""

    edge_id: str | int
    pattern: DriftPattern
    severity: DriftSeverity
    location_index: int | None = None  # Vertex index where pattern was detected
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftDetectionResult:
    """Result of GPS drift detection analysis."""

    total_edges: int
    edges_with_drift: int
    detections: list[DriftDetection]

    # Pattern counts
    zigzag_count: int = 0
    spike_count: int = 0
    loop_count: int = 0

    # Severity counts
    minor_count: int = 0
    severe_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_edges": self.total_edges,
            "edges_with_drift": self.edges_with_drift,
            "zigzag_count": self.zigzag_count,
            "spike_count": self.spike_count,
            "loop_count": self.loop_count,
            "minor_count": self.minor_count,
            "severe_count": self.severe_count,
            "detections": [
                {
                    "edge_id": str(d.edge_id),
                    "pattern": d.pattern.value,
                    "severity": d.severity.value,
                    "location_index": d.location_index,
                    "details": d.details,
                }
                for d in self.detections
            ],
        }


# Detection thresholds
ZIGZAG_MIN_VERTEX_DENSITY = 0.5  # vertices per meter
ZIGZAG_MIN_ALTERNATIONS = 3  # minimum consecutive direction changes
ZIGZAG_MAX_SEGMENT_LENGTH_M = 5.0  # max segment length for zigzag pattern

SPIKE_DISTANCE_THRESHOLD = 10.0  # meters from regression line
SPIKE_MIN_ANGLE_DEG = 90.0  # minimum angle at spike vertex

LOOP_MAX_LENGTH_M = 50.0  # maximum loop length
LOOP_CLOSURE_THRESHOLD_M = 10.0  # distance for loop closure detection


def detect_gps_drift(
    edges_gdf: gpd.GeoDataFrame,
    id_column: str | None = None,
    zigzag_vertex_density: float = ZIGZAG_MIN_VERTEX_DENSITY,
    spike_distance_m: float = SPIKE_DISTANCE_THRESHOLD,
    loop_max_length_m: float = LOOP_MAX_LENGTH_M,
) -> DriftDetectionResult:
    """Detect GPS drift patterns in road geometries.

    Analyzes each edge for common GPS recording artifacts and classifies
    them by pattern type and severity.

    Args:
        edges_gdf: GeoDataFrame with road edges (LineString geometries)
        id_column: Column to use as edge ID (auto-detected if None)
        zigzag_vertex_density: Min vertices/meter for zigzag detection
        spike_distance_m: Distance threshold for spike detection
        loop_max_length_m: Max length for loop detection

    Returns:
        DriftDetectionResult with pattern analysis
    """
    if len(edges_gdf) == 0:
        return DriftDetectionResult(
            total_edges=0,
            edges_with_drift=0,
            detections=[],
        )

    # Ensure metric CRS for accurate calculations
    if edges_gdf.crs is None:
        edges_gdf = edges_gdf.set_crs("EPSG:4326")

    metric_crs = edges_gdf.estimate_utm_crs()
    edges_metric = edges_gdf.to_crs(metric_crs)

    # Determine ID column
    id_col = id_column or _get_id_column(edges_gdf)

    detections: list[DriftDetection] = []
    edges_with_drift: set[str | int] = set()

    logger.info(f"Analyzing {len(edges_metric)} edges for GPS drift patterns")

    for _, row in edges_metric.iterrows():
        geom = row.geometry
        edge_id = row[id_col]

        if geom is None or geom.is_empty or not isinstance(geom, LineString):
            continue

        coords = np.array(geom.coords)
        if len(coords) < 3:
            continue

        # Check for zigzag pattern
        zigzag = _detect_zigzag(coords, geom.length, zigzag_vertex_density, edge_id)
        if zigzag:
            detections.append(zigzag)
            edges_with_drift.add(edge_id)

        # Check for spikes
        spikes = _detect_spikes(coords, spike_distance_m, edge_id)
        for spike in spikes:
            detections.append(spike)
            edges_with_drift.add(edge_id)

        # Check for small loops
        loop = _detect_small_loop(coords, loop_max_length_m, edge_id)
        if loop:
            detections.append(loop)
            edges_with_drift.add(edge_id)

    # Compute counts
    zigzag_count = sum(1 for d in detections if d.pattern == DriftPattern.ZIGZAG)
    spike_count = sum(1 for d in detections if d.pattern == DriftPattern.SPIKE)
    loop_count = sum(1 for d in detections if d.pattern == DriftPattern.SMALL_LOOP)
    minor_count = sum(1 for d in detections if d.severity == DriftSeverity.MINOR)
    severe_count = sum(1 for d in detections if d.severity == DriftSeverity.SEVERE)

    logger.info(
        f"GPS drift analysis: {len(edges_with_drift)} edges with drift, "
        f"{zigzag_count} zigzag, {spike_count} spike, {loop_count} loop"
    )

    return DriftDetectionResult(
        total_edges=len(edges_gdf),
        edges_with_drift=len(edges_with_drift),
        detections=detections,
        zigzag_count=zigzag_count,
        spike_count=spike_count,
        loop_count=loop_count,
        minor_count=minor_count,
        severe_count=severe_count,
    )


def _detect_zigzag(
    coords: np.ndarray,
    total_length: float,
    min_vertex_density: float,
    edge_id: str | int,
) -> DriftDetection | None:
    """Detect zigzag pattern in coordinates.

    Zigzag is characterized by:
    - High vertex density (many vertices per meter)
    - Alternating turn directions (left-right-left pattern)
    - Short segment lengths
    """
    n_vertices = len(coords)
    if total_length <= 0:
        return None

    vertex_density = n_vertices / total_length
    if vertex_density < min_vertex_density:
        return None

    # Calculate segment lengths
    diffs = np.diff(coords, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))

    # Check if segments are short enough
    if np.mean(segment_lengths) > ZIGZAG_MAX_SEGMENT_LENGTH_M:
        return None

    # Calculate turn angles and check for alternation
    if len(coords) < 4:
        return None

    # Compute cross products for turn direction
    turns = []
    for i in range(1, len(coords) - 1):
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        turns.append(np.sign(cross))

    # Count alternations
    alternations = 0
    for i in range(1, len(turns)):
        if turns[i] != 0 and turns[i - 1] != 0 and turns[i] != turns[i - 1]:
            alternations += 1

    if alternations < ZIGZAG_MIN_ALTERNATIONS:
        return None

    # Classify severity based on vertex density
    severity = (
        DriftSeverity.SEVERE if vertex_density > 2 * min_vertex_density else DriftSeverity.MINOR
    )

    return DriftDetection(
        edge_id=edge_id,
        pattern=DriftPattern.ZIGZAG,
        severity=severity,
        details={
            "vertex_density": round(vertex_density, 3),
            "alternations": alternations,
            "mean_segment_length_m": round(float(np.mean(segment_lengths)), 2),
        },
    )


def _detect_spikes(
    coords: np.ndarray,
    distance_threshold_m: float,
    edge_id: str | int,
) -> list[DriftDetection]:
    """Detect spike artifacts in coordinates.

    A spike is a single vertex that deviates significantly from the
    overall trajectory of the line.
    """
    if len(coords) < 5:
        return []

    spikes = []

    # Fit a regression line to overall trajectory
    # Use simplified approach: check each vertex against its neighbors
    for i in range(2, len(coords) - 2):
        # Get surrounding context (2 vertices on each side)
        context = np.array([coords[i - 2], coords[i - 1], coords[i + 1], coords[i + 2]])
        context_mean = np.mean(context, axis=0)

        # Calculate deviation from context mean
        deviation = np.sqrt(np.sum((coords[i] - context_mean) ** 2))

        if deviation < distance_threshold_m:
            continue

        # Calculate angle at this vertex
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]

        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 > 0 and mag2 > 0:
            cos_angle = np.clip(dot / (mag1 * mag2), -1, 1)
            angle_deg = np.degrees(np.arccos(cos_angle))

            if angle_deg < SPIKE_MIN_ANGLE_DEG:
                continue

        severity = (
            DriftSeverity.SEVERE if deviation > 2 * distance_threshold_m else DriftSeverity.MINOR
        )

        spikes.append(
            DriftDetection(
                edge_id=edge_id,
                pattern=DriftPattern.SPIKE,
                severity=severity,
                location_index=i,
                details={
                    "deviation_m": round(float(deviation), 2),
                    "angle_deg": round(float(angle_deg), 1) if "angle_deg" in dir() else None,
                },
            )
        )

    return spikes


def _detect_small_loop(
    coords: np.ndarray,
    max_length_m: float,
    edge_id: str | int,
) -> DriftDetection | None:
    """Detect small loop artifacts (U-turns) in coordinates.

    A small loop occurs when the track backtracks on itself briefly,
    creating a small enclosed area.
    """
    if len(coords) < 6:
        return None

    # Look for points that are close to each other but separated in the sequence
    min_skip = 4  # minimum vertices to skip

    for i in range(len(coords) - min_skip):
        for j in range(i + min_skip, len(coords)):
            # Check if these points are close
            dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            if dist > LOOP_CLOSURE_THRESHOLD_M:
                continue

            # Calculate path length between these points
            path_coords = coords[i : j + 1]
            diffs = np.diff(path_coords, axis=0)
            path_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

            if path_length > max_length_m:
                continue

            # Found a small loop
            severity = (
                DriftSeverity.MINOR if path_length < max_length_m / 2 else DriftSeverity.SEVERE
            )

            return DriftDetection(
                edge_id=edge_id,
                pattern=DriftPattern.SMALL_LOOP,
                severity=severity,
                location_index=i,
                details={
                    "loop_start_index": i,
                    "loop_end_index": j,
                    "path_length_m": round(float(path_length), 2),
                    "closure_distance_m": round(float(dist), 2),
                },
            )

    return None


def _get_id_column(gdf: gpd.GeoDataFrame) -> str:
    """Determine the ID column for a GeoDataFrame."""
    for col in ["id", "ID", "edge_id"]:
        if col in gdf.columns:
            return col
    if gdf.index.name and gdf.index.name in gdf.columns:
        return gdf.index.name
    raise ValueError(
        f"Could not determine ID column. Expected one of ['id', 'ID', 'edge_id']. "
        f"Available columns: {list(gdf.columns)}"
    )
