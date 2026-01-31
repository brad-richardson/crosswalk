"""Quality fingerprint dataclass for road network datasets.

Captures comprehensive metrics about a dataset's quality, topology,
and completeness for comparison and tracking.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> datetime:
    """Return current UTC time (for default_factory)."""
    return datetime.now(UTC)


@dataclass
class QualityFingerprint:
    """Quality fingerprint for a road network dataset.

    Captures metrics across several categories:
    - Basic stats: segment count, total length
    - Geometry: vertex density, invalid geometries
    - Topology: connectivity, dead ends, components
    - Attributes: name coverage, class distribution
    - Screen: results from screen tests (if run)
    """

    # Dataset identification
    dataset_name: str
    timestamp: datetime = field(default_factory=_utc_now)

    # Basic statistics
    total_segments: int = 0
    total_length_m: float = 0.0

    # Geometry metrics
    vertex_density_mean: float = 0.0
    vertex_density_std: float = 0.0
    invalid_geometry_count: int = 0

    # Topology metrics
    island_count: int = 0
    dead_end_count: int = 0
    dead_end_ratio: float = 0.0
    connected_components: int = 0
    largest_component_ratio: float = 0.0

    # Attribute metrics
    name_coverage_ratio: float = 0.0
    class_distribution: dict[str, int] = field(default_factory=dict)

    # Screen metrics (populated if screen tests were run)
    screen_fail_count: int = 0
    screen_fail_rate: float = 0.0
    screen_warn_count: int = 0
    screen_warn_rate: float = 0.0

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert fingerprint to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "total_segments": self.total_segments,
            "total_length_m": round(self.total_length_m, 2),
            "vertex_density_mean": round(self.vertex_density_mean, 4),
            "vertex_density_std": round(self.vertex_density_std, 4),
            "invalid_geometry_count": self.invalid_geometry_count,
            "island_count": self.island_count,
            "dead_end_count": self.dead_end_count,
            "dead_end_ratio": round(self.dead_end_ratio, 4),
            "connected_components": self.connected_components,
            "largest_component_ratio": round(self.largest_component_ratio, 4),
            "name_coverage_ratio": round(self.name_coverage_ratio, 4),
            "class_distribution": self.class_distribution,
            "screen_fail_count": self.screen_fail_count,
            "screen_fail_rate": round(self.screen_fail_rate, 4),
            "screen_warn_count": self.screen_warn_count,
            "screen_warn_rate": round(self.screen_warn_rate, 4),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityFingerprint":
        """Create fingerprint from dictionary."""
        # Parse timestamp
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            dataset_name=data.get("dataset_name", "unknown"),
            timestamp=timestamp,
            total_segments=data.get("total_segments", 0),
            total_length_m=data.get("total_length_m", 0.0),
            vertex_density_mean=data.get("vertex_density_mean", 0.0),
            vertex_density_std=data.get("vertex_density_std", 0.0),
            invalid_geometry_count=data.get("invalid_geometry_count", 0),
            island_count=data.get("island_count", 0),
            dead_end_count=data.get("dead_end_count", 0),
            dead_end_ratio=data.get("dead_end_ratio", 0.0),
            connected_components=data.get("connected_components", 0),
            largest_component_ratio=data.get("largest_component_ratio", 0.0),
            name_coverage_ratio=data.get("name_coverage_ratio", 0.0),
            class_distribution=data.get("class_distribution", {}),
            screen_fail_count=data.get("screen_fail_count", 0),
            screen_fail_rate=data.get("screen_fail_rate", 0.0),
            screen_warn_count=data.get("screen_warn_count", 0),
            screen_warn_rate=data.get("screen_warn_rate", 0.0),
            metadata=data.get("metadata", {}),
        )
