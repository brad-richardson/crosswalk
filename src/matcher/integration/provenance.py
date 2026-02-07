"""Provenance tracking for network integration.

Tracks the origin of edges through the integration pipeline,
supporting multi-dataset merging with priority-based conflict resolution.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import geopandas as gpd


class EdgeSource(Enum):
    """Origin of an edge in the integrated network."""

    REFERENCE = "reference"  # From Overture reference network (priority 0)
    TARGET_MATCHED = "target_matched"  # Target segment with confirmed match
    TARGET_UNMATCHED = "target_new"  # Target segment, no match (potential addition)


class ComponentStatus(Enum):
    """Status of a connected component after integration."""

    MAIN = "main"  # Connected to reference network
    DISCONNECTED = "disconnected"  # Not connected to reference network
    FILTERED = "filtered"  # Connected but insufficient net-new coverage


@dataclass
class TargetInput:
    """Input configuration for a target dataset to integrate.

    Attributes:
        name: Dataset identifier (e.g., "boston_streets", "boston_bikes")
        matched: GeoDataFrame of target segments that matched reference
        unmatched: GeoDataFrame of target segments without matches
        match_results: List of MatchResult objects from bridge file
        priority: Integration priority (lower = higher priority, 1 = highest)
    """

    name: str
    matched: gpd.GeoDataFrame
    unmatched: gpd.GeoDataFrame
    match_results: list[Any]  # list[MatchResult] - avoid circular import
    priority: int


@dataclass
class DroppedSegment:
    """Record of a segment dropped during conflict resolution.

    Attributes:
        original_id: Original ID from source dataset
        source_dataset: Name of the source dataset
        source_type: EdgeSource type
        geometry: The segment geometry
        dropped_reason: Why it was dropped
        overlapping_edge_id: Edge ID of higher-priority segment that superseded this
        overlap_iou: The IoU score that triggered the drop
        priority: The priority of the dropped segment
    """

    original_id: Any
    source_dataset: str
    source_type: EdgeSource
    geometry: Any  # LineString
    dropped_reason: str
    overlapping_edge_id: int | None = None
    overlap_iou: float | None = None
    priority: int = 0


@dataclass
class IntegrationStatistics:
    """Statistics from the integration pipeline.

    Attributes:
        reference_edges: Number of edges from reference
        target_edges_matched: Number of matched target edges included
        target_edges_unmatched: Number of unmatched target edges included
        dropped_overlaps: Number of segments dropped due to overlap
        total_nodes: Total nodes after planarization
        total_edges: Total edges after planarization
        main_component_edges: Edges in main connected component
        disconnected_edges: Edges truly disconnected from network
        filtered_edges: Edges connected but with insufficient net-new coverage
        datasets_integrated: List of dataset names integrated
    """

    reference_edges: int = 0
    target_edges_matched: int = 0
    target_edges_unmatched: int = 0
    dropped_overlaps: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    main_component_edges: int = 0
    disconnected_edges: int = 0
    filtered_edges: int = 0
    datasets_integrated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "reference_edges": self.reference_edges,
            "target_edges_matched": self.target_edges_matched,
            "target_edges_unmatched": self.target_edges_unmatched,
            "dropped_overlaps": self.dropped_overlaps,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "main_component_edges": self.main_component_edges,
            "disconnected_edges": self.disconnected_edges,
            "filtered_edges": self.filtered_edges,
            "datasets_integrated": self.datasets_integrated,
        }


@dataclass
class IntegrationResult:
    """Result of the integration pipeline.

    Attributes:
        nodes: GeoDataFrame of all nodes with component annotations
        edges: GeoDataFrame of all edges with provenance and component annotations
        disconnected_edges: GeoDataFrame of edges truly disconnected from network
        filtered_edges: GeoDataFrame of edges connected but insufficient net-new coverage
        dropped_overlaps: GeoDataFrame of segments dropped due to priority conflicts
        net_new_edges: GeoDataFrame of net-new geometry portions (for visualization)
        statistics: Integration statistics
        created_at: Timestamp of integration
    """

    nodes: gpd.GeoDataFrame
    edges: gpd.GeoDataFrame
    disconnected_edges: gpd.GeoDataFrame
    filtered_edges: gpd.GeoDataFrame
    dropped_overlaps: gpd.GeoDataFrame
    statistics: IntegrationStatistics
    net_new_edges: gpd.GeoDataFrame | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# Column names for provenance tracking (prefixed with _ to distinguish from source data)
PROVENANCE_COLUMNS = {
    "_source": "EdgeSource enum value",
    "_original_id": "Original ID from source dataset",
    "_source_dataset": "Name of source dataset (overture, boston_streets, etc.)",
    "_priority": "Integration priority (0 = reference, 1+ = targets)",
    "_match_ref_id": "GERS ID if this is a matched target segment",
    "_match_confidence": "Match confidence score if applicable",
}

# Column names for component tracking
COMPONENT_COLUMNS = {
    "component_id": "Connected component ID",
    "component_status": "main, disconnected, or filtered",
    "component_size": "Number of edges in the component",
}
