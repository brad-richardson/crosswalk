"""Network integration module.

Integrates unmatched target segments into the reference network,
handles multi-dataset merging with priority-based conflict resolution,
and identifies orphan components for QA review.
"""

from .combiner import combine_networks, separate_matched_unmatched
from .filters import detect_near_duplicates, filter_by_road_class, filter_short_segments
from .orphan_detector import (
    compute_reference_coverage,
    detect_orphan_components,
    detect_orphans_by_proximity,
    filter_fringe_segments,
    propagate_transitive_connectivity,
)
from .output import load_integration_result, write_integration_outputs
from .pipeline import TargetConfig, run_integration_from_config, run_integration_pipeline
from .provenance import (
    ComponentStatus,
    DroppedSegment,
    EdgeSource,
    IntegrationResult,
    IntegrationStatistics,
    TargetInput,
)

__all__ = [
    # Provenance
    "EdgeSource",
    "ComponentStatus",
    "TargetInput",
    "DroppedSegment",
    "IntegrationStatistics",
    "IntegrationResult",
    # Pipeline
    "TargetConfig",
    "run_integration_pipeline",
    "run_integration_from_config",
    # Combiner
    "combine_networks",
    "separate_matched_unmatched",
    # Filters
    "filter_short_segments",
    "detect_near_duplicates",
    "filter_by_road_class",
    # Orphan detection
    "detect_orphan_components",
    "detect_orphans_by_proximity",
    "propagate_transitive_connectivity",
    "compute_reference_coverage",
    "filter_fringe_segments",
    # Output
    "write_integration_outputs",
    "load_integration_result",
]
