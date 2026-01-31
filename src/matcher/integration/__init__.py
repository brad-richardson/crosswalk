"""Network integration module.

Integrates unmatched target segments into the reference network,
handles multi-dataset merging with priority-based conflict resolution,
and identifies orphan components for QA review.

Architecture:
- Pre-screening (fringe, water, buildings) is handled by the screen module
- Integration (combining, connectivity) is handled by this module
- Post-integration analysis (islands, drift) is handled by post_integration module
"""

# Re-export fringe functions from screen module for backward compatibility
from ..screen.tests.fringe_test import compute_reference_coverage, filter_fringe_segments
from .combiner import combine_networks, separate_matched_unmatched
from .filters import detect_near_duplicates, filter_by_road_class, filter_short_segments
from .orphan_detector import (
    detect_orphan_components,
    detect_orphans_by_proximity,
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
