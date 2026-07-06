"""Constants for post-integration analysis.

Centralized configuration for tuning post-integration behavior.
"""

# =============================================================================
# TOPOLOGY REPAIR
# =============================================================================

SNAP_TOLERANCE_M = 5.0  # Distance tolerance for endpoint snapping (meters)

# =============================================================================
# ISLAND DETECTION
# =============================================================================

ISLAND_SNAP_TOLERANCE_M = 5.0  # Distance to consider endpoints connected (meters)
SMALL_CLUSTER_THRESHOLD = 5  # Max edges for "small" cluster classification
FAR_DISTANCE_M = 100.0  # Distance threshold for "far from main network"
