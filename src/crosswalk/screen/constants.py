"""Constants for screen tests.

Centralized configuration for tuning screen test behavior.
"""

# =============================================================================
# MINIMUM AREA THRESHOLDS (square meters)
# =============================================================================
# Polygons smaller than these are filtered out to reduce noise

MIN_WATER_AREA_M2 = 100.0  # Small ponds, drainage
MIN_BUILDING_AREA_M2 = 20.0  # Sheds, small structures
MIN_LANDCOVER_AREA_M2 = 50.0  # Small patches

# =============================================================================
# BUFFER DISTANCES BY TRAVEL MODE (meters)
# =============================================================================
# Roads must not intersect buffered polygons. Larger buffers = stricter checks.
# Travel modes: "vehicle", "bicycle", "pedestrian" (from features.semantic)

# Water bodies - roads shouldn't be at water's edge
WATER_BUFFER_M: dict[str, float] = {
    "vehicle": 5.0,
    "bicycle": 2.0,
    "pedestrian": 1.0,
}

# Buildings - some tolerance for covered passages, arcades
BUILDING_BUFFER_M: dict[str, float] = {
    "vehicle": 3.0,
    "bicycle": 1.5,
    "pedestrian": 0.5,
}

# Landcover (wetlands, sports fields) - similar to buildings
LANDCOVER_BUFFER_M: dict[str, float] = {
    "vehicle": 3.0,
    "bicycle": 1.5,
    "pedestrian": 0.5,
}

# =============================================================================
# RESTRICTED LANDCOVER SUBTYPES
# =============================================================================
# Landcover types that roads should never cross

RESTRICTED_LANDCOVER_SUBTYPES = {
    # Wetlands - similar to water bodies
    "wetland",
    "marsh",
    "swamp",
    "bog",
    # Sports surfaces - no roads through playing fields
    "pitch",
    "sports_centre",
    "stadium",
    "track",
    "golf_course",
}

# =============================================================================
# FRINGE DETECTION (reference coverage)
# =============================================================================
# Segments outside the reference network coverage area are "fringe" segments
# that may be false positives at data boundaries.

FRINGE_BUFFER_M = 50.0  # Buffer around reference coverage polygon
FRINGE_MIN_INSIDE_LENGTH_M = 10.0  # Minimum length inside coverage to pass
FRINGE_HULL_RATIO = 0.3  # Concave hull ratio (0=convex, 1=very tight)
