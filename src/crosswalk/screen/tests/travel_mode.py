"""Travel mode classification for screen tests.

Uses the existing TRAFFIC_TIERS from features.semantic for consistency.
"""

from crosswalk.features.semantic import get_traffic_tier

# Default travel mode when road class is unknown or unmapped
DEFAULT_TRAVEL_MODE = "vehicle"


def get_travel_mode(road_class: str | None) -> str:
    """Determine travel mode from road class.

    Wraps get_traffic_tier from features.semantic, providing a default
    for unknown/unmapped road classes.

    Args:
        road_class: Road classification (e.g., "motorway", "footway")

    Returns:
        Travel mode: "vehicle", "bicycle", or "pedestrian"
    """
    tier = get_traffic_tier(road_class)
    if tier is None or tier == "neutral":
        return DEFAULT_TRAVEL_MODE
    return tier
