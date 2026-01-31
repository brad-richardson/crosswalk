"""Travel mode classification for screen tests."""

# Road classes by travel mode
VEHICLE_CLASSES = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "service",
    "unclassified", "living_street",
}
BIKE_CLASSES = {"cycleway"}
PEDESTRIAN_CLASSES = {"footway", "pedestrian", "path", "steps", "bridleway"}


def get_travel_mode(road_class: str | None) -> str:
    """Determine travel mode from road class.

    Args:
        road_class: Road classification (e.g., "motorway", "footway")

    Returns:
        Travel mode: "vehicle", "bike", or "pedestrian"
    """
    if road_class is None:
        return "vehicle"  # Default assumption
    rc = road_class.lower()
    if rc in PEDESTRIAN_CLASSES:
        return "pedestrian"
    if rc in BIKE_CLASSES:
        return "bike"
    return "vehicle"
