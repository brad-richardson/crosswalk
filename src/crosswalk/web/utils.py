"""Shared utilities for the crosswalk web UI."""

# Coordinate precision for all GeoJSON/geometry output (6 decimals ≈ 11cm)
UI_GEOM_PRECISION = 6


def round_geom(geom_mapping: dict, precision: int = UI_GEOM_PRECISION) -> dict:
    """Round coordinates in a GeoJSON geometry mapping to reduce payload size."""

    def _round(coords):
        if isinstance(coords[0], (list, tuple)):
            return [_round(c) for c in coords]
        return [round(v, precision) for v in coords]

    return {**geom_mapping, "coordinates": _round(geom_mapping["coordinates"])}
