"""Shared utilities for the matcher web UI."""


def round_geom(geom_mapping: dict, precision: int = 6) -> dict:
    """Round coordinates in a GeoJSON geometry mapping to reduce payload size."""

    def _round(coords):
        if isinstance(coords[0], (list, tuple)):
            return [_round(c) for c in coords]
        return [round(v, precision) for v in coords]

    return {**geom_mapping, "coordinates": _round(geom_mapping["coordinates"])}
