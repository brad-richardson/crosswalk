"""Tests for the GeoJSON coordinate-rounding helper in crosswalk.cli.data."""

from __future__ import annotations

from crosswalk.cli.data import _round_geojson_coords


def test_round_point():
    result = _round_geojson_coords({"type": "Point", "coordinates": [1.123456789, 2.987654321]})
    assert result == {"type": "Point", "coordinates": [1.123457, 2.987654]}


def test_round_linestring():
    result = _round_geojson_coords(
        {"type": "LineString", "coordinates": [[1.111111111, 2.2], [3.3, 4.444444444]]}
    )
    assert result == {
        "type": "LineString",
        "coordinates": [[1.111111, 2.2], [3.3, 4.444444]],
    }


def test_round_geometry_collection():
    """GeometryCollection uses a ``geometries`` key, not ``coordinates``."""
    result = _round_geojson_coords(
        {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "LineString", "coordinates": [[1.1234567, 2.0], [3.0, 4.0]]},
                {"type": "Point", "coordinates": [5.7654321, 6.0]},
            ],
        }
    )
    assert result == {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "LineString", "coordinates": [[1.123457, 2.0], [3.0, 4.0]]},
            {"type": "Point", "coordinates": [5.765432, 6.0]},
        ],
    }


def test_coordinateless_geometry_returned_unchanged():
    """A geometry with no ``coordinates`` key must not crash."""
    geojson = {"type": "Point"}
    assert _round_geojson_coords(geojson) == {"type": "Point"}


def test_round_nested_geometry_collection():
    """A GeometryCollection nested inside another must recurse."""
    result = _round_geojson_coords(
        {
            "type": "GeometryCollection",
            "geometries": [
                {
                    "type": "GeometryCollection",
                    "geometries": [
                        {"type": "Point", "coordinates": [1.1234567, 2.7654321]},
                    ],
                },
            ],
        }
    )
    assert result == {
        "type": "GeometryCollection",
        "geometries": [
            {
                "type": "GeometryCollection",
                "geometries": [
                    {"type": "Point", "coordinates": [1.123457, 2.765432]},
                ],
            },
        ],
    }
