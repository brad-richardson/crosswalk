"""Pytest configuration and fixtures."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely import LineString, Point


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_cross() -> gpd.GeoDataFrame:
    """Two crossing lines forming an X."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "name": ["Line A", "Line B"],
            "geometry": [
                LineString([(0, 0), (10, 10)]),
                LineString([(0, 10), (10, 0)]),
            ],
        },
        crs="EPSG:32610",  # UTM zone 10N (meters)
    )


@pytest.fixture
def simple_grid() -> gpd.GeoDataFrame:
    """4x4 grid of roads (5x5 = 25 intersections)."""
    lines = []
    ids = []
    names = []

    # Horizontal lines
    for i, y in enumerate(range(0, 50, 10)):
        lines.append(LineString([(0, y), (40, y)]))
        ids.append(f"h_{i}")
        names.append(f"Street {i+1}")

    # Vertical lines
    for i, x in enumerate(range(0, 50, 10)):
        lines.append(LineString([(x, 0), (x, 40)]))
        ids.append(f"v_{i}")
        names.append(f"Avenue {i+1}")

    return gpd.GeoDataFrame(
        {"id": ids, "name": names, "geometry": lines},
        crs="EPSG:32610",
    )


@pytest.fixture
def bridge_over_road() -> gpd.GeoDataFrame:
    """Two crossing lines where one is a bridge (should not intersect)."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "name": ["Main Street", "Highway Bridge"],
            "bridge": [False, True],
            "tunnel": [False, False],
            "layer": [0, 1],
            "geometry": [
                LineString([(0, 5), (10, 5)]),  # Ground level road
                LineString([(5, 0), (5, 10)]),  # Bridge crossing over
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def undershoot_lines() -> gpd.GeoDataFrame:
    """Lines with an undershoot that should be snapped."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "name": ["Main Road", "Side Street"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),  # Main road
                LineString([(50, 50), (50, 1.5)]),  # Side street undershoots by 1.5m
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def parallel_lines() -> gpd.GeoDataFrame:
    """Two parallel lines for feature testing."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "name": ["Road A", "Road B"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(0, 10), (100, 10)]),
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def perpendicular_lines() -> gpd.GeoDataFrame:
    """Two perpendicular lines for feature testing."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "name": ["East-West Road", "North-South Road"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(50, -50), (50, 50)]),
            ],
        },
        crs="EPSG:32610",
    )
