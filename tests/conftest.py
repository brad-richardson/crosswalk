"""Pytest configuration and fixtures."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely import LineString

# =============================================================================
# Label Data Fixtures
# =============================================================================


@pytest.fixture
def labels_dir() -> Path:
    """Return the path to the labels directory."""
    return Path(__file__).parent.parent / "labels"


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
        names.append(f"Street {i + 1}")

    # Vertical lines
    for i, x in enumerate(range(0, 50, 10)):
        lines.append(LineString([(x, 0), (x, 40)]))
        ids.append(f"v_{i}")
        names.append(f"Avenue {i + 1}")

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
    """Lines with an undershoot that should be snapped.

    The side street ends 1.5m from the main road. With snap_tolerance=2.0,
    it should be snapped to create a connected network.
    """
    return gpd.GeoDataFrame(
        {
            "local_id": [1, 2],
            "name": ["Main Road", "Side Street"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),  # Main road along x-axis
                LineString([(50, 1.5), (50, 50)]),  # Side street undershoots by 1.5m
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


@pytest.fixture
def t_junction() -> gpd.GeoDataFrame:
    """T-junction where one line ends on another (touches but doesn't cross)."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "name": ["Main Road", "Side Street"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),  # Main road
                LineString([(50, 50), (50, 0)]),  # Side street ends at main road
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def bridge_with_string_values() -> gpd.GeoDataFrame:
    """Bridge with OSM-style string values like 'yes', 'no'."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Ground Road", "Bridge", "Tunnel Exit"],
            "bridge": ["no", "yes", "no"],  # OSM style strings
            "tunnel": ["no", "no", "no"],
            "layer": [0, 1, 0],
            "geometry": [
                LineString([(0, 5), (10, 5)]),  # Ground level
                LineString([(5, 0), (5, 10)]),  # Bridge over
                LineString([(7, 0), (7, 10)]),  # Ground level, near bridge
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def lines_epsg4326() -> gpd.GeoDataFrame:
    """Lines in EPSG:4326 (geographic CRS) for auto-projection testing.

    These are roughly in Portland, OR area.
    """
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "name": ["Line A", "Line B"],
            "geometry": [
                LineString([(-122.6, 45.5), (-122.5, 45.5)]),  # ~8km line
                LineString([(-122.55, 45.45), (-122.55, 45.55)]),  # ~11km line
            ],
        },
        crs="EPSG:4326",
    )


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest.fixture
def reference_gdf() -> gpd.GeoDataFrame:
    """Reference network with two connected segments."""
    return gpd.GeoDataFrame(
        {
            "id": ["ref_1", "ref_2"],
            "name": ["Main St", "Oak Ave"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),  # Horizontal
                LineString([(100, 0), (100, 100)]),  # Vertical, connects at (100, 0)
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def target_matched_gdf() -> gpd.GeoDataFrame:
    """Target segments that match reference."""
    return gpd.GeoDataFrame(
        {
            "local_id": ["t_1"],
            "name": ["Main Street"],
            "geometry": [
                LineString([(0, 5), (100, 5)]),  # Parallel to Main St
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def target_unmatched_gdf() -> gpd.GeoDataFrame:
    """Target segments without matches."""
    return gpd.GeoDataFrame(
        {
            "local_id": ["t_new_1", "t_orphan"],
            "name": ["New Rd", "Orphan Rd"],
            "geometry": [
                LineString([(50, 0), (50, 50)]),  # Connects to Main St
                LineString([(200, 200), (250, 250)]),  # Disconnected orphan
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def match_results():
    """Match results for target_matched."""

    class MockMatchResult:
        def __init__(self, ref_id, target_id, confidence):
            self.ref_id = ref_id
            self.target_id = target_id
            self.confidence = confidence

    return [
        MockMatchResult("ref_1", "t_1", 0.9),
    ]
