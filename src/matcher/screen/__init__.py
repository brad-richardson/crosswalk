"""Screen framework for validating unmatched target segments.

This module provides a pluggable test framework for screening unmatched
target segments using external context (water bodies, buildings, etc.).
Screen tests identify segments that should not be added to the network.
"""

from .base import (
    CandidateContext,
    ScreenOutcome,
    ScreenResult,
    ScreenTest,
    get_registered_tests,
    get_test,
    register_test,
)
from .context import (
    fetch_overture_buildings,
    fetch_overture_landcover,
    fetch_overture_water,
    get_building_union,
    get_landcover_union,
    get_water_union,
)
from .runner import ScreenReport, run_screen
from .tests import BuildingTest, LandcoverTest, WaterBodyTest

__all__ = [
    # Base classes
    "ScreenOutcome",
    "ScreenResult",
    "ScreenTest",
    "CandidateContext",
    # Registry
    "register_test",
    "get_registered_tests",
    "get_test",
    # Context fetchers
    "fetch_overture_water",
    "get_water_union",
    "fetch_overture_buildings",
    "get_building_union",
    "fetch_overture_landcover",
    "get_landcover_union",
    # Test implementations
    "WaterBodyTest",
    "BuildingTest",
    "LandcoverTest",
    # Runner
    "run_screen",
    "ScreenReport",
]
