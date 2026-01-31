"""Screen tests for match validation.

Import this module to register all built-in screen tests.
"""

from .building_test import BuildingTest
from .travel_mode import get_travel_mode
from .water_body_test import WaterBodyTest

__all__ = [
    "WaterBodyTest",
    "BuildingTest",
    "get_travel_mode",
]
