"""Screen tests for match validation.

Import this module to register all built-in screen tests.
"""

from .building_test import BuildingTest
from .fringe_test import FringeTest, compute_reference_coverage, filter_fringe_segments
from .landcover_test import LandcoverTest
from .travel_mode import get_travel_mode
from .water_body_test import WaterBodyTest

__all__ = [
    "WaterBodyTest",
    "BuildingTest",
    "LandcoverTest",
    "FringeTest",
    "compute_reference_coverage",
    "filter_fringe_segments",
    "get_travel_mode",
]
