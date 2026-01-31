"""Screen framework for post-scoring match validation.

This module provides a pluggable test framework for validating matches
using external context (water bodies, buildings, etc.). Screen tests can
provide both positive and negative signals to confirm or rule out matches.
"""

from .base import (
    MatchContext,
    ScreenOutcome,
    ScreenResult,
    ScreenTest,
    get_registered_tests,
    get_test,
    register_test,
)
from .context import (
    fetch_overture_buildings,
    fetch_overture_water,
    get_building_union,
    get_water_union,
)

__all__ = [
    # Base classes
    "ScreenOutcome",
    "ScreenResult",
    "ScreenTest",
    "MatchContext",
    # Registry
    "register_test",
    "get_registered_tests",
    "get_test",
    # Context fetchers
    "fetch_overture_water",
    "get_water_union",
    "fetch_overture_buildings",
    "get_building_union",
]
