"""Base classes and registry for screen tests.

Screen tests use external context (water bodies, buildings, etc.) to validate
or invalidate matches with positive or negative signals.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from shapely.geometry import LineString


class ScreenOutcome(Enum):
    """Outcome of a screen test."""

    PASS = "pass"  # No evidence of bad match
    FAIL = "fail"  # Clear evidence match is wrong (e.g., road in water)
    WARN = "warn"  # Suspicious but not conclusive
    SKIP = "skip"  # Test not applicable (missing data, etc.)


@dataclass
class ScreenResult:
    """Result of running a screen test on a match."""

    outcome: ScreenOutcome
    test_name: str
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome != ScreenOutcome.PASS:
            if not self.reason or not self.reason.strip():
                raise ValueError(f"{self.outcome.name} outcome requires a reason")


@dataclass
class CandidateContext:
    """Context for an unmatched target segment being screened.

    These are segments from the target dataset that didn't match anything
    in the reference network. Screen tests determine if they're valid
    candidates for addition to the network.
    """

    target_id: str
    target_geom: LineString
    road_class: str | None = None  # e.g., "motorway", "residential", "footway"
    target_attrs: dict[str, Any] = field(default_factory=dict)


# Registry of all screen tests
_SCREEN_TESTS: dict[str, type["ScreenTest"]] = {}


def register_test(cls: type["ScreenTest"]) -> type["ScreenTest"]:
    """Decorator to register a screen test class.

    Example:
        @register_test
        class WaterBodyTest(ScreenTest):
            name = "water_body"
            ...
    """
    if not hasattr(cls, "name") or not isinstance(cls.name, str) or not cls.name.strip():
        raise ValueError(f"ScreenTest {cls.__name__} must have a non-empty string 'name' attribute")
    _SCREEN_TESTS[cls.name] = cls
    return cls


def get_registered_tests() -> dict[str, type["ScreenTest"]]:
    """Get all registered screen tests."""
    return _SCREEN_TESTS.copy()


def get_test(name: str) -> type["ScreenTest"]:
    """Get a screen test by name.

    Raises:
        KeyError: If test name not found
    """
    if name not in _SCREEN_TESTS:
        available = ", ".join(sorted(_SCREEN_TESTS.keys()))
        raise KeyError(f"Unknown screen test: {name}. Available: {available}")
    return _SCREEN_TESTS[name]


class ScreenTest(ABC):
    """Base class for screen tests.

    Screen tests validate unmatched target segments using external context
    (water bodies, buildings, etc.) to identify segments that should not
    be added to the network.

    Implement this class to create a new screen test:
    1. Set the `name` class attribute
    2. Implement `prepare()` to fetch context data
    3. Implement `test_candidate()` to check a single candidate
    4. Decorate the class with @register_test

    Example:
        @register_test
        class WaterBodyTest(ScreenTest):
            name = "water_body"

            def prepare(self, bbox: tuple[float, float, float, float]) -> None:
                self.water_union = fetch_water_bodies(bbox)

            def test_candidate(self, ctx: CandidateContext) -> ScreenResult:
                intersection = ctx.target_geom.intersection(self.water_union)
                if intersection.length > threshold:
                    return ScreenResult(
                        outcome=ScreenOutcome.FAIL,
                        test_name=self.name,
                        reason="Road intersects water body",
                    )
                return ScreenResult(
                    outcome=ScreenOutcome.PASS,
                    test_name=self.name,
                )
    """

    name: str = ""  # Must be overridden by subclasses

    @abstractmethod
    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Prepare the test by fetching context data.

        Called once before testing candidates. Implementations should fetch
        any external data needed (water bodies, buildings, etc.) and store
        it as instance attributes.

        Args:
            bbox: Bounding box as (xmin, ymin, xmax, ymax) in EPSG:4326
        """
        ...

    @abstractmethod
    def test_candidate(self, ctx: CandidateContext) -> ScreenResult:
        """Test a single candidate segment.

        Args:
            ctx: Candidate context with geometry and metadata

        Returns:
            ScreenResult with outcome and details
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
