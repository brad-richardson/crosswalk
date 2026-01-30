"""Base classes and registry for falsification tests.

Falsification tests are negative signal tests that can override even high-confidence
ML matches. They shift from "can we confirm this matches?" to "can we rule this out?"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from shapely.geometry import LineString


class FalsificationOutcome(Enum):
    """Outcome of a falsification test."""

    PASS = "pass"  # No evidence of bad match
    FAIL = "fail"  # Clear evidence match is wrong (e.g., road in water)
    WARN = "warn"  # Suspicious but not conclusive
    SKIP = "skip"  # Test not applicable (missing data, etc.)


@dataclass
class FalsificationResult:
    """Result of running a falsification test on a match."""

    outcome: FalsificationOutcome
    test_name: str
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome == FalsificationOutcome.FAIL and not self.reason:
            raise ValueError("FAIL outcome requires a reason")


@dataclass
class MatchContext:
    """Context for a match being tested.

    Provides all geometry and metadata needed for falsification tests.
    """

    match_id: str
    ref_id: str
    target_id: str
    ref_geom: LineString
    target_geom: LineString
    confidence: float
    ref_attrs: dict[str, Any] = field(default_factory=dict)
    target_attrs: dict[str, Any] = field(default_factory=dict)


# Registry of all falsification tests
_FALSIFICATION_TESTS: dict[str, type["FalsificationTest"]] = {}


def register_test(cls: type["FalsificationTest"]) -> type["FalsificationTest"]:
    """Decorator to register a falsification test class.

    Example:
        @register_test
        class WaterBodyTest(FalsificationTest):
            name = "water_body"
            ...
    """
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"FalsificationTest {cls.__name__} must have a 'name' attribute")
    _FALSIFICATION_TESTS[cls.name] = cls
    return cls


def get_registered_tests() -> dict[str, type["FalsificationTest"]]:
    """Get all registered falsification tests."""
    return _FALSIFICATION_TESTS.copy()


def get_test(name: str) -> type["FalsificationTest"]:
    """Get a falsification test by name.

    Raises:
        KeyError: If test name not found
    """
    if name not in _FALSIFICATION_TESTS:
        available = ", ".join(sorted(_FALSIFICATION_TESTS.keys()))
        raise KeyError(f"Unknown falsification test: {name}. Available: {available}")
    return _FALSIFICATION_TESTS[name]


class FalsificationTest(ABC):
    """Base class for falsification tests.

    Falsification tests are designed to detect invalid matches that slip through
    the ML scoring pipeline. They use external context (water bodies, buildings,
    etc.) to identify matches that are geometrically plausible but semantically
    impossible.

    Implement this class to create a new falsification test:
    1. Set the `name` class attribute
    2. Implement `prepare()` to fetch context data
    3. Implement `test_match()` to check a single match
    4. Decorate the class with @register_test

    Example:
        @register_test
        class WaterBodyTest(FalsificationTest):
            name = "water_body"

            def prepare(self, bbox: tuple[float, float, float, float]) -> None:
                self.water_polygons = fetch_water_bodies(bbox)

            def test_match(self, ctx: MatchContext) -> FalsificationResult:
                intersection = ctx.target_geom.intersection(self.water_union)
                if intersection.length > threshold:
                    return FalsificationResult(
                        outcome=FalsificationOutcome.FAIL,
                        test_name=self.name,
                        reason="Road intersects water body",
                    )
                return FalsificationResult(
                    outcome=FalsificationOutcome.PASS,
                    test_name=self.name,
                )
    """

    name: str = ""  # Must be overridden by subclasses

    @abstractmethod
    def prepare(self, bbox: tuple[float, float, float, float]) -> None:
        """Prepare the test by fetching context data.

        Called once before testing matches. Implementations should fetch
        any external data needed (water bodies, buildings, etc.) and store
        it as instance attributes.

        Args:
            bbox: Bounding box as (xmin, ymin, xmax, ymax) in EPSG:4326
        """
        ...

    @abstractmethod
    def test_match(self, ctx: MatchContext) -> FalsificationResult:
        """Test a single match for falsification.

        Args:
            ctx: Match context with geometries and metadata

        Returns:
            FalsificationResult with outcome and details
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
