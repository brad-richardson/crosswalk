"""Falsification framework for post-scoring match validation.

This module provides a pluggable test framework for detecting invalid matches
that slip through the ML scoring pipeline. Tests use external context
(water bodies, buildings, etc.) to identify matches that are geometrically
plausible but semantically impossible.
"""

from .base import (
    FalsificationOutcome,
    FalsificationResult,
    FalsificationTest,
    MatchContext,
    get_registered_tests,
    get_test,
    register_test,
)

__all__ = [
    # Base classes
    "FalsificationOutcome",
    "FalsificationResult",
    "FalsificationTest",
    "MatchContext",
    # Registry
    "register_test",
    "get_registered_tests",
    "get_test",
]
