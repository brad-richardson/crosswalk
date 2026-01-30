"""Tests for falsification base framework."""

import pytest
from shapely.geometry import LineString

from matcher.falsification import (
    FalsificationOutcome,
    FalsificationResult,
    FalsificationTest,
    MatchContext,
    get_registered_tests,
    get_test,
    register_test,
)


class TestFalsificationOutcome:
    """Tests for FalsificationOutcome enum."""

    def test_outcome_values(self):
        """Test that all expected outcomes exist."""
        assert FalsificationOutcome.PASS.value == "pass"
        assert FalsificationOutcome.FAIL.value == "fail"
        assert FalsificationOutcome.WARN.value == "warn"
        assert FalsificationOutcome.SKIP.value == "skip"


class TestFalsificationResult:
    """Tests for FalsificationResult dataclass."""

    def test_pass_result(self):
        """Test creating a PASS result."""
        result = FalsificationResult(
            outcome=FalsificationOutcome.PASS,
            test_name="test",
        )
        assert result.outcome == FalsificationOutcome.PASS
        assert result.test_name == "test"
        assert result.reason is None

    def test_fail_requires_reason(self):
        """Test that FAIL outcome requires a reason."""
        with pytest.raises(ValueError, match="FAIL outcome requires a reason"):
            FalsificationResult(
                outcome=FalsificationOutcome.FAIL,
                test_name="test",
            )

    def test_fail_with_reason(self):
        """Test creating a FAIL result with reason."""
        result = FalsificationResult(
            outcome=FalsificationOutcome.FAIL,
            test_name="test",
            reason="Road in water",
        )
        assert result.outcome == FalsificationOutcome.FAIL
        assert result.reason == "Road in water"

    def test_result_with_details(self):
        """Test creating a result with details."""
        result = FalsificationResult(
            outcome=FalsificationOutcome.WARN,
            test_name="test",
            reason="Suspicious",
            details={"intersection_m": 50.0},
        )
        assert result.details["intersection_m"] == 50.0


class TestMatchContext:
    """Tests for MatchContext dataclass."""

    def test_create_context(self):
        """Test creating a match context."""
        ref_geom = LineString([(0, 0), (1, 1)])
        target_geom = LineString([(0, 0.1), (1, 1.1)])

        ctx = MatchContext(
            match_id="match_1",
            ref_id="ref_123",
            target_id="target_456",
            ref_geom=ref_geom,
            target_geom=target_geom,
            confidence=0.95,
        )

        assert ctx.match_id == "match_1"
        assert ctx.ref_id == "ref_123"
        assert ctx.target_id == "target_456"
        assert ctx.confidence == 0.95
        assert ctx.ref_attrs == {}
        assert ctx.target_attrs == {}


class TestRegistry:
    """Tests for test registry."""

    def test_register_and_get_test(self):
        """Test registering and retrieving a test."""
        # Create a test class
        @register_test
        class DummyTest(FalsificationTest):
            name = "dummy_test"

            def prepare(self, bbox):
                pass

            def test_match(self, ctx):
                return FalsificationResult(
                    outcome=FalsificationOutcome.PASS,
                    test_name=self.name,
                )

        # Verify it's registered
        assert "dummy_test" in get_registered_tests()
        assert get_test("dummy_test") == DummyTest

    def test_get_unknown_test_raises(self):
        """Test that getting unknown test raises KeyError."""
        with pytest.raises(KeyError, match="Unknown falsification test"):
            get_test("nonexistent_test")

    def test_register_without_name_raises(self):
        """Test that registering without name raises ValueError."""
        with pytest.raises(ValueError, match="must have a 'name' attribute"):

            @register_test
            class NoNameTest(FalsificationTest):
                def prepare(self, bbox):
                    pass

                def test_match(self, ctx):
                    pass
