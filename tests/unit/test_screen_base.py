"""Tests for screen base framework."""

import pytest
from shapely.geometry import LineString

from matcher.screen import (
    CandidateContext,
    ScreenOutcome,
    ScreenResult,
    ScreenTest,
    get_registered_tests,
    get_test,
    register_test,
)
from matcher.screen.base import _SCREEN_TESTS


class TestScreenOutcome:
    def test_outcome_values(self):
        assert ScreenOutcome.PASS.value == "pass"
        assert ScreenOutcome.FAIL.value == "fail"
        assert ScreenOutcome.WARN.value == "warn"
        assert ScreenOutcome.SKIP.value == "skip"


class TestScreenResult:
    def test_pass_result(self):
        result = ScreenResult(outcome=ScreenOutcome.PASS, test_name="test")
        assert result.outcome == ScreenOutcome.PASS
        assert result.reason is None

    def test_non_pass_requires_reason(self):
        for outcome in [ScreenOutcome.FAIL, ScreenOutcome.WARN, ScreenOutcome.SKIP]:
            with pytest.raises(ValueError, match="requires a reason"):
                ScreenResult(outcome=outcome, test_name="test")

    def test_empty_reason_rejected(self):
        with pytest.raises(ValueError, match="requires a reason"):
            ScreenResult(outcome=ScreenOutcome.FAIL, test_name="test", reason="   ")

    def test_fail_with_reason(self):
        result = ScreenResult(outcome=ScreenOutcome.FAIL, test_name="test", reason="Road in water")
        assert result.outcome == ScreenOutcome.FAIL
        assert result.reason == "Road in water"

    def test_result_with_details(self):
        result = ScreenResult(
            outcome=ScreenOutcome.WARN,
            test_name="test",
            reason="Suspicious",
            details={"intersection_m": 50.0},
        )
        assert result.details["intersection_m"] == 50.0


class TestCandidateContext:
    def test_create_context(self):
        ctx = CandidateContext(
            target_id="target_456",
            target_geom=LineString([(0, 0.1), (1, 1.1)]),
            road_class="residential",
        )
        assert ctx.target_id == "target_456"
        assert ctx.road_class == "residential"
        assert ctx.target_attrs == {}

    def test_context_optional_fields(self):
        ctx = CandidateContext(
            target_id="target_1",
            target_geom=LineString([(0, 0), (1, 1)]),
        )
        assert ctx.road_class is None
        assert ctx.target_attrs == {}


class TestRegistry:
    @pytest.fixture(autouse=True)
    def clear_registry(self):
        _SCREEN_TESTS.clear()
        yield
        _SCREEN_TESTS.clear()

    def test_register_and_get_test(self):
        @register_test
        class DummyTest(ScreenTest):
            name = "dummy_test"

            def prepare(self, bbox):
                pass

            def test_candidate(self, ctx):
                return ScreenResult(outcome=ScreenOutcome.PASS, test_name=self.name)

        assert "dummy_test" in get_registered_tests()
        assert get_test("dummy_test") == DummyTest

    def test_get_unknown_test_raises(self):
        with pytest.raises(KeyError, match="Unknown screen test"):
            get_test("nonexistent_test")

    def test_register_invalid_name_raises(self):
        # No name
        with pytest.raises(ValueError, match="non-empty string"):

            @register_test
            class NoNameTest(ScreenTest):
                def prepare(self, bbox):
                    pass

                def test_candidate(self, ctx):
                    pass

        # Whitespace name
        with pytest.raises(ValueError, match="non-empty string"):

            @register_test
            class WhitespaceTest(ScreenTest):
                name = "   "

                def prepare(self, bbox):
                    pass

                def test_candidate(self, ctx):
                    pass

        # Non-string name
        with pytest.raises(ValueError, match="non-empty string"):

            @register_test
            class IntNameTest(ScreenTest):
                name = 123

                def prepare(self, bbox):
                    pass

                def test_candidate(self, ctx):
                    pass
