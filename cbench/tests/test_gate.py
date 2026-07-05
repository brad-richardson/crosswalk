"""CI regression test for the stitch-level quality gate and its metric machinery.

Runs entirely on committed fixtures (no ``data/raw`` / ``data/output``), so it
executes in GitHub Actions where the live pipeline outputs do not exist. It
guards two things the benchmark-time gate depends on:

1. The METRIC MACHINERY on a small set of REAL Boston groups + labels
   (``tests/fixtures/mini_*``): edge-overlap group mapping and sliver filtering
   must keep producing the anchored numbers. If mapping/sliver logic regresses,
   these assertions fail in CI before anyone runs the live gate.
2. The GATE LOGIC (arming, floor pass/fail, config parsing) across synthetic
   ``StitchEvalResult`` values — fast and exhaustive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cbench.eval.gate import (
    STATUS_FAIL,
    STATUS_NO_CONFIG,
    STATUS_PASS,
    STATUS_SKIP_UNARMED,
    GateFloors,
    evaluate_gate,
    load_gate_config,
)
from cbench.eval.stitch_metrics import StitchEvalResult, evaluate_stitch_groups

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# 1. Real-fixture metric-machinery regression (mapping + sliver + aggregation)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def mini_result() -> StitchEvalResult:
    groups = json.loads((FIXTURES / "mini_groups.json").read_text())["groups"]
    labels = pd.read_csv(FIXTURES / "mini_stitch_labels.csv")
    bridge = pd.read_csv(FIXTURES / "mini_bridge.csv")
    return evaluate_stitch_groups(bridge, labels, groups=groups)


def test_mini_fixture_metric_anchor(mini_result):
    """Anchored numbers from 5 real Boston groups (extracted 2026-07-05).

    These constrain the mapping + aggregation machinery. If a change here is
    intentional, re-extract the fixture and update the anchors.
    """
    r = mini_result
    assert r.groups_evaluated == 5
    assert r.precision == pytest.approx(1.0, abs=1e-4)
    assert r.recall == pytest.approx(0.5067, abs=1e-3)
    assert r.f1 == pytest.approx(0.6726, abs=1e-3)
    assert r.exact_match_rate == pytest.approx(0.2, abs=1e-4)
    assert r.total_curated_edges == 23
    # No sliver edges among these labeled groups -> filtered == raw.
    assert r.groups_sliver_affected == 0
    assert r.f1_filtered == pytest.approx(r.f1, abs=1e-9)


def test_mini_fixture_group_id_churn(mini_result):
    """Labels map by edge-overlap even when their stored group_id is stale.

    Rewrite every label's group_id to a bogus hash; edge-overlap recovery must
    still map all five groups and reproduce the same metrics.
    """
    groups = json.loads((FIXTURES / "mini_groups.json").read_text())["groups"]
    labels = pd.read_csv(FIXTURES / "mini_stitch_labels.csv")
    labels["group_id"] = [f"stale_{i}" for i in range(len(labels))]
    bridge = pd.read_csv(FIXTURES / "mini_bridge.csv")
    churned = evaluate_stitch_groups(bridge, labels, groups=groups)
    assert churned.groups_evaluated == mini_result.groups_evaluated
    assert churned.f1 == pytest.approx(mini_result.f1, abs=1e-9)


def test_mini_fixture_gate_pass_and_fail(mini_result):
    """The real fixture feeds the gate: floors below/above the anchor pass/fail."""
    below = GateFloors(min_mapped_groups=3, f1_filtered_floor=0.50, exact_filtered_floor=0.10)
    assert evaluate_gate("mini", mini_result, below).status == STATUS_PASS

    too_high_f1 = GateFloors(min_mapped_groups=3, f1_filtered_floor=0.90, exact_filtered_floor=0.10)
    out = evaluate_gate("mini", mini_result, too_high_f1)
    assert out.status == STATUS_FAIL
    assert "F1" in out.message


# --------------------------------------------------------------------------- #
# 2. Gate logic on synthetic StitchEvalResult values
# --------------------------------------------------------------------------- #


def _result(groups, f1_filtered, exact_filtered) -> StitchEvalResult:
    return StitchEvalResult(
        groups_evaluated=groups,
        precision=0.9,
        recall=0.9,
        f1=0.9,
        exact_match_rate=exact_filtered,
        total_curated_edges=10,
        total_extra_edges=0,
        precision_filtered=0.9,
        recall_filtered=0.9,
        f1_filtered=f1_filtered,
        exact_match_rate_filtered=exact_filtered,
    )


FLOORS = GateFloors(min_mapped_groups=30, f1_filtered_floor=0.78, exact_filtered_floor=0.45)


def test_gate_pass_when_above_floor():
    out = evaluate_gate("boston", _result(67, 0.8345, 0.5373), FLOORS)
    assert out.status == STATUS_PASS
    assert out.blocking is False


def test_gate_fail_on_f1():
    out = evaluate_gate("boston", _result(67, 0.70, 0.60), FLOORS)
    assert out.status == STATUS_FAIL
    assert out.blocking is True
    assert "F1" in out.message


def test_gate_fail_on_exact():
    out = evaluate_gate("boston", _result(67, 0.85, 0.30), FLOORS)
    assert out.status == STATUS_FAIL
    assert out.blocking is True
    assert "exact" in out.message


def test_gate_unarmed_below_min_mapped_groups():
    """Too few mapped groups -> skip (non-blocking). This is the arming switch."""
    out = evaluate_gate("boston", _result(12, 0.10, 0.10), FLOORS)
    assert out.status == STATUS_SKIP_UNARMED
    assert out.blocking is False


def test_gate_no_config_is_non_blocking():
    out = evaluate_gate("nowhere", _result(67, 0.10, 0.10), None)
    assert out.status == STATUS_NO_CONFIG
    assert out.blocking is False


def test_gate_none_result_with_floor_is_blocking():
    """A configured dataset that could not be evaluated must FAIL, not pass.

    Otherwise a swallowed stitch-eval error (missing labels/sidecar) would let
    --gate silently pass a dataset it was told to guard.
    """
    out = evaluate_gate("boston", None, FLOORS)
    assert out.status == STATUS_FAIL
    assert out.blocking is True


def test_gate_boundary_is_inclusive():
    """A dataset exactly on its floor passes (>=)."""
    out = evaluate_gate("boston", _result(30, 0.78, 0.45), FLOORS)
    assert out.status == STATUS_PASS


# --------------------------------------------------------------------------- #
# 3. Config parsing
# --------------------------------------------------------------------------- #


def test_load_gate_config_parses_blocks():
    cfg = {
        "gate": {
            "us_boston_streets": {
                "min_mapped_groups": 30,
                "f1_filtered_floor": 0.78,
                "exact_filtered_floor": 0.45,
            }
        }
    }
    floors = load_gate_config(cfg)
    assert set(floors) == {"us_boston_streets"}
    f = floors["us_boston_streets"]
    assert f.min_mapped_groups == 30
    assert f.f1_filtered_floor == 0.78
    assert f.exact_filtered_floor == 0.45


def test_load_gate_config_skips_malformed_and_empty():
    assert load_gate_config({}) == {}
    # Missing a required key -> skipped, not crashed.
    partial = {"gate": {"ds": {"min_mapped_groups": 30, "f1_filtered_floor": 0.5}}}
    assert load_gate_config(partial) == {}


def test_committed_toml_has_boston_gate():
    """The shipped datasets.toml must keep the Boston gate armed and configured."""
    import tomllib

    toml_path = Path(__file__).parent.parent / "datasets.toml"
    cfg = tomllib.loads(toml_path.read_text())
    floors = load_gate_config(cfg)
    assert "us_boston_streets" in floors
    # Re-derived 2026-07-05 post set-semantics reinterpretation (#295/#298):
    # baseline F1 0.8858 / exact 0.5946 over 111 mapped pair groups.
    assert floors["us_boston_streets"].f1_filtered_floor == pytest.approx(0.83)
    assert floors["us_boston_streets"].exact_filtered_floor == pytest.approx(0.50)
