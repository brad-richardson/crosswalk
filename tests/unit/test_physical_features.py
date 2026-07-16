from __future__ import annotations

import math

import pytest

from crosswalk.config import FEATURE_COLUMNS
from crosswalk.datasets.schema import FetchConfig
from crosswalk.features.physical import (
    PHYSICAL_EXPERIMENT_FEATURES,
    compute_physical_pair_features,
)


def _lr(value):
    return [{"between": [0.0, 1.0], "value": value}]


def test_experimental_features_are_disjoint_from_production_contract() -> None:
    """The experimental physical features must never leak into the production
    ML contract.

    ``PHYSICAL_EXPERIMENT_FEATURES`` is the canonical experiment-only feature
    list; ``config.FEATURE_COLUMNS`` is the production contract. They are kept
    disjoint deliberately per the go/no-go criteria in
    ``research/physical_feature_experiment_2026-07-15.md``.

    Graduating any of these features into ``FEATURE_COLUMNS`` requires ALL of the
    research note's preconditions, not just this test being edited:

    - each flag domain has enough active reviewed examples across at least three
      target providers (bridge/tunnel still short today);
    - the informative slice improves without leaning on the
      ``physical_comparable_count`` dataset-availability proxy;
    - the metrics are re-derived through the shared ``features/compute.py`` +
      ``crosswalk backfill`` path (the ablation harness numbers do not validate
      production wiring — see the module docstring in ``features/physical.py``);
    - the linear-referenced rule iteration is vectorized/pre-normalized before it
      runs on every candidate pair during inference.
    """
    overlap = set(PHYSICAL_EXPERIMENT_FEATURES) & set(FEATURE_COLUMNS)
    assert overlap == set(), (
        "Experimental physical features leaked into the production FEATURE_COLUMNS "
        f"contract without meeting graduation preconditions: {sorted(overlap)}"
    )


def test_physical_flag_domains_reports_bridge_and_tunnel() -> None:
    fetch = FetchConfig(bridge_column="BRIDGE", tunnel_column="TUNNEL")
    assert fetch.physical_flag_domains() == frozenset({"is_bridge", "is_tunnel"})


def test_physical_flag_domains_excludes_level_only_provenance() -> None:
    fetch = FetchConfig(level_column="LAYER")
    assert fetch.physical_flag_domains() == frozenset()


def test_physical_flag_domains_empty_when_no_physical_columns() -> None:
    assert FetchConfig().physical_flag_domains() == frozenset()


def test_tunnel_conflict_requires_target_domain_provenance() -> None:
    without_domain = compute_physical_pair_features(
        ref_level_lr=None,
        target_level_lr=None,
        ref_road_flags_lr=_lr([]),
        target_road_flags_lr=_lr(["is_tunnel"]),
    )
    with_domain = compute_physical_pair_features(
        ref_level_lr=None,
        target_level_lr=None,
        ref_road_flags_lr=_lr([]),
        target_road_flags_lr=_lr(["is_tunnel"]),
        target_flag_domains={"is_tunnel"},
    )

    assert math.isnan(without_domain["tunnel_fraction_delta"])
    assert with_domain["tunnel_fraction_delta"] == 1.0
    assert with_domain["physical_structure_conflict"] == 1.0
    assert with_domain["physical_comparable_count"] == 1.0


def test_unknown_bridge_domain_stays_missing_for_tunnel_only_provider() -> None:
    features = compute_physical_pair_features(
        ref_level_lr=None,
        target_level_lr=None,
        ref_road_flags_lr=_lr(["is_bridge"]),
        target_road_flags_lr=_lr([]),
        target_flag_domains={"is_tunnel"},
    )

    assert math.isnan(features["bridge_fraction_delta"])
    assert features["tunnel_fraction_delta"] == 0.0


def test_partial_linear_referencing_compares_only_aligned_ranges() -> None:
    ref_flags = [
        {"between": [0.0, 0.5], "value": []},
        {"between": [0.5, 1.0], "value": ["is_bridge"]},
    ]
    features = compute_physical_pair_features(
        ref_level_lr=None,
        target_level_lr=None,
        ref_road_flags_lr=ref_flags,
        target_road_flags_lr=_lr(["is_bridge"]),
        ref_start_frac=0.6,
        ref_end_frac=0.9,
        target_flag_domains={"is_bridge"},
    )

    assert features["bridge_fraction_delta"] == pytest.approx(0.0)
    assert features["physical_flag_positive_match"] == pytest.approx(1.0)
    assert features["physical_positive_match"] == pytest.approx(1.0)


def test_vertical_features_compare_sign_not_exact_level_number() -> None:
    same_direction = compute_physical_pair_features(
        ref_level_lr=_lr(1),
        target_level_lr=_lr(3),
        ref_road_flags_lr=None,
        target_road_flags_lr=None,
    )
    ground_conflict = compute_physical_pair_features(
        ref_level_lr=_lr(-1),
        target_level_lr=_lr(0),
        ref_road_flags_lr=None,
        target_road_flags_lr=None,
    )

    assert same_direction["vertical_nonzero_fraction_delta"] == 0.0
    assert same_direction["vertical_sign_delta"] == 0.0
    assert same_direction["vertical_positive_match"] == 1.0
    assert ground_conflict["vertical_nonzero_fraction_delta"] == 1.0
    assert ground_conflict["vertical_sign_delta"] == 0.5
    assert ground_conflict["physical_structure_conflict"] == 1.0


def test_overlapping_flag_rules_use_union_coverage() -> None:
    # Two differently-valued overlapping rules ([0,0.6]=bridge, [0.4,1.0]=tunnel)
    # cannot merge, so the known-coverage denominator must be their union (1.0),
    # not the raw sum (1.2). is_bridge covers [0,0.6] -> fraction 0.6, so the
    # delta against a fully-bridged target is 0.4 (raw-sum arithmetic gives 0.5).
    features = compute_physical_pair_features(
        ref_level_lr=None,
        target_level_lr=None,
        ref_road_flags_lr=[
            {"between": [0.0, 0.6], "value": ["is_bridge"]},
            {"between": [0.4, 1.0], "value": ["is_tunnel"]},
        ],
        target_road_flags_lr=_lr(["is_bridge"]),
        target_flag_domains={"is_bridge", "is_tunnel"},
    )

    assert features["bridge_fraction_delta"] == pytest.approx(0.4)
    # Target has no tunnel; ref tunnel covers [0.4,1.0]/union 1.0 -> 0.6.
    assert features["tunnel_fraction_delta"] == pytest.approx(0.6)


def test_missing_levels_are_unknown_not_ground() -> None:
    features = compute_physical_pair_features(
        ref_level_lr=_lr(0),
        target_level_lr=_lr(None),
        ref_road_flags_lr=None,
        target_road_flags_lr=None,
    )

    assert math.isnan(features["vertical_nonzero_fraction_delta"])
    assert math.isnan(features["vertical_sign_delta"])
    assert features["physical_comparable_count"] == 0.0
