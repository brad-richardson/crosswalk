from __future__ import annotations

import math

import pytest

from crosswalk.features.physical import compute_physical_pair_features


def _lr(value):
    return [{"between": [0.0, 1.0], "value": value}]


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
