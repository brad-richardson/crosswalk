"""Experimental pairwise physical-road features.

These features compare only attributes that both providers actually expose.
Missing target domains stay NaN; in particular, an empty flag list from a
tunnel-only source is not evidence that the source surveyed bridges. Vertical
comparison deliberately uses ground/non-ground fractions and sign rather than
exact numeric levels, whose scale and semantics vary across providers.

EXPERIMENT-ONLY: these features are deliberately excluded from the production ML
contract (``config.FEATURE_COLUMNS``); see the go/no-go criteria in
``research/physical_feature_experiment_2026-07-15.md``. The disjointness is
enforced by ``tests/unit/test_physical_features.py``.

Graduation skew — the ablation numbers do NOT validate production wiring. The
ablation harness (``scripts/physical_feature_experiment.py``) trains on
``[*FEATURE_COLUMNS, *physical_features]`` through its own alignment-fraction
sourcing and its own per-dataset ``target_flag_domains`` derivation. Production
inference would instead route through ``features/compute.py``'s worker path and
``crosswalk backfill``, where the linear-referenced physical-rule dicts these
features consume are not present in ``worker_data`` today. Graduating any of
these features therefore requires re-deriving the reported metrics through that
shared compute path (per the backfill-parity rule), not reusing the standalone
ablation JSON.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any

import numpy as np

from ..utils.physical import clip_lr_rules, interval_union_length

PHYSICAL_FLAG_FEATURES = (
    "bridge_fraction_delta",
    "tunnel_fraction_delta",
    "physical_flag_positive_match",
)
PHYSICAL_VERTICAL_FEATURES = (
    "vertical_nonzero_fraction_delta",
    "vertical_sign_delta",
    "vertical_positive_match",
)
PHYSICAL_COMPOSITE_FEATURES = (
    "physical_structure_conflict",
    "physical_positive_match",
    "physical_comparable_count",
)
PHYSICAL_EXPERIMENT_FEATURES = (
    *PHYSICAL_FLAG_FEATURES,
    *PHYSICAL_VERTICAL_FEATURES,
    *PHYSICAL_COMPOSITE_FEATURES,
)


def _known_duration(rules: list[dict[str, Any]]) -> float:
    # Union length, so overlapping rules do not inflate the known-coverage
    # denominator past the actually-covered fraction.
    return interval_union_length(rule["between"] for rule in rules)


def _flag_fraction(
    rules: Any,
    flag: str,
    start_frac: float,
    end_frac: float,
) -> float:
    clipped = clip_lr_rules(rules, start_frac, end_frac, flags=True)
    known = _known_duration(clipped)
    if known <= 0:
        return float("nan")
    positive = interval_union_length(rule["between"] for rule in clipped if flag in rule["value"])
    return min(max(positive / known, 0.0), 1.0)


def _level_profile(
    rules: Any,
    start_frac: float,
    end_frac: float,
) -> tuple[float, float]:
    """Return (nonzero fraction, signed mean) over known aligned level rules.

    Fractions are measured against the UNION-covered length, not the raw sum of
    rule spans, so overlapping ranges of different level values (which
    ``normalize_lr_rules`` cannot merge) do not inflate the denominator or push a
    ratio past 1.0. Same-value overlaps are already merged upstream. The
    missing-vs-ground doctrine is untouched: ``None``/NaN levels drop out as
    unknown, explicit level 0 stays a real ground observation (sign 0).
    """
    clipped = clip_lr_rules(rules, start_frac, end_frac)
    pos_intervals: list[tuple[float, float]] = []
    neg_intervals: list[tuple[float, float]] = []
    all_intervals: list[tuple[float, float]] = []
    for rule in clipped:
        try:
            value = float(rule["value"])
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        start, end = float(rule["between"][0]), float(rule["between"][1])
        if end <= start:
            continue
        interval = (start, end)
        all_intervals.append(interval)
        sign = float(np.sign(value))
        if sign > 0:
            pos_intervals.append(interval)
        elif sign < 0:
            neg_intervals.append(interval)
    known = interval_union_length(all_intervals)
    if known <= 0:
        return float("nan"), float("nan")
    nonzero = interval_union_length(pos_intervals + neg_intervals) / known
    # Signed mean via per-sign union coverage: an overlap covered by both a + and a
    # - rule cancels in the numerator, and |pos - neg| <= known keeps it in [-1, 1].
    signed_mean = (
        interval_union_length(pos_intervals) - interval_union_length(neg_intervals)
    ) / known
    return min(max(nonzero, 0.0), 1.0), min(max(signed_mean, -1.0), 1.0)


def _finite(values: Collection[float]) -> list[float]:
    return [float(value) for value in values if not math.isnan(float(value))]


def compute_physical_pair_features(
    *,
    ref_level_lr: Any,
    target_level_lr: Any,
    ref_road_flags_lr: Any,
    target_road_flags_lr: Any,
    ref_start_frac: float = 0.0,
    ref_end_frac: float = 1.0,
    target_start_frac: float = 0.0,
    target_end_frac: float = 1.0,
    target_flag_domains: Collection[str] = (),
    ref_flag_domains: Collection[str] = ("is_bridge", "is_tunnel"),
) -> dict[str, float]:
    """Compute comparable, alignment-aware physical evidence for one pair.

    ``target_flag_domains`` is required provenance, not inferred from observed
    positive flags. This keeps a provider's unmodeled domains unknown.
    """
    target_domains = set(target_flag_domains)
    ref_domains = set(ref_flag_domains)
    flag_deltas: dict[str, float] = {}
    positive_flag_matches: list[float] = []

    for flag, feature in (
        ("is_bridge", "bridge_fraction_delta"),
        ("is_tunnel", "tunnel_fraction_delta"),
    ):
        if flag not in target_domains or flag not in ref_domains:
            flag_deltas[feature] = float("nan")
            continue
        ref_fraction = _flag_fraction(ref_road_flags_lr, flag, ref_start_frac, ref_end_frac)
        target_fraction = _flag_fraction(
            target_road_flags_lr, flag, target_start_frac, target_end_frac
        )
        if math.isnan(ref_fraction) or math.isnan(target_fraction):
            flag_deltas[feature] = float("nan")
            continue
        flag_deltas[feature] = abs(ref_fraction - target_fraction)
        positive_flag_matches.append(min(ref_fraction, target_fraction))

    physical_flag_positive_match = (
        max(positive_flag_matches) if positive_flag_matches else float("nan")
    )

    ref_nonzero, ref_sign = _level_profile(ref_level_lr, ref_start_frac, ref_end_frac)
    target_nonzero, target_sign = _level_profile(
        target_level_lr, target_start_frac, target_end_frac
    )
    if any(math.isnan(value) for value in (ref_nonzero, ref_sign, target_nonzero, target_sign)):
        vertical_nonzero_delta = float("nan")
        vertical_sign_delta = float("nan")
        vertical_positive_match = float("nan")
    else:
        vertical_nonzero_delta = abs(ref_nonzero - target_nonzero)
        vertical_sign_delta = abs(ref_sign - target_sign) / 2.0
        same_nonzero_sign = ref_sign != 0.0 and target_sign != 0.0 and ref_sign * target_sign > 0
        vertical_positive_match = min(ref_nonzero, target_nonzero) if same_nonzero_sign else 0.0

    primitive_conflicts = _finite(
        [
            *flag_deltas.values(),
            vertical_nonzero_delta,
            vertical_sign_delta,
        ]
    )
    positive_matches = _finite([physical_flag_positive_match, vertical_positive_match])
    comparable_count = sum(
        not math.isnan(value)
        for value in (
            flag_deltas["bridge_fraction_delta"],
            flag_deltas["tunnel_fraction_delta"],
            vertical_nonzero_delta,
        )
    )

    return {
        **flag_deltas,
        "physical_flag_positive_match": physical_flag_positive_match,
        "vertical_nonzero_fraction_delta": vertical_nonzero_delta,
        "vertical_sign_delta": vertical_sign_delta,
        "vertical_positive_match": vertical_positive_match,
        "physical_structure_conflict": (
            max(primitive_conflicts) if primitive_conflicts else float("nan")
        ),
        "physical_positive_match": (max(positive_matches) if positive_matches else float("nan")),
        "physical_comparable_count": float(comparable_count),
    }
