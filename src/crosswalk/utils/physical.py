"""Physical road evidence with linear-referenced range preservation.

The fetch layer stores vertical level and road flags as ``*_lr`` columns whose
values are lists of ``{"between": [start, end], "value": ...}`` rules.  Parquet
round-trips those lists as numpy object arrays, while sidecars need plain JSON
types.  This module provides the small, shared normalization/clipping contract
used by the stitching sidecar, evidence packs, and review UI.

Missing metadata stays missing.  In particular, ``level=0`` and an empty flags
list are meaningful only when a source actually supplied those attributes; the
fetchers are responsible for emitting ``None`` when coverage is unknown.
"""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

import pandas as pd

PHYSICAL_LR_COLUMNS = ("level_lr", "road_flags_lr")


def _plain(value: Any) -> Any:
    """Convert numpy/pandas containers and scalars to JSON-compatible values."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        with suppress(TypeError, ValueError):
            value = value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    return value


def _missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, bool) and missing


def interval_union_length(intervals: Any) -> float:
    """Length covered by the union of ``[start, end]`` intervals.

    Overlapping intervals are counted once, so this is the correct measure of
    covered fraction when source LR rules overlap. Malformed or degenerate
    intervals are ignored.
    """
    spans: list[tuple[float, float]] = []
    for item in intervals or []:
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end > start:
            spans.append((start, end))
    if not spans:
        return 0.0
    spans.sort()
    total = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def normalize_lr_rules(raw: Any, *, flags: bool = False) -> list[dict[str, Any]]:
    """Return sorted, valid, JSON-compatible LR rules.

    Malformed ranges are ignored rather than converted to segment-wide facts.
    Rules are sorted first, then overlapping-or-adjacent equal-valued ranges are
    merged, but no boundary with a value change is lost. Overlapping ranges with
    different values are preserved as-is (coverage math must union them — see
    ``interval_union_length`` — rather than sum their raw lengths).
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    raw = _plain(raw)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    rules: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        between = _plain(item.get("between", [0.0, 1.0]))
        if not isinstance(between, list) or len(between) != 2:
            continue
        try:
            start, end = sorted((float(between[0]), float(between[1])))
        except (TypeError, ValueError):
            continue
        start, end = max(0.0, start), min(1.0, end)
        if end <= start:
            continue

        value = _plain(item.get("value"))
        if _missing_scalar(value):
            continue
        if flags:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                continue
            value = sorted({str(v) for v in value if not _missing_scalar(v)})

        rules.append({"between": [round(start, 7), round(end, 7)], "value": value})

    rules.sort(key=lambda r: (r["between"][0], r["between"][1]))
    merged: list[dict[str, Any]] = []
    for rule in rules:
        # Merge overlapping-or-adjacent ranges only when the value is identical;
        # a value change always starts a new range so no boundary is lost.
        if (
            merged
            and merged[-1]["value"] == rule["value"]
            and rule["between"][0] <= merged[-1]["between"][1]
        ):
            merged[-1]["between"][1] = max(merged[-1]["between"][1], rule["between"][1])
        else:
            merged.append(rule)
    return merged


def physical_attributes(level_lr: Any = None, road_flags_lr: Any = None) -> dict[str, Any]:
    """Build the canonical physical evidence block, omitting unknown fields."""
    out: dict[str, Any] = {}
    levels = normalize_lr_rules(level_lr)
    flags = normalize_lr_rules(road_flags_lr, flags=True)
    if levels:
        out["level_lr"] = levels
    if flags:
        out["road_flags_lr"] = flags
    return out


def clip_lr_rules(
    rules: Any,
    start_frac: float | None,
    end_frac: float | None,
    *,
    flags: bool = False,
) -> list[dict[str, Any]]:
    """Clip LR rules to an aligned interval in the segment's own fractions."""
    if start_frac is None or end_frac is None:
        return []
    try:
        start, end = sorted((float(start_frac), float(end_frac)))
    except (TypeError, ValueError):
        return []
    start, end = max(0.0, start), min(1.0, end)
    if end <= start:
        return []

    clipped: list[dict[str, Any]] = []
    for rule in normalize_lr_rules(rules, flags=flags):
        overlap_start = max(start, float(rule["between"][0]))
        overlap_end = min(end, float(rule["between"][1]))
        if overlap_end > overlap_start:
            clipped.append(
                {
                    "between": [round(overlap_start, 7), round(overlap_end, 7)],
                    "value": rule["value"],
                }
            )
    return clipped


def clip_physical_attributes(
    physical: dict[str, Any] | None,
    start_frac: float | None,
    end_frac: float | None,
) -> dict[str, Any]:
    """Return physical rules active over one aligned edge interval."""
    if not physical or start_frac is None or end_frac is None:
        return {}
    try:
        start, end = sorted((float(start_frac), float(end_frac)))
    except (TypeError, ValueError):
        return {}
    start, end = max(0.0, start), min(1.0, end)
    if end <= start:
        return {}

    out: dict[str, Any] = {"aligned_range": [round(start, 7), round(end, 7)]}
    levels = clip_lr_rules(physical.get("level_lr"), start, end)
    flags = clip_lr_rules(physical.get("road_flags_lr"), start, end, flags=True)
    if levels:
        out["level_lr"] = levels
    if flags:
        out["road_flags_lr"] = flags
    return out if len(out) > 1 else {}


def summarize_physical(physical: dict[str, Any] | None) -> str:
    """Compact human-readable bridge/tunnel/covered/indoor/layer summary."""
    if not physical:
        return ""
    parts: list[str] = []

    levels = normalize_lr_rules(physical.get("level_lr"))
    ordered_levels: list[Any] = []
    for rule in levels:
        value = rule["value"]
        if value not in ordered_levels:
            ordered_levels.append(value)
    if ordered_levels:
        parts.append("layer " + "→".join(str(v) for v in ordered_levels))

    flag_rules = normalize_lr_rules(physical.get("road_flags_lr"), flags=True)
    if flag_rules:
        total = 0.0
        aligned_range = _plain(physical.get("aligned_range"))
        if isinstance(aligned_range, list) and len(aligned_range) == 2:
            with suppress(TypeError, ValueError):
                total = abs(float(aligned_range[1]) - float(aligned_range[0]))
        if total == 0.0:
            # LR rule fractions are positions along the whole segment [0, 1], so
            # without an aligned range the denominator is the full segment (1.0).
            # Using the rule span instead would read a mid-segment flag as full
            # coverage and wrongly drop the "(partial)" suffix.
            total = 1.0
        for flag, label in (
            ("is_bridge", "bridge"),
            ("is_tunnel", "tunnel"),
            ("is_covered", "covered"),
            ("is_indoor", "indoor"),
        ):
            coverage = interval_union_length(r["between"] for r in flag_rules if flag in r["value"])
            if coverage > 0:
                suffix = " (partial)" if total and coverage < total - 1e-9 else ""
                parts.append(label + suffix)
        # The flag list records positive observations, but not which flag
        # domains a provider surveyed. Stay silent when no physical flag is
        # present rather than over-claiming both "not bridge" and "not tunnel".
    return "; ".join(parts)


def physical_is_informative(physical: dict[str, Any] | None) -> bool:
    """Whether a block contains a non-ground layer or positive physical flag."""
    if not physical:
        return False
    levels = normalize_lr_rules(physical.get("level_lr"))
    level_values = [rule["value"] for rule in levels]
    if len({str(value) for value in level_values}) > 1:
        return True
    for value in level_values:
        try:
            if float(value) != 0.0:
                return True
        except (TypeError, ValueError):
            return True
    return any(
        flag in rule["value"]
        for rule in normalize_lr_rules(physical.get("road_flags_lr"), flags=True)
        for flag in ("is_bridge", "is_tunnel", "is_covered", "is_indoor")
    )
