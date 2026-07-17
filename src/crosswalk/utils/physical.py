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

# Access/mode evidence channel (evidence-pack only; NOT an ML feature). Each
# ``access_lr`` span value is a per-mode map over exactly these three modes.
ACCESS_MODES = ("motor_vehicle", "bicycle", "foot")
_ACCESS_VALUES = frozenset({"allowed", "designated", "denied", "restricted"})
_ACCESS_SOURCES = frozenset({"tagged", "class_default"})
# Compact prompt labels. ``°`` marks a class_default; a bare value marks a
# tagged (surveyed) observation; ``?`` marks an unknown (unrendered) mode.
_ACCESS_MODE_LABELS = {"motor_vehicle": "mv", "bicycle": "bike", "foot": "foot"}
_ACCESS_VALUE_LABELS = {
    "allowed": "yes",
    "denied": "no",
    "designated": "designated",
    "restricted": "restricted",
}


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


def _clean_access_map(value: Any) -> dict[str, dict[str, str]]:
    """Keep only well-formed ``{mode: {value, source}}`` entries for known modes.

    Unknown modes and malformed/unknown value/source pairs are dropped (an
    ``unknown`` mode is never stored — it is simply absent).
    """
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for mode in ACCESS_MODES:
        entry = value.get(mode)
        if not isinstance(entry, dict):
            continue
        val = entry.get("value")
        src = entry.get("source")
        if val in _ACCESS_VALUES and src in _ACCESS_SOURCES:
            out[mode] = {"value": str(val), "source": str(src)}
    return out


def normalize_access_lr(raw: Any) -> list[dict[str, Any]]:
    """Return sorted, valid, JSON-compatible ``access_lr`` spans.

    Each surviving rule is ``{"between": [start, end], "value": {mode: {value,
    source}}}`` for the subset of :data:`ACCESS_MODES` that is known. Adjacent
    spans with an identical per-mode map are merged. Spans with no known mode are
    dropped (never rendered, never guessed).
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
        mode_map = _clean_access_map(_plain(item.get("value")))
        if not mode_map:
            continue
        rules.append({"between": [round(start, 7), round(end, 7)], "value": mode_map})

    rules.sort(key=lambda r: (r["between"][0], r["between"][1]))
    merged: list[dict[str, Any]] = []
    for rule in rules:
        if (
            merged
            and merged[-1]["value"] == rule["value"]
            and rule["between"][0] <= merged[-1]["between"][1]
        ):
            merged[-1]["between"][1] = max(merged[-1]["between"][1], rule["between"][1])
        else:
            merged.append(rule)
    return merged


def _clip_access_rules(rules: Any, start: float, end: float) -> list[dict[str, Any]]:
    """Clip normalized ``access_lr`` spans to an aligned interval."""
    clipped: list[dict[str, Any]] = []
    for rule in normalize_access_lr(rules):
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


def physical_attributes(
    level_lr: Any = None, road_flags_lr: Any = None, access_lr: Any = None
) -> dict[str, Any]:
    """Build the canonical physical evidence block, omitting unknown fields."""
    out: dict[str, Any] = {}
    levels = normalize_lr_rules(level_lr)
    flags = normalize_lr_rules(road_flags_lr, flags=True)
    access = normalize_access_lr(access_lr)
    if levels:
        out["level_lr"] = levels
    if flags:
        out["road_flags_lr"] = flags
    if access:
        out["access_lr"] = access
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
    access = _clip_access_rules(physical.get("access_lr"), start, end)
    if levels:
        out["level_lr"] = levels
    if flags:
        out["road_flags_lr"] = flags
    if access:
        out["access_lr"] = access
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


def _resolve_access_modes(rules: list[dict[str, Any]]) -> dict[str, tuple[str, str, bool]]:
    """Pick the summary ``(value, source, partial)`` per mode across the spans.

    Access is usually a single segment-wide span, so this collapses to a direct
    lookup; when a restriction varies along the segment the winning value is
    chosen as follows:

    - A **tagged** (surveyed) signal always outranks a ``class_default`` one,
      even at *lower* coverage — a class-default majority (e.g. ``allowed`` over
      60%) must never mask a real tagged restriction (e.g. a ``denied`` over 40%)
      and silently drop a separation signal.
    - Within the winning source tier, the longest-covered value wins (ties broken
      by first occurrence).

    ``partial`` is ``True`` when the chosen value does not cover the mode's full
    surveyed extent (so the summary can flag it, mirroring the road-flags
    ``(partial)`` suffix) — most relevant when a tagged restriction covers only
    part of a segment whose remainder falls back to a class default.
    """
    acc: dict[str, dict[tuple[str, str], float]] = {}
    for rule in rules:
        start, end = float(rule["between"][0]), float(rule["between"][1])
        length = max(0.0, end - start)
        for mode, entry in rule["value"].items():
            key = (entry["value"], entry["source"])
            acc.setdefault(mode, {})
            acc[mode][key] = acc[mode].get(key, 0.0) + length
    resolved: dict[str, tuple[str, str, bool]] = {}
    for mode, counts in acc.items():
        total = sum(counts.values())
        # Prefer a tagged signal even at lower coverage so a class-default
        # majority can't mask a surveyed restriction.
        tagged = {key: cov for key, cov in counts.items() if key[1] == "tagged"}
        pool = tagged if tagged else counts
        (value, source), coverage = max(pool.items(), key=lambda kv: kv[1])
        partial = coverage < total - 1e-9
        resolved[mode] = (value, source, partial)
    return resolved


def summarize_access(physical: dict[str, Any] | None, *, tagged_only: bool = False) -> str:
    """Compact per-mode access summary, e.g. ``mv:yes° bike:? foot:?``.

    ``°`` marks a class_default (implied by road class, never overriding a tag);
    a bare value marks a tagged (surveyed) observation; ``?`` marks an unknown
    (unrendered) mode. Returns ``""`` when no mode is known — or, with
    ``tagged_only``, when no *tagged* mode is present (so class-default-only
    blocks, already shown on the segment line, are not repeated at edge level).
    """
    if not physical:
        return ""
    resolved = _resolve_access_modes(normalize_access_lr(physical.get("access_lr")))
    if not resolved:
        return ""
    if tagged_only and not any(src == "tagged" for _, src, _ in resolved.values()):
        return ""
    parts: list[str] = []
    for mode in ACCESS_MODES:
        label = _ACCESS_MODE_LABELS[mode]
        if mode in resolved:
            value, source, partial = resolved[mode]
            suffix = "°" if source == "class_default" else ""
            partial_suffix = " (partial)" if partial else ""
            parts.append(
                f"{label}:{_ACCESS_VALUE_LABELS.get(value, value)}{suffix}{partial_suffix}"
            )
        else:
            parts.append(f"{label}:?")
    return " ".join(parts)


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
    if any(
        flag in rule["value"]
        for rule in normalize_lr_rules(physical.get("road_flags_lr"), flags=True)
        for flag in ("is_bridge", "is_tunnel", "is_covered", "is_indoor")
    ):
        return True
    # A tagged (surveyed) access restriction is decision-relevant at the edge
    # level; class-default inferences are already shown on the segment line and
    # are not, on their own, edge-level informative.
    return any(
        entry.get("source") == "tagged"
        for rule in normalize_access_lr(physical.get("access_lr"))
        for entry in rule["value"].values()
    )
