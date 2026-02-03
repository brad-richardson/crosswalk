"""Shared normalization functions for fetch modules.

Provides standardized conversions for road attributes from various source formats.
"""

import pandas as pd


def normalize_oneway_value(value: str | int | None) -> str | None:
    """Normalize one-way value to standard format.

    Common one-way values in datasets:
    - "yes", "Yes", "Y", "1", 1 -> "forward" (assume forward if just "yes")
    - "no", "No", "N", "0", 0, "B", "Both" -> "both"
    - "FT", "F", "forward" -> "forward"
    - "TF", "T", "backward" -> "backward"
    - "-1", "reverse" -> "backward"

    Args:
        value: Raw one-way value from source data

    Returns:
        Normalized value: "forward", "backward", "both", or None
    """
    if pd.isna(value) if hasattr(pd, "isna") else value is None:
        return None

    # Convert to string and normalize
    val_str = str(value).strip().lower()

    if val_str in ("yes", "y", "1", "ft", "f", "forward", "one-way", "oneway", "from-to"):
        return "forward"
    elif val_str in ("no", "n", "0", "b", "both", "two-way", "twoway"):
        return "both"
    elif val_str in ("-1", "tf", "t", "backward", "reverse", "to-from"):
        return "backward"
    elif val_str in ("", "null", "none", "nan"):
        return None

    return None


def normalize_speed_to_kph(value: int | float | str | None, unit: str) -> int | None:
    """Convert speed to kph.

    Args:
        value: Speed value (may be int, float, or string)
        unit: Unit string ("kph", "mph", etc.)

    Returns:
        Speed in kph as int, or None if invalid
    """
    if pd.isna(value) if hasattr(pd, "isna") else value is None:
        return None

    try:
        speed = float(value)
    except (TypeError, ValueError):
        return None

    if speed <= 0:
        return None

    if unit.lower() in ("mph", "mi/h"):
        return int(speed * 1.60934)
    return int(speed)
