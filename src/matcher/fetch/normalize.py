"""Shared normalization functions for fetch modules.

Provides standardized conversions for road attributes from various source formats.
"""

import geopandas as gpd
import pandas as pd


def resolve_column(gdf: gpd.GeoDataFrame, name: str | None) -> str | None:
    """Resolve a configured column name against actual DataFrame columns, case-insensitively.

    Data sources sometimes return columns in different cases than configured
    (e.g., ArcGIS returning 'objectid' when config says 'OBJECTID').

    Args:
        gdf: GeoDataFrame to search.
        name: Configured column name (may be None).

    Returns:
        The actual column name from gdf that matches case-insensitively,
        or None if name is None or no match found.
    """
    if name is None:
        return None
    if name in gdf.columns:
        return name
    lower = name.lower()
    for col in gdf.columns:
        if col.lower() == lower:
            return col
    return None


def _str_key(value) -> str:
    """Convert a value to a normalized string key.

    Handles the float-upcast problem: pandas often stores int-like values
    as float64 (e.g., 1 → 1.0), so str(1.0) gives "1.0" instead of "1".
    This normalizes integral floats back to their int representation.
    """
    if isinstance(value, float) and value == int(value):
        return str(int(value)).lower()
    return str(value).lower()


def map_column(series: pd.Series, mapping: dict, fallback: str | None = None):
    """Map a Series using a dict with automatic str/case normalization.

    Converts both mapping keys and series values to lowercase strings
    so that int-vs-string and case mismatches are handled transparently.
    Also normalizes integral floats (e.g., 1.0 → "1") so that YAML int
    keys match pandas float64 column values.

    Args:
        series: Pandas Series to map.
        mapping: Dict of source values to target values.
        fallback: Value to fill for unmatched rows (None = keep NaN).

    Returns:
        Numpy array of mapped values.
    """
    normalized = {_str_key(k): v for k, v in mapping.items()}
    result = series.fillna("").apply(_str_key).map(normalized)
    if fallback is not None:
        result = result.fillna(fallback)
    return result.values


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
