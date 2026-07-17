"""Cross-source normalization of physical road tags (bridge/tunnel/level).

This module is the single, source-agnostic contract for how *every* target
fetch path — ArcGIS, download, OGC API Features, WFS — turns raw upstream
attributes into the Overture-compatible physical fields:

- ``build_road_flags``   : bridge/tunnel indicators -> ``road_flags`` arrays
- ``build_level_rules``  : z-level values           -> ``level_rules`` arrays
- ``matches_flag_value`` : coded-domain / truthy matching for a single flag
- ``is_truthy``          : generic truthy detection across source encodings
- ``add_trivial_lr_columns`` : materialize trivial linear-referenced columns

These helpers were historically private to ``fetch/arcgis.py`` even though the
non-ArcGIS fetchers depended on them, which coupled every source to ArcGIS
internals and forced lockstep edits across duplicated copies. They live here as
the shared, public contract so a change to bridge/tunnel/level semantics happens
in exactly one place and applies identically to all sources.

**Missing-vs-ground doctrine:** ``None`` (unknown), level ``0`` (known ground),
and ``[]`` (known: no flags) are three distinct states and must never be
conflated. Empty ``level_rules`` mean the source did not establish a level; we
never fabricate ground level (0). This is pinned by
``tests/unit/test_physical_features.py::test_missing_levels_are_unknown_not_ground``.
"""

from typing import Any

import geopandas as gpd
import pandas as pd

from ..utils.linear_ref import create_trivial_lr


def build_level_rules(value: Any) -> list:
    """Build a ``level_rules`` array from a single z-level value.

    Source-agnostic contract shared by all target fetch paths.

    Args:
        value: Z-level value from source data.

    Returns:
        List with one level rule dict ``[{"value": level}]`` when the value is a
        parseable integer, or an empty list when the value is missing or
        non-integer. An empty list means *unknown level*, which is materially
        different from a known ground level (0).
    """
    if pd.isna(value):
        return []
    try:
        level = int(value)
        return [{"value": level}]
    except (ValueError, TypeError):
        return []


def build_road_flags(
    gdf: gpd.GeoDataFrame,
    bridge_column: str | None,
    tunnel_column: str | None,
    *,
    bridge_values: list[str | int | float] | None = None,
    tunnel_values: list[str | int | float] | None = None,
) -> list[list[str] | None]:
    """Build ``road_flags`` arrays from bridge/tunnel columns.

    Source-agnostic contract shared by all target fetch paths.

    Args:
        gdf: Input GeoDataFrame (or any frame exposing the flag columns).
        bridge_column: Column name for bridge indicator.
        tunnel_column: Column name for tunnel indicator.
        bridge_values: Explicit coded-domain values that count as a bridge. When
            ``None``, a generic truthy test is used.
        tunnel_values: Explicit coded-domain values that count as a tunnel. When
            ``None``, a generic truthy test is used.

    Returns:
        List (one entry per row) of flag lists (possibly empty), or ``None`` per
        row when neither flag column is configured — ``None`` (unknown) is
        distinct from ``[]`` (known: no flags).
    """
    configured = bool(
        (bridge_column and bridge_column in gdf.columns)
        or (tunnel_column and tunnel_column in gdf.columns)
    )
    if not configured:
        return [None for _ in range(len(gdf))]

    result = []

    for idx in range(len(gdf)):
        flags = []

        # Check bridge
        if bridge_column and bridge_column in gdf.columns:
            val = gdf.iloc[idx][bridge_column]
            if matches_flag_value(val, bridge_values):
                flags.append("is_bridge")

        # Check tunnel
        if tunnel_column and tunnel_column in gdf.columns:
            val = gdf.iloc[idx][tunnel_column]
            if matches_flag_value(val, tunnel_values):
                flags.append("is_tunnel")

        result.append(flags)

    return result


def matches_flag_value(value: Any, accepted: list[str | int | float] | None) -> bool:
    """Match a source flag against an explicit coded domain or truthy fallback."""
    if accepted is None:
        return is_truthy(value)
    if pd.isna(value):
        return False
    value_text = str(value).strip().casefold()
    for candidate in accepted:
        if value_text == str(candidate).strip().casefold():
            return True
        try:
            if float(value) == float(candidate):
                return True
        except (TypeError, ValueError):
            pass
    return False


def is_truthy(value: Any) -> bool:
    """Check if a value indicates True/Yes/1.

    Handles various representations from different data sources:
    - Boolean True/False
    - Numeric 1/0
    - String "Y"/"N", "Yes"/"No", "True"/"False", "1"/"0"

    Args:
        value: Value to check

    Returns:
        True if value indicates a truthy state
    """
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value == 1 or value > 0

    if isinstance(value, str):
        return value.upper() in ("Y", "YES", "TRUE", "1", "T")

    return bool(value)


def add_trivial_lr_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add trivial linear-referenced columns for target-side data.

    Source-agnostic contract shared by all target fetch paths (ArcGIS,
    download, OGC, WFS). Target-side data typically doesn't have
    linear-referenced attributes, so we create trivial LR columns with a single
    range ``[0.0, 1.0, value]`` for each attribute.

    This enables the same feature computation code to work with both
    Overture (which has LR attributes) and target data.

    Args:
        gdf: GeoDataFrame with flat attribute columns

    Returns:
        GeoDataFrame with added ``*_lr`` columns
    """

    # Get name from names struct or flat name column
    def get_name(row):
        names = row.get("names")
        if isinstance(names, dict):
            return names.get("primary")
        return row.get("name")

    # Names LR - extract primary from names struct
    gdf["names_lr"] = gdf.apply(
        lambda row: create_trivial_lr(get_name(row)).to_dict_list(),
        axis=1,
    )

    # Subclass LR
    if "subclass" in gdf.columns:
        gdf["subclass_lr"] = gdf["subclass"].apply(lambda x: create_trivial_lr(x).to_dict_list())
    else:
        gdf["subclass_lr"] = [[{"between": [0.0, 1.0], "value": None}] for _ in range(len(gdf))]

    # Level LR. Empty rules mean the source did not establish a level; do not
    # fabricate ground level (0), which is materially different from unknown.
    def get_level(row):
        level_rules = row.get("level_rules")
        if isinstance(level_rules, list) and len(level_rules) > 0:
            first = level_rules[0]
            if isinstance(first, dict):
                return first.get("value")
        return None

    gdf["level_lr"] = gdf.apply(
        lambda row: create_trivial_lr(get_level(row)).to_dict_list(),
        axis=1,
    )

    # Road flags LR - extract from road_flags if present
    def get_flags(row):
        road_flags = row.get("road_flags")
        if isinstance(road_flags, list):
            return sorted(road_flags)
        return None

    gdf["road_flags_lr"] = gdf.apply(
        lambda row: create_trivial_lr(get_flags(row)).to_dict_list(),
        axis=1,
    )

    # One-way LR - extract from oneway flat column
    if "oneway" in gdf.columns:
        gdf["oneway_lr"] = gdf["oneway"].apply(lambda x: create_trivial_lr(x).to_dict_list())
    else:
        gdf["oneway_lr"] = [[{"between": [0.0, 1.0], "value": None}] for _ in range(len(gdf))]

    # Access/mode LR - target sources carry no Overture access_restrictions, so
    # the channel is class-default only (e.g. a cycleway target ⇒ bike:designated°),
    # matching the reference-side inferences. Imported here to keep the shared
    # class-default table in one place (fetch.overture.parse_access_lr).
    from .overture import parse_access_lr

    access_class_col = (
        "class"
        if "class" in gdf.columns
        else ("road_class" if "road_class" in gdf.columns else None)
    )
    if access_class_col is not None:
        gdf["access_lr"] = gdf[access_class_col].apply(
            lambda c: parse_access_lr(None, c).to_dict_list()
        )
    else:
        gdf["access_lr"] = [[{"between": [0.0, 1.0], "value": None}] for _ in range(len(gdf))]

    # Speed limit LR - extract from speed_limit_kph flat column
    if "speed_limit_kph" in gdf.columns:
        gdf["speed_limit_kph_lr"] = gdf["speed_limit_kph"].apply(
            lambda x: create_trivial_lr(x).to_dict_list()
        )
    else:
        gdf["speed_limit_kph_lr"] = [
            [{"between": [0.0, 1.0], "value": None}] for _ in range(len(gdf))
        ]

    return gdf
