"""Feature audit inspection routes for the matcher web UI.

Provides a map-based page for browsing labeled pairs with per-feature
inspection, grouped by FEATURE_CATEGORIES, with inline histograms showing
match vs no_match distributions.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from ...config import FEATURE_CATEGORIES, FEATURE_COLUMNS
from ...features.semantic import display_name
from ...labeling.data_store import DataStore
from ...labeling.feature_store import FeatureStore
from ...labeling.label_store import LabelStore
from ..jinja import templates

logger = logging.getLogger(__name__)


def _display_name_from_raw(raw) -> str | None:
    """Derive display name from a raw names column value (JSON string or dict)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    return display_name(raw)


router = APIRouter(prefix="/audit")

PROJECT_ROOT = Path(__file__).parents[4]
LABELS_DIR = PROJECT_ROOT / "labels"

# Bounded cache for loaded audit data per dataset (max 8 datasets to limit memory)
_audit_cache: dict[str, dict] = {}
_AUDIT_CACHE_MAX = 8


def _load_audit_data(dataset: str) -> dict:
    """Load and join labels + features + data for a dataset. Cached."""
    if dataset in _audit_cache:
        return _audit_cache[dataset]

    # Load labels
    label_store = LabelStore(dataset, labels_dir=LABELS_DIR)
    labels_df = label_store.df.copy()
    if len(labels_df) == 0:
        _audit_cache[dataset] = {"pairs": [], "features_df": pd.DataFrame()}
        return _audit_cache[dataset]

    # Load features
    feature_store = FeatureStore(dataset, features_dir=LABELS_DIR / "features")
    features_df = feature_store.df.copy()

    # Load geometry data
    data_store = DataStore(dataset, data_dir=LABELS_DIR / "data")
    data_gdf = data_store.gdf.copy()

    # Join labels with features
    if len(features_df) > 0:
        merged = labels_df.merge(
            features_df,
            on=["gers_id", "target_id"],
            how="inner",
        )
    else:
        merged = labels_df

    # Join with geometry data
    if len(data_gdf) > 0:
        # Pick geometry + name columns from data_gdf
        data_cols = ["gers_id", "target_id"]
        for col in [
            "ref_geometry",
            "target_geometry",
            "ref_names",
            "target_names",
            "ref_class",
            "target_class",
        ]:
            if col in data_gdf.columns:
                data_cols.append(col)
        data_subset = data_gdf[data_cols].copy()

        merged = merged.merge(
            data_subset,
            on=["gers_id", "target_id"],
            how="left",
        )

    # Build pairs list
    pairs = []
    for _idx, row in merged.iterrows():
        pair = {
            "index": len(pairs),
            "gers_id": str(row.get("gers_id", "")),
            "target_id": str(row.get("target_id", "")),
            "label": row.get("label", "unknown"),
            "ref_name": _display_name_from_raw(row.get("ref_names")),
            "target_name": _display_name_from_raw(row.get("target_names")),
            "ref_class": row.get("ref_class") if pd.notna(row.get("ref_class")) else None,
            "target_class": row.get("target_class") if pd.notna(row.get("target_class")) else None,
            "features": {},
        }

        # Extract feature values
        for col in FEATURE_COLUMNS:
            if col in row.index and pd.notna(row[col]):
                pair["features"][col] = float(row[col])

        # Extract geometries for map display
        ref_geom = row.get("ref_geometry")
        target_geom = row.get("target_geometry")
        if ref_geom is not None and target_geom is not None:
            try:
                from shapely.geometry import mapping

                pair["ref_geom_json"] = mapping(ref_geom)
                pair["target_geom_json"] = mapping(target_geom)
            except Exception:
                pass  # Geometry may be invalid WKB; skip map preview

        pairs.append(pair)

    result = {"pairs": pairs, "features_df": merged}
    if len(_audit_cache) >= _AUDIT_CACHE_MAX:
        # Evict oldest entry
        _audit_cache.pop(next(iter(_audit_cache)))
    _audit_cache[dataset] = result
    return result


def _list_audit_datasets() -> list[str]:
    """List datasets that have labels."""
    human_dir = LABELS_DIR / "human"
    if not human_dir.exists():
        return []
    datasets = []
    for partition_dir in sorted(human_dir.glob("dataset=*")):
        if partition_dir.is_dir():
            ds = partition_dir.name.split("=")[1]
            csv_path = partition_dir / "data.csv"
            if csv_path.exists():
                datasets.append(ds)
    return datasets


@router.get("")
async def audit_page(
    request: Request,
    dataset: str | None = None,
    label: str | None = None,
    sort: str | None = None,
    order: str = "asc",
):
    """Main audit page."""
    datasets = _list_audit_datasets()
    pairs = []
    total_pairs = 0

    if dataset:
        data = _load_audit_data(dataset)
        pairs = data["pairs"]

        # Filter by label
        if label and label != "all":
            pairs = [p for p in pairs if p["label"] == label]

        # Sort by feature
        if sort and sort in FEATURE_COLUMNS:
            reverse = order == "desc"
            missing_sentinel = float("inf") if not reverse else float("-inf")
            pairs = sorted(
                pairs,
                key=lambda p: p["features"].get(sort, missing_sentinel),
                reverse=reverse,
            )

        total_pairs = len(pairs)

    return templates.TemplateResponse(
        request,
        "audit/page.html",
        {
            "mode": "audit",
            "datasets": datasets,
            "dataset": dataset,
            "pairs": pairs,
            "total_pairs": total_pairs,
            "label_filter": label or "all",
            "sort_feature": sort,
            "sort_order": order,
            "feature_categories": FEATURE_CATEGORIES,
            "feature_columns": FEATURE_COLUMNS,
        },
    )


@router.get("/pair")
async def audit_pair(
    request: Request,
    dataset: str = Query(...),
    index: int = Query(0),
):
    """HTMX fragment: pair detail with geometry + features by category."""
    data = _load_audit_data(dataset)
    pairs = data["pairs"]

    if index < 0 or index >= len(pairs):
        return templates.TemplateResponse(
            request,
            "audit/pair_detail.html",
            {"pair": None, "dataset": dataset},
        )

    pair = pairs[index]

    # Build geometry JSON for map
    geojson = {}
    if "ref_geom_json" in pair and pair["ref_geom_json"]:
        geojson["reference"] = pair["ref_geom_json"]
    if "target_geom_json" in pair and pair["target_geom_json"]:
        geojson["target"] = pair["target_geom_json"]

    return templates.TemplateResponse(
        request,
        "audit/pair_detail.html",
        {
            "pair": pair,
            "dataset": dataset,
            "pair_index": index,
            "geojson": json.dumps(geojson),
            "feature_categories": FEATURE_CATEGORIES,
        },
    )


@router.get("/distributions")
async def audit_distributions(
    dataset: str = Query(...),
    feature: str = Query(...),
):
    """Return histogram data for a feature: match vs no_match distributions."""
    data = _load_audit_data(dataset)
    df = data["features_df"]

    if feature not in df.columns or len(df) == 0:
        return JSONResponse(content={"bins": [], "match": [], "no_match": []})

    series = df[feature].dropna()
    if len(series) == 0:
        return JSONResponse(content={"bins": [], "match": [], "no_match": []})

    match_vals = df[df["label"] == "match"][feature].dropna()
    no_match_vals = df[df["label"] == "no_match"][feature].dropna()

    # Create 20 bins across full range
    vmin = float(series.min())
    vmax = float(series.max())
    if vmin == vmax:
        vmax = vmin + 1.0

    bins = np.linspace(vmin, vmax, 21)
    bin_centers = ((bins[:-1] + bins[1:]) / 2).tolist()

    match_hist = (
        np.histogram(match_vals, bins=bins)[0].tolist() if len(match_vals) > 0 else [0] * 20
    )
    no_match_hist = (
        np.histogram(no_match_vals, bins=bins)[0].tolist() if len(no_match_vals) > 0 else [0] * 20
    )

    # Normalize to percentages
    match_total = sum(match_hist) or 1
    no_match_total = sum(no_match_hist) or 1
    match_pct = [c / match_total * 100 for c in match_hist]
    no_match_pct = [c / no_match_total * 100 for c in no_match_hist]

    return JSONResponse(
        content={
            "bins": bin_centers,
            "match": match_pct,
            "no_match": no_match_pct,
            "range": [vmin, vmax],
        }
    )
