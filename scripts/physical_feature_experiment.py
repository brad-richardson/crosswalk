#!/usr/bin/env python3
"""Evaluate additive pairwise physical-road feature combinations.

The experiment reads the existing human pair corpus, refreshes physical LR
attributes from retained raw snapshots, and computes features at each label's
stored alignment fractions. It does not rewrite labels, feature parquets, or
the production model. Results include global grouped-CV metrics, the smaller
physical-comparable/informative slices, and an OOF disagreement queue for
manual review.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from crosswalk.config import FEATURE_COLUMNS
from crosswalk.datasets.schema import get_dataset_config
from crosswalk.features.physical import (
    PHYSICAL_COMPOSITE_FEATURES,
    PHYSICAL_EXPERIMENT_FEATURES,
    PHYSICAL_FLAG_FEATURES,
    PHYSICAL_VERTICAL_FEATURES,
    compute_physical_pair_features,
)
from crosswalk.fetch.overture import backfill_overture_physical_lr
from crosswalk.fetch.target import backfill_physical_lr_from_source_tags
from crosswalk.labeling.label_store import LabelStore
from crosswalk.matching.ml import DEFAULT_XGB_PARAMS, MLMatcher, create_segment_groups

TARGET_PROPERTY_DATASETS = (
    "au_sydney_roads",
    "fi_helsinki_roads",
    "gb_london_roads",
    "hk_hongkong_roads",
    "de_berlin_roads",
    "nl_amsterdam_roads",
    "us_utah_slc_roads",
)

FEATURE_VARIANTS = {
    "baseline": (),
    "flag_positive_only": ("physical_flag_positive_match",),
    "vertical_positive_only": ("vertical_positive_match",),
    "positive_only": ("physical_positive_match",),
    "conflict_only": ("physical_structure_conflict",),
    "coverage_only": ("physical_comparable_count",),
    "composite_no_coverage": (
        "physical_structure_conflict",
        "physical_positive_match",
    ),
    "flags": PHYSICAL_FLAG_FEATURES,
    "vertical": PHYSICAL_VERTICAL_FEATURES,
    "composite": PHYSICAL_COMPOSITE_FEATURES,
    "primitives": (*PHYSICAL_FLAG_FEATURES, *PHYSICAL_VERTICAL_FEATURES),
    "all_no_coverage": tuple(
        feature
        for feature in PHYSICAL_EXPERIMENT_FEATURES
        if feature != "physical_comparable_count"
    ),
    "all": PHYSICAL_EXPERIMENT_FEATURES,
}


def _plain_name(value: Any) -> str:
    if not isinstance(value, dict):
        return "" if value is None else str(value)
    primary = value.get("primary")
    return "" if primary is None else str(primary)


def _target_domains(dataset_id: str) -> tuple[set[str], bool, Any]:
    config = get_dataset_config(dataset_id)
    if config is None or config.fetch is None:
        return set(), False, None
    fetch = config.fetch
    return set(fetch.physical_flag_domains()), bool(fetch.level_column), fetch


def _physical_lookups(
    dataset_id: str, raw_dir: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str], bool]:
    ref_path = raw_dir / f"{dataset_id}_overture_segments_v1.0.parquet"
    target_path = raw_dir / f"{dataset_id}_v1.0.parquet"
    if not ref_path.exists() or not target_path.exists():
        return {}, {}, set(), False

    ref = pd.read_parquet(
        ref_path,
        columns=["id", "names", "road_flags", "level_rules"],
    )
    ref = backfill_overture_physical_lr(ref)

    target_domains, target_has_level, fetch = _target_domains(dataset_id)
    target = pd.read_parquet(target_path, columns=["id", "names", "source_tags"])
    if fetch is not None:
        target = backfill_physical_lr_from_source_tags(target, fetch)

    ref_lookup = {
        str(row.id): {
            "level_lr": row.level_lr,
            "road_flags_lr": row.road_flags_lr,
            "name": _plain_name(row.names),
        }
        for row in ref.itertuples(index=False)
    }
    target_lookup = {}
    for row in target.itertuples(index=False):
        target_lookup[str(row.id)] = {
            "level_lr": getattr(row, "level_lr", None) if target_has_level else None,
            "road_flags_lr": (getattr(row, "road_flags_lr", None) if target_domains else None),
            "name": _plain_name(row.names),
        }
    return ref_lookup, target_lookup, target_domains, target_has_level


def add_physical_features(frame: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Return a copy with physical experimental features and audit names."""
    result = frame.copy()
    for feature in PHYSICAL_EXPERIMENT_FEATURES:
        result[feature] = np.nan
    result["physical_ref_name"] = ""
    result["physical_target_name"] = ""
    result["physical_target_domains"] = ""

    for dataset_id in TARGET_PROPERTY_DATASETS:
        mask = result["dataset"].eq(dataset_id)
        if not mask.any():
            continue
        ref_lookup, target_lookup, target_domains, target_has_level = _physical_lookups(
            dataset_id, raw_dir
        )
        if not ref_lookup or not target_lookup:
            continue
        domains_text = ",".join(
            [*sorted(target_domains), *(("level",) if target_has_level else ())]
        )
        for idx, row in result.loc[mask].iterrows():
            ref = ref_lookup.get(str(row["gers_id"]))
            target = target_lookup.get(str(row["target_id"]))
            if ref is None or target is None:
                continue
            features = compute_physical_pair_features(
                ref_level_lr=ref["level_lr"],
                target_level_lr=target["level_lr"],
                ref_road_flags_lr=ref["road_flags_lr"],
                target_road_flags_lr=target["road_flags_lr"],
                ref_start_frac=float(row["ref_start_pct"]),
                ref_end_frac=float(row["ref_end_pct"]),
                target_start_frac=float(row["target_start_pct"]),
                target_end_frac=float(row["target_end_pct"]),
                target_flag_domains=target_domains,
            )
            for feature, value in features.items():
                result.at[idx, feature] = value
            result.at[idx, "physical_ref_name"] = ref["name"]
            result.at[idx, "physical_target_name"] = target["name"]
            result.at[idx, "physical_target_domains"] = domains_text
    return result


def _fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    seed: int,
) -> np.ndarray:
    import xgboost as xgb

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    params = {
        **DEFAULT_XGB_PARAMS,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": -1,
        "scale_pos_weight": n_neg / n_pos if n_pos else 1.0,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    return model.predict_proba(X_val)[:, 1]


def _metrics(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    y_slice = y[mask]
    p_slice = probability[mask]
    if len(y_slice) == 0:
        return {"n": 0, "positive": 0, "f1": None, "precision": None, "recall": None}
    prediction = (p_slice >= 0.5).astype(int)
    return {
        "n": int(len(y_slice)),
        "positive": int(y_slice.sum()),
        "f1": float(f1_score(y_slice, prediction, zero_division=0)),
        "precision": float(precision_score(y_slice, prediction, zero_division=0)),
        "recall": float(recall_score(y_slice, prediction, zero_division=0)),
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n_runs": len(runs)}
    for slice_name in ("global", "comparable", "informative"):
        result[slice_name] = {}
        for metric in ("n", "positive", "f1", "precision", "recall"):
            values = [
                run[slice_name][metric] for run in runs if run[slice_name][metric] is not None
            ]
            if not values:
                result[slice_name][metric + "_mean"] = None
                result[slice_name][metric + "_std"] = None
                continue
            result[slice_name][metric + "_mean"] = float(np.mean(values))
            result[slice_name][metric + "_std"] = float(np.std(values))
    result["per_dataset"] = {}
    dataset_ids = sorted({dataset_id for run in runs for dataset_id in run.get("per_dataset", {})})
    for dataset_id in dataset_ids:
        dataset_runs = [
            run["per_dataset"][dataset_id]
            for run in runs
            if dataset_id in run.get("per_dataset", {})
        ]
        result["per_dataset"][dataset_id] = {}
        for metric in ("n", "positive", "f1", "precision", "recall"):
            values = [item[metric] for item in dataset_runs if item[metric] is not None]
            result["per_dataset"][dataset_id][metric + "_mean"] = (
                float(np.mean(values)) if values else None
            )
            result["per_dataset"][dataset_id][metric + "_std"] = (
                float(np.std(values)) if values else None
            )
    return result


def _signal_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in PHYSICAL_EXPERIMENT_FEATURES:
        known = frame[frame[feature].notna()]
        active = known[known[feature] > 0]
        rows.append(
            {
                "feature": feature,
                "n_known": int(len(known)),
                "n_active": int(len(active)),
                "active_match_rate": (
                    float(active["label"].eq("match").mean()) if len(active) else None
                ),
                "active_by_dataset": [
                    {
                        "dataset_id": str(dataset_id),
                        "n": int(len(group)),
                        "n_match": int(group["label"].eq("match").sum()),
                        "n_no_match": int(group["label"].eq("no_match").sum()),
                    }
                    for dataset_id, group in active.groupby("dataset", sort=True)
                ],
            }
        )
    return rows


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _manual_queue(
    frame: pd.DataFrame,
    y: np.ndarray,
    baseline_probability: np.ndarray,
    physical_probability: np.ndarray,
    limit: int = 24,
) -> list[dict[str, Any]]:
    audit = frame[
        [
            "dataset",
            "gers_id",
            "target_id",
            "label",
            "original_confidence",
            "physical_ref_name",
            "physical_target_name",
            "physical_target_domains",
            *PHYSICAL_EXPERIMENT_FEATURES,
        ]
    ].copy()
    audit["baseline_oof_probability"] = baseline_probability
    audit["physical_oof_probability"] = physical_probability
    audit["probability_delta"] = physical_probability - baseline_probability
    audit["abs_probability_delta"] = audit["probability_delta"].abs()
    audit["category"] = "informative"
    audit.loc[
        (y == 0) & (baseline_probability >= 0.5) & (physical_probability < 0.5),
        "category",
    ] = "physical_variant_corrects_false_positive"
    audit.loc[
        (y == 1) & (baseline_probability >= 0.5) & (physical_probability < 0.5),
        "category",
    ] = "possible_physical_overreach"
    audit.loc[
        (audit["physical_structure_conflict"] >= 0.5) & (baseline_probability >= 0.7),
        "category",
    ] = "high_baseline_despite_physical_conflict"

    informative = audit[
        (audit["physical_structure_conflict"].fillna(0) > 0)
        | (audit["physical_positive_match"].fillna(0) > 0)
    ].copy()
    informative = informative.sort_values(
        ["abs_probability_delta", "physical_structure_conflict"], ascending=False
    )

    selected: list[pd.Series] = []
    per_dataset: dict[str, int] = {}
    per_category: dict[str, int] = {}
    for _, row in informative.iterrows():
        dataset = str(row["dataset"])
        category = str(row["category"])
        if per_dataset.get(dataset, 0) >= 5 or per_category.get(category, 0) >= 8:
            continue
        selected.append(row)
        per_dataset[dataset] = per_dataset.get(dataset, 0) + 1
        per_category[category] = per_category.get(category, 0) + 1
        if len(selected) >= limit:
            break

    return [
        {str(key): _json_scalar(value) for key, value in row.to_dict().items()} for row in selected
    ]


def run_experiment(
    labels_dir: Path,
    raw_dir: Path,
    seeds: tuple[int, ...],
    n_folds: int,
) -> dict[str, Any]:
    frame = LabelStore.load_all(labels_dir, skip_errors=False)
    frame = frame[frame["label"].isin({"match", "no_match"})].copy()
    frame = add_physical_features(frame, raw_dir)

    matcher = MLMatcher()
    matcher.feature_names = FEATURE_COLUMNS.copy()
    frame = matcher._validate_training_pairs(frame, max_hausdorff_m=1000.0).reset_index(drop=True)
    y = frame["label"].eq("match").astype(int).to_numpy()
    groups = create_segment_groups(frame)
    folds = list(GroupKFold(n_splits=n_folds).split(frame, y, groups=groups))

    comparable_mask = frame["physical_comparable_count"].fillna(0).gt(0).to_numpy()
    informative_mask = (
        frame["physical_structure_conflict"].fillna(0).gt(0)
        | frame["physical_positive_match"].fillna(0).gt(0)
    ).to_numpy()

    runs_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in FEATURE_VARIANTS}
    probabilities_by_variant: dict[str, list[np.ndarray]] = {
        variant: [] for variant in FEATURE_VARIANTS
    }
    for seed in seeds:
        for variant, physical_features in FEATURE_VARIANTS.items():
            feature_names = [*FEATURE_COLUMNS, *physical_features]
            X = frame.reindex(columns=feature_names).to_numpy(dtype=np.float32)
            X[np.isinf(X)] = np.nan
            oof_probability = np.full(len(frame), np.nan, dtype=float)
            for train_idx, val_idx in folds:
                oof_probability[val_idx] = _fit_predict(
                    X[train_idx], y[train_idx], X[val_idx], seed
                )
            if np.isnan(oof_probability).any():
                raise RuntimeError(f"OOF prediction incomplete for {variant}, seed={seed}")
            runs_by_variant[variant].append(
                {
                    "seed": seed,
                    "global": _metrics(y, oof_probability, np.ones(len(frame), dtype=bool)),
                    "comparable": _metrics(y, oof_probability, comparable_mask),
                    "informative": _metrics(y, oof_probability, informative_mask),
                    "per_dataset": {
                        dataset_id: _metrics(
                            y,
                            oof_probability,
                            frame["dataset"].eq(dataset_id).to_numpy(),
                        )
                        for dataset_id in TARGET_PROPERTY_DATASETS
                        if frame["dataset"].eq(dataset_id).any()
                    },
                }
            )
            probabilities_by_variant[variant].append(oof_probability)

    summary = {variant: _aggregate_runs(runs) for variant, runs in runs_by_variant.items()}
    baseline = summary["baseline"]
    for _variant, metrics in summary.items():
        for slice_name in ("global", "comparable", "informative"):
            f1 = metrics[slice_name]["f1_mean"]
            baseline_f1 = baseline[slice_name]["f1_mean"]
            metrics[slice_name]["f1_delta_vs_baseline"] = (
                None if f1 is None or baseline_f1 is None else float(f1 - baseline_f1)
            )
        for dataset_id, dataset_metrics in metrics["per_dataset"].items():
            f1 = dataset_metrics["f1_mean"]
            baseline_f1 = baseline["per_dataset"][dataset_id]["f1_mean"]
            dataset_metrics["f1_delta_vs_baseline"] = (
                None if f1 is None or baseline_f1 is None else float(f1 - baseline_f1)
            )

    coverage = []
    for dataset_id in TARGET_PROPERTY_DATASETS:
        subset = frame[frame["dataset"].eq(dataset_id)]
        if subset.empty:
            continue
        coverage.append(
            {
                "dataset_id": dataset_id,
                "n_labeled_pairs": int(len(subset)),
                "target_domains": sorted(
                    {
                        domain
                        for value in subset["physical_target_domains"].dropna()
                        for domain in str(value).split(",")
                        if domain
                    }
                ),
                "n_comparable": int(subset["physical_comparable_count"].gt(0).sum()),
                "n_informative": int(
                    (
                        subset["physical_structure_conflict"].fillna(0).gt(0)
                        | subset["physical_positive_match"].fillna(0).gt(0)
                    ).sum()
                ),
                "n_conflicts": int(subset["physical_structure_conflict"].fillna(0).gt(0).sum()),
                "n_positive_matches": int(subset["physical_positive_match"].fillna(0).gt(0).sum()),
            }
        )

    baseline_probability = np.mean(probabilities_by_variant["baseline"], axis=0)
    physical_probability = np.mean(probabilities_by_variant["all"], axis=0)
    return {
        "protocol": {
            "seeds": list(seeds),
            "n_folds": n_folds,
            "split": "segment-component GroupKFold",
            "threshold": 0.5,
            "n_rows": int(len(frame)),
            "n_groups": int(groups.nunique()),
            "feature_variants": {
                name: list(features) for name, features in FEATURE_VARIANTS.items()
            },
        },
        "coverage": coverage,
        "signal_summary": _signal_summary(frame),
        "summary": summary,
        "runs": runs_by_variant,
        "manual_review_queue": _manual_queue(frame, y, baseline_probability, physical_probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, default=Path("labels"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/physical_feature_ablation_2026-07-15.json"),
    )
    parser.add_argument("--seeds", default="42,73,999")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    result = run_experiment(args.labels_dir, args.raw_dir, seeds, args.folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    print(json.dumps(result["protocol"], indent=2))
    for variant, metrics in result["summary"].items():
        print(
            f"{variant:16s} global={metrics['global']['f1_mean']:.4f} "
            f"({metrics['global']['f1_delta_vs_baseline']:+.4f}) "
            f"comparable={metrics['comparable']['f1_mean']:.4f} "
            f"({metrics['comparable']['f1_delta_vs_baseline']:+.4f}) "
            f"informative={metrics['informative']['f1_mean']:.4f} "
            f"({metrics['informative']['f1_delta_vs_baseline']:+.4f})"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
