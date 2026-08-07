"""Measure the F1 / speed / memory tradeoff of widening SPARK_PORTABLE_FEATURES.

The shipped Spark-portable model trains on 28 of the 83 FEATURE_COLUMNS. The
stated cut is "computable from aligned geometry pairs (no topology, graph, or
spatial-index features required)", but 17 excluded features need nothing beyond
the two aligned geometries and the two name structs the Spark job already holds
(see tests/test_spark_feature_expansion.py for the proof-by-computation).

This script scores each candidate tier on the metrics that decide whether to
widen the contract:

  * F1  -- grouped-holdout + GroupKFold CV (the numbers `export-spark-model`
           itself prints) and LOO-by-type CV (the cross-dataset generalization
           metric from eval_utils.py), all under SPARK_PORTABLE_XGB_PARAMS so
           the only moving part is the feature set.
  * speed  -- booster inference throughput per row. The per-pair *feature
           computation* cost is measured separately, by
           tests/test_spark_feature_expansion.py::test_addable_feature_marginal_cost.
  * memory -- exported booster JSON size and peak RSS during inference (measured
           in a fresh subprocess so the peak is per-tier, not cumulative).

Findings live in research/spark_feature_expansion_2026-08-07.md.

Usage:
    uv run python research/spark_feature_expansion.py            # F1 + model tiers
    uv run python research/spark_feature_expansion.py --quick    # skip LOO CV
    uv run python research/spark_feature_expansion.py --seeds 42,1,2,3,4 \\
        --out research/results/spark_feature_expansion_2026-08-07.json
    uv run python research/spark_feature_expansion.py --per-feature --seeds 42,1,2
"""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# The 7 name features excluded from the Spark set. Every one is a pure function
# of the two resolved name strings -- the same strings name_levenshtein /
# name_token_sort / name_numeric_match (already shipped) are computed from.
ADDABLE_NAME_FEATURES = [
    "name_jaro_winkler",
    "name_soundex",
    "name_metaphone",
    "has_name_ref",
    "has_name_target",
    "name_is_generic",
    "route_prefix_match",
]

# The 10 excluded geometry features that need only the aligned geometry pair.
# max_coverage and sinuosity_delta are pure arithmetic on columns the 28-feature
# model already carries, so they cost literally nothing to add.
ADDABLE_GEOMETRY_FEATURES = [
    "max_coverage",
    "sinuosity_delta",
    "angle_histogram_similarity",
    "shape_complexity_ref",
    "shape_complexity_delta",
    "heading_consistency_ref",
    "heading_consistency_delta",
    "vertex_density_ref",
    "vertex_density_target",
    "vertex_density_ratio",
]

# Subset of the above that is derivable from the shipped 28 columns with no new
# geometry pass at all: max(ref_coverage, target_coverage) and
# abs(sinuosity_ref - sinuosity_target).
FREE_DERIVED_FEATURES = ["max_coverage", "sinuosity_delta"]


def build_tiers() -> dict[str, list[str]]:
    """Feature sets to compare, in increasing order of Spark-side work."""
    from crosswalk.config import FEATURE_COLUMNS, SPARK_PORTABLE_FEATURES

    base = list(SPARK_PORTABLE_FEATURES)
    return {
        "t0_baseline_28": base,
        "t1_free_derived_30": base + FREE_DERIVED_FEATURES,
        # Minimal tier: --per-feature shows has_name_target carries essentially
        # the whole name-block lift on its own, so it gets its own row.
        "t2a_name_presence_29": base + ["has_name_target"],
        "t2b_name_presence_30": base + ["has_name_ref", "has_name_target"],
        "t2_names_35": base + ADDABLE_NAME_FEATURES,
        "t3_geometry_38": base + ADDABLE_GEOMETRY_FEATURES,
        "t4_all_feasible_45": base + ADDABLE_NAME_FEATURES + ADDABLE_GEOMETRY_FEATURES,
        # Reference ceiling: everything, including the topology/graph/spatial-index
        # features a Spark job genuinely cannot compute pairwise.
        "ref_full_83": list(FEATURE_COLUMNS),
    }


def holdout_and_cv_f1(
    labels_dir: Path, features: list[str], seed: int, model_out: Path
) -> dict[str, Any]:
    """Train one tier the way `crosswalk export-spark-model` does.

    Same grouped holdout, same GroupKFold CV over the training rows, same
    hyperparams -- only ``exclude_features`` differs between tiers. The exported
    booster is written to ``model_out`` for the inference/memory bench.
    """
    from crosswalk.config import FEATURE_COLUMNS, SPARK_PORTABLE_XGB_PARAMS
    from crosswalk.matching.ml import MLMatcher

    keep = set(features)
    exclude = [f for f in FEATURE_COLUMNS if f not in keep]

    t0 = time.perf_counter()
    matcher = MLMatcher()
    metrics = matcher.train(
        labels_dir=labels_dir,
        test_size=0.2,
        binary=True,
        exclude_features=exclude,
        seed=seed,
        **SPARK_PORTABLE_XGB_PARAMS,
    )
    train_s = time.perf_counter() - t0

    model_out.parent.mkdir(parents=True, exist_ok=True)
    matcher.model.get_booster().save_model(str(model_out))

    return {
        "n_features": len(matcher.feature_names),
        "feature_names": list(matcher.feature_names),
        "cv_f1_mean": float(metrics["cv_f1_mean"]),
        "cv_f1_std": float(metrics["cv_f1_std"]),
        "test_f1_raw": float(metrics["test_f1_raw"]),
        "test_f1_production": float(metrics["test_f1_production"]),
        "test_accuracy": float(metrics["test_accuracy"]),
        "model_json_kb": model_out.stat().st_size / 1024,
        "train_seconds": train_s,
        "feature_importance": {
            k: float(v)
            for k, v in sorted(metrics["feature_importance"].items(), key=lambda kv: -kv[1])
        },
    }


def loo_f1(
    labels: Any,
    features: list[str],
    seed: int,
    xgb_params: dict[str, Any] | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """LOO-by-type CV (eval_utils.run_loo_by_type_cv) restricted to a feature set.

    ``run_loo_by_type_cv`` has no feature-subset knob, so this mirrors its fold
    construction and metric computation exactly while slicing the columns. Kept
    in sync by ``tests/test_spark_feature_expansion.py::
    test_loo_harness_reproduces_eval_utils_on_full_feature_set``.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    from xgboost import XGBClassifier

    from crosswalk.config import METRIC_AVERAGE, SPARK_PORTABLE_XGB_PARAMS
    from crosswalk.eval_utils import MIN_LOO_LABELS, build_type_groups
    from crosswalk.matching.ml import DEFAULT_XGB_PARAMS, MLMatcher

    params = {
        **DEFAULT_XGB_PARAMS,
        **(SPARK_PORTABLE_XGB_PARAMS if xgb_params is None else xgb_params),
    }
    for key in ("scale_pos_weight", "random_state", "n_jobs"):
        params.pop(key, None)

    df = labels[labels["label"].isin({"match", "no_match"})].copy()
    df = df.drop_duplicates(subset=["gers_id", "target_id", "dataset"], keep="last")
    counts = df.groupby("dataset").size()
    valid = counts[counts >= MIN_LOO_LABELS].index.tolist()
    df = df[df["dataset"].isin(valid)].copy()

    groups = build_type_groups(valid)
    ds_to_group = {ds: g for g, names in groups.items() for ds in names}

    matcher = MLMatcher()
    cols = [f for f in features if f in df.columns]
    rows = []
    for held in sorted(valid):
        train_df = df[df["dataset"] != held]
        test_df = df[df["dataset"] == held]
        X_train = matcher._cap_infinities(train_df[cols].to_numpy(dtype=np.float32))
        y_train = (train_df["label"] == "match").astype(int).to_numpy()
        n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        model = XGBClassifier(
            **params, scale_pos_weight=n_neg / n_pos, random_state=seed, n_jobs=n_jobs
        )
        model.fit(X_train, y_train)

        X_test = matcher._cap_infinities(test_df[cols].to_numpy(dtype=np.float32))
        y_test = (test_df["label"] == "match").astype(int).to_numpy()
        y_pred = model.predict(X_test)
        rows.append(
            {
                "dataset": held,
                "type_group": ds_to_group[held],
                "n_test": len(test_df),
                "f1": f1_score(y_test, y_pred, average=METRIC_AVERAGE, zero_division=0),
                "precision": precision_score(
                    y_test, y_pred, average=METRIC_AVERAGE, zero_division=0
                ),
                "recall": recall_score(y_test, y_pred, average=METRIC_AVERAGE, zero_division=0),
            }
        )

    f1s = np.array([r["f1"] for r in rows])
    by_group: dict[str, float] = {}
    for group in sorted({r["type_group"] for r in rows}):
        vals = [r["f1"] for r in rows if r["type_group"] == group]
        by_group[group] = float(np.mean(vals))
    return {
        "loo_f1_mean": float(f1s.mean()),
        "loo_f1_std": float(f1s.std(ddof=0)),
        "loo_n_folds": len(rows),
        "loo_f1_by_group": by_group,
        "loo_rows": rows,
    }


def per_feature_deltas(labels: Any, seeds: list[int]) -> list[dict[str, Any]]:
    """Rank each addable feature by its solo LOO-F1 lift over the 28 baseline.

    One feature at a time on top of ``SPARK_PORTABLE_FEATURES``, averaged over
    ``seeds``, reported as a paired per-dataset delta. Paired because the LOO
    folds are deterministic (one per dataset), so the same 33 datasets are
    compared like-for-like and the seed-to-seed noise floor (~0.001 LOO F1) is
    an order of magnitude below the interesting deltas.
    """
    from crosswalk.config import SPARK_PORTABLE_FEATURES

    base = list(SPARK_PORTABLE_FEATURES)
    baseline_runs = [loo_f1(labels, base, seed) for seed in seeds]
    baseline_by_ds: dict[str, list[float]] = {}
    for run in baseline_runs:
        for row in run["loo_rows"]:
            baseline_by_ds.setdefault(row["dataset"], []).append(row["f1"])

    ranked = []
    for feature in ADDABLE_NAME_FEATURES + ADDABLE_GEOMETRY_FEATURES:
        runs = [loo_f1(labels, base + [feature], seed) for seed in seeds]
        by_ds: dict[str, list[float]] = {}
        for run in runs:
            for row in run["loo_rows"]:
                by_ds.setdefault(row["dataset"], []).append(row["f1"])
        deltas = {
            ds: float(np.mean(vals)) - float(np.mean(baseline_by_ds[ds]))
            for ds, vals in by_ds.items()
        }
        arr = np.array(list(deltas.values()))
        ranked.append(
            {
                "feature": feature,
                "loo_f1_mean": float(np.mean([r["loo_f1_mean"] for r in runs])),
                "delta_vs_baseline": float(arr.mean()),
                "n_datasets_better": int((arr > 1e-9).sum()),
                "n_datasets_worse": int((arr < -1e-9).sum()),
                "n_datasets_unchanged": int((np.abs(arr) <= 1e-9).sum()),
                "best_dataset": max(deltas.items(), key=lambda kv: kv[1]),
                "worst_dataset": min(deltas.items(), key=lambda kv: kv[1]),
            }
        )
        print(
            f"  {feature:32s} loo_f1={ranked[-1]['loo_f1_mean']:.4f} "
            f"delta={ranked[-1]['delta_vs_baseline']:+.4f} "
            f"win/loss={ranked[-1]['n_datasets_better']}/{ranked[-1]['n_datasets_worse']}",
            flush=True,
        )

    ranked.sort(key=lambda r: -r["delta_vs_baseline"])
    return [
        {
            "baseline_loo_f1": float(np.mean([r["loo_f1_mean"] for r in baseline_runs])),
            "seeds": seeds,
        },
        *ranked,
    ]


def inference_bench(model_path: Path, n_features: int, n_rows: int, repeats: int) -> dict[str, Any]:
    """Booster throughput + peak RSS, measured the way a Spark executor sees it.

    Runs in this process against the *exported* JSON booster (not the sklearn
    wrapper), so the numbers describe what the Spark scorer actually loads.
    Called via ``--bench-model`` in a fresh subprocess so ``ru_maxrss`` is a
    per-tier peak rather than a running maximum over the whole sweep.
    """
    import xgboost as xgb

    rng = np.random.default_rng(0)
    X = rng.random((n_rows, n_features), dtype=np.float32)

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    booster.predict(xgb.DMatrix(X[:1024]))  # warm up

    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        booster.predict(xgb.DMatrix(X))
        timings.append(time.perf_counter() - t0)

    return {
        "inference_rows": n_rows,
        "inference_median_s": float(np.median(timings)),
        "inference_us_per_row": float(np.median(timings) / n_rows * 1e6),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def _bench_in_subprocess(
    model_path: Path, n_features: int, n_rows: int, repeats: int
) -> dict[str, Any]:
    """Run :func:`inference_bench` in a clean interpreter (isolated peak RSS)."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--bench-model",
            str(model_path),
            "--bench-n-features",
            str(n_features),
            "--inference-rows",
            str(n_rows),
            "--inference-repeats",
            str(repeats),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("labels"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated seeds to repeat every tier over. The per-tier F1 gaps "
            "are small enough that a single seed cannot separate them from XGBoost's "
            "own row/column-sampling noise; use this to get error bars."
        ),
    )
    parser.add_argument("--quick", action="store_true", help="skip LOO-by-type CV")
    parser.add_argument(
        "--per-feature",
        action="store_true",
        help="rank each addable feature by its solo LOO-F1 lift instead of running tiers",
    )
    parser.add_argument("--inference-rows", type=int, default=200_000)
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    # Internal: re-entrant mode used by _bench_in_subprocess for isolated RSS.
    parser.add_argument("--bench-model", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bench-n-features", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.bench_model is not None:
        print(
            json.dumps(
                inference_bench(
                    args.bench_model,
                    args.bench_n_features,
                    args.inference_rows,
                    args.inference_repeats,
                )
            )
        )
        return

    from crosswalk.labeling.label_store import LabelStore

    labels = LabelStore.load_all(args.labels)
    tiers = build_tiers()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    if args.per_feature:
        ranked = per_feature_deltas(labels, seeds)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
            print(f"\nWrote {args.out}")
        return

    results: dict[str, list[dict[str, Any]]] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for name, features in tiers.items():
            print(f"\n=== {name} ({len(features)} features) ===", flush=True)
            per_seed = []
            for seed in seeds:
                model_path = Path(tmp) / f"{name}_{seed}.json"
                entry = holdout_and_cv_f1(args.labels, features, seed, model_path)
                entry["seed"] = seed
                entry.update(
                    _bench_in_subprocess(
                        model_path,
                        entry["n_features"],
                        args.inference_rows,
                        args.inference_repeats,
                    )
                )
                if not args.quick:
                    t0 = time.perf_counter()
                    entry.update(loo_f1(labels, features, seed))
                    entry["loo_seconds"] = time.perf_counter() - t0
                per_seed.append(entry)
                print(
                    f"  seed={seed} cv_f1={entry['cv_f1_mean']:.4f}+-{entry['cv_f1_std']:.4f} "
                    f"test_f1={entry['test_f1_production']:.4f} "
                    f"loo_f1={entry.get('loo_f1_mean', float('nan')):.4f} "
                    f"model={entry['model_json_kb']:.0f}KB "
                    f"infer={entry['inference_us_per_row']:.2f}us/row "
                    f"rss={entry['peak_rss_mb']:.0f}MB",
                    flush=True,
                )
            results[name] = per_seed
            if len(seeds) > 1:
                for metric in ("cv_f1_mean", "test_f1_production", "loo_f1_mean"):
                    vals = [e[metric] for e in per_seed if metric in e]
                    if vals:
                        print(
                            f"  across {len(vals)} seeds {metric}: "
                            f"{np.mean(vals):.4f} +- {np.std(vals, ddof=0):.4f}",
                            flush=True,
                        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
