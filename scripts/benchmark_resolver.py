#!/usr/bin/env python
"""Benchmark: optimizer vs naive vs learned resolver edge selection.

Research-only — compares selection strategies on curated stitching labels
(`labels/stitching/dataset=*/data.csv`) + sidecar groups
(`data/output/*` fallback `data/factory/release=.../dataset=*/groups.json`).

Strategies:
  - optimizer (production): sidecar `selected` flag (keep-all + prune)
  - naive_keepall: keep every candidate (all ones)
  - conf_0.5 / conf_oracle: confidence threshold (fixed 0.5 vs oracle-tuned)
  - learned model (if `data/models/resolver_model.joblib` exists):
    model+threshold (0.5) and model+eF1 (structured expected-F1 per group)

Two views:
  1) In-sample (optimistic for the model, useful for ceiling)
  2) Grouped-CV OOF (leakage-free, the honest comparison — same harness as
     `crosswalk train-resolver`)

Usage:
    uv run python scripts/benchmark_resolver.py
    uv run python scripts/benchmark_resolver.py --dataset us_boston_streets
    uv run python scripts/benchmark_resolver.py --model-path data/models/resolver_model.joblib \\
        --output research/resolver_benchmark.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from crosswalk.resolver.evaluate import _eval_from_predictions, _prf
from crosswalk.resolver.features import (
    FEATURE_COLUMNS,
    featurize,
    group_key_columns,
    group_keys,
)
from crosswalk.resolver.round2 import (
    featurize_extended,
    select_expected_f1,
)
from crosswalk.resolver.train import (
    _build_combined_table,
    _discover_specs,
    predict_keep_probability,
)


def _featurize_for_cols(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, bool]:
    needs_ext = any(c not in FEATURE_COLUMNS for c in feature_cols)
    if needs_ext:
        return featurize_extended(df), True
    return featurize(df), False


def _conf_threshold_preds(conf: np.ndarray, thresh: float) -> np.ndarray:
    conf = np.where(np.isnan(conf), -1.0, conf)
    return (conf >= thresh).astype(int)


def _oracle_conf_threshold(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    y = df["keep"].to_numpy()
    conf = df["confidence"].to_numpy(dtype=float)
    best_t, best_f1 = 0.5, -1.0
    best_pred = _conf_threshold_preds(conf, 0.5)
    for t in np.arange(0.30, 1.0, 0.01):
        pred = _conf_threshold_preds(conf, float(t))
        _, _, f1 = _prf(pred, y)
        if f1 > best_f1:
            best_f1, best_t, best_pred = f1, float(t), pred
    return best_t, best_pred


def _model_preds_ef1(df: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby(group_key_columns(df)).indices.items():
        pred[idx] = select_expected_f1(proba[idx])
    return pred


def _model_payload(model_path: Path):
    if not model_path.exists():
        return None
    try:
        payload = joblib.load(str(model_path))
    except Exception as e:
        print(f"[warn] failed to load model {model_path}: {e}")
        return None
    if "model" not in payload or "feature_columns" not in payload:
        print(f"[warn] model payload missing keys: {list(payload.keys())}")
        return None
    return payload


def _row_dict(label: str, df: pd.DataFrame, pred: np.ndarray) -> dict:
    ev = _eval_from_predictions(label, df, pred)
    return ev.row()


def _print_table(rows: list[dict], title: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = ["model", "edges", "groups", "P", "R", "F1", "grp_exact", "F1_sliverfilt"]
    cols = [c for c in cols if c in df.columns]
    print(f"\n=== {title} ===")
    print(df[cols].to_string(index=False))


def _per_dataset_eval(
    raw_df: pd.DataFrame,
    feat_df: pd.DataFrame | None,
    proba: np.ndarray | None,
    include_model: bool,
) -> list[dict]:
    rows = []
    for ds, sub_raw in raw_df.groupby("dataset_id"):
        sel = sub_raw["selected"].to_numpy()
        rows.append(_row_dict(f"optimizer:{ds}", sub_raw, sel))
        rows.append(_row_dict(f"naive_keepall:{ds}", sub_raw, np.ones(len(sub_raw), dtype=int)))
        t_oracle, pred_oracle = _oracle_conf_threshold(sub_raw)
        rows.append(_row_dict(f"conf_oracle(t={t_oracle:.2f}):{ds}", sub_raw, pred_oracle))
        if include_model and feat_df is not None and proba is not None:
            if len(feat_df) != len(raw_df):
                continue
            sub_feat = feat_df[feat_df["dataset_id"] == ds]
            if len(sub_feat) != len(sub_raw):
                continue
            if not (len(feat_df) == len(raw_df) == len(proba)):
                print(
                    f"[warn] per-ds alignment drift: feat={len(feat_df)} raw={len(raw_df)} proba={len(proba)} — skipping remaining per-ds"
                )
                break
            mask_ds = raw_df["dataset_id"] == ds
            sub_proba = proba[mask_ds.to_numpy()]
            if len(sub_proba) != len(sub_raw):
                print(f"[warn] ds={ds} proba slice {len(sub_proba)} != raw {len(sub_raw)} — skip")
                continue
            rows.append(_row_dict(f"model_thr0.5:{ds}", sub_raw, (sub_proba >= 0.5).astype(int)))
            rows.append(_row_dict(f"model_ef1:{ds}", sub_raw, _model_preds_ef1(sub_raw, sub_proba)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", type=Path, default=Path("."), help="Project root or data root")
    ap.add_argument("--labels-root", type=Path, default=Path("labels/stitching"))
    ap.add_argument("--model-path", type=Path, default=Path("data/models/resolver_model.joblib"))
    ap.add_argument(
        "--dataset",
        "-d",
        action="append",
        dest="datasets",
        default=[],
        help="Only these datasets (repeatable)",
    )
    ap.add_argument("--output", "-o", type=Path, default=Path("research/resolver_benchmark.md"))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--include-split", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--repeat-seeds",
        default="0,1,2,3,4",
        help="Comma-separated split/model seeds for repeated grouped CV",
    )
    ap.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
        help="Whole-group paired bootstrap draws",
    )
    ap.add_argument(
        "--lodo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run leave-one-dataset-out transfer evaluation",
    )
    args = ap.parse_args()

    try:
        repeat_seeds = tuple(int(v.strip()) for v in args.repeat_seeds.split(",") if v.strip())
    except ValueError as exc:
        ap.error(f"--repeat-seeds must be comma-separated integers: {exc}")
    if not repeat_seeds:
        ap.error("--repeat-seeds must contain at least one integer")
    if args.bootstrap_resamples < 1:
        ap.error("--bootstrap-resamples must be >= 1")

    data_root = Path(args.data_root)
    labels_root = Path(args.labels_root)
    model_path = Path(args.model_path)
    dataset_filter = args.datasets if args.datasets else None

    specs = _discover_specs(data_root, labels_root, dataset_filter=dataset_filter)
    if not specs:
        print(f"No specs found under {labels_root}")
        return

    print(f"Discovered {len(specs)} dataset specs")
    for ds, gp, lp in specs:
        status = "exists" if Path(gp).exists() else "MISSING"
        print(f"  - {ds}: {gp} [{status}] labels={lp}")

    raw_df, per_ds_stats, groups_by_ds = _build_combined_table(
        specs,
        include_split=args.include_split,
        include_rejected=True,
        prefer_candidate_graph=True,
        filter_rule5=True,
        include_empty=True,
    )

    if raw_df.empty:
        print("Combined edge table empty — no labeled groups mapped")
        for s in per_ds_stats:
            print(f"  {s}")
        return

    raw_df = raw_df.reset_index(drop=True)
    print(
        f"\nTable: {len(raw_df)} edges / {group_keys(raw_df).nunique()} groups / "
        f"{raw_df['dataset_id'].nunique()} datasets — "
        f"keep=1:{int(raw_df['keep'].sum())} keep=0:{int((raw_df['keep'] == 0).sum())}"
    )

    # ---- In-sample baselines (no features needed) ----
    baselines = []
    baselines.append(_row_dict("optimizer (selected)", raw_df, raw_df["selected"].to_numpy()))
    baselines.append(_row_dict("naive_keepall (all 1)", raw_df, np.ones(len(raw_df), dtype=int)))
    baselines.append(
        _row_dict(
            "conf>=0.5", raw_df, _conf_threshold_preds(raw_df["confidence"].to_numpy(float), 0.5)
        )
    )
    t_oracle, pred_oracle = _oracle_conf_threshold(raw_df)
    baselines.append(
        _row_dict(f"conf_oracle(t={t_oracle:.2f}, optimistic in-sample)", raw_df, pred_oracle)
    )
    _print_table(baselines, "In-sample baselines (no model)")

    # ---- Learned model (in-sample, optimistic) ----
    payload = _model_payload(model_path)
    feat_df = None
    feat_cols: list[str] = []
    proba = None
    in_sample_rows = list(baselines)
    if payload is not None:
        feat_cols = list(payload.get("feature_columns", []))
        sel = payload.get("selector", "ef1")
        print(
            f"\nModel payload: {model_path} — {len(feat_cols)} feats, selector={sel}, keys={list(payload.keys())}"
        )
        print(f"  training_stats: {payload.get('training_stats', {})}")
        print(f"  cv_summary: {payload.get('cv_summary', {})}")
        try:
            feat_df, _is_extended = _featurize_for_cols(raw_df, feat_cols)
        except Exception as e:
            print(f"[warn] featurize failed: {e}")
            feat_df = None

        if feat_df is not None:
            missing = [c for c in feat_cols if c not in feat_df.columns]
            if missing:
                print(f"[warn] feature columns missing from featurized table: {missing[:10]}")
            else:
                X = feat_df[feat_cols].to_numpy(dtype=float)
                try:
                    proba = predict_keep_probability(payload["model"], X)
                except Exception as e:
                    print(f"[warn] model predict failed: {e}")
                    proba = None

        if proba is not None:
            in_sample_rows.append(
                _row_dict("model_thr0.5 (in-sample)", raw_df, (proba >= 0.5).astype(int))
            )
            in_sample_rows.append(
                _row_dict("model_ef1 (in-sample)", raw_df, _model_preds_ef1(raw_df, proba))
            )
            _print_table(in_sample_rows[-2:], "In-sample learned (optimistic)")
    else:
        print(f"\nNo model payload at {model_path} — benchmarking baselines only")

    _print_table(
        in_sample_rows, "All in-sample (optimizer vs naive vs model) — optimistic for model"
    )

    # ---- Grouped CV OOF (honest) ----
    cv_rows = []
    stability_result = None
    lodo_result = None
    if feat_df is not None and payload is not None:
        try:
            from crosswalk.resolver.train import evaluate_all

            feat_cols = list(payload.get("feature_columns", []))
            cv_model_sel = payload.get("selector", "ef1")
            eval_res = evaluate_all(
                feat_df, feat_cols, selector=cv_model_sel, n_splits=args.n_splits, threshold=0.5
            )
            for k in (
                "model",
                "baseline_production",
                "baseline_conf_oracle",
                "baseline_keepall",
                "baseline_conf",
            ):
                v = eval_res.get(k)
                if v is not None and hasattr(v, "row"):
                    cv_rows.append(v.row())
            _print_table(
                cv_rows, f"Grouped CV OOF ({args.n_splits} folds, selector={cv_model_sel}) — honest"
            )

            if cv_rows:
                model_r = eval_res.get("model")
                prod_r = eval_res.get("baseline_production")
                if model_r and prod_r:
                    print(
                        f"\nOOF headline: model F1={model_r.f1:.3f} vs prod F1={prod_r.f1:.3f} "
                        f"Δ={model_r.f1 - prod_r.f1:+.3f} | exact {model_r.group_exact_rate:.3f} vs {prod_r.group_exact_rate:.3f} "
                        f"Δ={model_r.group_exact_rate - prod_r.group_exact_rate:+.3f}"
                    )
        except Exception as e:
            print(f"[warn] grouped CV failed: {e}")

        try:
            from crosswalk.resolver.train import evaluate_repeated_grouped_cv

            stability_result = evaluate_repeated_grouped_cv(
                feat_df,
                feat_cols,
                selector=payload.get("selector", "ef1"),
                n_splits=args.n_splits,
                seeds=repeat_seeds,
                bootstrap_resamples=args.bootstrap_resamples,
            )
            print("\n=== Repeated grouped CV stability ===")
            for run in stability_result["runs"]:
                print(
                    f"seed={run['seed']:>3} F1={run['f1']:.3f} "
                    f"Δprod={run['f1_delta']:+.3f} exact={run['group_exact']:.3f} "
                    f"Δprod={run['group_exact_delta']:+.3f}"
                )
            boot = stability_result["paired_bootstrap"]
            for metric in ("f1", "group_exact"):
                summary = boot[metric]
                print(
                    f"paired bootstrap {metric}: Δ={summary['delta']:+.3f} "
                    f"95% CI [{summary['ci_low']:+.3f}, {summary['ci_high']:+.3f}] "
                    f"bootstrap support(Δ>0)="
                    f"{summary['bootstrap_support_candidate_better']:.3f}"
                )
        except Exception as e:
            raise RuntimeError("repeated grouped CV failed; refusing incomplete report") from e

        if args.lodo and feat_df["dataset_id"].nunique() >= 2:
            try:
                from crosswalk.resolver.train import evaluate_leave_one_dataset_out

                lodo_result = evaluate_leave_one_dataset_out(
                    feat_df,
                    feat_cols,
                    selector=payload.get("selector", "ef1"),
                    seed=args.seed,
                    bootstrap_resamples=args.bootstrap_resamples,
                )
                print("\n=== Leave-one-dataset-out transfer ===")
                for item in lodo_result["per_dataset"]:
                    model_result = item["model"]
                    production_result = item["baseline_production"]
                    print(
                        f"{item['dataset_id']}: edges={model_result.n_edges} "
                        f"F1={model_result.f1:.3f} vs {production_result.f1:.3f} "
                        f"exact={model_result.group_exact_rate:.3f} vs "
                        f"{production_result.group_exact_rate:.3f}"
                    )
                boot = lodo_result["paired_bootstrap"]
                for metric in ("f1", "group_exact"):
                    summary = boot[metric]
                    print(
                        f"LODO paired bootstrap {metric}: Δ={summary['delta']:+.3f} "
                        f"95% CI [{summary['ci_low']:+.3f}, {summary['ci_high']:+.3f}] "
                        f"bootstrap support(Δ>0)="
                        f"{summary['bootstrap_support_candidate_better']:.3f}"
                    )
            except Exception as e:
                raise RuntimeError(
                    "leave-one-dataset-out failed; refusing incomplete report"
                ) from e
        elif args.lodo:
            print("\nLODO skipped: evaluation contains fewer than two datasets")

    # ---- Per-dataset ----
    per_ds_rows = _per_dataset_eval(raw_df, feat_df, proba, include_model=proba is not None)
    _print_table(per_ds_rows, "Per-dataset (in-sample baselines + model if available)")

    # ---- Write markdown ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Resolver Benchmark — optimizer vs naive vs model\n")
    lines.append(
        "> Research-only. Compares M:N group edge selection strategies on curated stitching labels."
    )
    lines.append(
        f"> data_root={data_root} labels={labels_root} model={model_path.name} n_splits={args.n_splits}"
    )
    lines.append("")
    lines.append("## Inventory")
    lines.append("")
    lines.append(f"- specs discovered: {len(specs)}")
    for s in per_ds_stats:
        lines.append(
            f"  - {s.get('dataset_id')}: exists={s.get('exists')} sidecar_groups={s.get('n_sidecar_groups', 0)} "
            f"labels={s.get('n_labels', 0)} rows={s.get('rows', 0)} cand_groups={s.get('build_candidate_groups', 0)} "
            f"legacy_groups={s.get('build_legacy_groups', 0)} pos={s.get('build_positives', 0)} neg={s.get('build_negatives', 0)} "
            f"outside_candidate={s.get('build_human_selected_outside_candidate_graph', 0)} "
            f"outside_clean={s.get('build_human_selected_outside_candidate_graph_clean', 0)} "
            f"outside_split={s.get('build_human_selected_outside_candidate_graph_split', 0)} "
            f"parquet_rows={s.get('build_candidate_parquet_rows', 0)} enriched={s.get('build_candidate_parquet_enriched', 0)} "
            f"empty_rows={s.get('build_empty_rows', 0)} empty_legacy_skipped={s.get('build_empty_legacy_skipped', 0)}"
        )
    lines.append(f"- combined: {len(raw_df)} edges / {group_keys(raw_df).nunique()} groups")
    lines.append(
        f"  - keep=1:{int(raw_df['keep'].sum())} keep=0:{int((raw_df['keep'] == 0).sum())}"
    )
    if "provenance" in raw_df.columns:
        lines.append(f"  - provenance: {raw_df['provenance'].value_counts().to_dict()}")
    lines.append("")

    def _md_table(rows, title):
        if not rows:
            return []
        out = [
            f"### {title}",
            "",
            "| model | edges | groups | P | R | F1 | grp_exact | F1_sliverfilt |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            out.append(
                f"| {r.get('model', '')} | {r.get('edges', '')} | {r.get('groups', '')} | {r.get('P', '')} | {r.get('R', '')} | {r.get('F1', '')} | {r.get('grp_exact', '')} | {r.get('F1_sliverfilt', '')} |"
            )
        out.append("")
        return out

    lines.extend(
        _md_table(in_sample_rows, "In-sample (optimizer vs naive vs model) — optimistic for model")
    )
    if cv_rows:
        lines.extend(_md_table(cv_rows, f"Grouped CV OOF ({args.n_splits} folds) — honest"))
    if stability_result is not None:
        lines.extend(
            [
                "### Repeated stratified grouped CV",
                "",
                "Every edge is held out once per seed; rows from one dataset-scoped group never cross a fold.",
                "",
                "| seed | model F1 | Δ vs prod | model exact | Δ vs prod |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for run in stability_result["runs"]:
            lines.append(
                f"| {run['seed']} | {run['f1']:.4f} | {run['f1_delta']:+.4f} "
                f"| {run['group_exact']:.4f} | {run['group_exact_delta']:+.4f} |"
            )
        lines.extend(["", "Paired whole-group bootstrap on the mean OOF probability decision:", ""])
        lines.extend(
            [
                "| metric | observed Δ | 95% CI | bootstrap support (Δ > 0) |",
                "|---|---:|---:|---:|",
            ]
        )
        boot = stability_result["paired_bootstrap"]
        for metric in ("f1", "group_exact"):
            summary = boot[metric]
            lines.append(
                f"| {metric} | {summary['delta']:+.4f} "
                f"| [{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}] "
                f"| {summary['bootstrap_support_candidate_better']:.3f} |"
            )
        lines.append("")
    if lodo_result is not None:
        lines.extend(
            [
                "### Leave-one-dataset-out transfer",
                "",
                "| held-out dataset | edges | groups | model F1 | prod F1 | model exact | prod exact |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in lodo_result["per_dataset"]:
            model_result = item["model"]
            production_result = item["baseline_production"]
            lines.append(
                f"| {item['dataset_id']} | {model_result.n_edges} | {model_result.n_groups} "
                f"| {model_result.f1:.4f} | {production_result.f1:.4f} "
                f"| {model_result.group_exact_rate:.4f} "
                f"| {production_result.group_exact_rate:.4f} |"
            )
        lines.extend(["", "Paired dataset-cluster bootstrap over pooled LODO predictions:", ""])
        lines.extend(
            [
                "| metric | observed Δ | 95% CI | bootstrap support (Δ > 0) |",
                "|---|---:|---:|---:|",
            ]
        )
        boot = lodo_result["paired_bootstrap"]
        for metric in ("f1", "group_exact"):
            summary = boot[metric]
            lines.append(
                f"| {metric} | {summary['delta']:+.4f} "
                f"| [{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}] "
                f"| {summary['bootstrap_support_candidate_better']:.3f} |"
            )
        lines.append("")
    if per_ds_rows:
        lines.extend(_md_table(per_ds_rows, "Per-dataset in-sample"))

    lines.append("## Interpretation / limitations")
    lines.append("")
    lines.append(
        "- Four labeled datasets still use legacy sidecars without `candidate_edges`, so their under-selection universe is capped (64/group); Tunis is missing entirely."
    )
    lines.append(
        "- Legacy reject-all labels on legacy groups emit zero rows (honest cross-mode handling via `empty_legacy_skipped`)."
    )
    lines.append(
        "- In-sample model numbers are optimistic (data leakage). Use repeated grouped CV, its paired whole-group interval, and LODO transfer for the NO-GO decision."
    )
    lines.append(
        "- `optimizer (selected)` = sidecar `selected` flag: keep-all + vertex-junction tolerance + confidence prune (#284)."
    )
    lines.append("- `naive_keepall` = every candidate edge kept — precision floor, recall ceiling.")
    lines.append(
        "- Honest OOF F1 remains below production, while group exact is modestly higher. This is a useful prototype, not a production resolver."
    )
    lines.append(
        "- Typed `<ds>_candidates.parquet` data is joined fail-closed, but was locally available only for Seattle; the model still uses 33 sidecar/context features."
    )
    lines.append(
        "- Next evidence gate: regenerate fresh candidate graphs/parquets, recover the 20 Boston + 6 Seattle positive edges outside the candidate universe, and label ≥20 reject-all groups."
    )
    lines.append(
        "- Then run paired grouped-CV removal/permutation ablations by feature family; add panel votes only when candidate-display provenance makes unselected edges and NONE votes interpretable."
    )
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
