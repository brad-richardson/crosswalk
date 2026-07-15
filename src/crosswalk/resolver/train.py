"""Experimental resolver training: multi-dataset candidate-joined ablation harness.

Research-only — nothing in the production pipeline imports this module
(same guard as the rest of ``crosswalk.resolver``). Produces a joblib model
+ a ``research/learned_stitcher_round3.md`` report.

Data sources:
  - Curated labels ``labels/stitching/dataset=*/data.csv`` (178 pair + 48 set,
    52 empty ``[]`` reject-all on main as of 2026-07-10).
  - Sidecars ``data/output/*_groups.json`` if present else
    ``data/factory/release=…/dataset=*/groups.json`` (older release has no
    ``candidate_edges``, falls back to ``edges``+capped ``rejected_edges``).
  - Panel votes ``data/agents/stitching/batches/*/votes.csv`` as an opt-in soft extra.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crosswalk.config import FEATURE_VERSION
from crosswalk.resolver.extract import (
    build_edge_table,
    discover_candidates_parquet,
    load_candidates_parquet,
    load_sidecar_groups,
    load_stitching_labels,
)
from crosswalk.resolver.features import featurize, group_keys
from crosswalk.resolver.round2 import (
    TRAIN_LABEL_COLUMN,
    featurize_extended,
    run_cv2,
    select_group_predictions,
)
from crosswalk.resolver.votes import default_votes_paths, edge_soft_labels, load_votes


def _sanitize_ds(ds: str, max_len: int = 80) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ds)
    safe = safe.strip("_") or "dataset"
    return safe[:max_len]


def _discover_specs(
    data_root: Path,
    stitching_root: Path,
    dataset_filter: list[str] | None = None,
) -> list[tuple[str, Path, Path]]:
    data_root = Path(data_root)
    stitching_root = Path(stitching_root)

    label_files = sorted(stitching_root.glob("dataset=*/data.csv"))
    if not label_files:
        return []

    datasets: dict[str, Path] = {}
    for p in label_files:
        ds = p.parent.name.removeprefix("dataset=")
        if dataset_filter and ds not in dataset_filter:
            continue
        if p.stat().st_size > 0:
            datasets[ds] = p

    factory_roots: list[Path] = [
        data_root / "data" / "factory" / "release=2026-06-17.0",
        data_root / "factory" / "release=2026-06-17.0",
        data_root / "data" / "factory" / "release=2026-01-21.0",
        data_root / "factory" / "release=2026-01-21.0",
    ]
    output_roots: list[Path] = [data_root / "data" / "output", data_root / "output"]

    specs: list[tuple[str, Path, Path]] = []
    for ds, lpath in sorted(datasets.items()):
        groups_path: Path | None = None
        candidates: list[Path] = []
        for out_root in output_roots:
            candidates.append(out_root / f"{ds}_groups.json")
            candidates.append(out_root / f"{ds}_bridge_groups.json")
        for fr in factory_roots:
            candidates.append(fr / f"dataset={ds}" / "groups.json")
        for cand in candidates:
            if cand.exists():
                groups_path = cand
                break
        if groups_path is None:
            groups_path = Path(f"__missing_groups_{_sanitize_ds(ds)}.json")
        specs.append((ds, groups_path, lpath))
    return specs


def _load_groups_by_dataset(
    specs: list[tuple[str, Path, Path]],
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for ds, gpath, _ in specs:
        gp = Path(gpath)
        if not gp.exists():
            continue
        try:
            out[ds] = load_sidecar_groups(gp)
        except Exception:
            out[ds] = []
    return out


def _build_combined_table(
    specs: list[tuple[str, Path, Path]],
    include_split: bool = True,
    include_rejected: bool = True,
    prefer_candidate_graph: bool = True,
    filter_rule5: bool = True,
    include_empty: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, list[dict]]]:
    frames: list[pd.DataFrame] = []
    per_ds_stats: list[dict[str, Any]] = []
    groups_by_ds: dict[str, list[dict]] = {}

    for dataset_id, groups_path, labels_path in specs:
        gp = Path(groups_path)
        if not gp.exists():
            per_ds_stats.append(
                {
                    "dataset_id": dataset_id,
                    "groups_path": str(groups_path),
                    "exists": False,
                }
            )
            continue
        groups = load_sidecar_groups(gp)
        groups_by_ds[dataset_id] = groups
        human_df = load_stitching_labels(labels_path)
        candidates_path = discover_candidates_parquet(gp)
        candidates_df = (
            load_candidates_parquet(candidates_path) if candidates_path is not None else None
        )
        df = build_edge_table(
            groups,
            human_df,
            dataset_id,
            include_split=include_split,
            include_rejected=include_rejected,
            prefer_candidate_graph=prefer_candidate_graph,
            filter_rule5=filter_rule5,
            include_empty=include_empty,
            candidates_df=candidates_df,
        )
        bs = df.attrs.get("build_stats", {}) if hasattr(df, "attrs") else {}
        per_ds_stats.append(
            {
                "dataset_id": dataset_id,
                "groups_path": str(groups_path),
                "candidates_path": str(candidates_path) if candidates_path else "",
                "exists": True,
                "n_sidecar_groups": len(groups),
                "n_labels": len(human_df),
                **{f"build_{k}": v for k, v in bs.items()},
                "rows": len(df),
            }
        )
        if len(df):
            frames.append(df)

    if not frames:
        return pd.DataFrame(), per_ds_stats, groups_by_ds
    combined = pd.concat(frames, ignore_index=True)
    return combined, per_ds_stats, groups_by_ds


def _build_soft_extra(
    groups_by_dataset: dict[str, list[dict]],
    batches_root: Path,
) -> pd.DataFrame | None:
    batches_root = Path(batches_root)
    if not batches_root.exists():
        return None
    vote_paths = default_votes_paths(batches_root)
    if not vote_paths:
        return None
    votes_df = load_votes([str(p) for p in vote_paths])
    if votes_df.empty:
        return None

    frames: list[pd.DataFrame] = []
    for dataset_id, groups in groups_by_dataset.items():
        soft = edge_soft_labels(groups, votes_df, dataset_id=dataset_id)
        if not soft.empty:
            frames.append(soft)
    if not frames:
        return None
    soft = pd.concat(frames, ignore_index=True)
    if "group_id" in soft.columns and "ref_id" in soft.columns:
        soft = (
            soft.groupby(["dataset_id", "group_id", "ref_id", "target_id"], as_index=False).agg(
                {
                    "soft_keep": "mean",
                    "n_providers": "max",
                    "unanimous": "min",
                }
            )
            if "n_providers" in soft.columns
            else soft.drop_duplicates(subset=["group_id", "ref_id", "target_id"])
        )
    return soft


def _build_edge_lookup_for_group(group: dict) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for e in group.get("edges", []):
        lookup[(str(e["ref_id"]), str(e["target_id"]))] = e
    for e in group.get("rejected_edges", []):
        lookup.setdefault((str(e["ref_id"]), str(e["target_id"])), e)
    for e in group.get("candidate_edges", []):
        key = (str(e["ref_id"]), str(e["target_id"]))
        if key in lookup:
            merged = {**lookup[key], **e}
            lookup[key] = merged
        else:
            lookup[key] = e
    return lookup


def _prepare_soft_for_train(
    soft_df: pd.DataFrame,
    groups_by_dataset: dict[str, list[dict]],
    existing_group_ids: set[tuple[str, str]],
    feature_cols: list[str],
    extended: bool,
    *,
    use_float_label: bool = False,
) -> pd.DataFrame | None:
    if soft_df.empty:
        return None

    gmap: dict[tuple[str, str], dict] = {}
    for dataset_id, gs in groups_by_dataset.items():
        for g in gs:
            gmap[(dataset_id, str(g["group_id"]))] = g

    edge_lookup_cache: dict[tuple[str, str], dict[tuple[str, str], dict]] = {}

    rows: list[dict] = []
    for _, r in soft_df.iterrows():
        dataset_id = str(r.get("dataset_id", ""))
        gid = str(r["group_id"])
        group_key = (dataset_id, gid)
        if group_key in existing_group_ids:
            continue
        g = gmap.get(group_key)
        if g is None:
            continue
        key = (str(r["ref_id"]), str(r["target_id"]))
        if group_key not in edge_lookup_cache:
            edge_lookup_cache[group_key] = _build_edge_lookup_for_group(g)
        edge = edge_lookup_cache[group_key].get(key)
        if edge is None:
            continue

        sk = float(r["soft_keep"])
        keep_v = sk if use_float_label else float(sk >= 0.5)

        rows.append(
            {
                "dataset_id": dataset_id,
                "group_id": gid,
                "human_group_id": gid,
                "labeler": "panel",
                "provenance": "soft_vote",
                "match_type": g.get("match_type", ""),
                "ref_id": key[0],
                "target_id": key[1],
                "keep": keep_v,
                "soft_keep": sk,
                "selected": bool(edge.get("selected", True)),
                "pruned": bool(edge.get("pruned", False)),
                "confidence": float(edge.get("confidence", float("nan"))),
                "degree_ref": int(edge.get("degree_ref", 0)),
                "degree_tgt": int(edge.get("degree_tgt", 0)),
                "is_bridge": bool(
                    edge.get("candidate_graph_bridge", edge.get("is_bridge", False))
                ),
                "is_sliver": bool(edge.get("is_sliver", False)),
                "biconnected_block": int(edge.get("biconnected_block", -1)),
                "corridor_ref": int(edge.get("corridor_ref", -1)),
                "corridor_tgt": int(edge.get("corridor_tgt", -1)),
                "gers_start_frac": float(edge.get("gers_start_frac", float("nan"))),
                "gers_end_frac": float(edge.get("gers_end_frac", float("nan"))),
                "local_start_frac": float(edge.get("local_start_frac", float("nan"))),
                "local_end_frac": float(edge.get("local_end_frac", float("nan"))),
                "n_edges": int(g.get("n_edges", 1)),
                "n_corridors": int(g.get("n_corridors", 1)),
                "n_assignment_components": int(g.get("n_assignment_components", 1)),
                "largest_biconnected_block": int(g.get("largest_biconnected_block", 1)),
                "oversized_group": bool(g.get("oversized_group", False)),
                "num_refs": len(g.get("ref_ids", [])),
                "num_targets": len(g.get("target_ids", [])),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    if extended:
        df = featurize_extended(df)
    else:
        df = featurize(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return None
    if df[feature_cols].isna().all().all():
        return None
    return df


def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    soft_extra: pd.DataFrame | None = None,
    seed: int = 0,
):
    from crosswalk.resolver.evaluate import _make_model

    hard_df = df.copy()
    if TRAIN_LABEL_COLUMN not in hard_df.columns:
        hard_df[TRAIN_LABEL_COLUMN] = hard_df["keep"]
    frames = [hard_df]
    if soft_extra is not None and len(soft_extra):
        soft_df = soft_extra.copy()
        if TRAIN_LABEL_COLUMN not in soft_df.columns:
            soft_df[TRAIN_LABEL_COLUMN] = soft_df["keep"]
        frames.append(soft_df)
    train_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else hard_df

    X = train_df[feature_cols].to_numpy(dtype=float)
    y = train_df[TRAIN_LABEL_COLUMN].to_numpy(dtype=float)
    is_float = bool(np.any((y != 0) & (y != 1)))
    y_bin = (y >= 0.5).astype(int) if is_float else y.astype(int)
    n_pos = int(y_bin.sum())
    n_neg = int((y_bin == 0).sum())
    if len(train_df) == 0 or n_pos == 0 or n_neg == 0:
        raise ValueError(f"Cannot train: rows={len(train_df)} pos={n_pos} neg={n_neg}")
    dtrain_label = y if is_float else y_bin
    if is_float:
        try:
            import xgboost as xgb  # type: ignore

            model = xgb.XGBRegressor(
                objective="reg:logistic",
                eval_metric="logloss",
                n_estimators=120,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                reg_lambda=1.5,
                n_jobs=1,
                random_state=seed,
            )
        except Exception:
            model = _make_model(n_pos, n_neg, seed=seed)
            dtrain_label = y_bin
    else:
        model = _make_model(n_pos, n_neg, seed=seed)
    model.fit(X, dtrain_label)
    return model


def predict_keep_probability(model, X: np.ndarray) -> np.ndarray:
    """Return keep probabilities for classifier or soft-label regressor artifacts."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.clip(np.asarray(model.predict(X), dtype=float), 0.0, 1.0)


def evaluate_all(
    df: pd.DataFrame,
    feature_cols: list[str],
    selector: str = "ef1",
    n_splits: int = 5,
    threshold: float = 0.5,
    soft_extra: pd.DataFrame | None = None,
    seed: int = 0,
    use_float_soft: bool = False,
    per_type_ef1: bool = False,
) -> dict[str, Any]:
    return run_cv2(
        df,
        feature_cols=feature_cols,
        selector=selector,
        threshold=threshold,
        n_splits=n_splits,
        soft_extra=soft_extra,
        seed=seed,
        use_float_soft=use_float_soft,
        per_type_ef1=per_type_ef1,
    )


def evaluate_repeated_grouped_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    selector: str = "ef1",
    n_splits: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    threshold: float = 0.5,
    per_type_ef1: bool = False,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 1729,
) -> dict[str, Any]:
    """Repeat grouped CV, ensemble its OOF probabilities, and quantify deltas.

    Each row is held out once per seed and group membership is never split. The
    paired bootstrap then resamples complete groups, which is the independent
    unit for resolver decisions.
    """
    from crosswalk.resolver.evaluate import _eval_from_predictions, paired_group_bootstrap

    if not seeds:
        raise ValueError("at least one repeated-CV seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("repeated-CV seeds must be unique")

    frame = df.reset_index(drop=True)
    runs: list[dict[str, Any]] = []
    probabilities: list[np.ndarray] = []
    for run_seed in seeds:
        result = run_cv2(
            frame,
            feature_cols=feature_cols,
            selector=selector,
            threshold=threshold,
            n_splits=n_splits,
            seed=run_seed,
            per_type_ef1=per_type_ef1,
            split_seed=run_seed,
        )
        probabilities.append(np.asarray(result["oof_proba"], dtype=float))
        model_result = result["model"]
        baseline_result = result["baseline_production"]
        runs.append(
            {
                "seed": run_seed,
                "f1": model_result.f1,
                "group_exact": model_result.group_exact_rate,
                "f1_delta": model_result.f1 - baseline_result.f1,
                "group_exact_delta": (
                    model_result.group_exact_rate - baseline_result.group_exact_rate
                ),
            }
        )

    mean_proba = np.mean(np.vstack(probabilities), axis=0)
    ensemble_pred = select_group_predictions(
        frame,
        mean_proba,
        selector=selector,
        threshold=threshold,
        per_type_ef1=per_type_ef1,
    )
    production_pred = frame["selected"].to_numpy()
    ensemble_result = _eval_from_predictions(
        f"xgb repeated-{len(seeds)} grouped CV ensemble",
        frame,
        ensemble_pred,
    )
    production_result = _eval_from_predictions(
        "baseline: optimizer+prune (selected)",
        frame,
        production_pred,
    )
    bootstrap = paired_group_bootstrap(
        frame,
        ensemble_pred,
        production_pred,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )

    def _spread(key: str) -> dict[str, float]:
        values = np.asarray([run[key] for run in runs], dtype=float)
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return {
        "runs": runs,
        "run_spread": {
            "f1": _spread("f1"),
            "group_exact": _spread("group_exact"),
            "f1_delta": _spread("f1_delta"),
            "group_exact_delta": _spread("group_exact_delta"),
        },
        "model": ensemble_result,
        "baseline_production": production_result,
        "paired_bootstrap": bootstrap,
        "oof_proba": mean_proba,
        "oof_prediction": ensemble_pred,
        "seeds": list(seeds),
    }


def evaluate_leave_one_dataset_out(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    selector: str = "ef1",
    threshold: float = 0.5,
    seed: int = 0,
    per_type_ef1: bool = False,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 2718,
) -> dict[str, Any]:
    """Evaluate transfer by holding out each dataset from model fitting.

    This is intentionally harsher than random grouped CV: it answers whether
    the learned policy transfers to a geography/mode it never saw, rather than
    merely generalizing to another group from Boston or Seattle.
    """
    from crosswalk.resolver.evaluate import _eval_from_predictions, paired_group_bootstrap

    frame = df.reset_index(drop=True)
    if "dataset_id" not in frame.columns:
        raise ValueError("leave-one-dataset-out evaluation requires dataset_id")
    datasets = sorted(frame["dataset_id"].astype(str).unique())
    if len(datasets) < 2:
        raise ValueError("leave-one-dataset-out evaluation requires at least two datasets")

    oof_proba = np.full(len(frame), np.nan, dtype=float)
    oof_pred = np.zeros(len(frame), dtype=int)
    per_dataset: list[dict[str, Any]] = []
    for dataset_id in datasets:
        test_mask = frame["dataset_id"].astype(str).to_numpy() == dataset_id
        train_frame = frame.loc[~test_mask].reset_index(drop=True)
        test_frame = frame.loc[test_mask].reset_index(drop=True)
        train_truth = train_frame["keep"].to_numpy(dtype=float)
        if not np.isin(train_truth, [0.0, 1.0]).all():
            raise ValueError("keep must remain binary evaluation truth")
        if len(np.unique(train_truth)) < 2:
            raise ValueError(f"training fold for {dataset_id!r} does not contain both classes")

        model = train_model(train_frame, feature_cols, seed=seed)
        probability = predict_keep_probability(
            model,
            test_frame[feature_cols].to_numpy(dtype=float),
        )
        prediction = select_group_predictions(
            test_frame,
            probability,
            selector=selector,
            threshold=threshold,
            per_type_ef1=per_type_ef1,
        )
        positions = np.flatnonzero(test_mask)
        oof_proba[positions] = probability
        oof_pred[positions] = prediction

        production = test_frame["selected"].to_numpy()
        per_dataset.append(
            {
                "dataset_id": dataset_id,
                "model": _eval_from_predictions("learned LODO", test_frame, prediction),
                "baseline_production": _eval_from_predictions(
                    "optimizer+prune",
                    test_frame,
                    production,
                ),
            }
        )

    if np.isnan(oof_proba).any():
        raise AssertionError("leave-one-dataset-out predictions did not cover every row")
    production_pred = frame["selected"].to_numpy()
    model_result = _eval_from_predictions("learned (leave-one-dataset-out)", frame, oof_pred)
    production_result = _eval_from_predictions(
        "baseline: optimizer+prune (selected)",
        frame,
        production_pred,
    )
    return {
        "model": model_result,
        "baseline_production": production_result,
        "per_dataset": per_dataset,
        "paired_bootstrap": paired_group_bootstrap(
            frame,
            oof_pred,
            production_pred,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            resample_columns=["dataset_id"],
        ),
        "oof_proba": oof_proba,
        "oof_prediction": oof_pred,
    }


def build_report_text(
    per_ds_stats: list[dict[str, Any]],
    df: pd.DataFrame,
    feature_cols: list[str],
    eval_result: dict[str, Any],
    soft_extra: pd.DataFrame | None,
    selector: str,
    extended: bool,
    data_root: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Learned Stitcher Round 3 — candidate-joined experimental ablation")
    lines.append("")
    lines.append(
        "> Experimental only — not wired into production. Produces `data/models/resolver_model.joblib`"
    )
    lines.append(
        "> and a prototype eval. Uses legacy sidecars from `data/factory/release=2026-06-17.0` when"
    )
    lines.append(
        "> `data/output/*.json` is absent, so under-selection is partially capped (64/group)."
    )
    lines.append(
        "> Decision: NO-GO for production; draft PR #411 invalidated the proposed heuristic defaults"
    )
    lines.append("> on a fixed label universe. Keep this track experimental and guard-isolated.")
    lines.append("")
    lines.append("## Inventory")
    lines.append("")
    lines.append(f"- data_root: `{data_root}`")
    lines.append(f"- feature_version: {FEATURE_VERSION}")
    feat_label = f"extended ({len(feature_cols)})" if extended else f"base ({len(feature_cols)})"
    lines.append(f"- feature set: {feat_label} cols: {', '.join(feature_cols[:8])}…")
    lines.append(f"- selector: {selector}")
    lines.append(f"- total edge rows (hard labels): {len(df)}")
    if len(df):
        lines.append(
            f"  - keep=1: {int(df['keep'].sum())} / keep=0: {int((df['keep'] == 0).sum())}"
        )
        lines.append(
            f"  - groups: {group_keys(df).nunique()} / datasets: {df['dataset_id'].nunique()}"
        )
        proven = df["provenance"].value_counts().to_dict() if "provenance" in df.columns else {}
        lines.append(f"  - provenance: {proven}")
    if soft_extra is not None:
        lines.append(f"- soft extra rows: {len(soft_extra)} (featurized groups not in hard set)")
    else:
        lines.append("- soft extra rows: 0 (no votes or all overlapping)")
    lines.append("")
    lines.append("### Per-dataset build stats")
    lines.append("")
    lines.append(
        "| dataset | sidecar groups | labels | rows | candidate_groups | legacy_groups | parquet rows | enriched | missing keys | outside total | outside clean | outside split | pos | neg | empty_rows | empty_legacy_skipped |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in per_ds_stats:
        if not s.get("exists", True):
            lines.append(
                f"| {s['dataset_id']} | MISSING | - | - | - | - | - | - | - | - | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {s['dataset_id']} | {s.get('n_sidecar_groups', 0)} | {s.get('n_labels', 0)} | {s.get('rows', 0)} "
            f"| {s.get('build_candidate_groups', 0)} | {s.get('build_legacy_groups', 0)} "
            f"| {s.get('build_candidate_parquet_rows', 0)} | {s.get('build_candidate_parquet_enriched', 0)} "
            f"| {s.get('build_candidate_parquet_missing_keys', 0)} "
            f"| {s.get('build_human_selected_outside_candidate_graph', 0)} "
            f"| {s.get('build_human_selected_outside_candidate_graph_clean', 0)} "
            f"| {s.get('build_human_selected_outside_candidate_graph_split', 0)} "
            f"| {s.get('build_positives', 0)} | {s.get('build_negatives', 0)} "
            f"| {s.get('build_empty_rows', 0)} | {s.get('build_empty_legacy_skipped', 0)} |"
        )
    lines.append("")
    total_cand = sum(s.get("build_candidate_groups", 0) for s in per_ds_stats if s.get("exists"))
    total_legacy = sum(s.get("build_legacy_groups", 0) for s in per_ds_stats if s.get("exists"))
    lines.append(f"- total candidate_groups={total_cand} legacy_groups={total_legacy}")
    lines.append("")

    if eval_result:
        lines.append("## Eval (grouped CV, out-of-fold)")
        lines.append("")
        for k in (
            "model",
            "baseline_production",
            "baseline_conf_oracle",
            "baseline_keepall",
            "baseline_conf",
        ):
            v = eval_result.get(k)
            if v is not None and hasattr(v, "row"):
                lines.append(f"- {k}: {v.row()}")
        lines.append("")
        if "oof_proba" in eval_result:
            import numpy as np

            lines.append(f"- oof_proba: mean={float(np.mean(eval_result['oof_proba'])):.3f}")
            lines.append("")

        if "model" in eval_result and "baseline_production" in eval_result:
            mr = eval_result["model"]
            bp = eval_result["baseline_production"]
            lines.append(
                f"**Headline vs production:** model F1={mr.f1:.3f} P={mr.precision:.3f} R={mr.recall:.3f} "
                f"grp_exact={mr.group_exact_rate:.3f} | baseline F1={bp.f1:.3f} "
                f"P={bp.precision:.3f} R={bp.recall:.3f} exact={bp.group_exact_rate:.3f}"
            )
            lines.append("")

    lines.append("## Limitations / next steps")
    lines.append("")
    lines.append(
        "- P1 `<ds>_candidates.parquet` is persisted and joined with 83 typed pair features + signed lateral offset + class/length."
    )
    lines.append(
        "  This prototype still trains on the 25 sidecar + 8 competition/coverage features; candidate-feature family selection is the next ablation step."
    )
    lines.append(
        "- Candidate parquet integrity is fail-closed (non-empty, non-null unique join keys); per-dataset missing-key counts are reported above."
    )
    n_parquet_datasets = sum(
        bool(s.get("build_candidate_parquet_rows", 0)) for s in per_ds_stats if s.get("exists")
    )
    lines.append(
        f"- Typed candidate parquet was locally available for {n_parquet_datasets} dataset(s) in this run; "
        "regenerate the remaining fresh sidecars before drawing feature-family conclusions."
    )
    lines.append(
        "- Panel votes are opt-in. Only provider-selected edges are usable today because batch artifacts do not prove which unselected candidates were shown; NONE votes cannot safely create edge negatives."
    )
    lines.append(
        "- Cross-mode testset (Bogotá bike + SG footpaths NONE) needs ≥20 empty labels held out; currently partial."
    )
    lines.append("- De-anchored slice `deanchored_v1` exists (51 groups) — sliced in next eval.")
    lines.append(
        "- If GO (beats tuned prune on clean + group exact), follow `research/learned_optimizer_design.md` I1"
    )
    lines.append(
        "  runtime behind `learned_resolver_overrides` + shadow `resolver_score` + S1 Spark export."
    )
    lines.append("")
    return "\n".join(lines)


def save_model(
    model,
    feature_cols: list[str],
    output_path: Path,
    training_stats: dict[str, Any],
    cv_summary: dict[str, Any] | None = None,
    selector: str = "ef1",
) -> None:
    import joblib

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": list(feature_cols),
        "feature_version": FEATURE_VERSION,
        "training_stats": training_stats,
        "cv_summary": cv_summary or {},
        "selector": selector,
    }
    joblib.dump(payload, str(output_path))
