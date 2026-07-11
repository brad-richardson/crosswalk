"""Experimental resolver training: all datasets, extended feats + eF1 + soft votes.

Research-only — nothing in the production pipeline imports this module
(same guard as the rest of ``crosswalk.resolver``). Produces a joblib model
+ a ``research/learned_stitcher_round3.md`` report.

Data sources:
  - Curated labels ``labels/stitching/dataset=*/data.csv`` (178 pair + 48 set,
    52 empty ``[]`` reject-all on main as of 2026-07-10).
  - Sidecars ``data/output/*_groups.json`` if present else
    ``data/factory/release=…/dataset=*/groups.json`` (older release has no
    ``candidate_edges``, falls back to ``edges``+capped ``rejected_edges``).
  - Panel votes ``data/agents/stitching/batches/*/votes.csv`` for soft extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from crosswalk.config import FEATURE_VERSION
from crosswalk.resolver.extract import build_edge_table, load_sidecar_groups, load_stitching_labels
from crosswalk.resolver.features import featurize
from crosswalk.resolver.round2 import (
    featurize_extended,
    run_cv2,
)
from crosswalk.resolver.votes import default_votes_paths, edge_soft_labels, load_votes


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

    factory_roots = [
        data_root / "data" / "factory" / "release=2026-06-17.0",
        data_root / "factory" / "release=2026-06-17.0",
        Path("data/factory/release=2026-06-17.0"),
        data_root / "data" / "factory" / "release=2026-01-21.0",
        data_root / "factory" / "release=2026-01-21.0",
        Path("data/factory/release=2026-01-21.0"),
    ]
    output_roots = [data_root / "data" / "output", data_root / "output", Path("data/output")]

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
            groups_path = Path(f"__missing_groups_{ds}.json")
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
        df = build_edge_table(
            groups,
            human_df,
            dataset_id,
            include_split=include_split,
            include_rejected=include_rejected,
            prefer_candidate_graph=prefer_candidate_graph,
            filter_rule5=filter_rule5,
            include_empty=include_empty,
        )
        bs = df.attrs.get("build_stats", {}) if hasattr(df, "attrs") else {}
        per_ds_stats.append(
            {
                "dataset_id": dataset_id,
                "groups_path": str(groups_path),
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
        alt = Path("data/agents/stitching/batches")
        if alt.exists():
            batches_root = alt
        else:
            return None
    vote_paths = default_votes_paths(batches_root)
    if not vote_paths:
        return None
    votes_df = load_votes([str(p) for p in vote_paths])
    if votes_df.empty:
        return None

    all_groups: list[dict] = []
    for gs in groups_by_dataset.values():
        all_groups.extend(gs)

    soft = edge_soft_labels(all_groups, votes_df)
    if soft.empty:
        return None
    if "group_id" in soft.columns and "ref_id" in soft.columns:
        soft = (
            soft.groupby(["group_id", "ref_id", "target_id"], as_index=False).agg(
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
    existing_group_ids: set[str],
    feature_cols: list[str],
    extended: bool,
) -> pd.DataFrame | None:
    if soft_df.empty:
        return None

    gmap: dict[str, dict] = {}
    for gs in groups_by_dataset.values():
        for g in gs:
            gmap[g["group_id"]] = g

    edge_lookup_cache: dict[str, dict[tuple[str, str], dict]] = {}

    rows: list[dict] = []
    for _, r in soft_df.iterrows():
        gid = str(r["group_id"])
        if gid in existing_group_ids:
            continue
        g = gmap.get(gid)
        if g is None:
            continue
        key = (str(r["ref_id"]), str(r["target_id"]))
        if gid not in edge_lookup_cache:
            edge_lookup_cache[gid] = _build_edge_lookup_for_group(g)
        edge = edge_lookup_cache[gid].get(key)
        if edge is None:
            continue

        rows.append(
            {
                "dataset_id": "soft_vote",
                "group_id": gid,
                "human_group_id": gid,
                "labeler": "panel",
                "provenance": "soft_vote",
                "match_type": g.get("match_type", ""),
                "ref_id": key[0],
                "target_id": key[1],
                "keep": int(float(r["soft_keep"]) >= 0.5),
                "soft_keep": float(r["soft_keep"]),
                "selected": bool(edge.get("selected", True)),
                "pruned": bool(edge.get("pruned", False)),
                "confidence": float(edge.get("confidence", float("nan"))),
                "degree_ref": int(edge.get("degree_ref", 0)),
                "degree_tgt": int(edge.get("degree_tgt", 0)),
                "is_bridge": bool(edge.get("is_bridge", False)),
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

    frames = [df]
    if soft_extra is not None and len(soft_extra):
        frames.append(soft_extra)
    train_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else df

    X = train_df[feature_cols].to_numpy(dtype=float)
    y = train_df["keep"].to_numpy()
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if len(train_df) == 0 or n_pos == 0 or n_neg == 0:
        raise ValueError(f"Cannot train: rows={len(train_df)} pos={n_pos} neg={n_neg}")
    model = _make_model(n_pos, n_neg)
    model.set_params(random_state=seed)
    model.fit(X, y)
    return model


def evaluate_all(
    df: pd.DataFrame,
    feature_cols: list[str],
    selector: str = "ef1",
    n_splits: int = 5,
    threshold: float = 0.5,
    soft_extra: pd.DataFrame | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    return run_cv2(
        df,
        feature_cols=feature_cols,
        selector=selector,
        threshold=threshold,
        n_splits=n_splits,
        soft_extra=soft_extra,
    )


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
    lines.append("# Learned Stitcher Round 3 — experimental (all datasets, eF1+extended+soft)")
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
            f"  - groups: {df['group_id'].nunique()} / datasets: {df['dataset_id'].nunique()}"
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
        "| dataset | sidecar groups | labels | rows | candidate_groups | legacy_groups | pos | neg | empty_rows | empty_legacy_skipped |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in per_ds_stats:
        if not s.get("exists", True):
            lines.append(f"| {s['dataset_id']} | MISSING | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {s['dataset_id']} | {s.get('n_sidecar_groups', 0)} | {s.get('n_labels', 0)} | {s.get('rows', 0)} "
            f"| {s.get('build_candidate_groups', 0)} | {s.get('build_legacy_groups', 0)} "
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
        "- P1 parquet `<ds>_candidates.parquet` with 78 typed pair features + signed lateral offset + class/length"
    )
    lines.append(
        "  is NOT yet persisted — model uses only 26 sidecar + 8 competition/coverage features."
    )
    lines.append(
        "- Factory sidecars old → no `candidate_edges`, so under-selection positives under-counted (legacy path uses edges+rejected_edges capped 64)."
    )
    lines.append(
        "- Fresh `crosswalk stitch` with `stitch_persist_candidate_graph=True` needed for full-candidate training."
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
