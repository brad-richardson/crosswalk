#!/usr/bin/env python
"""Zero-churn ablation of compute_class_similarity constants.

class_similarity depends ONLY on the class/subclass strings — no geometry — so
each candidate constant change can be applied to the labeled feature matrix in
memory and evaluated with the real LOO-by-type CV, without a backfill, a
FEATURE_VERSION bump, a retrain, or anything shipped.

Reports, per variant: how many labeled pairs actually change, and the LOO
macro-F1 per type group against the same baseline run.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from crosswalk.eval_utils import run_loo_by_type_cv
from crosswalk.features import semantic
from crosswalk.labeling.label_store import LabelStore

CV_FOLDS, SEED = 5, 42
GROUPS = ["road_good", "road_poor", "sidewalk", "other"]


def raw_classes() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(glob.glob("data/raw/*_overture_segments_v1.0.parquet")):
        ds = f.split("/")[-1].replace("_overture_segments_v1.0.parquet", "")
        tp = f"data/raw/{ds}_v1.0.parquet"
        try:
            rs = set(pq.read_schema(f).names)
            ts = set(pq.read_schema(tp).names)
        except Exception:
            continue
        rc = [c for c in ("id", "class", "subclass") if c in rs]
        tc = [c for c in ("id", "class", "subclass") if c in ts]
        if "class" not in rc or "class" not in tc:
            continue
        try:
            r = pq.read_table(f, columns=rc).to_pandas().drop_duplicates("id").set_index("id")
            t = pq.read_table(tp, columns=tc).to_pandas().drop_duplicates("id").set_index("id")
        except Exception:
            continue
        out[ds] = (r, t)
    return out


def attach_classes(df: pd.DataFrame) -> pd.DataFrame:
    maps = raw_classes()
    df = df.copy()
    for col in ("ref_class", "ref_sub", "tgt_class", "tgt_sub"):
        df[col] = None
    for ds, (r, t) in maps.items():
        m = df["dataset"] == ds
        if not m.any():
            continue
        df.loc[m, "ref_class"] = df.loc[m, "gers_id"].map(r["class"])
        df.loc[m, "tgt_class"] = df.loc[m, "target_id"].map(t["class"])
        if "subclass" in r:
            df.loc[m, "ref_sub"] = df.loc[m, "gers_id"].map(r["subclass"])
        if "subclass" in t:
            df.loc[m, "tgt_sub"] = df.loc[m, "target_id"].map(t["subclass"])
    return df


def recompute(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            semantic.compute_class_similarity(a, b, sa, sb)
            for a, b, sa, sb in zip(df.ref_class, df.tgt_class, df.ref_sub, df.tgt_sub)
        ],
        index=df.index,
    )


VARIANTS: dict[str, dict] = {
    "baseline (as-shipped)": {},
    "A: veh<->bike 0.7->0.35": {"tier": {("vehicle", "bicycle"): 0.35}},
    "B: subclass-differs 0.85->0.35": {"subclass_differs": 0.35},
    "C: rank decay 0.2->0.1": {"rank_step": 0.1},
    "A+B": {"tier": {("vehicle", "bicycle"): 0.35}, "subclass_differs": 0.35},
    "A+B+C": {
        "tier": {("vehicle", "bicycle"): 0.35},
        "subclass_differs": 0.35,
        "rank_step": 0.1,
    },
}


def apply_variant(cfg: dict):
    """Monkeypatch semantic module constants; returns a restore callable."""
    orig_tier = dict(semantic.TIER_PENALTIES)
    orig_src = semantic.compute_class_similarity
    if "tier" in cfg:
        for (a, b), v in cfg["tier"].items():
            semantic.TIER_PENALTIES[(a, b)] = v
            semantic.TIER_PENALTIES[(b, a)] = v
    sd = cfg.get("subclass_differs")
    rs = cfg.get("rank_step")
    if sd is not None or rs is not None:

        def patched(ca, cb, sa=None, sb=None, _o=orig_src):
            v = _o(ca, cb, sa, sb)
            if v != v:  # NaN
                return v
            if sd is not None and v == 0.85:
                return sd
            if rs is not None and ca and cb:
                a, b = ca.lower().strip(), cb.lower().strip()
                ta, tb = semantic.get_traffic_tier(a), semantic.get_traffic_tier(b)
                if ta == tb and a != b and ta not in (None, "neutral"):
                    d = abs(
                        semantic.ROAD_CLASS_HIERARCHY.get(a, 6)
                        - semantic.ROAD_CLASS_HIERARCHY.get(b, 6)
                    )
                    return max(0.0, 1.0 - d * rs)
            return v

        semantic.compute_class_similarity = patched

    def restore():
        semantic.TIER_PENALTIES.clear()
        semantic.TIER_PENALTIES.update(orig_tier)
        semantic.compute_class_similarity = orig_src

    return restore


def main() -> None:
    base = LabelStore.load_all(Path("labels"))
    base = attach_classes(base)
    print(f"labeled pairs: {len(base):,}  (classes resolved: {base.ref_class.notna().sum():,})\n")

    shipped = recompute(base)
    stored = base["class_similarity"]
    drift = ((shipped.fillna(-9) - stored.fillna(-9)).abs() > 1e-9).sum()
    print(f"sanity: recomputed vs stored class_similarity differs on {drift} pairs\n")

    results = {}
    for name, cfg in VARIANTS.items():
        restore = apply_variant(cfg)
        try:
            vals = recompute(base)
            changed = int(((vals.fillna(-9) - shipped.fillna(-9)).abs() > 1e-9).sum())
            df = base.copy()
            df["class_similarity"] = vals
            res = run_loo_by_type_cv(labels=df, cv_folds=CV_FOLDS, seed=SEED)
            gm = res.group_metrics()
        finally:
            restore()
        row = {"changed_pairs": changed}
        for g in GROUPS:
            row[g] = round(gm[g]["f1_mean"], 4) if g in gm else None
        vals_ = [row[g] for g in GROUPS if row[g] is not None]
        row["mean"] = round(sum(vals_) / len(vals_), 4) if vals_ else None
        results[name] = row
        print(f"  {name:<34} changed={changed:<6} " + "  ".join(f"{g}={row[g]}" for g in GROUPS))

    out = pd.DataFrame(results).T
    print("\n=== LOO macro-F1 by type group ===")
    print(out.to_string())
    b = out.loc["baseline (as-shipped)"]
    print("\n=== delta vs baseline ===")
    for name in results:
        if name == "baseline (as-shipped)":
            continue
        d = {g: round(out.loc[name, g] - b[g], 4) for g in GROUPS if out.loc[name, g] is not None}
        print(f"  {name:<34} " + "  ".join(f"{k}={v:+.4f}" for k, v in d.items()))
    print("\nfloors: road_good 0.85  road_poor 0.87  sidewalk 0.82  other 0.88")


if __name__ == "__main__":
    main()
