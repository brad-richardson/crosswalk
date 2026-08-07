"""LOO-evaluate hyperparameter candidates for the 34-feature Spark set.

The epsilon-compact rule selects on inner-CV F1 and a n_est*max_depth cost proxy.
Neither is LOO, and on the 100-trial run the selected point (353x7) scored WORSE
on LOO than the old 28-feature params applied to the same 34 features. This scores
candidates on the metric the model is actually gated on.

  uv run python cand_loo.py <trials_log> <out.json>
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("sfe", "research/spark_feature_expansion.py")
sfe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sfe)
from crosswalk.config import SPARK_PORTABLE_FEATURES
from crosswalk.labeling.label_store import LabelStore

OLD_28_PARAMS = {
    "n_estimators": 224,
    "learning_rate": 0.01275299313255589,
    "max_depth": 10,
    "min_child_weight": 2,
    "subsample": 0.8019037612739637,
    "colsample_bytree": 0.9661600548038851,
    "gamma": 0.6021730351738508,
    "reg_alpha": 1.5439549237262677,
    "reg_lambda": 2.1882487406505136,
    "max_bin": 343,
}
EPS = 0.003
SEEDS = [42, 1, 2]  # 3 seeds, not 5 -- why the numbers here differ ~0.0002 from the 5-seed table

txt = Path(sys.argv[1]).read_text(errors="replace")
trials = []
for m in re.finditer(r"Trial (\d+) finished with value: ([\d.]+) and parameters: (\{.*?\})\.", txt):
    i, val, p = int(m.group(1)), float(m.group(2)), eval(m.group(3))
    trials.append((i, val + 1e-5 * max(0, p["n_estimators"] - 100), p))
best = max(t[1] for t in trials)
elig = [t for t in trials if t[1] >= best - EPS]
elig.sort(key=lambda t: t[2]["n_estimators"] * t[2]["max_depth"])
print(f"{len(trials)} trials, best_raw={best:.4f}, {len(elig)} within eps={EPS}", flush=True)

# 4 cheapest eligible + the best-F1 trial, deduped
cands = {f"eps_compact_rank{n}": t for n, t in enumerate(elig[:4], 1)}
cands["best_f1"] = max(trials, key=lambda t: t[1])
cands = {k: v for k, v in cands.items()}

labels = LabelStore.load_all(Path("labels"))
feats = list(SPARK_PORTABLE_FEATURES)
results = {}


def score(name, params):
    runs = [sfe.loo_f1(labels, feats, s, xgb_params=params) for s in SEEDS]
    m = [r["loo_f1_mean"] for r in runs]
    cost = params["n_estimators"] * params["max_depth"]
    results[name] = {
        "loo_mean": float(np.mean(m)),
        "loo_std": float(np.std(m)),
        "cost": cost,
        "params": params,
    }
    print(
        f"  {name:22s} loo={np.mean(m):.4f} +- {np.std(m):.4f}  "
        f"n_est={params['n_estimators']:4d} depth={params['max_depth']:2d} cost={cost:5d}",
        flush=True,
    )


score("old_28f_params", OLD_28_PARAMS)
for name, (idx, _raw, p) in cands.items():
    score(f"{name}_t{idx}", p)

Path(sys.argv[2]).write_text(json.dumps(results, indent=2))
best_name = max(results, key=lambda k: results[k]["loo_mean"])
print(
    f"\nBEST BY LOO: {best_name} -> {results[best_name]['loo_mean']:.4f} "
    f"(cost {results[best_name]['cost']})"
)
print("wrote", sys.argv[2])
