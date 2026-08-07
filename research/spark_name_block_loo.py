"""Before/after for the Spark feature widening, isolating feature vs param change.

Three configs, 5 seeds each, same LOO harness the research doc used:
  before   : 28 features + 28-tuned params   (main)
  features : 35 features + 28-tuned params   (feature change alone)
  after    : 35 features + 35-tuned params   (this PR)
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("sfe", "research/spark_feature_expansion.py")
sfe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sfe)

from crosswalk.config import SPARK_PORTABLE_FEATURES, SPARK_PORTABLE_XGB_PARAMS
from crosswalk.labeling.label_store import LabelStore

OLD_PARAMS = {
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
NEW = list(SPARK_PORTABLE_FEATURES)
NAME_BLOCK = {
    "name_jaro_winkler",
    "name_soundex",
    "name_metaphone",
    "has_name_ref",
    "has_name_target",
    "name_is_generic",
}
OLD = [f for f in NEW if f not in NAME_BLOCK]
assert len(OLD) == 28 and len(NEW) == 34, (len(OLD), len(NEW))

SEEDS = [42, 1, 2, 3, 4]
labels = LabelStore.load_all(Path("labels"))
configs = {
    "before_28f_28p": (OLD, OLD_PARAMS),
    "features_34f_28p": (NEW, OLD_PARAMS),
    "after_34f_34p": (NEW, dict(SPARK_PORTABLE_XGB_PARAMS)),
}
out = {}
for name, (feats, params) in configs.items():
    runs = [sfe.loo_f1(labels, feats, seed, xgb_params=params) for seed in SEEDS]
    means = [r["loo_f1_mean"] for r in runs]
    groups = {}
    for g in runs[0]["loo_f1_by_group"]:
        groups[g] = float(np.mean([r["loo_f1_by_group"][g] for r in runs]))
    out[name] = {
        "loo_mean": float(np.mean(means)),
        "loo_std": float(np.std(means)),
        "n_features": len(feats),
        "by_group": groups,
        "per_dataset": {r["dataset"]: r["f1"] for r in runs[0]["loo_rows"]},
    }
    print(
        f"{name:18s} n={len(feats)} loo={np.mean(means):.4f} +- {np.std(means):.4f} "
        f"groups={ {k: round(v, 4) for k, v in groups.items()} }",
        flush=True,
    )

Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
print("wrote", sys.argv[1])
