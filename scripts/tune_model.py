"""Hyperparameter tuning script using Optuna — leakage-free protocol.

LEAKAGE PREVENTION — READ BEFORE CHANGING
=========================================
This script holds out the EXACT same test set that ``MLMatcher.train()`` uses
for the configured seed (42 by default) BEFORE any tuning happens:

1. Labels are loaded and preprocessed identically to ``train()``:
   filter to {match, no_match}, the same ``_check_feature_versions()`` gate
   (override with ``--allow-stale-features``), then
   ``_validate_training_pairs()`` (same ``max_hausdorff_m`` default). This
   matters — the split indices are
   computed on the post-validation DataFrame, so any preprocessing drift
   between this script and ``train()`` changes which rows land in the test set.
2. ``segment_aware_split(df, test_size=0.2, random_state=<seed>)`` is applied and
   the test rows are DISCARDED. The Optuna study never sees them.
3. The inner GroupKFold cross-validation runs only over the training portion,
   with segment groups subset to the training rows (mirroring how ``train()``
   subsets ``groups`` for its in-training CV).

Consequently, hyperparameters produced by this script have NEVER seen the
configured-seed test set, and the ``test_accuracy`` / classification report that
``train()`` prints afterwards is an honest holdout estimate.

WARNING: This guarantee is anchored to matching ``test_size`` and ``seed``
values between tuning and training (defaults: 0.2 and 42). If either changes,
the holdout set changes and previously tuned params are no longer guaranteed
to be independent of it — re-run tuning with the same values.

The objective mirrors deployment as closely as possible: no early stopping
(``train()`` does not early-stop), per-fold ``scale_pos_weight`` computed from
the fold's training labels (``train()`` computes it from its training set),
and F1 scored with the project's ``METRIC_AVERAGE``.

FEATURE SETS
============
``--feature-set full`` (default) tunes the full ``FEATURE_COLUMNS`` model
(``DEFAULT_XGB_PARAMS`` in ml.py).

``--feature-set spark`` tunes the Spark-portable model
(``SPARK_PORTABLE_XGB_PARAMS``): the feature matrix is restricted to
``SPARK_PORTABLE_FEATURES`` exactly the way ``crosswalk export-spark-model``
does it (``train(exclude_features=...)`` restricts ``feature_names`` BEFORE
``_validate_training_pairs()``, so the post-validation row set — and hence the
configured-seed split — matches the exported model's training run). The objective
additionally subtracts a size penalty of 0.00001 F1 per tree above 100
(``n_estimators``) so tuning favors compact models for Spark deployment.
The leakage-free protocol (split, discard, inner CV) is identical.

EPSILON-COMPACT SELECTION
=========================
Inference speed matters for Spark deployment, and the in-objective size
penalty alone is too weak to prevent expensive winners (it also ignores tree
depth). Each trial therefore records a traversal-cost proxy
``cost = n_estimators * max_depth``, and after the study finishes the FINAL
selected params are the minimum-cost trial among all trials whose raw
(unpenalized) CV F1 is within ``--epsilon`` of the best raw CV F1. Defaults:
0.003 for ``--feature-set spark``, 0.0 (selection off — picks the best raw-F1
trial) for ``full``. Both the best-F1 trial and the selected trial are logged
and persisted in the results JSON.

Usage:
    uv run python scripts/tune_model.py --trials 100 --output best_params.json
    uv run python scripts/tune_model.py --feature-set spark --trials 100 \
        --output spark_params.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

try:
    import optuna
    import xgboost as xgb
except ImportError as e:
    raise SystemExit(
        f"Missing optional dependency '{e.name}'. Tuning requires the ML extras: "
        "uv pip install -e '.[dev,ml]' (plus optuna: uv pip install optuna)"
    ) from e

from crosswalk.config import FEATURE_COLUMNS, METRIC_AVERAGE, SPARK_PORTABLE_FEATURES
from crosswalk.matching.ml import (
    MLMatcher,
    _canonicalize_training_frame,
    segment_aware_split,
)

# Size penalty for the spark feature set: 0.00001 F1 per tree above 100.
# Matches the penalty used for the original Spark-portable tuning so the
# objective still favors compact models for Spark deployment.
SPARK_SIZE_PENALTY_PER_TREE = 0.00001
SPARK_SIZE_PENALTY_FREE_TREES = 100


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    size_penalty_per_tree: float = 0.0,
    seed: int = 42,
) -> float:
    """Optuna objective: mean F1 over segment-grouped CV folds (train rows only).

    When ``size_penalty_per_tree`` > 0, subtracts
    ``size_penalty_per_tree * max(0, n_estimators - SPARK_SIZE_PENALTY_FREE_TREES)``
    from the mean F1 so the search favors compact models (used for the
    Spark-portable feature set). The unpenalized mean CV F1 is stored on the
    trial as the ``raw_cv_f1`` user attribute, and a traversal-cost proxy
    (``n_estimators * max_depth``) as the ``cost`` user attribute for
    epsilon-compact selection.
    """

    # Define search space
    # Note: scale_pos_weight is computed dynamically (per fold here, on the
    # full training set in ml.py) — it is not a tuned parameter.
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "booster": "gbtree",
        "n_jobs": -1,
        "verbosity": 0,
        "random_state": seed,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "max_bin": trial.suggest_int("max_bin", 128, 512),
    }

    # Cross-validation with segment grouping to prevent within-CV leakage.
    # No early stopping: train() fits the full n_estimators, so the objective
    # must evaluate the same configuration that deployment will use.
    gkf = GroupKFold(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Mirror train(): class imbalance handled via scale_pos_weight
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        fold_params = {
            **params,
            "scale_pos_weight": n_neg / n_pos if n_pos > 0 else 1.0,
        }

        model = xgb.XGBClassifier(**fold_params)
        model.fit(X_train, y_train, verbose=False)

        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average=METRIC_AVERAGE, zero_division=0)
        scores.append(score)

    mean_f1 = float(np.mean(scores))
    trial.set_user_attr("raw_cv_f1", mean_f1)
    # Traversal-cost proxy for epsilon-compact selection: prediction latency
    # scales with trees * depth (worst-case node visits per row).
    trial.set_user_attr("cost", params["n_estimators"] * params["max_depth"])
    if size_penalty_per_tree > 0:
        mean_f1 -= size_penalty_per_tree * max(
            0, params["n_estimators"] - SPARK_SIZE_PENALTY_FREE_TREES
        )
    return mean_f1


# Default epsilon for epsilon-compact selection with --feature-set spark:
# the cheapest (n_estimators * max_depth) trial within this raw CV F1 margin
# of the best raw CV F1 is selected. 0.0 disables selection (best-F1 trial).
SPARK_DEFAULT_EPSILON = 0.003


def _trial_summary(trial: optuna.trial.FrozenTrial) -> dict:
    """JSON-serializable summary of a trial for logging/persistence."""
    return {
        "number": trial.number,
        "raw_cv_f1": trial.user_attrs.get("raw_cv_f1"),
        "objective": trial.value,
        "cost": trial.user_attrs.get("cost"),
        "n_estimators": trial.params.get("n_estimators"),
        "max_depth": trial.params.get("max_depth"),
        "params": trial.params,
    }


def select_epsilon_compact_trial(
    study: optuna.Study, epsilon: float
) -> tuple[optuna.trial.FrozenTrial, optuna.trial.FrozenTrial]:
    """Pick the cheapest trial within ``epsilon`` raw CV F1 of the best.

    Returns ``(best_f1_trial, selected_trial)`` where ``best_f1_trial``
    maximizes the raw (unpenalized) CV F1 and ``selected_trial`` is the
    minimum-cost (``n_estimators * max_depth``) trial among all completed
    trials with ``raw_cv_f1 >= best_raw_cv_f1 - epsilon``. Ties on cost are
    broken by higher raw CV F1, then earlier trial number (deterministic).
    """
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and "raw_cv_f1" in t.user_attrs
    ]
    if not completed:
        raise ValueError("No completed trials with raw_cv_f1 — nothing to select from")

    best_f1_trial = max(completed, key=lambda t: (t.user_attrs["raw_cv_f1"], -t.number))
    best_raw_f1 = best_f1_trial.user_attrs["raw_cv_f1"]

    eligible = [t for t in completed if t.user_attrs["raw_cv_f1"] >= best_raw_f1 - epsilon]
    selected = min(
        eligible,
        key=lambda t: (t.user_attrs["cost"], -t.user_attrs["raw_cv_f1"], t.number),
    )
    return best_f1_trial, selected


def run_tuning(
    labels_dir: str,
    output_path: str,
    n_trials: int = 100,
    allow_stale_features: bool = False,
    feature_set: str = "full",
    epsilon: float | None = None,
    seed: int = 42,
):
    """Run hyperparameter tuning on the training portion of a fixed-seed split."""
    from crosswalk.labeling.label_store import LabelStore

    logger.info("Loading data...")
    # Tuning must use the complete declared input set.  A skipped/corrupt
    # partition would silently change the configured-seed split and invalidate trial
    # comparisons.
    df = LabelStore.load_all(Path(labels_dir), skip_errors=False)

    # --- Mirror MLMatcher.train() preprocessing exactly ---
    # Any divergence here changes the row set the split is computed on, which
    # would silently change which rows are held out (see module docstring).
    df = df[df["label"].isin(["match", "no_match"])].copy()
    logger.info(f"Loaded {len(df)} valid labels")

    matcher = MLMatcher()
    if feature_set == "spark":
        # Mirror `crosswalk export-spark-model`: train() restricts feature_names
        # (FEATURE_COLUMNS order, minus non-portable features) BEFORE
        # _validate_training_pairs(), so validation's all-NaN check — and thus
        # the configured-seed split's row set — matches the exported model's run.
        matcher.feature_names = [f for f in FEATURE_COLUMNS if f in SPARK_PORTABLE_FEATURES]
        size_penalty_per_tree = SPARK_SIZE_PENALTY_PER_TREE
        logger.info(
            f"Feature set 'spark': {len(matcher.feature_names)} Spark-portable features, "
            f"size penalty {size_penalty_per_tree} F1/tree above "
            f"{SPARK_SIZE_PENALTY_FREE_TREES} trees"
        )
    else:
        matcher.feature_names = FEATURE_COLUMNS.copy()
        size_penalty_per_tree = 0.0
    if epsilon is None:
        epsilon = SPARK_DEFAULT_EPSILON if feature_set == "spark" else 0.0
    logger.info(f"Epsilon-compact selection: epsilon={epsilon}")
    # Same feature_version gate as train(): tuning on stale/mixed features
    # would produce params train() then refuses to reproduce.
    matcher._check_feature_versions(df, allow_stale_features=allow_stale_features)
    df = matcher._validate_training_pairs(df, max_hausdorff_m=1000.0)
    df = _canonicalize_training_frame(df, source="Tuning")

    # --- Hold out the configured-seed test set BEFORE tuning (leakage prevention) ---
    # Same call as train(): test rows are discarded, never seen by Optuna.
    train_idx, test_idx, groups = segment_aware_split(
        df, test_size=0.2, random_state=seed, return_groups=True
    )
    df_train = df.iloc[train_idx]
    # Subset groups to training rows (train_idx is positional), same as train()
    groups_train = groups.iloc[train_idx].to_numpy()
    logger.info(
        f"Held out {len(test_idx)} test rows (seed={seed}, test_size=0.2) — "
        f"tuning on {len(df_train)} training rows only"
    )

    # Extract features and labels (NaN preserved, inf capped — same as train())
    X, y = matcher._extract_features_and_labels(df_train, binary=True)
    X = matcher._cap_infinities(X)

    n_groups = len(np.unique(groups_train))
    logger.info(f"Using {n_groups} segment groups from {len(df_train)} training samples")

    # Mirror train()'s GroupKFold guard: n_splits can't exceed the group count,
    # and tuning without at least 2 groups has no validation signal at all.
    if n_groups < 2:
        raise SystemExit(
            f"Only {n_groups} segment group(s) in the training set — cannot "
            "cross-validate trials. Label more (distinct) segments before tuning."
        )
    n_splits = min(5, n_groups)
    if n_splits < 5:
        logger.warning(f"Fewer than 5 segment groups; using GroupKFold(n_splits={n_splits})")

    # Run optimization
    logger.info(f"Starting optimization with {n_trials} trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X,
            y,
            groups_train,
            n_splits=n_splits,
            size_penalty_per_tree=size_penalty_per_tree,
            seed=seed,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Epsilon-compact selection: cheapest trial within epsilon raw CV F1 of
    # the best raw CV F1 (see module docstring).
    best_f1_trial, selected_trial = select_epsilon_compact_trial(study, epsilon)
    best_summary = _trial_summary(best_f1_trial)
    selected_summary = _trial_summary(selected_trial)

    logger.info(f"Epsilon-compact selection (epsilon={epsilon}):")
    logger.info(f"  {'trial':>10} | {'raw CV F1':>9} | {'n_est':>5} | {'depth':>5} | {'cost':>6}")
    for label, s in [("best-F1", best_summary), ("selected", selected_summary)]:
        logger.info(
            f"  {label:>10} | {s['raw_cv_f1']:>9.4f} | {s['n_estimators']:>5} | "
            f"{s['max_depth']:>5} | {s['cost']:>6}"
        )
    if selected_trial.number != best_f1_trial.number:
        logger.info(
            f"  Selected trial {selected_trial.number} trades "
            f"{best_summary['raw_cv_f1'] - selected_summary['raw_cv_f1']:.4f} raw CV F1 for a "
            f"{best_summary['cost'] / selected_summary['cost']:.2f}x cheaper model"
        )
    else:
        logger.info("  Best-F1 trial is already the cheapest within epsilon")

    # Save results. best_trial maximizes raw (unpenalized) CV F1;
    # selected_trial is the epsilon-compact pick actually recommended.
    # best_objective includes the size penalty (if any).
    results = {
        "best_objective": study.best_trial.value,
        "best_raw_cv_f1": best_summary["raw_cv_f1"],
        "best_params": selected_trial.params,
        "best_trial": best_summary,
        "selected_trial": selected_summary,
        "epsilon": epsilon,
        "n_trials": n_trials,
        "n_train_samples": len(df_train),
        "n_test_held_out": len(test_idx),
        "feature_set": feature_set,
        "n_features": len(matcher.feature_names),
        "size_penalty_per_tree": size_penalty_per_tree,
        "seed": seed,
        "protocol": (
            f"tuned on seed-{seed}/test_size-0.2 training portion only (test set never seen)"
        ),
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results (selected params) to {output_path}")

    # Print the SELECTED params in a format ready to paste into the params dict
    target = (
        "config.py SPARK_PORTABLE_XGB_PARAMS"
        if feature_set == "spark"
        else "ml.py DEFAULT_XGB_PARAMS"
    )
    logger.info(f"\nSelected params for {target}:")
    for key, value in selected_trial.params.items():
        logger.info(f'    "{key}": {value},')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters")
    parser.add_argument("--labels", default="labels", help="Path to labels directory")
    parser.add_argument("--output", default="best_params.json", help="Output path for best params")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shared seed for the holdout split, XGBoost folds, and Optuna sampler",
    )
    parser.add_argument(
        "--feature-set",
        choices=["full", "spark"],
        default="full",
        help="Feature view to tune: 'full' (all FEATURE_COLUMNS, default) or "
        "'spark' (SPARK_PORTABLE_FEATURES subset with tree-count size penalty)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Epsilon-compact selection margin: pick the cheapest "
        "(n_estimators * max_depth) trial within this raw CV F1 of the best. "
        f"Default: {SPARK_DEFAULT_EPSILON} for --feature-set spark, 0.0 (off) for full",
    )
    parser.add_argument(
        "--allow-stale-features",
        action="store_true",
        help="Tune despite stale/mixed feature_version labels (same escape hatch as train())",
    )

    args = parser.parse_args()

    run_tuning(
        args.labels,
        args.output,
        args.trials,
        args.allow_stale_features,
        args.feature_set,
        args.epsilon,
        args.seed,
    )
