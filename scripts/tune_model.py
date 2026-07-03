"""Hyperparameter tuning script using Optuna — leakage-free protocol.

LEAKAGE PREVENTION — READ BEFORE CHANGING
=========================================
This script holds out the EXACT same test set that ``MLMatcher.train()`` uses
BEFORE any tuning happens:

1. Labels are loaded and preprocessed identically to ``train()``:
   filter to {match, no_match}, then ``_validate_training_pairs()``
   (same ``max_hausdorff_m`` default). This matters — the split indices are
   computed on the post-validation DataFrame, so any preprocessing drift
   between this script and ``train()`` changes which rows land in the test set.
2. ``segment_aware_split(df, test_size=0.2, random_state=42)`` is applied and
   the test rows are DISCARDED. The Optuna study never sees them.
3. The inner GroupKFold cross-validation runs only over the training portion,
   with segment groups subset to the training rows (mirroring how ``train()``
   subsets ``groups`` for its in-training CV).

Consequently, hyperparameters produced by this script have NEVER seen the
seed-42 test set, and the ``test_accuracy`` / classification report that
``train()`` prints afterwards is an honest holdout estimate.

WARNING: This guarantee is anchored to ``train()``'s defaults
(``test_size=0.2``, ``random_state=42``). If you change the seed or test_size
in ``MLMatcher.train()``, the holdout set changes and params tuned by this
script are no longer guaranteed to be independent of it — re-run tuning.

The objective mirrors deployment as closely as possible: no early stopping
(``train()`` does not early-stop), per-fold ``scale_pos_weight`` computed from
the fold's training labels (``train()`` computes it from its training set),
and F1 scored with the project's ``METRIC_AVERAGE``.

Usage:
    uv run python scripts/tune_model.py --trials 100 --output best_params.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from matcher.config import FEATURE_COLUMNS, METRIC_AVERAGE
from matcher.matching.ml import MLMatcher, segment_aware_split


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Optuna objective: mean F1 over segment-grouped CV folds (train rows only)."""

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
        "random_state": 42,
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
    gkf = GroupKFold(n_splits=5)
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

    return float(np.mean(scores))


def run_tuning(
    labels_dir: str,
    output_path: str,
    n_trials: int = 100,
):
    """Run hyperparameter tuning on the training portion of the seed-42 split."""
    from matcher.labeling.label_store import LabelStore

    logger.info("Loading data...")
    df = LabelStore.load_all(Path(labels_dir))

    # --- Mirror MLMatcher.train() preprocessing exactly ---
    # Any divergence here changes the row set the split is computed on, which
    # would silently change which rows are held out (see module docstring).
    df = df[df["label"].isin(["match", "no_match"])].copy()
    logger.info(f"Loaded {len(df)} valid labels")

    matcher = MLMatcher()
    matcher.feature_names = FEATURE_COLUMNS.copy()
    df = matcher._validate_training_pairs(df, max_hausdorff_m=1000.0)

    # --- Hold out the seed-42 test set BEFORE tuning (leakage prevention) ---
    # Same call as train(): test rows are discarded, never seen by Optuna.
    train_idx, test_idx, groups = segment_aware_split(
        df, test_size=0.2, random_state=42, return_groups=True
    )
    df_train = df.iloc[train_idx]
    # Subset groups to training rows (train_idx is positional), same as train()
    groups_train = groups.iloc[train_idx].to_numpy()
    logger.info(
        f"Held out {len(test_idx)} test rows (seed=42, test_size=0.2) — "
        f"tuning on {len(df_train)} training rows only"
    )

    # Extract features and labels (NaN preserved, inf capped — same as train())
    X, y = matcher._extract_features_and_labels(df_train, binary=True)
    X = matcher._cap_infinities(X)

    n_groups = len(np.unique(groups_train))
    logger.info(f"Using {n_groups} segment groups from {len(df_train)} training samples")

    # Run optimization
    logger.info(f"Starting optimization with {n_trials} trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible search
    )

    study.optimize(
        lambda trial: objective(trial, X, y, groups_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Log results
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  F1 Score: {trial.value:.4f}")
    logger.info("  Params:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save results
    results = {
        "best_f1": trial.value,
        "best_params": trial.params,
        "n_trials": n_trials,
        "n_train_samples": len(df_train),
        "n_test_held_out": len(test_idx),
        "protocol": "tuned on seed-42/test_size-0.2 training portion only (test set never seen)",
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved best params to {output_path}")

    # Print params in format ready for ml.py
    logger.info("\nParams for ml.py DEFAULT_XGB_PARAMS:")
    for key, value in trial.params.items():
        logger.info(f'    "{key}": {value},')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters")
    parser.add_argument("--labels", default="labels", help="Path to labels directory")
    parser.add_argument("--output", default="best_params.json", help="Output path for best params")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")

    args = parser.parse_args()

    run_tuning(args.labels, args.output, args.trials)
