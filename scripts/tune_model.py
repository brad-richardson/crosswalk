"""Hyperparameter tuning script using Optuna."""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from matcher.config import FEATURE_COLUMNS
from matcher.matching.ml import MLMatcher, create_segment_groups


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Optuna objective function."""

    # Define search space
    # Note: scale_pos_weight is computed dynamically in ml.py based on class balance
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "booster": "gbtree",
        "n_jobs": -1,
        "verbosity": 0,
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
        "early_stopping_rounds": 50,
    }

    # Cross-validation with segment grouping to prevent data leakage
    gkf = GroupKFold(n_splits=5)
    scores = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        score = f1_score(y_val, preds)
        scores.append(score)

    return np.mean(scores)


def run_tuning(
    labels_dir: str,
    output_path: str,
    n_trials: int = 100,
):
    """Run hyperparameter tuning."""
    from matcher.labeling.label_store import LabelStore

    logger.info("Loading data...")
    df = LabelStore.load_all(Path(labels_dir))

    # Filter to valid labels
    df = df[df["label"].isin(["match", "no_match"])].copy()
    logger.info(f"Loaded {len(df)} valid labels")

    # Extract features and labels
    matcher = MLMatcher()
    matcher.feature_names = FEATURE_COLUMNS
    X, y = matcher._extract_features_and_labels(df, binary=True)

    # Compute imputation medians from full dataset (acceptable for tuning)
    matcher.feature_medians = {}
    for i, feat_name in enumerate(matcher.feature_names):
        col_vals = X[:, i]
        median_val = np.nanmedian(col_vals)
        matcher.feature_medians[feat_name] = median_val if not np.isnan(median_val) else 0.0

    X = matcher._impute_missing(X)

    # Create segment groups for CV
    logger.info("Creating segment groups...")
    groups = create_segment_groups(df)
    logger.info(f"Created {len(np.unique(groups))} groups from {len(df)} samples")

    # Run optimization
    logger.info(f"Starting optimization with {n_trials} trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible search
    )

    study.optimize(
        lambda trial: objective(trial, X, y, groups),
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
        "n_samples": len(df),
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved best params to {output_path}")

    # Print params in format ready for ml.py
    logger.info("\nParams for ml.py default_params:")
    for key, value in trial.params.items():
        logger.info(f'            "{key}": {value},')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters")
    parser.add_argument("--labels", default="labels", help="Path to labels directory")
    parser.add_argument("--output", default="best_params.json", help="Output path for best params")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")

    args = parser.parse_args()

    run_tuning(args.labels, args.output, args.trials)
