"""Grouped-CV eval harness: learned resolver vs optimizer baseline.

Honesty constraints (see the prototype writeup):

* Grouped CV (``GroupKFold`` on ``group_id``): edges from one group never span
  the train/test boundary. ~40-60 labeled groups is small, so numbers are
  reported with per-fold spread, not a single point estimate.
* The production baseline is the optimizer's own selection (``selected``, which
  is keep-all on the persisted edges). A tuned confidence threshold is reported
  as a second, non-learned baseline.
* Metrics: per-edge precision/recall/F1 (raw + sliver-filtered) computed from
  out-of-fold predictions, plus group-level exact-match rate. Sliced by
  dataset and by clean/split provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from matcher.resolver.features import FEATURE_COLUMNS, featurize


def _prf(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _group_exact_rate(group_ids: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> float:
    df = pd.DataFrame({"g": group_ids, "p": pred.astype(int), "t": truth.astype(int)})
    exact = 0
    n = 0
    for _, sub in df.groupby("g"):
        n += 1
        if (sub["p"].to_numpy() == sub["t"].to_numpy()).all():
            exact += 1
    return exact / n if n else 0.0


@dataclass
class EvalResult:
    label: str
    n_edges: int
    n_groups: int
    precision: float
    recall: float
    f1: float
    group_exact_rate: float
    precision_filtered: float
    recall_filtered: float
    f1_filtered: float
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {
            "model": self.label,
            "edges": self.n_edges,
            "groups": self.n_groups,
            "P": round(self.precision, 3),
            "R": round(self.recall, 3),
            "F1": round(self.f1, 3),
            "grp_exact": round(self.group_exact_rate, 3),
            "F1_sliverfilt": round(self.f1_filtered, 3),
        }


def _make_model(n_pos: int, n_neg: int):
    import xgboost as xgb

    spw = (n_neg / n_pos) if n_pos else 1.0
    return xgb.XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.5,
        scale_pos_weight=spw,
        eval_metric="logloss",
        n_jobs=4,
        random_state=0,
    )


def _eval_from_predictions(
    label: str,
    df: pd.DataFrame,
    pred: np.ndarray,
) -> EvalResult:
    truth = df["keep"].to_numpy()
    gids = df["group_id"].to_numpy()
    p, r, f1 = _prf(pred, truth)
    ge = _group_exact_rate(gids, pred, truth)
    # sliver-filtered: drop rows flagged is_sliver from both sides
    mask = ~df["is_sliver"].astype(bool).to_numpy()
    pf, rf, f1f = _prf(pred[mask], truth[mask])
    return EvalResult(
        label=label,
        n_edges=len(df),
        n_groups=df["group_id"].nunique(),
        precision=p,
        recall=r,
        f1=f1,
        group_exact_rate=ge,
        precision_filtered=pf,
        recall_filtered=rf,
        f1_filtered=f1f,
    )


def run_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    threshold: float = 0.5,
    soft_extra: pd.DataFrame | None = None,
    seed: int = 0,
) -> dict:
    """Grouped-CV of the learned resolver vs baselines on out-of-fold predictions.

    Args:
        df: featurized edge table (must contain FEATURE_COLUMNS + keep, group_id).
        n_splits: GroupKFold splits.
        threshold: decision threshold on P(keep).
        soft_extra: optional featurized panel-soft edge table for groups NOT in
            ``df``, appended to each fold's training set with ``keep`` derived by
            rounding ``soft_keep`` (>=0.5 -> keep). Never used for evaluation.
        seed: reserved.

    Returns:
        dict with model/baseline EvalResults over pooled OOF predictions and a
        per-fold F1 list for the model.
    """
    df = df.reset_index(drop=True)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["keep"].to_numpy()
    groups = df["group_id"].to_numpy()

    n_groups = df["group_id"].nunique()
    n_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    oof_pred = np.zeros(len(df), dtype=int)
    oof_conf_pred = np.zeros(len(df), dtype=int)
    fold_f1 = []

    extra_X = extra_y = None
    if soft_extra is not None and len(soft_extra):
        extra_X = soft_extra[FEATURE_COLUMNS].to_numpy(dtype=float)
        extra_y = (soft_extra["soft_keep"].to_numpy() >= 0.5).astype(int)

    for tr, te in gkf.split(X, y, groups):
        Xtr, ytr = X[tr], y[tr]
        if extra_X is not None:
            Xtr = np.vstack([Xtr, extra_X])
            ytr = np.concatenate([ytr, extra_y])
        n_pos = int(ytr.sum())
        n_neg = int((ytr == 0).sum())
        if n_pos == 0 or n_neg == 0:
            # single-class training fold: fall back to majority (keep-all is the
            # production default and the majority class here).
            oof_pred[te] = 1
        else:
            model = _make_model(n_pos, n_neg)
            model.fit(Xtr, ytr)
            proba = model.predict_proba(X[te])[:, 1]
            oof_pred[te] = (proba >= threshold).astype(int)

        # per-fold tuned confidence-threshold baseline (tuned on train only)
        best_t, best_f1 = 0.5, -1.0
        conf_tr = df.iloc[tr]["confidence"].to_numpy()
        for t in np.arange(0.3, 1.0, 0.01):
            _, _, f1t = _prf((conf_tr >= t).astype(int), y[tr])
            if f1t > best_f1:
                best_f1, best_t = f1t, t
        conf_te = df.iloc[te]["confidence"].to_numpy()
        oof_conf_pred[te] = (conf_te >= best_t).astype(int)

        _, _, f1_fold = _prf(oof_pred[te], y[te])
        fold_f1.append(round(f1_fold, 3))

    keep_all = np.ones(len(df), dtype=int)
    results = {
        "model": _eval_from_predictions("learned (xgb, grouped CV)", df, oof_pred),
        "baseline_keepall": _eval_from_predictions(
            "baseline: optimizer selected (keep-all)", df, keep_all
        ),
        "baseline_conf": _eval_from_predictions(
            "baseline: tuned conf threshold", df, oof_conf_pred
        ),
        "fold_f1": fold_f1,
    }
    return results


def feature_importances(df: pd.DataFrame) -> pd.Series:
    """Train one model on all rows and return gain-based feature importances."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["keep"].to_numpy()
    model = _make_model(int(y.sum()), int((y == 0).sum()))
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)


def slice_report(df: pd.DataFrame, **cv_kwargs) -> pd.DataFrame:
    """Run CV on the full set and on informative slices; return a tidy table."""
    df = featurize(df) if "conf_rel_max" not in df.columns else df
    rows = []

    def _add(name: str, sub: pd.DataFrame):
        # need enough groups for grouped CV and both classes present
        if sub["group_id"].nunique() < 5 or sub["keep"].nunique() < 2:
            return
        res = run_cv(sub, **cv_kwargs)
        for key in ("model", "baseline_keepall", "baseline_conf"):
            r = res[key].row()
            r["slice"] = name
            rows.append(r)

    _add("all", df)
    _add("clean", df[df["provenance"] == "clean"])
    for ds, sub in df.groupby("dataset_id"):
        _add(f"dataset={ds}", sub)
    if "labeler" in df.columns:
        for lb, sub in df.groupby("labeler"):
            _add(f"labeler={lb}", sub)
    return pd.DataFrame(rows)
