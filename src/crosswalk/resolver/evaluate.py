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

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from crosswalk.resolver.features import FEATURE_COLUMNS, featurize, group_key_columns, group_keys


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


def _validated_binary_vector(
    values,
    *,
    name: str,
    expected_length: int | None = None,
) -> np.ndarray:
    """Return a 1-D integer binary vector without lossy coercion."""
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional binary vector")
    if expected_length is not None and len(raw) != expected_length:
        raise ValueError(f"{name} rows ({len(raw)}) != expected rows ({expected_length})")
    if pd.isna(raw).any() or not np.isin(raw, [0, 1]).all():
        raise ValueError(f"{name} must be binary with no null values")
    return raw.astype(int)


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


def _make_model(n_pos: int, n_neg: int, seed: int = 0):
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
        n_jobs=1,
        random_state=seed,
    )


def _eval_from_predictions(
    label: str,
    df: pd.DataFrame,
    pred: np.ndarray,
) -> EvalResult:
    truth = _validated_binary_vector(
        df["keep"].to_numpy(),
        name="keep (binary evaluation truth)",
        expected_length=len(df),
    )
    pred = _validated_binary_vector(
        pred,
        name="predictions",
        expected_length=len(df),
    )
    gids = group_keys(df).to_numpy()
    p, r, f1 = _prf(pred, truth)
    ge = _group_exact_rate(gids, pred, truth)
    # sliver-filtered: drop rows flagged is_sliver from both sides
    mask = ~df["is_sliver"].astype(bool).to_numpy()
    pf, rf, f1f = _prf(pred[mask], truth[mask])
    return EvalResult(
        label=label,
        n_edges=len(df),
        n_groups=int(group_keys(df).nunique()),
        precision=p,
        recall=r,
        f1=f1,
        group_exact_rate=ge,
        precision_filtered=pf,
        recall_filtered=rf,
        f1_filtered=f1f,
    )


def paired_group_bootstrap(
    df: pd.DataFrame,
    candidate_pred: np.ndarray,
    baseline_pred: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    resample_columns: list[str] | None = None,
) -> dict:
    """Estimate paired metric deltas by resampling dependent row clusters.

    Edge rows within a group are dependent, so an edge-level bootstrap would
    report intervals that are too narrow. Both strategies are evaluated on the
    same sampled units on every draw, preserving the paired comparison. The
    default unit is a dataset-scoped match group; LODO evaluation can instead
    pass ``["dataset_id"]`` to represent between-dataset transfer uncertainty.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    frame = df.reset_index(drop=True)
    truth = _validated_binary_vector(
        frame["keep"].to_numpy(),
        name="keep (binary evaluation truth)",
        expected_length=len(frame),
    )
    candidate = _validated_binary_vector(
        candidate_pred,
        name="candidate predictions",
        expected_length=len(frame),
    )
    baseline = _validated_binary_vector(
        baseline_pred,
        name="baseline predictions",
        expected_length=len(frame),
    )

    group_columns = group_key_columns(frame)
    missing_group_columns = [column for column in group_columns if column not in frame.columns]
    if missing_group_columns or frame[group_columns].isna().any().any():
        raise ValueError("bootstrap group keys must be present and non-null")
    if any(frame[column].astype("string").str.strip().eq("").any() for column in group_columns):
        raise ValueError("bootstrap group keys must be nonblank")
    grouped = list(frame.groupby(group_columns, sort=False).indices.values())
    if not grouped:
        raise ValueError("paired bootstrap needs at least one group")
    group_indices = [np.asarray(idx, dtype=int) for idx in grouped]
    n_groups = len(group_indices)

    unit_columns = list(resample_columns) if resample_columns is not None else group_columns
    missing_columns = [column for column in unit_columns if column not in frame.columns]
    if not unit_columns or missing_columns:
        raise ValueError(f"invalid bootstrap resample columns: {missing_columns or unit_columns}")
    if frame[unit_columns].isna().any().any():
        raise ValueError("bootstrap resample keys must be non-null")
    unit_index = pd.MultiIndex.from_frame(frame[unit_columns])
    unit_codes, unit_values = pd.factorize(unit_index, sort=False)
    n_units = len(unit_values)
    unit_indices = [np.flatnonzero(unit_codes == unit) for unit in range(n_units)]

    candidate_exact = np.asarray(
        [np.array_equal(candidate[idx], truth[idx]) for idx in group_indices], dtype=float
    )
    baseline_exact = np.asarray(
        [np.array_equal(baseline[idx], truth[idx]) for idx in group_indices], dtype=float
    )
    exact_delta_by_unit: list[list[float]] = [[] for _ in range(n_units)]
    for group_pos, idx in enumerate(group_indices):
        group_units = np.unique(unit_codes[idx])
        if len(group_units) != 1:
            raise ValueError("a match group crosses bootstrap resample units")
        exact_delta_by_unit[int(group_units[0])].append(
            float(candidate_exact[group_pos] - baseline_exact[group_pos])
        )

    rng = np.random.default_rng(seed)
    f1_delta = np.empty(n_resamples, dtype=float)
    exact_delta = np.empty(n_resamples, dtype=float)
    for draw in range(n_resamples):
        sampled_units = rng.integers(0, n_units, size=n_units)
        sampled_rows = np.concatenate([unit_indices[i] for i in sampled_units])
        candidate_f1 = _prf(candidate[sampled_rows], truth[sampled_rows])[2]
        baseline_f1 = _prf(baseline[sampled_rows], truth[sampled_rows])[2]
        f1_delta[draw] = candidate_f1 - baseline_f1
        sampled_exact = np.concatenate(
            [np.asarray(exact_delta_by_unit[i], dtype=float) for i in sampled_units]
        )
        exact_delta[draw] = float(sampled_exact.mean())

    observed_candidate = _eval_from_predictions("candidate", frame, candidate)
    observed_baseline = _eval_from_predictions("baseline", frame, baseline)
    alpha = (1.0 - confidence) / 2.0

    def _summary(values: np.ndarray, observed: float) -> dict[str, float]:
        return {
            "delta": float(observed),
            "ci_low": float(np.quantile(values, alpha)),
            "ci_high": float(np.quantile(values, 1.0 - alpha)),
            "bootstrap_support_candidate_better": float(
                np.mean(values > 0.0) + 0.5 * np.mean(values == 0.0)
            ),
        }

    return {
        "n_groups": n_groups,
        "n_resample_units": n_units,
        "resample_columns": unit_columns,
        "n_resamples": n_resamples,
        "confidence": confidence,
        "seed": seed,
        "f1": _summary(
            f1_delta,
            observed_candidate.f1 - observed_baseline.f1,
        ),
        "group_exact": _summary(
            exact_delta,
            observed_candidate.group_exact_rate - observed_baseline.group_exact_rate,
        ),
    }


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
    groups = group_keys(df).to_numpy()

    n_groups = int(group_keys(df).nunique())
    if n_groups < 2:
        raise ValueError(f"grouped CV needs >= 2 groups, got {n_groups}; nothing to hold out.")
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
            oof_pred[te] = 1
        else:
            model = _make_model(n_pos, n_neg, seed=seed)
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

    # Optimizer baseline = the sidecar's own per-edge `selected` flag (True for
    # ~all persisted edges, False for the handful of junction slivers), NOT a
    # blanket all-ones vector.
    keep_all = df["selected"].to_numpy()
    results = {
        "model": _eval_from_predictions("learned (xgb, grouped CV)", df, oof_pred),
        "baseline_keepall": _eval_from_predictions("baseline: optimizer selected", df, keep_all),
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


# Panel-labeler string -> rubric era. ``panel_unanimous_vN`` / ``panel_quorum_vN``
# roll up to ``vN`` (the rubric era that produced the vote); any other labeler
# (human names) rolls up to ``human``.
_PANEL_ERA_RE = re.compile(r"^panel_(?:unanimous|quorum)_(v\d+)$")


def _labeler_era(labeler) -> str:
    """Map a labeler string to its coarse rubric era for eval rollups.

    ``panel_unanimous_vN`` / ``panel_quorum_vN`` -> ``vN``; every other labeler
    (human names) -> ``human``. Lets eval judge whether an early era's share
    (e.g. v1's pre-access-channel votes) degrades quality, which the flat
    per-labeler slices cannot show.
    """
    m = _PANEL_ERA_RE.match(str(labeler))
    return m.group(1) if m else "human"


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
        # Per-era rollup: collapse panel_*_vN labelers to their rubric era so a
        # dominant early era (e.g. v1's pre-access-channel share) is judgeable.
        era = df["labeler"].map(_labeler_era)
        for e, sub in df.groupby(era):
            _add(f"era={e}", sub)
    # Anchored vs de-anchored: does hiding the optimizer's option menu (the
    # unbiased eval slice) change the picture vs the option-anchored majority?
    if "anchored" in df.columns:
        anchor_label = df["anchored"].map(lambda a: "anchored" if a else "deanchored")
        for name, sub in df.groupby(anchor_label):
            _add(name, sub)
    return pd.DataFrame(rows)
