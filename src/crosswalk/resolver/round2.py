"""Round-2 resolver experiments: richer group context + structured selection.

Round 1 (research/learned_group_resolver_prototype.md) was data-limited: the
sidecar persisted only optimizer-selected edges, so under-selection was
unlearnable and the keep-all baseline had recall 1.0 by construction. After
PR #282/#284 the sidecar carries every candidate (``rejected_edges``, incl.
``pruned`` records), so the table finally contains both error directions.

This module adds the two architecture axes round 1 never tested:

* **Extended group-context features** (:data:`EXTENDED_FEATURE_COLUMNS`):
  per-ref/per-target competition margins and coverage-complementarity — does
  this edge extend its segments' covered span, or duplicate coverage that
  higher-confidence edges already provide?
* **Structured per-group selection** (:func:`select_expected_f1`): instead of
  thresholding P(keep) per edge independently, choose each group's edge SET by
  maximizing expected F1 under the independence approximation
  ``E[F1](k) ~= 2 * sum_{i<=k} p_i / (k + sum_i p_i)`` over confidence-sorted
  prefixes (Lewis 1995 / GFM-style plug-in). This is a set-level objective, so
  it directly targets group exact-match, which per-edge thresholds ignore.

Experimental code: nothing in the production pipeline imports this module
(guarded by the resolver import-guard test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crosswalk.resolver.evaluate import _eval_from_predictions, _make_model, _prf
from crosswalk.resolver.features import FEATURE_COLUMNS, featurize

# Round-1 features + competition/coverage context (see featurize_extended).
EXTENDED_FEATURE_COLUMNS: list[str] = FEATURE_COLUMNS + [
    "conf_margin_ref",
    "conf_margin_tgt",
    "is_best_for_ref",
    "is_best_for_tgt",
    "n_higher_share_ref",
    "n_higher_share_tgt",
    "ref_span_overlap_higher",
    "tgt_span_overlap_higher",
]


def _span_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap length of [a0,a1] and [b0,b1] (fractions along a segment)."""
    lo, hi = max(min(a0, a1), min(b0, b1)), min(max(a0, a1), max(b0, b1))
    return max(0.0, hi - lo)


def featurize_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Round-1 features plus competition margins and coverage complementarity.

    All context aggregates are computed over the FULL candidate set of the
    group (selected + rejected), which is what makes them meaningful for the
    under-selection direction.
    """
    out = featurize(df)

    # Competition margins: confidence minus the best OTHER edge sharing the
    # same ref (resp. target) in the group. Positive => this edge is the
    # strongest claim on that segment.
    for side, col in (("ref_id", "ref"), ("target_id", "tgt")):
        key = ["group_id", side]
        gmax = out.groupby(key)["confidence"].transform("max")
        # second-best: max of others = max overall unless this row is the max,
        # then it's the second max.
        rank_desc = out.groupby(key)["confidence"].rank(method="first", ascending=False)
        second = (
            out.assign(_r=rank_desc)
            .groupby(key)["confidence"]
            .transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else np.nan)
        )
        best_other = np.where(out["confidence"] >= gmax, second, gmax)
        margin = out["confidence"] - pd.Series(best_other, index=out.index)
        out[f"conf_margin_{col}"] = margin.fillna(1.0)  # sole claimant => max margin
        out[f"is_best_for_{col}"] = (out["confidence"] >= gmax).astype(int)
        out[f"n_higher_share_{col}"] = rank_desc - 1

    # Coverage complementarity: fraction of this edge's span (on ref via
    # gers fracs, on target via local fracs) that is already covered by
    # HIGHER-confidence edges sharing the same segment. High overlap =>
    # redundant claim; low overlap => this edge extends coverage.
    for side, fr0, fr1, name in (
        ("ref_id", "gers_start_frac", "gers_end_frac", "ref_span_overlap_higher"),
        ("target_id", "local_start_frac", "local_end_frac", "tgt_span_overlap_higher"),
    ):
        vals = np.zeros(len(out))
        for (_, _), sub in out.groupby(["group_id", side]):
            if len(sub) == 1:
                continue
            rows = sub.sort_values("confidence", ascending=False)
            seen: list[tuple[float, float]] = []
            for idx, r in rows.iterrows():
                a0, a1 = r[fr0], r[fr1]
                span = abs(a1 - a0)
                if span <= 0 or np.isnan(span):
                    seen.append((a0, a1))
                    continue
                cov = 0.0
                for b0, b1 in seen:
                    if not (np.isnan(b0) or np.isnan(b1)):
                        cov += _span_overlap(a0, a1, b0, b1)
                vals[out.index.get_loc(idx)] = min(cov / span, 1.0)
                seen.append((a0, a1))
        out[name] = vals

    return out


PER_TYPE_EF1_PENALTY: dict[str, float] = {
    "1:N": 0.08,
    "N:1": -0.02,
    "M:N": 0.04,
}

# Optional training-only target. ``keep`` remains the hard evaluation truth so
# smoothing cannot silently turn every non-zero negative into a positive metric.
TRAIN_LABEL_COLUMN = "_train_keep"


def select_expected_f1(
    probs: np.ndarray,
    *,
    empty_bonus: float = 0.0,
    per_type_penalty: float = 0.0,
) -> np.ndarray:
    """Choose the subset of one group's edges maximizing plug-in expected F1.

    Sort by probability descending; over prefixes of size k (including k=0)
    pick argmax of ``2 * sum_{i<=k} p_i / (k + sum p_i)``.  empty_bonus biases
    toward non-empty (negative) or empty (positive); per_type_penalty biases
    k=0 threshold per match type (1:N more conservative).
    """
    order = np.argsort(-probs)
    total = probs.sum()
    best_k, best_v = 0, float(np.prod(1.0 - probs)) + empty_bonus + per_type_penalty
    csum = 0.0
    for k, i in enumerate(order, start=1):
        csum += probs[i]
        v = 2.0 * csum / (k + total) if (k + total) > 0 else 0.0
        if v > best_v:
            best_v, best_k = v, k
    sel = np.zeros(len(probs), dtype=int)
    sel[order[:best_k]] = 1
    return sel


def select_expected_f1_per_type(
    df: pd.DataFrame,
    proba: np.ndarray,
    *,
    penalty_map: dict[str, float] | None = None,
) -> np.ndarray:
    pm = penalty_map or PER_TYPE_EF1_PENALTY
    pred = np.zeros(len(df), dtype=int)
    mt_col = df["match_type"].to_numpy() if "match_type" in df.columns else None
    for _, idx in df.groupby("group_id").indices.items():
        if mt_col is not None:
            mt = str(mt_col[idx[0]]) if len(idx) else ""
            pen = pm.get(mt, 0.0)
        else:
            pen = 0.0
        pred[idx] = select_expected_f1(proba[idx], per_type_penalty=pen)
    return pred


def run_cv2(
    df: pd.DataFrame,
    feature_cols: list[str],
    selector: str = "threshold",
    threshold: float = 0.5,
    n_splits: int = 5,
    soft_extra: pd.DataFrame | None = None,
    seed: int = 0,
    use_float_soft: bool = False,
    per_type_ef1: bool = False,
) -> dict:
    """Grouped-CV eval with pluggable feature set and per-group selector.

    selector: threshold|ef1|ef1_per_type.  If use_float_soft, soft_extra keeps
    are trained as float BCE instead of binarized.  Seed propagated to model.
    """
    from sklearn.model_selection import GroupKFold

    df = df.reset_index(drop=True)
    X = df[feature_cols].to_numpy(dtype=float)
    raw_truth = df["keep"].to_numpy(dtype=float)
    if not np.isin(raw_truth, [0.0, 1.0]).all():
        raise ValueError("keep must remain binary evaluation truth")
    y_truth = raw_truth.astype(int)
    y_train = (
        df[TRAIN_LABEL_COLUMN].to_numpy(dtype=float)
        if TRAIN_LABEL_COLUMN in df.columns
        else y_truth.astype(float)
    )
    groups = df["group_id"].to_numpy()
    n_groups = df["group_id"].nunique()
    if n_groups < 2:
        raise ValueError("grouped CV needs >= 2 groups")
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))

    oof_proba = np.zeros(len(df))
    extra_X = extra_y = None
    if soft_extra is not None and len(soft_extra):
        extra_X = soft_extra[feature_cols].to_numpy(dtype=float)
        if use_float_soft and "soft_keep" in soft_extra.columns:
            extra_y = soft_extra["soft_keep"].to_numpy(dtype=float)
        else:
            src = (
                soft_extra["soft_keep"].to_numpy()
                if "soft_keep" in soft_extra.columns
                else soft_extra["keep"].to_numpy()
            )
            extra_y = (src >= 0.5).astype(int)

    def _seeded_make_model(np_count: int, nn_count: int, s: int):
        import contextlib

        try:
            return _make_model(np_count, nn_count, seed=s)
        except TypeError:
            m = _make_model(np_count, nn_count)
            with contextlib.suppress(Exception):
                m.set_params(random_state=s)
            return m

    for tr, te in gkf.split(X, y_truth, groups):
        Xtr, ytr = X[tr], y_train[tr]
        if extra_X is not None:
            Xtr = np.vstack([Xtr, extra_X])
            ytr = np.concatenate([ytr, extra_y])
        is_float_fold = bool(np.any((ytr != 0) & (ytr != 1)))
        ytr_bin = (ytr >= 0.5).astype(int) if is_float_fold else ytr.astype(int)
        n_pos = int((ytr_bin == 1).sum()) if is_float_fold else int(np.nansum(ytr >= 0.5))
        n_neg = int((ytr_bin == 0).sum()) if is_float_fold else int(np.nansum(ytr < 0.5))
        if int(ytr_bin.sum()) == 0 or int((ytr_bin == 0).sum()) == 0:
            oof_proba[te] = float(int(ytr_bin.sum()) > 0)
            continue
        if is_float_fold:
            try:
                import xgboost as xgb  # type: ignore

                fold_model = xgb.XGBRegressor(
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
                fold_model.fit(Xtr, ytr.astype(float))
                raw = fold_model.predict(X[te])
                oof_proba[te] = np.clip(raw, 0.0, 1.0)
                continue
            except Exception:
                pass
        model = _seeded_make_model(n_pos if n_pos else 1, n_neg if n_neg else 1, seed)
        model.fit(Xtr, ytr_bin if is_float_fold else ytr)
        oof_proba[te] = model.predict_proba(X[te])[:, 1]

    if selector == "threshold":
        pred = (oof_proba >= threshold).astype(int)
    elif selector == "ef1":
        pred = np.zeros(len(df), dtype=int)
        if per_type_ef1:
            pred = select_expected_f1_per_type(df, oof_proba)
        else:
            for _, idx in df.groupby("group_id").indices.items():
                pred[idx] = select_expected_f1(oof_proba[idx])
    elif selector == "ef1_per_type":
        pred = select_expected_f1_per_type(df, oof_proba)
    else:
        raise ValueError(f"unknown selector {selector!r}")

    label = f"xgb[{'ext' if len(feature_cols) > len(FEATURE_COLUMNS) else 'r1'}feats]+{selector}"
    if soft_extra is not None:
        label += "+soft"
    res = {
        "model": _eval_from_predictions(label, df, pred),
        "baseline_production": _eval_from_predictions(
            "baseline: optimizer+prune (selected)", df, df["selected"].astype(int).to_numpy()
        ),
        "oof_proba": oof_proba,
    }
    # tuned-conf baseline over the FULL candidate set (round-1 comparator)
    best_t, best_f1 = 0.5, -1.0
    conf = df["confidence"].to_numpy()
    for t in np.arange(0.3, 1.0, 0.01):
        _, _, f1t = _prf((conf >= t).astype(int), y_truth)
        if f1t > best_f1:
            best_f1, best_t = f1t, t
    res["baseline_conf_oracle"] = _eval_from_predictions(
        f"baseline: conf>={best_t:.2f} (oracle-tuned)", df, (conf >= best_t).astype(int)
    )
    return res
