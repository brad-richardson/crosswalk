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

from matcher.resolver.evaluate import _eval_from_predictions, _make_model, _prf
from matcher.resolver.features import FEATURE_COLUMNS, featurize

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


def select_expected_f1(probs: np.ndarray) -> np.ndarray:
    """Choose the subset of one group's edges maximizing plug-in expected F1.

    Sort by probability descending; over prefixes of size k (including k=0)
    pick argmax of ``2 * sum_{i<=k} p_i / (k + sum p_i)``.
    """
    order = np.argsort(-probs)
    total = probs.sum()
    # k=0 (predict the empty set) scores F1=1 exactly when the true set is
    # empty: E[F1 | empty] = P(no edge is a keep) = prod(1 - p_i).
    best_k, best_v = 0, float(np.prod(1.0 - probs))
    csum = 0.0
    for k, i in enumerate(order, start=1):
        csum += probs[i]
        v = 2.0 * csum / (k + total) if (k + total) > 0 else 0.0
        if v > best_v:
            best_v, best_k = v, k
    sel = np.zeros(len(probs), dtype=int)
    sel[order[:best_k]] = 1
    return sel


def run_cv2(
    df: pd.DataFrame,
    feature_cols: list[str],
    selector: str = "threshold",
    threshold: float = 0.5,
    n_splits: int = 5,
    soft_extra: pd.DataFrame | None = None,
) -> dict:
    """Grouped-CV eval with pluggable feature set and per-group selector.

    selector: ``threshold`` (independent per-edge, round-1 style) or ``ef1``
    (structured per-group expected-F1 subset selection).
    """
    from sklearn.model_selection import GroupKFold

    df = df.reset_index(drop=True)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["keep"].to_numpy()
    groups = df["group_id"].to_numpy()
    n_groups = df["group_id"].nunique()
    if n_groups < 2:
        raise ValueError("grouped CV needs >= 2 groups")
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))

    oof_proba = np.zeros(len(df))
    extra_X = extra_y = None
    if soft_extra is not None and len(soft_extra):
        extra_X = soft_extra[feature_cols].to_numpy(dtype=float)
        extra_y = (soft_extra["soft_keep"].to_numpy() >= 0.5).astype(int)

    for tr, te in gkf.split(X, y, groups):
        Xtr, ytr = X[tr], y[tr]
        if extra_X is not None:
            Xtr = np.vstack([Xtr, extra_X])
            ytr = np.concatenate([ytr, extra_y])
        n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
        if n_pos == 0 or n_neg == 0:
            oof_proba[te] = float(n_pos > 0)
            continue
        model = _make_model(n_pos, n_neg)
        model.fit(Xtr, ytr)
        oof_proba[te] = model.predict_proba(X[te])[:, 1]

    if selector == "threshold":
        pred = (oof_proba >= threshold).astype(int)
    elif selector == "ef1":
        pred = np.zeros(len(df), dtype=int)
        for _, idx in df.groupby("group_id").indices.items():
            pred[idx] = select_expected_f1(oof_proba[idx])
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
        _, _, f1t = _prf((conf >= t).astype(int), y)
        if f1t > best_f1:
            best_f1, best_t = f1t, t
    res["baseline_conf_oracle"] = _eval_from_predictions(
        f"baseline: conf>={best_t:.2f} (oracle-tuned)", df, (conf >= best_t).astype(int)
    )
    return res
