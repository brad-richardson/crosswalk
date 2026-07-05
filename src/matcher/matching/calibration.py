"""Probability calibration for the pairwise match classifier.

XGBoost margin scores squashed through the logistic are not guaranteed to be
well-calibrated probabilities, especially under ``scale_pos_weight`` class
reweighting. Downstream code (ML scoring decisions, the 1:N optimizer group
gates, the labeling UI bands, and the bridge-output confidence filter) all
apply hand-set thresholds to ``MatchResult.confidence`` as if it were a true
probability. This module fits an isotonic-regression calibrator so those
thresholds operate on genuine ``P(match)`` values.

Evaluation integrity: the calibrator MUST be fit on out-of-fold (OOF)
predictions from the *training* portion only — never on the holdout test set.
``fit_isotonic_oof`` is called from ``MLMatcher.train`` with the OOF predictions
collected from the in-training GroupKFold CV (the same folds that produce
``cv_f1_mean``), so each calibration input comes from a model that never saw
that row, and the seed-42 holdout never participates.

Portability: isotonic regression is a monotone piecewise-linear function fully
described by its interpolation knots (``X_thresholds_`` / ``y_thresholds_``).
``IsotonicCalibrator.to_knots`` serialises those knots so the calibrator can be
re-applied outside Python (e.g. the Spark scoring job) via a plain
``interp(x, xs, ys)`` with endpoint clipping — exactly what ``apply_knots``
does here and what ``np.interp`` computes natively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.isotonic import IsotonicRegression

# Minimum OOF rows (with both classes present) required to fit a calibrator.
# Below this the isotonic fit overfits sampling noise; callers fall back to raw.
MIN_CALIBRATION_ROWS = 100


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Portable isotonic calibrator: monotone piecewise-linear interpolation.

    Stores the interpolation knots directly (rather than a fitted sklearn
    object) so the artifact is small, dependency-light to apply, and portable
    to non-Python runtimes. ``transform`` reproduces
    ``sklearn.isotonic.IsotonicRegression(out_of_bounds="clip").transform``
    bit-for-bit (verified in tests).
    """

    x_thresholds: np.ndarray
    y_thresholds: np.ndarray
    method: str = "isotonic"

    @classmethod
    def from_sklearn(cls, iso: IsotonicRegression) -> IsotonicCalibrator:
        """Build a portable calibrator from a fitted ``IsotonicRegression``."""
        return cls(
            x_thresholds=np.asarray(iso.X_thresholds_, dtype=np.float64),
            y_thresholds=np.asarray(iso.y_thresholds_, dtype=np.float64),
        )

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities."""
        return apply_knots(
            np.asarray(probs, dtype=np.float64), self.x_thresholds, self.y_thresholds
        )

    def to_knots(self) -> dict[str, object]:
        """Serialise to JSON-friendly knot arrays (for manifests / Spark)."""
        return {
            "method": self.method,
            "x_thresholds": [float(x) for x in self.x_thresholds],
            "y_thresholds": [float(y) for y in self.y_thresholds],
        }

    @classmethod
    def from_knots(cls, knots: dict[str, object]) -> IsotonicCalibrator:
        """Reconstruct from :meth:`to_knots` output."""
        return cls(
            x_thresholds=np.asarray(knots["x_thresholds"], dtype=np.float64),
            y_thresholds=np.asarray(knots["y_thresholds"], dtype=np.float64),
            method=str(knots.get("method", "isotonic")),
        )


def apply_knots(
    probs: np.ndarray, x_thresholds: np.ndarray, y_thresholds: np.ndarray
) -> np.ndarray:
    """Piecewise-linear interpolation with endpoint clipping.

    ``np.interp`` clamps to the first/last knot value outside the knot range,
    matching ``IsotonicRegression(out_of_bounds="clip")``.
    """
    return np.interp(probs, x_thresholds, y_thresholds)


def fit_isotonic_oof(oof_probs: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator | None:
    """Fit an isotonic calibrator on out-of-fold predictions.

    Args:
        oof_probs: Out-of-fold predicted P(match) for the training rows. Rows
            with NaN (no OOF prediction) are dropped.
        y_true: True binary labels aligned with ``oof_probs``.

    Returns:
        An :class:`IsotonicCalibrator`, or ``None`` if there is insufficient
        data (fewer than ``MIN_CALIBRATION_ROWS`` valid rows, or only one class
        present) — in which case callers should fall back to raw scores.
    """
    from sklearn.isotonic import IsotonicRegression

    oof_probs = np.asarray(oof_probs, dtype=np.float64)
    y_true = np.asarray(y_true)
    finite = np.isfinite(oof_probs)
    oof_probs, y_true = oof_probs[finite], y_true[finite]

    if len(oof_probs) < MIN_CALIBRATION_ROWS or len(np.unique(y_true)) < 2:
        return None

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_probs, y_true)
    return IsotonicCalibrator.from_sklearn(iso)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin Expected Calibration Error (lower is better)."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if len(probs) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    n = len(probs)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(probs[mask].mean() - y_true[mask].mean())
    return float(ece)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and labels."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if len(probs) == 0:
        return float("nan")
    return float(np.mean((probs - y_true) ** 2))
