"""Exact, low-overhead scalar replacements for hot numpy statistics calls.

``np.percentile`` costs ~50 µs per call in pure-Python machinery
(``_ureduce`` / ``_quantile_unchecked`` / ``_get_indexes`` / ``_lerp``).
The per-pair feature path calls it hundreds of thousands of times per
dataset (perpendicular-offset statistics for every candidate pair and
every sibling-search survivor), which made it the single largest line
item in the feature-computation profile.

``percentile_sorted`` reproduces numpy's default linear-interpolation
method **bitwise** for 1-D float64 input (verified by
``tests/unit/test_exact_stats.py`` against ``np.percentile`` on random
arrays): same virtual-index computation, same ``_lerp`` formulation
including the ``gamma >= 0.5`` branch numpy uses for numerical symmetry.
NaN inputs are the caller's responsibility (sort places NaNs last, so
callers with possible NaNs must fall back to ``np.percentile`` — see
``_offsets_stats`` in ``relational.py``).

Do NOT "simplify" the interpolation to ``a + g * (b - a)`` only — the
two-branch form is what numpy implements, and dropping it breaks bitwise
parity with previously computed features.
"""

import math

import numpy as np


def percentile_sorted(sorted_values: np.ndarray, q: float) -> float:
    """Exact ``np.percentile(values, q)`` (linear method) on pre-sorted input.

    Args:
        sorted_values: 1-D float64 array sorted ascending, length >= 1,
            without NaNs.
        q: Percentile in [0, 100].

    Returns:
        The interpolated percentile, bitwise-equal to
        ``float(np.percentile(values, q))``.
    """
    n = sorted_values.shape[0]
    if n == 1:
        return float(sorted_values[0])

    # numpy: virtual_indexes = (n - 1) * quantiles, quantiles = q / 100
    virtual_index = (q / 100.0) * (n - 1)
    if virtual_index >= n - 1:
        return float(sorted_values[n - 1])

    previous = math.floor(virtual_index)
    gamma = virtual_index - previous
    a = float(sorted_values[int(previous)])
    b = float(sorted_values[int(previous) + 1])
    diff = b - a
    # numpy's _lerp: uses b - diff*(1-t) when t >= 0.5 for symmetry
    if gamma >= 0.5:
        return b - diff * (1.0 - gamma)
    return a + diff * gamma
