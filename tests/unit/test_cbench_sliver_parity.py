"""Parity guard: cbench's standalone sliver rule must match matcher's.

cbench is a standalone benchmarking package that cannot depend on matcher, so it
replicates the numeric junction-sliver classifier from
``matcher.config.is_sliver_edge`` in ``cbench.eval.sliver``. This test asserts
the two implementations agree across a representative grid of inputs (and share
the same thresholds), so they cannot drift silently.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pytest

# cbench lives in its own package tree with its own venv; add it to the path so
# the main test suite (which can import both) can compare the two classifiers.
_CBENCH_SRC = Path(__file__).resolve().parents[2] / "cbench" / "src"
if str(_CBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(_CBENCH_SRC))

from cbench.eval import sliver as cbench_sliver  # noqa: E402

from matcher import config as matcher_config  # noqa: E402


def test_thresholds_match():
    assert cbench_sliver.SLIVER_SPAN_THRESHOLD == matcher_config.SLIVER_SPAN_THRESHOLD
    assert cbench_sliver.SLIVER_ABS_OVERLAP_M == matcher_config.SLIVER_ABS_OVERLAP_M


# Representative inputs including the boundary values (0.10 span, 5 m overlap),
# missing (None) fracs/lengths, and NaN.
_FRACS = [None, 0.0, 0.02, 0.05, 0.099, 0.10, 0.11, 0.5, 1.0, float("nan")]
_LENS = [None, 0.0, 1.0, 4.9, 5.0, 50.0, 200.0, 2000.0, float("nan")]


@pytest.mark.parametrize("ref_span,tgt_span", list(product(_FRACS, _FRACS)))
def test_sliver_parity_over_grid(ref_span, tgt_span):
    for ref_len, tgt_len in product(_LENS, _LENS):
        expected = matcher_config.is_sliver_edge(ref_span, tgt_span, ref_len, tgt_len)
        actual = cbench_sliver.is_sliver_edge(ref_span, tgt_span, ref_len, tgt_len)
        assert actual == expected, (
            f"mismatch for span=({ref_span},{tgt_span}) len=({ref_len},{tgt_len}): "
            f"matcher={expected} cbench={actual}"
        )
