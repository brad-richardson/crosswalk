"""Keep the timing-sensitive performance tests off the parallel runner.

Every test under ``tests/performance/`` asserts a wall-clock budget
(``µs/line``, ``s per 100K constructions``, ...). Those budgets only mean
anything when the process has the machine roughly to itself. Under the repo
default ``addopts = "-n auto --dist loadscope"`` they run concurrently with
workers doing XGBoost training, so they measure *contention* rather than the
code under test -- and they fail or pass depending on what happens to be
co-scheduled.

That was already true before ``--dist loadscope`` (these tests have a long
history of failing under ``-n auto`` and passing under ``-n 0``); grouping a
module onto one worker just made the co-scheduling consistent enough to fail CI
reliably instead of intermittently. A flaky timing assertion is worse than no
assertion, because it trains everyone to re-run until green.

So: skip under xdist, with a reason that says exactly how to run them. CI runs
them in a dedicated serial step (``ci.yml``), which takes ~10s for all 42, and
CLAUDE.md documents the local invocation.

**This guard is scoped by DIRECTORY, not by property.** Wall-clock assertions
living outside ``tests/performance/`` still run in the parallel suite and still
have the pathology described above. Known remaining, as of 2026-08-07:

* ``tests/unit/test_spatial_context.py`` -- five ``elapsed <`` budgets
* ``tests/unit/test_relational.py`` -- one ``elapsed < 8``
* ``tests/test_spark_feature_expansion.py`` -- a us/pair budget

They are left alone deliberately: each is a small guard inside a mostly-functional
module, and moving them here would drag unrelated tests out of the parallel run.
If any of them starts flaking, the fix is to move the assertion, not to widen this
filter -- a property-based marker (e.g. ``@pytest.mark.timing``) would be the
cleaner shape if this list grows.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent

_XDIST_SKIP = pytest.mark.skip(
    reason=(
        "timing-sensitive: measures contention, not code, when run under xdist. "
        "Run serially instead: `uv run pytest tests/performance -n 0`"
    )
)


def pytest_collection_modifyitems(config, items):
    """Skip tests in THIS package when running on an xdist worker.

    ``PYTEST_XDIST_WORKER`` is set by xdist on each worker process and is absent
    for ``-n 0``, so this is a no-op for the serial invocation that is supposed
    to run these.

    The path filter is load-bearing, not defensive. ``pytest_collection_modifyitems``
    is a session-level hook: even when it is defined in a subdirectory conftest,
    pytest calls it once with the ENTIRE session's item list, not just the items
    under this directory. Without the ``_HERE`` check the first draft of this file
    skipped all 4,131 tests in the repo -- and the run still exited 0, because a
    fully-skipped suite is a passing suite. That is the failure mode worth guarding
    hardest: green CI that asserts nothing.
    """
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        return
    for item in items:
        path = Path(str(item.path)).resolve()
        if path == _HERE or _HERE in path.parents:
            item.add_marker(_XDIST_SKIP)
