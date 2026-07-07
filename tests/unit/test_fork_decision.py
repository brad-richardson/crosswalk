"""Tests for the fork/COW shortcut safety decision in compute_features_parallel.

The fork shortcut shares worker_data copy-on-write instead of pickling it to
every worker (huge win for the CLI), but forking a multi-threaded process can
deadlock the children — the web UI computes features from a daemon thread with
uvicorn's event loop alive, so it must NOT fork. _should_use_fork() is a pure
helper precisely so this decision is testable without actually forking.
"""

import sys
import threading

from crosswalk.features import pipeline as pipeline_mod
from crosswalk.features.pipeline import _should_use_fork


class _FakeThread:
    pass


def _patch_single_threaded_linux(monkeypatch):
    """Simulate a single-threaded main-thread Linux process."""
    main = _FakeThread()
    monkeypatch.setattr(pipeline_mod.sys, "platform", "linux", raising=False)
    monkeypatch.setattr(pipeline_mod.threading, "current_thread", lambda: main)
    monkeypatch.setattr(pipeline_mod.threading, "main_thread", lambda: main)
    monkeypatch.setattr(pipeline_mod.threading, "active_count", lambda: 1)


def test_fork_allowed_on_single_threaded_linux_main_thread(monkeypatch):
    _patch_single_threaded_linux(monkeypatch)
    assert _should_use_fork() is True


def test_fork_refused_when_other_threads_alive(monkeypatch):
    """The web UI scenario: extra live threads -> must take the pickling path."""
    _patch_single_threaded_linux(monkeypatch)
    monkeypatch.setattr(pipeline_mod.threading, "active_count", lambda: 3)
    assert _should_use_fork() is False


def test_fork_refused_off_main_thread(monkeypatch):
    """Calls from a non-main thread (daemon worker) must not fork."""
    _patch_single_threaded_linux(monkeypatch)
    monkeypatch.setattr(pipeline_mod.threading, "current_thread", lambda: _FakeThread())
    assert _should_use_fork() is False


def test_fork_refused_on_non_linux(monkeypatch):
    _patch_single_threaded_linux(monkeypatch)
    monkeypatch.setattr(pipeline_mod.sys, "platform", "darwin", raising=False)
    assert _should_use_fork() is False


def test_fork_refused_from_real_worker_thread():
    """End-to-end sanity: a real spawned thread never qualifies (it is not the
    main thread, and active_count >= 2 while it runs)."""
    result: list[bool] = []
    t = threading.Thread(target=lambda: result.append(_should_use_fork()))
    t.start()
    t.join()
    assert result == [False]


def test_helper_reads_real_environment():
    """No-patch smoke test: the helper must agree with its own definition in
    the live test process (whatever that environment is)."""
    expected = (
        sys.platform.startswith("linux")
        and threading.current_thread() is threading.main_thread()
        and threading.active_count() == 1
    )
    assert _should_use_fork() is expected
