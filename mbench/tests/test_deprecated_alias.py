"""The deprecated ``cbench`` console-script alias warns and forwards to ``mbench``.

The harness was renamed ``cbench`` -> ``mbench`` (2026-07-05). The old ``cbench``
entry point is kept as a thin shim (``mbench.cli.cbench_deprecated``) that emits a
deprecation warning to stderr and then forwards to the real Typer ``app``.
"""

from __future__ import annotations

import sys

import pytest

from mbench import cli


def test_cbench_alias_warns_and_forwards(capsys, monkeypatch):
    # `cbench --help` should warn on stderr, then forward to the real app,
    # which renders help to stdout and exits 0.
    monkeypatch.setattr(sys, "argv", ["cbench", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.cbench_deprecated()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    # Warned about the rename on stderr.
    assert "deprecated" in captured.err.lower()
    assert "mbench" in captured.err
    # Forwarded to the real command: the Typer app's own help (usage + its
    # command list) is rendered to stdout.
    assert "Usage" in captured.out
    assert "run-batch" in captured.out
