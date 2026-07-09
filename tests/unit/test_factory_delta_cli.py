"""CLI-level tests for ``crosswalk factory delta --format csv`` (#364).

Rich's ``Console.print`` re-flows text to console width even when stdout isn't a
TTY, which corrupts CSV rows containing long semicolon-joined GERS-id lists
(``from_gers``/``to_gers``): tokens get split across lines, so a naive
``csv.reader`` over the captured output produces phantom rows. The fix routes
``--format csv`` straight to stdout via plain text (``df.to_csv`` + a raw write),
never through Rich, and moves the human-readable summary line to stderr so
stdout-to-file (`> out.csv`) redirection captures nothing but clean CSV.
"""

from __future__ import annotations

import csv
import io

import pandas as pd
from typer.testing import CliRunner

from crosswalk.cli import app

runner = CliRunner()


def _write_bridge(path, rows: list[tuple[str, str]]) -> None:
    """rows: list of (local_id, gers_id)."""
    df = pd.DataFrame(rows, columns=["local_id", "gers_id"])
    df.to_parquet(path)


def _make_releases(tmp_path, n_changed: int = 60, n_gers_per_row: int = 8):
    """Build from/to bridge parquets under a factory root with many wide,
    multi-GERS ``changed`` rows — the shape that triggers Rich's console-width
    line wrapping on the semicolon-joined id lists."""
    root = tmp_path / "factory"
    from_dir = root / "release=r1" / "dataset=ds"
    to_dir = root / "release=r2" / "dataset=ds"
    from_dir.mkdir(parents=True)
    to_dir.mkdir(parents=True)

    from_rows: list[tuple[str, str]] = []
    to_rows: list[tuple[str, str]] = []
    for i in range(n_changed):
        lid = f"local_{i}"
        for j in range(n_gers_per_row):
            from_rows.append((lid, f"08f28{i:04d}{j}ffffff"))
            to_rows.append((lid, f"08f28{i:04d}{j}fffffe"))  # different -> changed

    _write_bridge(from_dir / "bridge.parquet", from_rows)
    _write_bridge(to_dir / "bridge.parquet", to_rows)
    return root


def test_delta_csv_to_stdout_is_clean_csv(tmp_path):
    """Wide multi-GERS rows must not be wrapped/mangled by Rich when the CSV
    goes to real stdout (no --output), and the summary banner must not leak
    into stdout alongside the CSV rows."""
    root = _make_releases(tmp_path)

    result = runner.invoke(
        app,
        [
            "factory",
            "delta",
            "ds",
            "--from",
            "r1",
            "--to",
            "r2",
            "--format",
            "csv",
            "--output-dir",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output

    stdout = result.stdout

    # No ANSI escape codes and no Rich box-drawing / markup artifacts.
    assert "\x1b[" not in stdout
    assert "[blue]" not in stdout and "[/blue]" not in stdout
    for box_char in ("│", "─", "╭", "╮", "╰", "╯"):
        assert box_char not in stdout

    # The human-readable summary belongs on stderr, not mixed into the CSV.
    assert "same=" not in stdout
    assert "same=" in result.stderr

    # stdout parses cleanly as CSV: header + exactly one row per changed local_id,
    # with a stable column count (no tokens split onto phantom rows).
    rows = list(csv.reader(io.StringIO(stdout)))
    assert rows[0] == ["local_id", "category", "from_gers", "to_gers"]
    data_rows = rows[1:]
    assert len(data_rows) == 60
    assert {len(r) for r in data_rows} == {4}
    assert {r[1] for r in data_rows} == {"changed"}

    # Every GERS id list round-trips as an unbroken semicolon-joined token —
    # this is what Rich's width-aware wrapping used to shred.
    for row in data_rows:
        from_gers, to_gers = row[2], row[3]
        assert len(from_gers.split(";")) == 8
        assert len(to_gers.split(";")) == 8
        assert "\n" not in from_gers
        assert "\n" not in to_gers


def test_delta_csv_to_stdout_matches_output_file_content(tmp_path):
    """The exact CSV bytes written via ``-o file.csv`` (never touched by Rich)
    must match what ends up on stdout when no ``-o`` is given."""
    root = _make_releases(tmp_path, n_changed=10, n_gers_per_row=3)

    file_result = runner.invoke(
        app,
        [
            "factory",
            "delta",
            "ds",
            "--from",
            "r1",
            "--to",
            "r2",
            "--format",
            "csv",
            "--output-dir",
            str(root),
            "--output",
            str(tmp_path / "delta.csv"),
        ],
    )
    assert file_result.exit_code == 0, file_result.output
    file_content = (tmp_path / "delta.csv").read_text()

    stdout_result = runner.invoke(
        app,
        [
            "factory",
            "delta",
            "ds",
            "--from",
            "r1",
            "--to",
            "r2",
            "--format",
            "csv",
            "--output-dir",
            str(root),
        ],
    )
    assert stdout_result.exit_code == 0, stdout_result.output

    assert stdout_result.stdout == file_content


def test_delta_md_format_still_uses_rich_console(tmp_path):
    """Non-CSV formats are out of scope for #364 and keep their current
    Rich-rendered behavior, including the leading summary line on stdout."""
    root = _make_releases(tmp_path, n_changed=2, n_gers_per_row=1)

    result = runner.invoke(
        app,
        [
            "factory",
            "delta",
            "ds",
            "--from",
            "r1",
            "--to",
            "r2",
            "--format",
            "md",
            "--output-dir",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "same=" in result.stdout
    assert "GERS churn delta" in result.stdout
