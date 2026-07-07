"""``crosswalk stitch`` must key the resolver-prune allowlist on DATASET IDENTITY.

Regression tests for #348: the prune allowlist used to be resolved from the
OUTPUT FILENAME (bridge stem minus ``_bridge``, with only exact ``before_`` /
``after_`` prefixes stripped), so a bridge written as e.g.
``after4_us_boston_streets_bridge.parquet`` silently skipped pruning and changed
match counts mid-measurement. The CLI now passes the dataset name it already
knows (the dataset argument / DatasetLoader key) as ``prune_dataset_key``; the
output path plays no part. Raw ``-r``/``-t`` path mode without a dataset name
has no dataset identity, so it passes None (prune off, logged).
"""

from __future__ import annotations

from typer.testing import CliRunner

from crosswalk.cli import app

runner = CliRunner()


class _FakeResult:
    n_matched = 0
    n_target = 0


def _capture_run_pipeline(monkeypatch):
    """Stub out the heavy pipeline; capture the kwargs the CLI passes."""
    import crosswalk.pipeline as pipeline_mod

    calls: list[dict] = []

    def _fake_run_pipeline(*args, **kwargs):
        calls.append(kwargs)
        return _FakeResult()

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_run_pipeline)
    return calls


class TestStitchPruneKeying:
    def test_nonstandard_output_filename_still_carries_dataset_identity(
        self, monkeypatch, tmp_path
    ):
        """An allowlisted dataset prunes regardless of the ``-o`` filename: the
        CLI keys the allowlist on the dataset argument, so a nonstandard bridge
        name (``after4_..._bridge.parquet``) changes nothing (#348)."""
        calls = _capture_run_pipeline(monkeypatch)
        out = tmp_path / "after4_us_boston_streets_bridge.parquet"

        result = runner.invoke(
            app,
            [
                "stitch",
                "us_boston_streets",
                "-r",
                str(tmp_path / "ref.parquet"),
                "-t",
                str(tmp_path / "tgt.parquet"),
                "-o",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0]["prune_dataset_key"] == "us_boston_streets"
        assert calls[0]["output_path"] == out

        # And that identity resolves the allowlist to its tuned floor — i.e. the
        # nonstandard filename still prunes for an allowlisted dataset.
        import crosswalk.matching.ml as ml_mod
        from crosswalk.config import settings
        from crosswalk.pipeline import runner as pipeline_runner

        class _FakeMatcher:
            def __init__(self, *a, **k):
                pass

            calibration_active = True

        monkeypatch.setattr(ml_mod, "MLMatcher", lambda *a, **k: _FakeMatcher())
        monkeypatch.setattr(settings, "enable_calibration", True)
        monkeypatch.setattr(settings, "resolver_prune_enabled", True)
        monkeypatch.setattr(settings, "resolver_prune_overrides", {"us_boston_streets": 0.96})
        assert pipeline_runner._effective_prune_threshold(calls[0]["prune_dataset_key"]) == 0.96

    def test_path_mode_without_dataset_name_passes_no_identity(self, monkeypatch, tmp_path):
        """Raw -r/-t mode with no dataset argument has NO dataset identity: the
        CLI must pass None (prune off) even when the reference filename or the
        output filename LOOKS like an allowlisted dataset — filenames never key
        the allowlist (#348)."""
        calls = _capture_run_pipeline(monkeypatch)
        # Both filenames deliberately mimic an allowlisted dataset.
        out = tmp_path / "us_boston_streets_bridge.parquet"

        result = runner.invoke(
            app,
            [
                "stitch",
                "-r",
                str(tmp_path / "us_boston_streets.parquet"),
                "-t",
                str(tmp_path / "tgt.parquet"),
                "-o",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0]["prune_dataset_key"] is None

    def test_dataset_argument_with_paths_keys_on_dataset(self, monkeypatch, tmp_path):
        """``crosswalk stitch <dataset> -r … -t …`` (explicit paths + explicit
        dataset name) keys the prune on the dataset argument."""
        calls = _capture_run_pipeline(monkeypatch)

        result = runner.invoke(
            app,
            [
                "stitch",
                "us_seattle_sidewalks",
                "-r",
                str(tmp_path / "ref.parquet"),
                "-t",
                str(tmp_path / "tgt.parquet"),
                "-o",
                str(tmp_path / "whatever.parquet"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0]["prune_dataset_key"] == "us_seattle_sidewalks"
