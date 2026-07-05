"""Tests for tool adapters."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cbench.adapters.base import EvalMode, ToolOutput
from cbench.adapters.matcher import DEFAULT_MATCHER_CMD, MatcherAdapter, _find_repo_root


class TestToolOutput:
    def test_valid_output(self):
        df = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
        out = ToolOutput(matches=df)
        assert len(out.matches) == 1

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"]})
        with pytest.raises(ValueError, match="missing required columns"):
            ToolOutput(matches=df)


class TestMatcherAdapter:
    def test_name_and_eval_mode(self):
        adapter = MatcherAdapter()
        assert adapter.name == "matcher"
        assert adapter.eval_mode == EvalMode.STITCH

    @patch("cbench.adapters.matcher.subprocess.run")
    def test_run_calls_matcher_cli(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        # Create fake output
        bridge_path = tmp_path / "bridge.parquet"
        pd.DataFrame(
            {
                "gers_id": ["r1"],
                "local_id": ["t1"],
                "confidence": [0.9],
                "match_type": ["1:1"],
            }
        ).to_parquet(bridge_path)

        adapter = MatcherAdapter()
        result = adapter.run(
            reference=tmp_path / "ref.parquet",
            target=tmp_path / "tgt.parquet",
            output_dir=tmp_path,
        )

        assert result == bridge_path.resolve()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        # Default invocation is `uv run matcher stitch ...` so it works from any
        # CWD without matcher being on PATH.
        assert cmd[: len(DEFAULT_MATCHER_CMD.split())] == DEFAULT_MATCHER_CMD.split()
        assert "stitch" in cmd
        # Subprocess runs from the repo root so matcher's relative model path
        # (data/models/...) resolves regardless of the caller's CWD.
        assert mock_run.call_args.kwargs["cwd"] == _find_repo_root()

    @patch("cbench.adapters.matcher.subprocess.run")
    def test_run_honors_matcher_cmd_and_repo_root(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        bridge_path = tmp_path / "bridge.parquet"
        pd.DataFrame({"gers_id": ["r1"], "local_id": ["t1"]}).to_parquet(bridge_path)

        repo_root = tmp_path / "myrepo"
        repo_root.mkdir()

        adapter = MatcherAdapter()
        adapter.run(
            reference=tmp_path / "ref.parquet",
            target=tmp_path / "tgt.parquet",
            output_dir=tmp_path,
            matcher_cmd="matcher",
            repo_root=repo_root,
        )

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "matcher"
        assert cmd[1] == "stitch"
        assert mock_run.call_args.kwargs["cwd"] == repo_root.resolve()

    @patch("cbench.adapters.matcher.subprocess.run")
    def test_run_passes_absolute_paths(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        bridge_path = tmp_path / "bridge.parquet"
        pd.DataFrame({"gers_id": ["r1"], "local_id": ["t1"]}).to_parquet(bridge_path)

        adapter = MatcherAdapter()
        adapter.run(
            reference=Path("ref.parquet"),
            target=Path("tgt.parquet"),
            output_dir=tmp_path,
        )
        cmd = mock_run.call_args[0][0]
        # All path args passed to matcher must be absolute since cwd != caller cwd.
        for flag in ("-r", "-t", "-o"):
            val = cmd[cmd.index(flag) + 1]
            assert Path(val).is_absolute(), f"{flag} path not absolute: {val}"

    def test_find_repo_root_locates_matcher_package(self):
        root = _find_repo_root()
        assert (root / "src" / "matcher").is_dir()

    def test_parse_output(self, tmp_path):
        bridge_path = tmp_path / "bridge.parquet"
        pd.DataFrame(
            {
                "gers_id": ["r1", "r2"],
                "local_id": ["t1", "t2"],
                "confidence": [0.95, 0.80],
                "match_type": ["1:1", "1:N"],
            }
        ).to_parquet(bridge_path)

        adapter = MatcherAdapter()
        output = adapter.parse_output(bridge_path)

        assert len(output.matches) == 2
        assert list(output.matches.columns) == ["ref_id", "target_id", "confidence"]
        assert output.matches["ref_id"].iloc[0] == "r1"
        assert output.matches["target_id"].iloc[0] == "t1"
        assert output.metadata["total_rows"] == 2


try:
    import geopandas  # noqa: F401

    _has_geopandas = True
except ImportError:
    _has_geopandas = False


@pytest.mark.skipif(not _has_geopandas, reason="geopandas required for hootenanny adapter")
class TestHootAdapter:
    def test_name_and_eval_mode(self):
        from cbench.adapters.hootenanny import HootAdapter

        adapter = HootAdapter()
        assert adapter.name == "hootenanny"
        assert adapter.eval_mode == EvalMode.STITCH

    def test_run_conflate_image_builds_docker_run_cmd(self, tmp_path):
        """Image mode drives `docker run` against a prebuilt image with the
        correct binary path, platform, mount, and namespaced creator flags."""
        from cbench.adapters import hootenanny as hoot

        ref = tmp_path / "ref.osm"
        tgt = tmp_path / "tgt.osm"
        out = tmp_path / "out.osm"
        ref.write_text("<osm/>")
        tgt.write_text("<osm/>")

        captured = {}

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                captured["cwd"] = kwargs.get("cwd")
                self.stdout = iter(["STATUS running", ""])
                # hoot writes the output inside the mounted dir
                out.write_text("<osm/>")

            def wait(self):
                return 0

        with patch.object(hoot.subprocess, "Popen", FakePopen):
            hoot._run_conflate_image(ref, tgt, out, image="hootenanny/run:0.2.41-1")

        cmd = captured["cmd"]
        assert cmd[0] == "docker" and cmd[1] == "run"
        assert "--platform" in cmd and "linux/amd64" in cmd
        assert "hootenanny/run:0.2.41-1" in cmd
        assert "/usr/bin/hoot" in cmd
        assert "conflate" in cmd
        assert "match.creators=hoot::HighwayMatchCreator" in cmd
        assert "merger.creators=hoot::HighwaySnapMergerCreator" in cmd
        # output dir is mounted at /data
        assert any(str(tmp_path) + ":/data" == c for c in cmd)
        assert cmd[-3:] == ["/data/ref.osm", "/data/tgt.osm", "/data/out.osm"]

    def test_run_image_mode_skips_compose(self, tmp_path):
        """run(hoot_image=...) must not touch the compose lifecycle."""
        # Minimal parquet inputs the converter can read.
        import geopandas as gpd
        from shapely.geometry import LineString

        from cbench.adapters.hootenanny import HootAdapter

        ref_pq = tmp_path / "ref.parquet"
        tgt_pq = tmp_path / "tgt.parquet"
        for p in (ref_pq, tgt_pq):
            gpd.GeoDataFrame(
                {"id": ["a"], "class": ["residential"], "names": [None]},
                geometry=[LineString([(0, 0), (0.001, 0.001)])],
                crs="EPSG:4326",
            ).to_parquet(p)

        with (
            patch("cbench.adapters.hootenanny._run_conflate_image") as mock_img,
            patch("cbench.adapters.hootenanny.ensure_compose_running") as mock_compose,
        ):
            HootAdapter().run(
                ref_pq, tgt_pq, tmp_path / "out", hoot_image="hootenanny/run:0.2.41-1"
            )
        mock_img.assert_called_once()
        mock_compose.assert_not_called()


@pytest.mark.skipif(not _has_geopandas, reason="geopandas required for naive adapter")
class TestNaiveAdapter:
    def _toy_data(self):
        import geopandas as gpd
        from shapely.geometry import LineString

        # A target road along y=0 from x=0..100 (lon/lat degrees near equator).
        # Reference r_near is nearly coincident; r_far is ~50m north; r_perp is
        # perpendicular. Coordinates are small degree offsets; the adapter
        # reprojects to UTM so distances are metric.
        target = gpd.GeoDataFrame(
            {"id": ["t1"]},
            geometry=[LineString([(0.0, 0.0), (0.001, 0.0)])],
            crs="EPSG:4326",
        )
        reference = gpd.GeoDataFrame(
            {"id": ["r_near", "r_far", "r_perp"]},
            geometry=[
                LineString([(0.0, 0.00001), (0.001, 0.00001)]),  # ~1m north, parallel
                LineString([(0.0, 0.01), (0.001, 0.01)]),  # ~1km north
                LineString([(0.0005, -0.001), (0.0005, 0.001)]),  # perpendicular
            ],
            crs="EPSG:4326",
        )
        return reference, target

    def test_name_and_eval_mode(self):
        from cbench.adapters.naive import NaiveAdapter

        adapter = NaiveAdapter()
        assert adapter.name == "naive"
        assert adapter.eval_mode == EvalMode.STITCH

    def test_registered(self):
        from cbench.adapters import REGISTRY

        assert "naive" in REGISTRY

    def test_matches_near_parallel_rejects_far_and_perpendicular(self):
        from cbench.adapters.naive import compute_naive_matches

        reference, target = self._toy_data()
        matches = compute_naive_matches(reference, target, buffer_m=15.0)
        assert set(matches.columns) == {"ref_id", "target_id", "confidence"}
        # The near-parallel reference matches; the far one is outside the buffer
        # and the perpendicular one fails the angle gate.
        assert "r_near" in set(matches["ref_id"])
        assert "r_far" not in set(matches["ref_id"])
        assert "r_perp" not in set(matches["ref_id"])
        assert (matches["target_id"] == "t1").all()

    def test_greedy_reference_assigned_once(self):
        import geopandas as gpd
        from shapely.geometry import LineString

        from cbench.adapters.naive import compute_naive_matches

        # One reference overlapping two candidate targets: greedy assigns it to
        # exactly one (its single best), never both.
        reference = gpd.GeoDataFrame(
            {"id": ["r1"]},
            geometry=[LineString([(0.0, 0.0), (0.001, 0.0)])],
            crs="EPSG:4326",
        )
        target = gpd.GeoDataFrame(
            {"id": ["t1", "t2"]},
            geometry=[
                LineString([(0.0, 0.00001), (0.001, 0.00001)]),
                LineString([(0.0, -0.00001), (0.001, -0.00001)]),
            ],
            crs="EPSG:4326",
        )
        matches = compute_naive_matches(reference, target, buffer_m=15.0)
        assert (matches["ref_id"] == "r1").sum() == 1

    def test_coverage_asymmetry_short_ref_matches_long_target(self):
        """A short reference covering part of a long target must still match.

        Regression guard for the removed Hausdorff "shape sanity" gate. The
        shipped gate used shapely's *symmetric* hausdorff_distance, which
        rejected any pair whose target extended more than buffer_m beyond the
        clipped reference — the coverage-asymmetry trap. Here a 10 m reference
        sits alongside one end of a ~110 m target: the reference is fully
        covered (overlap_frac ~ 1.0) and correctly aligned, so it must match.
        """
        import geopandas as gpd
        from shapely.geometry import LineString

        from cbench.adapters.naive import compute_naive_matches

        # ~110 m target along y=0 (near equator: 0.001 deg lon ~ 111 m).
        target = gpd.GeoDataFrame(
            {"id": ["t_long"]},
            geometry=[LineString([(0.0, 0.0), (0.001, 0.0)])],
            crs="EPSG:4326",
        )
        # ~10 m reference, parallel and ~1 m north, covering only the first ~9%
        # of the target. Its far end (the rest of the target) lies well beyond
        # buffer_m — the symmetric-Hausdorff gate would have wrongly rejected it.
        reference = gpd.GeoDataFrame(
            {"id": ["r_short"]},
            geometry=[LineString([(0.0, 0.00001), (0.0001, 0.00001)])],
            crs="EPSG:4326",
        )
        matches = compute_naive_matches(reference, target, buffer_m=15.0)
        assert "r_short" in set(matches["ref_id"])
        assert (matches["target_id"] == "t_long").all()

    def test_run_and_parse_roundtrip(self, tmp_path):
        from cbench.adapters.naive import NaiveAdapter

        reference, target = self._toy_data()
        ref_path = tmp_path / "ref.parquet"
        tgt_path = tmp_path / "tgt.parquet"
        reference.to_parquet(ref_path)
        target.to_parquet(tgt_path)

        adapter = NaiveAdapter()
        out = adapter.run(ref_path, tgt_path, tmp_path / "out")
        assert out.exists()
        parsed = adapter.parse_output(out)
        assert list(parsed.matches.columns) == ["ref_id", "target_id", "confidence"]
        # ids must be string-valued (dtype may be object or pandas StringDtype
        # depending on the pandas/pyarrow version — assert the values, not dtype)
        assert isinstance(parsed.matches["ref_id"].iloc[0], str)
        assert parsed.metadata["distinct_targets"] >= 1
