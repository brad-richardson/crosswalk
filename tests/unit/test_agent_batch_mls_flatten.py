"""Regression test for `crosswalk agent test-batch` MultiLineString handling.

Every other geometry-load site flattens MultiLineStrings to LineStrings once at
the load boundary via ``filter_to_linestrings`` (see #360, commit 7df47f3):
never drop, never flatten per-feature deep in consumers. The raw-parquet load
in ``generate_agent_test_batch`` (src/crosswalk/cli/agent.py) skipped that
boundary call, so a target row whose raw geometry is still a MultiLineString
(post-#360 these are matchable/labelable, no longer dropped at ingest) would
reach ``SampledCandidate.target_geometry`` unflattened.

This test builds a small on-disk dataset with a MultiLineString target row,
drives the actual `agent test-batch` CLI command against it, and asserts the
geometry that reaches the renderer boundary (``write_candidate_package``) has
already been flattened to a LineString.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from typer.testing import CliRunner

import crosswalk.agent_labeling.context_generator as context_generator_module
from crosswalk.cli import app

runner = CliRunner()


def test_agent_test_batch_flattens_multilinestring_target_geometry(tmp_path, monkeypatch):
    """A MultiLineString target row must be flattened before it reaches the renderer."""
    monkeypatch.chdir(tmp_path)

    # Reference (Overture) dataset: one clean LineString.
    reference_path = tmp_path / "overture_segments.parquet"
    gpd.GeoDataFrame(
        {
            "id": ["ref1"],
            "names": [None],
            "class": ["residential"],
            "geometry": [LineString([(-71.06, 42.36), (-71.05, 42.36)])],
        },
        crs="EPSG:4326",
    ).to_parquet(reference_path)

    # Target dataset: one row with a MultiLineString geometry. Post-#360 this
    # id is not dropped at ingest by the real pipeline, so it is a legitimate
    # candidate that can show up in human labels.
    mls = MultiLineString(
        [
            [(-71.06, 42.36), (-71.055, 42.36)],
            [(-71.055, 42.36), (-71.05, 42.36)],
        ]
    )
    data_raw = tmp_path / "data" / "raw"
    data_raw.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "id": ["target1"],
            "names": [None],
            "class": ["residential"],
            "geometry": [mls],
        },
        crs="EPSG:4326",
    ).to_parquet(data_raw / "test_dataset.parquet")

    # Human label referencing the multi-part target id.
    labels_dir = tmp_path / "labels" / "human" / "dataset=test_dataset"
    labels_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "gers_id": ["ref1"],
            "target_id": ["target1"],
            "label": ["match"],
            "labeler": ["test"],
        }
    ).to_csv(labels_dir / "data.csv", index=False)

    # Spy on write_candidate_package (the render-boundary entry point) to
    # capture the SampledCandidate actually built by the CLI command, without
    # needing to refactor the command's inline load logic.
    captured = {}
    original_write = context_generator_module.write_candidate_package

    def spy_write_candidate_package(*, output_dir, candidate, batch_id, fetch_satellite=True):
        captured["candidate"] = candidate
        return original_write(
            output_dir=output_dir,
            candidate=candidate,
            batch_id=batch_id,
            fetch_satellite=fetch_satellite,
        )

    monkeypatch.setattr(
        context_generator_module, "write_candidate_package", spy_write_candidate_package
    )

    result = runner.invoke(
        app,
        [
            "agent",
            "test-batch",
            "-n",
            "1",
            "-d",
            "test_dataset",
            "--labels",
            str(tmp_path / "labels" / "human"),
            "--reference",
            str(reference_path),
            "--output",
            str(tmp_path / "data" / "agents"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "candidate" in captured, "write_candidate_package was never called"

    candidate = captured["candidate"]
    assert candidate.target_geometry.geom_type == "LineString"
    assert not isinstance(candidate.target_geometry, MultiLineString)
