"""Backfill must resolve pairs whose gers_id vanished from the current Overture release.

GERS reference ids churn across Overture releases. When a labeled pair's gers_id
no longer exists in the freshly loaded reference parquet, backfill must fall back
to the stored reference geometry in labels/data (mirroring the target-side
augmentation) instead of silently skipping the pair.

Regression test for the 56-pair "stale non-US features" debt (helsinki 47 /
jp_tokyo 4 / ch_geneva_ped 4 / tn_tunis 1).
"""

import csv

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from typer.testing import CliRunner

from crosswalk.cli import app
from crosswalk.labeling.data_store import DataStore

runner = CliRunner()

DATASET = "us_testville_streets"

# WGS84 coordinates near Boston; ~100m segments
REF_PRESENT_ID = "aaaaaaaa-1111-2222-3333-444444444444"
REF_CHURNED_ID = "bbbbbbbb-5555-6666-7777-888888888888"
TARGET_A = f"{DATASET}_1_882a30641b"
TARGET_B = f"{DATASET}_2_882a30641b"

LINE_A = LineString([(-71.0600, 42.3600), (-71.0588, 42.3600)])
LINE_A_OFF = LineString([(-71.0600, 42.36002), (-71.0588, 42.36002)])
LINE_B = LineString([(-71.0570, 42.3610), (-71.0558, 42.3610)])
LINE_B_OFF = LineString([(-71.0570, 42.36102), (-71.0558, 42.36102)])


def _write_fixture(tmp_path):
    data_dir = tmp_path / "raw"
    labels_dir = tmp_path / "labels"
    data_dir.mkdir()
    (labels_dir / f"human/dataset={DATASET}").mkdir(parents=True)

    # Reference parquet contains ONLY the still-present gers_id.
    ref = gpd.GeoDataFrame(
        {
            "id": [REF_PRESENT_ID],
            "names": [{"primary": "Main Street"}],
            "class": ["residential"],
        },
        geometry=[LINE_A],
        crs="EPSG:4326",
    )
    ref.to_parquet(data_dir / f"{DATASET}_overture_segments_v1.0.parquet")

    # Target parquet contains both target segments.
    target = gpd.GeoDataFrame(
        {
            "id": [TARGET_A, TARGET_B],
            "names": [{"primary": "Main St"}, {"primary": "Oak St"}],
            "class": ["residential", "residential"],
        },
        geometry=[LINE_A_OFF, LINE_B_OFF],
        crs="EPSG:4326",
    )
    target.to_parquet(data_dir / f"{DATASET}_v1.0.parquet")

    # Human labels: one pair with a live gers_id, one whose gers_id churned away.
    with open(labels_dir / f"human/dataset={DATASET}/data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gers_id", "target_id", "label", "labeler", "labeled_at"])
        writer.writerow([REF_PRESENT_ID, TARGET_A, "match", "test", "2026-01-01T00:00:00+00:00"])
        writer.writerow([REF_CHURNED_ID, TARGET_B, "match", "test", "2026-01-01T00:00:00+00:00"])

    # DataStore holds stored geometries for BOTH pairs (captured at label time).
    store = DataStore(DATASET, data_dir=labels_dir / "data")
    store.add(
        gers_id=REF_PRESENT_ID,
        target_id=TARGET_A,
        ref_geometry=LINE_A,
        target_geometry=LINE_A_OFF,
        ref_class="residential",
        target_class="residential",
        ref_names={"primary": "Main Street"},
        target_names={"primary": "Main St"},
    )
    store.add(
        gers_id=REF_CHURNED_ID,
        target_id=TARGET_B,
        ref_geometry=LINE_B,
        target_geometry=LINE_B_OFF,
        ref_class="residential",
        target_class="residential",
        ref_names={"primary": "Oak Street"},
        target_names={"primary": "Oak St"},
    )
    store.save()

    return data_dir, labels_dir


class TestBackfillRefChurn:
    def test_churned_gers_id_backfills_from_stored_geometry(self, tmp_path):
        data_dir, labels_dir = _write_fixture(tmp_path)

        result = runner.invoke(
            app,
            [
                "backfill",
                "-l",
                str(labels_dir),
                "-d",
                str(data_dir),
                "-D",
                DATASET,
            ],
        )

        assert result.exit_code == 0, result.output
        # Both pairs must compute; nothing skipped.
        assert "Computed 2 features" in result.output, result.output
        assert "skipped=0" in result.output, result.output
        assert "missing from current" in result.output, result.output

        features = pd.read_parquet(labels_dir / f"features/dataset={DATASET}/data.parquet")
        assert len(features) == 2
        churned = features[features["gers_id"] == REF_CHURNED_ID]
        assert len(churned) == 1
        # Geometry-derived features must be real values, not skip-defaults.
        row = churned.iloc[0]
        assert pd.notna(row["hausdorff_distance_m"])
        assert row["hausdorff_distance_m"] < 50.0  # parallel ~2m-offset lines
