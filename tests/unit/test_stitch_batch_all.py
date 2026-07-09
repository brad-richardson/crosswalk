"""Unit tests for the combined cross-dataset stitching queue (``__all__``).

Covers the three pieces that make ``crosswalk data stitch-batch-all`` and the
web ``/stitching-review`` combined queue correct:

1. ``get_unreviewed_stitch_groups`` filters each group against ITS OWN dataset's
   labels (a group reviewed in dataset A must not be hidden in dataset B, even
   with a colliding group_id).
2. The ``stitch-batch-all --no-refresh`` combine stamps every group with its
   owning ``dataset_id``, orders by dataset, and writes the ``__all__`` batch.
3. The panel-CSV field-limit fix lifts csv's per-field cap.
"""

import csv
import json

from typer.testing import CliRunner

from crosswalk.cli import app
from crosswalk.filenames import STITCH_ALL_QUEUE

runner = CliRunner()


def _batch(dataset_id, group_ids):
    return {
        "dataset_id": dataset_id,
        "generated_at": "2026-07-09T00:00:00+00:00",
        "batch_size": len(group_ids),
        "groups": [{"group_id": gid, "match_type": "M:N"} for gid in group_ids],
    }


class TestGetUnreviewedPerGroupDataset:
    """The unreviewed filter must key on each group's owning dataset."""

    def test_normal_batch_filters_against_page_dataset(self, monkeypatch):
        from crosswalk.web import services

        class FakeStore:
            def __init__(self, dataset_id, *a, **k):
                self.dataset_id = dataset_id

            def get_reviewed_group_ids(self, dataset_id=None):
                return {"g_reviewed"}

        monkeypatch.setattr("crosswalk.labeling.stitching_store.StitchingLabelStore", FakeStore)
        groups = [{"group_id": "g_reviewed"}, {"group_id": "g_fresh"}]
        out = services.get_unreviewed_stitch_groups("ds_a", groups)
        assert [g["group_id"] for g in out] == ["g_fresh"]

    def test_aggregate_routes_each_group_to_its_own_dataset(self, monkeypatch):
        from crosswalk.web import services

        # ds_a has "g1" reviewed; ds_b has nothing reviewed. A colliding group_id
        # must stay visible for ds_b.
        reviewed = {"ds_a": {"g1"}, "ds_b": set()}

        class FakeStore:
            def __init__(self, dataset_id, *a, **k):
                self.dataset_id = dataset_id

            def get_reviewed_group_ids(self, dataset_id=None):
                return reviewed[self.dataset_id]

        monkeypatch.setattr("crosswalk.labeling.stitching_store.StitchingLabelStore", FakeStore)
        groups = [
            {"group_id": "g1", "dataset_id": "ds_a"},  # reviewed in ds_a -> hidden
            {"group_id": "g1", "dataset_id": "ds_b"},  # same id, ds_b -> visible
            {"group_id": "g2", "dataset_id": "ds_a"},  # fresh -> visible
        ]
        out = services.get_unreviewed_stitch_groups(STITCH_ALL_QUEUE, groups)
        assert [(g["group_id"], g["dataset_id"]) for g in out] == [
            ("g1", "ds_b"),
            ("g2", "ds_a"),
        ]

    def test_reviewed_set_loaded_once_per_dataset(self, monkeypatch):
        """Reviewed-id sets are cached per owning dataset, not per group."""
        from crosswalk.web import services

        calls = []

        class FakeStore:
            def __init__(self, dataset_id, *a, **k):
                self.dataset_id = dataset_id
                calls.append(dataset_id)

            def get_reviewed_group_ids(self, dataset_id=None):
                return set()

        monkeypatch.setattr("crosswalk.labeling.stitching_store.StitchingLabelStore", FakeStore)
        groups = [{"group_id": f"g{i}", "dataset_id": "ds_a"} for i in range(5)]
        services.get_unreviewed_stitch_groups(STITCH_ALL_QUEUE, groups)
        assert calls == ["ds_a"]  # one store construction, not five


class TestStitchBatchAllCombine:
    """The --no-refresh combine step (pure file IO)."""

    def _patch_real_datasets(self, monkeypatch, names):
        # The combine gates on DatasetLoader membership (CI has no data/raw), so
        # control the "real datasets" set explicitly.
        monkeypatch.setattr(
            "crosswalk.datasets.loader.DatasetLoader.list_available",
            lambda self: list(names),
        )

    def test_combine_stamps_and_orders(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "stitch"
        cache_dir.mkdir()
        # Write out-of-alphabetical order to prove the combine sorts by dataset.
        (cache_dir / "de_berlin_roads_batch.json").write_text(
            json.dumps(_batch("de_berlin_roads", ["b1", "b2"]))
        )
        (cache_dir / "co_bogota_roads_batch.json").write_text(
            json.dumps(_batch("co_bogota_roads", ["c1"]))
        )
        # An empty per-dataset queue contributes nothing.
        (cache_dir / "empty_ds_batch.json").write_text(json.dumps(_batch("empty_ds", [])))
        # A pre-existing aggregate must not fold into itself.
        (cache_dir / f"{STITCH_ALL_QUEUE}_batch.json").write_text(
            json.dumps(_batch(STITCH_ALL_QUEUE, ["stale"]))
        )

        monkeypatch.setattr("crosswalk.filenames.STITCH_CACHE_DIR", cache_dir)
        self._patch_real_datasets(monkeypatch, ["co_bogota_roads", "de_berlin_roads", "empty_ds"])

        result = runner.invoke(app, ["data", "stitch-batch-all", "--no-refresh"])
        assert result.exit_code == 0, result.output

        combined = json.loads((cache_dir / f"{STITCH_ALL_QUEUE}_batch.json").read_text())
        assert combined["dataset_id"] == STITCH_ALL_QUEUE
        rows = [(g["group_id"], g["dataset_id"]) for g in combined["groups"]]
        # Ordered by dataset (bogota < berlin), self-aggregate excluded, empty skipped.
        assert rows == [
            ("c1", "co_bogota_roads"),
            ("b1", "de_berlin_roads"),
            ("b2", "de_berlin_roads"),
        ]
        assert combined["batch_size"] == 3

    def test_combine_excludes_comparison_artifacts(self, tmp_path, monkeypatch):
        """before_/after_ batch caches must never fold into the queue — their
        labels would route to a junk labels/stitching/dataset=before_*/ partition."""
        cache_dir = tmp_path / "stitch"
        cache_dir.mkdir()
        (cache_dir / "us_boston_streets_batch.json").write_text(
            json.dumps(_batch("us_boston_streets", ["real1"]))
        )
        # A comparison artifact left behind by the change-tracking workflow.
        (cache_dir / "before_us_boston_streets_batch.json").write_text(
            json.dumps(_batch("before_us_boston_streets", ["junk1", "junk2"]))
        )

        monkeypatch.setattr("crosswalk.filenames.STITCH_CACHE_DIR", cache_dir)
        # Only the real dataset is known to DatasetLoader.
        self._patch_real_datasets(monkeypatch, ["us_boston_streets"])

        result = runner.invoke(app, ["data", "stitch-batch-all", "--no-refresh"])
        assert result.exit_code == 0, result.output

        combined = json.loads((cache_dir / f"{STITCH_ALL_QUEUE}_batch.json").read_text())
        owners = {g["dataset_id"] for g in combined["groups"]}
        assert owners == {"us_boston_streets"}  # artifact excluded
        assert [g["group_id"] for g in combined["groups"]] == ["real1"]

    def test_combine_empty_when_no_batches(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "stitch"
        cache_dir.mkdir()
        monkeypatch.setattr("crosswalk.filenames.STITCH_CACHE_DIR", cache_dir)
        self._patch_real_datasets(monkeypatch, ["us_boston_streets"])

        result = runner.invoke(app, ["data", "stitch-batch-all", "--no-refresh"])
        assert result.exit_code == 0, result.output
        combined = json.loads((cache_dir / f"{STITCH_ALL_QUEUE}_batch.json").read_text())
        assert combined["groups"] == []


class TestFindGroupCollisionSafe:
    """Writer/lookup paths must disambiguate a shared group_id by owning dataset."""

    def test_find_group_matches_owning_dataset(self):
        from crosswalk.web.routes.stitching import _find_group

        all_groups = [
            {"group_id": "g1", "dataset_id": "ds_a"},
            {"group_id": "g1", "dataset_id": "ds_b"},
        ]
        # Without a dataset hint, first match wins (per-dataset queue behavior).
        assert _find_group(all_groups, "g1")["dataset_id"] == "ds_a"
        # With the owning dataset, the correct occurrence is resolved.
        assert _find_group(all_groups, "g1", "ds_b")["dataset_id"] == "ds_b"
        assert _find_group(all_groups, "g1", "ds_a")["dataset_id"] == "ds_a"
        # Unknown (id, dataset) pair -> no match.
        assert _find_group(all_groups, "g1", "ds_c") is None
        assert _find_group(all_groups, "missing") is None


def test_panel_csv_field_limit_raised():
    """Importing panel_routing lifts csv's per-field limit above the default."""
    import crosswalk.agent_labeling.panel_routing  # noqa: F401

    # Default is 131072; the fix raises it far above that so large edge_set
    # cells in consensus.csv parse instead of raising "field larger than limit".
    assert csv.field_size_limit() > 131072
