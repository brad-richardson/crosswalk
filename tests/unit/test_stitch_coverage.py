"""Tests for the drift-aware review-queue filter (``labeling/stitch_coverage.py``).

Stitch group_ids are content hashes of the ref/target id sets, so a regenerated
sidecar re-mints ids for already-reviewed geometry. The motivating real case:
Bogotá group ``3c3e6853`` (64x191) was reviewed de-anchored (set label keeping
55x153); the re-optimized sidecar minted the same monster as ``8e32a935``
(70x239) and the exact-id reviewed filter re-queued it as brand new.

Semantics under test (build time, serve time, and the panel feed):

* exact-id survival        -> reviewed (excluded), identical to the old filter
* drift-mapped FULL cover  -> reviewed (excluded)
* drift-mapped PARTIAL     -> queued WITH ``prior_label`` delta metadata
* no mapping               -> queued untouched

Plus: pair-scope kept-membership semantics, the ``__all__`` combined queue,
mapping-tie-break parity with the eval-side ``recover_labeled_groups``, the
review-UI prefill, and the agent panel feed's ``--calibration`` escape hatch.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from crosswalk.agent_labeling.stitch_eval import recover_labeled_groups
from crosswalk.labeling.stitch_coverage import (
    PRIOR_LABEL_KEY,
    compute_prior_coverage,
    fully_covered_group_ids,
)

DATASET = "test_dataset"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _group(gid: str, edges: list[tuple[str, str]], **extra) -> dict:
    """A current-sidecar group with selected edges (rekey-test convention)."""
    return {
        "group_id": gid,
        "edges": [{"ref_id": r, "target_id": t, "selected": True} for r, t in edges],
        "ref_ids": sorted({r for r, _ in edges}),
        "target_ids": sorted({t for _, t in edges}),
        "match_type": "M:N",
        **extra,
    }


def _label_row(
    gid: str,
    *,
    edges: list[tuple[str, str]] | None = None,
    ref_ids: list[str] | None = None,
    target_ids: list[str] | None = None,
    semantics: str = "pair",
    labeler: str = "brad",
    labeled_at: str = "2026-07-07T12:00:00+00:00",
    dataset_id: str = DATASET,
) -> dict:
    """One stitching-label CSV row in the StitchingLabelStore schema."""
    return {
        "group_id": gid,
        "dataset_id": dataset_id,
        "selected_edges": json.dumps([{"ref_id": r, "target_id": t} for r, t in (edges or [])]),
        "match_type": "M:N",
        "num_refs": len(ref_ids or {r for r, _ in (edges or [])}),
        "num_targets": len(target_ids or {t for _, t in (edges or [])}),
        "labeler": labeler,
        "labeled_at": labeled_at,
        "session_id": "s1",
        "label_semantics": semantics,
        "ref_ids": json.dumps(sorted(ref_ids)) if ref_ids else "",
        "target_ids": json.dumps(sorted(target_ids)) if target_ids else "",
    }


def _labels_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_prior_coverage: core semantics
# ---------------------------------------------------------------------------


class TestExactIdCoverage:
    """Exact-id survival must behave identically to the old exact-id filter."""

    def test_exact_id_pair_label_fully_covered(self):
        groups = [_group("g1", [("r1", "t1"), ("r2", "t1")])]
        labels = _labels_df([_label_row("g1", edges=[("r1", "t1")])])
        cov = compute_prior_coverage(groups, labels)
        assert cov["g1"].exact_id
        assert cov["g1"].fully_covered
        assert fully_covered_group_ids(cov) == {"g1"}

    def test_exact_id_reject_all_fully_covered(self):
        """Reject-all (empty edges) survives on verbatim id, exactly as before."""
        groups = [_group("g1", [("r1", "t1")])]
        labels = _labels_df([_label_row("g1", edges=[])])
        cov = compute_prior_coverage(groups, labels)
        assert cov["g1"].exact_id and cov["g1"].fully_covered

    def test_exact_id_set_label_subset_membership_still_covered(self):
        """A set label that KEPT a subset, stored under the surviving id, is
        still reviewed: the id IS the membership hash, so the reviewer saw and
        adjudicated every current member (removals are not re-litigated)."""
        groups = [_group("g1", [("r1", "t1"), ("r2", "t2")])]
        labels = _labels_df([_label_row("g1", semantics="set", ref_ids=["r1"], target_ids=["t1"])])
        cov = compute_prior_coverage(groups, labels)
        assert cov["g1"].exact_id
        assert cov["g1"].fully_covered  # by fiat, despite kept ⊂ membership

    def test_exact_id_label_never_drift_covers_other_groups(self):
        """An exact-id label is pinned to its own group; a sibling group sharing
        members must not inherit its coverage."""
        groups = [
            _group("g1", [("r1", "t1")]),
            _group("g2", [("r1", "t2")]),  # shares r1 with g1's label
        ]
        labels = _labels_df(
            [_label_row("g1", semantics="set", ref_ids=["r1"], target_ids=["t1", "t2"])]
        )
        cov = compute_prior_coverage(groups, labels)
        assert cov["g1"].fully_covered
        assert "g2" not in cov  # not even partial coverage


class TestDriftedFullCover:
    def test_drifted_set_label_superset_kept_excluded(self):
        """Old group relabeled; new group's membership ⊆ kept -> reviewed."""
        groups = [_group("new1", [("r1", "t1"), ("r2", "t2")])]
        labels = _labels_df(
            [
                _label_row(
                    "old1",
                    semantics="set",
                    ref_ids=["r1", "r2", "r3"],  # r3 kept but no longer grouped here
                    target_ids=["t1", "t2"],
                )
            ]
        )
        cov = compute_prior_coverage(groups, labels)
        assert not cov["new1"].exact_id
        assert cov["new1"].fully_covered
        assert fully_covered_group_ids(cov) == {"new1"}

    def test_drifted_pair_label_endpoints_cover_excluded(self):
        """Pair-label kept membership = selected-edge endpoints."""
        groups = [_group("new1", [("r1", "t1"), ("r2", "t1")])]
        labels = _labels_df([_label_row("old1", edges=[("r1", "t1"), ("r2", "t1")])])
        cov = compute_prior_coverage(groups, labels)
        assert not cov["new1"].exact_id
        assert cov["new1"].fully_covered


class TestDriftedPartialCover:
    def test_bogota_miniature_delta_counts(self):
        """Kept 3x2; regrouped as 3x3 with 1 new ref and 1 new target."""
        groups = [
            _group(
                "new1",
                [("r1", "t1"), ("r2", "t2"), ("r4", "t3")],  # r4/t3 never seen
            )
        ]
        labels = _labels_df(
            [
                _label_row(
                    "old1",
                    semantics="set",
                    ref_ids=["r1", "r2", "r3"],  # r3 dropped from the new group
                    target_ids=["t1", "t2"],
                )
            ]
        )
        cov = compute_prior_coverage(groups, labels)
        c = cov["new1"]
        assert not c.fully_covered
        assert c.prior_group_id == "old1"
        assert c.covered_ref_ids == ("r1", "r2")
        assert c.new_ref_ids == ("r4",)
        assert c.covered_target_ids == ("t1", "t2")
        assert c.new_target_ids == ("t3",)
        d = c.to_batch_dict()
        assert (d["n_covered_refs"], d["n_total_refs"]) == (2, 3)
        assert (d["n_covered_targets"], d["n_total_targets"]) == (2, 3)
        assert d["labeler"] == "brad"
        assert d["labeled_at"].startswith("2026-07-07")
        assert fully_covered_group_ids(cov) == set()

    def test_pair_label_ref_ids_columns_extend_kept_universe(self):
        """Pair-scope kept membership = edge endpoints ∪ ref_ids/target_ids."""
        groups = [_group("new1", [("r1", "t1"), ("r2", "t2"), ("r3", "t3")])]
        labels = _labels_df(
            [
                _label_row(
                    "old1",
                    edges=[("r1", "t1")],
                    ref_ids=["r2"],  # adjudicated via the id columns
                    target_ids=["t2"],
                )
            ]
        )
        cov = compute_prior_coverage(groups, labels)
        c = cov["new1"]
        assert not c.fully_covered
        assert c.covered_ref_ids == ("r1", "r2")  # r2 via ref_ids column
        assert c.new_ref_ids == ("r3",)
        assert c.covered_target_ids == ("t1", "t2")
        assert c.new_target_ids == ("t3",)

    def test_merged_group_union_of_partial_labels_not_excluded(self):
        """Two drifted labels each half-cover a merged group: their UNION covers
        everything, but no single label does -> the merge stays queued (with the
        best-covering label's delta)."""
        groups = [_group("merged", [("r1", "t1"), ("r2", "t2"), ("r3", "t3"), ("r4", "t4")])]
        labels = _labels_df(
            [
                _label_row("oldA", edges=[("r1", "t1"), ("r2", "t2")]),
                _label_row("oldB", edges=[("r3", "t3"), ("r4", "t4")]),
            ]
        )
        cov = compute_prior_coverage(groups, labels)
        c = cov["merged"]
        assert not c.fully_covered
        # Equal coverage -> deterministic lexicographic tie-break on prior id.
        assert c.prior_group_id == "oldA"

    def test_best_covering_label_wins(self):
        groups = [_group("merged", [("r1", "t1"), ("r2", "t2"), ("r3", "t3")])]
        labels = _labels_df(
            [
                _label_row("oldA", edges=[("r1", "t1")]),
                _label_row("oldB", edges=[("r2", "t2"), ("r3", "t3")]),
            ]
        )
        cov = compute_prior_coverage(groups, labels)
        assert cov["merged"].prior_group_id == "oldB"


class TestNoMapping:
    def test_disjoint_label_produces_no_coverage(self):
        groups = [_group("g1", [("r1", "t1")])]
        labels = _labels_df(
            [_label_row("old1", semantics="set", ref_ids=["rX"], target_ids=["tX"])]
        )
        assert compute_prior_coverage(groups, labels) == {}

    def test_drifted_reject_all_never_maps(self):
        """Empty-edge pair labels carry nothing to overlap on; once their id is
        gone they are unrecoverable (recover_empty_reject_all semantics)."""
        groups = [_group("g1", [("r1", "t1")])]
        labels = _labels_df([_label_row("old1", edges=[])])
        assert compute_prior_coverage(groups, labels) == {}

    def test_empty_labels_df(self):
        groups = [_group("g1", [("r1", "t1")])]
        assert compute_prior_coverage(groups, _labels_df([]).reindex(columns=[])) == {}
        assert compute_prior_coverage(groups, None) == {}

    def test_no_groups(self):
        labels = _labels_df([_label_row("old1", edges=[("r1", "t1")])])
        assert compute_prior_coverage([], labels) == {}


class TestEvalMapperParity:
    """The coverage mapping must agree with the eval-side drift mapper —
    including its #354 deterministic tie-break — so the queue filter can never
    diverge from what stitch-eval / stitch-rekey resolve."""

    def test_set_label_tie_breaks_like_recover_labeled_groups(self):
        # Two groups overlap the set label equally (one member each); the
        # mapper must resolve to the lexicographically smallest group_id.
        groups = [
            _group("gB", [("r1", "tB")]),
            _group("gA", [("r2", "tA")]),
        ]
        labels = _labels_df(
            [_label_row("old1", semantics="set", ref_ids=["r1", "r2"], target_ids=[])]
        )
        rec = recover_labeled_groups(groups, labels)
        assert rec["set"] == [("old1", "gA")]  # eval-side pick
        cov = compute_prior_coverage(groups, labels)
        assert set(cov) == {"gA"}  # coverage keys on the same pick
        assert cov["gA"].prior_group_id == "old1"

    def test_pair_label_maps_to_recover_best_group(self):
        # Edges split 2-vs-1 across current groups: coverage must follow
        # recover's best-group choice (the 2-edge group).
        groups = [
            _group("gX", [("r1", "t1"), ("r2", "t2")]),
            _group("gY", [("r3", "t3")]),
        ]
        labels = _labels_df([_label_row("old1", edges=[("r1", "t1"), ("r2", "t2"), ("r3", "t3")])])
        rec = recover_labeled_groups(groups, labels)
        assert rec["split"][0][1] == "gX"
        cov = compute_prior_coverage(groups, labels)
        assert set(cov) == {"gX"}


# ---------------------------------------------------------------------------
# serve time: web/services.get_unreviewed_stitch_groups
# ---------------------------------------------------------------------------


def _patch_stores(monkeypatch, labels_by_ds: dict[str, pd.DataFrame], calls=None):
    class FakeStore:
        def __init__(self, dataset_id, *a, **k):
            self.dataset_id = dataset_id
            if calls is not None:
                calls.append(dataset_id)

        def load(self, dataset_id=None):
            df = labels_by_ds.get(self.dataset_id)
            if df is None:
                return _labels_df([]).reindex(columns=["group_id", "dataset_id", "selected_edges"])
            return df

    monkeypatch.setattr("crosswalk.labeling.stitching_store.StitchingLabelStore", FakeStore)


class TestServeTimeFilter:
    def test_exact_id_excluded_regression(self, monkeypatch):
        """The id-stable path must behave exactly as before."""
        _patch_stores(monkeypatch, {DATASET: _labels_df([_label_row("g1", edges=[])])})
        from crosswalk.web import services

        groups = [_group("g1", [("r1", "t1")]), _group("g2", [("r2", "t2")])]
        out = services.get_unreviewed_stitch_groups(DATASET, groups)
        assert [g["group_id"] for g in out] == ["g2"]

    def test_drifted_full_cover_excluded(self, monkeypatch):
        _patch_stores(
            monkeypatch,
            {
                DATASET: _labels_df(
                    [
                        _label_row(
                            "old1",
                            semantics="set",
                            ref_ids=["r1", "r2"],
                            target_ids=["t1"],
                        )
                    ]
                )
            },
        )
        from crosswalk.web import services

        groups = [_group("new1", [("r1", "t1"), ("r2", "t1")])]
        assert services.get_unreviewed_stitch_groups(DATASET, groups) == []

    def test_partial_cover_included_with_fresh_delta(self, monkeypatch):
        _patch_stores(
            monkeypatch,
            {
                DATASET: _labels_df(
                    [_label_row("old1", semantics="set", ref_ids=["r1"], target_ids=["t1"])]
                )
            },
        )
        from crosswalk.web import services

        group = _group("new1", [("r1", "t1"), ("r2", "t2")])
        # A stale build-time delta must be superseded by the fresh computation.
        group[PRIOR_LABEL_KEY] = {"prior_group_id": "stale"}
        out = services.get_unreviewed_stitch_groups(DATASET, [group])
        assert len(out) == 1
        d = out[0][PRIOR_LABEL_KEY]
        assert d["prior_group_id"] == "old1"
        assert (d["n_covered_refs"], d["n_total_refs"]) == (1, 2)
        assert d["new_ref_ids"] == ["r2"]
        assert d["new_target_ids"] == ["t2"]

    def test_unmapped_group_passthrough_drops_stale_delta(self, monkeypatch):
        _patch_stores(monkeypatch, {DATASET: _labels_df([])})
        from crosswalk.web import services

        group = _group("g1", [("r1", "t1")])
        group[PRIOR_LABEL_KEY] = {"prior_group_id": "stale"}
        out = services.get_unreviewed_stitch_groups(DATASET, [group])
        assert [g["group_id"] for g in out] == ["g1"]
        assert PRIOR_LABEL_KEY not in out[0]

    def test_all_queue_filters_per_owning_dataset(self, monkeypatch):
        """__all__: each group is coverage-checked against ITS OWN dataset's
        labels; a colliding group_id reviewed in ds_a stays visible for ds_b,
        and drift coverage in ds_a must not leak onto ds_b's groups."""
        from crosswalk.filenames import STITCH_ALL_QUEUE

        labels_a = _labels_df(
            [
                _label_row("g1", edges=[("r1", "t1")], dataset_id="ds_a"),  # exact
                _label_row(  # drift-covers ds_a's "new1"
                    "oldA",
                    semantics="set",
                    ref_ids=["r5"],
                    target_ids=["t5"],
                    dataset_id="ds_a",
                ),
            ]
        )
        _patch_stores(monkeypatch, {"ds_a": labels_a, "ds_b": _labels_df([])})
        from crosswalk.web import services

        groups = [
            _group("g1", [("r1", "t1")], dataset_id="ds_a"),  # exact -> hidden
            _group("g1", [("r1", "t1")], dataset_id="ds_b"),  # same id -> visible
            _group("new1", [("r5", "t5")], dataset_id="ds_a"),  # drift-cover -> hidden
            _group("new1", [("r5", "t5")], dataset_id="ds_b"),  # no labels -> visible
        ]
        out = services.get_unreviewed_stitch_groups(STITCH_ALL_QUEUE, groups)
        assert [(g["group_id"], g["dataset_id"]) for g in out] == [
            ("g1", "ds_b"),
            ("new1", "ds_b"),
        ]

    def test_labels_loaded_once_per_dataset(self, monkeypatch):
        from crosswalk.filenames import STITCH_ALL_QUEUE

        calls: list[str] = []
        _patch_stores(monkeypatch, {}, calls=calls)
        from crosswalk.web import services

        groups = [_group(f"g{i}", [(f"r{i}", f"t{i}")], dataset_id="ds_a") for i in range(5)]
        services.get_unreviewed_stitch_groups(STITCH_ALL_QUEUE, groups)
        assert calls == ["ds_a"]  # one store construction, not five


# ---------------------------------------------------------------------------
# review UI: prior-label prefill (routes/stitching._render_group)
# ---------------------------------------------------------------------------


class TestRenderGroupPrefill:
    def _group_with_prior(self):
        group = _group("new1", [("r1", "t1"), ("r2", "t2"), ("r3", "t3")])
        group[PRIOR_LABEL_KEY] = {
            "prior_group_id": "old1",
            "labeler": "brad",
            "labeled_at": "2026-07-07T12:00:00+00:00",
            "label_semantics": "set",
            "n_covered_refs": 2,
            "n_total_refs": 3,
            "n_covered_targets": 2,
            "n_total_targets": 3,
            "covered_ref_ids": ["r1", "r2"],
            "new_ref_ids": ["r3"],
            "covered_target_ids": ["t1", "t2"],
            "new_target_ids": ["t3"],
        }
        return group

    def test_anchored_prefill_covered_active_new_flagged(self):
        from crosswalk.web.routes.stitching import _render_group

        _, ctx = _render_group(self._group_with_prior(), DATASET, deanchored=False)
        assert ctx["preseed_active_refs"] == ["r1", "r2"]
        assert ctx["preseed_active_targets"] == ["t1", "t2"]
        assert ctx["preseed_edges"] == []  # membership prefill, never pair-scope
        assert sorted(ctx["preseed_inactive_ids"]) == ["r3", "t3"]
        assert sorted(ctx["prior_new_ids"]) == ["r3", "t3"]
        assert ctx["prior_label"]["prior_group_id"] == "old1"

    def test_deanchored_prefill_overrides_blank_slate(self):
        from crosswalk.web.routes.stitching import _render_group

        _, ctx = _render_group(self._group_with_prior(), DATASET, deanchored=True)
        # The prior label (the reviewer's OWN judgment) overrides the blank
        # slate; the optimizer's proposal stays hidden via deanchored=True.
        assert ctx["deanchored"] is True
        assert ctx["preseed_active_refs"] == ["r1", "r2"]
        assert sorted(ctx["preseed_inactive_ids"]) == ["r3", "t3"]

    def test_no_prior_label_keeps_existing_behavior(self):
        from crosswalk.web.routes.stitching import _render_group

        group = _group("g1", [("r1", "t1")])
        # Anchored: the optimizer preseed from _build_stitch_options is passed
        # through untouched (None here — the fixture has no optimizer
        # assignment — which the template reads as all-active pills).
        _, ctx = _render_group(group, DATASET, deanchored=False)
        assert ctx.get("preseed_active_refs") is None
        assert "prior_new_ids" not in ctx
        # De-anchored: blank slate, exactly as before.
        _, ctx = _render_group(group, DATASET, deanchored=True)
        assert ctx["preseed_active_refs"] == []
        assert sorted(ctx["preseed_inactive_ids"]) == ["r1", "t1"]
        assert "prior_new_ids" not in ctx


class TestGroupTemplateRender:
    """The banner + new-pill flags must actually render through the route."""

    def test_fragment_renders_banner_prefill_and_new_flags(self):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from crosswalk.web.app import create_app

        group = _group("new1", [("r1", "t1"), ("r2", "t2")])
        for e in group["edges"]:
            e["confidence"] = 0.9
        group[PRIOR_LABEL_KEY] = {
            "prior_group_id": "old1",
            "labeler": "brad",
            "labeled_at": "2026-07-07T12:00:00+00:00",
            "label_semantics": "set",
            "n_covered_refs": 1,
            "n_total_refs": 2,
            "n_covered_targets": 1,
            "n_total_targets": 2,
            "covered_ref_ids": ["r1"],
            "new_ref_ids": ["r2"],
            "covered_target_ids": ["t1"],
            "new_target_ids": ["t2"],
        }
        batch = {"dataset_id": DATASET, "groups": [group]}
        patches = [
            patch("crosswalk.web.routes.stitching.list_datasets", return_value=[DATASET]),
            patch("crosswalk.web.routes.stitching.load_stitch_batch", return_value=batch),
            patch(
                "crosswalk.web.routes.stitching.get_unreviewed_stitch_groups",
                return_value=[group],
            ),
        ]
        for p in patches:
            p.start()
        try:
            client = TestClient(create_app())
            resp = client.get(f"/stitching-review/group?dataset={DATASET}&group_index=0")
            assert resp.status_code == 200
            html = resp.text
            assert 'id="prior-label-notice"' in html
            assert "1/2 refs" in html
            assert "old1" in html
            # New-since-label members are class-flagged; titles stay bare ids
            # (the submit path collects ids from pill titles).
            assert "segment-pill-new" in html
            assert 'title="r2"' in html
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# build time: cli/data._generate_stitch_batch_for_dataset
# ---------------------------------------------------------------------------


class TestBuildTimeQueue:
    def _setup(self, tmp_path, monkeypatch, labels_rows, failed_ids):
        """Sidecar with three groups; labels + panel-failure gate as given."""
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "data" / "output"
        output_dir.mkdir(parents=True)
        groups = [
            _group("covered1", [("r1", "t1"), ("r2", "t1")]),
            _group("partial1", [("r3", "t3"), ("r4", "t4")]),
            _group("fresh1", [("r5", "t5")]),
        ]
        (output_dir / f"{DATASET}_groups.json").write_text(json.dumps({"groups": groups}))

        # Labels live under CWD-relative labels/stitching (store default).
        part = tmp_path / "labels" / "stitching" / f"dataset={DATASET}"
        part.mkdir(parents=True)
        _labels_df(labels_rows).to_csv(part / "data.csv", index=False)

        cache_dir = tmp_path / "cache_stitch"
        cache_dir.mkdir()
        monkeypatch.setattr("crosswalk.filenames.STITCH_CACHE_DIR", cache_dir)
        monkeypatch.setattr(
            "crosswalk.agent_labeling.panel_routing.panel_failed_group_ids",
            lambda ds: set(failed_ids),
        )
        monkeypatch.setattr(
            "crosswalk.agent_labeling.panel_routing.attach_panel_route_reasons",
            lambda selected, ds: 0,
        )
        monkeypatch.setattr("crosswalk.cli.data._fill_spatial_context", lambda *a, **k: None)
        return output_dir, cache_dir

    def test_covered_excluded_partial_gets_delta(self, tmp_path, monkeypatch):
        from crosswalk.cli.data import _generate_stitch_batch_for_dataset

        labels_rows = [
            # Drift-covers "covered1" fully (kept ⊇ current membership).
            _label_row("oldC", semantics="set", ref_ids=["r1", "r2", "rGone"], target_ids=["t1"]),
            # Partially covers "partial1" (r4/t4 never seen).
            _label_row("oldP", semantics="set", ref_ids=["r3"], target_ids=["t3"]),
        ]
        output_dir, cache_dir = self._setup(
            tmp_path,
            monkeypatch,
            labels_rows,
            failed_ids=["covered1", "partial1", "fresh1"],
        )
        ok = _generate_stitch_batch_for_dataset(
            DATASET,
            output_dir=output_dir,
            batch_size=10,
            k_alternatives=3,
            include_unvoted=False,
        )
        assert ok
        batch = json.loads((cache_dir / f"{DATASET}_batch.json").read_text())
        by_id = {g["group_id"]: g for g in batch["groups"]}
        assert "covered1" not in by_id  # drift-mapped full cover -> excluded
        assert set(by_id) == {"partial1", "fresh1"}
        delta = by_id["partial1"][PRIOR_LABEL_KEY]
        assert delta["prior_group_id"] == "oldP"
        assert (delta["n_covered_refs"], delta["n_total_refs"]) == (1, 2)
        assert delta["new_ref_ids"] == ["r4"]
        assert PRIOR_LABEL_KEY not in by_id["fresh1"]

    def test_exact_id_reviewed_still_excluded(self, tmp_path, monkeypatch):
        """Regression: the pre-drift exact-id path is unchanged."""
        from crosswalk.cli.data import _generate_stitch_batch_for_dataset

        labels_rows = [_label_row("covered1", edges=[("r1", "t1"), ("r2", "t1")])]
        output_dir, cache_dir = self._setup(
            tmp_path, monkeypatch, labels_rows, failed_ids=["covered1", "fresh1"]
        )
        ok = _generate_stitch_batch_for_dataset(
            DATASET,
            output_dir=output_dir,
            batch_size=10,
            k_alternatives=3,
            include_unvoted=False,
        )
        assert ok
        batch = json.loads((cache_dir / f"{DATASET}_batch.json").read_text())
        assert {g["group_id"] for g in batch["groups"]} == {"fresh1"}


# ---------------------------------------------------------------------------
# panel feed: crosswalk agent stitch-batch --calibration
# ---------------------------------------------------------------------------


class TestCalibrationFlag:
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("crosswalk.filenames.PROJECT_ROOT", tmp_path)
        output_dir = tmp_path / "data" / "output"
        output_dir.mkdir(parents=True)
        groups = [
            _group("labeled1", [("r1", "t1")]),
            _group("covered1", [("r2", "t2")]),
            _group("fresh1", [("r3", "t3")]),
        ]
        (output_dir / f"{DATASET}_groups.json").write_text(json.dumps({"groups": groups}))

        part = tmp_path / "labels" / "stitching" / f"dataset={DATASET}"
        part.mkdir(parents=True)
        _labels_df(
            [
                _label_row("labeled1", edges=[("r1", "t1")]),  # exact-id reviewed
                _label_row(  # drift-covers covered1
                    "oldC", semantics="set", ref_ids=["r2"], target_ids=["t2"]
                ),
            ]
        ).to_csv(part / "data.csv", index=False)

        captured: dict = {}

        def _fake_select(groups, reviewed_group_ids, k=15, **kwargs):
            captured["reviewed"] = set(reviewed_group_ids)
            return []  # command exits 1 ("No groups selected") after capture

        monkeypatch.setattr(
            "crosswalk.matching.batch_selection.select_stitching_batch", _fake_select
        )
        return captured

    def test_default_excludes_exact_and_drift_covered(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from crosswalk.cli import app

        captured = self._setup(tmp_path, monkeypatch)
        result = CliRunner().invoke(app, ["agent", "stitch-batch", DATASET])
        assert result.exit_code == 1  # fake select returned [] after capturing
        # Stale label ids (oldC) may sit in the exclusion set — harmless, they
        # match no sidecar group (pre-existing behavior). What matters is which
        # CURRENT groups are excluded vs votable.
        current = {"labeled1", "covered1", "fresh1"}
        assert captured["reviewed"] & current == {"labeled1", "covered1"}

    def test_calibration_keeps_labeled_groups_votable(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from crosswalk.cli import app

        captured = self._setup(tmp_path, monkeypatch)
        result = CliRunner().invoke(app, ["agent", "stitch-batch", DATASET, "--calibration"])
        assert result.exit_code == 1
        assert captured["reviewed"] == set()


# ---------------------------------------------------------------------------
# eval-side non-interference: extraction-free reuse
# ---------------------------------------------------------------------------


def test_recover_labeled_groups_untouched_by_coverage_import():
    """The queue filter REUSES the eval mapper rather than shadowing it: the
    symbol consumed by stitch_coverage must be the very function stitch_eval
    exports (no parallel implementation whose tie-breaks could diverge)."""
    import crosswalk.agent_labeling.stitch_eval as se
    import crosswalk.labeling.stitch_coverage as sc

    assert sc.recover_labeled_groups is se.recover_labeled_groups


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
