"""Pairwise stitch-review geometry and web-flow tests."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, mapping
from starlette.requests import Request


def _import_mbench_stitch_metrics():
    """Import mbench's evaluator (own package tree, path-shimmed like the parity tests)."""
    mbench_src = Path(__file__).resolve().parents[2] / "mbench" / "src"
    if str(mbench_src) not in sys.path:
        sys.path.insert(0, str(mbench_src))
    from mbench.eval.stitch_metrics import evaluate_stitch_groups

    return evaluate_stitch_groups


from crosswalk.labeling.stitch_pair_review import (
    candidate_edge_union,
    enrich_candidate_endpoints,
    missing_candidate_endpoint_ids,
)
from crosswalk.labeling.stitching_store import (
    ADJUDICATION_SCOPE_EXACT_IDENTITY,
    ADJUDICATION_SCOPE_PARTIAL_IDENTITY,
    StitchingLabelStore,
)
from crosswalk.web.routes.stitching import (
    _build_pairwise_candidates,
    stitching_pair_features,
    stitching_review,
    stitching_select,
)
from crosswalk.web.services import (
    get_pairwise_revisit_groups,
    record_partial_identity_progress,
)


def _line(y: float) -> dict:
    return mapping(LineString([(0, y), (1, y)]))


def _edge(ref_id: str, target_id: str, confidence: float = 0.8) -> dict:
    return {
        "ref_id": ref_id,
        "target_id": target_id,
        "confidence": confidence,
        "gers_start_frac": 0.25,
        "gers_end_frac": 0.75,
        "local_start_frac": 0.1,
        "local_end_frac": 0.9,
    }


def test_candidate_edge_union_is_stable_and_deduplicated():
    selected = _edge("r1", "t1", 0.9)
    duplicate = _edge("r1", "t1", 0.2)
    rejected = _edge("r2", "t1", 0.4)
    group = {"edges": [selected], "rejected_edges": [duplicate, rejected]}

    assert candidate_edge_union(group) == [selected, rejected]


def test_enrich_candidate_endpoints_attaches_external_geometry(monkeypatch):
    group = {
        "ref_ids": ["r1"],
        "target_ids": ["t1"],
        "ref_geometries": {"r1": _line(0)},
        "target_geometries": {"t1": _line(1)},
        "edges": [_edge("r1", "t1")],
        "rejected_edges": [_edge("r2", "t1")],
    }
    assert missing_candidate_endpoint_ids([group]) == ({"r2"}, set())

    ref_rows = gpd.GeoDataFrame(
        [{"id": "r2", "names": "External Road", "class": "primary"}],
        geometry=[LineString([(2, 0), (3, 0)])],
        crs="EPSG:4326",
    )
    calls = []

    def fake_read(path, *, columns, filters):
        calls.append((path, columns, filters))
        return ref_rows

    monkeypatch.setattr(
        "crosswalk.labeling.stitch_pair_review.find_overture_segments",
        lambda *_: Path("ref.parquet"),
    )
    monkeypatch.setattr(
        "crosswalk.labeling.stitch_pair_review.find_target_file",
        lambda *_: Path("target.parquet"),
    )
    monkeypatch.setattr(gpd, "read_parquet", fake_read)

    stats = enrich_candidate_endpoints([group], "ds", data_dir=Path("raw"))

    assert stats == {
        "requested_ref": 1,
        "requested_target": 0,
        "attached_ref": 1,
        "attached_target": 0,
    }
    assert calls[0][2] == [("id", "in", ["r2"])]
    assert group["candidate_ref_geometries"]["r2"]["type"] == "LineString"
    assert group["candidate_ref_names"]["r2"] == "External Road"
    assert group["candidate_ref_classes"]["r2"] == "primary"


def test_pairwise_candidates_include_aligned_external_pair_and_prior_defaults():
    group = {
        "ref_ids": ["r1"],
        "target_ids": ["t1"],
        "ref_geometries": {"r1": _line(0)},
        "target_geometries": {"t1": _line(1)},
        "candidate_ref_geometries": {"r2": _line(2)},
        "candidate_ref_names": {"r2": "External"},
        "candidate_ref_classes": {"r2": "secondary"},
        "edges": [_edge("r1", "t1", 0.9)],
        "rejected_edges": [_edge("r2", "t1", 0.4)],
    }
    context = {
        "preseed_edges": [],
        "preseed_active_refs": ["r1"],
        "preseed_active_targets": ["t1"],
    }

    pairs = _build_pairwise_candidates(group, context)

    assert [(pair["ref_id"], pair["target_id"]) for pair in pairs] == [
        ("r1", "t1"),
        ("r2", "t1"),
    ]
    assert pairs[0]["default_resolution"] == "keep"
    assert pairs[0]["default_identity"] == "match"
    assert pairs[1]["default_resolution"] == "drop"
    assert pairs[1]["default_identity"] == "unsure"
    assert pairs[1]["is_external"] is True
    assert pairs[1]["geometry_available"] is True
    coords = pairs[1]["geometry"]["reference"]["coordinates"]
    assert [list(coord) for coord in coords] == [[0.25, 2.0], [0.75, 2.0]]


def test_pairwise_cards_confirm_first_confidence_desc_slivers_last():
    """Keep-seeded cards lead, confidence descends per bucket, slivers trail."""
    sliver = _edge("r2", "t1", 0.99)
    sliver.update(
        {
            "gers_start_frac": 0.5,
            "gers_end_frac": 0.5 + 1e-9,
            "local_start_frac": 0.5,
            "local_end_frac": 0.5 + 1e-9,
        }
    )
    group = {
        "ref_ids": ["r1", "r2"],
        "target_ids": ["t1", "t2"],
        "ref_geometries": {"r1": _line(0), "r2": _line(2)},
        "target_geometries": {"t1": _line(1), "t2": _line(3)},
        "edges": [_edge("r1", "t1", 0.6), _edge("r2", "t2", 0.95)],
        "rejected_edges": [_edge("r1", "t2", 0.9), sliver],
    }
    context = {
        "preseed_edges": [
            {"ref_id": "r1", "target_id": "t1"},
            {"ref_id": "r2", "target_id": "t2"},
        ]
    }

    pairs = _build_pairwise_candidates(group, context)

    assert [(pair["ref_id"], pair["target_id"]) for pair in pairs] == [
        ("r2", "t2"),  # keep-seeded, confidence 0.95
        ("r1", "t1"),  # keep-seeded, confidence 0.6
        ("r1", "t2"),  # drop, confidence 0.9
        ("r2", "t1"),  # drop sliver trails despite confidence 0.99
    ]
    assert pairs[3]["is_sliver"] is True


def test_pairwise_cards_prefill_saved_partial_dispositions():
    """A prior partial save overrides seed defaults and flags cards reviewed."""
    group = {
        "ref_ids": ["r1"],
        "target_ids": ["t1", "t2"],
        "ref_geometries": {"r1": _line(0)},
        "target_geometries": {"t1": _line(1), "t2": _line(2)},
        "edges": [_edge("r1", "t1", 0.9)],
        "rejected_edges": [_edge("r1", "t2", 0.4)],
    }
    context = {
        "prior_label": {
            "edge_dispositions": [
                {
                    "ref_id": "r1",
                    "target_id": "t1",
                    "resolution": "drop",
                    "identity": "no_match",
                },
                # Malformed entries are ignored, never crash the wizard.
                {"ref_id": "r1", "target_id": "t2", "resolution": "keep", "identity": "bogus"},
                # keep => match invariant: never prefill a state the UI can't produce.
                {"ref_id": "r1", "target_id": "t2", "resolution": "keep", "identity": "no_match"},
            ]
        }
    }

    pairs = _build_pairwise_candidates(group, context)
    by_key = {(pair["ref_id"], pair["target_id"]): pair for pair in pairs}

    saved = by_key[("r1", "t1")]
    assert saved["saved"] is True
    assert saved["default_resolution"] == "drop"
    assert saved["default_identity"] == "no_match"
    fresh = by_key[("r1", "t2")]
    assert fresh["saved"] is False
    assert fresh["default_resolution"] == "drop"


def test_pairwise_card_order_frozen_across_partial_saves():
    """Saved flips change prefill but never reorder cards or chunk boundaries."""
    group = {
        "ref_ids": ["r1"],
        "target_ids": ["t1", "t2"],
        "ref_geometries": {"r1": _line(0)},
        "target_geometries": {"t1": _line(1), "t2": _line(2)},
        "edges": [_edge("r1", "t1", 0.9)],
        "rejected_edges": [_edge("r1", "t2", 0.4)],
    }
    seed_order = [
        (pair["ref_id"], pair["target_id"]) for pair in _build_pairwise_candidates(group, {})
    ]
    assert seed_order == [("r1", "t1"), ("r1", "t2")]

    # A partial save that flips the keep-seeded card to drop and the
    # drop-seeded card to keep must not migrate either across buckets.
    context = {
        "prior_label": {
            "edge_dispositions": [
                {"ref_id": "r1", "target_id": "t1", "resolution": "drop", "identity": "no_match"},
                {"ref_id": "r1", "target_id": "t2", "resolution": "keep", "identity": "match"},
            ]
        }
    }
    pairs = _build_pairwise_candidates(group, context)

    assert [(pair["ref_id"], pair["target_id"]) for pair in pairs] == seed_order
    assert pairs[0]["default_resolution"] == "drop"
    assert pairs[1]["default_resolution"] == "keep"


def test_pairwise_candidate_with_null_alignment_uses_full_geometry():
    edge = _edge("r1", "t1")
    edge["gers_start_frac"] = None
    group = {
        "ref_ids": ["r1"],
        "target_ids": ["t1"],
        "ref_geometries": {"r1": _line(0)},
        "target_geometries": {"t1": _line(1)},
        "edges": [edge],
    }

    pair = _build_pairwise_candidates(group, {})[0]

    assert pair["geometry_available"] is True
    assert pair["geometry"]["reference"] == pair["geometry"]["reference_full"]


def test_pairwise_queue_orders_groups_easy_first(tmp_path, monkeypatch):
    """Queue serves groups by ascending candidate-card count, id tiebreak."""
    monkeypatch.chdir(tmp_path)
    store = StitchingLabelStore("ds")
    store.add("big", [], "M:N", 1, 3, "brad", "s1")
    store.add("small", [], "1:N", 1, 1, "brad", "s2")

    big = {
        "dataset_id": "ds",
        "group_id": "big",
        "ref_ids": ["r1"],
        "target_ids": ["t1", "t2", "t3"],
        "edges": [_edge("r1", "t1"), _edge("r1", "t2")],
        "rejected_edges": [_edge("r1", "t3")],
    }
    small = {
        "dataset_id": "ds",
        "group_id": "small",
        "ref_ids": ["r1"],
        "target_ids": ["t1"],
        "edges": [_edge("r1", "t1")],
        "rejected_edges": [],
    }

    queue = get_pairwise_revisit_groups("__pairwise__", [big, small])

    assert [group["group_id"] for group in queue] == ["small", "big"]


def test_partial_progress_preserves_prior_label_and_keeps_group_queued(tmp_path, monkeypatch):
    """A partial save stores decided-so-far dispositions without completing."""
    monkeypatch.chdir(tmp_path)
    store = StitchingLabelStore("ds")
    store.add(
        "g1",
        [],
        "M:N",
        2,
        1,
        "panel_unanimous_v7",
        "s1",
        label_semantics="set",
        ref_ids=["r1", "r2"],
        target_ids=["t1"],
        notes="prior note",
        adjudication_scope="membership",
    )
    prior_labeled_at = StitchingLabelStore("ds").load("ds").iloc[0]["labeled_at"]

    dispositions = [
        {"ref_id": "r1", "target_id": "t1", "resolution": "keep", "identity": "match"},
    ]
    record_partial_identity_progress("ds", "g1", dispositions, notes="prior note")

    row = StitchingLabelStore("ds").load("ds").iloc[0]
    assert row["adjudication_scope"] == ADJUDICATION_SCOPE_PARTIAL_IDENTITY
    assert json.loads(row["edge_dispositions"]) == dispositions
    # Resolver-facing truth of the prior label is preserved verbatim.
    assert row["label_semantics"] == "set"
    assert json.loads(row["ref_ids"]) == ["r1", "r2"]
    assert json.loads(row["target_ids"]) == ["t1"]
    assert row["selected_edges"] == "[]"
    assert row["match_type"] == "M:N"
    assert int(row["num_refs"]) == 2
    assert int(row["num_targets"]) == 1
    assert row["notes"] == "prior note"
    # Provenance is preserved: a progress save is not re-authorship. The
    # original labeler classification (panel vs human) and decision time must
    # survive for eval slicing.
    assert row["labeler"] == "panel_unanimous_v7"
    assert row["session_id"] == "s1"
    assert row["labeled_at"] == prior_labeled_at

    group = {
        "dataset_id": "ds",
        "group_id": "g1",
        "ref_ids": ["r1", "r2"],
        "target_ids": ["t1"],
        "edges": [_edge("r1", "t1"), _edge("r2", "t1")],
        "rejected_edges": [],
    }
    queue = get_pairwise_revisit_groups("__pairwise__", [group])
    assert [g["group_id"] for g in queue] == ["g1"]
    prior = queue[0]["prior_label"]
    assert prior["edge_dispositions"] == dispositions
    assert prior["covered_ref_ids"] == ["r1", "r2"]
    assert prior["notes"] == "prior note"

    # Completing the review (full universe under exact scope) leaves the queue.
    StitchingLabelStore("ds").add(
        "g1",
        [{"ref_id": "r1", "target_id": "t1"}],
        "M:N",
        2,
        1,
        "brad",
        "s2",
        adjudication_scope=ADJUDICATION_SCOPE_EXACT_IDENTITY,
        edge_dispositions=[
            {"ref_id": "r1", "target_id": "t1", "resolution": "keep", "identity": "match"},
            {"ref_id": "r2", "target_id": "t1", "resolution": "drop", "identity": "no_match"},
        ],
    )
    assert get_pairwise_revisit_groups("__pairwise__", [group]) == []


def test_partial_progress_refuses_overwriting_completed_adjudication(tmp_path, monkeypatch):
    """A progress save must never destroy a finished exact-identity label."""
    monkeypatch.chdir(tmp_path)
    completed = [
        {"ref_id": "r1", "target_id": "t1", "resolution": "keep", "identity": "match"},
        {"ref_id": "r2", "target_id": "t1", "resolution": "drop", "identity": "no_match"},
        {"ref_id": "r3", "target_id": "t1", "resolution": "drop", "identity": "no_match"},
    ]
    StitchingLabelStore("ds").add(
        "g1",
        [{"ref_id": "r1", "target_id": "t1"}],
        "M:N",
        3,
        1,
        "panel_unanimous_v7",
        "s1",
        adjudication_scope=ADJUDICATION_SCOPE_EXACT_IDENTITY,
        edge_dispositions=completed,
    )

    with pytest.raises(ValueError, match="completed exact-identity"):
        record_partial_identity_progress(
            "ds",
            "g1",
            [{"ref_id": "r1", "target_id": "t1", "resolution": "keep", "identity": "match"}],
        )

    row = StitchingLabelStore("ds").load("ds").iloc[0]
    assert row["adjudication_scope"] == ADJUDICATION_SCOPE_EXACT_IDENTITY
    assert json.loads(row["edge_dispositions"]) == completed
    assert row["labeler"] == "panel_unanimous_v7"


def test_partial_progress_requires_prior_row(tmp_path, monkeypatch):
    """No prior row means no legitimate caller: never mint a half-session label."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="no prior label row"):
        record_partial_identity_progress(
            "ds",
            "g1",
            [{"ref_id": "r1", "target_id": "t1", "resolution": "keep", "identity": "match"}],
        )

    assert StitchingLabelStore("ds").load("ds").empty


def test_partial_identity_row_is_stitch_metric_neutral(tmp_path, monkeypatch):
    """mbench stitch metrics are identical before and after a partial save."""
    evaluate_stitch_groups = _import_mbench_stitch_metrics()
    monkeypatch.chdir(tmp_path)
    StitchingLabelStore("ds").add(
        "g1",
        [{"ref_id": "r1", "target_id": "t1"}],
        "1:N",
        1,
        1,
        "panel_unanimous_v7",
        "s1",
        adjudication_scope="exact_resolution",
    )
    bridge = pd.DataFrame([{"ref_id": "r1", "target_id": "t1", "confidence": 0.9}])
    groups_sidecar = [
        {
            "group_id": "g1",
            "edges": [{"ref_id": "r1", "target_id": "t1"}],
            "rejected_edges": [{"ref_id": "r2", "target_id": "t1"}],
        }
    ]

    before = evaluate_stitch_groups(bridge, StitchingLabelStore("ds").load("ds"), groups_sidecar)

    record_partial_identity_progress(
        "ds",
        "g1",
        [{"ref_id": "r1", "target_id": "t1", "resolution": "keep", "identity": "match"}],
    )
    labels_after = StitchingLabelStore("ds").load("ds")
    assert labels_after.iloc[0]["adjudication_scope"] == ADJUDICATION_SCOPE_PARTIAL_IDENTITY

    after = evaluate_stitch_groups(bridge, labels_after, groups_sidecar)

    assert after == before


class TestPairwiseWizardRoutes:
    DATASET = "owner_ds"
    QUEUE = "__pairwise__"

    def _group(self):
        return {
            "dataset_id": self.DATASET,
            "group_id": "gpair",
            "match_type": "1:N",
            "ref_ids": ["r1"],
            "target_ids": ["t1"],
            "ref_geometries": {"r1": _line(0)},
            "target_geometries": {"t1": _line(1)},
            "ref_names": {"r1": "Reference"},
            "target_names": {"t1": "Target"},
            "ref_classes": {"r1": "primary"},
            "target_classes": {"t1": "primary"},
            "edges": [_edge("r1", "t1", 0.9)],
            "rejected_edges": [],
            "prior_label": {
                "covered_ref_ids": ["r1"],
                "covered_target_ids": ["t1"],
                "selected_edges": [{"ref_id": "r1", "target_id": "t1"}],
                "notes": "prior note",
            },
        }

    @staticmethod
    def _request(path: str, query: bytes = b"") -> Request:
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": path,
                "headers": [],
                "query_string": query,
                "server": ("test", 80),
                "client": ("test", 1),
                "scheme": "http",
                "root_path": "",
                "http_version": "1.1",
            }
        )
        request.state.labeler_name = "tester"
        request.state.dataset_label_counts = {}
        request.state.min_labels_per_dataset = 1
        request.state.dataset_stitching_counts = {}
        request.state.min_stitching_labels = 1
        return request

    def _patches(self):
        group = self._group()
        batch = {"dataset_id": self.QUEUE, "groups": [group]}
        patches = [
            patch(
                "crosswalk.web.routes.stitching.list_datasets",
                return_value=[self.DATASET],
            ),
            patch(
                "crosswalk.web.routes.stitching.load_stitch_batch",
                return_value=batch,
            ),
            patch(
                "crosswalk.web.routes.stitching.get_pairwise_revisit_groups",
                return_value=[group],
            ),
        ]
        for item in patches:
            item.start()
        return patches

    @staticmethod
    def _stop(patches):
        for item in patches:
            item.stop()

    def test_pairwise_queue_renders_dedicated_wizard_and_pair_map_payload(self):
        patches = self._patches()
        try:
            response = asyncio.run(
                stitching_review(
                    self._request("/stitching-review"),
                    dataset=self.QUEUE,
                )
            )
        finally:
            self._stop(patches)

        assert response.status_code == 200
        html = response.body.decode()
        assert 'id="pairwise-card"' in html
        assert 'id="pairwise-candidates"' in html
        assert 'id="group-card"' not in html
        assert "stitching-pairwise.js" in html
        assert "Reference" in html
        assert "prior note" in html

    def test_pair_features_rejects_non_candidate(self):
        patches = self._patches()
        try:
            response = asyncio.run(
                stitching_pair_features(
                    self._request("/stitching-review/pair-features"),
                    dataset=self.QUEUE,
                    group_id="gpair",
                    group_dataset=self.DATASET,
                    ref_id="forged",
                    target_id="t1",
                )
            )
        finally:
            self._stop(patches)

        assert response.status_code == 404

    def test_pair_features_falls_back_to_edge_details(self):
        patches = self._patches()
        try:
            with patch(
                "crosswalk.web.routes.stitching.candidates_sidecar_path",
                return_value=Path("does-not-exist.parquet"),
            ):
                response = asyncio.run(
                    stitching_pair_features(
                        self._request("/stitching-review/pair-features"),
                        dataset=self.QUEUE,
                        group_id="gpair",
                        group_dataset=self.DATASET,
                        ref_id="r1",
                        target_id="t1",
                    )
                )
        finally:
            self._stop(patches)

        assert response.status_code == 200
        html = response.body.decode()
        assert "confidence" in html
        assert "ref_coverage" in html

    def _two_candidate_group(self):
        group = self._group()
        group["target_ids"] = ["t1", "t2"]
        group["target_geometries"]["t2"] = _line(2)
        group["target_names"]["t2"] = "Second target"
        group["target_classes"]["t2"] = "primary"
        group["rejected_edges"] = [_edge("r1", "t2", 0.4)]
        return group

    def _select_kwargs(self, **overrides):
        kwargs = {
            "dataset": self.QUEUE,
            "group_id": "gpair",
            "group_dataset": self.DATASET,
            "group_index": 0,
            "included_refs": "",
            "included_targets": "",
            "selected_edges": "",
            "adjudication_scope": "exact_identity",
            "edge_dispositions": "",
            "exclude_slivers": "",
            "deanchored": False,
            "confirm_reject_all": "",
            "partial_save": "",
            "notes": "",
        }
        kwargs.update(overrides)
        return kwargs

    def test_partial_save_records_progress_and_reserves_same_group(self):
        group = self._two_candidate_group()
        batch = {"dataset_id": self.QUEUE, "groups": [group]}
        subset = [
            {"ref_id": "r1", "target_id": "t1", "resolution": "keep", "identity": "match"},
        ]
        calls = []
        with (
            patch(
                "crosswalk.web.routes.stitching.list_datasets",
                return_value=[self.DATASET],
            ),
            patch(
                "crosswalk.web.routes.stitching.load_stitch_batch",
                return_value=batch,
            ),
            patch(
                "crosswalk.web.routes.stitching.get_pairwise_revisit_groups",
                return_value=[group],
            ),
            patch(
                "crosswalk.web.routes.stitching.record_partial_identity_progress",
                side_effect=lambda **kw: calls.append(kw),
            ),
        ):
            response = asyncio.run(
                stitching_select(
                    self._request("/stitching-review/select"),
                    **self._select_kwargs(
                        edge_dispositions=json.dumps(subset),
                        partial_save="1",
                        notes="halfway",
                    ),
                )
            )

        assert response.status_code == 200
        assert calls == [
            {
                "dataset_id": self.DATASET,
                "group_id": "gpair",
                "dispositions": subset,
                "notes": "halfway",
            }
        ]
        html = response.body.decode()
        # The SAME group is re-served (partial saves never advance the queue).
        assert 'id="pairwise-card"' in html
        assert "gpair" in html

    def test_partial_save_rejected_outside_pairwise_queue(self):
        group = self._two_candidate_group()
        batch = {"dataset_id": self.DATASET, "groups": [group]}
        subset = [
            {"ref_id": "r1", "target_id": "t1", "resolution": "keep", "identity": "match"},
        ]
        calls = []
        with (
            patch(
                "crosswalk.web.routes.stitching.list_datasets",
                return_value=[self.DATASET],
            ),
            patch(
                "crosswalk.web.routes.stitching.load_stitch_batch",
                return_value=batch,
            ),
            patch(
                "crosswalk.web.routes.stitching.record_partial_identity_progress",
                side_effect=lambda **kw: calls.append(kw),
            ),
        ):
            response = asyncio.run(
                stitching_select(
                    self._request("/stitching-review/select"),
                    **self._select_kwargs(
                        dataset=self.DATASET,
                        edge_dispositions=json.dumps(subset),
                        partial_save="1",
                    ),
                )
            )

        assert response.status_code == 400
        assert calls == []

    def test_full_universe_partial_save_is_rejected(self):
        """A replayed full payload must go through the completion path."""
        group = self._two_candidate_group()
        batch = {"dataset_id": self.QUEUE, "groups": [group]}
        everything = [
            {"ref_id": "r1", "target_id": "t1", "resolution": "keep", "identity": "match"},
            {"ref_id": "r1", "target_id": "t2", "resolution": "drop", "identity": "no_match"},
        ]
        calls = []
        with (
            patch(
                "crosswalk.web.routes.stitching.list_datasets",
                return_value=[self.DATASET],
            ),
            patch(
                "crosswalk.web.routes.stitching.load_stitch_batch",
                return_value=batch,
            ),
            patch(
                "crosswalk.web.routes.stitching.record_partial_identity_progress",
                side_effect=lambda **kw: calls.append(kw),
            ),
        ):
            response = asyncio.run(
                stitching_select(
                    self._request("/stitching-review/select"),
                    **self._select_kwargs(
                        edge_dispositions=json.dumps(everything),
                        partial_save="1",
                    ),
                )
            )

        assert response.status_code == 400
        assert calls == []

    def test_partial_subset_without_flag_is_rejected(self):
        group = self._two_candidate_group()
        batch = {"dataset_id": self.QUEUE, "groups": [group]}
        subset = [
            {"ref_id": "r1", "target_id": "t1", "resolution": "keep", "identity": "match"},
        ]
        with (
            patch(
                "crosswalk.web.routes.stitching.list_datasets",
                return_value=[self.DATASET],
            ),
            patch(
                "crosswalk.web.routes.stitching.load_stitch_batch",
                return_value=batch,
            ),
        ):
            response = asyncio.run(
                stitching_select(
                    self._request("/stitching-review/select"),
                    **self._select_kwargs(edge_dispositions=json.dumps(subset)),
                )
            )

        assert response.status_code == 400
