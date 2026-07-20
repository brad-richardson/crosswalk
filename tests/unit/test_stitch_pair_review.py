"""Pairwise stitch-review geometry and web-flow tests."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import LineString, mapping
from starlette.requests import Request

from crosswalk.labeling.stitch_pair_review import (
    candidate_edge_union,
    enrich_candidate_endpoints,
    missing_candidate_endpoint_ids,
)
from crosswalk.web.routes.stitching import (
    _build_pairwise_candidates,
    stitching_pair_features,
    stitching_review,
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
