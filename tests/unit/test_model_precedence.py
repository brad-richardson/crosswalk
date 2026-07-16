"""Production model-selection and provenance invariants."""

from __future__ import annotations

import hashlib

import geopandas as gpd
import pytest
from shapely import LineString

from crosswalk.config import MatcherSettings, bundled_model_path, settings
from crosswalk.matching.ml import select_model_for_dataset
from crosswalk.pipeline import runner
from crosswalk.utils.crs import ProjectionResult


def test_production_model_defaults_to_bundle_even_when_local_model_exists(tmp_path, monkeypatch):
    local_model = tmp_path / "matcher_model_combined.joblib"
    local_model.write_bytes(b"local experiment")
    monkeypatch.setattr(settings, "model_path", bundled_model_path())
    monkeypatch.setattr(settings, "local_model_path", local_model)

    assert runner._default_model_path() == bundled_model_path()
    assert runner._resolve_model_path() == bundled_model_path()


def test_matcher_model_path_environment_is_explicit_override(tmp_path, monkeypatch):
    local_model = tmp_path / "local.joblib"
    local_model.write_bytes(b"local experiment")
    monkeypatch.setenv("MATCHER_MODEL_PATH", str(local_model))

    configured = MatcherSettings(_env_file=None)

    assert configured.model_path == local_model


def test_advisory_auto_selection_remains_local_first(tmp_path, monkeypatch):
    local_model = tmp_path / "local.joblib"
    local_model.write_bytes(b"local experiment")
    monkeypatch.setattr(settings, "local_model_path", local_model)
    target = gpd.GeoDataFrame(
        {"names": ["Main Street"]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs="EPSG:3857",
    )

    assert select_model_for_dataset(target) == str(local_model)


def test_explicit_model_path_is_validated_and_hashed(tmp_path):
    local_model = tmp_path / "local.joblib"
    payload = b"explicit local model"
    local_model.write_bytes(payload)

    assert runner._resolve_model_path(local_model) == local_model
    assert runner._active_model_hash(local_model) == hashlib.sha256(payload).hexdigest()

    with pytest.raises(FileNotFoundError, match="Active ML model not found"):
        runner._resolve_model_path(tmp_path / "missing.joblib")


def test_calibration_dependent_thresholds_receive_the_resolved_model(tmp_path, monkeypatch):
    local_model = tmp_path / "local.joblib"
    local_model.write_bytes(b"explicit local model")
    seen = []

    def _active(model_path=None):
        seen.append(model_path)
        return True

    monkeypatch.setattr(runner, "_calibration_active", _active)
    monkeypatch.setattr(settings, "resolver_prune_enabled", True)
    monkeypatch.setattr(settings, "resolver_prune_overrides", {"ds": 0.91})

    assert runner._effective_glue_min_confidence(local_model) == pytest.approx(
        settings.optimizer_glue_min_confidence
    )
    assert runner._effective_prune_threshold("ds", local_model) == pytest.approx(0.91)
    assert seen == [local_model, local_model]


def test_run_pipeline_threads_one_model_path_through_score_and_export(tmp_path, monkeypatch):
    local_model = tmp_path / "local.joblib"
    local_model.write_bytes(b"explicit local model")
    frame = gpd.GeoDataFrame(
        {"id": ["x"]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs="EPSG:3857",
    )
    projection = ProjectionResult(frame, frame, frame.crs, frame.crs, False)
    captured = {}

    monkeypatch.setattr(runner, "load_and_filter_inputs", lambda *_, **__: (frame, frame))

    def _score(**kwargs):
        captured["score"] = kwargs["model_path"]
        return [], projection

    expected = object()

    def _optimize(**kwargs):
        captured["optimize"] = kwargs["model_path"]
        return expected

    monkeypatch.setattr(runner, "score_candidates_from_geodataframes", _score)
    monkeypatch.setattr(runner, "optimize_and_export", _optimize)

    result = runner.run_pipeline(
        tmp_path / "reference.parquet",
        tmp_path / "target.parquet",
        tmp_path / "bridge.parquet",
        model_path=local_model,
    )

    assert result is expected
    assert captured == {"score": local_model, "optimize": local_model}


def test_run_pipeline_threads_dataset_identity_for_physical_backfill(tmp_path, monkeypatch):
    """Behavior pin: dataset-name runs thread identity into input loading.

    Pins pre-existing behavior (this held before the ``load_kwargs`` ternary was
    made unconditional): whenever ``run_pipeline`` receives a dataset identity
    via ``prune_dataset_key``, it reaches ``load_and_filter_inputs`` as
    ``dataset_id`` so the target physical backfill can look up the dataset's
    FetchConfig. This is a regression guard for that plumbing, not a fix guard.
    """
    frame = gpd.GeoDataFrame(
        {"id": ["x"]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs="EPSG:3857",
    )
    projection = ProjectionResult(frame, frame, frame.crs, frame.crs, False)
    captured = {}

    def _load(reference_path, target_path, dataset_id=None):
        captured["dataset_id"] = dataset_id
        return frame, frame

    monkeypatch.setattr(runner, "load_and_filter_inputs", _load)
    monkeypatch.setattr(runner, "score_candidates_from_geodataframes", lambda **k: ([], projection))
    monkeypatch.setattr(runner, "optimize_and_export", lambda **k: object())

    runner.run_pipeline(
        tmp_path / "reference.parquet",
        tmp_path / "target.parquet",
        tmp_path / "bridge.parquet",
        prune_dataset_key="some_dataset",
    )

    assert captured["dataset_id"] == "some_dataset"
