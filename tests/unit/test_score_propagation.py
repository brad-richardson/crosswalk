"""Unit tests for structure-aware score propagation (EXPERIMENTAL).

Tests the propagation math on synthetic mini-networks:

* a consistent corridor gets its borderline pair boosted,
* a parallel-road trap gets its borderline pair dampened,
* logit-space adjustments respect the ``delta_cap`` bound,
* an isolated pair (no neighbors, no competitors) is left unchanged,
* the flag-off pipeline path never touches results (byte-identical).
"""

import math

import geopandas as gpd
import pytest
from shapely import LineString

from crosswalk.matching.score_propagation import (
    PropagationParams,
    _logit,
    propagate_scores,
)
from crosswalk.matching.types import MatchDecision, MatchResult


def _gdf(rows):
    """Build a projected (meters) GeoDataFrame from {id: LineString} rows."""
    ids = list(rows.keys())
    geoms = [rows[i] for i in ids]
    return gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs="EPSG:3857")


def _mr(ref_id, target_id, conf):
    # Mirror the scorer's decision bands via settings so a no-op propagation
    # leaves the decision unchanged even if the thresholds are reconfigured.
    from crosswalk.config import settings

    if conf >= settings.scoring_match_threshold:
        dec = MatchDecision.MATCH
    elif conf >= settings.scoring_review_threshold:
        dec = MatchDecision.REVIEW
    else:
        dec = MatchDecision.NO_MATCH
    return MatchResult(ref_id, target_id, dec, conf, {}, {})


def test_consistent_corridor_boosts_borderline_pair():
    # Collinear corridor sharing the junction at (100, 0).
    reference = _gdf(
        {
            "R1": LineString([(0, 0), (100, 0)]),
            "R2": LineString([(100, 0), (200, 0)]),
        }
    )
    target = _gdf(
        {
            "T1": LineString([(0, 0), (100, 0)]),
            "T2": LineString([(100, 0), (200, 0)]),
        }
    )
    # (R1,T1) is borderline; its topological continuation (R2,T2) is confident.
    results = [_mr("R1", "T1", 0.55), _mr("R2", "T2", 0.95)]

    params = PropagationParams(n_rounds=2, alpha=0.6, beta=0.6)
    out, stats = propagate_scores(results, reference, target, params=params)

    assert stats.n_consistent_edges >= 1
    assert out[0].confidence > 0.55  # borderline pair boosted by confident neighbor


def test_parallel_road_trap_dampens_borderline_pair():
    # A single target sits between two parallel reference roads. The wrong,
    # parallel decoy (R_decoy) is (mis)scored confidently; the correct road
    # (R_true) is borderline. They share the target but are non-adjacent on the
    # ref side -> competitors, so the borderline pair should be dampened.
    reference = _gdf(
        {
            "R_true": LineString([(0, 0), (100, 0)]),
            "R_decoy": LineString([(0, 60), (100, 60)]),
        }
    )
    target = _gdf({"T": LineString([(0, 30), (100, 30)])})
    results = [_mr("R_true", "T", 0.60), _mr("R_decoy", "T", 0.92)]

    params = PropagationParams(n_rounds=2, alpha=0.6, beta=0.6)
    out, stats = propagate_scores(results, reference, target, params=params)

    assert stats.n_competitor_edges >= 1
    assert out[0].confidence < 0.60  # borderline true pair dampened by confident decoy


def test_boost_only_ablation_disables_dampening():
    reference = _gdf(
        {
            "R_true": LineString([(0, 0), (100, 0)]),
            "R_decoy": LineString([(0, 60), (100, 60)]),
        }
    )
    target = _gdf({"T": LineString([(0, 30), (100, 30)])})
    results = [_mr("R_true", "T", 0.60), _mr("R_decoy", "T", 0.92)]

    params = PropagationParams(n_rounds=2, boost_only=True)
    out, stats = propagate_scores(results, reference, target, params=params)

    assert stats.n_competitor_edges == 0
    # No corner between these pairs either -> nothing to boost -> unchanged.
    assert out[0].confidence == pytest.approx(0.60)


def test_adjustment_respects_delta_cap():
    reference = _gdf(
        {
            "R1": LineString([(0, 0), (100, 0)]),
            "R2": LineString([(100, 0), (200, 0)]),
        }
    )
    target = _gdf(
        {
            "T1": LineString([(0, 0), (100, 0)]),
            "T2": LineString([(100, 0), (200, 0)]),
        }
    )
    results = [_mr("R1", "T1", 0.50), _mr("R2", "T2", 0.999)]

    cap = 1.0
    params = PropagationParams(n_rounds=25, alpha=5.0, beta=5.0, damping=1.0, delta_cap=cap)
    out, _ = propagate_scores(results, reference, target, params=params)

    drift = _logit(out[0].confidence) - math.log(0.50 / 0.50)  # logit(0.5) == 0
    assert abs(drift) <= cap + 1e-6


def test_isolated_pair_is_unchanged():
    reference = _gdf({"R1": LineString([(0, 0), (100, 0)])})
    target = _gdf({"T1": LineString([(0, 0), (100, 0)])})
    results = [_mr("R1", "T1", 0.42)]

    out, stats = propagate_scores(results, reference, target, params=PropagationParams())

    assert stats.n_consistent_edges == 0
    assert stats.n_competitor_edges == 0
    assert out[0].confidence == pytest.approx(0.42)
    assert out[0].decision == MatchDecision.REVIEW  # unchanged from input band


def test_decision_recomputed_after_boost():
    # Borderline REVIEW-band pair pushed above the match threshold flips decision.
    reference = _gdf(
        {
            "R1": LineString([(0, 0), (100, 0)]),
            "R2": LineString([(100, 0), (200, 0)]),
        }
    )
    target = _gdf(
        {
            "T1": LineString([(0, 0), (100, 0)]),
            "T2": LineString([(100, 0), (200, 0)]),
        }
    )
    r1 = MatchResult("R1", "T1", MatchDecision.REVIEW, 0.45, {}, {})
    r2 = MatchResult("R2", "T2", MatchDecision.MATCH, 0.98, {}, {})
    out, _ = propagate_scores(
        [r1, r2], reference, target, params=PropagationParams(alpha=1.0, n_rounds=3)
    )
    assert out[0].confidence > 0.5
    assert out[0].decision == MatchDecision.MATCH


def test_zero_junction_coincidence_rejected():
    reference = _gdf({"R1": LineString([(0, 0), (100, 0)])})
    target = _gdf({"T1": LineString([(0, 1), (100, 1)])})
    r = _mr("R1", "T1", 0.9)
    with pytest.raises(ValueError, match="junction_coincidence_m"):
        propagate_scores(
            [r], reference, target, params=PropagationParams(junction_coincidence_m=0.0)
        )


def test_pipeline_gates_propagation_on_flag(monkeypatch, tmp_path):
    """run_pipeline must invoke propagate_scores iff the settings flag is on."""
    from types import SimpleNamespace

    import crosswalk.pipeline.runner as runner
    from crosswalk.config import settings

    ref = _gdf({"R1": LineString([(0, 0), (100, 0)])})
    tgt = _gdf({"T1": LineString([(0, 1), (100, 1)])})
    ref_path = tmp_path / "ref.parquet"
    tgt_path = tmp_path / "tgt.parquet"
    ref.to_parquet(ref_path)
    tgt.to_parquet(tgt_path)

    fake_results = [_mr("R1", "T1", 0.9)]

    # run_pipeline reads projection_result.reference/.target after scoring
    def _fake_score(reference, target, **kwargs):
        proj = SimpleNamespace(reference=reference, target=target)
        return list(fake_results), proj

    # Stub the heavy stages around the gating point.
    monkeypatch.setattr(runner, "score_candidates_from_geodataframes", _fake_score)
    monkeypatch.setattr(runner, "optimize_matches_with_grouping", lambda results, *a, **k: results)
    monkeypatch.setattr(runner, "_export_groups_sidecar", lambda *a, **k: None)
    monkeypatch.setattr(runner, "generate_bridge_file", lambda *a, **k: None)
    monkeypatch.setattr(runner, "generate_unmatched_report", lambda *a, **k: None)

    calls = {"n": 0}

    def _fake_propagate(results, **kwargs):
        calls["n"] += 1
        return results, None

    # The runner lazy-imports propagate_scores inside the flag guard, so
    # patching the source module attribute intercepts the real call site.
    monkeypatch.setattr("crosswalk.matching.score_propagation.propagate_scores", _fake_propagate)

    assert settings.enable_score_propagation is False
    runner.run_pipeline(ref_path, tgt_path, tmp_path / "off_bridge.parquet")
    assert calls["n"] == 0, "flag off must not invoke propagation"

    monkeypatch.setattr(settings, "enable_score_propagation", True)
    runner.run_pipeline(ref_path, tgt_path, tmp_path / "on_bridge.parquet")
    assert calls["n"] == 1, "flag on must invoke propagation exactly once"
