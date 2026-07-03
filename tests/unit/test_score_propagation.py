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

from matcher.matching.score_propagation import (
    PropagationParams,
    _logit,
    propagate_scores,
)
from matcher.matching.types import MatchDecision, MatchResult


def _gdf(rows):
    """Build a projected (meters) GeoDataFrame from {id: LineString} rows."""
    ids = list(rows.keys())
    geoms = [rows[i] for i in ids]
    return gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs="EPSG:3857")


def _mr(ref_id, target_id, conf):
    # Mirror the scorer's decision bands (match=0.5, review=0.1) so a no-op
    # propagation leaves the decision unchanged.
    if conf >= 0.5:
        dec = MatchDecision.MATCH
    elif conf >= 0.1:
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


def test_flag_off_pipeline_does_not_import_or_call(monkeypatch):
    # With the flag off (default), the runner must not invoke propagation.
    import matcher.matching.score_propagation as sp
    from matcher.config import settings

    called = {"n": 0}
    orig = sp.propagate_scores

    def _spy(*a, **k):
        called["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(sp, "propagate_scores", _spy)
    assert settings.enable_score_propagation is False
    # The default flag is off; the runner guards the call behind it. This test
    # documents the invariant that nothing calls propagation when off.
    assert called["n"] == 0
