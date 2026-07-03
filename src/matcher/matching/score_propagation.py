"""Structure-aware score propagation (EXPERIMENTAL).

A post-scoring, pre-optimizer step that adjusts per-pair match confidences using
the topological structure of the two road networks. The pairwise XGBoost scorer is
locally blind: whether local segment ``T`` matches reference segment ``R`` depends
on whether ``T``'s topological neighbors match ``R``'s neighbors, yet all such
global reasoning currently lives in hand-tuned optimizer heuristics.

Score propagation is the label-free way to inject that structure:

* **Boost** a candidate pair ``(R, T)`` when a *consistent neighbor* pair
  ``(R', T')`` — where ``R'`` continues from ``R`` and ``T'`` continues from ``T``
  through the *same physical corner* — is a confident match. Confident, structurally
  consistent corridors reinforce each other.
* **Dampen** a candidate pair when a *competitor* pair (same target ``T`` claimed by
  a different, non-adjacent reference ``R''``, or vice versa) is a confident match.
  Two roads that are not topological continuations but fight over the same segment
  are a classic parallel-road trap.

The math runs in logit space and every adjustment is bounded (``delta_cap``) so the
scorer's calibrated-ish ordering is perturbed, not destroyed. Iterated over a few
damped rounds so support flows a couple of hops along corridors.

This module is gated behind ``settings.enable_score_propagation`` (default False).
With the flag off it is never called and pipeline output is byte-identical.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import shapely
from loguru import logger

from ..config import settings
from .types import MatchDecision, MatchResult


@dataclass
class PropagationParams:
    """Tunable knobs for score propagation.

    Defaults are read from ``settings`` but can be overridden for ablation/tests.
    """

    n_rounds: int = 2
    alpha: float = 0.6  # boost strength (logit units at full neighbor agreement)
    beta: float = 0.6  # dampen strength (logit units at full competitor confidence)
    damping: float = 0.5  # per-round contraction of the propagated signal
    delta_cap: float = 1.5  # max |logit drift| from the original score
    junction_coincidence_m: float = 20.0  # grid size for corner coincidence
    boost_only: bool = False  # ablation: disable the dampen term

    @classmethod
    def from_settings(cls) -> PropagationParams:
        return cls(
            n_rounds=settings.score_propagation_rounds,
            alpha=settings.score_propagation_alpha,
            beta=settings.score_propagation_beta,
            damping=settings.score_propagation_damping,
            delta_cap=settings.score_propagation_delta_cap,
            junction_coincidence_m=settings.score_propagation_junction_m,
            boost_only=settings.score_propagation_boost_only,
        )


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _endpoint_coords(geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (start_xy, end_xy) arrays of shape (N, 2) for LineString geoms.

    Uses vectorized shapely so it stays fast on 100K+ geometries.
    """
    start = shapely.get_point(geoms, 0)
    end = shapely.get_point(geoms, -1)
    start_xy = shapely.get_coordinates(start)
    end_xy = shapely.get_coordinates(end)
    return start_xy, end_xy


def _snap_cell(xy: np.ndarray, g: float) -> np.ndarray:
    """Snap coordinates to a grid of size ``g`` (round-to-nearest centering).

    Returns an (N, 2) int array of grid-cell indices. Two points within ~g of
    each other land in the same cell.
    """
    return np.round(xy / g).astype(np.int64)


@dataclass
class PropagationStats:
    """Diagnostics returned alongside the adjusted results."""

    n_pairs: int
    n_consistent_edges: int
    n_competitor_edges: int
    n_boosted: int
    n_dampened: int
    n_decisions_changed: int
    mean_abs_delta_logit: float
    max_abs_delta_logit: float
    seconds: float


def propagate_scores(
    results: list[MatchResult],
    reference,
    target,
    ref_id_column: str = "id",
    target_id_column: str = "id",
    params: PropagationParams | None = None,
) -> tuple[list[MatchResult], PropagationStats]:
    """Adjust per-pair confidences using network structure (in place).

    Args:
        results: MatchResult objects from the scorer (mutated in place).
        reference: Projected reference GeoDataFrame (meters) with id + geometry.
        target: Projected target GeoDataFrame (meters) with id + geometry.
        ref_id_column: ID column in ``reference``.
        target_id_column: ID column in ``target``.
        params: Optional override; defaults to values from ``settings``.

    Returns:
        (results, stats). ``results`` is the same list, with ``confidence`` and
        ``decision`` updated for pairs whose score moved.
    """
    t0 = time.perf_counter()
    if params is None:
        params = PropagationParams.from_settings()

    n = len(results)
    if n == 0:
        return results, PropagationStats(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)

    g = params.junction_coincidence_m
    if not g > 0:
        raise ValueError(f"junction_coincidence_m must be > 0, got {g!r}")

    # --- Geometry lookups by id ---------------------------------------------
    ref_id_arr = reference[ref_id_column].astype(str).to_numpy()
    tgt_id_arr = target[target_id_column].astype(str).to_numpy()
    ref_geoms = reference.geometry.to_numpy()
    tgt_geoms = target.geometry.to_numpy()

    ref_start, ref_end = _endpoint_coords(ref_geoms)
    tgt_start, tgt_end = _endpoint_coords(tgt_geoms)
    ref_start_c = _snap_cell(ref_start, g)
    ref_end_c = _snap_cell(ref_end, g)
    tgt_start_c = _snap_cell(tgt_start, g)
    tgt_end_c = _snap_cell(tgt_end, g)

    ref_row = {rid: i for i, rid in enumerate(ref_id_arr)}
    tgt_row = {tid: i for i, tid in enumerate(tgt_id_arr)}

    # --- Per-pair structural fingerprints -----------------------------------
    # For each candidate pair collect: the ref segment row, target segment row,
    # the two ref junction cells, the two target junction cells, and the set of
    # "corner cells" where a ref endpoint and a target endpoint coincide (the
    # physical intersection where both continue on the same side).
    ref_rows = np.full(n, -1, dtype=np.int64)
    tgt_rows = np.full(n, -1, dtype=np.int64)
    conf = np.array([r.confidence for r in results], dtype=np.float64)

    corner_pairs: dict[tuple, list[int]] = defaultdict(list)
    same_target: dict[int, list[int]] = defaultdict(list)  # tgt_row -> pair idxs
    same_ref: dict[int, list[int]] = defaultdict(list)  # ref_row -> pair idxs
    pair_ref_cells: list[frozenset] = [frozenset()] * n
    pair_tgt_cells: list[frozenset] = [frozenset()] * n

    n_missing = 0
    for p, res in enumerate(results):
        ri = ref_row.get(str(res.ref_id), -1)
        ti = tgt_row.get(str(res.target_id), -1)
        if ri < 0 or ti < 0:
            n_missing += 1
            continue
        ref_rows[p] = ri
        tgt_rows[p] = ti

        r_cells = {tuple(ref_start_c[ri]), tuple(ref_end_c[ri])}
        t_cells = {tuple(tgt_start_c[ti]), tuple(tgt_end_c[ti])}
        pair_ref_cells[p] = frozenset(r_cells)
        pair_tgt_cells[p] = frozenset(t_cells)

        # A corner is a cell where a ref endpoint and a target endpoint coincide.
        for c in r_cells & t_cells:
            corner_pairs[c].append(p)

        same_target[ti].append(p)
        same_ref[ri].append(p)

    # --- Consistent-neighbor edges (boost) ----------------------------------
    # Two pairs sharing a corner cell, with different ref AND different target
    # segments, are structurally consistent continuations through that corner.
    edge_src: list[int] = []
    edge_dst: list[int] = []
    for _cell, plist in corner_pairs.items():
        if len(plist) < 2:
            continue
        for i in range(len(plist)):
            pi = plist[i]
            for j in range(i + 1, len(plist)):
                pj = plist[j]
                if ref_rows[pi] == ref_rows[pj] or tgt_rows[pi] == tgt_rows[pj]:
                    continue  # same segment on one side -> not a neighbor
                edge_src.append(pi)
                edge_dst.append(pj)
                edge_src.append(pj)
                edge_dst.append(pi)

    # --- Competitor edges (dampen) ------------------------------------------
    # Same target claimed by two non-adjacent refs (or same ref, two non-adjacent
    # targets): a genuine alternative, not a 1:N continuation.
    comp_src: list[int] = []
    comp_dst: list[int] = []

    def _add_competitors(groups, is_ref_group: bool):
        for _key, plist in groups.items():
            if len(plist) < 2:
                continue
            for i in range(len(plist)):
                pi = plist[i]
                for j in range(i + 1, len(plist)):
                    pj = plist[j]
                    # Competitors share one side (same ref or same target). They
                    # compete only if their *other* sides are NOT adjacent (do not
                    # share a junction cell) -> truly different roads.
                    if is_ref_group:
                        other_i, other_j = pair_tgt_cells[pi], pair_tgt_cells[pj]
                    else:
                        other_i, other_j = pair_ref_cells[pi], pair_ref_cells[pj]
                    if other_i & other_j:
                        continue  # adjacent on the other side -> a continuation
                    comp_src.append(pi)
                    comp_dst.append(pj)
                    comp_src.append(pj)
                    comp_dst.append(pi)

    if not params.boost_only:
        _add_competitors(same_ref, is_ref_group=True)
        _add_competitors(same_target, is_ref_group=False)

    edge_src_a = np.asarray(edge_src, dtype=np.int64)
    edge_dst_a = np.asarray(edge_dst, dtype=np.int64)
    comp_src_a = np.asarray(comp_src, dtype=np.int64)
    comp_dst_a = np.asarray(comp_dst, dtype=np.int64)

    # Precompute per-node neighbor counts for the boost mean.
    boost_deg = np.zeros(n, dtype=np.float64)
    if edge_src_a.size:
        np.add.at(boost_deg, edge_src_a, 1.0)
    has_boost = boost_deg > 0

    # --- Iterative damped propagation in logit space ------------------------
    l0 = _logit(conf)
    logit = l0.copy()
    cap = params.delta_cap

    for _round in range(params.n_rounds):
        s = _sigmoid(logit)
        agree = 2.0 * s - 1.0  # in [-1, 1]; positive == confident match

        step = np.zeros(n, dtype=np.float64)
        if edge_src_a.size:
            summed = np.zeros(n, dtype=np.float64)
            np.add.at(summed, edge_src_a, agree[edge_dst_a])
            mean_agree = np.zeros(n, dtype=np.float64)
            mean_agree[has_boost] = summed[has_boost] / boost_deg[has_boost]
            step += params.alpha * mean_agree

        if not params.boost_only and comp_src_a.size:
            comp_pos = np.maximum(agree[comp_dst_a], 0.0)
            max_comp = np.zeros(n, dtype=np.float64)
            np.maximum.at(max_comp, comp_src_a, comp_pos)
            step -= params.beta * max_comp

        # Accumulate this round's damped contribution, bounded relative to l0.
        drift = (logit - l0) + params.damping * step
        drift = np.clip(drift, -cap, cap)
        logit = l0 + drift

    new_conf = _sigmoid(logit)
    delta_logit = logit - l0

    # --- Write back confidence + recompute decisions ------------------------
    match_thr = settings.scoring_match_threshold
    review_thr = settings.scoring_review_threshold
    n_changed = 0
    n_boosted = 0
    n_dampened = 0
    for p, res in enumerate(results):
        if ref_rows[p] < 0:
            continue  # geometry missing -> leave untouched
        if delta_logit[p] > 1e-9:
            n_boosted += 1
        elif delta_logit[p] < -1e-9:
            n_dampened += 1
        res.confidence = float(new_conf[p])
        if res.confidence >= match_thr:
            new_dec = MatchDecision.MATCH
        elif res.confidence >= review_thr:
            new_dec = MatchDecision.REVIEW
        else:
            new_dec = MatchDecision.NO_MATCH
        if new_dec != res.decision:
            n_changed += 1
            res.decision = new_dec

    abs_delta = np.abs(delta_logit[ref_rows >= 0])
    stats = PropagationStats(
        n_pairs=n,
        n_consistent_edges=int(edge_src_a.size // 2),
        n_competitor_edges=int(comp_src_a.size // 2),
        n_boosted=n_boosted,
        n_dampened=n_dampened,
        n_decisions_changed=n_changed,
        mean_abs_delta_logit=float(abs_delta.mean()) if abs_delta.size else 0.0,
        max_abs_delta_logit=float(abs_delta.max()) if abs_delta.size else 0.0,
        seconds=time.perf_counter() - t0,
    )

    if n_missing:
        logger.warning(
            f"Score propagation: {n_missing} pairs had ids not found in geometry "
            "lookups; left unchanged."
        )
    logger.info(
        f"Score propagation: {stats.n_pairs:,} pairs, "
        f"{stats.n_consistent_edges:,} consistent edges, "
        f"{stats.n_competitor_edges:,} competitor edges, "
        f"boosted={stats.n_boosted:,} dampened={stats.n_dampened:,}, "
        f"decisions changed={stats.n_decisions_changed:,}, "
        f"mean|delta_logit|={stats.mean_abs_delta_logit:.3f} "
        f"max|delta_logit|={stats.max_abs_delta_logit:.3f} "
        f"in {stats.seconds:.2f}s"
    )
    return results, stats
