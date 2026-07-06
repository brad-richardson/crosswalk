"""Score once, then ablate score-propagation params by re-running only the
post-scoring stages (propagate -> optimize -> bridge). Writes one bridge per
config so they can be evaluated together with research/eval_bridges.py.

Usage:
    python research/ablate.py <ref.parquet> <target.parquet> <out_dir>
"""

import copy
import sys
from pathlib import Path

from loguru import logger

from crosswalk.config import DEFAULT_SNAP_TOLERANCE_M, settings
from crosswalk.matching import optimize_matches_with_grouping
from crosswalk.matching.score_propagation import PropagationParams, propagate_scores
from crosswalk.pipeline.runner import score_candidates_from_geodataframes
from crosswalk.resolution import generate_bridge_file
from crosswalk.utils import ensure_projected_crs

CONFIGS = {
    "off": None,
    "default_r2": PropagationParams(n_rounds=2, alpha=0.6, beta=0.6),
    "boost_only_r2": PropagationParams(n_rounds=2, alpha=0.6, beta=0.6, boost_only=True),
    "rounds1": PropagationParams(n_rounds=1, alpha=0.6, beta=0.6),
    "rounds3": PropagationParams(n_rounds=3, alpha=0.6, beta=0.6),
    "lowbeta_r2": PropagationParams(n_rounds=2, alpha=0.6, beta=0.3),
    "gentle_r2": PropagationParams(n_rounds=2, alpha=0.4, beta=0.4, delta_cap=1.0),
}


def main():
    ref_path, tgt_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    import geopandas as gpd

    from crosswalk.utils.geometry import filter_to_linestrings

    reference = filter_to_linestrings(gpd.read_parquet(ref_path), source_name="reference")
    target = filter_to_linestrings(gpd.read_parquet(tgt_path), source_name="target")

    logger.info("Scoring candidates once...")
    results, _proj = score_candidates_from_geodataframes(
        reference=reference, target=target, ref_id_column="id", target_id_column="id"
    )
    proj = ensure_projected_crs(reference, target)
    reference_p, target_p = proj.reference, proj.target
    logger.info(f"Scored {len(results)} candidates; running {len(CONFIGS)} configs")

    for name, params in CONFIGS.items():
        res = copy.deepcopy(results)
        if params is not None:
            res, stats = propagate_scores(res, reference_p, target_p, params=params)
            logger.info(f"[{name}] {stats}")
        optimized = optimize_matches_with_grouping(
            res,
            reference=reference_p,
            target=target_p,
            min_confidence=0.1,
            contiguity_tolerance=DEFAULT_SNAP_TOLERANCE_M,
            ref_id_column="id",
            target_id_column="id",
        )
        bridge_path = out_dir / f"ablate_{name}.parquet"
        generate_bridge_file(
            matches=optimized,
            output_path=bridge_path,
            match_method="xgboost",
            bridge_min_confidence=settings.bridge_min_confidence,
        )
        logger.info(f"[{name}] wrote {bridge_path}")


if __name__ == "__main__":
    main()
