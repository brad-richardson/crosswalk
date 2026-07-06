"""Naive geometric baseline adapter.

A deliberately simple buffer-overlap conflation baseline. It exists to give the
benchmark a *floor*: any real conflation tool (Hootenanny, crosswalk) should beat
pure geometry with no learning, no name matching, and no topology reasoning.

Algorithm (per target segment ``t``):

1. Reproject both datasets to a local metric CRS (UTM estimated from the
   target extent) so buffers and lengths are in meters.
2. Buffer ``t`` by ``buffer_m`` and query the reference spatial index for
   segments intersecting the buffer.
3. For each candidate reference ``r``:
   - Clip ``r`` to the target buffer and measure the clipped length. The
     *overlap fraction* is ``clipped_len / r.length`` — how much of the
     reference lies alongside the target.
   - Reject if the overlap fraction is below ``min_overlap``.
   - Compute the absolute bearing difference between the two segments'
     end-to-end orientation. Reject if it exceeds ``angle_tol_deg`` (treating
     opposing directions as aligned — roads are undirected here).
   - Score = ``overlap_fraction * cos(angle_diff)``.

   The overlap-within-buffer test and the bearing test carry all the
   discrimination here. An earlier version added a Hausdorff "shape sanity"
   guard, but it was removed: the intended directed check (clipped ref -> target)
   is a mathematical no-op — the clipped reference is by construction the part of
   the reference inside ``target.buffer(buffer_m)``, so its directed Hausdorff to
   the target never exceeds ``buffer_m`` — while shapely's ``hausdorff_distance``
   is *symmetric*, so the guard as shipped instead rejected any pair whose target
   extended more than ``buffer_m`` beyond the clipped reference. That is the
   coverage-asymmetry trap (a short reference legitimately covering part of a long
   local segment), and it silently cut naive recall by nearly half on roads
   (0.95 -> 0.54) and worsened the sidewalk collapse — so no shape guard is used.
4. Greedy assignment: sort all surviving (ref, target, score) triples by score
   descending and assign each *reference* to at most one target — its single
   best. Because Overture references are typically segmented much finer than
   local targets, this still yields the natural 1:N (many refs -> one target)
   shape while preventing a single reference from being claimed by several
   targets.

Everything here is pure geopandas/shapely — no crosswalk imports. mbench stays
crosswalk-free by design.
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely.geometry import LineString, MultiLineString

from mbench.adapters.base import EvalMode, ToolOutput

# Tuned conservatively for city road networks. Documented and overridable via
# --opt so runs stay reproducible and the thresholds are explicit in results.
DEFAULT_BUFFER_M = 15.0
DEFAULT_MIN_OVERLAP = 0.30
DEFAULT_ANGLE_TOL_DEG = 35.0


def _segment_bearing(geom) -> float | None:
    """End-to-end bearing of a line in degrees [0, 180).

    Uses the first and last coordinate only (naive by design). Direction is
    folded into [0, 180) so that a road and its reverse are treated as aligned.
    """
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiLineString):
        # Use the longest constituent part.
        parts = [g for g in geom.geoms if not g.is_empty]
        if not parts:
            return None
        geom = max(parts, key=lambda g: g.length)
    if not isinstance(geom, LineString):
        return None
    coords = list(geom.coords)
    if len(coords) < 2:
        return None
    x0, y0 = coords[0][0], coords[0][1]
    x1, y1 = coords[-1][0], coords[-1][1]
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
    return ang % 180.0


def _bearing_diff(a: float, b: float) -> float:
    """Absolute difference of two undirected bearings in degrees [0, 90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def compute_naive_matches(
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    buffer_m: float = DEFAULT_BUFFER_M,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    angle_tol_deg: float = DEFAULT_ANGLE_TOL_DEG,
    id_column: str = "id",
) -> pd.DataFrame:
    """Compute naive buffer-overlap matches between reference and target.

    Returns a DataFrame with columns ``[ref_id, target_id, confidence]``.
    """
    if id_column not in reference.columns or id_column not in target.columns:
        raise ValueError(f"Both datasets must have an '{id_column}' column")

    # Project to a local metric CRS so buffers/lengths are in meters.
    if target.crs is None:
        raise ValueError("Target GeoDataFrame has no CRS; cannot project to meters")
    metric_crs = target.estimate_utm_crs()
    ref = reference.to_crs(metric_crs)
    tgt = target.to_crs(metric_crs)
    logger.info(f"Projected to {metric_crs.to_epsg()} for metric buffering")

    ref_geoms = ref.geometry.values
    ref_ids = ref[id_column].astype(str).values
    ref_lengths = ref.geometry.length.values
    ref_bearings = [_segment_bearing(g) for g in ref_geoms]

    sindex = ref.sindex

    # Collect all surviving candidate triples, then greedily assign.
    candidates: list[tuple[float, str, str]] = []  # (score, ref_id, target_id)

    for tgt_geom, tgt_id in zip(tgt.geometry.values, tgt[id_column].astype(str).values):
        if tgt_geom is None or tgt_geom.is_empty:
            continue
        tgt_bearing = _segment_bearing(tgt_geom)
        buf = tgt_geom.buffer(buffer_m)

        for ri in sindex.query(buf, predicate="intersects"):
            r_geom = ref_geoms[ri]
            r_len = ref_lengths[ri]
            if r_geom is None or r_geom.is_empty or r_len <= 0:
                continue

            clipped = r_geom.intersection(buf)
            clipped_len = clipped.length
            if clipped_len <= 0:
                continue
            overlap_frac = min(clipped_len / r_len, 1.0)
            if overlap_frac < min_overlap:
                continue

            angle_factor = 1.0
            if tgt_bearing is not None and ref_bearings[ri] is not None:
                angle_diff = _bearing_diff(tgt_bearing, ref_bearings[ri])
                if angle_diff > angle_tol_deg:
                    continue
                angle_factor = math.cos(math.radians(angle_diff))

            score = overlap_frac * angle_factor
            candidates.append((score, ref_ids[ri], tgt_id))

    # Greedy 1:1-ish assignment: each reference goes to its single best target.
    candidates.sort(key=lambda c: c[0], reverse=True)
    ref_assigned: dict[str, tuple[str, float]] = {}
    for score, r_id, t_id in candidates:
        if r_id in ref_assigned:
            continue
        ref_assigned[r_id] = (t_id, score)

    rows = [(r_id, t_id, score) for r_id, (t_id, score) in ref_assigned.items()]
    matches = pd.DataFrame(rows, columns=["ref_id", "target_id", "confidence"])
    logger.info(
        f"Naive baseline produced {len(matches)} matches "
        f"({matches['target_id'].nunique()} distinct targets) "
        f"[buffer={buffer_m}m min_overlap={min_overlap} angle_tol={angle_tol_deg}deg]"
    )
    return matches


class NaiveAdapter:
    """Naive geometric buffer-overlap baseline.

    Produces the benchmark floor. Reads the reference/target GeoParquet
    directly (no external process), computes matches with
    :func:`compute_naive_matches`, and writes them to a parquet the same way
    other adapters emit their raw output.
    """

    name: str = "naive"
    eval_mode: EvalMode = EvalMode.STITCH

    def run(self, reference: Path, target: Path, output_dir: Path, **kwargs) -> Path:
        """Compute naive matches and write them to ``matches.parquet``.

        Args:
            reference: Path to reference (Overture) GeoParquet.
            target: Path to target GeoParquet.
            output_dir: Directory for the output parquet.
            **kwargs:
                buffer_m: Buffer radius in meters (default 15).
                min_overlap: Minimum reference-length overlap fraction (default 0.30).
                angle_tol_deg: Maximum undirected bearing difference (default 35).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = (output_dir / "matches.parquet").resolve()

        buffer_m = float(kwargs.get("buffer_m", DEFAULT_BUFFER_M))
        min_overlap = float(kwargs.get("min_overlap", DEFAULT_MIN_OVERLAP))
        angle_tol_deg = float(kwargs.get("angle_tol_deg", DEFAULT_ANGLE_TOL_DEG))

        logger.info(f"Loading reference {reference}")
        ref = gpd.read_parquet(reference)
        logger.info(f"Loading target {target}")
        tgt = gpd.read_parquet(target)

        matches = compute_naive_matches(
            ref,
            tgt,
            buffer_m=buffer_m,
            min_overlap=min_overlap,
            angle_tol_deg=angle_tol_deg,
        )
        matches.to_parquet(out_path)
        logger.success(f"Wrote naive matches to {out_path}")
        return out_path

    def parse_output(self, output_path: Path) -> ToolOutput:
        """Parse the matches parquet into a standardized ToolOutput."""
        matches = pd.read_parquet(output_path)
        matches = matches.astype({"ref_id": str, "target_id": str})
        metadata = {
            "total_matches": len(matches),
            "distinct_targets": int(matches["target_id"].nunique()) if len(matches) else 0,
        }
        return ToolOutput(matches=matches, metadata=metadata)
