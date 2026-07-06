"""Valhalla Meili map-matching baseline adapter.

The one *modern, actively maintained* external baseline in mbench (Hootenanny is a
frozen 2018 image; naive is the floor). It pilots the **path-based formulation**
bet from ``docs/EVAL_ROADMAP.md`` (§Architecture assessment #3): instead of
scoring candidate segment pairs, treat each local segment as a synthetic GPS
trace and *map-match* it onto an Overture-derived routable graph. The matched
edge sequence **is** the segment-correspondence set — which handles segmentation
mismatch natively (a single long local segment snaps across many short Overture
segments, and vice-versa) with no candidate generation or pairwise scoring.

Pipeline:

1. **Overture -> routable graph** (``convert/pbf.py``): the Overture segments
   parquet becomes an OSM PBF where each segment is a way carrying its GERS id as
   the OSM ``way_id`` (via a JSON sidecar), and shared connector coordinates
   collapse to shared nodes so the graph is routable. Tiles are built once per
   reference file and cached (keyed by file identity) so repeat runs are cheap.
2. **Build + match** with Valhalla's own engine via the ``pyvalhalla`` binding
   (``valhalla_build_tiles`` + ``valhalla.Actor``). This is the *same* Valhalla /
   Meili engine as the Docker image, run **in-process and ARM-native** — so the
   wall time is a *valid* datapoint (unlike the x86-emulated Hootenanny row).
3. **Match** each local target segment: densify its vertices to a trace, call
   ``Actor.trace_attributes`` with ``shape_match=map_snap``, read the matched
   ``edges[].way_id`` back to GERS ids, and aggregate per target with an
   overlap-length-weighted filter to drop spuriously-touched edges.

Runtime backend note: the design target was the maintained multi-arch Valhalla
*Docker* image, but on this machine both Docker routes are blocked — the ARM
image on ghcr.io stalls on blob download (registry CDN), and the amd64 image on
Docker Hub segfaults under qemu emulation during ``valhalla_build_tiles``. The
``pyvalhalla`` wheel ships the identical native-ARM Valhalla binaries, so it is
the working ARM-native path and gives valid timing. See ``docs/BENCHMARKING.md``.

Direction handling: map-matching a local segment is direction-agnostic (a road /
sidewalk and its reverse are the same physical feature), so we default to
``pedestrian`` costing — it is bidirectional (ignores ``oneway``, the documented
Meili failure mode) and traverses every local road class (residential .. footway),
which covers both the road and sidewalk datasets. Override with ``--opt
costing=auto`` for a directional, roads-only comparison.

No matcher imports — mbench stays crosswalk-free.
"""

from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from mbench.adapters.base import EvalMode, ToolOutput
from mbench.adapters.mapmatch_common import aggregate_edges, densify_lonlat
from mbench.convert.pbf import convert_overture_to_pbf

DEFAULT_COSTING = "pedestrian"
DEFAULT_DENSIFY_M = 10.0  # trace point spacing (Meili expects trace-like density)
DEFAULT_SEARCH_RADIUS_M = 25.0
DEFAULT_GPS_ACCURACY_M = 10.0
# An Overture way is kept as a match for a target only if the matched edge length
# attributed to it is at least this fraction of the target length OR this many
# meters — drops spuriously-touched edges (e.g. a snap that clips one node of a
# crossing street) without penalizing legitimate short slivers.
DEFAULT_MIN_MATCH_FRAC = 0.10
DEFAULT_MIN_MATCH_M = 8.0
DEFAULT_WORKERS = 8


# ---------------------------------------------------------------------------
# Graph cache + build
# ---------------------------------------------------------------------------


def _graph_cache_dir(base: Path, reference: Path) -> Path:
    """Deterministic per-reference cache dir (keyed by name + size + mtime).

    Reusing a built tileset across runs avoids repaying the (dominant) tile-build
    cost. The key changes if the reference file changes, forcing a rebuild.
    """
    st = reference.stat()
    key = f"{reference.stem}_{st.st_size}_{int(st.st_mtime)}"
    return (base / key).resolve()


def _native_bin_dir() -> Path:
    """Directory holding pyvalhalla's bundled native (ARM) Valhalla executables."""
    import valhalla

    return Path(valhalla.PYVALHALLA_DIR) / "bin"


def _valhalla_config(tiles_dir: Path) -> dict:
    """Build a Valhalla config dict serving from ``tiles_dir``.

    Built directly from pyvalhalla's default-config template (dropping ``Optional``
    placeholders) rather than ``valhalla.get_config`` — the latter has a bug where
    it strips the ``logging`` block during sanitization and then tries to set it,
    raising ``KeyError``.
    """
    import copy

    from valhalla.valhalla_build_config import Optional
    from valhalla.valhalla_build_config import config as default_config

    def _drop_optional(d: dict) -> dict:
        for k in list(d):
            v = d[k]
            if isinstance(v, Optional):
                del d[k]
            elif isinstance(v, dict):
                _drop_optional(v)
        return d

    cfg = _drop_optional(copy.deepcopy(default_config))
    cfg["mjolnir"]["tile_dir"] = str(tiles_dir)
    cfg["mjolnir"]["tile_extract"] = ""
    return cfg


def build_valhalla_graph(pbf_path: Path, cache_dir: Path, rebuild: bool = False) -> Path:
    """Build (or reuse) a Valhalla tileset from ``pbf_path`` and return the config path.

    The config is produced by ``valhalla.get_config`` (the ``valhalla_build_config``
    CLI is Python-side, not a native binary); tiles are built by the bundled native
    ``valhalla_build_tiles`` executable (invoked directly, bypassing the pyvalhalla
    PATH-based wrapper). The built tileset is cached under ``cache_dir/tiles``.
    """
    bin_dir = _native_bin_dir()
    tiles_dir = cache_dir / "tiles"
    config_path = cache_dir / "valhalla.json"

    have_tiles = tiles_dir.exists() and any(tiles_dir.rglob("*.gph"))
    if have_tiles and not rebuild and config_path.exists():
        logger.info(f"Reusing cached Valhalla tiles in {tiles_dir}")
        return config_path

    tiles_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Building Valhalla config...")
    config_path.write_text(json.dumps(_valhalla_config(tiles_dir)))

    logger.info("Building Valhalla tiles from PBF (this is the dominant cost)...")
    r = subprocess.run(
        [str(bin_dir / "valhalla_build_tiles"), "-c", str(config_path), str(pbf_path)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"valhalla_build_tiles failed: {r.stderr[-2000:]}")
    n_tiles = sum(1 for _ in tiles_dir.rglob("*.gph"))
    logger.success(f"Built Valhalla tileset ({n_tiles} tiles) in {tiles_dir}")
    return config_path


# ---------------------------------------------------------------------------
# Trace building + matching
# ---------------------------------------------------------------------------


# Trace densification is shared with the GraphHopper baseline (identical
# formulation); re-exported under the private name the Meili unit tests import.
_densify_lonlat = densify_lonlat


def _trace_request_payload(
    shape: list[tuple[float, float]], costing: str, search_radius: float, gps_acc: float
) -> dict:
    """Build the Valhalla ``trace_attributes`` request for one densified trace."""
    return {
        "shape": [{"lon": lon, "lat": lat} for lon, lat in shape],
        "costing": costing,
        "shape_match": "map_snap",
        "search_radius": search_radius,
        "gps_accuracy": gps_acc,
        "trace_options": {"turn_penalty_factor": 0},
        # Pin distance units explicitly: _aggregate_edges converts edge.length
        # km -> m assuming kilometers. Valhalla defaults to km today, but leaving
        # it implicit means a default change would silently corrupt the km->m
        # conversion (and thus every overlap fraction). Keep this in sync.
        "units": "kilometers",
        "filters": {"attributes": ["edge.way_id", "edge.length"], "action": "include"},
    }


def _aggregate_edges(
    resp: dict, id_map: dict[str, str], target_len_m: float, min_frac: float, min_m: float
) -> list[tuple[str, float]]:
    """Aggregate a Valhalla ``trace_attributes`` response into (gers_id, confidence).

    Thin Valhalla-specific wrapper over ``mapmatch_common.aggregate_edges``: it
    pulls ``(way_id, edge.length)`` from the response and converts Valhalla's edge
    length km -> m (matching the ``units=kilometers`` pinned in the request), then
    delegates the summing + overlap filter to the shared aggregator.
    """
    if not resp:
        return []
    edges = (
        (e.get("way_id"), float(e.get("length", 0.0)) * 1000.0)  # km -> m
        for e in resp.get("edges", [])
    )
    return aggregate_edges(edges, id_map, target_len_m, min_frac, min_m)


def match_targets(
    target: gpd.GeoDataFrame,
    id_map: dict[str, str],
    config_path: Path,
    costing: str,
    densify_m: float,
    search_radius: float,
    gps_acc: float,
    min_frac: float,
    min_m: float,
    workers: int,
    id_column: str = "id",
) -> pd.DataFrame:
    """Map-match every target segment and return [ref_id, target_id, confidence].

    Each worker thread gets its own ``valhalla.Actor`` (the actor is not safe for
    concurrent calls, but multiple actors share the read-only tileset on disk).
    """
    import valhalla

    # The Valhalla trace shape must be lon/lat (EPSG:4326), and _densify_lonlat
    # assumes its input geometry is already 4326. Overture parquets are 4326, but
    # a target in another CRS would otherwise be fed to Valhalla as raw
    # projected coordinates. Reproject up front, mirroring convert/pbf.py.
    if target.crs is None:
        raise ValueError("Target GeoDataFrame has no CRS; cannot map-match")
    if target.crs.to_epsg() != 4326:
        logger.info(f"Reprojecting target from {target.crs.to_epsg()} to EPSG:4326 for matching")
        target = target.to_crs(epsg=4326)

    metric_crs = target.estimate_utm_crs()
    geoms = target.geometry.values
    tgt_ids = target[id_column].astype(str).values
    lengths_m = target.to_crs(metric_crs).geometry.length.values

    cfg = str(config_path)
    _local = threading.local()

    def _actor():
        a = getattr(_local, "actor", None)
        if a is None:
            a = valhalla.Actor(cfg)
            _local.actor = a
        return a

    def _one(i: int):
        shape = _densify_lonlat(geoms[i], metric_crs, densify_m)
        if len(shape) < 2:
            return []
        payload = _trace_request_payload(shape, costing, search_radius, gps_acc)
        try:
            resp = json.loads(_actor().trace_attributes(json.dumps(payload)))
        except Exception:
            # Valhalla raises when map_snap cannot snap the trace (unmatched
            # local segment) — a legitimate "no match" outcome, not an error.
            return []
        pairs = _aggregate_edges(resp, id_map, float(lengths_m[i]), min_frac, min_m)
        return [(gers, tgt_ids[i], conf) for gers, conf in pairs]

    rows: list[tuple[str, str, float]] = []
    n = len(geoms)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, res in enumerate(pool.map(_one, range(n))):
            rows.extend(res)
            if (k + 1) % 5000 == 0:
                logger.info(f"  matched {k + 1}/{n} targets ({len(rows)} pairs so far)")

    matches = pd.DataFrame(rows, columns=["ref_id", "target_id", "confidence"])
    logger.info(
        f"Meili produced {len(matches)} match pairs "
        f"({matches['target_id'].nunique() if len(matches) else 0} distinct targets matched "
        f"of {n} targets)"
    )
    return matches


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class MeiliAdapter:
    """Valhalla Meili map-matching baseline (in-process via pyvalhalla)."""

    name: str = "meili"
    eval_mode: EvalMode = EvalMode.STITCH

    def run(self, reference: Path, target: Path, output_dir: Path, **kwargs) -> Path:
        """Build the Overture graph, then map-match every target with Valhalla.

        **kwargs (all via ``--opt key=value``):
            costing: Valhalla costing model (default ``pedestrian``; ``auto`` for
                directional roads-only).
            densify_m / search_radius / gps_accuracy: trace + snap params.
            min_match_frac / min_match_m: overlap filter thresholds.
            workers: match concurrency (default 8; one Actor per thread).
            graph_cache_dir: where the built tileset lives (default under
                output_dir; always OUTSIDE the crosswalk data/output tree).
            rebuild: force a graph rebuild even if a cached tileset exists.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = (output_dir / "matches.parquet").resolve()

        costing = kwargs.get("costing", DEFAULT_COSTING)
        densify_m = float(kwargs.get("densify_m", DEFAULT_DENSIFY_M))
        search_radius = float(kwargs.get("search_radius", DEFAULT_SEARCH_RADIUS_M))
        gps_acc = float(kwargs.get("gps_accuracy", DEFAULT_GPS_ACCURACY_M))
        min_frac = float(kwargs.get("min_match_frac", DEFAULT_MIN_MATCH_FRAC))
        min_m = float(kwargs.get("min_match_m", DEFAULT_MIN_MATCH_M))
        workers = int(kwargs.get("workers", DEFAULT_WORKERS))
        rebuild = str(kwargs.get("rebuild", "")).lower() in ("1", "true", "yes")

        cache_base = Path(kwargs.get("graph_cache_dir", output_dir / "graph_cache"))
        cache_dir = _graph_cache_dir(cache_base, reference)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pbf_path = cache_dir / "graph.osm.pbf"
        id_map_path = cache_dir / "id_map.json"

        # 1. Overture -> PBF (+ id-map), cached.
        if pbf_path.exists() and id_map_path.exists() and not rebuild:
            logger.info(f"Reusing cached PBF/id-map in {cache_dir}")
        else:
            convert_overture_to_pbf(reference, pbf_path, id_map_path)
        id_map = json.loads(id_map_path.read_text())

        # 2. Build the Valhalla tileset (cached).
        config_path = build_valhalla_graph(pbf_path, cache_dir, rebuild=rebuild)

        # 3. Match.
        logger.info(f"Loading target {target}")
        tgt = gpd.read_parquet(target)
        logger.info(
            f"Map-matching {len(tgt)} target segments "
            f"(costing={costing}, densify={densify_m}m, search_radius={search_radius}m, "
            f"workers={workers})"
        )
        matches = match_targets(
            tgt,
            id_map,
            config_path=config_path,
            costing=costing,
            densify_m=densify_m,
            search_radius=search_radius,
            gps_acc=gps_acc,
            min_frac=min_frac,
            min_m=min_m,
            workers=workers,
        )

        matches.to_parquet(out_path)
        meta = {
            "backend": "pyvalhalla",
            "costing": costing,
            "densify_m": densify_m,
            "search_radius_m": search_radius,
            "min_match_frac": min_frac,
            "min_match_m": min_m,
            "n_ways": len(id_map),
        }
        (output_dir / "run_meta.json").write_text(json.dumps(meta))
        logger.success(f"Wrote Meili matches to {out_path}")
        return out_path

    def parse_output(self, output_path: Path) -> ToolOutput:
        """Parse the matches parquet into a standardized ToolOutput."""
        matches = pd.read_parquet(output_path)
        matches = matches.astype({"ref_id": str, "target_id": str})
        meta_path = output_path.parent / "run_meta.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        metadata.update(
            {
                "total_matches": len(matches),
                "distinct_targets": int(matches["target_id"].nunique()) if len(matches) else 0,
            }
        )
        return ToolOutput(matches=matches, metadata=metadata)
