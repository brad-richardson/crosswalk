"""GraphHopper map-matching baseline adapter.

The *second* live, actively-maintained external map-matching baseline in mbench
(alongside Valhalla Meili). Where Meili runs Valhalla in-process via a Python
wheel, GraphHopper is a JVM library — so this adapter shells out to a tiny
single-file Java runner (``GraphHopperRunner.java``) executed with
`jbang <https://jbang.dev>`, which resolves the pinned ``graphhopper-map-matching``
jar (+ transitive deps) from Maven Central and manages the JDK. There is **no
GraphHopper server** — the engine is embedded in that one JVM process.

It pilots the same **path-based formulation** as Meili (see ``meili.py`` and
``docs/EVAL_ROADMAP.md``): each local target segment is densified into a synthetic
GPS trace and map-matched onto the Overture-derived routable graph; the matched
reference-edge sequence *is* the segment↔GERS correspondence set, which handles
segmentation mismatch natively. Running a *second, independent* map-matcher over
the identical formulation is a controlled test of whether Meili's signature
(perfect recall, precision lost to parallel geometry) is a property of the
**formulation** or of Valhalla specifically.

Pipeline (~80% shared with Meili):

1. **Overture -> routable graph** (``convert/pbf.py``, shared): the Overture
   segments parquet becomes an OSM PBF; each segment is a way whose synthetic
   ``way_id`` is carried both as the OSM way id (for Valhalla) and in the ``name``
   tag (for GraphHopper, which does not expose OSM way ids on matched edges). A
   JSON sidecar maps ``way_id -> GERS id``.
2. **Import + match** in one JVM invocation (``GraphHopperRunner.java``):
   ``GraphHopper.importOrLoad()`` builds/loads the graph (cached per reference
   file), then ``MapMatching.match()`` snaps each densified trace. The matched
   ``edge.getName()`` recovers the ``way_id``.
3. **Aggregate** per target with the shared overlap-length filter
   (``mapmatch_common.aggregate_edges``) — identical to Meili.

Java is optional: if ``jbang`` (and a JVM) are absent the adapter raises a clear
"install jbang" error, and its unit tests skip cleanly, so mbench stays
pip-installable and CI stays green.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from mbench.adapters.base import EvalMode, ToolOutput
from mbench.adapters.mapmatch_common import aggregate_edges, densify_lonlat
from mbench.convert.pbf import convert_overture_to_pbf

# The foot/pedestrian analogue of Meili's default: bidirectional and traverses
# every walkable local class (residential..footway), covering roads and sidewalks.
# Override with ``--opt vehicle=car`` for a directional roads-only comparison.
DEFAULT_VEHICLE = "foot"
DEFAULT_DENSIFY_M = 10.0
# GraphHopper's GPS-noise std-dev (meters); the candidate search radius scales with
# it. Set to mirror Meili's 25 m search_radius so offset local geometry still snaps.
DEFAULT_SIGMA_M = 25.0
# Keep every edge snappable (no subnetwork pruning), mirroring Valhalla.
DEFAULT_MIN_NETWORK_SIZE = 0
DEFAULT_MIN_MATCH_FRAC = 0.10
DEFAULT_MIN_MATCH_M = 8.0
DEFAULT_WORKERS = 8

_RUNNER = Path(__file__).with_name("GraphHopperRunner.java")


def _require_jbang() -> str:
    """Return the jbang executable path, or raise a clear install message."""
    jbang = shutil.which("jbang")
    if jbang is None:
        raise RuntimeError(
            "GraphHopper adapter needs 'jbang' (which runs the JVM map-matching "
            "library) and a JVM. Install jbang: `brew install jbang` (macOS) or see "
            "https://www.jbang.dev/download/ . jbang manages the JDK 17 itself. "
            "Then re-run. mbench stays usable without it — every other adapter is "
            "pure Python."
        )
    return jbang


def build_traces_tsv(
    target: gpd.GeoDataFrame,
    tsv_path: Path,
    densify_m: float,
    id_column: str = "id",
) -> tuple[list[str], dict[str, float]]:
    """Write one densified lon/lat trace per target to ``tsv_path``.

    Returns ``(target_ids, target_len_m)`` — the id order and per-target length in
    meters (used later for the overlap filter). Line format:
    ``<target_id>\\t<lon>,<lat>;<lon>,<lat>;...``
    """
    if target.crs is None:
        raise ValueError("Target GeoDataFrame has no CRS; cannot map-match")
    if target.crs.to_epsg() != 4326:
        logger.info(f"Reprojecting target from {target.crs.to_epsg()} to EPSG:4326")
        target = target.to_crs(epsg=4326)

    metric_crs = target.estimate_utm_crs()
    geoms = target.geometry.values
    tgt_ids = target[id_column].astype(str).tolist()
    lengths_m = target.to_crs(metric_crs).geometry.length.values

    len_by_id: dict[str, float] = {}
    n_written = 0
    with open(tsv_path, "w") as f:
        for i, tid in enumerate(tgt_ids):
            len_by_id[tid] = float(lengths_m[i])
            shape = densify_lonlat(geoms[i], metric_crs, densify_m)
            if len(shape) < 2:
                continue
            coords = ";".join(f"{lon},{lat}" for lon, lat in shape)
            f.write(f"{tid}\t{coords}\n")
            n_written += 1
    logger.info(f"Wrote {n_written} traces to {tsv_path}")
    return tgt_ids, len_by_id


def parse_matches_tsv(
    tsv_path: Path,
    id_map: dict[str, str],
    len_by_id: dict[str, float],
    min_frac: float,
    min_m: float,
    densify_m: float,
) -> pd.DataFrame:
    """Parse the runner's TSV output into a [ref_id, target_id, confidence] frame.

    Each line is ``<target_id>\\t<way_id>,<edge_m>,<n_states>;...``.

    **Matched-length estimate.** GraphHopper's public map-matching API exposes only
    the *full* length of each matched edge (``EdgeIteratorState.getDistance()``), not
    the sub-length actually overlapped by the trace, which Valhalla's
    ``trace_attributes`` returns directly. Feeding full edge lengths to the overlap
    filter would unfairly inflate a parallel/crossing edge clipped by a couple of
    trace points up to its full length. We recover a faithful matched-length from the
    trace density: observations are ~``densify_m`` apart, so an edge carrying
    ``n_states`` of them spans ~``n_states * densify_m`` meters, capped at the edge's
    full length. This is the honest equivalent of Valhalla's matched sub-length, and
    it naturally zeroes out "bridged" edges (routed *between* observations with none
    snapped onto them: ``n_states == 0`` -> 0 m -> filtered), which are GraphHopper's
    parallel/connecting false positives.
    """
    rows: list[tuple[str, str, float]] = []
    if tsv_path.exists():
        with open(tsv_path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                tid, payload = line.split("\t", 1)
                edges: list[tuple[str, float]] = []
                for tok in payload.split(";"):
                    parts = tok.split(",")
                    if len(parts) < 3:
                        continue
                    way_id, meters, n_states = parts[0], parts[1], parts[2]
                    matched_m = min(float(meters), int(n_states) * densify_m)
                    edges.append((way_id, matched_m))
                pairs = aggregate_edges(edges, id_map, len_by_id.get(tid, 0.0), min_frac, min_m)
                rows.extend((gers, tid, conf) for gers, conf in pairs)

    matches = pd.DataFrame(rows, columns=["ref_id", "target_id", "confidence"])
    logger.info(
        f"GraphHopper produced {len(matches)} match pairs "
        f"({matches['target_id'].nunique() if len(matches) else 0} distinct targets matched)"
    )
    return matches


def _graph_cache_dir(base: Path, reference: Path) -> Path:
    """Deterministic per-reference cache dir (keyed by name + size + mtime)."""
    st = reference.stat()
    key = f"{reference.stem}_{st.st_size}_{int(st.st_mtime)}"
    return (base / key).resolve()


class GraphHopperAdapter:
    """GraphHopper map-matching baseline (embedded JVM via jbang)."""

    name: str = "graphhopper"
    eval_mode: EvalMode = EvalMode.STITCH

    def run(self, reference: Path, target: Path, output_dir: Path, **kwargs) -> Path:
        """Build the Overture graph, then map-match every target with GraphHopper.

        **kwargs (all via ``--opt key=value``):
            vehicle: GraphHopper encoded-value prefix / costing (default ``foot``;
                ``car`` for directional roads-only).
            densify_m: trace point spacing (default 10).
            sigma_m: GraphHopper measurement error sigma / snap radius (default 25).
            min_network_size: subnetwork prune floor (default 0 = keep all edges).
            min_match_frac / min_match_m: overlap filter thresholds (0.10 / 8 m),
                applied to the density-based matched-length (see parse_matches_tsv).
            workers: match concurrency (default 8; one MapMatching per thread).
            graph_cache_dir: where the PBF + built GH graph live (default under
                output_dir; always OUTSIDE the matcher data/output tree).
            rebuild: force a PBF + graph rebuild even if cached.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = (output_dir / "matches.parquet").resolve()

        vehicle = str(kwargs.get("vehicle", DEFAULT_VEHICLE))
        densify_m = float(kwargs.get("densify_m", DEFAULT_DENSIFY_M))
        sigma_m = float(kwargs.get("sigma_m", DEFAULT_SIGMA_M))
        min_network = int(kwargs.get("min_network_size", DEFAULT_MIN_NETWORK_SIZE))
        min_frac = float(kwargs.get("min_match_frac", DEFAULT_MIN_MATCH_FRAC))
        min_m = float(kwargs.get("min_match_m", DEFAULT_MIN_MATCH_M))
        workers = int(kwargs.get("workers", DEFAULT_WORKERS))
        rebuild = str(kwargs.get("rebuild", "")).lower() in ("1", "true", "yes")

        jbang = _require_jbang()

        cache_base = Path(kwargs.get("graph_cache_dir", output_dir / "graph_cache"))
        cache_dir = _graph_cache_dir(cache_base, reference)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pbf_path = cache_dir / "graph.osm.pbf"
        id_map_path = cache_dir / "id_map.json"
        gh_dir = cache_dir / "gh_graph"

        # 1. Overture -> PBF (+ id-map), cached. A rebuild also clears the GH graph
        #    dir so importOrLoad re-imports from the fresh PBF.
        if pbf_path.exists() and id_map_path.exists() and not rebuild:
            logger.info(f"Reusing cached PBF/id-map in {cache_dir}")
        else:
            convert_overture_to_pbf(reference, pbf_path, id_map_path)
            if gh_dir.exists():
                shutil.rmtree(gh_dir)
        id_map = json.loads(id_map_path.read_text())

        # 2. Densify targets into traces.
        logger.info(f"Loading target {target}")
        tgt = gpd.read_parquet(target)
        traces_path = output_dir / "traces.tsv"
        matches_tsv = output_dir / "gh_matches.tsv"
        _tids, len_by_id = build_traces_tsv(tgt, traces_path, densify_m)

        # 3. Import graph + match (one JVM process via jbang).
        logger.info(
            f"Map-matching {len(tgt)} target segments with GraphHopper "
            f"(vehicle={vehicle}, densify={densify_m}m, sigma={sigma_m}m, workers={workers})"
        )
        cmd = [
            jbang,
            str(_RUNNER),
            str(pbf_path),
            str(gh_dir),
            str(traces_path),
            str(matches_tsv),
            vehicle,
            str(sigma_m),
            str(min_network),
            str(workers),
        ]
        logger.info("Running GraphHopper JVM runner (jbang; first run resolves the jar)...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"GraphHopper runner failed (exit {r.returncode}):\n{r.stderr[-3000:]}"
            )
        # The runner logs its match count on stderr; surface it.
        for tail in r.stderr.strip().splitlines()[-3:]:
            logger.info(f"[graphhopper] {tail}")

        # 4. Aggregate + overlap filter (shared with Meili).
        matches = parse_matches_tsv(
            matches_tsv, id_map, len_by_id, min_frac, min_m, densify_m=densify_m
        )
        matches.to_parquet(out_path)

        meta = {
            "backend": "graphhopper-map-matching:10.2 (jbang)",
            "vehicle": vehicle,
            "densify_m": densify_m,
            "sigma_m": sigma_m,
            "min_network_size": min_network,
            "min_match_frac": min_frac,
            "min_match_m": min_m,
            "n_ways": len(id_map),
        }
        (output_dir / "run_meta.json").write_text(json.dumps(meta))
        logger.success(f"Wrote GraphHopper matches to {out_path}")
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
