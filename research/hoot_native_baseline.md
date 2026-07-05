# Hootenanny native x86 baseline (hoot 0.2.87)

> _Harness renamed `cbench` → `mbench` (2026-07-05). Mentions of "cbench" below are historical; the harness is now invoked as `mbench`._

This note records the one-shot **native x86 Linux** run of Hootenanny that closes
the "wall time is n/a — emulated" caveat on the frozen Hootenanny rows in
[`docs/BENCHMARK_RESULTS.md`](../docs/BENCHMARK_RESULTS.md). It documents the
exact version, host, commands, valid wall times / peak memory, quality, and — in
detail — the merge-phase crash that prevented a faithful reproduction of the
0.2.41 snap-merge quality row on 0.2.87.

## TL;DR

- **Version: Hootenanny `0.2.87_3_g3eb7beb` (built 2026-01-22)** — the *current*
  release family (vs the emulated `0.2.41` from 2018), from the user's
  compose-built `hootenanny-core-services:latest` image + the source checkout it
  mounts at `$HOOT_HOME`.
- **Valid native wall time (closes the caveat).** hoot conflation is
  **single-threaded** (99% of one core throughout), so timing is honest and the
  12-core cap was non-binding:
  | Run | Dataset | Wall | Peak RSS |
  |-----|---------|------|----------|
  | full conflation (tag-only merge) | us_boston_streets | **5m47s** | 1.79 GB |
  | match-only (`conflate.match.only`) | us_boston_streets | **3m26s** | 1.64 GB |
  | full conflation (tag-only merge) | us_fort_collins_sidewalks | **4m50s** | 1.48 GB |
- **Faithful snap-merge quality could NOT be reproduced on 0.2.87** — the default
  / Unifying `LinearSnapMerger` **aborts** in the merge phase
  (`No node ID specified for RemoveNodeByEid`) on this synthetic OSM topology. The
  only completing merge mode is **`LinearTagOnlyMerger`** (hoot's Attribute
  Conflation merge), which surfaces matches more coarsely.
- **Native 0.2.87 quality (tag-only merge, same `HighwayMatchCreator` matcher):**
  - us_fort_collins_sidewalks: **P 1.000 / R 0.887 / F1 0.940** (TP 188, FP 0,
    FN 24) — comparable to, and slightly above, the 0.2.41 emulated F1 0.927.
  - us_boston_streets: P 0.988 / R 0.634 / F1 0.773 (TP 170, FP 2, FN 98) — the
    **recall is a merge-representation artifact, not a matcher regression** (see
    below); the 0.2.41 emulated snap-merge row (F1 0.973) remains the Boston
    quality reference.

## Host

- **CPU:** Intel Core Ultra 7 265K, 20 cores / 20 threads, x86_64
- **RAM:** 62 GiB
- **OS:** Ubuntu 24.04.4 LTS (kernel 6.17)
- **Docker:** 29.5.2
- The box is a shared always-on media server, so the conflation container was
  **capped to 12 CPUs** (`docker run --cpus 12`) and 40 GiB RAM. Because hoot
  conflation is single-threaded, the cap did not affect wall time.

## Exact commands

hoot ran in a standalone container off `hootenanny-core-services:latest` with the
source checkout bind-mounted at `$HOOT_HOME` (mirroring the compose layout), which
avoids starting the OOM-prone postgres/frontend/tomcat services — the `hoot` CLI
does not need them:

```bash
docker run --rm --cpus 12 --memory 40g --user 1000:1000 \
  -e HOOT_HOME=/var/lib/hootenanny \
  -v ~/dev/hootenanny:/var/lib/hootenanny \
  -v <workdir>:/data \
  --entrypoint /usr/bin/time \
  hootenanny-core-services:latest \
  -v /var/lib/hootenanny/bin/hoot conflate --warn \
    -D match.creators=HighwayMatchCreator \
    -D geometry.linear.merger.default=LinearTagOnlyMerger \
    /data/<ref>.osm /data/<tgt>.osm /data/<out>.osm
```

Inputs are the identical Boston/Fort-Collins OSM the cbench Hootenanny adapter
builds (`cbench.convert.osm.convert_parquet_to_osm`, `matcher_ref_*`/`matcher_tgt_*`
provenance tags; reference with connectors for topology, target without). The
conflated OSM was scored with the adapter's own `parse_output()` +
`cbench.eval.metrics.evaluate(match_level="target")` + stitch eval — the exact
path `cbench run hootenanny` uses.

## The merge-phase crash (why the faithful row could not be reproduced)

Every attempt to run a full conflation with a **geometry-snapping** highway merger
aborts at ~52% ("Merging feature matches") with:

```
Error running conflate:
No node ID specified for RemoveNodeByEid.
```

`--info` logging pins it to the merger loop:
`Applying merger: LinearSnapMerger 1,611 / 13,121` → crash. `LinearSnapMerger`
splits/relinks ways during the snap and, via `RecursiveElementRemover`, calls
`RemoveNodeByEid` with a default (unset) element id. The trigger is
coincident-coordinate nodes in the synthetic target OSM (hoot also logs
`FindNodesInWayFactory: Two nodes were found with the same coordinate`); 0.2.41
tolerated this, 0.2.87 does not. Note the matcher itself found **13,121 match
sets** — matching is unaffected; only the snap-*merge* crashes.

Reproduced (all abort identically) with:

1. Default config (no overrides).
2. `-D merger.creators=HighwayMergerCreator` (0.2.87's 2nd-gen Unifying merger;
   the 0.2.41 `HighwaySnapMergerCreator` no longer exists).
3. `-D match.creators=NetworkMatchCreator -D merger.creators=NetworkMergerCreator`
   (the alternative Network algorithm — same executor, same crash; also ~5× slower).
4. Custom `conflate.pre.ops`/`conflate.post.ops` with the roundabout ops removed
   (`RemoveRoundabouts`/`ReplaceRoundabouts`) — the crash is in the merger, not the
   ops.
5. `-C AttributeConflation.conf` — still selected a snap merger in this path.

Two non-crashing modes exist, neither a faithful snap-merge substitute:

- **`conflate.match.only=true`** completes (3m26s) but writes matches back only as
  per-element `hoot:status`/provenance tags with **no review relations and no
  combined ref+tgt way** — i.e. no observable correspondence to extract. Confirms
  the adapter's long-standing note that match-only "discards results."
- **`geometry.linear.merger.default=LinearTagOnlyMerger`** (Attribute Conflation
  merge) completes because it transfers secondary tags onto the reference way
  instead of snapping geometry (and, as a side effect, disables roundabout
  removal). Matched targets surface as reference ways carrying both
  `matcher_ref_*` and `matcher_tgt_*` tags, plus review relations — extractable.

### Why tag-only recall is dataset-dependent

Tag-only keeps whole reference ways and never splits them, so a match where a short
local segment overlaps only *part* of a long Overture way cannot be surfaced as a
sub-segment correspondence — exactly the segmentation-mismatch case the snap merger
handled by splitting. Hence:

- **Boston roads** (short local segments vs long Overture ways → heavy
  segmentation mismatch): tag-only surfaces ~11.6K pairs and only 170/268 labeled
  targets → **R 0.634**. This is a *merge-representation* shortfall, not the
  0.2.87 matcher matching worse than 0.2.41.
- **Fort Collins sidewalks** (finely segmented on both sides, close to 1:1):
  tag-only surfaces ~25.8K pairs (incl. 4,943 reviews) and 188/212 labeled targets
  → **R 0.887, F1 0.940**, on par with the emulated 0.2.41 row.

## Verdict

The native run **closes the timing caveat** (valid single-threaded wall times on
current-family hoot 0.2.87) and yields a solid native quality row for footways
(FC F1 0.940). For roads it confirms that the 2018→2024 hoot line has a
merge-phase regression on synthetic (non-topological) target OSM, so the faithful
Boston snap-merge quality row stays pinned at the 0.2.41 emulated numbers, exactly
as the "frozen baseline" policy intends. Per the benchmarking guidance, we did not
rebuild hoot from source to chase the `LinearSnapMerger` bug.
