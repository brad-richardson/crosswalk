# Dataset Inventory Repair — 2026-07 (M4 remainder)

Follow-up to `scaleout_readiness_2026_07.md` (PR #274) and the bridge-table
factory (PR #292, `docs/FACTORY.md`). That audit found **10 labeled datasets
missing their local target parquet** (re-fetch debt) plus a **bogota bike
class-vocabulary mapping bug**, both listed in `docs/FACTORY.md` as the "M4
remainder" prerequisites for running the full inventory through
`matcher factory run --all`. This note records the repair.

All target parquets were re-fetched with the existing `matcher data fetch target`
tooling from the documented public sources in each `datasets/<name>.yaml`, and
written to the shared gitignored store `data/raw/` (Overture segments +
`.meta.yaml` with the `release:` field the factory reads were already present for
all 10). Model: `data/models/matcher_model_combined.joblib`.

## Re-fetch outcome (8 of 10 recovered)

| Dataset | Type | Source (from YAML) | Rows fetched | Factory-ready |
|---|---|---|---:|---|
| co_bogota_bike_network | bike | ArcGIS `catastrobogota` MapServer/18 (polygon→centerline) | 6,191 | **yes** |
| hk_hongkong_roads | road | data.gov.hk `RdNet_IRNP.gdb.zip` (download) | 36,398 | **yes** |
| ke_nairobi_roads | road | energydata.info `roads.zip` (shp, national subset) | 7,405 | **yes** |
| ke_kisumu_roads | road | energydata.info `roads.zip` (shp, national subset) | 722 | **yes** |
| nl_amsterdam_roads | road | PDOK NWB `nwb_wegen.gpkg` (981 MB national, bbox-filtered) | 55,710 | **yes** |
| us_utah_slc_roads | road | ArcGIS Utah SGID UtahRoads FeatureServer/0 | 74,718 | **yes** |
| us_montana_missoula | road | ArcGIS `gisservicemt.gov` MontanaTransportation/0 | 9,421 | **yes** |
| us_usfs_flathead | road | ArcGIS `apps.fs.usda.gov` EDW_RoadBasic_01/0 | 414 | **yes** |
| sg_singapore_roads | road | LTA DataMall `RoadSectionLine.zip` (API-key) | 15,319 | **yes** (see below) |
| sg_singapore_footpaths | sidewalk | LTA DataMall `Footpath.zip` (API-key) | 110,811 | **yes** (see below) |

### Correction to the audit's fetchability read

`matcher data fetch verify` reported the two Kenya sources (energydata.info) as
**HTTP 403**, which read as a dead source. That is a **false negative from the
verify HEAD probe** — a real `GET` (the fetch path) returns 200, and both Kenya
datasets fetched cleanly. Do not treat a `verify` 403 as terminal for
download-type sources; try the actual fetch.

### Singapore (2) — recovered in a follow-up with the LTA key

Both Singapore datasets come from **LTA DataMall**, which requires an account
API key passed as the `AccountKey` header (`api_key_env_var: LTA_API_KEY`).
They were initially blocked on the missing credential (not a dead source); once
an `LTA_API_KEY` was supplied via the environment, both fetched cleanly. **The
key is never stored in the repo** — it must be exported in the environment for
any re-fetch (`export LTA_API_KEY=…`, then `matcher data fetch target sg_…`).

**sg_singapore_footpaths** needed nothing else: 110,811 features (OBJECTID ids,
Mar 2026 release).

**sg_singapore_roads — the empty-RD_CD resolution.** The configured
`id_column: RD_CD` ships **100% null** (verified on the Mar 2026 release, same
as Aug 2025). With an empty upstream id component, every generated id collapsed
to `sg_road_None_{h3}` and fetch-time dedup silently dropped ~95% of segments
(15,319 → 664, one survivor per H3 cell) — the stored labels' target_ids
(`sg_road_None_…`) show the original Feb fetch had the identical defect, so the
existing 199 labels were built on the same collapsed universe and resolve via
stored geometry regardless. No other column is a unique id (`RD_CD_DESC` is the
road name, ~3.8k unique over ~15k segments). Fix: switch `id_column` to the
repo's established **synthetic geometry-hash id** `_geom_hash`
(md5 of WKT rounded to 7 dp, first 12 hex — the same mechanism already used by
`jp_tokyo_emergency_roads` and `tn_tunis_ml_roads`). Properties: deterministic
and collision-checked (15,319/15,319 unique), stable across releases while a
geometry is unchanged, and geometry edits surface as id churn — which the
stored-geometry fallback (PR #273) already absorbs. The old 664-segment
`quality_fingerprint` (an artifact of the collapse) was refreshed to the honest
15,319-segment fingerprint; the intended 664 → 15,319 jump trips the quality
regression gate, hence the one-time `--force --skip-quality-check` on this
re-fetch. Revisit if LTA ever restores `RD_CD`.

## bogota bike class-vocab fix

`co_bogota_bike_network` used `class_column: CICTSUPERF` with **no
`class_mapping`**, so the raw integer surface codes were written straight into the
semantic `class` field. `CICTSUPERF` is the **surface material** ("Tipo de
superficie"), not a road/path class — its ArcGIS coded-value domain is:

```
0 Sin dato · 1 Concreto · 2 Asfalto · 3 Mixtos · 4 Adoquin concreto · 5 Adoquin arcilla
```

`semantic.py` cannot resolve those codes to the class vocabulary, so
`class_similarity` was **100% NaN** (audit §1d — the only genuinely-actionable
class-NaN case of the four). Every feature in this layer is a dedicated cicloruta,
so the correct semantic class is `cycleway` for all surface codes (the bike-type
default). Fix: add a `class_mapping` in `datasets/co_bogota_bike_network.yaml`
mapping all six CICTSUPERF codes → `cycleway`. Re-fetch confirms **class =
cycleway for all 6,191 features** (raw code distribution was
`2: 5,657 · 1: 359 · 4: 81 · 5: 49 · 3: 45`), recovering `class_similarity`
against Overture's cycleway references.

### Label-store propagation (coordinated backfill)

The config fix alone only affects future fetches; the 29 existing labeled pairs
in `labels/data` still carried the raw codes (`2`×25, `1`×3, `5`×1) and their
`labels/features` rows were 100%-NaN on `class_similarity`. The re-fetched
OBJECTIDs share **zero** overlap with the stored target_ids (ArcGIS OBJECTID
churn), so backfill resolves these pairs entirely from stored data — meaning the
stored `target_class` had to be corrected in place (all 29 → `cycleway`),
followed by `matcher backfill -D co_bogota_bike_network` (29/29 computed,
0 skipped). Result: `class_similarity` **29/29 non-NaN** (mean 0.82, range
0.5–1.0). The recompute also refreshed some relational/topology columns
(graphlet, crossing-angle, endpoint-degree, clustering) because the restored raw
target network now provides real neighborhood context where previously only the
29 stored segments existed — the same legitimate spillover documented for
geneva_ped in the audit's Task 2. Both label parquets ride in this PR
(coordinated-backfill pattern of #256/#273).

## Factory validation

`matcher factory run` (outputs → `data/factory/release=2026-01-21.0/…`, the safe
factory root; `data/output` and `data/cache` untouched) on three small restored
datasets, all `done`:

| Dataset | Wall | Matched | Review | Unmatched | Groups | Oversized |
|---|---:|---:|---:|---:|---:|---:|
| us_usfs_flathead | 17.2 s | 316 (76%) | 28 | 70 | 131 | 0 |
| us_montana_missoula | 94.2 s | 9,002 (95%) | 316 | 103 | 1,897 | 3 |
| co_bogota_bike_network | 79.0 s | 4,712 (76%) | 577 | 902 | 1,349 | 3 |
| sg_singapore_roads | 104.5 s | 13,899 (90.7%) | 451 | 969 | 6,121 | 0 |
| sg_singapore_footpaths | 222.9 s | 53,229 (48.0%) | 5,379 | 52,203 | 14,567 | 62 |

(The footpaths match rate is data reality, not a pipeline failure: Overture's
pedestrian coverage of Singapore is partial, so ~half the LTA footpath network
has no Overture counterpart — a rich honest-negative source, like Tunis.)

Factory discovery (`matcher.factory.discovery.discover_pairs`, the routine
behind `factory run --all`) now resolves **34 stitchable pairs** (was 24) — the
8 restored datasets plus both Singapore datasets, all discovered with
`release=2026-01-21.0`.

## `factory run --all` viability for the overnight box sweep

With this repair, `matcher factory run --all` covers **34 datasets** (24 prior +
8 restored + 2 Singapore), up from the 24 the audit measured. It skips only the
two locals-without-Overture (`ch_grand_geneva_cycle_schema`,
`fr_france_winter_hiking_traces`). The audit's cost model (≈3.5–5 h serial on the
10-core dev machine; well within one overnight run at `--workers 12` on the
20-core box) holds; the incremental additions are mostly small-to-mid datasets,
with `sg_singapore_footpaths` (110.8 k local, 223 s measured),
`us_utah_slc_roads` (74.7 k local) and `nl_amsterdam_roads` (55.7 k local) the
largest new entries — all far below the `jp_tokyo` memory canary. Overnight
`--all` is viable. Note the Singapore locals in `data/raw` were fetched with a
private `LTA_API_KEY` that lives only in the operator's environment; a re-fetch
on another machine (e.g. the box) needs that env var set, or the two parquets
copied over.
