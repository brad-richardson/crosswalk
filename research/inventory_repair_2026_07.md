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
| sg_singapore_roads | road | LTA DataMall `RoadSectionLine.zip` (API-key) | — | **blocked** |
| sg_singapore_footpaths | sidewalk | LTA DataMall `Footpath.zip` (API-key) | — | **blocked** |

### Correction to the audit's fetchability read

`matcher data fetch verify` reported the two Kenya sources (energydata.info) as
**HTTP 403**, which read as a dead source. That is a **false negative from the
verify HEAD probe** — a real `GET` (the fetch path) returns 200, and both Kenya
datasets fetched cleanly. Do not treat a `verify` 403 as terminal for
download-type sources; try the actual fetch.

### Still blocked (2) — and why

Both Singapore datasets come from **LTA DataMall**, which requires an account
API key passed as the `AccountKey` header (`api_key_env_var: LTA_API_KEY`). No
`LTA_API_KEY` is present in the environment, so neither can be fetched here. This
is a **credential gap, not a dead source** (the DataMall endpoints are up). To
unblock: obtain an LTA DataMall account key, `export LTA_API_KEY=…`, then
`matcher data fetch target sg_singapore_footpaths` / `sg_singapore_roads`.

Additional caveat for **sg_singapore_roads** specifically: even with a key, its
configured `id_column: RD_CD` ships **completely empty** in the current LTA
release (documented in the dataset YAML `notes`), so it has no stable upstream ID
for label linkage. It needs an ID-column revisit (or LTA restoring `RD_CD`)
before it is truly stitchable, independent of the key.

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
cycleway for all 6,191 features** (was raw `1`/`2`/`5`), recovering
`class_similarity` against Overture's cycleway references.

## Factory validation

`matcher factory run` (outputs → `data/factory/release=2026-01-21.0/…`, the safe
factory root; `data/output` and `data/cache` untouched) on three small restored
datasets, all `done`:

| Dataset | Wall | Matched | Review | Unmatched | Groups | Oversized |
|---|---:|---:|---:|---:|---:|---:|
| us_usfs_flathead | 17.2 s | 316 (76%) | 28 | 70 | 131 | 0 |
| us_montana_missoula | 94.2 s | 9,002 (95%) | 316 | 103 | 1,897 | 3 |
| co_bogota_bike_network | 79.0 s | 4,712 (76%) | 577 | 902 | 1,349 | 3 |

`matcher factory discovery` now resolves **32 stitchable pairs** (was 24) — the 8
restored datasets are all discovered with `release=2026-01-21.0`.

## `factory run --all` viability for the overnight box sweep

With this repair, `matcher factory run --all` covers **32 datasets** (24 prior +
8 restored), up from the 24 the audit measured. It skips only the 2 Singapore
datasets (missing until an LTA key is supplied) and the two locals-without-Overture
(`ch_grand_geneva_cycle_schema`, `fr_france_winter_hiking_traces`). The audit's
cost model (≈3.5–5 h serial on the 10-core dev machine; well within one overnight
run at `--workers 12` on the 20-core box) holds; the incremental additions are
mostly small-to-mid datasets, with `us_utah_slc_roads` (74.7 k local) and
`nl_amsterdam_roads` (55.7 k local) the largest new entries — both far below the
`jp_tokyo` memory canary. Overnight `--all` is viable.
