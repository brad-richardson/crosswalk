# Cross-Mode Match Audit — 2026-08-06

Audit of all 36 datasets in factory `release=2026-06-17.0` for the known
cross-mode defect: cycleway/footpath features matched to parallel ROAD
centerlines (or vice versa) at high confidence. Positive control:
`co_bogota_bike_network` (on `quality_hold` since 2026-07-06 for exactly this).

- Script: `scripts/cross_mode_audit.py` (re-runnable; writes full per-dataset
  JSON with `--json`)
- Inputs: `data/factory/release=2026-06-17.0/dataset=*/bridge.parquet`
  (rows with `match_decision == "match"`), joined to `class` from
  `data/raw/<name>_v1.0.parquet` (target, keyed by `local_id`) and
  `data/raw/<name>_overture_segments_v1.0.parquet` (ref, keyed by `gers_id`).
  Join coverage was 100% of matched rows for every dataset.

## Methodology

The mode taxonomy is **reused verbatim** from the deterministic
class-consistency gate in the panel runner
(`src/crosswalk/agent_labeling/stitch_runner.py`, imported directly — not
re-implemented):

- `PEDESTRIAN_CLASSES` = {footway, sidewalk, path, pedestrian, steps, crossing}
- `VEHICULAR_CLASSES` = {motorway, trunk, primary, secondary, tertiary,
  residential, service, unclassified, living_street, driveway, road}
- `CYCLEWAY_CLASSES` = {cycleway}
- Everything else (track, unknown, "", None) is **neutral** and never flags
  (`road_class_mode` / `is_cross_mode_edge` semantics: a pair is cross-mode iff
  both sides are non-neutral and the modes differ).

Two counters:

- **xmode** — any cross-mode pair per the gate (includes the gate's
  conservative ped↔bike extra).
- **xveh** — cross-mode with a **vehicular side** (ped/cycle on one side, road
  class on the other). This is the dangerous parallel-road-centerline pattern
  the bogota hold describes; ped↔bike pairs are usually the same shared-use
  path with divergent class vocab (Overture `cycleway` vs local `path`/
  `footway`), per the memory note that cycleway class is a weak cross-mode
  signal.

`hi` = confidence ≥ 0.8 (the dangerous bucket). One audit-only extension: for
datasets typed `sidewalk`/`bike` in `datasets/<name>.yaml` whose target rows
have neutral/missing class, target mode is inferred from the dataset type
(targets are pedestrian/bike by construction). This only affected
`us_seattle_sidewalks` (no class column) and small slices of the other
sidewalk sets. No inference for `trail`/`road` types.

**Positive control validated**: `co_bogota_bike_network` lights up with 1,105
road↔cycleway matches, 580 at conf ≥ 0.8 (10.5% of the dataset), confidence
median 0.873, max 0.998 — matching the hold's "0.82–0.95 confidence"
description.

## Results (sorted by high-confidence vehicular cross-mode)

| Dataset | Matches | xmode | xmode hi | xveh | xveh hi | xveh hi % | Dominant pattern |
|---|---:|---:|---:|---:|---:|---:|---|
| fi_helsinki_roads | 227,481 | 62,872 | 58,073 | 62,872 | 58,073 | 25.53 | ref cycleway ↔ tgt "service" (mis-mapped, see below) |
| co_bogota_roads | 135,598 | 11,917 | 8,201 | 11,899 | 8,185 | 6.04 | ref footway ↔ tgt tertiary/residential |
| us_boston_bike_network | 4,589 | 4,155 | 3,887 | 3,653 | 3,444 | 75.05 | ref road ↔ tgt cycleway |
| br_sao_paulo_roads | 205,982 | 4,453 | 3,287 | 4,453 | 3,287 | 1.60 | ref footway ↔ tgt road |
| gb_london_roads | 300,244 | 5,380 | 3,244 | 5,380 | 3,244 | 1.08 | ref footway/cycleway ↔ tgt road |
| ch_grand_geneva_cycle_schema | 3,802 | 3,580 | 3,145 | 3,406 | 2,993 | 78.72 | ref road ↔ tgt cycleway (route overlay, by design) |
| de_berlin_roads | 47,156 | 2,757 | 1,834 | 2,757 | 1,834 | 3.89 | ref footway ↔ tgt road |
| nl_amsterdam_roads | 60,536 | 3,011 | 2,222 | 2,187 | 1,626 | 2.69 | ref footway ↔ tgt road/bike |
| ca_toronto_roads | 68,798 | 3,379 | 3,097 | 883 | 668 | 0.97 | ref cycleway ↔ tgt path (soft) |
| au_sydney_roads | 173,560 | 4,994 | 4,378 | 1,039 | 614 | 0.35 | ref cycleway ↔ tgt path (soft) |
| co_bogota_bike_network | 5,503 | 1,794 | 1,102 | 1,105 | 580 | 10.54 | ref road ↔ tgt cycleway (**positive control**) |
| sg_singapore_footpaths | 57,552 | 4,940 | 3,518 | 1,085 | 456 | 0.79 | ref cycleway ↔ tgt footway (soft) + veh subset |
| ch_geneva_pedestrian_network | 2,219 | 655 | 505 | 560 | 424 | 19.11 | ref residential/uncl/service ↔ tgt footway |
| ch_geneva_hiking_routes | 631 | 262 | 229 | 253 | 220 | 34.87 | ref residential/uncl ↔ tgt path (route traces) |
| us_boston_sidewalks | 52,830 | 1,491 | 1,215 | 368 | 190 | 0.36 | ref cycleway ↔ tgt footway (soft) |
| jp_tokyo_emergency_roads | 18,478 | 268 | 103 | 268 | 103 | 0.56 | ref footway ↔ tgt road |
| us_utah_slc_roads | 80,985 | 185 | 87 | 185 | 87 | 0.11 | ref footway ↔ tgt road |
| us_philadelphia_sidewalks | 54,980 | 981 | 745 | 286 | 84 | 0.15 | ref bike ↔ tgt ped (soft) |
| sg_singapore_roads | 22,658 | 178 | 71 | 178 | 71 | 0.31 | ref footway ↔ tgt road |
| us_seattle_sidewalks | 29,352 | 341 | 194 | 201 | 71 | 0.24 | ref road ↔ tgt sidewalk (type-inferred) |
| in_mumbai_streets | 56,589 | 108 | 47 | 108 | 47 | 0.08 | ref footway ↔ tgt road |
| us_frisco_trails | 423 | 141 | 138 | 40 | 39 | 9.22 | ref service ↔ tgt path; ref cycleway ↔ tgt path (soft) |
| au_melbourne_roads | 19,151 | 45 | 38 | 45 | 38 | 0.20 | ref cycleway ↔ tgt road |
| us_boston_streets | 14,393 | 41 | 34 | 41 | 34 | 0.24 | mixed |
| us_montana_missoula | 10,117 | 37 | 18 | 37 | 18 | 0.18 | ref footway ↔ tgt road |
| us_fort_collins_streets | 11,841 | 23 | 14 | 23 | 14 | 0.12 | mixed |
| us_montana_helena | 8,655 | 7 | 4 | 7 | 4 | 0.05 | ref footway ↔ tgt road |
| us_fort_collins_sidewalks | 19,154 | 234 | 214 | 3 | 1 | 0.01 | ref bike ↔ tgt ped (soft) |
| us_usfs_flathead | 490 | 3 | 1 | 3 | 1 | 0.20 | — |
| us_austin_sidewalks | 11,335 | 76 | 70 | 1 | 0 | 0.00 | ref bike ↔ tgt ped (soft) |
| fr_france_winter_hiking_traces | 3,720 | 0 | 0 | 0 | 0 | 0.00 | **audit-blind** (tgt class 100% unknown) |
| hk_hongkong_roads | 34,267 | 0 | 0 | 0 | 0 | 0.00 | audit-blind (tgt class 100% unknown) |
| ke_kisumu_roads | 3,858 | 0 | 0 | 0 | 0 | 0.00 | audit-blind (tgt class 100% unknown) |
| ke_nairobi_roads | 29,698 | 0 | 0 | 0 | 0 | 0.00 | audit-blind (tgt class 100% unknown) |
| tn_tunis_ml_roads | 126,609 | 0 | 0 | 0 | 0 | 0.00 | audit-blind (tgt class 100% unknown) |
| us_usfs_lolo | 810 | 0 | 0 | 0 | 0 | 0.00 | 43–48% neutral both sides |

## Key findings and examples

### us_boston_bike_network — bogota's defect signature, but 82% is legitimate

3,444 high-conf road↔bike matches (75% of the dataset). Facility-type
breakdown of the 3,412 high-conf road-ref↔cycleway-target pairs (via
`source_tags.ExisFacil`): **82.2% are painted/shared-lane facilities** (BL
1,255, SLM 1,072, BLSL 205, BFBL 147, ...) that share the road geometry —
matching the road centerline is *correct* for those. **607 pairs (17.8%) are
separated facilities** (SBL 535, SBLBL 43, SBLSL 24, CFSBL 5) — the genuine
bogota-style defect, ≈13% of all matches. Note `datasets/us_boston_bike_network.yaml`
maps painted lanes (BL etc.) to `cycleway` in `fetch.class_mapping` even though
its own `classification.class_mapping_rules` maps BL→`unknown`.

Examples (all conf ≈ 1.0): `boston_bike_799_882a3064e7` ↔ `1442458d-25ad-483b-a3c3-7e6404f6121b`
(tertiary↔cycleway); `boston_bike_10_882a30655b` ↔ `ac4c59a0-61f4-4a44-9046-938a1c4a5670`
(primary↔cycleway); `boston_bike_781_882a3065db` ↔ `e72f446c-1f9d-4239-8347-ef46702d3425`
(secondary↔cycleway).

### fi_helsinki_roads — class-mapping defect, not (primarily) a geometry defect

58,073 high-conf cross-mode (25.5%), dominated by ref `cycleway` ↔ target
`service` (49,798). Verified via `source_tags`: those targets carry
`toiminn_lk: 8` / `linkkityyp: 8` (Digiroad functional class 8 = kevyen
liikenteen väylä, the combined foot/cycle path class), which
`datasets/fi_helsinki_roads.yaml` `class_mapping` folds into `service`
(`8: service`, `9: service`). The Overture cycleway is most likely the *same
physical path*; the match is right, the target class is wrong. Fix the yaml
mapping (8 → cycleway/path) and re-audit; the residual footway↔residential
(1,230) and path↔residential (1,372) slices then need a normal spot-check.
Examples (conf 1.0): `hel_road_638f61f4-...:4_8808997b23` ↔
`1a9da8d4-4da4-4d8b-8861-3d85c6e480d8` (cycleway↔service);
`hel_road_c923f6cf-...:4_881126d0ed` ↔ `bf445cf5-...` (cycleway↔residential);
`hel_road_0f72724d-...:3_881126d753` ↔ `a50fa7c6-...` (cycleway↔tertiary).

### ch_grand_geneva_cycle_schema — cross-mode by design

2,993 high-conf road↔bike (78.7%), but `datasets/ch_grand_geneva_cycle_schema.yaml`
documents `matching.target_kind: route_network`: targets are signed cycle
route overlays (Liaison/Maillage/Voie Verte) — "a route that follows a road
matches the road". The flags are expected, not a defect. Examples (conf 1.0):
`ch_geneva_bike_126_881f91a0e3` ↔ `acb685df-...` (residential↔cycleway);
`ch_geneva_bike_1530_881f91a449` ↔ `e5b7c114-...` (primary↔cycleway).

### co_bogota_bike_network — positive control (hold confirmed)

580 high-conf road↔cycleway (10.5%; conf median 0.873, max 0.998); plus 689
footway-ref↔cycleway soft flags. Top class pairs: footway↔cycleway 678,
tertiary↔cycleway 412, secondary↔cycleway 269, residential↔cycleway 268.
Examples: `bog_bike_53197_8866e0904d` ↔ `8cfc325d-0edf-4fd6-a59e-c46f166ae207`
(residential↔cycleway, 0.998); `bog_bike_57287_8866e092eb` ↔
`48a71dc5-f0f1-4ca3-9bdb-5f41585c393f` (secondary↔cycleway, 0.998);
`bog_bike_54598_8866e08205` ↔ `9ca576b4-...` (footway↔cycleway, 0.998).

### co_bogota_roads — the same defect mirrored from the roads side

8,185 high-conf vehicular cross-mode (6.0%): ref footway ↔ target
tertiary/residential (4,200 + 3,298), plus residential-ref ↔ pedestrian-target
1,423 and steps↔tertiary 629. This is Overture sidewalk/steps geometry being
consumed by road targets (or Bogotá pedestrian streets matched to residential
refs) — same neighborhood, same parallel-geometry failure. Examples:
`bog_road_190447_8866e0918b` ↔ `6fa4cfec-...` (cycleway↔pedestrian, 0.998);
`bog_road_272922_8866e09047` ↔ `50a9abb6-...` (cycleway↔primary, 0.992).

### ch_geneva_pedestrian_network / ch_geneva_hiking_routes

Pedestrian network: 424 high-conf veh↔ped (19.1%) — residential↔footway 203,
unclassified↔footway 164, service↔footway 154. Some may be legit shared-street
links (network edges along streets without separate sidewalk geometry), but
the rate is bogota-magnitude. Example: `ch_geneva_ped_216_881f91af13` ↔
`bde48733-...` (service↔footway, 0.982). Hiking routes: 220 high-conf veh↔path
(34.9%) — residential↔path 101, unclassified↔path 63 — but these are route
traces that legitimately follow roads (no `target_kind` annotation in the yaml,
unlike the cycle schema). Example: `ch_geneva_hike_4_881f91a8a7` ↔
`cc6a2d31-...` (residential↔path, 0.982).

### sg_singapore_footpaths / us_frisco_trails

Singapore: 456 high-conf veh↔footway (0.79%) — residential↔footway 716,
primary↔footway 76, **motorway↔footway 29**, trunk↔footway 18; the 3,855
cycleway↔footway flags are mostly PCN shared paths (soft). Example:
`sg_footpath_41533_886526adeb` ↔ `793d1af9-...` (primary↔footway, 0.958).
Frisco: 39 high-conf veh↔path out of 423 matches (9.2%), all service↔path
(40) — service-road vs trail alignments. Example: `frisco_trail_198_8826c87761`
↔ `fe369fd3-...` (cycleway↔path, 1.0).

### Audit-blind datasets

`fr_france_winter_hiking_traces`, `hk_hongkong_roads`, `ke_kisumu_roads`,
`ke_nairobi_roads`, `tn_tunis_ml_roads` store target class = `unknown` for
100% of rows — their zero counts are **not** evidence of cleanliness. For the
one ped-flavored blind dataset, fr_france_winter_hiking_traces, a ref-side
proxy: 516 matches to vehicular refs, 433 at conf ≥ 0.8 (11.6% of matches;
mostly unclassified/tertiary rural roads, which winter routes may legitimately
follow). `us_seattle_sidewalks` was rescued via type inference (all targets
pedestrian by construction).

## Recommendations

| Dataset | Recommendation | Rationale |
|---|---|---|
| co_bogota_bike_network | **keep quality_hold** (control) | 580 high-conf road↔cycleway (10.5%); hold confirmed |
| us_boston_bike_network | **recommend quality_hold** | 607 high-conf separated-facility (SBL*) road matches ≈13% of dataset; remaining 82% painted-lane matches are legit — a facility-aware fix could release most of it |
| co_bogota_roads | **recommend quality_hold** (or urgent spot-check) | 8,185 high-conf veh↔ped (6.0%), same geometry as the held bike network |
| fi_helsinki_roads | **needs fix + re-audit** | not a geometry defect: yaml class_mapping folds Digiroad class 8 (foot/cycle path) into `service`; fix mapping, then spot-check residual ~2k ped↔veh |
| ch_geneva_pedestrian_network | **needs spot-check (high priority)** | 424 high-conf veh↔ped (19.1%); shared-street links plausible but rate is bogota-magnitude |
| ch_geneva_hiking_routes | needs spot-check | 220 high-conf veh↔path (34.9%) but route-trace semantics make road-following legitimate; consider a `target_kind: route_network` annotation if confirmed |
| ch_grand_geneva_cycle_schema | clean (by design) | route overlay, documented `target_kind: route_network`; flags expected |
| us_frisco_trails | needs spot-check | 39 high-conf service↔path (9.2%) in a tiny dataset |
| sg_singapore_footpaths | needs spot-check | 456 high-conf veh↔footway (0.79%) incl. motorway/trunk↔footway; cycleway↔footway bulk is PCN vocab overlap |
| fr_france_winter_hiking_traces | needs spot-check (audit-blind) | no target classes; 433 high-conf matches to vehicular refs (11.6%) by ref-side proxy |
| nl_amsterdam_roads | needs spot-check | 1,626 high-conf veh-involved (2.7%) |
| de_berlin_roads | needs spot-check | 1,834 high-conf (3.9%), footway-ref↔road-target |
| br_sao_paulo_roads, gb_london_roads | needs spot-check (low) | 1.6% / 1.1% high-conf veh-involved; large absolute counts (3.2k each) |
| ca_toronto_roads, au_sydney_roads | clean-ish / spot-check (low) | veh-involved ≤1%; bulk is cycleway-ref↔path-target shared-use vocab overlap |
| us_boston_sidewalks | clean-ish | 190 high-conf veh↔ped (0.36%) |
| us_seattle_sidewalks | clean-ish | 71 high-conf (0.24%, type-inferred) |
| us_philadelphia_sidewalks, us_austin_sidewalks, us_fort_collins_sidewalks | clean | ≤0.15% high-conf veh-involved |
| jp_tokyo_emergency_roads, sg_singapore_roads, us_utah_slc_roads, in_mumbai_streets, au_melbourne_roads, us_boston_streets, us_montana_missoula, us_fort_collins_streets, us_montana_helena, us_usfs_flathead, us_usfs_lolo | clean | ≤0.6% high-conf, single/double-digit counts |
| hk_hongkong_roads, ke_kisumu_roads, ke_nairobi_roads, tn_tunis_ml_roads | clean (audit-blind, road-type) | no target classes, but road datasets matched to ≥98% non-neutral road refs; low risk |

## Actions taken (2026-08-06)

The three highest-priority rows above are now implemented in the repo; the
remaining rows are open follow-ups.

| Recommendation | Status |
|---|---|
| `us_boston_bike_network` → quality_hold | **done** — `quality_hold` block in `datasets/us_boston_bike_network.yaml` (since 2026-08-06) |
| `co_bogota_roads` → quality_hold | **done** — `quality_hold` block in `datasets/co_bogota_roads.yaml` (since 2026-08-06) |
| `fi_helsinki_roads` → fix class mapping | **partially done** — `fetch.class_mapping` now maps `toiminn_lk: 8` to `path` instead of `service`. `class_mapping` is applied at *fetch* time (`fetch/target.py:293`) and baked into `data/raw/`, so this is **inert until Helsinki is re-fetched**. Nothing detects the resulting yaml/parquet drift automatically. Re-fetch → re-run factory → re-run `scripts/cross_mode_audit.py`, then spot-check the residual footway↔residential (1,230) and path↔residential (1,372) slices |
| `ch_geneva_pedestrian_network` spot-check | open (highest-priority remaining) |
| everything else in the table above | open |

Mapping evidence for the Helsinki fix, from `data/raw/fi_helsinki_roads_v1.0.parquet`
`source_tags` (219,106 rows): `toiminn_lk = 8` covers 57,900 rows, 57,860 of which
carry `linkkityyp = 8` (*kevyen liikenteen väylä*, the combined foot/cycle way) and
39 `linkkityyp = 9` (*jalankulkualue*, pedestrian area) — none are service roads.
`toiminn_lk = 9` is a genuinely different, much smaller class (194 rows, all
`linkkityyp` 10/13/14/15 = *huolto-/pelastustie*, *huoltoaukko*) and correctly
stays mapped to `service`. `path` is the right target for 8 because the class is
shared foot+cycle, so neither `footway` nor `cycleway` fits.

**Cross-cutting**: the defect concentrates where a ped/bike/trail dataset
overlaps a dense road network (bogota pattern) — and the audit's `xveh hi`
metric reproduces the held dataset's signature and finds two more materially
affected datasets (us_boston_bike_network's separated-facility subset,
co_bogota_roads). Datasets whose targets are route overlays (grand geneva
cycle schema, hiking routes/traces) flag heavily but are legitimate by design;
a `target_kind: route_network` annotation (already used by the cycle schema)
would let future audits and the panel gate exempt them explicitly.
