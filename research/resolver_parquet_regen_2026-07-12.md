# Resolver parquet regen verification — 2026-07-12

Branch: `feat/combined-resolver-r1-plus-votes` (combines #416 parquet join + #417 vote recovery)

## Regenerated artifacts

All 13 labeled datasets stitched with `stitch_persist_candidates=True` (default since #414) from `feat/combined-resolver-r1-plus-votes`:

| Dataset | candidates.parquet | rows | cols | groups.json |
|---|---|---:|---:|---:|
| co_bogota_bike_network | 2.3 MiB | 6,351 | 123 | 15 MiB |
| co_bogota_roads | 42 MiB | 107,868 | 123 | 247 MiB |
| de_berlin_roads | 14 MiB | 33,955 | 123 | 82 MiB |
| fi_helsinki_roads | 85 MiB | 209,310 | 123 | 719 MiB |
| ke_nairobi_roads | 11 MiB | 28,434 | 123 | 93 MiB |
| nl_amsterdam_roads | 23 MiB | 57,647 | 123 | 144 MiB |
| sg_singapore_footpaths | 31 MiB | 79,829 | 123 | 167 MiB |
| sg_singapore_roads | 9.9 MiB | 23,916 | 123 | 64 MiB |
| tn_tunis_ml_roads | 35 MiB | 95,582 | 123 | 201 MiB |
| us_boston_streets | 4.4 MiB | 10,584 | 123 | 26 MiB |
| us_montana_missoula | 2.6 MiB | 6,496 | 123 | 23 MiB |
| us_seattle_sidewalks | 14 MiB | 33,582 | 123 | 87 MiB |
| us_usfs_flathead | 251 KiB | 438 | 123 | 5.8 MiB |

All parquets validated:
- 123 cols = 83 FEATURE_COLUMNS + 40 structural/provenance (lateral_offset_signed_m, ref_class/target_class, ref_length_m, degree_ref/tgt, is_bridge, biconnected_block, corridor_ref/tgt, n_edges, optimizer_decision, decision_reason, etc.)
- FEATURE_COLUMNS present 83/83, 0 missing
- Unique keys on (group_id, ref_id, target_id), 0 dup
- Non-null signed offsets = row count (Seattle 33,582 / 33,582)

## Training table after R1 join

`build_multi_dataset_table(..., auto_discover_candidates=True)` from `extract.py` with new `_enrich_with_candidate_parquet`:

- Total rows: 945 (was 850 in PR #410 report / 850 in gap validation)
- Per-dataset (fresh):

| dataset | rows | pos | neg | cand_groups | legacy | parquet_rows | enriched | missing_keys | outside |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| co_bogota_bike_network | 6 | 5 | 1 | 2 | 0 | 6351 | 6 | 0 | 0 |
| co_bogota_roads | 23 | 6 | 17 | 4 | 0 | 107868 | 23 | 0 | 0 |
| de_berlin_roads | 8 | 8 | 0 | 4 | 0 | 33955 | 8 | 0 | 0 |
| fi_helsinki_roads | 9 | 7 | 2 | 4 | 0 | 209310 | 9 | 0 | 0 |
| ke_nairobi_roads | 5 | 5 | 0 | 2 | 0 | 28434 | 5 | 0 | 5 |
| nl_amsterdam_roads | 7 | 6 | 1 | 3 | 0 | 57647 | 7 | 0 | 0 |
| sg_singapore_footpaths | 18 | 10 | 8 | 6 | 0 | 79829 | 18 | 0 | 0 |
| sg_singapore_roads | 42 | 39 | 3 | 6 | 0 | 23916 | 42 | 0 | 0 |
| tn_tunis_ml_roads | 6 | 5 | 1 | 3 | 0 | 95582 | 6 | 0 | 0 |
| us_boston_streets | 681 | 478 | 203 | 111 | 0 | 10584 | 681 | 0 | 23 |
| us_montana_missoula | 28 | 26 | 2 | 3 | 0 | 6496 | 28 | 0 | 0 |
| us_seattle_sidewalks | 91 | 57 | 34 | 22 | 0 | 33582 | 91 | 0 | 5 |
| us_usfs_flathead | 21 | 20 | 1 | 5 | 0 | 438 | 21 | 0 | 0 |

Improvements vs gap analysis 2026-07-11:

- Seattle: 0 → 91 rows (rekey #415 + fresh sidecar fixes mapping)
- Tunis: missing → 6 rows (now generated, was absent)
- Bogotá roads: 14 legacy → 23 fresh uncapped
- Helsinki: 3 → 9
- Total: 850 → 945 (+95, +11%)
- All legacy_groups 0 (was 4 factory fallbacks)
- All candidate_parquet_missing_keys 0

Remaining recall ceiling:

- Boston 23 outside (human selected outside candidate graph) – same as before, needs candidate floor audit
- Seattle 5 outside, Nairobi 5 outside – new, small

## Vote recovery (second half of data plumbing)

`votes.py` now:

- edge_groups indexed over `edges + candidate_edges + rejected_edges + optimizer_assignment` (was only `edges`)
- seg_groups fallback for churned / NONE votes
- `edge_soft_labels` emits soft keep for full candidate universe (candidate_edges when present) – previously only selected assignment, so rejected candidate B->T was invisible. Now B->T appears with weighted soft_keep 0.33 (claude 1.0 + codex 0.5).

Manual synthetic verification:

- mapping `vg1` with union `[A->T]` + `[A->T,B->T]` → `g1`, soft table 2 rows `A:1.0, B:0.333`
- segment fallback `X->W` where `W` not in group but `X` segment overlaps → maps via segment (was unmappable)

## Next steps (per learned_optimizer_design §8)

- Land #416 + #417 (zero prod risk)
- Keep fresh parquets as baseline (this branch generated them, but parquet files are gitignored under `data/`; code branch `feat/combined-resolver-r1-plus-votes` contains the generation logic, not the artifacts)
- Update `feature_ablation_study` coverage mode to confirm dead zones now have typed features
- Rebase `feat/train-resolver-experimental` (#410) onto main+these fixes and re-run `benchmark_resolver.py` with full 83 features + vote soft rows → produce `research/resolver_round4_parquet_joined.md`
- Grow clean balanced ground truth to 200-300 groups before per-type threshold tuning
