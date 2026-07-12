# Resolver Benchmark — production optimizer vs experimental candidate policy

> Research-only. Compares multi-edge stitching-group selection strategies on curated labels.
> The saved resolver artifact is scored only in the optimistic in-sample section. Grouped CV and LODO freshly fit the candidate XGBoost architecture using the artifact's feature manifest and selector; they do not evaluate that saved fit.

## Reproducibility manifest

- generated_utc: `2026-07-12T22:28:39+00:00`
- canonical replay command: `uv run python scripts/benchmark_resolver.py --data-root . --labels-root labels/stitching --model-path data/models/resolver_model.joblib --output research/resolver_benchmark.md --n-splits 5 --repeat-seeds 0,1,2,3,4 --bootstrap-resamples 2000 --seed 0 --include-split --lodo`
- base source commit: `0bb06d651732984c101fb4fe819ef42c42ae957e`
- tracked working-tree dirty: `true`
- untracked files present: `true`
- tracked source patch SHA256 (generated report excluded): `e49d39a39ea29f1644ec0b3552a0d9ec2f650db6b35055c3fd2145e0ac6a7fea`
- untracked non-ignored files hashed below: `8`

| role | path | bytes | SHA256 | tracked | ignored |
|---|---|---:|---|---|---|
| saved resolver model | `data/models/resolver_model.joblib` | 143485 | `46122f8e5764dcd33d9c3fa06929a76dee546cf8c5c51cc2237b2350af61b7ed` | False | True |
| us_boston_streets groups | `data/output/us_boston_streets_groups.json` | 23170442 | `5eaeadc4ac7ad1d0d061c14c58fbdecbe501ce07144ff3686ac5a77218807f28` | False | True |
| us_boston_streets candidates | `data/output/us_boston_streets_candidates.parquet` | 4121868 | `a3d426c85c62ae37d34f7e7a03bb7316f1d240714e7a676cfb7c4c0a49cbfccd` | False | True |
| us_boston_streets bridge | `data/output/us_boston_streets_bridge.parquet` | 962274 | `497216c73c8a9d25fed9f65b5eec34de984105c7b3705014ce22d715a24e3ff8` | False | True |
| us_seattle_sidewalks groups | `data/output/us_seattle_sidewalks_groups.json` | 37052871 | `6310d7c4180b0d212d67eae757c57a8d4dbb290847d2c9a1017c3448d8983b00` | False | True |
| us_seattle_sidewalks candidates | `data/output/us_seattle_sidewalks_candidates.parquet` | 7779434 | `cc3fdc81dd290c6aeacb14d6f68f4df08a2889aa2c67250d241901c072a1bbe2` | False | True |
| us_seattle_sidewalks bridge | `data/output/us_seattle_sidewalks_bridge.parquet` | 2061803 | `02a49eb7c3e20c558ad72b34f0c8240960b9a66306254005e494c908df52e07d` | False | True |

| untracked workspace file | bytes | SHA256 |
|---|---:|---|
| `src/crosswalk/model_export.py` | 1605 | `4e2f29bf83c291a0bdf004f43a95455a1b7193133f97b68b5fed5749fe01e596` |
| `src/crosswalk/provenance.py` | 4770 | `a988e7877576f0e394166f2e06d4683f62530469570291c4ad6827de99983f84` |
| `tests/fixtures/optimizer_boston_corridor_rescue.json` | 15150 | `268d5eb35918b53b10d926c2214ac42039dfde865ea32a08df96afcf6bf55ef2` |
| `tests/fixtures/optimizer_boston_willow_decomposition.json` | 4046 | `7ee7350a97f3e03c2c0884fc3ff28ffb07ce6fb7abfb63dae71b7b0cc45d4095` |
| `tests/regression/test_optimizer_boston_corridor_rescue.py` | 4909 | `7168a70a40bda722d79a4b2708cb9f19cfd9656d51f503ea68d4a2ca150f8fbf` |
| `tests/unit/test_model_precedence.py` | 4103 | `1459c9aa9c32408240d94e4b7717f5c3139102282ee689562385c78780de3d75` |
| `tests/unit/test_pipeline_runner.py` | 887 | `4fbc3e1f23aa4143b24e118905f2646804d9559911cc3d6d7834d992e0c1d79d` |
| `tests/unit/test_source_commit_provenance.py` | 3593 | `5232210d6209fed8fe7ef44b4f3b35715e3998fb52a54ae44e3b12c7c118e7e4` |

> These model/output artifacts are ignored local files. A clean checkout does not contain them and cannot reproduce the inventory or metrics until byte-identical artifacts with the hashes above are restored. The source commit identifies the base checkout; the tracked source-patch hash identifies uncommitted tracked changes used for this run while deliberately excluding this generated report to avoid a self-referential digest. Every untracked, non-ignored workspace file is listed and content-hashed separately above.

## Inventory

- specs discovered: 13
  - co_bogota_bike_network: exists=True sidecar_groups=1380 labels=2 rows=5 cand_groups=2 legacy_groups=0 pos=5 neg=0 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=0 raw_rows=5 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - co_bogota_roads: exists=True sidecar_groups=29355 labels=8 rows=14 cand_groups=0 legacy_groups=3 pos=6 neg=8 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=1 raw_rows=14 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - de_berlin_roads: exists=True sidecar_groups=11210 labels=6 rows=8 cand_groups=4 legacy_groups=0 pos=8 neg=0 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=0 raw_rows=8 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - fi_helsinki_roads: exists=True sidecar_groups=17652 labels=5 rows=3 cand_groups=0 legacy_groups=1 pos=2 neg=1 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=1 raw_rows=3 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - ke_nairobi_roads: exists=True sidecar_groups=4751 labels=5 rows=24 cand_groups=0 legacy_groups=2 pos=10 neg=14 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=0 raw_rows=24 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - nl_amsterdam_roads: exists=True sidecar_groups=16629 labels=5 rows=7 cand_groups=3 legacy_groups=0 pos=6 neg=1 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=0 raw_rows=7 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - sg_singapore_footpaths: exists=True sidecar_groups=14567 labels=7 rows=17 cand_groups=0 legacy_groups=4 pos=10 neg=7 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=2 raw_rows=17 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - sg_singapore_roads: exists=True sidecar_groups=6355 labels=9 rows=44 cand_groups=6 legacy_groups=0 pos=43 neg=1 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=0 raw_rows=44 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - tn_tunis_ml_roads: exists=False sidecar_groups=0 labels=0 rows=0 cand_groups=0 legacy_groups=0 pos=0 neg=0 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=0 raw_rows=0 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - us_boston_streets: exists=True sidecar_groups=2934 labels=119 rows=558 cand_groups=107 legacy_groups=0 pos=413 neg=145 legacy_known_omit_occ=16 (clean=0,split=16) legacy_known_omit_unique_raw=14 (clean=0,split=14) legacy_known_omit_unique_retained=13 (clean=0,split=13) parquet_rows=9699 enriched=558 empty_rows=0 empty_legacy_skipped=0 raw_rows=690 duplicate_surplus_rows=67 duplicate_keys=67 conflicting_keys=30 quarantined_groups=3 quarantined_rows=130 deduplicated_rows=2
  - us_montana_missoula: exists=True sidecar_groups=2064 labels=3 rows=28 cand_groups=3 legacy_groups=0 pos=26 neg=2 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=0 raw_rows=28 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - us_seattle_sidewalks: exists=True sidecar_groups=6644 labels=49 rows=92 cand_groups=14 legacy_groups=0 pos=49 neg=43 legacy_known_omit_occ=1 (clean=0,split=1) legacy_known_omit_unique_raw=1 (clean=0,split=1) legacy_known_omit_unique_retained=1 (clean=0,split=1) parquet_rows=18236 enriched=92 empty_rows=0 empty_legacy_skipped=0 raw_rows=92 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
  - us_usfs_flathead: exists=True sidecar_groups=142 labels=5 rows=21 cand_groups=5 legacy_groups=0 pos=20 neg=1 legacy_known_omit_occ=0 (clean=0,split=0) legacy_known_omit_unique_raw=0 (clean=0,split=0) legacy_known_omit_unique_retained=0 (clean=0,split=0) parquet_rows=0 enriched=0 empty_rows=0 empty_legacy_skipped=0 raw_rows=21 duplicate_surplus_rows=0 duplicate_keys=0 conflicting_keys=0 quarantined_groups=0 quarantined_rows=0 deduplicated_rows=0
- combined: 821 edges / 147 groups
  - keep=1:598 keep=0:223
  - provenance: {'clean': 532, 'split': 289}

### In-sample (optimizer, naive, saved artifact) — optimistic for artifact

| model | edges | groups | P | R | F1 | grp_exact | F1_sliverfilt |
|---|---|---:|---:|---:|---:|---:|---:|
| optimizer (selected) | 821 | 147 | 0.867 | 0.957 | 0.909 | 0.701 | 0.909 |
| naive_keepall (all 1) | 821 | 147 | 0.728 | 1.0 | 0.843 | 0.571 | 0.849 |
| conf>=0.5 | 821 | 147 | 0.758 | 0.982 | 0.856 | 0.592 | 0.856 |
| conf_oracle(t=0.90, optimistic in-sample) | 821 | 147 | 0.847 | 0.945 | 0.893 | 0.694 | 0.893 |
| saved_artifact_thr0.5 (in-sample) | 821 | 147 | 0.945 | 0.863 | 0.902 | 0.694 | 0.902 |
| saved_artifact_ef1 (in-sample) | 821 | 147 | 0.918 | 0.938 | 0.928 | 0.741 | 0.928 |

### Freshly retrained candidate architecture: grouped CV OOF (5 folds) — fold-held-out

| model | edges | groups | P | R | F1 | grp_exact | F1_sliverfilt |
|---|---|---:|---:|---:|---:|---:|---:|
| xgb[extfeats]+ef1 | 821 | 147 | 0.882 | 0.925 | 0.903 | 0.721 | 0.903 |
| baseline: optimizer+prune (selected) | 821 | 147 | 0.867 | 0.957 | 0.909 | 0.701 | 0.909 |
| baseline: conf>=0.90 (oracle-tuned) | 821 | 147 | 0.847 | 0.945 | 0.893 | 0.694 | 0.893 |

### Repeated stratified grouped CV

Every edge is held out once per seed; rows from one dataset-scoped group never cross a fold. Each seed is one CV realization pooling predictions from five separately fitted fold models.

| seed | model F1 | Δ vs prod | model exact | Δ vs prod |
|---:|---:|---:|---:|---:|
| 0 | 0.8945 | -0.0149 | 0.7075 | +0.0068 |
| 1 | 0.8952 | -0.0142 | 0.7279 | +0.0272 |
| 2 | 0.9057 | -0.0037 | 0.7007 | +0.0000 |
| 3 | 0.8912 | -0.0181 | 0.7279 | +0.0272 |
| 4 | 0.8967 | -0.0127 | 0.7415 | +0.0408 |

Paired whole-group bootstrap on the five-seed mean-OOF probability ensemble decision (a research ensemble, not one deployable full-data model):

| metric | observed Δ | 95% CI | bootstrap support (Δ > 0) |
|---|---:|---:|---:|
| f1 | -0.0095 | [-0.0330, +0.0112] | 0.228 |
| group_exact | +0.0272 | [-0.0410, +0.1020] | 0.774 |

### Leave-one-dataset-out transfer

| held-out dataset | edges | groups | model F1 | prod F1 | model exact | prod exact |
|---|---:|---:|---:|---:|---:|---:|
| co_bogota_bike_network | 5 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| co_bogota_roads | 14 | 3 | 1.0000 | 0.9231 | 1.0000 | 0.6667 |
| de_berlin_roads | 8 | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| fi_helsinki_roads | 3 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ke_nairobi_roads | 24 | 2 | 0.5714 | 1.0000 | 0.0000 | 1.0000 |
| nl_amsterdam_roads | 7 | 3 | 1.0000 | 0.9231 | 1.0000 | 0.6667 |
| sg_singapore_footpaths | 17 | 4 | 0.8000 | 1.0000 | 0.2500 | 1.0000 |
| sg_singapore_roads | 44 | 6 | 1.0000 | 0.9885 | 1.0000 | 0.8333 |
| us_boston_streets | 558 | 100 | 0.8899 | 0.9001 | 0.8000 | 0.6900 |
| us_montana_missoula | 28 | 3 | 0.9600 | 0.9811 | 0.6667 | 0.6667 |
| us_seattle_sidewalks | 92 | 14 | 0.7379 | 0.8000 | 0.4286 | 0.4286 |
| us_usfs_flathead | 21 | 5 | 0.9474 | 0.9756 | 0.8000 | 0.8000 |

Paired dataset-cluster bootstrap over pooled LODO predictions:

| metric | observed Δ | 95% CI | bootstrap support (Δ > 0) |
|---|---:|---:|---:|
| f1 | -0.0240 | [-0.1132, -0.0074] | 0.011 |
| group_exact | +0.0612 | [-0.1905, +0.1092] | 0.753 |

### Per-dataset in-sample

| model | edges | groups | P | R | F1 | grp_exact | F1_sliverfilt |
|---|---|---:|---:|---:|---:|---:|---:|
| optimizer:co_bogota_bike_network | 5 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| naive_keepall:co_bogota_bike_network | 5 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| conf_oracle(t=0.30):co_bogota_bike_network | 5 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_thr0.5:co_bogota_bike_network | 5 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_ef1:co_bogota_bike_network | 5 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| optimizer:co_bogota_roads | 14 | 3 | 0.857 | 1.0 | 0.923 | 0.667 | 0.923 |
| naive_keepall:co_bogota_roads | 14 | 3 | 0.429 | 1.0 | 0.6 | 0.0 | 0.8 |
| conf_oracle(t=0.52):co_bogota_roads | 14 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_thr0.5:co_bogota_roads | 14 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_ef1:co_bogota_roads | 14 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| optimizer:de_berlin_roads | 8 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| naive_keepall:de_berlin_roads | 8 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| conf_oracle(t=0.30):de_berlin_roads | 8 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_thr0.5:de_berlin_roads | 8 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_ef1:de_berlin_roads | 8 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| optimizer:fi_helsinki_roads | 3 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| naive_keepall:fi_helsinki_roads | 3 | 1 | 0.667 | 1.0 | 0.8 | 0.0 | 0.8 |
| conf_oracle(t=0.56):fi_helsinki_roads | 3 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_thr0.5:fi_helsinki_roads | 3 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_ef1:fi_helsinki_roads | 3 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| optimizer:ke_nairobi_roads | 24 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| naive_keepall:ke_nairobi_roads | 24 | 2 | 0.417 | 1.0 | 0.588 | 0.0 | 0.588 |
| conf_oracle(t=0.56):ke_nairobi_roads | 24 | 2 | 0.571 | 0.8 | 0.667 | 0.0 | 0.667 |
| saved_artifact_thr0.5:ke_nairobi_roads | 24 | 2 | 0.667 | 0.2 | 0.308 | 0.0 | 0.308 |
| saved_artifact_ef1:ke_nairobi_roads | 24 | 2 | 0.778 | 0.7 | 0.737 | 0.0 | 0.737 |
| optimizer:nl_amsterdam_roads | 7 | 3 | 0.857 | 1.0 | 0.923 | 0.667 | 0.923 |
| naive_keepall:nl_amsterdam_roads | 7 | 3 | 0.857 | 1.0 | 0.923 | 0.667 | 0.923 |
| conf_oracle(t=0.65):nl_amsterdam_roads | 7 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_thr0.5:nl_amsterdam_roads | 7 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_ef1:nl_amsterdam_roads | 7 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| optimizer:sg_singapore_footpaths | 17 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| naive_keepall:sg_singapore_footpaths | 17 | 4 | 0.588 | 1.0 | 0.741 | 0.0 | 0.741 |
| conf_oracle(t=0.82):sg_singapore_footpaths | 17 | 4 | 0.714 | 1.0 | 0.833 | 0.5 | 0.833 |
| saved_artifact_thr0.5:sg_singapore_footpaths | 17 | 4 | 0.714 | 1.0 | 0.833 | 0.5 | 0.833 |
| saved_artifact_ef1:sg_singapore_footpaths | 17 | 4 | 0.667 | 1.0 | 0.8 | 0.25 | 0.8 |
| optimizer:sg_singapore_roads | 44 | 6 | 0.977 | 1.0 | 0.989 | 0.833 | 0.989 |
| naive_keepall:sg_singapore_roads | 44 | 6 | 0.977 | 1.0 | 0.989 | 0.833 | 0.989 |
| conf_oracle(t=0.30):sg_singapore_roads | 44 | 6 | 1.0 | 0.977 | 0.988 | 0.833 | 0.988 |
| saved_artifact_thr0.5:sg_singapore_roads | 44 | 6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_ef1:sg_singapore_roads | 44 | 6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| optimizer:us_boston_streets | 558 | 100 | 0.856 | 0.949 | 0.9 | 0.69 | 0.9 |
| naive_keepall:us_boston_streets | 558 | 100 | 0.74 | 1.0 | 0.851 | 0.64 | 0.854 |
| conf_oracle(t=0.90):us_boston_streets | 558 | 100 | 0.846 | 0.971 | 0.904 | 0.73 | 0.904 |
| saved_artifact_thr0.5:us_boston_streets | 558 | 100 | 0.97 | 0.857 | 0.91 | 0.69 | 0.91 |
| saved_artifact_ef1:us_boston_streets | 558 | 100 | 0.931 | 0.947 | 0.939 | 0.77 | 0.939 |
| optimizer:us_montana_missoula | 28 | 3 | 0.963 | 1.0 | 0.981 | 0.667 | 0.981 |
| naive_keepall:us_montana_missoula | 28 | 3 | 0.929 | 1.0 | 0.963 | 0.333 | 0.981 |
| conf_oracle(t=0.65):us_montana_missoula | 28 | 3 | 1.0 | 0.962 | 0.98 | 0.667 | 0.98 |
| saved_artifact_thr0.5:us_montana_missoula | 28 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| saved_artifact_ef1:us_montana_missoula | 28 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| optimizer:us_seattle_sidewalks | 92 | 14 | 0.721 | 0.898 | 0.8 | 0.429 | 0.8 |
| naive_keepall:us_seattle_sidewalks | 92 | 14 | 0.533 | 1.0 | 0.695 | 0.143 | 0.695 |
| conf_oracle(t=0.78):us_seattle_sidewalks | 92 | 14 | 0.635 | 0.959 | 0.764 | 0.357 | 0.764 |
| saved_artifact_thr0.5:us_seattle_sidewalks | 92 | 14 | 0.725 | 0.755 | 0.74 | 0.429 | 0.74 |
| saved_artifact_ef1:us_seattle_sidewalks | 92 | 14 | 0.741 | 0.816 | 0.777 | 0.429 | 0.777 |
| optimizer:us_usfs_flathead | 21 | 5 | 0.952 | 1.0 | 0.976 | 0.8 | 0.976 |
| naive_keepall:us_usfs_flathead | 21 | 5 | 0.952 | 1.0 | 0.976 | 0.8 | 0.976 |
| conf_oracle(t=0.30):us_usfs_flathead | 21 | 5 | 0.947 | 0.9 | 0.923 | 0.6 | 0.923 |
| saved_artifact_thr0.5:us_usfs_flathead | 21 | 5 | 1.0 | 0.85 | 0.919 | 0.6 | 0.919 |
| saved_artifact_ef1:us_usfs_flathead | 21 | 5 | 1.0 | 0.85 | 0.919 | 0.6 | 0.919 |

## Interpretation / limitations

- Four labeled datasets still use legacy sidecars without `candidate_edges`, so their under-selection universe is capped (64/group); Tunis is missing entirely.
- Legacy reject-all labels on legacy groups emit zero rows (honest cross-mode handling via `empty_legacy_skipped`).
- In-sample model numbers are optimistic (data leakage). Use repeated grouped CV, its paired whole-group interval, and LODO transfer for the NO-GO decision.
- `optimizer (selected)` is the final production sidecar assignment after corridor-aware grouping, symmetric coverage validation, strict adjacent-alignment/name rescue, decomposition/review policy, and the dataset-tuned confidence prune. It is an edge-set comparator: MATCH versus REVIEW publication decisions are not scored separately.
- `naive_keepall` = every candidate edge kept — precision floor, recall ceiling.
- Fold-held-out OOF F1 remains below production, while group exact is modestly higher. This is a useful candidate architecture, not a production resolver.
- The OOF rows hold label groups out of each fit, but they are not an untouched or nested model-selection estimate: the 33-feature manifest and eF1 selector were developed on overlapping historical versions of this label corpus. Any apparent candidate gain is therefore still exploratory (and potentially optimistic); this limitation only strengthens the present NO-GO.
- Typed `<ds>_candidates.parquet` data is joined fail-closed and was locally available for: us_boston_streets, us_seattle_sidewalks; the model still uses 33 sidecar/context features.
- Legacy-known omission audit (not a universal candidate-recall claim): 17 historical-label occurrences (clean 0, split 17), 15 unique raw current-group/edge keys (clean 0, split 15), and 14 unique keys after collision quarantine (clean 0, split 14). These count only human-selected edges known to the mapped group's legacy `edges`/`rejected_edges` view but absent from its emitted candidate universe; split labels can legitimately contain edges owned by another current group. Dataset detail: us_boston_streets occ=16/raw_unique=14/retained_unique=13, us_seattle_sidewalks occ=1/raw_unique=1/retained_unique=1.
- Label-integrity audit: primary metrics quarantine 3 current group(s) / 130 raw row(s) with 30 contradictory edge key(s), then collapse 2 remaining cross-historical same-truth duplicate row(s); the retained table has unique candidate keys. The quarantine is 130/953 (13.6%) of emitted raw edge-row occurrences. CV/LODO therefore estimate performance only on the adjudication-clean subset, not all mapped labels. Quarantined groups require human adjudication plus a sensitivity analysis before any population-level promotion claim.
- Comparator identity: `saved_artifact_*` rows score the supplied joblib fit and are optimistic/in-sample. Grouped CV and LODO freshly fit the XGBoost candidate architecture from the artifact's 33-column feature manifest and eF1 selector, without the artifact's historical soft-vote training extras.
- Repeated grouped CV averages five held-out probabilities per edge and reports the resulting five-seed mean-OOF ensemble. Each seed row is one CV realization (five fold fits); the mean-OOF ensemble is not a single deployable full-data fit.
- Next evidence gate: adjudicate the contradictory groups and legacy-known split omission keys—correcting ground truth where source-segmentation research warrants it—then label ≥20 reject-all groups.
- Then run paired grouped-CV removal/permutation ablations by feature family; add panel votes only when candidate-display provenance makes unselected edges and NONE votes interpretable.
