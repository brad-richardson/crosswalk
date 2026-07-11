# Resolver Benchmark — optimizer vs naive vs model

> Research-only. Compares M:N group edge selection strategies on curated stitching labels.
> data_root=. labels=labels/stitching model=data/models/resolver_model.joblib n_splits=5

## Inventory

- specs discovered: 13
  - co_bogota_bike_network: exists=True sidecar_groups=1359 labels=2 rows=14 cand_groups=0 legacy_groups=2 pos=5 neg=9 empty_rows=0 empty_legacy_skipped=0
  - co_bogota_roads: exists=True sidecar_groups=29355 labels=8 rows=14 cand_groups=0 legacy_groups=3 pos=6 neg=8 empty_rows=0 empty_legacy_skipped=1
  - de_berlin_roads: exists=False sidecar_groups=0 labels=0 rows=0 cand_groups=0 legacy_groups=0 pos=0 neg=0 empty_rows=0 empty_legacy_skipped=0
  - fi_helsinki_roads: exists=True sidecar_groups=17652 labels=5 rows=3 cand_groups=0 legacy_groups=1 pos=2 neg=1 empty_rows=0 empty_legacy_skipped=1
  - ke_nairobi_roads: exists=True sidecar_groups=4751 labels=5 rows=24 cand_groups=0 legacy_groups=2 pos=10 neg=14 empty_rows=0 empty_legacy_skipped=0
  - nl_amsterdam_roads: exists=True sidecar_groups=15481 labels=5 rows=7 cand_groups=0 legacy_groups=3 pos=6 neg=1 empty_rows=0 empty_legacy_skipped=1
  - sg_singapore_footpaths: exists=True sidecar_groups=14567 labels=7 rows=17 cand_groups=0 legacy_groups=4 pos=10 neg=7 empty_rows=0 empty_legacy_skipped=2
  - sg_singapore_roads: exists=True sidecar_groups=6121 labels=9 rows=47 cand_groups=0 legacy_groups=6 pos=43 neg=4 empty_rows=0 empty_legacy_skipped=0
  - tn_tunis_ml_roads: exists=False sidecar_groups=0 labels=0 rows=0 cand_groups=0 legacy_groups=0 pos=0 neg=0 empty_rows=0 empty_legacy_skipped=0
  - us_boston_streets: exists=True sidecar_groups=3365 labels=119 rows=809 cand_groups=0 legacy_groups=111 pos=501 neg=308 empty_rows=0 empty_legacy_skipped=0
  - us_montana_missoula: exists=True sidecar_groups=1914 labels=3 rows=30 cand_groups=0 legacy_groups=3 pos=26 neg=4 empty_rows=0 empty_legacy_skipped=0
  - us_seattle_sidewalks: exists=False sidecar_groups=0 labels=0 rows=0 cand_groups=0 legacy_groups=0 pos=0 neg=0 empty_rows=0 empty_legacy_skipped=0
  - us_usfs_flathead: exists=True sidecar_groups=131 labels=5 rows=25 cand_groups=0 legacy_groups=5 pos=20 neg=5 empty_rows=0 empty_legacy_skipped=0
- combined: 990 edges / 137 groups
  - keep=1:629 keep=0:361
  - provenance: {'clean': 599, 'split': 391}

### In-sample (optimizer vs naive vs model) — optimistic for model

| model | edges | groups | P | R | F1 | grp_exact | F1_sliverfilt |
|---|---|---:|---:|---:|---:|---:|---:|
| optimizer (selected) | 990 | 137 | 0.849 | 0.94 | 0.892 | 0.708 | 0.896 |
| naive_keepall (all 1) | 990 | 137 | 0.635 | 1.0 | 0.777 | 0.423 | 0.794 |
| conf>=0.5 | 990 | 137 | 0.734 | 0.96 | 0.832 | 0.562 | 0.836 |
| conf_oracle t=0.94 | 990 | 137 | 0.842 | 0.916 | 0.877 | 0.672 | 0.881 |
| model_thr0.5 (in-sample) | 990 | 137 | 0.967 | 0.887 | 0.925 | 0.73 | 0.93 |
| model_ef1 (in-sample) | 990 | 137 | 0.926 | 0.941 | 0.934 | 0.745 | 0.938 |

### Grouped CV OOF (5 folds) — honest

| model | edges | groups | P | R | F1 | grp_exact | F1_sliverfilt |
|---|---|---:|---:|---:|---:|---:|---:|
| xgb[extfeats]+ef1 | 990 | 137 | 0.853 | 0.903 | 0.877 | 0.664 | 0.881 |
| baseline: optimizer+prune (selected) | 990 | 137 | 0.849 | 0.94 | 0.892 | 0.708 | 0.896 |
| baseline: conf>=0.94 (oracle-tuned) | 990 | 137 | 0.842 | 0.916 | 0.877 | 0.672 | 0.881 |

### Per-dataset in-sample

| model | edges | groups | P | R | F1 | grp_exact | F1_sliverfilt |
|---|---|---:|---:|---:|---:|---:|---:|
| optimizer:co_bogota_bike_network | 14 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| naive_keepall:co_bogota_bike_network | 14 | 2 | 0.357 | 1.0 | 0.526 | 0.5 | 0.526 |
| conf_oracle(t=0.94):co_bogota_bike_network | 14 | 2 | 0.625 | 1.0 | 0.769 | 0.5 | 0.769 |
| model_thr0.5:co_bogota_bike_network | 14 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| model_ef1:co_bogota_bike_network | 14 | 2 | 0.833 | 1.0 | 0.909 | 0.5 | 0.909 |
| optimizer:co_bogota_roads | 14 | 3 | 0.857 | 1.0 | 0.923 | 0.667 | 0.923 |
| naive_keepall:co_bogota_roads | 14 | 3 | 0.429 | 1.0 | 0.6 | 0.0 | 0.8 |
| conf_oracle(t=0.52):co_bogota_roads | 14 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| model_thr0.5:co_bogota_roads | 14 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| model_ef1:co_bogota_roads | 14 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| optimizer:fi_helsinki_roads | 3 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| naive_keepall:fi_helsinki_roads | 3 | 1 | 0.667 | 1.0 | 0.8 | 0.0 | 0.8 |
| conf_oracle(t=0.56):fi_helsinki_roads | 3 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| model_thr0.5:fi_helsinki_roads | 3 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| model_ef1:fi_helsinki_roads | 3 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| optimizer:ke_nairobi_roads | 24 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| naive_keepall:ke_nairobi_roads | 24 | 2 | 0.417 | 1.0 | 0.588 | 0.0 | 0.588 |
| conf_oracle(t=0.56):ke_nairobi_roads | 24 | 2 | 0.571 | 0.8 | 0.667 | 0.0 | 0.667 |
| model_thr0.5:ke_nairobi_roads | 24 | 2 | 0.833 | 0.5 | 0.625 | 0.5 | 0.625 |
| model_ef1:ke_nairobi_roads | 24 | 2 | 0.889 | 0.8 | 0.842 | 0.5 | 0.842 |
| optimizer:nl_amsterdam_roads | 7 | 3 | 0.857 | 1.0 | 0.923 | 0.667 | 0.923 |
| naive_keepall:nl_amsterdam_roads | 7 | 3 | 0.857 | 1.0 | 0.923 | 0.667 | 0.923 |
| conf_oracle(t=0.52):nl_amsterdam_roads | 7 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| model_thr0.5:nl_amsterdam_roads | 7 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| model_ef1:nl_amsterdam_roads | 7 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| optimizer:sg_singapore_footpaths | 17 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| naive_keepall:sg_singapore_footpaths | 17 | 4 | 0.588 | 1.0 | 0.741 | 0.0 | 0.741 |
| conf_oracle(t=0.82):sg_singapore_footpaths | 17 | 4 | 0.714 | 1.0 | 0.833 | 0.5 | 0.833 |
| model_thr0.5:sg_singapore_footpaths | 17 | 4 | 0.833 | 1.0 | 0.909 | 0.75 | 0.909 |
| model_ef1:sg_singapore_footpaths | 17 | 4 | 0.714 | 1.0 | 0.833 | 0.5 | 0.833 |
| optimizer:sg_singapore_roads | 47 | 6 | 0.977 | 1.0 | 0.989 | 0.833 | 0.989 |
| naive_keepall:sg_singapore_roads | 47 | 6 | 0.915 | 1.0 | 0.956 | 0.667 | 0.977 |
| conf_oracle(t=0.34):sg_singapore_roads | 47 | 6 | 1.0 | 0.977 | 0.988 | 0.833 | 0.988 |
| model_thr0.5:sg_singapore_roads | 47 | 6 | 1.0 | 0.977 | 0.988 | 0.833 | 0.988 |
| model_ef1:sg_singapore_roads | 47 | 6 | 1.0 | 0.977 | 0.988 | 0.833 | 0.988 |
| optimizer:us_boston_streets | 809 | 108 | 0.822 | 0.924 | 0.87 | 0.676 | 0.875 |
| naive_keepall:us_boston_streets | 809 | 108 | 0.619 | 1.0 | 0.765 | 0.454 | 0.781 |
| conf_oracle(t=0.94):us_boston_streets | 809 | 108 | 0.822 | 0.924 | 0.87 | 0.676 | 0.875 |
| model_thr0.5:us_boston_streets | 809 | 108 | 0.965 | 0.876 | 0.918 | 0.704 | 0.924 |
| model_ef1:us_boston_streets | 809 | 108 | 0.92 | 0.936 | 0.928 | 0.741 | 0.933 |
| optimizer:us_montana_missoula | 30 | 3 | 0.963 | 1.0 | 0.981 | 0.667 | 0.981 |
| naive_keepall:us_montana_missoula | 30 | 3 | 0.867 | 1.0 | 0.929 | 0.0 | 0.945 |
| conf_oracle(t=0.47):us_montana_missoula | 30 | 3 | 1.0 | 0.962 | 0.98 | 0.667 | 0.98 |
| model_thr0.5:us_montana_missoula | 30 | 3 | 1.0 | 0.962 | 0.98 | 0.667 | 0.98 |
| model_ef1:us_montana_missoula | 30 | 3 | 1.0 | 0.962 | 0.98 | 0.667 | 0.98 |
| optimizer:us_usfs_flathead | 25 | 5 | 0.952 | 1.0 | 0.976 | 0.8 | 0.976 |
| naive_keepall:us_usfs_flathead | 25 | 5 | 0.8 | 1.0 | 0.889 | 0.4 | 0.889 |
| conf_oracle(t=0.47):us_usfs_flathead | 25 | 5 | 1.0 | 0.9 | 0.947 | 0.8 | 0.947 |
| model_thr0.5:us_usfs_flathead | 25 | 5 | 1.0 | 0.9 | 0.947 | 0.8 | 0.947 |
| model_ef1:us_usfs_flathead | 25 | 5 | 1.0 | 0.95 | 0.974 | 0.8 | 0.974 |

## Interpretation / limitations

- Factory sidecars (`release=2026-06-17.0`) have no `candidate_edges`, so under-selection is capped (64/group).
- Legacy reject-all labels on legacy groups emit zero rows (honest cross-mode handling via `empty_legacy_skipped`).
- In-sample model numbers are optimistic (data leakage). Use the Grouped CV OOF block for the NO-GO decision.
- `optimizer (selected)` = sidecar `selected` flag: keep-all + vertex-junction tolerance + confidence prune (#284).
- `naive_keepall` = every candidate edge kept — precision floor, recall ceiling.
- Production currently beats learned on clean slices at small label scale (see round2/round3 reports);
  model needs P1 `<ds>_candidates.parquet` (78 feats + signed lateral offset) + fresh stitch
  + cross-mode empty testset ≥20 to have a fair shot.
