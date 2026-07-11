# Learned Stitcher Round 3 — experimental (all datasets, eF1+extended+soft)

> Experimental only — not wired into production. Produces `data/models/resolver_model.joblib`
> and a prototype eval. Uses legacy sidecars from `data/factory/release=2026-06-17.0` when
> `data/output/*.json` is absent, so under-selection is partially capped (64/group).

## Inventory

- data_root: `.`
- feature_version: 2026-07-07.2
- feature set: extended (33) cols: confidence, conf_rel_max, conf_rel_mean, conf_rank_frac, conf_is_group_min, gers_span, local_span, max_span…
- selector: ef1
- total edge rows (hard labels): 990
  - keep=1: 629 / keep=0: 361
  - groups: 137 / datasets: 10
  - provenance: {'clean': 599, 'split': 391}
- soft extra rows: 561 (featurized groups not in hard set)

### Per-dataset build stats

| dataset | sidecar groups | labels | rows | candidate_groups | legacy_groups | pos | neg | empty_rows | empty_legacy_skipped |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| co_bogota_bike_network | 1359 | 2 | 14 | 0 | 2 | 5 | 9 | 0 | 0 |
| co_bogota_roads | 29355 | 8 | 14 | 0 | 3 | 6 | 8 | 0 | 1 |
| de_berlin_roads | MISSING | - | - | - | - | - | - | - | - |
| fi_helsinki_roads | 17652 | 5 | 3 | 0 | 1 | 2 | 1 | 0 | 1 |
| ke_nairobi_roads | 4751 | 5 | 24 | 0 | 2 | 10 | 14 | 0 | 0 |
| nl_amsterdam_roads | 15481 | 5 | 7 | 0 | 3 | 6 | 1 | 0 | 1 |
| sg_singapore_footpaths | 14567 | 7 | 17 | 0 | 4 | 10 | 7 | 0 | 2 |
| sg_singapore_roads | 6121 | 9 | 47 | 0 | 6 | 43 | 4 | 0 | 0 |
| tn_tunis_ml_roads | MISSING | - | - | - | - | - | - | - | - |
| us_boston_streets | 3365 | 119 | 809 | 0 | 111 | 501 | 308 | 0 | 0 |
| us_montana_missoula | 1914 | 3 | 30 | 0 | 3 | 26 | 4 | 0 | 0 |
| us_seattle_sidewalks | MISSING | - | - | - | - | - | - | - | - |
| us_usfs_flathead | 131 | 5 | 25 | 0 | 5 | 20 | 5 | 0 | 0 |

- total candidate_groups=0 legacy_groups=140

## Eval (grouped CV, out-of-fold)

- model: {'model': 'xgb[extfeats]+ef1+soft', 'edges': 990, 'groups': 137, 'P': 0.867, 'R': 0.9, 'F1': 0.883, 'grp_exact': 0.672, 'F1_sliverfilt': 0.887}
- baseline_production: {'model': 'baseline: optimizer+prune (selected)', 'edges': 990, 'groups': 137, 'P': 0.849, 'R': 0.94, 'F1': 0.892, 'grp_exact': 0.708, 'F1_sliverfilt': 0.896}
- baseline_conf_oracle: {'model': 'baseline: conf>=0.94 (oracle-tuned)', 'edges': 990, 'groups': 137, 'P': 0.842, 'R': 0.916, 'F1': 0.877, 'grp_exact': 0.672, 'F1_sliverfilt': 0.881}

- oof_proba: mean=0.556

**Headline vs production:** model F1=0.883 P=0.867 R=0.900 grp_exact=0.672 | baseline F1=0.892 P=0.849 R=0.940 exact=0.708

## Limitations / next steps

- P1 parquet `<ds>_candidates.parquet` with 78 typed pair features + signed lateral offset + class/length
  is NOT yet persisted — model uses only 26 sidecar + 8 competition/coverage features.
- Factory sidecars old → no `candidate_edges`, so under-selection positives under-counted (legacy path uses edges+rejected_edges capped 64).
- Fresh `crosswalk stitch` with `stitch_persist_candidate_graph=True` needed for full-candidate training.
- Cross-mode testset (Bogotá bike + SG footpaths NONE) needs ≥20 empty labels held out; currently partial.
- De-anchored slice `deanchored_v1` exists (51 groups) — sliced in next eval.
- If GO (beats tuned prune on clean + group exact), follow `research/learned_optimizer_design.md` I1
  runtime behind `learned_resolver_overrides` + shadow `resolver_score` + S1 Spark export.
