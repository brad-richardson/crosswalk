# Learned Stitcher Round 3 — candidate-joined experimental ablation

> **Superseded evidence snapshot.** This 949-edge/163-group run predates collision
> quarantine, the final Boston/Seattle #424 artifacts, and the corrected omission
> accounting. The canonical current comparison is
> [`resolver_benchmark.md`](resolver_benchmark.md); the historical numbers below
> remain only to document the experiment's evolution.

> Experimental only — not wired into production. This is a meaningful training and
> evaluation prototype, but the current evidence is a production NO-GO. Draft PR #411
> invalidated the proposed heuristic defaults on a fixed label universe, and this model
> still trails the production optimizer on its historical grouped-CV edge F1.

## Inventory

- Feature version: `2026-07-07.2`
- Feature set used by the model: 25 sidecar + 8 competition/coverage features (33 total)
- Selector: expected-F1 (`ef1`)
- Hard labels: 949 edges (675 keep / 274 drop), 163 groups, 12 datasets
- Provenance: 637 clean / 312 split
- Candidate graph coverage: 156 groups; 10 groups still use legacy capped sidecars
- Soft panel rows: 0; votes are opt-in and were disabled for this benchmark

### Per-dataset build stats

| dataset | groups | labels | rows | candidate | legacy | parquet rows | enriched | outside candidate | pos | neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| co_bogota_bike_network | 1,380 | 2 | 5 | 2 | 0 | 0 | 0 | 0 | 5 | 0 |
| co_bogota_roads | 29,355 | 8 | 14 | 0 | 3 | 0 | 0 | 0 | 6 | 8 |
| de_berlin_roads | 11,210 | 6 | 8 | 4 | 0 | 0 | 0 | 0 | 8 | 0 |
| fi_helsinki_roads | 17,652 | 5 | 3 | 0 | 1 | 0 | 0 | 0 | 2 | 1 |
| ke_nairobi_roads | 4,751 | 5 | 24 | 0 | 2 | 0 | 0 | 0 | 10 | 14 |
| nl_amsterdam_roads | 16,629 | 5 | 7 | 3 | 0 | 0 | 0 | 0 | 6 | 1 |
| sg_singapore_footpaths | 14,567 | 7 | 17 | 0 | 4 | 0 | 0 | 0 | 10 | 7 |
| sg_singapore_roads | 6,355 | 9 | 44 | 6 | 0 | 0 | 0 | 0 | 43 | 1 |
| tn_tunis_ml_roads | missing | - | - | - | - | - | - | - | - | - |
| us_boston_streets | 3,649 | 119 | 679 | 111 | 0 | 0 | 0 | 20 | 482 | 197 |
| us_montana_missoula | 2,064 | 3 | 28 | 3 | 0 | 0 | 0 | 0 | 26 | 2 |
| us_seattle_sidewalks | 19,114 | 49 | 99 | 22 | 0 | 34,401 | 99 | 6 | 57 | 42 |
| us_usfs_flathead | 142 | 5 | 21 | 5 | 0 | 0 | 0 | 0 | 20 | 1 |

Seattle's persisted candidate parquet joined all 99 labeled rows to all 83 typed
candidate columns with zero missing keys. It is the only typed candidate parquet
available locally in this run, so the 83-feature bundle is not yet in the model.

## Honest grouped-CV result

| model | edges | groups | precision | recall | F1 | group exact |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost, 33 features + eF1 | 949 | 163 | 0.853 | 0.899 | 0.875 | 0.748 |
| Production optimizer | 949 | 163 | 0.845 | 0.960 | 0.899 | 0.730 |
| Oracle-tuned confidence ≥0.98 | 949 | 163 | 0.858 | 0.911 | 0.884 | 0.687 |

The learned prototype trades 6.1 points of recall for 0.8 points of precision. It
loses 0.023 edge F1 to production while gaining 0.018 group exact. In-sample eF1 is
0.946, so the gap between fit and OOF performance is a strong small-data/overfit signal.

## Decision and next evidence gates

1. Keep the model research-only. Do not wire it into production or ship the heuristic
   thresholds from the gap-analysis hypotheses.
2. Regenerate fresh candidate graphs and typed parquets for the four legacy datasets;
   add Tunis. Nairobi currently reaches relational feature generation but does not
   finish in a practical local run, so profile/chunk that stage before bulk regen.
3. Fix candidate recall before model tuning: 20 Boston and 6 Seattle labeled positives
   are outside the current candidate universe.
4. Label at least 20 reliably mapped reject-all groups, then add 150–300 diverse groups
   emphasizing non-US M:N ambiguity and the low-confidence/parallel-sibling failure modes.
5. Run paired grouped-CV removal and permutation ablations by candidate-feature family
   (`lateral`, geometry distance, names, class/length, graph/competition), with dataset
   holdouts and bootstrap intervals. Promote families, not isolated correlated columns.
6. Record the exact candidate set shown to every voting provider. Until then, selected
   edges are usable positive evidence, but unseen/unselected edges and NONE votes cannot
   safely become negatives.

See `research/feature_ablation_strategy_2026-07-11.md` for the removal/permutation gate
and `research/resolver_gap_analysis_2026-07-11.md` for the underlying failure analysis.
