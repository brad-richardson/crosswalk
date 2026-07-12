# Resolver benchmark — optimizer vs naive vs learned

> Research-only, regenerated 2026-07-12 from 949 curated edge labels in 163
> dataset-scoped groups across 12 datasets. The model is evaluated with five-fold
> grouped CV; in-sample model numbers are shown only as an overfit diagnostic.

## Inventory

| dataset | labeled rows | mapped groups | candidate groups | legacy groups | outside candidate | typed parquet |
|---|---:|---:|---:|---:|---:|---:|
| co_bogota_bike_network | 5 | 2 | 2 | 0 | 0 | 0 |
| co_bogota_roads | 14 | 3 | 0 | 3 | 0 | 0 |
| de_berlin_roads | 8 | 4 | 4 | 0 | 0 | 0 |
| fi_helsinki_roads | 3 | 1 | 0 | 1 | 0 | 0 |
| ke_nairobi_roads | 24 | 2 | 0 | 2 | 0 | 0 |
| nl_amsterdam_roads | 7 | 3 | 3 | 0 | 0 | 0 |
| sg_singapore_footpaths | 17 | 4 | 0 | 4 | 0 | 0 |
| sg_singapore_roads | 44 | 6 | 6 | 0 | 0 | 0 |
| tn_tunis_ml_roads | missing | - | - | - | - | - |
| us_boston_streets | 679 | 108 | 111 | 0 | 20 | 0 |
| us_montana_missoula | 28 | 3 | 3 | 0 | 0 | 0 |
| us_seattle_sidewalks | 99 | 22 | 22 | 0 | 6 | 34,401 |
| us_usfs_flathead | 21 | 5 | 5 | 0 | 0 | 0 |

Combined labels contain 675 keep and 274 drop edges (637 clean / 312 split).
Seattle's typed parquet enriched all 99 labeled rows with zero missing join keys.

## In-sample comparison

| model | precision | recall | F1 | group exact |
|---|---:|---:|---:|---:|
| Production optimizer | 0.845 | 0.960 | 0.899 | 0.730 |
| Naive keep-all | 0.711 | 1.000 | 0.831 | 0.589 |
| Confidence ≥0.5 | 0.736 | 0.979 | 0.840 | 0.607 |
| Oracle confidence ≥0.98 | 0.858 | 0.911 | 0.884 | 0.687 |
| Learned threshold 0.5 | 0.976 | 0.913 | 0.943 | 0.798 |
| Learned expected-F1 | 0.937 | 0.954 | 0.946 | 0.798 |

The model's in-sample lead does not survive held-out groups.

## Grouped-CV OOF — decision metric

| model | precision | recall | F1 | group exact |
|---|---:|---:|---:|---:|
| Learned XGBoost + expected-F1 | 0.853 | 0.899 | 0.875 | 0.748 |
| Production optimizer | 0.845 | 0.960 | 0.899 | 0.730 |
| Oracle confidence ≥0.98 | 0.858 | 0.911 | 0.884 | 0.687 |

Decision: production wins edge F1 by 0.023; the model wins group exact by 0.018.
This supports keeping the training/evaluation harness as a prototype and rejects a
production resolver flip at the current label and feature coverage.

## Production baseline by dataset

| dataset | edges | groups | precision | recall | F1 | group exact |
|---|---:|---:|---:|---:|---:|---:|
| co_bogota_bike_network | 5 | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| co_bogota_roads | 14 | 3 | 0.857 | 1.000 | 0.923 | 0.667 |
| de_berlin_roads | 8 | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| fi_helsinki_roads | 3 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| ke_nairobi_roads | 24 | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| nl_amsterdam_roads | 7 | 3 | 0.857 | 1.000 | 0.923 | 0.667 |
| sg_singapore_footpaths | 17 | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| sg_singapore_roads | 44 | 6 | 0.977 | 1.000 | 0.989 | 0.833 |
| us_boston_streets | 679 | 108 | 0.826 | 0.952 | 0.884 | 0.704 |
| us_montana_missoula | 28 | 3 | 0.963 | 1.000 | 0.981 | 0.667 |
| us_seattle_sidewalks | 99 | 22 | 0.757 | 0.930 | 0.835 | 0.682 |
| us_usfs_flathead | 21 | 5 | 0.952 | 1.000 | 0.976 | 0.800 |

Boston and Seattle supply almost all useful ambiguity; most other datasets have too
few mapped groups or only easy positives. Aggregate improvements will remain fragile
until the non-US and reject-all label base expands.

## Remaining gaps

- Four datasets still use legacy capped candidate universes; Tunis has no mapped sidecar.
- Twenty Boston and six Seattle positive edges are outside the candidate graph.
- Typed 83-column candidate features are joined, but available only for Seattle and not
  yet selected into the 33-feature model.
- Reject-all evaluation is underpowered because legacy empty labels cannot express their
  full candidate universe.
- Panel votes remain opt-in: old batch artifacts do not prove which unselected candidates
  were displayed, so they cannot safely manufacture negative edge labels.

Next gate: fresh sidecars/parquets, candidate-recall fixes, ≥20 mapped empty groups, and
paired grouped-CV feature-family ablations with dataset holdouts and bootstrap intervals.
