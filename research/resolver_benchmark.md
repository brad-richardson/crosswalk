# Resolver benchmark — optimizer vs naive vs learned

> Research-only, regenerated 2026-07-12 from 949 curated edge labels in 163
> dataset-scoped groups across 12 datasets. The model is evaluated with repeated
> five-fold grouped CV, paired whole-group bootstrap intervals, and
> leave-one-dataset-out (LODO) transfer. In-sample numbers are shown only as an
> overfit diagnostic.

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
| Learned threshold 0.5 | 0.949 | 0.855 | 0.899 | 0.724 |
| Learned expected-F1 | 0.916 | 0.939 | 0.928 | 0.773 |

The model's in-sample lead does not survive held-out groups.

## Grouped-CV OOF — decision metric

| model | precision | recall | F1 | group exact |
|---|---:|---:|---:|---:|
| Learned XGBoost + expected-F1 | 0.853 | 0.899 | 0.875 | 0.748 |
| Production optimizer | 0.845 | 0.960 | 0.899 | 0.730 |
| Oracle confidence ≥0.98 | 0.858 | 0.911 | 0.884 | 0.687 |

The single deterministic split favors production edge F1 by 0.023 and the model's
group exact by 0.018. The repeated and dataset-held-out checks below show why neither
point delta is sufficient evidence for a production flip.

## Repeated grouped CV + paired uncertainty

Each seed uses shuffled stratified group folds. Every edge is held out once per seed,
and a dataset-scoped group never crosses the train/test boundary.

| seed | model F1 | Δ vs production | model exact | Δ vs production |
|---:|---:|---:|---:|---:|
| 0 | 0.8805 | -0.0183 | 0.7546 | +0.0245 |
| 1 | 0.8757 | -0.0230 | 0.7485 | +0.0184 |
| 2 | 0.8620 | -0.0368 | 0.7362 | +0.0061 |
| 3 | 0.8751 | -0.0237 | 0.7546 | +0.0245 |
| 4 | 0.8615 | -0.0373 | 0.6994 | -0.0307 |

The final decision is made from the mean held-out probability per edge. A paired
bootstrap resamples all 163 complete groups (2,000 draws), never individual edges:

| metric | observed Δ | 95% interval | bootstrap support (Δ > 0) |
|---|---:|---:|---:|
| edge F1 | -0.0271 | [-0.0581, +0.0015] | 0.033 |
| group exact | +0.0245 | [-0.0368, +0.0859] | 0.784 |

The F1 loss is stable across all five fold assignments. The apparent exact-match
gain is not: one seed reverses it, and the interval spans a meaningful loss through a
meaningful gain.

## Leave-one-dataset-out transfer

LODO trains on 11 datasets and predicts the twelfth, so Boston or Seattle examples
cannot teach the model how to resolve another group from that same dataset.

| held-out slice | learned F1 | production F1 | learned exact | production exact |
|---|---:|---:|---:|---:|
| Pooled LODO | 0.8784 | 0.8988 | 0.7546 | 0.7301 |
| Nairobi roads | 0.5556 | 1.0000 | 0.5000 | 1.0000 |
| Singapore footpaths | 0.8000 | 1.0000 | 0.2500 | 1.0000 |
| Boston streets | 0.8729 | 0.8844 | 0.7685 | 0.7037 |
| Seattle sidewalks | 0.8070 | 0.8346 | 0.6818 | 0.6818 |

The paired LODO bootstrap resamples the 12 held-out datasets as clusters and gives
edge-F1 Δ `-0.0204` (95% interval `[-0.0874, -0.0097]`, bootstrap support
`0.003`) and exact Δ `+0.0245` (`[-0.1731, +0.0638]`, support `0.669`).
Transfer therefore reinforces the edge-F1 NO-GO and again leaves exact-match
improvement unresolved.

**Decision: NO-GO for production.** Keep the learned resolver as an experimental
ablation harness. Reconsider only after candidate recall and multi-dataset label
coverage improve, then require paired intervals that clear zero on both edge F1 and
group exact (or an explicit product decision to trade one for the other).

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
paired feature-family ablations evaluated with the repeated grouped/LODO/bootstrap
harness added here.
