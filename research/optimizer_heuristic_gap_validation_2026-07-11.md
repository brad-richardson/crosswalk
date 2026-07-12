# Optimizer heuristic gap validation — 2026-07-11

## Decision

Do **not** ship the proposed per-type glue/prune defaults, 0.05 confidence margin,
or bridge-backbone guard yet. The fixed-universe evaluation does not validate them
as production wins:

- Current production selection remains best on pooled and clean-label edge F1.
- The handed-off prune thresholds regress pooled F1 from **0.9049 to 0.8969** and
  clean F1 from **0.9546 to 0.9437**. Exact-group rate is unchanged.
- On Boston, which supplies 111 of the 144 mapped groups, the proposed per-type
  thresholds, 0.05 margin, and bridge guard are exactly identical to the current
  scalar 0.96 behavior.
- A 0.02 margin is mildly better on Boston but worse pooled and on clean labels.
  That is a useful candidate for more ground truth, not a safe default change.
- Glue sweeps changed which historical labels could be recovered. Their apparent
  score gains were retention-biased and cannot support a production change.

This conclusion supersedes the optimistic "low-risk wins" wording in the experimental
resolver gap analysis. That report identified plausible hypotheses; it did not establish
them on a stable evaluation universe.

## Reproducible evaluation

The new offline harness reconstructs the assignment immediately before confidence
pruning from `selected OR pruned`, applies each policy once per unique candidate edge,
and evaluates every policy on the same recovered labels and candidate universe. It
reports pooled, clean/split, match-type, and per-dataset slices plus a separate coverage
table.

```bash
uv run python scripts/ablate_optimizer_heuristics.py \
  --data-root . \
  --output /tmp/optimizer-heuristic-ablation
```

The default sweep includes scalar thresholds 0.90–0.98, margins 0.02/0.05/0.08,
the handed-off per-type thresholds (`1:N=0.92`, `N:1=0.94`, `M:N=0.96`), and
the per-type + margin + bridge-guard combination.

### Coverage

| Coverage item | Result |
|---|---:|
| Labeled datasets | 13 |
| Fresh sidecars | 8 |
| Factory-snapshot fallbacks | 4 |
| Missing sidecars | 1 (`tn_tunis_ml_roads`) |
| Curated label rows | 226 |
| Mapped evaluation groups | 144 |
| Evaluation edges | 850 |
| Positive evaluation edges | 618 |
| Boston share | 111/144 groups; 679/850 rows |

Coverage is not equivalent to independent ground truth. Split recovery means an old
human group now spans multiple current groups, so its within-group negative labels are
partial/noisy. The primary decision slice is therefore `provenance=clean`, with pooled
and split results reported to expose sensitivity.

Major holes:

- Seattle has 49 label rows and a fresh sidecar but maps to **zero** evaluation rows;
  its v1→v2 identifiers need rekeying or relabeling.
- Tunis has three labels but no current or factory sidecar available to the harness.
- Bogotá roads, Helsinki, Nairobi, and Singapore footpaths use legacy factory groups.
  Four reject-all labels are skipped because capped legacy edges cannot represent the
  full candidate universe.
- Boston has 20 human-selected edges outside the persisted candidate graph, which is
  an explicit candidate-recall ceiling.

### Fixed-universe results

| Policy | Pooled F1 | Pooled exact | Clean F1 | Clean exact | Boston F1 | Boston exact |
|---|---:|---:|---:|---:|---:|---:|
| Current persisted optimizer | **0.9049** | **0.7292** | **0.9546** | 0.8182 | 0.8844 | 0.6937 |
| Reconstructed pre-prune | 0.8652 | 0.6806 | 0.9455 | 0.7727 | 0.8370 | 0.6306 |
| Scalar 0.98 | 0.9005 | 0.7292 | 0.9409 | 0.8182 | 0.8898 | 0.6937 |
| Margin 0.02 | 0.9023 | **0.7431** | 0.9433 | **0.8364** | **0.8919** | **0.7117** |
| Margin 0.05 | 0.8969 | 0.7292 | 0.9437 | 0.8182 | 0.8844 | 0.6937 |
| Handed-off per-type thresholds | 0.8969 | 0.7292 | 0.9437 | 0.8182 | 0.8844 | 0.6937 |
| Per-type + 0.05 + bridge guard | 0.8969 | 0.7292 | 0.9437 | 0.8182 | 0.8844 | 0.6937 |

The 0.02 margin's pooled exact-match gain comes with lower pooled and clean edge F1.
The apparent scalar-0.98 gain is confined to the noisy split slice (F1 +0.015); clean
F1 falls by 0.0137. Neither is a ship signal.

By type, current production is already strong on 1:N (F1 0.9781) and N:1
(F1 0.9600). The only tentative margin signal is M:N: margin 0.02 moves F1 from
0.8668 to 0.8732 and exact from 0.4717 to 0.4906 over just 53 groups. This matches
the panel's qualitative large-interchange over-selection finding, but the sample is too
small and Boston-heavy to select a default.

## Why the proposed glue result is not comparable

A scalar glue sweep was also run by reoptimizing the available Boston scored-candidate
cache and exporting temporary sidecars. Across glue 0.575–0.98, only 99 historical
groups and 336–357 edge rows recovered, versus 111 groups and 679 rows in the current
fresh Boston sidecar. The clean slice shrank to 36–37 groups, and human-selected edges
outside the candidate graph rose from 18 to 25 as glue increased.

Some configurations consequently reported clean F1/exact of 1.0, but only after most
of the evaluation universe disappeared. Those numbers are not wins. A grouping/glue
experiment must satisfy all of these before its quality metrics are compared:

1. Same raw inputs, scored-candidate snapshot, calibration, and labels.
2. Same recovered human-group count and provenance mix, or an explicit paired-label
   comparison over the intersection plus a separate attrition penalty.
3. No regression in represented human-positive edges or candidate-graph recall.
4. Per-dataset reporting so Boston cannot hide smaller-dataset failures.

There is also an implementation-design gap: glue forms connected components before a
final `MatchType` exists. A "per-type glue threshold" needs a deterministic pre-group
type definition or a documented two-pass grouping algorithm. Dispatching on the final
type at the current seam is circular.

## Feature-ablation context

The existing 83-feature pair-model ablation (`benchmarks/ablation_2026_07`) already
identifies useful families: class, alignment coverage, name, topology, sinuosity,
intersection overlap, heading, and crossing angle all hurt held-out F1 when removed.
Lateral offset is the only category whose removal slightly helps (+0.0005 F1), while
graphlet removal is a small regression (-0.0016) and remains below the project's
10K-label pruning threshold. This supports expanding evaluation around *availability
and dataset dead zones*, not stripping broad feature families globally.

The resolver's current persisted stage-1 candidate graph still lacks the full typed
pair-feature matrix. A learned-resolver ablation cannot honestly decide the value of
those families until a stage-2 candidate parquet persists all 78 model features plus
the signed lateral offset for every candidate edge, not only previously pair-labeled
edges.

## Ranked next steps

1. **Repair the evaluation universe before tuning.** Rekey or relabel Seattle, generate
   Tunis, and produce fresh candidate-graph sidecars for Bogotá roads, Helsinki,
   Nairobi, and Singapore footpaths. Run large datasets in bounded batches rather than
   loading all sidecars together on the 17 GB host.
2. **Persist stage-2 candidate features and recall accounting.** Write one typed parquet
   row per `(dataset_id, group_id, ref_id, target_id)` with the 78 pair features,
   signed lateral offset, confidence/calibration version, optimizer decision/reason,
   and structural fields. Lower or separately audit the export floor responsible for
   Boston's 20 missing human-positive edges.
3. **Grow and rebalance clean ground truth.** Target at least 200–300 clean groups before
   choosing per-type thresholds, with deliberate quotas for M:N interchanges, 1:N/N:1
   corridors, cross-mode reject-all groups, and non-Boston datasets. Keep split labels
   as a sensitivity slice rather than threshold-tuning truth.
4. **Run targeted panels, then settle disagreements with humans.** The existing 35-group
   early-signal wave found M:N over-selection and cross-class false matches. Run fresh
   waves on Tunis, Berlin, Bogotá bike, Singapore footpaths/roads, and Missoula, sampling
   low margin, large edge count, bridge-backbone, cross-class, and optimizer/panel
   disagreement strata. A fourth-voter wave should remain opt-in until it demonstrates
   decorrelation outside the small Boston sample. Panel consensus is label-acquisition
   evidence, not a substitute for a held-out human set.
5. **Recover reject-all evidence.** Build at least 20 current candidate-graph-backed
   empty-set groups across modes; legacy empty labels without candidate membership are
   not evaluable. Gate any prune change on candidate recall regression no worse than
   0.01.
6. **Re-run paired ablations.** Sweep scalar and per-type prune/margin policies with
   bootstrap confidence intervals and leave-one-dataset-out results. Only then consider
   the small M:N margin candidate. Define and test a non-circular glue algorithm
   separately.

## Production recommendation

Keep the existing optimizer defaults. Land evaluation/data-capture infrastructure first.
The bridge guard can be reconsidered as a safety invariant if future labeled failures
show it protects necessary backbone edges, but current data shows zero decision changes;
adding structural computation before prune today would add complexity without measured
benefit.
