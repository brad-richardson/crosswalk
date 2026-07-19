# Global resolver GO/NO-GO — 2026-07-18

## Decision

**NO-GO for replacing the production optimizer today. GO for continued shadow
experimentation and targeted exact-edge labeling.** The current learned resolver
still loses to the production optimizer under both repeated grouped CV and
leave-one-dataset-out transfer. Adding the pair matcher's candidate features
narrows the gap slightly, but does not reverse it.

The likely end-state is not a monolithic replacement for the pair matcher. It is
a global resolver that consumes pair-identity evidence plus group/corridor
context, while deterministic graph constraints and an optimizer fallback remain
as safety rails until transfer performance is established.

## Current label inventory

- 417 stitch rows: 336 pair-semantics and 81 set-semantics.
- 325 usable mapped groups, 1,817 candidate edges, 24 datasets.
- 1,507 keep and 310 drop edge targets.
- 294 mapped panel groups and 31 mapped human pair-exact groups.
- Only 9 de-anchored exact reviews. The corpus is therefore large enough to fit
  models, but still weak on independent counterfactual decisions.

The key distinction is now stored explicitly: a candidate can be the same
physical feature (`identity=match`) but still be omitted from the final global
resolution (`resolution=drop`). Historical unselected stitch edges cannot be
treated as pairwise negatives.

## Candidate architecture comparison

All numbers use the same 325 groups, five repeated grouped-CV seeds, the eF1
set selector, and a paired whole-group bootstrap. Production is the current
optimizer+prune assignment on the identical labeled edge universe.

| Features | Count | OOF edge F1 | Group exact | F1 delta vs production |
|---|---:|---:|---:|---:|
| Resolver context only | 33 | 0.9261 | 0.7262 | -0.0202 |
| + pair geometry/coverage | 72 | 0.9250 | 0.7200 | -0.0213 |
| + pair nonsemantic topology/representation | 105 | **0.9301** | 0.7262 | **-0.0162** |
| + all 83 pair features | 116 | 0.9280 | 0.7200 | -0.0184 |
| Production optimizer+prune | — | **0.9463** | **0.8031** | — |

For the best learned variant (`nonsemantic`), the F1 delta 95% CI is
`[-0.0288, -0.0048]` with bootstrap support `P(delta > 0) = 0.001`. Its LODO
edge F1 is 0.9185 and group exact is 0.7046, versus production at 0.9463 and
0.8031. The transfer result is an unambiguous NO-GO.

Interpretation: pair topology and representation features add about 0.004 edge
F1 over the 33-feature model. Raw semantic features do not help in this sample.
The dominant limitation is supervision quality and anchoring, not lack of model
capacity.

## New supervision paths

The durable panel evidence now expands to 8,316 edge rows over 471 groups and 25
datasets: 666 zero, 906 fractional, and 6,744 one-valued soft targets. Of those,
7,311 rows come from a complete recorded displayed-edge universe. Reject-all
ballots and never-selected displayed candidates can therefore contribute safe
zeros; legacy NONE/insufficient-evidence ballots remain abstentions. Historical
`no_exact_option` desired sets are resolved only through their originating pack
label maps.

The audited stitch-to-pair preview currently derives 1,828 weak selected-edge
positives, leaves all complements unlabeled, removes 47 pairs already covered by
human truth, and retains 1,753 rows. Candidate features are available for 1,645;
107 are missing. There are no explicit identity rows yet because the new exact
review UI has not been used. This preview remains outside production pair labels.

## Evidence gate for the next decision

Before reconsidering replacement:

1. Collect at least 30–50 de-anchored exact-identity groups and at least 20
   human-confirmed reject-all groups.
2. Include dropped-but-identity-match examples; these teach the residual
   distinction the current labels collapse.
3. Hold out by parent/corridor and dataset, not by edge, and keep one untouched
   promotion set.
4. Rerun repeated grouped CV, LODO, exact-match, negative-heavy slices, and a
   learning curve. Promotion requires a non-negative paired CI against
   production plus no safety-slice regression.

A useful shadow model should target roughly 150–250 independent human exact
groups across modes/failure strata. A credible first production decision is
more likely at 500–1,000 such groups, including 100–200 hard reject/conflict
cases and enough examples to expose dataset transfer. Repeated panel ballots on
one group improve uncertainty estimates but still count as one independent
training group.
