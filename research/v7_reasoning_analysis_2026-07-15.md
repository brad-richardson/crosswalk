# V7 canonical-rubric reasoning analysis

**Date:** 2026-07-15

**Status:** analysis complete; human evaluation pending

**Panel:** Claude Opus 4.8 high, Codex GPT-5.6 Sol high, Muse Spark 1.1 high

## Scope

This analysis covers the effective v7 breadth replay: 26 stitching groups from
13 datasets and 78 valid ballots. The effective set substitutes the following
clean retries for timed-out rows in the original batch directories:

The manual-review pack is v7-only. It contains no v6-only cases or v6 fallback
rows; a group seen in v6 is present only when that group also reappeared in one
of these v7 batches.

| Group | Effective batch |
|---|---|
| `e67638a5` | `br_sao_paulo_roads_breadth_v7_retry1_20260714` |
| `dd106a0f` | `ch_grand_geneva_cycle_schema_breadth_v7_retry1_20260714` |
| `dc818edc` | `de_berlin_roads_breadth_v7_retry1_20260714` |
| `b7f57035` | `us_philadelphia_sidewalks_breadth_v7_retry2_20260715` |

`pack_feedback` was not enabled for this wave, so its 78 fields are empty. The
findings below come from the post-vote `reasoning` field. All 78 ballots contain
reasoning (50,560 characters total): Claude averaged 830 characters, Codex 313,
and Muse 802.

This is a qualitative diagnosis before human truth is available. Disagreement
and confidence are not correctness measurements.

## Result profile

| Wave | Unanimous | Majority | No majority | Auto-accept | Human review |
|---|---:|---:|---:|---:|---:|
| v6 | 17 | 7 | 2 | 8 | 18 |
| v7 | 11 | 12 | 3 | 4 | 22 |

The v6 and v7 consensus edge sets agree for 17 of 26 groups and differ for 9.
V7 is more conservative, but human labels are required before interpreting that
as an improvement or regression.

Pairwise v7 choice agreement was Claude-Muse 17/26, Claude-Codex 14/26, and
Codex-Muse 14/26. Codex was the sole dissenter in 6 of the 12 majority groups;
Claude and Muse were each the sole dissenter in 3.

## What the canonical rubric improved

The canonical instructions are visibly active in the reasoning. Agents
consistently cite MI-1 through MI-6 and SA-1 through SA-6, establish physical
identity before scoring similarity, preserve legitimate M:N segmentation, and
apply the all-and-only requirement to displayed options.

Most disagreements are now localized boundary questions rather than competing
definitions of a match. Six of the twelve majority decisions differ by exactly
one edge. Short junction anchors, endpoint kisses, clips, and slivers occur in
the reasoning for 23 of 26 groups. Representative cases are:

- Sydney `7a09bb4b`
- Sao Paulo `7cc10db3`
- France `18644de1`
- Fort Collins sidewalks `e4961c86`
- Fort Collins streets `42099176`
- Frisco `e3cfbfc7`

The canonical rubric should be retained. The principal remaining weaknesses are
evidence presentation and option coverage, plus one corridor-label wording bug.

## Instruction ambiguity: corridor labels

The evidence prompt currently says that `R#/T#` names the corridor each side
belongs to and that shared corridors tend to be one physical through-route.
Several agents then use “same corridor R0/T0” as positive cross-side identity
evidence.

That inference is not warranted: R labels are local to the reference side and T
labels are local to the target side. The shared numeric suffix does not assert
that R0 corresponds to T0. The wording should explicitly say:

> R# labels compare reference segments only; T# labels compare target segments
> only. R0 and T0 do not imply cross-side identity. Corridor membership is
> continuity context after physical identity has been independently established.

This is the highest-priority instruction-clarity change for the next wave.

## `NONE` conflates two different outcomes

There are 19 `NONE` ballots across 11 groups:

- 10 mean that the correct accepted edge set is genuinely empty.
- 9, across five groups, describe a nonempty desired set for which no exact
  displayed option exists.

The five nonempty option-gap groups are:

| Group | Failure pattern |
|---|---|
| `dc818edc` | Berlin core/partition with no exact pruned option |
| `3df39ddf` | Large France fan minus weak junction edges |
| `72cd975f` | Mumbai surface-road fan minus flyover/low-confidence edges |
| `1b629724` | Fort Collins fan minus two role-incompatible edges |
| `b7f57035` | Philadelphia continuation minus branch-conflict clips |

`1b629724` is the clearest generator failure: all three voters reject the menu,
not the existence of a match. The desired shape is “dense assignment minus a
small bad subset.” This differs from the earlier per-target multi-reference gap,
which the current generator can now express.

The structured output should add a `none_reason` enum:

- `all_edges_no_match`
- `no_exact_option`
- `insufficient_evidence`

For diagnostic runs, `no_exact_option` could optionally carry a proposed edge
set. Today both meanings serialize as `edge_set=[]`, so the distinction survives
only in prose.

The option generator should add pruned alternatives that remove small clusters
of slivers, ramps/flyovers, endpoint branch conflicts, or role-incompatible
edges from an otherwise dense assignment.

## Missing physical-facility evidence

A recurring ambiguity is whether two aligned geometries represent the same
physical pavement or parallel/grade-separated facilities. The most informative
cases are:

- Sydney `66e22055`: on-road cycleway versus separated cycle track
- Geneva `dd106a0f` and `ee608970`: surface cycleway/road versus tunnel or
  parallel facility
- Hong Kong `4eed5e80`: tunnel versus surface road

Names, classes, offsets, and two-dimensional geometry are not enough for these
cases. Evidence packs should expose `layer`, `bridge`, `tunnel`, `covered`,
facility subtype/separation, and relevant road flags when available. Ambiguous
class/grade combinations should remain human-routed when those facts are absent.

## Coverage arithmetic is valuable evidence

Agents repeatedly reconstruct interval partitions manually. Examples include a
0.932 + 0.066 sidewalk partition, a 0.705 + 0.295 bend partition, and dense fan
coverage sums used to isolate only two bad edges. This reasoning often
distinguishes real fine-to-coarse segmentation from duplicate, parallel, or
junction-only candidates.

Evidence packs should precompute a compact per-reference/per-target table with:

- union coverage;
- gaps and overlapping spans;
- tiling versus duplicate intervals;
- bearing or turn difference;
- endpoint versus interior intersection; and
- junction degree and branch role.

## Provider behavior and confidence

Provider confidence is not cross-calibrated:

| Provider | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Claude | 0.659 | 0.610 | 0.480 | 0.900 |
| Codex | 0.970 | 0.980 | 0.890 | 0.995 |
| Muse | 0.847 | 0.865 | 0.620 | 0.980 |

Codex remains highly confident when isolated or when evidence is contested.
Claude expresses nearly all explicit uncertainty. Muse and Claude provide the
most inspectable prose; Muse often performs the strongest visual/coverage
arithmetic but sometimes overweights collinearity and corridor continuity in
grade-separated cases. Codex is concise and, in v7, supplies most of the panel's
choice diversity.

Raw confidence should not weight ballots as if it were a shared probability.
The minimum-confidence routing gate should remain in place. It correctly held
four unanimous but low-confidence nonempty results for human review:

- Sao Paulo `e67638a5`
- Austin `6948c621`
- Fort Collins sidewalks `59fdcc1c`
- Philadelphia `8f948c89`

## Recommended human-review sequence

Review the highest-information groups first:

1. Option coverage: `dc818edc`, `3df39ddf`, `72cd975f`, `1b629724`, `b7f57035`.
2. Facility identity: `66e22055`, `dd106a0f`, `ee608970`, `4eed5e80`.
3. Local topology: `7a09bb4b`, `7cc10db3`, `18644de1`, `e4961c86`,
   `42099176`, `e3cfbfc7`.
4. Remaining human-routed groups, then the four auto-accept spot checks.

The generated `__all__` stitching-review cache follows this sequence and holds
all 26 effective v7 evidence snapshots, including the four retry substitutions.
Use de-anchored review mode to avoid inheriting the optimizer's preselection.

## Follow-up priority

1. Keep the canonical identity rubric unchanged through the human evaluation.
2. Clarify the side-local meaning of corridor labels.
3. Add structured `none_reason` output.
4. Generate “dense assignment minus small bad subset” alternatives.
5. Add interval-partition and topology facts to the evidence pack.
6. Add facility/grade-separation metadata, or route those cases to humans.
7. Re-evaluate confidence policy using human truth; do not weight by raw
   provider confidence in the meantime.

## Post-human evaluation

Brad completed the v7-only combined queue on 2026-07-15 in de-anchored mode.
The review required OSM inspection and satellite imagery for nearly every case;
all but perhaps two were judged legitimately ambiguous from the evidence shown
in the pack. Frontage roads and vertically layered roads were the dominant
ambiguity theme.

### Label-integrity audit

- 26/26 v7 groups have a label; the combined queue has zero groups remaining.
- All 26 rows are `labeler=brad` and `session_id=deanchored_v1`.
- The review produced 21 set-membership labels and 5 pair-semantics reject-all
  labels.
- Ten labels contain notes. Every R/T shorthand in those notes is in range and
  maps back to the exact v7 group snapshot.
- Every stored member belongs to its group, every stored pair belongs to its
  candidate universe, and no duplicate or malformed v7 label was found.

Set rows assert which reference and target segments belong to the matched group;
they do not adjudicate every candidate pair. “Exact” below therefore means exact
membership for a set row and exact edge set for a pair row. Pair-level errors
between already-included members can remain invisible.

### Aggregate comparison

| Predictor | Mixed-semantics exact | Rate |
|---|---:|---:|
| v7 consensus | 12/26 | 46.2% |
| Claude v7 | 13/26 | 50.0% |
| Codex v7 | 13/26 | 50.0% |
| Muse v7 | 12/26 | 46.2% |
| Optimizer proposal | 10/26 | 38.5% |
| v6 consensus on the same groups | 10/26 | 38.5% |

V7 and v6 were both exact on 9 groups. V7 alone was exact on `e4961c86`,
`59fdcc1c`, and `8f948c89`; v6 alone was exact on `66e22055`. Thirteen groups
matched neither wave. This is a net +2 exact groups for v7, but the replay
changed instructions, Codex model/route, effort, and generation time together,
so the gain cannot be attributed to one factor.

For the 21 set labels, v7 membership was exact on 10, with mean boundary
precision 0.936 and mean member coverage 0.834. For the five reject-all pair
labels, v7 was exact on two.

### Routing calibration is the strongest result

| Slice | Exact |
|---|---:|
| Auto-accept | 4/4 |
| Human review | 8/22 |
| Unanimous | 8/11 |
| Majority | 3/12 |
| No majority | 1/3 |

The panel did what the system needs most: all four auto-accepts survived human
review, while the difficult cases stayed with the human. The auto sample is
small, but there is no evidence here to loosen the routing gates. Majority
consensus in this hard slice is not safe to export.

The low-confidence gate held four unanimous nonempty groups. Two were exact
(`59fdcc1c`, `8f948c89`) and two were not (`e67638a5`, `6948c621`), so the gate
caught real risk without merely duplicating the unanimity rule. Of the three
unanimous `NONE` cases, two were correct reject-alls (`d4d2e782`, `ae47a8f3`)
and one was an inexpressible nonempty match (`1b629724`). Keeping all `NONE`
results human-routed remains correct.

### Menu coverage versus voter choice

The human judgment was expressible by a displayed option (counting `NONE` as
the reject-all choice) for 18/26 groups. Eight groups had no exact option:

- `dc818edc`
- `3df39ddf`
- `72cd975f`
- `1b629724`
- `b7f57035`
- `dd106a0f`
- `7a09bb4b`
- `e67638a5`

The pre-human reasoning analysis identified the first five of these five; human
review confirmed all five predictions. The other three add facility/layer and
large-fan examples to the same generator weakness.

The best offered option had mean member coverage 1.000 but boundary precision
0.967. In other words, the menus almost always contained all desired members
but sometimes forced a few unwanted ones too. This strongly supports generating
“dense assignment minus a small bad subset” options rather than broadening the
entire search.

The 14 v7 differences split into:

- 8 expressibility failures: the human answer was absent from the menu.
- 6 selection failures despite an expressible answer: `66e22055`, `ee608970`,
  `4eed5e80`, `7cc10db3`, `18644de1`, and `6948c621`.

Three of those six were reject-all facility-identity errors: the panel retained
an edge where the human found separated infrastructure, a tunnel/surface
conflict, or a non-identical junction continuation.

### Frontage and vertical-layer evidence gap

The human notes localize the missing information:

- Sydney `66e22055`: separated cycle infrastructure and an underground target.
- Geneva `dd106a0f`: dedicated cycle lanes versus a motorway/trench/possible
  bridge where overlapping roads are not one facility.
- Hong Kong `4eed5e80`: tunnel/surface identity conflict.
- Philadelphia `b7f57035`: a physical bridge at `layer=1`, plus a false
  `R2`–`T7` relationship.
- Hong Kong `da49a4e0`: frontage road interwoven with the actual match.
- Mumbai `72cd975f`: sibling/parallel named-road ambiguity around a dense fan.

There is also a terminology bug in the current pack. The displayed per-edge
`is_bridge` value is computed with `networkx.bridges` on the bipartite candidate
graph. It means **graph-theoretic bridge edge**, not a physical road bridge. The
prompt renders it as the bare word “bridge,” which is misleading precisely in
the vertical-layer cases that need physical bridge evidence.

The data loaders already extract physical `is_bridge`, `is_tunnel`, and
`level`/`layer` attributes, but the stitching group sidecar currently carries
only names and classes for each segment. The next evidence schema should:

1. Rename the current structural flag to `candidate_graph_bridge` and display
   it as “graph bridge,” never just “bridge.”
2. Propagate side-specific physical `is_bridge`, `is_tunnel`, `layer`/`level`,
   and compact road-role flags into group sidecars and evidence packs.
3. Include useful frontage/link/service/access/one-way role tags when available.
4. Treat coincident or parallel roads with conflicting layer/facility roles as
   an explicit human-routing trigger when the evidence is incomplete.

### Limitation exposed by frontage roads

Set membership is appropriate for training group resolution, but it cannot say
which already-included reference should pair with which already-included target.
Hong Kong `da49a4e0` demonstrates this: all three providers have exact human
membership, yet their edge sets differ and the note calls out a frontage road
interwoven with the true match.

For future diagnostic evaluations, the UI should offer an optional exact-pair
adjudication mode after membership is chosen. This is especially valuable for
frontage, braided, and layered networks. The existing v7 labels remain valid
membership truth; they simply should not be reported as pair-level exactness.
