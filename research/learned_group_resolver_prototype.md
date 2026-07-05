# Learned group resolver — prototype + honest eval

Prototype for the flagship milestone: replace/augment the hand-written optimizer
edge-selection inside M:N stitching groups with a trained per-edge keep/drop
classifier. This document reports the **data reality**, a grouped-CV eval of a
prototype XGBoost classifier **vs the optimizer baseline**, and a **go/no-go**
recommendation with a concrete productionization plan.

Code (all experimental, **not imported by production at all** — no runtime flag
path yet, **zero production behavior change**):
`src/matcher/resolver/` (extract / features / votes / evaluate),
`scripts/build_resolver_dataset.py`, `tests/unit/test_resolver_extract.py`.
Reproduce:

```bash
uv run python scripts/build_resolver_dataset.py \
  --data-root /path/to/matcher \
  --dataset us_boston_streets --dataset us_seattle_sidewalks --with-votes
```

---

## TL;DR — go/no-go

**Conditional NO-GO on a full learned resolver at current label scale; GO on a
cheap confidence-based drop-filter as the interim step.**

There *is* measurable per-edge headroom over the keep-all optimizer baseline
(clean labels: F1 0.828 → 0.852 learned, +0.02; all mapped labels: 0.781 →
0.895, +0.11). **But three findings temper this into a "not yet":**

1. On the *cleanest* labels a trivial **tuned confidence threshold beats the
   structured model** (0.872 vs 0.852 F1). Almost all the signal is "edge
   confidence relative to its group"; the 20 structural/context features add
   little at n≈40 groups.
2. **Group-level exact-match — the metric that actually matters for replacing a
   group decision — barely moves** (clean 0.805 → 0.829; all labels 0.691 →
   0.691). A per-edge F1 gain that does not translate into more fully-correct
   groups is not yet worth the optimizer-rewrite risk.
3. **The data can only teach over-selection correction, not under-selection.**
   The sidecar persists only the optimizer's *selected* edges (see §1), so the
   classifier never sees a rejected-but-correct edge. A true keep/drop resolver
   needs the full candidate graph persisted first.

The highest-leverage next step is **more labels, not more model**: adding 62
panel-soft-labeled groups to training lifts clean F1 0.852 → 0.903 (+0.05),
a bigger gain than any feature work. See §5 milestone plan.

---

## 1. Data reality (what is actually available per edge)

Verified against `data/output/{ds}_groups.json`, `labels/stitching/`, and
`data/agents/stitching/batches/*/votes.csv` (2026-07):

| fact | consequence for the resolver |
|---|---|
| The sidecar `edges` list persists **only the optimizer's selected assignment** (Boston: 10,579/10,590 edges `selected=True`; the 11 False are junction slivers; Seattle: 35,288/35,288 True). | The rejected candidates are **gone**. A keep/drop task over persisted edges is really *over-selection correction*: "which optimizer-kept edges should be dropped." Keep-all recall = 1.0 by construction; errors are all false positives. Under-selection (drop-but-should-keep) is **unlearnable** from this data. |
| The 78 pairwise ML features (`labels/features/`) are keyed `(gers_id, target_id)` but exist only for *pair*-labeled pairs. Coverage of group edges = **20/364 ≈ 5%**. | Per-edge features must come from the **sidecar itself** (confidence + PR #267 structural layer + alignment fractions), not the pairwise parquet. No geometry recompute was needed or done. |
| Per edge the sidecar carries: `confidence`, `degree_ref/tgt`, `is_bridge`, `is_sliver`, `biconnected_block`, `corridor_ref/tgt`, `{gers,local}_{start,end}_frac`, `selected`. Per group: `n_edges`, `n_corridors`, `n_assignment_components`, `largest_biconnected_block`, `oversized_group`, `match_type`. | This is the full feature backbone the prototype uses (25 derived columns; see `resolver/features.py`). |
| Ground truth = `labels/stitching/` curated `selected_edges`. Boston 73 labels (24 human `brad`, 49 `panel_unanimous_v1`), Seattle 9. group_id churns on any component shift. | Labels mapped to current sidecar groups by **edge overlap**, reusing `stitch_eval.recover_labeled_groups` verbatim. |

**Label→group mapping outcome** (Boston): 40 **clean** (all selected edges land
in one current group), 27 **split** (human edge set spans ≥2 current groups → the
within-group keep set is *partial*, so drop labels are noisy), 6 **lost** (edges
no longer survive), 0 empty. Seattle: 1 clean, 1 split, 2 empty, 5 lost — too
small to eval alone.

**Resulting per-edge dataset:** 376 edges / 68 groups (241 keep, 135 drop;
`PYTHONHASHSEED=0` — the split-label count varies by a few edges otherwise, see
the reproducibility note in §3).
Clean-only: 157 edges / 41 groups (109 keep, 46 drop = 29% drop rate). Slivers in
labeled groups ≈ 0, so sliver-filtered metrics equal raw here.

The **clean** slice is the trustworthy eval; **split** rows are reported but
carry mapping noise (a within-group partial-truth inflates the apparent drop
count) and should be read as an upper bound on available signal, not ground
truth.

---

## 2. Setup

- **Task:** per-edge binary keep (1) / drop (0), where keep = edge ∈ human
  `selected_edges`. Baseline = optimizer `selected` (= keep-all on persisted
  edges).
- **Model:** XGBoost (`max_depth=3`, 120 trees, `lr=0.08`,
  `scale_pos_weight=n_neg/n_pos`), 25 sidecar-derived features. Deliberately
  small given n.
- **CV:** `GroupKFold(5)` on `group_id` — **edges from one group never span
  train/test**. Metrics computed on pooled out-of-fold predictions. Single-class
  training folds fall back to keep-all (the majority + production default).
- **Baselines:** (a) **keep-all** = the optimizer's own selection (the real
  production baseline); (b) **tuned confidence threshold** — threshold picked on
  each fold's *train* split only, a non-learned control that isolates "is there
  anything beyond confidence?".
- **Metrics:** per-edge P/R/F1 (raw + sliver-filtered) from pooled OOF preds, and
  **group-level exact-match rate** (fraction of held-out groups whose predicted
  keep-set exactly equals the human within-group keep-set). Sliced by dataset,
  clean/split provenance, and labeler.

---

## 3. Results — model vs optimizer baseline (identical held-out groups)

All figures below are the deterministic `PYTHONHASHSEED=0` run.

### Per-edge P/R/F1 and group-exact (pooled OOF, GroupKFold-5)

| slice | model | edges | groups | P | R | F1 | grp-exact | F1 (sliver-filt) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **clean** | optimizer keep-all (baseline) | 157 | 41 | 0.707 | 1.000 | **0.828** | **0.805** | 0.828 |
| **clean** | tuned conf threshold | 157 | 41 | 0.853 | 0.892 | **0.872** | 0.780 | 0.872 |
| **clean** | learned XGB | 157 | 41 | 0.908 | 0.802 | **0.852** | **0.829** | 0.852 |
| all mapped | optimizer keep-all | 376 | 68 | 0.641 | 1.000 | 0.781 | 0.691 | 0.781 |
| all mapped | tuned conf threshold | 376 | 68 | 0.833 | 0.909 | 0.869 | 0.662 | 0.869 |
| all mapped | learned XGB | 376 | 68 | 0.886 | 0.905 | **0.895** | 0.691 | 0.895 |
| Boston only | optimizer keep-all | 369 | 66 | 0.640 | 1.000 | 0.780 | 0.697 | 0.780 |
| Boston only | learned XGB | 369 | 66 | 0.883 | 0.928 | **0.905** | 0.727 | 0.905 |

Seattle alone (2 clean, tiny) is not separately evaluable — folded into "all".
The "all mapped" / "Boston" rows include the noisier *split* labels; the
**clean** rows are the trustworthy comparison.

### Per-labeler provenance (all mapped labels)

| labeler | model | edges | groups | F1 | grp-exact |
|---|---|---:|---:|---:|---:|
| brad (human) | keep-all | 91 | 21 | 0.795 | 0.810 |
| brad (human) | tuned conf | 91 | 21 | 0.797 | 0.619 |
| brad (human) | learned XGB | 91 | 21 | **0.885** | 0.810 |
| panel_unanimous_v1 | keep-all | 285 | 47 | 0.777 | 0.638 |
| panel_unanimous_v1 | tuned conf | 285 | 47 | 0.872 | 0.638 |
| panel_unanimous_v1 | learned XGB | 285 | 47 | **0.906** | 0.681 |

Notable: on **human** labels the learned model crushes the confidence threshold
(0.885 vs 0.797) — the structural features earn their keep against a human's more
holistic judgement, where confidence alone is a poor proxy. On **panel** labels
(mostly split-inclusive) the model still leads (0.906 vs 0.872) but by less,
since the panel's own decisions track confidence more closely.

### Stability (per-fold F1 is noisy; pooled OOF is stable)

Individual CV folds swing hard (clean per-fold F1 e.g. `[0.42, 0.91, 0.86, 0.79,
0.98]`) — expected with ~8 groups/fold. But the **pooled OOF F1 is stable across
8 different group-shuffle CV partitions**: all-labels mean 0.889 (sd 0.006), clean
mean 0.850 (sd 0.006). So the point estimates in §3 are trustworthy; the
per-fold spread is a small-sample artifact, not model instability.

**Reproducibility note.** The **clean** slice is bit-stable (157 edges, 41
groups). The **all-labels** count wobbles by a few edges run-to-run because
`stitch_eval.recover_labeled_groups` breaks ties among equal-overlap groups by
set-iteration order, which is subject to string-hash randomization — this only
touches *split* labels (whose mapping is inherently ambiguous). Read the clean
slice + shuffle-averaged numbers as the trustworthy figures; set
`PYTHONHASHSEED=0` for an exactly reproducible split mapping. Digits are
reported to ±0.01.

### Panel soft labels add real signal (with/without)

Adding 62 panel-soft-labeled groups (not in the curated set) to the *training*
folds, with per-edge soft keep = reliability-weighted provider vote share
(codex down-weighted 0.5, see `resolver/votes.py`), evaluated only on curated
clean labels:

| training data | clean OOF F1 | grp-exact |
|---|---:|---:|
| curated clean only | 0.852 | 0.829 |
| curated clean + 62 panel-soft groups | **0.903** | 0.805 |

+0.05 F1 from soft labels alone — larger than any feature-engineering gain
observed, and the cheapest lever available (panels already run).

### What the model keys on (gain importance, full-data fit)

`conf_rel_max` (0.24, confidence vs group max) ≫ `match_type_MN` (0.14),
`confidence` (0.09), `num_targets` (0.09), `local_span`/`min_span` (~0.06),
`degree_tgt`, `n_share_tgt`. **Confidence-relative-to-group dominates**; the
structural layer contributes at the margin. This is *why* the tuned threshold is
so competitive on clean data.

---

## 4. Interpretation / honest verdict

- **Is there headroom over the strong optimizer baseline?** Yes, but modest and
  mostly confidence-shaped. The optimizer over-selects ~29% of edges (by clean
  human truth); a learned filter recovers precision 0.71→0.91 while giving back
  some recall (1.0→0.80). Net clean F1 +0.02 over keep-all, and it *loses* to a
  one-parameter confidence threshold (0.872).
- **Does it help the decision that matters?** Only slightly. Group-exact clean
  0.805→0.829 (+2 pts, ≈1 extra correct group in 41); on all mapped labels it is
  flat (0.691→0.691). Replacing `optimizer.py` edge selection with a model that
  doesn't move group-exact is not justified yet.
- **Why negative-ish is the right read:** at 40–60 labeled groups, a 25-feature
  model is in the regime where a single strong feature (confidence) plus
  regularization is near-optimal, and the extra features mostly add variance.
  The human-label slice hints the structure *will* pay off with more/holistic
  labels (0.885 vs 0.797 there), but not yet at scale.

**What would flip the answer:**
1. **Persist the full candidate graph** (rejected edges included) so
   under-selection becomes learnable — today's biggest blind spot.
2. **~150–300 labeled groups** (vs 68), ideally human or human-audited, so the
   structural features clear the noise floor. Panel-soft labels are a cheap
   accelerant (+0.05 F1 already).
3. A **group-level objective** (structured/constrained prediction enforcing
   non-overlap) rather than independent per-edge calls, to move group-exact.

---

## 5. Productionization milestone plan

Where it hooks in: `matching/optimizer.py::_classify_and_resolve_component`
(the M:N branch, ~lines 748–807) produces the per-group assignment that becomes
`selected`. A learned resolver would run as a **post-assignment prune/re-score**
of that component's edges, behind a config flag, emitting the same
`selected` bool the sidecar already carries.

**Phase 0 — data plumbing (prerequisite, ~0.5 wk).** Persist the **full
candidate edge set** (not just selected) into the sidecar with a `selected`
flag per edge, plus a per-edge alternative-rank/margin. This is the single most
important unblock: it makes under-selection learnable and turns every future
pipeline run into resolver training data. Cheap — the candidate graph already
exists in-memory in `find_match_components`.

**Phase 1 — interim confidence-drop filter (~0.5 wk, shippable now).** A
flag-gated post-optimizer rule: within a group, drop an edge if its
group-relative confidence is below a tuned margin (respect sliver + single-
corridor exemptions). Captures most of the measured headroom (clean F1 ≈ 0.87)
with **no model**, no train/serve skew, trivially auditable. Ship behind
`--resolver-prune` default-off; A/B on Boston/Seattle sidecars.

**Phase 2 — learned per-edge resolver (gated on ~150+ groups).** Promote this
prototype: same features + the now-persisted rejected-edge negatives + panel-
soft labels. Train offline, export like the existing XGBoost model, load in the
optimizer behind a flag. Ship only when it beats Phase 1 on **group-exact** (not
just edge F1) on held-out groups.

**Phase 3 — structured group model (gated on several hundred groups).**
GNN/CRF over the candidate bipartite graph with a coverage/non-overlap
constraint (the design doc's endgame), to directly optimize group-exact and
capture "if edge e kept, its conflicting sibling should drop." Hierarchical
resolution for oversized cores (resolve sub-corridors, then a boundary
stitching pass).

**Label targets to unblock each phase:** Phase 2 ≈ 150–300 groups (mix human +
panel-soft, full candidate graph); Phase 3 ≈ 500+. Panels are the scalable
source; keep down-weighting codex and prefer post-clip-fix unanimous vintages.

---

## 6. Deliverables / reproducibility

- `src/matcher/resolver/extract.py` — label→group mapping + per-edge table with
  provenance (dataset, labeler, clean/split).
- `src/matcher/resolver/features.py` — 25 sidecar-derived features
  (`FEATURE_COLUMNS`), confidence-relative-to-group central.
- `src/matcher/resolver/votes.py` — panel-vote soft labels, reliability-weighted
  (codex down-weighted), churn-robust vote→sidecar mapping.
- `src/matcher/resolver/evaluate.py` — GroupKFold harness, keep-all + tuned-conf
  baselines, per-edge + group-exact metrics, slice report.
- `scripts/build_resolver_dataset.py` — CLI driver (see top).
- `tests/unit/test_resolver_extract.py` — data-contract tests + a guard that no
  production module imports `matcher.resolver`.

Nothing here is imported by the pipeline; there is **zero production behavior
change**.
