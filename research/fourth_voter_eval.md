# Fourth Panel Voter — Invoker, Dissenter Replay & 4-Voter Validation

> _Historical document. The project was renamed `matcher` → `crosswalk` (PyPI `crosswalk-py`) on 2026-07-05; the original name is preserved below unchanged._

**Date:** 2026-07-05
**Goal:** Add a decorrelated 4th voter to the stitching consensus panel (claude +
codex + agy) for quorum robustness, decide the 4-voter **export rule** (3/4
majority vs 4/4-of-responding), and validate the candidate model — all without
touching production defaults, labels, or live output.

## The 4th voter

- **Provider:** `opencode` (v1.17.13, OpenRouter-backed), model
  `openrouter/qwen/qwen3-vl-235b-a22b-instruct`.
- **Why this model:** a distinct family (Qwen vs the incumbents' Claude / GPT /
  Gemini) so its vote is *decorrelated* and adds signal rather than echoing an
  existing voice. Vision-capable (reads the evidence PNGs).
- **Invocation contract:** `opencode run "PROMPT" -m <model> -f img1 -f img2 …`.
  **The prompt MUST precede every `-f`** — `-f` takes a file array and swallows a
  trailing positional as another filename, leaving the model with no instruction.
  opencode prints the answer to stdout (TUI framing goes to stderr); the existing
  `parse_vote` JSON extraction handles it unchanged.
- **Ships OFF by default.** `DEFAULT_PANEL` is unchanged (still 3 voters). The 4th
  voter is only present under the opt-in `--panel v3-candidate` config; production
  waves are unaffected until the export rule is validated and the labeler tag is
  bumped.

Usage:

```bash
# 3-voter production panel (unchanged default)
matcher agent stitch-run --batch <dir>

# 4-voter candidate (opt-in; adds opencode/Qwen3-VL)
matcher agent stitch-run --batch <dir> --panel v3-candidate
# resumable: persists per group, --resume skips completed groups
matcher agent stitch-run --batch <dir> --panel v3-candidate --resume
```

Resume safety: `--resume` only reuses partial rows whose recorded provider set
matches the current panel — a `v3-candidate` resume over partials written by the
3-voter default ignores them and re-runs everything, so it can never silently
return cached 3-voter votes with the 4th voter skipped. Carried-forward groups
are also restricted to the current `--group-ids`/`--limit` selection (no leakage
into the final output), while unselected groups' rows are preserved in the
partial files for later crash recovery.

---

## Task 2 — Lone-dissenter accuracy (offline, 0 API calls)

**Question:** should a 4-voter panel auto-accept on **3/4 majority**, or require
**4/4-of-responding** (min quorum 3)? Equivalently: when the panel splits 2-vs-1,
is the lone dissenter's option ever the *better* one? If lone dissents carry real
signal, overriding them (3/4) ships errors.

**Method.** Across **all 11 historical batches** (`data/agents/stitching/
batches/*`), map each 3-voter group to a **settled human label** (`labeler=brad`,
the only true ground truth — `panel_unanimous_v1` rows are agent exports and were
excluded) via `stitch_eval`'s edge-overlap mapping. Keep groups with a clean 2-1
choice split. For each, compare the **dissenter's** edge set vs the **majority's**
edge set against the settled label using **sliver-filtered edge F1** (junction
slivers removed from both sides). Dissenter "right" = strictly higher F1.

**Results** (11 dissent instances across 10 distinct groups; all mapped to
Boston-roads brad labels — the 2 brad sidewalk labels didn't intersect any 2-1
split group):

| Split | dissenter-right | notes |
|---|---|---|
| **Overall** | **4/11 = 36%** (+1 tie) | well above the 20-25% ship-threshold |
| by distinct group | 4/10 = 40% | |

| Dissenting provider | n | dissenter-right | mean F1 (diss / maj) |
|---|---|---|---|
| **claude** | 4 | **3/4 (75%)** | 0.45 / 0.29 |
| **agy** | 1 | 1/1 (100%) | 0.16 / 0.15 |
| **codex** | 6 | **0/6 (0%)** | 0.27 / 0.62 |

Dataset split: all 11 instances are **roads** (Boston). No brad-labeled *sidewalk*
group had a 2-1 split, so the "agy-on-sidewalks" axis is measured in Task 3
(qwen's decorrelation) rather than here.

**Reading the math.** A lone dissent is *not noise* — 36% of the time the single
holdout is closer to the settled label than the two that agree. Crucially the
signal is **provider-dependent**: when **claude** is the holdout it is right 75%
of the time; when **codex** is the holdout it is right 0% of the time (codex's
lone dissents are consistently the worse option — often a NONE/ABSTAIN or an
over-trimmed set). A blanket 3/4 rule overrides *all* of them equally, so it would
ship the claude-holdout errors.

**Decision.** 36% > 25% ⇒ by the stated rule, **3/4 majority auto-accept ships
real errors**. Recommend **4/4-of-responding, min quorum 3**.

This is also the *natural extension of the current rule*: the 3-voter production
panel already requires **3/3 unanimity** to auto-accept. "4/4-of-responding
(min-3)" simply preserves that unanimity guarantee at panel size 4 (all responders
must agree; tolerate at most one abstention). A 3/4 rule would be a **weakening**
of today's bar — exactly the direction Task 2 says is unsafe. A lone dissent
should route to human review, not be overridden.

*(Caveat: n=11 is small and roads-only. The direction is robust — lone dissents
are far from noise — and it errs safe: 4/4 only ever demotes a would-be
auto-accept to human review; it never emits a wrong label.)*

---

## Task 3 — Quota-frugal 4-voter validation (12 opencode calls)

**Method.** No re-run of the full panel. Reused the **stored incumbent votes** for
the two `*_panelv2check` batches (8 Boston + 4 Seattle = 12 groups; see
`panel_v2_validation.md`) and ran **only qwen** on the same packs, then simulated
the 4-voter panel offline.

### qwen vote quality

| Metric | Value |
|---|---|
| valid-vote rate | **12/12 (100%)** |
| format compliance | 100% (clean JSON, parsed first try, no retries) |
| abstentions / errors | 0 / 0 |
| latency (mean / median) | **5.2s / 5.4s** — faster than codex (10s), agy (16s), claude (30s) |

qwen's rationales are on-topic and evidence-grounded — they cite alignment
fractions, sliver artifacts, road class (footway vs residential), and cross-mode
concerns, showing it actually reads the pack. Verbatim examples:

- **166ce59a → A:** *"Both R1 and R2 are residential segments named 'Salcombe
  Street' that geometrically align with T1 ('SALCOMBE STREET'). The overview shows
  they form a continuous physical road that T1 represents. Option A correctly
  includes both valid matches, while B and C exclude one valid alignment."*
- **0b3a4f7d → B (a wrong dissent):** *"Option B correctly matches the physical
  road alignment by including only the high-confidence edges … It excludes
  R4->T1, R5->T1, and R8->T2 which appear to be sliver artifacts or have low
  target alignment scores."* — over-trimmed; the settled label was A (all edges).
- **e919f4ab → NONE (a wrong dissent):** *"All segments are footways … no option
  represents a clear physical road correspondence … [option A's] T1 and T2 appear
  to be parallel but separate paths, violating the guidance against matching
  parallel-but-separate roads."* — the incumbents unanimously (correctly) chose A.

**Bias note:** qwen leans toward **under-inclusion** (dropping edges it labels
"slivers"). Both of its wrong dissents on clean Boston groups were over-trims.

### Decorrelation check

Does qwen just echo an incumbent? Choice agreement across the 12 groups:

| qwen agrees with | rate |
|---|---|
| agy | **9/12 (75%)** |
| claude | 7/12 (58%) |
| codex | 7/12 (58%) |

qwen is **most correlated with agy**, and specifically on **sidewalks**: on the
two Seattle groups where agy was the lone incumbent holdout (46e57794, 670e939f),
qwen sided with agy — turning agy's 1-vote dissent into a 2-2 tie. So qwen is a
genuinely independent voice on roads but partially **amplifies agy on sidewalks**
rather than being fully orthogonal there. It still adds value (it breaks false
majorities on ambiguous sidewalk groups → more human review, the safe direction),
but the decorrelation is imperfect on the sidewalk axis.

### 4-voter simulation (incumbents + qwen)

| type | group | claude | codex | agy | qwen | incumbent | 4v top (n) | 3/4? | 4/4? | F1 vs settled |
|---|---|---|---|---|---|---|---|---|---|---|
| roads | 07632e1f | A|A|A|A | unanimous | A (4) | ✅ | ✅ | 1.00 |
| roads | 0b3a4f7d | A|A|A|**B** | unanimous | A (3) | ✅ | ✖ | 1.00 |
| roads | 0e3e10ad | A|**B**|A|A | majority | A (3) | ✅ | ✖ | 1.00 |
| roads | 166ce59a | A|A|A|A | unanimous | A (4) | ✅ | ✅ | 1.00 |
| roads | 37a546e3 | A|A|A|A | unanimous | A (4) | ✅ | ✅ | 0.95* |
| roads | 461ebf00 | A|A|A|A | unanimous | A (4) | ✅ | ✅ | 1.00 |
| roads | 4bcea059 | A|A|A|A | unanimous | A (4) | ✅ | ✅ | 1.00 |
| roads | c3a963e9 | A|A|A|A | unanimous | A (4) | ✅ | ✅ | 1.00 |
| sidewalks | 2b99c180 | C|**B**|C|**B** | majority | C (2) | ✖ | ✖ | 0.71 |
| sidewalks | 46e57794 | E|E|**A**|**A** | majority | E (2) | ✖ | ✖ | 0.93 |
| sidewalks | 670e939f | D|D|**A**|**A** | majority | D (2) | ✖ | ✖ | 0.89 |
| sidewalks | e919f4ab | A|A|A|**NONE** | unanimous | A (3) | ✅ | ✖ | 1.00 |

*\*37a546e3 is an option-coverage gap (settled 11-edge set not offered), not a
disagreement — carried over from the v2 validation.*

**Export-rule comparison on the 12 mapped groups:**

| rule | fires on | exact-vs-settled | mean F1 |
|---|---|---|---|
| **3/4 majority** | 9/12 | 8/9 (89%) | 0.995 |
| **4/4-of-responding** | 6/12 | 5/6 (83%) | 0.992 |

On *this clean sample*, 3/4 happens to fire on 3 extra groups that are all exactly
correct (0b3a4f7d, 0e3e10ad, e919f4ab) — because in each, the lone dissent was
**qwen (or codex) being wrong** and the majority right. That is the *opposite* of
Task 2's broader finding, and it is a **selection artifact**: `panelv2check` groups
were hand-picked as *clean settled examples* for v2 validation, so their splits
are dominated by a wrong-dissenter. The broad, unfiltered brad set (Task 2) is the
representative population, and there lone dissents are right 36% of the time.

**Net:** on this sample the 4th voter caught **zero** incumbent false-unanimities
(there were none — all 8 incumbent-unanimous groups were correct) and produced
**2 false demotions** (0b3a4f7d, e919f4ab, where qwen wrongly broke a correct
unanimity). The 4th voter's value is therefore *not* demonstrated on this easy
sample; it rests on Task 2's evidence that dissents carry signal on harder groups,
plus qwen's clean decorrelation on roads.

---

## Recommended export rule

**Adopt 4/4-of-responding, min quorum 3.** Concretely, for a 4-voter panel:

- Auto-accept **iff** all *responding* voters agree on the same non-NONE option
  **and** ≥3 voters responded (≤1 abstention tolerated).
- Any 3/4 (or 2-2) split → **human review**, regardless of which voter dissents.

Rationale: preserves the current unanimity guarantee (today's rule is 3/3); Task 2
shows lone dissents are right 36% of the time (>25%), so overriding them ships
errors; the cost of 4/4 is bounded (demotion to review, never a wrong label).

**Do NOT adopt 3/4 majority** — it weakens today's unanimity bar in the one
direction Task 2 flags as unsafe.

**What ships now vs the recommended rule:** `compute_consensus` is intentionally
UNCHANGED in this PR. The current code is *stricter* than the recommendation:
unanimity requires all panelists to respond AND agree (zero abstentions
tolerated), so with 4 voters a single opencode abstention (e.g. an OpenRouter
quota blip) blocks auto-accept for that group. That is safe (demotion to review,
never a wrong label) but wasteful at scale; the "≤1 abstention tolerated"
relaxation is implemented at rollout step 4 together with the default flip, once
the wider-wave evidence is in.

## Rollout plan (no default flip in this PR)

1. **Now (this PR):** invoker + `v3-candidate` panel + resumable driver land, all
   **OFF by default**. No labels exported, no default changed.
2. **Widen the evidence base:** run `--panel v3-candidate` on the next real wave
   (e.g. Berlin roads) and any new sidewalk batch, so qwen's decorrelation and the
   4/4 rule are tested on *unselected* groups (the panelv2check sample is too
   clean/small to be decisive on its own).
3. **Sidewalk decorrelation:** confirm whether qwen's agy-correlation on sidewalks
   persists at scale; if qwen mostly echoes agy there, consider a different 4th
   model for sidewalk waves (the road decorrelation is already good).
4. **Flip the default** to `v3-candidate` + implement the 4/4-of-responding
   consensus rule **only after** (2)-(3) confirm the 4th voter breaks real
   false-unanimities more often than it false-demotes.
5. **Labeler identity bump:** adopting 4 voters changes panel composition ⇒ a new
   export labeler tag (`panel_*_v4…`, exact name TBD). **Deferred** — pack
   enrichment is being A/B-tested concurrently (do not touch
   `stitch_evidence` / `stitch_options`); the v4 tag is decided jointly with that
   outcome.

## API budget

**13 opencode calls total** (1 format probe + 12 panelv2check groups). Well under
the ~20 budget. All other analysis was offline over stored votes + settled labels.
