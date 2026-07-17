# v7 physical/frontage stitching panel wave — Codex findings

**Analysis date:** 2026-07-17  
**Analyst seat:** Codex (`gpt-5.6-sol`)  
**Source:** `data/agents/stitching/batches/*physical_context_v7_20260715*/{votes.csv,consensus.csv,batch.json}`

## Executive summary

- The raw files contain **8 datasets, 23 batch directories, 65 group-runs, 50 unique groups, and 195 ballots** (65 × 3 seats), not 282 ballots. All 195 expected provider keys are present, with **0 duplicate seat keys, 0 errors, and 0 abstains**. The prompt's 282-ballot count is therefore not reproducible from the specified files; all ballot rates below use the auditable denominator of 195.
- The controls provide **qualified support**, not validation, for the context hypothesis. Relative to enriched, removing physical context changed **4/15 choices**; removing coincidence changed **3/15**; minimal changed **7/15**. The important changes are concentrated in `fb8f359f`, `1b90f03b`, and `92c0997f`; `7bac1f1d` and `18ef284e` were completely choice-stable.
- The 2×2 result is an interaction, not a clean main effect. NONE ballots by cell were **8/15 enriched, 5/15 no-physical, 8/15 no-coincidence, and 2/15 minimal**. On the NONE indicator, the descriptive physical contrast is +30 percentage points and the coincidence contrast +10 points, but the difference-in-differences is -20 points. With only five repeated groups and no truth labels, those figures are behavioral—not accuracy estimates.
- Minimal is meaningfully worse on identity safety despite similar confidence: its panel choice differs from enriched on **3/5 controls**, and it turns `fb8f359f` from unanimous NONE to unanimous A and `1b90f03b` from unanimous NONE to majority A. The accompanying text says the minimal panel is treating ramp/mainline or coincident named roads as full M:N continuity. Mean control confidence barely moves (**0.799 enriched vs 0.787 minimal**), so confidence does not warn about the regression.
- `92c0997f` is the strongest caution against declaring success. Only enriched produces majority C; the other three cells produce majority A. Yet enriched Codex says every option—including C—contains `e15`, conflating coincident tunnel T3 and surface service-road T7: **“Because every offered set contains e15, all options contain at least one false edge.”** Context changed the answer, but the menu may still prevent a correct answer.
- NONE is predominantly an expressibility signal, not abstention or uncertainty. Of **74 decisive NONE ballots**, I classify **56 (75.7%) as no-exact-option/expressibility gaps, 11 (14.9%) as genuine empty-set reject-alls, and 7 (9.5%) as insufficient evidence**. True abstains are **0**. This strongly supports both a `none_reason` enum and exact-edge-set option work while preserving NONE's existing reject-all semantics.
- The seats have clear directional personalities. Claude is the least confident (**0.625**) and agrees with a defined majority **50/59 (84.7%)**. Codex is highly confident (**0.928**) and is usually the conservative dissenter—**8/12** of its majority-run dissents are NONE against an option majority. Muse is more inclusionary: only **15/65 NONE**, **25/65 A**, and **10/12** of its dissents are an option against a NONE majority.
- Splits are not concentrated in ablations: **30/50 enriched (60%)** and **9/15 ablated (60%)** group-runs are non-unanimous. The five controls alone are somewhat noisier when ablated (**2/5 enriched splits vs 9/15 ablated splits**), but there is no wave-wide concentration signal.
- **Disposition: fix expressibility; do not bless v7 yet.** Context appears useful enough to retain, particularly for `fb8f359f`, `1b90f03b`, `7bac1f1d`, and vertical reject-all cases. But 56 expressibility NONEs, 59/65 human-review routes, six three-way splits, and the unresolved `92c0997f` false-edge problem make the current panel/menu combination unsuitable for blessing. Rubric clarification should follow, especially for duplicate-vs-split-carriageway and anchor-vs-clip decisions.

## Scope, denominators, and checks

The 23 expected batch directories are present: eight enriched dataset batches, plus three ablations for each of the five control datasets. Enriched contains 50 group-runs; each ablation contains the five controls. Across controls, each `group_id` has one stable `option_menu_sha256` and one stable `displayed_candidate_universe_sha256` across all four variants, so letter choices are directly comparable.

The files contain 65 `consensus.csv` rows and exactly three `votes.csv` rows per group-run. This yields 195 ballots. The user prompt mentions 282, but neither the authoritative document's “one row per seat per group” semantics nor the specified raw files support that number. I did not substitute archived data or count retries as extra ballots.

Consensus over all 65 group-runs is **26 unanimous, 33 majority, and 6 no-majority**. Routing is **6 auto-accept and 59 human-review**; enriched alone is 5 auto-accept and 45 human-review. NONE is always counted as a valid, decisive choice. No row has `abstain_reason` or `error`.

## 1. Per-seat behavior

### Choice distribution

Counts are out of 65 ballots per provider.

| Provider | A | B | C | D | E | F | G | H | I | NONE | NONE rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claude | 12 | 5 | 5 | 2 | 1 | 0 | 1 | 8 | 4 | 27 | 41.5% |
| codex | 15 | 6 | 0 | 1 | 1 | 1 | 2 | 2 | 5 | 32 | 49.2% |
| muse | 25 | 10 | 3 | 1 | 0 | 0 | 2 | 5 | 4 | 15 | 23.1% |

### Confidence, abstention, and majority agreement

“Majority agreement” is calculated on the 59 group-runs with a modal choice supported by at least two seats; the six all-different runs have no panel majority and are excluded from that rate.

| Provider | Mean confidence | Median | NONE confidence | Option confidence | Abstains | Agrees with majority |
|---|---:|---:|---:|---:|---:|---:|
| claude | 0.625 | 0.580 | 0.578 | 0.658 | 0/65 | 50/59 (84.7%) |
| codex | 0.928 | 0.940 | 0.927 | 0.929 | 0/65 | 47/59 (79.7%) |
| muse | 0.801 | 0.820 | 0.791 | 0.804 | 0/65 | 47/59 (79.7%) |

Pairwise agreement is low for a three-seat adjudication panel: Claude–Codex **38/65 (58.5%)**, Claude–Muse **38/65 (58.5%)**, and Codex–Muse **35/65 (53.8%)**.

### Characteristic reasoning style and directional dissent

- **Claude** averages **140.5 reasoning words** and **96.5 feedback words**. It explicitly cites MI/SA rules in 58/65 ballots, commonly enumerates the exact edge set, and calibrates down when zooms or semantics are missing. Its nine majority-run dissents are mixed: 3 NONE against option, 2 option against NONE, and 4 option-vs-option. Claude is not the systematic directional pole.
- **Codex** is the most concise (**66.7 reasoning words**, **50.3 feedback words**) and most confident. It explicitly discusses exact/menu semantics in **41/65** ballots but cites MI/SA labels only 2/65 times. Its confidence is almost invariant to verdict (0.927 NONE vs 0.929 option), which is a calibration concern. Of 12 majority-run dissents, **8 are NONE against an option majority**, 3 are option-vs-option, and 1 is an option against a NONE majority.
- **Muse** is the most expansive (**169.1 reasoning words**) and numerical: 56/65 reasonings contain percentages/fractions and 64/65 cite MI/SA rules. It favors inclusion (A on 25/65; NONE on only 15/65). Of 12 majority-run dissents, **10 are an option against a NONE majority**. Muse is the systematic inclusionary pole opposite Codex.

There is therefore no single seat that simply dissents most: Codex and Muse tie at 12 majority-run dissents. The meaningful pattern is directional—Codex tends to demand an exact menu match and return NONE, while Muse tends to preserve a broad M:N option.

## 2. Enriched vs no-physical: physical-evidence effect

Across the 15 paired control ballots, removing physical evidence changed **4 choices** and **10 confidence values**. Mean confidence fell from 0.799 to 0.776: paired mean delta (no-physical minus enriched) **-0.0233**, mean absolute delta 0.0433, range -0.14 to +0.10.

| Provider | Choice changes | Changed ballots | Mean confidence delta | Per-group confidence deltas |
|---|---:|---|---:|---|
| claude | 0/5 | none | -0.014 | `1b90f03b` +0.05; `18ef284e` -0.04; `92c0997f` 0; `fb8f359f` -0.08; `7bac1f1d` 0 |
| codex | 1/5 | `92c0997f` NONE → A | -0.036 | `1b90f03b` -0.07; `18ef284e` -0.03; `92c0997f` 0; `fb8f359f` -0.08; `7bac1f1d` 0 |
| muse | 3/5 | `1b90f03b` NONE → G; `92c0997f` C → A; `fb8f359f` NONE → A | -0.020 | `1b90f03b` -0.14; `18ef284e` 0; `92c0997f` -0.04; `fb8f359f` +0.10; `7bac1f1d` -0.02 |

Physical evidence therefore has a visible effect, but it is heavily seat- and group-concentrated: Muse supplies three of four choice changes, and `92c0997f` supplies two. Claude never changes choice.

NONE count falls from **8/15 enriched to 5/15 no-physical**. Every NONE in both of these cells is an expressibility-gap flavor; no paired ballot retains NONE but changes its flavor. The substantive transitions are three expressibility NONEs becoming offered options: Codex on `92c0997f`, Muse on `1b90f03b`, and Muse on `fb8f359f`.

The text makes the intended mechanism plausible:

- In enriched `fb8f359f`, Claude rejects `e10` because **“R2 layer-0 ground → T8 bridge is a grade-separation pass-under”** and Muse flags layer mismatch plus same-side duplication. Without physical context, Muse switches to A and treats the block as legitimate M:N.
- In enriched `92c0997f`, Codex uses the tunnel/ground conflict between T3 and T7 to return NONE. With physical evidence removed it switches to A: **“The tertiary/service class differences do not outweigh the continuous same-name, same-direction geometry.”**
- `18ef284e` and `7bac1f1d` do not move. Their service/road roles and names alone appear sufficient for all three seats to reconstruct the same decision even when the explicit physical panel is absent.

This supports a physical-evidence effect on some hard cases, but not a general one across the five controls.

## 3. Enriched vs no-coincidence: coincidence-context effect

Removing coincidence context changes **3/15 choices**, all on `92c0997f`, and changes 13 confidence values. Mean confidence is effectively unchanged: 0.799 enriched vs 0.798 no-coincidence; paired mean delta **-0.0013**, mean absolute delta 0.0533, range -0.17 to +0.14.

| Provider | Choice changes | Changed ballot | Mean confidence delta | Per-group confidence deltas |
|---|---:|---|---:|---|
| claude | 1/5 | `92c0997f` C → NONE | -0.034 | `1b90f03b` +0.08; `18ef284e` -0.17; `92c0997f` -0.07; `fb8f359f` -0.08; `7bac1f1d` +0.07 |
| codex | 1/5 | `92c0997f` NONE → A | +0.014 | `1b90f03b` -0.01; `18ef284e` +0.05; `92c0997f` +0.02; `fb8f359f` 0; `7bac1f1d` +0.01 |
| muse | 1/5 | `92c0997f` C → A | +0.016 | `1b90f03b` -0.04; `18ef284e` -0.04; `92c0997f` +0.02; `fb8f359f` +0.14; `7bac1f1d` 0 |

NONE remains **8/15**, but flavor changes matter:

- Claude on `fb8f359f` remains NONE but moves from expressibility to **insufficient evidence**: the no-coincidence ballot says the exact ramp/mainline boundary is not determinable.
- Claude on `92c0997f` moves from option C to **insufficient-evidence NONE**, saying the option images are physically indistinguishable and the R1/R2 relationship cannot be determined.
- Codex on `92c0997f` moves from expressibility NONE to A, while Muse also moves C → A.

Thus the measurable coincidence effect is entirely one control's three-seat re-interpretation, plus a flavor shift on `fb8f359f`. It is not a broad main effect.

## 4. Full physical × coincidence 2×2 interaction

Cells below are `choice @ confidence`. P=physical, C=coincidence.

| Group | Provider | P+C enriched | C only no-physical | P only no-coincidence | Neither minimal |
|---|---|---|---|---|---|
| `18ef284e` | claude | H @ 0.72 | H @ 0.68 | H @ 0.55 | H @ 0.70 |
|  | codex | NONE @ 0.91 | NONE @ 0.88 | NONE @ 0.96 | NONE @ 0.96 |
|  | muse | H @ 0.88 | H @ 0.88 | H @ 0.84 | H @ 0.91 |
| `1b90f03b` | claude | NONE @ 0.50 | NONE @ 0.55 | NONE @ 0.58 | A @ 0.66 |
|  | codex | NONE @ 0.97 | NONE @ 0.90 | NONE @ 0.96 | NONE @ 0.86 |
|  | muse | NONE @ 0.82 | G @ 0.68 | NONE @ 0.78 | A @ 0.68 |
| `7bac1f1d` | claude | I @ 0.72 | I @ 0.72 | I @ 0.79 | I @ 0.72 |
|  | codex | I @ 0.98 | I @ 0.98 | I @ 0.99 | I @ 0.96 |
|  | muse | I @ 0.93 | I @ 0.91 | I @ 0.93 | I @ 0.86 |
| `92c0997f` | claude | C @ 0.57 | C @ 0.57 | NONE @ 0.50 | C @ 0.40 |
|  | codex | NONE @ 0.94 | A @ 0.94 | A @ 0.96 | A @ 0.96 |
|  | muse | C @ 0.82 | A @ 0.78 | A @ 0.84 | A @ 0.84 |
| `fb8f359f` | claude | NONE @ 0.58 | NONE @ 0.50 | NONE @ 0.50 | A @ 0.58 |
|  | codex | NONE @ 0.97 | NONE @ 0.89 | NONE @ 0.97 | A @ 0.93 |
|  | muse | NONE @ 0.68 | A @ 0.78 | NONE @ 0.82 | A @ 0.78 |

### Cell summaries

| Cell | NONE ballots | Mean confidence | Unanimous runs | Panel choices by group order (`18ef`, `1b90`, `7bac`, `92c0`, `fb8f`) |
|---|---:|---:|---:|---|
| enriched P+C | 8/15 | 0.799 | 3/5 | H, NONE, I, C, NONE |
| no-physical C only | 5/15 | 0.776 | 1/5 | H, NONE, I, A, NONE |
| no-coincidence P only | 8/15 | 0.798 | 3/5 | H, NONE, I, A, NONE |
| minimal neither | 2/15 | 0.787 | 2/5 | H, A, I, A, A |

Using NONE as a descriptive anti-overmerge indicator, physical's average contrast is **+30 points**: mean NONE rate is 53.3% with P versus 23.3% without P. Coincidence's contrast is **+10 points**: 43.3% with C versus 33.3% without C. The interaction contrast is **-20 points**: physical adds 20 points when coincidence is on but 40 when it is off. These are repeated ballots from only five groups and must not be treated as independent-binomial or accuracy estimates.

The group-level interaction is more informative:

- `fb8f359f` shows **redundant protection**: either factor alone sustains majority/unanimous NONE, while neither yields unanimous A. Physical plus coincidence makes all three seats reject the menu.
- `1b90f03b` is similar: either factor sustains panel NONE, full context makes it unanimous, and minimal yields majority A. Codex's persistent minimal NONE shows names/corridor identity can sometimes substitute for both experimental fields.
- `92c0997f` shows a **both-needed interaction for majority C**. Neither single factor reproduces enriched C. That is not proof C is correct; Codex's enriched expressibility objection says C still carries a tunnel-to-service false edge.
- `7bac1f1d` and `18ef284e` show **no choice effect at all**. `7bac1f1d` is unanimous I in all cells; `18ef284e` is H/H/NONE in all cells.

Only **7/15 seat/group pairs** retain one choice across all four cells. However, no factor is cleanly dominant: physical produces a larger NONE contrast and four enriched-pair changes versus coincidence's three, but those changes are concentrated in Muse and `92c0997f`. The defensible conclusion is “case-specific interaction,” not “physical dominates” or “coincidence dominates.”

Minimal differs from enriched on **7/15 choices** and **3/5 panel choices**. Its mean confidence is only 0.013 lower, and unanimity falls by just one run (2/5 vs 3/5), illustrating why confidence and consensus strength alone would not reveal the nature of the likely identity regression.

## 5. Frontage/service-road and vertically layered ambiguity

### Text search scope

I searched `reasoning + pack_feedback` case-insensitively. “Frontage/service” means `frontage` or `service-road/service road`. “Vertical” means overpass, underpass, bridge, tunnel, vertical, layer, elevated, pass-under, or grade-separation. These are mention counts, not independently verified group labels; the vertical regex is intentionally broad because layer evidence is often discussed in missing-info feedback.

- Across all variants, frontage/service terms occur in **32/195 ballots**, **21 group-runs**, and **12 unique groups**. In enriched they occur in **16/150 ballots across 11/50 groups**.
- Vertical terms occur in **136/195 ballots**, **55 group-runs**, and **43 unique groups**. In enriched they occur in **107/150 ballots across 43/50 groups**.

Enriched frontage/service groups are:

- Sydney: `35329743`
- Berlin: `7bac1f1d`
- Helsinki: `4dc33ddd`, `7175635e`, `92c0997f`, `e085519d`
- London: `91570f54`
- Hong Kong: `18ef284e`, `4eed5e80`, `8582dd97`, `b049e0de`

Enriched vertical-term groups are:

- Sydney: `1de025b8`, `35329743`, `53bb11d7`, `66e22055`, `750ae089`, `8f152b92`, `fb8f359f`
- Geneva: `a451bf05`, `ca8d1f92`, `dd106a0f`, `e4746a04`
- Berlin: `33a36ca5`, `3b876df0`, `3f53c7e7`, `4148382c`, `422d5d7b`, `7bac1f1d`, `d4d2e782`, `dc818edc`
- Helsinki: `4dc33ddd`, `5faa0b72`, `7175635e`, `7680bb19`, `92c0997f`, `e085519d`, `f4f3387b`
- London: `1b90f03b`, `729f879b`, `91570f54`, `e8a39e6d`, `ea25e0bd`
- Hong Kong: `00e8e9fd`, `18ef284e`, `4eed5e80`, `8582dd97`, `b049e0de`, `cd320a3c`, `e0099fb8`
- Amsterdam: `5e31936e`, `61a926e3`, `6775ade1`, `bdbdf792`
- Philadelphia: `b7f57035`

### Evidence that context can prevent bad planar merges

- **`fb8f359f`** is the cleanest control signal. Minimal is unanimous A; enriched is unanimous NONE. Enriched Muse identifies target duplication and says, **“when two segments occupy same centerline you must pick the representation whose role/network continuity fits, not keep both.”** Claude independently rejects a layer-0-to-bridge pass-under edge. Either physical or coincidence alone keeps the panel away from unanimous A.
- **`1b90f03b`** changes from minimal majority A to enriched unanimous NONE. All enriched seats isolate `e1`, where Harrier Avenue T2 coincides with the Eastern Avenue tunnel tile T4. Codex states: **“T2 is a separate Harrier Avenue corridor coincident with T4.”** No menu offers “all except e1.”
- **`7bac1f1d`** is unanimous I in all four cells, so it is not evidence of a treatment effect, but it is a strong correctness-pattern example. Enriched Codex rejects Mäusetunnel edges because **“the underground footway/steps ... are different physical features and network roles from the surface road.”** The stable result suggests name/role data already carried most of this case.
- **`8f152b92`** is enriched-only and unanimously B. Codex rejects `e4` because surface Craigend Street and the Cross City Tunnel are **“vertically and functionally distinct traveled ways.”** This is exactly the desired anti-planar-merge reasoning.
- The genuine reject-all clusters `422d5d7b` and `d4d2e782` are unanimous NONE: all seats recognize layer -1 indoor footways under layer-0 vehicular roads. `4eed5e80` has two NONEs against Muse A for Kai Tak Tunnel under Kowloon City Road; it merits human adjudication because the same evidence did not produce unanimity.

### Evidence the panel can preserve legitimate continuity

The enriched panel does not blindly split every layer/class transition:

- `8582dd97` is unanimous A. Codex says the Tsing Tsuen Road **“name and layer transitions near the interchange do not change movement intent and are consistent with segmentation and attribute-boundary differences.”** Muse backs that with full coverage sums and treats two short overlaps as junction anchors.
- `7175635e` is unanimous A despite cycleway/service and partial-bridge metadata. Muse notes the three reference-alignment fractions sum to 1.005 and treats them as an exact corridor partition.
- `91570f54` is unanimous A: all seats retain a short 13 m service-road boundary anchor because it completes the same-name Blue Lion Place corridor.
- `18ef284e` stays H/H/NONE across all variants. Claude and Muse exclude layer-0 service clips against elevated T6 while retaining the long tunnel/bypass matches; Codex's NONE is not over-splitting but a missing-menu dispute over three short mainline anchors.

### Skeptical conclusion

The treatment steers the panel away from merge-all A on `fb8f359f`, `1b90f03b`, and `92c0997f`, and the text ties those shifts to the intended physical/coincidence evidence. It also coexists with unanimous continuity-preserving decisions on `8582dd97`, `7175635e`, and `91570f54`. That is encouraging.

It is not enough to claim that context improves accuracy. There are no human truth labels in this wave; only five controls support paired comparison; two controls do not change; and enriched `92c0997f` may still over-merge tunnel T3 with service T7 because the correct exclusion is unavailable. The strongest defensible claim is that context changes reasoning in the intended direction on selected cases without inducing universal over-splitting—not that it has been proven correct.

## 6. NONE forensics

### Classification method

All 74 NONE ballots are decisive. I used this precedence:

1. **G = genuine reject-all:** the voter explicitly says the accepted edge set is empty/every candidate edge is a no-match.
2. **I = insufficient evidence:** the voter says the evidence cannot determine the accepted set or decide among identity interpretations.
3. **E = expressibility/no exact option:** the voter identifies a nonempty accepted set, a required omitted edge, or mandatory false edges in every menu option. Residual uncertainty does not override E if the voter identifies a menu defect that makes every option inexact.

This distinction matters because “every option is wrong” alone is ambiguous: it is G only when the desired set is empty; it is E when a nonempty correct set is missing.

### Counts

| Provider | Expressibility E | Genuine reject-all G | Insufficient I | Total NONE |
|---|---:|---:|---:|---:|
| claude | 18 | 4 | 5 | 27 |
| codex | 25 | 5 | 2 | 32 |
| muse | 13 | 2 | 0 | 15 |
| **Total** | **56 (75.7%)** | **11 (14.9%)** | **7 (9.5%)** | **74** |

| Variant | E | G | I | Total NONE |
|---|---:|---:|---:|---:|
| enriched | 43 | 11 | 5 | 59 |
| no-physical | 5 | 0 | 0 | 5 |
| no-coincidence | 6 | 0 | 2 | 8 |
| minimal | 2 | 0 | 0 | 2 |

Representative intent is unusually explicit:

- Expressibility: `1b90f03b`/codex/enriched—**“No option contains exactly those 13 edges.”**
- Genuine reject-all: `d4d2e782`/codex/enriched—**“The correct accepted edge set is empty, which no offered option represents.”**
- Insufficient evidence: `35329743`/codex/enriched—**“The evidence cannot establish an exact final set.”**

### Every NONE ballot

Rows list only providers that voted NONE. This table covers all 74 ballots. Dataset names are retained to disambiguate provenance.

| Dataset | Variant | Group | NONE providers and flavor |
|---|---|---|---|
| au_sydney_roads | enriched | `1de025b8` | codex=E, muse=E |
| au_sydney_roads | enriched | `35329743` | codex=I |
| au_sydney_roads | enriched | `53bb11d7` | codex=E, muse=E |
| au_sydney_roads | enriched | `66e22055` | codex=G |
| au_sydney_roads | enriched | `750ae089` | claude=E, codex=E |
| au_sydney_roads | enriched | `fb8f359f` | claude=E, codex=E, muse=E |
| au_sydney_roads | no-coincidence | `fb8f359f` | claude=I, codex=E, muse=E |
| au_sydney_roads | no-physical | `fb8f359f` | claude=E, codex=E |
| ch_grand_geneva_cycle_schema | enriched | `ca8d1f92` | claude=I |
| ch_grand_geneva_cycle_schema | enriched | `dd106a0f` | claude=E, codex=E |
| ch_grand_geneva_cycle_schema | enriched | `e4746a04` | claude=G, codex=G |
| de_berlin_roads | enriched | `3b876df0` | claude=I, codex=E |
| de_berlin_roads | enriched | `422d5d7b` | claude=G, codex=G, muse=G |
| de_berlin_roads | enriched | `d4d2e782` | claude=G, codex=G, muse=G |
| de_berlin_roads | enriched | `dc818edc` | codex=E |
| fi_helsinki_roads | enriched | `4dc33ddd` | claude=E |
| fi_helsinki_roads | enriched | `5faa0b72` | codex=I |
| fi_helsinki_roads | enriched | `7680bb19` | muse=E |
| fi_helsinki_roads | enriched | `92c0997f` | codex=E |
| fi_helsinki_roads | enriched | `e085519d` | claude=E, codex=E, muse=E |
| fi_helsinki_roads | no-coincidence | `92c0997f` | claude=I |
| gb_london_roads | enriched | `1b90f03b` | claude=E, codex=E, muse=E |
| gb_london_roads | enriched | `e8a39e6d` | claude=I, muse=E |
| gb_london_roads | enriched | `ea25e0bd` | muse=E |
| gb_london_roads | minimal | `1b90f03b` | codex=E |
| gb_london_roads | no-coincidence | `1b90f03b` | claude=E, codex=E, muse=E |
| gb_london_roads | no-physical | `1b90f03b` | claude=E, codex=E |
| hk_hongkong_roads | enriched | `00e8e9fd` | claude=E, codex=E |
| hk_hongkong_roads | enriched | `18ef284e` | codex=E |
| hk_hongkong_roads | enriched | `4eed5e80` | claude=G, codex=G |
| hk_hongkong_roads | enriched | `b049e0de` | claude=E |
| hk_hongkong_roads | enriched | `b8b5da4a` | claude=E, codex=E, muse=E |
| hk_hongkong_roads | enriched | `cd320a3c` | claude=E, codex=E |
| hk_hongkong_roads | minimal | `18ef284e` | codex=E |
| hk_hongkong_roads | no-coincidence | `18ef284e` | codex=E |
| hk_hongkong_roads | no-physical | `18ef284e` | codex=E |
| nl_amsterdam_roads | enriched | `6775ade1` | claude=E |
| nl_amsterdam_roads | enriched | `bdbdf792` | claude=E, codex=E |
| us_philadelphia_sidewalks | enriched | `17053a69` | claude=E, codex=E, muse=E |
| us_philadelphia_sidewalks | enriched | `b7f57035` | claude=E |
| us_philadelphia_sidewalks | enriched | `c8da4c08` | claude=E, codex=E, muse=E |

The most direct product implication is that a three-value `none_reason` is not optional metadata; it captures materially different decisions. The 56 E ballots also show that `none_reason` alone will not fix throughput. The option generator must expose the asserted exact set or permit a structured edge-level override. The 11 G ballots confirm that NONE's existing decisive empty-set/reject-all meaning must remain first-class rather than being conflated with abstention or a missing menu.

## 7. Disagreement map

### Concentration by variant

| Variant | Unanimous | Majority split | Three-way/no-majority | Any split |
|---|---:|---:|---:|---:|
| enriched | 20 | 24 | 6 | 30/50 (60%) |
| no-physical | 1 | 4 | 0 | 4/5 (80%) |
| no-coincidence | 3 | 2 | 0 | 2/5 (40%) |
| minimal | 2 | 3 | 0 | 3/5 (60%) |
| **Total** | **26** | **33** | **6** | **39/65 (60%)** |

The 15 ablated runs have 9/15 splits, exactly the enriched wave's 60% rate. On the controls alone, enriched is quieter (2/5 splits) than the combined ablations (9/15), but no individual ablation shows a monotone degradation. Splits are therefore not concentrated enough to claim that ablation generally destabilizes the panel.

### Majority-run dissenters

Notation is `group[variant]: majority → dissenter`, with E=enriched, NP=no-physical, NC=no-coincidence, M=minimal.

- **Claude, 9 dissents:** `4148382c[E]` B→H; `b049e0de[E]` B→NONE; `53bb11d7[E]` NONE→H; `1de025b8[E]` NONE→C; `92c0997f[NP]` A→C; `92c0997f[M]` A→C; `61a926e3[E]` A→B; `ca8d1f92[E]` A→NONE; `92c0997f[NC]` A→NONE.
- **Codex, 12 dissents:** `18ef284e[E/NP/NC/M]` H→NONE in all four runs; `1b90f03b[M]` A→NONE; `66e22055[E]` B→NONE; `729f879b[E]` A→G; `e8a39e6d[E]` NONE→B; `8dae3675[E]` A→H; `f4f3387b[E]` C→A; `5faa0b72[E]` H→NONE; `92c0997f[E]` C→NONE.
- **Muse, 12 dissents:** `3b876df0[E]` NONE→B; `4eed5e80[E]` NONE→A; `00e8e9fd[E]` NONE→A; `cd320a3c[E]` NONE→B; `fb8f359f[NP]` NONE→A; `750ae089[E]` NONE→B; `bdbdf792[E]` NONE→A; `1b90f03b[NP]` NONE→G; `dd106a0f[E]` NONE→C; `e4746a04[E]` NONE→A; `ea25e0bd[E]` D→NONE; `9f56d71d[E]` H→A.

Across the 33 majority splits, the fault-line types are **13 majority-NONE vs option**, **12 majority-option vs NONE**, and **8 option-vs-option**. Muse accounts for 10/13 inclusionary challenges to a NONE majority; Codex accounts for 8/12 conservative NONE challenges to an option majority.

### Six all-different enriched groups

| Dataset/group | Claude | Codex | Muse | Main fault line from text |
|---|---|---|---|---|
| Sydney `35329743` | E | NONE | A | Elevated-vs-ground representation; Codex says evidence cannot identify whether T2 represents both coincident R1/R5 or only elevated R5. |
| Berlin `dc818edc` | D | NONE | B | Tunnel mainline edge `e24`, ramp branch `e13`, and missing exact combination. |
| Helsinki `4dc33ddd` | NONE | F | A | Service-vs-cycleway same-side coincidence and asymmetric menu treatment of equivalent stubs. |
| Helsinki `7680bb19` | A | I | NONE | Same-corridor continuity versus duplicate target coverage and a 2.2 m sliver. |
| Amsterdam `6775ade1` | NONE | E | A | Tunnel portal/side-street ownership and a vertical-crossing edge. |
| Philadelphia `b7f57035` | NONE | A | D | Whether bridge-continuation `e10` is required and which endpoint clips are false. |

These six should not be summarized as random seat noise. Five involve a missing exact combination or representation/anchor policy; `35329743` is a direct insufficient-evidence case. They are enriched-only, so the full pack did not resolve them.

## 8. `pack_feedback` synthesis

All **195/195** `pack_feedback` values parse as JSON. They contain **388 `missing_info` items** (1.99/ballot) and **382 `ambiguities` items** (1.96/ballot). The theme coding below is nonexclusive and searches both arrays.

| Recurring theme | Ballots | Rate | Interpretation |
|---|---:|---:|---|
| Zoom/resolution/close-up | 145/195 | 74.4% | The dominant evidence request; option differences often occur at junction-scale geometry not visible in the supplied overview. |
| Anchor vs clip/sliver/segmentation | 136/195 | 69.7% | The dominant rubric decision: whether short complementary overlaps are true boundary anchors or endpoint artifacts. |
| Coincidence/carriageway/duplicate/offset | 125/195 | 64.1% | Panels cannot consistently distinguish duplicate digitization, abstract centerline-to-split carriageway M:N, and parallel facilities. |
| Role/class/name/tags | 121/195 | 62.1% | Missing or conflicting target roles/names repeatedly decide frontage, cycleway, ramp, and service-road identity. |
| Vertical/physical attributes | 111/195 | 56.9% | Even in a physical-context wave, voters request layer breakpoints, bridge/deck boundaries, tunnel portals, and confirmation that tags describe the aligned span. |
| Direction/connectivity/topology | 82/195 | 42.1% | Travel direction, endpoint connectivity, and adjacency are needed to separate same-way continuations from crossings/branches. |
| Option/menu/visualization | 52/195 | 26.7% | Menus omit asserted exact sets, and some option images are pixel-identical or do not expose the changed edges. |

The same themes are visible in direct feedback:

- **Junction detail:** `18ef284e`/claude/enriched requests a **“junction zoom ... to confirm layer separation vs true overlap.”** `1b90f03b` repeatedly requests zooms to determine whether `e6/e8` are straddles or clips.
- **Pixel-identical options:** `92c0997f`/claude/no-coincidence says **“all 9 option images look identical at this zoom.”** `fb8f359f`/claude/enriched says the option images **“gave no visual discrimination.”**
- **Carriageway semantics:** `92c0997f`/muse/enriched asks for heading/bearing to confirm whether R1/R2 are opposite carriageways. This is not merely missing evidence; the rubric needs a stable rule for centerline-to-carriageway representation versus duplicate suppression.
- **Physical tags at the aligned span:** `8582dd97`/muse/enriched names **“bridge tagging vs real level diff”** as the ambiguity. Segment-wide bridge/layer attributes can overstate a conflict if the aligned subline lies at an attribute transition.
- **Target capability gaps:** `fb8f359f`/claude/no-physical notes **“physical=unknown for all segments,”** while `18ef284e`/claude/no-physical asks for target class/physical roles. `batch.json` also shows uneven target capabilities: some datasets lack level entirely and Geneva/Philadelphia expose no bridge/tunnel flag domain.
- **Menu expressibility:** `1b90f03b`/muse/enriched explicitly requests an option with exactly 13 edges excluding only `e1`; `fb8f359f`/claude wants all except `e10`; `18ef284e`/codex wants H plus short anchors. These are generator inputs, not just prose complaints.

Three rubric ambiguities recur strongly enough to revise explicitly:

1. **Boundary anchor vs endpoint clip:** define the evidentiary threshold for complementary alignment fractions, same-direction geometry, and connectivity. `18ef284e`, `1b90f03b`, and many no-exact-option cases turn on this.
2. **Duplicate vs legitimate M:N representation:** specify when one centerline may map to both split carriageways and when same-side coincidence requires choosing only one representation. `92c0997f`, `fb8f359f`, `7680bb19`, and `e8a39e6d` expose inconsistent applications.
3. **Physical-attribute precedence:** clarify whether a layer/bridge/tunnel mismatch is decisive at the candidate's aligned subspan, how to treat segment-wide transition tags, and when names/roles can substitute for missing physical fields.

## What a human should review first

1. **`92c0997f` (Helsinki), all four variants.** This is the only control where all three seats change when coincidence is removed and where full P+C uniquely creates majority C. Verify the R1/R2 carriageway relationship and, most importantly, whether mandatory `e15` incorrectly merges tunnel T3 with surface service T7. This determines whether the apparent treatment success is real or merely a different wrong option.
2. **`fb8f359f` (Sydney), all four variants.** It has the starkest flip: enriched unanimous NONE versus minimal unanimous A. Adjudicate Western Motorway Onramp vs Homebush Bay Drive, T1/T5 same-side duplication, and R2/T8 bridge grade separation. Also determine the exact nonempty edge set so the option generator can learn from the case.
3. **`1b90f03b` (London), all four variants.** Verify whether Harrier Avenue T2 is a distinct coincident feature from Eastern Avenue tunnel T4 and whether `e6/e8` are true complementary boundary anchors. Three enriched NONEs independently assert “all except `e1`,” making this a high-value exact-option test.
4. **The unanimous empty-set cases `422d5d7b` and `d4d2e782` (Berlin), then `4eed5e80` (Hong Kong), `e4746a04` (Geneva), and `66e22055` (Sydney).** Confirm whether the empty edge set is correct. The first two are the clearest regression cases for a `reject_all_empty` NONE reason; the latter three test whether Muse or the option majority is over-merging vertical/parallel facilities.
5. **The six all-different groups:** `35329743`, `dc818edc`, `4dc33ddd`, `7680bb19`, `6775ade1`, and `b7f57035`. They remain unresolved despite full context and expose the exact evidence/rubric gaps most likely to survive future enrichment.
6. **`18ef284e` (Hong Kong), all variants.** The invariant H/H/NONE result isolates a menu/rubric disagreement rather than a treatment effect. Review Codex's three short mainline anchors (`e3`, `e15`, `e18`) against H; this is a clean anchor-vs-clip calibration case.
7. **Repeated unanimous expressibility groups:** `e085519d`, `17053a69`, and `c8da4c08`. The panel broadly agrees on what is wrong but cannot select the exact set. These are ideal regression fixtures for exact-pair option generation.

## Disposition and concrete recommendations

**Recommended disposition: fix expressibility; do not bless v7 yet.** Retain the physical and coincidence context fields while making the following changes before another calibration wave:

1. Add a required `none_reason` enum with at least `reject_all_empty`, `exact_set_not_offered`, and `insufficient_evidence`. Preserve NONE as a decisive vote; never map it to abstain.
2. Generate exact-pair alternatives from seat-identified required/forbidden edges, and preferably allow a structured edge-level override when no enumerated option is exact. Keep NONE itself as the explicit, decisive reject-all-menu choice; use `reject_all_empty` when the intended edge set is actually empty.
3. Revise the rubric for anchor-vs-clip, duplicate-vs-split-carriageway M:N, and subspan physical-attribute precedence. Include worked examples from `18ef284e`, `92c0997f`, and `fb8f359f`.
4. Improve packs with junction-scale crops, highlighted option deltas, travel direction/connectivity, aligned-subspan layer breakpoints, and target role/class capabilities. Avoid option images that differ only in invisible edge selections.
5. Human-adjudicate the five controls and rerun the same 2×2 after the menu/rubric changes. Add targeted controls from the genuine reject-all and all-different groups. A future claim that context “helps” should measure agreement with those human labels, not NONE rate, unanimity, or confidence alone.

The wave supplies credible evidence that physical/coincidence context can interrupt continuity-based over-merging. It does not yet show that the candidate panel reliably expresses or selects the correct edge set. Expressibility is the dominant blocker; rubric/evidence refinement is the next blocker; blessing should come only after both are tested against human truth.
