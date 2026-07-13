# Kimi K2.6 OpenRouter routing + Gemini 3.5 Flash replacement — 2026-07-12 → 2026-07-13

## 2026-07-13 continuation: Gemini 3.5 Flash via OpenRouter (flex) + Kimi remaining sweep

### Context for resume

User asked: "also considering using gemini 3.5 flash via openrouter? i stopped using it before via agy because i blew through quota so quickly"

Goal: finish the Kimi provider sweep at <90s latency + correct on the veto pack,
then evaluate Gemini 3.5 Flash as the v6 replacement for Kimi. The desired
Gemini seat may use agy first and OpenRouter flex as its paid quota/infra
fallback; OpenRouter billing bypasses agy's personal Google daily cap.

### Isolation harness (critical — do NOT edit live config)

Blessed v5 unchanged: `kimi/openrouter/moonshotai/kimi-k2.6 opencode agent vote 480s` in `opencode.json`.
All probes used `OPENCODE_CONFIG_CONTENT` env injection (verified via `opencode debug config` in `@opencode-ai/sdk/dist/server.js`) + per-invocation `OPENCODE_DB` temp dir to avoid sqlite lock. Harness at `/tmp/parallel_probe.py`, `/tmp/gemini_probe.py`, `/tmp/flex_probe.py`.

### Kimi K2.6 provider sweep — final 90s gate (8bf6c63b veto, settled=A)

| provider | latency | choice | vs A | notes |
|---|---|---|---|---|
| siliconflow fp8 | 90s TIMEOUT | — | — | `{"only":["siliconflow"]}` |
| streamlake fp8 | 90s TIMEOUT | — | — |
| cloudflare | 90s TIMEOUT | — | — |
| chutes | 90s TIMEOUT (90+180) | — | — |
| parasail int4 | 180s TIMEOUT | — | — |
| inceptron int4 | 180s TIMEOUT | — | — |
| atlas-cloud int4 | 90s TIMEOUT | — | — |
| baseten fp4 | 90s TIMEOUT | — | — |
| fireworks | 90s TIMEOUT | — | — |
| deepinfra fp4 | 1.4s error | — | `No allowed providers are available` |
| digitalocean | 50.9s None | — | — | no tool use |
| nebius int4 | 1.5s error | — | `No endpoints found that support tool use` |
| phala | 20.9s None | — | — |
| novita | 178.8s | F | wrong | drops R1->T7 sliver (ref_aln 0.016) as sliver-like |
| modelrun fp4 | 92s | B | wrong | drops e1 R1->T1 conf 0.515 |
| venice int4 | 156s first, then 180s TIMEOUT on repeat | A then TIMEOUT | **correct but unstable + >90s** |
| together | 31s | F | wrong |
| wandb fp4 | 22.6s | F | wrong |
| wandb/together repeat 90s | ~14-22s | F | wrong | p90 fast but quality fails|
| kimi-k2.6 default OR routing | 90s TIMEOUT (was 388s at 480s budget earlier) | — | — |
| kimi-k2.5 venice 90s | 22.3s | F | wrong |
| kimi-k2.7-code venice 90s | 57.6s | F | wrong |
| k2.7-code default/siliconflow/together/wandb | 90s TIMEOUT or F | — | — |
| quant fp8 filter | 90s TIMEOUT | — | — |
| ordered list venice+together+wandb | 90s TIMEOUT | — | — |

**Veto pattern**: fast int4/fp4 Kimi hosts consistently pick F (drops e2 R1->T7 16.2m overlap ref_aln=0.016 as sliver-like) or B (drops e1 R1->T1 conf 0.515). Only Venice once picked A correctly, but flaked on repeat.

**Conclusion on Kimi**: No provider passes <90s + correct. Do NOT promote Kimi routing to production. Keep blessed 480s seat or replace voter.

### Gemini via OpenRouter — breakthrough

#### Models and pricing (from /api/v1/models and /endpoints 2026-07-13)

- `google/gemini-3.5-flash` ctx 1,048,576 pricing standard $0.0000015/$0.000009 = $1.5/M in, $9/M out
- flex tier: `google-ai-studio/flex` and `google-vertex/global/flex` = **$0.75/M in, $4.5/M out (50% off)**
- priority tier: $2.7/M in, $16.2/M out
- 6 Google endpoints for 3.5: vertex global standard/flex/priority + ai-studio standard/flex/priority
- `google/gemini-3-flash-preview` also available, similar pricing
- Invalid: `moonshotai/kimi-latest` = `not a valid model ID`

#### Flex routing syntax

OpenRouter: `provider.only` accepts exact endpoint tags like `google-ai-studio/flex`. Base `google-vertex` matches all regions, but tier suffix requires explicit opt-in. Verified working:
```json
{"only": ["google-ai-studio/flex"]}
```

#### Results on calibration packs (settled: small A, medium A, timeout H)

| test | model | provider | latency | choice | correct |
|---|---|---|---:|---|---|
| g35 default | 3.5-flash | default OR | 14.4s | A | ✓ veto |
| g35 flex ai-studio | 3.5-flash | only flex-ai-studio | 16.4s | A | ✓ |
| g35 std ai-studio | 3.5-flash | only ai-studio | 17.7s | A | ✓ |
| g35 std vertex | 3.5-flash | only vertex global | 18.1s | A | ✓ |
| g35 flex vertex | 3.5-flash | only flex-vertex | 111.5s (111s → None) + 30.6s small | A small / timeout med | ⚠ avoid vertex flex |
| g3 preview default | 3-flash-preview | default | 16.8s | A | ✓ |
| g3 preview flex ai-studio | 3-flash-preview | flex | 8.8s | A | ✓ fastest |
| small acfa90c9 | 3.5 flex ai-studio | 10.6s | A | ✓ |
| small acfa90c9 | 3.5 std | 7-7.8s | A | ✓ |
| timeout 99911d68 | 3.5 flex/default | 73.2s | H | ✓ |

**Gemini crushed the veto where all fast Kimi hosts fail.** 14-18s on 8bf6c63b vs Kimi 156s+ timeout.

#### Real token counts and cost (measured via direct OR API same payload as voter, 2026-07-13)

Base64 image payload = vote accurate.

- Small pack `acfa90c9` (4 PNGs, 5,350 char prompt): **5,917 prompt / 383 completion (329 reasoning)** = 6,300 total → flex **$0.00616**, standard $0.0123
- Medium veto `8bf6c63b` (9 PNGs, 7,276 char prompt): **12,366 prompt / 996 completion (960 reasoning)** → flex **$0.01376**, standard $0.0275
- Text token est ~2,100, image ~600/PNG at our 50KB res, but actual billing is higher (maybe higher-res tokenization or overhead) — trust measured 5.9k/12.3k not estimate.

**Cost math:**
- Tunis sweep 15 groups ≈ $0.10 flex / $0.20 std
- Boston 14 groups ≈ $0.10
- 100-group wave: **$0.61–1.38 flex** vs $1.22–2.75 std
- Vs Kimi venice $0.75/M similar prompt but reasoning heavy + unstable.

#### Why this solves agy quota issue

- `agy` CLI (`@agent/gemini`) bills directly to personal Google AI Studio free-tier quota → user hit daily cap.
- `openrouter/google/gemini-3.5-flash` via OR uses Google Vertex + AI Studio **paid** tier under OR's enterprise quota, billed to `OPENCODE_DB` auth's OpenRouter key (sk-or-v1-...). No daily free cap, scales with OR billing, $OPENROUTER_API_KEY.

#### OpenRouter route recommendation (to test in the v6 candidate)

Configure the paid route of the logical Gemini seat as OpenRouter Gemini 3.5
Flash flex:

```json
{
  "provider": {
    "openrouter": {
      "models": {
        "google/gemini-3.5-flash": {
          "options": {
            "provider": {
              "only": ["google-ai-studio/flex"],
              "allow_fallbacks": false
            }
          }
        }
      }
    }
  }
}
```

Or for balance, ordered list with fallbacks:
```json
{"order": ["google-ai-studio/flex", "google-ai-studio", "google-vertex/global"], "allow_fallbacks": true}
```

Avoid `google-vertex/global/flex` (30.6s small + 111s timeout medium). Prefer `google-ai-studio/flex` — fastest + cheapest + correct.

`allow_fallbacks` is deliberately false in the implemented route: the panel's
own logical Gemini seat performs the agy -> OpenRouter fallback and records the
physical route. Allowing OpenRouter to select an unrecorded endpoint would make
the ballot's route provenance false.

**Next steps pending:**
1. Run a full human-labeled calibration wave with Gemini 3.5 flex and agy on the same packs, alongside the current v5 ballots, to measure route parity and quorum lift.
2. If Gemini passes broader accuracy, bless a four-seat v6 panel replacing Kimi with one route-aware Gemini seat (add provenanced routing-policy and actual-route identity).
3. Keep Kimi in the reproducible v5 panel; do not retain it as a fifth v6 seat unless separate evidence shows that the added correlated cost/latency improves safe accepts.
4. Add token/cost logging to `stitch_runner.py` for real votes.csv cost column (currently not logged).

### 2026-07-13 proposed course: v6 replaces Kimi with one route-aware Gemini seat

The intended composition change is **Kimi -> Gemini**, not an additional fifth
seat and not a return to the v3 panel. The v6 candidate remains four voters:

```text
claude / Opus 4.8
codex / GPT-5.6 Terra
gemini / Gemini 3.5 Flash
muse / Muse Spark 1.1
```

Treat Gemini as one logical voter with a transport policy, rather than seating
`agy` and OpenRouter as two correlated votes:

```text
1. agy CLI / Gemini 3.5 Flash (Medium)
2. on quota or provider/infrastructure failure, OpenRouter
   google/gemini-3.5-flash via google-ai-studio/flex
3. if both routes fail, preserve partial progress and halt; do not silently run
   a three-voter wave
```

`google-ai-studio/flex` is the preferred OpenRouter route: it has the same
lowest published flex price as Vertex flex, while the calibration was much
faster and more reliable. Do not include `google-vertex/global/flex` in the
initial fallback chain. A later standard AI Studio fallback (2x flex price) can
be considered separately if availability data justifies it.

#### Important gate before making agy primary

The overnight result establishes **model quality through the OpenRouter
transport**, where every PNG is force-attached. It does not erase the historical
agy defect: in committed v3-era provenance agy chose the first-listed option A
on 11/12 valid ballots, reported constant 0.95 confidence, and only inspects pack
images if it chooses to follow the file-pointer instruction. Quota was a blocker,
but it was not the only blocker.

Run agy and OpenRouter flex as independently recorded candidate routes on the
same human-labeled calibration packs before selecting the primary route. Require:

- comparable exact edge-set accuracy and no new settled-veto regression;
- no position-anchor or constant-confidence monitor alarm;
- evidence that agy actually consumed the visual pack, not just prompt text;
- acceptable valid-ballot, timeout, and p90 latency rates.

If agy passes, ship agy-first/flex-fallback. If it does not, ship flex-first and
retain agy only as an explicitly non-blessed experiment. The v6 model decision
does not need to wait on an unsafe transport preference.

#### Implementation scope for handoff

1. Add a v6 candidate panel while leaving `default`/`v5` and all historical
   panels reproducible. Add the v6 export-era voter set and labeler constants;
   do not change the blessed default until calibration passes.
2. Introduce a logical `gemini` provider identity. Its canonical model identity
   must be stable across transports, while each ballot also records the actual
   route used (at minimum `agy` or `openrouter/google-ai-studio/flex`). Include
   the ordered route policy in `panel_invocation_sha256` so resume cannot reuse
   ballots from a different policy.
3. Implement fallback at the **single-seat invocation layer**. Empty exit-0
   output (agy's observed quota-cap signal), nonzero quota/rate/auth/network
   errors, missing CLI, and provider-level timeouts should try flex immediately.
   Once agy has a provider-scoped failure, open its circuit for the rest of that
   wave so every later group does not pay for the same doomed probe. Malformed
   model output remains a normal parse retry/abstention unless testing shows it
   is a transport failure.
4. Preserve the existing safety contract: if flex also fails, raise
   `ProviderInvocationError`, flush partial rows, and require `--resume`. Never
   convert a dual-route outage into a silent abstention on every remaining group.
5. Add focused tests for route selection, sticky circuit state, actual-route
   provenance, invocation-signature changes, dual-route failure, partial/resume
   compatibility, the v6 export gate, and unchanged v3/v4/v5 reproduction.
6. Calibration should have two stages: first the discriminating three-pack smoke
   test for both routes; then the same human-labeled multi-dataset set used to
   bless v5 (or a documented stratified equivalent), reporting per-route exact
   accuracy, edge F1, abstains, p50/p90 latency, position bias, confidence bias,
   panel auto-accept lift/regressions, and measured/estimated cost.
7. Only after those gates pass, make v6 the default and update CLI help, module
   docs, export provenance comments, and the research conclusion. Kimi remains
   addressable through `v5` for exact historical reproduction.

Token/cost logging is useful but should not block v6. If OpenCode does not expose
stable usage metadata, record the route and maintain a clearly labeled estimate;
do not fabricate exact per-ballot cost from prompt character counts.

#### Local implementation status — 2026-07-13

Implemented locally without changing the blessed v5 default:

- `--panel v6-candidate`: lean Claude + Codex + Muse trio;
- `--panel v6-agy-calibration`: the lean trio plus an agy-only Gemini seat;
- `--panel v6-flex-calibration`: the lean trio plus an OpenRouter AI Studio flex
  Gemini seat;
- wave-local sticky circuit from agy provider failure to flex;
- `invocation_route` on Gemini ballots, strict export validation, and route
  policy in `panel_invocation_sha256`;
- isolated OpenCode config injection pinning `google-ai-studio/flex` with
  OpenRouter fallbacks disabled, immune to a caller's config override;
- distinct v6 labeler tags and era resolution, while v6 remains nonstandard and
  requires `--allow-nonstandard-panel` until calibration passes;
- historical v3/v4/v5 panels and the v5 default remain unchanged.

Verification: focused routing/export/monitor suite passed (313 tests), Ruff
passed, and the full
unit run reached 3,406 passed / 1 xfailed with four unrelated environment/data
failures (subprocess import path plus pre-existing pandas/schema/backfill fixture
issues). Effective OpenCode config inspection confirmed the exact flex endpoint
and tool-less vote agent. No live quota-consuming calibration was run as part of
the implementation.

#### Five-pack route smoke — 2026-07-13

Ran the route-isolated v6 calibration panels on the same five Tunis packs:
`49d609f2`, `8bf6c63b`, `99911d68`, `acfa90c9`, and `f971c741`.
These include the known Kimi veto pack and prior settled choices A/H/I. The
stitch evaluator could not map these group packs to rows in the human label
table, so this is a transport/bias smoke, **not** the human-labeled parity wave.

| route | choices | vs prior settled panel choice | A share | latency p50 / p90 | transport errors |
|---|---|---:|---:|---:|---:|
| agy / AI Studio | A, A, A, A, A | 3/5 | 100% | 24.01s / 140.87s | 0 |
| OpenRouter AI Studio flex | A, A, A, A, C | 3/5 | 80% | 17.52s / 42.55s | 0 |

Both routes produced valid ballots with the expected `invocation_route` and
Gemini passed the known `8bf6c63b` veto on both. However, agy reproduced the
position-anchor defect exactly (5/5 A, with four 1.00 confidences and one 0.95)
and missed settled H/I. Flex also missed H/I in this fresh sample (the H result
regressed from the earlier one-pack probe), so neither route cleared the quality
gate. The other voters kept both groups in human review. Do not run or promote
`v6-candidate` on this evidence; proceed to a genuinely human-labeled,
position-diverse calibration set first.

OpenCode 1.17.15 effective-config inspection also established that
`OPENCODE_CONFIG_CONTENT` is a highest-precedence merge, not a literal full
replacement. The injected Gemini model/provider and tool-denial keys win over
project config, so the flex endpoint remains pinned with fallbacks disabled;
the implementation comment was corrected to match the observed semantics.

#### Gemini 3.1 Pro + lean three-seat replay — 2026-07-13

Repeated the same five-pack smoke with `google/gemini-3.1-pro-preview`.
OpenRouter endpoint metadata confirmed `google-ai-studio/flex` support. AGY
does not expose Pro Medium, so the route comparison is AGY **High** versus flex
at the model's default **Medium** effort.

| route | choices | vs prior settled panel choice | A share | latency p50 / p90 | transport errors |
|---|---|---:|---:|---:|---:|
| agy / Pro High | A, A, G, A, A | 3/5 | 80% | 93.67s / 121.62s | 0 |
| OpenRouter flex / Pro Medium | A, A, A, A, I | 4/5 | 80% | 11.86s / 28.60s | 0 |

Pro flex recovered the difficult settled-I pack and was substantially faster
than AGY High, but still missed settled H. Pro AGY broke Flash AGY's 5/5-A
signature, showing the transport does not mechanically force A, but did not
improve exact accuracy.

Also replayed every recorded smoke wave after removing Gemini, leaving Claude
+ Codex + Muse. Across all four independent waves (Flash AGY, Flash flex, Pro
AGY, Pro flex), the lean panel chose the prior settled answer on all 5/5 packs
and produced the same routing yield every time: 2 auto-accept, 3 human-review.
Gemini therefore added no auto-accept lift in this smoke and sometimes converted
correct three-seat unanimity into four-seat dissent. This is preliminary and
still not human-table-mapped evidence, but it supports evaluating a cheaper
three-seat Claude/Codex/Muse candidate instead of forcing a Kimi replacement.

Decision: v6-candidate is the lean Claude/Codex/Muse trio. Keep the route-aware
Gemini implementation and both four-seat Gemini calibration panels as an
experimental harness for future models, but do not include Gemini in the v6
export voter set. The blessed default remains v5 until the lean candidate's
broader gate passes.

The retained route harness deliberately pins canonical OpenRouter model IDs to
their exact agy aliases. A future Gemini model should be added as an explicit
mapping and calibration panel before use; do not accept a free-form model
override that could record one canonical model while agy invokes another.

---
# Original investigation below — 2026-07-12


## Outcome

Do **not** change the blessed v5 Kimi routing yet. Keep the existing seat:

```text
provider: kimi
model: openrouter/moonshotai/kimi-k2.6
opencode agent: vote
timeout: 480 seconds
```

Soft performance routing substantially improved latency on three calibration
packs, but caused a repeatable quality regression on a coverage-sensitive group.
Pinning the official Moonshot and Baidu endpoints avoided anonymous routing but
timed out on the discriminating medium pack. No tested policy passed both the
quality and latency gates, so the experimental production configuration was
withdrawn rather than published.

## Why this was investigated

The first 15-group Tunis v5 wave showed that the OpenCode transport itself was
healthy, but the Kimi seat had a severe long tail:

| voter | median latency | mean latency | max recorded latency |
|---|---:|---:|---:|
| codex / Terra | 9.78s | 13.38s | 33.17s |
| claude / Opus | 28.56s | 35.15s | 74.50s |
| muse / Meta | 29.12s | 39.61s | 103.99s |
| kimi / OpenRouter | 153.32s | 187.25s | 409.12s |

Kimi also produced two empty exit-0 first attempts and one 480-second timeout.
Muse uses the same OpenCode executable and tool-less `vote` agent but calls Meta
directly, so the evidence pointed to Kimi/OpenRouter inference or routing rather
than a general OpenCode configuration failure.

## Configuration validation

The proposed model-scoped OpenCode configuration used:

```json
{
  "provider": {
    "openrouter": {
      "models": {
        "moonshotai/kimi-k2.6": {
          "options": {
            "provider": {
              "preferred_max_latency": {"p90": 3},
              "preferred_min_throughput": {"p90": 50},
              "allow_fallbacks": true
            }
          }
        }
      }
    }
  }
}
```

This shape was not merely schema-accepted. Source tracing confirmed that
OpenCode 1.17.15 merges model options into `providerOptions.openrouter`, and
`@openrouter/ai-sdk-provider` 2.9.0 passes the OpenRouter options into the JSON
request body. The snake-case keys match the OpenRouter wire API. Scoping the
policy to the exact Kimi model leaves the residual OpenRouter/Qwen voter
unchanged and preserves the blessed base model string used by export provenance.

Focused tests passed throughout (272 tests after the routing regression test was
added). Those tests validated configuration, scoping, identity, consensus, and
export behavior, but a live calibration was still required because unit tests
cannot establish endpoint-level inference equivalence.

## Live calibration packs

Three existing open-data Tunis evidence packs were reused:

| group | role | settled Kimi/panel choice | relevant prior behavior |
|---|---|---|---|
| `acfa90c9` | small healthy N:1 | `A` | normal completion |
| `8bf6c63b` | medium coverage-sensitive M:N | unanimous `A` | all seven edges are true; dropping either end edge is wrong |
| `99911d68` | prior timeout case | panel majority `H` | Kimi timed out at 480s |

The discriminating group matters because option `F` drops one true edge from the
settled seven-edge option `A`. A Kimi `F` vote cannot silently mint a label under
the v5 unanimity rule, but systematic underselection would increase human-review
load and reduce the value of Kimi as a voter.

## Results

### Soft p90 performance preferences

| group | latency | choice | exact vs settled |
|---|---:|---|---|
| `acfa90c9` | 13.15s | `A` | yes |
| `8bf6c63b` | 14.53s | `F` | **no** |
| `8bf6c63b` repeat | 70.30s | `F` | **no** |
| `99911d68` | 51.14s | `H` | yes |

The repeat makes the quality drift difficult to dismiss as ordinary stochastic
noise. The routing policy solved much of the observed latency but failed the
quality gate.

### Fixed low reasoning effort

Low reasoning was tested before provider pinning:

- small pack: 41.79s at low effort versus 27.42s at default; both chose `A`;
- prior-timeout pack: low effort still timed out at 180s.

Reasoning effort was therefore not retained. OpenRouter support and mapping can
vary by endpoint, and a hard output/reasoning token cap risks consuming the
budget before the required JSON answer, producing truncation or empty output.

### Official Moonshot endpoint only

| group | latency | choice | result |
|---|---:|---|---|
| `acfa90c9` | 33.59s | `A` | correct |
| `8bf6c63b` | 240s | — | timeout |

The third call was cancelled to conserve quota. The model creator's endpoint is
not currently suitable as the only production host under a 240-second
calibration cap.

### WandB endpoint only

| group | latency | choice | exact vs settled |
|---|---:|---|---|
| `acfa90c9` | 14.16s | `A` | yes |
| `8bf6c63b` | 22.66s | `F` | **no** |
| `99911d68` | 53.61s | `H` | yes |

WandB was fast and reliable but reproduced the systematic underselection.

### Together endpoint only

| group | latency | choice | exact vs settled |
|---|---:|---|---|
| `acfa90c9` | 6.91s | `A` | yes |
| `8bf6c63b` | 30.97s | `F` | **no** |
| `99911d68` | 48.44s | `H` | yes |

Together likewise failed the quality gate on the discriminating group.

### Baidu endpoint only

`8bf6c63b` timed out at 240 seconds. No further Baidu calls were made.

## Public endpoint evidence

OpenRouter's public endpoint inventory exposed 21 K2.6 hosts with provider tag,
quantization, price, supported parameters, and recent uptime. Its live provider
page additionally reported rolling latency and throughput. At investigation
time, WandB, Decart, and Together were among the fastest published endpoints;
WandB and Together nevertheless failed the discriminating quality case.

This demonstrates why latency/throughput telemetry alone is insufficient for
ground-truth acquisition. Independently hosted copies may differ in
quantization, serving stack, reasoning behavior, or other inference details.

## Adversarial review findings

1. **Quality blocker:** soft speed routing changed a settled unanimous `A` to
   incorrect `F` twice.
2. **Provenance gap:** the base model slug does not distinguish the routing
   policy or the underlying served endpoint. Pre- and post-policy ballots would
   otherwise look identical.
3. **Soft thresholds are not a bound:** they use rolling provider metrics and
   only deprioritize slow endpoints. Slow fallbacks, long reasoning, and stalls
   can still reach the caller timeout.
4. **Endpoint metadata is not reliably exposed by the current answer-only
   OpenCode invocation.** JSON event mode does not provide a stable documented
   contract for the underlying OpenRouter host. Do not add a speculative schema
   field until capture is reliable.
5. **The v5 safety rule contains individual drift:** a dissent routes to human
   review and cannot mint a label, but a systematically weak voter wastes quota
   and increases review load.

## Decision

- Withdraw all experimental `opencode.json` routing changes.
- Do not publish the routing branch as a production behavior change.
- Keep the original blessed v5 Kimi seat and timeout unchanged.
- Pause broader panel sweeps while weekly quota is constrained.
- Preserve PR #427 (Tunis labels and vote provenance) independently; it does not
  depend on any experimental routing policy.

## Next work

1. Build a reusable **provider calibration harness** instead of editing
   `opencode.json` between probes. It should accept `only`/`order` policies,
   replay archived packs, capture latency/choice/edge set/errors, and emit a
   manifest without writing labels.
2. Assemble at least 10–20 settled packs, stratified across small/large,
   M:N/1:N/N:1, reject-all, low-margin, coverage-sensitive, and prior-timeout
   cases. The three-pack probe is a useful veto, not a promotion sample.
3. Test remaining plausible endpoints one at a time on the discriminating pack
   before spending the full calibration budget. Exclude any endpoint that does
   not reproduce settled `A`.
4. For survivors, require acceptable exact-choice and edge-set agreement across
   the full calibration set, plus bounded p90 latency and valid-JSON rate.
5. Prefer an explicit ordered list of multiple quality-validated hosts with
   fallbacks among that list. Do not restore anonymous throughput/latency routing
   unless its paired quality metrics pass.
6. Add routing-policy identity to panel invocation provenance. Add the served
   endpoint only if it can be captured reliably from a supported API contract.
7. Re-run a small four-voter calibration wave before resuming Bogotá, Helsinki,
   Singapore footpaths, and Nairobi voting.
