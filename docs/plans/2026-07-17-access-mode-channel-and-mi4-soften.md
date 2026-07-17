# Access/Mode Evidence Channel + MI-4 Soften — Scoping & Design (2026-07-17)

**Status:** design / pre-implementation. Bundled rubric + feature change.
**Origin:** v8 stitch-wave analysis (`research/v8_wave_analysis_{codex,fable,synthesis}_2026-07-17.md`).
**Analyst input:** Claude Fable seat (scoping proposal + MI-4 wording); orchestrator (Opus) framing + decisions.

## Motivation

The v8 wave folded in a MI-4 cycleway/separated-infra **uncertainty gate** (Fix A,
rubric era `2026-07-17+9463c80a0f77`). Both v8 analysts independently found it
**over-triggers**: it flipped its own non-regression guard `7175635e` (v7 unanimous
merge → v8 majority `insufficient_evidence` NONE) and ≥6 previously-decided groups,
because its trigger is *"the pack lacks the attributes"* and the pack essentially
**always** lacks physical-separation evidence — so the lead clause fires on every
cycleway-vs-road pair and swamps its own coverage-partition exemption.

Two root causes, addressed together here:

1. **The gate's polarity is backwards** — absence of evidence should not force
   uncertainty. → **MI-4 soften** (invert polarity; positive separation signal
   required).
2. **A genuinely useful, usually-available signal is missing from the pack** —
   mode/access restrictions. The pack today surfaces only `level` (layer) and
   `road_flags` (is_bridge/is_tunnel). Overture already ships `access_restrictions[]`
   (we parse it for *direction*/oneway only; the mode dimension is discarded at
   extraction). → **Access/mode channel.**

### The Geneva category error (verified in pack data)

`ch_grand_geneva_cycle_schema` targets are class=`cycleway` but are **cycle
route/itinerary overlays, not physical infrastructure**: `a451bf05` targets named
"Chancy-Bellegarde" (a signed itinerary) along Overture `secondary` "Route de
Bellegarde"/"Route de la Douane"; `b33a27f5` along "Route du Mandement" (`tertiary`).
`target_physical` is `{}`; Geneva `target_physical_capabilities` is
`{has_level: false, flag_domains: []}` — zero physical attributes. Applying the
painted-vs-separated frame here is a category error; the correct disposition is
**route-follows-road (merge)**. This class of case is what both fixes must resolve.

> **Key semantic invariant:** access/mode restrictions resolve **role/mode** (who
> may use the way), **not** physical same-pavement-vs-separated (geometry). Mode is
> the more decidable and more common axis: a cycle route following a road inherits
> `motor_vehicle=allowed` ⇒ it is the road; a real dedicated track is
> `motor_vehicle=denied`. Access is used to decide which representation a coincident
> facility belongs to — never as standalone proof of physical separation.

## Part A — MI-4 soften (canonical rubric change)

Replace the appended MI-4 sentence (added by Fix A `3f9b488`) with the integrated
wording below. This is **one MI-4 addition, no new rule ID**, and must be applied
**byte-identically** to both `docs/MATCHING_MERGING_RULES.md` and the code mirror
`src/crosswalk/agent_labeling/matching_rubric.py` (CI enforces parity), with a
`MATCHING_RUBRIC_VERSION` bump (content-addressed → new era).

Proposed content (to be **restructured into 3–4 sentences before shipping** —
lead rule / affirming signals / mode≠geometry caveat / NONE-last-resort list —
with zero content change; the canonical doc is diff-checked and a 230-word single
sentence is a readability/parse risk):

> When the identity decision between such a bike or pedestrian facility and a
> roadway candidate hinges on whether it is same-pavement or physically separated,
> apply the structural and access evidence first: if naming continuity, endpoint
> topology, or a clean coverage partition (aligned fractions tiling either side to
> ~1.0) supports the match, or the access evidence affirms shared use — motor
> vehicles allowed on the aligned roadway while the coincident facility is a cycle
> lane, signed route, or itinerary overlay rather than separately-built
> infrastructure — accept it: a route or lane designation that follows a road *is*
> the road, and the mere absence of separation attributes, offset metrics, or
> imagery does not defeat this; shared-pavement facilities are the common case, so
> with no affirmative sign of separation the same-pavement reading stands. Bear in
> mind that access restrictions describe mode and role (who may use the way), not
> geometry — use them to decide which representation a coincident facility belongs
> to, never as standalone proof of physical separation. Treat identity as
> unresolved (unsure per PL-3 at the pair level, or insufficient-evidence NONE per
> SA-5 at the stitch level) only when there is a positive indication that the
> facility may be physically separate — a visible lateral offset from the roadway
> centerline, a conflicting layer/bridge/tunnel stack over the aligned span, tagged
> (not class-default) access marking the facility as a distinct-mode way while the
> roadway carries motor traffic, a competing role-compatible representation of the
> same corridor such as a separately mapped path or footway that also matches, or a
> segment too short or fragmentary for the structural tests to apply — and that
> indication is not itself resolved by the rest of the evidence, including the
> access channel.

Polarity, restated: **structural + access evidence lead; unresolved status requires
a POSITIVE separation signal**, and a *tagged* (not class-default) access denial is
one such signal. Absence of separation attributes is explicitly non-triggering.

## Part B — Access/mode evidence channel (feature)

### B.1 Data model

Add `access_lr` alongside `level_lr` / `road_flags_lr` in `ref_physical` /
`target_physical`, LR-scoped (`between: [a, b]`). Each span value = per-mode map
over exactly three modes `{motor_vehicle, bicycle, foot}`:

```
{ value: allowed | designated | denied | restricted | unknown,
  source: tagged | class_default }
```

- `restricted` (distinct from `denied`) for permit/private/time-scoped entries.
- `unknown` spans are **omitted from rendering, never guessed**.
- Provenance tiers: **tagged** (explicit data) > **class_default** (entailed by the
  class definition) > absent (= unknown). Class defaults **never override tags**.

### B.2 Overture mapping (extraction)

Extend the `access_restrictions[]` parse next to `parse_oneway_lr` in
`fetch/overture.py`:

- Entries with `when.heading` → stay in the direction/oneway channel (unchanged).
- Remaining entries → map `access_type × when.mode[]` to per-mode values
  (`allowed`/`denied` direct; `designated` → designated; no `when.mode` = blanket →
  all modes; `when.during` / recognized-user scoping → `restricted`), LR-scoped.
- Then one **class-default pass** fills unset modes from a small fixed table:
  - driveable classes (motorway … residential/service) ⇒ `motor_vehicle=allowed`
  - `motorway` ⇒ `bicycle=denied`, `foot=denied` (legally entailed — the one safe
    inferred denial)
  - `cycleway` ⇒ `bicycle=designated`
  - `footway` / `sidewalk` ⇒ `foot=designated`

Carry through `fetch/physical_tags.py` (trivial LR for target side where absent) →
per-edge attribute struct → prompt, same plumbing as `road_flags`.

### B.3 Implied-scoping policy (the brittleness guard)

**Never emit an inferred denial except the motorway entailment.** `motor_vehicle`
is only ever inferred as `allowed` on driveable classes — never inferred `denied`
from `pedestrian`/`cycleway`/`footway` class (the European permit-traffic
counterexample; and unnecessary, since the decision-relevant merge signal is
`motor_vehicle=allowed` on the *road* side, reliably class-entailed, and the
decision-relevant *separation* signal must be **tagged** to count per MI-4). Every
inference is thus tautological or legally entailed; everything brittle renders
`unknown`.

### B.4 Prompt format

Extend the SEGMENTS line; `°` marks class_default, bare = tagged, `?` = unknown:

```
R3: name='Route de Bellegarde' class='secondary' physical='layer 0' access='mv:yes° bike:? foot:?'
```

Edge lines gain an access clause only when a tagged restriction varies along the
aligned span (rare), rendered like `road_flags_lr`. Add one legend line under
SEGMENTS:

```
access: °=class-default (implied by road class; never overrides tagged data), ?=unknown
        — access describes who may use the way (mode/role), not physical separation.
```

### B.5 Geneva / route-network datasets

Access alone resolves ~90% (ref `mv:yes°` vs target `bike:designated°` +
`physical=unknown` → merge under new MI-4). For the rest, add a deterministic
**per-dataset config flag** `target_kind: route_network` next to
`target_physical_capabilities` (Geneva qualifies on inspection: itinerary-style
names, `{}` physical, empty flag_domains) — **not** runtime name-parsing. Surface as
one prompt header line:

```
NOTE: targets in this dataset are signed route/itinerary designations, not
separately-built infrastructure; a route that follows a road matches the road.
```

This makes `a451bf05`'s competing footway ref (R10) a non-issue (a route overlay can
legitimately coincide with both road and path; the roadway alignment wins where the
itinerary follows the road). **Action item:** audit the other target datasets and
set `target_kind` where warranted.

## Touch-points (implementation order)

0. **Backfill determination — RESOLVED 2026-07-17: re-extract, no re-fetch.**
   The saved reference parquets retain the raw `access_restrictions` column
   (verified in `fi_helsinki_roads` 123k/374k nonnull and
   `ch_geneva_pedestrian_network` 53k/110k nonnull). Backfilling existing datasets
   is a re-**extract** (re-run `extract_lr_attributes` + physical_tags), no S3
   re-**fetch**. The raw structure confirms the B.2 mapping directly — observed
   values include `{access_type: denied, when.mode: [motor_vehicle]}` +
   `{access_type: allowed, when.mode: [foot]}` (a footway) and
   `{access_type: allowed, when.recognized: [as_private]}` (→ `restricted`).
   Note `when.mode` can be null with a `recognized`/`during` scope (blanket
   permit/private) — map to `restricted` across modes, not `allowed`.
1. `fetch/overture.py` — `parse_access_lr` + class-default table + wire into
   `extract_lr_attributes`.
2. `fetch/physical_tags.py` — trivial/target-side `access_lr`.
3. Group/evidence builder (`stitch_evidence.py` and the batch group struct) —
   carry `access_lr` into `ref_physical`/`target_physical`; render the prompt line +
   legend; render the `target_kind: route_network` header.
4. `matching_rubric.py` + `docs/MATCHING_MERGING_RULES.md` — MI-4 soften (Part A),
   byte-identical, `MATCHING_RUBRIC_VERSION` bump.
5. Per-dataset config — `target_kind` field + Geneva set; audit others.
6. Version bumps: `MATCHING_RUBRIC_VERSION` (rubric); `FEATURE_VERSION` if
   `access_lr` becomes an ML feature (else evidence-pack-only). Consider whether the
   class-default table belongs in `config.py`.

## Validation plan

Pre-register as a fix hypothesis (the `codex_deep_v1` pattern) with the **blind
per-group checklist** below, validate on the sealed holdout, **then** run a
**targeted** rerun of only the ≥6 regressed cycleway groups + guard — not the full
65 (quota).

| group | predicted outcome | rationale |
|-------|-------------------|-----------|
| `7175635e` (guard) | **merge** | coverage partition ≈1.005 |
| `a451bf05` (Geneva) | **merge** | route-overlay + mv-allowed; R10 non-issue |
| `b33a27f5` (Geneva) | **merge** | route-overlay + mv-allowed |
| `66e22055` | **NONE** (preserved) | observed lateral offset + vertical stack + 15m fragment |
| `5faa0b72` | **NONE** (preserved) | competing same-centerline reps + underpass conflict |
| `92c0997f` | **NONE** (preserved) | vertical tunnel/surface conflict (+ partial expressibility) |

Net intent: recover the decidable-by-convention merges (incl. the guard and the
Geneva route overlays), keep the genuine-ambiguity NONEs as the last resort.
