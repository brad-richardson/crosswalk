# v8 Stitch Wave — Orchestrator Synthesis (2026-07-17)

Synthesizes the two independent analyst reports:
- `research/v8_wave_analysis_codex_2026-07-17.md` (Codex `gpt-5.6-sol`, xhigh)
- `research/v8_wave_analysis_fable_2026-07-17.md` (Claude Fable)

Both analyzed the archived v8 ballots (`labels/votes/`, committed in `122671c`) +
the batch working dirs, independently, without reading each other. Orchestrator
(Opus 4.8) reconciliation + one routing spot-check below.

## Bottom line

- **Bless v8 as a composition/rubric era? NO.** Two-analyst consensus. The wave
  failed its own explicit non-regression guard (`7175635e`) and the MI-4
  cycleway gate over-triggers systemically.
- **Bless v8's *mechanisms*? YES.** The #451 none_reason/desired_edges contract,
  #450 exact-pair seeds, and #446 retries all performed cleanly. Keep them; run
  future production waves **enriched-only**.
- **Mint the deferred v7+v8 auto-accepts? Curated YES — human-gated, by
  cross-wave-unanimous choice, NOT by the v8 `auto_accept` flag** (the flag is
  demonstrably miscalibrated this wave — see the divergence section).

## Where the two analysts independently agree (high-confidence findings)

| # | Finding | Codex | Fable |
|---|---------|-------|-------|
| 1 | **Fix-A guard `7175635e` FAILED** — v7 unanimous A → v8 majority NONE (claude+muse `insufficient_evidence`, only codex held A). Same evidence pack both waves ⇒ rubric-induced. | ✓ | ✓ |
| 2 | **MI-4 gate over-triggers broadly** — a surge of `insufficient_evidence` NONEs (~3→37) flipping ≥6 previously-decided cycleway groups. Shared flip set: `66e22055`, `a451bf05`, `b33a27f5`, `5faa0b72`, `92c0997f`, `7175635e`. | ✓ (cycleway NONE 15/45→31/45, +35.6pt) | ✓ (34/37 ins NONEs use MI-4 language) |
| 3 | **Exact-pair seeds (#450) reduced but did NOT close the gap** — gap-NONE share ~72–78%→**44.7%** of NONEs; but total NONE rate *rose* (36.8%→43.6%). 38 residual `no_exact_option`, 13/38 one edge away (combinatorial, in large groups). | ✓ | ✓ |
| 4 | **#451 contract flawless** — 100% enum coverage; all 38 `no_exact_option` `desired_edges` mapped via batch.json with **0 mapping failures, 0 degenerate sets, 0 misfired NONEs**. Machine-analyzable channel is trustworthy. | ✓ | ✓ |
| 5 | **2×2: `minimal` is dangerous** — strip both context channels and the panel *confidently accepts wrong optimizer output*. `1b90f03b/minimal` unanimously auto-accepted optimizer A (incl. the false Harrier Ave edge) at 0.893 — the exact set every informed cell + all of v7 rejected. Physical/coincidence are substitutes, and context prevents confident wrong merges. | ✓ | ✓ |
| 6 | **Withhold `1b90f03b/minimal`; spot-check `6775ade1`** before any mint. | ✓ | ✓ |
| 7 | **Per-seat:** claude = MI-4 gate-responder (NONE rate ↑ to 55%, lowest conf); codex = exact-set perfectionist + most frequent dissenter, highest/least-calibrated confidence; muse = accept-pole (under-weights coincidence/role gates). | ✓ | ✓ |

## The one genuine divergence: minting timing

Both converge on **which** labels are safe (the cross-wave-unanimous, non-cycleway
subset). They split on **when**:

- **Codex: hold the whole tranche** until MI-4 is corrected + rerun — "don't
  promote under a wave that failed its explicit non-regression condition." Also
  flags `8f152b92` (v7 auto-accept → v8 `human_review/low_confidence`) as needing
  reconciliation first.
- **Fable: mint the curated 9 now** — they're unanimous across **two** rubric eras
  (v7 *and* v8), and the failed guard is in the cycleway cohort, which is
  **disjoint** from the mint set. Adopt "ablation-variant ballots never mint."

### Orchestrator reconciliation (routing spot-check)

Verified from the committed consensus CSVs:

- `8f152b92`: v7 unanimous B **auto_accept** (0.877) → v8 unanimous B, **human_review/low_confidence** (0.867). *Same verdict both waves*; confidence merely dipped below the bar.
- `1b90f03b`: v8 flag **auto-accepted the wrong cell** (minimal→A, 0.893) and **demoted the correct one** (enriched→J, human_review/low_confidence).

⇒ The v8 `routing=auto_accept` flag is **not** a safe mint selector this wave
(variant-blind, and it inverted correct/incorrect on `1b90f03b`). This *supports
Fable's mechanism* (mint by cross-wave-unanimous **choice**, human-curated) while
*honoring Codex's caution* (don't trust the wave's automatic promotion). The two
positions are closer than they read: neither would mint automatically.

The mint set is **disjoint** from the regressed cycleway cohort, so the guard
failure does not taint it. The remaining judgment — mint the stable subset now vs.
wait for the MI-4-fixed rerun — is **Brad's call** (a provenance decision).

## Recommended disposition

1. **Do not bless the v8 rubric era.** Register MI-4 revision as the blocker:
   either add separation evidence to the pack (attributes / lateral-offset metric
   / imagery) or give the gate a default convention for `physical='unknown'` +
   strong coverage partition. Re-run the ≥6 regressed cycleway groups as a
   targeted non-regression panel.
2. **Bless + keep the mechanisms** (#450/#451/#446); make **auto-accept
   variant-aware (enriched-only)** so a context-blind unanimity like
   `1b90f03b/minimal` can never mint.
3. **Add a consensus-`desired_edges` path** — route cross-seat-identical desired
   sets (e.g. `bdbdf792` all three seats; `00e8e9fd`, `fb8f359f`) to a one-click
   confirm/seed. Attacks the 38 residual expressibility NONEs menu enumeration
   can't reach.
4. **Curated mint (human-gated, by cross-wave-unanimous choice):**
   - **Mint:** `33a36ca5`(A), `3f53c7e7`(A), `7bac1f1d`(I), `91570f54`(A),
     `e0099fb8`(A), `ee358f5a`(B) — unanimous in both v7 and v8, disjoint from the
     regression.
   - **Spot-check then mint:** `6775ade1`(E) — only mint candidate with a
     contradicting prior-wave ballot (v7 no-consensus).
   - **Reconcile then mint:** `8f152b92`(B) — same unanimous verdict both waves,
     but v8 routing dropped to low_confidence; confirm the confidence policy.
   - **Withhold:** `1b90f03b/minimal`(A) — context-blind artifact; the correct
     label for this group is **J** (enriched), currently human_review.
   - **Policy:** ablation-variant ballots are experiment data, never labels.

## Top human-review priorities (both analysts converge)

1. `7175635e` — adjudicate the Fix-A guard; its call drives the whole MI-4 disposition.
2. `1b90f03b` — confirm **J** as the label; formally withhold the minimal-A auto-accept.
3. MI-4 flip set — `66e22055`, `a451bf05`, `b33a27f5`, `5faa0b72`, `92c0997f` (v7 consensus → v8 NONE); human labels double as MI-4-revision calibration.
4. Consensus-desired-edges harvest — `bdbdf792`, `00e8e9fd`, `fb8f359f`.
5. `6775ade1` spot-check; `8582dd97` layer question; the 4 no-consensus knots (`750ae089`, `1de025b8`, `dc818edc`, `7680bb19`).

_No labels minted by this pass — minting is a separate, explicitly-authorized action._
