# V6 breadth wave — 2026-07-14

## Scope

The first production-style breadth pass for the lean v6 panel used the
Claude/Codex/Muse composition from `PANEL_V6_CANDIDATE`:

- `claude-opus-4-8`, medium effort
- `gpt-5.6-terra`, medium effort
- `meta/muse-spark-1.1`, high reasoning effort

The wave sampled 26 fresh groups across 13 datasets rebuilt against Overture
`2026-06-17.0`: two groups per dataset, with 13 large, 7 borderline, and 6
low-confidence groups. The run intentionally excluded groups over the 40-edge
export backstop and groups already covered by human labels.

Ignored evidence packs and raw outputs are under
`data/agents/stitching/batches/breadth_v6_20260714_*` on the Mac and factory
machine. Every selected group has an evidence-pack hash, displayed-candidate
universe hash, option-menu hash, panel-invocation hash, and consensus-policy
hash in its vote/consensus rows.

## Results

All 78 requested ballots completed: 26 each from Claude, Codex, and Muse. There
were no provider errors, timeouts, parse failures, or abstentions.

| Outcome | Groups |
|---|---:|
| Unanimous | 17 |
| 2–1 majority | 7 |
| No majority | 2 |
| Auto-accept | 8 |
| Human review | 18 |

Routing reasons for the 18 review groups were seven dissenting majorities, two
three-way/no-majority results, two unanimous `NONE` results, six confidence
gates, and one class-mismatch gate. The latter nine show why panel agreement
and export eligibility must remain separate.

| Stratum | Groups | Unanimous | Majority | None | Auto-accept | Review |
|---|---:|---:|---:|---:|---:|---:|
| Large | 13 | 8 | 4 | 1 | 4 | 9 |
| Borderline | 7 | 4 | 3 | 0 | 3 | 4 |
| Low-confidence | 6 | 5 | 0 | 1 | 1 | 5 |

Among the seven majority decisions, Muse was the sole dissenter four times,
Claude twice, and Codex once. This is evidence that Muse remains decorrelated;
without human labels it is not evidence about which voter was correct. Mean
latencies were 42.1 s for Claude, 15.3 s for Codex, and 52.7 s for Muse. Mean
self-reported confidences were 0.696, 0.935, and 0.837 respectively.

## Harness finding

The initial factory smoke exposed two repo-local Meta configuration problems
that had been masked by a corrected global OpenCode config:

1. `@ai-sdk/openai-compatible` does not expose the Responses API method needed
   by Muse reasoning and failed with `responses is not a function`.
2. The factory process needed `META_API_KEY` in the repo's ignored `.env` so the
   explicit `{env:META_API_KEY}` provider option did not override stored auth
   with an empty value.

`opencode.json` now uses `@ai-sdk/openai`, declares Muse's actual multimodal
limits, and requests high reasoning effort. A two-group live smoke then
completed all six ballots before the remaining wave was launched.

## Interpretation and next step

This wave validates harness reliability and produces useful cross-dataset
acquisition evidence, but it is not an accuracy evaluation: these groups do not
yet have human labels. The highest-value next action is to adjudicate the 18
review-routed groups and spot-check the eight auto-accept groups. After that,
run a deeper panel pass targeted by the observed disagreements—especially large
groups and Muse sole dissents—rather than immediately drawing accuracy or bias
conclusions from unlabeled votes.
