# Seattle stitching-label rekey — 2026-07-11/12

## Outcome

Seattle resolver coverage is restored from zero to **22 mapped pair-label groups /
99 candidate rows**. The root cause was a mixed identifier vintage: stitching
labels had already moved from volatile ArcGIS `OBJECTID` values to stable SDOT
`COMPKEY` values, while the local raw target and groups sidecar were still
OBJECTID-keyed.

The local target was rekeyed with the audited `scripts/rekey_seattle_target.py`
flow. PR #415 initially applied 18 collision-free 1:1 group-ID moves, but that
regeneration used a stale local Overture `2026-01-21.0` file even though the
dataset YAML recorded the later release.

The reference was therefore force-refetched and verified as Overture
`2026-06-17.0` (169,457 road segments), the current release as of July 11, and
the stitch output was regenerated again. Sixteen of #415's mappings remained
verbatim current. The June sidecar produced one safe correction
(`29558394` -> `f3edbcad`) and recovered the previously lost pair
(`03c78ff0` -> `02a96d29`). One #415 label (`60b7872d`) now spans group
boundaries and remains review-only rather than being rewritten speculatively.

Across both passes, `selected_edges`, membership sets, label provenance, and
human decisions are unchanged. The append-only `rekey_log.csv` records every
old/new group ID.

## Reproduction

```bash
uv run python scripts/rekey_seattle_target.py --apply
uv run crosswalk data fetch reference us_seattle_sidewalks \
  --source overture --force
uv run crosswalk stitch us_seattle_sidewalks -w 1
uv run crosswalk data stitch-rekey us_seattle_sidewalks \
  --sidecar data/output/us_seattle_sidewalks_groups.json --apply
uv run python scripts/ablate_optimizer_heuristics.py \
  --data-root . --dataset us_seattle_sidewalks \
  --output /tmp/seattle-rekey-eval
```

The raw target, bridge, and groups sidecar are generated/ignored data. The
committed result is the label rekey plus its audit log.

## Coverage after rekey

| item | result |
|---|---:|
| stitching label rows | 49 |
| #415 safe 1:1 rekeys | 18 |
| June-reference safe follow-up rekeys | 2 |
| mapped pair-label groups | 22 |
| candidate edge rows | 99 |
| represented positives / negatives | 57 / 42 |
| human positives outside candidate graph | 6 |
| production edge F1 | 0.8346 |
| production historical-group exact | 0.6818 |

The 22 mapped evaluation groups include four split labels that recovery can
still evaluate as a sensitivity slice. Seattle's validated
0.90 confidence prune remains the best tested production policy here. Scalar
0.92–0.98 and the handed-off per-type thresholds all regress F1 and exact match;
the 0.05 margin loses 0.0071 F1 and adds no exact-match wins.

## Deliberately not auto-rekeyed

Four pair labels span current group boundaries and require stitching review:

- `c5bf38f6`: 3/16 selected edges in the largest current group `9ea4edd2`;
- `24bad503`: 1/2 in `668724f8`;
- `60b7872d`: 10/15 in `edd31072`;
- `670e939f`: 4/5 in `534c466a`.

There are 27 SET-semantics labels. Eighteen have all members contained in exactly
one current group, but the rekey command intentionally treats membership-based
mapping as review-only. Five more have full recovery but span multiple groups;
four have missing members. None are blind-rekey candidates.

- full but split: `3fdf30d0`, `b5a16db8`, `d7c194e2`, `bf6d0adc`,
  `d35b4619`;
- missing members: `e99d63e3` (9/15), `9465a639` (2/3), `d6c70928`
  (4/10), `85b903ba` (1/9).

## Next actions

1. Review the four split pair labels and nine ambiguous SET labels in the
   stitching UI.
2. Decide explicitly whether the 18 exact single-group SET mappings should gain
   a guarded rekey path; do not rewrite them ad hoc.
3. Investigate the six human-positive edges outside Seattle's current candidate
   graph before using Seattle to tune a learned selector.
4. Keep Seattle as a separate dataset slice in multi-seed and
   leave-one-dataset-out resolver evaluation; its optimal prune floor differs
   materially from Boston's.
