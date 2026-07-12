# Seattle stitching-label rekey — 2026-07-11

## Outcome

Seattle resolver coverage is restored from zero to **21 mapped pair-label groups /
105 candidate rows**. The root cause was a mixed identifier vintage: stitching
labels had already moved from volatile ArcGIS `OBJECTID` values to stable SDOT
`COMPKEY` values, while the local raw target and groups sidecar were still
OBJECTID-keyed.

The local target was rekeyed with the audited `scripts/rekey_seattle_target.py`
flow and the stitch output was regenerated. This PR then applies only the 18
collision-free 1:1 group-ID moves produced by `crosswalk data stitch-rekey`.
`selected_edges`, membership sets, label provenance, and human decisions are
unchanged. The append-only `rekey_log.csv` records every old/new group ID.

## Reproduction

```bash
uv run python scripts/rekey_seattle_target.py --apply
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
| safe 1:1 rekeys applied | 18 |
| mapped pair-label groups | 21 |
| candidate edge rows | 105 |
| represented positives / negatives | 61 / 44 |
| human positives outside candidate graph | 6 |
| production edge F1 | 0.8769 |
| production historical-group exact | 0.6667 |

The 21 mapped evaluation groups are the 18 clean rekeys plus three split labels
that recovery can still evaluate as a sensitivity slice. Seattle's validated
0.90 confidence prune remains the best tested production policy here. Scalar
0.92–0.98 and the handed-off per-type thresholds all regress F1 and exact match;
the 0.05 margin is nearly neutral on F1 but adds no exact-match wins.

## Deliberately not auto-rekeyed

Three pair labels span current group boundaries and require stitching review:

- `c5bf38f6`: 3/16 selected edges in the largest current group `9ea4edd2`;
- `24bad503`: 1/2 in `668724f8`;
- `670e939f`: 4/5 in `d3bef09c`.

One pair label is genuinely lost and should be re-paneled/re-labeled:
`03c78ff0` (`022ecdd0-…` → `sea_sidewalk_658573_8828d550bd`).

There are 27 SET-semantics labels. Twenty have all members contained in exactly
one current group, but the rekey command intentionally treats membership-based
mapping as review-only. The remaining seven have missing members or span more
than one current group and are clearly not blind-rekey candidates:

- `e99d63e3` (14/15 members), `9465a639` (1/3), `bf6d0adc` (7/8),
  `d35b4619` (9/12);
- `3fdf30d0`, `b5a16db8`, and `d7c194e2` have full member recovery but span
  two current groups.

## Next actions

1. Review the three split pair labels and seven ambiguous SET labels in the
   stitching UI; re-panel the single lost pair.
2. Decide explicitly whether the 20 exact single-group SET mappings should gain
   a guarded rekey path; do not rewrite them ad hoc.
3. Investigate the six human-positive edges outside Seattle's current candidate
   graph before using Seattle to tune a learned selector.
4. Keep Seattle as a separate dataset slice in multi-seed and
   leave-one-dataset-out resolver evaluation; its optimal prune floor differs
   materially from Boston's.
