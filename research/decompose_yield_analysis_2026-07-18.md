# --decompose yield analysis (2026-07-18 bulk v7 wave)

**Question (Brad):** was there a bug in `--decompose`, or was the low output
just undesirable?

**Answer: no bug.** `--decompose` did exactly what it's designed to do. The low
label yield is a structural property of how decomposed groups export, plus a
cost-modeling miss in the wave plan. Both are worth remembering; neither is a
defect.

## What happened

The wave ran `stitch-batch -n 15 --decompose` on 5 datasets. `--decompose`
splits any group over the export backstop into panel-sized sub-problems, each
its own votable pack. Result:

| dataset | selected groups | packs generated |
|---|---|---|
| us_boston_streets | 15 | 15 |
| us_seattle_sidewalks | 15 | 15 |
| co_bogota_bike_network | 15 | 16 |
| au_sydney_roads | 15 | **131** |
| fi_helsinki_roads | 15 | **207** |

Sydney + Helsinki have many monster groups; decompose expanded them into 338 of
the 384 total packs. The plan budgeted `5 × 15 × 3 ≈ 225` invocations (by
*group* count); actual was 384 packs × 3 = **1,152** (by *pack* count). ~5× over.

## Why yield was low (the important part)

A decomposed parent group exports **only if every one of its sub-problems
reaches `auto_accept`**. One dissenting (`human_review`) or `none` sub-problem
sinks the entire parent, which then skips as `subproblem_failed`.

Wave result: **8 of 9 decomposed parents failed to export.** Examples:
`c8b85bbc` passed 48/68 sub-problems and still skipped; `bfbf4f93` 37/68.
Only `50b7da83` (2/2) exported. So the entire Sydney+Helsinki decompose
blow-up — 338 packs, ~1,014 ballots — produced exactly **one** exportable
label.

Total wave mint: **25 labels** (24 whole-group `panel_unanimous_v7` + 1
decomposed), almost all from the non-decomposed whole groups. Failed parents
contribute **zero** resolver-training rows (`extract.py` consumes only exported
labels).

## Is the all-or-nothing recomposition itself wrong?

It's a *defensible conservative choice*: don't mint a large M:N group unless the
panel cleanly agreed on every piece of it. But it makes `--decompose` a poor
fit for **label-minting** waves. Its real value is the **human-review queue** —
it turns an un-votable monster group into tractable sub-problems whose dissents
route to Brad. So decompose is a review-queue tool, not a mint tool.

**Open enhancement (not filed):** a "partial recomposition" mode could mint the
union of the sub-problems that *did* pass, salvaging yield from big groups. This
changes label semantics (a partial monster group is a different assertion than
the whole) and needs a deliberate design decision before implementing. Logged
here so it isn't rediscovered from scratch.

## How future waves should budget

1. Budget by **pack count**, not group count. After `stitch-batch`, run
   `find <batchdir> -mindepth 1 -maxdepth 1 -type d | wc -l` and ×3.
2. Project **exportable** yield with `stitch-export --dry-run` before firing —
   the unanimous/pack counts massively overstate what will mint.
3. Default `--decompose` **off** for monster-heavy datasets (Sydney, Helsinki)
   unless the explicit goal is populating the human-review queue.

See `research/bulk_wave_plan_2026-07.md` and the committed wave
(`b9e0920`).
