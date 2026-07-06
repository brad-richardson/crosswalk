# Panel vote provenance

Committed audit trail of the LLM consensus panel's **raw ballots** and
per-group **consensus** — the evidence behind every exported
`panel_unanimous_*` label in `labels/stitching/`.

## Why this exists

The panel (`stitch_runner.run_batch`) writes `votes.csv` (every raw ballot) and
`consensus.csv` per batch **under the batch dir**, which lives in the
git-ignored `data/` tree. That means the provenance was never committed: the
resolved labels survived in `labels/stitching/`, but *who voted what* did not.

`stitch_export.write_vote_provenance()` snapshots those two files into this
tracked tree whenever a dataset is exported (`crosswalk agent stitch-export`),
keyed by dataset and tagged with a `source_batch` column:

```
labels/votes/dataset=<name>/votes.csv       # raw per-voter ballots
labels/votes/dataset=<name>/consensus.csv   # per-group consensus rows
```

Writes are idempotent (deduped by `source_batch`/`group_id`/`provider`), so
re-exporting the same batches is a no-op.

## Scope / limitations

- Provenance is archived **at export time**, so it covers every batch that
  produced a committed label. Batches whose groups were never exported (e.g.
  all routed to human review) are not captured here.
- Historical vote data from panel runs **before** this mechanism landed lives
  only on the machine that ran them; back-fill by re-running
  `stitch-export` against those batch dirs if they can be recovered.
