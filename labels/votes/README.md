# Panel vote provenance

Committed audit trail of the LLM consensus panel's **raw ballots** and
per-group **consensus** — the evidence behind every exported
`panel_unanimous_*` label in `labels/stitching/`.

## Why this exists

The panel (`stitch_runner.run_batch`) writes `votes.csv` (every raw ballot) and
`consensus.csv` per batch **under the batch dir**, which lives in the
git-ignored `data/` tree. That means the provenance was never committed: the
resolved labels survived in `labels/stitching/`, but *who voted what* did not.

`stitch_export.write_vote_provenance()` snapshots those files plus the compact
per-group evidence manifest into this
tracked tree whenever a dataset is exported (`crosswalk agent stitch-export`),
keyed by dataset and tagged with a `source_batch` column:

```
labels/votes/dataset=<name>/votes.csv       # raw per-voter ballots
labels/votes/dataset=<name>/consensus.csv   # per-group consensus rows
labels/votes/dataset=<name>/evidence.csv    # exact offered menus + pack hashes
```

Writes are idempotent (deduped by `source_batch`/`group_id`/`provider`), so
re-exporting the same batches is a no-op.

## Scope / limitations

- Provenance is archived **at export time**, so it covers every batch that
  produced a committed label. Batches whose groups were never exported (e.g.
  all routed to human review) are not captured here.
- Historical vote data from panel runs **before** this mechanism landed lives
  only on the machine that ran them. Recovered legacy batches can be archived
  for audit with `write_vote_provenance(..., require_evidence=False)`, but the
  strict `stitch-export` CLI will not mint a new label when it cannot prove the
  ballots were cast against the recovered menu.
- New ballots and consensus rows carry an `evidence_id`, exact pack/menu hashes,
  a panel invocation signature, and a consensus-policy signature. `evidence.csv`
  stores the corresponding menu (including `NONE`), every displayed edge, and
  the full upstream source-candidate list, so an unselected or pruned edge can
  be interpreted without recovering the ignored batch.
- Legacy packs can be archived with `source_artifacts.status` explicitly marked
  unavailable. That preserves what was displayed without claiming a model or
  sidecar identity that the old pack never recorded.
