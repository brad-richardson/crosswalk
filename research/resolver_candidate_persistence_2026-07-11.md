# Resolver candidate persistence — 2026-07-11

## Outcome

The stitch pipeline now writes a typed `<dataset>_candidates.parquet` beside the
bridge and groups sidecar when `stitch_persist_candidates=True` (default). Factory
runs normalize the same artifact to `dataset=<name>/candidates.parquet` and treat a
missing artifact as stale when the manifest says groups were produced.

This is additive. It does not change scoring, optimization, pruning, the bridge,
or the groups JSON. The parquet contains exactly one row for every edge in the
uncapped resolver `candidate_edges` universe, attributed to exactly one group.
Pure 1:1 components remain out of scope because the group resolver never reads
them.

## Schema and provenance

Each row is keyed by `(dataset_id, group_id, ref_id, target_id)` and includes:

- positional `ref_idx` / `target_idx` and calibrated confidence;
- all 83 current `FEATURE_COLUMNS` as native floating-point columns with NaNs
  preserved;
- alignment fractions and the full candidate-graph structural layer;
- group type/size/corridor/oversize context;
- optimizer status (`selected`, `pruned`, `selected_elsewhere`, decision and
  reason), so training can distinguish rejection from confidence pruning;
- reference/target class and metric length;
- `lateral_offset_signed_m`, computed on the aligned portions, with positive meaning left of the stored reference
  direction and negative meaning right; corridor consumers must normalize
  reference direction before comparing sign consistency;
- `feature_version`, SHA-256 `model_hash`, and sidecar `schema_version`.

The candidate parquet deliberately omits geometry. `groups.json` remains the
geometry/display artifact; the parquet is the typed training and future scoring
substrate.

## Validation contract

- Parquet keys must equal the full JSON `candidate_edges` keys.
- Every declared runtime feature column must exist and have floating dtype.
- Candidate persistence must work even if JSON candidate-graph persistence is
  disabled; the two additive outputs have independent flags.
- Toggling persistence must not change existing groups JSON keys or matcher
  decisions.
- Factory staleness keys include both candidate persistence flags.

## Remaining resolver work, in order

1. Join the parquet in the resolver training-table builder and assert runtime /
   training feature parity on a committed small fixture.
2. Regenerate fresh uncapped sidecars for Bogotá roads, Helsinki, Nairobi, and
   Singapore footpaths; generate Tunis rather than using a missing/legacy view.
3. Investigate the 20 Boston human-positive edges outside the persisted candidate
   graph before tuning any learned selector or confidence threshold.
4. Grow to 200–300 clean, balanced groups. Prioritize M:N interchanges,
   cross-class/reject-all cases, low margins, bridges, and optimizer/panel
   disagreements; retain split labels as a sensitivity slice.
5. Capture at least 20 current candidate-backed reject-all groups and settle
   panel disagreements with human review rather than treating votes as held-out
   ground truth.
6. Re-run multi-seed/bootstrap and leave-one-dataset-out evaluation. Report both
   historical-human exact match and current-sidecar exact match explicitly.
7. Keep the learned resolver and proposed per-type heuristic defaults in NO-GO
   state until those data/coverage gates are met.
