# Resolver Gap Analysis — why learned ties but not beats prod (2026-07-11)

> **Validation update:** The optimizer values proposed below were hypotheses from
> the initial Boston error analysis, not measured wins. A subsequent fixed-universe
> ablation in draft PR #411 found the proposed per-type prune + 0.05 margin + bridge
> guard regressed pooled F1 from 0.9049 to 0.8969 and clean F1 from 0.9546 to
> 0.9437; on Boston they are identical to the current scalar 0.96 behavior. Glue
> sweeps also changed label retention and were not comparable. Keep production
> defaults until fresh sidecars and more clean M:N ground truth validate a policy.

> **2026-07-12 multi-dataset update:** The prototype now evaluates 949 edges in
> 163 dataset-scoped groups across 12 datasets. It joins persisted typed candidate
> parquets fail-closed; only Seattle has one locally (34,401 rows, all 99 labeled
> edges enriched, zero missing keys). Honest OOF learned F1 is 0.875 versus 0.899
> production, while group exact is 0.748 versus 0.730. This keeps the track useful
> as an ablation harness but confirms NO-GO for production. Four datasets still use
> legacy candidate universes, Tunis is missing, and 20 Boston + 6 Seattle positives
> remain outside the candidate graph. Panel votes default off because historical
> batches do not record the displayed candidate universe; selected edges are usable
> evidence, but unselected edges and NONE votes are not safe negatives.

## Exec summary
- Scope: Boston only, 679 edges / 108 groups (356 clean / 323 split). Prod = keep-all+prune `selected` flag.
- Prod: F1 0.883 P0.822 R0.953 exact 0.731. Best conf oracle t=0.98 F1 0.886 (+0.003). Model OOF eF1 F1 0.878-0.885 (tie ±0.001). In-sample ceiling 0.914 / 0.796 exact → headroom exists but OOF can't use it.
- Root cause: confidence is 95% of signal; remaining 5% is under-powered (small N, noisy split labels, missing geometry, candidate recall ceiling 20 human positives outside candidate graph).

## Evidence
- `build_edge_table` Boston: rows=679 pos=482 neg=197 candidate_seen=681 rule5_filtered=2 human_selected_outside=20. 20 positives unlearnable.
- Crosstab `selected` vs `keep`: 100 TN, 97 FP (over-keep), 23 FN (under-keep), 459 TP. Model eF1: 122 TN, 75 FP, 40 FN, 442 TP → reduces FP 97→75 (+22 precision) but increases FN 23→40 (-17 recall). Net F1 flat.
- Clean slice: prod F1 0.942 vs conf-oracle 0.943 vs model CV 0.916 (model *hurts* clean). Split slice: prod 0.796 vs oracle 0.810 vs model 0.824 (model *helps* split).
- Importances: conf_rel_max 0.299, conf_rank_frac 0.103, confidence 0.043, num_refs 0.043 → relative confidence dominates; coverage/bridge/sliver <0.03.
- Base-only failures (prod correct, model wrong): groups 0e3e10ad, 4bcea059, 85ead29b, 9abce40c, dac6dc5d — high-conf (0.98-1.0) edges get oof 0.08-0.5 due to fold variance / competition features (margin_tgt -0.02) memorizing split noise.
- Model-only wins (10 groups): all small sliver-like stubs with conf 0.977-0.981, margin -0.02, small gers_span 0.04-0.15 local_span 0.05-0.2, bridge=True → correctly dropped. That's the parallel-sibling stub class already targeted by `MN_CONTESTED_EDGE_MAX_SPAN=0.3 + MAX_ABS 75m` demote-to-REVIEW guard (#367).

## Why confidence = prod
- `find_match_components`: sliver edges + weak edges below `glue_min_confidence` don't weld components → already prunes monsters.
- `apply_confidence_drop_prune`: absolute threshold, keep top per group → same as oracle t.
- `MN_CONTESTED` + `_validate_assignment_coverage` handle overlap redundancy.
- No lateral/name/length features in sidecar → no extra signal left.

## Cross-product artifact / noisy labels
- Empty `[]` reject-all: 52 on main, 0 rows emitted here (all candidate_groups, no empty labels mapped in this run — needs recovery fix).
- Split provenance 323/679 edges: human set spans multiple groups → within-group keep label is partial/noisy. Model trained on split hurts clean.
- 15-20 human_selected_outside_candidate_graph = sidecar candidate floor too high or glue prune removed them → label says keep but edge not in universe.

## Training improvements (ranked by expected delta)

### S — fix data plumbing (no model change)
- **Regen sidecars** for all 13 labeled datasets with `stitch_persist_candidate_graph=True`. Currently Boston-only. Target 150-300 groups. Δ +0.01-0.02 exact.
- **Fix candidate recall**: lower `min_confidence` for candidate graph export (keep prod threshold for selection, lower for candidate_edges) to recover 20 missing positives. Measure `human_selected_outside` → 0.
- **Empty handling**: ensure empty labels map via `recover_empty_reject_all` not just verbatim gid; emit full candidate universe with keep=0.

### M — robust to noisy labels (small)
- **Drop split from training, eval split separately**: train on clean only (`provenance=clean`), add split as soft or eval slice. Clean CV 0.916 → baseline 0.942 with more data may close. Already proven split hurts.
- **Label smoothing / symmetric CE**: `y_smooth = 0.9*keep + 0.1*(1-keep)` or `L = CE + 0.1*RCE` for split rows. Or confident learning: prune rows where oof disagrees >0.8.
- **Per-labeler weights**: `brad` 1.0, `panel_unanimous_v3` 1.0, `panel_split` 0.5, `deanchored` excluded or down-weighted.
- **Soft votes as true soft**: train on `soft_keep` float with BCE, not binarized `>=0.5`. Current 290 soft rows help but binarized.
- **Group-size stratification**: GroupKFold by group size / match_type to avoid folds with only 1:N etc.

### L — model + features
- **Per-type eF1 thresholds**: `select_expected_f1` currently argmax 2*sum_p/(k+total). Extend to per-match_type learned offset: M:N more conservative, 1:N keep-all. Calibrate k=0 empty probability using `prod(1-p)`.
- **Per-type prune as baseline**: Learn `t(|R|,|T|, match_type)` instead of single oracle t=0.98. Current config has one `optimizer_glue_min_confidence`.
- **Persist stage-2 features**: `<ds>_candidates.parquet` with 78 typed pair feats + signed lateral offset + length/class. Coverage of `labels/features` is 5% today — need 100% via runner.
- **Top new features** (from error groups): `signed lateral offset`, `hausdorff / frechet`, `name equality`, `endpoint dist to nearest kept`, `coverage complementarity * already present but needs refinement`, `bridge centrality`.

Risk: adding features at 108 groups overfits; need >=200 groups first.

## Optimizer heuristic hypotheses (not validated; do not ship yet)

Ranked as experiment candidates from the initial error analysis. The concrete
values are retained for reproducibility, not as recommendations:

1. **Per-match_type glue_min_confidence**: `1:N` may need higher glue to avoid monsters, while `N:1` may tolerate lower. The initial sweep candidates were `1:N:0.95, N:1:0.9, M:N:0.92`. Match type is only known after grouping at the current seam, so this also needs a non-circular algorithm definition.
2. **Per-match_type confidence_drop_prune**: the initial candidates were `1:N:0.92, N:1:0.94, M:N:0.96`. Fixed-universe evaluation did not beat the existing dataset-calibrated production selection.
3. **Relative margin prune** (S, medium): drop if `conf < max_group_conf - delta` (e.g. 0.05) instead of absolute; handles calibration drift. Already partially in `conf_rel_max`.
4. **Bridge-aware keep**: if `is_bridge` and `degree_ref=1` and `degree_tgt=1` → never prune (backbone). This changed zero decisions in the available labels and would require moving structural computation before prune.
5. **Sliver already correct**: `is_sliver` all keep=0 (0.0 mean) and `is_sliver=True` never selected; no change needed. Keep `SLIVER_SPAN_THRESHOLD=0.10` + 5m hybrid.
6. **Contested stub demote**: `MN_CONTESTED_EDGE_MAX_SPAN 0.3 + 75m` already catches 7/7 mode-A stubs. Could lower to 0.25 for more precision, but watch cascade: rescue logic keeps at least one per node.

## Experiment order (do in sequence, measure sliver-filtered F1 + exact + empty-exact)

1. `crosswalk stitch --all-datasets` with `stitch_persist_candidate_graph=True` → rebuild `data/output/*_groups.json` (unblocks multi-dataset).
2. Re-run `scripts/benchmark_resolver.py` + `uv run crosswalk train-resolver --clean-only` → measure clean exact.
3. Ablation: `ef1` vs `thr0.5` vs per-type thr vs `conf>=t` oracle on clean.
4. If clean still < prod, keep production defaults and use the fixed-universe harness from draft PR #411 to test paired prune/margin candidates without label-retention bias.
5. Then stage-2 parquet + XGBoost with lateral/name.

## Metrics to track (beyond pair F1)
- `F1_sliverfilt` (already) + `group_exact` + `F1_clean` + `F1_split`.
- Empty recall: % groups where model predicts empty set when label empty.
- `human_selected_outside_candidate_graph` → 0.
- Over-keep vs under-keep confusion matrix.

## Anti-patterns
- Don't train on split as hard labels; don't use `rejected_edges` capped 64 universe (legacy).
- Don't let `conf_is_group_min` dominate on size=1 groups.
- Don't threshold oof 0.5 — use eF1 per group.
- Don't add 78 feats without full candidate persistence (train/test skew per CLAUDE.md Backfill Architecture).
