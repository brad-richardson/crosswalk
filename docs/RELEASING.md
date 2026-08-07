# Releasing `crosswalk-py` to PyPI

The package is built and validated locally; **publishing is a deliberate human
act** — CI publishes only when a maintainer pushes a `v*` tag (the `publish` job
in `ci.yml`, via PyPI Trusted Publishing). Distribution name: **`crosswalk-py`**
(verified available 2026-07; bare `crosswalk` is squatted). Import package and
console script are `crosswalk`; a deprecated `matcher` console script warns and
forwards.

## What ships in the wheel

- The `crosswalk` package (`src/crosswalk/`), including the **pretrained model**
  `src/crosswalk/_model/matcher_model_combined.joblib` (~466 KB, isotonic
  calibration bundled).
- The **Spark-portable model** `src/crosswalk/_model/spark_model.json` (~1.1 MB
  XGBoost-native booster) + `src/crosswalk/_model/spark_manifest.json` (feature
  list, feature_version, hyperparams, isotonic calibration knots). Spark
  consumers import these from the wheel via `from crosswalk.spark import
  spark_model_json, spark_manifest` (see docs/ARCHITECTURE.md).
- The sdist is trimmed to `src/` + `README.md` + `pyproject.toml`
  (`[tool.hatch.build.targets.sdist]`) — without that it drags labels/research/
  mbench (~11 MB).

## Pre-release checklist

1. **Model lockstep** — the bundled model's `feature_version` must equal
   `config.FEATURE_VERSION`. CI enforces this (`tests/unit/test_shipped_model.py`);
   if you bumped features since the last reship:

   ```bash
   uv run crosswalk train -o src/crosswalk/_model/matcher_model_combined.joblib
   uv run pytest tests/unit/test_shipped_model.py -q
   ```

   The **Spark-portable model** has its own lockstep gate
   (`tests/unit/test_shipped_spark_model.py`): its manifest `feature_version`
   must equal `config.FEATURE_VERSION` and its `features` must exactly match
   `config.SPARK_PORTABLE_FEATURES`. If you bumped features (or the
   Spark-portable feature set / hyperparams) since the last reship, re-export
   and copy the two artifacts into the package:

   ```bash
   uv run crosswalk export-spark-model -o data/models/export
   cp data/models/export/model.json    src/crosswalk/_model/spark_model.json
   cp data/models/export/manifest.json src/crosswalk/_model/spark_manifest.json
   uv run pytest tests/unit/test_shipped_spark_model.py -q
   ```

   > **If you added to `SPARK_PORTABLE_FEATURES`, insert in `FEATURE_COLUMNS`
   > order — do not append.** The `features` comparison above is an *ordered*
   > list comparison, and the manifest is written from `matcher.feature_names`,
   > which the exporter rebuilds in `FEATURE_COLUMNS` order regardless of how the
   > config list is ordered. Appending therefore fails the lockstep test with a
   > diff that reads like a stale export rather than an ordering mistake.
   > `test_spark_portable_features_follow_feature_columns_order` catches it first.

   Widening the Spark feature set also means **retuning**
   `SPARK_PORTABLE_XGB_PARAMS` before the re-export — the committed values are
   fitted to a specific feature count:

   ```bash
   uv run python scripts/tune_model.py --feature-set spark --trials 300
   ```

   > **Use >= 300 trials, and sanity-check the selected point on LOO before
   > shipping it.** A 100-trial run of this exact search (the count used through
   > 2026-07-03) selected a point that was *worse than not retuning at all*:
   > LOO-by-type F1 0.8763 against 0.8777 for simply reusing the previous feature
   > set's hyperparameters, and ~30% slower to score. Epsilon-compact selection
   > optimizes inner-CV F1 -- a within-distribution metric -- and breaks ties on
   > `n_estimators * max_depth`, a proxy that mispredicted traversal cost by 4x on
   > that trial. CV F1 alone does not catch this.

2. **Version bump** — update `version` in `pyproject.toml` AND `__version__` in
   `src/crosswalk/__init__.py` (keep them equal), then `uv lock`.

3. **Full test suite + lint**:

   ```bash
   uv run ruff format src/ tests/ && uv run ruff check src/ tests/
   uv run pytest tests/ -q
   ```

4. **Build**:

   ```bash
   rm -rf dist && uv build
   ```

5. **Cold-start smoke test from the built wheel** (throwaway venv — this is the
   test that catches missing runtime deps; it caught `networkx` in 0.2.0):

   ```bash
   uv venv --python 3.12 /tmp/rm-smoke && VIRTUAL_ENV=/tmp/rm-smoke \
       uv pip install dist/crosswalk_py-*.whl
   /tmp/rm-smoke/bin/crosswalk fetch-overture --bbox -71.06,42.35,-71.05,42.36 -o /tmp/ref.parquet
   /tmp/rm-smoke/bin/crosswalk stitch -r /tmp/ref.parquet -t <some_local.parquet> -o /tmp/bridge.parquet
   ```

6. **Tag** the release commit: `git tag v<version> && git push --tags`. **This
   is the publish trigger** — see below.

## Publishing

**PyPI Trusted Publishing (configured, OIDC — no API tokens):** the PyPI project
`crosswalk-py` trusts this repo's **`ci.yml`** workflow with the **`pypi`**
GitHub environment. Pushing a `v*` tag runs the full CI matrix and then the
`publish` job, which verifies the tag matches `pyproject.toml`'s version,
builds with `uv build`, and uploads via `pypa/gh-action-pypi-publish`. Nothing
publishes on branch pushes or PRs.

**Manual fallback** (API token, scope it to the project):

```bash
uv publish  # prompts for credentials; or: uvx twine upload dist/*
```

## Post-publish

- Verify: `pip install crosswalk-py==<version>` in a clean venv, then
  `crosswalk --help` and the cold-start smoke test above.
- Update `docs/BENCHMARK_RESULTS.md` / README if install instructions changed.

## Retrain cadence / repo bloat note

The model is committed in-repo (not a release asset) because at <0.5 MB per
retrain the history cost is trivial next to the LFS label parquets, and it keeps
the wheel build hermetic (no fetch code path, no asset/commit version skew). If
retrain frequency ever makes this painful, the documented alternative is a
GitHub release asset + first-run download (`research/engine_dx_comparison.md`,
Part 2 #4).
