# Releasing `crosswalk-py` to PyPI

The package is built and validated locally; **publishing is a deliberate human
act** — nothing in CI pushes to PyPI. Distribution name: **`crosswalk-py`**
(verified available 2026-07; `matcher` is taken). Import package and console
script remain `matcher`.

## What ships in the wheel

- The `matcher` package (`src/crosswalk/`), including the **pretrained model**
  `src/crosswalk/_model/matcher_model_combined.joblib` (~466 KB, isotonic
  calibration bundled).
- The **Spark-portable model** `src/crosswalk/_model/spark_model.json` (~1.1 MB
  XGBoost-native booster) + `src/crosswalk/_model/spark_manifest.json` (feature
  list, feature_version, hyperparams, isotonic calibration knots). Spark
  consumers import these from the wheel via `from matcher.spark import
  spark_model_json, spark_manifest` (see docs/ARCHITECTURE.md).
- The sdist is trimmed to `src/` + `README.md` + `pyproject.toml`
  (`[tool.hatch.build.targets.sdist]`) — without that it drags labels/research/
  cbench (~11 MB).

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

6. **Tag** the release commit: `git tag v<version> && git push --tags`.

## Publishing

**Recommended: PyPI Trusted Publishing** (OIDC, no long-lived API tokens):

1. On PyPI: project settings → "Publishing" → add a trusted publisher for the
   GitHub repo (`brad-richardson/matcher`), a dedicated workflow filename (e.g.
   `release.yml`), and (recommended) a `pypi` GitHub environment.
2. Add a `release.yml` workflow triggered on the version tag that runs
   `uv build` and `pypa/gh-action-pypi-publish@release/v1` with
   `permissions: id-token: write`. The first publish registers the project name.

**Manual fallback** (API token, scope it to the project after first upload):

```bash
uv publish  # prompts for credentials; or: uvx twine upload dist/*
```

## Post-publish

- Verify: `pip install crosswalk-py==<version>` in a clean venv, then
  `matcher --help` and the cold-start smoke test above.
- Update `docs/BENCHMARK_RESULTS.md` / README if install instructions changed.

## Retrain cadence / repo bloat note

The model is committed in-repo (not a release asset) because at <0.5 MB per
retrain the history cost is trivial next to the LFS label parquets, and it keeps
the wheel build hermetic (no fetch code path, no asset/commit version skew). If
retrain frequency ever makes this painful, the documented alternative is a
GitHub release asset + first-run download (`research/engine_dx_comparison.md`,
Part 2 #4).
