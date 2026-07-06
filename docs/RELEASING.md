# Releasing `road-matcher` to PyPI

The package is built and validated locally; **publishing is a deliberate human
act** — nothing in CI pushes to PyPI. Distribution name: **`road-matcher`**
(verified available 2026-07; `matcher` is taken). Import package and console
script remain `matcher`.

## What ships in the wheel

- The `matcher` package (`src/matcher/`), including the **pretrained model**
  `src/matcher/_model/matcher_model_combined.joblib` (~466 KB, isotonic
  calibration bundled). Total wheel size ~930 KB.
- The sdist is trimmed to `src/` + `README.md` + `pyproject.toml`
  (`[tool.hatch.build.targets.sdist]`) — without that it drags labels/research/
  cbench (~11 MB).

## Pre-release checklist

1. **Model lockstep** — the bundled model's `feature_version` must equal
   `config.FEATURE_VERSION`. CI enforces this (`tests/unit/test_shipped_model.py`);
   if you bumped features since the last reship:

   ```bash
   uv run matcher train -o src/matcher/_model/matcher_model_combined.joblib
   uv run pytest tests/unit/test_shipped_model.py -q
   ```

2. **Version bump** — update `version` in `pyproject.toml` AND `__version__` in
   `src/matcher/__init__.py` (keep them equal), then `uv lock`.

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
       uv pip install dist/road_matcher-*.whl
   /tmp/rm-smoke/bin/matcher fetch-overture --bbox -71.06,42.35,-71.05,42.36 -o /tmp/ref.parquet
   /tmp/rm-smoke/bin/matcher stitch -r /tmp/ref.parquet -t <some_local.parquet> -o /tmp/bridge.parquet
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

- Verify: `pip install road-matcher==<version>` in a clean venv, then
  `matcher --help` and the cold-start smoke test above.
- Update `docs/BENCHMARK_RESULTS.md` / README if install instructions changed.

## Retrain cadence / repo bloat note

The model is committed in-repo (not a release asset) because at <0.5 MB per
retrain the history cost is trivial next to the LFS label parquets, and it keeps
the wheel build hermetic (no fetch code path, no asset/commit version skew). If
retrain frequency ever makes this painful, the documented alternative is a
GitHub release asset + first-run download (`research/engine_dx_comparison.md`,
Part 2 #4).
