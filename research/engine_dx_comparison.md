# Engine developer-experience comparison — what it costs to run each matcher

> _Harness renamed `cbench` → `mbench` (2026-07-05). Mentions of "cbench" below are historical; the harness is now invoked as `mbench` (a deprecated `cbench` alias still forwards)._

**Trigger:** the head-to-head in [`docs/BENCHMARK_RESULTS.md`](../docs/BENCHMARK_RESULTS.md)
showed Valhalla **Meili** reaching **F1 0.994** on Boston roads from *one pip
dependency, no training, no labels, ~12 s* — while `matcher` needs a clone, a heavy
install, a trained model, and a data fetch before it does anything. This note makes
that friction visible: it documents the **cold-start path** for every engine in
`cbench` (a new user holding two input parquets who wants a GERS bridge table),
scores each on a rubric, records which alternative engines were surveyed and
*rejected*, and lays out a ranked plan to make `matcher` itself as easy to run as
the best baseline.

All cold-start claims for `matcher` were **re-enacted in a throwaway venv** (see
"Cold-start reenactment" below); the numbers are measured, not remembered.

---

## Part 1 — the process comparison

### Rubric (1 = painful, 5 = effortless)

Five axes: **Steps** to first result, **Dep weight** (pip vs Docker vs emulation),
**Config** burden (knobs before a first answer), **Time** to first result,
**Maint** (retraining / image aging / upstream liveness).

| Engine | Steps | Dep weight | Config | Time | Maint | **Σ/25** | One-line verdict |
|--------|:-----:|:----------:|:------:|:----:|:-----:|:--------:|------------------|
| **Valhalla Meili** | 5 | 4 | 4 | 4 | 4 | **21** | One pip extra, auto-builds the graph, ARM-native, no model. The DX bar to beat. |
| **naive floor** | 5 | 5 | 4 | 5 | 5 | **24** | Trivial to run — but it *is* the floor (F1 0.839 roads / 0.365 sidewalks). |
| **matcher (post top-3 fixes, 2026-07-05)** | 5 | 3 | 4 | 3 | 4 | **19** | `pip install road-matcher` → `fetch-overture --clip-target` → `stitch`. No train, no YAML, no clone. See "Post-fix update" below. |
| ~~matcher (pre-fix)~~ | ~~2~~ | ~~3~~ | ~~4~~ | ~~2~~ | ~~3~~ | ~~**14**~~ | Best quality, worst cold-start: clone + heavy install + **train** + **fetch** + stitch. |
| **Hootenanny 0.2.41** (emulated) | 3 | 2 | 3 | 1 | 1 | **10** | Prebuilt amd64 image under x86 emulation on ARM; wall time invalid; frozen 2018. |
| **Hootenanny 0.2.87** (native x86) | 1 | 1 | 2 | 2 | 2 | **8** | No runnable image — multi-hour source build or a native-x86 box; snap-merge aborts. |

> The naive floor scores highest on pure DX precisely because it does the least.
> Read the rubric as "effort to first result," not "quality" — quality is in
> `BENCHMARK_RESULTS.md`. The interesting comparison was **Meili (21) vs matcher
> (14)**: a 7-point DX gap at near-identical quality (0.994 vs 0.996). The whole of
> Part 2 was about closing it — and the top-3 fixes landed (see "Post-fix update"
> at the end), re-scoring matcher to **19** with the residual 2-point gap (dep
> weight + stitch time) inherent to the ML stack rather than setup friction.

### Cold-start narratives

Assume the user already has the two input parquets (reference = Overture segments,
target = local roads) — the same starting point `cbench` gives every adapter.

#### naive (the floor) — `Σ 24`

- **Deps:** `geopandas` + `shapely` (the `cbench[hootenanny]` extra), pure pip, pure Python.
- **Steps:** one — `cbench run naive us_boston_streets`. Runs **in-process**, no subprocess, no image.
- **Config:** three knobs (`buffer_m=15`, `min_overlap=0.30`, `angle_tol_deg=35`), all with sane defaults; zero required.
- **Time:** ~4 s.
- **What breaks:** nothing operationally. It just *collapses on quality*: recall 0.24 / F1 0.365 on dense parallel sidewalks (one target buffer swallows many parallel Overture edges; end-to-end bearing can't separate them). On roads it is honest (F1 0.839).
- **Maint:** none — it lives in this repo (`cbench/src/cbench/adapters/naive.py`), no external anything.

#### Valhalla Meili — `Σ 21` (the bar to beat)

- **Deps:** `cbench[meili]` = `geopandas` + `shapely` + `osmium` + **`pyvalhalla`**. The `pyvalhalla` wheel **bundles native-ARM Valhalla 3.7.0 binaries**, so there is *no Docker, no service, no emulation*. The only catch: it needs **Python ≥ 3.12** (abi3 wheel), so cbench must run from a 3.12 env (`uv run --python 3.12 cbench …`).
- **Steps:** one — `cbench run meili us_boston_streets`. **No training, no labels, no model.**
- **Data prep (the key DX win):** the adapter does the Overture→routable-graph conversion *itself and automatically* — `cbench/src/cbench/convert/pbf.py` turns the reference parquet into an OSM PBF (GERS id carried as the OSM `way_id`, connector coords collapsed to shared nodes), then `valhalla_build_tiles` builds the tileset. **Tiles are cached per reference file** (keyed by name+size+mtime), so the dominant cost is paid once; repeat runs reuse them. The user supplies nothing but the two parquets `cbench` already resolves.
- **Config:** all defaulted (`costing=pedestrian` covers roads *and* sidewalks, `densify_m=10`, `search_radius=25`, overlap filter `0.10`/`8 m`). Zero knobs required.
- **Time:** **12.0 s** Boston cold (7 s of which is the one-time tile build), **5.0 s** match-only with cached tiles; 15 s FC, 29 s Seattle. ~6–17× faster than matcher.
- **What breaks (operational routes from `docs/BENCHMARKING.md`; quality analysis in `research/meili_baseline.md`):**
  - Both **Docker** routes are dead ends on Apple Silicon — the ghcr.io ARM image **stalls at 0 bytes/s** on blob download, and the Docker Hub amd64 image **segfaults under qemu** in `valhalla_build_tiles`. The in-process `pyvalhalla` wheel is the *only* working path (and the reason it scores 4 not 5 on deps).
  - Py 3.11 can't install `pyvalhalla`, so the adapter is unavailable on the 3.11 cbench CI env — its pure logic is unit-tested there instead.
  - **Quality ceiling, not an operational break:** no first-class no-match (map-matching snaps every trace onto *something*), so precision is lost to parallel-geometry snaps (sidewalk → adjacent road centerline 2.4–3.0 m away). Perfect recall, precision tax.
- **Maint:** upstream Valhalla is **actively maintained** and multi-arch; the pinned wheel ages gracefully. No model to retrain, no labels to curate.

#### matcher — `Σ 14` (be honest)

The **re-enacted** fresh-clone path to a Boston bridge:

1. `uv pip install -e ".[dev,ml]"` — the `ml` extra adds `lightgbm` + `pygeoops` on top of an already-heavy core (`xgboost`, `numba`, `optuna`, `duckdb`, `geopandas`, `overturemaps`, `osmium`, `scikit-learn`). **~2 s with a warm uv cache; minutes on a truly cold machine** (numba/xgboost/geopandas wheels are large).
2. `matcher train` — **required after every fresh clone: `data/models/` is gitignored, so a fresh clone has no model and cannot stitch at all.** The good news the reenactment surfaced: **labels *are* committed** (`labels/human/dataset=*`, 34 datasets, 639 Boston pair labels), so training works offline with zero setup. Measured: **~35 s**, emits a **465 KB** `matcher_model_combined.joblib` (the task estimated ~5 MB — it is an order of magnitude smaller, which matters for Part 2).
3. `matcher data fetch all us_boston_streets` — **also required: `data/` is gitignored, so `data/raw/` is empty on a fresh clone.** For a *configured* dataset this is one automatic command (bbox + ArcGIS URL live in `datasets/us_boston_streets.yaml`); it pulls the target from a Boston ArcGIS FeatureServer and the reference from Overture (via the `overturemaps` lib over S3 — both are *core* deps, so no extra install). But it needs **network**, and for an *arbitrary* dataset the user must first author a YAML (source URL + bbox) or skip fetch entirely and bring their own two parquets via `-r/-t`.
4. `matcher stitch us_boston_streets -o bridge.parquet` — **~85 s** (2965 MB peak), or `matcher stitch -r ref.parquet -t target.parquet -o bridge.parquet` if you brought your own parquets (skips step 3).

- **Config:** genuinely light at the happy path — `stitch` runs on defaults; `-p recall|balanced|precision` and `-b buffer_m` are the only knobs most users touch. This axis is *not* the problem (scores 4).
- **What breaks (reenacted):**
  - `matcher stitch us_boston_streets` on a fresh clone → **`Could not find reference (Overture) file for 'us_boston_streets'`** (data/raw empty). This is the first wall a new user hits.
  - No model file → `stitch` can't score. (You must `train` first; there is no shipped model.)
  - **Correction to a common assumption:** loading a model whose `feature_version` ≠ current `FEATURE_VERSION` is only a **`logger.warning` ("Consider retraining")**, *not* a hard error (`ml.py::load_model`). The hard error is at **train** time, on *labels* carrying a stale `feature_version` (`_check_feature_versions`). So a shipped model would keep loading across releases and silently warn — which is a **risk** for Part 2, not a safety net.
- **Maint:** the model must be retrained whenever `FEATURE_VERSION` bumps (currently `2026-07-04.2`) or the label base grows; that is the standing tax a shipped-model story has to manage.

Net: matcher's steps score (2) and time score (2) are dragged down entirely by **train + fetch**, not by stitch itself. Everything Part 2 proposes targets those two steps.

#### Hootenanny 0.2.41, emulated — `Σ 10`

- **Deps:** a **prebuilt amd64 Docker image** (`hootenanny/run:0.2.41-1`, 2018) pulled with `--platform linux/amd64` and run **under x86 emulation** on Apple Silicon. Docker + qemu is a heavy, slow dependency (dep score 2).
- **Steps:** `docker pull …` then `cbench run hootenanny … --opt hoot_image=…` (the adapter drives `docker run` and the OSM conversion for you). Full `hoot conflate` is run only because match-only discards results; cbench extracts correspondences from provenance tags.
- **Config:** version-specific creator classes (`hoot::HighwayMatchCreator` / `hoot::HighwaySnapMergerCreator`) — defaulted by the adapter but a known trap across versions.
- **Time:** **invalid** — emulated, so only quality (F1 0.973 roads / 0.927 sidewalks) is reported, never wall time (time score 1).
- **What breaks:** repeated non-fatal `Two nodes … same coordinate` merge warnings (0.2.41 tolerates them). The bigger issue is that 0.2.41 is simply *old*.
- **Maint:** **frozen 2018 baseline**, Hootenanny is out of active maintenance; recorded once with exact version+config and never re-run (maint score 1).

#### Hootenanny 0.2.87, native x86 — `Σ 8`

- **Deps:** the current release **has no runnable prebuilt image** — the `rpmbuild-*` images are build *environments* with no `hoot` binary staged. So you either do a **multi-hour, high-risk source build** (EL/CentOS, GDAL/GEOS/PROJ/v8/node from source, under emulation on ARM) or, as the `†` rows did, run a source-built `core-services:latest` on an actual **native-x86 Linux box** (dep score 1, steps score 1).
- **Config:** must override the merger to `LinearTagOnlyMerger` (see below).
- **Time:** valid and honest at last — **5m47s** Boston full / **3m26s** match-only / **4m50s** FC, single-threaded, ~1.8 GB peak (time score 2).
- **What breaks (from `research/hoot_native_baseline.md`):** the default/Unifying **`LinearSnapMerger` aborts** at ~52% (`No node ID specified for RemoveNodeByEid`) on synthetic connector-less target OSM — 0.2.41 tolerated the coincident-coordinate nodes, 0.2.87 does not. Every geometry-snapping merger (Unifying, Network, AttributeConflation.conf) hits it. Only `LinearTagOnlyMerger` completes, but it never splits reference ways, so Boston's short-local-vs-long-Overture segmentation collapses recall to 0.634 (a merge-representation artifact, not a matcher regression — FC stays at F1 0.940). The faithful snap-merge Boston quality stays pinned to the 0.2.41 emulated numbers.
- **Maint:** ~9-month release gap, effectively frozen for our purposes (maint score 2).

---

### Rejected-engines survey — "is there another engine besides Valhalla?"

Recorded answer, from the baseline-landscape verification in `docs/BENCHMARKING.md`
(the #275 arc). Weighted by **MATCH-stage output only** (can it emit
segment↔segment correspondences headlessly?).

| Tool | Status (Jul 2026) | Verdict | Why rejected / deferred |
|------|-------------------|---------|--------------------------|
| **GraphHopper map-matching** | **Actively maintained** (2025 releases, on Maven Central) | **The one live alternative to Valhalla** — *not* rejected, deferred | Same segment-as-trace paradigm as Meili, as an **embeddable JVM library (no server)**. Shares ~80% of the work (the Overture→OSM-PBF tax) with the Meili adapter already built. Pick it if avoiding a running service matters more than Valhalla's stronger conflation precedent. **This is the concrete "yes, there is another" answer.** |
| **SharedStreets `shst match`** | **Effectively abandoned** (last real release v0.15.2, May 2020) | **Rejected** | Matches to *global SharedStreets reference tiles*, not an arbitrary supplied reference — impedance mismatch with our ref/target contract. Needs ancient Node 10–14 (`node-gyp` fails on Node 20/22) and an unmaintained tile backend. Explicit "recommend against." |
| **OpenLR** (`tomtom-international/openlr`) | Maintained (2025 commits) | **Rejected** | A location-referencing *codec* (encode on map A, decode on map B), not a conflation engine. You'd hand-build an encode→decode harness on a tool never meant for bulk conflation, and decode quality needs consistent FRC/FOW attributes local data usually lacks. Low priority. |
| **RoadMatcher** (vividsolutions) | Dormant | **Rejected** | Java tool, no active maintenance. |
| **JOSM Conflation Plugin** | Maintained | **Rejected** | Interactive in-editor conflation, not headless. |
| **Overture `match-inspector`** | Maintained | **N/A** | Conflates *building footprints*, not roads; useful only as a labeling-UI reference. |

**Ecosystem note (the reason matcher exists):** there is **no official Overture
drop-in "match two road-linework sets → GERS links" open-source tool**. Overture's
own road conflation is an internal pipeline whose *outputs* (GERS bridge files) are
published, not the matcher. GERS bridge files map GERS↔source IDs for a *fixed* set
of upstream datasets (OSM, Esri, Meta/Microsoft) — eval scaffolding for OSM-sourced
roads, not a tool for arbitrary local data. Commercial options (TomTom GEM,
CARTO/Databricks, Wherobots/Sedona) are closed.

**Bottom line for the user:** yes — **GraphHopper** is the second live map-matching
engine, and it reuses the Overture→PBF converter Meili already needs. Everything
else in the landscape is abandoned, closed, or the wrong tool shape.

---

## Part 2 — matcher DX improvement plan (ranked)

Goal: shrink matcher's cold-start from *clone + install + train + fetch + stitch*
toward Meili's *install + run*. Ranked by **(new-user impact × inverse effort)**.

### The ranking

| # | Improvement | New-user impact | Effort | Notes |
|---|-------------|:---------------:|:------:|-------|
| **1** | **Ship a pretrained model** | **Very high** | **Low** | Removes the mandatory `train` step. Model is **465 KB** — trivially committable. |
| **2** | **Overture reference fetch from a bbox, no YAML** | **High** | **Medium** | Removes the "author a dataset YAML" wall for arbitrary areas; `matcher data fetch` already wraps the `overturemaps` lib. |
| **3** | **PyPI packaging (`pip install road-matcher` / `uvx`)** | **High** | **Medium** | Removes the clone + editable-install step; pairs with #1 (a wheel that bundles the model = install-and-run). |
| 4 | **`matcher init` / first-run model auto-download** | Medium | Low–Med | Alternative to committing the model if repo bloat is a concern (GitHub release asset). |
| 5 | **Leaner default memory** (~3 GB Boston) | Medium | Medium | `--workers` already exists; make the lean path the default or auto-scale. |
| 6 | **`--no-ml` degraded mode** | Low | Low–Med | Honesty cost is high (F1 0.839 roads / **0.365 sidewalks**); see below. |

### Detail on the top candidates

**#1 Ship a pretrained model — the single biggest gap, and cheap.**
`data/models/` is gitignored, so a fresh clone literally cannot stitch; `train` is a
*mandatory* 35 s + dependency step, not an optional one. The reenactment shows the
artifact is only **465 KB** (task estimate: ~5 MB — even more shippable). Options,
easiest first:
- **Commit the joblib** under a tracked path (e.g. `src/matcher/_model/…` so it ships
  in the wheel too). Simplest; adds <0.5 MB to the repo.
- **GitHub release asset auto-downloaded on first run** (`matcher init` or lazy in
  `load_model`). Keeps the repo lean; adds a network dependency and a fetch code
  path.

**Critical compatibility caveat this must handle:** a shipped model has a stored
`feature_version`, and today `load_model` only **warns** (not errors) when it
mismatches current `FEATURE_VERSION` (`2026-07-04.2`). So a stale shipped model
would keep loading and **silently degrade** after any feature change. A shipped-model
story therefore needs one of: (a) **pin the model's `feature_version` in CI** — a
test that fails when `FEATURE_VERSION` changes without re-exporting the shipped
model, forcing the commit to stay in lockstep; or (b) **upgrade the mismatch to a
hard error** with a clear "run `matcher train` or update matcher" message. (a) is the
lower-friction choice and keeps the shipped-model + release process honest.

**#2 Overture reference fetch from a bare bbox.** The machinery already exists —
`matcher data fetch reference` calls `overturemaps.core.geodataframe(bbox, release)`
and `get_latest_release()`. The friction is that it is gated on a *configured
dataset* (bbox pulled from `datasets/<name>.yaml`). A `matcher fetch-overture
--bbox <…> -o ref.parquet` that skips the YAML would let a new user get the Overture
half of *any* area in one command, matching Meili's "bring only your local data"
ergonomics. (The local/target half genuinely must be user-supplied — that is
inherent, not a matcher wart.)

**#3 PyPI packaging.** Clone + `uv pip install -e` is a real step Meili doesn't have
(`pip install pyvalhalla`-class). A published `road-matcher` wheel — ideally
**bundling the #1 model** — collapses matcher's cold-start to `pip install
road-matcher && matcher stitch -r ref.parquet -t tgt.parquet -o bridge.parquet`,
i.e. **two lines, no train, no clone**. The heavy transitive deps (numba, xgboost,
geopandas) remain, but that is one `pip` resolve, same shape as Meili's.

**#5 Memory.** Boston stitch peaks ~2965 MB vs Meili's ~838 MB. `--workers` already
lets users trade memory for time; the improvement is making a lean profile the
default (or auto-scaling workers to available RAM) so the happy path doesn't OOM a
small laptop. Medium effort, medium payoff.

**#6 `--no-ml` degraded mode — measure the honesty cost.** The naive adapter's logic
(`cbench/src/cbench/adapters/naive.py`) is a zero-training buffer-overlap matcher; exposing it
in `matcher` proper would give an instant, model-free first result. **But the
honesty cost is steep:** F1 **0.839 on roads** and **0.365 on sidewalks** (recall
0.24 — it collapses on dense parallel footways). That is a *floor*, not a product; a
`--no-ml` flag risks users benchmarking matcher at floor quality. If shipped, it must
be loudly labeled "geometric floor, unlearned." Given #1 makes the *real* model a
zero-setup default anyway, `--no-ml` is low priority — #1 removes the reason to want
it.

### Recommended "do now" — top 3

1. **Ship the 465 KB pretrained model** (commit it, or release-asset + first-run
   fetch) **+ a `FEATURE_VERSION` lockstep CI test**. *Effort: low (~half a day).*
   Impact: eliminates the mandatory `train` step and the "fresh clone can't stitch"
   wall — the biggest single DX cliff. The CI test keeps the shipped model honest
   across feature changes (the current warn-don't-error behavior is a latent trap).
2. **PyPI wheel that bundles the model** (`pip install road-matcher`). *Effort:
   medium (~1–2 days incl. packaging the model as package data).* Impact: with #1,
   collapses cold-start to `pip install` + one `stitch -r/-t` line — matching Meili's
   two-line ergonomics for users who bring their own two parquets.
3. **`matcher fetch-overture --bbox … -o ref.parquet`** (YAML-free reference fetch).
   *Effort: medium (~1 day; the `overturemaps` call already exists).* Impact: removes
   the last acquisition wall for arbitrary areas, so a new user only has to supply
   their *local* data — the same starting line Meili gives.

Together these three turn matcher's five-step cold-start (clone → install → train →
fetch → stitch) into **install → (fetch-overture) → stitch** with no training and no
clone — DX parity with Meili, at matcher's higher quality.

---

## Cold-start reenactment — what actually happened

Re-enacted in a throwaway venv (`uv venv --python 3.11`, editable install from this
checkout), recording each stumble as data:

1. **Install** `uv pip install -e ".[dev,ml]"` — **~2 s (warm uv cache)**; would be
   minutes cold (large numba/xgboost/geopandas/optuna wheels). No failures.
2. **`matcher train`** — **succeeded in ~35 s** offline. Surprise (good): **labels
   are committed** (34 datasets, 639 Boston labels), so training needs no data fetch.
   Emitted `data/models/matcher_model_combined.joblib` at **465 KB** (Test acc 0.907,
   CV F1 0.929).
3. **`matcher stitch us_boston_streets`** (before fetch) — **failed**: `Could not
   find reference (Overture) file for 'us_boston_streets'`. `data/raw/` is gitignored
   and empty on a fresh clone — the first wall.
4. **Overture acquisition** is built in: `overturemaps` and `osmium` are *core* deps
   (not extras), and `datasets/us_boston_streets.yaml` carries the bbox + ArcGIS
   source, so `matcher data fetch all us_boston_streets` is one (networked) command
   for configured datasets. Arbitrary areas need a hand-authored YAML or user-supplied
   parquets.

Two surprises worth flagging:
- The shipped-model gap is **cheaper to close than assumed** — the model is **465 KB**,
  not ~5 MB.
- The "code hard-errors on stale feature versions" premise is **only half true**:
  the hard error is at **train** time on stale *label* features; **model load only
  *warns*** on a `feature_version` mismatch. A shipped model would keep loading and
  silently degrade — hence the recommended CI lockstep test in #1.

---

## Post-fix update (2026-07-05) — the top-3 landed

All three recommended fixes shipped; matcher's rubric row is re-scored **14 → 19**.

**What shipped:**

1. **Pretrained model committed into the package** —
   `src/matcher/_model/matcher_model_combined.joblib` (466 KB, isotonic
   calibration included). `stitch` uses it automatically whenever
   `data/models/` has no locally trained model (a local model always takes
   precedence). *Committed-in-repo* was chosen over a release-asset fetch: at
   <0.5 MB/retrain the repo-bloat cost is trivial for a hobby-scale retrain
   cadence (git history grows by one small blob per reship; the labels' LFS
   parquets dwarf it), while a first-run download would add a network dependency,
   a fetch code path, and a "which asset matches this commit?" versioning problem
   that the in-tree copy solves for free.
   The **silent-degradation trap is closed twice over**: model load now
   **hard-errors** on a `feature_version` mismatch (escape hatches:
   `--allow-version-mismatch` / `MATCHER_ALLOW_MODEL_VERSION_MISMATCH=1`; the
   trusted bundled path is exempt), and the CI lockstep test
   (`tests/unit/test_shipped_model.py`) fails any PR that bumps
   `FEATURE_VERSION` without reshipping the bundled artifact — retrain + reship
   must land in the same PR (`matcher train -o src/matcher/_model/…`). The test
   also asserts the shipped calibration knots are present and the feature list
   matches `config.FEATURE_COLUMNS`.
2. **PyPI packaging** — the distribution is **`road-matcher`** (verified
   available on PyPI 2026-07; `matcher` is taken), import package and console
   script stay `matcher`. The wheel is **930 KB including the model**; the sdist
   is trimmed to the package (was dragging labels/research/cbench, 11 MB → 0.9 MB).
   Core deps were slimmed to what stitch needs: `optuna` (tuning-script-only) and
   `pillow`/`mercantile` (imagery) moved to extras; unused `lightgbm` dropped;
   `networkx` — a real stitch dependency the cold-start test caught — moved *into*
   core. Not yet published (publishing is the user's act); `uv build` artifacts +
   `docs/RELEASING.md` checklist, with PyPI trusted publishing as the recommended
   path.
3. **`matcher fetch-overture`** — YAML-free reference fetch:
   `--bbox xmin,ymin,xmax,ymax` or `--clip-target my_roads.parquet` (bbox derived
   from the target's extent — the zero-thought path), `--release` pinning
   (default: latest), 1 km topology buffer (`--buffer-m`), optional
   `--connectors`, and the `.meta.yaml` sidecar with the `release` field the
   factory needs.

**Measured cold start** (throwaway venv, wheel install, 389-segment downtown-Boston
slice as the user's "local data"; Apple Silicon, warm uv cache):

```text
uv venv --python 3.12 && uv pip install road_matcher-0.2.0-py3-none-any.whl   # 0.3 s
matcher fetch-overture --clip-target my_roads.parquet -o ref.parquet          # 33 s (network)
matcher stitch -r ref.parquet -t my_roads.parquet -o bridge.parquet           # 46 s
```

Total **~80 s to a 537-row bridge parquet (370/389 targets matched) with zero
training, zero YAML, zero clone** — vs the pre-fix path of clone + install +
train (35 s) + hand-authored YAML or config-gated fetch + stitch.

**Honest re-score:** Steps 2→**5** (install → fetch → stitch; with parquets already
in hand it is install → stitch, Meili-equal). Time 2→**3** (no train step, but
stitch itself is still ~6× Meili's match). Dep weight stays **3** (the pip resolve
is still numba/xgboost/geopandas-heavy — one resolve, but a heavy one). Config
stays **4**. Maint 3→**4** (the retrain tax on `FEATURE_VERSION` bumps remains,
but it is now CI-enforced and versioned in-repo rather than silent). Σ = **19/25**
vs Meili's 21; the residual gap is engine-inherent (dependency mass + match time),
not cold-start friction.
