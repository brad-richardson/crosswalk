"""Demo + guard for the boundary of SPARK_PORTABLE_FEATURES.

`crosswalk export-spark-model` ships a subset of the 83 FEATURE_COLUMNS, on the
stated rationale that they are "the subset computable from aligned geometry pairs
(no topology, graph, or spatial-index features required)". That rationale was
over-broad: 17 then-excluded features also need nothing but the two aligned
geometries and the two name structs the Spark job already holds to compute
`name_levenshtein` / `sinuosity_ref` / `ref_coverage`.

6 of those 17 shipped on 2026-08-07 and the set is now 34: the name block minus
`route_prefix_match`, which is out on fill rate (non-NaN on 1 of 5,532 stored
labels -- see test_route_prefix_match_is_almost_always_nan). The other 11 remain
excluded on measured value, not feasibility. This module keeps proving the
feasibility of all 17 regardless of which currently ship -- that is what makes the
exclusion an argument about value.

This module *proves* that by computation rather than by reading:
:func:`compute_spark_addable_features` takes only ``(ref_geom, target_geom,
ref_names, target_names)`` -- no STRtree, no connector graph, no topology dicts,
no native-target degrees -- and reproduces the authoritative
``compute_pair_features()`` output for all 17 columns on real labelled pairs --
bit-for-bit for the 15 that vary across the fixture, and equal-but-degenerate for
the 2 that are constant on it (see DEGENERATE_ON_FIXTURE). The same fixture shows the genuinely context-dependent columns
(topology, graphlet, clustering, crossing angle, parallel sibling, endpoint
proximity) collapsing to NaN when that context is withheld, which is what makes
them un-addable regardless of importance score.

Nothing here mutates the shipped artifacts. The F1 / model-size sweep across
feature tiers lives in ``research/spark_feature_expansion.py``; the marginal
feature-computation cost is measured by ``test_addable_feature_marginal_cost``
below.

Run:
    uv run pytest tests/test_spark_feature_expansion.py -p no:randomly
    uv run pytest tests/test_spark_feature_expansion.py -m slow -s   # timings
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from crosswalk.config import FEATURE_CATEGORIES, FEATURE_COLUMNS, SPARK_PORTABLE_FEATURES
from crosswalk.features.alignment import (
    compute_coverage_features,
    create_subline,
    linestring_alignment,
)
from crosswalk.features.compute import compute_pair_features
from crosswalk.features.geometric import (
    compute_angle_histogram_similarity,
    compute_heading_consistency,
    compute_shape_complexity,
    compute_sinuosity,
    compute_vertex_density,
)
from crosswalk.features.semantic import (
    compute_name_similarity,
    compute_route_prefix_match,
    resolve_best_name_variant,
)

# Mirrors compute_pair_features(): below this coverage the aligned sub-portion is
# extracted, at or above it the full geometry is reused (identical result, no
# subline allocation). Duplicated rather than imported because the Spark job has
# to reproduce the constant on its side too -- that is the point of the demo.
HIGH_COVERAGE_THRESHOLD = 0.995

# The 7 name features excluded from the Spark set. Every one falls out of the
# same `compute_name_similarity` / `compute_route_prefix_match` calls that
# already produce the shipped `name_levenshtein` and `name_token_sort`.
ADDABLE_NAME_FEATURES = [
    "name_jaro_winkler",
    "name_soundex",
    "name_metaphone",
    "has_name_ref",
    "has_name_target",
    "name_is_generic",
    "route_prefix_match",
]

# The 10 excluded geometry features that need only the aligned geometry pair.
ADDABLE_GEOMETRY_FEATURES = [
    "max_coverage",
    "sinuosity_delta",
    "angle_histogram_similarity",
    "shape_complexity_ref",
    "shape_complexity_delta",
    "heading_consistency_ref",
    "heading_consistency_delta",
    "vertex_density_ref",
    "vertex_density_target",
    "vertex_density_ratio",
]

ADDABLE_FEATURES = ADDABLE_NAME_FEATURES + ADDABLE_GEOMETRY_FEATURES

# Of the 7 feasible name features, 6 shipped on 2026-08-07. `route_prefix_match`
# did not: it is non-NaN on 1 of 5,532 stored labelled pairs (0.02%) because it
# needs BOTH names to canonicalize to a route designation, and it is the only name
# feature requiring a call of its own. Excluding it is what takes the block from
# 1.18 us/pair to literally free.
SPARSE_NAME_FEATURES = ["route_prefix_match"]
SHIPPED_NAME_FEATURES = [f for f in ADDABLE_NAME_FEATURES if f not in SPARSE_NAME_FEATURES]

# Derivable from columns the shipped 28-feature model already carries, with no
# new geometry pass whatsoever: max(ref_coverage, target_coverage) and
# abs(sinuosity_ref - sinuosity_target).
FREE_DERIVED_FEATURES = ["max_coverage", "sinuosity_delta"]

# Features that genuinely need network context (a spatial index over the whole
# target layer, the Overture connector graph, or a corridor/sibling search).
# Withholding that context must NaN them out -- that is the evidence they cannot
# move into a per-pair Spark UDF.
CONTEXT_DEPENDENT_FEATURES = [
    "min_endpoint_proximity_m",  # cKDTree over every target endpoint
    "max_endpoint_proximity_m",
    "shared_endpoint_count",
    "graphlet_similarity",  # connector graph
    "endpoint_degree_similarity",
    "clustering_coef_ref",
    "clustering_coef_target",
    "clustering_coef_delta",
    "crossing_angle_min_ref",  # STRtree over the Overture layer
    "transverse_neighbor_fraction_ref",
    "crossing_angle_min_target",
    "transverse_neighbor_fraction_target",
    "has_parallel_sibling_ref",  # corridor/sibling search over an STRtree
    "parallel_fraction_ref",
    "offset_vs_half_corridor_ratio",
    "likely_representation_mismatch",
]

# The Topology category, split out because it fails for a different reason: it
# needs Overture connectors projected onto both sides plus the target network's
# own endpoint-cluster Union-Find, neither of which exists per candidate row.
TOPOLOGY_FEATURES = FEATURE_CATEGORIES["Topology"]

# Datasets sampled for the parity fixture. Three different segmentation styles
# (dense US grid, European road network, sidewalk layer) so the comparison
# exercises reversed alignments, partial coverage, and missing names.
SAMPLE_DATASETS = ["us_boston_streets", "de_berlin_roads", "us_seattle_sidewalks"]
SAMPLE_PAIRS_PER_DATASET = 60

# Features that are legitimately CONSTANT across the whole fixture, so the parity
# test below cannot distinguish a correct implementation from a stub returning
# that constant. Measured 2026-08-07 on the 180-pair fixture:
#   route_prefix_match  -- NaN on all 180 (needs a route designation on BOTH sides;
#                          non-NaN on 1 of 5,532 stored labels repo-wide)
#   name_is_generic     -- 0.0 on all 180 (no generic-pattern names in these three
#                          urban street/sidewalk layers)
# They are still compared for equality; the point is that "proven bit-for-bit" is
# a strong claim for the other 15 and a weak one for these two. Pinned by
# test_degenerate_fixture_features_are_exactly_the_known_two so the set cannot
# grow silently. The fix, if it ever needs one, is a fourth sample dataset
# carrying route designations and generic names -- that covers both.
DEGENERATE_ON_FIXTURE = ["route_prefix_match", "name_is_generic"]


def _repo_root() -> Path:
    """Repo root, resolved from this file rather than the CWD."""
    return Path(__file__).resolve().parent.parent


# =============================================================================
# The demo: everything below computes from a bare geometry + name pair
# =============================================================================


def compute_spark_addable_features(
    ref_geom,
    target_geom,
    ref_names=None,
    target_names=None,
) -> dict[str, float]:
    """Compute the 17 Spark-addable features from a candidate pair alone.

    This is the reference implementation a Spark scorer would port. Inputs are
    exactly what ``MatchLayerToNetworkV2`` already has per candidate row:

    * ``ref_geom`` / ``target_geom`` -- LineStrings in a projected CRS (meters),
      the same pair the shipped 28 features are computed from.
    * ``ref_names`` / ``target_names`` -- the Overture-format name structs the
      shipped ``name_levenshtein`` already needs.

    No spatial index, no connector graph, no topology dict, no neighbour lookup.
    ``linestring_alignment`` is a pure pairwise function, and the aligned
    sublines it produces are already required by the shipped ``sinuosity_ref`` /
    ``min_length_m`` / ``ref_coverage``, so the marginal cost here is only the
    per-geometry passes, not the alignment.

    Returns:
        Dict keyed by feature name, matching ``compute_pair_features()`` exactly.
    """
    alignment = linestring_alignment(ref_geom, target_geom)

    # Aligned-portion selection, identical to compute_pair_features().
    ref_coverage = alignment.overture_end_frac - alignment.overture_start_frac
    target_coverage = alignment.dataset_end_frac - alignment.dataset_start_frac
    if ref_coverage >= HIGH_COVERAGE_THRESHOLD:
        ref_aligned = ref_geom
    else:
        ref_aligned = (
            create_subline(ref_geom, alignment.overture_start_frac, alignment.overture_end_frac)
            or ref_geom
        )
    if target_coverage >= HIGH_COVERAGE_THRESHOLD:
        target_aligned = target_geom
    else:
        target_aligned = (
            create_subline(target_geom, alignment.dataset_start_frac, alignment.dataset_end_frac)
            or target_geom
        )

    # Extract coords once and thread them through every per-geometry call -- the
    # repo's perf rule (no repeated np.array(line.coords), no per-element GEOS
    # round-trips). Every helper below is either vectorized numpy or numba JIT.
    coords_ref = np.array(ref_aligned.coords)
    coords_target = np.array(target_aligned.coords)

    # --- Name features: pure string ops on the resolved variant pair ---------
    name_ref, name_target = resolve_best_name_variant(ref_names, target_names)
    name_sim = compute_name_similarity(name_ref, name_target)

    # --- Geometry features: one pass per aligned geometry -------------------
    coverage = compute_coverage_features(alignment)

    sinuosity_ref = compute_sinuosity(ref_aligned, coords=coords_ref)
    sinuosity_target = compute_sinuosity(target_aligned, coords=coords_target)

    heading_ref = compute_heading_consistency(ref_aligned)
    heading_target = compute_heading_consistency(target_aligned)

    complexity_ref = compute_shape_complexity(ref_aligned, coords=coords_ref)
    complexity_target = compute_shape_complexity(target_aligned, coords=coords_target)

    density_ref = compute_vertex_density(ref_aligned, coords=coords_ref)
    density_target = compute_vertex_density(target_aligned, coords=coords_target)
    if math.isnan(density_ref) or math.isnan(density_target):
        density_ratio = float("nan")
    elif density_ref > 0 and density_target > 0:
        density_ratio = min(density_ref, density_target) / max(density_ref, density_target)
    else:
        density_ratio = 0.0

    return {
        # Name Similarity (7)
        "name_jaro_winkler": name_sim["jaro_winkler"],
        "name_soundex": name_sim["soundex_match"],
        "name_metaphone": name_sim["metaphone_similarity"],
        "has_name_ref": name_sim["has_name_ref"],
        "has_name_target": name_sim["has_name_target"],
        "name_is_generic": name_sim["name_is_generic"],
        "route_prefix_match": compute_route_prefix_match(name_ref, name_target),
        # Free derivations of columns the 28-feature model already carries (2)
        "max_coverage": coverage["max_coverage"],
        "sinuosity_delta": abs(sinuosity_ref - sinuosity_target),
        # One extra pass over the aligned geometry pair (8)
        "angle_histogram_similarity": compute_angle_histogram_similarity(
            ref_aligned, target_aligned, coords_a=coords_ref, coords_b=coords_target
        ),
        "shape_complexity_ref": complexity_ref,
        "shape_complexity_delta": abs(complexity_ref - complexity_target),
        "heading_consistency_ref": heading_ref,
        "heading_consistency_delta": abs(heading_ref - heading_target),
        "vertex_density_ref": density_ref,
        "vertex_density_target": density_target,
        "vertex_density_ratio": density_ratio,
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def sample_pairs():
    """Real labelled pairs (projected to UTM) from the committed label data.

    ``labels/data`` stores the geometries captured at labelling time in
    EPSG:4326 -- the same per-row payload a Spark candidate table carries.
    """
    import geopandas as gpd

    from crosswalk.labeling.data_store import DataStore

    pairs = []
    loaded = {}
    for dataset in SAMPLE_DATASETS:
        gdf = DataStore(dataset_id=dataset).gdf
        loaded[dataset] = len(gdf)
        if len(gdf) == 0:
            continue
        gdf = gdf.head(SAMPLE_PAIRS_PER_DATASET)
        utm = gdf.set_geometry("ref_geometry").estimate_utm_crs()
        ref_proj = gpd.GeoSeries(gdf["ref_geometry"], crs="EPSG:4326").to_crs(utm)
        target_proj = gpd.GeoSeries(gdf["target_geometry"], crs="EPSG:4326").to_crs(utm)
        for i in range(len(gdf)):
            ref_geom, target_geom = ref_proj.iloc[i], target_proj.iloc[i]
            if ref_geom is None or target_geom is None or ref_geom.is_empty:
                continue
            # DataStore round-trips the name structs as JSON strings.
            pairs.append(
                {
                    "dataset": dataset,
                    "ref_geom": ref_geom,
                    "target_geom": target_geom,
                    "ref_names": _parse_names(gdf["ref_names"].iloc[i]),
                    "target_names": _parse_names(gdf["target_names"].iloc[i]),
                    "ref_class": gdf["ref_class"].iloc[i],
                    "target_class": gdf["target_class"].iloc[i],
                }
            )

    # DataStore._load swallows every exception and returns an empty frame, so an
    # unmaterialized Git LFS pointer or a corrupt parquet reaches us as "no pairs"
    # rather than an error. Without this assert that silently skips the whole
    # module -- or worse, passes at reduced coverage if only one of the three
    # datasets failed. Fail loudly: every claim in this file rests on the fixture
    # being complete.
    expected = len(SAMPLE_DATASETS) * SAMPLE_PAIRS_PER_DATASET
    assert len(pairs) == expected, (
        f"Fixture is incomplete: {len(pairs)} pairs, expected {expected}. "
        f"Rows loaded per dataset: {loaded}. A zero means "
        "labels/data/dataset=<name>/data.parquet is missing, is an unmaterialized "
        "Git LFS pointer, or failed to parse (DataStore._load swallows the "
        "exception). Run `git lfs pull` from the repo root. Asserted rather than "
        "skipped because a partial fixture still passes every test here while "
        "proving much less."
    )
    return pairs


def _parse_names(value):
    """DataStore persists Overture name structs as JSON text; decode to a dict."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _authoritative_features(pair) -> dict[str, float]:
    """Run the authoritative ``compute_pair_features`` with geometry-only context.

    Every network-context argument is withheld (no graphlet data, no sibling
    context, no connectors, empty endpoint features). Topology is passed as an
    empty dict purely to get past the ``MissingContextError`` guard -- it has no
    influence on any column under test.
    """
    return compute_pair_features(
        ref_geom_full=pair["ref_geom"],
        target_geom_full=pair["target_geom"],
        ref_class=pair["ref_class"],
        target_class=pair["target_class"],
        endpoint_features={},
        ref_topology={},
        target_topology={},
        alignment=linestring_alignment(pair["ref_geom"], pair["target_geom"]),
        ref_names_raw=pair["ref_names"],
        target_names_raw=pair["target_names"],
    )


@pytest.fixture(scope="module")
def authoritative(sample_pairs):
    """``compute_pair_features`` output per sample pair, computed once."""
    return [_authoritative_features(pair) for pair in sample_pairs]


# =============================================================================
# Classification: what the 55 excluded features actually need
# =============================================================================


def test_addable_features_are_currently_excluded():
    """Sanity: the 17 proposed features are real, and none ships today."""
    for feature in ADDABLE_FEATURES:
        assert feature in FEATURE_COLUMNS, f"{feature} is not a declared feature"
    assert sorted(SHIPPED_NAME_FEATURES + SPARSE_NAME_FEATURES) == sorted(ADDABLE_NAME_FEATURES)
    for feature in SHIPPED_NAME_FEATURES:
        assert feature in SPARK_PORTABLE_FEATURES, (
            f"{feature} is part of the name block, which shipped 2026-08-07"
        )
    for feature in SPARSE_NAME_FEATURES + ADDABLE_GEOMETRY_FEATURES:
        assert feature not in SPARK_PORTABLE_FEATURES, (
            f"{feature} is excluded on measured value; adding it needs a "
            "re-measurement, not just a config edit"
        )
    assert len(ADDABLE_FEATURES) == len(set(ADDABLE_FEATURES)) == 17, (
        f"Expected 17 distinct addable features, got {len(ADDABLE_FEATURES)} "
        f"({len(set(ADDABLE_FEATURES))} distinct). If a feature genuinely became "
        "Spark-addable, add it to ADDABLE_NAME_FEATURES or ADDABLE_GEOMETRY_FEATURES, "
        "prove it in test_addable_features_match_authoritative_computation, and update "
        "the counts in research/spark_feature_expansion_2026-08-07.md."
    )


def test_excluded_features_partition_into_three_buckets():
    """Every excluded feature lands in exactly one bucket.

    11 still-addable (10 geometry + route_prefix_match) + 16 network-context +
    22 topology = 49. A new feature landing in FEATURE_COLUMNS without a
    Spark-feasibility verdict fails here, which is the point: the classification
    has to stay exhaustive to be trustworthy. The 6 shipped name features are
    absent because they now ship.
    """
    excluded = [f for f in FEATURE_COLUMNS if f not in SPARK_PORTABLE_FEATURES]
    still_addable = [f for f in ADDABLE_FEATURES if f not in SPARK_PORTABLE_FEATURES]
    buckets = still_addable + CONTEXT_DEPENDENT_FEATURES + list(TOPOLOGY_FEATURES)

    assert len(buckets) == len(set(buckets)), "A feature was classified twice"
    assert sorted(buckets) == sorted(excluded), (
        f"Unclassified: {sorted(set(excluded) - set(buckets))}; "
        f"stale: {sorted(set(buckets) - set(excluded))}"
    )
    counts = (len(still_addable), len(CONTEXT_DEPENDENT_FEATURES), len(TOPOLOGY_FEATURES))
    assert counts == (11, 16, 22), (
        f"Bucket sizes are {counts}, expected (11, 16, 22). The partition assert above "
        "still passed, so every excluded feature is classified -- meaning a feature "
        "MOVED between buckets, which changes its Spark-feasibility verdict. Re-derive "
        "the affected claim in research/spark_feature_expansion_2026-08-07.md before "
        "updating these numbers."
    )


def test_free_derived_features_need_no_new_computation(authoritative):
    """``max_coverage`` / ``sinuosity_delta`` are arithmetic on shipped columns.

    A consumer can add these two without touching its feature code at all --
    they are ``max(ref_coverage, target_coverage)`` and
    ``abs(sinuosity_ref - sinuosity_target)`` over columns already in the
    28-feature vector.
    """
    for feature in FREE_DERIVED_FEATURES:
        assert feature in ADDABLE_GEOMETRY_FEATURES

    for shipped in authoritative:
        assert shipped["max_coverage"] == max(shipped["ref_coverage"], shipped["target_coverage"])
        expected_delta = abs(shipped["sinuosity_ref"] - shipped["sinuosity_target"])
        assert shipped["sinuosity_delta"] == pytest.approx(expected_delta, nan_ok=True)


def test_addable_features_match_authoritative_computation(sample_pairs, authoritative):
    """The geometry-only path reproduces ``compute_pair_features`` exactly.

    This is the load-bearing claim: with nothing but two LineStrings and two
    name structs, all 17 columns come out identical to the value the full
    pipeline stores. Exact equality (not approx) because both paths run the same
    helpers on the same aligned sublines.

    Caveat, pinned by test_degenerate_fixture_features_are_exactly_the_known_two:
    2 of the 17 are constant across this fixture, so for those this asserts
    agreement on a single value rather than across a distribution. A stub
    returning that constant would pass. The claim is strong for 15 and weak for
    ``route_prefix_match`` / ``name_is_generic``.
    """
    mismatches = []
    for pair, expected in zip(sample_pairs, authoritative, strict=True):
        computed = compute_spark_addable_features(
            pair["ref_geom"], pair["target_geom"], pair["ref_names"], pair["target_names"]
        )
        for feature in ADDABLE_FEATURES:
            want, got = expected[feature], computed[feature]
            if isinstance(want, float) and math.isnan(want) and math.isnan(got):
                continue
            if want != got:
                mismatches.append((pair["dataset"], feature, want, got))

    assert not mismatches, f"Geometry-only path diverged on {len(mismatches)}: {mismatches[:5]}"


def test_route_prefix_match_is_almost_always_nan():
    """Pin how sparse ``route_prefix_match`` is -- the reason it is not shipped.

    Measured 2026-08-07: non-NaN on **1 of 5,532** stored labelled pairs (0.02%),
    the single hit being in ``ca_toronto_roads``. It needs both sides to
    canonicalize to a route designation (I-90, US-101, SR-520), which street and
    sidewalk layers do not carry. XGBoost cannot split on a column that is missing
    in 5,531 of 5,532 rows.

    It is also the *only* member of the name block that costs new computation --
    1.18 us/pair, i.e. 100% of the block's marginal cost, since the other 6 are
    dict keys ``compute_name_similarity()`` already returns. Excluding it is
    therefore what takes the widening from "close to free" to **0.00 us/pair**,
    for a measured solo lift of +0.0002 LOO F1 (noise).

    Fails loudly in either direction: if the label base grows enough highway data
    to make the feature real, re-measure and reconsider shipping it; if it silently
    goes 100% NaN, ``canonicalize_route_name()`` probably broke.
    """
    import glob

    import pandas as pd

    paths = glob.glob("labels/features/dataset=*/data.parquet")
    if not paths:
        pytest.skip("No feature store found — run from repo root")

    total = non_nan = 0
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=["route_prefix_match"])
        except Exception:  # partition predates the feature
            continue
        total += len(df)
        non_nan += int(df["route_prefix_match"].notna().sum())

    assert total > 1000, f"Only {total} rows found — feature store looks truncated"
    rate = non_nan / total
    assert 0 < rate < 0.01, (
        f"route_prefix_match non-NaN rate is {rate:.4%} ({non_nan}/{total}), outside the "
        "expected 'present but vanishingly rare' band. If it rose, the feature may now be "
        "worth its 1.18 us/pair and the name block should be re-measured. If it hit zero, "
        "check canonicalize_route_name()."
    )


def test_degenerate_fixture_features_are_exactly_the_known_two(authoritative):
    """Pin which addable features are constant across the fixture.

    The parity test compares all 17 columns, but a column taking the same value on
    all 180 pairs cannot distinguish a correct implementation from a stub
    returning that constant -- returning ``float("nan")`` for
    ``route_prefix_match``, or ``0.0`` for ``name_is_generic``, passes it. So
    "proven bit-for-bit" is a strong claim for 15 of the 17 and a weak one for
    these two, and that distinction belongs in the codebase rather than in a
    reviewer's head.

    Fails if the degenerate set grows (a new feature became untestable here) or
    shrinks (the fixture now exercises one -- upgrade the claim).
    """
    constant = []
    for feature in ADDABLE_FEATURES:
        values = [f[feature] for f in authoritative]
        first = values[0]
        first_nan = isinstance(first, float) and math.isnan(first)
        if all(
            (first_nan and isinstance(v, float) and math.isnan(v)) or (not first_nan and v == first)
            for v in values
        ):
            constant.append(feature)

    assert sorted(constant) == sorted(DEGENERATE_ON_FIXTURE), (
        f"Constant-on-fixture features changed: {sorted(constant)}, expected "
        f"{sorted(DEGENERATE_ON_FIXTURE)}. GROWN => the parity proof covers fewer "
        "features than claimed; add a sample dataset exercising the new one. "
        "SHRUNK => the fixture now varies it, so drop it from DEGENERATE_ON_FIXTURE "
        "and strengthen the claim in the module docstring."
    )


def test_context_dependent_features_nan_without_network_context(authoritative):
    """Withholding the spatial index / connector graph NaNs the other features.

    Complement to the test above, but NOT the same class of evidence, and the
    difference matters. ``compute_pair_features`` branches on ``is not None`` for
    graphlet data and sibling context and falls back to NaN defaults, so this is
    substantially true by construction: it asserts the function returns its
    documented no-context default when given no context.

    What it is good for is the one-way direction -- if a feature listed here ever
    starts computing without network context, it was misclassified and belongs in
    the addable set. That is worth guarding. What it is NOT is proof of
    infeasibility on a par with the parity test above, which actually recomputes
    17 columns from a bare pair. Do not cite it as such.
    """
    checked = CONTEXT_DEPENDENT_FEATURES + list(TOPOLOGY_FEATURES)
    always_nan = dict.fromkeys(checked, True)
    for features in authoritative:
        for feature in checked:
            value = features[feature]
            if not (isinstance(value, float) and math.isnan(value)):
                always_nan[feature] = False

    computable = [name for name, is_nan in always_nan.items() if not is_nan]
    assert not computable, (
        f"These were assumed to need network context but computed anyway: {computable}. "
        "Re-check whether they belong in the addable set."
    )


def test_offset_over_expected_halfwidth_ships_despite_sibling_category(authoritative):
    """Guard against over-reading the category names.

    ``offset_over_expected_halfwidth`` sits in the Parallel Sibling category but
    is ``lateral_offset / class-expected half width`` -- no sibling search. It
    already ships, and this pins why the other four Parallel Sibling features do
    not follow it.
    """
    assert "offset_over_expected_halfwidth" in SPARK_PORTABLE_FEATURES
    for features in authoritative:
        assert not math.isnan(features["offset_over_expected_halfwidth"])


def test_shipped_28_are_also_context_free(authoritative):
    """Baseline check on the other side of the line.

    None of the 28 already-shipping features secretly depends on withheld
    context either -- every one produces a real value on at least some pairs
    with no index, graph, or topology available. (Individual NaNs are legitimate
    data signals: an unnamed segment, a name with no digits, an alignment that
    reaches a geometry end.) Without this, the NaN test above would pass just as
    happily if ``compute_pair_features`` were broken outright.
    """
    computed = dict.fromkeys(SPARK_PORTABLE_FEATURES, False)
    for features in authoritative:
        for feature in SPARK_PORTABLE_FEATURES:
            value = features[feature]
            if not (isinstance(value, float) and math.isnan(value)):
                computed[feature] = True

    never_computed = [name for name, ok in computed.items() if not ok]
    assert not never_computed, (
        f"Shipped Spark features that never produced a value without network "
        f"context: {never_computed}"
    )


# =============================================================================
# The research harness reproduces eval_utils
# =============================================================================


def _load_research_module():
    """Import ``research/spark_feature_expansion.py`` by path.

    ``research/`` is not a package and is not on ``pythonpath``, so the sweep
    script is loaded from its file location rather than imported by name.
    """
    import importlib.util

    path = _repo_root() / "research" / "spark_feature_expansion.py"
    if not path.exists():
        pytest.skip(f"{path} not found — run from repo root")
    spec = importlib.util.spec_from_file_location("spark_feature_expansion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_loo_harness_reproduces_eval_utils_on_full_feature_set():
    """``spark_feature_expansion.loo_f1`` must equal ``run_loo_by_type_cv``.

    Every LOO number in the research writeup comes from ``loo_f1``, which
    reimplements the fold construction rather than calling ``run_loo_by_type_cv``
    (that function has no feature-subset knob). A reimplementation is only worth
    as much as its parity check: if it drifts from ``eval_utils`` the whole tier
    table silently stops meaning what it says.

    The input frame is deliberately doctored to exercise branches real data does
    not. Verified by mutation 2026-08-07 -- each of these, applied to ``loo_f1``,
    fails this test:

    * fold composition (train on same type_group only)
    * per-fold ``scale_pos_weight`` (pin to 1.0)
    * ``METRIC_AVERAGE`` ("binary" -> "macro")
    * column selection (truncate ``cols``)
    * the ``xgb_params is None`` check (-> truthiness, so ``{}`` would silently
      fall through to SPARK_PORTABLE_XGB_PARAMS)
    * the ``MIN_LOO_LABELS`` filter -- guarded by ``truncated`` below
    * the dedup call -- guarded by ``dup`` below

    The last two need that help: all 33 real datasets have zero duplicate pairs,
    and the smallest still hold 29+ labels, far above MIN_LOO_LABELS. An earlier
    version of this docstring claimed to pin them and did not. Note the dedup
    guard catches *removing* the dedup (row count diverges); it does not pin
    ``keep="last"`` vs ``keep="first"``, which would need the flipped label to
    move a fold metric and is not guaranteed on one row in fifty.

    Trained with a deliberately tiny booster on both sides -- parity is about fold
    construction and metric computation, not model quality, and the full-size
    default made this the second-most expensive test in the repo under ``-n auto``.
    """
    import pandas as pd

    from crosswalk.config import FEATURE_COLUMNS
    from crosswalk.eval_utils import MIN_LOO_LABELS, run_loo_by_type_cv
    from crosswalk.labeling.label_store import LabelStore

    labels_dir = _repo_root() / "labels"
    if not labels_dir.exists():
        pytest.skip(f"Labels directory not found at {labels_dir}")

    module = _load_research_module()

    labels = LabelStore.load_all(labels_dir)
    labels = labels[labels["label"].isin({"match", "no_match"})]
    counts = labels.groupby("dataset").size()
    eligible = counts[counts >= MIN_LOO_LABELS].sort_values()
    if len(eligible) < 5:
        pytest.skip(f"Only {len(eligible)} datasets clear MIN_LOO_LABELS — nothing to compare")

    subset = labels[labels["dataset"].isin(list(eligible.index[:4]))].copy()

    # Guard the MIN_LOO_LABELS filter: a real dataset cut below the threshold.
    # Both harnesses must drop it, so it must NOT appear as a fold. Reuse an
    # existing dataset name so build_type_groups can still classify it.
    truncated_name = eligible.index[4]
    truncated = labels[labels["dataset"] == truncated_name].head(MIN_LOO_LABELS - 1)
    assert len(truncated) == MIN_LOO_LABELS - 1

    # Guard the dedup: one exact (gers_id, target_id, dataset) repeat with the
    # label flipped, so dropping the dedup changes that fold's row count.
    dup = subset.iloc[[0]].copy()
    dup["label"] = "no_match" if dup.iloc[0]["label"] == "match" else "match"

    subset = pd.concat([subset, truncated, dup], ignore_index=True)

    # Tiny booster, identical on both sides. Both harnesses merge these over
    # DEFAULT_XGB_PARAMS, so the comparison stays apples-to-apples.
    cheap = {"n_estimators": 24, "max_depth": 3}

    mine = module.loo_f1(subset, list(FEATURE_COLUMNS), seed=42, xgb_params=cheap, n_jobs=2)
    theirs = run_loo_by_type_cv(labels=subset, seed=42, xgb_params=cheap).to_frame()

    assert mine["loo_n_folds"] == len(theirs), (
        f"Fold count diverged: harness ran {mine['loo_n_folds']}, eval_utils ran {len(theirs)}"
    )
    evaluated = {r["dataset"] for r in mine["loo_rows"]}
    assert evaluated == set(theirs["dataset"]), (
        "Harness and eval_utils evaluated different datasets"
    )
    assert truncated_name not in evaluated, (
        f"{truncated_name} was cut to {MIN_LOO_LABELS - 1} labels and must be filtered "
        "by MIN_LOO_LABELS, but it was evaluated as a fold"
    )

    reference = theirs.set_index("dataset")
    for row in mine["loo_rows"]:
        expected = reference.loc[row["dataset"]]
        assert row["type_group"] == expected["type_group"], row["dataset"]
        assert row["n_test"] == expected["n_test"], row["dataset"]
        for metric in ("f1", "precision", "recall"):
            assert row[metric] == pytest.approx(float(expected[metric]), abs=1e-12), (
                f"{row['dataset']}.{metric}: harness {row[metric]} vs eval_utils {expected[metric]}"
            )


# =============================================================================
# Tradeoff measurement: marginal feature-computation cost
# =============================================================================


def _median_us_per_pair(fn, pairs, repeats: int) -> float:
    """Median wall time per pair (microseconds) over ``repeats`` sweeps."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for pair in pairs:
            fn(pair)
        timings.append((time.perf_counter() - start) / len(pairs) * 1e6)
    return float(np.median(timings))


def _prepare(pair):
    """The alignment + aligned sublines + coords the shipped 28 already build."""
    ref_geom, target_geom = pair["ref_geom"], pair["target_geom"]
    alignment = linestring_alignment(ref_geom, target_geom)
    ref_cov = alignment.overture_end_frac - alignment.overture_start_frac
    target_cov = alignment.dataset_end_frac - alignment.dataset_start_frac
    ref_aligned = (
        ref_geom
        if ref_cov >= HIGH_COVERAGE_THRESHOLD
        else (
            create_subline(ref_geom, alignment.overture_start_frac, alignment.overture_end_frac)
            or ref_geom
        )
    )
    target_aligned = (
        target_geom
        if target_cov >= HIGH_COVERAGE_THRESHOLD
        else (
            create_subline(target_geom, alignment.dataset_start_frac, alignment.dataset_end_frac)
            or target_geom
        )
    )
    return (
        ref_aligned,
        target_aligned,
        np.array(ref_aligned.coords),
        np.array(target_aligned.coords),
    )


@pytest.mark.slow
def test_addable_feature_marginal_cost(sample_pairs):
    """Measure the *marginal* per-pair cost of the 17 extra features.

    Only genuinely new work counts. The alignment, the aligned sublines, and the
    coordinate extraction are already paid by the shipped 28 (``ref_coverage``,
    ``min_length_m``, ``sinuosity_ref`` all need them), as are
    ``compute_name_similarity`` (which computes all 10 name metrics and discards
    7) and the target-side ``sinuosity`` / ``heading_consistency`` /
    ``shape_complexity``. So the marginal cost is:

    * names: one extra ``compute_route_prefix_match`` call -- the other 6 name
      features are already-computed dict entries the exporter throws away.
    * geometry: 5 new calls (ref-side heading consistency / shape complexity,
      both vertex densities, the pairwise angle histogram). ``max_coverage``,
      ``sinuosity_delta``, ``shape_complexity_delta``,
      ``heading_consistency_delta`` and ``vertex_density_ratio`` are arithmetic
      on values already in hand and cost nothing measurable.

    Asserts only a generous ceiling; the printed table is the deliverable.
    """
    repeats = 7
    prepared = [_prepare(pair) for pair in sample_pairs]
    names = [resolve_best_name_variant(p["ref_names"], p["target_names"]) for p in sample_pairs]

    def prepare_only(pair):
        return _prepare(pair)

    def baseline_names(name_pair):
        return compute_name_similarity(*name_pair)

    def added_names(name_pair):
        return compute_route_prefix_match(*name_pair)

    def baseline_geometry(prep):
        ref_aligned, target_aligned, coords_ref, coords_target = prep
        compute_sinuosity(ref_aligned, coords=coords_ref)
        compute_sinuosity(target_aligned, coords=coords_target)
        compute_heading_consistency(target_aligned)
        compute_shape_complexity(target_aligned, coords=coords_target)

    def added_geometry(prep):
        ref_aligned, target_aligned, coords_ref, coords_target = prep
        compute_heading_consistency(ref_aligned)
        compute_shape_complexity(ref_aligned, coords=coords_ref)
        compute_vertex_density(ref_aligned, coords=coords_ref)
        compute_vertex_density(target_aligned, coords=coords_target)
        compute_angle_histogram_similarity(
            ref_aligned, target_aligned, coords_a=coords_ref, coords_b=coords_target
        )

    # Warm the numba JIT so compilation does not land in the first timing sweep.
    compute_spark_addable_features(
        sample_pairs[0]["ref_geom"],
        sample_pairs[0]["target_geom"],
        sample_pairs[0]["ref_names"],
        sample_pairs[0]["target_names"],
    )

    prepare_us = _median_us_per_pair(prepare_only, sample_pairs, repeats)
    baseline_name_us = _median_us_per_pair(baseline_names, names, repeats)
    baseline_geom_us = _median_us_per_pair(baseline_geometry, prepared, repeats)
    added_name_us = _median_us_per_pair(added_names, names, repeats)
    added_geom_us = _median_us_per_pair(added_geometry, prepared, repeats)

    already_paid = prepare_us + baseline_name_us + baseline_geom_us
    added = added_name_us + added_geom_us
    print(
        f"\nn_pairs={len(sample_pairs)} repeats={repeats}\n"
        f"  align + subline + coords (already paid) : {prepare_us:8.2f} us/pair\n"
        f"  name similarity call     (already paid) : {baseline_name_us:8.2f} us/pair\n"
        f"  target-side geometry     (already paid) : {baseline_geom_us:8.2f} us/pair\n"
        f"  = shipped-28 per-pair subtotal          : {already_paid:8.2f} us/pair\n"
        f"  + route_prefix_match     (7 name feats) : {added_name_us:8.2f} us/pair\n"
        f"  + 5 new geometry calls  (10 geom feats) : {added_geom_us:8.2f} us/pair\n"
        f"  = marginal cost of all 17               : {added:8.2f} us/pair "
        f"({added / already_paid * 100:.1f}% of subtotal)"
    )

    assert added < already_paid, (
        "Added feature computation is no longer cheap relative to the per-pair "
        f"work the 28-feature model already does: +{added:.1f}us vs {already_paid:.1f}us"
    )


# NOTE: a `test_tier_model_sizes_and_inference` used to live here, training three
# feature tiers and printing an F1/size/latency table. Removed 2026-08-07: it was
# the most expensive test in the repo under the default `-n auto` (three
# concurrent XGBoost trainings at n_jobs=-1 oversubscribe OpenMP across xdist
# workers -- this file went 6s -> ~10min with it present), and its only assertions
# were `n_features == 45`, already implied by the partition test, and
# `model_kb < 4 * baseline_kb` against a measured ratio of 1.03: ~290% slack, so it
# could not fire short of a catastrophic regression. The table it printed is the
# real deliverable and `research/spark_feature_expansion.py` already produces it
# with more seeds and LOO CV. Cost without detection is not a test.
