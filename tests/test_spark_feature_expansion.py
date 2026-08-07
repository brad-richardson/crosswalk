"""Demo + guard for the boundary of SPARK_PORTABLE_FEATURES.

`crosswalk export-spark-model` ships a subset of the 83 FEATURE_COLUMNS, on the
stated rationale that they are "the subset computable from aligned geometry pairs
(no topology, graph, or spatial-index features required)". That rationale was
over-broad: 17 then-excluded features also need nothing but the two aligned
geometries and the two name structs the Spark job already holds to compute
`name_levenshtein` / `sinuosity_ref` / `ref_coverage`.

6 of those 17 (the name block minus the near-always-NaN `route_prefix_match`)
shipped on 2026-08-07 and the set is now 34. The other 11 remain excluded on
measured value, not feasibility.
This module keeps proving the feasibility of all 17 regardless of which are
currently shipped -- that is what makes the exclusion an argument about value.

This module *proves* that by computation rather than by reading:
:func:`compute_spark_addable_features` takes only ``(ref_geom, target_geom,
ref_names, target_names)`` -- no STRtree, no connector graph, no topology dicts,
no native-target degrees -- and reproduces the authoritative
``compute_pair_features()`` output for all 17 columns bit-for-bit on real
labelled pairs. The same fixture shows the genuinely context-dependent columns
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
# "1.18 us/pair" to literally free. See test_route_prefix_match_is_almost_always_nan.
SPARSE_NAME_FEATURES = ["route_prefix_match"]
SHIPPED_NAME_FEATURES = [f for f in ADDABLE_NAME_FEATURES if f not in SPARSE_NAME_FEATURES]

# Derivable from columns the shipped Spark model already carries, with no
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
      the same pair the shipped Spark features are computed from.
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
        # Free derivations of columns the shipped model already carries (2)
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
    for dataset in SAMPLE_DATASETS:
        gdf = DataStore(dataset_id=dataset).gdf
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

    if not pairs:
        pytest.skip("No stored pair data under labels/data — run from repo root")
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


def test_addable_features_are_declared_and_correctly_split():
    """The 17 feasible features are real, and the shipped/excluded split is the
    decision recorded in the research doc.

    6 of the 7 name features ship (2026-08-07, 0.00 us/pair -- they are dict keys
    ``compute_name_similarity()`` already returns). ``route_prefix_match`` and the
    10 geometry features do not. All 17 stay listed here because the classification
    is about *feasibility*, which did not change -- only the value verdict did.
    """
    for feature in ADDABLE_FEATURES:
        assert feature in FEATURE_COLUMNS, f"{feature} is not a declared feature"
    assert len(ADDABLE_FEATURES) == len(set(ADDABLE_FEATURES)) == 17
    assert sorted(SHIPPED_NAME_FEATURES + SPARSE_NAME_FEATURES) == sorted(ADDABLE_NAME_FEATURES)

    for feature in SHIPPED_NAME_FEATURES:
        assert feature in SPARK_PORTABLE_FEATURES, (
            f"{feature} is part of the name block, which shipped 2026-08-07"
        )
    for feature in SPARSE_NAME_FEATURES:
        assert feature not in SPARK_PORTABLE_FEATURES, (
            f"{feature} is excluded for near-total NaN, not infeasibility -- "
            "keeping it out is what makes the name block cost 0.00 us/pair. "
            "See test_route_prefix_match_is_almost_always_nan before adding it."
        )
    for feature in ADDABLE_GEOMETRY_FEATURES:
        assert feature not in SPARK_PORTABLE_FEATURES, (
            f"{feature} is in the geometry block, excluded on measured value "
            "(-0.0034 LOO F1). Adding it needs a re-measurement, not just a config edit."
        )


def test_spark_portable_features_follow_feature_columns_order():
    """``SPARK_PORTABLE_FEATURES`` must be a FEATURE_COLUMNS-ordered subsequence.

    Subtle and easy to break. The exporter passes this list to ``MLMatcher`` as an
    *exclusion* set, so ``_extract_from_columns`` rebuilds ``feature_names`` in
    ``FEATURE_COLUMNS`` order and the model itself is indifferent to how this list
    is ordered. But ``build_spark_model_manifest`` writes the manifest from
    ``feature_names``, and ``tests/unit/test_shipped_spark_model.py`` compares
    ``manifest["features"] == SPARK_PORTABLE_FEATURES`` as an **ordered list**.

    So appending a new feature to the end of this list -- the obvious move, and
    what an earlier draft of the research doc explicitly recommended -- breaks the
    shipped-manifest test with a diff that looks like a re-export problem rather
    than an ordering one. Fail here instead, with the reason.
    """
    expected = [f for f in FEATURE_COLUMNS if f in set(SPARK_PORTABLE_FEATURES)]
    assert list(SPARK_PORTABLE_FEATURES) == expected, (
        "SPARK_PORTABLE_FEATURES is not in FEATURE_COLUMNS order. Reorder it to "
        f"match (insert in place, do not append):\n{expected}"
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
    assert (len(still_addable), len(CONTEXT_DEPENDENT_FEATURES), len(TOPOLOGY_FEATURES)) == (
        11,
        16,
        22,
    )
    assert sorted(still_addable) == sorted(ADDABLE_GEOMETRY_FEATURES + SPARSE_NAME_FEATURES)


def test_free_derived_features_need_no_new_computation(authoritative):
    """``max_coverage`` / ``sinuosity_delta`` are arithmetic on shipped columns.

    A consumer can add these two without touching its feature code at all --
    they are ``max(ref_coverage, target_coverage)`` and
    ``abs(sinuosity_ref - sinuosity_target)`` over columns already in the
    shipped Spark feature vector.
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


def test_context_dependent_features_nan_without_network_context(authoritative):
    """Withholding the spatial index / connector graph NaNs the other features.

    Complement to the test above: it is not that everything is computable
    per-pair. These 16 columns have no value at all without a network-wide
    structure, so they cannot follow the 17 into a Spark UDF however useful they
    are locally.
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


def test_shipped_spark_features_are_also_context_free(authoritative):
    """Baseline check on the other side of the line.

    None of the already-shipping Spark features secretly depends on withheld
    context either -- every one produces a real value on at least some pairs
    with no index, graph, or topology available. (Individual NaNs are legitimate
    data signals: an unnamed segment, a name with no digits, an alignment that
    reaches a geometry end.) Without this, the NaN test above would pass just as
    happily if ``compute_pair_features`` were broken outright.

    No exemptions: every shipped feature must produce a real value somewhere in
    the fixture. ``route_prefix_match`` is the one feasible name feature that
    cannot clear this bar (NaN on all 180 pairs, and on all but 1 of the 5,532
    stored labels), which is part of why it is not shipped.
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


# =============================================================================
# The research harness reproduces eval_utils
# =============================================================================


def _load_research_module():
    """Import ``research/spark_feature_expansion.py`` by path.

    ``research/`` is not a package and is not on ``pythonpath``, so the sweep
    script is loaded from its file location rather than imported by name.
    """
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "research" / "spark_feature_expansion.py"
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
    as much as its parity check: if it drifts from ``eval_utils`` -- fold
    composition, the MIN_LOO_LABELS filter, dedup, per-fold ``scale_pos_weight``,
    the METRIC_AVERAGE choice -- the whole tier table silently stops meaning what
    it says.

    Run on the smallest eligible datasets so this stays a parity check rather
    than a second full sweep; the code under test does not branch on dataset
    size, and both harnesses see the identical pre-filtered frame.
    """
    from crosswalk.config import FEATURE_COLUMNS
    from crosswalk.eval_utils import MIN_LOO_LABELS, run_loo_by_type_cv
    from crosswalk.labeling.label_store import LabelStore

    labels_dir = Path("labels")
    if not labels_dir.exists():
        pytest.skip("Labels directory not found — run from repo root")

    module = _load_research_module()

    labels = LabelStore.load_all(labels_dir)
    labels = labels[labels["label"].isin({"match", "no_match"})]
    counts = labels.groupby("dataset").size()
    eligible = counts[counts >= MIN_LOO_LABELS].sort_values()
    if len(eligible) < 3:
        pytest.skip(f"Only {len(eligible)} datasets clear MIN_LOO_LABELS — nothing to compare")
    subset = labels[labels["dataset"].isin(eligible.index[:6])].copy()

    # xgb_params={} pins DEFAULT_XGB_PARAMS on both sides; loo_f1 would otherwise
    # default to SPARK_PORTABLE_XGB_PARAMS and the comparison would be vacuous.
    mine = module.loo_f1(subset, list(FEATURE_COLUMNS), seed=42, xgb_params={})
    theirs = run_loo_by_type_cv(labels=subset, seed=42).to_frame()

    assert mine["loo_n_folds"] == len(theirs), (
        f"Fold count diverged: harness ran {mine['loo_n_folds']}, eval_utils ran {len(theirs)}"
    )

    reference = theirs.set_index("dataset")
    assert {r["dataset"] for r in mine["loo_rows"]} == set(reference.index), (
        "Harness and eval_utils evaluated different datasets"
    )
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
    """The alignment + aligned sublines + coords the shipped set already builds."""
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
    """Measure the *marginal* per-pair cost of the 17 feasible features.

    Only genuinely new work counts. The alignment, the aligned sublines, and the
    coordinate extraction are already paid by any Spark feature set worth the name
    (``ref_coverage``, ``min_length_m``, ``sinuosity_ref`` all need them), as are
    ``compute_name_similarity`` (which computes 9 of the 10 name metrics in one
    call) and the target-side ``sinuosity`` / ``heading_consistency`` /
    ``shape_complexity``.

    The ``route_prefix_match`` row is what the *name block* cost to add, measured
    against the pre-2026-08-07 shipped set of 28. It is now itself shipped, so it
    has moved from the "added" side of the ledger to the "already paid" side; the
    row is kept because it is the number that justified the widening and the split
    is what makes the geometry comparison legible.

    The live question is the geometry block: 5 new calls (ref-side heading
    consistency / shape complexity, both vertex densities, the pairwise angle
    histogram) for 10 features. ``max_coverage``, ``sinuosity_delta``,
    ``shape_complexity_delta``, ``heading_consistency_delta`` and
    ``vertex_density_ratio`` are arithmetic on values already in hand and cost
    nothing measurable.

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
        f"  = per-pair subtotal                     : {already_paid:8.2f} us/pair\n"
        f"  + route_prefix_match  (7 name, SHIPPED) : {added_name_us:8.2f} us/pair\n"
        f"  + 5 new geometry calls (10 geom, OUT)   : {added_geom_us:8.2f} us/pair\n"
        f"  = marginal cost of all 17               : {added:8.2f} us/pair "
        f"({added / already_paid * 100:.1f}% of subtotal)"
    )

    assert added < already_paid, (
        "Added feature computation is no longer cheap relative to the per-pair "
        f"work the shipped Spark model already does: +{added:.1f}us vs {already_paid:.1f}us"
    )


@pytest.mark.slow
def test_tier_model_sizes_and_inference(tmp_path):
    """Train three tiers; report holdout F1, artifact size, and inference cost.

    A compact version of ``research/spark_feature_expansion.py`` (single seed,
    no LOO CV) so the size/latency side of the tradeoff is reproducible from the
    test suite. The two widened tiers bracket the decision space: the smallest
    the shipped 34 and the 44 that adding the geometry block back would produce.

    Asserts only that widening the feature set does not blow up the artifact the
    Spark job has to ship -- the F1 verdict needs LOO CV over 5 seeds, which is
    what the research script is for. Nothing here endorses a tier.
    """
    import xgboost as xgb

    from crosswalk.config import SPARK_PORTABLE_XGB_PARAMS
    from crosswalk.matching.ml import MLMatcher

    labels_dir = Path("labels")
    if not labels_dir.exists():
        pytest.skip("Labels directory not found — run from repo root")

    tiers = {
        "shipped_34": list(SPARK_PORTABLE_FEATURES),
        "plus_geometry_44": list(SPARK_PORTABLE_FEATURES) + ADDABLE_GEOMETRY_FEATURES,
    }

    report = {}
    for name, features in tiers.items():
        keep = set(features)
        matcher = MLMatcher()
        metrics = matcher.train(
            labels_dir=labels_dir,
            test_size=0.2,
            binary=True,
            exclude_features=[f for f in FEATURE_COLUMNS if f not in keep],
            seed=42,
            **SPARK_PORTABLE_XGB_PARAMS,
        )
        model_path = tmp_path / f"{name}.json"
        matcher.model.get_booster().save_model(str(model_path))

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        batch = np.random.default_rng(0).random((50_000, len(features)), dtype=np.float32)
        booster.predict(xgb.DMatrix(batch[:1024]))  # warm up
        start = time.perf_counter()
        booster.predict(xgb.DMatrix(batch))
        elapsed = time.perf_counter() - start

        report[name] = {
            "n_features": len(matcher.feature_names),
            "cv_f1": metrics["cv_f1_mean"],
            "test_f1": metrics["test_f1_production"],
            "model_kb": model_path.stat().st_size / 1024,
            "us_per_row": elapsed / len(batch) * 1e6,
        }

    print("")
    for name, row in report.items():
        print(
            f"  {name:16s} n={row['n_features']:3d} cv_f1={row['cv_f1']:.4f} "
            f"test_f1={row['test_f1']:.4f} model={row['model_kb']:7.1f}KB "
            f"infer={row['us_per_row']:.2f}us/row"
        )

    assert report["plus_geometry_44"]["n_features"] == 44
    assert report["plus_geometry_44"]["model_kb"] < 4 * report["shipped_34"]["model_kb"]
