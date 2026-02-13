"""Sweep tests verifying features respond correctly to controlled degradation.

Each test creates a series of geometry pairs with one variable changing monotonically,
then asserts the corresponding features respond in the expected direction.

Features NOT covered by sweep tests (26 of 67) — these depend on external context
(topology graph, alignment data, graphlet computation, clustering, or spatial queries)
that cannot be constructed from simple geometry pairs:

  Topology (12): from_degree_ref, to_degree_ref, from_degree_target, to_degree_target,
    degree_match_score, degree_signature_similarity, is_dead_end_ref, is_dead_end_target,
    dead_end_match, is_intersection_ref, is_intersection_target, intersection_match

  Alignment Coverage (4): ref_coverage, target_coverage, min_coverage, coverage_ratio

  Graphlet (2): graphlet_similarity, endpoint_degree_similarity

  Clustering (3): clustering_coef_ref, clustering_coef_target, clustering_coef_delta

  Parallel Sibling (5): has_parallel_sibling_ref, parallel_fraction_ref,
    offset_vs_half_corridor_ratio, offset_over_expected_halfwidth,
    likely_representation_mismatch

These features are validated by bounds tests in conftest.py FEATURE_BOUNDS instead.
"""

import math

import numpy as np
import pytest

from .conftest import compute_features_simple, make_projected_line


def _feature_series(pairs, feature_name):
    """Compute a feature across a series of geometry pairs."""
    values = []
    for ref, target in pairs:
        feats = compute_features_simple(ref, target)
        values.append(feats[feature_name])
    return values


def _feature_series_with_names(pairs, feature_name, names, classes=None):
    """Compute a feature across geometry pairs with varying names/classes.

    Args:
        pairs: List of (ref, target) LineString pairs
        feature_name: Feature to extract
        names: List of (ref_name, target_name) tuples
        classes: Optional list of (ref_class, target_class) tuples
    """
    values = []
    for i, (ref, target) in enumerate(pairs):
        ref_name, target_name = names[i]
        ref_class = classes[i][0] if classes else "residential"
        target_class = classes[i][1] if classes else "residential"
        feats = compute_features_simple(
            ref,
            target,
            ref_name=ref_name,
            target_name=target_name,
            ref_class=ref_class,
            target_class=target_class,
        )
        values.append(feats[feature_name])
    return values


def _assert_monotonic_increasing(values, feature_name, tolerance=0.0):
    """Assert values are non-decreasing (allowing tolerance for floating point)."""
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1] - tolerance, (
            f"{feature_name} not monotonically increasing: "
            f"values[{i - 1}]={values[i - 1]:.4f} > values[{i}]={values[i]:.4f} "
            f"(series: {[f'{v:.4f}' for v in values]})"
        )


def _assert_monotonic_decreasing(values, feature_name, tolerance=0.0):
    """Assert values are non-increasing (allowing tolerance for floating point)."""
    for i in range(1, len(values)):
        assert values[i] <= values[i - 1] + tolerance, (
            f"{feature_name} not monotonically decreasing: "
            f"values[{i - 1}]={values[i - 1]:.4f} < values[{i}]={values[i]:.4f} "
            f"(series: {[f'{v:.4f}' for v in values]})"
        )


def _assert_first_better_than_last(values, feature_name, higher_is_better=True):
    """Assert first value is better than last (weaker than full monotonicity)."""
    if higher_is_better:
        assert values[0] > values[-1], (
            f"{feature_name}: first ({values[0]:.4f}) should be > last ({values[-1]:.4f})"
        )
    else:
        assert values[0] < values[-1], (
            f"{feature_name}: first ({values[0]:.4f}) should be < last ({values[-1]:.4f})"
        )


def _assert_all_constant(values, feature_name, tolerance=1e-6):
    """Assert all values are the same (within tolerance)."""
    for i in range(1, len(values)):
        assert abs(values[i] - values[0]) <= tolerance, (
            f"{feature_name} not constant: "
            f"values[0]={values[0]:.4f}, values[{i}]={values[i]:.4f} "
            f"(series: {[f'{v:.4f}' for v in values]})"
        )


class TestLateralOffsetSweep:
    """Parallel lines at increasing lateral offsets."""

    @pytest.fixture
    def offset_pairs(self):
        offsets = [0, 2, 5, 10, 20, 50]
        ref = make_projected_line([(0, 0), (100, 0)])
        return [(ref, make_projected_line([(0, d), (100, d)])) for d in offsets]

    def test_lateral_offset_increases(self, offset_pairs):
        values = _feature_series(offset_pairs, "lateral_offset_m")
        _assert_monotonic_increasing(values, "lateral_offset_m", tolerance=0.5)

    def test_buffer_iou_5m_decreases(self, offset_pairs):
        values = _feature_series(offset_pairs, "buffer_iou_5m")
        _assert_first_better_than_last(values, "buffer_iou_5m", higher_is_better=True)

    def test_edge_distance_rmse_increases(self, offset_pairs):
        values = _feature_series(offset_pairs, "edge_distance_rmse_m")
        _assert_monotonic_increasing(values, "edge_distance_rmse_m", tolerance=0.5)

    def test_hausdorff_increases(self, offset_pairs):
        values = _feature_series(offset_pairs, "hausdorff_distance_m")
        _assert_monotonic_increasing(values, "hausdorff_distance_m", tolerance=0.5)

    def test_mean_hausdorff_increases(self, offset_pairs):
        values = _feature_series(offset_pairs, "mean_hausdorff_distance_m")
        _assert_monotonic_increasing(values, "mean_hausdorff_distance_m", tolerance=0.5)

    def test_hausdorff_p95_increases(self, offset_pairs):
        values = _feature_series(offset_pairs, "hausdorff_p95_m")
        _assert_monotonic_increasing(values, "hausdorff_p95_m", tolerance=0.5)

    def test_buffer_iou_15m_decreases(self, offset_pairs):
        values = _feature_series(offset_pairs, "buffer_iou_15m")
        _assert_first_better_than_last(values, "buffer_iou_15m", higher_is_better=True)

    def test_lateral_offset_iqr_increases(self, offset_pairs):
        values = _feature_series(offset_pairs, "lateral_offset_iqr_m")
        # IQR is 0 for uniform offset but should not decrease
        _assert_monotonic_increasing(values, "lateral_offset_iqr_m", tolerance=0.5)

    def test_lateral_offset_p95_increases(self, offset_pairs):
        values = _feature_series(offset_pairs, "lateral_offset_p95_m")
        _assert_monotonic_increasing(values, "lateral_offset_p95_m", tolerance=0.5)


class TestAngularRotationSweep:
    """Same center, rotated by increasing angles."""

    @pytest.fixture
    def rotation_pairs(self):
        """Use 3+ point lines so angle histogram comparison is exercised."""
        angles = [0, 5, 15, 30, 45, 90]
        center_x, center_y = 50, 50
        length = 50
        # Multi-point ref line (3 points for angle histogram)
        ref = make_projected_line([(0, 50), (50, 50), (100, 50)])
        pairs = []
        for angle_deg in angles:
            angle_rad = math.radians(angle_deg)
            dx = length * math.cos(angle_rad)
            dy = length * math.sin(angle_rad)
            # 3-point target with midpoint
            start = (center_x - dx, center_y - dy)
            end = (center_x + dx, center_y + dy)
            mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            target = make_projected_line([start, mid, end])
            pairs.append((ref, target))
        return pairs

    def test_heading_delta_increases(self, rotation_pairs):
        values = _feature_series(rotation_pairs, "heading_delta")
        _assert_monotonic_increasing(values, "heading_delta", tolerance=1.0)

    def test_angle_histogram_similarity_decreases(self, rotation_pairs):
        """Angle histogram should decrease as shapes diverge."""
        values = _feature_series(rotation_pairs, "angle_histogram_similarity")
        # At minimum, 0-degree should score >= 90-degree.
        assert values[0] >= values[-1] - 0.01, (
            f"angle_histogram_similarity: 0deg ({values[0]:.4f}) should be >= "
            f"90deg ({values[-1]:.4f})"
        )


class TestAlongTrackShiftSweep:
    """Collinear lines shifted along their axis.

    This is the key test for collinear_gap_ratio - if it's near-zero
    importance in ablation, this test will reveal whether the feature
    actually responds to the geometry it claims to measure.
    """

    @pytest.fixture
    def shift_pairs(self):
        """Reference: [0, 100], Target shifted by [0, 10, 50, 90, 100, 150]m."""
        shifts = [0, 10, 50, 90, 100, 150]
        ref = make_projected_line([(0, 0), (100, 0)])
        pairs = []
        for shift in shifts:
            target = make_projected_line([(shift, 0), (shift + 100, 0)])
            pairs.append((ref, target))
        return pairs

    def test_collinear_gap_responds_to_shift(self, shift_pairs):
        """Collinear gap ratio (overlap fraction) should decrease as lines slide apart."""
        values = _feature_series(shift_pairs, "collinear_gap_ratio")
        _assert_monotonic_decreasing(values, "collinear_gap_ratio", tolerance=0.01)

    def test_centroid_distance_increases_with_shift(self, shift_pairs):
        values = _feature_series(shift_pairs, "centroid_distance_m")
        _assert_monotonic_increasing(values, "centroid_distance_m", tolerance=1.0)


class TestShapeComplexitySweep:
    """Straight line vs progressively zigzag."""

    @pytest.fixture
    def complexity_pairs(self):
        """Lines with 0, 2, 4, 8, 16 turns."""
        turn_counts = [0, 2, 4, 8, 16]
        target = make_projected_line([(0, 2), (100, 2)])  # straight reference
        pairs = []
        for n_turns in turn_counts:
            if n_turns == 0:
                ref = make_projected_line([(0, 0), (100, 0)])
            else:
                # Create zigzag with n_turns
                n_segments = n_turns + 1
                segment_len = 100.0 / n_segments
                coords = [(0, 0)]
                for i in range(n_segments):
                    x = (i + 1) * segment_len
                    y = 5 * (1 if i % 2 == 0 else -1)
                    coords.append((x, y))
                ref = make_projected_line(coords)
            pairs.append((ref, target))
        return pairs

    def test_shape_complexity_ref_increases(self, complexity_pairs):
        values = _feature_series(complexity_pairs, "shape_complexity_ref")
        _assert_first_better_than_last(values, "shape_complexity_ref", higher_is_better=False)

    def test_heading_consistency_ref_decreases(self, complexity_pairs):
        values = _feature_series(complexity_pairs, "heading_consistency_ref")
        # Straight line should have highest consistency
        _assert_first_better_than_last(values, "heading_consistency_ref", higher_is_better=True)

    def test_sinuosity_ref_increases(self, complexity_pairs):
        values = _feature_series(complexity_pairs, "sinuosity_ref")
        _assert_first_better_than_last(values, "sinuosity_ref", higher_is_better=False)

    def test_sinuosity_delta_increases(self, complexity_pairs):
        values = _feature_series(complexity_pairs, "sinuosity_delta")
        _assert_first_better_than_last(values, "sinuosity_delta", higher_is_better=False)

    def test_heading_consistency_delta_increases(self, complexity_pairs):
        values = _feature_series(complexity_pairs, "heading_consistency_delta")
        _assert_first_better_than_last(values, "heading_consistency_delta", higher_is_better=False)

    def test_shape_complexity_delta_increases(self, complexity_pairs):
        values = _feature_series(complexity_pairs, "shape_complexity_delta")
        _assert_first_better_than_last(values, "shape_complexity_delta", higher_is_better=False)


class TestLengthRatioSweep:
    """Reference 100m, target progressively shorter."""

    @pytest.fixture
    def length_pairs(self):
        lengths = [100, 80, 60, 40, 20]
        ref = make_projected_line([(0, 0), (100, 0)])
        return [(ref, make_projected_line([(0, 2), (length, 2)])) for length in lengths]

    def test_length_ratio_decreases(self, length_pairs):
        values = _feature_series(length_pairs, "length_ratio")
        _assert_monotonic_decreasing(values, "length_ratio", tolerance=0.01)

    def test_min_coverage_decreases(self, length_pairs):
        """Shorter target means less overlap.

        Note: Without alignment data, coverage features default to 0.0.
        This test checks length_ratio instead, which doesn't require alignment.
        """
        values = _feature_series(length_pairs, "length_ratio")
        _assert_first_better_than_last(values, "length_ratio", higher_is_better=True)

    def test_min_length_decreases(self, length_pairs):
        values = _feature_series(length_pairs, "min_length_m")
        _assert_monotonic_decreasing(values, "min_length_m", tolerance=0.5)


class TestVertexDensitySweep:
    """Same physical line with increasing vertex counts."""

    @pytest.fixture
    def density_pairs(self):
        vertex_counts = [2, 5, 10, 50, 200]
        target = make_projected_line([(0, 2), (100, 2)])
        pairs = []
        for n in vertex_counts:
            xs = np.linspace(0, 100, n)
            coords = [(x, 0) for x in xs]
            ref = make_projected_line(coords)
            pairs.append((ref, target))
        return pairs

    def test_vertex_density_ref_increases(self, density_pairs):
        values = _feature_series(density_pairs, "vertex_density_ref")
        _assert_monotonic_increasing(values, "vertex_density_ref", tolerance=0.001)

    def test_vertex_density_target_constant(self, density_pairs):
        """Target is always the same 2-vertex line."""
        values = _feature_series(density_pairs, "vertex_density_target")
        _assert_all_constant(values, "vertex_density_target", tolerance=0.001)

    def test_vertex_density_ratio_decreases(self, density_pairs):
        """Ratio = min/max diverges as ref density increases."""
        values = _feature_series(density_pairs, "vertex_density_ratio")
        _assert_monotonic_decreasing(values, "vertex_density_ratio", tolerance=0.01)


class TestNameSimilaritySweep:
    """Same geometry, varying name similarity from identical to random."""

    @pytest.fixture
    def name_pairs(self):
        """Fixed geometry, names degrading: identical → abbreviation → partial → different → random."""
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 2), (100, 2)])
        pairs = [(ref, target)] * 5
        names = [
            ("Main Street", "Main Street"),  # identical
            ("Main Street", "Main St"),  # abbreviation
            ("North Main Street", "Main Street"),  # partial
            ("Main Street", "Oak Avenue"),  # different
            ("Main Street", "Xylophone Quartz"),  # random
        ]
        return pairs, names

    def test_name_levenshtein_decreases(self, name_pairs):
        pairs, names = name_pairs
        values = _feature_series_with_names(pairs, "name_levenshtein", names)
        _assert_first_better_than_last(values, "name_levenshtein", higher_is_better=True)

    def test_name_jaro_winkler_decreases(self, name_pairs):
        pairs, names = name_pairs
        values = _feature_series_with_names(pairs, "name_jaro_winkler", names)
        _assert_first_better_than_last(values, "name_jaro_winkler", higher_is_better=True)

    def test_name_token_sort_decreases(self, name_pairs):
        pairs, names = name_pairs
        values = _feature_series_with_names(pairs, "name_token_sort", names)
        _assert_first_better_than_last(values, "name_token_sort", higher_is_better=True)

    def test_name_soundex_decreases(self, name_pairs):
        pairs, names = name_pairs
        values = _feature_series_with_names(pairs, "name_soundex", names)
        _assert_first_better_than_last(values, "name_soundex", higher_is_better=True)

    def test_name_metaphone_decreases(self, name_pairs):
        pairs, names = name_pairs
        values = _feature_series_with_names(pairs, "name_metaphone", names)
        _assert_first_better_than_last(values, "name_metaphone", higher_is_better=True)


class TestNameFlagTests:
    """Point tests for name-related flag features."""

    def test_has_name_ref_with_name(self):
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 2), (100, 2)])
        feats = compute_features_simple(ref, target, ref_name="Main Street")
        assert feats["has_name_ref"] == 1.0

    def test_has_name_ref_without_name(self):
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 2), (100, 2)])
        feats = compute_features_simple(ref, target, ref_name=None)
        assert feats["has_name_ref"] == 0.0

    def test_has_name_target_without_name(self):
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 2), (100, 2)])
        feats = compute_features_simple(ref, target, target_name=None)
        assert feats["has_name_target"] == 0.0

    def test_name_is_generic(self):
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 2), (100, 2)])
        feats = compute_features_simple(ref, target, ref_name="Service Road")
        assert feats["name_is_generic"] == 1.0

    def test_route_prefix_match_same(self):
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 2), (100, 2)])
        feats = compute_features_simple(ref, target, ref_name="I-90", target_name="Interstate 90")
        assert feats["route_prefix_match"] == 1.0

    def test_route_prefix_match_different(self):
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 2), (100, 2)])
        feats = compute_features_simple(ref, target, ref_name="I-90", target_name="US-90")
        assert feats["route_prefix_match"] == 0.0

    def test_name_numeric_match_non_numeric(self):
        """Non-numeric names should return NaN (no numeric signal)."""
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 2), (100, 2)])
        feats = compute_features_simple(
            ref, target, ref_name="Main Street", target_name="Main Street"
        )
        assert math.isnan(feats["name_numeric_match"])


class TestClassSimilaritySweep:
    """Same geometry, varying class similarity from same to cross-tier."""

    @pytest.fixture
    def class_pairs(self):
        """Fixed geometry, classes degrading: same → adjacent → distant → cross-tier."""
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 2), (100, 2)])
        pairs = [(ref, target)] * 4
        classes = [
            ("residential", "residential"),  # same
            ("residential", "tertiary"),  # adjacent rank
            ("residential", "motorway"),  # distant rank
            ("residential", "footway"),  # cross-tier
        ]
        return pairs, classes

    def test_class_similarity_decreases(self, class_pairs):
        pairs, classes = class_pairs
        values = _feature_series_with_names(
            pairs,
            "class_similarity",
            names=[("Test Road", "Test Road")] * len(classes),
            classes=classes,
        )
        _assert_first_better_than_last(values, "class_similarity", higher_is_better=True)


class TestEndpointProximitySweep:
    """Lines with endpoints at increasing distances."""

    @pytest.fixture
    def endpoint_pairs(self):
        """Ref at origin, target endpoints shifted by [0, 2, 5, 10, 30, 100]m."""
        shifts = [0, 2, 5, 10, 30, 100]
        ref = make_projected_line([(0, 0), (100, 0)])
        pairs = []
        for s in shifts:
            target = make_projected_line([(s, s), (100 + s, s)])
            pairs.append((ref, target))
        return pairs

    def test_min_endpoint_proximity_increases(self, endpoint_pairs):
        values = _feature_series(endpoint_pairs, "min_endpoint_proximity_m")
        _assert_monotonic_increasing(values, "min_endpoint_proximity_m", tolerance=0.5)

    def test_max_endpoint_proximity_increases(self, endpoint_pairs):
        values = _feature_series(endpoint_pairs, "max_endpoint_proximity_m")
        _assert_monotonic_increasing(values, "max_endpoint_proximity_m", tolerance=0.5)

    def test_shared_endpoint_count_first_gt_last(self, endpoint_pairs):
        values = _feature_series(endpoint_pairs, "shared_endpoint_count")
        _assert_first_better_than_last(values, "shared_endpoint_count", higher_is_better=True)


class TestSinuositySweep:
    """Ref lines from straight to sinusoidal, target always straight."""

    @pytest.fixture
    def sinuosity_pairs(self):
        """Ref with increasing sinusoidal amplitude [0, 2, 5, 10, 20], target straight."""
        amplitudes = [0, 2, 5, 10, 20]
        target = make_projected_line([(0, 2), (100, 2)])
        pairs = []
        for amp in amplitudes:
            n_points = 50
            xs = np.linspace(0, 100, n_points)
            coords = [(x, amp * math.sin(2 * math.pi * x / 100)) for x in xs]
            ref = make_projected_line(coords)
            pairs.append((ref, target))
        return pairs

    def test_sinuosity_ref_increases(self, sinuosity_pairs):
        values = _feature_series(sinuosity_pairs, "sinuosity_ref")
        _assert_monotonic_increasing(values, "sinuosity_ref", tolerance=0.001)

    def test_sinuosity_target_constant(self, sinuosity_pairs):
        values = _feature_series(sinuosity_pairs, "sinuosity_target")
        _assert_all_constant(values, "sinuosity_target", tolerance=0.001)

    def test_sinuosity_delta_increases(self, sinuosity_pairs):
        values = _feature_series(sinuosity_pairs, "sinuosity_delta")
        _assert_monotonic_increasing(values, "sinuosity_delta", tolerance=0.001)

    def test_heading_consistency_ref_decreases(self, sinuosity_pairs):
        values = _feature_series(sinuosity_pairs, "heading_consistency_ref")
        _assert_first_better_than_last(values, "heading_consistency_ref", higher_is_better=True)

    def test_heading_consistency_delta_increases(self, sinuosity_pairs):
        """Delta increases as ref becomes more curvy while target stays straight."""
        values = _feature_series(sinuosity_pairs, "heading_consistency_delta")
        _assert_first_better_than_last(values, "heading_consistency_delta", higher_is_better=False)


class TestAngleHistogramTests:
    """Tests specific to angle_histogram_similarity after the fix."""

    def test_both_straight_lines(self):
        """Two 2-point lines should return 1.0 (both straight)."""
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 2), (100, 2)])
        feats = compute_features_simple(ref, target)
        assert feats["angle_histogram_similarity"] == 1.0

    def test_straight_vs_curvy(self):
        """Straight line vs zigzag should score low."""
        ref = make_projected_line([(0, 0), (100, 0)])
        # Zigzag target
        target = make_projected_line([(0, 2), (25, 20), (50, 2), (75, 20), (100, 2)])
        feats = compute_features_simple(ref, target)
        assert feats["angle_histogram_similarity"] < 0.5, (
            f"Straight vs zigzag should score low, got {feats['angle_histogram_similarity']:.4f}"
        )


class TestBugHuntingTests:
    """Specific tests targeting suspected implementation bugs."""

    def test_collinear_gap_bidirectional(self):
        """Reversed copy should get similar collinear_gap_ratio."""
        ref = make_projected_line([(0, 0), (100, 0)])
        target_fwd = make_projected_line([(50, 0), (150, 0)])
        target_rev = make_projected_line([(150, 0), (50, 0)])

        feats_fwd = compute_features_simple(ref, target_fwd)
        feats_rev = compute_features_simple(ref, target_rev)

        assert abs(feats_fwd["collinear_gap_ratio"] - feats_rev["collinear_gap_ratio"]) < 0.1, (
            f"collinear_gap_ratio not direction-invariant: "
            f"fwd={feats_fwd['collinear_gap_ratio']:.4f}, "
            f"rev={feats_rev['collinear_gap_ratio']:.4f}"
        )

    def test_angle_histogram_direction_invariant(self):
        """Forward vs reversed line should have same angle_histogram_similarity."""
        ref = make_projected_line([(0, 0), (30, 10), (60, 0), (100, 5)])
        target_fwd = make_projected_line([(0, 2), (30, 12), (60, 2), (100, 7)])
        target_rev = make_projected_line([(100, 7), (60, 2), (30, 12), (0, 2)])

        feats_fwd = compute_features_simple(ref, target_fwd)
        feats_rev = compute_features_simple(ref, target_rev)

        assert (
            abs(feats_fwd["angle_histogram_similarity"] - feats_rev["angle_histogram_similarity"])
            < 0.15
        ), (
            f"angle_histogram_similarity not direction-invariant: "
            f"fwd={feats_fwd['angle_histogram_similarity']:.4f}, "
            f"rev={feats_rev['angle_histogram_similarity']:.4f}"
        )

    def test_edge_distance_rmse_correlates_with_hausdorff(self):
        """On dense geometries, RMSE should track mean hausdorff."""
        # Dense line (many vertices) - RMSE and mean hausdorff should agree
        n = 50
        xs = np.linspace(0, 100, n)

        offsets = [2, 5, 10, 20]
        rmse_values = []
        hausdorff_values = []

        for offset in offsets:
            ref = make_projected_line([(x, 0) for x in xs])
            target = make_projected_line([(x, offset) for x in xs])
            feats = compute_features_simple(ref, target)
            rmse_values.append(feats["edge_distance_rmse_m"])
            hausdorff_values.append(feats["mean_hausdorff_distance_m"])

        # Both should increase monotonically
        for i in range(1, len(offsets)):
            assert rmse_values[i] >= rmse_values[i - 1] - 0.5, f"RMSE not increasing: {rmse_values}"
            assert hausdorff_values[i] >= hausdorff_values[i - 1] - 0.5, (
                f"Mean Hausdorff not increasing: {hausdorff_values}"
            )

    def test_vertex_density_reasonable_range(self):
        """Typical road vertex density: 0.01-1.0 vertices/m."""
        # A 100m line with 10 vertices = 0.1 v/m
        ref = make_projected_line([(i * 10, 0) for i in range(11)])
        target = make_projected_line([(0, 5), (100, 5)])
        feats = compute_features_simple(ref, target)

        vd = feats["vertex_density_ref"]
        assert 0.01 <= vd <= 2.0, (
            f"vertex_density_ref={vd:.4f} outside reasonable range [0.01, 2.0] "
            f"for 100m line with 11 vertices"
        )

    def test_identical_lines_perfect_scores(self):
        """Identical lines should produce optimal values for key features."""
        line = make_projected_line([(0, 0), (50, 5), (100, 0)])
        feats = compute_features_simple(line, line)

        # Similarity features should be at or near maximum
        assert feats["buffer_iou_5m"] > 0.99, f"buffer_iou_5m={feats['buffer_iou_5m']}"
        assert feats["length_ratio"] > 0.99, f"length_ratio={feats['length_ratio']}"
        assert feats["heading_delta"] < 1.0, f"heading_delta={feats['heading_delta']}"

        # Distance features should be at or near zero
        assert feats["hausdorff_distance_m"] < 1.0, (
            f"hausdorff_distance_m={feats['hausdorff_distance_m']}"
        )
        assert feats["lateral_offset_m"] < 1.0, f"lateral_offset_m={feats['lateral_offset_m']}"
