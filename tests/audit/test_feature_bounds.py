"""Test that all features stay within expected bounds.

Catches NaN/inf leaks, sign errors, and out-of-range values.
"""

import numpy as np

from matcher.config import FEATURE_COLUMNS

from .conftest import FEATURE_BOUNDS, compute_features_simple, make_projected_line


class TestBoundsOnSyntheticPairs:
    """Verify feature bounds on controlled synthetic geometry pairs."""

    # Features that are NaN by design when optional context is missing.
    # compute_features_simple() does not pass alignment, graphlet data, or
    # sibling context, so these will be NaN.
    # XGBoost handles NaN natively.
    NAN_WITHOUT_CONTEXT = {
        "post_node_continuation_m",
        "endpoint_heading_divergence",
        # No graphlet data in simple test
        "graphlet_similarity",
        "endpoint_degree_similarity",
        # No graphlet/alignment data — clustering unavailable
        "clustering_coef_ref",
        "clustering_coef_target",
        "clustering_coef_delta",
        # No sibling context — crossing angle defaults to NaN
        "crossing_angle_min_ref",
        "transverse_neighbor_fraction_ref",
        "crossing_angle_min_target",
        "transverse_neighbor_fraction_target",
        # "Test Road" has no numeric suffix or route prefix
        "name_numeric_match",
        "route_prefix_match",
    }

    # Additional features that are NaN when names are None
    NAN_WITHOUT_NAMES = {
        "name_levenshtein",
        "name_jaro_winkler",
        "name_token_sort",
        "name_soundex",
        "name_metaphone",
    }

    def test_bounds_on_identical_lines(self):
        """Perfect match pair: both lines are the same."""
        line = make_projected_line([(0, 0), (50, 0), (100, 0)])
        features = compute_features_simple(line, line)

        for name in FEATURE_COLUMNS:
            val = features.get(name)
            assert val is not None, f"Feature {name} is None"
            if name in self.NAN_WITHOUT_CONTEXT:
                continue  # NaN is valid when no alignment provided
            assert not np.isnan(val), f"Feature {name} is NaN for identical lines"
            assert not np.isinf(val), f"Feature {name} is inf for identical lines"

            bounds = FEATURE_BOUNDS.get(name)
            if bounds:
                lo, hi = bounds
                if lo is not None:
                    assert val >= lo - 1e-9, (
                        f"Feature {name}={val} below min {lo} for identical lines"
                    )
                if hi is not None:
                    assert val <= hi + 1e-9, (
                        f"Feature {name}={val} above max {hi} for identical lines"
                    )

    def test_bounds_on_parallel_close_lines(self):
        """Close parallel lines: typical match scenario."""
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 3), (100, 3)])
        features = compute_features_simple(ref, target)

        for name in FEATURE_COLUMNS:
            val = features.get(name)
            assert val is not None, f"Feature {name} is None"
            if name in self.NAN_WITHOUT_CONTEXT:
                continue
            assert not np.isnan(val), f"Feature {name} is NaN for parallel lines"
            assert not np.isinf(val), f"Feature {name} is inf for parallel lines"

    def test_bounds_on_far_apart_lines(self):
        """Worst case: lines far apart and perpendicular."""
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(500, 500), (500, 600)])
        features = compute_features_simple(ref, target, ref_name=None, target_name=None)

        nan_expected = self.NAN_WITHOUT_CONTEXT | self.NAN_WITHOUT_NAMES
        for name in FEATURE_COLUMNS:
            val = features.get(name)
            assert val is not None, f"Feature {name} is None"
            if name in nan_expected:
                continue
            assert not np.isnan(val), f"Feature {name} is NaN for far-apart lines"
            assert not np.isinf(val), f"Feature {name} is inf for far-apart lines"

            bounds = FEATURE_BOUNDS.get(name)
            if bounds:
                lo, hi = bounds
                if lo is not None:
                    assert val >= lo - 1e-9, (
                        f"Feature {name}={val} below min {lo} for far-apart lines"
                    )
                if hi is not None:
                    assert val <= hi + 1e-9, (
                        f"Feature {name}={val} above max {hi} for far-apart lines"
                    )

    def test_bounds_on_short_line(self):
        """Degenerate: very short line (1 meter)."""
        ref = make_projected_line([(0, 0), (1, 0)])
        target = make_projected_line([(0, 0.5), (1, 0.5)])
        features = compute_features_simple(ref, target)

        for name in FEATURE_COLUMNS:
            val = features.get(name)
            assert val is not None, f"Feature {name} is None"
            if name in self.NAN_WITHOUT_CONTEXT:
                continue
            assert not np.isnan(val), f"Feature {name} is NaN for short line"
            assert not np.isinf(val), f"Feature {name} is inf for short line"

    def test_bounds_on_two_vertex_line(self):
        """Degenerate: minimum vertex count (2 vertices)."""
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(0, 5), (100, 5)])
        features = compute_features_simple(ref, target)

        for name in FEATURE_COLUMNS:
            val = features.get(name)
            assert val is not None, f"Feature {name} is None"
            if name in self.NAN_WITHOUT_CONTEXT:
                continue
            assert not np.isnan(val), f"Feature {name} is NaN for 2-vertex line"
            assert not np.isinf(val), f"Feature {name} is inf for 2-vertex line"

    def test_bounds_on_reversed_line(self):
        """Direction shouldn't cause out-of-bounds values."""
        ref = make_projected_line([(0, 0), (50, 0), (100, 0)])
        target = make_projected_line([(100, 2), (50, 2), (0, 2)])  # reversed
        features = compute_features_simple(ref, target)

        for name in FEATURE_COLUMNS:
            val = features.get(name)
            assert val is not None, f"Feature {name} is None"
            if name in self.NAN_WITHOUT_CONTEXT:
                continue
            assert not np.isnan(val), f"Feature {name} is NaN for reversed line"
            assert not np.isinf(val), f"Feature {name} is inf for reversed line"

    def test_bounds_on_zigzag_line(self):
        """Complex shape: zigzag pattern."""
        ref_coords = [(i * 10, (i % 2) * 10) for i in range(11)]
        target_coords = [(i * 10, (i % 2) * 10 + 3) for i in range(11)]
        ref = make_projected_line(ref_coords)
        target = make_projected_line(target_coords)
        features = compute_features_simple(ref, target)

        for name in FEATURE_COLUMNS:
            val = features.get(name)
            assert val is not None, f"Feature {name} is None"
            if name in self.NAN_WITHOUT_CONTEXT:
                continue
            assert not np.isnan(val), f"Feature {name} is NaN for zigzag line"
            assert not np.isinf(val), f"Feature {name} is inf for zigzag line"


class TestErrorFeatureBounds:
    """Verify _get_error_features() returns NaN for all features."""

    def test_error_features_all_nan(self, error_features):
        """All error feature defaults should be NaN (XGBoost handles natively)."""
        for name in FEATURE_COLUMNS:
            val = error_features.get(name)
            assert val is not None, f"Error feature {name} missing"
            assert np.isnan(val), f"Error feature {name} should be NaN, got {val}"

    def test_error_features_cover_all_columns(self, error_features):
        """Error features must define every declared feature."""
        for name in FEATURE_COLUMNS:
            assert name in error_features, f"Error features missing {name}"


class TestBoundsOnRealData:
    """Verify no NaN/inf in real labeled data."""

    # Features that may legitimately be NaN in labeled data:
    # - Topology: NaN when target data is unavailable
    # - Graphlet/clustering: NaN when graphlet data not precomputed
    # - Crossing angle: NaN when no sibling context or no different-tier neighbors
    # - Intersection overlap: NaN without alignment/topology
    # - Name similarity: NaN when names are missing
    # - Class similarity: NaN when class is missing/unknown
    # - Route/numeric: NaN when not a route or no numeric component
    NAN_ALLOWED_FEATURES = {
        # Topology
        "from_degree_ref",
        "to_degree_ref",
        "from_degree_target",
        "to_degree_target",
        "degree_match_score",
        "degree_signature_similarity",
        "is_dead_end_ref",
        "is_dead_end_target",
        "dead_end_match",
        "is_intersection_ref",
        "is_intersection_target",
        "intersection_match",
        # Intersection overlap
        "post_node_continuation_m",
        "endpoint_heading_divergence",
        # Graphlet / clustering
        "graphlet_similarity",
        "endpoint_degree_similarity",
        "clustering_coef_ref",
        "clustering_coef_target",
        "clustering_coef_delta",
        # Crossing angle
        "crossing_angle_min_ref",
        "transverse_neighbor_fraction_ref",
        "crossing_angle_min_target",
        "transverse_neighbor_fraction_target",
        # Name similarity (NaN when names missing)
        "name_levenshtein",
        "name_jaro_winkler",
        "name_token_sort",
        "name_soundex",
        "name_metaphone",
        "name_numeric_match",
        "route_prefix_match",
        # Class similarity (NaN when class missing/unknown)
        "class_similarity",
    }

    def test_no_nan_in_real_data(self, labeled_features):
        """Assert no NaN per feature column in real labeled data (except allowed)."""
        for col in FEATURE_COLUMNS:
            if col not in labeled_features.columns:
                continue
            if col in self.NAN_ALLOWED_FEATURES:
                continue  # NaN is expected when optional context unavailable
            nan_count = labeled_features[col].isna().sum()
            total = len(labeled_features)
            assert nan_count == 0, (
                f"Feature {col} has {nan_count}/{total} NaN values in labeled data"
            )

    def test_no_inf_in_real_data(self, labeled_features):
        """Assert no inf per feature column in real labeled data."""
        for col in FEATURE_COLUMNS:
            if col not in labeled_features.columns:
                continue
            series = labeled_features[col]
            inf_count = np.isinf(series).sum()
            assert inf_count == 0, f"Feature {col} has {inf_count} inf values in labeled data"

    def test_real_data_within_bounds(self, labeled_features):
        """Check that real data values respect declared bounds."""
        violations = []
        for col in FEATURE_COLUMNS:
            if col not in labeled_features.columns:
                continue
            bounds = FEATURE_BOUNDS.get(col)
            if not bounds:
                continue
            lo, hi = bounds
            series = labeled_features[col].dropna()
            if len(series) == 0:
                continue

            if lo is not None:
                below = (series < lo - 1e-6).sum()
                if below > 0:
                    worst = series.min()
                    violations.append(f"{col}: {below} values below {lo} (min={worst:.4f})")

            if hi is not None:
                above = (series > hi + 1e-6).sum()
                if above > 0:
                    worst = series.max()
                    violations.append(f"{col}: {above} values above {hi} (max={worst:.4f})")

        assert len(violations) == 0, "Bound violations in real data:\n" + "\n".join(violations)
