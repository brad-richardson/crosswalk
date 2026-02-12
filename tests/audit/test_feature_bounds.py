"""Test that all features stay within expected bounds.

Catches NaN/inf leaks, sign errors, and out-of-range values.
"""

import numpy as np

from matcher.config import FEATURE_COLUMNS

from .conftest import FEATURE_BOUNDS, compute_features_simple, make_projected_line


class TestBoundsOnSyntheticPairs:
    """Verify feature bounds on controlled synthetic geometry pairs."""

    # Features that are NaN by design when no alignment is provided.
    # compute_features_simple() does not pass alignment, so these will be NaN.
    # XGBoost handles NaN natively (same pattern as topology features).
    NAN_WITHOUT_ALIGNMENT = {
        "post_node_continuation_m",
    }

    def test_bounds_on_identical_lines(self):
        """Perfect match pair: both lines are the same."""
        line = make_projected_line([(0, 0), (50, 0), (100, 0)])
        features = compute_features_simple(line, line)

        for name in FEATURE_COLUMNS:
            val = features.get(name)
            assert val is not None, f"Feature {name} is None"
            if name in self.NAN_WITHOUT_ALIGNMENT:
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
            if name in self.NAN_WITHOUT_ALIGNMENT:
                continue
            assert not np.isnan(val), f"Feature {name} is NaN for parallel lines"
            assert not np.isinf(val), f"Feature {name} is inf for parallel lines"

    def test_bounds_on_far_apart_lines(self):
        """Worst case: lines far apart and perpendicular."""
        ref = make_projected_line([(0, 0), (100, 0)])
        target = make_projected_line([(500, 500), (500, 600)])
        features = compute_features_simple(ref, target, ref_name=None, target_name=None)

        for name in FEATURE_COLUMNS:
            val = features.get(name)
            assert val is not None, f"Feature {name} is None"
            if name in self.NAN_WITHOUT_ALIGNMENT:
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
            if name in self.NAN_WITHOUT_ALIGNMENT:
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
            if name in self.NAN_WITHOUT_ALIGNMENT:
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
            if name in self.NAN_WITHOUT_ALIGNMENT:
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
            if name in self.NAN_WITHOUT_ALIGNMENT:
                continue
            assert not np.isnan(val), f"Feature {name} is NaN for zigzag line"
            assert not np.isinf(val), f"Feature {name} is inf for zigzag line"


class TestErrorFeatureBounds:
    """Verify _get_error_features() returns values within bounds."""

    # Error defaults that are NaN by design (XGBoost handles natively)
    # Topology features: unknown when target data unavailable
    # Intersection overlap features: unknown without alignment/topology
    TOPOLOGY_FEATURES = {
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
        "post_node_continuation_m",
    }

    def test_error_features_within_bounds(self, error_features):
        for name in FEATURE_COLUMNS:
            val = error_features.get(name)
            assert val is not None, f"Error feature {name} missing"
            if name in self.TOPOLOGY_FEATURES:
                assert np.isnan(val), f"Topology error feature {name} should be NaN"
            else:
                assert not np.isnan(val), f"Error feature {name} is NaN"
                assert not np.isinf(val), f"Error feature {name} is inf"

    def test_error_features_cover_all_columns(self, error_features):
        """Error features must define every declared feature."""
        for name in FEATURE_COLUMNS:
            assert name in error_features, f"Error features missing {name}"


class TestBoundsOnRealData:
    """Verify no NaN/inf in real labeled data."""

    # Topology features may be NaN when target data is unavailable
    # (e.g., us_boston_streets_osm has no fetchable target). XGBoost handles NaN natively.
    TOPOLOGY_FEATURES = TestErrorFeatureBounds.TOPOLOGY_FEATURES

    def test_no_nan_in_real_data(self, labeled_features):
        """Assert no NaN per feature column in real labeled data (except topology)."""
        for col in FEATURE_COLUMNS:
            if col not in labeled_features.columns:
                continue
            if col in self.TOPOLOGY_FEATURES:
                continue  # Topology NaN is expected when target data unavailable
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
