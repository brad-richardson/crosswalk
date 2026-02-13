"""Test that features actually discriminate between match and no_match labels.

Uses real labeled data to check distribution separation, variance, and
degenerate-value detection.
"""

import pandas as pd
import pytest
from scipy import stats

from matcher.config import FEATURE_COLUMNS
from matcher.features.compute import _get_error_features


class TestFeatureSeparationByLabel:
    """Check if features separate match vs no_match in real data."""

    def test_feature_separation_by_label(self, match_features, no_match_features):
        """Per feature: Mann-Whitney U test, compare medians.

        Not every feature needs to be discriminative, but this reports which
        ones are and which aren't.
        """
        results = []
        for col in FEATURE_COLUMNS:
            if col not in match_features.columns or col not in no_match_features.columns:
                continue

            match_vals = match_features[col].dropna()
            no_match_vals = no_match_features[col].dropna()

            if len(match_vals) < 5 or len(no_match_vals) < 5:
                continue

            match_med = match_vals.median()
            no_match_med = no_match_vals.median()

            try:
                u_stat, p_value = stats.mannwhitneyu(
                    match_vals, no_match_vals, alternative="two-sided"
                )
            except ValueError:
                p_value = 1.0

            results.append(
                {
                    "feature": col,
                    "match_median": match_med,
                    "no_match_median": no_match_med,
                    "median_diff": abs(match_med - no_match_med),
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
            )

        # At least SOME features should be discriminative
        df = pd.DataFrame(results)
        significant_count = df["significant"].sum()
        assert significant_count > 5, (
            f"Only {significant_count} features are statistically significant "
            f"(p < 0.05) at separating match from no_match. "
            f"Expected at least 5."
        )


class TestSuspiciousFeaturesHaveVariance:
    """Features should have non-zero variance (not all the same value)."""

    def test_suspicious_features_have_variance(self, labeled_features):
        """Assert non-zero variance for features that SHOULD vary."""
        # Features that require full graph/spatial context and may be constant
        # in backfilled data (computed without full Overture graph)
        context_dependent = {
            "graphlet_similarity",
            "endpoint_degree_similarity",
            "clustering_coef_ref",
            "clustering_coef_target",
            "clustering_coef_delta",
            "min_endpoint_proximity_m",
            "max_endpoint_proximity_m",
            "shared_endpoint_count",
            # Topology degree features - require connector graph
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
        }
        zero_variance = []
        for col in FEATURE_COLUMNS:
            if col in context_dependent:
                continue
            if col not in labeled_features.columns:
                continue
            series = labeled_features[col].dropna()
            if len(series) < 10:
                continue
            if series.var() < 1e-12:
                zero_variance.append(col)

        # Allow some features to be constant (e.g., coverage when no alignment)
        # but flag if more than 10 are degenerate
        assert len(zero_variance) <= 10, (
            f"{len(zero_variance)} features have zero variance: {zero_variance}"
        )


class TestDegenerateValueDetection:
    """Check for features stuck at their default/error values."""

    def _pct_at_default(self, series: pd.Series, default_val: float, tol: float = 1e-6) -> float:
        """Return fraction of values matching the default."""
        if len(series) == 0:
            return 0.0
        close = (abs(series - default_val) < tol).sum()
        return close / len(series)

    def test_graphlet_features_not_all_default(self, labeled_features):
        """graphlet_similarity should not be all 0.5 (the error default)."""
        if "graphlet_similarity" not in labeled_features.columns:
            pytest.skip("graphlet_similarity not in data")
        series = labeled_features["graphlet_similarity"].dropna()
        pct = self._pct_at_default(series, 0.5)
        # Allow up to 80% at default (sparse graph data), but flag > 80%
        assert pct < 0.80, (
            f"graphlet_similarity: {pct:.1%} of values are at default 0.5 "
            f"(suggests feature rarely fires)"
        )

    def test_clustering_coef_not_all_zero(self, labeled_features):
        """clustering_coef_ref should not be all 0.0."""
        if "clustering_coef_ref" not in labeled_features.columns:
            pytest.skip("clustering_coef_ref not in data")
        series = labeled_features["clustering_coef_ref"].dropna()
        pct = self._pct_at_default(series, 0.0)
        assert pct < 0.95, (
            f"clustering_coef_ref: {pct:.1%} of values are 0.0 (suggests feature rarely fires)"
        )

    def test_parallel_sibling_detection_fires(self, labeled_features):
        """has_parallel_sibling_ref should not be all 0.0."""
        if "has_parallel_sibling_ref" not in labeled_features.columns:
            pytest.skip("has_parallel_sibling_ref not in data")
        series = labeled_features["has_parallel_sibling_ref"].dropna()
        nonzero = (series > 0).sum()
        # Expect at least some sibling detection
        assert nonzero > 0, (
            f"has_parallel_sibling_ref is all 0.0 across {len(series)} samples. "
            f"Sibling detection may be disabled or broken."
        )

    def test_collinear_gap_not_all_one(self, labeled_features):
        """collinear_gap_ratio should not be all 1.0 (the error default)."""
        if "collinear_gap_ratio" not in labeled_features.columns:
            pytest.skip("collinear_gap_ratio not in data")
        series = labeled_features["collinear_gap_ratio"].dropna()
        pct = self._pct_at_default(series, 1.0)
        assert pct < 0.95, f"collinear_gap_ratio: {pct:.1%} of values are at default 1.0"

    def test_angle_histogram_not_all_default(self, labeled_features):
        """angle_histogram_similarity should not be all 0.5."""
        if "angle_histogram_similarity" not in labeled_features.columns:
            pytest.skip("angle_histogram_similarity not in data")
        series = labeled_features["angle_histogram_similarity"].dropna()
        pct = self._pct_at_default(series, 0.5)
        assert pct < 0.80, f"angle_histogram_similarity: {pct:.1%} of values are at default 0.5"

    def test_endpoint_degree_similarity_not_all_default(self, labeled_features):
        """endpoint_degree_similarity should not be all 0.5."""
        if "endpoint_degree_similarity" not in labeled_features.columns:
            pytest.skip("endpoint_degree_similarity not in data")
        series = labeled_features["endpoint_degree_similarity"].dropna()
        pct = self._pct_at_default(series, 0.5)
        assert pct < 0.80, f"endpoint_degree_similarity: {pct:.1%} of values are at default 0.5"

    def test_edge_distance_rmse_not_all_max(self, labeled_features):
        """edge_distance_rmse_m should not be all MAX_DISTANCE."""
        if "edge_distance_rmse_m" not in labeled_features.columns:
            pytest.skip("edge_distance_rmse_m not in data")
        series = labeled_features["edge_distance_rmse_m"].dropna()
        pct = self._pct_at_default(series, 10000.0)
        assert pct < 0.10, f"edge_distance_rmse_m: {pct:.1%} of values are at MAX_DISTANCE 10000"


class TestErrorDefaultPercentage:
    """Check what fraction of real data matches error defaults."""

    def test_low_error_default_percentage(self, labeled_features):
        """Report features with high error-default percentages.

        Many features legitimately have high default rates:
        - Name features: many roads are unnamed (has_name_ref, name_is_generic)
        - Shape features: many roads are straight (shape_complexity=0, sinuosity=1.0)
        - Route features: most roads aren't numbered routes (route_prefix_match=0.5)
        - Clustering: sparse graph data (clustering_coef=0.0)

        This test flags truly problematic features (>90% at default AND the feature
        claims to measure something that should vary for most road pairs).
        """
        error_defaults = _get_error_features()

        # Features with known bugs (tracked separately)
        known_broken: set[str] = set()

        # Features where high default rate is expected/acceptable
        expected_high_default = {
            "collinear_gap_ratio",  # Returns 1.0 for non-collinear pairs (~93% of data)
            "name_soundex",
            "name_metaphone",  # Many unnamed roads
            "has_name_ref",
            "has_name_target",  # Binary: unnamed roads common
            "name_is_generic",  # Most names aren't generic
            "route_prefix_match",  # Most roads aren't numbered routes
            "clustering_coef_ref",
            "clustering_coef_target",
            "clustering_coef_delta",  # Sparse graphs
            "sinuosity_ref",
            "sinuosity_target",  # Many straight roads
            "heading_consistency_ref",
            "heading_consistency_target",  # Many straight roads
            "shape_complexity_ref",
            "shape_complexity_target",
            "shape_complexity_delta",  # Many straight roads
            "likely_representation_mismatch",  # Most pairs aren't mismatched
            "has_parallel_sibling_ref",  # Most segments don't have siblings
            "parallel_fraction_ref",  # Related to sibling detection
            # Context-dependent features: require full graph/spatial context
            # that backfill doesn't have, so they stay at defaults
            "graphlet_similarity",  # Requires connector graph
            "endpoint_degree_similarity",  # Requires connector graph
            "min_endpoint_proximity_m",  # Requires spatial index
            "max_endpoint_proximity_m",  # Requires spatial index
            "shared_endpoint_count",  # Requires spatial index
        }

        truly_broken = []
        for col in FEATURE_COLUMNS:
            if col in expected_high_default or col in known_broken:
                continue
            if col not in labeled_features.columns:
                continue
            default_val = error_defaults.get(col)
            if default_val is None:
                continue

            series = labeled_features[col].dropna()
            if len(series) == 0:
                continue

            pct = (abs(series - default_val) < 1e-6).sum() / len(series)
            if pct > 0.90:
                truly_broken.append(f"{col}: {pct:.1%} at default {default_val}")

        assert len(truly_broken) == 0, (
            "Features with >90% values at error default (unexpected):\n" + "\n".join(truly_broken)
        )
