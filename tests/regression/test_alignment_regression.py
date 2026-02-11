"""Regression tests for alignment - ensures good matches stay good after changes.

These tests verify that the alignment algorithm maintains expected behavior
across various scenarios. They use synthetic geometries to ensure tests are
reproducible regardless of data file availability.
"""

import math

import pyproj
import pytest
from shapely.geometry import LineString
from shapely.ops import transform as shapely_transform

from matcher.features.alignment import create_subline, linestring_alignment


class TestSyntheticAlignmentRegression:
    """Synthetic tests to ensure basic alignment behavior doesn't regress."""

    def test_identical_lines_full_coverage(self):
        """Identical lines should have 100% coverage."""
        line = LineString([(0, 0), (50, 0), (100, 0)])
        result = linestring_alignment(line, line)

        assert result.overture_coverage >= 0.99
        assert result.dataset_coverage >= 0.99

    def test_parallel_offset_full_coverage(self):
        """Parallel offset lines should have full coverage (no divergence)."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 5), (100, 5)])  # 5m offset

        result = linestring_alignment(ref, target)

        assert result.overture_coverage >= 0.95
        assert result.dataset_coverage >= 0.95

    def test_partial_overlap_correct_coverage(self):
        """Partial overlap should report correct coverage fractions."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(25, 0), (75, 0)])  # 50% overlap

        result = linestring_alignment(ref, target)

        # Reference should show ~50% coverage (the overlapping portion)
        assert 0.4 <= result.overture_coverage <= 0.6
        # Target should show ~100% coverage (it's fully aligned)
        assert result.dataset_coverage >= 0.9

    def test_reversed_line_full_coverage(self):
        """Reversed lines should still have full coverage."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(100, 0), (0, 0)])  # Reversed

        result = linestring_alignment(ref, target)

        assert result.overture_coverage >= 0.95
        assert result.dataset_coverage >= 0.95

    def test_slight_lateral_variation_not_truncated(self):
        """Lines with slight lateral variations shouldn't be truncated."""
        ref = LineString([(0, 0), (25, 0), (50, 0), (75, 0), (100, 0)])
        # Target has small lateral deviations (GPS jitter simulation)
        target = LineString([(0, 1), (25, -1), (50, 2), (75, -1), (100, 1)])

        result = linestring_alignment(ref, target)

        # Should maintain high coverage despite small deviations
        assert result.overture_coverage >= 0.9
        assert result.dataset_coverage >= 0.9

    def test_curvy_road_maintains_coverage(self):
        """Curvy roads that follow the same path should maintain coverage."""
        # Create a curvy road (sinusoidal)
        ref_coords = [(x, 5 * math.sin(x / 10)) for x in range(0, 101, 5)]
        target_coords = [(x, 5 * math.sin(x / 10) + 2) for x in range(0, 101, 5)]

        ref = LineString(ref_coords)
        target = LineString(target_coords)

        result = linestring_alignment(ref, target)

        # Curvy but parallel should have high coverage
        assert result.overture_coverage >= 0.9
        assert result.dataset_coverage >= 0.9

    def test_different_lengths_partial_coverage(self):
        """Lines of different lengths should show proportional coverage."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (200, 0)])  # Target is 2x longer

        result = linestring_alignment(ref, target)

        # Reference should be fully covered
        assert result.overture_coverage >= 0.95
        # Target should be ~50% covered
        assert 0.4 <= result.dataset_coverage <= 0.6

    def test_multi_segment_alignment(self):
        """Complex multi-segment lines should still align properly."""
        ref = LineString([(0, 0), (30, 10), (60, 0), (90, 10), (120, 0)])
        target = LineString([(0, 2), (30, 12), (60, 2), (90, 12), (120, 2)])

        result = linestring_alignment(ref, target)

        # Should maintain high coverage for parallel zigzag pattern
        assert result.overture_coverage >= 0.9
        assert result.dataset_coverage >= 0.9


class TestDivergenceDetectionRegression:
    """Regression tests specifically for divergence detection behavior."""

    def test_sharp_end_divergence_detected(self):
        """Sharp divergence at the end should be detected and truncated."""
        ref = LineString([(0, 0), (80, 0), (100, 0)])
        target = LineString([(0, 0), (80, 0), (100, 50)])  # 50m divergence at end

        result = linestring_alignment(ref, target)

        # Should detect divergence and truncate
        assert result.overture_coverage < 0.95
        assert result.dataset_coverage < 0.95

    def test_sharp_start_divergence_detected(self):
        """Sharp divergence at the start should be detected and truncated."""
        ref = LineString([(0, 0), (20, 0), (100, 0)])
        target = LineString([(0, 50), (20, 0), (100, 0)])  # 50m divergence at start

        result = linestring_alignment(ref, target)

        # Should detect divergence at start
        assert result.overture_start_frac > 0.05

    def test_both_ends_divergence_detected(self):
        """Divergence at both ends should truncate both."""
        ref = LineString([(0, 0), (50, 0), (100, 0)])
        target = LineString([(0, 30), (50, 0), (100, 30)])

        result = linestring_alignment(ref, target)

        # Should truncate both ends
        assert result.overture_coverage < 0.8

    def test_gradual_curve_tolerated(self):
        """Gradual curves (small angle) should not be truncated."""
        # 10 degree angle over 100m = ~17m divergence
        angle_rad = math.radians(10)
        end_y = 100 * math.tan(angle_rad)

        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 0), (100, end_y)])

        result = linestring_alignment(ref, target)

        # Gradual curve should still have decent coverage
        assert result.overture_coverage >= 0.7

    def test_15m_constant_offset_no_truncation(self):
        """15m constant offset should not trigger divergence truncation."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 15), (100, 15)])

        result = linestring_alignment(ref, target)

        # Constant offset is not divergence
        assert result.overture_coverage >= 0.95
        assert result.dataset_coverage >= 0.95

    def test_divergence_detection_can_be_disabled(self):
        """Divergence detection should be disableable."""
        ref = LineString([(0, 0), (80, 0), (100, 0)])
        target = LineString([(0, 0), (80, 0), (100, 50)])

        # With divergence detection
        with_detection = linestring_alignment(ref, target, detect_divergence=True)

        # Without divergence detection
        without_detection = linestring_alignment(ref, target, detect_divergence=False)

        # Without detection should show higher coverage
        assert without_detection.overture_coverage >= with_detection.overture_coverage


class TestShortSegmentOnLongReference:
    """Tests for alignment of short target segments on long references.

    Real-world case: phila_sidewalk_899976 (11m) on Overture d4cc8e07 (817m).
    The alignment must correctly identify WHERE the short target sits on the
    long reference, not just that it overlaps somewhere.

    The subline center should be within 1m of the target's actual projected
    position on the reference.
    """

    @staticmethod
    def _expected_center_frac(ref, target):
        """Compute where the target's midpoint projects onto the reference."""
        target_mid = target.interpolate(0.5, normalized=True)
        return ref.project(target_mid, normalized=True)

    @staticmethod
    def _actual_center_frac(result):
        """Compute the center of the aligned subline on the reference."""
        return (result.overture_start_frac + result.overture_end_frac) / 2

    def test_real_pair_phila_sidewalk_on_long_ref(self):
        """Alignment should place subline where target actually sits.

        Target endpoint is 0.94m from reference start. The aligned subline
        center should be within 1m of the projected position (~4.7m along ref).
        """
        ref = LineString(
            [
                (-75.1356235, 40.0952469),
                (-75.1356788, 40.0952247),
                (-75.1372502, 40.0945944),
                (-75.1376063, 40.0944512),
                (-75.1376626, 40.0944041),
                (-75.1376921, 40.0943446),
                (-75.1375271, 40.0934991),
                (-75.1375029, 40.0934643),
                (-75.1373363, 40.0931817),
                (-75.1373013, 40.0931319),
                (-75.1372605, 40.0930737),
                (-75.1371757, 40.0929502),
                (-75.1370376, 40.0927564),
                (-75.1370148, 40.0927618),
                (-75.1363652, 40.0929186),
                (-75.136224, 40.0929478),
                (-75.135057, 40.0932222),
                (-75.1344723, 40.0933597),
                (-75.1339933, 40.0934766),
                (-75.1337999, 40.0935373),
            ]
        )
        target = LineString(
            [
                (-75.13572513699995, 40.09521999800006),
                (-75.13563115199997, 40.095253005000075),
            ]
        )

        result = linestring_alignment(ref, target)

        expected = self._expected_center_frac(ref, target)
        actual = self._actual_center_frac(result)
        error_m = abs(actual - expected) * ref.length
        assert error_m < 1.0 / 111000, (  # 1m tolerance in degrees
            f"Subline center off by {error_m * 111000:.1f}m "
            f"(expected frac={expected:.6f}, got={actual:.6f})"
        )
        assert result.dataset_coverage >= 0.9

    def test_synthetic_short_segment_at_bend(self):
        """Short target near start of a long L-shaped reference (projected meters)."""
        ref = LineString(
            [
                (0, 0),
                (100, 0),
                (200, 0),
                (300, 0),
                (400, 0),
                (400, -100),
                (400, -200),
                (400, -300),
                (400, -400),
            ]
        )
        target = LineString([(5, 3), (16, 3)])

        result = linestring_alignment(ref, target)

        expected = self._expected_center_frac(ref, target)
        actual = self._actual_center_frac(result)
        error_m = abs(actual - expected) * ref.length
        assert error_m < 1.0, (
            f"Subline center off by {error_m:.1f}m (expected frac={expected:.6f}, got={actual:.6f})"
        )
        assert result.dataset_coverage >= 0.9

    def test_synthetic_short_segment_at_midpoint(self):
        """Short target near midpoint of a long straight reference."""
        ref = LineString([(0, 0), (800, 0)])
        target = LineString([(400, 3), (411, 3)])

        result = linestring_alignment(ref, target)

        expected = self._expected_center_frac(ref, target)
        actual = self._actual_center_frac(result)
        error_m = abs(actual - expected) * ref.length
        assert error_m < 1.0, (
            f"Subline center off by {error_m:.1f}m (expected frac={expected:.6f}, got={actual:.6f})"
        )
        assert result.dataset_coverage >= 0.9

    def test_synthetic_short_segment_at_end(self):
        """Short target near end of a long straight reference."""
        ref = LineString([(0, 0), (800, 0)])
        target = LineString([(789, 3), (800, 3)])

        result = linestring_alignment(ref, target)

        expected = self._expected_center_frac(ref, target)
        actual = self._actual_center_frac(result)
        error_m = abs(actual - expected) * ref.length
        assert error_m < 1.0, (
            f"Subline center off by {error_m:.1f}m (expected frac={expected:.6f}, got={actual:.6f})"
        )
        assert result.dataset_coverage >= 0.9


# -- Shared geometry fixtures for overalignment tests --

# Mumbai pair: target end barely meets reference start (~5m overlap on 214m ref)
_MUMBAI_REF_WGS = [
    (72.8673478, 19.0670971),
    (72.8673947, 19.0671258),
    (72.8674923, 19.0671855),
    (72.8680551, 19.0675296),
    (72.8682888, 19.0675866),
    (72.868365, 19.0675517),
    (72.8684056, 19.0675331),
    (72.8688529, 19.0668792),
]
_MUMBAI_TGT_WGS = [
    (72.8662591430328, 19.0664357013479),
    (72.8665335555516, 19.0666071109827),
    (72.8668575929, 19.0667960206963),
    (72.8671447645236, 19.0669840156765),
    (72.867384158045, 19.0671211095619),
]

_UTM43N = pyproj.Transformer.from_crs(
    "EPSG:4326", "+proj=utm +zone=43 +north +datum=WGS84", always_xy=True
).transform


def _subline_overshoot(ref, target, result):
    """Compute how far each subline extends past the other line's extent.

    Returns (ref_overshoot, target_overshoot) in the same units as the input
    geometries (degrees for WGS84, meters for projected).
    """
    ref_sub = create_subline(ref, result.overture_start_frac, result.overture_end_frac)
    target_sub = create_subline(target, result.dataset_start_frac, result.dataset_end_frac)
    if ref_sub is None or target_sub is None:
        return float("inf"), float("inf")

    # Project target subline endpoints onto ref to find valid ref range
    t_start_proj = ref.project(target_sub.interpolate(0), normalized=True)
    t_end_proj = ref.project(target_sub.interpolate(1, normalized=True), normalized=True)
    ref_clipped_start = max(result.overture_start_frac, min(t_start_proj, t_end_proj))
    ref_clipped_end = min(result.overture_end_frac, max(t_start_proj, t_end_proj))

    ref_overshoot = (
        max(
            abs(result.overture_start_frac - ref_clipped_start),
            abs(result.overture_end_frac - ref_clipped_end),
        )
        * ref.length
    )

    # Project ref subline endpoints onto target to find valid target range
    r_start_proj = target.project(ref_sub.interpolate(0), normalized=True)
    r_end_proj = target.project(ref_sub.interpolate(1, normalized=True), normalized=True)
    tgt_clipped_start = max(result.dataset_start_frac, min(r_start_proj, r_end_proj))
    tgt_clipped_end = min(result.dataset_end_frac, max(r_start_proj, r_end_proj))

    target_overshoot = (
        max(
            abs(result.dataset_start_frac - tgt_clipped_start),
            abs(result.dataset_end_frac - tgt_clipped_end),
        )
        * target.length
    )

    return ref_overshoot, target_overshoot


class TestOveralignment:
    """Tests for barely-overlapping segments that should have small alignment.

    When two lines barely overlap at one endpoint (e.g. collinear roads meeting
    at a junction), the standard midpoint seed projects far from the actual
    overlap zone, so the grid search converges to a suboptimal offset that
    overshoots. The fix evaluates endpoint-based seeds and switches when they
    score significantly better (5x threshold), giving the refinement a starting
    point near the true overlap.
    """

    # -- Parameterized: barely-overlapping pairs should have small coverage --

    @pytest.mark.parametrize(
        "ref_coords, tgt_coords, max_ref_cov, max_tgt_cov, max_overshoot, case_id",
        [
            pytest.param(
                _MUMBAI_REF_WGS,
                _MUMBAI_TGT_WGS,
                0.15,
                0.15,
                1.0 / 111000,  # 1m in degrees
                "mumbai_wgs84",
                id="mumbai-degree-space",
            ),
            pytest.param(
                list(shapely_transform(_UTM43N, LineString(_MUMBAI_REF_WGS)).coords),
                list(shapely_transform(_UTM43N, LineString(_MUMBAI_TGT_WGS)).coords),
                0.05,
                0.05,
                2.0,  # 2m in projected meters
                "mumbai_projected",
                id="mumbai-projected-meters",
            ),
            pytest.param(
                [(95, 0), (200, 0), (300, 0)],
                [(0, 3), (50, 3), (100, 3)],
                0.10,
                0.15,
                1.0,  # 1m in synthetic meters
                "synthetic_collinear",
                id="synthetic-collinear-5m-overlap",
            ),
            pytest.param(
                # Bogota pair bog_road_165093: 360m ref, 72m target at edge
                [
                    (0, 0),
                    (-19, -3.2),
                    (-39.3, -5.6),
                    (-84.3, -7.1),
                    (-93.5, -8),
                    (-101, -8.8),
                    (-112.2, -9.9),
                    (-128.4, -12.8),
                    (-144.9, -18.5),
                    (-180.3, -31.4),
                    (-199.5, -34.5),
                    (-286.5, -17.4),
                    (-353, -4.5),
                ],
                [(-2.2, -3), (7.6, 2.2), (18.1, 10), (28.7, 22.9), (36.7, 40.6), (39.9, 51.5)],
                0.05,
                0.10,
                10.0,  # More generous for angled pair
                "bogota_asymmetric",
                id="bogota-asymmetric-360m-ref",
            ),
        ],
    )
    def test_barely_overlapping_small_coverage(
        self, ref_coords, tgt_coords, max_ref_cov, max_tgt_cov, max_overshoot, case_id
    ):
        """Barely-overlapping pairs should have small coverage and limited overshoot."""
        ref = LineString(ref_coords)
        target = LineString(tgt_coords)

        result = linestring_alignment(ref, target)

        assert result.overture_coverage < max_ref_cov, (
            f"[{case_id}] ref_cov={result.overture_coverage:.4f}, expected < {max_ref_cov}"
        )
        assert result.dataset_coverage < max_tgt_cov, (
            f"[{case_id}] tgt_cov={result.dataset_coverage:.4f}, expected < {max_tgt_cov}"
        )

        ref_overshoot, target_overshoot = _subline_overshoot(ref, target, result)
        assert ref_overshoot < max_overshoot, (
            f"[{case_id}] ref overshoots target by {ref_overshoot:.2f}"
        )
        assert target_overshoot < max_overshoot, (
            f"[{case_id}] target overshoots ref by {target_overshoot:.2f}"
        )

    # -- Parameterized: well-aligned pairs should maintain high coverage --

    @pytest.mark.parametrize(
        "ref_coords, tgt_coords, min_cov",
        [
            pytest.param(
                [(0, 0), (100, 0)],
                [(0, 5), (100, 5)],
                0.90,
                id="parallel-5m-offset",
            ),
            pytest.param(
                [(0, 0), (200, 0)],
                [(0, 5), (200, 5)],
                0.90,
                id="parallel-long",
            ),
            pytest.param(
                [(0, 0), (100, 0)],
                [(100, 0), (0, 0)],
                0.95,
                id="reversed-direction",
            ),
        ],
    )
    def test_high_coverage_not_degraded(self, ref_coords, tgt_coords, min_cov):
        """Well-aligned pairs must maintain high coverage (multi-seed stability)."""
        ref = LineString(ref_coords)
        target = LineString(tgt_coords)

        result = linestring_alignment(ref, target)

        assert result.overture_coverage >= min_cov
        assert result.dataset_coverage >= min_cov

    # -- Individual edge-case tests --

    def test_synthetic_collinear_no_overlap(self):
        """Two collinear parallel lines with a gap — should have near-zero coverage."""
        ref = LineString([(120, 0), (200, 0), (300, 0)])
        target = LineString([(0, 3), (50, 3), (100, 3)])

        result = linestring_alignment(ref, target)
        assert result.overture_coverage < 0.10

    def test_angled_junction_not_clipped(self):
        """Lines meeting at 45 degrees should produce a small but non-zero alignment."""
        ref = LineString([(0, 0), (200, 0)])
        target = LineString([(-70.7, -70.7), (0, 0)])

        result = linestring_alignment(ref, target)

        assert result.overture_coverage > 0.01
        assert result.dataset_coverage > 0.01

    def test_angled_junction_with_parallel_section(self):
        """Road approaching at 45 degrees then running parallel should align the parallel part."""
        ref = LineString([(0, 0), (200, 0)])
        target = LineString([(-30, -30), (0, 3), (20, 3)])

        result = linestring_alignment(ref, target)

        ref_aligned_m = result.overture_coverage * ref.length
        assert ref_aligned_m >= 15, f"Expected ~20m aligned, got {ref_aligned_m:.1f}m"

    def test_small_overshoot_not_clipped(self):
        """A 12m overlap should not be over-trimmed."""
        ref = LineString([(88, 0), (200, 0)])
        target = LineString([(0, 3), (100, 3)])

        result = linestring_alignment(ref, target)

        ref_aligned_m = result.overture_coverage * ref.length
        assert ref_aligned_m >= 10, f"Expected ~12m aligned, got {ref_aligned_m:.1f}m"

    def test_curved_barely_overlapping_not_collapsed(self):
        """Curved roads that barely overlap should still produce an alignment."""
        ref_pts = [(x, 5 * math.sin(x / 30)) for x in range(0, 201, 10)]
        tgt_pts = [(x, -5 * math.sin((-x) / 20) + 2) for x in range(-100, 6, 10)]
        ref = LineString(ref_pts)
        target = LineString(tgt_pts)

        result = linestring_alignment(ref, target)

        assert result.overture_coverage > 0.005 or result.dataset_coverage > 0.005
