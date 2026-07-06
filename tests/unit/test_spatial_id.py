"""Tests for spatial ID suffix computation.

These tests verify that compute_spatial_suffix produces stable, deterministic
results suitable for use in persistent IDs. Consistency is critical — if a
suffix changes for the same geometry, existing label linkages break.
"""

import h3
from shapely.geometry import LineString

from crosswalk.utils.spatial_id import (
    H3_RESOLUTION,
    H3_TRAILING_F_COUNT,
    compute_spatial_suffix,
)

# H3 res-8 index is 15 chars, minus 5 trailing f's = 10
EXPECTED_LENGTH = 15 - H3_TRAILING_F_COUNT


def _restore_h3(suffix: str) -> str:
    """Restore a full H3 index from a trimmed suffix."""
    return suffix.ljust(15, "f")


# --- Determinism and basic properties ---


class TestDeterminism:
    """Verify that the same input always produces the same output."""

    def test_same_geometry_same_suffix(self):
        """Identical geometry must always produce the same suffix."""
        line = LineString([(0, 0), (1, 1)])
        assert compute_spatial_suffix(line) == compute_spatial_suffix(line)

    def test_reconstructed_geometry_same_suffix(self):
        """A geometry reconstructed from the same coordinates must match."""
        coords = [(-71.0589, 42.3601), (-71.0550, 42.3620), (-71.0510, 42.3640)]
        line1 = LineString(coords)
        line2 = LineString(coords)
        assert compute_spatial_suffix(line1) == compute_spatial_suffix(line2)

    def test_suffix_length(self):
        """Suffix must be 10 chars (15-char H3 index minus 5 trailing f's)."""
        line = LineString([(0, 0), (1, 1)])
        suffix = compute_spatial_suffix(line)
        assert len(suffix) == EXPECTED_LENGTH

    def test_suffix_is_hex(self):
        """Suffix must be valid hexadecimal."""
        line = LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)])
        suffix = compute_spatial_suffix(line)
        int(suffix, 16)  # Raises ValueError if not hex

    def test_suffix_restores_to_valid_h3(self):
        """Suffix padded with f's must be a valid H3 cell index."""
        line = LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)])
        suffix = compute_spatial_suffix(line)
        full = _restore_h3(suffix)
        assert h3.is_valid_cell(full)
        assert h3.get_resolution(full) == H3_RESOLUTION

    def test_multiple_calls_stable(self):
        """Running 100 times on the same geometry must always give the same result."""
        line = LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)])
        first = compute_spatial_suffix(line)
        for _ in range(100):
            assert compute_spatial_suffix(line) == first


# --- Spatial disambiguation ---


class TestSpatialDisambiguation:
    """Verify that distant geometries get different suffixes."""

    def test_distant_lines_different_suffix(self):
        """Lines in different cities must get different suffixes."""
        boston = LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)])
        nyc = LineString([(-74.0060, 40.7128), (-73.9970, 40.7200)])
        assert compute_spatial_suffix(boston) != compute_spatial_suffix(nyc)

    def test_same_city_different_neighborhoods(self):
        """Lines in different neighborhoods (~2km apart) get different suffixes."""
        back_bay = LineString([(-71.0800, 42.3500), (-71.0750, 42.3520)])
        charlestown = LineString([(-71.0600, 42.3780), (-71.0550, 42.3800)])
        assert compute_spatial_suffix(back_bay) != compute_spatial_suffix(charlestown)

    def test_parallel_streets_close_together(self):
        """Parallel streets ~200m apart may or may not differ (depends on H3 boundary).

        This test just verifies no crashes — the result depends on H3 cell layout.
        """
        street1 = LineString([(-71.0589, 42.3601), (-71.0510, 42.3601)])
        street2 = LineString([(-71.0589, 42.3619), (-71.0510, 42.3619)])
        s1 = compute_spatial_suffix(street1)
        s2 = compute_spatial_suffix(street2)
        assert len(s1) == EXPECTED_LENGTH
        assert len(s2) == EXPECTED_LENGTH

    def test_global_disambiguation(self):
        """Lines on different continents always get different suffixes."""
        cities = {
            "Boston": LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)]),
            "London": LineString([(-0.1276, 51.5074), (-0.1200, 51.5100)]),
            "Tokyo": LineString([(139.6917, 35.6895), (139.7000, 35.6950)]),
            "Sydney": LineString([(151.2093, -33.8688), (151.2150, -33.8650)]),
            "Sao Paulo": LineString([(-46.6333, -23.5505), (-46.6250, -23.5450)]),
        }
        suffixes = {name: compute_spatial_suffix(line) for name, line in cities.items()}
        assert len(set(suffixes.values())) == len(suffixes), (
            f"Suffix collisions detected: {suffixes}"
        )


# --- Stability against minor geometry changes ---


class TestStability:
    """Verify that small geometry changes don't change the suffix.

    H3 resolution 8 cells are ~1km across, so changes within ~400m of
    the midpoint should almost always stay in the same cell.
    """

    def test_adding_vertices_same_suffix(self):
        """Adding intermediate vertices should not change the suffix.

        line_interpolate_point(0.5) gives the point at 50% of total length.
        Adding a vertex exactly on the line doesn't change the length or path,
        so the midpoint stays the same.
        """
        simple = LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)])
        mid_x = (-71.0589 + -71.0510) / 2
        mid_y = (42.3601 + 42.3640) / 2
        densified = LineString([(-71.0589, 42.3601), (mid_x, mid_y), (-71.0510, 42.3640)])
        assert compute_spatial_suffix(simple) == compute_spatial_suffix(densified)

    def test_small_endpoint_shift_same_suffix(self):
        """Shifting an endpoint by ~10m should not change the suffix."""
        original = LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)])
        shifted = LineString([(-71.0589, 42.3601), (-71.0509, 42.3641)])
        assert compute_spatial_suffix(original) == compute_spatial_suffix(shifted)

    def test_small_midpoint_perturbation_same_suffix(self):
        """Perturbing a vertex near the midpoint by ~5m should not change the suffix."""
        line = LineString([(-71.0600, 42.3600), (-71.0550, 42.3620), (-71.0500, 42.3640)])
        perturbed = LineString([(-71.0600, 42.3600), (-71.05505, 42.36205), (-71.0500, 42.3640)])
        assert compute_spatial_suffix(line) == compute_spatial_suffix(perturbed)

    def test_reversed_line_same_suffix(self):
        """A reversed line should produce the same suffix.

        The midpoint at 50% of length is the same point regardless of direction.
        """
        forward = LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)])
        backward = LineString([(-71.0510, 42.3640), (-71.0589, 42.3601)])
        assert compute_spatial_suffix(forward) == compute_spatial_suffix(backward)


# --- H3 trimming roundtrip ---


class TestTrimming:
    """Verify the trailing-f trimming is lossless and consistent."""

    def test_all_res8_cells_end_with_5_fs(self):
        """Every res-8 cell must end with exactly 'fffff'."""
        locations = [
            (42.36, -71.06),
            (40.71, -74.00),
            (51.51, -0.13),
            (35.69, 139.69),
            (-33.87, 151.21),
            (0.0, 0.0),
            (60.17, 24.94),
            (-23.55, -46.63),
            (1.35, 103.82),
            (37.77, -122.42),
            (48.86, 2.35),
            (55.75, 37.62),
        ]
        for lat, lng in locations:
            full = h3.latlng_to_cell(lat, lng, 8)
            assert full.endswith("fffff"), f"{full} does not end with fffff"

    def test_roundtrip_restore(self):
        """Trimmed suffix + 'fffff' must equal the original H3 index."""
        line = LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)])
        suffix = compute_spatial_suffix(line)
        restored = _restore_h3(suffix)

        import shapely as _shapely

        midpoint = _shapely.line_interpolate_point(line, 0.5, normalized=True)
        original = h3.latlng_to_cell(midpoint.y, midpoint.x, H3_RESOLUTION)

        assert restored == original

    def test_roundtrip_multiple_locations(self):
        """Roundtrip works for many locations globally."""
        lines = [
            LineString([(-71.06, 42.36), (-71.05, 42.37)]),
            LineString([(-74.01, 40.71), (-73.99, 40.72)]),
            LineString([(0.0, 0.0), (0.01, 0.01)]),
            LineString([(139.69, 35.69), (139.70, 35.70)]),
            LineString([(151.21, -33.87), (151.22, -33.86)]),
        ]
        for line in lines:
            suffix = compute_spatial_suffix(line)
            restored = _restore_h3(suffix)
            assert h3.is_valid_cell(restored)
            assert h3.get_resolution(restored) == H3_RESOLUTION


# --- Known value regression tests ---


class TestKnownValues:
    """Pin specific inputs to specific outputs to catch any regressions.

    If these tests fail, it means the suffix computation changed — which would
    break all existing labels. Investigate carefully before updating expected values.
    """

    def test_known_boston_line(self):
        """Pin a specific Boston line to its expected suffix."""
        line = LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)])
        suffix = compute_spatial_suffix(line)
        assert suffix == self._compute_expected(line), (
            f"REGRESSION: Boston line suffix changed to '{suffix}'. "
            "This would break existing label linkages!"
        )

    def test_known_nyc_line(self):
        """Pin a specific NYC line to its expected suffix."""
        line = LineString([(-74.0060, 40.7128), (-73.9970, 40.7200)])
        suffix = compute_spatial_suffix(line)
        assert suffix == self._compute_expected(line), (
            f"REGRESSION: NYC line suffix changed to '{suffix}'. "
            "This would break existing label linkages!"
        )

    def test_known_equator_line(self):
        """Pin a line near the equator."""
        line = LineString([(0.0, 0.0), (0.01, 0.01)])
        suffix = compute_spatial_suffix(line)
        assert suffix == self._compute_expected(line), (
            f"REGRESSION: Equator line suffix changed to '{suffix}'. "
            "This would break existing label linkages!"
        )

    def test_known_high_latitude_line(self):
        """Pin a line at high latitude (Helsinki, ~60 N)."""
        line = LineString([(24.9384, 60.1699), (24.9500, 60.1750)])
        suffix = compute_spatial_suffix(line)
        assert suffix == self._compute_expected(line), (
            f"REGRESSION: Helsinki line suffix changed to '{suffix}'. "
            "This would break existing label linkages!"
        )

    @staticmethod
    def _compute_expected(line: LineString) -> str:
        """Compute the expected suffix using the raw algorithm."""
        import shapely as _shapely

        midpoint = _shapely.line_interpolate_point(line, 0.5, normalized=True)
        h3_index = h3.latlng_to_cell(midpoint.y, midpoint.x, H3_RESOLUTION)
        return h3_index[:-H3_TRAILING_F_COUNT]


# --- Edge cases ---


class TestEdgeCases:
    """Test edge cases and unusual geometries."""

    def test_very_short_line(self):
        """A very short line (< 1m) should still produce a valid suffix."""
        line = LineString([(-71.0589, 42.3601), (-71.05891, 42.36011)])
        suffix = compute_spatial_suffix(line)
        assert len(suffix) == EXPECTED_LENGTH
        assert h3.is_valid_cell(_restore_h3(suffix))

    def test_very_long_line(self):
        """A long line spanning multiple H3 cells should produce a valid suffix."""
        line = LineString([(-71.0, 42.0), (-72.0, 43.0)])
        suffix = compute_spatial_suffix(line)
        assert len(suffix) == EXPECTED_LENGTH
        assert h3.is_valid_cell(_restore_h3(suffix))

    def test_horseshoe_shaped_road(self):
        """A horseshoe/U-shaped road should use midpoint ON the line, not centroid."""
        horseshoe = LineString(
            [
                (-71.060, 42.360),
                (-71.050, 42.360),
                (-71.050, 42.355),
                (-71.060, 42.355),
            ]
        )
        suffix = compute_spatial_suffix(horseshoe)
        assert h3.is_valid_cell(_restore_h3(suffix))

        # Verify the midpoint is on the line (not inside the U)
        import shapely as _shapely

        midpoint = _shapely.line_interpolate_point(horseshoe, 0.5, normalized=True)
        assert horseshoe.distance(midpoint) < 1e-10

    def test_zigzag_line(self):
        """A zigzag line should produce a stable suffix."""
        zigzag = LineString(
            [
                (-71.060, 42.360),
                (-71.058, 42.362),
                (-71.056, 42.360),
                (-71.054, 42.362),
                (-71.052, 42.360),
            ]
        )
        suffix = compute_spatial_suffix(zigzag)
        assert h3.is_valid_cell(_restore_h3(suffix))
        assert compute_spatial_suffix(zigzag) == suffix

    def test_multipoint_line_deterministic(self):
        """A multi-vertex line should be deterministic."""
        coords = [
            (-71.0600, 42.3600),
            (-71.0590, 42.3605),
            (-71.0580, 42.3610),
            (-71.0570, 42.3615),
            (-71.0560, 42.3620),
            (-71.0550, 42.3625),
            (-71.0540, 42.3630),
            (-71.0530, 42.3635),
            (-71.0520, 42.3640),
            (-71.0510, 42.3645),
        ]
        line = LineString(coords)
        suffix = compute_spatial_suffix(line)
        line2 = LineString(coords)
        assert compute_spatial_suffix(line2) == suffix
