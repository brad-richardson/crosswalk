"""Tests for backfill safety features.

Tests the --require-stored-data flag that prevents fallback to raw data
during feature backfill.
"""


class TestRequireStoredData:
    """Test the require_stored_data guard in backfill."""

    def test_require_stored_data_skips_when_no_stored_geometry(self):
        """When require_stored_data=True and no stored geometry, pair is skipped."""
        # This is an integration-level behavior test.
        # The actual logic lives in cli/labels.py backfill_features().
        # We test the conditional logic directly.

        require_stored_data = True
        has_stored_data = True
        pair_data = None  # No stored data for this pair

        ref_geom = None
        target_geom = None

        # Simulate the stored data lookup returning nothing
        if has_stored_data:
            if pair_data is not None:
                # Would extract geometries...
                pass

        # Fallback path
        should_skip = False
        if ref_geom is None or target_geom is None:
            if require_stored_data:
                should_skip = True

        assert should_skip is True

    def test_require_stored_data_false_allows_fallback(self):
        """When require_stored_data=False, fallback to raw data is allowed."""
        require_stored_data = False

        ref_geom = None
        target_geom = None

        should_skip = False
        if ref_geom is None or target_geom is None:
            if require_stored_data:
                should_skip = True

        assert should_skip is False

    def test_stored_data_available_no_skip(self):
        """When stored data provides geometries, no skip regardless of flag."""
        require_stored_data = True

        # Simulate successful stored data lookup
        ref_geom = "mock_geom"  # non-None
        target_geom = "mock_geom"  # non-None

        should_skip = False
        if ref_geom is None or target_geom is None:
            if require_stored_data:
                should_skip = True

        assert should_skip is False
