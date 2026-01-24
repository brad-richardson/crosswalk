"""Tests for the filenames module."""

from pathlib import Path

import pytest

from matcher.config import DATA_VERSION
from matcher.filenames import (
    bridge_filename,
    extract_version_from_filename,
    find_osm_segments,
    find_overture_segments,
    find_target_file,
    osm_connectors_filename,
    osm_segments_filename,
    overture_connectors_filename,
    overture_segments_filename,
    target_filename,
)


class TestFilenamePatterns:
    """Test filename generation functions."""

    def test_target_filename(self):
        """Target filename should include version suffix."""
        result = target_filename("us_boston_streets")
        assert result == f"us_boston_streets_{DATA_VERSION}.parquet"

    def test_overture_segments_filename(self):
        """Overture segments filename should include version suffix."""
        result = overture_segments_filename("us_boston")
        assert result == f"us_boston_overture_segments_{DATA_VERSION}.parquet"

    def test_overture_connectors_filename(self):
        """Overture connectors filename should include version suffix."""
        result = overture_connectors_filename("us_boston")
        assert result == f"us_boston_overture_connectors_{DATA_VERSION}.parquet"

    def test_osm_segments_filename(self):
        """OSM segments filename should include version suffix."""
        result = osm_segments_filename("us_boston_streets")
        assert result == f"us_boston_streets_osm_segments_{DATA_VERSION}.parquet"

    def test_osm_connectors_filename(self):
        """OSM connectors filename should include version suffix."""
        result = osm_connectors_filename("us_boston_streets")
        assert result == f"us_boston_streets_osm_connectors_{DATA_VERSION}.parquet"

    def test_bridge_filename_no_version(self):
        """Bridge filename should NOT include version suffix (output file)."""
        result = bridge_filename("us_boston_streets")
        assert result == "us_boston_streets_bridge.parquet"
        assert DATA_VERSION not in result


class TestExtractVersion:
    """Test version extraction from filenames."""

    def test_extract_version_from_versioned_file(self):
        """Should extract version from versioned filename."""
        path = Path("us_boston_streets_v1.0.parquet")
        assert extract_version_from_filename(path) == "1.0"

    def test_extract_version_from_complex_name(self):
        """Should extract version from complex filename."""
        path = Path("us_fort_collins_sidewalks_overture_segments_v2.3.parquet")
        assert extract_version_from_filename(path) == "2.3"

    def test_extract_version_returns_none_for_unversioned(self):
        """Should return None for unversioned filename."""
        path = Path("us_boston_streets.parquet")
        assert extract_version_from_filename(path) is None

    def test_extract_version_handles_multiple_underscores(self):
        """Should handle filenames with multiple underscores."""
        path = Path("us_boston_bike_network_osm_segments_v1.0.parquet")
        assert extract_version_from_filename(path) == "1.0"

    def test_extract_version_with_longer_version(self):
        """Should handle longer version numbers."""
        path = Path("dataset_v12.345.parquet")
        assert extract_version_from_filename(path) == "12.345"

    def test_extract_version_rejects_invalid_version_format(self):
        """Should return None if version part contains non-version chars."""
        # This tests the case where _v is followed by something that's not a version
        path = Path("some_variant_file.parquet")  # Has 'v' but not in version format
        assert extract_version_from_filename(path) is None


class TestFileDiscovery:
    """Test file discovery functions."""

    def test_find_overture_segments_exact_match(self, tmp_path):
        """Should find exact match for versioned Overture file."""
        # Create a versioned file
        file_path = tmp_path / overture_segments_filename("us_boston_streets")
        file_path.touch()

        result = find_overture_segments(tmp_path, "us_boston_streets")
        assert result == file_path

    def test_find_overture_segments_prefix_match(self, tmp_path):
        """Should find prefix match when exact match doesn't exist."""
        # Create file for shorter prefix
        file_path = tmp_path / overture_segments_filename("us_boston")
        file_path.touch()

        result = find_overture_segments(tmp_path, "us_boston_streets")
        assert result == file_path

    def test_find_overture_segments_returns_none_when_not_found(self, tmp_path):
        """Should return None when no matching file exists."""
        result = find_overture_segments(tmp_path, "nonexistent_dataset")
        assert result is None

    def test_find_osm_segments(self, tmp_path):
        """Should find OSM segments file."""
        file_path = tmp_path / osm_segments_filename("us_boston_streets")
        file_path.touch()

        result = find_osm_segments(tmp_path, "us_boston_streets")
        assert result == file_path

    def test_find_osm_segments_returns_none_when_not_found(self, tmp_path):
        """Should return None when no matching file exists."""
        result = find_osm_segments(tmp_path, "nonexistent_dataset")
        assert result is None

    def test_find_target_file(self, tmp_path):
        """Should find target file."""
        file_path = tmp_path / target_filename("us_boston_streets")
        file_path.touch()

        result = find_target_file(tmp_path, "us_boston_streets")
        assert result == file_path

    def test_find_target_file_returns_none_when_not_found(self, tmp_path):
        """Should return None when no matching file exists."""
        result = find_target_file(tmp_path, "nonexistent_dataset")
        assert result is None
