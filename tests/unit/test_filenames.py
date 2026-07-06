"""Tests for filenames module."""

import ast
import inspect

from crosswalk.filenames import find_overture_segments


class TestFindOvertureSegments:
    """Tests for find_overture_segments function."""

    def test_no_glob_fallback_in_source(self):
        """Ensure find_overture_segments does not use glob fallback.

        The function should only return files matching the exact DATA_VERSION,
        not fall back to glob patterns like v*.parquet which could return
        files with mismatched versions.
        """
        # Get the source code of the function
        source = inspect.getsource(find_overture_segments)

        # Parse the source to check for glob calls
        tree = ast.parse(source)

        # Find all calls to .glob() method
        glob_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "glob":
                        glob_calls.append(node)

        assert len(glob_calls) == 0, (
            "find_overture_segments should not use glob() fallback. "
            "This defeats the purpose of versioned filenames - use exact version match only."
        )

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        """Returns None when no matching file exists."""
        result = find_overture_segments(tmp_path, "us_boston_streets")
        assert result is None

    def test_finds_exact_version_match(self, tmp_path):
        """Finds file with exact version match."""
        from crosswalk.config import DATA_VERSION

        # Create a file with the current version
        expected_file = tmp_path / f"us_boston_overture_segments_{DATA_VERSION}.parquet"
        expected_file.touch()

        result = find_overture_segments(tmp_path, "us_boston_streets")
        assert result == expected_file

    def test_does_not_find_different_version(self, tmp_path):
        """Does NOT find files with different version suffix."""
        # Create a file with a different version
        wrong_version_file = tmp_path / "us_boston_overture_segments_v99.99.parquet"
        wrong_version_file.touch()

        result = find_overture_segments(tmp_path, "us_boston_streets")
        # Should return None because exact version doesn't exist
        assert result is None

    def test_progressive_prefix_matching(self, tmp_path):
        """Finds file with progressively shorter prefixes."""
        from crosswalk.config import DATA_VERSION

        # Create a file with shorter prefix (us_boston instead of us_boston_streets)
        expected_file = tmp_path / f"us_boston_overture_segments_{DATA_VERSION}.parquet"
        expected_file.touch()

        result = find_overture_segments(tmp_path, "us_boston_streets")
        assert result == expected_file

    def test_prefers_longer_prefix_match(self, tmp_path):
        """Prefers longer prefix match over shorter one."""
        from crosswalk.config import DATA_VERSION

        # Create both a longer and shorter prefix match
        longer_match = tmp_path / f"us_boston_streets_overture_segments_{DATA_VERSION}.parquet"
        shorter_match = tmp_path / f"us_boston_overture_segments_{DATA_VERSION}.parquet"
        longer_match.touch()
        shorter_match.touch()

        result = find_overture_segments(tmp_path, "us_boston_streets")
        # Should find the longer prefix first (exact dataset name match)
        assert result == longer_match
