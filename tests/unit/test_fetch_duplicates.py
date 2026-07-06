"""Tests for duplicate ID handling in the fetch pipeline."""

import geopandas as gpd
from shapely.geometry import LineString

from crosswalk.datasets.schema import FetchConfig


def _make_gdf(ids: list[str], geometries: list[LineString] | None = None) -> gpd.GeoDataFrame:
    """Create a simple GeoDataFrame for testing dedup."""
    n = len(ids)
    if geometries is None:
        geometries = [LineString([(i, 0), (i + 1, 1)]) for i in range(n)]
    return gpd.GeoDataFrame(
        {
            "id": ids,
            "names": [None] * n,
            "class": ["road"] * n,
            "subtype": ["road"] * n,
            "sources": [[{"dataset": "test", "record_id": str(i)}] for i in range(n)],
            "road_flags": [[] for _ in range(n)],
            "level_rules": [[] for _ in range(n)],
            "source_tags": [{}] * n,
            "subclass": [None] * n,
            "status": [None] * n,
            "oneway": [None] * n,
            "speed_limit_kph": [None] * n,
        },
        geometry=geometries,
        crs="EPSG:4326",
    )


class TestArcgisDedup:
    """Test dedup behavior in arcgis.py."""

    def test_unique_ids_no_changes(self):
        """No duplicates -> no dedup, no log."""
        gdf = _make_gdf(["a_1", "a_2", "a_3"])
        n_before = len(gdf)
        result = gdf.drop_duplicates(subset=["id"], keep="first")
        assert len(result) == n_before
        assert len(result) == 3

    def test_duplicate_ids_deduped(self):
        """Duplicate IDs should be deduplicated, keeping first occurrence."""
        gdf = _make_gdf(
            ["a_1", "a_2", "a_1"],
            [
                LineString([(0, 0), (1, 1)]),
                LineString([(2, 2), (3, 3)]),
                LineString([(4, 4), (5, 5)]),  # Different geom, same ID
            ],
        )
        result = gdf.drop_duplicates(subset=["id"], keep="first")
        assert len(result) == 2
        # First occurrence should be kept
        assert result.iloc[0].geometry.equals(LineString([(0, 0), (1, 1)]))


class TestTargetTransformDedup:
    """Test dedup behavior in target.py _transform_download_data."""

    def test_transform_deduplicates_true_duplicates(self):
        """_transform_download_data should deduplicate when same upstream ID + same H3 cell."""
        from crosswalk.fetch.target import _transform_download_data

        # Create input with duplicate OBJECTID values at the SAME location
        # (same H3 cell -> same composite ID -> deduped)
        gdf = gpd.GeoDataFrame(
            {
                "OBJECTID": [100, 200, 100],
                "NAME": ["First St", "Second St", "First St Dup"],
            },
            geometry=[
                LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)]),
                LineString([(-74.0060, 40.7128), (-73.9970, 40.7200)]),
                LineString([(-71.0590, 42.3602), (-71.0511, 42.3641)]),  # ~10m from first
            ],
            crs="EPSG:4326",
        )

        fetch_config = FetchConfig(
            id_prefix="test",
            name_column="NAME",
            id_column="OBJECTID",
        )
        result = _transform_download_data(
            gdf,
            fetch_config=fetch_config,
            source_name="Test",
        )

        # Same upstream ID + same H3 cell -> same composite ID -> deduped to 2
        assert len(result) == 2
        id_list = result["id"].tolist()
        assert all(id_.startswith("test_100_") or id_.startswith("test_200_") for id_ in id_list)

    def test_transform_disambiguates_different_locations(self):
        """Duplicate upstream IDs at different locations get different composite IDs."""
        from crosswalk.fetch.target import _transform_download_data

        # Same OBJECTID but very different locations (Boston vs NYC)
        gdf = gpd.GeoDataFrame(
            {
                "OBJECTID": [100, 200, 100],
                "NAME": ["First St", "Second St", "First St NYC"],
            },
            geometry=[
                LineString([(-71.0589, 42.3601), (-71.0510, 42.3640)]),  # Boston
                LineString([(-74.0060, 40.7128), (-73.9970, 40.7200)]),  # NYC
                LineString([(-74.0050, 40.7130), (-73.9960, 40.7190)]),  # Also NYC
            ],
            crs="EPSG:4326",
        )

        fetch_config = FetchConfig(
            id_prefix="test",
            name_column="NAME",
            id_column="OBJECTID",
        )
        result = _transform_download_data(
            gdf,
            fetch_config=fetch_config,
            source_name="Test",
        )

        # Different H3 cells -> different suffixes -> all 3 kept
        assert len(result) == 3
        id_list = result["id"].tolist()
        assert len(id_list) == len(set(id_list))  # All unique

    def test_transform_no_dedup_for_unique(self):
        """No dedup needed when all IDs are unique."""
        from crosswalk.fetch.target import _transform_download_data

        gdf = gpd.GeoDataFrame(
            {
                "OBJECTID": [100, 200, 300],
                "NAME": ["A", "B", "C"],
            },
            geometry=[
                LineString([(0, 0), (1, 1)]),
                LineString([(2, 2), (3, 3)]),
                LineString([(4, 4), (5, 5)]),
            ],
            crs="EPSG:4326",
        )

        fetch_config = FetchConfig(
            id_prefix="test",
            name_column="NAME",
            id_column="OBJECTID",
        )
        result = _transform_download_data(
            gdf,
            fetch_config=fetch_config,
            source_name="Test",
        )

        assert len(result) == 3
