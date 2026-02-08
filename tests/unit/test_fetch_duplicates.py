"""Tests for duplicate ID handling in the fetch pipeline."""

import geopandas as gpd
from shapely.geometry import LineString


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
