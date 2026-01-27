"""Tests for the DatasetLoader utility."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from matcher.config import DATA_VERSION
from matcher.datasets.loader import DatasetLoader, LoadedPair

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_linestring_gdf(
    n: int = 5,
    *,
    crs: str | None = "EPSG:4326",
    include_names: bool = True,
    include_class: bool = True,
) -> gpd.GeoDataFrame:
    """Create a minimal GeoDataFrame with LineString geometries."""
    data = {
        "id": [f"seg_{i}" for i in range(n)],
        "geometry": [LineString([(i, 0), (i + 1, 0)]) for i in range(n)],
    }
    if include_names:
        data["names"] = [f"Road {i}" for i in range(n)]
    if include_class:
        data["class"] = ["residential"] * n
    gdf = gpd.GeoDataFrame(data)
    if crs:
        gdf = gdf.set_crs(crs)
    return gdf


def _write_parquet(gdf: gpd.GeoDataFrame, path):
    """Write a GeoDataFrame to parquet."""
    gdf.to_parquet(path)


@pytest.fixture()
def data_dir(tmp_path):
    """Create a temporary data directory with a standard dataset pair."""
    raw = tmp_path / "raw"
    raw.mkdir()

    ref_gdf = _make_linestring_gdf(5)
    target_gdf = _make_linestring_gdf(5)

    # Overture reference: us_boston_overture_segments_v1.0.parquet
    _write_parquet(ref_gdf, raw / f"us_boston_overture_segments_{DATA_VERSION}.parquet")

    # Target: us_boston_streets_v1.0.parquet
    _write_parquet(target_gdf, raw / f"us_boston_streets_{DATA_VERSION}.parquet")

    return raw


@pytest.fixture()
def data_dir_osm(data_dir):
    """Extend data_dir with an OSM segments file."""
    osm_gdf = _make_linestring_gdf(3)
    _write_parquet(
        osm_gdf,
        data_dir / f"us_boston_streets_osm_segments_{DATA_VERSION}.parquet",
    )
    return data_dir


# ---------------------------------------------------------------------------
# Path resolution tests
# ---------------------------------------------------------------------------


class TestResolvePaths:
    def test_resolve_paths_standard_dataset(self, data_dir):
        """Standard dataset resolves to correct ref + target paths."""
        loader = DatasetLoader(data_dir)
        ref, target = loader._resolve_paths("us_boston_streets")
        assert ref.name == f"us_boston_overture_segments_{DATA_VERSION}.parquet"
        assert target.name == f"us_boston_streets_{DATA_VERSION}.parquet"

    def test_resolve_paths_osm_variant(self, data_dir_osm):
        """OSM suffix dataset uses base Overture as reference and OSM segments as target."""
        loader = DatasetLoader(data_dir_osm)
        ref, target = loader._resolve_paths("us_boston_streets_osm")
        assert "overture_segments" in ref.name
        assert "osm_segments" in target.name

    def test_resolve_paths_missing_reference(self, tmp_path):
        """FileNotFoundError with actionable message when reference missing."""
        raw = tmp_path / "raw"
        raw.mkdir()
        # Only create target, no reference
        _write_parquet(
            _make_linestring_gdf(2),
            raw / f"us_nowhere_streets_{DATA_VERSION}.parquet",
        )
        loader = DatasetLoader(raw)
        with pytest.raises(FileNotFoundError, match="Reference file not found"):
            loader._resolve_paths("us_nowhere_streets")

    def test_resolve_paths_missing_target(self, data_dir):
        """FileNotFoundError with actionable message when target missing."""
        loader = DatasetLoader(data_dir)
        with pytest.raises(FileNotFoundError, match="Target file not found"):
            loader._resolve_paths("us_boston_sidewalks")


# ---------------------------------------------------------------------------
# GDF loading tests
# ---------------------------------------------------------------------------


class TestLoadGdf:
    def test_load_gdf_sets_default_crs(self, tmp_path):
        """Default CRS to WGS84 when source has no CRS."""
        gdf = _make_linestring_gdf(3, crs=None)
        path = tmp_path / "no_crs.parquet"
        # Write without CRS — geopandas will not embed CRS metadata
        gdf.to_parquet(path)

        loader = DatasetLoader(tmp_path)
        loaded = loader._load_gdf(path)
        assert loaded.crs is not None
        assert loaded.crs.to_epsg() == 4326

    def test_load_gdf_filters_linestrings(self, tmp_path):
        """Non-LineString geometries are dropped."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["ls", "ml", "pt"],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    MultiLineString([[(2, 2), (3, 3)], [(4, 4), (5, 5)]]),
                    Point(6, 6),
                ],
            },
            crs="EPSG:4326",
        )
        path = tmp_path / "mixed.parquet"
        gdf.to_parquet(path)

        loader = DatasetLoader(tmp_path)
        loaded = loader._load_gdf(path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["id"] == "ls"

    def test_load_gdf_empty_after_filter(self, tmp_path):
        """ValueError when no LineStrings remain after filtering."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["pt"],
                "geometry": [Point(0, 0)],
            },
            crs="EPSG:4326",
        )
        path = tmp_path / "points_only.parquet"
        gdf.to_parquet(path)

        loader = DatasetLoader(tmp_path)
        with pytest.raises(ValueError, match="No LineString geometries remaining"):
            loader._load_gdf(path)

    def test_load_gdf_raises_missing_required_columns(self, tmp_path):
        """ValueError raised when required column 'id' is missing."""
        gdf = gpd.GeoDataFrame(
            {
                "other": ["a"],
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            crs="EPSG:4326",
        )
        path = tmp_path / "no_id.parquet"
        gdf.to_parquet(path)

        loader = DatasetLoader(tmp_path)
        with pytest.raises(ValueError, match="missing required columns"):
            loader._load_gdf(path)


# ---------------------------------------------------------------------------
# load_pair tests
# ---------------------------------------------------------------------------


class TestLoadPair:
    def test_load_pair_returns_loaded_pair(self, data_dir):
        """load_pair returns a LoadedPair with correct fields."""
        loader = DatasetLoader(data_dir)
        pair = loader.load_pair("us_boston_streets")

        assert isinstance(pair, LoadedPair)
        assert pair.dataset_id == "us_boston_streets"
        assert len(pair.reference) == 5
        assert len(pair.target) == 5
        assert pair.reference_path.exists()
        assert pair.target_path.exists()
        # Default: geographic CRS (WGS84)
        assert pair.reference.crs.is_geographic

    def test_load_pair_project(self, data_dir):
        """project=True returns data in a projected (non-geographic) CRS."""
        loader = DatasetLoader(data_dir)
        pair = loader.load_pair("us_boston_streets", project=True)

        assert not pair.reference.crs.is_geographic
        assert not pair.target.crs.is_geographic


# ---------------------------------------------------------------------------
# Session caching tests
# ---------------------------------------------------------------------------


class TestSessionCaching:
    def test_session_caching(self, data_dir):
        """Within a session, the same file is loaded from cache."""
        loader = DatasetLoader(data_dir)
        with loader.session():
            ref1 = loader.load_reference("us_boston_streets")
            ref2 = loader.load_reference("us_boston_streets")
            # Same object identity means cache hit
            assert ref1 is ref2

    def test_session_cleanup(self, data_dir):
        """Cache is cleared after exiting the session context."""
        loader = DatasetLoader(data_dir)
        with loader.session():
            _ = loader.load_reference("us_boston_streets")
            assert loader._cache is not None
            assert len(loader._cache) > 0

        # After context exit
        assert loader._cache is None


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestListAvailable:
    def test_list_available(self, data_dir_osm, monkeypatch):
        """list_available includes both standard and OSM datasets."""
        # list_available() does: from .schema import list_dataset_configs as _list_yaml
        # so we monkeypatch the function on the schema module
        monkeypatch.setattr(
            "matcher.datasets.schema.list_dataset_configs",
            lambda: ["us_boston_streets"],
        )
        loader = DatasetLoader(data_dir_osm)
        available = loader.list_available()

        assert "us_boston_streets" in available
        assert "us_boston_streets_osm" in available

    def test_find_reference_path_not_found(self, tmp_path):
        """find_reference_path returns None when no file exists."""
        raw = tmp_path / "empty"
        raw.mkdir()
        loader = DatasetLoader(raw)
        assert loader.find_reference_path("nonexistent") is None
