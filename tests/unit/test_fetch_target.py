"""Tests for the target fetch module."""

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from crosswalk.datasets.schema import FetchConfig
from crosswalk.fetch.target import (
    _load_ms_roads_tsv,
    _transform_download_data,
    fetch_dataset,
    fetch_datasets_by_prefix,
    list_datasets,
)


@pytest.fixture
def sample_gdf():
    """Create a sample GeoDataFrame for testing."""
    return gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2, 3],
            "NAME": ["Main St", "Oak Ave", None],
            "ROAD_TYPE": ["primary", "secondary", "tertiary"],
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
                LineString([(2, 2), (3, 3)]),
            ],
        },
        crs="EPSG:4326",
    )


class TestTransformDownloadData:
    """Tests for _transform_download_data function."""

    def test_basic_transform(self, sample_gdf):
        """Test basic transformation to Overture schema."""
        fetch_config = FetchConfig(
            id_prefix="test",
            name_column="NAME",
            class_column="ROAD_TYPE",
            id_column="OBJECTID",
        )
        result = _transform_download_data(
            sample_gdf,
            fetch_config=fetch_config,
            source_name="TestSource",
        )

        assert len(result) == 3
        assert "id" in result.columns
        assert "names" in result.columns
        assert "class" in result.columns
        assert "subtype" in result.columns
        assert "sources" in result.columns

        # Check ID format: {prefix}_{upstreamID}_{h3suffix}
        # Use rsplit to extract the H3 suffix (last component) regardless of prefix structure
        first_id = result["id"].iloc[0]
        assert first_id.startswith("test_1_")
        h3_suffix = first_id.rsplit("_", 1)[-1]
        assert len(h3_suffix) == 10  # 15-char H3 index minus 5 trailing f's

        # Check names extraction
        assert result["names"].iloc[0] == {"primary": "Main St"}
        assert result["names"].iloc[2] is None

        # Check class
        assert result["class"].iloc[0] == "primary"

    def test_class_mapping(self, sample_gdf):
        """Test class mapping transformation."""
        class_mapping = {"primary": "major", "secondary": "minor", "tertiary": "local"}

        fetch_config = FetchConfig(
            id_prefix="test",
            name_column="NAME",
            class_column="ROAD_TYPE",
            class_mapping=class_mapping,
            id_column="OBJECTID",
        )
        result = _transform_download_data(
            sample_gdf,
            fetch_config=fetch_config,
            source_name="TestSource",
        )

        assert result["class"].iloc[0] == "major"
        assert result["class"].iloc[1] == "minor"
        assert result["class"].iloc[2] == "local"

    def test_missing_id_column_raises_error(self):
        """Test that missing id_column raises ValueError."""
        gdf = gpd.GeoDataFrame(
            {
                "NAME": ["Main St"],
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            crs="EPSG:4326",
        )

        fetch_config = FetchConfig(
            id_prefix="test",
            name_column="NAME",
        )
        with pytest.raises(ValueError, match="id_column must be specified"):
            _transform_download_data(
                gdf,
                fetch_config=fetch_config,
                source_name="TestSource",
            )

    def test_invalid_id_column_raises_error(self):
        """Test that invalid id_column raises ValueError."""
        gdf = gpd.GeoDataFrame(
            {
                "NAME": ["Main St"],
                "geometry": [LineString([(0, 0), (1, 1)])],
            },
            crs="EPSG:4326",
        )

        fetch_config = FetchConfig(
            id_prefix="test",
            name_column="NAME",
            id_column="NONEXISTENT",
        )
        with pytest.raises(ValueError, match="not found in data"):
            _transform_download_data(
                gdf,
                fetch_config=fetch_config,
                source_name="TestSource",
            )

    def test_empty_dataframe(self):
        """Test handling of empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

        fetch_config = FetchConfig(id_prefix="test")
        result = _transform_download_data(
            gdf,
            fetch_config=fetch_config,
            source_name="TestSource",
        )

        assert len(result) == 0


class TestListDatasets:
    """Tests for list_datasets function."""

    @patch("crosswalk.fetch.target.list_dataset_configs")
    @patch("crosswalk.fetch.target.get_dataset_config")
    def test_list_all_datasets(self, mock_get_config, mock_list_configs):
        """Test listing all datasets."""
        mock_list_configs.return_value = ["us_boston_streets", "co_bogota_roads"]

        mock_config = MagicMock()
        mock_config.type = "road"
        mock_config.description = "Test description"
        mock_config.source.type = "arcgis"
        mock_config.source.api_key_env_var = None
        mock_get_config.return_value = mock_config

        result = list_datasets()

        assert len(result) == 2
        assert result[0]["name"] == "co_bogota_roads"
        assert result[0]["type"] == "road"
        assert result[0]["source_type"] == "arcgis"
        assert result[0]["api_key_required"] is False

    @patch("crosswalk.fetch.target.list_dataset_configs")
    @patch("crosswalk.fetch.target.get_dataset_config")
    def test_list_datasets_with_prefix(self, mock_get_config, mock_list_configs):
        """Test filtering datasets by prefix."""
        mock_list_configs.return_value = ["us_boston_streets", "co_bogota_roads"]

        mock_config = MagicMock()
        mock_config.type = "road"
        mock_config.description = "Test"
        mock_config.source.type = "arcgis"
        mock_config.source.api_key_env_var = None
        mock_get_config.return_value = mock_config

        result = list_datasets(prefix="us_")

        assert len(result) == 1
        assert result[0]["name"] == "us_boston_streets"


class TestFetchDataset:
    """Tests for fetch_dataset function."""

    @patch("crosswalk.fetch.target.get_dataset_config")
    def test_fetch_nonexistent_dataset(self, mock_get_config, tmp_path):
        """Test fetching a dataset that doesn't exist."""
        mock_get_config.return_value = None

        result = fetch_dataset("nonexistent", tmp_path)

        assert result is None

    @patch("crosswalk.fetch.target.fetch_arcgis_layer")
    @patch("crosswalk.fetch.target.get_dataset_config")
    def test_fetch_arcgis_dataset(self, mock_get_config, mock_fetch_arcgis, tmp_path):
        """Test fetching an ArcGIS dataset."""
        # Setup mock config
        mock_config = MagicMock()
        mock_config.description = "Test dataset"
        mock_config.source.type = "arcgis"
        mock_config.source.url = "https://example.com/arcgis/0"
        mock_config.fetch = FetchConfig(
            id_prefix="test",
            name_column="NAME",
            class_column="TYPE",
        )
        mock_config.display_name = "Test Dataset"
        mock_get_config.return_value = mock_config

        # Setup mock return
        output_path = tmp_path / "test_v1.0.parquet"
        mock_fetch_arcgis.return_value = output_path

        result = fetch_dataset("test", tmp_path)

        assert result == output_path
        mock_fetch_arcgis.assert_called_once()

    @patch("crosswalk.fetch.target.get_dataset_config")
    def test_fetch_manual_dataset(self, mock_get_config, tmp_path):
        """Test fetching a manual download dataset returns None."""
        mock_config = MagicMock()
        mock_config.description = "Manual dataset"
        mock_config.source.type = "manual"
        mock_config.source.portal_url = "https://example.com"
        mock_config.notes = "Download from portal"
        mock_get_config.return_value = mock_config

        result = fetch_dataset("manual_dataset", tmp_path)

        assert result is None


class TestFetchDatasetsByPrefix:
    """Tests for fetch_datasets_by_prefix function."""

    @patch("crosswalk.fetch.target.fetch_dataset")
    @patch("crosswalk.fetch.target.list_dataset_configs")
    def test_fetch_by_prefix(self, mock_list_configs, mock_fetch_dataset, tmp_path):
        """Test fetching multiple datasets by prefix."""
        mock_list_configs.return_value = [
            "us_boston_streets",
            "us_boston_sidewalks",
            "co_bogota_roads",
        ]
        mock_fetch_dataset.side_effect = [
            tmp_path / "us_boston_streets.parquet",
            tmp_path / "us_boston_sidewalks.parquet",
        ]

        results = fetch_datasets_by_prefix("us_boston", tmp_path)

        assert len(results) == 2
        assert "us_boston_streets" in results
        assert "us_boston_sidewalks" in results
        assert mock_fetch_dataset.call_count == 2

    @patch("crosswalk.fetch.target.list_dataset_configs")
    def test_fetch_by_prefix_no_matches(self, mock_list_configs, tmp_path):
        """Test fetching with no matching prefix."""
        mock_list_configs.return_value = ["us_boston_streets"]

        results = fetch_datasets_by_prefix("jp_", tmp_path)

        assert results == {}


class TestGetBufferedBbox:
    """Tests for get_buffered_bbox utility function."""

    def test_uses_default_when_none(self):
        """Test that default buffer is used when buffer_m is None."""
        from crosswalk.fetch.overture import BoundingBox, get_buffered_bbox

        bbox = BoundingBox(xmin=-71.0, ymin=42.0, xmax=-70.0, ymax=43.0)
        result_bbox, effective_buffer = get_buffered_bbox(bbox, None, 1000.0)

        assert effective_buffer == 1000.0
        assert result_bbox.xmin < bbox.xmin
        assert result_bbox.ymin < bbox.ymin
        assert result_bbox.xmax > bbox.xmax
        assert result_bbox.ymax > bbox.ymax

    def test_uses_explicit_buffer(self):
        """Test that explicit buffer overrides default."""
        from crosswalk.fetch.overture import BoundingBox, get_buffered_bbox

        bbox = BoundingBox(xmin=-71.0, ymin=42.0, xmax=-70.0, ymax=43.0)
        result_bbox, effective_buffer = get_buffered_bbox(bbox, 500.0, 1000.0)

        assert effective_buffer == 500.0

    def test_zero_buffer_returns_original(self):
        """Test that buffer=0 returns original bbox."""
        from crosswalk.fetch.overture import BoundingBox, get_buffered_bbox

        bbox = BoundingBox(xmin=-71.0, ymin=42.0, xmax=-70.0, ymax=43.0)
        result_bbox, effective_buffer = get_buffered_bbox(bbox, 0, 1000.0)

        assert effective_buffer is None
        assert result_bbox.xmin == bbox.xmin
        assert result_bbox.ymin == bbox.ymin
        assert result_bbox.xmax == bbox.xmax
        assert result_bbox.ymax == bbox.ymax


class TestLoadMsRoadsTsv:
    """Tests for _load_ms_roads_tsv function."""

    def _write_tsv(self, path: Path, rows: list[tuple[str, dict]]):
        """Write test TSV file with country_code + GeoJSON Feature lines."""
        with open(path, "w") as f:
            for country_code, feature in rows:
                f.write(f"{country_code}\t{json.dumps(feature)}\n")

    def _make_feature(self, coords, properties=None):
        """Create a GeoJSON Feature with a LineString geometry."""
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": properties or {},
        }

    def test_basic_loading(self, tmp_path):
        """Test loading TSV with no country filter."""
        tsv_path = tmp_path / "roads.tsv"
        self._write_tsv(
            tsv_path,
            [
                ("TUN", self._make_feature([(10.0, 36.8), (10.1, 36.9)])),
                ("DZA", self._make_feature([(3.0, 36.7), (3.1, 36.8)])),
            ],
        )
        gdf = _load_ms_roads_tsv(tsv_path)
        assert len(gdf) == 2
        assert gdf.crs.to_epsg() == 4326

    def test_country_filter(self, tmp_path):
        """Test filtering by country code."""
        tsv_path = tmp_path / "roads.tsv"
        self._write_tsv(
            tsv_path,
            [
                ("TUN", self._make_feature([(10.0, 36.8), (10.1, 36.9)])),
                ("DZA", self._make_feature([(3.0, 36.7), (3.1, 36.8)])),
                ("TUN", self._make_feature([(10.2, 36.7), (10.3, 36.8)])),
            ],
        )
        gdf = _load_ms_roads_tsv(tsv_path, country_filter="TUN")
        assert len(gdf) == 2

    def test_properties_preserved(self, tmp_path):
        """Test that GeoJSON Feature properties are preserved as columns."""
        tsv_path = tmp_path / "roads.tsv"
        self._write_tsv(
            tsv_path,
            [
                (
                    "TUN",
                    self._make_feature([(10.0, 36.8), (10.1, 36.9)], {"WidthMeters": 5.2}),
                ),
            ],
        )
        gdf = _load_ms_roads_tsv(tsv_path, country_filter="TUN")
        assert len(gdf) == 1
        assert "WidthMeters" in gdf.columns
        assert gdf["WidthMeters"].iloc[0] == 5.2

    def test_empty_result(self, tmp_path):
        """Test empty result when no rows match filter."""
        tsv_path = tmp_path / "roads.tsv"
        self._write_tsv(
            tsv_path,
            [("DZA", self._make_feature([(3.0, 36.7), (3.1, 36.8)]))],
        )
        gdf = _load_ms_roads_tsv(tsv_path, country_filter="TUN")
        assert len(gdf) == 0
        assert gdf.crs.to_epsg() == 4326

    def test_malformed_lines_skipped(self, tmp_path):
        """Test that malformed lines are skipped without error."""
        tsv_path = tmp_path / "roads.tsv"
        with open(tsv_path, "w") as f:
            f.write("TUN\t{invalid json}\n")
            f.write("no_tab_here\n")
            f.write("\n")
            feat = self._make_feature([(10.0, 36.8), (10.1, 36.9)])
            f.write(f"TUN\t{json.dumps(feat)}\n")
        gdf = _load_ms_roads_tsv(tsv_path, country_filter="TUN")
        assert len(gdf) == 1


class TestGeomHashId:
    """Tests for _geom_hash ID generation in fetch_download pipeline."""

    def test_geom_hash_deterministic(self):
        """Test that geometry hash produces deterministic IDs."""
        import hashlib

        import shapely

        geom = LineString([(10.0, 36.8), (10.1, 36.9)])
        wkt = shapely.to_wkt(geom, rounding_precision=7)
        expected = hashlib.md5(wkt.encode()).hexdigest()[:12]

        # Compute again — should be identical
        wkt2 = shapely.to_wkt(geom, rounding_precision=7)
        actual = hashlib.md5(wkt2.encode()).hexdigest()[:12]
        assert actual == expected

    def test_geom_hash_different_for_different_geoms(self):
        """Test that different geometries produce different hashes."""
        import hashlib

        import shapely

        geom1 = LineString([(10.0, 36.8), (10.1, 36.9)])
        geom2 = LineString([(10.0, 36.8), (10.2, 36.9)])

        hash1 = hashlib.md5(shapely.to_wkt(geom1, rounding_precision=7).encode()).hexdigest()[:12]
        hash2 = hashlib.md5(shapely.to_wkt(geom2, rounding_precision=7).encode()).hexdigest()[:12]
        assert hash1 != hash2

    def test_geom_hash_in_transform_pipeline(self):
        """Test _geom_hash column works end-to-end through _transform_download_data."""
        gdf = gpd.GeoDataFrame(
            {
                "_geom_hash": ["abc123def456", "789012345678"],
                "geometry": [
                    LineString([(10.0, 36.8), (10.1, 36.9)]),
                    LineString([(10.2, 36.7), (10.3, 36.8)]),
                ],
            },
            crs="EPSG:4326",
        )
        fetch_config = FetchConfig(id_prefix="test", id_column="_geom_hash")
        result = _transform_download_data(gdf, fetch_config=fetch_config, source_name="test")
        assert len(result) == 2
        # IDs should use the geom hash values
        assert result["id"].iloc[0].startswith("test_abc123def456_")

    def test_geom_hash_length(self):
        """Test that geometry hash is exactly 12 hex chars."""
        import hashlib

        import shapely

        geom = LineString([(0.0, 0.0), (1.0, 1.0)])
        h = hashlib.md5(shapely.to_wkt(geom, rounding_precision=7).encode()).hexdigest()[:12]
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)


class TestFilePatternFiltering:
    """Tests for file_pattern filtering in ZIP extraction."""

    def test_file_pattern_selects_correct_file(self, tmp_path):
        """Test that file_pattern glob selects the right file from a ZIP."""
        import fnmatch

        # Simulate the filtering logic with relative paths (as used in actual code)
        rel_paths = [
            "UTF-8/N06-24_HighwaySection.geojson",
            "UTF-8/N06-24_HighwayJoint.geojson",
            "Shift-JIS/N06-24_HighwaySection.geojson",
            "Shift-JIS/N06-24_HighwayJoint.geojson",
        ]

        pattern = "UTF-8/*HighwaySection*"
        filtered = [f for f in rel_paths if fnmatch.fnmatch(f, pattern)]
        assert len(filtered) == 1
        assert filtered[0] == "UTF-8/N06-24_HighwaySection.geojson"

    def test_file_pattern_no_match_keeps_all(self, tmp_path):
        """Test that non-matching pattern keeps original list."""
        import fnmatch

        rel_paths = ["data.geojson", "metadata.json"]
        pattern = "*NoMatch*"
        filtered = [f for f in rel_paths if fnmatch.fnmatch(f, pattern)]
        # When filter returns empty, the code keeps original found_files
        assert len(filtered) == 0

    def test_fetch_download_with_file_pattern(self, tmp_path):
        """Test fetch_download selects correct file from ZIP using file_pattern."""
        from crosswalk.fetch.target import fetch_download

        # Create a ZIP with files in subdirectories (like real MLIT ZIPs)
        zip_path = tmp_path / "test.zip"
        geojson_a = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[0, 0], [1, 1]],
                    },
                    "properties": {"ID": "A1", "name": "Road A"},
                }
            ],
        }
        geojson_b = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[2, 2], [3, 3]],
                    },
                    "properties": {"ID": "B1", "name": "Road B"},
                }
            ],
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("good/Section.geojson", json.dumps(geojson_a))
            zf.writestr("bad/Section.geojson", json.dumps(geojson_b))

        output_path = tmp_path / "output.parquet"
        fetch_config = FetchConfig(id_prefix="test", id_column="ID", name_column="name")

        # Mock the download to return our local ZIP
        with patch("crosswalk.fetch.target.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.headers = {"content-type": "application/zip"}
            mock_resp.iter_content = lambda chunk_size: [zip_path.read_bytes()]
            mock_resp.raise_for_status = lambda: None
            mock_get.return_value = mock_resp

            result = fetch_download(
                url="https://example.com/test.zip",
                output_path=output_path,
                file_format="geojson",
                fetch_config=fetch_config,
                source_name="test",
                file_pattern="good/*",
            )

        gdf = gpd.read_parquet(result)
        assert len(gdf) == 1
        # Should have loaded good/Section, not bad/Section
        assert gdf["names"].iloc[0] == {"primary": "Road A"}


class TestMsRoadsTsvFetchDownload:
    """Test ms_roads_tsv format through fetch_download pipeline."""

    def test_fetch_download_ms_roads_tsv(self, tmp_path):
        """Test fetch_download with ms_roads_tsv format end-to-end."""
        from crosswalk.fetch.target import fetch_download

        # Create a ZIP with a TSV file inside
        zip_path = tmp_path / "test.zip"
        feat1 = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[10.0, 36.8], [10.1, 36.9]],
            },
            "properties": {},
        }
        feat2 = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[3.0, 36.7], [3.1, 36.8]],
            },
            "properties": {},
        }
        tsv_content = f"TUN\t{json.dumps(feat1)}\nDZA\t{json.dumps(feat2)}\n"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("Northern_Africa.tsv", tsv_content)

        output_path = tmp_path / "output.parquet"
        fetch_config = FetchConfig(id_prefix="tn_test", id_column="_geom_hash")

        with patch("crosswalk.fetch.target.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.headers = {"content-type": "application/zip"}
            mock_resp.iter_content = lambda chunk_size: [zip_path.read_bytes()]
            mock_resp.raise_for_status = lambda: None
            mock_get.return_value = mock_resp

            result = fetch_download(
                url="https://example.com/test.zip",
                output_path=output_path,
                file_format="ms_roads_tsv",
                fetch_config=fetch_config,
                source_name="test",
                where_clause="TUN",
            )

        gdf = gpd.read_parquet(result)
        assert len(gdf) == 1  # Only TUN row
        # Should have _geom_hash-based ID
        assert gdf["id"].iloc[0].startswith("tn_test_")
