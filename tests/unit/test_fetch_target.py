"""Tests for the target fetch module."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from matcher.datasets.schema import FetchConfig
from matcher.fetch.target import (
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
        assert result["id"].iloc[0].startswith("test_1_")
        assert len(result["id"].iloc[0].split("_")) == 3  # prefix_id_suffix

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

    @patch("matcher.fetch.target.list_dataset_configs")
    @patch("matcher.fetch.target.get_dataset_config")
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

    @patch("matcher.fetch.target.list_dataset_configs")
    @patch("matcher.fetch.target.get_dataset_config")
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

    @patch("matcher.fetch.target.get_dataset_config")
    def test_fetch_nonexistent_dataset(self, mock_get_config, tmp_path):
        """Test fetching a dataset that doesn't exist."""
        mock_get_config.return_value = None

        result = fetch_dataset("nonexistent", tmp_path)

        assert result is None

    @patch("matcher.fetch.target.fetch_arcgis_layer")
    @patch("matcher.fetch.target.get_dataset_config")
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

    @patch("matcher.fetch.target.get_dataset_config")
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

    @patch("matcher.fetch.target.fetch_dataset")
    @patch("matcher.fetch.target.list_dataset_configs")
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

    @patch("matcher.fetch.target.list_dataset_configs")
    def test_fetch_by_prefix_no_matches(self, mock_list_configs, tmp_path):
        """Test fetching with no matching prefix."""
        mock_list_configs.return_value = ["us_boston_streets"]

        results = fetch_datasets_by_prefix("jp_", tmp_path)

        assert results == {}


class TestGetBufferedBbox:
    """Tests for get_buffered_bbox utility function."""

    def test_uses_default_when_none(self):
        """Test that default buffer is used when buffer_m is None."""
        from matcher.fetch.overture import BoundingBox, get_buffered_bbox

        bbox = BoundingBox(xmin=-71.0, ymin=42.0, xmax=-70.0, ymax=43.0)
        result_bbox, effective_buffer = get_buffered_bbox(bbox, None, 1000.0)

        assert effective_buffer == 1000.0
        assert result_bbox.xmin < bbox.xmin
        assert result_bbox.ymin < bbox.ymin
        assert result_bbox.xmax > bbox.xmax
        assert result_bbox.ymax > bbox.ymax

    def test_uses_explicit_buffer(self):
        """Test that explicit buffer overrides default."""
        from matcher.fetch.overture import BoundingBox, get_buffered_bbox

        bbox = BoundingBox(xmin=-71.0, ymin=42.0, xmax=-70.0, ymax=43.0)
        result_bbox, effective_buffer = get_buffered_bbox(bbox, 500.0, 1000.0)

        assert effective_buffer == 500.0

    def test_zero_buffer_returns_original(self):
        """Test that buffer=0 returns original bbox."""
        from matcher.fetch.overture import BoundingBox, get_buffered_bbox

        bbox = BoundingBox(xmin=-71.0, ymin=42.0, xmax=-70.0, ymax=43.0)
        result_bbox, effective_buffer = get_buffered_bbox(bbox, 0, 1000.0)

        assert effective_buffer is None
        assert result_bbox.xmin == bbox.xmin
        assert result_bbox.ymin == bbox.ymin
        assert result_bbox.xmax == bbox.xmax
        assert result_bbox.ymax == bbox.ymax
