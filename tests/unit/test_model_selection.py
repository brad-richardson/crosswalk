"""Tests for automatic model selection based on dataset attributes."""

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString


class TestSelectModelForDataset:
    """Tests for select_model_for_dataset function."""

    @pytest.fixture
    def sample_gdf_with_names(self):
        """Create a GeoDataFrame with names."""
        lines = [LineString([(0, 0), (10, 0)]) for _ in range(10)]
        names = [
            "Main St",
            "Oak Ave",
            "Elm St",
            None,
            "Pine Rd",
            "",
            "Cedar Ln",
            "Maple Dr",
            "Birch Way",
            "Walnut St",
        ]
        return gpd.GeoDataFrame(
            {"id": list(range(10)), "names": names, "geometry": lines},
            crs="EPSG:4326",
        )

    @pytest.fixture
    def sample_gdf_no_names(self):
        """Create a GeoDataFrame without names."""
        lines = [LineString([(0, 0), (10, 0)]) for _ in range(10)]
        return gpd.GeoDataFrame(
            {"id": list(range(10)), "geometry": lines},
            crs="EPSG:4326",
        )

    @pytest.fixture
    def sample_gdf_low_name_coverage(self):
        """Create a GeoDataFrame with low name coverage (<50%)."""
        lines = [LineString([(0, 0), (10, 0)]) for _ in range(10)]
        # Only 3 out of 10 have names (30%)
        names = ["Main St", None, None, "Oak Ave", None, None, None, "Elm St", None, None]
        return gpd.GeoDataFrame(
            {"id": list(range(10)), "names": names, "geometry": lines},
            crs="EPSG:4326",
        )

    @pytest.fixture
    def model_paths(self):
        """Create temporary model files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            full_path = Path(tmpdir) / "full_model.joblib"
            geom_path = Path(tmpdir) / "geom_model.joblib"
            yield {"full": str(full_path), "geom": str(geom_path)}

    def test_high_name_coverage_selects_full_model(self, sample_gdf_with_names, model_paths):
        """With >50% name coverage, full model should be selected."""
        from crosswalk.matching.ml import select_model_for_dataset

        # Create the full model file
        Path(model_paths["full"]).touch()

        result = select_model_for_dataset(
            sample_gdf_with_names,
            full_model_path=model_paths["full"],
            geom_only_model_path=model_paths["geom"],
        )

        assert result == model_paths["full"]

    def test_no_names_selects_geom_model_if_exists(self, sample_gdf_no_names, model_paths):
        """Without name column, geometry-only model should be selected if available."""
        from crosswalk.matching.ml import select_model_for_dataset

        # Create both model files
        Path(model_paths["full"]).touch()
        Path(model_paths["geom"]).touch()

        result = select_model_for_dataset(
            sample_gdf_no_names,
            full_model_path=model_paths["full"],
            geom_only_model_path=model_paths["geom"],
        )

        assert result == model_paths["geom"]

    def test_no_names_falls_back_to_full_model(self, sample_gdf_no_names, model_paths):
        """Without name column and no geom model, falls back to full model."""
        from crosswalk.matching.ml import select_model_for_dataset

        # Only create the full model file
        Path(model_paths["full"]).touch()

        result = select_model_for_dataset(
            sample_gdf_no_names,
            full_model_path=model_paths["full"],
            geom_only_model_path=model_paths["geom"],
        )

        assert result == model_paths["full"]

    def test_low_name_coverage_selects_geom_model(self, sample_gdf_low_name_coverage, model_paths):
        """With <50% name coverage, geometry-only model should be selected."""
        from crosswalk.matching.ml import select_model_for_dataset

        # Create both model files
        Path(model_paths["full"]).touch()
        Path(model_paths["geom"]).touch()

        result = select_model_for_dataset(
            sample_gdf_low_name_coverage,
            full_model_path=model_paths["full"],
            geom_only_model_path=model_paths["geom"],
        )

        assert result == model_paths["geom"]

    @pytest.mark.parametrize(
        "name_coverage,min_threshold,expected_model",
        [
            (0.6, 0.5, "full"),  # 60% coverage, 50% threshold -> full
            (0.4, 0.5, "geom"),  # 40% coverage, 50% threshold -> geom
            (0.5, 0.5, "full"),  # 50% coverage, 50% threshold -> full (boundary)
            (0.49, 0.5, "geom"),  # 49% coverage, 50% threshold -> geom
            (0.3, 0.2, "full"),  # 30% coverage, 20% threshold -> full
        ],
    )
    def test_threshold_boundary_cases(
        self, model_paths, name_coverage, min_threshold, expected_model
    ):
        """Test various threshold boundary cases."""
        from crosswalk.matching.ml import select_model_for_dataset

        # Create both model files
        Path(model_paths["full"]).touch()
        Path(model_paths["geom"]).touch()

        # Create GDF with specific coverage
        num_with_names = int(10 * name_coverage)
        lines = [LineString([(0, 0), (10, 0)]) for _ in range(10)]
        names = ["Name"] * num_with_names + [None] * (10 - num_with_names)
        gdf = gpd.GeoDataFrame(
            {"id": list(range(10)), "names": names, "geometry": lines},
            crs="EPSG:4326",
        )

        result = select_model_for_dataset(
            gdf,
            full_model_path=model_paths["full"],
            geom_only_model_path=model_paths["geom"],
            min_name_coverage=min_threshold,
        )

        expected_path = model_paths["full"] if expected_model == "full" else model_paths["geom"]
        assert result == expected_path

    def test_empty_strings_not_counted_as_names(self, model_paths):
        """Empty strings should not count as having a name."""
        from crosswalk.matching.ml import select_model_for_dataset

        # Create both model files
        Path(model_paths["full"]).touch()
        Path(model_paths["geom"]).touch()

        # Create GDF where most "names" are empty strings
        lines = [LineString([(0, 0), (10, 0)]) for _ in range(10)]
        names = ["Real Name", "", "", "", "", "   ", "", "", "", "Another Name"]  # Only 2/10 = 20%
        gdf = gpd.GeoDataFrame(
            {"id": list(range(10)), "names": names, "geometry": lines},
            crs="EPSG:4326",
        )

        result = select_model_for_dataset(
            gdf,
            full_model_path=model_paths["full"],
            geom_only_model_path=model_paths["geom"],
        )

        # Empty strings should not count, so coverage is 20% < 50%
        assert result == model_paths["geom"]

    def test_alternative_name_column(self, model_paths):
        """Test with 'name' column instead of 'names'."""
        from crosswalk.matching.ml import select_model_for_dataset

        Path(model_paths["full"]).touch()

        lines = [LineString([(0, 0), (10, 0)]) for _ in range(10)]
        names = ["Main St"] * 8 + [None] * 2  # 80% coverage
        gdf = gpd.GeoDataFrame(
            {"id": list(range(10)), "name": names, "geometry": lines},
            crs="EPSG:4326",
        )

        result = select_model_for_dataset(
            gdf,
            full_model_path=model_paths["full"],
            geom_only_model_path=model_paths["geom"],
        )

        assert result == model_paths["full"]
