"""Integration tests for linear-referenced attribute handling.

Tests the full flow from Overture data parsing through feature computation.
"""

import geopandas as gpd
from shapely.geometry import LineString

from matcher.features.alignment import AlignmentResult
from matcher.fetch.overture import (
    extract_lr_attributes,
    parse_names_lr,
)
from matcher.utils.linear_ref import (
    AttributeRange,
    LinearReferencedAttribute,
    create_trivial_lr,
    extract_aligned_attributes,
)


class TestLRDataFlowIntegration:
    """Test LR data flow from parsing to feature extraction."""

    def test_overture_names_extraction_end_to_end(self):
        """Test extracting names from Overture-style names dict through to LR."""
        # Simulate Overture names structure with rules
        names_dict = {
            "primary": "Main Street",
            "rules": [
                {"value": "Oak Avenue", "between": [0.0, 0.4], "variant": "common"},
                {"value": "Elm Boulevard", "between": [0.6, 1.0], "variant": "common"},
            ],
        }

        # Parse to LR
        lr = parse_names_lr(names_dict)

        # Verify structure
        assert len(lr.ranges) >= 3  # Oak, Main (gap fill), Elm

        # Extract for alignment covering first half
        attrs1 = extract_aligned_attributes({"name": lr}, 0.0, 0.3)
        assert attrs1["name"] == "Oak Avenue"

        # Extract for alignment covering middle (gap = default)
        attrs2 = extract_aligned_attributes({"name": lr}, 0.4, 0.6)
        assert attrs2["name"] == "Main Street"  # Default fills gap

        # Extract for alignment covering end
        attrs3 = extract_aligned_attributes({"name": lr}, 0.7, 1.0)
        assert attrs3["name"] == "Elm Boulevard"

    def test_trivial_lr_for_target_data(self):
        """Test that target data without LR rules gets trivial LR."""
        # Simulate target data with simple flat name
        name = "Local Road"

        # Create trivial LR
        lr = create_trivial_lr(name)

        # Any alignment should return the same value
        attrs1 = extract_aligned_attributes({"name": lr}, 0.0, 0.5)
        attrs2 = extract_aligned_attributes({"name": lr}, 0.5, 1.0)
        attrs3 = extract_aligned_attributes({"name": lr}, 0.0, 1.0)

        assert attrs1["name"] == name
        assert attrs2["name"] == name
        assert attrs3["name"] == name

    def test_gdf_extract_lr_attributes(self):
        """Test extracting LR attributes from a GeoDataFrame."""
        # Create minimal GeoDataFrame with Overture-like structure
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1", "seg2"],
                "names": [
                    {"primary": "First Street"},
                    {
                        "primary": "Second Street",
                        "rules": [{"value": "Alt Name", "between": [0.0, 0.5]}],
                    },
                ],
                "subclass": ["residential", "primary"],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                ],
            },
            crs="EPSG:4326",
        )

        # Extract LR attributes
        result = extract_lr_attributes(gdf)

        # Verify LR columns were added
        assert "names_lr" in result.columns
        assert "subclass_lr" in result.columns
        assert "level_lr" in result.columns
        assert "road_flags_lr" in result.columns

        # Verify first segment (no rules, should be trivial)
        lr1 = result.iloc[0]["names_lr"]
        assert len(lr1) == 1
        assert lr1[0]["value"] == "First Street"

        # Verify second segment (has rules)
        lr2 = result.iloc[1]["names_lr"]
        assert len(lr2) >= 1

    def test_alignment_affects_name_extraction(self):
        """Test that alignment fractions change which name is extracted."""
        # Create LR with name change at midpoint
        lr = LinearReferencedAttribute(
            ranges=[
                AttributeRange(start=0.0, end=0.5, value="First Half Name"),
                AttributeRange(start=0.5, end=1.0, value="Second Half Name"),
            ],
            default_value="First Half Name",
        )

        # Alignment covers first half
        alignment1 = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=0.4,
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )
        attrs1 = extract_aligned_attributes(
            {"name": lr},
            alignment1.overture_start_frac,
            alignment1.overture_end_frac,
        )
        assert attrs1["name"] == "First Half Name"

        # Alignment covers second half
        alignment2 = AlignmentResult(
            overture_start_frac=0.6,
            overture_end_frac=1.0,
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )
        attrs2 = extract_aligned_attributes(
            {"name": lr},
            alignment2.overture_start_frac,
            alignment2.overture_end_frac,
        )
        assert attrs2["name"] == "Second Half Name"


class TestMissingLRColumns:
    """Test that extract_lr_attributes handles GeoDataFrames without LR columns."""

    def test_missing_lr_columns_handled(self):
        """GeoDataFrames with flat 'name' column (no 'names' dict) should get trivial LR."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["seg1"],
                "name": "Oak Street",  # Flat name, not 'names' dict
                "geometry": [LineString([(0, 0), (1, 0)])],
            },
            crs="EPSG:4326",
        )

        result = extract_lr_attributes(gdf)

        assert "names_lr" in result.columns
        lr_data = result.iloc[0]["names_lr"]
        assert len(lr_data) == 1
        assert lr_data[0]["between"] == [0.0, 1.0]


class TestMultipleAttributes:
    """Test handling of multiple LR attributes together."""

    def test_extract_all_aligned_attributes(self):
        """Test extracting multiple attributes with different LR structures."""
        # Name changes at 0.5
        names_lr = LinearReferencedAttribute.from_dict_list(
            [
                {"between": [0.0, 0.5], "value": "First St"},
                {"between": [0.5, 1.0], "value": "Second St"},
            ]
        )

        # Subclass is uniform
        subclass_lr = create_trivial_lr("residential")

        # Level has elevated section
        level_lr = LinearReferencedAttribute.from_dict_list(
            [
                {"between": [0.0, 0.3], "value": 0},
                {"between": [0.3, 0.7], "value": 1},
                {"between": [0.7, 1.0], "value": 0},
            ]
        )

        lr_data = {
            "name": names_lr,
            "subclass": subclass_lr,
            "level": level_lr,
        }

        # Extract for alignment covering mostly the second half of names
        # Query [0.45, 0.65]: First St = 0.05 (from 0.45 to 0.5), Second St = 0.15 (from 0.5 to 0.65)
        # Second St wins with more coverage
        attrs = extract_aligned_attributes(lr_data, 0.45, 0.65)

        assert attrs["name"] == "Second St"  # Second half wins (0.15 > 0.05)
        assert attrs["subclass"] == "residential"  # Uniform
        assert attrs["level"] == 1  # Elevated section (0.3-0.7)
