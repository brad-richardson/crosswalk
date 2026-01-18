"""Shared dataset configurations for ArcGIS fetch scripts.

This module contains standardized configurations for fetching road and sidewalk
data from various municipal GIS portals. Each configuration follows the schema
expected by matcher.fetch.arcgis.fetch_arcgis_layer().

Usage:
    from dataset_configs import FORT_COLLINS_DATASETS, FRISCO_DATASETS

    for dataset in FORT_COLLINS_DATASETS:
        fetch_arcgis_layer(output_path=..., **dataset)
"""

# =============================================================================
# Fort Collins, Colorado
# =============================================================================
# Source: https://data-fcgov.opendata.arcgis.com/
# Organization ID: dLpFH5mwVvxSN4OE

FORT_COLLINS_SIDEWALK_TYPES = {
    1: "attached",
    2: "detached",
    3: "missing",
    4: "no_roadway",
    5: "drive_approach",
}

FORT_COLLINS_STREET_TYPES = {
    "INTERSTATE": "motorway",
    "ARTERIAL": "primary",
    "ARTERIAL MAJOR": "primary",
    "HIGHWAY": "primary",
    "HIGHWAY DIVIDED": "primary",
    "COLLECTOR": "tertiary",
    "LOCAL": "residential",
    "RAMP": "motorway_link",
    "ROUNDABOUT": "tertiary",
    "ALLEY": "service",
}

FORT_COLLINS_DATASETS = [
    {
        "name": "fort_collins_sidewalks",
        "url": "https://services1.arcgis.com/dLpFH5mwVvxSN4OE/arcgis/rest/services/Sidewalks/FeatureServer/0",
        "id_prefix": "fc_sidewalk",
        "name_column": None,  # Sidewalks don't have names
        "class_column": "TYPE",
        "class_mapping": {
            1: "footway",
            2: "footway",
            3: "footway",  # Missing - useful for gap analysis
            4: "footway",
            5: "footway",
        },
        "subclass_column": "TYPE",
        "subclass_mapping": {
            1: "sidewalk",
            2: "sidewalk",
            3: "sidewalk",
            4: "sidewalk",
            5: "crossing",  # Drive approach
        },
        "source_name": "Fort Collins Sidewalks",
        "description": "540 miles of sidewalk inventory with attachment type",
    },
    {
        "name": "fort_collins_streets",
        "url": "https://services1.arcgis.com/dLpFH5mwVvxSN4OE/arcgis/rest/services/Street_Centerlines/FeatureServer/0",
        "id_prefix": "fc_street",
        "name_column": "STRNAME",
        "class_column": "STREETTYPE",
        "class_mapping": FORT_COLLINS_STREET_TYPES,
        "source_name": "Fort Collins Street Centerlines",
        "description": "Street centerlines with 10 road type classifications",
    },
]


# =============================================================================
# Frisco, Texas
# =============================================================================
# Source: https://geodata-frisco.hub.arcgis.com/
# Server: maps.friscotexas.gov

FRISCO_TRAIL_TYPES = {
    "Bike Only": "cycleway",
    "Parkway": "path",
    "Regional": "path",
    "Local": "footway",
    "Nature": "path",
    "Golf Course": "path",
}

FRISCO_ROAD_SUBTYPES = {
    1: "motorway",  # Tollway
    2: "primary",  # State Highway
    3: "primary",  # US Highway
    4: "secondary",  # Major Thoroughfare
    5: "tertiary",  # Minor Thoroughfare
    6: "tertiary",  # Collector
    7: "residential",  # Residential
    8: "motorway_link",  # Ramp
    9: "service",  # Driveway
}

FRISCO_DATASETS = [
    {
        "name": "frisco_trails",
        "url": "https://maps.friscotexas.gov/gis/rest/services/Public/FriscoCommunityMaps/FeatureServer/10",
        "id_prefix": "frisco_trail",
        "name_column": "Name",
        "class_column": "TrailType",
        "class_mapping": FRISCO_TRAIL_TYPES,
        "source_name": "Frisco Trails",
        "description": "Trail network including pedestrian paths and bike routes",
    },
    {
        "name": "frisco_roads",
        "url": "https://maps.friscotexas.gov/gis/rest/services/Public/FriscoCommunityMaps/FeatureServer/8",
        "id_prefix": "frisco_road",
        "name_column": "ROAD_NAME",
        "class_column": "SUBTYPE",
        "class_mapping": FRISCO_ROAD_SUBTYPES,
        "source_name": "Frisco Road Centerlines",
        "description": "Road centerlines with 9 classifications and lifecycle status",
    },
]


# =============================================================================
# Salt Lake City, Utah
# =============================================================================
# Source: https://gis-slcgov.opendata.arcgis.com/datasets/sidewalk-1
# Note: Uses ArcGIS Hub download - endpoint derived from portal

# The sidewalk dataset is available via the Open Data portal
# Portal page: https://gis-slcgov.opendata.arcgis.com/datasets/sidewalk-1
# The actual FeatureServer endpoint needs to be extracted from the portal
SALT_LAKE_CITY_DATASETS = [
    {
        "name": "salt_lake_city_sidewalks",
        # This URL will be populated after querying the Hub API
        "url": None,
        "portal_url": "https://gis-slcgov.opendata.arcgis.com/datasets/sidewalk-1",
        "id_prefix": "slc_sidewalk",
        "name_column": None,
        "class_column": None,  # Will determine from schema
        "class_mapping": None,
        "source_name": "Salt Lake City Sidewalks",
        "description": "Sidewalk centerlines from Salt Lake City GIS",
    },
]


# =============================================================================
# Ada County, Idaho (Boise Area) - Roads Only
# =============================================================================
# Source: https://maps.achdidaho.org/

ADA_COUNTY_DATASETS = [
    {
        "name": "ada_county_roads",
        "url": "https://maps.achdidaho.org/server/rest/services/Assessor/roadcenterline/MapServer/0",
        "id_prefix": "ada_road",
        "name_column": "StName",
        "class_column": "FuncClass",
        "class_mapping": None,  # Need to inspect actual values
        "source_name": "Ada County Road Centerlines",
        "description": "Road centerlines for Boise metro area",
    },
]


# =============================================================================
# Utah Statewide Trails
# =============================================================================
# Source: https://gis.utah.gov/products/sgid/recreation/trails-pathways/

UTAH_TRAILS_DATASETS = [
    {
        "name": "utah_trails",
        "url": "https://services1.arcgis.com/99lidPhWCzftIe9K/ArcGIS/rest/services/TrailsAndPathways/FeatureServer/0",
        "id_prefix": "utah_trail",
        "name_column": "PrimaryName",
        "class_column": "DesignatedUses",
        "class_mapping": None,  # Complex - multiple uses per trail
        "source_name": "Utah Trails and Pathways",
        "description": "Statewide trails with ADA accessibility data",
    },
]


# =============================================================================
# All datasets for bulk operations
# =============================================================================
ALL_DATASETS = {
    "fort_collins": FORT_COLLINS_DATASETS,
    "frisco": FRISCO_DATASETS,
    "salt_lake_city": SALT_LAKE_CITY_DATASETS,
    "ada_county": ADA_COUNTY_DATASETS,
    "utah_trails": UTAH_TRAILS_DATASETS,
}
