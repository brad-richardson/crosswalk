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
# Boston, Massachusetts
# =============================================================================
# Source: https://data.boston.gov/
# Also: https://gisportal.boston.gov/

# MassDOT functional classification codes
# Based on F_CLASS cross-reference: CLASS 3/4/5 were mismapped
BOSTON_STREET_CLASS_MAPPING = {
    1: "motorway",  # Interstate (not in Boston data)
    2: "primary",  # Principal arterial
    3: "primary",  # Also Principal arterial (79% are F_CLASS 3)
    4: "tertiary",  # Major Collector (65% are F_CLASS 5)
    5: "residential",  # Local/Minor Collector (83% F_CLASS 0/unknown)
    6: "service",  # Service roads (not in Boston data)
}

# Sidewalk type codes -> Overture "footway" class
BOSTON_SIDEWALK_CLASS_MAPPING = {
    "SWALK-CL": "footway",
    "CWALK-CL": "footway",  # Crosswalks are pedestrian paths
    "CWALK-CL-UM": "footway",  # Unmarked crosswalks
    "PWALK-CL": "footway",  # Private walkway
    "Sidewalk centerline": "footway",
    "Crosswalk centerline": "footway",
    "Privatewalk centerline": "footway",
}

# Sidewalk type codes -> subclass (to match Overture schema)
BOSTON_SIDEWALK_SUBCLASS_MAPPING = {
    "SWALK-CL": "sidewalk",
    "CWALK-CL": "crosswalk",
    "CWALK-CL-UM": "crosswalk",  # Unmarked crosswalks
    "PWALK-CL": "sidewalk",  # Private walkway -> sidewalk
    "Sidewalk centerline": "sidewalk",
    "Crosswalk centerline": "crosswalk",
    "Privatewalk centerline": "sidewalk",
}

# Bike facility type codes -> Overture classes
# Key distinction: facilities on road surface vs physically separated
BOSTON_BIKE_CLASS_MAPPING = {
    # Physically separated infrastructure -> cycleway
    "SBL": "cycleway",  # Separated bike lane (raised/curbed)
    "SBLBL": "cycleway",  # Separated + bike lane
    "SBLSL": "cycleway",  # Separated + shared lane
    "CFSBL": "cycleway",  # Contraflow separated bike lane
    # On-road painted facilities -> unknown (same surface as road)
    "BL": "unknown",  # Bike lane (painted on road)
    "BL-PEAKBUS": "unknown",  # Bike lane (peak bus hours)
    "BFBL": "unknown",  # Buffered bike lane (paint only)
    "BLSL": "unknown",  # Bike lane + shared lane
    "CFBL": "unknown",  # Contraflow bike lane (painted)
    "CFBS": "unknown",  # Contraflow bike street
    # Shared use paths -> path (separate from road)
    "SUP": "path",  # Shared use path
    "SUPN": "path",  # Natural surface shared use path
    "SUPM": "path",  # Minor shared use path
    # Shared lane markings (not dedicated) -> unknown
    "SLM": "unknown",  # Shared lane markings (sharrows)
    "SLMTC": "unknown",  # Shared lane traffic calmed
    # Other infrastructure types
    "PED": "pedestrian",  # Pedestrianized street
    "WALK": "footway",  # Walkway
}

BOSTON_DATASETS = [
    {
        "name": "boston_streets",
        "url": "https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/City_of_Boston_Managed_Streets/FeatureServer/0",
        "id_prefix": "boston_streets",
        "name_column": "STREETNAME",
        "class_column": "CLASS",
        "class_mapping": BOSTON_STREET_CLASS_MAPPING,
        "level_column": None,
        "source_name": "Boston Managed Streets",
        "description": "Street centerlines with MassDOT functional classification",
    },
    {
        "name": "boston_sidewalks",
        "url": "https://gisportal.boston.gov/arcgis/rest/services/Infrastructure/OpenData/MapServer/5",
        "id_prefix": "boston_sidewalk",
        "name_column": None,  # Sidewalks unnamed
        "class_column": "TYPE",
        "class_mapping": BOSTON_SIDEWALK_CLASS_MAPPING,
        "subclass_column": "TYPE",
        "subclass_mapping": BOSTON_SIDEWALK_SUBCLASS_MAPPING,
        "level_column": None,
        "source_name": "Boston Sidewalk Centerlines",
        "description": "Sidewalk and crosswalk centerlines",
    },
    {
        "name": "boston_bike_network",
        "url": "https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/Boston_Bicycle_Network_2024/FeatureServer/0",
        "id_prefix": "boston_bike",
        "name_column": "STREET_NAM",
        "class_column": "ExisFacil",
        "class_mapping": BOSTON_BIKE_CLASS_MAPPING,
        "level_column": None,
        "source_name": "Boston Bicycle Network 2024",
        "description": "Bike lanes, paths, and shared facilities",
    },
]


# =============================================================================
# Fort Collins, Colorado
# =============================================================================
# Source: https://data-fcgov.opendata.arcgis.com/
# Organization ID: dLpFH5mwVvxSN4OE

# Fort Collins sidewalk TYPE codes (for reference in the source data):
#   1: attached (attached to roadway)
#   2: detached (separated from roadway by buffer)
#   3: missing (gap in sidewalk network)
#   4: no_roadway (path not adjacent to road)
#   5: drive_approach (driveway crossing)

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
# Fresno County, California (uses custom fetch script)
# =============================================================================
# Source: Caltrans Functional Classification
# Note: Uses WHERE clause filter for county, custom name parsing - see fetch_fresno.py

# FHWA Functional Classification to Overture class mapping
FRESNO_F_SYSTEM_MAPPING = {
    1: "motorway",  # Interstate
    2: "motorway",  # Principal Arterial - Freeways/Expressways
    3: "primary",  # Principal Arterial - Other
    4: "secondary",  # Minor Arterial
    5: "tertiary",  # Major Collector
    6: "tertiary",  # Minor Collector
    7: "residential",  # Local
}

FRESNO_DATASETS = [
    {
        "name": "fresno_roads",
        "url": "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/CHhighway/CRS_Functional_Classification/FeatureServer/0",
        "id_prefix": "fresno_roads",
        "name_column": "RouteID",  # Requires custom parsing
        "class_column": "F_System",
        "class_mapping": FRESNO_F_SYSTEM_MAPPING,
        "where_clause": "County_label = 'FRESNO'",
        "source_name": "Caltrans Functional Classification",
        "description": "Fresno County roads from Caltrans (FHWA classification)",
    },
]


# =============================================================================
# Utah Salt Lake County (uses custom fetch script)
# =============================================================================
# Source: Utah SGID
# Note: Uses WHERE clause filter for county - see fetch_utah.py

# CARTOCODE to Overture class mapping
UTAH_CARTOCODE_MAPPING = {
    "1": "motorway",  # Interstate
    "2": "trunk",  # US Highway
    "3": "trunk",  # US Highway
    "4": "primary",  # Parkway
    "5": "primary",  # State Route (major)
    "6": "secondary",  # State Route (minor)
    "7": "motorway_link",  # Ramps
    "8": "secondary",  # Major street
    "9": "tertiary",  # Minor road
    "10": "tertiary",  # Secondary street
    "11": "residential",  # Local street
    "12": "track",  # Unpaved/dirt road
    "14": "service",  # Private road
    "15": "service",  # Private road
    "16": "path",  # Trail/path
    "17": "unclassified",  # Other
}

UTAH_SALT_LAKE_DATASETS = [
    {
        "name": "utah_roads",
        "url": "https://services1.arcgis.com/99lidPhWCzftIe9K/ArcGIS/rest/services/UtahRoads/FeatureServer/0",
        "id_prefix": "utah_roads",
        "name_column": "FULLNAME",
        "class_column": "CARTOCODE",
        "class_mapping": UTAH_CARTOCODE_MAPPING,
        "where_clause": "COUNTY_L = '49035'",  # Salt Lake County FIPS
        "source_name": "Utah SGID Roads",
        "description": "Salt Lake County roads from Utah SGID",
    },
]


# =============================================================================
# Cape Town, South Africa
# =============================================================================
# Source: https://odp-cctegis.opendata.arcgis.com/
# ArcGIS Hub portal with road centerlines

CAPE_TOWN_DATASETS = [
    {
        "name": "cape_town_roads",
        "url": None,  # Using portal_url for Hub discovery
        "portal_url": "https://odp-cctegis.opendata.arcgis.com/datasets/road-centrelines",
        "id_prefix": "ct_road",
        "name_column": "STR_NAME",  # Common column name for street name
        "class_column": None,  # Will use discover-classes
        "class_mapping": None,
        "source_name": "City of Cape Town Road Centrelines",
        "description": "Road centerlines from City of Cape Town Open Data Portal",
        "bbox": (18.30, -34.10, 18.65, -33.85),  # Cape Town bbox
    },
]


# =============================================================================
# Bogotá, Colombia
# =============================================================================
# Source: https://www.ideca.gov.co/ and https://serviciosgis.catastrobogota.gov.co/
# ArcGIS MapServer with road network (Malla Vial)

# MVITCLA classification mapping (numeric codes for road types)
# Based on Bogotá road classification system
BOGOTA_MALLA_VIAL_TYPE_MAPPING = {
    1: "primary",  # Malla vial arterial principal
    2: "secondary",  # Malla vial arterial complementaria
    3: "tertiary",  # Malla vial intermedia
    4: "residential",  # Malla vial local
    5: "unclassified",  # Malla vial rural
    6: "pedestrian",  # Malla vial peatonal
    7: "unclassified",  # Other
}

BOGOTA_DATASETS = [
    {
        "name": "bogota_roads",
        "url": "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/Mapa_Referencia/Mapa_Referencia/MapServer/13",
        "id_prefix": "bog_road",
        "name_column": "MVINPRINCI",  # Principal name (e.g., "006ABISA00")
        "class_column": "MVITCLA",  # Numeric class code (1-7)
        "class_mapping": BOGOTA_MALLA_VIAL_TYPE_MAPPING,
        "source_name": "IDECA Bogotá Malla Vial",
        "description": "Road network from Bogotá cadastre GIS (Malla Vial)",
        "bbox": (-74.25, 4.45, -73.95, 4.85),  # Bogotá bbox
    },
    {
        "name": "bogota_sidewalks",
        "url": "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/Mapa_Referencia/Mapa_Referencia/MapServer/16",
        "id_prefix": "bog_sidewalk",
        "name_column": None,  # Sidewalks typically unnamed
        "class_column": None,
        "class_mapping": None,
        "source_name": "IDECA Bogotá Andenes",
        "description": "Sidewalks (andenes) from Bogotá cadastre GIS",
        "bbox": (-74.25, 4.45, -73.95, 4.85),
    },
    {
        "name": "bogota_bike_network",
        "url": "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/Mapa_Referencia/Mapa_Referencia/MapServer/18",
        "id_prefix": "bog_bike",
        "name_column": None,
        "class_column": None,
        "class_mapping": None,
        "source_name": "IDECA Bogotá Ciclorutas",
        "description": "Bike paths (ciclorutas) from Bogotá cadastre GIS",
        "bbox": (-74.25, 4.45, -73.95, 4.85),
    },
]


# =============================================================================
# Buenos Aires, Argentina
# =============================================================================
# Source: https://data.buenosaires.gob.ar/dataset/calles
# GeoJSON download from city open data portal

BUENOS_AIRES_DATASETS = [
    {
        "name": "buenos_aires_streets",
        # Buenos Aires open data portal - requires manual download
        # Portal: https://data.buenosaires.gob.ar/dataset/calles
        # GeoJSON: https://data.buenosaires.gob.ar/dataset/calles/resource/2941f731-0a2e-4391-b8c9-a2912a80c081
        "url": "https://data.buenosaires.gob.ar/dataset/calles",  # Portal URL for reference
        "fetch_type": "manual",  # Requires manual download - portal blocks programmatic access
        "id_prefix": "ba_street",
        "name_column": "nombre",  # Street name in Spanish
        "class_column": "tipo",  # Street type
        "class_mapping": None,  # Use discover-classes
        "source_name": "Buenos Aires Calles",
        "description": "Street centerlines - requires manual download from portal",
        "bbox": (-58.55, -34.75, -58.30, -34.50),  # Buenos Aires bbox
        "portal_url": "https://data.buenosaires.gob.ar/dataset/calles",  # Manual download portal
        "notes": "Portal blocks programmatic access. Download GeoJSON manually.",
    },
]


# =============================================================================
# Amsterdam, Netherlands
# =============================================================================
# Source: https://maps.amsterdam.nl/open_geodata/
# Also: https://geodata.nationaalgeoregister.nl/nwbwegen/wfs (national road database)

AMSTERDAM_DATASETS = [
    {
        "name": "amsterdam_roads",
        # WFS endpoint for Dutch national road database (NWB)
        "url": "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0",
        "fetch_type": "wfs",  # Signal to use WFS fetcher
        "wfs_typename": "nwbwegen:wegvakken",  # Road segments layer
        "id_prefix": "ams_road",
        "name_column": "stt_naam",  # Street name
        "class_column": "wegbehsrt",  # Road manager type (proxy for class)
        "class_mapping": None,  # Use discover-classes
        "source_name": "PDOK NWB Wegen",
        "description": "Road segments from Dutch National Road Database (NWB)",
        "bbox": (4.75, 52.30, 5.00, 52.45),  # Amsterdam bbox
    },
]


# =============================================================================
# Helsinki, Finland
# =============================================================================
# Source: https://vayla.fi/en/transport-network/data/digiroad/data
# Digiroad national road and street database - download as GeoPackage

HELSINKI_DATASETS = [
    {
        "name": "helsinki_roads",
        # Digiroad download - K format (pre-segmented for GIS use)
        "url": "https://ava.vaylapilvi.fi/ava/Tie/Digiroad/Aineistojulkaisut/latest/DIGIROAD_K_KOKOMAA.gpkg.zip",
        "fetch_type": "download",  # Signal to download and extract
        "file_format": "gpkg",
        "id_prefix": "hel_road",
        "name_column": "TIENIMI_SU",  # Finnish road name
        "class_column": "TOIMINN_LK",  # Functional class
        "class_mapping": None,  # Use discover-classes
        "source_name": "Digiroad Finland",
        "description": "Road network from Finnish national Digiroad database",
        "bbox": (24.80, 60.10, 25.20, 60.30),  # Helsinki bbox
        "bbox_filter": True,  # Apply bbox filter after loading
    },
]


# =============================================================================
# Singapore
# =============================================================================
# Source: https://datamall.lta.gov.sg/content/datamall/en/static-data.html
# LTA DataMall static datasets (requires free API key registration)

SINGAPORE_DATASETS = [
    {
        "name": "singapore_roads",
        # LTA static data download URL (requires API key header)
        "url": "https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/RoadSectionLine.zip",
        "fetch_type": "download",
        "file_format": "shp",
        "id_prefix": "sg_road",
        "name_column": "RD_NAME",  # Road name
        "class_column": "RD_TYPE",  # Road type
        "class_mapping": None,  # Use discover-classes
        "source_name": "LTA Singapore Roads",
        "description": "Road section lines from Singapore LTA DataMall",
        "bbox": (103.60, 1.20, 104.05, 1.50),  # Singapore bbox
        "api_key_required": True,  # Note: needs LTA API key
        "api_key_header": "AccountKey",
    },
    {
        "name": "singapore_footpaths",
        "url": "https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/Footpath.zip",
        "fetch_type": "download",
        "file_format": "shp",
        "id_prefix": "sg_footpath",
        "name_column": None,
        "class_column": None,
        "class_mapping": None,
        "source_name": "LTA Singapore Footpaths",
        "description": "Footpaths from Singapore LTA DataMall",
        "bbox": (103.60, 1.20, 104.05, 1.50),
        "api_key_required": True,
        "api_key_header": "AccountKey",
    },
]


# =============================================================================
# Seoul, South Korea
# =============================================================================
# Source: https://data.seoul.go.kr (Seoul Open Data Plaza)
# Road centerlines - Korean portal

SEOUL_DATASETS = [
    {
        "name": "seoul_roads",
        # Seoul Open Data - road centerlines (도로중심선)
        # Note: Portal is primarily in Korean; URL subject to change
        "url": "https://data.seoul.go.kr/dataList/OA-164/S/1/datasetView.do",
        "fetch_type": "download",
        "file_format": "shp",
        "id_prefix": "seoul_road",
        "name_column": "ROAD_NAME",  # May vary based on actual schema
        "class_column": "ROAD_TYPE",  # May vary
        "class_mapping": None,  # Use discover-classes
        "source_name": "Seoul Open Data Roads",
        "description": "Road centerlines from Seoul Open Data Plaza (도로중심선)",
        "bbox": (126.75, 37.40, 127.20, 37.70),  # Seoul bbox
        "notes": "Korean portal - may require manual download and schema inspection",
    },
]


# =============================================================================
# Nairobi, Kenya
# =============================================================================
# Sources:
# - https://maps.krb.go.ke/ (Kenya Roads Board - official, newer data)
# - https://data.humdata.org/dataset/kenya-roads (HDX - humanitarian)

NAIROBI_DATASETS = [
    {
        "name": "nairobi_roads",
        # HDX dataset - Kenya roads shapefile (covers Nairobi region)
        # Source: World Food Programme / Humanitarian Data Exchange
        "url": "https://data.humdata.org/dataset/3d10cb5f-f56e-4924-b1fd-32931e0ddb41/resource/0170d6b6-7ad3-439a-ae1e-e84060886502/download/kenroads.zip",
        "fetch_type": "download",
        "file_format": "shp",
        "id_prefix": "nrb_road",
        "name_column": "NAME",
        "class_column": "CLASS",
        "class_mapping": None,  # Use discover-classes
        "source_name": "HDX Kenya Roads",
        "description": "Kenya roads from Humanitarian Data Exchange (clip to Nairobi bbox)",
        "bbox": (36.70, -1.40, 37.00, -1.15),  # Nairobi bbox
        "bbox_filter": True,  # Apply bbox filter after loading
    },
]


# =============================================================================
# All datasets for bulk operations
# =============================================================================
ALL_DATASETS = {
    # Active datasets with fetch scripts (US cities)
    "boston": BOSTON_DATASETS,
    "fort_collins": FORT_COLLINS_DATASETS,
    "frisco": FRISCO_DATASETS,
    "salt_lake_city": SALT_LAKE_CITY_DATASETS,
    "fresno": FRESNO_DATASETS,
    "utah_salt_lake": UTAH_SALT_LAKE_DATASETS,
    # Future datasets (configurations only, no fetch scripts yet)
    "ada_county": ADA_COUNTY_DATASETS,
    "utah_trails": UTAH_TRAILS_DATASETS,
    # International cities
    "cape_town": CAPE_TOWN_DATASETS,
    "bogota": BOGOTA_DATASETS,
    "buenos_aires": BUENOS_AIRES_DATASETS,
    "amsterdam": AMSTERDAM_DATASETS,
    "helsinki": HELSINKI_DATASETS,
    "singapore": SINGAPORE_DATASETS,
    "seoul": SEOUL_DATASETS,
    "nairobi": NAIROBI_DATASETS,
}


# =============================================================================
# City bounding boxes (for Overture reference data fetch)
# =============================================================================
CITY_BBOXES = {
    # US cities
    "boston": (-71.19, 42.21, -70.92, 42.40),
    "fort_collins": (-105.15, 40.45, -104.95, 40.65),
    "frisco": (-96.90, 33.10, -96.75, 33.20),
    "salt_lake_city": (-112.05, 40.70, -111.80, 40.85),
    "fresno": (-119.95, 36.65, -119.65, 36.90),
    # International cities
    "cape_town": (18.30, -34.10, 18.65, -33.85),
    "bogota": (-74.25, 4.45, -73.95, 4.85),
    "buenos_aires": (-58.55, -34.75, -58.30, -34.50),
    "amsterdam": (4.75, 52.30, 5.00, 52.45),
    "helsinki": (24.80, 60.10, 25.20, 60.30),
    "singapore": (103.60, 1.20, 104.05, 1.50),
    "seoul": (126.75, 37.40, 127.20, 37.70),
    "nairobi": (36.70, -1.40, 37.00, -1.15),
}
