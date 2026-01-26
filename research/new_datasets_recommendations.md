# Road/Sidewalk Dataset Recommendations for Network Extension

## Research Summary

Verified data availability for cities with focus on sidewalk/pedestrian infrastructure data.

### Original Selections Assessment (UPDATED)

| City | Sidewalk Data | Road Data | Status |
|------|--------------|-----------|--------|
| Frisco, TX | Trails only | Roads available | Both confirmed |
| Fort Collins, CO | **Sidewalks available** | Streets available | **Both confirmed** |
| Salt Lake City, UT | Dedicated sidewalk layer | Streets available | Both confirmed |
| Edmonton, Canada | Survey data only | Not found | NO geospatial data |
| Ada County/Boise, ID | Not available | Roads available | Roads confirmed |

---

## Top 3 Recommended Datasets (with Sidewalk Focus)

### 1. Fort Collins, CO - Sidewalks AND Streets (CONFIRMED)

**Sidewalks FeatureServer:**
```
https://services1.arcgis.com/dLpFH5mwVvxSN4OE/arcgis/rest/services/Sidewalks/FeatureServer/0
```

**Sidewalk Fields:**
| Field | Type | Values |
|-------|------|--------|
| TYPE | SmallInteger | 1=Attached, 2=Detached, 3=Missing, 4=No Roadway Improvements, 5=Drive Approach |
| WIDTH | Double | Sidewalk width in feet |

**Street Centerlines FeatureServer:**
```
https://services1.arcgis.com/dLpFH5mwVvxSN4OE/arcgis/rest/services/Street_Centerlines/FeatureServer/0
```

**Street Fields:**
| Field | Type | Description |
|-------|------|-------------|
| STRNAME | String | Street name |
| STREETTYPE | String | INTERSTATE, ARTERIAL, ARTERIAL MAJOR, HIGHWAY, HIGHWAY DIVIDED, COLLECTOR, LOCAL, RAMP, ROUNDABOUT, ALLEY |
| ONEWAY | String | One-way designation |
| PRIVSTRT | String | Private street indicator |
| STATUS | String | Status indicator |

**Why Good:**
- **540 miles of sidewalk inventory** with attachment type classification
- Excellent sidewalk data quality (attached vs detached distinction)
- Street classification with 10 road types
- College town with diverse infrastructure
- Active development and GIS department

---

### 2. Salt Lake City, UT - Sidewalk Network (CONFIRMED)

**Data Portal:** https://gis-slcgov.opendata.arcgis.com/datasets/sidewalk-1

**Data Type:** Dedicated sidewalk centerlines

**Download Formats:** Shapefile, GeoJSON, CSV, KML (via portal)

**Why Good:**
- Explicit sidewalk infrastructure layer (rare)
- High-growth Utah market (2nd in US for new construction)
- Downloads available in multiple formats
- Maintained by city GIS department

---

### 3. Frisco, TX - Trails AND Roads (CONFIRMED)

**Trails FeatureServer:**
```
https://maps.friscotexas.gov/gis/rest/services/Public/FriscoCommunityMaps/FeatureServer/10
```

**Trail Fields:**
| Field | Type | Values |
|-------|------|--------|
| TrailType | String | Bike Only, Parkway, Regional, Local, Nature, Golf Course |
| BikeRouteType | String | Bike Lane, Buffered Bike Lane, Shared Roadway |
| Surface | String | Concrete, Unimproved |
| Width | String | 4' through 12' |
| WalkingLoop | String | Yes/No |
| LifeCycleStatus | String | Existing, Future, Under Construction, Abandoned |

**Road Centerlines FeatureServer:**
```
https://maps.friscotexas.gov/gis/rest/services/Public/FriscoCommunityMaps/FeatureServer/8
```

**Road Fields:**
| Field | Type | Description |
|-------|------|-------------|
| ROAD_NAME | String | Street name |
| SUBTYPE | Integer | 1=Tollway, 2=State Hwy, 3=US Hwy, 4=Major Thoroughfare, 5=Minor Thoroughfare, 6=Collector, 7=Residential, 8=Ramp, 9=Driveway |
| SURFACE_CODE | String | Concrete, HMAC, Unimproved, Gravel |
| MPH | Integer | Speed limit (0-70) |
| LANES | Integer | Number of lanes (1-10) |
| LIFECYCLESTATUS | String | Proposed, Completed, Constructed |

**Why Good:**
- Explosive growth suburb (18.2% annual)
- High OSM gap likelihood for new subdivisions
- Trail data includes pedestrian infrastructure
- Excellent road classification with 9 types
- Lifecycle status captures new construction

---

## Additional Road-Only Datasets

### Ada County, ID (Boise Area) - Road Centerlines (CONFIRMED)

**MapServer Endpoint:**
```
https://maps.achdidaho.org/server/rest/services/Assessor/roadcenterline/MapServer/0
```

**Key Fields:**
| Field | Type | Description |
|-------|------|-------------|
| StName | String | Street name |
| FuncClass | String (50) | Functional classification |
| PostSpeed | Integer | Posted speed limit |
| OneWay | String (2) | One-way designation |
| Private | String (1) | Private road indicator |

**Why Good:**
- Idaho has highest rate of new construction in US (21.2 units/1000 homes)
- Covers Boise metro area
- Functional classification field for road hierarchy

**Note:** No sidewalk data available for Boise area.

---

### Utah Statewide - Trails and Pathways (CONFIRMED)

**FeatureServer Endpoint:**
```
https://services1.arcgis.com/99lidPhWCzftIe9K/ArcGIS/rest/services/TrailsAndPathways/FeatureServer/0
```

**Key Fields:**
| Field | Description |
|-------|-------------|
| PrimaryName | Trail name |
| DesignatedUses | Permitted uses (hiking, biking, etc.) |
| SurfaceType | Trail surface material |
| Class | Trail classification |
| ADAAccessible | Accessibility status |
| TransNetwork | Part of transportation network |

**Coverage:** Statewide including Salt Lake City metro area

---

## Summary: All Confirmed Endpoints

| City | Layer | Endpoint |
|------|-------|----------|
| Fort Collins | Sidewalks | `https://services1.arcgis.com/dLpFH5mwVvxSN4OE/arcgis/rest/services/Sidewalks/FeatureServer/0` |
| Fort Collins | Streets | `https://services1.arcgis.com/dLpFH5mwVvxSN4OE/arcgis/rest/services/Street_Centerlines/FeatureServer/0` |
| Salt Lake City | Sidewalk | Portal download: https://gis-slcgov.opendata.arcgis.com/datasets/sidewalk-1 |
| Frisco TX | Trails | `https://maps.friscotexas.gov/gis/rest/services/Public/FriscoCommunityMaps/FeatureServer/10` |
| Frisco TX | Roads | `https://maps.friscotexas.gov/gis/rest/services/Public/FriscoCommunityMaps/FeatureServer/8` |
| Ada County | Roads | `https://maps.achdidaho.org/server/rest/services/Assessor/roadcenterline/MapServer/0` |
| Utah State | Trails | `https://services1.arcgis.com/99lidPhWCzftIe9K/ArcGIS/rest/services/TrailsAndPathways/FeatureServer/0` |

---

## Fetch Script Templates

### Fort Collins (Sidewalks + Streets)

```python
FORT_COLLINS_DATASETS = [
    {
        "name": "fort_collins_sidewalks",
        "url": "https://services1.arcgis.com/dLpFH5mwVvxSN4OE/arcgis/rest/services/Sidewalks/FeatureServer/0",
        "id_prefix": "fc_sidewalk",
        "name_column": None,  # Sidewalks don't have names
        "class_column": "TYPE",
        "class_mapping": {
            1: "footway",  # Attached
            2: "footway",  # Detached
            3: "footway",  # Missing (for gap analysis)
            4: "footway",  # No Roadway Improvements
            5: "footway",  # Drive Approach
        },
        "subclass_column": "TYPE",
        "subclass_mapping": {
            1: "sidewalk",
            2: "sidewalk",
            3: "sidewalk",
            4: "sidewalk",
            5: "crossing",
        },
        "source_name": "Fort Collins Sidewalks",
    },
    {
        "name": "fort_collins_streets",
        "url": "https://services1.arcgis.com/dLpFH5mwVvxSN4OE/arcgis/rest/services/Street_Centerlines/FeatureServer/0",
        "id_prefix": "fc_street",
        "name_column": "STRNAME",
        "class_column": "STREETTYPE",
        "class_mapping": {
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
        },
        "source_name": "Fort Collins Street Centerlines",
    },
]
```

### Frisco TX (Trails + Roads)

```python
FRISCO_DATASETS = [
    {
        "name": "frisco_trails",
        "url": "https://maps.friscotexas.gov/gis/rest/services/Public/FriscoCommunityMaps/FeatureServer/10",
        "id_prefix": "frisco_trail",
        "name_column": "Name",
        "class_column": "TrailType",
        "class_mapping": {
            "Bike Only": "cycleway",
            "Parkway": "path",
            "Regional": "path",
            "Local": "footway",
            "Nature": "path",
            "Golf Course": "path",
        },
        "source_name": "Frisco Trails",
    },
    {
        "name": "frisco_roads",
        "url": "https://maps.friscotexas.gov/gis/rest/services/Public/FriscoCommunityMaps/FeatureServer/8",
        "id_prefix": "frisco_road",
        "name_column": "ROAD_NAME",
        "class_column": "SUBTYPE",
        "class_mapping": {
            1: "motorway",      # Tollway
            2: "primary",       # State Highway
            3: "primary",       # US Highway
            4: "secondary",     # Major Thoroughfare
            5: "tertiary",      # Minor Thoroughfare
            6: "tertiary",      # Collector
            7: "residential",   # Residential
            8: "motorway_link", # Ramp
            9: "service",       # Driveway
        },
        "source_name": "Frisco Road Centerlines",
    },
]
```

### Ada County/Boise (Roads Only)

```python
ADA_COUNTY_DATASETS = [
    {
        "name": "ada_county_roads",
        "url": "https://maps.achdidaho.org/server/rest/services/Assessor/roadcenterline/MapServer/0",
        "id_prefix": "ada_road",
        "name_column": "StName",
        "class_column": "FuncClass",
        "class_mapping": {
            # Will need to inspect actual FuncClass values
        },
        "source_name": "Ada County Road Centerlines",
    },
]
```

---

## Next Steps

1. **Fort Collins:** Create fetch script for sidewalks + streets
2. **Salt Lake City:** Download sidewalk shapefile from portal, analyze schema
3. **Frisco TX:** Create fetch script for trails + roads
4. Run matching pipeline against Overture to quantify gaps

---

## Data Portal Links

| Dataset | Portal URL |
|---------|-----------|
| Fort Collins GIS | https://data-fcgov.opendata.arcgis.com/ |
| Salt Lake City Sidewalk | https://gis-slcgov.opendata.arcgis.com/datasets/sidewalk-1 |
| Frisco Hub | https://geodata-frisco.hub.arcgis.com/ |
| Utah Trails | https://gis.utah.gov/products/sgid/recreation/trails-pathways/ |
| Ada County GIS | https://maps.achdidaho.org/ |
| Edmonton Open Data | https://data.edmonton.ca/ (no sidewalk geospatial data) |
