# European Pedestrian & Cycling Dataset Research

This document outlines potentially high-value, non-OSM authoritative datasets for pedestrian and cycling infrastructure in Europe (and select hiking trails globally/regionally).

Research conducted on: **February 8, 2026**

## 1. Verified & Ready for Ingestion
These datasets have been fully verified with working download links or API access.

### 🇳🇱 Netherlands: NWB Wegen (National Road Database)
*   **Status:** **VERIFIED**
*   **Format:** GeoPackage (`.gpkg`)
*   **Config:** `datasets/nl_netherlands_paths.yaml`
*   **Direct Link:** Verified via `curl`. Includes dense path/cycleway network.

### 🇳🇴 Norway: NVDB Sidewalks (Fortau)
*   **Status:** **VERIFIED**
*   **Format:** Custom Fetch (JSON -> GPKG)
*   **Config:** `datasets/no_norway_sidewalks.yaml`
*   **Access:** Verified via custom fetcher `src/matcher/fetch/norway_nvdb.py`. Feature Type 48.

### 🇨🇭 Switzerland: Swisstopo Hiking Trails
*   **Status:** **VERIFIED**
*   **Format:** GeoPackage (`.gpkg`)
*   **Config:** `datasets/ch_switzerland_hiking.yaml`
*   **Direct Link:** Verified. Official high-quality trail network.

---

## 2. In-Progress / Premium
### 🇬🇧 Great Britain: OS NGD Pavement Link
*   **Status:** **PENDING / PREMIUM?**
*   **Issue:** API Key is valid for Names API but returns 401 for NGD Features API.
*   **Note:** If access cannot be granted via the free tier, we may need to use **OS Open Roads** (limited sidewalk data) or a manual export from the OS Data Hub.

---

## 2. Verified Direct Download Candidates
These datasets have been checked for availability and have direct download links verified.

### 🇳🇱 Netherlands: PDOK NWB (Nationaal Wegen Bestand)
*   **Dataset Name:** `NWB Wegen` / `Fietspaden`
*   **Description:** The national road database including separate footpaths and cycle paths (if they have street names).
*   **Type:** Cycle / Road / Pedestrian
*   **Format:** GeoPackage (`.gpkg`)
*   **URL:** `https://geo.rijkswaterstaat.nl/services/ogc/gdr/nwb_wegen/ows?service=WFS&version=2.0.0&request=GetFeature&typeName=wegvakken&outputFormat=application/geopackage%2Bsqlite3&content-disposition=attachment`
*   **Status:** **VERIFIED** (200 OK)

### 🇨🇭 Switzerland: Swisstopo Hiking Trails
*   **Dataset Name:** `swissTLM3D hiking trails`
*   **Description:** Official hiking trail network for Switzerland and Liechtenstein. Produced by Swisstopo in collaboration with Swiss Hiking Federation.
*   **Type:** Hiking / Trails (Authoritative)
*   **Format:** GeoPackage (`.gpkg`) inside ZIP
*   **URL:** `https://data.geo.admin.ch/ch.swisstopo.swisstlm3d-wanderwege/swisstlm3d-wanderwege/swisstlm3d-wanderwege_2056_5728.gpkg.zip`
*   **License:** Open Data (requires attribution).
*   **Status:** **VERIFIED** (200 OK, ~186MB)

### 🇫🇷 Paris: Cycle Paths
*   **Dataset Name:** `Réseau des itinéraires cyclables à Paris`
*   **Description:** Detailed cycle network for Paris and surrounding region.
*   **Type:** Cycling / Bike Lanes
*   **Format:** Shapefile (`.shp`) inside ZIP
*   **URL:** `https://www.data.gouv.fr/fr/datasets/r/f4555e8c-cdaf-4d08-8d22-a39a28e57927`
*   **License:** Open License (Licence Ouverte).
*   **Status:** **VERIFIED** (200 OK)

---

## 2. Large Scale / Pan-European Initiatives

### 🇪🇺 EuroGeographics: Open Maps For Europe (OME2)
*   **Description:** Harmonized, high-value large-scale geospatial data (1:10,000 scale).
*   **Coverage:** Prototype covers Belgium, France, and the Netherlands.
*   **Type:** Transport / Administrative
*   **Format:** GeoPackage
*   **Access:** [mapsforeurope.org](https://www.mapsforeurope.org/access-data) (Requires email registration for download links).

### 🇬🇧 UK: Ordnance Survey NGD (National Geographic Database)
*   **Description:** OS NGD Transport Network Collection. Contains `Pavement Link`, `Cycle Lane`, `Path`, and `Path Link` features.
*   **Type:** Very high resolution pedestrian and cycle infrastructure.
*   **Access:** [OS Data Hub](https://osdatahub.os.uk/downloads/open).
*   **Note:** The `NGD` is the successor to `OS MasterMap` and provides much better sidewalk/pavement connectivity than `OS Open Roads`.

---

## 3. High-Potential City Sources (Portals & Services)

### 🇧🇪 Brussels, Belgium
*   **Agency:** Brussels Mobility
*   **Dataset:** `Trottoirs` / `Accessibilité piétonne`
*   **Keywords:** PAVE (Plans d'Accessibilité de la Voirie et de l'Espace public).
*   **Access:** [opendata.brussels.be](https://opendata.brussels.be/) / [mobility.brussels](https://mobility.brussels/en/metadata-catalogue)

### 🇫🇮 Helsinki, Finland
*   **Source:** HRI (Helsinki Region Infoshare)
*   **Dataset:** `YLRE` (Register of public areas / Julkisten alueiden rekisteri).
*   **Features:** Contains detailed layers for street areas, green areas, and maintenance categories.
*   **Access:** WFS/WMS via [HRI.fi](https://hri.fi/).

### 🇦🇹 Vienna, Austria
*   **Dataset:** `Gehsteigkataster` (Sidewalk register) & `Hauptradverkehrsnetz`.
*   **Portal:** [data.wien.gv.at](https://data.wien.gv.at/)

### 🇩🇪 Berlin, Germany
*   **Portals:** [Berlin Open Data](https://daten.berlin.de/) / [FIS-Broker](https://fbinter.stadt-berlin.de/fb/index.jsp)
*   **Datasets:** `Radverkehrsanlagen` (Cycle facilities) & `Gehwege` (Sidewalks).

---

## 4. Recommendations for Ingestion

1.  **Netherlands (NWB):** High priority. The GeoPackage link is direct and verified.
2.  **Swisstopo (Hiking):** High priority for "reputable hiking trails" requirement. Direct link verified.
3.  **UK OS NGD:** Samples should be fetched to see if the `Pavement Link` geometry is usable for the matcher without excessive preprocessing.
4.  **Brussels (Sidewalks):** Potentially the most detailed sidewalk dataset if the PAVE data is accessible in vector format.