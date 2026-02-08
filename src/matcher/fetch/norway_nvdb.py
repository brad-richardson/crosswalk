import requests
import json
import geopandas as gpd
from shapely import wkt
import pandas as pd
from pathlib import Path
from typing import Dict, Any

def fetch_nvdb_fortau(config: Dict[str, Any], output_path: Path) -> None:
    """
    Fetches 'Fortau' (Sidewalks) from Norway NVDB API v3.
    Feature Type ID: 48
    """
    base_url = "https://nvdbapiles-v3.atlas.vegvesen.no/vegobjekter/48"
    
    # User-Agent is required
    headers = {
        "Accept": "application/vnd.vegvesen.nvdb-v3+json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Example BBOX (Oslo Center) - In production, this should come from config 'bbox'
    # If config bbox is provided (minx, miny, maxx, maxy), use it.
    bbox = config.get('fetch', {}).get('bbox')
    
    params = {
        "inkluder": "egenskaper,geometri",
        "srid": 4326
    }
    
    if bbox:
        # NVDB expects min_lon,min_lat,max_lon,max_lat
        params["kartutsnitt"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    else:
        # Default to a small area in Oslo if no bbox is specified to avoid huge download
        print("No bbox specified, defaulting to Oslo center sample.")
        params["kartutsnitt"] = "10.7,59.9,10.8,59.92"

    print(f"Fetching NVDB data from {base_url} with params {params}...")
    
    # Handle pagination
    features = []
    next_url = base_url
    
    while next_url:
        try:
            # If next_url is absolute (from 'neste' link), use it directly, else use params
            if next_url == base_url:
                response = requests.get(next_url, headers=headers, params=params)
            else:
                response = requests.get(next_url, headers=headers)
                
            response.raise_for_status()
            data = response.json()
            
            for obj in data.get('objekter', []):
                geom_wkt = obj.get('geometri', {}).get('wkt')
                if geom_wkt:
                    features.append({
                        'id': obj['id'],
                        'geometry': wkt.loads(geom_wkt),
                        'properties': {p['navn']: p.get('verdi') for p in obj.get('egenskaper', [])}
                    })
            
            # Check for next page
            metadata = data.get('metadata', {})
            next_start = metadata.get('neste', {}).get('start')
            if next_start:
                 next_url = metadata['neste']['href']
            else:
                next_url = None
                
            print(f"Fetched {len(features)} features so far...")
            
            # Safety limit for testing
            if len(features) > 10000:
                print("Hit safety limit of 10,000 features. Stopping.")
                break
                
        except Exception as e:
            print(f"Error fetching NVDB page: {e}")
            break

    if not features:
        print("No features fetched.")
        return

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(features)
    gdf.set_geometry('geometry', inplace=True)
    gdf.crs = "EPSG:4326"
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")
    print(f"Saved {len(gdf)} features to {output_path}")

