#!/usr/bin/env python3
"""
Rebuild kabupaten_boundaries.json from GADM Level 2 with proper ID formatting.
This ensures all ~502 kabupaten/kota from GADM are included in the GeoJSON.
"""
import json
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import mapping
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Paths
GADM_FILE = Path("data/raw/gadm41_IDN_2.json")
OUTPUT_FILE = Path("data/processed/kabupaten_boundaries.json")

if HAS_GEOPANDAS:
    print(f"Loading GADM from {GADM_FILE} (using GeoPandas)...")
    gdf = gpd.read_file(GADM_FILE)
    print(f"[OK] Loaded {len(gdf)} features from GADM")

    # Build GeoJSON FeatureCollection with proper ID
    features = []
    for idx, row in gdf.iterrows():
        # Extract GID_2 which contains hierarchical codes (e.g., "IDN.1.2_1")
        gid_2 = row.get("GID_2") or f"IDN.{idx}_{idx}"
        nama_kabupaten = row.get("NAME_2") or f"Region_{idx}"
        luas_wilayah = float(row.geometry.area) if row.geometry else 0.0
        
        feature = {
            "type": "Feature",
            "properties": {
                "id_kabupaten": gid_2.strip().upper(),
                "nama_kabupaten": nama_kabupaten.strip(),
                "luas_wilayah_km2": luas_wilayah,
            },
            "geometry": mapping(row.geometry),
        }
        features.append(feature)
else:
    print("(Warning) GeoPandas not available. Using raw JSON copy instead...")
    print(f"Loading GADM from {GADM_FILE}...")
    with open(GADM_FILE, "r", encoding="utf8") as f:
        gadm_data = json.load(f)
    
    features = []
    for idx, feature in enumerate(gadm_data.get("features", [])):
        props = feature.get("properties", {})
        gid_2 = props.get("GID_2") or f"IDN.{idx}_{idx}"
        nama_kabupaten = props.get("NAME_2") or f"Region_{idx}"
        
        # Update properties
        feature["properties"] = {
            "id_kabupaten": gid_2.strip().upper(),
            "nama_kabupaten": nama_kabupaten.strip(),
            "luas_wilayah_km2": props.get("AREA", 0.0),
        }
        features.append(feature)
    
    print(f"[OK] Loaded {len(features)} features from GADM")

geojson = {
    "type": "FeatureCollection",
    "name": "kabupaten_boundaries",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": features,
}

# Write output
print(f"Writing {len(features)} features to {OUTPUT_FILE}...")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=None)
print(f"[OK] Wrote {len(features)} features to {OUTPUT_FILE}")

