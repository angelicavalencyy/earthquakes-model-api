"""Risk map endpoints for choropleth visualization."""
import traceback
from pathlib import Path
import json
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from fastapi import APIRouter, HTTPException, Query
from functools import lru_cache

# Use the new static risk data provider
from app.services.inference_risk_zone import (
    scaler as static_feature_scaler,
    feature_columns as static_feature_columns,
    get_static_risk_data,
)

router = APIRouter()

BOUNDARY_GEOJSON_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "gadm41_IDN_2.json"
)


@lru_cache(maxsize=1)
def _load_boundary_features() -> list[dict]:
    if not BOUNDARY_GEOJSON_PATH.exists():
        raise FileNotFoundError(f"Boundary geojson not found: {BOUNDARY_GEOJSON_PATH}")

    with BOUNDARY_GEOJSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return payload.get("features", [])


def _normalize_id(val):
    if not val:
        return None
    return str(val).strip().upper()


def _normalize_name(val):
    if not val:
        return None
    return str(val).strip().lower()


def _format_area(value):
    if value is None:
        return None
    return round(float(value), 2)


def _format_whole_number(value):
    if value is None:
        return None
    return int(round(float(value)))


@lru_cache(maxsize=1)
def _build_boundary_area_lookup() -> dict[str, float]:
    boundary_features = _load_boundary_features()
    boundary_area_lookup: dict[str, float] = {}

    if not boundary_features:
        return boundary_area_lookup

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)

    for feature in boundary_features:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        properties = feature.get("properties", {})
        projected_geometry = transform(transformer.transform, shape(geometry))
        area_value = projected_geometry.area / 10**6
        b_id = properties.get("id_kabupaten") or properties.get("GID_2")
        b_name = properties.get("nama_kabupaten") or properties.get("NAME_2")
        boundary_area_lookup[_normalize_id(b_id)] = float(area_value)
        boundary_area_lookup[_normalize_name(b_name)] = float(area_value)

    return boundary_area_lookup


def _build_cluster_row(
    item: dict,
    boundary_area_lookup: dict[str, float],
) -> dict:
    """Format one risk row. item is a dict from get_static_risk_data()["data"]."""
    
    normalized_id = _normalize_id(item.get("id_kabupaten"))
    normalized_name = _normalize_name(item.get("nama_kabupaten"))
    
    area_value = boundary_area_lookup.get(normalized_id)
    if area_value is None and normalized_name:
        area_value = boundary_area_lookup.get(normalized_name)

    # Note: Aliasing new ML features (mag_p95, mag_std) to old frontend expected names (mag_max, mag_mean)
    return {
        "id_kabupaten": item.get("id_kabupaten"),
        "nama_kabupaten": item.get("nama_kabupaten"),
        "luas_wilayah_km2": _format_area(area_value),
        "frekuensi_gempa": _format_whole_number(item.get("frekuensi_gempa", 0)),
        "depth_mean": _format_area(item.get("depth_std", 0.0)),
        "mag_max": _format_area(item.get("mag_p95", 0.0)),
        "mag_mean": _format_area(item.get("mag_std", 0.0)),
        "korban_total": _format_whole_number(item.get("korban_total", 0)),
        "rumah_rusak_total": _format_whole_number(item.get("rumah_rusak_total", 0)),
        "fasum_rusak_total": _format_whole_number(item.get("fasum_rusak_total", 0)),
        "cluster": item.get("cluster_label"),
        "cluster_label": item.get("cluster_label"),
        "risk_score": float(item.get("risk_score")) if item.get("risk_score") is not None else None,
        "risk_level": item.get("risk_level"),
    }


@router.get("/risk-map/table")
async def get_risk_table(
    limit: int | None = Query(default=None, ge=0),
    offset: int = Query(0),
    risk_level: str | None = Query(None),
):
    try:
        static_data = get_static_risk_data()
        data = static_data.get("data", [])

        if risk_level:
            data = [item for item in data if item.get("risk_level") == risk_level]
            
        total_count = len(data)

        if limit is not None:
            data = data[offset:offset + limit]
        else:
            data = data[offset:]

        boundary_area_lookup = _build_boundary_area_lookup()

        return {
            "count": total_count,
            "data": [
                _build_cluster_row(item, boundary_area_lookup)
                for item in data
            ],
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/kmed/risk-map")
async def get_kmed_risk_map():
    """Get the static choropleth GeoJSON-equivalent and risk data for KMedoids."""
    try:
        static_data = get_static_risk_data()
        data = static_data.get("data", [])

        boundary_area_lookup = _build_boundary_area_lookup()

        risk_lookup = {_normalize_id(item.get("id_kabupaten")): item for item in data}
        name_lookup = {_normalize_name(item.get("nama_kabupaten")): item for item in data}

        features = []

        for boundary_feature in _load_boundary_features():
            boundary_properties = boundary_feature.get("properties", {})
            b_id = boundary_properties.get("id_kabupaten") or boundary_properties.get("GID_2")
            b_name = boundary_properties.get("nama_kabupaten") or boundary_properties.get("NAME_2")
            
            region_id = _normalize_id(b_id)
            risk_item = risk_lookup.get(region_id)
            
            # fallback: try matching by name if id-based match failed
            if risk_item is None:
                region_name = _normalize_name(b_name)
                risk_item = name_lookup.get(region_name)
                
            has_risk_data = risk_item is not None
            
            cluster_label = risk_item.get("cluster_label") if has_risk_data else -1
            risk_score = float(risk_item.get("risk_score", 0.0)) if has_risk_data else 0.0
            risk_level = risk_item.get("risk_level", "Tidak Ada Data") if has_risk_data else "Tidak Ada Data"

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "id_kabupaten": region_id,
                        "GID_2": region_id,
                        "nama_kabupaten": b_name,
                        "luas_wilayah_km2": _format_area(
                            boundary_area_lookup.get(region_id)
                            or boundary_area_lookup.get(_normalize_name(b_name))
                        ),
                        "frekuensi_gempa": _format_whole_number(risk_item.get("frekuensi_gempa", 0)) if has_risk_data else 0,
                        "mag_max": _format_area(risk_item.get("mag_p95", 0.0)) if has_risk_data else 0.0,
                        "mag_mean": _format_area(risk_item.get("mag_std", 0.0)) if has_risk_data else 0.0,
                        "depth_mean": _format_area(risk_item.get("depth_std", 0.0)) if has_risk_data else 0.0,
                        "korban_total": _format_whole_number(risk_item.get("korban_total", 0)) if has_risk_data else 0,
                        "rumah_rusak_total": _format_whole_number(risk_item.get("rumah_rusak_total", 0)) if has_risk_data else 0,
                        "fasum_rusak_total": _format_whole_number(risk_item.get("fasum_rusak_total", 0)) if has_risk_data else 0,
                        "cluster": cluster_label,
                        "cluster_label": cluster_label,
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "has_risk_data": has_risk_data,
                    },
                    "geometry": boundary_feature.get("geometry"),
                }
            )

        return {"type": "FeatureCollection", "features": features}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e