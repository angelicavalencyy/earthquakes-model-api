"""Risk map endpoints for choropleth visualization."""
import traceback
from pathlib import Path
import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.db.session import get_session
from app.db.models import RegionRiskStatic
from app.services.inference_risk_zone import (
    scaler as static_feature_scaler,
    feature_columns as static_feature_columns,
)

router = APIRouter()

BOUNDARY_GEOJSON_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "kabupaten_boundaries.json"
)


def _load_boundary_features() -> list[dict]:
    if not BOUNDARY_GEOJSON_PATH.exists():
        raise FileNotFoundError(f"Boundary geojson not found: {BOUNDARY_GEOJSON_PATH}")

    import json

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


def _build_boundary_area_lookup(boundary_features: list[dict]) -> dict[str, float]:
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
        boundary_area_lookup[_normalize_id(properties.get("id_kabupaten"))] = float(area_value)
        boundary_area_lookup[_normalize_name(properties.get("nama_kabupaten"))] = float(area_value)

    return boundary_area_lookup


def _inverse_scale_static_features(risk_item: RegionRiskStatic) -> dict[str, float | None]:
    feature_values = {
        "frekuensi_gempa": float(risk_item.frekuensi_gempa)
        if risk_item.frekuensi_gempa is not None
        else None,
        "depth_mean": float(risk_item.depth_mean)
        if risk_item.depth_mean is not None
        else None,
        "mag_max": float(risk_item.mag_max)
        if risk_item.mag_max is not None
        else None,
        "mag_mean": float(risk_item.mag_mean)
        if risk_item.mag_mean is not None
        else None,
        "korban_total": float(risk_item.korban_total)
        if risk_item.korban_total is not None
        else None,
        "rumah_rusak_total": float(risk_item.rumah_rusak_total)
        if risk_item.rumah_rusak_total is not None
        else None,
        "fasum_rusak_total": float(risk_item.fasum_rusak_total)
        if risk_item.fasum_rusak_total is not None
        else None,
    }

    if static_feature_scaler is None or static_feature_columns is None:
        return feature_values

    ordered_columns = list(static_feature_columns)
    candidate_values = [feature_values.get(column, 0.0) or 0.0 for column in ordered_columns]
    if not all(0.0 <= value <= 1.0 for value in candidate_values):
        return feature_values

    inverted = static_feature_scaler.inverse_transform(
        pd.DataFrame([candidate_values], columns=ordered_columns)
    )[0]

    return {
        column: float(value)
        for column, value in zip(ordered_columns, inverted, strict=False)
    }


def _build_cluster_row(
    item: RegionRiskStatic,
    boundary_area_lookup: dict[str, float],
) -> dict[str, float | int | str | None]:
    """Format one risk row to match clustered_k3.csv plus cluster metadata."""
    normalized = _inverse_scale_static_features(item)

    normalized_id = _normalize_id(item.id_kabupaten)
    normalized_name = _normalize_name(item.nama_kabupaten)
    area_value = boundary_area_lookup.get(normalized_id)
    if area_value is None:
        area_value = boundary_area_lookup.get(normalized_name)

    return {
        "id_kabupaten": item.id_kabupaten,
        "nama_kabupaten": item.nama_kabupaten,
        "luas_wilayah_km2": _format_area(area_value if area_value is not None else item.luas_wilayah_km2),
        "frekuensi_gempa": _format_whole_number(normalized.get("frekuensi_gempa")),
        "depth_mean": _format_area(normalized.get("depth_mean")),
        "mag_max": _format_area(normalized.get("mag_max")),
        "mag_mean": _format_area(normalized.get("mag_mean")),
        "korban_total": _format_whole_number(normalized.get("korban_total")),
        "rumah_rusak_total": _format_whole_number(normalized.get("rumah_rusak_total")),
        "fasum_rusak_total": _format_whole_number(normalized.get("fasum_rusak_total")),
        "cluster_label": item.cluster_label,
        "risk_score": float(item.risk_score) if item.risk_score is not None else None,
        "risk_level": item.risk_level,
        "PC1": float(item.PC1) if item.PC1 is not None else None,
        "PC2": float(item.PC2) if item.PC2 is not None else None,
    }


@router.get("/risk-map/geojson")
async def get_risk_map_geojson(session: AsyncSession = Depends(get_session)):
    """Get choropleth GeoJSON with region risk data."""
    try:
        query = select(RegionRiskStatic)
        result = await session.exec(query)
        data = result.all()

        boundary_area_lookup = _build_boundary_area_lookup(_load_boundary_features())

        risk_lookup = {_normalize_id(item.id_kabupaten): item for item in data}
        name_lookup = {_normalize_name(item.nama_kabupaten): item for item in data}

        features = []

        for boundary_feature in _load_boundary_features():
            boundary_properties = boundary_feature.get("properties", {})
            region_id = _normalize_id(boundary_properties.get("id_kabupaten"))
            risk_item = risk_lookup.get(region_id)
            # fallback: try matching by name if id-based match failed
            if risk_item is None:
                region_name = _normalize_name(boundary_properties.get("nama_kabupaten"))
                risk_item = name_lookup.get(region_name)
            has_risk_data = risk_item is not None

            cluster_label = (
                risk_item.cluster_label
                if has_risk_data and risk_item.cluster_label is not None
                else -1
            )
            risk_score = (
                float(risk_item.risk_score)
                if has_risk_data and risk_item.risk_score is not None
                else 0.0
            )
            risk_level = (
                risk_item.risk_level
                if has_risk_data and risk_item.risk_level is not None
                else "Tidak Ada Data"
            )

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "id_kabupaten": region_id,
                        "GID_2": region_id,
                        "nama_kabupaten": boundary_properties.get("nama_kabupaten"),
                        "luas_wilayah_km2": _format_area(
                            boundary_area_lookup.get(region_id)
                            or boundary_area_lookup.get(_normalize_name(boundary_properties.get("nama_kabupaten")))
                        ),
                        # fitur risiko dari DB
                        "frekuensi_gempa": _format_whole_number(_inverse_scale_static_features(risk_item).get("frekuensi_gempa")) if has_risk_data else 0,
                        "mag_max": _format_area(_inverse_scale_static_features(risk_item).get("mag_max")) if has_risk_data else 0.0,
                        "mag_mean": _format_area(_inverse_scale_static_features(risk_item).get("mag_mean")) if has_risk_data else 0.0,
                        "depth_mean": _format_area(_inverse_scale_static_features(risk_item).get("depth_mean")) if has_risk_data else 0.0,
                        "korban_total": _format_whole_number(_inverse_scale_static_features(risk_item).get("korban_total")) if has_risk_data else 0,
                        "rumah_rusak_total": _format_whole_number(_inverse_scale_static_features(risk_item).get("rumah_rusak_total")) if has_risk_data else 0,
                        "fasum_rusak_total": _format_whole_number(_inverse_scale_static_features(risk_item).get("fasum_rusak_total")) if has_risk_data else 0,
                        # hasil ML
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


@router.get("/risk-map/table")
async def get_risk_table(
    session: AsyncSession = Depends(get_session),
    limit: int | None = Query(default=None, ge=0),
    offset: int = Query(0),
    risk_level: str | None = Query(None),
):
    try:
        query = select(RegionRiskStatic)

        if risk_level:
            query = query.where(RegionRiskStatic.risk_level == risk_level)

        if limit is not None:
            query = query.limit(limit)
        query = query.offset(offset)

        result = await session.exec(query)
        data = result.all()

        boundary_area_lookup = _build_boundary_area_lookup(_load_boundary_features())

        return {
            "count": len(data),
            "data": [
                _build_cluster_row(item, boundary_area_lookup)
                for item in data
            ],
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
