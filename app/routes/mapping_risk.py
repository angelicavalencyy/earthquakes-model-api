"""Risk map endpoints for choropleth visualization."""
import traceback
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.db.session import get_session
from app.db.models import RegionRiskStatic

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


@router.get("/risk-map/geojson")
async def get_risk_map_geojson(session: AsyncSession = Depends(get_session)):
    """Get choropleth GeoJSON with region risk data."""
    try:
        query = select(RegionRiskStatic)
        result = await session.exec(query)
        data = result.all()

        def normalize_id(val):
            if not val:
                return None
            return str(val).strip().upper()

        def normalize_name(val):
            if not val:
                return None
            return str(val).strip().lower()

        risk_lookup = {normalize_id(item.id_kabupaten): item for item in data}
        name_lookup = {normalize_name(item.nama_kabupaten): item for item in data}

        features = []

        for boundary_feature in _load_boundary_features():
            boundary_properties = boundary_feature.get("properties", {})
            region_id = normalize_id(boundary_properties.get("id_kabupaten"))
            risk_item = risk_lookup.get(region_id)
            # fallback: try matching by name if id-based match failed
            if risk_item is None:
                region_name = normalize_name(boundary_properties.get("nama_kabupaten"))
                risk_item = name_lookup.get(region_name)
            has_risk_data = risk_item is not None

            frekuensi_gempa = (
                float(risk_item.frekuensi_gempa)
                if has_risk_data and risk_item.frekuensi_gempa is not None
                else 0.0
            )
            mag_max = (
                float(risk_item.mag_max)
                if has_risk_data and risk_item.mag_max is not None
                else 0.0
            )
            mag_mean = (
                float(risk_item.mag_mean)
                if has_risk_data and risk_item.mag_mean is not None
                else 0.0
            )
            depth_mean = (
                float(risk_item.depth_mean)
                if has_risk_data and risk_item.depth_mean is not None
                else 0.0
            )

            korban_total = (
                float(risk_item.korban_total)
                if has_risk_data and risk_item.korban_total is not None
                else 0.0
            )
            rumah_rusak_total = (
                float(risk_item.rumah_rusak_total)
                if has_risk_data and risk_item.rumah_rusak_total is not None
                else 0.0
            )
            fasum_rusak_total = (
                float(risk_item.fasum_rusak_total)
                if has_risk_data and risk_item.fasum_rusak_total is not None
                else 0.0
            )

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
                        "luas_wilayah_km2": (
                            float(boundary_properties["luas_wilayah_km2"])
                            if boundary_properties.get("luas_wilayah_km2") is not None
                            else None
                        ),
                        # fitur risiko dari DB
                        "frekuensi_gempa": frekuensi_gempa,
                        "mag_max": mag_max,
                        "mag_mean": mag_mean,
                        "depth_mean": depth_mean,
                        "korban_total": korban_total,
                        "rumah_rusak_total": rumah_rusak_total,
                        "fasum_rusak_total": fasum_rusak_total,
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

        return {
            "count": len(data),
            "data": [
                {
                    "id_kabupaten": item.id_kabupaten,
                    "nama_kabupaten": item.nama_kabupaten,
                    # fitur
                    "luas_wilayah_km2": (
                        float(item.luas_wilayah_km2) if item.luas_wilayah_km2 else None
                    ),
                    "frekuensi_gempa": (
                        float(item.frekuensi_gempa) if item.frekuensi_gempa else None
                    ),
                    "mag_max": float(item.mag_max) if item.mag_max else None,
                    "mag_mean": float(item.mag_mean) if item.mag_mean else None,
                    "depth_mean": float(item.depth_mean) if item.depth_mean else None,
                    "korban_total": (
                        float(item.korban_total) if item.korban_total else None
                    ),
                    "rumah_rusak_total": (
                        float(item.rumah_rusak_total)
                        if item.rumah_rusak_total
                        else None
                    ),
                    "fasum_rusak_total": (
                        float(item.fasum_rusak_total)
                        if item.fasum_rusak_total
                        else None
                    ),
                    # hasil ML
                    "cluster": item.cluster_label,
                    "risk_score": float(item.risk_score) if item.risk_score else None,
                    "risk_level": item.risk_level,
                    # PCA
                    "PC1": float(item.PC1) if item.PC1 else None,
                    "PC2": float(item.PC2) if item.PC2 else None,
                }
                for item in data
            ],
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
