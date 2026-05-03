"""Realtime prediction API routes.

Provides endpoints to predict and list realtime earthquake predictions
using the ML inference helpers.
"""

import re
import traceback
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select, desc
from sqlmodel.ext.asyncio.session import AsyncSession
from app.db.session import get_session
from app.db.models import RealtimePredict
from app.services.inference_realtime import load_model, predict_realtime_from_bmkg
from app.services.bmkg_realtime import fetch_bmkg_data

router = APIRouter()

load_model()


def parse_float(val: str | None) -> float | None:
    """Parse a numeric string to float, return None on failure."""
    if val is None:
        return None
    cleaned = re.sub(r"[^\d.\-]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return None
    
def parse_lat(val: str | None) -> float | None:
    """Parse latitude text (may include hemisphere) into signed float."""
    if val is None:
        return None
    is_south = "LS" in str(val).upper()
    cleaned = re.sub(r"[^\d.\-]", "", str(val))
    try:
        num = float(cleaned)
        return -num if is_south else num
    except ValueError:
        return None

def parse_lon(val: str | None) -> float | None:
    """Parse longitude text into float, return None on failure."""
    if val is None:
        return None
    cleaned = re.sub(r"[^\d.\-]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return None


@router.post("/realtime")
def realtime_prediction(data: dict):
    """Synchronous wrapper endpoint to predict from BMKG payload."""
    return predict_realtime_from_bmkg(data)


@router.get("/realtime/auto")
async def realtime_auto(session: AsyncSession = Depends(get_session)):
    """Fetch BMKG data, predict, and persist realtime predictions.

    Returns a summary with counts and the prediction results.
    """
    try:
        raw_data = fetch_bmkg_data()
        results = predict_realtime_from_bmkg(raw_data)

        gempa_list = raw_data["Infogempa"]["gempa"]
        enriched_results = []
        inserted_count = 0
        updated_count = 0

        for bmkg_item, pred in zip(gempa_list, results["results"]):
            combined = {
                "Tanggal":    bmkg_item.get("Tanggal"),
                "Jam":        bmkg_item.get("Jam"),
                "Koordinat":  bmkg_item.get("Coordinates"),
                "Lintang":    bmkg_item.get("Lintang"),
                "Bujur":      bmkg_item.get("Bujur"),
                "Magnitude":  bmkg_item.get("Magnitude"),
                "Kedalaman":  bmkg_item.get("Kedalaman"),
                "Wilayah":    bmkg_item.get("Wilayah"),
                "cluster":    pred["cluster"],
                "risk_score": pred["risk_score"],
                "risk_level": pred["risk_level"],
            }
            enriched_results.append(combined)

            #  Cek apakah data sudah ada di DB
            existing_result = await session.exec(
                select(RealtimePredict).where(
                    RealtimePredict.tanggal  == combined["Tanggal"],
                    RealtimePredict.jam      == combined["Jam"],
                    RealtimePredict.koordinat == combined["Koordinat"],
                    RealtimePredict.latitude  == parse_lat(combined["Lintang"]),   
                    RealtimePredict.longitude == parse_lon(combined["Bujur"]),     
                )

            )
            
            existing_item = existing_result.first()
            if existing_item is not None:
                existing_item.cluster = combined["cluster"]
                existing_item.risk_score = combined["risk_score"]
                existing_item.risk_level = combined["risk_level"]
                existing_item.updated_at = datetime.now(timezone.utc)
                updated_count += 1
            else:
                now = datetime.now(timezone.utc)
                obj = RealtimePredict(
                    tanggal=combined["Tanggal"],
                    jam=combined["Jam"],
                    koordinat=combined["Koordinat"],
                    latitude=parse_lat(combined["Lintang"]),
                    longitude=parse_lon(combined["Bujur"]),
                    magnitude=parse_float(combined["Magnitude"]),
                    depth=parse_float(combined["Kedalaman"]),
                    wilayah=combined["Wilayah"],
                    cluster=combined["cluster"],
                    risk_score=combined["risk_score"],
                    risk_level=combined["risk_level"],
                    created_at=now,
                    updated_at=now,
                )
                session.add(obj)
                inserted_count += 1

        await session.commit()

        return {
            "count": len(enriched_results),        # total dari BMKG
            "inserted": inserted_count,
            "updated": updated_count,
            "skipped": 0,
            "results": enriched_results
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/realtime/history")
async def get_history(
    session: AsyncSession = Depends(get_session),
    limit: int | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    risk_level: str | None = Query(default=None),  # filter opsional
):
    """Return paginated realtime prediction history from the database."""
    try:
        query = select(RealtimePredict).order_by(desc(RealtimePredict.updated_at))

        #  Filter by risk_level kalau ada
        if risk_level:
            query = query.where(RealtimePredict.risk_level == risk_level)

        if limit is not None:
            query = query.limit(limit)
        query = query.offset(offset)
        result = await session.exec(query)
        data = result.all()

        #  Total count
        count_query = select(RealtimePredict)
        if risk_level:
            count_query = count_query.where(RealtimePredict.risk_level == risk_level)
        count_result = await session.exec(count_query)
        total = len(count_result.all())

        return {
            "total":      total,
            "returned":   len(data),
            "offset":     offset,
            "risk_level": risk_level or "all",
            "data": [
                {
                    "id":         str(item.id),
                    "tanggal":    item.tanggal,
                    "jam":        item.jam,
                    "koordinat":  item.koordinat,
                    "latitude":   float(item.latitude)  if item.latitude  else None,
                    "longitude":  float(item.longitude) if item.longitude else None,
                    "magnitude":  float(item.magnitude) if item.magnitude else None,
                    "depth":      float(item.depth)     if item.depth     else None,
                    "wilayah":    item.wilayah,
                    "cluster":    item.cluster,
                    "risk_score": float(item.risk_score) if item.risk_score else None,
                    "risk_level": item.risk_level,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in data
            ]
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e