"""Realtime prediction API routes.

Provides endpoints to predict and list realtime earthquake predictions
using the ML inference helpers.
"""

import re
import traceback
from datetime import datetime, timezone
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select, desc
from sqlmodel.ext.asyncio.session import AsyncSession
from app.db.session import get_session
from app.db.models import RealtimePredict
from app.services.inference_realtime import load_model, predict_realtime_from_bmkg, _map_to_kabupaten, get_region_frequency
from app.services.bmkg_realtime import fetch_bmkg_data
from app.db.bmkg_realtime import BMKGRealtimePayload

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
def realtime_prediction(payload: BMKGRealtimePayload):
    """Synchronous wrapper endpoint to predict from BMKG payload."""
    try:
        return predict_realtime_from_bmkg(payload.model_dump())
    except RuntimeError as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    except ValueError as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/realtime/auto")
async def realtime_auto(session: AsyncSession = Depends(get_session)):
    """Fetch BMKG data, predict, and persist realtime predictions."""
    try:
        raw_data = await fetch_bmkg_data()
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
            "count": len(enriched_results),
            "inserted": inserted_count,
            "updated": updated_count,
            "skipped": 0,
            "results": enriched_results
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e

async def _recompute_realtime_history(session: AsyncSession) -> dict:
    """Internal function to run predictions on historical data and update the database."""
    query = select(RealtimePredict).order_by(desc(RealtimePredict.updated_at))
    result = await session.exec(query)
    data = result.all()
    
    updated_count = 0
    errors = []
    
    for item in data:
        try:
            payload = {
                "Lintang": str(item.latitude),
                "Bujur": str(item.longitude),
                "Magnitude": str(item.magnitude),
                "Kedalaman": str(item.depth) + " km",
                "Wilayah": item.wilayah,
                "Tanggal": item.tanggal,
                "Jam": item.jam,
                "Coordinates": item.koordinat
            }
            
            pred = predict_realtime_from_bmkg(payload)
            
            changed = False
            if item.cluster != pred.get("cluster_label"):
                item.cluster = pred.get("cluster_label")
                changed = True
            if float(item.risk_score) != float(pred.get("risk_score", 0)) if item.risk_score else True:
                item.risk_score = pred.get("risk_score")
                changed = True
            if item.risk_level != pred.get("risk_level"):
                item.risk_level = pred.get("risk_level")
                changed = True
                
            if changed:
                session.add(item)
                updated_count += 1
                
        except Exception as e:
            errors.append({"id": str(item.id), "error": str(e)})
            
    if updated_count > 0:
        await session.commit()
        
    return {
        "total_processed": len(data),
        "updated_count": updated_count,
        "errors": errors
    }


@router.get("/realtime/history")
async def get_history(
    limit: int | None = Query(None, description="Max records to return. Default is all.", ge=1),
    offset: int = Query(0, description="Pagination offset", ge=0),
    risk_level: str | None = Query(None, description="Filter by exact risk_level string"),
    recompute: bool = Query(False, description="If true, re-run model predictions on all history before returning"),
    session: AsyncSession = Depends(get_session)
):
    """Get historical realtime predictions from the database."""
    try:
        recompute_summary = None
        if recompute:
            recompute_summary = await _recompute_realtime_history(session)

        count_query = select(RealtimePredict)
        if risk_level:
            count_query = count_query.where(RealtimePredict.risk_level == risk_level)
        count_result = await session.exec(count_query)
        total = len(count_result.all())

        query = select(RealtimePredict).order_by(desc(RealtimePredict.updated_at))
        if risk_level:
            query = query.where(RealtimePredict.risk_level == risk_level)
            
        if limit is not None:
            query = query.limit(limit)
        query = query.offset(offset)

        result = await session.exec(query)
        data = result.all()

        response = {
            "total":      total,
            "returned":   len(data),
            "offset":     offset,
            "risk_level": risk_level or "all",
            "data":       []
        }
        if recompute_summary is not None:
            response["recompute"] = recompute_summary

        for item in data:
            row = {
                "id":         str(item.id),
                "tanggal":    item.tanggal,
                "jam":        item.jam,
                "koordinat":  item.koordinat,
                "latitude":   float(item.latitude)  if item.latitude  else None,
                "longitude":  float(item.longitude) if item.longitude else None,
                "magnitude":  float(item.magnitude) if item.magnitude else None,
                "depth":      float(item.depth)     if item.depth     else None,
                "wilayah":    item.wilayah,
                "gadm_id":    "offshore",
                "gadm_name":  "Lepas Pantai / Luar Wilayah",
                "frekuensi_gempa": 0,
                "cluster":    item.cluster,
                "risk_score": float(item.risk_score) if item.risk_score is not None else None,
                "risk_level": item.risk_level,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            }
            
            lat = row.get("latitude")
            lon = row.get("longitude")
            if lat is not None and lon is not None:
                val_id, val_name = _map_to_kabupaten(lat, lon)
                if val_id is not None:
                    row["gadm_id"] = str(val_id)
                    row["frekuensi_gempa"] = get_region_frequency(str(val_id)) or 0
                if val_name is not None:
                    row["gadm_name"] = str(val_name)
                    
            response["data"].append(row)

        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/realtime/train-status")
async def get_realtime_train_status():
    """Get the latest training logs and metrics for the realtime risk model."""
    try:
        from app.services.inference_realtime import (
            model_version, trained_at, model_hash, total_data_trained, total_earthquakes_trained
        )
        
        # Parse cluster count from model_version if possible (e.g. "kmedoids-manhattan-k3" -> 3)
        cluster_count = 3  # Default fallback
        if model_version and "-k" in model_version:
            try:
                cluster_count = int(model_version.split("-k")[-1].split("-")[0])
            except ValueError:
                pass
                
                
        return {
            "model_version": "Model Versi 1",  # "nanti di count" -> for now hardcode 1 or a placeholder
            "cluster": cluster_count,
            "trained_at": trained_at,
            "model_hash": model_hash,
            "total_data_trained": total_data_trained,
            "total_earthquakes_trained": total_earthquakes_trained
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e