#!/usr/bin/env python3
"""Diagnose boundary vs DB region ID matching.

Checks intersection count between boundary GeoJSON IDs and DB region_risk_static IDs.
"""
import json
import re
import asyncio
from pathlib import Path

try:
    from sqlmodel import select
    from app.db.main import async_engine
    from app.db.models import RegionRiskStatic
    HAS_DB_MODULES = True
except ImportError as e:
    print(f"(Warning) Cannot import DB modules: {e}")
    print("   Will only check boundary file against itself.")
    HAS_DB_MODULES = False


def norm_name(s):
    """Normalize region name for matching."""
    return (
        re.sub(
            r"[^0-9a-z]", "",
            (s or "").lower()
            .replace("kabupaten", "")
            .replace("kota", "")
        ).strip()
    )


async def diagnose_boundary_match():
    """Compare boundary file features with DB rows."""
    # Load boundary GeoJSON
    boundary_file = Path("data/processed/kabupaten_boundaries.json")
    if not boundary_file.exists():
        print(f"[ERROR] Boundary file not found: {boundary_file}")
        return
    
    with open(boundary_file, "r", encoding="utf8") as f:
        boundary_data = json.load(f)
    
    boundary_features = boundary_data.get("features", [])
    boundary_ids = {(f["properties"].get("id_kabupaten") or "").strip().upper() for f in boundary_features}
    boundary_names = {norm_name(f["properties"].get("nama_kabupaten") or "") for f in boundary_features}
    
    print("\n" + "="*60)
    print("BOUNDARY DIAGNOSTIC")
    print("="*60)
    print(f"Boundary file:        {boundary_file}")
    print(f"Boundary features:    {len(boundary_features):3d}")
    print(f"Unique IDs:           {len(boundary_ids):3d}")
    print(f"Unique names:         {len(boundary_names):3d}")
    
    if not HAS_DB_MODULES:
        print("\n(DB modules not available for full comparison)")
        print("="*60 + "\n")
        return
    
    # Query DB
    try:
        async with async_engine.connect() as conn:
            res = await conn.execute(
                select(RegionRiskStatic.id_kabupaten, RegionRiskStatic.nama_kabupaten)
            )
            rows = res.all()
    except (ConnectionError, ValueError, OSError) as e:
        print(f"\n[ERROR] Cannot query DB: {e}")
        print("="*60 + "\n")
        return
    
    db_ids = {(str(r[0]) or "").strip().upper() for r in rows}
    db_names = {norm_name(r[1]) for r in rows}
    
    # Report
    print(f"DB rows:              {len(rows):3d}")
    print(f"DB IDs:               {len(db_ids):3d}")
    print(f"DB names:             {len(db_names):3d}")
    print(f"IDs intersection:     {len(db_ids & boundary_ids):3d}")
    print(f"Names intersection:   {len(db_names & boundary_names):3d}")
    print("="*60)
    
    # Show mismatches if any
    id_missing_in_boundary = db_ids - boundary_ids
    id_missing_in_db = boundary_ids - db_ids
    
    if id_missing_in_boundary or id_missing_in_db:
        print("\nID MISMATCHES:")
        if id_missing_in_boundary:
            print(f"  In DB but NOT in boundary: {len(id_missing_in_boundary)} IDs")
            for uid in sorted(list(id_missing_in_boundary)[:5]):
                print(f"    - {uid}")
            if len(id_missing_in_boundary) > 5:
                print(f"    ... and {len(id_missing_in_boundary) - 5} more")
        if id_missing_in_db:
            print(f"  In boundary but NOT in DB: {len(id_missing_in_db)} IDs")
            for uid in sorted(list(id_missing_in_db)[:5]):
                print(f"    - {uid}")
            if len(id_missing_in_db) > 5:
                print(f"    ... and {len(id_missing_in_db) - 5} more")
    else:
        print("\n[OK] All IDs matched perfectly!")
    print()


if __name__ == "__main__":
    asyncio.run(diagnose_boundary_match())
