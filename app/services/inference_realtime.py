"""Realtime earthquake inference service.

Loads a pre-trained region-level clustering model and predicts risk for
incoming BMKG earthquake events by mapping them to kabupaten regions.

Dynamic behaviour: each new event updates the region's running statistics
(frekuensi, magnitude stats, depth, shallow ratio), so risk levels can
change over time as earthquakes accumulate in a region.
"""

import re
import pickle
import logging
import traceback
from pathlib import Path

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import json

# Add src to path so pickle can resolve 'realtime_helpers' custom classes
src_path = Path(__file__).resolve().parents[2] / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Remove legacy CLUSTER_FEATURE_COLUMNS, we use the ones from the loaded model

# ── Global model cache ──────────────────────────────────────────────
model = None
scaler = None
feature_columns = None
feature_weights = None
cluster_risk_map = None
model_version = None
trained_at = None
model_hash = None
gadm_gdf = None
log_transform_cols = []
metrics = None
total_data_trained = 0
total_earthquakes_trained = 0
clip_bounds = {}
_pkl_mtime = 0.0  # Track file modification time to avoid wasteful reloads

# Running region statistics for dynamic updates
_region_stats = None  # DataFrame keyed by id_kabupaten
_region_events = {}   # id_kabupaten -> list of {magnitude, depth_km}


def _derive_risk_thresholds() -> tuple[float, float]:
    """Derive (low/med, med/high) thresholds from trained cluster scores."""
    scores = []
    if isinstance(cluster_risk_map, dict):
        for info in cluster_risk_map.values():
            if not isinstance(info, dict):
                continue
            s = info.get("risk_score")
            if s is not None:
                try:
                    scores.append(float(s))
                except (TypeError, ValueError):
                    continue
    scores = sorted(set(scores))
    if len(scores) >= 3:
        return (scores[0] + scores[1]) / 2.0, (scores[1] + scores[2]) / 2.0
    return 1.0 / 3.0, 2.0 / 3.0


def _risk_level(score: float, thresholds: tuple[float, float]) -> str:
    if score < thresholds[0]:
        return "Rendah"
    if score < thresholds[1]:
        return "Sedang"
    return "Tinggi"


def load_model():
    """Load the pre-trained realtime model into module-level globals.
    
    Smart reload: only re-reads the .pkl file when its modification time
    has changed on disk (e.g. after the weekly scheduler retrain).
    """
    global model, scaler, feature_columns, feature_weights
    global cluster_risk_map, model_version, trained_at, model_hash
    global gadm_gdf, log_transform_cols, _region_stats
    global _pkl_mtime, _region_events

    # Cari model pkl: prioritaskan best
    pkl_path = None
    for candidate in [
        Path("app/ml/kmed/realtime/realtime_model_best.pkl"),
    ]:
        if candidate.exists():
            pkl_path = candidate
            break

    if pkl_path is None:
        logging.warning("No realtime model pickle found")
        return {"error": "model file not found"}

    # ── Smart reload: skip jika file belum berubah ───────────────────
    current_mtime = pkl_path.stat().st_mtime
    if model is not None and current_mtime == _pkl_mtime:
        return  # Model sudah loaded dan file belum berubah

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in ('__main__', '__mp_main__'):
                try:
                    import src.train_realtime as tr
                    if hasattr(tr, name):
                        return getattr(tr, name)
                except Exception:
                    pass
            return super().find_class(module, name)

    try:
        with pkl_path.open("rb") as f:
            data = CustomUnpickler(f).load()
    except Exception as exc:
        logging.error("Failed to load model %s: %s", pkl_path, exc)
        return {"error": str(exc)}

    model = data.get("model")
    scaler = data.get("scaler")
    feature_columns = data.get("feature_columns", [])
    feature_weights = data.get("feature_weights", {})
    cluster_risk_map = data.get("cluster_risk_map", {})
    model_version = data.get("model_version", "unknown")
    trained_at = data.get("trained_at")
    model_hash = data.get("model_hash")
    log_transform_cols = data.get("log_transform_cols", [])
    
    # Load winsorization clip bounds for consistent preprocessing
    global clip_bounds
    clip_bounds = data.get("clip_bounds", {})
    
    # Store metrics for tracking changes
    global metrics, total_data_trained, total_earthquakes_trained
    metrics = data.get("metrics", {})
    
    # Load base region features from model
    region_records = data.get("region_features")
    if region_records:
        _region_stats = pd.DataFrame(region_records)
        total_data_trained = len(_region_stats)
        if "frekuensi_gempa" in _region_stats.columns:
            total_earthquakes_trained = int(_region_stats["frekuensi_gempa"].sum())
        else:
            total_earthquakes_trained = 0
    else:
        _region_stats = None
        total_data_trained = 0
        total_earthquakes_trained = 0

    # Reset dynamic event cache (model baru = statistik baru)
    _region_events.clear()
    
    # Update mtime tracker
    _pkl_mtime = current_mtime

    # Load GADM boundaries
    if gadm_gdf is None:
        try:
            with open("data/raw/gadm41_IDN_2.json", "r", encoding="utf-8") as gf:
                gadm_raw = json.load(gf)
            gdf = gpd.GeoDataFrame.from_features(gadm_raw["features"])
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf.rename(columns={"GID_2": "id_kabupaten", "NAME_2": "nama_kabupaten"})
            gadm_gdf = gdf[["id_kabupaten", "nama_kabupaten", "geometry"]]
        except Exception:
            gadm_gdf = None

    logging.info("Loaded realtime model: %s from %s (mtime=%.0f)", model_version, pkl_path, _pkl_mtime)


# ── Parsing helpers ──────────────────────────────────────────────────
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _parse_float(value, *, field_name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required field: {field_name}")
    if isinstance(value, (int, float)):
        return float(value)
    m = _FLOAT_RE.search(str(value).strip())
    if not m:
        raise ValueError(f"Invalid numeric value for {field_name}: {value!r}")
    return float(m.group(0))

def _parse_lat(text) -> float | None:
    if text is None:
        return None
    t = str(text).upper()
    is_south = "LS" in t
    m = _FLOAT_RE.search(t)
    if not m:
        return None
    num = float(m.group(0))
    return -num if is_south else num

def _parse_lon(text) -> float | None:
    if text is None:
        return None
    m = _FLOAT_RE.search(str(text))
    return float(m.group(0)) if m else None


def _map_to_kabupaten(lat: float, lon: float) -> tuple[str | None, str | None]:
    """Map coordinates to kabupaten using GADM spatial join."""
    if gadm_gdf is None or lat is None or lon is None:
        return None, None
    try:
        pt = gpd.GeoDataFrame(
            [{"geometry": gpd.points_from_xy([lon], [lat])[0]}],
            geometry="geometry", crs="EPSG:4326"
        )
        # Try point-in-polygon first
        join = gpd.sjoin(pt, gadm_gdf, how="left", predicate="within")
        if not join.empty and pd.notna(join.iloc[0].get("id_kabupaten")):
            return join.iloc[0]["id_kabupaten"], join.iloc[0]["nama_kabupaten"]
        # Fallback: nearest (sea events)
        gm = gadm_gdf.to_crs(epsg=3395)
        pm = pt.to_crs(epsg=3395)
        near = gpd.sjoin_nearest(pm, gm, how="left", distance_col="_d")
        if not near.empty:
            return near.iloc[0]["id_kabupaten"], near.iloc[0]["nama_kabupaten"]
    except Exception:
        pass
    return None, None


def _get_region_features(kab_id: str) -> dict | None:
    """Get current aggregated features for a kabupaten."""
    if _region_stats is None or kab_id is None:
        return None
    row = _region_stats[_region_stats["id_kabupaten"] == kab_id]
    if row.empty:
        return None
    
    # Extract only the feature columns known to the loaded model
    feats = {}
    for col in feature_columns:
        if col in row:
            feats[col] = float(row.iloc[0][col])
        else:
            feats[col] = 0.0
    return feats

def get_region_frequency(kab_id: str) -> int | None:
    """Get the current dynamic frequency count for a kabupaten."""
    if _region_stats is None or kab_id is None:
        return None
    row = _region_stats[_region_stats["id_kabupaten"] == kab_id]
    if row.empty:
        return None
    return int(row.iloc[0].get("frekuensi_gempa", 0))


def _update_region_stats(kab_id: str, magnitude: float, depth_km: float):
    """Dynamically update region statistics with a new earthquake event."""
    global _region_stats
    if _region_stats is None or kab_id is None:
        return

    # Track raw events for accurate recalculation
    if kab_id not in _region_events:
        # Initialize from base stats
        row = _region_stats[_region_stats["id_kabupaten"] == kab_id]
        if not row.empty:
            def _safe_get(k1, k2, default_val):
                v = row.iloc[0].get(k1)
                if pd.notna(v): return v
                v = row.iloc[0].get(k2)
                if pd.notna(v): return v
                return default_val

            n = int(_safe_get("_event_count", "frekuensi_gempa", 0))
            mag_val = float(_safe_get("mag_mean", "mag_p95", magnitude))
            depth_val = float(_safe_get("depth_mean", "depth_std", depth_km))
            _region_events[kab_id] = {
                "magnitudes": [mag_val] * n if n > 0 else [],
                "depths": [depth_val] * n if n > 0 else [],
            }
        else:
            _region_events[kab_id] = {"magnitudes": [], "depths": []}

    # Add new event
    _region_events[kab_id]["magnitudes"].append(magnitude)
    _region_events[kab_id]["depths"].append(depth_km)

    mags = _region_events[kab_id]["magnitudes"]
    deps = _region_events[kab_id]["depths"]

    # Recompute features dynamically if they exist in the model's feature set
    mask = _region_stats["id_kabupaten"] == kab_id
    if mask.any():
        idx = _region_stats[mask].index[0]
        n_events = len(mags)
        _region_stats.loc[idx, "frekuensi_gempa"] = n_events
        _region_stats.loc[idx, "_event_count"] = n_events
        
        if "seismic_density" in feature_columns and "luas_wilayah_km2" in _region_stats.columns:
            luas = _region_stats.loc[idx, "luas_wilayah_km2"]
            _region_stats.loc[idx, "seismic_density"] = n_events / max(luas, 1.0)
            
        if "mag_p95" in feature_columns:
            _region_stats.loc[idx, "mag_p95"] = float(np.percentile(mags, 95)) if mags else 0.0
        if "mag_std" in feature_columns:
            _region_stats.loc[idx, "mag_std"] = float(np.std(mags)) if len(mags) > 1 else 0.0
            
        if "depth_std" in feature_columns:
            _region_stats.loc[idx, "depth_std"] = float(np.std(deps)) if len(deps) > 1 else 0.0
        if "shallow_ratio" in feature_columns:
            shallow = sum(1 for d in deps if d < 70)
            _region_stats.loc[idx, "shallow_ratio"] = shallow / n_events if n_events else 0.0

        if "mag_max" in feature_columns:
            current_max = _region_stats.loc[idx, "mag_max"] if "mag_max" in _region_stats.columns else 0.0
            _region_stats.loc[idx, "mag_max"] = max(float(current_max), float(np.max(mags)))
        if "mag_mean" in feature_columns:
            _region_stats.loc[idx, "mag_mean"] = float(np.mean(mags)) if mags else 0.0
        if "depth_mean" in feature_columns:
            _region_stats.loc[idx, "depth_mean"] = float(np.mean(deps)) if deps else 0.0


def _predict_region(features: dict) -> dict:
    """Scale features and predict cluster + risk for a region."""
    df = pd.DataFrame([features])
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Apply winsorization clipping (konsisten dengan preprocessing saat training)
    for col, bounds in clip_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(bounds["low"], bounds["high"])

    # Apply log transformation
    for col in log_transform_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].astype(float))

    # Apply RobustScaler (already trained)
    scaled = scaler.transform(df)

    # Weight features
    scaled_dict = dict(zip(feature_columns, scaled[0]))
    weighted = []
    for col in feature_columns:
        val = scaled_dict[col] * feature_weights.get(col, 1.0)
        weighted.append(val)

    # Predict cluster
    cluster = int(model.predict(np.array([weighted]))[0])

    # Fetch risk score and level directly from the cluster map 
    if cluster_risk_map and cluster in cluster_risk_map:
        cluster_info = cluster_risk_map[cluster]
        risk_score = cluster_info.get("risk_score", 0.0)
        risk_lvl = cluster_info.get("risk_level", "Unknown")
    else:
        # Fallback if map missing
        risk_score = 0.0
        risk_lvl = "Unknown"

    return {"cluster": cluster, "risk_score": round(risk_score, 4), "risk_level": risk_lvl}


def transform_bmkg_data(raw_data: dict):
    """Transform BMKG payload into predictions with dynamic region updates."""
    gempa_list = raw_data["Infogempa"]["gempa"]
    results = []

    for gempa in gempa_list:
        try:
            magnitude = _parse_float(gempa.get("Magnitude"), field_name="Magnitude")
            depth_km = _parse_float(gempa.get("Kedalaman"), field_name="Kedalaman")
        except Exception as exc:
            raise ValueError(f"Invalid gempa item: {exc}") from exc

        # Parse coordinates
        lat, lon = None, None
        if "Lintang" in gempa or "Bujur" in gempa:
            lat = _parse_lat(gempa.get("Lintang"))
            lon = _parse_lon(gempa.get("Bujur"))
        elif "Coordinates" in gempa:
            coords = gempa.get("Coordinates")
            if isinstance(coords, str):
                parts = coords.split(",")
                if len(parts) >= 2:
                    try:
                        lat, lon = float(parts[0].strip()), float(parts[1].strip())
                    except Exception:
                        pass

        # Map to kabupaten
        kab_id, kab_name = _map_to_kabupaten(lat, lon)

        # Update region stats dynamically
        _update_region_stats(kab_id, magnitude, depth_km)

        # Get region features and predict
        region_feat = _get_region_features(kab_id)
        if region_feat and model is not None and scaler is not None:
            pred = _predict_region(region_feat)
        else:
            pred = {"cluster": -1, "risk_score": 0.0, "risk_level": "Unknown"}

        results.append({
            "magnitude": magnitude,
            "depth_km": depth_km,
            "cluster": pred["cluster"],
            "risk_score": pred["risk_score"],
            "risk_level": pred["risk_level"],
            "gadm_id": kab_id,
            "gadm_name": kab_name,
        })

    return results


def predict_realtime_from_bmkg(raw_data: dict):
    """Return model predictions and metadata for a BMKG payload."""
    load_model()  # Dynamically reload latest model
    if model is None or scaler is None:
        raise RuntimeError("Realtime model is not loaded or unavailable")

    results = transform_bmkg_data(raw_data)
    return {
        "total_data": len(results),
        "model_version": model_version,
        "trained_at": trained_at,
        "model_hash": model_hash,
        "results": results,
    }


# Load model on import (best-effort)
try:
    _res = load_model()
    if isinstance(_res, dict) and _res.get("error"):
        logging.warning("Realtime model unavailable: %s", _res.get("error"))
except Exception:
    logging.exception("Error loading realtime model")
