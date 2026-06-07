import logging
import pickle
from pathlib import Path
import traceback

# Global model cache (for backward compatibility if anything needs it)
model = None
scaler = None
feature_columns = None

# New global cache for static serving
model_version = None
feature_weights = None
cluster_risk_map = None
region_features = None
trained_at = None
model_hash = None
metrics = None

def load_static_model(pkl_path: str | Path = None) -> dict:
    global model_version, feature_columns, feature_weights, cluster_risk_map, region_features, trained_at, model_hash, metrics

    # Cari model pkl: prioritaskan yang diberikan, lalu best, k4, legacy k3
    if pkl_path is not None:
        candidates = [Path(pkl_path)]
    else:
        candidates = [
            Path("app/ml/kmed/static/static_best_k4.pkl"),
            Path("app/ml/kmed/static/static_KMedoids_euclidean_k4.pkl"),
            Path("app/ml/kmed/static/static_KMedoids_manhattan_k4.pkl"),
        ]

    p = None
    for c in candidates:
        if c.exists():
            p = c
            break
    if p is None:
        msg = f"Static model file not found. Searched: {[str(c) for c in candidates]}"
        logging.warning(msg)
        return {"error": msg}

    try:
        with p.open("rb") as f:
            data = pickle.load(f)
            
        model_version = data.get("model_version")
        feature_columns = data.get("feature_columns")
        feature_weights = data.get("feature_weights")
        cluster_risk_map = data.get("cluster_risk_map", {})
        region_features = data.get("region_features", [])
        trained_at = data.get("trained_at")
        model_hash = data.get("model_hash")
        metrics = data.get("metrics")
        
        logging.info("Successfully loaded static model version %s", model_version)
        return data
    except Exception as exc:
        tb = traceback.format_exc()
        logging.error("Failed to load static model pickle %s: %s", p, exc)
        return {"error": str(exc), "traceback": tb}

# Attempt to load on import
try:
    load_static_model()
except Exception:
    logging.exception("Unexpected error during static model import")

def get_static_risk_data():
    """Return the precomputed static region risk map directly from the loaded model."""
    load_static_model()  # Reload in case it was updated
    if region_features is None:
        raise RuntimeError("Static model is not loaded or unavailable")
        
    return {
        "model_version": model_version,
        "trained_at": trained_at,
        "model_hash": model_hash,
        "metrics": metrics,
        "total_regions": len(region_features),
        "feature_weights": feature_weights,
        "cluster_risk_map": cluster_risk_map,
        "data": region_features
    }