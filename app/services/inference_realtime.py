

# pylint: disable=global-statement,invalid-name
import pickle
import pandas as pd


CLUSTER_FEATURE_COLUMNS = [
    "frekuensi_gempa",
    "depth_mean",
    "mag_max",
    "mag_mean",
]

# Global model cache (populated by load_model)
model = None
scaler = None
feature_columns = None
cluster_risk_map = None
model_version = None


def load_model():
    """Load the pre-trained realtime model into module-level globals."""
    global model, scaler, feature_columns, cluster_risk_map, model_version

    with open("app/ml/realtime_model_k3.pkl", "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    scaler = data["scaler"]
    feature_columns = data["feature_columns"]
    cluster_risk_map = data.get("cluster_risk_map", {})
    model_version = data.get("model_version", "unknown")


def transform_single_gempa(gempa: dict):
    """Convert a single BMKG gempa entry to model feature dict."""
    magnitude = float(gempa["Magnitude"])
    depth = float(gempa["Kedalaman"].replace(" km", ""))

    return {
        "frekuensi_gempa": 1.0,
        "depth_mean": depth,
        "mag_max": magnitude,
        "mag_mean": magnitude,
    }


def transform_bmkg_data(raw_data: dict):
    """Transform BMKG payload into a list of model predictions.

    Uses the module-level `scaler` and `model` objects; ensure `load_model()`
    was called before invoking this function.
    """
    gempa_list = raw_data["Infogempa"]["gempa"]

    results = []

    for gempa in gempa_list:
        clean_data = transform_single_gempa(gempa)

        df = pd.DataFrame([clean_data])
        df = df.reindex(columns=feature_columns, fill_value=0)

        df_scaled = scaler.transform(df)
        cluster = model.predict(df_scaled)[0]

        risk_info = cluster_risk_map.get(cluster, {})

        risk_score = risk_info.get("risk_score")

        results.append({
            "frekuensi_gempa": clean_data["frekuensi_gempa"],
            "depth_mean": clean_data["depth_mean"],
            "mag_max": clean_data["mag_max"],
            "mag_mean": clean_data["mag_mean"],
            "cluster": int(cluster),
            "risk_score": round(risk_score, 2) if risk_score is not None else None,
            "risk_level": risk_info.get("risk_level"),
        })

    return results


def predict_realtime_from_bmkg(raw_data: dict):
    """Return model predictions and metadata for a BMKG payload."""
    results = transform_bmkg_data(raw_data)

    return {
        "total_data": len(results),
        "model_version": model_version,
        "results": results,
    }


# Load model into cache on import (intended behavior for service)
load_model()
