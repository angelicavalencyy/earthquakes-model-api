import pickle

# Global model cache
model = None
scaler = None
feature_columns = None
cluster_risk_map = None


def load_model():
    """Load the region risk model and return the raw model data."""

    with open("app/ml/model_v3.pkl", "rb") as f:
        data = pickle.load(f)

    return data


# Load on import for service availability and populate module cache
_loaded = load_model()
model = _loaded.get("model")
scaler = _loaded.get("scaler")
feature_columns = _loaded.get("feature_columns")
cluster_risk_map = _loaded.get("cluster_risk_map", {})
    