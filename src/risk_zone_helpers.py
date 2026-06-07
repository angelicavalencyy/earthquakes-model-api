import numpy as np
import pandas as pd
import mlflow
import json
import geopandas as gpd
from rapidfuzz import process, fuzz
from sklearn.preprocessing import RobustScaler, QuantileTransformer

# Import the exact aggregation logic used by realtime models for the 11 BMKG features
from realtime_helpers import (
    aggregate_to_regions, winsorize_features, compute_kmedoids_params,
    log_outlier_handling
)

CLUSTER_FEATURE_COLUMNS = [
    "frekuensi_gempa",
    "seismic_density",
    "mag_p95",
    "mag_std",
    "b_value",
    "depth_std",
    "depth_skew",
    "shallow_ratio",
    "event_concentration",
    "centroid_lat",
    "centroid_lon",
    "korban_total",
    "rumah_rusak_total",
    "fasum_rusak_total",
]

FEATURE_WEIGHTS = {
    "frekuensi_gempa": 0.070,
    "seismic_density": 0.140,
    "event_concentration": 0.080,
    "mag_p95": 0.160,
    "mag_std": 0.050,
    "b_value": 0.080,
    "depth_std": 0.030,
    "depth_skew": 0.020,
    "shallow_ratio": 0.120,
    "korban_total": 0.120,
    "rumah_rusak_total": 0.100,
    "fasum_rusak_total": 0.020,
    "centroid_lat": 0.005,
    "centroid_lon": 0.005,
}

# ── Feature Weights V2 (Optimized: reduced spatial bias, boosted hazard+impact) ─
# Alasan perubahan:
#   - centroid_lat/lon tetap rendah (0.005) karena di risk_zone sudah kecil
#   - Fitur hazard (seismic_density, mag_p95, shallow_ratio) dinaikkan
#   - Fitur dampak (korban_total, rumah_rusak_total) dinaikkan
#   - Total bobot = 1.000
FEATURE_WEIGHTS_V2 = {
    "frekuensi_gempa": 0.080,
    "seismic_density": 0.160,
    "event_concentration": 0.070,
    "mag_p95": 0.170,
    "mag_std": 0.040,
    "b_value": 0.070,
    "depth_std": 0.025,
    "depth_skew": 0.015,
    "shallow_ratio": 0.130,
    "korban_total": 0.130,
    "rumah_rusak_total": 0.080,
    "fasum_rusak_total": 0.020,
    "centroid_lat": 0.005,
    "centroid_lon": 0.005,
}

# ── Feature Weights V3 (K-Medoids Optimized: max hazard+impact separation) ─
# Alasan perubahan:
#   - Fitur hazard utama (seismic_density, mag_p95, shallow_ratio) dinaikkan
#     lebih agresif untuk memperkuat separasi klaster berbasis risiko
#   - Fitur dampak (korban_total) dinaikkan
#   - depth_std, depth_skew, fasum_rusak_total dikurangi (noise)
#   - centroid_lat/lon tetap minimal
#   - Total bobot = 1.000
FEATURE_WEIGHTS_V3 = {
    "frekuensi_gempa": 0.090,
    "seismic_density": 0.175,
    "event_concentration": 0.075,
    "mag_p95": 0.185,
    "mag_std": 0.035,
    "b_value": 0.065,
    "depth_std": 0.015,
    "depth_skew": 0.010,
    "shallow_ratio": 0.140,
    "korban_total": 0.120,
    "rumah_rusak_total": 0.070,
    "fasum_rusak_total": 0.015,
    "centroid_lat": 0.003,
    "centroid_lon": 0.002,
}

def load_data():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bmkg = pd.read_csv(root / "data" / "raw" / "bmkg_raw.csv")
    dibi = pd.read_csv(root / "data" / "raw" / "dibi_raw.csv")
    
    import json as js
    with open(root / "data" / "raw" / "gadm41_IDN_2.json", "r", encoding="utf-8") as f:
        raw = js.load(f)
    gadm = gpd.GeoDataFrame.from_features(raw["features"])
    if gadm.crs is None:
        gadm.set_crs("EPSG:4326", inplace=True)
    gadm = gadm.to_crs("EPSG:4326")
    gadm["luas_wilayah_km2"] = gadm.to_crs(epsg=3395).geometry.area / 1e6
    gadm = gadm.rename(columns={"GID_2": "id_kabupaten", "NAME_2": "nama_kabupaten"})
    
    return bmkg, dibi, gadm

def process_dibi(dibi_raw, gadm):
    missing_counts = dibi_raw[['korban_meninggal', 'korban_hilang', 'korban_terluka', 'rumah_rusak', 'rumah_terendam', 'fasum_rusak']].isnull().sum()
    total_rows = len(dibi_raw)
    for col, count in missing_counts.items():
        mlflow.log_metric(f"dibi_missing_{col}", int(count))
    
    summary_df = pd.DataFrame({"column": missing_counts.index, "missing_count": missing_counts.values})
    mlflow.log_text(summary_df.to_csv(index=False), "eda/dibi_missing_values_summary.csv")
    
    dibi_raw["korban_total"] = (
        dibi_raw["korban_meninggal"].fillna(0)
        + dibi_raw["korban_hilang"].fillna(0)
        + dibi_raw["korban_terluka"].fillna(0)
    )
    dibi_raw["rumah_rusak_total"] = dibi_raw["rumah_rusak"].fillna(0)
    dibi_raw["fasum_rusak_total"] = dibi_raw["fasum_rusak"].fillna(0)
    
    gadm_names_original = gadm["nama_kabupaten"].unique()
    gadm_names_processed = [name.lower().replace("kabupaten", "").replace("kota", "").strip() for name in gadm_names_original]
    gadm_name_to_processed = dict(zip(gadm_names_processed, gadm_names_original))

    dibi_names = dibi_raw["kabupaten"].dropna().unique()
    mapping = {}
    for original_dibi_name in dibi_names:
        processed_dibi_name = original_dibi_name.lower().replace("kota", "").replace("kabupaten", "").strip()
        match, score, _ = process.extractOne(processed_dibi_name, gadm_names_processed, scorer=fuzz.WRatio)
        if score >= 75:
            mapping[original_dibi_name] = gadm_name_to_processed[match]
        else:
            mapping[original_dibi_name] = None
            
    mapping_df = pd.DataFrame(mapping.items(), columns=["kabupaten", "nama_kabupaten"])
    
    if "id_kabupaten" in dibi_raw.columns:
        dibi_raw = dibi_raw.drop(columns=["id_kabupaten"])
        
    dibi = dibi_raw.merge(mapping_df, on="kabupaten", how="left")
    dibi = dibi.merge(gadm[["id_kabupaten", "nama_kabupaten"]], on="nama_kabupaten", how="left")
    
    dibi_agg = (
        dibi.groupby("id_kabupaten", dropna=True)
        .agg(
            korban_total=("korban_total", "sum"),
            rumah_rusak_total=("rumah_rusak_total", "sum"),
            fasum_rusak_total=("fasum_rusak_total", "sum"),
        )
        .reset_index()
    )
    return dibi_agg

def prepare_data():
    bmkg_raw, dibi_raw, gadm_raw = load_data()
    bmkg_raw = bmkg_raw.drop_duplicates().dropna(subset=['latitude', 'longitude'])
    
    # Gunakan fungsi agregasi yang sama dengan model realtime agar 11 fiturnya identik
    bmkg_agg = aggregate_to_regions(bmkg_raw, gadm_raw)
    dibi_agg = process_dibi(dibi_raw, gadm_raw)
    
    df = bmkg_agg.merge(dibi_agg, on="id_kabupaten", how="left")
    
    cols_zero = ["korban_total", "rumah_rusak_total", "fasum_rusak_total"]
    for col in cols_zero:
        df[col] = df[col].fillna(0)
        
    return df

def preprocess(df: pd.DataFrame):
    raw = df[CLUSTER_FEATURE_COLUMNS].copy()
    # Step 1: Winsorize — cap outlier di persentil 1 dan 99
    clipped, clip_bounds = winsorize_features(raw)
    # Step 2: Log transform fitur skewed
    transformed = clipped.copy()
    transformed["frekuensi_gempa"] = np.log1p(transformed["frekuensi_gempa"])
    transformed["seismic_density"] = np.log1p(transformed["seismic_density"])
    transformed["korban_total"] = np.log1p(transformed["korban_total"])
    transformed["rumah_rusak_total"] = np.log1p(transformed["rumah_rusak_total"])
    # Step 3: QuantileTransformer — distribusi uniform [0,1]
    scaler = QuantileTransformer(
        output_distribution='uniform',
        random_state=42,
        n_quantiles=min(len(transformed), 1000)
    )
    scaled = pd.DataFrame(scaler.fit_transform(transformed), columns=CLUSTER_FEATURE_COLUMNS)
    return raw, scaled, scaler, clip_bounds

def weight_features(scaled_df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Apply feature weighting. Accepts optional custom weights dict."""
    if weights is None:
        weights = FEATURE_WEIGHTS
    w = scaled_df.copy()
    for col in CLUSTER_FEATURE_COLUMNS:
        w[col] *= weights.get(col, 1.0)
    return w

def sort_model_clusters(model, df_scaled: pd.DataFrame, weights: dict = None):
    """Sort cluster labels by risk score (ascending). Accepts optional weights."""
    if weights is None:
        weights = FEATURE_WEIGHTS
    df = df_scaled.copy()
    df["cluster_label"] = model.labels_
    cs = df.groupby("cluster_label")[CLUSTER_FEATURE_COLUMNS].mean()
    hazard_features = ["frekuensi_gempa", "seismic_density", "mag_p95", "shallow_ratio", "korban_total", "rumah_rusak_total"]
    risk_scores = pd.Series(0.0, index=cs.index)
    for feat in hazard_features:
        if feat in cs.columns:
            weight = weights.get(feat, 0.1)
            risk_scores += cs[feat] * weight
    cs["risk_score"] = risk_scores
    cs = cs.sort_values("risk_score")
    
    mapping = {old: new for new, old in enumerate(cs.index)}
    new_labels = np.array([mapping[l] for l in model.labels_])
    
    new_centers = np.zeros_like(model.cluster_centers_)
    for old, new in mapping.items():
        if old < len(model.cluster_centers_):
            new_centers[new] = model.cluster_centers_[old]
            
    model.labels_ = new_labels
    model.cluster_centers_ = new_centers
    return model

def generate_risk_table(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Generate risk classification table. Accepts optional custom weights."""
    if weights is None:
        weights = FEATURE_WEIGHTS
    cs = df.groupby("cluster_label")[CLUSTER_FEATURE_COLUMNS].mean()
    
    hazard_features = ["frekuensi_gempa", "seismic_density", "mag_p95", "shallow_ratio", "korban_total", "rumah_rusak_total"]
    
    risk_scores = pd.Series(0.0, index=cs.index)
    for feat in hazard_features:
        if feat in cs.columns:
            weight = weights.get(feat, 0.1)
            risk_scores += cs[feat] * weight
            
    cs["risk_score"] = risk_scores
    cs = cs.sort_values("risk_score")
    
    n = len(cs)
    if n == 2:
        levels = ["Rendah", "Tinggi"]
    elif n == 3:
        levels = ["Rendah", "Sedang", "Tinggi"]
    elif n == 4:
        levels = ["Rendah", "Sedang", "Tinggi", "Ekstrem"]
    else:
        levels = ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Ekstrem"][:n]
        
    cs["risk_level"] = levels
    info = cs[["risk_score", "risk_level"]].reset_index().rename(columns={"index": "cluster_label"})
    counts = df["cluster_label"].value_counts().reset_index()
    counts.columns = ["cluster_label", "count"]
    return info.merge(counts, on="cluster_label", how="left")
