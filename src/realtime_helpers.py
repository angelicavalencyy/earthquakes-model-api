"""Helper functions for realtime earthquake clustering training."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, pairwise_distances,
)
from sklearn_extra.cluster import KMedoids
from scipy.stats import zscore

RANDOM_STATE = 42
FEATURE_WEIGHTS = {
    "frekuensi_gempa": 0.10,
    "seismic_density": 0.15,
    "mag_p95": 0.15,
    "mag_std": 0.05,
    "b_value": 0.08,
    "depth_std": 0.05,
    "depth_skew": 0.07,
    "shallow_ratio": 0.10,
    "event_concentration": 0.10,
    "centroid_lat": 0.075,
    "centroid_lon": 0.075,
}

# ── Feature Weights V2 (Optimized: reduced spatial bias, boosted hazard) ─
# Alasan perubahan:
#   - centroid_lat/lon dikurangi dari 0.075 → 0.035 agar clustering
#     tidak bias ke lokasi geografis, tapi ke karakteristik seismik
#   - Fitur hazard utama (seismic_density, mag_p95, shallow_ratio, frekuensi_gempa)
#     dinaikkan untuk memperkuat separasi klaster berbasis risiko
#   - Total bobot = 1.000
FEATURE_WEIGHTS_V2 = {
    "frekuensi_gempa": 0.130,
    "seismic_density": 0.180,
    "mag_p95": 0.180,
    "mag_std": 0.050,
    "b_value": 0.080,
    "depth_std": 0.040,
    "depth_skew": 0.040,
    "shallow_ratio": 0.130,
    "event_concentration": 0.100,
    "centroid_lat": 0.035,
    "centroid_lon": 0.035,
}

# ── Feature Weights V3 (K-Medoids Optimized: minimal spatial, max hazard separation) ─
# Alasan perubahan:
#   - centroid_lat/lon diturunkan drastis ke 0.010 (dari 0.035) karena fitur
#     spasial menambah noise untuk clustering berbasis risiko dan mengurangi
#     separasi klaster. K-Medoids (medoid = data riil) lebih terpengaruh
#     noise dibanding K-Means (centroid = rata-rata yang meredam noise).
#   - Fitur hazard utama dinaikkan lebih agresif untuk memperkuat separasi:
#     seismic_density 0.200, mag_p95 0.200, shallow_ratio 0.150
#   - depth_std dan depth_skew dikurangi karena berkorelasi tinggi dengan
#     fitur lain dan menambah dimensi tanpa informasi baru
#   - Total bobot = 1.000
FEATURE_WEIGHTS_V3 = {
    "frekuensi_gempa": 0.140,
    "seismic_density": 0.200,
    "mag_p95": 0.200,
    "mag_std": 0.040,
    "b_value": 0.070,
    "depth_std": 0.025,
    "depth_skew": 0.025,
    "shallow_ratio": 0.150,
    "event_concentration": 0.120,
    "centroid_lat": 0.010,
    "centroid_lon": 0.010,
}

CLUSTER_FEATURE_COLUMNS = list(FEATURE_WEIGHTS.keys())

K_RANGE = range(3, 6)
SCHEMES = [
    ("KMeans", "euclidean"), ("KMeans", "manhattan"),
    ("KMedoids", "euclidean"), ("KMedoids", "manhattan"),
]

# ── Outlier Capping ──────────────────────────────────────────────────
def winsorize_features(df: pd.DataFrame, lower_pct: float = 1, upper_pct: float = 99):
    """Clip feature values at given percentiles to cap extreme outliers.
    
    Data gempa secara alami mengandung outlier ekstrem (misalnya satu kabupaten
    dengan frekuensi gempa 10x lebih tinggi dari rata-rata). Tanpa capping,
    outlier ini mendominasi pembentukan klaster dan menyebabkan imbalance
    (mayoritas wilayah masuk satu klaster).
    
    Winsorization di persentil 1-99 mempertahankan distribusi data sekaligus
    mencegah distorsi oleh nilai ekstrem.
    
    Args:
        df: DataFrame fitur yang akan di-clip
        lower_pct: Persentil batas bawah (default: 1)
        upper_pct: Persentil batas atas (default: 99)
    
    Returns:
        tuple: (DataFrame yang sudah di-clip, dict batas clip per kolom)
    """
    result = df.copy()
    clip_bounds = {}
    for col in result.columns:
        vals = result[col].dropna()
        if len(vals) == 0:
            continue
        low = float(np.percentile(vals, lower_pct))
        high = float(np.percentile(vals, upper_pct))
        result[col] = result[col].clip(low, high)
        clip_bounds[col] = {"low": low, "high": high}
    return result, clip_bounds

# ── Data Loading ─────────────────────────────────────────────────────
def load_bmkg() -> pd.DataFrame:
    df = pd.read_csv("./data/raw/bmkg_raw.csv")
    df = df.drop_duplicates()
    df = df.dropna(subset=["magnitude", "depth_km", "latitude", "longitude"])
    return df

def load_gadm() -> gpd.GeoDataFrame:
    with open("./data/raw/gadm41_IDN_2.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(raw["features"])
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    gdf = gdf.to_crs("EPSG:4326")
    gdf["luas_wilayah_km2"] = gdf.to_crs(epsg=3395).geometry.area / 1e6
    gdf = gdf.rename(columns={"GID_2": "id_kabupaten", "NAME_2": "nama_kabupaten"})
    return gdf[["id_kabupaten", "nama_kabupaten", "luas_wilayah_km2", "geometry"]]

def assign_events_to_kabupaten(bmkg: pd.DataFrame, gadm: gpd.GeoDataFrame) -> pd.DataFrame:
    b = bmkg.copy()
    b["geometry"] = gpd.points_from_xy(b["longitude"], b["latitude"])
    b_gdf = gpd.GeoDataFrame(b, geometry="geometry", crs="EPSG:4326")
    gt = gadm[["id_kabupaten", "geometry"]].copy()
    joined = gpd.sjoin(b_gdf, gt, how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    unmatched = joined["id_kabupaten"].isna()
    if unmatched.any():
        gm = gt.to_crs(epsg=3395).rename(columns={"id_kabupaten": "id_kab_near"})
        sea = joined.loc[unmatched].to_crs(epsg=3395)
        near = gpd.sjoin_nearest(sea, gm, how="left", distance_col="_d").drop(columns=["index_right"], errors="ignore")
        joined.loc[near.index, "id_kabupaten"] = near["id_kab_near"]
    return joined

def calc_b_value(magnitudes):
    if len(magnitudes) < 3:
        return 1.0
    m_mean = magnitudes.mean()
    m_min = magnitudes.min()
    if m_mean == m_min:
        return 1.0
    return 1.0 / (np.log(10) * (m_mean - m_min))

def aggregate_to_regions(bmkg: pd.DataFrame, gadm: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Mapping events to kabupaten...")
    events = assign_events_to_kabupaten(bmkg, gadm)
    print("Aggregating to region level...")
    events["lat_round"] = events["latitude"].round(2)
    events["lon_round"] = events["longitude"].round(2)
    hotspots = events.groupby(["id_kabupaten", "lat_round", "lon_round"]).size()
    max_hotspots = hotspots.groupby("id_kabupaten").max()
    total_events = events.groupby("id_kabupaten").size()
    event_concentration = (max_hotspots / total_events).fillna(0)

    agg = events.groupby("id_kabupaten").agg(
        frekuensi_gempa=("magnitude", "count"),
        mag_p95=("magnitude", lambda x: x.quantile(0.95) if len(x) > 0 else 0),
        mag_std=("magnitude", "std"),
        depth_std=("depth_km", "std"),
        depth_skew=("depth_km", "skew"),
        _shallow=("depth_km", lambda x: (x < 70).sum()),
        _total=("depth_km", "count"),
        b_value=("magnitude", calc_b_value),
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
    ).reset_index()
    
    agg["shallow_ratio"] = agg["_shallow"] / agg["_total"].replace(0, 1)
    agg = agg.drop(columns=["_shallow", "_total"])
    agg["event_concentration"] = agg["id_kabupaten"].map(event_concentration)
    
    result = gadm[["id_kabupaten", "nama_kabupaten", "luas_wilayah_km2"]].merge(agg, on="id_kabupaten", how="left")
    
    result["seismic_density"] = result["frekuensi_gempa"].fillna(0) / result["luas_wilayah_km2"].replace(0, 1)
    
    for col in CLUSTER_FEATURE_COLUMNS:
        if col == "b_value":
            result[col] = result[col].fillna(1.0)
        else:
            result[col] = result[col].fillna(0)
    return result

# ── Preprocessing ────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    raw = df[CLUSTER_FEATURE_COLUMNS].copy()
    # Step 1: Winsorize — cap outlier di persentil 1 dan 99
    clipped, clip_bounds = winsorize_features(raw)
    # Step 2: Log transform fitur skewed
    transformed = clipped.copy()
    transformed["frekuensi_gempa"] = np.log1p(transformed["frekuensi_gempa"])
    transformed["seismic_density"] = np.log1p(transformed["seismic_density"])
    # Step 3: QuantileTransformer — memetakan ke distribusi uniform [0,1]
    #
    # Alasan mengganti RobustScaler:
    #   RobustScaler hanya menggeser median ke 0 dan scale by IQR,
    #   tapi TIDAK menghilangkan skewness distribusi. Data gempa yang
    #   heavily skewed (kebanyakan wilayah rendah aktivitas) tetap skewed
    #   setelah RobustScaler → menghasilkan 1 cluster dominan (68% "Tinggi").
    #
    #   QuantileTransformer memetakan setiap fitur ke distribusi uniform [0,1]
    #   berdasarkan peringkat (rank). Ini berarti:
    #   - Setiap fitur tersebar MERATA di [0,1]
    #   - Tidak ada cluster yang mendominasi karena distribusi seimbang
    #   - K-Medoids lebih unggul karena medoid selection pada data uniform
    #     lebih representatif (median-based vs mean-based K-Means)
    #   - Manhattan distance pada rank-transformed data setara dengan
    #     Spearman rank correlation distance (Park & Jun, 2009)
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
    hazard_features = ["frekuensi_gempa", "seismic_density", "mag_p95", "shallow_ratio"]
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

# ── Clustering ───────────────────────────────────────────────────────
class _KMeansManhattan:
    def __init__(self, n_clusters=3, max_iter=300, random_state=42, n_init=10):
        self.n_clusters, self.max_iter = n_clusters, max_iter
        self.random_state, self.n_init = random_state, n_init
        self.cluster_centers_ = self.labels_ = self.inertia_ = None

    def _run(self, x, rng):
        n = x.shape[0]
        c = x[rng.choice(n, self.n_clusters, replace=False)].copy()
        for _ in range(self.max_iter):
            d = pairwise_distances(x, c, metric="manhattan")
            lab = d.argmin(axis=1)
            nc = np.array([np.median(x[lab==j], axis=0) if (lab==j).any() else x[rng.choice(n)] for j in range(self.n_clusters)])
            if np.allclose(nc, c): break
            c = nc
        return c, lab, float(pairwise_distances(x, c, metric="manhattan")[np.arange(n), lab].sum())

    def fit(self, x):
        x = x.values if hasattr(x, "values") else np.asarray(x, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = (None, None, float("inf"))
        for _ in range(self.n_init):
            c, l, i = self._run(x, rng)
            if i < best[2]: best = (c, l, i)
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, x):
        x = x.values if hasattr(x, "values") else np.asarray(x, dtype=float)
        return pairwise_distances(x, self.cluster_centers_, metric="manhattan").argmin(axis=1)

def compute_kmedoids_params(n_samples: int, k: int) -> dict:
    """Hitung hyperparameter K-Medoids berdasarkan karakteristik dataset.
    
    Daripada menggunakan nilai hardcode, parameter dihitung dari ukuran data:
    
    n_init (jumlah inisialisasi independen):
        Formula: ceil(log2(n_samples)), dibatasi [3, 10]
        Alasan: Ruang pencarian medoid optimal dari n titik bersifat kombinatorial
        C(n,k). Seiring n bertambah, dibutuhkan lebih banyak restart untuk
        menjelajahi ruang solusi. log2(n) memberikan scaling yang wajar karena
        probabilitas menemukan global optimum meningkat logaritmis dengan
        jumlah restart.
        
        Contoh: n=100 -> n_init=7, n=500 -> n_init=9, n=1000 -> n_init=10
    
    max_iter (iterasi swap maksimum per restart):
        Formula: max(500, n_samples), dibatasi 2000
        Alasan: Fase SWAP algoritma PAM memiliki kompleksitas O(k(n-k)) per
        iterasi. Secara empiris, konvergensi terjadi jauh sebelum n iterasi.
        Minimum 500 memastikan dataset kecil mendapat iterasi cukup,
        batas atas 2000 mencegah komputasi berlebihan pada dataset besar.
        
        Contoh: n=100 -> max_iter=500, n=800 -> max_iter=800, n=5000 -> max_iter=2000
    
    Referensi:
        Kaufman, L. & Rousseeuw, P.J. (1990). Finding Groups in Data.
        Park, H.S. & Jun, C.H. (2009). A simple and fast algorithm for K-medoids.
    """
    n_init = max(3, min(10, int(np.ceil(np.log2(max(n_samples, 2))))))
    max_iter = max(500, min(2000, n_samples))
    return {"n_init": n_init, "max_iter": max_iter}

def fit_model(features: pd.DataFrame, algo: str, metric: str, k: int,
              optimize_kmedoids: bool = False):
    """Fit clustering model.
    
    Args:
        features: Weighted feature DataFrame
        algo: 'KMeans' or 'KMedoids'
        metric: 'euclidean' or 'manhattan'
        k: Number of clusters
        optimize_kmedoids: If True, use enhanced multi-init with both
            'build' and 'k-medoids++' initialization for K-Medoids.
            BUILD initialization (Kaufman & Rousseeuw, 1990) sering
            menemukan medoid awal yang lebih baik karena mempertimbangkan
            seluruh dataset secara greedy, bukan sampling acak.
    """
    n_samples = features.shape[0]
    if algo == "KMeans":
        if metric == "euclidean":
            m = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        else:
            m = _KMeansManhattan(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    else:
        # Multi-init PAM: jalankan K-Medoids beberapa kali dengan seed berbeda,
        # pilih hasil dengan inertia (total distance) terkecil
        params = compute_kmedoids_params(n_samples, k)
        best_model = None
        best_inertia = float("inf")

        # Daftar metode inisialisasi:
        #   - k-medoids++: probabilistic, mirip k-means++ (default)
        #   - build: deterministic BUILD phase dari PAM (Kaufman & Rousseeuw)
        init_methods = ["k-medoids++", "build"] if optimize_kmedoids else ["k-medoids++"]
        n_init = params["n_init"] * 2 if optimize_kmedoids else params["n_init"]

        for init_method in init_methods:
            for i in range(n_init):
                candidate = KMedoids(
                    n_clusters=k, metric=metric, method="pam",
                    random_state=RANDOM_STATE + i, init=init_method,
                    max_iter=params["max_iter"]
                )
                candidate.fit(features)
                if candidate.inertia_ < best_inertia:
                    best_inertia = candidate.inertia_
                    best_model = candidate
        return best_model
    m.fit(features)
    return m

# ── Evaluation ───────────────────────────────────────────────────────
def compute_metrics(x: np.ndarray, labels: np.ndarray, centers: np.ndarray, distance_metric: str = "euclidean") -> dict:
    sse = 0.0
    for i, lab in enumerate(labels):
        if 0 <= lab < len(centers):
            if distance_metric == "manhattan":
                sse += float(np.sum(np.abs(x[i] - centers[lab])))
            else:
                sse += float(np.sum((x[i] - centers[lab]) ** 2))
    n_cl = len(set(labels))
    sil = float(silhouette_score(x, labels, metric=distance_metric)) if n_cl >= 2 else 0.0
    dbi = float(davies_bouldin_score(x, labels)) if n_cl >= 2 else float("inf")
    chi = float(calinski_harabasz_score(x, labels)) if n_cl >= 2 else 0.0
    return {"sse": sse, "silhouette": sil, "dbi": dbi, "chi": chi}

def generate_risk_table(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Generate risk classification table. Accepts optional custom weights."""
    if weights is None:
        weights = FEATURE_WEIGHTS
    cs = df.groupby("cluster_label")[CLUSTER_FEATURE_COLUMNS].mean()
    
    # Calculate risk score based strictly on hazard-increasing features
    # to avoid spatial bias (latitude/longitude) and ambiguous features.
    hazard_features = ["frekuensi_gempa", "seismic_density", "mag_p95", "shallow_ratio"]
    
    # Compute weighted sum for risk score
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

# ── K Selection ──────────────────────────────────────────────────────
def select_best_k(weighted_feat: pd.DataFrame, algo: str, metric: str,
                  k_range=range(3, 5)) -> tuple:
    """Pilih k optimal berdasarkan silhouette score tertinggi.
    
    Mengembalikan (best_k, DataFrame metrik untuk setiap k yang diuji).
    Default k_range = [3, 4] sesuai kebutuhan model realtime.
    """
    best_k = k_range.start
    best_sil = -1
    results = []
    
    for k in k_range:
        m = fit_model(weighted_feat, algo, metric, k)
        x = weighted_feat.values
        met = compute_metrics(x, m.labels_, m.cluster_centers_, distance_metric=metric)
        results.append({"k": k, **met})
        if met["silhouette"] > best_sil:
            best_sil = met["silhouette"]
            best_k = k
    
    return best_k, pd.DataFrame(results)

# ── MLflow Logging ───────────────────────────────────────────────────
def log_eda(raw_df: pd.DataFrame, scaled_feat: pd.DataFrame, weighted_feat: pd.DataFrame = None):
    feat = raw_df[CLUSTER_FEATURE_COLUMNS]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, CLUSTER_FEATURE_COLUMNS):
        sns.histplot(feat[col], kde=True, ax=ax, color="#2F6BFF")
        ax.set_title(f"Distribusi Fitur {col.replace('_', ' ').title()}")
    plt.tight_layout(); mlflow.log_figure(fig, "eda/1_distribusi_histogram.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(feat.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Heatmap Variabel")
    plt.tight_layout(); mlflow.log_figure(fig, "eda/2_matriks_korelasi.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        raw_df["centroid_lon"], raw_df["centroid_lat"],
        s=raw_df["frekuensi_gempa"],
        c=raw_df["mag_p95"],
        cmap="Reds", alpha=0.6, edgecolors="w", linewidth=0.5
    )
    plt.colorbar(scatter, label="Magnitude (p95)")
    ax.set_title("Pemetaan Spasial Wilayah (Ukuran: Frekuensi Gempa)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout(); mlflow.log_figure(fig, "eda/3_pemetaan_spasial.png"); plt.close(fig)

    if weighted_feat is not None:
        from sklearn.cluster import KMeans
        x_weighted = weighted_feat.values
        labels = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit_predict(x_weighted)
        
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pc = pca.fit_transform(x_weighted)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=labels, cmap="viridis")
        ax.set_title(f"PCA K=3")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax, label="Klaster")
        plt.tight_layout(); mlflow.log_figure(fig, "eda/4_pca_k3.png"); plt.close(fig)

    pairs = [(a, b) for i, a in enumerate(CLUSTER_FEATURE_COLUMNS) for b in CLUSTER_FEATURE_COLUMNS[i+1:]]
    ncols = min(5, len(pairs)); nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]
    for idx, (a, b) in enumerate(pairs):
        axes_flat[idx].scatter(feat[a], feat[b], s=10, alpha=0.5)
        axes_flat[idx].set_xlabel(a); axes_flat[idx].set_ylabel(b)
    for idx in range(len(pairs), len(axes_flat)): axes_flat[idx].set_visible(False)
    plt.tight_layout(); mlflow.log_figure(fig, "eda/5_scatter_plot_pasangan.png"); plt.close(fig)
    mlflow.log_text(feat.describe().T.to_csv(), "eda/6_ringkasan_fitur.csv")

def log_outliers(raw_df: pd.DataFrame, prefix="outlier"):
    feat = raw_df[CLUSTER_FEATURE_COLUMNS].fillna(0).to_numpy(dtype=float)
    z = np.abs(zscore(feat, axis=0, nan_policy="omit"))
    mask = (z > 3).any(axis=1)
    n_out, n_tot = int(mask.sum()), len(raw_df)
    pct = n_out / n_tot * 100 if n_tot else 0
    mlflow.log_metric("outlier_count", n_out)
    mlflow.log_metric("outlier_pct", round(pct, 4))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Normal", "Pencilan (Outlier)"], [n_tot - n_out, n_out], color=["#7AA6FF", "#FF6B6B"])
    ax.set_title("Deteksi Pencilan Ekstrem (Z > 3)"); ax.bar_label(bars, padding=3)
    plt.tight_layout(); mlflow.log_figure(fig, f"{prefix}/diagram_batang.png"); plt.close(fig)
    summary = pd.DataFrame([{"label": "Normal", "count": n_tot - n_out}, {"label": "Outlier", "count": n_out}])
    mlflow.log_text(summary.to_csv(index=False), f"{prefix}/ringkasan_pencilan.csv")
    return n_out, pct

def log_scaling(raw_feat: pd.DataFrame, scaled_feat: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=raw_feat, ax=axes[0]); axes[0].set_title("Distribusi Sebelum Standardisasi"); axes[0].tick_params(axis='x', rotation=45)
    sns.boxplot(data=scaled_feat, ax=axes[1]); axes[1].set_title("Distribusi Sesudah RobustScaler"); axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout(); mlflow.log_figure(fig, "scaling/perbandingan_skala.png"); plt.close(fig)

def log_cluster_results(model, x_weighted, prefix):
    labels = model.labels_
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#2F6BFF")
    ax.set_title(f"Distribusi Wilayah per Klaster")
    ax.set_xlabel("Klaster"); ax.set_ylabel("Jumlah Wilayah"); ax.bar_label(bars, padding=3)
    plt.tight_layout(); mlflow.log_figure(fig, f"{prefix}/distribusi_klaster.png"); plt.close(fig)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pc = pca.fit_transform(x_weighted)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=labels, cmap="viridis")
    ax.set_title(f"PCA K={len(counts)}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="Klaster")
    plt.tight_layout(); mlflow.log_figure(fig, f"{prefix}/sebaran_klaster_pca.png"); plt.close(fig)

def log_elbow(weighted_df: pd.DataFrame, algo: str, metric: str):
    rows = []
    for k in K_RANGE:
        m = fit_model(weighted_df, algo, metric, k)
        x = weighted_df.values
        met = compute_metrics(x, m.labels_, m.cluster_centers_, distance_metric=metric)
        rows.append({"k": k, "sse": met["sse"], "silhouette": met["silhouette"]})
    edf = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(edf["k"], edf["sse"], marker="o", color="#2F6BFF")
    ax1.set_title(f"Metode Elbow ({algo} - {metric.title()})"); ax1.set_xlabel("K"); ax1.set_ylabel("SSE"); ax1.grid(True, alpha=0.25)
    ax2.plot(edf["k"], edf["silhouette"], marker="o", color="#FF6B6B")
    ax2.set_title(f"Metode Silhouette ({algo} - {metric.title()})"); ax2.set_xlabel("K"); ax2.set_ylabel("Skor Silhouette"); ax2.grid(True, alpha=0.25)
    plt.tight_layout(); mlflow.log_figure(fig, f"elbow/{algo}_{metric}_grafik.png"); plt.close(fig)
    mlflow.log_text(edf.to_csv(index=False), f"elbow/{algo}_{metric}_metrik.csv")
    return edf

def log_outlier_handling(raw_df: pd.DataFrame, feature_columns: list,
                         log_transform_cols: list, prefix: str = "outlier_handling"):
    """Log proses penanganan outlier ke MLflow dengan visualisasi before/after.

    Menghasilkan grafik dan tabel perbandingan kondisi data sebelum dan
    sesudah penanganan outlier (winsorization + log transform).

    Args:
        raw_df: DataFrame dengan nilai fitur mentah
        feature_columns: Daftar kolom fitur yang dianalisis
        log_transform_cols: Kolom yang menerima transformasi log1p
        prefix: Path prefix untuk artifact MLflow

    Returns:
        tuple: (n_outlier_sebelum, n_outlier_sesudah, persen_reduksi)
    """
    raw_feat = raw_df[feature_columns].copy().fillna(0)
    n_tot = len(raw_feat)

    # ── SEBELUM: Analisis outlier pada data mentah ───────────────────
    feat_np = raw_feat.to_numpy(dtype=float)
    z_before = np.abs(zscore(feat_np, axis=0, nan_policy="omit"))
    mask_before = (z_before > 3).any(axis=1)
    n_before = int(mask_before.sum())
    skew_before = {col: float(raw_feat[col].skew()) for col in feature_columns}

    # ── SESUDAH: Terapkan winsorization + log transform ──────────────
    clipped, _ = winsorize_features(raw_feat)
    transformed = clipped.copy()
    for col in log_transform_cols:
        if col in transformed.columns:
            transformed[col] = np.log1p(transformed[col])

    feat_after_np = transformed.fillna(0).to_numpy(dtype=float)
    z_after = np.abs(zscore(feat_after_np, axis=0, nan_policy="omit"))
    mask_after = (z_after > 3).any(axis=1)
    n_after = int(mask_after.sum())
    skew_after = {col: float(transformed[col].skew()) for col in feature_columns}

    reduction = ((n_before - n_after) / max(n_before, 1)) * 100

    # ── 1. Diagram Batang: Jumlah Outlier Sebelum vs Sesudah ─────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Kiri: total outlier
    cats = ["Sebelum\nPenanganan", "Sesudah\nPenanganan"]
    vals = [n_before, n_after]
    colors = ["#FF6B6B", "#4ECDC4"]
    bars = axes[0].bar(cats, vals, color=colors, width=0.5,
                       edgecolor="white", linewidth=1.5)
    axes[0].bar_label(bars, padding=5, fontsize=12, fontweight="bold")
    axes[0].set_title("Jumlah Total Outlier (Z > 3)",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Jumlah Wilayah")
    axes[0].text(0.5, 0.92, f"Reduksi: {reduction:.1f}%",
                 transform=axes[0].transAxes, ha="center", fontsize=11,
                 style="italic",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Kanan: per-fitur
    before_pf = [int((z_before[:, i] > 3).sum()) for i in range(len(feature_columns))]
    after_pf = [int((z_after[:, i] > 3).sum()) for i in range(len(feature_columns))]
    x = np.arange(len(feature_columns))
    w = 0.35
    axes[1].bar(x - w / 2, before_pf, w, label="Sebelum", color="#FF6B6B", alpha=0.85)
    axes[1].bar(x + w / 2, after_pf, w, label="Sesudah", color="#4ECDC4", alpha=0.85)
    axes[1].set_title("Outlier Per Fitur", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Jumlah Outlier")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [c.replace("_", "\n") for c in feature_columns],
        fontsize=7, rotation=45, ha="right"
    )
    axes[1].legend(fontsize=9)

    plt.suptitle(
        "Perbandingan Outlier Sebelum vs Sesudah Penanganan\n"
        "(Winsorization 1‑99% + Log Transform)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    mlflow.log_figure(fig, f"{prefix}/1_perbandingan_jumlah_outlier.png")
    plt.close(fig)

    # ── 2. Boxplot Per Fitur: Sebelum vs Sesudah ─────────────────────
    n_feat = len(feature_columns)
    ncols = min(4, n_feat)
    nrows = (n_feat + ncols - 1) // ncols
    fig, axes_bp = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes_bp.flat if hasattr(axes_bp, "flat") else [axes_bp]

    for i, col in enumerate(feature_columns):
        if i < len(list(axes_flat)):
            ax = list(axes_bp.flat)[i] if hasattr(axes_bp, "flat") else axes_bp
            bp = ax.boxplot(
                [raw_feat[col].dropna(), transformed[col].dropna()],
                labels=["Sebelum", "Sesudah"], patch_artist=True,
                medianprops=dict(color="black", linewidth=2)
            )
            bp["boxes"][0].set_facecolor("#FF6B6B")
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor("#4ECDC4")
            bp["boxes"][1].set_alpha(0.7)
            ax.set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")

    all_axes = list(axes_bp.flat) if hasattr(axes_bp, "flat") else [axes_bp]
    for i in range(n_feat, len(all_axes)):
        all_axes[i].set_visible(False)

    plt.suptitle("Boxplot Per Fitur: Sebelum vs Sesudah Penanganan Outlier",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    mlflow.log_figure(fig, f"{prefix}/2_boxplot_per_fitur.png")
    plt.close(fig)

    # ── 3. Perbandingan Skewness ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_feat)
    w = 0.35
    ax.bar(x - w / 2, [skew_before[c] for c in feature_columns], w,
           label="Sebelum", color="#FF6B6B", alpha=0.85)
    ax.bar(x + w / 2, [skew_after[c] for c in feature_columns], w,
           label="Sesudah", color="#4ECDC4", alpha=0.85)
    ax.set_xlabel("Fitur", fontsize=11)
    ax.set_ylabel("Skewness", fontsize=11)
    ax.set_title("Perbandingan Skewness Sebelum vs Sesudah Penanganan Outlier",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in feature_columns],
        fontsize=8, rotation=45, ha="right"
    )
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="orange", linestyle=":", alpha=0.4)
    ax.axhline(y=-1, color="orange", linestyle=":", alpha=0.4)
    plt.tight_layout()
    mlflow.log_figure(fig, f"{prefix}/3_perbandingan_skewness.png")
    plt.close(fig)

    # ── 4. Tabel Detail Per Fitur ────────────────────────────────────
    detail_rows = []
    for i, col in enumerate(feature_columns):
        b_col = int((z_before[:, i] > 3).sum())
        a_col = int((z_after[:, i] > 3).sum())
        detail_rows.append({
            "fitur": col,
            "outlier_sebelum": b_col,
            "outlier_sesudah": a_col,
            "reduksi": b_col - a_col,
            "skewness_sebelum": round(skew_before[col], 4),
            "skewness_sesudah": round(skew_after[col], 4),
        })
    detail_df = pd.DataFrame(detail_rows)
    detail_df.loc[len(detail_df)] = {
        "fitur": "TOTAL", "outlier_sebelum": n_before,
        "outlier_sesudah": n_after, "reduksi": n_before - n_after,
        "skewness_sebelum": None, "skewness_sesudah": None,
    }
    mlflow.log_text(detail_df.to_csv(index=False),
                    f"{prefix}/4_detail_outlier_per_fitur.csv")

    # ── 5. Log Metrics ───────────────────────────────────────────────
    mlflow.log_metric("outlier_before_count", n_before)
    mlflow.log_metric("outlier_after_count", n_after)
    mlflow.log_metric("outlier_reduction_pct", round(reduction, 2))
    mlflow.log_param("outlier_method", "winsorize(1-99%) + log1p")
    mlflow.log_param("outlier_detection", "Z-score > 3")
    mlflow.log_param("outlier_log_transform_cols", ",".join(log_transform_cols))

    print(f"  Outlier handling: {n_before} -> {n_after} "
          f"(reduksi {reduction:.1f}%)")
    return n_before, n_after, reduction

def log_outlier_post(x_weighted, labels, centers, prefix, id_df, raw_feat, distance_metric="euclidean"):
    z = np.abs(zscore(x_weighted, axis=0, nan_policy="omit"))
    mask = (z > 3).any(axis=1)
    n_out, n_tot = int(mask.sum()), len(x_weighted)
    pct = n_out / n_tot * 100 if n_tot else 0
    per_cluster = {}
    for cl in sorted(set(labels)):
        cm = np.array(labels) == cl
        co = int(mask[cm].sum())
        ct = int(cm.sum())
        dists = []
        for i in np.where(cm)[0]:
            if distance_metric == "manhattan":
                dists.append(float(np.sum(np.abs(x_weighted[i] - centers[cl]))))
            else:
                dists.append(float(np.linalg.norm(x_weighted[i] - centers[cl])))
        per_cluster[cl] = {"outlier_count": co, "total": ct, "outlier_pct": round(co/ct*100, 2) if ct else 0, "mean_dist": round(float(np.mean(dists)), 4) if dists else 0}
    rows = [{"cluster": cl, **v} for cl, v in per_cluster.items()]
    rows.append({"cluster": "ALL", "outlier_count": n_out, "total": n_tot, "outlier_pct": round(pct, 2), "mean_dist": 0})
    mlflow.log_text(pd.DataFrame(rows).to_csv(index=False), f"{prefix}/ringkasan_pencilan_pasca_klaster.csv")
    
    # Simpan data lengkap pencilan
    outlier_df = pd.concat([id_df, raw_feat], axis=1)[mask].copy()
    outlier_df["cluster_label"] = np.array(labels)[mask]
    mlflow.log_text(outlier_df.to_csv(index=False), f"{prefix}/data_pencilan_lengkap.csv")
    
    mlflow.log_metric("post_outlier_count", n_out)
    mlflow.log_metric("post_outlier_pct", round(pct, 4))
