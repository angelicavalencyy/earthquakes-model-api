import os
from pathlib import Path

import mlflow
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

from rapidfuzz import process, fuzz
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn_extra.cluster import KMedoids

from sklearn.metrics import  silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score


CLUSTER_FEATURE_COLUMNS = [
    "frekuensi_gempa",
    "depth_mean",
    "mag_max",
    "mag_mean",
    "korban_total",
    "rumah_rusak_total",
    "fasum_rusak_total",
]
def load_data():
    bmkg_raw = pd.read_csv("./data/raw/bmkg_raw.csv")
    dibi_raw = pd.read_csv("./data/raw/dibi_raw.csv")
    gadm_raw = gpd.read_file("./data/raw/gadm41_IDN_2.json")

    return bmkg_raw, dibi_raw, gadm_raw

def prepare_gadm(gadm):
    gadm = gadm.to_crs("EPSG:4326")

    gadm["luas_wilayah_km2"] = gadm.to_crs(epsg=3395).geometry.area / 10**6

    gadm = gadm.rename(columns={
        "GID_2": "id_kabupaten",
        "NAME_2": "nama_kabupaten"
    })

    return gadm[["id_kabupaten", "nama_kabupaten", "luas_wilayah_km2", "geometry"]]

def process_bmkg(bmkg, gadm):
    # Wilayah dihitung per kabupaten.
    bmkg["geometry"] = bmkg.apply(
        lambda r: Point(r["longitude"], r["latitude"]), axis=1
    )

    bmkg_gdf = gpd.GeoDataFrame(bmkg, geometry="geometry", crs="EPSG:4326")

    bmkg_join = gpd.sjoin(
        bmkg_gdf,
        gadm[["id_kabupaten", "geometry"]],
        how="left",
        predicate="within"
    )

    bmkg_agg = (
        bmkg_join.groupby("id_kabupaten")
        .agg(
            frekuensi_gempa=("magnitude", "count"),
            mag_max=("magnitude", "max"),
            mag_mean=("magnitude", "mean"),
            depth_mean=("depth_km", "mean"),
        )
        .reset_index()
    )

    return bmkg_agg

def map_dibi_to_gadm(dibi_raw, gadm, threshold=75):
    gadm_names_original = gadm["nama_kabupaten"].unique()
    gadm_names_processed = [
        name.lower().replace("kabupaten", "").replace("kota", "").strip()
        for name in gadm_names_original
    ]
    gadm_name_to_processed = dict(zip(gadm_names_processed, gadm_names_original))

    dibi_names = dibi_raw["kabupaten"].dropna().unique()
    mapping = {}

    for original_dibi_name in dibi_names:
        processed_dibi_name = original_dibi_name.lower().replace("kota", "").replace("kabupaten", "").strip()

        match, score, _ = process.extractOne(processed_dibi_name, gadm_names_processed, scorer=fuzz.WRatio)

        if score >= threshold:
            mapping[original_dibi_name] = gadm_name_to_processed[match]
        else:
            mapping[original_dibi_name] = None

    return pd.DataFrame(mapping.items(), columns=["kabupaten", "nama_kabupaten"])

def process_dibi(dibi_raw, gadm):
    dibi_raw["korban_total"] = (
        dibi_raw["korban_meninggal"].fillna(0)
        + dibi_raw["korban_hilang"].fillna(0)
        + dibi_raw["korban_terluka"].fillna(0)
    )

    mapping_df = map_dibi_to_gadm(dibi_raw, gadm)

    dibi = dibi_raw.merge(mapping_df, on="kabupaten", how="left")
    dibi = dibi.merge(
        gadm[["id_kabupaten", "nama_kabupaten"]],
        on="nama_kabupaten",
        how="left",
        suffixes=('_original', '_gadm')
    )

    if 'id_kabupaten_gadm' in dibi.columns:
        dibi = dibi.rename(columns={'id_kabupaten_gadm': 'id_kabupaten'})
    elif 'id_kabupaten' not in dibi.columns:
        dibi['id_kabupaten'] = dibi['id_kabupaten_original'] if 'id_kabupaten_original' in dibi.columns else pd.NA

    if dibi.empty or dibi['id_kabupaten'].isnull().all():
        return pd.DataFrame(columns=['id_kabupaten', 'korban_total', 'rumah_rusak_total', 'fasum_rusak_total'])

    dibi_agg = (
        dibi.groupby("id_kabupaten", dropna=True)
        .agg(
            korban_total=("korban_total", "sum"),
            rumah_rusak_total=("rumah_rusak", "sum"),
            fasum_rusak_total=("fasum_rusak", "sum"),
        )
        .reset_index()
    )

    # print(dibi_agg.head())
    return dibi_agg

def merge_all(gadm, bmkg_agg, dibi_agg):
    df = (
        gadm.merge(bmkg_agg, on="id_kabupaten", how="left")
            .merge(dibi_agg, on="id_kabupaten", how="left")
    )

    cols_zero = ["frekuensi_gempa", "mag_max", "mag_mean", "korban_total", "rumah_rusak_total", "fasum_rusak_total"]
    df[cols_zero] = df[cols_zero].fillna(0)

    if not df["depth_mean"].isnull().all():
        max_depth = df["depth_mean"].max()
        df["depth_mean"] = df["depth_mean"].fillna(max_depth)
    else:
        df["depth_mean"] = df["depth_mean"].fillna(0)

    return df.drop(columns="geometry")


def log_feature_artifacts(raw_features_df, feature_columns, artifact_prefix):
    feature_df = raw_features_df[feature_columns].copy()
    corr_df = feature_df.corr(numeric_only=True)
    summary_df = feature_df.describe().T.reset_index().rename(columns={"index": "feature"})

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
    )
    ax.set_title(f"Correlation Heatmap - {artifact_prefix}")
    plt.tight_layout()
    mlflow.log_text(summary_df.to_csv(index=False), f"{artifact_prefix}_feature_summary.csv")
    mlflow.log_text(corr_df.to_csv(), f"{artifact_prefix}_feature_correlation.csv")
    mlflow.log_figure(fig, f"{artifact_prefix}_feature_correlation_heatmap.png")
    plt.close(fig)

    return None


def log_cluster_distribution_artifact(cluster_counts, artifact_prefix):
    cluster_series = cluster_counts.sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cluster_series.index.astype(str), cluster_series.values, color="#2F6BFF")
    ax.set_title(f"Distribusi Cluster - {artifact_prefix}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah Data")
    ax.bar_label(bars, padding=3)
    plt.tight_layout()
    mlflow.log_figure(fig, f"{artifact_prefix}_cluster_distribution_bar.png")
    plt.close(fig)

    return None


def log_preprocessing_artifact(artifact_prefix, steps):
    artifact_text = "\n".join(f"{step_name}: {step_detail}" for step_name, step_detail in steps)
    mlflow.log_text(artifact_text, f"{artifact_prefix}_preprocessing_steps.txt")
    return None


def log_model_clustering_artifact(artifact_prefix, k_val, feature_columns):
    artifact_text = "\n".join([
        "Model Clustering Summary",
        "Algorithm: K-Medoids",
        f"Number of clusters (k): {k_val}",
        f"Input features: {', '.join(feature_columns)}",
        "Training data: historical earthquake data",
        "Testing data: realtime earthquake data",
        "Output: cluster label, risk score, and risk level per kabupaten",
    ])
    mlflow.log_text(artifact_text, f"{artifact_prefix}_model_clustering_summary.txt")
    return None


def impute_dibi_features(df):

    df["korban_total"] = df["korban_total"].fillna(df["korban_total"].median())
    df["rumah_rusak_total"] = df["rumah_rusak_total"].fillna(df["rumah_rusak_total"].median())
    df["fasum_rusak_total"] = df["fasum_rusak_total"].fillna(df["fasum_rusak_total"].median())

    return df

def cleaning():
    # Ambil data mentah dan buang baris yang tidak valid.
    bmkg_raw_df, dibi_raw_df, gadm_raw_df = load_data()

    bmkg_raw_df = bmkg_raw_df.drop_duplicates()
    bmkg_raw_df = bmkg_raw_df.dropna(subset=['latitude', 'longitude'])

    gadm_processed_df = prepare_gadm(gadm_raw_df)

    # Model wilayah tetap memakai agregasi kabupaten.
    bmkg_aggregated_df = process_bmkg(bmkg_raw_df, gadm_processed_df)
    dibi_aggregated_df = process_dibi(dibi_raw_df, gadm_processed_df)
    
    raw_features_df = merge_all(gadm_processed_df, bmkg_aggregated_df, dibi_aggregated_df)
    raw_features_df = impute_dibi_features(raw_features_df)

    raw_features_df = raw_features_df[["id_kabupaten", "nama_kabupaten", *CLUSTER_FEATURE_COLUMNS]]

    # raw_features_df.to_csv("/content/drive/MyDrive/dataset_PA/fitur_risiko_ex_1.csv", index=False)
    return raw_features_df


def normalize_data(raw_features_df):
    scaler = MinMaxScaler()

    # scale only the features used for clustering
    numerical_features_to_scale_df = raw_features_df[CLUSTER_FEATURE_COLUMNS].copy()

    # scaled_numerical_features_array: NumPy array containing the scaled numerical features.
    scaled_numerical_features_array = scaler.fit_transform(numerical_features_to_scale_df)

    # print("\nData setelah Normalisasi Min-Max:")

    # scaled_numerical_features_df: DataFrame containing the scaled numerical features with original column names.
    scaled_numerical_features_df = pd.DataFrame(scaled_numerical_features_array, columns=numerical_features_to_scale_df.columns)

    # print(scaled_numerical_features_df.head())

    return scaled_numerical_features_df, scaler



def run_clustering(final_processed_df, k=3):
    clustering_input_df = final_processed_df[CLUSTER_FEATURE_COLUMNS].copy()

    model = KMedoids(n_clusters=k, random_state=42)
    model.fit(clustering_input_df)

    labels = model.labels_

    final_processed_df["cluster_label"] = labels

    cluster_characteristics_df = (
        final_processed_df
        .groupby("cluster_label")
        .mean(numeric_only=True)
    )

    return final_processed_df, model, cluster_characteristics_df
def run_pca(final_processed_df, clustering_input_df=None, n_components=2):
    """
    Menjalankan PCA dan menambahkan PC1, PC2 ke final_processed_df.

    Args:
        final_processed_df (pd.DataFrame)
        clustering_input_df (pd.DataFrame, optional): jika None akan auto ambil fitur numerik
        n_components (int): jumlah komponen PCA

    Return:
        final_processed_df (updated)
        pca_result_df (identifiers + PC only)
        pca_model
    """

    from sklearn.decomposition import PCA

    # 🔹 1. Pastikan hanya fitur numerik yang dipakai
    if clustering_input_df is None:
        clustering_input_df = final_processed_df[CLUSTER_FEATURE_COLUMNS].copy()

    # 🔹 2. Handle missing values (biar PCA aman)
    clustering_input_df = clustering_input_df.fillna(0)

    # 🔹 3. PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca_components_array = pca.fit_transform(clustering_input_df)

    # 🔹 4. Buat dataframe PCA
    pca_columns = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_components_array, columns=pca_columns)

    # 🔹 5. Reset index biar aman saat concat
    final_processed_df = final_processed_df.reset_index(drop=True)
    pca_df = pca_df.reset_index(drop=True)

    # 🔹 6. Gabungkan ke dataframe utama
    for col in pca_columns:
        final_processed_df[col] = pca_df[col]

    # 🔹 7. Data khusus PCA (untuk visualisasi)
    pca_result_df = pd.concat([
        final_processed_df[["id_kabupaten", "nama_kabupaten"]],
        pca_df
    ], axis=1)

    # 🔹 8. (Tambahan penting) Variance PCA
    explained_variance = pca.explained_variance_ratio_

    print(f"PCA Variance: {explained_variance}")
    print(f"Total Variance Explained: {explained_variance.sum():.4f}")

    return final_processed_df, pca_result_df, pca
def generate_risk_table(final_processed_df):
    """
    Menghitung:
    - cluster_counts
    - risk_score
    - risk_level
    - cluster_info_df
    - kabupaten_cluster_table
    - final_with_risk_df

    Return:
        final_processed_df (updated with risk)
        cluster_info_df
        kabupaten_cluster_table
        final_with_risk_df
    """

    # Count items per cluster
    cluster_counts = final_processed_df.groupby("cluster_label").size().rename("count")

    # Mean per cluster for the selected risk features only
    cluster_summary = final_processed_df.groupby("cluster_label")[CLUSTER_FEATURE_COLUMNS].mean()

    # INVERSI FITUR KEDALAMAN
    # Kedalaman kecil berarti gempa lebih berbahaya, jadi kontribusinya harus berlawanan arah.
    risk_calc_df = cluster_summary.copy()
    if "depth_mean" in risk_calc_df.columns:
        risk_calc_df["depth_mean"] = 1 - risk_calc_df["depth_mean"]

    # DIBI impact features tetap masuk sebagai komponen risiko karena mewakili dampak nyata.
    for feature_name in ["korban_total", "rumah_rusak_total", "fasum_rusak_total"]:
        if feature_name in risk_calc_df.columns:
            risk_calc_df[feature_name] = risk_calc_df[feature_name]

    # Skor risiko dihitung dari rerata fitur risiko yang sudah diarahkan dengan benar.
    cluster_summary["risk_score"] = risk_calc_df.mean(axis=1)

    # Sort by risk
    cluster_summary_no_pca = cluster_summary.sort_values("risk_score")

    # Assign risk levels
    n_clusters = len(cluster_summary_no_pca)

    if n_clusters >= 3:
        cluster_summary_no_pca["risk_level"] = pd.qcut(
            cluster_summary_no_pca["risk_score"],
            q=3,
            labels=["Rendah", "Sedang", "Tinggi"]
        )
    elif n_clusters == 2:
        cluster_summary_no_pca["risk_level"] = pd.qcut(
            cluster_summary_no_pca["risk_score"],
            q=2,
            labels=["Rendah", "Tinggi"]
        )
    else:
        cluster_summary_no_pca["risk_level"] = ["Rendah"] * n_clusters

    # Build cluster info table
    cluster_info_df = (
        cluster_summary_no_pca[["risk_score", "risk_level"]]
        .reset_index()
        .rename(columns={"index": "cluster_label"})
    )

    cluster_info_df = cluster_info_df.merge(
        cluster_counts.reset_index(),
        on="cluster_label",
        how="left"
    )

    # Kabupaten cluster table
    kabupaten_cluster_table = (
        final_processed_df[["id_kabupaten", "nama_kabupaten", "cluster_label"]]
        .merge(cluster_info_df, on="cluster_label", how="left")
    )

    kabupaten_cluster_table = kabupaten_cluster_table[
        ["id_kabupaten", "nama_kabupaten", "cluster_label",
         "risk_score", "risk_level", "count"]
    ]

    # Merge risk info into full dataframe
    final_processed_df = final_processed_df.merge(
        cluster_info_df[["cluster_label", "risk_score", "risk_level"]],
        on="cluster_label",
        how="left"
    )

    # Compact final view
    final_with_risk_df = final_processed_df[
        ["id_kabupaten", "nama_kabupaten",
         "cluster_label", "risk_score", "risk_level"]
    ].copy()

    return (
        final_processed_df,
        cluster_info_df,
        kabupaten_cluster_table,
        final_with_risk_df
    )

def save_model(
    cluster_info_df,
    train_df,
    model_version: str,
):
    # Simpan dataset hasil clustering yang memang dipakai pada model ini.
    final_df = train_df.merge(
        cluster_info_df[["cluster_label", "risk_score", "risk_level"]],
        on="cluster_label",
        how="left"
    )

    mlflow.log_dict({
        "model_version": model_version,
        "feature_columns": list(train_df.columns),
        "cluster_risk_map": cluster_info_df[
            ["cluster_label", "risk_score", "risk_level"]
        ].to_dict(orient="records"),
        "dataset_rows": len(final_df),
    }, f"{model_version}_model_metadata.json")

    print(f"Model {model_version} logged to MLflow")


def build_human_readable_cluster_df(raw_features_df, clustered_df):
    """Return cluster output with raw feature values for reporting/export."""
    export_columns = [
        "id_kabupaten",
        "nama_kabupaten",
        *CLUSTER_FEATURE_COLUMNS,
    ]
    cluster_columns = [
        "id_kabupaten",
        "nama_kabupaten",
        "cluster_label",
        "risk_score",
        "risk_level",
        "PC1",
        "PC2",
    ]

    return raw_features_df[export_columns].merge(
        clustered_df[cluster_columns],
        on=["id_kabupaten", "nama_kabupaten"],
        how="left",
    )


def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)


def get_mlflow_tracking_uri() -> str:
    tracking_dir = Path(__file__).resolve().parents[1] / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    return tracking_dir.resolve().as_uri()

def main():
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment("Earthquake_Clustering_KMedoids_v8")
    
    # 1. Load & Prepare Data
    _, _, gadm_raw = load_data()
    gadm_processed = prepare_gadm(gadm_raw) 

    # 2. Cleaning Fitur
    raw_features_df = cleaning()

    # 3. Simpan raw_features & Polygon
    mlflow.log_text(raw_features_df.to_csv(index=False), "raw_features_v2.csv")
    log_feature_artifacts(
        raw_features_df=raw_features_df,
        feature_columns=CLUSTER_FEATURE_COLUMNS,
        artifact_prefix="kmed",
    )

    log_preprocessing_artifact(
        artifact_prefix="kmed",
        steps=[
            ("Data cleaning", "drop_duplicates pada data gempa; dropna pada latitude dan longitude"),
            ("Spatial join", "point-in-polygon ke poligon GADM Level 2 untuk label kabupaten/kota"),
            ("Feature merge", "penggabungan fitur gempa dan DIBI ke level kabupaten"),
            ("Imputation", "median imputation untuk korban_total, rumah_rusak_total, dan fasum_rusak_total"),
            ("Normalization", "MinMaxScaler pada fitur clustering"),
        ],
    )

    mlflow.log_text(gadm_processed.to_json(), "kabupaten_boundaries.geojson")

    # 5. Normalization
    scaled_numerical_features_df, _ = normalize_data(raw_features_df)

    # 6. Merge back base dataframe
    base_df = pd.concat([
        raw_features_df[['id_kabupaten', 'nama_kabupaten']].reset_index(drop=True),
        scaled_numerical_features_df.reset_index(drop=True)
    ], axis=1)

    # --- LOOP UNTUK VARIASI K ---
    for k_val in range(2, 6):
        with mlflow.start_run(run_name=f"Run_K_{k_val}", nested=True):
            print(f"\n--- Processing k={k_val} ---")

            log_model_clustering_artifact(
                artifact_prefix=f"kmed_k{k_val}",
                k_val=k_val,
                feature_columns=CLUSTER_FEATURE_COLUMNS,
            )

            current_df = base_df.copy()

            dataset = getattr(mlflow.data, "from_pandas")(current_df, name=f"features_k_{k_val}")
            mlflow.log_input(dataset, context="training")

            mlflow.log_param("k_clusters", k_val)
            mlflow.log_param("algorithm", "K-Medoids")
            mlflow.log_param("preprocessing_cleaning", "drop_duplicates; dropna latitude/longitude")
            mlflow.log_param("preprocessing_spatial_join", "GADM Level 2 point-in-polygon")
            mlflow.log_param("preprocessing_scaling", "MinMaxScaler")
            mlflow.log_param("scaler", "MinMaxScaler")
            mlflow.log_param("n_features", len(CLUSTER_FEATURE_COLUMNS))
            mlflow.log_param("n_samples", len(current_df))

            current_df, model, _ = run_clustering(current_df, k=k_val)
            current_df, cluster_info_df, _, _ = generate_risk_table(current_df)
            current_df, pca_df, pca_model = run_pca(current_df)

            features_only = current_df[CLUSTER_FEATURE_COLUMNS].copy()
            silhouette = silhouette_score(features_only, current_df['cluster_label'])
            dbi = davies_bouldin_score(features_only, current_df['cluster_label'])
            chi = calinski_harabasz_score(features_only, current_df['cluster_label'])

            mlflow.log_metric("silhouette_score", silhouette)
            mlflow.log_metric("davies_bouldin_index", dbi)
            mlflow.log_metric("calinski_harabasz_index", chi)

            explained_var = pca_model.explained_variance_ratio_
            mlflow.log_metric("pca_pc1_variance", explained_var[0])
            mlflow.log_metric("pca_pc2_variance", explained_var[1])
            mlflow.log_metric("pca_total_variance", explained_var.sum())

            cluster_counts = current_df['cluster_label'].value_counts().to_dict()
            mlflow.log_dict(cluster_counts, f"cluster_distribution_k{k_val}.json")
            log_cluster_distribution_artifact(
                pd.Series(cluster_counts),
                artifact_prefix=f"kmed_k{k_val}",
            )

            human_readable_df = build_human_readable_cluster_df(raw_features_df, current_df)
            mlflow.log_text(human_readable_df.to_csv(index=False), f"clustered_k{k_val}.csv")
            mlflow.log_text(current_df.to_csv(index=False), f"clustered_k{k_val}_scaled.csv")
            mlflow.log_text(cluster_info_df.to_csv(index=False), f"cluster_summary_k{k_val}.csv")
            mlflow.log_text(pca_df.to_csv(index=False), f"pca_k{k_val}.csv")

            fig, ax = plt.subplots()
            ax.scatter(current_df['PC1'], current_df['PC2'], c=current_df['cluster_label'])
            ax.set_title(f"PCA K={k_val}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            mlflow.log_figure(fig, f"pca_plot_k{k_val}.png")
            plt.close(fig)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model_k{k_val}",
                registered_model_name="Earthquake_KMedoids_Registry"
            )

            print(f"Iteration k={k_val} finished | model_version=k{k_val}")


if __name__ == "__main__":
    main()