"""Realtime earthquake clustering training script."""

# pylint: disable=missing-function-docstring,missing-module-docstring,line-too-long,trailing-whitespace

import os
from pathlib import Path

import mlflow
import pandas as pd
import geopandas as gpd
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
]
def load_data():
    bmkg_raw = pd.read_csv("./data/raw/bmkg_raw.csv")
    gadm_raw = gpd.read_file("./data/raw/gadm41_IDN_2.json")

    return bmkg_raw, gadm_raw

def prepare_gadm(gadm):
    gadm = gadm.to_crs("EPSG:4326")

    gadm["luas_wilayah_km2"] = gadm.to_crs(epsg=3395).geometry.area / 10**6

    gadm = gadm.rename(columns={
        "GID_2": "id_kabupaten",
        "NAME_2": "nama_kabupaten"
    })

    return gadm[["id_kabupaten", "nama_kabupaten", "luas_wilayah_km2", "geometry"]]


def assign_kabupaten_for_events(bmkg: pd.DataFrame, gadm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign each BMKG event to a kabupaten.

    Events inside a kabupaten use point-in-polygon mapping.
    Events in the sea fall back to the nearest kabupaten so they are not dropped.
    """

    bmkg_copy = bmkg.copy()
    bmkg_copy["geometry"] = gpd.points_from_xy(bmkg_copy["longitude"], bmkg_copy["latitude"])

    bmkg_gdf = gpd.GeoDataFrame(bmkg_copy, geometry="geometry", crs="EPSG:4326")
    gadm_target = gadm[["id_kabupaten", "geometry"]].copy()

    within_join = gpd.sjoin(
        bmkg_gdf,
        gadm_target,
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    unmatched_mask = within_join["id_kabupaten"].isna()
    if unmatched_mask.any():
        gadm_metric = gadm_target.to_crs(epsg=3395).rename(columns={"id_kabupaten": "id_kabupaten_nearest"})
        bmkg_metric = within_join.loc[unmatched_mask].to_crs(epsg=3395)

        nearest_join = gpd.sjoin_nearest(
            bmkg_metric,
            gadm_metric,
            how="left",
            distance_col="_nearest_distance",
        ).drop(columns=["index_right"], errors="ignore")

        within_join.loc[nearest_join.index, "id_kabupaten"] = nearest_join["id_kabupaten_nearest"]

    return within_join
def process_bmkg(bmkg, gadm):
    # Realtime dihitung per gempa, bukan per kabupaten.
    bmkg_join = assign_kabupaten_for_events(bmkg, gadm)

    bmkg_join["frekuensi_gempa"] = 1.0
    bmkg_join["mag_max"] = bmkg_join["magnitude"]
    bmkg_join["mag_mean"] = bmkg_join["magnitude"]
    bmkg_join["depth_mean"] = bmkg_join["depth_km"]

    return bmkg_join

def merge_all(gadm, bmkg_agg):

    df = (
        bmkg_agg.merge(
            gadm[["id_kabupaten", "nama_kabupaten", "luas_wilayah_km2"]],
            on="id_kabupaten",
            how="left",
        )
    )

    # Fitur realtime dipakai per event.
    bmkg_cols = ["frekuensi_gempa", "mag_max", "mag_mean", "depth_mean"]
    df[bmkg_cols] = df[bmkg_cols].fillna(0)

    # 🔹 DIBI JANGAN 0 → biarkan NaN dulu (akan diimputasi nanti)
    # korban_total, rumah_rusak_total, fasum_rusak_total

    df = df.drop(columns="geometry")

    kabupaten_level_df = (
        df.groupby(["id_kabupaten", "nama_kabupaten"], as_index=False)
        .agg(
            luas_wilayah_km2=("luas_wilayah_km2", "first"),
            frekuensi_gempa=("frekuensi_gempa", "sum"),
            depth_mean=("depth_mean", "mean"),
            mag_max=("mag_max", "max"),
            mag_mean=("mag_mean", "mean"),
        )
    )

    return kabupaten_level_df[["id_kabupaten", "nama_kabupaten", "luas_wilayah_km2", *CLUSTER_FEATURE_COLUMNS]]


def log_feature_artifacts(raw_features_df, feature_columns, artifact_prefix):
    feature_df = raw_features_df[feature_columns].copy()
    corr_df = feature_df.corr(numeric_only=True)
    summary_df = feature_df.describe().T.reset_index().rename(columns={"index": "feature"})

    fig, ax = plt.subplots(figsize=(8, 6))
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


# def impute_dibi_features(df):

#     df["korban_total"] = df["korban_total"].fillna(df["korban_total"].median())
#     df["rumah_rusak_total"] = df["rumah_rusak_total"].fillna(df["rumah_rusak_total"].median())
#     df["fasum_rusak_total"] = df["fasum_rusak_total"].fillna(df["fasum_rusak_total"].median())

#     return df

def cleaning():
    # Ambil data mentah dan buang baris yang tidak lengkap.
    bmkg_raw_df, gadm_raw_df = load_data()

    bmkg_raw_df = bmkg_raw_df.drop_duplicates()
    bmkg_raw_df = bmkg_raw_df.dropna(subset=['latitude', 'longitude'])

    gadm_processed_df = prepare_gadm(gadm_raw_df)

    # Realtime tetap memakai event-level feature.
    bmkg_aggregated_df = process_bmkg(bmkg_raw_df, gadm_processed_df)
    # dibi_aggregated_df = process_dibi(dibi_raw_df, gadm_processed_df)
    
    raw_features_df = merge_all(gadm_processed_df, bmkg_aggregated_df)
    return raw_features_df


def normalize_data(raw_features_df):
    scaler = MinMaxScaler()

    numerical_features_to_scale_df = raw_features_df[CLUSTER_FEATURE_COLUMNS].copy()

    scaled_numerical_features_array = scaler.fit_transform(numerical_features_to_scale_df)

    scaled_numerical_features_df = pd.DataFrame(scaled_numerical_features_array, columns=numerical_features_to_scale_df.columns)

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
    """Tambahkan PC1 dan PC2 untuk visualisasi cluster."""

    from sklearn.decomposition import PCA

    if clustering_input_df is None:
        clustering_input_df = final_processed_df[CLUSTER_FEATURE_COLUMNS].copy()

    clustering_input_df = clustering_input_df.fillna(0)

    pca = PCA(n_components=n_components, random_state=42)
    pca_components_array = pca.fit_transform(clustering_input_df)

    pca_columns = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_components_array, columns=pca_columns)

    final_processed_df = final_processed_df.reset_index(drop=True)
    pca_df = pca_df.reset_index(drop=True)

    for col in pca_columns:
        final_processed_df[col] = pca_df[col]

    pca_result_df = pd.concat([
        final_processed_df[["id_kabupaten", "nama_kabupaten"]],
        pca_df
    ], axis=1)

    explained_variance = pca.explained_variance_ratio_

    print(f"PCA Variance: {explained_variance}")
    print(f"Total Variance Explained: {explained_variance.sum():.4f}")

    return final_processed_df, pca_result_df, pca


def generate_risk_table(final_processed_df):
    """Hitung skor dan level risiko tiap cluster."""

    cluster_counts = final_processed_df.groupby("cluster_label").size().rename("count")
    cluster_summary = final_processed_df.groupby("cluster_label")[CLUSTER_FEATURE_COLUMNS].mean()

    risk_calc_df = cluster_summary.copy()
    if "depth_mean" in risk_calc_df.columns:
        risk_calc_df["depth_mean"] = 1 - risk_calc_df["depth_mean"]

    cluster_summary["risk_score"] = risk_calc_df.mean(axis=1)
    cluster_summary_no_pca = cluster_summary.sort_values("risk_score")
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

    kabupaten_cluster_table = (
        final_processed_df[["id_kabupaten", "nama_kabupaten", "cluster_label"]]
        .merge(cluster_info_df, on="cluster_label", how="left")
    )

    kabupaten_cluster_table = kabupaten_cluster_table[
        ["id_kabupaten", "nama_kabupaten", "cluster_label", "risk_score", "risk_level", "count"]
    ]

    final_processed_df = final_processed_df.merge(
        cluster_info_df[["cluster_label", "risk_score", "risk_level"]],
        on="cluster_label",
        how="left"
    )

    final_with_risk_df = final_processed_df[
        ["id_kabupaten", "nama_kabupaten", "cluster_label", "risk_score", "risk_level"]
    ].copy()

    return (
        final_processed_df,
        cluster_info_df,
        kabupaten_cluster_table,
        final_with_risk_df
    )

def save_model(
    cluster_info_df,
    model_version: str,
    feature_columns,
):
    mlflow.log_dict({
        "model_version": model_version,
        "feature_columns": list(feature_columns),
        "cluster_risk_map": cluster_info_df.set_index("cluster_label").to_dict(orient="index"),
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
    mlflow.set_experiment("Earthquake_Realtime_Clustering_KMedoids_v6")
    
    _, gadm_raw = load_data()
    gadm_processed = prepare_gadm(gadm_raw) 

    raw_features_df = cleaning()

    mlflow.log_text(raw_features_df.to_csv(index=False), "raw_features_realtime.csv")
    log_feature_artifacts(
        raw_features_df=raw_features_df,
        feature_columns=CLUSTER_FEATURE_COLUMNS,
        artifact_prefix="realtime",
    )

    log_preprocessing_artifact(
        artifact_prefix="realtime",
        steps=[
            ("Data cleaning", "drop_duplicates pada data gempa; dropna pada latitude dan longitude"),
            ("Spatial join", "assign event gempa ke kabupaten dengan point-in-polygon lalu nearest fallback"),
            ("Feature merge", "agregasi fitur gempa ke level kabupaten"),
            ("Normalization", "MinMaxScaler pada fitur clustering"),
        ],
    )

    mlflow.log_text(gadm_processed.to_json(), "kabupaten_boundaries.geojson")

    scaled_numerical_features_df, _ = normalize_data(raw_features_df)

    base_df = pd.concat([
        raw_features_df[['id_kabupaten', 'nama_kabupaten']].reset_index(drop=True),
        scaled_numerical_features_df.reset_index(drop=True)
    ], axis=1)

    for k_val in range(2, 6):
        with mlflow.start_run(run_name=f"Run_K_{k_val}", nested=True):
            print(f"\n--- Processing k={k_val} ---")

            log_model_clustering_artifact(
                artifact_prefix=f"realtime_k{k_val}",
                k_val=k_val,
                feature_columns=CLUSTER_FEATURE_COLUMNS,
            )

            current_df = base_df.copy() 

            # Log dataset
            dataset = getattr(mlflow.data, "from_pandas")(current_df, name=f"features_k_{k_val}")
            mlflow.log_input(dataset, context="training")

            # Log parameter
            mlflow.log_param("k_clusters", k_val)
            mlflow.log_param("algorithm", "K-Medoids")
            mlflow.log_param("preprocessing_cleaning", "drop_duplicates; dropna latitude/longitude")
            mlflow.log_param("preprocessing_spatial_join", "point-in-polygon + nearest fallback")
            mlflow.log_param("preprocessing_scaling", "MinMaxScaler")
            mlflow.log_param("scaler", "MinMaxScaler")
            mlflow.log_param("n_features", len(CLUSTER_FEATURE_COLUMNS))
            mlflow.log_param("n_samples", len(current_df))

            # Clustering + Risk Table + PCA
            current_df, model, _ = run_clustering(current_df, k=k_val)

            current_df, cluster_info_df, _, _ = generate_risk_table(current_df)

            current_df, pca_df, pca_model = run_pca(current_df)

            features_only = current_df[CLUSTER_FEATURE_COLUMNS].copy()

            # Evaluation
            silhouette = silhouette_score(features_only, current_df['cluster_label'])
            dbi = davies_bouldin_score(features_only, current_df['cluster_label'])
            chi = calinski_harabasz_score(features_only, current_df['cluster_label'])

            # Log evaluation
            mlflow.log_metric("silhouette_score", silhouette)
            mlflow.log_metric("davies_bouldin_index", dbi)
            mlflow.log_metric("calinski_harabasz_index", chi)

            # Log PCA variance
            explained_var = pca_model.explained_variance_ratio_

            # Log PCA
            mlflow.log_metric("pca_pc1_variance", explained_var[0])
            mlflow.log_metric("pca_pc2_variance", explained_var[1])
            mlflow.log_metric("pca_total_variance", explained_var.sum())

            # Log cluster distribution
            cluster_counts = current_df['cluster_label'].value_counts().to_dict()
            mlflow.log_dict(cluster_counts, f"cluster_distribution_k{k_val}.json")

            log_cluster_distribution_artifact(
                pd.Series(cluster_counts),
                artifact_prefix=f"realtime_k{k_val}",
            )

            # Log clustered data
            # Log clustered data
            human_readable_df = build_human_readable_cluster_df(raw_features_df, current_df)
            mlflow.log_text(human_readable_df.to_csv(index=False), f"clustered_k{k_val}.csv")
            mlflow.log_text(current_df.to_csv(index=False), f"clustered_k{k_val}_scaled.csv")

            # Log cluster summary
            mlflow.log_text(cluster_info_df.to_csv(index=False), f"cluster_summary_k{k_val}.csv")

            # Log PCA data
            mlflow.log_text(pca_df.to_csv(index=False), f"pca_k{k_val}.csv")

            fig, ax = plt.subplots()
            ax.scatter(
                current_df['PC1'],
                current_df['PC2'],
                c=current_df['cluster_label']
            )
            ax.set_title(f"PCA K={k_val}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            mlflow.log_figure(fig, f"pca_plot_k{k_val}.png")
            plt.close(fig)
           
            # Save model metadata to MLflow
            save_model(
                cluster_info_df=cluster_info_df,
                model_version=f"k{k_val}",
                feature_columns=scaled_numerical_features_df.columns,
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model_k{k_val}"
            )

            print(f"Iteration k={k_val} finished | model_version=k{k_val}")

if __name__ == "__main__":
    main()
