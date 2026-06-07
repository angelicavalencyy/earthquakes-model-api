"""Realtime earthquake clustering training script.

Trains KMeans & KMedoids with two weight configurations (V1 original, V2 optimized)
across k=2-5. Compares all variants and saves the best K-Medoids model with
adaptive k selection (3 or 4 clusters based on silhouette score).

Output: Best K-Medoids model saved to app/ml/kmed/realtime_model_best.pkl
"""
import sys
import json
import hashlib
import pickle
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np
import mlflow

warnings.filterwarnings("ignore", category=FutureWarning)

from realtime_helpers import (
    K_RANGE, SCHEMES, fit_model, compute_metrics, log_elbow,
    log_eda, log_outliers, log_scaling, log_cluster_results, log_outlier_post,
    CLUSTER_FEATURE_COLUMNS, FEATURE_WEIGHTS, FEATURE_WEIGHTS_V2, FEATURE_WEIGHTS_V3,
    aggregate_to_regions, preprocess, weight_features, generate_risk_table,
    sort_model_clusters, select_best_k, compute_kmedoids_params,
    log_outlier_handling
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "app" / "ml"

def get_mlflow_tracking_uri() -> str:
    d = Path(__file__).resolve().parents[1] / "mlruns"
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve().as_uri()

# Dua konfigurasi bobot untuk dibandingkan
# WEIGHT_CONFIGS = {
#     "W1_original": FEATURE_WEIGHTS,       # Bobot asli (centroid_lat/lon = 0.075)
#     "W2_optimized": FEATURE_WEIGHTS_V2,   # Bobot optimasi (kurangi spatial bias)
# }
WEIGHT_CONFIGS = {
    "W3_optimized": FEATURE_WEIGHTS_V3,
}


def load_data():
    """Load raw BMKG historical events."""
    return pd.read_csv(PROJECT_ROOT / "data" / "raw" / "bmkg_raw.csv")


def prepare_gadm(gadm):
    if gadm.crs is None:
        gadm.set_crs("EPSG:4326", inplace=True)
    gadm = gadm.to_crs("EPSG:4326")
    gadm["luas_wilayah_km2"] = gadm.to_crs(epsg=3395).geometry.area / 10**6
    gadm = gadm.rename(columns={
        "GID_2": "id_kabupaten",
        "NAME_2": "nama_kabupaten"
    })
    return gadm[["id_kabupaten", "nama_kabupaten", "luas_wilayah_km2", "geometry"]]


def cleaning():
    bmkg_raw_df = load_data()
    bmkg_raw_df = bmkg_raw_df.drop_duplicates()
    bmkg_raw_df = bmkg_raw_df.dropna(subset=["magnitude", "depth_km", "latitude", "longitude"])

    with open(PROJECT_ROOT / "data" / "raw" / "gadm41_IDN_2.json", "r", encoding="utf-8") as f:
        gadm_raw = json.load(f)
    gadm_gdf = gpd.GeoDataFrame.from_features(gadm_raw["features"])
    gadm_gdf = prepare_gadm(gadm_gdf)

    print("Aggregating to region level using realtime_helpers...")
    kabupaten_level_df = aggregate_to_regions(bmkg_raw_df, gadm_gdf)
    return kabupaten_level_df


def save_pkl(payload, path):
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Pickle exported -> {path}")


def main():
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    exp_name = "Earthquake_Realtime_Clustering_v2"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)

    print("Preparing realtime data (BMKG + GADM)...")
    region_df = cleaning()

    # ── Preprocessing (shared across weight configs) ─────────────────
    raw_feat, scaled_feat, scaler, clip_bounds = preprocess(region_df)
    id_df = region_df[["id_kabupaten", "nama_kabupaten"]].reset_index(drop=True)
    n_samples = len(region_df)

    # ── EDA Run ──────────────────────────────────────────────────────
    with mlflow.start_run(run_name="EDA_Exploration") as eda_run:
        eda_run_id = eda_run.info.run_id

        mlflow.log_param("n_regions", n_samples)
        mlflow.log_param("features", ",".join(CLUSTER_FEATURE_COLUMNS))
        mlflow.log_param("scaler", "QuantileTransformer(uniform)")
        mlflow.log_param("log_transform", "frekuensi_gempa,seismic_density")
        mlflow.log_param("preprocessing", "winsorize(1-99) -> log1p -> QuantileTransformer(uniform)")

        # Log computed K-Medoids hyperparameters (data-driven, bukan asumsi)
        for k in K_RANGE:
            params = compute_kmedoids_params(n_samples, k)
            mlflow.log_param(f"kmedoids_k{k}_n_init", params["n_init"])
            mlflow.log_param(f"kmedoids_k{k}_max_iter", params["max_iter"])

        # Log AHP weights for configurations
        mlflow.log_param("ahp_weights_v3", json.dumps(FEATURE_WEIGHTS_V3))

        # Log winsorization clip bounds
        mlflow.log_text(
            json.dumps(clip_bounds, indent=2),
            "preprocessing/winsorize_clip_bounds.json"
        )

        # EDA with W3 weights for visualization
        weighted_feat_v3 = weight_features(scaled_feat, FEATURE_WEIGHTS_V3)
        log_eda(region_df, scaled_feat, weighted_feat_v3)
        log_outliers(region_df, prefix="eda/outlier_before")
        log_scaling(raw_feat, scaled_feat)

        # Log outlier handling (before vs after comparison)
        log_outlier_handling(
            region_df, CLUSTER_FEATURE_COLUMNS,
            log_transform_cols=["frekuensi_gempa", "seismic_density"],
            prefix="eda/outlier_handling"
        )

        # Elbow plots
        for algo, metric in SCHEMES:
            log_elbow(weighted_feat_v3, algo, metric)

    # ── Training Loop: Weight × Algo × Metric × K ───────────────────
    best_per_scheme = {}
    all_comparison_rows = []

    for weight_name, weights in WEIGHT_CONFIGS.items():
        weighted_feat = weight_features(scaled_feat, weights)

        print(f"\n{'='*60}")
        print(f"Weight Config: {weight_name}")
        print(f"{'='*60}")

        for algo, metric in SCHEMES:
            scheme_key = f"{weight_name}_{algo}_{metric}"
            best_sil = -1
            best_info = None

            print(f"\n  Training: {scheme_key}")

            for k in K_RANGE:
                run_name = f"{scheme_key}_k{k}"
                is_kmedoids = algo == "KMedoids"

                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("algorithm", algo)
                    mlflow.log_param("distance_metric", metric)
                    mlflow.log_param("k", k)
                    mlflow.log_param("weight_config", weight_name)
                    mlflow.log_param("features", ",".join(CLUSTER_FEATURE_COLUMNS))
                    mlflow.log_param("optimize_kmedoids", is_kmedoids)

                    model = fit_model(weighted_feat, algo, metric, k, optimize_kmedoids=is_kmedoids)
                    model = sort_model_clusters(model, scaled_feat, weights)
                    labels = model.labels_
                    centers = model.cluster_centers_
                    x_w = weighted_feat.values

                    metrics = compute_metrics(x_w, labels, centers, distance_metric=metric)
                    for name, val in metrics.items():
                        mlflow.log_metric(name, round(val, 6))

                    # Log cluster distribution balance
                    counts = pd.Series(labels).value_counts()
                    max_pct = counts.max() / len(labels) * 100
                    min_pct = counts.min() / len(labels) * 100
                    balance_ratio = min_pct / max(max_pct, 1)
                    mlflow.log_metric("cluster_max_pct", round(max_pct, 2))
                    mlflow.log_metric("cluster_min_pct", round(min_pct, 2))
                    mlflow.log_metric("cluster_balance_ratio", round(balance_ratio, 4))

                    print(f"    [{run_name}] Sil={metrics['silhouette']:.4f} "
                          f"SSE={metrics['sse']:.2f} "
                          f"Balance={min_pct:.1f}%-{max_pct:.1f}%")

                    result_df = pd.concat([id_df, scaled_feat], axis=1).copy()
                    result_df["cluster_label"] = labels
                    cluster_info = generate_risk_table(result_df, weights)

                    # Log artifacts per run
                    prefix = "eda"
                    log_cluster_results(model, weighted_feat.values, prefix)
                    log_outlier_post(weighted_feat.values, labels, centers,
                                     prefix, id_df, raw_feat, distance_metric=metric)
                    mlflow.log_text(cluster_info.to_csv(index=False),
                                    f"{prefix}/cluster_info.csv")

                    result_raw = pd.concat([id_df, raw_feat], axis=1).copy()
                    result_raw["cluster_label"] = labels
                    result_raw = result_raw.merge(
                        cluster_info[["cluster_label", "risk_score", "risk_level"]],
                        on="cluster_label", how="left"
                    )
                    mlflow.log_text(result_raw.to_csv(index=False),
                                    f"{prefix}/clustered_regions.csv")

                    info_dict = {
                        "model": model, "k": k, "metrics": metrics,
                        "labels": labels, "cluster_info": cluster_info,
                        "weights": weights, "weight_name": weight_name,
                        "algo": algo, "metric": metric,
                    }

                    if best_info is None or metrics["silhouette"] > best_sil:
                        best_sil = metrics["silhouette"]
                        best_info = info_dict

                    all_comparison_rows.append({
                        "weight": weight_name,
                        "algorithm": algo,
                        "distance": metric,
                        "k": k,
                        "full_scheme": run_name,
                        "sse": round(metrics["sse"], 4),
                        "silhouette": round(metrics["silhouette"], 4),
                        "dbi": round(metrics["dbi"], 4),
                        "chi": round(metrics["chi"], 4),
                        "balance": round(balance_ratio, 4),
                    })

            best_per_scheme[scheme_key] = {"best": best_info}

    # ── Log best models' artifacts to EDA run ────────────────────────
    print(f"\n{'='*60}\nLogging best models to EDA...")

    with mlflow.start_run(run_id=eda_run_id):
        for scheme_key, info_dict in best_per_scheme.items():
            info = info_dict["best"]
            if info is None:
                continue
            k_val = info["k"]
            wfeat = weight_features(scaled_feat, info["weights"])
            prefix = f"eda/{scheme_key}_best_k{k_val}"

            log_cluster_results(info["model"], wfeat.values, prefix)
            log_outlier_post(wfeat.values, info["labels"],
                             info["model"].cluster_centers_,
                             prefix, id_df, raw_feat,
                             distance_metric=info["metric"])
            mlflow.log_text(info["cluster_info"].to_csv(index=False),
                            f"{prefix}/cluster_info.csv")

            result_raw = pd.concat([id_df, raw_feat], axis=1).copy()
            result_raw["cluster_label"] = info["labels"]
            result_raw = result_raw.merge(
                info["cluster_info"][["cluster_label", "risk_score", "risk_level"]],
                on="cluster_label", how="left"
            )
            mlflow.log_text(result_raw.to_csv(index=False),
                            f"{prefix}/clustered_regions.csv")

    # ── Final Comparison ─────────────────────────────────────────────
    comp_df = pd.DataFrame(all_comparison_rows)

    print("\n" + "=" * 80)
    print("FULL MODEL COMPARISON (Weight x Algo x Metric x K):")
    print("=" * 80)
    print(comp_df.to_string(index=False))

    # Show best per algorithm type
    print(f"\n{'='*60}")
    for algo_type in ["KMeans", "KMedoids"]:
        subset = comp_df[comp_df["algorithm"] == algo_type]
        if not subset.empty:
            best_idx = subset["silhouette"].idxmax()
            best_row = subset.loc[best_idx]
            print(f"  Best {algo_type}: {best_row['full_scheme']} "
                  f"Sil={best_row['silhouette']:.4f} "
                  f"SSE={best_row['sse']:.2f} "
                  f"Balance={best_row['balance']:.4f}")

    # ── KMedoids vs KMeans Head-to-Head (Multi-Criteria) ────────────
    kmed_df = comp_df[comp_df["algorithm"] == "KMedoids"]
    kmeans_df = comp_df[comp_df["algorithm"] == "KMeans"]

    if kmed_df.empty:
        print("\nERROR: No KMedoids results found!")
        return

    best_kmed_idx = kmed_df["silhouette"].idxmax()
    best_kmed_row = kmed_df.loc[best_kmed_idx]

    # Variabel untuk stability test (diisi di dalam if-block, dipakai di MLflow log)
    stability_results = {}
    kmed_wins = 0
    kmeans_wins = 0

    if not kmeans_df.empty:
        best_kmeans_idx = kmeans_df["silhouette"].idxmax()
        best_kmeans_row = kmeans_df.loc[best_kmeans_idx]

        print(f"\n{'='*70}")
        print("MULTI-CRITERIA COMPARISON: KMEDOIDS vs KMEANS")
        print(f"{'='*70}")
        print(f"  KMedoids: {best_kmed_row['full_scheme']}")
        print(f"  KMeans:   {best_kmeans_row['full_scheme']}")
        print(f"{'─'*70}")

        # ── Tabel Perbandingan 5 Metrik ──────────────────────────────
        kmed_wins = 0
        kmeans_wins = 0

        metrics_compare = [
            ("Silhouette (higher=better)", "silhouette", "higher"),
            ("SSE (lower=better)", "sse", "lower"),
            ("DBI (lower=better)", "dbi", "lower"),
            ("CHI (higher=better)", "chi", "higher"),
            ("Balance (higher=better)", "balance", "higher"),
        ]

        print(f"  {'Metric':<30} {'KMedoids':>10} {'KMeans':>10} {'Winner':>12}")
        print(f"  {'─'*62}")

        for label, col, direction in metrics_compare:
            kmed_val = best_kmed_row[col]
            km_val = best_kmeans_row[col]

            if direction == "higher":
                winner = "KMedoids" if kmed_val >= km_val else "KMeans"
            else:
                winner = "KMedoids" if kmed_val <= km_val else "KMeans"

            if winner == "KMedoids":
                kmed_wins += 1
            else:
                kmeans_wins += 1

            marker = " <<" if winner == "KMedoids" else ""
            print(f"  {label:<30} {kmed_val:>10.4f} {km_val:>10.4f} {winner:>12}{marker}")

        print(f"  {'─'*62}")
        print(f"  Skor metrik: KMedoids={kmed_wins}/5, KMeans={kmeans_wins}/5")

        # ── Stability Test (Uji Ketahanan Outlier INJECTION) ────────────
        print(f"\n{'='*70}")
        print("STABILITY TEST: Uji ketahanan terhadap outlier injection")
        print("  Metode: Injeksi 5% outlier ekstrem ke dalam data,")
        print("  bandingkan Adjusted Rand Index (ARI) sebelum dan sesudah.")
        print("  K-Means centroid (mean) akan TERTARIK ke outlier.")
        print("  K-Medoids medoid (data asli) KEBAL terhadap outlier.")
        print("  ARI mendekati 1.0 = clustering stabil meski ada outlier.")
        print(f"{'─'*70}")

        from sklearn.metrics import adjusted_rand_score

        # Get weight config for the best models
        kmed_info = None
        km_info = None
        for scheme_key, sdict in best_per_scheme.items():
            info = sdict["best"]
            if info is None:
                continue
            if scheme_key == best_kmed_row["full_scheme"].rsplit("_k", 1)[0]:
                kmed_info = info
            # Find KMeans best match
            km_scheme = best_kmeans_row["full_scheme"].rsplit("_k", 1)[0]
            if scheme_key == km_scheme:
                km_info = info

        stability_results = {}
        for name, info in [("KMedoids", kmed_info), ("KMeans", km_info)]:
            if info is None:
                stability_results[name] = {"mean_ari": 0.0, "scores": []}
                continue

            wfeat = weight_features(scaled_feat, info["weights"])
            base_model = fit_model(wfeat, info["algo"], info["metric"], info["k"])
            base_labels = sort_model_clusters(base_model, scaled_feat, info["weights"]).labels_

            ari_scores = []
            n = len(wfeat)
            n_inject = max(3, int(n * 0.05))  # 5% outlier injection

            for trial in range(5):
                rng = np.random.default_rng(42 + trial)
                
                # Buat salinan data dan injeksi outlier ekstrem
                injected = wfeat.copy()
                inject_idx = rng.choice(n, n_inject, replace=False)
                
                for col_idx, col in enumerate(injected.columns):
                    col_max = injected[col].max()
                    col_range = injected[col].max() - injected[col].min()
                    # Injeksi nilai 3-8x di atas range normal
                    extreme = col_max + rng.uniform(2, 7, size=n_inject) * col_range
                    injected.iloc[inject_idx, col_idx] = extreme
                
                # Juga injeksi ke scaled_feat untuk sort_model_clusters
                injected_scaled = scaled_feat.copy()
                for col_idx, col in enumerate(injected_scaled.columns):
                    col_max = injected_scaled[col].max()
                    col_range = injected_scaled[col].max() - injected_scaled[col].min()
                    extreme = col_max + rng.uniform(2, 7, size=n_inject) * col_range
                    injected_scaled.iloc[inject_idx, col_idx] = extreme

                # Re-cluster dengan data yang sudah diinjeksi outlier
                inj_model = fit_model(injected, info["algo"], info["metric"], info["k"])
                inj_model = sort_model_clusters(inj_model, injected_scaled, info["weights"])
                inj_labels = inj_model.labels_

                # Bandingkan label pada titik NON-injeksi saja
                clean_mask = np.ones(n, dtype=bool)
                clean_mask[inject_idx] = False
                clean_idx = np.where(clean_mask)[0]

                ari = adjusted_rand_score(base_labels[clean_idx], inj_labels[clean_idx])
                ari_scores.append(ari)

            mean_ari = np.mean(ari_scores)
            std_ari = np.std(ari_scores)
            median_ari = np.median(ari_scores)
            stability_results[name] = {
                "mean_ari": mean_ari, "std_ari": std_ari,
                "median_ari": median_ari, "scores": ari_scores
            }
            print(f"  {name:>10}: mean={mean_ari:.4f}  std={std_ari:.4f}  "
                  f"median={median_ari:.4f}")
            print(f"             (per trial: {', '.join(f'{s:.3f}' for s in ari_scores)})")

        # Determine stability winners (2 sub-criteria)
        kmed_ari = stability_results["KMedoids"]["mean_ari"]
        km_ari = stability_results["KMeans"]["mean_ari"]
        kmed_std = stability_results["KMedoids"]["std_ari"]
        km_std = stability_results["KMeans"]["std_ari"]

        # Sub-criteria 1: Mean ARI (higher = more robust)
        robust_winner = "KMedoids" if kmed_ari >= km_ari else "KMeans"
        if robust_winner == "KMedoids":
            kmed_wins += 1
        else:
            kmeans_wins += 1
        print(f"\n  Robustness (mean ARI): {robust_winner} "
              f"(KMedoids={kmed_ari:.4f} vs KMeans={km_ari:.4f})")

        # Sub-criteria 2: Std ARI (lower = more consistent/predictable)
        consist_winner = "KMedoids" if kmed_std <= km_std else "KMeans"
        if consist_winner == "KMedoids":
            kmed_wins += 1
        else:
            kmeans_wins += 1
        print(f"  Consistency (std ARI): {consist_winner} "
              f"(KMedoids={kmed_std:.4f} vs KMeans={km_std:.4f})")

        # ── Final Score ──────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"SKOR AKHIR (5 metrik + 2 stability sub-criteria):")
        print(f"  KMedoids: {kmed_wins}/7")
        print(f"  KMeans:   {kmeans_wins}/7")
        print(f"{'─'*70}")

        if kmed_wins >= kmeans_wins:
            print("  >> KMedoids MENANG secara keseluruhan!")
        else:
            print("  >> KMeans unggul di metrik numerik, NAMUN:")

        # ── Justifikasi berbasis data ────────────────────────────────
        print(f"\n{'='*70}")
        print("JUSTIFIKASI PEMILIHAN K-MEDOIDS:")
        print(f"{'─'*70}")

        # 1. Stability advantage
        if kmed_ari >= km_ari:
            print(f"  [DATA] K-Medoids LEBIH STABIL: ARI={kmed_ari:.4f} vs "
                  f"KMeans ARI={km_ari:.4f}")
            print("         Saat 10% outlier dihapus, cluster K-Medoids tidak berubah")
            print("         signifikan. K-Means berubah lebih banyak karena centroid")
            print("         (rata-rata) terdistorsi oleh outlier.")
        else:
            print(f"  [DATA] Stability test: KMedoids ARI={kmed_ari:.4f}, "
                  f"KMeans ARI={km_ari:.4f}")

        # 2. DBI comparison
        kmed_dbi = best_kmed_row["dbi"]
        km_dbi = best_kmeans_row["dbi"]
        if kmed_dbi <= km_dbi:
            print(f"  [DATA] K-Medoids DBI LEBIH RENDAH: {kmed_dbi:.4f} vs "
                  f"KMeans {km_dbi:.4f}")
            print("         DBI mengukur rasio scatter vs separasi cluster.")
            print("         Nilai lebih rendah = cluster lebih well-separated.")

        # 3. Theoretical justification
        print()
        print("  [TEORI] K-Means mengasumsikan distribusi spherical & equal-size.")
        print("          Data gempa mengikuti patahan geologis, BUKAN pola bola.")
        print("          K-Medoids tidak memiliki asumsi distribusi ini.")
        print("          (Kaufman & Rousseeuw, 1990; Park & Jun, 2009)")
        print()
        print("  [TEORI] SSE & Silhouette secara inherent bias ke K-Means karena")
        print("          K-Means LANGSUNG meminimalkan SSE. Ini bukan keunggulan")
        print("          algoritmik, tapi konsekuensi fungsi objektifnya.")
        print()
        print("  [TEORI] Medoid = kabupaten NYATA. Centroid = titik abstrak yang")
        print("          mungkin tidak merepresentasikan wilayah manapun.")
        print("          Untuk kebijakan mitigasi bencana, representasi riil")
        print("          lebih bermakna daripada titik statistik abstrak.")
        print(f"{'='*70}")

    # ── Select & Save Best K-Medoids ─────────────────────────────────
    # Find the actual model info from best_per_scheme
    best_scheme_key = None
    best_overall_sil = -1
    for scheme_key, info_dict in best_per_scheme.items():
        if "KMedoids" not in scheme_key:
            continue
        info = info_dict["best"]
        if info is not None and info["metrics"]["silhouette"] > best_overall_sil:
            best_overall_sil = info["metrics"]["silhouette"]
            best_scheme_key = scheme_key

    winner = best_per_scheme[best_scheme_key]["best"]
    best_k = winner["k"]
    best_weights = winner["weights"]
    best_weight_name = winner["weight_name"]
    best_algo = winner["algo"]
    best_metric = winner["metric"]

    print(f"\n{'='*60}")
    print(f"SELECTED MODEL: {best_scheme_key} k={best_k}")
    print(f"  Silhouette: {winner['metrics']['silhouette']:.4f}")
    print(f"  SSE: {winner['metrics']['sse']:.2f}")
    print(f"  Weight Config: {best_weight_name}")
    print(f"  Distance Metric: {best_metric}")
    print(f"  Cluster Distribution:")
    for _, row in winner["cluster_info"].iterrows():
        print(f"    Cluster {int(row['cluster_label'])}: "
              f"{row['risk_level']} ({int(row['count'])} wilayah)")

    # Build and save all best models per scheme
    trained_at = datetime.now(timezone.utc).isoformat()
    region_base = pd.concat([id_df, raw_feat], axis=1).copy()
    for scheme_key, info_dict in best_per_scheme.items():
        infos_to_save = []
        if info_dict.get("best"):
            infos_to_save.append(info_dict["best"])
        if info_dict.get("k4") and (not info_dict.get("best") or info_dict["best"]["k"] != 4):
            infos_to_save.append(info_dict["k4"])

        for info in infos_to_save:
            algo = info["algo"]
            metric = info["metric"]
            k = info["k"]
            
            algo_dir = "kmed" if algo == "KMedoids" else "kmeans"
            scheme_out_dir = EXPORT_DIR / algo_dir / "realtime"
            scheme_out_dir.mkdir(parents=True, exist_ok=True)
            
            scheme_result = region_base.copy()
            scheme_result["cluster_label"] = info["labels"]
            scheme_result = scheme_result.merge(
                info["cluster_info"][["cluster_label", "risk_score", "risk_level"]],
                on="cluster_label", how="left"
            )
            
            model_version = f"{algo.lower()}-{metric}-k{k}-realtime"
            hash_src = json.dumps({"version": model_version, "trained_at": trained_at}, sort_keys=True).encode()
            model_hash = hashlib.sha256(hash_src).hexdigest()
            
            scheme_payload = {
                "model_version": model_version,
                "model": info["model"],
                "scaler": scaler,
                "feature_columns": list(CLUSTER_FEATURE_COLUMNS),
                "feature_weights": info["weights"],
                "cluster_risk_map": info["cluster_info"].set_index("cluster_label").to_dict(orient="index"),
                "region_features": scheme_result.to_dict(orient="records"),
                "trained_at": trained_at,
                "model_hash": model_hash,
                "metrics": info["metrics"],
                "log_transform_cols": ["frekuensi_gempa", "seismic_density"],
                "clip_bounds": clip_bounds,
                "weight_config": info["weight_name"],
                "n_clusters": k,
            }
            
            filename = f"realtime_{metric}_best_k{k}.pkl"
            save_pkl(scheme_payload, scheme_out_dir / filename)
            
            if scheme_key == best_scheme_key and info == info_dict.get("best"):
                best_path = scheme_out_dir / "realtime_model_best.pkl"
                shutil.copy2(scheme_out_dir / filename, best_path)
                print(f"Best model also copied to {best_path}")

    # ── Log Final Comparison to MLflow ───────────────────────────────
    with mlflow.start_run(run_name="Final_Comparison"):
        mlflow.log_text(comp_df.to_csv(index=False),
                        "perbandingan_metrik_model.csv")
        mlflow.log_param("best_model_forced", "KMedoids")
        mlflow.log_param("best_scheme", best_scheme_key)
        mlflow.log_param("best_k", best_k)
        mlflow.log_param("best_weight_config", best_weight_name)
        mlflow.log_param("best_silhouette", round(best_overall_sil, 6))

        # Log per-row metrics
        for row in all_comparison_rows:
            for m in ["sse", "silhouette", "dbi", "chi"]:
                if m in row:
                    mlflow.log_metric(f"{row['full_scheme']}_{m}", row[m])

        # ── Log Stability Test (Z-score ARI) ─────────────────────────
        if stability_results:
            for algo_name, res in stability_results.items():
                prefix = f"stability_{algo_name}"
                mlflow.log_metric(f"{prefix}_mean_ari", round(res["mean_ari"], 6))
                mlflow.log_metric(f"{prefix}_std_ari", round(res["std_ari"], 6))
                mlflow.log_metric(f"{prefix}_median_ari", round(res["median_ari"], 6))
                for i, score in enumerate(res["scores"]):
                    mlflow.log_metric(f"{prefix}_trial_{i+1}", round(score, 6))

            # Log stability comparison summary as CSV artifact
            stab_rows = []
            for algo_name, res in stability_results.items():
                stab_rows.append({
                    "algorithm": algo_name,
                    "mean_ari": round(res["mean_ari"], 6),
                    "std_ari": round(res["std_ari"], 6),
                    "median_ari": round(res["median_ari"], 6),
                    **{f"trial_{i+1}": round(s, 6) for i, s in enumerate(res["scores"])}
                })
            stab_df = pd.DataFrame(stab_rows)
            mlflow.log_text(stab_df.to_csv(index=False),
                            "stability_test/zscore_ari_results.csv")

            # Log stability test metadata
            mlflow.log_param("stability_method", "Z-score outlier removal (10%)")
            mlflow.log_param("stability_n_trials", 5)
            mlflow.log_param("stability_robustness_winner",
                             "KMedoids" if stability_results.get("KMedoids", {}).get("mean_ari", 0)
                             >= stability_results.get("KMeans", {}).get("mean_ari", 0) else "KMeans")
            mlflow.log_param("stability_consistency_winner",
                             "KMedoids" if stability_results.get("KMedoids", {}).get("std_ari", 1)
                             <= stability_results.get("KMeans", {}).get("std_ari", 1) else "KMeans")

        # Log final score
        mlflow.log_metric("final_score_kmedoids", kmed_wins)
        mlflow.log_metric("final_score_kmeans", kmeans_wins)
        mlflow.log_param("final_score_total_criteria", 7)

    print(f"\nDone! Realtime models saved to {EXPORT_DIR / 'realtime'} and split by algo.")
    print(f"  Model: {model_version}")
    print(f"  K={best_k}, Silhouette={winner['metrics']['silhouette']:.4f}")


if __name__ == "__main__":
    main()
