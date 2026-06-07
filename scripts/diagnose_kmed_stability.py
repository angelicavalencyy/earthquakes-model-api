import traceback
import numpy as np
import types
import sys

# Inject a minimal mlflow stub
if 'mlflow' not in sys.modules:
    mlflow_stub = types.SimpleNamespace(
        log_text=lambda *a, **k: None,
        log_figure=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
        log_dict=lambda *a, **k: None,
        set_tracking_uri=lambda *a, **k: None,
        get_experiment_by_name=lambda *a, **k: None,
        create_experiment=lambda *a, **k: None,
        set_experiment=lambda *a, **k: None,
        start_run=lambda *a, **k: types.SimpleNamespace(__enter__=lambda *a: None, __exit__=lambda *a, **k: None)
    )
    sys.modules['mlflow'] = mlflow_stub

try:
    from src import train_realtime as tr
except Exception:
    sys.path.append('.')
    from src import train_realtime as tr

print('Loading and preprocessing (this may take a few seconds)...')
raw = None
try:
    raw = tr.cleaning()
except Exception as e:
    print('Full cleaning failed or geopandas missing - falling back to processed CSV:', e)
    for path in ["./data/processed/raw_features_realtime.csv", "./data/processed/raw_features_v2.csv"]:
        try:
            import pandas as _pd
            import os
            if os.path.exists(path):
                raw = _pd.read_csv(path)
                print(f'Loaded processed features from {path}')
                break
        except Exception:
            continue

if raw is None:
    raise RuntimeError('Unable to load feature data for diagnostics')

scaled, scaler = tr.normalize_data(raw)
weighted = tr.weight_clustering_features(scaled)

from sklearn.metrics import silhouette_score

seeds = [0, 1, 2, 42, 99]
methods = [
    ("KMedoids_Euclidean", tr.fit_kmedoids_euclidean),
    ("KMedoids_Manhattan", tr.fit_kmedoids_manhattan),
    ("KMeans_Euclidean", tr.fit_kmeans_euclidean),
    ("KMeans_Manhattan", tr.fit_kmeans_manhattan),
]

all_results = {}

for method_name, fit_fn in methods:
    print(f"\n=== Method: {method_name} ===")
    labels_list = []
    errors = []
    silhouettes = []
    uniques = []

    for seed in seeds:
        try:
            model, labels = fit_fn(weighted, k=4, random_state=seed)
            labels = np.asarray(labels)
            labels_list.append(labels)
            uniques.append(len(np.unique(labels)))
            try:
                # Use cluster features only for silhouette
                s = silhouette_score(weighted[tr.CLUSTER_FEATURE_COLUMNS].to_numpy(), labels)
            except Exception as e:
                s = f'silhouette_error:{e}'
            silhouettes.append(s)
            print(f'Run seed={seed}: unique_labels={uniques[-1]}, silhouette={s}')
        except Exception as e:
            errors.append((seed, traceback.format_exc()))
            print(f'Run seed={seed} failed: {e}')

    # Compare pairwise equality for this method
    pairwise_equal = 0
    total_pairs = 0
    if len(labels_list) > 1:
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                total_pairs += 1
                if np.array_equal(labels_list[i], labels_list[j]):
                    pairwise_equal += 1
        print(f'Pairwise equal runs for {method_name}: {pairwise_equal}/{total_pairs}')
    else:
        print(f'Not enough successful runs for {method_name} to check stability.')

    if errors:
        print('\nErrors encountered:')
        for seed, tb in errors:
            print('---')
            print(f'Seed={seed}')
            print(tb)

print('\nAll diagnostics complete')
