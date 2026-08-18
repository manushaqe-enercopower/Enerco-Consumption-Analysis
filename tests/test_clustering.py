import numpy as np
import pandas as pd

from src.clustering import (
    CLUSTER_FEATURES,
    cluster_company_profiles,
    evaluate_k_values,
    prepare_clustering_features,
    scale_features,
    select_best_k,
)


def make_profiles() -> pd.DataFrame:
    rows = []

    for i in range(12):
        high_group = i >= 6
        base = 5.0 if high_group else 1.0

        rows.append(
            {
                "company_code": f"Kompania {i + 1}",
                "peak_ratio": base + i * 0.03,
                "weekday_weekend_ratio": base + i * 0.02,
                "cv": base * 0.5 + i * 0.01,
                "load_factor": base * 0.2 + i * 0.005,
                "seasonality_index": base + i * 0.025,
                "trend_percent": base * 10 + i,
                "business_sector": None,
            }
        )

    return pd.DataFrame(rows)


def test_prepare_clustering_features_uses_required_columns():
    profiles = make_profiles()

    valid_profiles, features, excluded = prepare_clustering_features(profiles)

    assert len(valid_profiles) == 12
    assert excluded.empty
    assert list(features.columns) == CLUSTER_FEATURES


def test_scale_features_standardizes_values():
    profiles = make_profiles()

    _, features, _ = prepare_clustering_features(profiles)
    scaled, _ = scale_features(features)

    assert scaled.shape == (12, len(CLUSTER_FEATURES))
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-7)


def test_evaluate_k_values():
    profiles = make_profiles()

    _, features, _ = prepare_clustering_features(profiles)
    scaled, _ = scale_features(features)

    evaluation = evaluate_k_values(
        scaled,
        k_min=2,
        k_max=5,
    )

    assert evaluation["k"].tolist() == [2, 3, 4, 5]
    assert evaluation["inertia"].notna().all()
    assert evaluation["silhouette_score"].notna().all()


def test_select_best_k_uses_highest_silhouette():
    evaluation = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "inertia": [100.0, 70.0, 55.0],
            "silhouette_score": [0.40, 0.65, 0.50],
        }
    )

    assert select_best_k(evaluation) == 3


def test_cluster_company_profiles_assigns_cluster_id():
    profiles = make_profiles()

    clustered, evaluation, excluded, _, model = cluster_company_profiles(profiles)

    assert len(clustered) == 12
    assert "cluster_id" in clustered.columns
    assert clustered["cluster_id"].notna().all()
    assert clustered["cluster_id"].nunique() == model.n_clusters
    assert not evaluation.empty
    assert excluded.empty
