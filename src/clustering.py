from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROFILE_PATH = Path("data/processed/profile_metrics/company_profiles.parquet")
REPORT_PATH = Path("reports/hapi_6_clustering.xlsx")
ELBOW_PLOT_PATH = Path("reports/hapi_6_elbow.png")
SILHOUETTE_PLOT_PATH = Path("reports/hapi_6_silhouette.png")

CLUSTER_FEATURES = [
    "peak_ratio",
    "weekday_weekend_ratio",
    "cv",
    "load_factor",
    "seasonality_index",
    "trend_percent",
]


def prepare_clustering_features(
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_columns = [
        column for column in CLUSTER_FEATURES if column not in profiles.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing clustering columns: {', '.join(missing_columns)}")

    numeric_features = (
        profiles[CLUSTER_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    valid_mask = numeric_features.notna().all(axis=1)

    valid_profiles = profiles.loc[valid_mask].copy().reset_index(drop=True)
    valid_features = numeric_features.loc[valid_mask].copy().reset_index(drop=True)
    excluded_profiles = profiles.loc[~valid_mask].copy().reset_index(drop=True)

    if len(valid_profiles) < 3:
        raise ValueError("At least 3 valid companies are required for clustering.")

    return valid_profiles, valid_features, excluded_profiles


def scale_features(
    features: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler


def evaluate_k_values(
    scaled_features: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    n_samples = len(scaled_features)

    effective_k_max = min(k_max, n_samples - 1)

    if effective_k_max < k_min:
        raise ValueError("Not enough companies to evaluate clustering.")

    results = []

    for k in range(k_min, effective_k_max + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

        labels = model.fit_predict(scaled_features)

        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(
                    scaled_features,
                    labels,
                ),
            }
        )

    return pd.DataFrame(results)


def select_best_k(evaluation: pd.DataFrame) -> int:
    if evaluation.empty:
        raise ValueError("Clustering evaluation is empty.")

    best_row = evaluation.sort_values(
        by=["silhouette_score", "k"],
        ascending=[False, True],
    ).iloc[0]

    return int(best_row["k"])


def cluster_company_profiles(
    profiles: pd.DataFrame,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
    KMeans,
]:
    valid_profiles, features, excluded_profiles = prepare_clustering_features(profiles)

    scaled_features, scaler = scale_features(features)

    evaluation = evaluate_k_values(
        scaled_features,
        random_state=random_state,
    )

    best_k = select_best_k(evaluation)

    model = KMeans(
        n_clusters=best_k,
        random_state=random_state,
        n_init=10,
    )

    labels = model.fit_predict(scaled_features)

    clustered_profiles = valid_profiles.copy()
    clustered_profiles["cluster_id"] = labels

    return (
        clustered_profiles,
        evaluation,
        excluded_profiles,
        scaler,
        model,
    )


def create_cluster_summary(
    clustered_profiles: pd.DataFrame,
) -> pd.DataFrame:
    aggregation = {
        "company_code": "count",
        **{feature: "mean" for feature in CLUSTER_FEATURES},
    }

    summary = (
        clustered_profiles.groupby("cluster_id")
        .agg(aggregation)
        .reset_index()
        .rename(columns={"company_code": "company_count"})
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )

    return summary


def create_sector_comparison_status(
    clustered_profiles: pd.DataFrame,
) -> pd.DataFrame:
    if (
        "business_sector" not in clustered_profiles.columns
        or clustered_profiles["business_sector"].dropna().empty
    ):
        return pd.DataFrame(
            [
                {
                    "status": "SKIPPED",
                    "reason": (
                        "Business-sector metadata is not mapped in the "
                        "available dataset, so Cluster_ID cannot be "
                        "reliably compared with declared business sector."
                    ),
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "status": "AVAILABLE",
                "reason": "Business-sector metadata is available.",
            }
        ]
    )


def save_cluster_report(
    clustered_profiles: pd.DataFrame,
    evaluation: pd.DataFrame,
    excluded_profiles: pd.DataFrame,
    output_path: Path = REPORT_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = create_cluster_summary(clustered_profiles)
    sector_status = create_sector_comparison_status(clustered_profiles)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        clustered_profiles.to_excel(
            writer,
            sheet_name="Clustered_Companies",
            index=False,
        )

        evaluation.to_excel(
            writer,
            sheet_name="K_Evaluation",
            index=False,
        )

        summary.to_excel(
            writer,
            sheet_name="Cluster_Summary",
            index=False,
        )

        excluded_profiles.to_excel(
            writer,
            sheet_name="Excluded_Companies",
            index=False,
        )

        sector_status.to_excel(
            writer,
            sheet_name="Sector_Comparison_Status",
            index=False,
        )


def plot_clustering_evaluation(
    evaluation: pd.DataFrame,
    elbow_path: Path = ELBOW_PLOT_PATH,
    silhouette_path: Path = SILHOUETTE_PLOT_PATH,
) -> None:
    elbow_path.parent.mkdir(parents=True, exist_ok=True)
    silhouette_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(
        evaluation["k"],
        evaluation["inertia"],
        marker="o",
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.xticks(evaluation["k"])
    plt.tight_layout()
    plt.savefig(elbow_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        evaluation["k"],
        evaluation["silhouette_score"],
        marker="o",
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score")
    plt.xticks(evaluation["k"])
    plt.tight_layout()
    plt.savefig(silhouette_path, dpi=150)
    plt.close()


def run_clustering(
    profile_path: Path = PROFILE_PATH,
    report_path: Path = REPORT_PATH,
) -> pd.DataFrame:
    profiles = pd.read_parquet(profile_path)

    (
        clustered_profiles,
        evaluation,
        excluded_profiles,
        _,
        model,
    ) = cluster_company_profiles(profiles)

    save_cluster_report(
        clustered_profiles,
        evaluation,
        excluded_profiles,
        report_path,
    )

    plot_clustering_evaluation(evaluation)

    best_k = model.n_clusters

    best_silhouette = evaluation.loc[
        evaluation["k"] == best_k,
        "silhouette_score",
    ].iloc[0]

    print()
    print("=" * 60)
    print("HAPI 6 - COMPANY PROFILE CLUSTERING")
    print("=" * 60)
    print(f"Companies available: {len(profiles):,}")
    print(f"Companies clustered: {len(clustered_profiles):,}")
    print(f"Companies excluded: {len(excluded_profiles):,}")
    print(f"Features used: {', '.join(CLUSTER_FEATURES)}")
    print(f"Evaluated k: {evaluation['k'].min()}-{evaluation['k'].max()}")
    print(f"Best k by Silhouette Score: {best_k}")
    print(f"Best Silhouette Score: {best_silhouette:.4f}")
    print()
    print("Cluster distribution:")

    distribution = clustered_profiles["cluster_id"].value_counts().sort_index()

    for cluster_id, count in distribution.items():
        print(f"  Cluster {cluster_id}: {count} companies")

    print()
    print("Sector comparison: SKIPPED - business_sector is not mapped.")
    print(f"Report saved: {report_path}")
    print(f"Elbow plot saved: {ELBOW_PLOT_PATH}")
    print(f"Silhouette plot saved: {SILHOUETTE_PLOT_PATH}")

    return clustered_profiles


def main() -> None:
    run_clustering()


if __name__ == "__main__":
    main()
