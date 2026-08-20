from __future__ import annotations
import plotly.express as px

from pathlib import Path

import pandas as pd
import streamlit as st

UI_DIR = Path(__file__).resolve().parents[1]
ROOT = UI_DIR.parent

CLUSTERING_REPORT_PATH = ROOT / "reports" / "hapi_6_clustering.xlsx"


@st.cache_data(show_spinner=False)
def load_clustering_report(path: str) -> dict[str, pd.DataFrame]:
    return pd.read_excel(
        path,
        sheet_name=[
            "Clustered_Companies",
            "K_Evaluation",
            "Cluster_Summary",
            "Sector_Comparison_Status",
            "Excluded_Companies",
        ],
    )


report = load_clustering_report(str(CLUSTERING_REPORT_PATH))

clustered = report["Clustered_Companies"]
k_evaluation = report["K_Evaluation"]
cluster_summary = report["Cluster_Summary"]
sector_status = report["Sector_Comparison_Status"]
excluded = report["Excluded_Companies"]


best_k_row = k_evaluation.loc[k_evaluation["silhouette_score"].idxmax()]

best_k = int(best_k_row["k"])
best_silhouette = float(best_k_row["silhouette_score"])


st.title("Analiza e klasterizimit")

st.caption(
    "Hapi 6 — grupimi i kompanive sipas karakteristikave të profilit " "të konsumit."
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Kompanitë e klasterizuara",
    f"{len(clustered):,}",
)

col2.metric(
    "Kompanitë e përjashtuara",
    f"{len(excluded):,}",
)

col3.metric(
    "Numri optimal i klasterëve",
    f"{best_k}",
)

col4.metric(
    "Silhouette Score",
    f"{best_silhouette:.3f}",
)


st.info(
    f"Modeli ka zgjedhur **k = {best_k}** si ndarjen më të përshtatshme, "
    f"me Silhouette Score **{best_silhouette:.3f}**."
)
overview_tab, evaluation_tab, profiles_tab, companies_tab, methodology_tab = st.tabs(
    [
        "Përmbledhje",
        "Vlerësimi i k",
        "Profilet e klasterëve",
        "Kompanitë",
        "Metodologjia",
    ]
)
with overview_tab:
    st.subheader("Përmbledhje")

    st.markdown(
        "Klasterizimi grupon kompanitë me sjellje të ngjashme të konsumit "
        "duke përdorur metrikat e profilit vjetor."
    )

    st.caption(
        "Qëllimi analitik: Të identifikohen grupe kompanish me karakteristika "
        "të ngjashme të ngarkesës, sezonalitetit dhe trendit."
    )

    cluster_counts = (
        clustered.groupby("cluster_id")
        .size()
        .reset_index(name="company_count")
        .sort_values("cluster_id")
    )

    cluster_counts["cluster_label"] = "Klasteri " + cluster_counts["cluster_id"].astype(
        str
    )

    col1, col2 = st.columns(2)

    for index, row in cluster_counts.iterrows():
        target_col = col1 if index % 2 == 0 else col2

        target_col.metric(
            row["cluster_label"],
            f"{int(row['company_count'])} kompani",
        )

    st.markdown("### Shpërndarja e kompanive sipas klasterit")

    st.bar_chart(cluster_counts.set_index("cluster_label")["company_count"])

    st.caption("Grafiku paraqet numrin e kompanive të përfshira në secilin klaster.")

with evaluation_tab:
    st.subheader("Vlerësimi i numrit të klasterëve")

    st.markdown(
        "Numri optimal i klasterëve vlerësohet duke krahasuar "
        "**Silhouette Score** dhe **inertia** për vlera të ndryshme të `k`."
    )

    st.caption(
        "Qëllimi analitik: Të zgjidhet një numër klasterësh që krijon "
        "grupe sa më të dallueshme dhe koherente."
    )

    st.info(
        f"Vlera më e lartë e Silhouette Score është "
        f"**{best_silhouette:.3f}** dhe arrihet me **k = {best_k}**."
    )

    st.markdown("### Silhouette Score")

    fig_silhouette = px.line(
        k_evaluation,
        x="k",
        y="silhouette_score",
        markers=True,
    )

    fig_silhouette.update_traces(
        hovertemplate=("k = %{x}<br>" "Silhouette Score: %{y:.3f}" "<extra></extra>")
    )

    fig_silhouette.add_vline(
        x=best_k,
        line_dash="dash",
        annotation_text=f"k optimal = {best_k}",
        annotation_position="top",
    )

    fig_silhouette.update_layout(
        title="Silhouette Score sipas numrit të klasterëve",
        xaxis_title="Numri i klasterëve (k)",
        yaxis_title="Silhouette Score",
        height=500,
    )

    fig_silhouette.update_xaxes(
        tickmode="array",
        tickvals=k_evaluation["k"],
    )

    st.plotly_chart(
        fig_silhouette,
        width="stretch",
    )

    st.caption(
        "Vlera më e lartë tregon ndarje më të mirë ndërmjet klasterëve. "
        "Në këtë analizë maksimumi arrihet me k = 2."
    )

    st.divider()

    st.markdown("### Elbow / Inertia")

    fig_inertia = px.line(
        k_evaluation,
        x="k",
        y="inertia",
        markers=True,
    )

    fig_inertia.update_traces(
        hovertemplate=("k = %{x}<br>" "Inertia: %{y:.2f}" "<extra></extra>")
    )

    fig_inertia.add_vline(
        x=best_k,
        line_dash="dash",
        annotation_text=f"k = {best_k}",
        annotation_position="top",
    )

    fig_inertia.update_layout(
        title="Inertia sipas numrit të klasterëve",
        xaxis_title="Numri i klasterëve (k)",
        yaxis_title="Inertia",
        height=500,
    )

    fig_inertia.update_xaxes(
        tickmode="array",
        tickvals=k_evaluation["k"],
    )

    st.plotly_chart(
        fig_inertia,
        width="stretch",
    )

    st.caption(
        "Inertia zvogëlohet me rritjen e numrit të klasterëve. "
        "Ajo përdoret si kontroll shtesë krahas Silhouette Score, "
        "jo si kriter i vetëm për zgjedhjen e k."
    )

with profiles_tab:
    st.subheader("Profilet e klasterëve")

    st.markdown(
        "Kjo pamje krahason karakteristikat mesatare të kompanive "
        "brenda secilit klaster."
    )

    st.caption(
        "Qëllimi analitik: Të kuptohet se çfarë e dallon secilin grup "
        "nga grupet e tjera sipas profilit të konsumit."
    )

    metric_labels = {
        "peak_ratio": "Raporti Peak / Off-Peak",
        "weekday_weekend_ratio": "Raporti Ditë Pune / Fundjavë",
        "cv": "Koeficienti i variacionit (CV)",
        "load_factor": "Load Factor",
        "seasonality_index": "Indeksi i sezonalitetit",
        "trend_percent": "Trendi (%)",
    }

    profile_table = cluster_summary[
        [
            "cluster_id",
            "company_count",
            "peak_ratio",
            "weekday_weekend_ratio",
            "cv",
            "load_factor",
            "seasonality_index",
            "trend_percent",
        ]
    ].copy()

    profile_table["cluster_id"] = "Klasteri " + profile_table["cluster_id"].astype(str)

    display_profile = profile_table.rename(
        columns={
            "cluster_id": "Klasteri",
            "company_count": "Kompanitë",
            "peak_ratio": "Peak / Off-Peak",
            "weekday_weekend_ratio": "Ditë Pune / Fundjavë",
            "cv": "CV",
            "load_factor": "Load Factor",
            "seasonality_index": "Sezonaliteti",
            "trend_percent": "Trendi (%)",
        }
    )

    st.dataframe(
        display_profile,
        width="stretch",
        hide_index=True,
        column_config={
            "Peak / Off-Peak": st.column_config.NumberColumn(
                "Peak / Off-Peak",
                format="%.2f",
            ),
            "Ditë Pune / Fundjavë": st.column_config.NumberColumn(
                "Ditë Pune / Fundjavë",
                format="%.2f",
            ),
            "CV": st.column_config.NumberColumn(
                "CV",
                format="%.2f",
            ),
            "Load Factor": st.column_config.NumberColumn(
                "Load Factor",
                format="%.2f",
            ),
            "Sezonaliteti": st.column_config.NumberColumn(
                "Sezonaliteti",
                format="%.2f",
            ),
            "Trendi (%)": st.column_config.NumberColumn(
                "Trendi (%)",
                format="%.2f%%",
            ),
        },
    )

    st.divider()

    selected_metric = st.selectbox(
        "Zgjidh metrikën për krahasim",
        options=list(metric_labels.keys()),
        format_func=lambda value: metric_labels[value],
        key="cluster-profile-metric",
    )

    comparison_df = cluster_summary[
        [
            "cluster_id",
            selected_metric,
        ]
    ].copy()

    comparison_df["cluster_label"] = "Klasteri " + comparison_df["cluster_id"].astype(
        str
    )

    fig_profile = px.bar(
        comparison_df,
        x="cluster_label",
        y=selected_metric,
        text=selected_metric,
    )

    fig_profile.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{metric_labels[selected_metric]}: "
            "%{y:.2f}"
            "<extra></extra>"
        ),
    )

    fig_profile.update_layout(
        title=f"Krahasimi — {metric_labels[selected_metric]}",
        xaxis_title="Klasteri",
        yaxis_title=metric_labels[selected_metric],
        height=500,
    )

    st.plotly_chart(
        fig_profile,
        width="stretch",
    )

    st.caption(
        "Vlerat paraqesin mesataren e kompanive brenda secilit klaster "
        "për metrikën e zgjedhur."
    )

with companies_tab:
    st.subheader("Kompanitë sipas klasterit")

    st.markdown(
        "Kjo tabelë paraqet kompanitë e përfshira në analizë, "
        "klasterin e caktuar dhe metrikat kryesore të profilit."
    )

    st.caption(
        "Qëllimi analitik: Të mundësohet kontrolli i përbërjes së secilit "
        "klaster dhe identifikimi i kompanive që kanë profile të ngjashme."
    )

    company_view = clustered.copy()

    company_view["cluster_label"] = "Klasteri " + company_view["cluster_id"].astype(str)

    company_view["company_number"] = (
        company_view["company_code"].str.extract(r"(\d+)").astype(int)
    )

    cluster_options = [
        "Të gjithë",
        *[
            f"Klasteri {cluster_id}"
            for cluster_id in sorted(company_view["cluster_id"].unique())
        ],
    ]

    selected_cluster = st.selectbox(
        "Filtro sipas klasterit",
        options=cluster_options,
        key="cluster-company-filter",
    )

    filtered_companies = company_view.copy()

    if selected_cluster != "Të gjithë":
        filtered_companies = filtered_companies[
            filtered_companies["cluster_label"] == selected_cluster
        ]

    filtered_companies = filtered_companies.sort_values("company_number")

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Kompanitë e shfaqura",
        f"{len(filtered_companies):,}",
    )

    if selected_cluster == "Të gjithë":
        metric_col2.metric(
            "Klasterë të përfshirë",
            f"{filtered_companies['cluster_id'].nunique()}",
        )
    else:
        metric_col2.metric(
            "Klasteri",
            selected_cluster.replace("Klasteri ", ""),
        )

    display_companies = filtered_companies[
        [
            "company_code",
            "cluster_label",
            "meter_count_used",
            "peak_ratio",
            "weekday_weekend_ratio",
            "cv",
            "load_factor",
            "seasonality",
            "seasonality_index",
            "trend_percent",
        ]
    ].rename(
        columns={
            "company_code": "Kompania",
            "cluster_label": "Klasteri",
            "meter_count_used": "Njehsorë",
            "peak_ratio": "Peak / Off-Peak",
            "weekday_weekend_ratio": "Ditë Pune / Fundjavë",
            "cv": "CV",
            "load_factor": "Load Factor",
            "seasonality": "Sezonaliteti",
            "seasonality_index": "Indeksi sezonal",
            "trend_percent": "Trendi (%)",
        }
    )

    st.dataframe(
        display_companies,
        width="stretch",
        hide_index=True,
        column_config={
            "Peak / Off-Peak": st.column_config.NumberColumn(
                "Peak / Off-Peak",
                format="%.2f",
            ),
            "Ditë Pune / Fundjavë": st.column_config.NumberColumn(
                "Ditë Pune / Fundjavë",
                format="%.2f",
            ),
            "CV": st.column_config.NumberColumn(
                "CV",
                format="%.2f",
            ),
            "Load Factor": st.column_config.NumberColumn(
                "Load Factor",
                format="%.2f",
            ),
            "Indeksi sezonal": st.column_config.NumberColumn(
                "Indeksi sezonal",
                format="%.2f",
            ),
            "Trendi (%)": st.column_config.NumberColumn(
                "Trendi (%)",
                format="%.2f%%",
            ),
        },
    )

    st.divider()

    st.subheader("Kompanitë e përjashtuara")

    if excluded.empty:
        st.success("Nuk ka kompani të përjashtuara nga klasterizimi.")
    else:
        st.warning(
            f"{len(excluded)} kompani nuk është përfshirë në klasterizim "
            "sepse nuk kishte metrika të mjaftueshme të vlefshme."
        )

        excluded_view = excluded[
            [
                "company_code",
                "observation_count",
                "total_kwh",
                "meter_count_used",
            ]
        ].rename(
            columns={
                "company_code": "Kompania",
                "observation_count": "Vëzhgime",
                "total_kwh": "Konsumi total (kWh)",
                "meter_count_used": "Njehsorë",
            }
        )

        st.dataframe(
            excluded_view,
            width="stretch",
            hide_index=True,
            column_config={
                "Konsumi total (kWh)": st.column_config.NumberColumn(
                    "Konsumi total (kWh)",
                    format="%.2f",
                ),
            },
        )

with methodology_tab:
    st.subheader("Metodologjia")

    st.markdown(
        "Hapi 6 përdor **K-Means clustering** për të grupuar kompanitë "
        "sipas karakteristikave të profilit të tyre vjetor të konsumit."
    )

    st.markdown("### Metrikat e përdorura")

    methodology_features = pd.DataFrame(
        {
            "Metrika": [
                "Peak / Off-Peak",
                "Ditë Pune / Fundjavë",
                "CV",
                "Load Factor",
                "Indeksi i sezonalitetit",
                "Trendi (%)",
            ],
            "Kolona": [
                "peak_ratio",
                "weekday_weekend_ratio",
                "cv",
                "load_factor",
                "seasonality_index",
                "trend_percent",
            ],
            "Qëllimi": [
                "Mat dallimin ndërmjet konsumit në peak dhe off-peak.",
                "Mat dallimin ndërmjet ditëve të punës dhe fundjavës.",
                "Mat variabilitetin e konsumit.",
                "Mat sa rregullisht përdoret kapaciteti maksimal.",
                "Mat fuqinë e sezonalitetit të profilit.",
                "Mat ndryshimin e konsumit gjatë periudhës.",
            ],
        }
    )

    st.dataframe(
        methodology_features,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.markdown("### Përgatitja e të dhënave")

    st.markdown("""
Përpara klasterizimit:

1. Të gjashtë metrikat konvertohen në vlera numerike.
2. Vlerat `inf` dhe `-inf` trajtohen si të pavlefshme.
3. Kompanitë që kanë të paktën një metrikë të munguar përjashtohen.
4. Metrikat standardizohen me **StandardScaler**.
""")

    st.info(
        "Standardizimi është i nevojshëm sepse metrikat kanë shkallë "
        "të ndryshme. Pas standardizimit, secila metrikë kontribuon "
        "në mënyrë më të krahasueshme në distancën e përdorur nga K-Means."
    )

    st.divider()

    st.markdown("### Zgjedhja e numrit të klasterëve")

    st.markdown("""
Pipeline-i teston vlerat:

**k = 2 deri në 10**

Për secilën vlerë të `k`:

- trajnohet një model **KMeans**;
- llogaritet **inertia**;
- llogaritet **Silhouette Score**.

Numri final i klasterëve zgjidhet sipas **Silhouette Score më të lartë**.
Në rast barazimi preferohet vlera më e vogël e `k`.
""")

    st.code("KMeans(n_clusters=k, random_state=42, n_init=10)")

    st.success(
        f"Në dataset-in aktual, rezultati më i mirë është "
        f"k = {best_k}, me Silhouette Score {best_silhouette:.3f}."
    )

    st.divider()

    st.markdown("### Rezultati aktual")

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Kompanitë e klasterizuara",
        f"{len(clustered):,}",
    )

    result_col2.metric(
        "Kompanitë e përjashtuara",
        f"{len(excluded):,}",
    )

    result_col3.metric(
        "Klasterë",
        f"{best_k}",
    )

    st.markdown("""
ID-të **Klasteri 0**, **Klasteri 1**, etj. janë etiketa teknike të
gjeneruara nga modeli. Numri i klasterit nuk nënkupton renditje,
performancë më të mirë ose më të dobët.
""")

    st.divider()

    st.markdown("### Krahasimi me sektorët")

    if (
        not sector_status.empty
        and str(sector_status.iloc[0]["status"]).upper() == "SKIPPED"
    ):
        st.warning(
            "Krahasimi ndërmjet klasterëve dhe sektorëve të biznesit "
            "nuk është realizuar sepse metadata `business_sector` "
            "nuk është e mapuar në dataset-in aktual."
        )
    else:
        st.success("Metadata e sektorëve është e disponueshme për krahasim.")

    st.caption(
        "Nuk janë bërë supozime për sektorin e kompanive pa të dhëna "
        "burimore të verifikuara."
    )
