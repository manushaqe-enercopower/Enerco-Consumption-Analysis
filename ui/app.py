from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

QUALITY_REPORT_PATH = ROOT / "reports" / "hapi_1_data_quality.xlsx"

PROFILE_DIR = ROOT / "data" / "processed" / "profile_metrics"
COMPANY_PROFILES_PATH = PROFILE_DIR / "company_profiles.parquet"
METER_PROFILES_PATH = PROFILE_DIR / "meter_profiles.parquet"

OUTLIERS_DIR = ROOT / "data" / "processed" / "outliers"
OUTLIERS_PATH = OUTLIERS_DIR / "company_hourly_outliers.parquet"
OUTLIER_SUMMARY_PATH = OUTLIERS_DIR / "company_outlier_summary.parquet"


st.set_page_config(
    page_title="EnerCo Consumption Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_overview_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    meter_quality = pd.read_excel(
        QUALITY_REPORT_PATH,
        sheet_name="Meter Quality",
    )

    company_profiles = pd.read_parquet(COMPANY_PROFILES_PATH)

    meter_profiles = pd.read_parquet(METER_PROFILES_PATH)

    outliers = pd.read_parquet(OUTLIERS_PATH)

    outlier_summary = pd.read_parquet(OUTLIER_SUMMARY_PATH)

    return (
        meter_quality,
        company_profiles,
        meter_profiles,
        outliers,
        outlier_summary,
    )


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1600px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 12px;
            padding: 14px 16px;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.90rem;
        }

        div[data-testid="stExpander"] {
            border-radius: 10px;
        }

        .analysis-note {
            border-left: 3px solid #2563EB;
            padding: 0.6rem 1rem;
            margin: 0.75rem 0 1.5rem 0;
            background: rgba(37, 99, 235, 0.06);
            border-radius: 0 8px 8px 0;
        }

        .section-caption {
            opacity: 0.75;
            margin-top: -0.5rem;
            margin-bottom: 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def overview_page() -> None:
    st.title("Analiza e Konsumit të Energjisë — EnerCo")

    st.caption(
        "Të dhënat burimore: Qershor 2025 – Qershor 2026 · "
        "Dritarja vjetore e profilit: Korrik 2025 – Qershor 2026"
    )

    (
        meter_quality,
        company_profiles,
        meter_profiles,
        outliers,
        outlier_summary,
    ) = load_overview_data()

    source_companies = meter_quality["company_code"].nunique()
    source_series = len(meter_quality)

    profile_companies = company_profiles["company_code"].nunique()
    usable_meters = len(meter_profiles)

    st.markdown("""
        Kjo ndërfaqe përmbledh analizën e plotë të konsumit të energjisë
        për portofolin e EnerCo, nga kontrolli i cilësisë së të dhënave
        deri te profilet e konsumit, outlier-at, faktorët e jashtëm,
        klasterizimi dhe analiza e prosumerëve.

        Përdorni menunë anësore për të eksploruar secilën pjesë të analizës
        në nivel portofoli, kompanie dhe njehsori.
        """)

    cols = st.columns(4)

    cols[0].metric(
        "Kompanitë në burim",
        f"{source_companies:,}",
    )

    cols[1].metric(
        "Kompanitë në profilin vjetor",
        f"{profile_companies:,}",
    )

    cols[2].metric(
        "Seri/njehsorë në burim",
        f"{source_series:,}",
    )

    cols[3].metric(
        "Njehsorë konsumi të përdorshëm",
        f"{usable_meters:,}",
    )

    st.divider()

    st.subheader("Rezultatet kryesore")

    st.markdown("### Cilësia e të dhënave")

    quality_counts = meter_quality["quality_status"].value_counts()

    clean_meters = int(quality_counts.get("clean", 0))

    review_meters = int(quality_counts.get("review", 0))

    unusable_quality_meters = int(quality_counts.get("unusable", 0))

    quality_col1, quality_col2, quality_col3 = st.columns(3)

    quality_col1.metric(
        "Njehsorë clean",
        f"{clean_meters:,}",
    )

    quality_col2.metric(
        "Për shqyrtim",
        f"{review_meters:,}",
    )

    quality_col3.metric(
        "Të papërdorshëm",
        f"{unusable_quality_meters:,}",
    )

    st.caption(
        "Klasifikimi vjen direkt nga Hapi 1. "
        "Vetëm njehsorët që kalojnë filtrat përkatës përdoren "
        "në analizat e mëtejshme."
    )

    st.divider()

    st.markdown("### Profilet e konsumit")

    seasonality_counts = company_profiles["seasonality"].value_counts()

    winter_companies = int(seasonality_counts.get("winter", 0))

    summer_companies = int(seasonality_counts.get("summer", 0))

    no_seasonality_companies = int(seasonality_counts.get("none", 0))

    profile_col1, profile_col2, profile_col3 = st.columns(3)

    profile_col1.metric(
        "Profil dimëror",
        f"{winter_companies:,}",
    )

    profile_col2.metric(
        "Profil veror",
        f"{summer_companies:,}",
    )

    profile_col3.metric(
        "Pa sezonalitet të fortë",
        f"{no_seasonality_companies:,}",
    )

    st.caption(
        "Klasifikimi i sezonalitetit bazohet në profilin vjetor "
        "të konsumit të secilës kompani."
    )

    st.divider()

    st.markdown("### Outlier-at")

    total_outliers = len(outliers)

    companies_with_outliers = outliers["company_code"].nunique()

    total_hours = int(outlier_summary["total_hours"].sum())

    outlier_rate = total_outliers / total_hours * 100 if total_hours else 0

    max_z = float(outliers["absolute_z_score"].max()) if not outliers.empty else 0

    outlier_col1, outlier_col2, outlier_col3, outlier_col4 = st.columns(4)

    outlier_col1.metric(
        "Orë outlier",
        f"{total_outliers:,}",
    )

    outlier_col2.metric(
        "Kompanitë me outlier-a",
        f"{companies_with_outliers:,}",
    )

    outlier_col3.metric(
        "Pjesa e vëzhgimeve",
        f"{outlier_rate:.2f}%",
    )

    outlier_col4.metric(
        "Maksimumi |Z|",
        f"{max_z:.2f}",
    )

    st.caption(
        "Outlier konsiderohet një vëzhgim orar me |Z| > 3 "
        "kundrejt sjelljes tipike të vetë kompanisë."
    )


pages = {
    "Analiza": [
        st.Page(
            overview_page,
            title="Përmbledhje",
            icon="🏠",
        ),
        st.Page(
            "pages/data_quality.py",
            title="Cilësia e të dhënave",
            icon="✅",
        ),
        st.Page(
            "pages/company_profiles.py",
            title="Profilet e kompanive",
            icon="📈",
        ),
        st.Page(
            "pages/meter_analysis.py",
            title="Analiza e njehsorëve",
            icon="⚡",
        ),
        st.Page(
            "pages/outliers.py",
            title="Outlier-at",
            icon="⚠️",
        ),
        st.Page(
            "pages/clustering.py",
            title="Klasterizimi",
            icon="🧩",
        ),
        st.Page(
            "pages/prosumers.py",
            title="Prosumerët",
            icon="☀️",
        ),
        st.Page(
            "pages/external_factors.py",
            title="Faktorët shtesë",
            icon="🌡️",
        ),
    ],
}


navigation = st.navigation(
    pages,
    expanded=True,
)

navigation.run()
