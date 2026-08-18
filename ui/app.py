from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


st.set_page_config(
    page_title="EnerCo Consumption Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
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

    st.markdown("""
        Kjo ndërfaqe paraqet rezultatet analitike të ndërtuara
        sipas metodologjisë së analizës së profilit të konsumit.

        Përdorni menunë anësore për të eksploruar kontrollin e
        cilësisë, profilet e konsumit, outlier-at, faktorët shtesë
        dhe klasterizimin.
        """)

    cols = st.columns(4)

    cols = st.columns(4)

    cols[0].metric(
        "Kompanitë në burim",
        "81",
    )

    cols[1].metric(
        "Kompanitë në profilin vjetor",
        "66",
    )

    cols[2].metric(
        "Seri/njehsorë në burim",
        "517",
    )

    cols[3].metric(
        "Njehsorë konsumi të përdorshëm",
        "300",
    )

    st.info(
        "Faqet analitike shtohen dhe validohen në mënyrë "
        "progresive sipas hapave të metodologjisë."
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
