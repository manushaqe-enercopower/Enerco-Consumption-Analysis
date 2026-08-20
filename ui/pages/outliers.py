from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

UI_DIR = Path(__file__).resolve().parents[1]
ROOT = UI_DIR.parent

OUTLIERS_DIR = ROOT / "data" / "processed" / "outliers"

COMPANY_HOURLY_PATH = OUTLIERS_DIR / "company_hourly.parquet"
OUTLIERS_PATH = OUTLIERS_DIR / "company_hourly_outliers.parquet"
SUMMARY_PATH = OUTLIERS_DIR / "company_outlier_summary.parquet"


@st.cache_data(show_spinner=False)
def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


company_hourly = read_parquet(str(COMPANY_HOURLY_PATH))
outliers = read_parquet(str(OUTLIERS_PATH))
company_summary = read_parquet(str(SUMMARY_PATH))


st.title("Analiza e outlier-ave")

st.caption(
    "Identifikimi i konsumit orar jo të zakonshëm "
    "në nivel kompanie duke përdorur Z-score."
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Kompanitë e analizuara",
    f"{company_summary['company_code'].nunique():,}",
)

col2.metric(
    "Orë të analizuara",
    f"{len(company_hourly):,}",
)

col3.metric(
    "Outlier-a të identifikuar",
    f"{len(outliers):,}",
)

col4.metric(
    "Kompanitë me outlier-a",
    f"{outliers['company_code'].nunique():,}",
)


overview_tab, company_tab, time_tab, severity_tab, table_tab, methodology_tab = st.tabs(
    [
        "Përmbledhje",
        "Analiza sipas kompanisë",
        "Shpërndarja kohore",
        "Ashpërsia",
        "Tabela e outlier-ave",
        "Metodologjia",
    ]
)


with overview_tab:
    st.subheader("Përmbledhje")

    total_observations = int(company_summary["total_hours"].sum())
    total_outliers = int(len(outliers))

    outlier_rate = (
        total_outliers / total_observations * 100 if total_observations else 0
    )

    max_z = float(outliers["absolute_z_score"].max()) if not outliers.empty else 0

    st.markdown(
        "Outlier konsiderohet çdo orë ku konsumi i kompanisë devijon "
        "më shumë se **3 devijime standarde** nga mesatarja e saj."
    )

    st.caption(
        "Qëllimi analitik: Të identifikohen kompanitë dhe periudhat "
        "që mund të kërkojnë verifikim teknik ose operacional."
    )

    st.info(
        f"Janë identifikuar {total_outliers:,} orë outlier, "
        f"që përfaqësojnë {outlier_rate:.2f}% të "
        f"{total_observations:,} vëzhgimeve të analizuara."
    )

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Pjesa e vëzhgimeve outlier",
        f"{outlier_rate:.2f}%",
    )

    metric_col2.metric(
        "Devijimi maksimal |Z|",
        f"{max_z:.2f}",
    )

    top_companies = (
        company_summary[company_summary["outlier_hours"] > 0]
        .sort_values(
            "outlier_hours",
            ascending=False,
        )
        .head(20)
    )

    fig = px.bar(
        top_companies,
        x="company_code",
        y="outlier_hours",
        custom_data=[
            "outlier_percent",
            "max_abs_z_score",
        ],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Orë outlier: %{y:,.0f}<br>"
            "Përqindja: %{customdata[0]:.2f}%<br>"
            "Maks. |Z|: %{customdata[1]:.2f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        title="20 kompanitë me më së shumti orë outlier",
        xaxis_title="Kompania",
        yaxis_title="Orë outlier",
        height=500,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.caption(
        "Grafiku rendit kompanitë sipas numrit absolut të orëve "
        "të identifikuara si outlier."
    )


with company_tab:
    st.subheader("Analiza sipas kompanisë")

    companies = sorted(
        company_summary["company_code"].astype(str).unique(),
        key=lambda value: int(value.split()[-1]),
    )

    selected_company = st.selectbox(
        "Zgjidh kompaninë",
        companies,
        key="outlier-company-selector",
    )

    selected_summary = company_summary[
        company_summary["company_code"].astype(str) == selected_company
    ].iloc[0]

    selected_hourly = company_hourly[
        company_hourly["company_code"].astype(str) == selected_company
    ].copy()

    selected_outliers = outliers[
        outliers["company_code"].astype(str) == selected_company
    ].copy()

    selected_hourly["timestamp"] = pd.to_datetime(selected_hourly["timestamp"])

    selected_outliers["timestamp"] = pd.to_datetime(selected_outliers["timestamp"])

    st.markdown(
        "Kjo analizë paraqet konsumin orar të kompanisë së zgjedhur "
        "dhe momentet që janë identifikuar si outlier."
    )

    st.caption(
        "Qëllimi analitik: Të shihet nëse devijimet janë incidente "
        "të izoluara apo përsëriten në periudha të caktuara."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Orë të analizuara",
        f"{int(selected_summary['total_hours']):,}",
    )

    col2.metric(
        "Orë outlier",
        f"{int(selected_summary['outlier_hours']):,}",
    )

    col3.metric(
        "Pjesa outlier",
        f"{selected_summary['outlier_percent']:.2f}%",
    )

    max_company_z = selected_summary["max_abs_z_score"]

    col4.metric(
        "Maksimumi |Z|",
        (f"{max_company_z:.2f}" if pd.notna(max_company_z) else "0.00"),
    )

    fig = px.line(
        selected_hourly.sort_values("timestamp"),
        x="timestamp",
        y="energy_kwh",
    )

    fig.update_traces(
        name="Konsumi orar",
        hovertemplate=(
            "%{x|%d.%m.%Y %H:%M}<br>" "Konsumi: %{y:,.2f} kWh" "<extra></extra>"
        ),
    )

    if not selected_outliers.empty:
        fig.add_scatter(
            x=selected_outliers["timestamp"],
            y=selected_outliers["energy_kwh"],
            mode="markers",
            name="Outlier",
            marker={
                "size": 8,
                "symbol": "circle-open",
            },
            customdata=selected_outliers[
                [
                    "absolute_z_score",
                    "z_score",
                ]
            ],
            hovertemplate=(
                "%{x|%d.%m.%Y %H:%M}<br>"
                "Konsumi: %{y:,.2f} kWh<br>"
                "|Z|: %{customdata[0]:.2f}<br>"
                "Z-score: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )

    fig.update_layout(
        title=f"Konsumi orar dhe outlier-at — {selected_company}",
        xaxis_title="Koha",
        yaxis_title="Konsumi (kWh)",
        height=550,
        hovermode="closest",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    if selected_outliers.empty:
        st.info(
            "Për këtë kompani nuk janë identifikuar outlier-a " "me pragun |Z| > 3."
        )
    else:
        st.caption(
            "Pikat e shënuara mbi vijën e konsumit paraqesin "
            "orët që kanë kaluar pragun |Z| > 3."
        )


with time_tab:
    st.subheader("Shpërndarja kohore")

    st.markdown(
        "Kjo analizë tregon se si shpërndahen outlier-at "
        "gjatë periudhës së analizuar."
    )

    st.caption(
        "Qëllimi analitik: Të identifikohen muajt dhe orët ku "
        "devijimet e pazakonta të konsumit paraqiten më shpesh."
    )

    time_outliers = outliers.copy()

    time_outliers["timestamp"] = pd.to_datetime(time_outliers["timestamp"])

    time_outliers["month"] = time_outliers["timestamp"].dt.to_period("M").astype(str)

    monthly_outliers = (
        time_outliers.groupby("month")
        .size()
        .reset_index(name="outlier_hours")
        .sort_values("month")
    )

    fig_month = px.bar(
        monthly_outliers,
        x="month",
        y="outlier_hours",
        text="outlier_hours",
    )

    fig_month.update_traces(
        textposition="outside",
        hovertemplate=("<b>%{x}</b><br>" "Orë outlier: %{y:,.0f}" "<extra></extra>"),
    )

    fig_month.update_layout(
        title="Outlier-at sipas muajit",
        xaxis_title="Muaji",
        yaxis_title="Orë outlier",
        height=500,
    )

    fig_month.update_xaxes(
        tickmode="array",
        tickvals=monthly_outliers["month"],
        ticktext=monthly_outliers["month"],
    )

    st.plotly_chart(
        fig_month,
        width="stretch",
    )

    st.caption(
        "Vlerat paraqesin numrin total të orëve të identifikuara "
        "si outlier për të gjitha kompanitë në secilin muaj."
    )

    st.divider()

    st.subheader("Shpërndarja sipas orës së ditës")

    st.markdown(
        "Grafiku tregon në cilat orë të ditës janë identifikuar "
        "më shpesh devijimet e pazakonta të konsumit."
    )

    time_outliers["hour"] = time_outliers["timestamp"].dt.hour

    hourly_outliers = (
        time_outliers.groupby("hour")
        .size()
        .reset_index(name="outlier_hours")
        .sort_values("hour")
    )

    fig_hour = px.bar(
        hourly_outliers,
        x="hour",
        y="outlier_hours",
        text="outlier_hours",
    )

    fig_hour.update_traces(
        textposition="outside",
        hovertemplate=("Ora %{x}:00<br>" "Orë outlier: %{y:,.0f}" "<extra></extra>"),
    )

    fig_hour.update_layout(
        title="Outlier-at sipas orës së ditës",
        xaxis_title="Ora",
        yaxis_title="Orë outlier",
        height=500,
    )

    fig_hour.update_xaxes(
        tickmode="array",
        tickvals=list(range(24)),
        ticktext=[f"{hour:02d}:00" for hour in range(24)],
    )

    st.plotly_chart(
        fig_hour,
        width="stretch",
    )

    st.caption(
        "Ky profil ndihmon të identifikohen orët ku devijimet "
        "shfaqen më shpesh në të gjitha kompanitë."
    )


with severity_tab:
    st.subheader("Ashpërsia e outlier-ave")

    st.markdown(
        "Outlier-at ndahen sipas madhësisë së devijimit absolut "
        "**|Z|** për të dalluar rastet më ekstreme."
    )

    st.caption(
        "Qëllimi analitik: Të prioritizohen devijimet që kanë "
        "largësinë më të madhe nga sjellja normale e konsumit."
    )

    severity_outliers = outliers.copy()

    severity_order = [
        "I moderuar (3–4)",
        "I lartë (4–6)",
        "Shumë i lartë (≥6)",
    ]

    severity_outliers["severity"] = pd.cut(
        severity_outliers["absolute_z_score"],
        bins=[3, 4, 6, float("inf")],
        labels=severity_order,
        right=False,
    )

    severity_counts = (
        severity_outliers.groupby(
            "severity",
            observed=False,
        )
        .size()
        .reindex(
            severity_order,
            fill_value=0,
        )
        .rename("outlier_hours")
        .reset_index()
    )

    severity_counts["percent"] = (
        severity_counts["outlier_hours"] / severity_counts["outlier_hours"].sum() * 100
    )

    moderate_count = int(
        severity_counts.loc[
            severity_counts["severity"] == "I moderuar (3–4)",
            "outlier_hours",
        ].iloc[0]
    )

    high_count = int(
        severity_counts.loc[
            severity_counts["severity"] == "I lartë (4–6)",
            "outlier_hours",
        ].iloc[0]
    )

    extreme_count = int(
        severity_counts.loc[
            severity_counts["severity"] == "Shumë i lartë (≥6)",
            "outlier_hours",
        ].iloc[0]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Të moderuar |Z| 3–4",
        f"{moderate_count:,}",
    )

    col2.metric(
        "Të lartë |Z| 4–6",
        f"{high_count:,}",
    )

    col3.metric(
        "Shumë të lartë |Z| ≥ 6",
        f"{extreme_count:,}",
    )

    fig_severity = px.bar(
        severity_counts,
        x="severity",
        y="outlier_hours",
        text="outlier_hours",
        custom_data=["percent"],
    )

    fig_severity.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Orë outlier: %{y:,.0f}<br>"
            "Pjesa: %{customdata[0]:.2f}%"
            "<extra></extra>"
        ),
    )

    fig_severity.update_layout(
        title="Shpërndarja sipas nivelit të devijimit",
        xaxis_title="Niveli i devijimit",
        yaxis_title="Orë outlier",
        height=500,
    )

    st.plotly_chart(
        fig_severity,
        width="stretch",
    )

    st.caption(
        "Sa më e lartë vlera absolute e Z-score, aq më larg është "
        "vëzhgimi nga sjellja tipike e konsumit të kompanisë."
    )


with table_tab:
    st.subheader("Tabela e outlier-ave")

    st.markdown(
        "Tabela paraqet të gjitha orët e identifikuara si outlier "
        "dhe mundëson filtrimin sipas kompanisë dhe nivelit të devijimit."
    )

    table_outliers = outliers.copy()

    table_outliers["timestamp"] = pd.to_datetime(table_outliers["timestamp"])

    severity_order = [
        "I moderuar (3–4)",
        "I lartë (4–6)",
        "Shumë i lartë (≥6)",
    ]

    table_outliers["severity"] = pd.cut(
        table_outliers["absolute_z_score"],
        bins=[3, 4, 6, float("inf")],
        labels=severity_order,
        right=False,
    )

    companies = sorted(
        table_outliers["company_code"].astype(str).unique(),
        key=lambda value: int(value.split()[-1]),
    )

    filter_col1, filter_col2 = st.columns(2)

    selected_companies = filter_col1.multiselect(
        "Filtro sipas kompanisë",
        options=companies,
        default=[],
        key="outlier-table-company-filter",
    )

    selected_severity = filter_col2.multiselect(
        "Filtro sipas nivelit",
        options=severity_order,
        default=[],
        key="outlier-table-severity-filter",
    )

    filtered_outliers = table_outliers.copy()

    if selected_companies:
        filtered_outliers = filtered_outliers[
            filtered_outliers["company_code"].astype(str).isin(selected_companies)
        ]

    if selected_severity:
        filtered_outliers = filtered_outliers[
            filtered_outliers["severity"].isin(selected_severity)
        ]

    filtered_outliers = filtered_outliers.sort_values(
        ["absolute_z_score", "timestamp"],
        ascending=[False, True],
    )

    st.metric(
        "Rreshta të shfaqur",
        f"{len(filtered_outliers):,}",
    )

    display_columns = [
        "company_code",
        "timestamp",
        "energy_kwh",
        "z_score",
        "absolute_z_score",
        "severity",
    ]

    display_df = filtered_outliers[display_columns].rename(
        columns={
            "company_code": "Kompania",
            "timestamp": "Koha",
            "energy_kwh": "Konsumi (kWh)",
            "z_score": "Z-score",
            "absolute_z_score": "|Z|",
            "severity": "Ashpërsia",
        }
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Koha": st.column_config.DatetimeColumn(
                "Koha",
                format="DD.MM.YYYY HH:mm",
            ),
            "Konsumi (kWh)": st.column_config.NumberColumn(
                "Konsumi (kWh)",
                format="%.2f",
            ),
            "Z-score": st.column_config.NumberColumn(
                "Z-score",
                format="%.2f",
            ),
            "|Z|": st.column_config.NumberColumn(
                "|Z|",
                format="%.2f",
            ),
        },
    )


with methodology_tab:
    st.subheader("Metodologjia")

    st.markdown(
        "Analiza e outlier-ave është ndërtuar mbi të dhënat e përpunuara "
        "nga hapat paraprakë të pipeline-it."
    )

    st.markdown("### Outlier-at orarë")

    st.markdown("""
Procesi i analizës është:

1. Përdoren vetëm të dhënat e **konsumit** (`flow_type = consumption`).
2. Analiza mbulon periudhën **Korrik 2025 – Qershor 2026**.
3. Njehsorët e klasifikuar si **unusable** në Hapin 1 përjashtohen.
4. Konsumi i njehsorëve agregohet në nivel **kompani + orë**.
5. Për secilën kompani llogariten:
   - konsumi mesatar orar;
   - devijimi standard;
   - Z-score për secilën orë.
6. Një vëzhgim klasifikohet si outlier kur **|Z| > 3**.
""")

    st.markdown("#### Formula")

    st.code(
        "Z = (konsumi_orar - mesatarja_e_kompanisë) / " "devijimi_standard_i_kompanisë"
    )

    st.info(
        "Z-score llogaritet veçmas për secilën kompani. "
        "Prandaj devijimi matet kundrejt sjelljes tipike të vetë kompanisë, "
        "jo kundrejt një mesatareje të përbashkët të portofolit."
    )

    st.divider()

    st.markdown("### Klasifikimi i ashpërsisë")

    severity_methodology = pd.DataFrame(
        {
            "Niveli": [
                "I moderuar",
                "I lartë",
                "Shumë i lartë",
            ],
            "Pragu": [
                "3 < |Z| < 4",
                "4 ≤ |Z| < 6",
                "|Z| ≥ 6",
            ],
            "Interpretimi": [
                "Devijim i dukshëm nga sjellja tipike",
                "Devijim i fortë që kërkon verifikim",
                "Devijim ekstrem me prioritet të lartë",
            ],
        }
    )

    st.dataframe(
        severity_methodology,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.markdown("### Krahasimi sipas sektorit")

    st.warning(
        "Hapi 4.2 nuk është ekzekutuar sepse mapping-u i kompanive "
        "me sektorët e biznesit nuk është i disponueshëm. "
        "Nuk janë bërë supozime për sektorin e kompanive."
    )

    st.caption(
        "Ky kufizim ruan integritetin e analizës dhe shmang klasifikimin "
        "e kompanive në sektorë pa të dhëna burimore të verifikuara."
    )

    st.divider()

    st.markdown("### Raporti final")

    st.markdown(
        "Për secilin outlier të vlefshëm, pipeline-i final ruan "
        "nivelin e ashpërsisë, arsyen e identifikimit dhe rekomandimin "
        "për trajtim."
    )

    st.markdown("""
Rekomandimet e backend-it ndahen në:

- **Për shqyrtim teknik** — rastet më ekstreme ose kompanitë me përqindje të lartë outlier-ash.
- **Për verifikim operacional** — devijime të larta që kërkojnë kontroll shtesë.
- **Monitorim** — devijime më të moderuara që mund të përfaqësojnë sjellje legjitime të biznesit.
""")

    st.caption(
        "Outlier nuk nënkupton automatikisht gabim në matje. "
        "Ai identifikon një vëzhgim statistikisht të pazakontë që duhet "
        "interpretuar në kontekst teknik ose operacional."
    )
