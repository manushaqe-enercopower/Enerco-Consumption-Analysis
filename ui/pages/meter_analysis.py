from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

UI_DIR = Path(__file__).resolve().parents[1]
ROOT = UI_DIR.parent

if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))


from components.charting import (
    PLOTLY_CONFIG,
    company_sort_number,
    format_energy,
    sort_by_company,
    style_figure,
)

PROFILE_DIR = ROOT / "data" / "processed" / "profile_metrics"

METER_PROFILES_PATH = PROFILE_DIR / "meter_profiles.parquet"
METER_MONTHLY_PATH = PROFILE_DIR / "meter_monthly.parquet"
METER_HOURLY_PATH = PROFILE_DIR / "meter_hourly_profile.parquet"


SEASONALITY_LABELS = {
    "winter": "Dimër",
    "summer": "Verë",
    "none": "Pa sezonalitet të qartë",
}

SIMILARITY_LABELS = {
    "similar": "I ngjashëm me njehsorët tjerë",
    "different": "Profil i ndryshëm",
    "single_meter": "Kompani me një njehsor",
}

SIMILARITY_SHORT_LABELS = {
    "similar": "I ngjashëm",
    "different": "I ndryshëm",
    "single_meter": "Njehsor i vetëm",
}


@st.cache_data(show_spinner=False)
def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def render_plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        theme="streamlit",
        config=PLOTLY_CONFIG,
        key=key,
    )


def analysis_description(
    title: str,
    description: str,
    purpose: str,
    finding: str | None = None,
) -> None:
    st.markdown(f"### {title}")
    st.markdown(description)
    st.caption(f"Qëllimi analitik: {purpose}")

    if finding:
        st.info(finding)


def value_card(
    label: str,
    value: str,
    help_text: str | None = None,
) -> None:
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"**{value}**")

        if help_text:
            st.caption(help_text)


def format_ratio(value: float) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{value:.2f}"


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{value:.2f}%"


def format_decimal_percent(value: float) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.1f}%"


def format_kwh(value: float) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f} kWh"


def seasonality_label(value: str) -> str:
    return SEASONALITY_LABELS.get(str(value), str(value))


def similarity_label(value: str) -> str:
    return SIMILARITY_LABELS.get(str(value), str(value))


def similarity_short_label(value: str) -> str:
    return SIMILARITY_SHORT_LABELS.get(str(value), str(value))


def selected_meter_mask(
    df: pd.DataFrame,
    selected: pd.Series,
) -> pd.Series:
    mask = (
        (df["company_code"] == selected["company_code"])
        & (df["meter_id"] == selected["meter_id"])
        & (df["source_sheet"] == selected["source_sheet"])
    )

    if "source_column" in df.columns:
        mask &= df["source_column"] == selected["source_column"]

    return mask


def meter_display_name(row: pd.Series) -> str:
    meter_id = str(row["meter_id"])
    source_column = str(row["source_column"])

    if source_column and source_column != meter_id:
        return f"{meter_id} · {source_column}"

    return meter_id


def peak_finding(row: pd.Series) -> str:
    ratio = row["peak_ratio"]

    if pd.isna(ratio):
        return "Raporti Peak/Off-Peak nuk mund të llogaritet për këtë njehsor."

    if ratio > 1:
        difference = (ratio - 1) * 100
        return (
            f"Ky njehsor konsumon mesatarisht {difference:.1f}% më shumë "
            "gjatë orëve Peak sesa gjatë orëve Off-Peak."
        )

    if ratio < 1:
        difference = (1 - ratio) * 100
        return (
            f"Ky njehsor konsumon mesatarisht {difference:.1f}% më pak "
            "gjatë orëve Peak sesa gjatë orëve Off-Peak."
        )

    return "Konsumi Peak dhe Off-Peak është pothuajse i barabartë."


def weekday_finding(row: pd.Series) -> str:
    ratio = row["weekday_weekend_ratio"]

    if pd.isna(ratio):
        return "Raporti Ditë Pune/Fundjavë nuk mund të llogaritet për këtë njehsor."

    if ratio > 1:
        difference = (ratio - 1) * 100
        return (
            f"Konsumi gjatë ditëve të punës është mesatarisht "
            f"{difference:.1f}% më i lartë se gjatë fundjavës."
        )

    if ratio < 1:
        difference = (1 - ratio) * 100
        return (
            f"Konsumi gjatë ditëve të punës është mesatarisht "
            f"{difference:.1f}% më i ulët se gjatë fundjavës."
        )

    return "Nuk ka dallim material ndërmjet ditëve të punës dhe fundjavës."


def trend_finding(row: pd.Series) -> str:
    trend = row["trend_percent"]

    if pd.isna(trend):
        return "Trendi nuk mund të vlerësohet për këtë njehsor."

    if trend > 0:
        return (
            f"Muaji i fundit ka konsum mesatar {trend:.1f}% më të lartë "
            "se mesatarja e tre muajve të parë aktivë."
        )

    if trend < 0:
        return (
            f"Muaji i fundit ka konsum mesatar {abs(trend):.1f}% më të ulët "
            "se mesatarja e tre muajve të parë aktivë."
        )

    return "Nuk vërehet ndryshim material ndërmjet fillimit dhe fundit të periudhës."


def hourly_profile_figure(hourly: pd.DataFrame) -> go.Figure:
    data = hourly.sort_values("hour")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["hour"],
            y=data["p10_kwh"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["hour"],
            y=data["p90_kwh"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.16)",
            name="Intervali P10–P90",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["hour"],
            y=data["median_kwh"],
            mode="lines+markers",
            name="Mediana",
            line={"width": 3},
            marker={"size": 7},
            hovertemplate=("Ora %{x}<br>" "Mediana: %{y:,.2f} kWh" "<extra></extra>"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["hour"],
            y=data["mean_kwh"],
            mode="lines",
            name="Mesatarja",
            line={"width": 2, "dash": "dash"},
            hovertemplate=("Ora %{x}<br>" "Mesatarja: %{y:,.2f} kWh" "<extra></extra>"),
        )
    )

    fig.add_vrect(
        x0=7,
        x1=18,
        opacity=0.08,
        layer="below",
        line_width=0,
        annotation_text="Peak",
        annotation_position="top left",
    )

    fig.update_xaxes(tickmode="linear", dtick=1)

    return style_figure(
        fig,
        title="Profili tipik 24-orësh i njehsorit",
        x_title="Ora",
        y_title="Energji për orë (kWh)",
        height=560,
        hovermode="x unified",
    )


def company_meter_hourly_figure(
    company_hourly: pd.DataFrame,
    company_profiles: pd.DataFrame,
    selected: pd.Series,
) -> go.Figure:
    data = company_hourly.merge(
        company_profiles[
            [
                "company_code",
                "meter_id",
                "source_sheet",
                "source_column",
                "profile_similarity",
            ]
        ],
        on=[
            "company_code",
            "meter_id",
            "source_sheet",
            "source_column",
        ],
        how="left",
    )

    data["meter_label"] = data["meter_id"].astype(str)

    fig = go.Figure()

    meter_keys = (
        data[
            [
                "meter_id",
                "source_sheet",
                "source_column",
            ]
        ]
        .drop_duplicates()
        .sort_values(["meter_id", "source_sheet", "source_column"])
    )

    selected_key = (
        str(selected["meter_id"]),
        str(selected["source_sheet"]),
        str(selected["source_column"]),
    )

    for meter in meter_keys.itertuples(index=False):
        meter_data = data[
            (data["meter_id"].astype(str) == str(meter.meter_id))
            & (data["source_sheet"].astype(str) == str(meter.source_sheet))
            & (data["source_column"].astype(str) == str(meter.source_column))
        ].sort_values("hour")

        current_key = (
            str(meter.meter_id),
            str(meter.source_sheet),
            str(meter.source_column),
        )

        is_selected = current_key == selected_key

        fig.add_trace(
            go.Scatter(
                x=meter_data["hour"],
                y=meter_data["mean_kwh"],
                mode="lines",
                name=str(meter.meter_id),
                line={
                    "width": 4 if is_selected else 1.5,
                },
                opacity=1.0 if is_selected else 0.45,
                hovertemplate=(
                    f"<b>{meter.meter_id}</b><br>"
                    "Ora %{x}<br>"
                    "Mesatarja: %{y:,.2f} kWh"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(tickmode="linear", dtick=1)

    return style_figure(
        fig,
        title="Krahasimi i profileve 24-orëshe brenda kompanisë",
        x_title="Ora",
        y_title="Konsumi mesatar për orë (kWh)",
        height=600,
        hovermode="x unified",
    )


def metric_comparison_figure(
    company_profiles: pd.DataFrame,
    selected: pd.Series,
) -> go.Figure:
    data = company_profiles.copy()

    data["selected"] = np.where(
        selected_meter_mask(data, selected),
        "Njehsori i zgjedhur",
        "Njehsorët tjerë",
    )

    data["similarity_label"] = (
        data["profile_similarity"]
        .map(SIMILARITY_SHORT_LABELS)
        .fillna(data["profile_similarity"])
    )

    fig = px.scatter(
        data,
        x="peak_ratio",
        y="load_factor",
        size="total_kwh",
        color="similarity_label",
        symbol="selected",
        hover_name="meter_id",
        hover_data={
            "weekday_weekend_ratio": ":.3f",
            "cv": ":.3f",
            "trend_percent": ":.2f",
            "total_kwh": ":,.0f",
            "similarity_label": False,
            "selected": False,
        },
        labels={
            "peak_ratio": "Peak / Off-Peak",
            "load_factor": "Load Factor",
            "weekday_weekend_ratio": "Ditë Pune / Fundjavë",
            "cv": "CV",
            "trend_percent": "Trendi (%)",
            "total_kwh": "Konsumi total (kWh)",
            "similarity_label": "Ngjashmëria",
        },
        size_max=38,
    )

    return style_figure(
        fig,
        title="Pozicionimi i njehsorëve të kompanisë",
        x_title="Raporti Peak / Off-Peak",
        y_title="Load Factor",
        height=580,
    )


def monthly_profile_figure(
    monthly: pd.DataFrame,
    annual_mean: float,
) -> go.Figure:
    data = monthly.copy()
    data["month_date"] = pd.to_datetime(data["month"] + "-01")
    data = data.sort_values("month_date")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["month_date"],
            y=data["mean_kwh"],
            mode="lines+markers",
            name="Konsumi mesatar mujor",
            line={"width": 3},
            marker={"size": 8},
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>" "Mesatarja: %{y:,.2f} kWh" "<extra></extra>"
            ),
        )
    )

    if pd.notna(annual_mean):
        fig.add_hline(
            y=annual_mean,
            line_dash="dash",
            annotation_text=f"Mesatarja vjetore {annual_mean:,.2f} kWh",
            annotation_position="top left",
        )

    return style_figure(
        fig,
        title="Profili mujor i njehsorit",
        x_title="Muaji",
        y_title="Konsumi mesatar për orë (kWh)",
        height=560,
        hovermode="x unified",
    )


def company_similarity_bar(company_profiles: pd.DataFrame) -> go.Figure:
    counts = (
        company_profiles["profile_similarity"]
        .value_counts()
        .rename_axis("profile_similarity")
        .reset_index(name="meters")
    )

    counts["label"] = (
        counts["profile_similarity"]
        .map(SIMILARITY_SHORT_LABELS)
        .fillna(counts["profile_similarity"])
    )

    fig = px.bar(
        counts,
        x="label",
        y="meters",
        text="meters",
        labels={
            "label": "Klasifikimi",
            "meters": "Numri i njehsorëve",
        },
    )

    fig.update_traces(textposition="outside")

    return style_figure(
        fig,
        title="Ngjashmëria e njehsorëve brenda kompanisë",
        x_title=None,
        y_title="Numri i njehsorëve",
        height=460,
    )


required_paths = [
    METER_PROFILES_PATH,
    METER_MONTHLY_PATH,
    METER_HOURLY_PATH,
]

missing_paths = [path for path in required_paths if not path.exists()]

if missing_paths:
    st.error("Mungojnë disa output-e të Hapit 3.7.")
    st.code("\n".join(str(path.relative_to(ROOT)) for path in missing_paths))
    st.caption("Ekzekutoni: python -m src.metrics")
    st.stop()


profiles = read_parquet(str(METER_PROFILES_PATH))
monthly = read_parquet(str(METER_MONTHLY_PATH))
hourly = read_parquet(str(METER_HOURLY_PATH))

for df in (profiles, monthly, hourly):
    df["company_code"] = df["company_code"].astype(str)
    df["meter_id"] = df["meter_id"].astype(str)
    df["source_sheet"] = df["source_sheet"].astype(str)

if "source_column" in profiles.columns:
    profiles["source_column"] = profiles["source_column"].astype(str)

if "source_column" in hourly.columns:
    hourly["source_column"] = hourly["source_column"].astype(str)


companies = sorted(
    profiles["company_code"].dropna().unique().tolist(),
    key=company_sort_number,
)


st.title("Analiza e njehsorëve")
st.caption(
    "Hapi 3.7 — analiza e profileve të konsumit në nivel njehsori "
    "për Korrik 2025 – Qershor 2026"
)

selector_cols = st.columns([2, 3, 4])

with selector_cols[0]:
    selected_company = st.selectbox(
        "Zgjidh kompaninë",
        companies,
        key="meter-analysis-company-selector",
    )

company_profiles = profiles[profiles["company_code"] == selected_company].copy()

company_profiles = company_profiles.sort_values(
    ["meter_id", "source_sheet", "source_column"]
).reset_index(drop=True)

company_profiles["meter_display"] = company_profiles.apply(
    meter_display_name,
    axis=1,
)

with selector_cols[1]:
    selected_meter_display = st.selectbox(
        "Zgjidh njehsorin",
        company_profiles["meter_display"].tolist(),
        key="meter-analysis-meter-selector",
    )

selected = company_profiles[
    company_profiles["meter_display"] == selected_meter_display
].iloc[0]

with selector_cols[2]:
    st.caption(
        "Analiza përfshin vetëm njehsorët e konsumit që kalojnë filtrin "
        "e cilësisë së Hapit 1. Krahasimi i ngjashmërisë bëhet brenda "
        "së njëjtës kompani."
    )

selected_monthly = monthly[
    (monthly["company_code"] == selected["company_code"])
    & (monthly["meter_id"] == selected["meter_id"])
    & (monthly["source_sheet"] == selected["source_sheet"])
].copy()

selected_hourly = hourly[selected_meter_mask(hourly, selected)].copy()

company_hourly = hourly[hourly["company_code"] == selected_company].copy()


(
    overview_tab,
    hourly_tab,
    comparison_tab,
    metrics_tab,
    seasonality_tab,
    similarity_tab,
    table_tab,
) = st.tabs(
    [
        "Përmbledhje",
        "Profili 24-orësh",
        "Krahasimi i njehsorëve",
        "Metrikat",
        "Sezonaliteti dhe trendi",
        "Ngjashmëria",
        "Tabela e njehsorëve",
    ]
)


with overview_tab:
    analysis_description(
        title="Kartela e njehsorit",
        description=(
            "Kjo kartelë përmbledh karakteristikat kryesore të profilit "
            "të konsumit për njehsorin e zgjedhur."
        ),
        purpose=(
            "Të identifikohet sjellja e njehsorit dhe roli i tij brenda "
            "profilit të përgjithshëm të kompanisë."
        ),
        finding=(
            f"Njehsori klasifikohet si "
            f"'{similarity_label(selected['profile_similarity'])}' dhe ka "
            f"sezonalitet '{seasonality_label(selected['seasonality'])}'."
        ),
    )

    row_1 = st.columns(5)

    with row_1[0]:
        value_card("Konsumi vjetor", format_energy(float(selected["total_kwh"])))

    with row_1[1]:
        value_card("Mesatarja", format_kwh(selected["mean_kwh"]))

    with row_1[2]:
        value_card("Piku maksimal", format_kwh(selected["max_kwh"]))

    with row_1[3]:
        value_card("Peak / Off-Peak", format_ratio(selected["peak_ratio"]))

    with row_1[4]:
        value_card(
            "Ditë Pune / Fundjavë",
            format_ratio(selected["weekday_weekend_ratio"]),
        )

    row_2 = st.columns(5)

    with row_2[0]:
        value_card("CV", format_ratio(selected["cv"]))

    with row_2[1]:
        value_card("Load Factor", format_decimal_percent(selected["load_factor"]))

    with row_2[2]:
        value_card("Sezonaliteti", seasonality_label(selected["seasonality"]))

    with row_2[3]:
        value_card("Trendi", format_percent(selected["trend_percent"]))

    with row_2[4]:
        value_card(
            "Ngjashmëria",
            similarity_short_label(selected["profile_similarity"]),
        )

    st.markdown("#### Identifikimi teknik")

    metadata_cols = st.columns(4)

    with metadata_cols[0]:
        value_card("Kompania", selected_company)

    with metadata_cols[1]:
        value_card("Njehsori", str(selected["meter_id"]))

    with metadata_cols[2]:
        value_card("Sheet-i burimor", str(selected["source_sheet"]))

    with metadata_cols[3]:
        value_card("Kolona burimore", str(selected["source_column"]))

    st.caption(
        "Identifikuesit teknikë ruhen për gjurmueshmëri nga output-i analitik "
        "deri te kolona origjinale në workbook."
    )


with hourly_tab:
    analysis_description(
        title="Profili tipik 24-orësh",
        description=(
            "Grafiku paraqet mesataren, medianën dhe intervalin P10–P90 për "
            "secilën orë të ditës."
        ),
        purpose=(
            "Të identifikohet forma e profilit të njehsorit dhe stabiliteti "
            "i konsumit në secilën orë."
        ),
        finding=peak_finding(selected),
    )

    render_plot(
        hourly_profile_figure(selected_hourly),
        "meter-analysis-hourly-profile",
    )

    cols = st.columns(4)

    with cols[0]:
        value_card("Mesatarja Peak", format_kwh(selected["peak_mean_kwh"]))

    with cols[1]:
        value_card("Mesatarja Off-Peak", format_kwh(selected["off_peak_mean_kwh"]))

    with cols[2]:
        value_card("Peak / Off-Peak", format_ratio(selected["peak_ratio"]))

    with cols[3]:
        value_card("Load Factor", format_decimal_percent(selected["load_factor"]))


with comparison_tab:
    meter_count = len(company_profiles)

    analysis_description(
        title="Krahasimi i njehsorëve brenda kompanisë",
        description=(
            "Të gjithë njehsorët e përdorshëm të kompanisë paraqiten së bashku "
            "për të krahasuar formën e tyre 24-orëshe. Njehsori i zgjedhur "
            "theksohet me vijë më të fortë."
        ),
        purpose=(
            "Të identifikohen njehsorët që ndjekin një sjellje të përbashkët "
            "dhe ata që devijojnë nga profili i kompanisë."
        ),
        finding=(
            f"{selected_company} ka {meter_count} njehsorë të përdorshëm në "
            "analizën e Hapit 3.7."
        ),
    )

    render_plot(
        company_meter_hourly_figure(
            company_hourly,
            company_profiles,
            selected,
        ),
        "meter-analysis-company-hourly-comparison",
    )

    if meter_count == 1:
        st.info(
            "Kjo kompani ka vetëm një njehsor të përdorshëm, prandaj nuk ka "
            "bazë për krahasim të brendshëm të profileve."
        )


with metrics_tab:
    analysis_description(
        title="Pozicionimi sipas metrikave të profilit",
        description=(
            "Grafiku krahason njehsorët e së njëjtës kompani sipas raportit "
            "Peak/Off-Peak dhe Load Factor. Madhësia e pikës përfaqëson "
            "konsumin total."
        ),
        purpose=(
            "Të shihet nëse njehsori i zgjedhur ka karakteristika të ngjashme "
            "me njehsorët tjerë të kompanisë apo zë një pozicion të veçantë."
        ),
        finding=(
            f"Statusi i ngjashmërisë për njehsorin e zgjedhur është "
            f"'{similarity_label(selected['profile_similarity'])}'."
        ),
    )

    render_plot(
        metric_comparison_figure(company_profiles, selected),
        "meter-analysis-metric-comparison",
    )

    metric_cols = st.columns(4)

    with metric_cols[0]:
        value_card("CV", format_ratio(selected["cv"]))

    with metric_cols[1]:
        value_card("Load Factor", format_decimal_percent(selected["load_factor"]))

    with metric_cols[2]:
        value_card(
            "Ditë Pune / Fundjavë",
            format_ratio(selected["weekday_weekend_ratio"]),
        )

    with metric_cols[3]:
        value_card("Trendi", format_percent(selected["trend_percent"]))

    st.info(weekday_finding(selected))


with seasonality_tab:
    analysis_description(
        title="Sezonaliteti dhe trendi mujor",
        description=(
            "Profili mujor tregon si ndryshon konsumi mesatar i njehsorit gjatë "
            "dritares vjetore dhe e krahason atë me mesataren vjetore."
        ),
        purpose=(
            "Të vlerësohet nëse njehsori ka varësi sezonale dhe nëse konsumi "
            "po rritet apo bie drejt fundit të periudhës."
        ),
        finding=(
            f"Sezonaliteti është '{seasonality_label(selected['seasonality'])}'. "
            f"{trend_finding(selected)}"
        ),
    )

    render_plot(
        monthly_profile_figure(selected_monthly, selected["mean_kwh"]),
        "meter-analysis-monthly-profile",
    )

    cols = st.columns(5)

    with cols[0]:
        value_card("Mesatarja verë", format_kwh(selected["summer_mean_kwh"]))

    with cols[1]:
        value_card("Mesatarja dimër", format_kwh(selected["winter_mean_kwh"]))

    with cols[2]:
        value_card("Indeksi i verës", format_ratio(selected["summer_index"]))

    with cols[3]:
        value_card("Indeksi i dimrit", format_ratio(selected["winter_index"]))

    with cols[4]:
        value_card("Indeksi sezonal", format_ratio(selected["seasonality_index"]))

    period_cols = st.columns(3)

    with period_cols[0]:
        value_card(
            "3 muajt e parë",
            format_kwh(selected["first_three_month_mean_kwh"]),
        )

    with period_cols[1]:
        value_card("Muaji i fundit", format_kwh(selected["last_month_mean_kwh"]))

    with period_cols[2]:
        value_card(
            "Periudha aktive",
            f"{selected['first_active_month']} → {selected['last_active_month']}",
        )


with similarity_tab:
    similar_count = int((company_profiles["profile_similarity"] == "similar").sum())
    different_count = int((company_profiles["profile_similarity"] == "different").sum())
    single_count = int((company_profiles["profile_similarity"] == "single_meter").sum())

    analysis_description(
        title="Ngjashmëria e profileve brenda kompanisë",
        description=(
            "Backend-i i Hapit 3.7 klasifikon secilin njehsor si të ngjashëm, "
            "të ndryshëm ose si njehsor të vetëm të kompanisë."
        ),
        purpose=(
            "Të identifikohen njehsorët që mund të përfaqësojnë procese, zona "
            "ose sjellje të ndryshme operative brenda të njëjtës kompani."
        ),
        finding=(
            f"Në {selected_company}: {similar_count} njehsorë janë klasifikuar "
            f"si të ngjashëm, {different_count} si të ndryshëm dhe "
            f"{single_count} si njehsor i vetëm."
        ),
    )

    render_plot(
        company_similarity_bar(company_profiles),
        "meter-analysis-similarity-bar",
    )

    similarity_cols = st.columns(3)

    with similarity_cols[0]:
        value_card("Të ngjashëm", str(similar_count))

    with similarity_cols[1]:
        value_card("Të ndryshëm", str(different_count))

    with similarity_cols[2]:
        value_card("Njehsor i vetëm", str(single_count))

    different_meters = company_profiles[
        company_profiles["profile_similarity"] == "different"
    ].copy()

    if not different_meters.empty:
        st.markdown("#### Njehsorët me profil të ndryshëm")

        different_display = different_meters[
            [
                "meter_id",
                "source_sheet",
                "source_column",
                "total_kwh",
                "peak_ratio",
                "weekday_weekend_ratio",
                "cv",
                "load_factor",
                "seasonality",
                "trend_percent",
            ]
        ].copy()

        different_display["seasonality"] = (
            different_display["seasonality"]
            .map(SEASONALITY_LABELS)
            .fillna(different_display["seasonality"])
        )

        different_display = different_display.rename(
            columns={
                "meter_id": "Njehsori",
                "source_sheet": "Sheet-i",
                "source_column": "Kolona burimore",
                "total_kwh": "Konsumi total (kWh)",
                "peak_ratio": "Peak / Off-Peak",
                "weekday_weekend_ratio": "Ditë Pune / Fundjavë",
                "cv": "CV",
                "load_factor": "Load Factor",
                "seasonality": "Sezonaliteti",
                "trend_percent": "Trendi (%)",
            }
        )

        st.dataframe(
            different_display,
            width="stretch",
            hide_index=True,
        )
    else:
        st.success(
            "Nuk ka njehsorë të klasifikuar me profil të ndryshëm në këtë kompani."
        )


with table_tab:
    analysis_description(
        title="Tabela e njehsorëve",
        description=(
            "Tabela përmbledh metrikat e Hapit 3.7 për të gjithë njehsorët e "
            "përdorshëm të konsumit."
        ),
        purpose=(
            "Të mundësohet filtrimi dhe inspektimi i drejtpërdrejtë i profileve "
            "në nivel njehsori."
        ),
        finding=(
            f"Në total janë {len(profiles)} njehsorë të përdorshëm në "
            f"{profiles['company_code'].nunique()} kompani."
        ),
    )

    filter_cols = st.columns(4)

    with filter_cols[0]:
        company_filter = st.selectbox(
            "Kompania",
            ["Të gjitha"] + companies,
            key="meter-table-company-filter",
        )

    with filter_cols[1]:
        similarity_filter = st.selectbox(
            "Ngjashmëria",
            ["Të gjitha", "I ngjashëm", "I ndryshëm", "Njehsor i vetëm"],
            key="meter-table-similarity-filter",
        )

    with filter_cols[2]:
        seasonality_filter = st.selectbox(
            "Sezonaliteti",
            ["Të gjitha", "Dimër", "Verë", "Pa sezonalitet të qartë"],
            key="meter-table-seasonality-filter",
        )

    with filter_cols[3]:
        trend_filter = st.selectbox(
            "Trendi",
            ["Të gjitha", "Rritje", "Rënie"],
            key="meter-table-trend-filter",
        )

    table_data = profiles.copy()

    table_data["similarity_label"] = (
        table_data["profile_similarity"]
        .map(SIMILARITY_SHORT_LABELS)
        .fillna(table_data["profile_similarity"])
    )

    table_data["seasonality_label"] = (
        table_data["seasonality"]
        .map(SEASONALITY_LABELS)
        .fillna(table_data["seasonality"])
    )

    if company_filter != "Të gjitha":
        table_data = table_data[table_data["company_code"] == company_filter]

    similarity_reverse = {
        "I ngjashëm": "similar",
        "I ndryshëm": "different",
        "Njehsor i vetëm": "single_meter",
    }

    if similarity_filter != "Të gjitha":
        table_data = table_data[
            table_data["profile_similarity"] == similarity_reverse[similarity_filter]
        ]

    if seasonality_filter != "Të gjitha":
        table_data = table_data[table_data["seasonality_label"] == seasonality_filter]

    if trend_filter == "Rritje":
        table_data = table_data[table_data["trend_percent"] > 0]
    elif trend_filter == "Rënie":
        table_data = table_data[table_data["trend_percent"] < 0]

    table_data = sort_by_company(table_data)

    display = table_data[
        [
            "company_code",
            "meter_id",
            "source_sheet",
            "source_column",
            "total_kwh",
            "peak_ratio",
            "weekday_weekend_ratio",
            "cv",
            "load_factor",
            "seasonality_label",
            "trend_percent",
            "similarity_label",
        ]
    ].rename(
        columns={
            "company_code": "Kompania",
            "meter_id": "Njehsori",
            "source_sheet": "Sheet-i",
            "source_column": "Kolona burimore",
            "total_kwh": "Konsumi total (kWh)",
            "peak_ratio": "Peak / Off-Peak",
            "weekday_weekend_ratio": "Ditë Pune / Fundjavë",
            "cv": "CV",
            "load_factor": "Load Factor",
            "seasonality_label": "Sezonaliteti",
            "trend_percent": "Trendi (%)",
            "similarity_label": "Ngjashmëria",
        }
    )

    st.caption(f"Po shfaqen {len(display)} njehsorë.")

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Konsumi total (kWh)": st.column_config.NumberColumn(format="%.2f"),
            "Peak / Off-Peak": st.column_config.NumberColumn(format="%.3f"),
            "Ditë Pune / Fundjavë": st.column_config.NumberColumn(format="%.3f"),
            "CV": st.column_config.NumberColumn(format="%.3f"),
            "Load Factor": st.column_config.NumberColumn(format="%.3f"),
            "Trendi (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
