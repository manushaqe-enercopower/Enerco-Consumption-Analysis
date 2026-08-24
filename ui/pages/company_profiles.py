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
    sys.path.insert(
        0,
        str(UI_DIR),
    )


from components.charting import (
    PLOTLY_CONFIG,
    company_sort_number,
    format_energy,
    sort_by_company,
    style_figure,
)

PROFILE_DIR = ROOT / "data" / "processed" / "profile_metrics"

COMPANY_PROFILES_PATH = PROFILE_DIR / "company_profiles.parquet"

COMPANY_MONTHLY_PATH = PROFILE_DIR / "company_monthly.parquet"

COMPANY_HOURLY_PATH = PROFILE_DIR / "company_hourly_profile.parquet"
PORTFOLIO_PROFILE_PATH = PROFILE_DIR / "portfolio_profile.parquet"
PORTFOLIO_MONTHLY_PATH = PROFILE_DIR / "portfolio_monthly.parquet"
PORTFOLIO_HOURLY_PATH = PROFILE_DIR / "portfolio_hourly_profile.parquet"

ALL_COMPANIES = "Të gjitha kompanitë"


SEASONALITY_LABELS = {
    "winter": "Dimër",
    "summer": "Verë",
    "none": "Pa sezonalitet të qartë",
}


@st.cache_data(show_spinner=False)
def read_parquet(
    path: str,
) -> pd.DataFrame:
    return pd.read_parquet(path)


def render_plot(
    fig: go.Figure,
    key: str,
) -> None:
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


def safe_percentile(
    series: pd.Series,
    value: float,
) -> float:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty or pd.isna(value):
        return np.nan

    return float((numeric <= value).mean() * 100)


def format_ratio(
    value: float,
) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{value:.2f}"


def format_percent(
    value: float,
) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{value:.2f}%"


def format_decimal_percent(
    value: float,
) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.1f}%"


def seasonality_label(
    value: str,
) -> str:
    return SEASONALITY_LABELS.get(
        str(value),
        str(value),
    )


def peak_finding(
    row: pd.Series,
) -> str:
    value = row["peak_ratio"]

    if pd.isna(value):
        return "Raporti Peak/Off-Peak nuk mund të " "llogaritet për këtë kompani."

    if value > 1:
        difference = (value - 1) * 100

        return (
            f"Konsumi mesatar gjatë orëve Peak është "
            f"rreth {difference:.1f}% më i lartë se gjatë "
            "orëve Off-Peak."
        )

    if value < 1:
        difference = (1 - value) * 100

        return (
            f"Konsumi mesatar gjatë orëve Peak është "
            f"rreth {difference:.1f}% më i ulët se gjatë "
            "orëve Off-Peak."
        )

    return "Konsumi mesatar Peak dhe Off-Peak është " "pothuajse i njëjtë."


def weekday_finding(
    row: pd.Series,
) -> str:
    value = row["weekday_weekend_ratio"]

    if pd.isna(value):
        return "Raporti Ditë Pune/Fundjavë nuk mund të " "llogaritet për këtë kompani."

    if value > 1:
        difference = (value - 1) * 100

        return (
            f"Konsumi gjatë ditëve të punës është mesatarisht "
            f"{difference:.1f}% më i lartë se gjatë fundjavës."
        )

    if value < 1:
        difference = (1 - value) * 100

        return (
            f"Konsumi gjatë ditëve të punës është mesatarisht "
            f"{difference:.1f}% më i ulët se gjatë fundjavës."
        )

    return (
        "Kompania ka konsum pothuajse të njëjtë gjatë " "ditëve të punës dhe fundjavës."
    )


def trend_finding(
    row: pd.Series,
) -> str:
    trend = row["trend_percent"]

    if pd.isna(trend):
        return "Trendi nuk mund të vlerësohet për këtë kompani."

    if trend > 0:
        return (
            f"Muaji i fundit ka konsum mesatar "
            f"{trend:.1f}% më të lartë se mesatarja e "
            "tre muajve të parë aktivë."
        )

    if trend < 0:
        return (
            f"Muaji i fundit ka konsum mesatar "
            f"{abs(trend):.1f}% më të ulët se mesatarja e "
            "tre muajve të parë aktivë."
        )

    return (
        "Nuk vërehet ndryshim material ndërmjet muajit "
        "të fundit dhe periudhës fillestare."
    )


def load_factor_finding(
    row: pd.Series,
    profiles: pd.DataFrame,
) -> str:
    value = row["load_factor"]

    median = profiles["load_factor"].median()

    if pd.isna(value):
        return "Load Factor nuk mund të llogaritet."

    relation = "mbi" if value > median else "nën"

    return (
        f"Load Factor është {value:.3f} "
        f"({value * 100:.1f}% e pikut), "
        f"{relation} medianës së portofolit prej "
        f"{median:.3f}."
    )


def cv_finding(
    row: pd.Series,
    profiles: pd.DataFrame,
) -> str:
    value = row["cv"]

    median = profiles["cv"].median()

    if pd.isna(value):
        return "CV nuk mund të llogaritet."

    if value > median:
        return (
            f"CV = {value:.3f}, mbi medianën e portofolit "
            f"prej {median:.3f}. Profili i kësaj kompanie "
            "është relativisht më i ndryshueshëm."
        )

    return (
        f"CV = {value:.3f}, nën medianën e portofolit "
        f"prej {median:.3f}. Profili i kësaj kompanie "
        "është relativisht më stabil."
    )


def hourly_profile_figure(
    hourly: pd.DataFrame,
) -> go.Figure:
    data = hourly[hourly["profile_type"] == "all_days"].sort_values("hour")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["hour"],
            y=data["p10_kwh"],
            mode="lines",
            line={
                "width": 0,
            },
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["hour"],
            y=data["p90_kwh"],
            mode="lines",
            line={
                "width": 0,
            },
            fill="tonexty",
            fillcolor=("rgba(37, 99, 235, 0.16)"),
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
            line={
                "width": 3,
            },
            marker={
                "size": 7,
            },
            hovertemplate=("Ora %{x}<br>" "Mediana: %{y:,.2f} kWh" "<extra></extra>"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["hour"],
            y=data["mean_kwh"],
            mode="lines",
            name="Mesatarja",
            line={
                "width": 2,
                "dash": "dash",
            },
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

    fig.update_xaxes(
        tickmode="linear",
        dtick=1,
    )

    return style_figure(
        fig,
        title="Profili tipik 24-orësh i konsumit",
        x_title="Ora",
        y_title="Energji për orë (kWh)",
        height=560,
        hovermode="x unified",
    )


def weekday_weekend_figure(
    hourly: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    mapping = {
        "weekday": "Ditë pune",
        "weekend": "Fundjavë",
    }

    for profile_type, label in mapping.items():
        data = hourly[hourly["profile_type"] == profile_type].sort_values("hour")

        fig.add_trace(
            go.Scatter(
                x=data["hour"],
                y=data["mean_kwh"],
                mode="lines+markers",
                name=label,
                line={
                    "width": 3,
                },
                marker={
                    "size": 6,
                },
                hovertemplate=(
                    f"{label}<br>"
                    "Ora %{x}<br>"
                    "Mesatarja: %{y:,.2f} kWh"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(
        tickmode="linear",
        dtick=1,
    )

    return style_figure(
        fig,
        title="Profili orar: ditë pune kundrejt fundjavës",
        x_title="Ora",
        y_title="Konsumi mesatar për orë (kWh)",
        height=560,
        hovermode="x unified",
    )


def monthly_profile_figure(
    monthly: pd.DataFrame,
    annual_mean: float,
) -> go.Figure:
    data = monthly.copy()

    data["month_date"] = pd.to_datetime(data["month"] + "-01")

    data["month_number"] = data["month_date"].dt.month

    data["season"] = "Muaj neutral"

    data.loc[
        data["month_number"].isin(
            [
                6,
                7,
                8,
            ]
        ),
        "season",
    ] = "Verë"

    data.loc[
        data["month_number"].isin(
            [
                12,
                1,
                2,
            ]
        ),
        "season",
    ] = "Dimër"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["month_date"],
            y=data["mean_kwh"],
            mode="lines+markers",
            name="Konsumi mesatar mujor",
            line={
                "width": 3,
            },
            marker={
                "size": 9,
                "color": data["month_number"],
                "colorscale": "Viridis",
                "showscale": False,
            },
            text=data["season"],
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>"
                "Mesatarja: %{y:,.2f} kWh<br>"
                "Periudha: %{text}"
                "<extra></extra>"
            ),
        )
    )

    if pd.notna(annual_mean):
        fig.add_hline(
            y=annual_mean,
            line_dash="dash",
            annotation_text=(f"Mesatarja vjetore " f"{annual_mean:,.2f} kWh"),
            annotation_position="top left",
        )

    return style_figure(
        fig,
        title="Profili mujor dhe sezonaliteti",
        x_title="Muaji",
        y_title="Konsumi mesatar për orë (kWh)",
        height=560,
        hovermode="x unified",
    )


def trend_figure(
    monthly: pd.DataFrame,
    profile_row: pd.Series,
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
            line={
                "width": 3,
            },
            marker={
                "size": 8,
            },
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>" "Mesatarja: %{y:,.2f} kWh" "<extra></extra>"
            ),
        )
    )

    first_three = profile_row["first_three_month_mean_kwh"]

    if pd.notna(first_three):
        fig.add_hline(
            y=first_three,
            line_dash="dash",
            annotation_text=("Mesatarja e 3 muajve të parë"),
            annotation_position="bottom left",
        )

    if not data.empty:
        last = data.iloc[-1]

        fig.add_annotation(
            x=last["month_date"],
            y=last["mean_kwh"],
            text=(f"Muaji i fundit<br>" f"{last['mean_kwh']:,.1f} kWh"),
            showarrow=True,
            arrowhead=2,
            ax=-60,
            ay=-50,
        )

    return style_figure(
        fig,
        title="Ecuria mujore e konsumit",
        x_title="Muaji",
        y_title="Konsumi mesatar për orë (kWh)",
        height=560,
        hovermode="x unified",
    )


def metric_percentile_figure(
    profiles: pd.DataFrame,
    selected: pd.Series,
) -> go.Figure:
    metrics = [
        (
            "peak_ratio",
            "Peak / Off-Peak",
        ),
        (
            "weekday_weekend_ratio",
            "Ditë Pune / Fundjavë",
        ),
        (
            "cv",
            "CV",
        ),
        (
            "load_factor",
            "Load Factor",
        ),
        (
            "seasonality_index",
            "Indeksi sezonal",
        ),
        (
            "trend_percent",
            "Trendi",
        ),
    ]

    rows = []

    for column, label in metrics:
        value = selected[column]

        percentile = safe_percentile(
            profiles[column],
            value,
        )

        rows.append(
            {
                "metric": label,
                "percentile": percentile,
                "value": value,
            }
        )

    data = pd.DataFrame(rows).dropna(
        subset=[
            "percentile",
        ]
    )

    fig = go.Figure(
        go.Bar(
            x=data["percentile"],
            y=data["metric"],
            orientation="h",
            text=[
                (f"{value:.2f}" if pd.notna(value) else "N/A")
                for value in data["value"]
            ],
            textposition="outside",
            customdata=data["value"],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Vlera: %{customdata:.3f}<br>"
                "Percentili në portofol: %{x:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=50,
        line_dash="dash",
        annotation_text="Mediana e portofolit",
        annotation_position="top",
    )

    fig.update_xaxes(
        range=[
            0,
            105,
        ]
    )

    return style_figure(
        fig,
        title="Pozicioni i kompanisë në shpërndarjen e portofolit",
        x_title="Percentili (%)",
        y_title=None,
        height=560,
    )


def portfolio_metric_scatter(
    profiles: pd.DataFrame,
    selected_company: str,
) -> go.Figure:
    data = profiles.copy()

    data["seasonality_label"] = (
        data["seasonality"].map(SEASONALITY_LABELS).fillna(data["seasonality"])
    )

    if selected_company == ALL_COMPANIES:
        data["selected"] = "Të gjitha kompanitë"
    else:
        data["selected"] = np.where(
            data["company_code"] == selected_company,
            "Kompania e zgjedhur",
            "Kompanitë tjera",
        )

    fig = px.scatter(
        data,
        x="peak_ratio",
        y="load_factor",
        size="total_kwh",
        color="seasonality_label",
        symbol="selected",
        hover_name="company_code",
        hover_data={
            "weekday_weekend_ratio": ":.3f",
            "cv": ":.3f",
            "trend_percent": ":.2f",
            "meter_count_used": True,
            "total_kwh": ":,.0f",
            "seasonality_label": False,
            "selected": False,
        },
        labels={
            "peak_ratio": ("Raporti Peak / Off-Peak"),
            "load_factor": "Load Factor",
            "seasonality_label": "Sezonaliteti",
            "weekday_weekend_ratio": ("Ditë Pune / Fundjavë"),
            "cv": "CV",
            "trend_percent": "Trendi (%)",
            "meter_count_used": ("Njehsorë të përdorur"),
            "total_kwh": ("Konsumi total (kWh)"),
        },
        size_max=36,
    )

    return style_figure(
        fig,
        title="Pozicionimi i profileve të kompanive",
        x_title="Raporti Peak / Off-Peak",
        y_title="Load Factor",
        height=600,
    )


required_paths = [
    COMPANY_PROFILES_PATH,
    COMPANY_MONTHLY_PATH,
    COMPANY_HOURLY_PATH,
    PORTFOLIO_PROFILE_PATH,
    PORTFOLIO_MONTHLY_PATH,
    PORTFOLIO_HOURLY_PATH,
]

missing_paths = [path for path in required_paths if not path.exists()]

if missing_paths:
    st.error("Mungojnë disa output-e të Hapit 3.")

    st.code("\n".join(str(path.relative_to(ROOT)) for path in missing_paths))

    st.caption("Ekzekutoni: python -m src.metrics")

    st.stop()


profiles = read_parquet(str(COMPANY_PROFILES_PATH))

monthly = read_parquet(str(COMPANY_MONTHLY_PATH))

hourly = read_parquet(str(COMPANY_HOURLY_PATH))
portfolio_profile = read_parquet(str(PORTFOLIO_PROFILE_PATH))
portfolio_monthly = read_parquet(str(PORTFOLIO_MONTHLY_PATH))
portfolio_hourly = read_parquet(str(PORTFOLIO_HOURLY_PATH))


profiles["company_code"] = profiles["company_code"].astype(str)

monthly["company_code"] = monthly["company_code"].astype(str)

hourly["company_code"] = hourly["company_code"].astype(str)


companies = sorted(
    profiles["company_code"].dropna().unique().tolist(),
    key=company_sort_number,
)


st.title("Profilet e konsumit të kompanive")

st.caption(
    "Karakterizimi i profilit vjetor të konsumit " "për Korrik 2025 – Qershor 2026"
)


selector_left, selector_right = st.columns(
    [
        2,
        5,
    ]
)

with selector_left:
    selected_company = st.selectbox(
        "Zgjidh kompaninë",
        [ALL_COMPANIES, *companies],
        key="company-profile-selector",
    )

with selector_right:
    st.caption(
        "Metrikat llogariten vetëm nga njehsorët e konsumit "
        "që kalojnë kontrollin e cilësisë së Hapit 1."
    )


is_portfolio = selected_company == ALL_COMPANIES

if is_portfolio:
    selected = portfolio_profile.iloc[0]
    selected_monthly = portfolio_monthly.copy()
    selected_hourly = portfolio_hourly.copy()
    selected_label = f"Portofoli ({len(profiles)} kompani)"
    st.info(f"Po shfaqet analiza e kombinuar për të gjitha {len(profiles)} kompanitë.")
else:
    selected = profiles[profiles["company_code"] == selected_company].iloc[0]
    selected_monthly = monthly[monthly["company_code"] == selected_company].copy()
    selected_hourly = hourly[hourly["company_code"] == selected_company].copy()
    selected_label = selected_company


(
    overview_tab,
    hourly_tab,
    weekday_tab,
    seasonality_tab,
    trend_tab,
    comparison_tab,
    table_tab,
) = st.tabs(
    [
        "Përmbledhje",
        "Profili 24-orësh",
        "Ditë pune / Fundjavë",
        "Sezonaliteti",
        "Trendi mujor",
        "Krahasimi i metrikave",
        "Tabela e kompanive",
    ]
)


with overview_tab:
    analysis_description(
        title="Kartela e identitetit të konsumit",
        description=(
            "Kjo kartelë përmbledh treguesit kryesorë që "
            "përshkruajnë mënyrën se si kompania konsumon "
            "energji gjatë vitit."
        ),
        purpose=(
            "Të krijohet një identitet analitik i kompanisë "
            "që mund të përdoret për krahasim, outlier-a dhe "
            "klasterizim."
        ),
        finding=(
            f"{selected_label} klasifikohet me sezonalitet "
            f"'{seasonality_label(selected['seasonality'])}', "
            f"përdor {int(selected['meter_count_used'])} njehsorë "
            "konsumi dhe ka "
            f"{trend_finding(selected).lower()}"
        ),
    )

    top_cols = st.columns(5)

    with top_cols[0]:
        value_card(
            "Konsumi vjetor",
            format_energy(float(selected["total_kwh"])),
        )

    with top_cols[1]:
        value_card(
            "Njehsorë të përdorur",
            str(int(selected["meter_count_used"])),
        )

    with top_cols[2]:
        value_card(
            "Peak / Off-Peak",
            format_ratio(selected["peak_ratio"]),
        )

    with top_cols[3]:
        value_card(
            "Ditë Pune / Fundjavë",
            format_ratio(selected["weekday_weekend_ratio"]),
        )

    with top_cols[4]:
        value_card(
            "Trendi",
            format_percent(selected["trend_percent"]),
        )

    second_cols = st.columns(5)

    with second_cols[0]:
        value_card(
            "CV",
            format_ratio(selected["cv"]),
        )

    with second_cols[1]:
        value_card(
            "Load Factor",
            format_decimal_percent(selected["load_factor"]),
        )

    with second_cols[2]:
        value_card(
            "Sezonaliteti",
            seasonality_label(selected["seasonality"]),
        )

    with second_cols[3]:
        value_card(
            "Indeksi i verës",
            format_ratio(selected["summer_index"]),
        )

    with second_cols[4]:
        value_card(
            "Indeksi i dimrit",
            format_ratio(selected["winter_index"]),
        )

    st.markdown("#### Metadata e kompanisë")

    metadata_cols = st.columns(4)

    with metadata_cols[0]:
        value_card(
            "Kodi",
            selected_label,
        )

    with metadata_cols[1]:
        value_card(
            "Veprimtaria",
            "Nuk është hartëzuar",
        )

    with metadata_cols[2]:
        value_card(
            "Madhësia",
            "Nuk është hartëzuar",
        )

    with metadata_cols[3]:
        value_card(
            "Niveli i tensionit",
            "Nuk është hartëzuar",
        )

    st.caption(
        "Veprimtaria, madhësia dhe niveli i tensionit nuk "
        "plotësohen me supozime. Këto fusha kërkojnë mapping-un "
        "konfidencial të kompanive."
    )

    analysis_description(
        title="Pozicionimi në portofol",
        description=(
            "Grafiku vendos të gjitha kompanitë sipas raportit "
            "Peak/Off-Peak dhe Load Factor. Madhësia e pikës "
            "përfaqëson konsumin total."
        ),
        purpose=(
            "Të shihet nëse kompania e zgjedhur ka profil të "
            "ngjashëm me shumicën e portofolit apo qëndron në "
            "një zonë më të pazakontë."
        ),
    )

    render_plot(
        portfolio_metric_scatter(
            profiles,
            selected_company,
        ),
        "company-profile-overview-scatter",
    )


with hourly_tab:
    analysis_description(
        title="Profili tipik 24-orësh",
        description=(
            "Grafiku paraqet formën mesatare të konsumit gjatë "
            "24 orëve të ditës. Mediana dhe mesatarja tregohen "
            "së bashku me intervalin P10–P90."
        ),
        purpose=(
            "Të identifikohet kur kompania e rrit ose e ul "
            "konsumin dhe sa stabile është kjo sjellje nga dita "
            "në ditë."
        ),
        finding=(peak_finding(selected)),
    )

    render_plot(
        hourly_profile_figure(selected_hourly),
        "company-profile-hourly",
    )

    metric_cols = st.columns(4)

    with metric_cols[0]:
        value_card(
            "Mesatarja Peak",
            (
                f"{selected['peak_mean_kwh']:,.2f} kWh"
                if pd.notna(selected["peak_mean_kwh"])
                else "N/A"
            ),
        )

    with metric_cols[1]:
        value_card(
            "Mesatarja Off-Peak",
            (
                f"{selected['off_peak_mean_kwh']:,.2f} kWh"
                if pd.notna(selected["off_peak_mean_kwh"])
                else "N/A"
            ),
        )

    with metric_cols[2]:
        value_card(
            "Raporti Peak / Off-Peak",
            format_ratio(selected["peak_ratio"]),
        )

    with metric_cols[3]:
        value_card(
            "Load Factor",
            format_decimal_percent(selected["load_factor"]),
        )

    st.info(
        load_factor_finding(
            selected,
            profiles,
        )
    )


with weekday_tab:
    analysis_description(
        title="Ditët e punës kundrejt fundjavës",
        description=(
            "Dy profilet orare krahasojnë sjelljen tipike të "
            "kompanisë gjatë ditëve të punës dhe fundjavës."
        ),
        purpose=(
            "Të identifikohet nëse aktiviteti i kompanisë "
            "ndryshon dukshëm gjatë fundjavës."
        ),
        finding=(weekday_finding(selected)),
    )

    render_plot(
        weekday_weekend_figure(selected_hourly),
        "company-profile-weekday-weekend",
    )

    cols = st.columns(3)

    with cols[0]:
        value_card(
            "Mesatarja ditë pune",
            (
                f"{selected['weekday_mean_kwh']:,.2f} kWh"
                if pd.notna(selected["weekday_mean_kwh"])
                else "N/A"
            ),
        )

    with cols[1]:
        value_card(
            "Mesatarja fundjavë",
            (
                f"{selected['weekend_mean_kwh']:,.2f} kWh"
                if pd.notna(selected["weekend_mean_kwh"])
                else "N/A"
            ),
        )

    with cols[2]:
        value_card(
            "Raporti",
            format_ratio(selected["weekday_weekend_ratio"]),
        )


with seasonality_tab:
    analysis_description(
        title="Sezonaliteti i konsumit",
        description=(
            "Konsumi mesatar mujor krahasohet me mesataren "
            "vjetore. Muajt Qershor–Gusht përfaqësojnë verën, "
            "ndërsa Dhjetor–Shkurt dimrin."
        ),
        purpose=(
            "Të përcaktohet nëse kompania ka kërkesë dukshëm "
            "më të lartë gjatë verës, dimrit apo nuk shfaq "
            "sezonalitet të fortë."
        ),
        finding=(
            f"Profili klasifikohet si "
            f"'{seasonality_label(selected['seasonality'])}'. "
            f"Indeksi i verës është "
            f"{selected['summer_index']:.2f} dhe indeksi i dimrit "
            f"{selected['winter_index']:.2f}."
        ),
    )

    render_plot(
        monthly_profile_figure(
            selected_monthly,
            annual_mean=selected["mean_kwh"],
        ),
        "company-profile-seasonality",
    )

    cols = st.columns(4)

    with cols[0]:
        value_card(
            "Mesatarja vjetore",
            f"{selected['mean_kwh']:,.2f} kWh",
        )

    with cols[1]:
        value_card(
            "Mesatarja verë",
            (
                f"{selected['summer_mean_kwh']:,.2f} kWh"
                if pd.notna(selected["summer_mean_kwh"])
                else "N/A"
            ),
        )

    with cols[2]:
        value_card(
            "Mesatarja dimër",
            (
                f"{selected['winter_mean_kwh']:,.2f} kWh"
                if pd.notna(selected["winter_mean_kwh"])
                else "N/A"
            ),
        )

    with cols[3]:
        value_card(
            "Klasifikimi",
            seasonality_label(selected["seasonality"]),
        )

    st.markdown("#### Sezonaliteti në të gjithë portofolin")

    seasonality_counts = (
        profiles["seasonality"]
        .value_counts()
        .rename_axis("seasonality")
        .reset_index(name="companies")
    )

    seasonality_counts["label"] = (
        seasonality_counts["seasonality"]
        .map(SEASONALITY_LABELS)
        .fillna(seasonality_counts["seasonality"])
    )

    fig = px.bar(
        seasonality_counts,
        x="label",
        y="companies",
        text="companies",
        labels={
            "label": "Sezonaliteti",
            "companies": "Kompanitë",
        },
    )

    fig.update_traces(
        textposition="outside",
    )

    fig = style_figure(
        fig,
        title="Shpërndarja e sezonalitetit në portofol",
        x_title=None,
        y_title="Numri i kompanive",
        height=470,
    )

    render_plot(
        fig,
        "company-profile-seasonality-portfolio",
    )


with trend_tab:
    analysis_description(
        title="Trendi i konsumit",
        description=(
            "Trendi krahason konsumin mesatar të muajit të "
            "fundit me mesataren e tre muajve të parë aktivë "
            "në dritaren vjetore."
        ),
        purpose=(
            "Të dallohet nëse konsumi i kompanisë është në "
            "rritje, në rënie apo relativisht stabil."
        ),
        finding=(trend_finding(selected)),
    )

    render_plot(
        trend_figure(
            selected_monthly,
            selected,
        ),
        "company-profile-trend",
    )

    cols = st.columns(4)

    with cols[0]:
        value_card(
            "3 muajt e parë",
            (
                f"{selected['first_three_month_mean_kwh']:,.2f} kWh"
                if pd.notna(selected["first_three_month_mean_kwh"])
                else "N/A"
            ),
        )

    with cols[1]:
        value_card(
            "Muaji i fundit",
            (
                f"{selected['last_month_mean_kwh']:,.2f} kWh"
                if pd.notna(selected["last_month_mean_kwh"])
                else "N/A"
            ),
        )

    with cols[2]:
        value_card(
            "Trendi",
            format_percent(selected["trend_percent"]),
        )

    with cols[3]:
        value_card(
            "Periudha aktive",
            (f"{selected['first_active_month']} → " f"{selected['last_active_month']}"),
        )


with comparison_tab:
    if is_portfolio:
        analysis_description(
            title="Pamja e plotë e portofolit",
            description=(
                "Kur zgjidhen të gjitha kompanitë, kjo pamje paraqet "
                "shpërndarjen e plotë të profileve individuale në portofol."
            ),
            purpose=(
                "Të shihet si shpërndahen të gjitha kompanitë pa e trajtuar "
                "portofolin e kombinuar si një kompani individuale."
            ),
            finding=(f"Në këtë pamje janë përfshirë {len(profiles)} kompani."),
        )

        render_plot(
            portfolio_metric_scatter(
                profiles,
                ALL_COMPANIES,
            ),
            "company-profile-all-companies-scatter",
        )

        portfolio_summary = pd.DataFrame(
            {
                "Metrika": [
                    "Peak / Off-Peak",
                    "Ditë Pune / Fundjavë",
                    "CV",
                    "Load Factor",
                    "Indeksi sezonal",
                    "Trendi (%)",
                ],
                "Portofoli i kombinuar": [
                    selected["peak_ratio"],
                    selected["weekday_weekend_ratio"],
                    selected["cv"],
                    selected["load_factor"],
                    selected["seasonality_index"],
                    selected["trend_percent"],
                ],
                "Mediana e kompanive": [
                    profiles["peak_ratio"].median(),
                    profiles["weekday_weekend_ratio"].median(),
                    profiles["cv"].median(),
                    profiles["load_factor"].median(),
                    profiles["seasonality_index"].median(),
                    profiles["trend_percent"].median(),
                ],
            }
        )

        st.dataframe(
            portfolio_summary,
            width="stretch",
            hide_index=True,
            column_config={
                "Portofoli i kombinuar": st.column_config.NumberColumn(format="%.3f"),
                "Mediana e kompanive": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    else:
        analysis_description(
            title="Krahasimi me portofolin",
            description=(
                "Secila metrikë vendoset në percentilin e saj "
                f"kundrejt {len(profiles)} kompanive që kaluan "
                "kontrollin e cilësisë."
            ),
            purpose=(
                "Të kuptohet se cilat karakteristika të kompanisë "
                "janë tipike dhe cilat janë relativisht ekstreme "
                "në portofol."
            ),
            finding=(
                cv_finding(
                    selected,
                    profiles,
                )
                + " "
                + load_factor_finding(
                    selected,
                    profiles,
                )
            ),
        )

        render_plot(
            metric_percentile_figure(
                profiles,
                selected,
            ),
            "company-profile-percentiles",
        )

        st.caption(
            "Percentili paraqet pozicionin relativ të kompanisë "
            "në portofol. Percentil më i lartë nuk do të thotë "
            "automatikisht performancë më e mirë ose më e keqe."
        )


with table_tab:
    analysis_description(
        title="Tabela e profileve të kompanive",
        description=(
            "Tabela përmbledh metrikat e Hapit 3 për të gjitha "
            "kompanitë që kanë të dhëna të mjaftueshme për "
            "analizën vjetore."
        ),
        purpose=(
            "Të mundësohet krahasimi dhe inspektimi i drejtpërdrejtë "
            "i profileve të kompanive."
        ),
        finding=(f"Në këtë analizë janë përfshirë " f"{len(profiles)} kompani."),
    )

    filter_cols = st.columns(2)

    seasonality_options = [
        "Të gjitha",
        "Dimër",
        "Verë",
        "Pa sezonalitet të qartë",
    ]

    selected_seasonality = filter_cols[0].selectbox(
        "Sezonaliteti",
        seasonality_options,
        key="company-profile-seasonality-filter",
    )

    trend_options = [
        "Të gjitha",
        "Rritje",
        "Rënie",
    ]

    selected_trend = filter_cols[1].selectbox(
        "Drejtimi i trendit",
        trend_options,
        key="company-profile-trend-filter",
    )

    table_data = profiles.copy()

    table_data["seasonality_label"] = (
        table_data["seasonality"]
        .map(SEASONALITY_LABELS)
        .fillna(table_data["seasonality"])
    )

    if selected_seasonality != "Të gjitha":
        table_data = table_data[table_data["seasonality_label"] == selected_seasonality]

    if selected_trend == "Rritje":
        table_data = table_data[table_data["trend_percent"] > 0]

    elif selected_trend == "Rënie":
        table_data = table_data[table_data["trend_percent"] < 0]

    table_data = sort_by_company(table_data)

    display = table_data[
        [
            "company_code",
            "meter_count_used",
            "total_kwh",
            "peak_ratio",
            "weekday_weekend_ratio",
            "cv",
            "load_factor",
            "seasonality_label",
            "seasonality_index",
            "trend_percent",
        ]
    ].rename(
        columns={
            "company_code": "Kompania",
            "meter_count_used": "Njehsorë",
            "total_kwh": "Konsumi total (kWh)",
            "peak_ratio": "Peak / Off-Peak",
            "weekday_weekend_ratio": ("Ditë Pune / Fundjavë"),
            "cv": "CV",
            "load_factor": "Load Factor",
            "seasonality_label": "Sezonaliteti",
            "seasonality_index": ("Indeksi sezonal"),
            "trend_percent": "Trendi (%)",
        }
    )

    st.caption(f"Po shfaqen {len(display)} kompani.")

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Konsumi total (kWh)": (
                st.column_config.NumberColumn(
                    format="%.2f",
                )
            ),
            "Peak / Off-Peak": (
                st.column_config.NumberColumn(
                    format="%.3f",
                )
            ),
            "Ditë Pune / Fundjavë": (
                st.column_config.NumberColumn(
                    format="%.3f",
                )
            ),
            "CV": (
                st.column_config.NumberColumn(
                    format="%.3f",
                )
            ),
            "Load Factor": (
                st.column_config.NumberColumn(
                    format="%.3f",
                )
            ),
            "Indeksi sezonal": (
                st.column_config.NumberColumn(
                    format="%.3f",
                )
            ),
            "Trendi (%)": (
                st.column_config.NumberColumn(
                    format="%.2f",
                )
            ),
        },
    )
