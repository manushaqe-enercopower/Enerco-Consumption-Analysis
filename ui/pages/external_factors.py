from __future__ import annotations

import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

UI_DIR = Path(__file__).resolve().parents[1]

ROOT = UI_DIR.parent

if str(UI_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(UI_DIR),
    )


from components.charting import (
    PLOTLY_CONFIG,
    discrete_colorscale,
    format_energy,
    hex_to_rgba,
    series_color,
    sort_by_company,
    style_figure,
    tariff_color,
)

TARIFF_DIR = ROOT / "data" / "processed" / "factors" / "tariff"

HOLIDAY_DIR = ROOT / "data" / "processed" / "factors" / "holidays"

WEATHER_DIR = ROOT / "data" / "processed" / "factors" / "weather"


DATA_FILES = {
    "tariff_portfolio": (TARIFF_DIR / "tariff_portfolio_summary.parquet"),
    "tariff_company": (TARIFF_DIR / "tariff_company.parquet"),
    "tariff_monthly": (TARIFF_DIR / "tariff_monthly.parquet"),
    "tariff_hourly": (TARIFF_DIR / "tariff_hourly_profile.parquet"),
    "tariff_schedule": (TARIFF_DIR / "tariff_schedule.parquet"),
    "holiday_portfolio": (HOLIDAY_DIR / "holiday_portfolio_impact.parquet"),
    "holiday_company": (HOLIDAY_DIR / "holiday_company_impact.parquet"),
    "holiday_hourly": (HOLIDAY_DIR / "holiday_hourly_impact.parquet"),
    "holiday_outliers": (HOLIDAY_DIR / "holiday_outliers.parquet"),
    "holiday_outlier_summary": (HOLIDAY_DIR / "holiday_outlier_summary.parquet"),
    "weather_portfolio": (WEATHER_DIR / "weather_portfolio_daily.parquet"),
    "weather_summary": (WEATHER_DIR / "weather_portfolio_summary.parquet"),
    "weather_company": (WEATHER_DIR / "weather_company_sensitivity.parquet"),
    "weather_monthly": (WEATHER_DIR / "weather_monthly.parquet"),
    "weather_response": (WEATHER_DIR / "weather_temperature_response.parquet"),
}


@st.cache_data(show_spinner=False)
def read_parquet(
    path: str,
) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_data() -> dict[str, pd.DataFrame]:
    missing = [path for path in DATA_FILES.values() if not path.exists()]

    if missing:
        st.error(
            "Some analytical outputs required by this page " "have not been generated."
        )

        st.code("\n".join(str(path.relative_to(ROOT)) for path in missing))

        st.stop()

    return {key: read_parquet(str(path)) for key, path in DATA_FILES.items()}


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


def tariff_monthly_figure(
    monthly: pd.DataFrame,
) -> go.Figure:
    data = monthly.copy()

    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )

    tariffs = sorted(data["tariff"].dropna().unique())

    for index, tariff in enumerate(tariffs):
        current = data[data["tariff"] == tariff].sort_values("month")

        fig.add_trace(
            go.Bar(
                x=current["month"],
                y=(current["total_kwh"] / 1_000_000),
                name=f"{tariff} energji",
                marker_color=tariff_color(
                    tariff,
                    index,
                ),
                customdata=np.column_stack(
                    [
                        current["tariff_share_percent"],
                        current["active_hours"],
                    ]
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>" + tariff + ": %{y:.2f} GWh<br>"
                    "Pjesa mujore: %{customdata[0]:.1f}%<br>"
                    "Orë aktive: %{customdata[1]:,.0f}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if "T1" in tariffs:
        t1 = data[data["tariff"] == "T1"].sort_values("month")

        fig.add_trace(
            go.Scatter(
                x=t1["month"],
                y=t1["tariff_share_percent"],
                mode="lines+markers",
                name="Pjesa T1",
                line={
                    "width": 3,
                    "color": tariff_color("T1"),
                },
                hovertemplate=(
                    "<b>%{x}</b><br>" "Pjesa T1: %{y:.1f}%" "<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        barmode="stack",
    )

    fig.update_yaxes(
        title_text="Energji (GWh)",
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="Pjesa T1 (%)",
        range=[
            0,
            100,
        ],
        secondary_y=True,
        showgrid=False,
    )

    return style_figure(
        fig,
        title="Përbërja mujore sipas tarifës dhe pjesa T1",
        x_title="Muaji",
        height=520,
        hovermode="x unified",
    )


def tariff_schedule_figure(
    schedule: pd.DataFrame,
) -> go.Figure:
    data = schedule.copy()

    data = data[data["is_dominant_tariff"].astype(bool)].copy()

    data = data.sort_values(
        "occurrence_count",
        ascending=False,
    ).drop_duplicates(
        [
            "month_number",
            "hour",
        ]
    )

    tariffs = sorted(data["tariff"].astype(str).unique())

    tariff_codes = {tariff: index for index, tariff in enumerate(tariffs)}

    data["tariff_code"] = data["tariff"].astype(str).map(tariff_codes)

    pivot = data.pivot(
        index="month_number",
        columns="hour",
        values="tariff_code",
    )

    text = data.assign(tariff_text=data["tariff"].astype(str)).pivot(
        index="month_number",
        columns="hour",
        values="tariff_text",
    )

    pivot = pivot.sort_index()
    text = text.reindex(pivot.index)

    month_labels = [calendar.month_abbr[int(month)] for month in pivot.index]

    colors = [
        tariff_color(
            tariff,
            index,
        )
        for index, tariff in enumerate(tariffs)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=month_labels,
            text=text.values,
            zmin=-0.5,
            zmax=max(
                len(tariffs) - 0.5,
                0.5,
            ),
            colorscale=discrete_colorscale(colors),
            colorbar={
                "title": "Tarifa",
                "tickvals": list(tariff_codes.values()),
                "ticktext": list(tariff_codes.keys()),
            },
            hovertemplate=(
                "Muaji: %{y}<br>"
                "Ora: %{x}<br>"
                "Tarifa dominuese: %{text}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_yaxes(autorange="reversed")

    return style_figure(
        fig,
        title="Orari tarifor i vërejtur — muaj × orë",
        x_title="Ora",
        y_title="Muaji",
        height=500,
    )


def tariff_hourly_figure(
    hourly: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    tariffs = sorted(hourly["tariff"].dropna().unique())

    for index, tariff in enumerate(tariffs):
        current = hourly[hourly["tariff"] == tariff].sort_values("hour")

        color = tariff_color(
            tariff,
            index,
        )

        fig.add_trace(
            go.Scatter(
                x=current["hour"],
                y=(current["p10_portfolio_kwh"] / 1_000),
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
                x=current["hour"],
                y=(current["p90_portfolio_kwh"] / 1_000),
                mode="lines",
                fill="tonexty",
                fillcolor=hex_to_rgba(
                    color,
                    0.15,
                ),
                line={
                    "width": 0,
                },
                name=f"{tariff} P10–P90",
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=current["hour"],
                y=(current["mean_portfolio_kwh"] / 1_000),
                mode="lines+markers",
                name=f"{tariff} mesatarja",
                line={
                    "width": 3,
                    "color": color,
                },
                marker={
                    "size": 6,
                },
                customdata=current["observation_hours"],
                hovertemplate=(
                    "Ora %{x}<br>"
                    "Mesatarja: %{y:,.1f} MWh<br>"
                    "Vëzhgime: %{customdata:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    return style_figure(
        fig,
        title="Ngarkesa orare e portofolit sipas tarifës me intervalin P10–P90",
        x_title="Ora",
        y_title="Ngarkesa e portofolit (MWh)",
        height=520,
        hovermode="x unified",
    )


def tariff_company_figure(
    company_tariff: pd.DataFrame,
    portfolio_t1_share: float,
) -> go.Figure:
    data = company_tariff.copy()

    shares = data.pivot_table(
        index="company_code",
        columns="tariff",
        values="tariff_share_percent",
        aggfunc="first",
        fill_value=0,
    )

    totals = data.groupby("company_code")["company_total_kwh"].first()

    plot_data = shares.join(totals).reset_index()

    plot_data["annual_gwh"] = plot_data["company_total_kwh"] / 1_000_000

    plot_data["t1_share"] = plot_data["T1"] if "T1" in plot_data.columns else 0.0

    plot_data["t2_share"] = plot_data["T2"] if "T2" in plot_data.columns else 0.0

    plot_data["dominant_tariff"] = np.where(
        plot_data["t1_share"] >= plot_data["t2_share"],
        "T1",
        "T2",
    )

    plot_data = plot_data[plot_data["annual_gwh"] > 0].copy()

    fig = px.scatter(
        plot_data,
        x="annual_gwh",
        y="t1_share",
        size="annual_gwh",
        color="dominant_tariff",
        hover_name="company_code",
        hover_data={
            "annual_gwh": ":.3f",
            "t1_share": ":.2f",
            "t2_share": ":.2f",
            "company_total_kwh": False,
        },
        labels={
            "annual_gwh": "Konsumi vjetor (GWh)",
            "t1_share": "Pjesa T1 (%)",
            "dominant_tariff": "Tarifa dominuese",
        },
        color_discrete_map={
            "T1": tariff_color("T1"),
            "T2": tariff_color("T2"),
        },
        size_max=32,
    )

    fig.add_hline(
        y=portfolio_t1_share,
        line_dash="dash",
        annotation_text=(f"Pjesa T1 e portofolit {portfolio_t1_share:.1f}%"),
        annotation_position="top left",
    )

    fig.update_xaxes(type="log")

    fig.update_yaxes(
        range=[
            0,
            100,
        ]
    )

    return style_figure(
        fig,
        title="Pozicionimi tarifor i kompanive",
        x_title="Konsumi vjetor (GWh, shkallë logaritmike)",
        y_title="Pjesa T1 (%)",
        height=560,
    )


def holiday_impact_figure(
    impact: pd.DataFrame,
) -> go.Figure:
    data = impact.sort_values("date").copy()

    labels = data["date"].dt.strftime("%d %b %Y") + " · " + data["holiday_name"]

    colors = np.where(
        data["impact_percent"] < 0,
        "#EF4444",
        "#10B981",
    )

    custom = np.column_stack(
        [
            data["portfolio_kwh"] / 1_000_000,
            data["baseline_mean"] / 1_000_000,
        ]
    )

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=data["impact_percent"],
            marker_color=colors,
            customdata=custom,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Ndikimi: %{y:.1f}%<br>"
                "Konsumi real: %{customdata[0]:.3f} GWh<br>"
                "Baseline i krahasueshëm: %{customdata[1]:.3f} GWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_width=1,
    )

    return style_figure(
        fig,
        title="Ndikimi i konsumit gjatë festave kundrejt baseline-it",
        x_title=None,
        y_title="Ndikimi (%)",
        height=560,
    )


def holiday_hourly_figure(
    hourly: pd.DataFrame,
) -> go.Figure:
    data = hourly.sort_values("hour")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["hour"],
            y=(data["baseline_p10_kwh"] / 1_000),
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
            y=(data["baseline_p90_kwh"] / 1_000),
            mode="lines",
            fill="tonexty",
            fillcolor=("rgba(128,128,128,0.18)"),
            line={
                "width": 0,
            },
            name="Ditë normale P10–P90",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["hour"],
            y=(data["baseline_mean_kwh"] / 1_000),
            mode="lines",
            name="Mesatarja e ditëve normale",
            line={
                "width": 2,
                "dash": "dash",
            },
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["hour"],
            y=(data["portfolio_kwh"] / 1_000),
            mode="lines+markers",
            name="Konsumi real i festës",
            line={
                "width": 3,
            },
        )
    )

    return style_figure(
        fig,
        title="Profili orar i festës kundrejt ditëve normale të krahasueshme",
        x_title="Ora",
        y_title="Ngarkesa e portofolit (MWh)",
        height=520,
        hovermode="x unified",
    )


def holiday_company_distribution(
    company_impact: pd.DataFrame,
) -> go.Figure:
    data = company_impact.dropna(subset=["impact_percent"])

    fig = go.Figure(
        go.Violin(
            y=data["impact_percent"],
            text=data["company_code"],
            box_visible=True,
            meanline_visible=True,
            points="all",
            jitter=0.25,
            pointpos=0,
            hovertemplate=("%{text}<br>" "Ndikimi: %{y:.1f}%" "<extra></extra>"),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
    )

    return style_figure(
        fig,
        title="Shpërndarja e ndikimit të festës sipas kompanisë",
        y_title="Ndikimi kundrejt baseline-it (%)",
        height=520,
    )


def weather_response_figure(
    portfolio: pd.DataFrame,
    response: pd.DataFrame,
) -> go.Figure:
    daily = portfolio.sort_values("date")

    curve = response.dropna(
        subset=[
            "mean_temperature_c",
            "median_portfolio_kwh",
        ]
    ).sort_values("mean_temperature_c")

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=daily["temperature_mean_c"],
            y=(daily["portfolio_kwh"] / 1_000_000),
            mode="markers",
            name="Vëzhgimet ditore",
            marker={
                "size": 7,
                "opacity": 0.40,
            },
            text=daily["date"].dt.strftime("%d %b %Y"),
            customdata=np.column_stack(
                [
                    daily["hdd"],
                    daily["cdd"],
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Temperatura: %{x:.1f}°C<br>"
                "Konsumi: %{y:.3f} GWh<br>"
                "HDD: %{customdata[0]:.1f}<br>"
                "CDD: %{customdata[1]:.1f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=curve["mean_temperature_c"],
            y=(curve["p10_portfolio_kwh"] / 1_000_000),
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
            x=curve["mean_temperature_c"],
            y=(curve["p90_portfolio_kwh"] / 1_000_000),
            mode="lines",
            fill="tonexty",
            fillcolor=("rgba(37,99,235,0.15)"),
            line={
                "width": 0,
            },
            name="Intervali i temperaturës P10–P90",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=curve["mean_temperature_c"],
            y=(curve["median_portfolio_kwh"] / 1_000_000),
            mode="lines+markers",
            name="Mediana sipas intervalit të temperaturës",
            line={
                "width": 3,
            },
        )
    )

    return style_figure(
        fig,
        title="Reagimi i konsumit të portofolit ndaj temperaturës",
        x_title="Temperatura mesatare ditore (°C)",
        y_title="Konsumi ditor (GWh)",
        height=580,
    )


def weather_monthly_figure(
    monthly: pd.DataFrame,
) -> go.Figure:
    data = monthly.sort_values("month")

    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )

    fig.add_trace(
        go.Bar(
            x=data["month"],
            y=data["total_hdd"],
            name="HDD",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=data["month"],
            y=data["total_cdd"],
            name="CDD",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=data["month"],
            y=(data["total_portfolio_kwh"] / 1_000_000),
            name="Konsumi",
            mode="lines+markers",
            line={
                "width": 3,
            },
        ),
        secondary_y=True,
    )

    fig.update_layout(barmode="group")

    fig.update_yaxes(
        title_text="Degree days",
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="Konsumi mujor (GWh)",
        secondary_y=True,
        showgrid=False,
    )

    return style_figure(
        fig,
        title="Kërkesa mujore për ngrohje/ftohje kundrejt konsumit",
        x_title="Muaji",
        height=540,
        hovermode="x unified",
    )


def weather_sensitivity_figure(
    sensitivity: pd.DataFrame,
) -> go.Figure:
    data = sensitivity.dropna(
        subset=[
            "hdd_correlation",
            "cdd_correlation",
            "mean_daily_kwh",
        ]
    ).copy()

    data = data[data["mean_daily_kwh"] > 0]

    fig = px.scatter(
        data,
        x="hdd_correlation",
        y="cdd_correlation",
        color=("dominant_weather_response"),
        size="mean_daily_kwh",
        size_max=32,
        hover_name="company_code",
        hover_data={
            "temperature_correlation": ":.3f",
            "hdd_correlation": ":.3f",
            "cdd_correlation": ":.3f",
            "heating_kwh_per_hdd": ":.2f",
            "cooling_kwh_per_cdd": ":.2f",
            "mean_daily_kwh": ":,.1f",
        },
        labels={
            "hdd_correlation": "Korelacioni HDD",
            "cdd_correlation": "Korelacioni CDD",
            "dominant_weather_response": "Reagimi dominues",
            "mean_daily_kwh": "Konsumi mesatar ditor (kWh)",
        },
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
    )

    fig.update_xaxes(
        range=[
            -1,
            1,
        ]
    )

    fig.update_yaxes(
        range=[
            -1,
            1,
        ]
    )

    return style_figure(
        fig,
        title="Ndjeshmëria e kompanive ndaj ngrohjes dhe ftohjes",
        x_title="Korelacioni me HDD",
        y_title="Korelacioni me CDD",
        height=600,
    )


def weather_timeline_figure(
    portfolio: pd.DataFrame,
) -> go.Figure:
    data = portfolio.sort_values("date").copy()

    data["consumption_7d_gwh"] = (
        data["portfolio_kwh"]
        .rolling(
            7,
            center=True,
            min_periods=3,
        )
        .mean()
        / 1_000_000
    )

    data["temperature_7d"] = (
        data["temperature_mean_c"]
        .rolling(
            7,
            center=True,
            min_periods=3,
        )
        .mean()
    )

    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["consumption_7d_gwh"],
            mode="lines",
            name="Konsumi — mesatare 7-ditore",
            line={
                "width": 3,
            },
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["temperature_7d"],
            mode="lines",
            name="Temperatura — mesatare 7-ditore",
            line={
                "width": 2,
            },
        ),
        secondary_y=True,
    )

    fig.update_yaxes(
        title_text="Konsumi ditor (GWh)",
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="Temperatura (°C)",
        secondary_y=True,
        showgrid=False,
    )

    return style_figure(
        fig,
        title="Konsumi dhe temperatura gjatë periudhës së analizuar",
        x_title="Data",
        height=520,
        hovermode="x unified",
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


data = load_data()


st.title("Faktorët shtesë")

st.caption(
    "Analiza e tarifave, festave zyrtare " "dhe ndikimit të motit në konsum"
)


tariff_portfolio = data["tariff_portfolio"]

holiday_portfolio = data["holiday_portfolio"]

weather_summary = data["weather_summary"]


t1_rows = tariff_portfolio[tariff_portfolio["tariff"] == "T1"]

t1_share = (
    float(t1_rows.iloc[0]["energy_share_percent"]) if not t1_rows.empty else np.nan
)

holiday_average = float(holiday_portfolio["impact_percent"].mean())

summary_row = weather_summary.iloc[0]

hdd_corr = float(summary_row["hdd_correlation"])

cdd_corr = float(summary_row["cdd_correlation"])


summary_cols = st.columns(4)

summary_cols[0].metric(
    "Pjesa e energjisë në T1",
    f"{t1_share:.2f}%",
)

summary_cols[1].metric(
    "Ndikimi mesatar i festave",
    f"{holiday_average:.2f}%",
)

summary_cols[2].metric(
    "Korelacioni HDD",
    f"{hdd_corr:.3f}",
)

summary_cols[3].metric(
    "Korelacioni CDD",
    f"{cdd_corr:.3f}",
)


st.markdown(
    """
    <div class="analysis-note">
        Këta faktorë përdoren për të shpjeguar sjelljen sistematike
        të konsumit dhe për të zvogëluar interpretimin e gabuar të
        anomalive. Korelacioni paraqet lidhje statistikore dhe nuk
        duhet të interpretohet automatikisht si shkakësi.
    </div>
    """,
    unsafe_allow_html=True,
)


tariff_tab, holiday_tab, weather_tab = st.tabs(
    [
        "Tarifat",
        "Festat zyrtare",
        "Moti / HDD / CDD",
    ]
)


with tariff_tab:
    st.subheader("Analiza e tarifave")

    st.caption(
        "Analiza e konsumit të portofolit dhe kompanive sipas "
        "periudhave tarifore të regjistruara në të dhënat burimore."
    )

    total_kwh = float(tariff_portfolio["total_kwh"].sum())

    t2_rows = tariff_portfolio[tariff_portfolio["tariff"] == "T2"]

    t2_share = (
        float(t2_rows.iloc[0]["energy_share_percent"]) if not t2_rows.empty else np.nan
    )

    (
        tariff_overview_tab,
        tariff_monthly_tab,
        tariff_schedule_tab,
        tariff_hourly_tab,
        tariff_company_tab,
    ) = st.tabs(
        [
            "Përmbledhje",
            "Përbërja mujore",
            "Orari tarifor",
            "Profili orar",
            "Pozicionimi i kompanive",
        ]
    )

    with tariff_overview_tab:
        analysis_description(
            title="Struktura tarifore e portofolit",
            description=(
                "Kjo pamje përmbledh mënyrën se si konsumi total vjetor "
                "i energjisë elektrike shpërndahet ndërmjet periudhave "
                "tarifore të disponueshme në të dhënat e EnerCo."
            ),
            purpose=(
                "Të përcaktohet se cila periudhë tarifore përfaqëson "
                "pjesën më të madhe të konsumit të portofolit."
            ),
            finding=(
                f"T1 përfaqëson {t1_share:.2f}% të konsumit total, "
                f"ndërsa T2 përfaqëson {t2_share:.2f}%. "
                "Portofoli është më i përqendruar në T1."
            ),
        )

        cols = st.columns(4)

        cols[0].metric(
            "Energjia e portofolit",
            format_energy(total_kwh),
        )

        cols[1].metric(
            "Pjesa T1",
            f"{t1_share:.2f}%",
        )

        cols[2].metric(
            "Pjesa T2",
            f"{t2_share:.2f}%",
        )

        cols[3].metric(
            "Kompanitë",
            int(data["tariff_company"]["company_code"].nunique()),
        )

        summary_fig = px.pie(
            tariff_portfolio,
            names="tariff",
            values="total_kwh",
            hole=0.58,
            color="tariff",
            color_discrete_map={
                "T1": tariff_color("T1"),
                "T2": tariff_color("T2"),
            },
        )

        summary_fig.update_traces(
            textposition="inside",
            textinfo="label+percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Energji: %{value:,.0f} kWh<br>"
                "Pjesa: %{percent}"
                "<extra></extra>"
            ),
        )

        summary_fig = style_figure(
            summary_fig,
            title="Përbërja vjetore e konsumit sipas tarifës",
            height=480,
        )

        render_plot(
            summary_fig,
            "tariff-overview",
        )

    with tariff_monthly_tab:
        analysis_description(
            title="Përbërja mujore sipas tarifës",
            description=(
                "Konsumi mujor në T1 dhe T2 paraqitet së bashku me "
                "përqindjen e energjisë mujore që i përket tarifës T1."
            ),
            purpose=(
                "Të identifikohet nëse struktura tarifore e konsumit "
                "ndryshon sipas sezonit apo mbetet relativisht stabile."
            ),
            finding=(
                "Linja e pjesës T1 tregon nëse ndryshimet sezonale në "
                "konsumin total shoqërohen edhe me ndryshim të "
                "strukturës tarifore."
            ),
        )

        render_plot(
            tariff_monthly_figure(data["tariff_monthly"]),
            "tariff-monthly",
        )

        with st.expander("Shfaq të dhënat mujore të tarifave"):
            st.dataframe(
                data["tariff_monthly"],
                width="stretch",
                hide_index=True,
            )

    with tariff_schedule_tab:
        analysis_description(
            title="Orari tarifor i vërejtur",
            description=(
                "Heatmap-i rindërton tarifën e regjistruar për çdo muaj "
                "dhe orë drejtpërdrejt nga të dhënat burimore."
            ),
            purpose=(
                "Të verifikohet se kur aplikohen T1 dhe T2 dhe nëse "
                "orari tarifor ndryshon gjatë vitit."
            ),
            finding=(
                "Grafiku ndërtohet nga kolona reale Tariff dhe nuk "
                "supozon paraprakisht një orar tarifor të paracaktuar."
            ),
        )

        render_plot(
            tariff_schedule_figure(data["tariff_schedule"]),
            "tariff-schedule",
        )

    with tariff_hourly_tab:
        analysis_description(
            title="Ngarkesa orare sipas tarifës",
            description=(
                "Linjat paraqesin konsumin mesatar të portofolit për "
                "secilën orë dhe tarifë. Zona e hijezuar paraqet "
                "intervalin P10–P90 të ngarkesës së vërejtur."
            ),
            purpose=(
                "Të krahasohet niveli tipik dhe ndryshueshmëria e "
                "ngarkesës gjatë periudhave të ndryshme tarifore."
            ),
            finding=(
                "Intervali P10–P90 ndihmon të dallohen orët me konsum "
                "relativisht stabil nga ato me ndryshueshmëri më të lartë."
            ),
        )

        render_plot(
            tariff_hourly_figure(data["tariff_hourly"]),
            "tariff-hourly",
        )

    with tariff_company_tab:
        analysis_description(
            title="Pozicionimi tarifor i kompanive",
            description=(
                "Çdo pikë paraqet një kompani. Boshti horizontal tregon "
                "konsumin vjetor, ndërsa boshti vertikal tregon përqindjen "
                "e konsumit të kompanisë që ndodh në T1."
            ),
            purpose=(
                "Të identifikohen kompanitë me strukturë tarifore që "
                "ndryshon dukshëm nga portofoli i përgjithshëm i EnerCo."
            ),
            finding=(
                f"Linja referente paraqet pjesën T1 të portofolit prej "
                f"{t1_share:.2f}%. Kompanitë dukshëm mbi ose nën këtë "
                "linjë kanë profile tarifore të ndryshme nga mesatarja."
            ),
        )

        render_plot(
            tariff_company_figure(
                data["tariff_company"],
                portfolio_t1_share=t1_share,
            ),
            "tariff-company",
        )

        with st.expander("Shfaq të dhënat e tarifave sipas kompanisë"):
            st.dataframe(
                sort_by_company(data["tariff_company"]),
                width="stretch",
                hide_index=True,
            )


with holiday_tab:
    st.subheader("Analiza e festave zyrtare")

    holiday_company = data["holiday_company"]

    holiday_outliers = data["holiday_outliers"]

    (
        holiday_overview_tab,
        holiday_impact_tab,
        holiday_hourly_tab,
        holiday_company_tab,
        holiday_outlier_tab,
    ) = st.tabs(
        [
            "Përmbledhje",
            "Ndikimi i festave",
            "Profili orar",
            "Ndikimi sipas kompanisë",
            "Ndërlidhja me outlier-at",
        ]
    )

    with holiday_overview_tab:
        analysis_description(
            title="Sjellja e konsumit gjatë festave",
            description=(
                "Festat zyrtare të Kosovës krahasohen me ditë normale "
                "që kanë të njëjtin muaj dhe të njëjtën ditë të javës."
            ),
            purpose=(
                "Të ndahen efektet e përsëritshme të kalendarit nga "
                "devijimet reale dhe të pazakonta të konsumit."
            ),
            finding=(
                f"Gjatë periudhës së analizuar, festat zyrtare e kanë "
                f"ulur konsumin e portofolit mesatarisht me "
                f"{abs(holiday_portfolio['impact_percent'].mean()):.2f}% "
                f"dhe medianën me "
                f"{abs(holiday_portfolio['impact_percent'].median()):.2f}%."
            ),
        )

        cols = st.columns(4)

        cols[0].metric(
            "Festa të analizuara",
            int(holiday_portfolio["date"].nunique()),
        )

        cols[1].metric(
            "Ndikimi mesatar",
            f"{holiday_portfolio['impact_percent'].mean():.2f}%",
        )

        cols[2].metric(
            "Ndikimi median",
            f"{holiday_portfolio['impact_percent'].median():.2f}%",
        )

        cols[3].metric(
            "Orë outlier gjatë festave",
            f"{len(holiday_outliers):,}",
        )

    with holiday_impact_tab:
        analysis_description(
            title="Ndikimi i festave individuale",
            description=(
                "Secila festë krahasohet me baseline-in e saj të "
                "përshtatur nga ditë normale të krahasueshme."
            ),
            purpose=(
                "Të përcaktohet se cilat festa shkaktojnë ndryshimet "
                "më të mëdha sistematike në konsumin e portofolit."
            ),
            finding=(
                "Vlerat negative tregojnë konsum më të ulët se niveli "
                "i pritur për një ditë normale të krahasueshme."
            ),
        )

        render_plot(
            holiday_impact_figure(holiday_portfolio),
            "holiday-impact",
        )

    with holiday_hourly_tab:
        analysis_description(
            title="Profili orar i konsumit gjatë festës",
            description=(
                "Zgjidhni një festë zyrtare për të krahasuar kurbën "
                "e saj orare të konsumit me profilin normal të pritur."
            ),
            purpose=(
                "Të identifikohen orët që kontribuojnë më së shumti "
                "në uljen ose rritjen e konsumit gjatë festës."
            ),
        )

        holiday_options = (
            holiday_portfolio[
                [
                    "date",
                    "holiday_name",
                ]
            ]
            .drop_duplicates()
            .sort_values("date")
        )

        holiday_options["label"] = (
            holiday_options["date"].dt.strftime("%d %b %Y")
            + " — "
            + holiday_options["holiday_name"]
        )

        option_map = dict(
            zip(
                holiday_options["label"],
                holiday_options["date"],
            )
        )

        selected_label = st.selectbox(
            "Zgjidh festën",
            list(option_map.keys()),
            key="holiday-hourly-selector",
        )

        selected_date = option_map[selected_label]

        selected_hourly = data["holiday_hourly"][
            data["holiday_hourly"]["date"] == selected_date
        ]

        render_plot(
            holiday_hourly_figure(selected_hourly),
            "holiday-hourly",
        )

    with holiday_company_tab:
        analysis_description(
            title="Ndikimi i festës sipas kompanisë",
            description=(
                "Shpërndarja tregon se si ka reaguar secila kompani "
                "gjatë festës krahasuar me baseline-in e vet."
            ),
            purpose=(
                "Të përcaktohet nëse ndikimi i përgjithshëm i festës "
                "është i përhapur në shumicën e kompanive apo shkaktohet "
                "vetëm nga një grup i vogël."
            ),
        )

        selected_company = holiday_company[holiday_company["date"] == selected_date]

        render_plot(
            holiday_company_distribution(selected_company),
            "holiday-company-distribution",
        )

    with holiday_outlier_tab:
        analysis_description(
            title="Ndërlidhja e festave me outlier-at",
            description=(
                "Ky vizualizim lidh festat zyrtare me rezultatet e "
                "detektimit të outlier-ave nga Hapi 4. Për secilën festë "
                "paraqitet numri i orëve të klasifikuara si outlier dhe "
                "numri i kompanive të prekura."
            ),
            purpose=(
                "Të identifikohen rastet ku një devijim i konsumit mund "
                "të shpjegohet nga një festë zyrtare, në vend që të "
                "interpretohet menjëherë si problem teknik ose sjellje "
                "e pazakontë e kompanisë."
            ),
            finding=(
                f"Gjatë festave zyrtare janë identifikuar "
                f"{len(holiday_outliers):,} orë outlier, të shpërndara në "
                f"{holiday_outliers['company_code'].nunique()} kompani. "
                "Këto raste duhet të interpretohen në kontekst të kalendarit "
                "përpara se të konsiderohen problem teknik."
            ),
        )

        outlier_summary = data["holiday_outlier_summary"].copy()

        if not outlier_summary.empty:
            outlier_summary = outlier_summary.sort_values(
                "outlier_hours",
                ascending=False,
            )

            fig = px.bar(
                outlier_summary,
                x="holiday_name",
                y="outlier_hours",
                text="companies",
                hover_data={
                    "date": True,
                    "companies": True,
                    "max_abs_z_score": ":.2f",
                },
                labels={
                    "holiday_name": "Festa zyrtare",
                    "outlier_hours": "Orë outlier",
                    "companies": "Kompanitë e prekura",
                    "date": "Data",
                    "max_abs_z_score": "Maksimumi |Z|",
                },
            )

            fig.update_traces(
                texttemplate="%{text} kompani",
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Orë outlier: %{y:,.0f}<br>"
                    "Kompanitë e prekura: %{text}<br>"
                    "<extra></extra>"
                ),
            )

            fig = style_figure(
                fig,
                title=("Outlier-at e detektuar gjatë festave zyrtare"),
                x_title="Festa zyrtare",
                y_title="Numri i orëve outlier",
                height=520,
            )

            render_plot(
                fig,
                "holiday-outliers",
            )


with weather_tab:
    st.subheader("Analiza e motit / HDD / CDD")

    weather_portfolio = data["weather_portfolio"]

    weather_company = data["weather_company"]

    (
        weather_overview_tab,
        weather_timeline_tab,
        weather_response_tab,
        weather_degree_days_tab,
        weather_company_tab,
    ) = st.tabs(
        [
            "Përmbledhje",
            "Konsumi dhe temperatura",
            "Reagimi ndaj temperaturës",
            "HDD / CDD",
            "Ndjeshmëria e kompanive",
        ]
    )

    with weather_overview_tab:
        analysis_description(
            title="Lidhja e portofolit me motin",
            description=(
                "Temperatura historike e Prishtinës përdoret si pikë "
                "referimi e përbashkët, sepse të dhënat për distriktin "
                "e secilës kompani nuk janë të disponueshme."
            ),
            purpose=(
                "Të matet nëse konsumi i portofolit ndryshon në mënyrë "
                "sistematike me nevojën për ngrohje ose ftohje."
            ),
            finding=(
                f"Korelacioni i konsumit me temperaturën është "
                f"{summary_row['temperature_correlation']:.3f}. "
                f"Korelacioni HDD është {hdd_corr:.3f}, ndërsa "
                f"CDD është {cdd_corr:.3f}. Ndikimi i lidhur me "
                "ngrohjen është më i fortë në këtë portofol."
            ),
        )

        cols = st.columns(5)

        cols[0].metric(
            "Lokacioni referues",
            "Prishtina",
        )

        cols[1].metric(
            "Temperatura mesatare",
            f"{summary_row['mean_temperature_c']:.2f}°C",
        )

        cols[2].metric(
            "Korelacioni me temperaturën",
            f"{summary_row['temperature_correlation']:.3f}",
        )

        cols[3].metric(
            "Korelacioni HDD",
            f"{hdd_corr:.3f}",
        )

        cols[4].metric(
            "Korelacioni CDD",
            f"{cdd_corr:.3f}",
        )

    with weather_timeline_tab:
        analysis_description(
            title="Konsumi dhe temperatura gjatë kohës",
            description=(
                "Mesataret lëvizëse 7-ditore zvogëlojnë zhurmën ditore "
                "dhe e bëjnë më të dukshme lidhjen sezonale ndërmjet "
                "temperaturës dhe konsumit të energjisë."
            ),
            purpose=(
                "Të kontrollohet nëse rritjet dhe rëniet e konsumit "
                "përputhen me ndryshimet kryesore të temperaturës gjatë vitit."
            ),
        )

        render_plot(
            weather_timeline_figure(weather_portfolio),
            "weather-timeline",
        )

    with weather_response_tab:
        analysis_description(
            title="Kurba e reagimit ndaj temperaturës",
            description=(
                "Çdo pikë përfaqëson një ditë të konsumit të portofolit. "
                "Zona e reagimit përmbledh konsumin sipas intervaleve të "
                "temperaturës duke përdorur medianën dhe intervalin P10–P90."
            ),
            purpose=(
                "Të identifikohen lidhje jolineare, si rritja e kërkesës "
                "për energji gjatë kushteve shumë të ftohta ose shumë të nxehta."
            ),
            finding=(
                "Korelacioni negativ me temperaturën dhe korelacioni "
                "pozitiv me HDD tregojnë se efekti i motit të ftohtë "
                "është më i rëndësishëm për këtë portofol."
            ),
        )

        render_plot(
            weather_response_figure(
                weather_portfolio,
                data["weather_response"],
            ),
            "weather-response",
        )

    with weather_degree_days_tab:
        analysis_description(
            title="Heating dhe Cooling Degree Days",
            description=(
                "HDD mat temperaturën ditore nën bazën 18°C, ndërsa "
                "CDD mat temperaturën mbi të njëjtën bazë."
            ),
            purpose=(
                "Të ndahet ndikimi i motit në kërkesë për ngrohje dhe "
                "ftohje, në vend që të përdoret vetëm temperatura e papërpunuar."
            ),
            finding=(
                f"HDD ka korelacion {hdd_corr:.3f} me konsumin e portofolit, "
                f"krahasuar me {cdd_corr:.3f} për CDD."
            ),
        )
        render_plot(
            weather_monthly_figure(data["weather_monthly"]),
            "weather-monthly",
        )

    with weather_company_tab:
        analysis_description(
            title="Ndjeshmëria e kompanive ndaj motit",
            description=(
                "Çdo kompani pozicionohet sipas korelacionit të saj me "
                "HDD dhe CDD. Madhësia e pikës përfaqëson konsumin "
                "mesatar ditor."
            ),
            purpose=(
                "Të identifikohen kompanitë që reagojnë më shumë ndaj "
                "kushteve të ngrohjes ose ftohjes."
            ),
            finding=(
                "Kompanitë më djathtas në grafik kanë lidhje më të fortë "
                "me kërkesën për ngrohje, ndërsa kompanitë më lart kanë "
                "lidhje më të fortë me kërkesën për ftohje."
            ),
        )

        render_plot(
            weather_sensitivity_figure(weather_company),
            "weather-company",
        )

        st.markdown("#### Kompanitë me ndjeshmërinë më të lartë ndaj motit")

        st.dataframe(
            weather_company[
                [
                    "company_code",
                    "weather_sensitivity_strength",
                    "dominant_weather_response",
                    "temperature_correlation",
                    "hdd_correlation",
                    "cdd_correlation",
                    "heating_kwh_per_hdd",
                    "cooling_kwh_per_cdd",
                ]
            ]
            .sort_values(
                "weather_sensitivity_strength",
                ascending=False,
            )
            .head(20),
            width="stretch",
            hide_index=True,
        )
