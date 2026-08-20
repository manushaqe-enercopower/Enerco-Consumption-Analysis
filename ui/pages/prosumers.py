from __future__ import annotations
import plotly.express as px

from pathlib import Path

import pandas as pd
import streamlit as st

UI_DIR = Path(__file__).resolve().parents[1]
ROOT = UI_DIR.parent

PROSUMERS_DIR = ROOT / "data" / "processed" / "prosumers"

SUMMARY_PATH = PROSUMERS_DIR / "prosumer_summary.parquet"
MONTHLY_PATH = PROSUMERS_DIR / "prosumer_monthly.parquet"
HOURLY_PROFILE_PATH = PROSUMERS_DIR / "prosumer_hourly_profile.parquet"
PORTFOLIO_MONTHLY_PATH = PROSUMERS_DIR / "prosumer_portfolio_monthly.parquet"
PORTFOLIO_HOURLY_PROFILE_PATH = (
    PROSUMERS_DIR / "prosumer_portfolio_hourly_profile.parquet"
)
HOURLY_PATH = PROSUMERS_DIR / "prosumer_hourly.parquet"


@st.cache_data(show_spinner=False)
def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


summary = read_parquet(str(SUMMARY_PATH))
monthly = read_parquet(str(MONTHLY_PATH))
hourly_profile = read_parquet(str(HOURLY_PROFILE_PATH))
portfolio_monthly = read_parquet(str(PORTFOLIO_MONTHLY_PATH))
portfolio_hourly_profile = read_parquet(str(PORTFOLIO_HOURLY_PROFILE_PATH))
hourly = read_parquet(str(HOURLY_PATH))


total_consumption_kwh = float(summary["total_consumption_kwh"].sum())

total_injection_kwh = float(summary["total_injection_kwh"].sum())

total_net_grid_kwh = float(summary["net_grid_kwh"].sum())

injection_ratio = (
    total_injection_kwh / total_consumption_kwh * 100 if total_consumption_kwh else 0
)


st.title("Analiza e prosumerëve")

st.caption(
    "Analiza e konsumit dhe injektimit të energjisë për njehsorët "
    "prosumer gjatë periudhës Korrik 2025 – Qershor 2026."
)

st.info(
    "Në të dhënat burimore, **A+** përfaqëson konsumin nga rrjeti, "
    "ndërsa **A-** përfaqëson energjinë e injektuar në rrjet."
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Prosumer të analizuar",
    f"{len(summary):,}",
)

col2.metric(
    "Konsumi total",
    f"{total_consumption_kwh / 1_000_000:.2f} GWh",
)

col3.metric(
    "Injektimi total",
    f"{total_injection_kwh / 1_000_000:.2f} GWh",
)

col4.metric(
    "Injektim / konsum",
    f"{injection_ratio:.2f}%",
)


st.caption(
    f"Bilanci neto i energjisë me rrjetin për portofolin është "
    f"{total_net_grid_kwh / 1_000_000:.2f} GWh."
)
overview_tab, monthly_tab, hourly_tab, prosumer_tab, table_tab = st.tabs(
    [
        "Përmbledhje",
        "Profili mujor",
        "Profili 24-orësh",
        "Analiza sipas prosumerit",
        "Tabela e prosumerëve",
    ]
)
with overview_tab:
    st.subheader("Përmbledhje")

    st.markdown(
        "Kjo pamje përmbledh marrëdhënien ndërmjet konsumit nga rrjeti "
        "dhe energjisë së injektuar nga prosumerët."
    )

    st.caption(
        "Qëllimi analitik: Të shihet sa energji konsumon portofoli, "
        "sa injekton në rrjet dhe si ndryshon ky raport gjatë vitit."
    )

    total_export_hours = int(summary["net_export_hours"].sum())

    prosumers_with_injection = int((summary["total_injection_kwh"] > 0).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Bilanci neto me rrjetin",
        f"{total_net_grid_kwh / 1_000_000:.2f} GWh",
    )

    col2.metric(
        "Prosumerë me injektim",
        f"{prosumers_with_injection:,}",
    )

    col3.metric(
        "Orë net-export në njehsorë",
        f"{total_export_hours:,}",
    )

    st.divider()

    portfolio_monthly_view = portfolio_monthly.copy()

    portfolio_monthly_view["month"] = portfolio_monthly_view["month"].astype(str)

    monthly_chart = portfolio_monthly_view.melt(
        id_vars="month",
        value_vars=[
            "consumption_kwh",
            "injection_kwh",
        ],
        var_name="flow",
        value_name="energy_kwh",
    )

    monthly_chart["flow"] = monthly_chart["flow"].map(
        {
            "consumption_kwh": "Konsum",
            "injection_kwh": "Injektim",
        }
    )

    fig = px.bar(
        monthly_chart,
        x="month",
        y="energy_kwh",
        color="flow",
        barmode="group",
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>" "%{fullData.name}: %{y:,.0f} kWh" "<extra></extra>"
        )
    )

    fig.update_layout(
        title="Konsumi dhe injektimi sipas muajit",
        xaxis_title="Muaji",
        yaxis_title="Energjia (kWh)",
        legend_title="",
        height=500,
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=portfolio_monthly_view["month"],
        ticktext=portfolio_monthly_view["month"],
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.caption(
        "Grafiku krahason konsumin total nga rrjeti me energjinë "
        "e injektuar nga prosumerët për secilin muaj."
    )

with monthly_tab:
    st.subheader("Profili mujor")

    st.markdown(
        "Kjo pamje analizon ndryshimin mujor të konsumit, injektimit "
        "dhe bilancit neto me rrjetin."
    )

    st.caption(
        "Qëllimi analitik: Të identifikohen muajt me prodhim më të lartë "
        "dhe periudhat kur prosumerët mbulojnë pjesë më të madhe të konsumit "
        "përmes injektimit."
    )

    monthly_view = portfolio_monthly.copy()
    monthly_view["month"] = monthly_view["month"].astype(str)

    max_injection_row = monthly_view.loc[monthly_view["injection_kwh"].idxmax()]

    max_ratio_row = monthly_view.loc[
        monthly_view["injection_import_ratio_percent"].idxmax()
    ]

    min_grid_row = monthly_view.loc[monthly_view["net_grid_kwh"].idxmin()]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Muaji me injektimin më të lartë",
        str(max_injection_row["month"]),
        f"{max_injection_row['injection_kwh'] / 1000:.1f} MWh",
        delta_color="off",
    )

    col2.metric(
        "Raporti maksimal injektim / konsum",
        f"{max_ratio_row['injection_import_ratio_percent']:.2f}%",
        str(max_ratio_row["month"]),
        delta_color="off",
    )

    col3.metric(
        "Konsumi neto minimal nga rrjeti",
        f"{min_grid_row['net_grid_kwh'] / 1000:.1f} MWh",
        str(min_grid_row["month"]),
        delta_color="off",
    )

    st.divider()

    fig_ratio = px.bar(
        monthly_view,
        x="month",
        y="injection_import_ratio_percent",
        text="injection_import_ratio_percent",
    )

    fig_ratio.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>" "Injektim / konsum: %{y:.2f}%" "<extra></extra>"
        ),
    )

    fig_ratio.update_layout(
        title="Raporti mujor i injektimit ndaj konsumit",
        xaxis_title="Muaji",
        yaxis_title="Injektim / konsum (%)",
        height=500,
    )

    fig_ratio.update_xaxes(
        tickmode="array",
        tickvals=monthly_view["month"],
        ticktext=monthly_view["month"],
    )

    st.plotly_chart(
        fig_ratio,
        width="stretch",
    )

    st.caption(
        "Raporti tregon sa përqind e konsumit mujor korrespondon "
        "me energjinë e injektuar në rrjet."
    )

    st.divider()

    fig_net = px.line(
        monthly_view,
        x="month",
        y="net_grid_kwh",
        markers=True,
    )

    fig_net.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>" "Bilanci neto: %{y:,.0f} kWh" "<extra></extra>"
        ),
    )

    fig_net.update_layout(
        title="Bilanci neto mujor me rrjetin",
        xaxis_title="Muaji",
        yaxis_title="Energjia neto nga rrjeti (kWh)",
        height=500,
    )

    fig_net.update_xaxes(
        tickmode="array",
        tickvals=monthly_view["month"],
        ticktext=monthly_view["month"],
    )

    st.plotly_chart(
        fig_net,
        width="stretch",
    )

    st.caption(
        "Bilanci neto llogaritet si konsum minus injektim. "
        "Vlera më e ulët nënkupton varësi më të vogël neto nga rrjeti."
    )

with hourly_tab:
    st.subheader("Profili 24-orësh")

    st.markdown(
        "Kjo pamje tregon profilin mesatar 24-orësh të konsumit, "
        "injektimit dhe bilancit neto me rrjetin për të gjithë prosumerët."
    )

    st.caption(
        "Qëllimi analitik: Të identifikohen orët kur konsumi dhe injektimi "
        "janë më të larta dhe si ndryshon varësia nga rrjeti gjatë ditës."
    )

    hourly_view = portfolio_hourly_profile.copy()

    peak_consumption_row = hourly_view.loc[hourly_view["mean_consumption_kwh"].idxmax()]

    peak_injection_row = hourly_view.loc[hourly_view["mean_injection_kwh"].idxmax()]

    min_net_row = hourly_view.loc[hourly_view["mean_net_grid_kwh"].idxmin()]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Ora me konsumin më të lartë",
        f"{int(peak_consumption_row['hour']):02d}:00",
        f"{peak_consumption_row['mean_consumption_kwh']:.1f} kWh",
        delta_color="off",
    )

    col2.metric(
        "Ora me injektimin më të lartë",
        f"{int(peak_injection_row['hour']):02d}:00",
        f"{peak_injection_row['mean_injection_kwh']:.1f} kWh",
        delta_color="off",
    )

    col3.metric(
        "Varësia neto minimale nga rrjeti",
        f"{int(min_net_row['hour']):02d}:00",
        f"{min_net_row['mean_net_grid_kwh']:.1f} kWh",
        delta_color="off",
    )

    st.divider()

    hourly_chart = hourly_view.melt(
        id_vars="hour",
        value_vars=[
            "mean_consumption_kwh",
            "mean_injection_kwh",
            "mean_net_grid_kwh",
        ],
        var_name="metric",
        value_name="energy_kwh",
    )

    hourly_chart["metric"] = hourly_chart["metric"].map(
        {
            "mean_consumption_kwh": "Konsum",
            "mean_injection_kwh": "Injektim",
            "mean_net_grid_kwh": "Bilanci neto",
        }
    )

    fig_hourly = px.line(
        hourly_chart,
        x="hour",
        y="energy_kwh",
        color="metric",
        markers=True,
    )

    fig_hourly.update_traces(
        hovertemplate=(
            "Ora %{x}:00<br>" "%{fullData.name}: %{y:,.2f} kWh" "<extra></extra>"
        )
    )

    fig_hourly.update_layout(
        title="Profili mesatar 24-orësh",
        xaxis_title="Ora",
        yaxis_title="Energjia mesatare (kWh)",
        legend_title="",
        height=550,
    )

    fig_hourly.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 25)),
        ticktext=[f"{hour:02d}:00" for hour in range(1, 25)],
    )

    st.plotly_chart(
        fig_hourly,
        width="stretch",
    )

    st.caption(
        "Bilanci neto llogaritet si konsum minus injektim. "
        "Kur diferenca zvogëlohet, injektimi mbulon pjesë më të madhe "
        "të shkëmbimit të energjisë me rrjetin."
    )

    st.divider()

    st.subheader("Profili i injektimit gjatë ditës")

    fig_injection = px.bar(
        hourly_view,
        x="hour",
        y="mean_injection_kwh",
        text="mean_injection_kwh",
    )

    fig_injection.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        hovertemplate=(
            "Ora %{x}:00<br>" "Injektimi mesatar: %{y:,.2f} kWh" "<extra></extra>"
        ),
    )

    fig_injection.update_layout(
        title="Injektimi mesatar sipas orës",
        xaxis_title="Ora",
        yaxis_title="Injektimi mesatar (kWh)",
        height=500,
    )

    fig_injection.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 25)),
        ticktext=[f"{hour:02d}:00" for hour in range(1, 25)],
    )

    st.plotly_chart(
        fig_injection,
        width="stretch",
    )

    st.caption(
        "Grafiku izolon profilin e injektimit për të treguar qartë "
        "orët kur prodhimi i prosumerëve është më aktiv."
    )

with prosumer_tab:
    st.subheader("Analiza sipas prosumerit")

    st.markdown(
        "Kjo pamje mundëson analizën individuale të secilit njehsor prosumer, "
        "duke krahasuar konsumin, injektimin dhe bilancin neto me rrjetin."
    )

    st.caption(
        "Qëllimi analitik: Të identifikohen prosumerët me injektim më të lartë, "
        "periudhat e net-export dhe ndryshimet sezonale gjatë vitit."
    )

    prosumer_options = (
        summary[
            [
                "company_code",
                "meter_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "company_code",
                "meter_id",
            ]
        )
    )

    prosumer_options["label"] = (
        prosumer_options["company_code"] + " — " + prosumer_options["meter_id"]
    )

    selected_label = st.selectbox(
        "Zgjidh prosumerin",
        options=prosumer_options["label"].tolist(),
        key="prosumer-selector",
    )

    selected_row = prosumer_options[prosumer_options["label"] == selected_label].iloc[0]

    selected_company = selected_row["company_code"]
    selected_meter = selected_row["meter_id"]

    selected_summary = summary[
        (summary["company_code"] == selected_company)
        & (summary["meter_id"] == selected_meter)
    ].iloc[0]

    selected_monthly = monthly[
        (monthly["company_code"] == selected_company)
        & (monthly["meter_id"] == selected_meter)
    ].copy()

    selected_profile = hourly_profile[
        (hourly_profile["company_code"] == selected_company)
        & (hourly_profile["meter_id"] == selected_meter)
    ].copy()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Konsumi total",
        f"{selected_summary['total_consumption_kwh'] / 1000:.1f} MWh",
    )

    col2.metric(
        "Injektimi total",
        f"{selected_summary['total_injection_kwh'] / 1000:.1f} MWh",
    )

    col3.metric(
        "Injektim / konsum",
        f"{selected_summary['injection_import_ratio_percent']:.2f}%",
    )

    col4.metric(
        "Orë net-export",
        f"{int(selected_summary['net_export_hours']):,}",
    )

    st.info(
        f"Bilanci neto me rrjetin për **{selected_company} / {selected_meter}** "
        f"është **{selected_summary['net_grid_kwh'] / 1000:.1f} MWh**."
    )

    st.divider()

    selected_monthly["month"] = selected_monthly["month"].astype(str)

    monthly_selected_chart = selected_monthly.melt(
        id_vars="month",
        value_vars=[
            "consumption_kwh",
            "injection_kwh",
        ],
        var_name="flow",
        value_name="energy_kwh",
    )

    monthly_selected_chart["flow"] = monthly_selected_chart["flow"].map(
        {
            "consumption_kwh": "Konsum",
            "injection_kwh": "Injektim",
        }
    )

    fig_selected_monthly = px.bar(
        monthly_selected_chart,
        x="month",
        y="energy_kwh",
        color="flow",
        barmode="group",
    )

    fig_selected_monthly.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>" "%{fullData.name}: %{y:,.2f} kWh" "<extra></extra>"
        ),
    )

    fig_selected_monthly.update_layout(
        title=f"Konsumi dhe injektimi mujor — {selected_meter}",
        xaxis_title="Muaji",
        yaxis_title="Energjia (kWh)",
        legend_title="",
        height=500,
    )

    fig_selected_monthly.update_xaxes(
        tickmode="array",
        tickvals=selected_monthly["month"],
        ticktext=selected_monthly["month"],
    )

    st.plotly_chart(
        fig_selected_monthly,
        width="stretch",
    )

    st.caption(
        "Grafiku tregon ndryshimin mujor të konsumit dhe injektimit "
        "për prosumerin e zgjedhur."
    )

    st.divider()

    selected_hourly_chart = selected_profile.melt(
        id_vars="hour",
        value_vars=[
            "mean_consumption_kwh",
            "mean_injection_kwh",
            "mean_net_grid_kwh",
        ],
        var_name="metric",
        value_name="energy_kwh",
    )

    selected_hourly_chart["metric"] = selected_hourly_chart["metric"].map(
        {
            "mean_consumption_kwh": "Konsum",
            "mean_injection_kwh": "Injektim",
            "mean_net_grid_kwh": "Bilanci neto",
        }
    )

    fig_selected_hourly = px.line(
        selected_hourly_chart,
        x="hour",
        y="energy_kwh",
        color="metric",
        markers=True,
    )

    fig_selected_hourly.update_traces(
        hovertemplate=(
            "Ora %{x}:00<br>" "%{fullData.name}: %{y:,.2f} kWh" "<extra></extra>"
        ),
    )

    fig_selected_hourly.update_layout(
        title=f"Profili mesatar 24-orësh — {selected_meter}",
        xaxis_title="Ora",
        yaxis_title="Energjia mesatare (kWh)",
        legend_title="",
        height=500,
    )

    fig_selected_hourly.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 25)),
        ticktext=[f"{hour:02d}:00" for hour in range(1, 25)],
    )

    st.plotly_chart(
        fig_selected_hourly,
        width="stretch",
    )

    st.caption(
        "Profili 24-orësh tregon se në cilat orë prosumeri konsumon "
        "nga rrjeti dhe kur injektimi ndikon më shumë në bilancin neto."
    )

with table_tab:
    st.subheader("Tabela e prosumerëve")

    st.markdown(
        "Tabela përmbledh rezultatet vjetore për secilin njehsor prosumer "
        "dhe mundëson krahasimin e konsumit, injektimit dhe shkëmbimit neto me rrjetin."
    )

    st.caption(
        "Qëllimi analitik: Të identifikohen prosumerët me injektim më të lartë, "
        "raport më të madh injektim/konsum dhe më shumë orë net-export."
    )

    table_view = summary.copy()

    table_view["company_number"] = (
        table_view["company_code"].str.extract(r"(\d+)").astype(int)
    )

    table_view = table_view.sort_values(
        [
            "company_number",
            "meter_id",
        ]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Prosumerë",
        f"{len(table_view):,}",
    )

    col2.metric(
        "Me injektim",
        f"{int((table_view['total_injection_kwh'] > 0).sum()):,}",
    )

    col3.metric(
        "Me orë net-export",
        f"{int((table_view['net_export_hours'] > 0).sum()):,}",
    )

    display_table = table_view[
        [
            "company_code",
            "meter_id",
            "total_consumption_kwh",
            "total_injection_kwh",
            "net_grid_kwh",
            "injection_import_ratio_percent",
            "hours_with_injection",
            "net_export_hours",
            "max_consumption_kwh",
            "max_injection_kwh",
        ]
    ].rename(
        columns={
            "company_code": "Kompania",
            "meter_id": "Njehsori",
            "total_consumption_kwh": "Konsumi total (kWh)",
            "total_injection_kwh": "Injektimi total (kWh)",
            "net_grid_kwh": "Bilanci neto (kWh)",
            "injection_import_ratio_percent": "Injektim / konsum (%)",
            "hours_with_injection": "Orë me injektim",
            "net_export_hours": "Orë net-export",
            "max_consumption_kwh": "Konsumi maksimal (kWh)",
            "max_injection_kwh": "Injektimi maksimal (kWh)",
        }
    )

    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Konsumi total (kWh)": st.column_config.NumberColumn(
                "Konsumi total (kWh)",
                format="%.2f",
            ),
            "Injektimi total (kWh)": st.column_config.NumberColumn(
                "Injektimi total (kWh)",
                format="%.2f",
            ),
            "Bilanci neto (kWh)": st.column_config.NumberColumn(
                "Bilanci neto (kWh)",
                format="%.2f",
            ),
            "Injektim / konsum (%)": st.column_config.NumberColumn(
                "Injektim / konsum (%)",
                format="%.2f%%",
            ),
            "Konsumi maksimal (kWh)": st.column_config.NumberColumn(
                "Konsumi maksimal (kWh)",
                format="%.2f",
            ),
            "Injektimi maksimal (kWh)": st.column_config.NumberColumn(
                "Injektimi maksimal (kWh)",
                format="%.2f",
            ),
        },
    )

    st.caption(
        "Bilanci neto llogaritet si konsum minus injektim. "
        "Vlerat negative tregojnë se injektimi është më i madh se konsumi "
        "për periudhën përkatëse."
    )
