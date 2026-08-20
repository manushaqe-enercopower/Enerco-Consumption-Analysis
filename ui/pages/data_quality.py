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
    sort_by_company,
    style_figure,
)

QUALITY_REPORT = ROOT / "reports" / "hapi_1_data_quality.xlsx"


QUALITY_LABELS = {
    "clean": "I pastër",
    "review": "Për shqyrtim",
    "unusable": "I papërdorshëm",
}

COVERAGE_LABELS = {
    "full_period": "Periudhë e plotë",
    "partial_period": "Periudhë e pjesshme",
    "no_data": "Pa të dhëna",
}

FLOW_LABELS = {
    "consumption": "Konsum / A+",
    "solar_injection": "Injektim / A−",
}


@st.cache_data(show_spinner=False)
def load_quality_report(
    path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meter_quality = pd.read_excel(
        path,
        sheet_name="Meter Quality",
    )

    timeline_quality = pd.read_excel(
        path,
        sheet_name="Timeline Quality",
    )

    return (
        meter_quality,
        timeline_quality,
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


def prepare_quality_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    required_columns = {
        "company_code",
        "meter_id",
        "flow_type",
        "quality_status",
        "coverage_status",
        "profile_total_hours",
        "profile_observed_hours",
        "profile_missing_hours",
        "profile_missing_percent",
    }

    missing = required_columns - set(result.columns)

    if missing:
        st.error(
            "Raporti i cilësisë nuk përmban kolonat e reja "
            "të dritares vjetore të profilit."
        )

        st.code("python -m src.quality")

        st.caption("Kolonat që mungojnë: " + ", ".join(sorted(missing)))

        st.stop()

    numeric_columns = [
        "total_hours",
        "observed_hours",
        "missing_hours",
        "missing_percent",
        "profile_total_hours",
        "profile_observed_hours",
        "profile_missing_hours",
        "profile_missing_percent",
        "coverage_percent",
        "negative_count",
        "zero_run_count_over_48h",
        "zero_run_day_count",
        "max_zero_run_hours",
        "extreme_value_count",
        "mean_kwh",
        "max_kwh",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    result["profile_coverage_percent"] = np.where(
        result["profile_total_hours"] > 0,
        result["profile_observed_hours"] / result["profile_total_hours"] * 100,
        0.0,
    )

    result["quality_label"] = (
        result["quality_status"].map(QUALITY_LABELS).fillna(result["quality_status"])
    )

    result["coverage_label"] = (
        result["coverage_status"].map(COVERAGE_LABELS).fillna(result["coverage_status"])
    )

    result["flow_label"] = (
        result["flow_type"].map(FLOW_LABELS).fillna(result["flow_type"])
    )

    result["max_mean_ratio"] = np.where(
        result["mean_kwh"] > 0,
        result["max_kwh"] / result["mean_kwh"],
        np.nan,
    )

    return result


if not QUALITY_REPORT.exists():
    st.error("Raporti i Hapit 1 nuk u gjet.")

    st.code("python -m src.quality")

    st.stop()


meter_quality, timeline_quality = load_quality_report(str(QUALITY_REPORT))

meter_quality = prepare_quality_data(meter_quality)


source_series = len(meter_quality)

source_companies = int(meter_quality["company_code"].nunique())

usable_consumption = meter_quality[
    (meter_quality["flow_type"] == "consumption")
    & (meter_quality["quality_status"] != "unusable")
].copy()

usable_consumption_series = len(usable_consumption)

usable_companies = int(usable_consumption["company_code"].nunique())

excluded_companies = source_companies - usable_companies

clean_count = int((meter_quality["quality_status"] == "clean").sum())

review_count = int((meter_quality["quality_status"] == "review").sum())

unusable_count = int((meter_quality["quality_status"] == "unusable").sum())


st.title("Kontrolli i cilësisë së të dhënave")

st.caption(
    "Validimi i integritetit të të dhënave "
    "përpara llogaritjes së profileve të konsumit"
)


(
    overview_tab,
    missing_tab,
    negative_tab,
    zero_tab,
    extreme_tab,
    timeline_tab,
    meter_table_tab,
) = st.tabs(
    [
        "Përmbledhje",
        "Mungesat dhe mbulimi",
        "Vlerat negative",
        "Zero >48h",
        "Vlerat ekstreme",
        "Konsistenca kohore",
        "Tabela e njehsorëve",
    ]
)


with overview_tab:
    analysis_description(
        title="Gjendja e përgjithshme e cilësisë",
        description=(
            "Çdo seri e njehsorit kontrollohet për mungesa, "
            "vlera negative, periudha zero mbi 48 orë, "
            "skaje statistikore ekstreme dhe mbulim të periudhës."
        ),
        purpose=(
            "Të përcaktohet cilat seri janë të besueshme për "
            "analizën vjetore dhe cilat duhet të përjashtohen "
            "ose të shqyrtohen."
        ),
        finding=(
            f"Nga {source_series:,} seri burimore, "
            f"{clean_count:,} janë të pastra, "
            f"{review_count:,} kërkojnë shqyrtim dhe "
            f"{unusable_count:,} janë të papërdorshme sipas "
            "rregullave të cilësisë. "
            f"{usable_companies} nga {source_companies} kompani "
            "mbeten të përdorshme për analizën vjetore të konsumit."
        ),
    )

    cols = st.columns(5)

    cols[0].metric(
        "Seri burimore",
        f"{source_series:,}",
    )

    cols[1].metric(
        "Të pastra",
        f"{clean_count:,}",
    )

    cols[2].metric(
        "Për shqyrtim",
        f"{review_count:,}",
    )

    cols[3].metric(
        "Të papërdorshme",
        f"{unusable_count:,}",
    )

    cols[4].metric(
        "Kompanitë e përdorshme",
        f"{usable_companies}/{source_companies}",
    )

    left, right = st.columns(2)

    with left:
        analysis_description(
            title="Shpërndarja sipas statusit të cilësisë",
            description=(
                "Çdo seri klasifikohet si e pastër, për shqyrtim " "ose e papërdorshme."
            ),
            purpose=(
                "Të shihet menjëherë sa pjesë e të dhënave mund "
                "të hyjë në analizë pa problem dhe sa kërkon trajtim."
            ),
        )

        status_data = pd.DataFrame(
            {
                "Statusi": [
                    "I pastër",
                    "Për shqyrtim",
                    "I papërdorshëm",
                ],
                "Seri": [
                    clean_count,
                    review_count,
                    unusable_count,
                ],
            }
        )

        fig = px.pie(
            status_data,
            names="Statusi",
            values="Seri",
            hole=0.58,
        )

        fig.update_traces(
            textposition="inside",
            textinfo="label+percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Seri: %{value:,.0f}<br>"
                "Pjesa: %{percent}"
                "<extra></extra>"
            ),
        )

        fig = style_figure(
            fig,
            title="Statusi i cilësisë së serive",
            height=480,
        )

        render_plot(
            fig,
            "quality-status-overview",
        )

    with right:
        full_count = int((meter_quality["coverage_status"] == "full_period").sum())

        partial_count = int(
            (meter_quality["coverage_status"] == "partial_period").sum()
        )

        analysis_description(
            title="Mbulimi i periudhës burimore",
            description=(
                "Ky kontroll dallon seritë që kanë të dhëna nga "
                "fillimi deri në fund të skedarit nga seritë që "
                "hyjnë ose dalin gjatë periudhës."
            ),
            purpose=(
                "Të dallohet mbulimi i pjesshëm nga boshllëqet "
                "reale brenda dritares vjetore të analizës."
            ),
            finding=(
                f"{full_count:,} seri mbulojnë të gjithë periudhën "
                f"burimore, ndërsa {partial_count:,} kanë mbulim "
                "të pjesshëm. Mbulimi i pjesshëm nuk e bën "
                "automatikisht një seri të papërdorshme."
            ),
        )

        coverage_data = pd.DataFrame(
            {
                "Mbulimi": [
                    "Periudhë e plotë",
                    "Periudhë e pjesshme",
                ],
                "Seri": [
                    full_count,
                    partial_count,
                ],
            }
        )

        fig = px.bar(
            coverage_data,
            x="Mbulimi",
            y="Seri",
            text="Seri",
        )

        fig.update_traces(
            textposition="outside",
        )

        fig = style_figure(
            fig,
            title="Mbulimi i periudhës burimore",
            x_title=None,
            y_title="Numri i serive",
            height=480,
        )

        render_plot(
            fig,
            "quality-coverage-overview",
        )

    st.info(
        f"Pas filtrimit të cilësisë, analiza e profilit përdor "
        f"{usable_consumption_series:,} seri konsumi nga "
        f"{usable_companies} kompani. "
        f"{excluded_companies} kompani nuk kanë seri konsumi "
        "të mjaftueshme për profilin vjetor."
    )


with missing_tab:
    over_threshold = meter_quality[meter_quality["profile_missing_percent"] > 10].copy()

    missing_hours_total = int(meter_quality["profile_missing_hours"].sum())

    analysis_description(
        title="Mungesat në dritaren vjetore",
        description=(
            "Për çdo seri llogaritet përqindja e orëve që mungojnë "
            "brenda dritares Korrik 2025 – Qershor 2026."
        ),
        purpose=(
            "Sipas metodologjisë, një seri me më shumë se 10% "
            "orë të munguara gjatë vitit konsiderohet e "
            "pabesueshme për atë periudhë."
        ),
        finding=(
            f"{len(over_threshold):,} seri tejkalojnë pragun 10%. "
            f"Në total janë regjistruar {missing_hours_total:,} "
            "orë të munguara në dritaren vjetore në të gjitha seritë."
        ),
    )

    cols = st.columns(3)

    cols[0].metric(
        "Seri mbi pragun 10%",
        f"{len(over_threshold):,}",
    )

    cols[1].metric(
        "Pragu metodologjik",
        "10%",
    )

    cols[2].metric(
        "Orë të munguara",
        f"{missing_hours_total:,}",
    )

    analysis_description(
        title="Shpërndarja e përqindjes së mungesave",
        description=(
            "Histogrami tregon sa seri kanë nivele të ndryshme "
            "mungesash në dritaren vjetore."
        ),
        purpose=(
            "Të shihet nëse mungesat janë të përqendruara në "
            "pak seri apo janë problem i përhapur në dataset."
        ),
        finding=(
            "Vija e kuqe tregon pragun 10% që ndan seritë "
            "e pranueshme nga seritë e papërdorshme për "
            "profilin vjetor."
        ),
    )

    fig = px.histogram(
        meter_quality,
        x="profile_missing_percent",
        nbins=40,
        labels={
            "profile_missing_percent": ("Orë të munguara në profil (%)"),
        },
    )

    fig.add_vline(
        x=10,
        line_dash="dash",
        line_color="red",
        annotation_text="Pragu 10%",
        annotation_position="top right",
    )

    fig = style_figure(
        fig,
        title="Shpërndarja e mungesave në dritaren vjetore",
        x_title="Orë të munguara (%)",
        y_title="Numri i serive",
        height=520,
    )

    render_plot(
        fig,
        "quality-missing-histogram",
    )

    analysis_description(
        title="Mbulimi vjetor kundrejt mbulimit burimor",
        description=(
            "Çdo pikë është një seri. Boshti horizontal paraqet "
            "mbulimin e periudhës burimore, ndërsa boshti vertikal "
            "paraqet mungesat brenda dritares vjetore."
        ),
        purpose=(
            "Të dallohen seritë që janë thjesht të pjesshme në "
            "13 muajt burimorë nga ato që kanë boshllëqe të "
            "rëndësishme brenda vetë vitit analitik."
        ),
    )

    fig = px.scatter(
        meter_quality,
        x="coverage_percent",
        y="profile_missing_percent",
        color="quality_label",
        hover_name="company_code",
        hover_data={
            "meter_id": True,
            "flow_label": True,
            "profile_coverage_percent": ":.2f",
            "coverage_percent": ":.2f",
            "profile_missing_percent": ":.2f",
            "quality_label": False,
        },
        labels={
            "coverage_percent": ("Mbulimi i periudhës burimore (%)"),
            "profile_missing_percent": ("Mungesa në dritaren vjetore (%)"),
            "quality_label": "Statusi",
            "meter_id": "Njehsori",
            "flow_label": "Rrjedha",
            "profile_coverage_percent": ("Mbulimi vjetor (%)"),
        },
    )

    fig.add_hline(
        y=10,
        line_dash="dash",
        line_color="red",
        annotation_text="Pragu 10%",
    )

    fig = style_figure(
        fig,
        title="Mbulimi dhe mungesat sipas serisë",
        x_title="Mbulimi i periudhës burimore (%)",
        y_title="Mungesa vjetore (%)",
        height=580,
    )

    render_plot(
        fig,
        "quality-missing-scatter",
    )

    st.markdown("#### Seritë me mungesat më të larta")

    missing_table = (
        meter_quality[
            [
                "company_code",
                "meter_id",
                "flow_label",
                "profile_observed_hours",
                "profile_missing_hours",
                "profile_missing_percent",
                "quality_label",
            ]
        ]
        .sort_values(
            "profile_missing_percent",
            ascending=False,
        )
        .head(30)
        .rename(
            columns={
                "company_code": "Kompania",
                "meter_id": "Njehsori",
                "flow_label": "Rrjedha",
                "profile_observed_hours": "Orë të vëzhguara",
                "profile_missing_hours": "Orë të munguara",
                "profile_missing_percent": "Mungesa (%)",
                "quality_label": "Statusi",
            }
        )
    )

    st.dataframe(
        missing_table,
        width="stretch",
        hide_index=True,
    )


with negative_tab:
    negative_series = meter_quality[meter_quality["negative_count"] > 0].copy()

    negative_values = int(meter_quality["negative_count"].sum())

    analysis_description(
        title="Kontrolli i vlerave negative",
        description=(
            "Vlerat negative nuk priten as te konsumi A+ dhe "
            "as te injektimi A−. Të dyja ruhen si seri pozitive "
            "dhe të ndara."
        ),
        purpose=(
            "Të identifikohet çdo gabim i mundshëm i eksportimit "
            "ose defekt i të dhënave që do të shtrembëronte analizën."
        ),
        finding=(
            f"Janë gjetur {negative_values:,} vlera negative "
            f"në {len(negative_series):,} seri."
        ),
    )

    cols = st.columns(2)

    cols[0].metric(
        "Vlera negative",
        f"{negative_values:,}",
    )

    cols[1].metric(
        "Seri të prekura",
        f"{len(negative_series):,}",
    )

    if negative_values == 0:
        st.success(
            "Nuk është detektuar asnjë vlerë negative. " "Ky kontroll kalon pa problem."
        )

    else:
        negative_table = (
            negative_series[
                [
                    "company_code",
                    "meter_id",
                    "flow_label",
                    "negative_count",
                    "quality_label",
                ]
            ]
            .sort_values(
                "negative_count",
                ascending=False,
            )
            .rename(
                columns={
                    "company_code": "Kompania",
                    "meter_id": "Njehsori",
                    "flow_label": "Rrjedha",
                    "negative_count": "Vlera negative",
                    "quality_label": "Statusi",
                }
            )
        )

        st.dataframe(
            negative_table,
            width="stretch",
            hide_index=True,
        )


with zero_tab:
    zero_series = meter_quality[meter_quality["zero_run_count_over_48h"] > 0].copy()

    zero_events = int(meter_quality["zero_run_count_over_48h"].sum())

    zero_days = int(meter_quality["zero_run_day_count"].sum())

    max_zero_run = int(meter_quality["max_zero_run_hours"].max())

    analysis_description(
        title="Zero të vazhdueshme mbi 48 orë",
        description=(
            "Periudhat me konsum zero për më shumë se 48 orë "
            "rresht shënohen për shqyrtim, sepse mund të tregojnë "
            "problem komunikimi të njehsorit dhe jo konsum real zero."
        ),
        purpose=(
            "Të dallohen boshllëqet teknike që janë regjistruar "
            "si zero nga periudhat normale të konsumit."
        ),
        finding=(
            f"{len(zero_series):,} seri kanë të paktën një periudhë "
            f"zero mbi 48 orë. Janë identifikuar {zero_events:,} "
            f"periudha të tilla, që prekin {zero_days:,} ditë në total."
        ),
    )

    cols = st.columns(3)

    cols[0].metric(
        "Seri të prekura",
        f"{len(zero_series):,}",
    )

    cols[1].metric(
        "Periudha >48h",
        f"{zero_events:,}",
    )

    cols[2].metric(
        "Periudha më e gjatë",
        f"{max_zero_run:,} orë",
    )

    if zero_series.empty:
        st.success("Nuk është detektuar asnjë periudhë zero mbi 48 orë.")

    else:
        analysis_description(
            title="Seritë me periudhat zero më të gjata",
            description=(
                "Grafiku rendit seritë sipas periudhës më të gjatë "
                "të regjistruar me zero të vazhdueshme."
            ),
            purpose=("Të prioritizohen rastet që kërkojnë kontroll teknik."),
        )

        plot_data = (
            zero_series.sort_values(
                "max_zero_run_hours",
                ascending=False,
            )
            .head(25)
            .copy()
        )

        plot_data["label"] = (
            plot_data["company_code"].astype(str)
            + " · "
            + plot_data["meter_id"].astype(str)
        )

        plot_data = plot_data.sort_values("max_zero_run_hours")

        fig = px.bar(
            plot_data,
            x="max_zero_run_hours",
            y="label",
            orientation="h",
            hover_data={
                "zero_run_count_over_48h": True,
                "zero_run_day_count": True,
            },
            labels={
                "max_zero_run_hours": ("Periudha më e gjatë (orë)"),
                "label": "Seria",
                "zero_run_count_over_48h": ("Numri i periudhave"),
                "zero_run_day_count": ("Ditë të prekura"),
            },
        )

        fig = style_figure(
            fig,
            title="Periudhat zero më të gjata",
            x_title="Orë të vazhdueshme me zero",
            y_title=None,
            height=650,
        )

        render_plot(
            fig,
            "quality-zero-runs",
        )


with extreme_tab:
    extreme_series = meter_quality[meter_quality["extreme_value_count"] > 0].copy()

    extreme_values = int(meter_quality["extreme_value_count"].sum())

    analysis_description(
        title="Skajet statistikore ekstreme",
        description=(
            "Një vlerë shënohet për shqyrtim kur është mbi "
            "50 herë më e lartë se mesatarja historike e vetë serisë."
        ),
        purpose=(
            "Të parandalohet që vlera potencialisht të gabuara "
            "të shtrembërojnë mesataret, CV, Load Factor dhe "
            "metrikat e tjera."
        ),
        finding=(
            f"Janë identifikuar {extreme_values:,} vlera ekstreme "
            f"në {len(extreme_series):,} seri."
        ),
    )

    cols = st.columns(2)

    cols[0].metric(
        "Vlera ekstreme",
        f"{extreme_values:,}",
    )

    cols[1].metric(
        "Seri të prekura",
        f"{len(extreme_series):,}",
    )

    if extreme_series.empty:
        st.success("Nuk është detektuar asnjë vlerë mbi pragun 50×.")

    else:
        analysis_description(
            title="Maksimumi kundrejt mesatares historike",
            description=(
                "Raporti Max/Mesatare tregon sa larg është "
                "vlera maksimale e secilës seri nga niveli i saj tipik."
            ),
            purpose=(
                "Të dallohen shpejt seritë me skaje statistikore "
                "që kërkojnë verifikim."
            ),
            finding=(
                "Vija referente në 50× përfaqëson pragun e përdorur "
                "nga kontrolli i cilësisë."
            ),
        )

        ratio_data = (
            extreme_series.dropna(subset=["max_mean_ratio"])
            .sort_values(
                "max_mean_ratio",
                ascending=False,
            )
            .head(30)
            .copy()
        )

        ratio_data["label"] = (
            ratio_data["company_code"].astype(str)
            + " · "
            + ratio_data["meter_id"].astype(str)
        )

        ratio_data = ratio_data.sort_values("max_mean_ratio")

        fig = px.bar(
            ratio_data,
            x="max_mean_ratio",
            y="label",
            orientation="h",
            hover_data={
                "mean_kwh": ":.3f",
                "max_kwh": ":.3f",
                "extreme_value_count": True,
            },
            labels={
                "max_mean_ratio": "Raporti Max/Mesatare",
                "label": "Seria",
                "mean_kwh": "Mesatarja (kWh)",
                "max_kwh": "Maksimumi (kWh)",
                "extreme_value_count": "Vlera ekstreme",
            },
        )

        fig.add_vline(
            x=50,
            line_dash="dash",
            line_color="red",
            annotation_text="Pragu 50×",
        )

        fig = style_figure(
            fig,
            title="Seritë me skajet statistikore më të larta",
            x_title="Maksimumi / Mesatarja",
            y_title=None,
            height=680,
        )

        render_plot(
            fig,
            "quality-extreme-values",
        )


with timeline_tab:
    timeline_numeric_columns = [
        "invalid_date_rows",
        "invalid_hour_rows",
        "duplicate_date_hour_rows",
        "incomplete_days",
        "days_with_23_hours",
        "days_with_25_hours",
        "missing_calendar_hours",
    ]

    for column in timeline_numeric_columns:
        timeline_quality[column] = pd.to_numeric(
            timeline_quality[column],
            errors="coerce",
        ).fillna(0)

    invalid_dates = int(timeline_quality["invalid_date_rows"].sum())

    invalid_hours = int(timeline_quality["invalid_hour_rows"].sum())

    duplicate_hours = int(timeline_quality["duplicate_date_hour_rows"].sum())

    missing_calendar = int(timeline_quality["missing_calendar_hours"].sum())

    analysis_description(
        title="Konsistenca e sekuencës kohore",
        description=(
            "Çdo fletë kontrollohet për data të pavlefshme, "
            "orë jashtë intervalit 1–24, dublime, ditë jo të plota "
            "dhe orë që mungojnë në kalendar."
        ),
        purpose=(
            "Të sigurohet që sekuenca orare është e rregullt "
            "përpara analizës së profileve dhe outlier-ave."
        ),
        finding=(
            f"Janë gjetur {invalid_dates} rreshta me data të pavlefshme, "
            f"{invalid_hours} me orë të pavlefshme, "
            f"{duplicate_hours} dublime dhe "
            f"{missing_calendar} orë kalendarike që mungojnë."
        ),
    )

    cols = st.columns(4)

    cols[0].metric(
        "Data të pavlefshme",
        f"{invalid_dates:,}",
    )

    cols[1].metric(
        "Orë të pavlefshme",
        f"{invalid_hours:,}",
    )

    cols[2].metric(
        "Dublime",
        f"{duplicate_hours:,}",
    )

    cols[3].metric(
        "Orë kalendarike që mungojnë",
        f"{missing_calendar:,}",
    )

    analysis_description(
        title="Kontrollet sipas fletës burimore",
        description=(
            "Heatmap-i paraqet numrin e problemeve të sekuencës "
            "kohore të detektuara në secilën prej gjashtë fletëve."
        ),
        purpose=(
            "Të identifikohet nëse një problem kohor është lokal "
            "në një fletë apo i përhapur në të gjithë skedarin."
        ),
    )

    heatmap_columns = {
        "invalid_date_rows": "Data të pavlefshme",
        "invalid_hour_rows": "Orë të pavlefshme",
        "duplicate_date_hour_rows": "Dublime",
        "incomplete_days": "Ditë jo të plota",
        "days_with_23_hours": "Ditë me 23 orë",
        "days_with_25_hours": "Ditë me 25 orë",
        "missing_calendar_hours": "Orë që mungojnë",
    }

    z = timeline_quality[list(heatmap_columns.keys())].to_numpy()

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(heatmap_columns.values()),
            y=timeline_quality["source_sheet"],
            text=z,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=max(
                float(np.nanmax(z)),
                1.0,
            ),
            hovertemplate=(
                "Fleta: %{y}<br>" "Kontrolli: %{x}<br>" "Raste: %{z}<extra></extra>"
            ),
        )
    )

    fig = style_figure(
        fig,
        title="Problemet e sekuencës kohore sipas fletës",
        x_title=None,
        y_title="Fleta burimore",
        height=520,
    )

    render_plot(
        fig,
        "quality-timeline-heatmap",
    )

    timeline_display = timeline_quality.rename(
        columns={
            "source_sheet": "Fleta",
            "row_count": "Rreshta",
            "min_date": "Data e parë",
            "max_date": "Data e fundit",
            "invalid_date_rows": "Data të pavlefshme",
            "invalid_hour_rows": "Orë të pavlefshme",
            "duplicate_date_hour_rows": "Dublime",
            "incomplete_days": "Ditë jo të plota",
            "days_with_23_hours": "Ditë me 23 orë",
            "days_with_25_hours": "Ditë me 25 orë",
            "missing_calendar_hours": "Orë që mungojnë",
            "timeline_status": "Statusi",
        }
    )

    st.dataframe(
        timeline_display,
        width="stretch",
        hide_index=True,
    )


with meter_table_tab:
    analysis_description(
        title="Raporti i detajuar sipas njehsorit",
        description=(
            "Tabela përmban rezultatet e kontrolleve të cilësisë "
            "për çdo seri individuale."
        ),
        purpose=(
            "Të lejohet inspektimi i drejtpërdrejtë i një kompanie "
            "ose njehsori dhe arsyes pse është klasifikuar si i pastër, "
            "për shqyrtim ose i papërdorshëm."
        ),
    )

    filter_cols = st.columns(4)

    status_options = [
        "Të gjitha",
        "I pastër",
        "Për shqyrtim",
        "I papërdorshëm",
    ]

    selected_status = filter_cols[0].selectbox(
        "Statusi",
        status_options,
    )

    flow_options = [
        "Të gjitha",
        *sorted(meter_quality["flow_label"].dropna().unique().tolist()),
    ]

    selected_flow = filter_cols[1].selectbox(
        "Rrjedha",
        flow_options,
    )

    companies = sorted(
        meter_quality["company_code"].dropna().unique().tolist(),
        key=company_sort_number,
    )

    selected_company = filter_cols[2].selectbox(
        "Kompania",
        [
            "Të gjitha",
            *companies,
        ],
    )

    coverage_options = [
        "Të gjitha",
        "Periudhë e plotë",
        "Periudhë e pjesshme",
        "Pa të dhëna",
    ]

    selected_coverage = filter_cols[3].selectbox(
        "Mbulimi",
        coverage_options,
    )

    filtered = meter_quality.copy()

    if selected_status != "Të gjitha":
        filtered = filtered[filtered["quality_label"] == selected_status]

    if selected_flow != "Të gjitha":
        filtered = filtered[filtered["flow_label"] == selected_flow]

    if selected_company != "Të gjitha":
        filtered = filtered[filtered["company_code"] == selected_company]

    if selected_coverage != "Të gjitha":
        filtered = filtered[filtered["coverage_label"] == selected_coverage]

    filtered = sort_by_company(filtered)

    display_columns = [
        "company_code",
        "meter_id",
        "flow_label",
        "quality_label",
        "coverage_label",
        "profile_observed_hours",
        "profile_missing_hours",
        "profile_missing_percent",
        "negative_count",
        "zero_run_count_over_48h",
        "zero_run_day_count",
        "max_zero_run_hours",
        "extreme_value_count",
        "first_valid_timestamp",
        "last_valid_timestamp",
    ]

    display = filtered[display_columns].rename(
        columns={
            "company_code": "Kompania",
            "meter_id": "Njehsori",
            "flow_label": "Rrjedha",
            "quality_label": "Statusi",
            "coverage_label": "Mbulimi",
            "profile_observed_hours": "Orë të vëzhguara",
            "profile_missing_hours": "Orë të munguara",
            "profile_missing_percent": "Mungesa (%)",
            "negative_count": "Vlera negative",
            "zero_run_count_over_48h": "Periudha zero >48h",
            "zero_run_day_count": "Ditë me zero >48h",
            "max_zero_run_hours": "Zero max (orë)",
            "extreme_value_count": "Vlera ekstreme",
            "first_valid_timestamp": "Leximi i parë",
            "last_valid_timestamp": "Leximi i fundit",
        }
    )

    st.caption(f"Po shfaqen {len(display):,} nga " f"{len(meter_quality):,} seri.")

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )
