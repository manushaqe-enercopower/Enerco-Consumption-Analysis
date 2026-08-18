from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
    ],
}

SERIES_COLORS = [
    "#2563EB",
    "#F59E0B",
    "#10B981",
    "#EF4444",
    "#8B5CF6",
    "#06B6D4",
    "#EC4899",
    "#84CC16",
]

TARIFF_COLORS = {
    "T1": "#F59E0B",
    "T2": "#2563EB",
    "T3": "#8B5CF6",
}


def company_sort_number(
    company_code: str,
) -> int:
    match = pd.Series([str(company_code)]).str.extract(r"(\d+)").iloc[0, 0]

    if pd.isna(match):
        return 999999

    return int(match)


def series_color(
    index: int,
) -> str:
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def tariff_color(
    tariff: str,
    index: int = 0,
) -> str:
    return TARIFF_COLORS.get(
        str(tariff),
        series_color(index),
    )


def hex_to_rgba(
    hex_color: str,
    alpha: float,
) -> str:
    value = hex_color.lstrip("#")

    r = int(
        value[0:2],
        16,
    )

    g = int(
        value[2:4],
        16,
    )

    b = int(
        value[4:6],
        16,
    )

    return f"rgba({r},{g},{b},{alpha})"


def discrete_colorscale(
    colors: list[str],
) -> list[list[float | str]]:
    if not colors:
        return []

    count = len(colors)

    scale = []

    for index, color in enumerate(colors):
        start = index / count
        end = (index + 1) / count

        scale.append(
            [
                start,
                color,
            ]
        )

        scale.append(
            [
                max(
                    start,
                    end - 1e-6,
                ),
                color,
            ]
        )

    return scale


def style_figure(
    fig: go.Figure,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 500,
    hovermode: str | None = None,
) -> go.Figure:
    fig.update_layout(
        title={
            "text": title,
            "x": 0.01,
            "xanchor": "left",
        },
        height=height,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 30,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        hovermode=hovermode,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(
        title_text=x_title,
        showgrid=False,
    )

    fig.update_yaxes(
        title_text=y_title,
        gridcolor="rgba(128,128,128,0.15)",
        zerolinecolor="rgba(128,128,128,0.25)",
    )

    return fig


def format_energy(
    value_kwh: float,
) -> str:
    if abs(value_kwh) >= 1_000_000:
        return f"{value_kwh / 1_000_000:,.2f} GWh"

    if abs(value_kwh) >= 1_000:
        return f"{value_kwh / 1_000:,.2f} MWh"

    return f"{value_kwh:,.2f} kWh"


def sort_by_company(
    data: pd.DataFrame,
    column: str = "company_code",
) -> pd.DataFrame:
    result = data.copy()

    if column not in result.columns:
        return result

    result["_company_sort"] = result[column].astype(str).map(company_sort_number)

    result = (
        result.sort_values(
            [
                "_company_sort",
                column,
            ]
        )
        .drop(
            columns=[
                "_company_sort",
            ]
        )
        .reset_index(drop=True)
    )

    return result
