import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

PROFILE_START = pd.Timestamp("2025-07-01")
PROFILE_END = pd.Timestamp("2026-07-01")

PRISTINA_LATITUDE = 42.6629
PRISTINA_LONGITUDE = 21.1655
WEATHER_TIMEZONE = "Europe/Belgrade"

HEATING_BASE_C = 18.0
COOLING_BASE_C = 18.0

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_INPUT_DIR = Path("data/processed/hourly_long")
DEFAULT_QUALITY_REPORT = Path("reports/hapi_1_data_quality.xlsx")
DEFAULT_OUTPUT_DIR = Path("data/processed/factors/weather")
DEFAULT_REPORT_PATH = Path("reports/hapi_5_weather.xlsx")
DEFAULT_FIGURE_DIR = Path("reports/figures/weather")
DEFAULT_CACHE_PATH = Path("data/reference/weather/prishtina_daily_2025-07_2026-06.csv")


def calculate_degree_days(
    temperature_mean_c: pd.Series,
    heating_base_c: float = HEATING_BASE_C,
    cooling_base_c: float = COOLING_BASE_C,
) -> tuple[pd.Series, pd.Series]:
    temperature = pd.to_numeric(
        temperature_mean_c,
        errors="coerce",
    )

    hdd = (heating_base_c - temperature).clip(lower=0)

    cdd = (temperature - cooling_base_c).clip(lower=0)

    return hdd, cdd


def parse_open_meteo_response(
    payload: dict,
) -> pd.DataFrame:
    if "daily" not in payload:
        raise ValueError("Open-Meteo response does not contain daily data.")

    daily = payload["daily"]

    required_fields = [
        "time",
        "temperature_2m_mean",
        "temperature_2m_min",
        "temperature_2m_max",
    ]

    missing_fields = [field for field in required_fields if field not in daily]

    if missing_fields:
        raise ValueError("Open-Meteo response is missing fields: " f"{missing_fields}")

    lengths = {len(daily[field]) for field in required_fields}

    if len(lengths) != 1:
        raise ValueError("Open-Meteo daily arrays have inconsistent lengths.")

    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(
                daily["time"],
                errors="coerce",
            ),
            "temperature_mean_c": pd.to_numeric(
                daily["temperature_2m_mean"],
                errors="coerce",
            ),
            "temperature_min_c": pd.to_numeric(
                daily["temperature_2m_min"],
                errors="coerce",
            ),
            "temperature_max_c": pd.to_numeric(
                daily["temperature_2m_max"],
                errors="coerce",
            ),
        }
    )

    weather = weather[
        weather["date"].notna() & weather["temperature_mean_c"].notna()
    ].copy()

    hdd, cdd = calculate_degree_days(weather["temperature_mean_c"])

    weather["hdd"] = hdd
    weather["cdd"] = cdd

    weather["temperature_range_c"] = (
        weather["temperature_max_c"] - weather["temperature_min_c"]
    )

    weather["weather_location"] = "Prishtina"
    weather["latitude"] = PRISTINA_LATITUDE
    weather["longitude"] = PRISTINA_LONGITUDE

    return weather.sort_values("date").reset_index(drop=True)


def fetch_from_open_meteo(
    start: pd.Timestamp,
    end: pd.Timestamp,
    retries: int = 3,
    timeout: int = 30,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    params = {
        "latitude": PRISTINA_LATITUDE,
        "longitude": PRISTINA_LONGITUDE,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": ("temperature_2m_mean," "temperature_2m_min," "temperature_2m_max"),
        "timezone": WEATHER_TIMEZONE,
    }

    url = f"{OPEN_METEO_URL}?" f"{urlencode(params)}"

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):
        try:
            print(f"Fetching weather " f"(attempt {attempt}/{retries})...")

            request = Request(
                url,
                headers={"User-Agent": ("Enerco-Consumption-Analysis/1.0")},
            )

            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))

            return parse_open_meteo_response(payload)

        except Exception as exc:
            last_error = exc

            if attempt < retries:
                time.sleep(retry_delay * attempt)

    raise RuntimeError(
        "Open-Meteo request failed after " f"{retries} attempts."
    ) from last_error


def validate_weather_coverage(
    weather: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    data = weather.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data = data[
        data["date"].notna() & (data["date"] >= start) & (data["date"] <= end)
    ].copy()

    expected_dates = pd.date_range(
        start,
        end,
        freq="D",
    )

    actual_dates = pd.DatetimeIndex(data["date"].drop_duplicates())

    missing_dates = expected_dates.difference(actual_dates)

    if len(missing_dates) > 0:
        raise ValueError(
            "Weather data does not cover the full "
            f"analysis period. Missing {len(missing_dates)} days."
        )

    return data.sort_values("date").reset_index(drop=True)


def load_or_fetch_weather(
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    start: pd.Timestamp = PROFILE_START,
    end: pd.Timestamp = PROFILE_END - pd.Timedelta(days=1),
    refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    cache_path = Path(cache_path)

    if cache_path.exists() and not refresh:
        try:
            cached = pd.read_csv(
                cache_path,
                parse_dates=["date"],
            )

            cached = validate_weather_coverage(
                cached,
                start,
                end,
            )

            return cached, "local_cache"

        except Exception:
            pass

    try:
        weather = fetch_from_open_meteo(
            start=start,
            end=end,
        )

        weather = validate_weather_coverage(
            weather,
            start,
            end,
        )

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        weather.to_csv(
            cache_path,
            index=False,
        )

        return weather, "open_meteo"

    except Exception:
        if cache_path.exists():
            cached = pd.read_csv(
                cache_path,
                parse_dates=["date"],
            )

            cached = validate_weather_coverage(
                cached,
                start,
                end,
            )

            print("Weather API unavailable. " "Using local cache fallback.")

            return cached, "local_cache_fallback"

        raise


def load_quality_report(
    quality_path: str | Path | None,
) -> pd.DataFrame | None:
    if quality_path is None:
        return None

    quality_path = Path(quality_path)

    if not quality_path.exists():
        return None

    quality = pd.read_excel(
        quality_path,
        sheet_name="Meter Quality",
    )

    required_columns = {
        "source_sheet",
        "source_column",
        "quality_status",
    }

    missing_columns = required_columns - set(quality.columns)

    if missing_columns:
        raise ValueError(
            "Quality report is missing columns: " f"{sorted(missing_columns)}"
        )

    return quality[
        [
            "source_sheet",
            "source_column",
            "quality_status",
        ]
    ].copy()


def prepare_consumption_data(
    data: pd.DataFrame,
    quality_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_columns = {
        "date",
        "company_code",
        "flow_type",
        "energy_kwh",
        "source_sheet",
        "source_column",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Weather analysis input is missing columns: " f"{sorted(missing_columns)}"
        )

    result = data.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result["energy_kwh"] = pd.to_numeric(
        result["energy_kwh"],
        errors="coerce",
    )

    result = result[
        result["date"].notna()
        & (result["date"] >= PROFILE_START)
        & (result["date"] < PROFILE_END)
        & (result["flow_type"] == "consumption")
        & result["energy_kwh"].notna()
    ].copy()

    if quality_df is not None:
        result = result.merge(
            quality_df,
            on=[
                "source_sheet",
                "source_column",
            ],
            how="left",
            validate="many_to_one",
        )

        result = result[result["quality_status"] != "unusable"].copy()

    return result.reset_index(drop=True)


def create_company_daily(
    consumption: pd.DataFrame,
) -> pd.DataFrame:
    return (
        consumption.groupby(
            [
                "company_code",
                "date",
            ],
            as_index=False,
        )["energy_kwh"]
        .sum(min_count=1)
        .rename(
            columns={
                "energy_kwh": "daily_kwh",
            }
        )
    )


def create_portfolio_daily(
    company_daily: pd.DataFrame,
) -> pd.DataFrame:
    return (
        company_daily.groupby(
            "date",
            as_index=False,
        )["daily_kwh"]
        .sum(min_count=1)
        .rename(
            columns={
                "daily_kwh": "portfolio_kwh",
            }
        )
    )


def merge_weather(
    consumption_daily: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    result = consumption_daily.copy()

    result["date"] = pd.to_datetime(result["date"])

    weather_data = weather[
        [
            "date",
            "temperature_mean_c",
            "temperature_min_c",
            "temperature_max_c",
            "temperature_range_c",
            "hdd",
            "cdd",
        ]
    ].copy()

    result = result.merge(
        weather_data,
        on="date",
        how="left",
        validate="many_to_one",
    )

    missing_weather = result["temperature_mean_c"].isna().sum()

    if missing_weather:
        raise ValueError(f"Weather missing for " f"{missing_weather} consumption rows.")

    result["month"] = result["date"].dt.to_period("M").astype(str)

    return result


def safe_correlation(
    x: pd.Series,
    y: pd.Series,
) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(
                x,
                errors="coerce",
            ),
            pd.to_numeric(
                y,
                errors="coerce",
            ),
        ],
        axis=1,
    ).dropna()

    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan

    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def safe_slope(
    x: pd.Series,
    y: pd.Series,
    minimum_points: int = 10,
) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(
                x,
                errors="coerce",
            ),
            pd.to_numeric(
                y,
                errors="coerce",
            ),
        ],
        axis=1,
    ).dropna()

    if len(pair) < minimum_points or pair.iloc[:, 0].nunique() < 2:
        return np.nan

    slope = np.polyfit(
        pair.iloc[:, 0],
        pair.iloc[:, 1],
        1,
    )[0]

    return float(slope)


def create_company_weather_sensitivity(
    company_weather: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for company_code, group in company_weather.groupby("company_code"):
        heating_days = group[group["hdd"] > 0]

        cooling_days = group[group["cdd"] > 0]

        temperature_corr = safe_correlation(
            group["temperature_mean_c"],
            group["daily_kwh"],
        )

        hdd_corr = safe_correlation(
            group["hdd"],
            group["daily_kwh"],
        )

        cdd_corr = safe_correlation(
            group["cdd"],
            group["daily_kwh"],
        )

        heating_slope = safe_slope(
            heating_days["hdd"],
            heating_days["daily_kwh"],
        )

        cooling_slope = safe_slope(
            cooling_days["cdd"],
            cooling_days["daily_kwh"],
        )

        correlation_values = [
            value
            for value in [
                hdd_corr,
                cdd_corr,
            ]
            if pd.notna(value)
        ]

        sensitivity_strength = (
            max(abs(value) for value in correlation_values)
            if correlation_values
            else np.nan
        )

        if pd.notna(hdd_corr) and pd.notna(cdd_corr):
            if abs(hdd_corr) > abs(cdd_corr):
                dominant_response = "heating"
            elif abs(cdd_corr) > abs(hdd_corr):
                dominant_response = "cooling"
            else:
                dominant_response = "balanced"

        elif pd.notna(hdd_corr):
            dominant_response = "heating"

        elif pd.notna(cdd_corr):
            dominant_response = "cooling"

        else:
            dominant_response = "insufficient_data"

        rows.append(
            {
                "company_code": company_code,
                "days_analyzed": len(group),
                "mean_daily_kwh": (group["daily_kwh"].mean()),
                "temperature_correlation": (temperature_corr),
                "hdd_correlation": hdd_corr,
                "cdd_correlation": cdd_corr,
                "heating_kwh_per_hdd": (heating_slope),
                "cooling_kwh_per_cdd": (cooling_slope),
                "weather_sensitivity_strength": (sensitivity_strength),
                "dominant_weather_response": (dominant_response),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "weather_sensitivity_strength",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def create_monthly_weather_summary(
    portfolio_weather: pd.DataFrame,
) -> pd.DataFrame:
    return (
        portfolio_weather.groupby("month")
        .agg(
            total_portfolio_kwh=(
                "portfolio_kwh",
                "sum",
            ),
            mean_daily_portfolio_kwh=(
                "portfolio_kwh",
                "mean",
            ),
            temperature_mean_c=(
                "temperature_mean_c",
                "mean",
            ),
            temperature_min_c=(
                "temperature_min_c",
                "min",
            ),
            temperature_max_c=(
                "temperature_max_c",
                "max",
            ),
            total_hdd=(
                "hdd",
                "sum",
            ),
            total_cdd=(
                "cdd",
                "sum",
            ),
            days=(
                "date",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values("month")
        .reset_index(drop=True)
    )


def create_temperature_response(
    portfolio_weather: pd.DataFrame,
    bin_width: float = 5.0,
) -> pd.DataFrame:
    data = portfolio_weather.copy()

    minimum = np.floor(data["temperature_mean_c"].min() / bin_width) * bin_width

    maximum = np.ceil(data["temperature_mean_c"].max() / bin_width) * bin_width

    bins = np.arange(
        minimum,
        maximum + bin_width,
        bin_width,
    )

    data["temperature_bin"] = pd.cut(
        data["temperature_mean_c"],
        bins=bins,
        include_lowest=True,
    )

    response = (
        data.groupby(
            "temperature_bin",
            observed=True,
        )
        .agg(
            days=(
                "date",
                "count",
            ),
            mean_temperature_c=(
                "temperature_mean_c",
                "mean",
            ),
            mean_portfolio_kwh=(
                "portfolio_kwh",
                "mean",
            ),
            median_portfolio_kwh=(
                "portfolio_kwh",
                "median",
            ),
            p10_portfolio_kwh=(
                "portfolio_kwh",
                lambda x: x.quantile(0.10),
            ),
            p90_portfolio_kwh=(
                "portfolio_kwh",
                lambda x: x.quantile(0.90),
            ),
        )
        .reset_index()
    )

    response["temperature_bin"] = response["temperature_bin"].astype(str)

    return response


def create_portfolio_weather_summary(
    portfolio_weather: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "days_analyzed": (portfolio_weather["date"].nunique()),
                "mean_temperature_c": (portfolio_weather["temperature_mean_c"].mean()),
                "min_temperature_c": (portfolio_weather["temperature_min_c"].min()),
                "max_temperature_c": (portfolio_weather["temperature_max_c"].max()),
                "total_hdd": (portfolio_weather["hdd"].sum()),
                "total_cdd": (portfolio_weather["cdd"].sum()),
                "temperature_correlation": (
                    safe_correlation(
                        portfolio_weather["temperature_mean_c"],
                        portfolio_weather["portfolio_kwh"],
                    )
                ),
                "hdd_correlation": (
                    safe_correlation(
                        portfolio_weather["hdd"],
                        portfolio_weather["portfolio_kwh"],
                    )
                ),
                "cdd_correlation": (
                    safe_correlation(
                        portfolio_weather["cdd"],
                        portfolio_weather["portfolio_kwh"],
                    )
                ),
            }
        ]
    )


def plot_weather_findings(
    portfolio_weather: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    temperature_response: pd.DataFrame,
    company_sensitivity: pd.DataFrame,
    figure_dir: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Daily temperature-consumption response.
    plt.figure(figsize=(10, 6))

    x = portfolio_weather["temperature_mean_c"].to_numpy()

    y = portfolio_weather["portfolio_kwh"].to_numpy()

    plt.scatter(
        x,
        y,
        alpha=0.45,
    )

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() >= 10 and np.unique(x[valid]).size >= 3:
        coefficients = np.polyfit(
            x[valid],
            y[valid],
            2,
        )

        x_curve = np.linspace(
            x[valid].min(),
            x[valid].max(),
            200,
        )

        y_curve = np.polyval(
            coefficients,
            x_curve,
        )

        plt.plot(
            x_curve,
            y_curve,
            linewidth=2,
        )

    plt.xlabel("Mean daily temperature (°C)")

    plt.ylabel("Daily portfolio consumption (kWh)")

    plt.title("Weather response of portfolio consumption")

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        figure_dir / "weather_consumption_response.png",
        dpi=180,
    )

    plt.close()

    # 2. Temperature-bin response with P10-P90 range.
    plt.figure(figsize=(10, 6))

    response = temperature_response.dropna(
        subset=[
            "mean_temperature_c",
            "median_portfolio_kwh",
        ]
    ).sort_values("mean_temperature_c")

    plt.fill_between(
        response["mean_temperature_c"],
        response["p10_portfolio_kwh"],
        response["p90_portfolio_kwh"],
        alpha=0.2,
        label="P10-P90",
    )

    plt.plot(
        response["mean_temperature_c"],
        response["median_portfolio_kwh"],
        marker="o",
        label="Median consumption",
    )

    plt.xlabel("Mean daily temperature (°C)")

    plt.ylabel("Daily portfolio consumption (kWh)")

    plt.title("Consumption response by temperature band")

    plt.grid(alpha=0.2)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figure_dir / "weather_temperature_response_bands.png",
        dpi=180,
    )

    plt.close()

    # 3. Monthly HDD/CDD.
    monthly = monthly_summary.copy()

    positions = np.arange(len(monthly))

    width = 0.38

    plt.figure(figsize=(12, 6))

    plt.bar(
        positions - width / 2,
        monthly["total_hdd"],
        width=width,
        label="HDD",
    )

    plt.bar(
        positions + width / 2,
        monthly["total_cdd"],
        width=width,
        label="CDD",
    )

    plt.xticks(
        positions,
        monthly["month"],
        rotation=45,
        ha="right",
    )

    plt.ylabel("Degree days")

    plt.title("Monthly Heating and Cooling Degree Days")

    plt.grid(
        axis="y",
        alpha=0.2,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figure_dir / "weather_monthly_hdd_cdd.png",
        dpi=180,
    )

    plt.close()

    # 4. Companies with strongest measured weather association.
    sensitivity = (
        company_sensitivity.dropna(subset=["weather_sensitivity_strength"])
        .head(20)
        .sort_values("weather_sensitivity_strength")
    )

    if not sensitivity.empty:
        plt.figure(figsize=(10, 8))

        plt.barh(
            sensitivity["company_code"],
            sensitivity["weather_sensitivity_strength"],
        )

        plt.xlabel("Maximum |correlation| with HDD/CDD")

        plt.ylabel("Company")

        plt.title("Companies with strongest weather association")

        plt.grid(
            axis="x",
            alpha=0.2,
        )

        plt.tight_layout()

        plt.savefig(
            figure_dir / "weather_company_sensitivity.png",
            dpi=180,
        )

        plt.close()


def run_weather_analysis(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    quality_path: str | Path | None = DEFAULT_QUALITY_REPORT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    refresh_weather: bool = False,
) -> dict[str, pd.DataFrame]:
    input_dir = Path(input_dir)

    output_dir = Path(output_dir)

    report_path = Path(report_path)

    parquet_files = sorted(input_dir.glob("part_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Hapi 2 parquet files found in " f"{input_dir}")

    weather, weather_source = load_or_fetch_weather(
        cache_path=cache_path,
        refresh=refresh_weather,
    )

    quality_df = load_quality_report(quality_path)

    required_columns = [
        "date",
        "company_code",
        "flow_type",
        "energy_kwh",
        "source_sheet",
        "source_column",
    ]

    parts = []

    for parquet_file in parquet_files:
        print(f"Loading: " f"{parquet_file.name}")

        data = pd.read_parquet(
            parquet_file,
            columns=required_columns,
        )

        parts.append(
            prepare_consumption_data(
                data,
                quality_df=quality_df,
            )
        )

    consumption = pd.concat(
        parts,
        ignore_index=True,
    )

    if consumption.empty:
        raise ValueError("No valid consumption data available " "for weather analysis.")

    company_daily = create_company_daily(consumption)

    portfolio_daily = create_portfolio_daily(company_daily)

    company_weather = merge_weather(
        company_daily,
        weather,
    )

    portfolio_weather = merge_weather(
        portfolio_daily,
        weather,
    )

    company_sensitivity = create_company_weather_sensitivity(company_weather)

    monthly_summary = create_monthly_weather_summary(portfolio_weather)

    temperature_response = create_temperature_response(portfolio_weather)

    portfolio_summary = create_portfolio_weather_summary(portfolio_weather)

    portfolio_summary["weather_source"] = weather_source

    portfolio_summary["weather_location"] = "Prishtina"

    portfolio_summary["heating_base_c"] = HEATING_BASE_C

    portfolio_summary["cooling_base_c"] = COOLING_BASE_C

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "weather_daily": weather,
        "portfolio_weather": portfolio_weather,
        "company_weather": company_weather,
        "portfolio_summary": portfolio_summary,
        "company_sensitivity": company_sensitivity,
        "monthly_summary": monthly_summary,
        "temperature_response": temperature_response,
    }

    filenames = {
        "weather_daily": ("weather_daily.parquet"),
        "portfolio_weather": ("weather_portfolio_daily.parquet"),
        "company_weather": ("weather_company_daily.parquet"),
        "portfolio_summary": ("weather_portfolio_summary.parquet"),
        "company_sensitivity": ("weather_company_sensitivity.parquet"),
        "monthly_summary": ("weather_monthly.parquet"),
        "temperature_response": ("weather_temperature_response.parquet"),
    }

    for key, filename in filenames.items():
        outputs[key].to_parquet(
            output_dir / filename,
            index=False,
        )

    with pd.ExcelWriter(
        report_path,
        engine="openpyxl",
    ) as writer:
        portfolio_summary.to_excel(
            writer,
            sheet_name="Portfolio Summary",
            index=False,
        )

        weather.to_excel(
            writer,
            sheet_name="Daily Weather",
            index=False,
        )

        monthly_summary.to_excel(
            writer,
            sheet_name="Monthly Weather",
            index=False,
        )

        company_sensitivity.to_excel(
            writer,
            sheet_name="Company Sensitivity",
            index=False,
        )

        temperature_response.to_excel(
            writer,
            sheet_name="Temperature Response",
            index=False,
        )

    plot_weather_findings(
        portfolio_weather=portfolio_weather,
        monthly_summary=monthly_summary,
        temperature_response=temperature_response,
        company_sensitivity=company_sensitivity,
        figure_dir=figure_dir,
    )

    summary = portfolio_summary.iloc[0]

    print()
    print("=" * 60)
    print("WEATHER / HDD / CDD ANALYSIS")
    print("=" * 60)

    print(f"Companies analyzed: " f"{company_daily['company_code'].nunique()}")

    print(f"Weather days: " f"{len(weather)}")

    print(f"Weather source: " f"{weather_source}")

    print("Reference location: " "Prishtina")

    print(f"HDD/CDD base temperature: " f"{HEATING_BASE_C:.1f}°C")

    print(f"Mean temperature: " f"{summary['mean_temperature_c']:.2f}°C")

    print(
        f"Temperature range: "
        f"{summary['min_temperature_c']:.2f}°C "
        f"to "
        f"{summary['max_temperature_c']:.2f}°C"
    )

    print(
        f"Portfolio temperature correlation: "
        f"{summary['temperature_correlation']:.4f}"
    )

    print(f"Portfolio HDD correlation: " f"{summary['hdd_correlation']:.4f}")

    print(f"Portfolio CDD correlation: " f"{summary['cdd_correlation']:.4f}")

    print(f"Report saved: " f"{report_path}")

    print(f"Figures saved: " f"{figure_dir}")

    print(f"Processed data saved: " f"{output_dir}")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--quality-report",
        default=DEFAULT_QUALITY_REPORT,
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
    )

    parser.add_argument(
        "--figure-dir",
        default=DEFAULT_FIGURE_DIR,
    )

    parser.add_argument(
        "--cache",
        default=DEFAULT_CACHE_PATH,
    )

    parser.add_argument(
        "--refresh-weather",
        action="store_true",
    )

    args = parser.parse_args()

    run_weather_analysis(
        input_dir=args.input_dir,
        quality_path=args.quality_report,
        output_dir=args.output_dir,
        report_path=args.report,
        figure_dir=args.figure_dir,
        cache_path=args.cache,
        refresh_weather=args.refresh_weather,
    )


if __name__ == "__main__":
    main()
