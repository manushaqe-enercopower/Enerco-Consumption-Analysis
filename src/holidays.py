import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_START = pd.Timestamp("2025-07-01")
PROFILE_END = pd.Timestamp("2026-07-01")

DEFAULT_INPUT_DIR = Path("data/processed/hourly_long")
DEFAULT_QUALITY_REPORT = Path("reports/hapi_1_data_quality.xlsx")
DEFAULT_OUTLIER_PATH = Path("data/processed/outliers/company_hourly_outliers.parquet")
DEFAULT_OUTPUT_DIR = Path("data/processed/factors/holidays")
DEFAULT_REPORT_PATH = Path("reports/hapi_5_holidays.xlsx")


HOLIDAY_DATA = [
    {
        "holiday_name": "Krishtlindjet Katolike",
        "holiday_date": "2025-12-25",
        "observed_date": "2025-12-25",
    },
    {
        "holiday_name": "Viti i Ri - dita 1",
        "holiday_date": "2026-01-01",
        "observed_date": "2026-01-01",
    },
    {
        "holiday_name": "Viti i Ri - dita 2",
        "holiday_date": "2026-01-02",
        "observed_date": "2026-01-02",
    },
    {
        "holiday_name": "Krishtlindjet Ortodokse",
        "holiday_date": "2026-01-07",
        "observed_date": "2026-01-07",
    },
    {
        "holiday_name": "Dita e Pavaresise se Republikes se Kosoves",
        "holiday_date": "2026-02-17",
        "observed_date": "2026-02-17",
    },
    {
        "holiday_name": "Bajrami i Madh - dita e pare",
        "holiday_date": "2026-03-20",
        "observed_date": "2026-03-20",
    },
    {
        "holiday_name": "Dita e Kushtetutes se Republikes se Kosoves",
        "holiday_date": "2026-04-09",
        "observed_date": "2026-04-09",
    },
    {
        "holiday_name": "Pashket Katolike",
        "holiday_date": "2026-04-05",
        "observed_date": "2026-04-06",
    },
    {
        "holiday_name": "Pashket Ortodokse",
        "holiday_date": "2026-04-12",
        "observed_date": "2026-04-13",
    },
    {
        "holiday_name": "Dita Nderkombetare e Punes",
        "holiday_date": "2026-05-01",
        "observed_date": "2026-05-01",
    },
    {
        "holiday_name": "Dita e Evropes",
        "holiday_date": "2026-05-09",
        "observed_date": "2026-05-11",
    },
    {
        "holiday_name": "Bajrami i Vogel - dita e pare",
        "holiday_date": "2026-05-27",
        "observed_date": "2026-05-27",
    },
]


def get_kosovo_holiday_calendar(
    start: pd.Timestamp = PROFILE_START,
    end: pd.Timestamp = PROFILE_END,
) -> pd.DataFrame:
    calendar = pd.DataFrame(HOLIDAY_DATA)

    calendar["holiday_date"] = pd.to_datetime(calendar["holiday_date"])

    calendar["observed_date"] = pd.to_datetime(calendar["observed_date"])

    calendar = calendar[
        (calendar["observed_date"] >= start) & (calendar["observed_date"] < end)
    ].copy()

    calendar["is_shifted"] = calendar["holiday_date"] != calendar["observed_date"]

    calendar["holiday_weekday"] = calendar["holiday_date"].dt.day_name()

    calendar["observed_weekday"] = calendar["observed_date"].dt.day_name()

    return calendar.sort_values("observed_date").reset_index(drop=True)


def load_quality_report(
    quality_path: str | Path | None,
) -> pd.DataFrame | None:
    if quality_path is None:
        return None

    quality_path = Path(quality_path)

    if not quality_path.exists():
        return None

    quality_df = pd.read_excel(
        quality_path,
        sheet_name="Meter Quality",
    )

    required_columns = {
        "source_sheet",
        "source_column",
        "quality_status",
    }

    missing_columns = required_columns - set(quality_df.columns)

    if missing_columns:
        raise ValueError(
            "Quality report is missing columns: " f"{sorted(missing_columns)}"
        )

    return quality_df[
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
        "hour",
        "timestamp",
        "company_code",
        "flow_type",
        "energy_kwh",
        "source_sheet",
        "source_column",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Holiday input is missing columns: " f"{sorted(missing_columns)}"
        )

    result = data.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    result["hour"] = pd.to_numeric(
        result["hour"],
        errors="coerce",
    )

    result["energy_kwh"] = pd.to_numeric(
        result["energy_kwh"],
        errors="coerce",
    )

    result = result[
        result["date"].notna()
        & result["timestamp"].notna()
        & result["hour"].between(1, 24)
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


def attach_holiday_flags(
    data: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    result["date"] = pd.to_datetime(result["date"])

    holiday_lookup = calendar[
        [
            "holiday_name",
            "holiday_date",
            "observed_date",
            "is_shifted",
        ]
    ].copy()

    result = result.merge(
        holiday_lookup,
        left_on="date",
        right_on="observed_date",
        how="left",
        validate="many_to_one",
    )

    result["is_holiday"] = result["holiday_name"].notna()

    result["month_number"] = result["date"].dt.month

    result["weekday_number"] = result["date"].dt.weekday

    result["weekday_name"] = result["date"].dt.day_name()

    return result


def create_company_daily(
    consumption_data: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    daily = (
        consumption_data.groupby(
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

    return attach_holiday_flags(
        daily,
        calendar,
    )


def create_portfolio_daily(
    company_daily: pd.DataFrame,
) -> pd.DataFrame:
    portfolio = (
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

    calendar_columns = company_daily[
        [
            "date",
            "holiday_name",
            "holiday_date",
            "observed_date",
            "is_shifted",
            "is_holiday",
            "month_number",
            "weekday_number",
            "weekday_name",
        ]
    ].drop_duplicates("date")

    return portfolio.merge(
        calendar_columns,
        on="date",
        how="left",
        validate="one_to_one",
    )


def calculate_holiday_impacts(
    daily_data: pd.DataFrame,
    value_column: str,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    if group_columns is None:
        group_columns = []

    required_columns = {
        "date",
        "is_holiday",
        "holiday_name",
        "month_number",
        "weekday_number",
        value_column,
        *group_columns,
    }

    missing_columns = required_columns - set(daily_data.columns)

    if missing_columns:
        raise ValueError(
            "Holiday impact input is missing columns: " f"{sorted(missing_columns)}"
        )

    normal_days = daily_data[~daily_data["is_holiday"]].copy()

    baseline_keys = [
        *group_columns,
        "month_number",
        "weekday_number",
    ]

    baseline = (
        normal_days.groupby(baseline_keys)[value_column]
        .agg(
            baseline_mean="mean",
            baseline_median="median",
            baseline_std="std",
            baseline_days="count",
        )
        .reset_index()
    )

    holidays = daily_data[daily_data["is_holiday"]].copy()

    holidays = holidays.merge(
        baseline,
        on=baseline_keys,
        how="left",
        validate="many_to_one",
    )

    holidays["impact_kwh"] = holidays[value_column] - holidays["baseline_mean"]

    holidays["impact_percent"] = np.where(
        holidays["baseline_mean"] != 0,
        holidays["impact_kwh"] / holidays["baseline_mean"] * 100,
        np.nan,
    )

    valid_std = holidays["baseline_std"].notna() & (holidays["baseline_std"] > 0)

    holidays["z_vs_baseline"] = np.nan

    holidays.loc[
        valid_std,
        "z_vs_baseline",
    ] = (
        holidays.loc[
            valid_std,
            value_column,
        ]
        - holidays.loc[
            valid_std,
            "baseline_mean",
        ]
    ) / holidays.loc[
        valid_std,
        "baseline_std",
    ]

    return holidays.sort_values(
        [
            *group_columns,
            "date",
        ]
    ).reset_index(drop=True)


def create_portfolio_hourly_impact(
    consumption_data: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    portfolio_hourly = (
        consumption_data.groupby(
            [
                "date",
                "hour",
                "timestamp",
            ],
            as_index=False,
        )["energy_kwh"]
        .sum(min_count=1)
        .rename(
            columns={
                "energy_kwh": "portfolio_kwh",
            }
        )
    )

    portfolio_hourly = attach_holiday_flags(
        portfolio_hourly,
        calendar,
    )

    normal = portfolio_hourly[~portfolio_hourly["is_holiday"]].copy()

    baseline = (
        normal.groupby(
            [
                "month_number",
                "weekday_number",
                "hour",
            ]
        )["portfolio_kwh"]
        .agg(
            baseline_mean_kwh="mean",
            baseline_median_kwh="median",
            baseline_p10_kwh=lambda x: x.quantile(0.10),
            baseline_p90_kwh=lambda x: x.quantile(0.90),
            baseline_hours="count",
        )
        .reset_index()
    )

    holidays = portfolio_hourly[portfolio_hourly["is_holiday"]].copy()

    holidays = holidays.merge(
        baseline,
        on=[
            "month_number",
            "weekday_number",
            "hour",
        ],
        how="left",
        validate="many_to_one",
    )

    holidays["impact_kwh"] = holidays["portfolio_kwh"] - holidays["baseline_mean_kwh"]

    holidays["impact_percent"] = np.where(
        holidays["baseline_mean_kwh"] != 0,
        holidays["impact_kwh"] / holidays["baseline_mean_kwh"] * 100,
        np.nan,
    )

    return holidays.sort_values(
        [
            "date",
            "hour",
        ]
    ).reset_index(drop=True)


def create_holiday_outlier_detail(
    outliers: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    if outliers.empty:
        return outliers.copy()

    result = outliers.copy()

    if "date" not in result.columns:
        if "timestamp" not in result.columns:
            raise ValueError("Outliers require date or timestamp.")

        result["date"] = pd.to_datetime(
            result["timestamp"],
            errors="coerce",
        ).dt.normalize()
    else:
        result["date"] = pd.to_datetime(
            result["date"],
            errors="coerce",
        )

    holiday_lookup = calendar[
        [
            "holiday_name",
            "holiday_date",
            "observed_date",
            "is_shifted",
        ]
    ]

    result = result.merge(
        holiday_lookup,
        left_on="date",
        right_on="observed_date",
        how="inner",
        validate="many_to_one",
    )

    return result.sort_values(
        [
            "date",
            "absolute_z_score",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def create_holiday_outlier_summary(
    holiday_outliers: pd.DataFrame,
) -> pd.DataFrame:
    if holiday_outliers.empty:
        return pd.DataFrame(
            columns=[
                "holiday_name",
                "date",
                "outlier_hours",
                "companies",
                "max_abs_z_score",
            ]
        )

    return (
        holiday_outliers.groupby(
            [
                "holiday_name",
                "date",
            ]
        )
        .agg(
            outlier_hours=(
                "company_code",
                "size",
            ),
            companies=(
                "company_code",
                "nunique",
            ),
            max_abs_z_score=(
                "absolute_z_score",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "outlier_hours",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def run_holiday_analysis(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    quality_path: str | Path | None = DEFAULT_QUALITY_REPORT,
    outlier_path: str | Path = DEFAULT_OUTLIER_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, pd.DataFrame]:
    input_dir = Path(input_dir)
    outlier_path = Path(outlier_path)
    output_dir = Path(output_dir)
    report_path = Path(report_path)

    parquet_files = sorted(input_dir.glob("part_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Hapi 2 parquet files found in {input_dir}")

    quality_df = load_quality_report(quality_path)

    required_columns = [
        "date",
        "hour",
        "timestamp",
        "company_code",
        "flow_type",
        "energy_kwh",
        "source_sheet",
        "source_column",
    ]

    parts = []

    for parquet_file in parquet_files:
        print(f"Loading: {parquet_file.name}")

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

    consumption_data = pd.concat(
        parts,
        ignore_index=True,
    )

    if consumption_data.empty:
        raise ValueError("No valid consumption data available " "for holiday analysis.")

    calendar = get_kosovo_holiday_calendar()

    company_daily = create_company_daily(
        consumption_data,
        calendar,
    )

    portfolio_daily = create_portfolio_daily(company_daily)

    portfolio_impact = calculate_holiday_impacts(
        portfolio_daily,
        value_column="portfolio_kwh",
    )

    company_impact = calculate_holiday_impacts(
        company_daily,
        value_column="daily_kwh",
        group_columns=["company_code"],
    )

    hourly_impact = create_portfolio_hourly_impact(
        consumption_data,
        calendar,
    )

    if outlier_path.exists():
        outliers = pd.read_parquet(outlier_path)

        holiday_outliers = create_holiday_outlier_detail(
            outliers,
            calendar,
        )
    else:
        holiday_outliers = pd.DataFrame()

    holiday_outlier_summary = create_holiday_outlier_summary(holiday_outliers)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "calendar": calendar,
        "portfolio_daily": portfolio_daily,
        "company_daily": company_daily,
        "portfolio_impact": portfolio_impact,
        "company_impact": company_impact,
        "hourly_impact": hourly_impact,
        "holiday_outliers": holiday_outliers,
        "holiday_outlier_summary": (holiday_outlier_summary),
    }

    filenames = {
        "calendar": "holiday_calendar.parquet",
        "portfolio_daily": "holiday_portfolio_daily.parquet",
        "company_daily": "holiday_company_daily.parquet",
        "portfolio_impact": "holiday_portfolio_impact.parquet",
        "company_impact": "holiday_company_impact.parquet",
        "hourly_impact": "holiday_hourly_impact.parquet",
        "holiday_outliers": "holiday_outliers.parquet",
        "holiday_outlier_summary": ("holiday_outlier_summary.parquet"),
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
        calendar.to_excel(
            writer,
            sheet_name="Holiday Calendar",
            index=False,
        )

        portfolio_impact.to_excel(
            writer,
            sheet_name="Portfolio Impact",
            index=False,
        )

        company_impact.to_excel(
            writer,
            sheet_name="Company Impact",
            index=False,
        )

        hourly_impact.to_excel(
            writer,
            sheet_name="Hourly Impact",
            index=False,
        )

        holiday_outlier_summary.to_excel(
            writer,
            sheet_name="Outlier Summary",
            index=False,
        )

        holiday_outliers.to_excel(
            writer,
            sheet_name="Holiday Outliers",
            index=False,
        )

    observed_dates = portfolio_daily.loc[
        portfolio_daily["is_holiday"],
        "date",
    ].nunique()

    print()
    print("=" * 60)
    print("HAPI 5.2 - HOLIDAY ANALYSIS")
    print("=" * 60)

    print(f"Companies analyzed: " f"{company_daily['company_code'].nunique()}")

    print(f"Official observed holiday dates: " f"{len(calendar)}")

    print(f"Holiday dates found in data: " f"{observed_dates}")

    if not portfolio_impact.empty:
        print(
            "Average portfolio holiday impact: "
            f"{portfolio_impact['impact_percent'].mean():.2f}%"
        )

        print(
            "Median portfolio holiday impact: "
            f"{portfolio_impact['impact_percent'].median():.2f}%"
        )

    print(f"Outlier hours occurring on holidays: " f"{len(holiday_outliers):,}")

    if not holiday_outliers.empty:
        print(
            f"Companies with holiday outliers: "
            f"{holiday_outliers['company_code'].nunique()}"
        )

    print(f"Report saved: " f"{report_path}")

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
        "--outliers",
        default=DEFAULT_OUTLIER_PATH,
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
    )

    args = parser.parse_args()

    run_holiday_analysis(
        input_dir=args.input_dir,
        quality_path=args.quality_report,
        outlier_path=args.outliers,
        output_dir=args.output_dir,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
