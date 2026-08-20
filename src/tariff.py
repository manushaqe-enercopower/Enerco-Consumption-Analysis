import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_START = pd.Timestamp("2025-07-01")
PROFILE_END = pd.Timestamp("2026-07-01")

DEFAULT_INPUT_DIR = Path("data/processed/hourly_long")
DEFAULT_QUALITY_REPORT = Path("reports/hapi_1_data_quality.xlsx")
DEFAULT_OUTPUT_DIR = Path("data/processed/factors/tariff")
DEFAULT_REPORT_PATH = Path("reports/hapi_5_tariff.xlsx")


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


def prepare_tariff_data(
    data: pd.DataFrame,
    quality_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_columns = {
        "date",
        "hour",
        "timestamp",
        "tariff",
        "company_code",
        "flow_type",
        "energy_kwh",
        "source_sheet",
        "source_column",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Tariff input is missing columns: " f"{sorted(missing_columns)}"
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

    result["tariff"] = result["tariff"].astype("string").str.strip().str.upper()

    result = result[
        result["date"].notna()
        & result["timestamp"].notna()
        & result["hour"].between(1, 24)
        & (result["date"] >= PROFILE_START)
        & (result["date"] < PROFILE_END)
        & (result["flow_type"] == "consumption")
        & result["energy_kwh"].notna()
        & result["tariff"].notna()
        & (result["tariff"] != "")
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

    result["month"] = result["date"].dt.to_period("M").astype(str)

    result["month_number"] = result["date"].dt.month

    return result.reset_index(drop=True)


def create_portfolio_hourly(
    tariff_data: pd.DataFrame,
) -> pd.DataFrame:
    return (
        tariff_data.groupby(
            [
                "date",
                "hour",
                "timestamp",
                "tariff",
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


def create_tariff_portfolio_summary(
    tariff_data: pd.DataFrame,
) -> pd.DataFrame:
    portfolio_hourly = create_portfolio_hourly(tariff_data)

    energy_totals = (
        tariff_data.groupby(
            "tariff",
            as_index=False,
        )["energy_kwh"]
        .sum(min_count=1)
        .rename(
            columns={
                "energy_kwh": "total_kwh",
            }
        )
    )

    hourly_stats = (
        portfolio_hourly.groupby("tariff")
        .agg(
            active_hours=(
                "timestamp",
                "nunique",
            ),
            mean_hourly_portfolio_kwh=(
                "portfolio_kwh",
                "mean",
            ),
            median_hourly_portfolio_kwh=(
                "portfolio_kwh",
                "median",
            ),
            p10_hourly_portfolio_kwh=(
                "portfolio_kwh",
                lambda x: x.quantile(0.10),
            ),
            p90_hourly_portfolio_kwh=(
                "portfolio_kwh",
                lambda x: x.quantile(0.90),
            ),
            max_hourly_portfolio_kwh=(
                "portfolio_kwh",
                "max",
            ),
        )
        .reset_index()
    )

    summary = energy_totals.merge(
        hourly_stats,
        on="tariff",
        how="left",
        validate="one_to_one",
    )

    total_energy = summary["total_kwh"].sum()

    summary["energy_share_percent"] = np.where(
        total_energy > 0,
        summary["total_kwh"] / total_energy * 100,
        np.nan,
    )

    return summary.sort_values(
        "total_kwh",
        ascending=False,
    ).reset_index(drop=True)


def create_company_tariff_summary(
    tariff_data: pd.DataFrame,
) -> pd.DataFrame:
    company_hourly = (
        tariff_data.groupby(
            [
                "company_code",
                "date",
                "hour",
                "timestamp",
                "tariff",
            ],
            as_index=False,
        )["energy_kwh"]
        .sum(min_count=1)
        .rename(
            columns={
                "energy_kwh": "company_hourly_kwh",
            }
        )
    )

    summary = (
        company_hourly.groupby(
            [
                "company_code",
                "tariff",
            ]
        )
        .agg(
            total_kwh=(
                "company_hourly_kwh",
                "sum",
            ),
            active_hours=(
                "timestamp",
                "nunique",
            ),
            mean_hourly_kwh=(
                "company_hourly_kwh",
                "mean",
            ),
            median_hourly_kwh=(
                "company_hourly_kwh",
                "median",
            ),
            max_hourly_kwh=(
                "company_hourly_kwh",
                "max",
            ),
        )
        .reset_index()
    )

    company_totals = (
        summary.groupby("company_code")["total_kwh"]
        .sum()
        .rename("company_total_kwh")
        .reset_index()
    )

    summary = summary.merge(
        company_totals,
        on="company_code",
        how="left",
        validate="many_to_one",
    )

    summary["tariff_share_percent"] = np.where(
        summary["company_total_kwh"] > 0,
        summary["total_kwh"] / summary["company_total_kwh"] * 100,
        np.nan,
    )

    return summary.sort_values(
        [
            "company_code",
            "total_kwh",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def create_monthly_tariff_summary(
    tariff_data: pd.DataFrame,
) -> pd.DataFrame:
    portfolio_hourly = create_portfolio_hourly(tariff_data)

    portfolio_hourly["month"] = portfolio_hourly["date"].dt.to_period("M").astype(str)

    summary = (
        portfolio_hourly.groupby(
            [
                "month",
                "tariff",
            ]
        )
        .agg(
            total_kwh=(
                "portfolio_kwh",
                "sum",
            ),
            active_hours=(
                "timestamp",
                "nunique",
            ),
            mean_hourly_portfolio_kwh=(
                "portfolio_kwh",
                "mean",
            ),
            max_hourly_portfolio_kwh=(
                "portfolio_kwh",
                "max",
            ),
        )
        .reset_index()
    )

    month_totals = (
        summary.groupby("month")["total_kwh"]
        .sum()
        .rename("month_total_kwh")
        .reset_index()
    )

    summary = summary.merge(
        month_totals,
        on="month",
        how="left",
        validate="many_to_one",
    )

    summary["tariff_share_percent"] = np.where(
        summary["month_total_kwh"] > 0,
        summary["total_kwh"] / summary["month_total_kwh"] * 100,
        np.nan,
    )

    return summary.sort_values(
        [
            "month",
            "tariff",
        ]
    ).reset_index(drop=True)


def create_hourly_tariff_profile(
    tariff_data: pd.DataFrame,
) -> pd.DataFrame:
    portfolio_hourly = create_portfolio_hourly(tariff_data)

    return (
        portfolio_hourly.groupby(
            [
                "tariff",
                "hour",
            ]
        )
        .agg(
            observation_hours=(
                "timestamp",
                "nunique",
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
            max_portfolio_kwh=(
                "portfolio_kwh",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "tariff",
                "hour",
            ]
        )
        .reset_index(drop=True)
    )


def create_tariff_schedule(
    tariff_data: pd.DataFrame,
) -> pd.DataFrame:
    unique_schedule = tariff_data[
        [
            "date",
            "hour",
            "timestamp",
            "tariff",
        ]
    ].drop_duplicates()

    unique_schedule["month_number"] = unique_schedule["date"].dt.month

    counts = (
        unique_schedule.groupby(
            [
                "month_number",
                "hour",
                "tariff",
            ]
        )
        .size()
        .rename("occurrence_count")
        .reset_index()
    )

    slot_totals = (
        counts.groupby(
            [
                "month_number",
                "hour",
            ]
        )["occurrence_count"]
        .sum()
        .rename("slot_occurrence_count")
        .reset_index()
    )

    counts = counts.merge(
        slot_totals,
        on=[
            "month_number",
            "hour",
        ],
        how="left",
        validate="many_to_one",
    )

    counts["occurrence_percent"] = np.where(
        counts["slot_occurrence_count"] > 0,
        counts["occurrence_count"] / counts["slot_occurrence_count"] * 100,
        np.nan,
    )

    max_occurrence = counts.groupby(
        [
            "month_number",
            "hour",
        ]
    )[
        "occurrence_count"
    ].transform("max")

    counts["is_dominant_tariff"] = counts["occurrence_count"] == max_occurrence

    return counts.sort_values(
        [
            "month_number",
            "hour",
            "tariff",
        ]
    ).reset_index(drop=True)


def analyze_tariffs(
    tariff_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "portfolio_summary": (create_tariff_portfolio_summary(tariff_data)),
        "company_tariff": (create_company_tariff_summary(tariff_data)),
        "monthly_tariff": (create_monthly_tariff_summary(tariff_data)),
        "hourly_profile": (create_hourly_tariff_profile(tariff_data)),
        "tariff_schedule": (create_tariff_schedule(tariff_data)),
    }


def run_tariff_analysis(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    quality_path: str | Path | None = DEFAULT_QUALITY_REPORT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, pd.DataFrame]:
    input_dir = Path(input_dir)
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
        "tariff",
        "company_code",
        "flow_type",
        "energy_kwh",
        "source_sheet",
        "source_column",
    ]

    tariff_parts = []

    for parquet_file in parquet_files:
        print(f"Loading: {parquet_file.name}")

        data = pd.read_parquet(
            parquet_file,
            columns=required_columns,
        )

        tariff_parts.append(
            prepare_tariff_data(
                data,
                quality_df=quality_df,
            )
        )

    tariff_data = pd.concat(
        tariff_parts,
        ignore_index=True,
    )

    if tariff_data.empty:
        raise ValueError("No valid consumption data available " "for tariff analysis.")

    results = analyze_tariffs(tariff_data)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = {
        "portfolio_summary": ("tariff_portfolio_summary.parquet"),
        "company_tariff": ("tariff_company.parquet"),
        "monthly_tariff": ("tariff_monthly.parquet"),
        "hourly_profile": ("tariff_hourly_profile.parquet"),
        "tariff_schedule": ("tariff_schedule.parquet"),
    }

    for key, filename in output_files.items():
        results[key].to_parquet(
            output_dir / filename,
            index=False,
        )

    with pd.ExcelWriter(
        report_path,
        engine="openpyxl",
    ) as writer:
        results["portfolio_summary"].to_excel(
            writer,
            sheet_name="Portfolio Summary",
            index=False,
        )

        results["company_tariff"].to_excel(
            writer,
            sheet_name="Company Tariff",
            index=False,
        )

        results["monthly_tariff"].to_excel(
            writer,
            sheet_name="Monthly Tariff",
            index=False,
        )

        results["hourly_profile"].to_excel(
            writer,
            sheet_name="Hourly Profile",
            index=False,
        )

        results["tariff_schedule"].to_excel(
            writer,
            sheet_name="Tariff Schedule",
            index=False,
        )

    portfolio_summary = results["portfolio_summary"]

    print()
    print("=" * 60)
    print("TARIFF ANALYSIS")
    print("=" * 60)

    print(f"Companies analyzed: " f"{tariff_data['company_code'].nunique()}")

    print(f"Consumption observations used: " f"{len(tariff_data):,}")

    print("Tariffs found: " + ", ".join(portfolio_summary["tariff"].astype(str)))

    for _, row in portfolio_summary.iterrows():
        print(
            f"  {row['tariff']}: "
            f"{row['total_kwh']:,.2f} kWh "
            f"({row['energy_share_percent']:.2f}%)"
        )

    print(f"Report saved: " f"{report_path}")

    print(f"Processed data saved: " f"{output_dir}")

    return {
        "tariff_data": tariff_data,
        **results,
    }


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

    args = parser.parse_args()

    run_tariff_analysis(
        input_dir=args.input_dir,
        quality_path=args.quality_report,
        output_dir=args.output_dir,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
