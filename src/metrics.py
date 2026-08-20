import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROFILE_END, PROFILE_START

PEAK_HOURS = set(range(7, 19))

SUMMER_MONTHS = {6, 7, 8}
WINTER_MONTHS = {12, 1, 2}

SEASONALITY_THRESHOLD = 1.10


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan

    return float(numerator / denominator)


def _prepare_hourly_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data["hour"] = pd.to_numeric(
        data["hour"],
        errors="coerce",
    )

    data["energy_kwh"] = pd.to_numeric(
        data["energy_kwh"],
        errors="coerce",
    )

    data = data[
        data["date"].notna()
        & data["hour"].between(1, 24)
        & (data["date"] >= PROFILE_START)
        & (data["date"] < PROFILE_END)
    ].copy()

    return data


def _monthly_means(
    hourly_df: pd.DataFrame,
) -> pd.DataFrame:
    data = hourly_df.copy()

    data["month"] = data["date"].dt.to_period("M").astype(str)

    monthly = (
        data.groupby(
            "month",
            as_index=False,
        )["energy_kwh"]
        .mean()
        .rename(
            columns={
                "energy_kwh": "mean_kwh",
            }
        )
    )

    monthly["month_number"] = pd.to_datetime(monthly["month"] + "-01").dt.month

    return monthly


def _calculate_seasonality(
    monthly_df: pd.DataFrame,
    annual_mean: float,
) -> dict:
    if monthly_df.empty or pd.isna(annual_mean) or annual_mean == 0:
        return {
            "summer_mean_kwh": np.nan,
            "winter_mean_kwh": np.nan,
            "summer_index": np.nan,
            "winter_index": np.nan,
            "seasonality_index": np.nan,
            "seasonality": "none",
        }

    summer = monthly_df[monthly_df["month_number"].isin(SUMMER_MONTHS)]

    winter = monthly_df[monthly_df["month_number"].isin(WINTER_MONTHS)]

    summer_mean = float(summer["mean_kwh"].mean()) if not summer.empty else np.nan

    winter_mean = float(winter["mean_kwh"].mean()) if not winter.empty else np.nan

    summer_index = _safe_ratio(
        summer_mean,
        annual_mean,
    )

    winter_index = _safe_ratio(
        winter_mean,
        annual_mean,
    )

    valid_indices = [
        value
        for value in [
            summer_index,
            winter_index,
        ]
        if pd.notna(value)
    ]

    seasonality_index = max(valid_indices) if valid_indices else np.nan

    seasonality = "none"

    if pd.notna(summer_index) and pd.notna(winter_index):
        summer_vs_winter = _safe_ratio(
            summer_mean,
            winter_mean,
        )

        winter_vs_summer = _safe_ratio(
            winter_mean,
            summer_mean,
        )

        if (
            summer_index >= SEASONALITY_THRESHOLD
            and pd.notna(summer_vs_winter)
            and summer_vs_winter >= SEASONALITY_THRESHOLD
        ):
            seasonality = "summer"

        elif (
            winter_index >= SEASONALITY_THRESHOLD
            and pd.notna(winter_vs_summer)
            and winter_vs_summer >= SEASONALITY_THRESHOLD
        ):
            seasonality = "winter"

    return {
        "summer_mean_kwh": summer_mean,
        "winter_mean_kwh": winter_mean,
        "summer_index": summer_index,
        "winter_index": winter_index,
        "seasonality_index": seasonality_index,
        "seasonality": seasonality,
    }


def _calculate_trend(
    monthly_df: pd.DataFrame,
) -> dict:
    if monthly_df.empty:
        return {
            "first_three_month_mean_kwh": np.nan,
            "last_month_mean_kwh": np.nan,
            "trend_percent": np.nan,
            "first_active_month": None,
            "last_active_month": None,
        }

    monthly = monthly_df.sort_values("month").reset_index(drop=True)

    first_three = monthly.head(3)

    first_three_mean = float(first_three["mean_kwh"].mean())

    last_month_mean = float(monthly.iloc[-1]["mean_kwh"])

    if first_three_mean == 0:
        trend_percent = np.nan
    else:
        trend_percent = (last_month_mean - first_three_mean) / first_three_mean * 100

    return {
        "first_three_month_mean_kwh": (first_three_mean),
        "last_month_mean_kwh": (last_month_mean),
        "trend_percent": (float(trend_percent) if pd.notna(trend_percent) else np.nan),
        "first_active_month": monthly.iloc[0]["month"],
        "last_active_month": monthly.iloc[-1]["month"],
    }


def calculate_profile_metrics(
    hourly_df: pd.DataFrame,
) -> dict:
    data = _prepare_hourly_data(hourly_df)

    valid = data.dropna(subset=["energy_kwh"]).copy()

    if valid.empty:
        return {
            "observation_count": 0,
            "total_kwh": np.nan,
            "mean_kwh": np.nan,
            "max_kwh": np.nan,
            "peak_mean_kwh": np.nan,
            "off_peak_mean_kwh": np.nan,
            "peak_ratio": np.nan,
            "weekday_mean_kwh": np.nan,
            "weekend_mean_kwh": np.nan,
            "weekday_weekend_ratio": np.nan,
            "cv": np.nan,
            "load_factor": np.nan,
            "summer_mean_kwh": np.nan,
            "winter_mean_kwh": np.nan,
            "summer_index": np.nan,
            "winter_index": np.nan,
            "seasonality_index": np.nan,
            "seasonality": "none",
            "first_three_month_mean_kwh": np.nan,
            "last_month_mean_kwh": np.nan,
            "trend_percent": np.nan,
            "first_active_month": None,
            "last_active_month": None,
        }

    mean_kwh = float(valid["energy_kwh"].mean())

    max_kwh = float(valid["energy_kwh"].max())

    total_kwh = float(valid["energy_kwh"].sum())

    peak_values = valid.loc[
        valid["hour"].isin(PEAK_HOURS),
        "energy_kwh",
    ]

    off_peak_values = valid.loc[
        ~valid["hour"].isin(PEAK_HOURS),
        "energy_kwh",
    ]

    peak_mean = float(peak_values.mean()) if not peak_values.empty else np.nan

    off_peak_mean = (
        float(off_peak_values.mean()) if not off_peak_values.empty else np.nan
    )

    valid["day_of_week"] = valid["date"].dt.dayofweek

    weekday_values = valid.loc[
        valid["day_of_week"] < 5,
        "energy_kwh",
    ]

    weekend_values = valid.loc[
        valid["day_of_week"] >= 5,
        "energy_kwh",
    ]

    weekday_mean = float(weekday_values.mean()) if not weekday_values.empty else np.nan

    weekend_mean = float(weekend_values.mean()) if not weekend_values.empty else np.nan

    standard_deviation = float(valid["energy_kwh"].std(ddof=0))

    cv = _safe_ratio(
        standard_deviation,
        mean_kwh,
    )

    load_factor = _safe_ratio(
        mean_kwh,
        max_kwh,
    )

    monthly = _monthly_means(valid)

    seasonality = _calculate_seasonality(
        monthly,
        mean_kwh,
    )

    trend = _calculate_trend(monthly)

    return {
        "observation_count": len(valid),
        "total_kwh": total_kwh,
        "mean_kwh": mean_kwh,
        "max_kwh": max_kwh,
        "peak_mean_kwh": peak_mean,
        "off_peak_mean_kwh": off_peak_mean,
        "peak_ratio": _safe_ratio(
            peak_mean,
            off_peak_mean,
        ),
        "weekday_mean_kwh": weekday_mean,
        "weekend_mean_kwh": weekend_mean,
        "weekday_weekend_ratio": (
            _safe_ratio(
                weekday_mean,
                weekend_mean,
            )
        ),
        "cv": cv,
        "load_factor": load_factor,
        **seasonality,
        **trend,
    }


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
        "coverage_status",
    }

    missing = required_columns - set(quality_df.columns)

    if missing:
        raise ValueError("Quality report is missing columns: " f"{sorted(missing)}")

    return quality_df[
        [
            "source_sheet",
            "source_column",
            "quality_status",
            "coverage_status",
        ]
    ].copy()


def _add_quality_status(
    data: pd.DataFrame,
    quality_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if quality_df is None:
        data = data.copy()
        data["quality_status"] = "unknown"
        data["coverage_status"] = "unknown"

        return data

    return data.merge(
        quality_df,
        on=[
            "source_sheet",
            "source_column",
        ],
        how="left",
        validate="many_to_one",
    )


def _calculate_meter_profiles(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = []
    monthly_rows = []

    group_columns = [
        "source_sheet",
        "source_column",
        "company_code",
        "meter_id",
    ]

    for keys, group in data.groupby(
        group_columns,
        sort=False,
    ):
        (
            source_sheet,
            source_column,
            company_code,
            meter_id,
        ) = keys

        metrics = calculate_profile_metrics(group)

        quality_status = (
            group["quality_status"].dropna().iloc[0]
            if group["quality_status"].notna().any()
            else "unknown"
        )

        coverage_status = (
            group["coverage_status"].dropna().iloc[0]
            if group["coverage_status"].notna().any()
            else "unknown"
        )

        profiles.append(
            {
                "company_code": company_code,
                "meter_id": meter_id,
                "source_sheet": source_sheet,
                "source_column": source_column,
                "flow_type": "consumption",
                "quality_status": quality_status,
                "coverage_status": coverage_status,
                **metrics,
            }
        )

        monthly = _monthly_means(
            _prepare_hourly_data(group).dropna(subset=["energy_kwh"])
        )

        for _, row in monthly.iterrows():
            monthly_rows.append(
                {
                    "company_code": company_code,
                    "meter_id": meter_id,
                    "source_sheet": source_sheet,
                    "month": row["month"],
                    "mean_kwh": row["mean_kwh"],
                }
            )

    return (
        pd.DataFrame(profiles),
        pd.DataFrame(monthly_rows),
    )


def _aggregate_company_hourly(
    data: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "company_code",
        "date",
        "hour",
        "timestamp",
    ]

    return data.groupby(
        group_columns,
        as_index=False,
    )[
        "energy_kwh"
    ].sum(min_count=1)


def _calculate_company_profiles(
    company_hourly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = []
    monthly_rows = []

    for company_code, group in company_hourly.groupby(
        "company_code",
        sort=False,
    ):
        metrics = calculate_profile_metrics(group)

        profiles.append(
            {
                "company_code": company_code,
                **metrics,
            }
        )

        monthly = _monthly_means(
            _prepare_hourly_data(group).dropna(subset=["energy_kwh"])
        )

        for _, row in monthly.iterrows():
            monthly_rows.append(
                {
                    "company_code": company_code,
                    "month": row["month"],
                    "mean_kwh": row["mean_kwh"],
                }
            )

    return (
        pd.DataFrame(profiles),
        pd.DataFrame(monthly_rows),
    )


def _calculate_company_hourly_profiles(
    company_hourly: pd.DataFrame,
) -> pd.DataFrame:
    data = _prepare_hourly_data(company_hourly)

    data = data.dropna(
        subset=[
            "energy_kwh",
        ]
    ).copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "company_code",
                "profile_type",
                "hour",
                "observations",
                "mean_kwh",
                "median_kwh",
                "p10_kwh",
                "p90_kwh",
            ]
        )

    data["day_type"] = np.where(
        data["date"].dt.dayofweek < 5,
        "weekday",
        "weekend",
    )

    def aggregate_profile(
        profile_data: pd.DataFrame,
        profile_type: str,
    ) -> pd.DataFrame:
        profile = (
            profile_data.groupby(
                [
                    "company_code",
                    "hour",
                ]
            )
            .agg(
                observations=(
                    "energy_kwh",
                    "count",
                ),
                mean_kwh=(
                    "energy_kwh",
                    "mean",
                ),
                median_kwh=(
                    "energy_kwh",
                    "median",
                ),
                p10_kwh=(
                    "energy_kwh",
                    lambda x: x.quantile(0.10),
                ),
                p90_kwh=(
                    "energy_kwh",
                    lambda x: x.quantile(0.90),
                ),
            )
            .reset_index()
        )

        profile["profile_type"] = profile_type

        return profile

    overall = aggregate_profile(
        data,
        "all_days",
    )

    weekday = aggregate_profile(
        data[data["day_type"] == "weekday"],
        "weekday",
    )

    weekend = aggregate_profile(
        data[data["day_type"] == "weekend"],
        "weekend",
    )

    result = pd.concat(
        [
            overall,
            weekday,
            weekend,
        ],
        ignore_index=True,
    )

    return (
        result[
            [
                "company_code",
                "profile_type",
                "hour",
                "observations",
                "mean_kwh",
                "median_kwh",
                "p10_kwh",
                "p90_kwh",
            ]
        ]
        .sort_values(
            [
                "company_code",
                "profile_type",
                "hour",
            ]
        )
        .reset_index(drop=True)
    )


def _calculate_meter_hourly_profiles(
    data: pd.DataFrame,
) -> pd.DataFrame:
    prepared = _prepare_hourly_data(data)

    prepared = prepared.dropna(
        subset=[
            "energy_kwh",
        ]
    ).copy()

    columns = [
        "company_code",
        "meter_id",
        "source_sheet",
        "source_column",
        "hour",
        "observations",
        "mean_kwh",
        "median_kwh",
        "p10_kwh",
        "p90_kwh",
    ]

    if prepared.empty:
        return pd.DataFrame(columns=columns)

    profile = (
        prepared.groupby(
            [
                "company_code",
                "meter_id",
                "source_sheet",
                "source_column",
                "hour",
            ]
        )
        .agg(
            observations=(
                "energy_kwh",
                "count",
            ),
            mean_kwh=(
                "energy_kwh",
                "mean",
            ),
            median_kwh=(
                "energy_kwh",
                "median",
            ),
            p10_kwh=(
                "energy_kwh",
                lambda x: x.quantile(0.10),
            ),
            p90_kwh=(
                "energy_kwh",
                lambda x: x.quantile(0.90),
            ),
        )
        .reset_index()
    )

    return (
        profile[columns]
        .sort_values(
            [
                "company_code",
                "meter_id",
                "source_sheet",
                "source_column",
                "hour",
            ]
        )
        .reset_index(drop=True)
    )


def _add_meter_similarity_flags(
    meter_profiles: pd.DataFrame,
) -> pd.DataFrame:
    result = meter_profiles.copy()

    result["profile_similarity"] = "single_meter"

    metric_columns = [
        "peak_ratio",
        "weekday_weekend_ratio",
        "cv",
        "load_factor",
    ]

    for company_code, indexes in result.groupby("company_code").groups.items():
        indexes = list(indexes)

        if len(indexes) <= 1:
            continue

        company_meters = result.loc[
            indexes,
            metric_columns,
        ]

        medians = company_meters.median(numeric_only=True)

        for index in indexes:
            row = result.loc[
                index,
                metric_columns,
            ]

            differences = []

            for column in [
                "peak_ratio",
                "weekday_weekend_ratio",
                "cv",
            ]:
                median = medians[column]
                value = row[column]

                if pd.isna(value) or pd.isna(median):
                    continue

                if median == 0:
                    difference = abs(value - median)
                else:
                    difference = abs(value - median) / abs(median)

                differences.append(difference > 0.50)

            load_value = row["load_factor"]

            load_median = medians["load_factor"]

            if pd.notna(load_value) and pd.notna(load_median):
                differences.append(abs(load_value - load_median) > 0.20)

            result.loc[
                index,
                "profile_similarity",
            ] = (
                "different" if any(differences) else "similar"
            )

    return result


def run_profile_analysis(
    input_dir: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    quality_path: str | Path | None = None,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    report_path = Path(report_path)

    parquet_files = sorted(input_dir.glob("part_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Hapi 2 parquet files found in " f"{input_dir}")

    quality_df = load_quality_report(quality_path)

    if quality_df is None:
        print("Quality report not found. " "No unusable meters will be excluded.")

    meter_profile_parts = []
    meter_monthly_parts = []
    meter_hourly_profile_parts = []
    company_hourly_parts = []

    required_columns = [
        "date",
        "hour",
        "timestamp",
        "company_code",
        "meter_id",
        "flow_type",
        "energy_kwh",
        "source_sheet",
        "source_column",
    ]

    for parquet_file in parquet_files:
        print(f"Processing: " f"{parquet_file.name}")

        data = pd.read_parquet(
            parquet_file,
            columns=required_columns,
        )

        data = data[data["flow_type"] == "consumption"].copy()

        data = _add_quality_status(
            data,
            quality_df,
        )

        before_count = (
            data[
                [
                    "source_sheet",
                    "source_column",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        data = data[data["quality_status"] != "unusable"].copy()

        after_count = (
            data[
                [
                    "source_sheet",
                    "source_column",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        print(f"  Consumption meters used: " f"{after_count}/{before_count}")

        meter_hourly_profiles = _calculate_meter_hourly_profiles(data)

        (
            meter_profiles,
            meter_monthly,
        ) = _calculate_meter_profiles(data)

        meter_profile_parts.append(meter_profiles)

        meter_monthly_parts.append(meter_monthly)

        meter_hourly_profile_parts.append(meter_hourly_profiles)

        company_hourly_parts.append(_aggregate_company_hourly(data))

    meter_profiles_df = pd.concat(
        meter_profile_parts,
        ignore_index=True,
    )

    meter_monthly_df = pd.concat(
        meter_monthly_parts,
        ignore_index=True,
    )

    meter_hourly_profile_df = pd.concat(
        meter_hourly_profile_parts,
        ignore_index=True,
    )

    meter_key_columns = [
        "company_code",
        "meter_id",
        "source_sheet",
        "source_column",
    ]

    profile_meter_keys = set(
        map(
            tuple,
            meter_profiles_df[meter_key_columns].drop_duplicates().to_numpy(),
        )
    )

    hourly_meter_keys = set(
        map(
            tuple,
            meter_hourly_profile_df[meter_key_columns].drop_duplicates().to_numpy(),
        )
    )

    if profile_meter_keys != hourly_meter_keys:
        missing_hourly = profile_meter_keys - hourly_meter_keys

        unexpected_hourly = hourly_meter_keys - profile_meter_keys

        raise ValueError(
            "Meter hourly profile population does not match "
            "the Hapi 3 meter profile population. "
            f"Missing hourly meters: {len(missing_hourly)}; "
            f"unexpected hourly meters: "
            f"{len(unexpected_hourly)}."
        )

    company_hourly = pd.concat(
        company_hourly_parts,
        ignore_index=True,
    )

    company_hourly = company_hourly.groupby(
        [
            "company_code",
            "date",
            "hour",
            "timestamp",
        ],
        as_index=False,
    )["energy_kwh"].sum(min_count=1)

    (
        company_profiles_df,
        company_monthly_df,
    ) = _calculate_company_profiles(company_hourly)

    company_hourly_profile_df = _calculate_company_hourly_profiles(company_hourly)

    meter_profiles_df = _add_meter_similarity_flags(meter_profiles_df)

    meter_counts = (
        meter_profiles_df.groupby("company_code")["meter_id"]
        .nunique()
        .rename("meter_count_used")
    )

    company_profiles_df = company_profiles_df.merge(
        meter_counts,
        on="company_code",
        how="left",
    )

    company_profiles_df["business_sector"] = pd.NA

    company_profiles_df["company_size"] = pd.NA

    company_profiles_df["voltage_level"] = pd.NA

    company_profiles_df["metadata_status"] = "pending_mapping"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    company_profiles_df.to_parquet(
        output_dir / "company_profiles.parquet",
        index=False,
    )

    meter_profiles_df.to_parquet(
        output_dir / "meter_profiles.parquet",
        index=False,
    )

    company_monthly_df.to_parquet(
        output_dir / "company_monthly.parquet",
        index=False,
    )

    meter_monthly_df.to_parquet(
        output_dir / "meter_monthly.parquet",
        index=False,
    )

    company_hourly_profile_df.to_parquet(
        output_dir / "company_hourly_profile.parquet",
        index=False,
    )

    meter_hourly_profile_df.to_parquet(
        output_dir / "meter_hourly_profile.parquet",
        index=False,
    )

    with pd.ExcelWriter(
        report_path,
        engine="openpyxl",
    ) as writer:
        company_profiles_df.to_excel(
            writer,
            sheet_name="Company Profiles",
            index=False,
        )

        meter_profiles_df.to_excel(
            writer,
            sheet_name="Meter Profiles",
            index=False,
        )

        company_monthly_df.to_excel(
            writer,
            sheet_name="Company Monthly",
            index=False,
        )

        meter_monthly_df.to_excel(
            writer,
            sheet_name="Meter Monthly",
            index=False,
        )
        company_hourly_profile_df.to_excel(
            writer,
            sheet_name="Company Hourly",
            index=False,
        )

        meter_hourly_profile_df.to_excel(
            writer,
            sheet_name="Meter Hourly",
            index=False,
        )

    print()
    print("=" * 60)
    print("CONSUMPTION PROFILE METRICS")
    print("=" * 60)

    print(f"Companies analyzed: " f"{len(company_profiles_df)}")

    print(f"Consumption meters analyzed: " f"{len(meter_profiles_df)}")

    print()
    print("Seasonality:")
    print(company_profiles_df["seasonality"].value_counts(dropna=False).to_string())

    print()
    print(f"Report saved: " f"{report_path}")

    return {
        "company_profiles": (company_profiles_df),
        "meter_profiles": (meter_profiles_df),
        "company_monthly": (company_monthly_df),
        "meter_monthly": (meter_monthly_df),
        "company_hourly_profile": (company_hourly_profile_df),
        "meter_hourly_profile": (meter_hourly_profile_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default=("data/processed/" "hourly_long"),
    )

    parser.add_argument(
        "--output-dir",
        default=("data/processed/" "profile_metrics"),
    )

    parser.add_argument(
        "--quality-report",
        default=("reports/" "hapi_1_data_quality.xlsx"),
    )

    parser.add_argument(
        "--report",
        default=("reports/" "hapi_3_profile_metrics.xlsx"),
    )

    args = parser.parse_args()

    run_profile_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        quality_path=args.quality_report,
    )


if __name__ == "__main__":
    main()
