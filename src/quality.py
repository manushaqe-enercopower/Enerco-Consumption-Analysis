import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROFILE_END, PROFILE_START
from src.loader import load_sheet

EXPECTED_HOURS = set(range(1, 25))


def _find_zero_runs(values: pd.Series) -> list[tuple[int, int, int]]:
    zero_mask = values.eq(0).fillna(False).to_numpy()

    if not zero_mask.any():
        return []

    starts = np.flatnonzero(zero_mask & ~np.r_[False, zero_mask[:-1]])

    ends = np.flatnonzero(zero_mask & ~np.r_[zero_mask[1:], False])

    return [
        (int(start), int(end), int(end - start + 1)) for start, end in zip(starts, ends)
    ]


def analyze_timeline(
    base_df: pd.DataFrame,
    sheet_name: str,
) -> dict:
    dates = pd.to_datetime(
        base_df["Date"],
        errors="coerce",
    )

    hours = pd.to_numeric(
        base_df["Hour"],
        errors="coerce",
    )

    invalid_date_count = int(dates.isna().sum())

    invalid_hour_count = int((~hours.isin(EXPECTED_HOURS)).sum())

    timeline = pd.DataFrame(
        {
            "Date": dates,
            "Hour": hours,
        }
    )

    valid_timeline = timeline[
        timeline["Date"].notna() & timeline["Hour"].isin(EXPECTED_HOURS)
    ].copy()

    duplicate_count = int(
        valid_timeline.duplicated(
            subset=["Date", "Hour"],
            keep=False,
        ).sum()
    )

    incomplete_days = 0
    days_with_23_hours = 0
    days_with_25_hours = 0

    for _, day_df in valid_timeline.groupby("Date"):
        hour_count = len(day_df)

        unique_hours = set(day_df["Hour"].astype(int))

        if hour_count == 23:
            days_with_23_hours += 1

        if hour_count == 25:
            days_with_25_hours += 1

        if unique_hours != EXPECTED_HOURS or hour_count != 24:
            incomplete_days += 1

    if valid_timeline.empty:
        min_date = None
        max_date = None
        missing_calendar_hours = 0
    else:
        min_date = valid_timeline["Date"].min()
        max_date = valid_timeline["Date"].max()

        calendar_days = (max_date - min_date).days + 1

        expected_rows = calendar_days * 24

        unique_date_hours = valid_timeline[["Date", "Hour"]].drop_duplicates()

        missing_calendar_hours = max(
            0,
            expected_rows - len(unique_date_hours),
        )

    timeline_status = (
        "clean"
        if (
            invalid_date_count == 0
            and invalid_hour_count == 0
            and duplicate_count == 0
            and incomplete_days == 0
            and missing_calendar_hours == 0
        )
        else "review"
    )

    return {
        "source_sheet": sheet_name,
        "row_count": len(base_df),
        "min_date": min_date,
        "max_date": max_date,
        "invalid_date_rows": invalid_date_count,
        "invalid_hour_rows": invalid_hour_count,
        "duplicate_date_hour_rows": duplicate_count,
        "incomplete_days": incomplete_days,
        "days_with_23_hours": days_with_23_hours,
        "days_with_25_hours": days_with_25_hours,
        "missing_calendar_hours": missing_calendar_hours,
        "timeline_status": timeline_status,
    }


def analyze_meter(
    values: pd.Series,
    dates: pd.Series,
    metadata: dict,
    hours: pd.Series | None = None,
) -> dict:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).reset_index(drop=True)

    dates = pd.to_datetime(
        dates,
        errors="coerce",
    ).reset_index(drop=True)

    if hours is not None:
        hours = pd.to_numeric(
            hours,
            errors="coerce",
        ).reset_index(drop=True)

        timestamps = dates + pd.to_timedelta(
            hours - 1,
            unit="h",
        )
    else:
        timestamps = dates

    total_hours = len(numeric)

    missing_hours = int(numeric.isna().sum())

    missing_percent = missing_hours / total_hours * 100 if total_hours else 100.0

    # ---------------------------------------------------------
    # Annual profile-window completeness
    # July 2025 -> June 2026, as defined by the methodology.
    # ---------------------------------------------------------
    profile_mask = dates.notna() & (dates >= PROFILE_START) & (dates < PROFILE_END)

    profile_values = numeric.loc[profile_mask].reset_index(drop=True)

    profile_total_hours = len(profile_values)

    profile_missing_hours = int(profile_values.isna().sum())

    profile_missing_percent = (
        profile_missing_hours / profile_total_hours * 100
        if profile_total_hours
        else 100.0
    )

    profile_observed_hours = int(profile_values.notna().sum())

    valid_mask = numeric.notna()

    valid_positions = np.flatnonzero(valid_mask.to_numpy())

    if len(valid_positions) == 0:
        first_valid_timestamp = pd.NaT
        last_valid_timestamp = pd.NaT

        observed_hours = 0
        active_span_hours = 0

        leading_missing_hours = total_hours

        trailing_missing_hours = 0

        internal_missing_hours = total_hours

        internal_missing_percent = 100.0

        coverage_percent = 0.0
        coverage_status = "no_data"

    else:
        first_valid_index = int(valid_positions[0])

        last_valid_index = int(valid_positions[-1])

        first_valid_timestamp = timestamps.iloc[first_valid_index]

        last_valid_timestamp = timestamps.iloc[last_valid_index]

        observed_hours = int(valid_mask.sum())

        leading_missing_hours = first_valid_index

        trailing_missing_hours = total_hours - last_valid_index - 1

        active_values = numeric.iloc[first_valid_index : last_valid_index + 1]

        active_span_hours = len(active_values)

        internal_missing_hours = int(active_values.isna().sum())

        internal_missing_percent = (
            internal_missing_hours / active_span_hours * 100
            if active_span_hours
            else 100.0
        )

        coverage_percent = active_span_hours / total_hours * 100 if total_hours else 0.0

        if leading_missing_hours == 0 and trailing_missing_hours == 0:
            coverage_status = "full_period"
        else:
            coverage_status = "partial_period"

    negative_count = int((numeric < 0).sum())

    zero_runs = _find_zero_runs(numeric)

    long_zero_runs = [run for run in zero_runs if run[2] > 48]

    zero_run_count = len(long_zero_runs)

    max_zero_run_hours = max(
        (run[2] for run in long_zero_runs),
        default=0,
    )

    zero_run_indices = []

    for (
        start,
        end,
        _,
    ) in long_zero_runs:
        zero_run_indices.extend(
            range(
                start,
                end + 1,
            )
        )

    if zero_run_indices:
        zero_dates = dates.iloc[zero_run_indices]

        zero_run_day_count = int(zero_dates.dropna().dt.normalize().nunique())

    else:
        zero_run_day_count = 0

    valid_values = numeric.dropna()

    mean_consumption = float(valid_values.mean()) if not valid_values.empty else np.nan

    max_consumption = float(valid_values.max()) if not valid_values.empty else np.nan

    if pd.notna(mean_consumption) and mean_consumption > 0:
        extreme_threshold = mean_consumption * 50

        extreme_value_count = int((numeric > extreme_threshold).sum())

    else:
        extreme_value_count = 0

    # ---------------------------------------------------------
    # Methodology reliability rule:
    # >10% missing within the annual profile window
    # means unusable for annual profile analysis.
    # ---------------------------------------------------------
    #if profile_observed_hours == 0 or profile_missing_percent > 10:
    if active_span_hours == 0 or internal_missing_percent > 10:
        quality_status = "unusable"

    elif negative_count > 0 or zero_run_count > 0 or extreme_value_count > 0:
        quality_status = "review"

    else:
        quality_status = "clean"

    return {
        **metadata,
        "total_hours": total_hours,
        "observed_hours": observed_hours,
        "missing_hours": (missing_hours),
        "missing_percent": round(
            missing_percent,
            4,
        ),
        "profile_total_hours": (profile_total_hours),
        "profile_observed_hours": (profile_observed_hours),
        "profile_missing_hours": (profile_missing_hours),
        "profile_missing_percent": (
            round(
                profile_missing_percent,
                4,
            )
        ),
        "first_valid_timestamp": (first_valid_timestamp),
        "last_valid_timestamp": (last_valid_timestamp),
        "active_span_hours": (active_span_hours),
        "leading_missing_hours": (leading_missing_hours),
        "trailing_missing_hours": (trailing_missing_hours),
        "internal_missing_hours": (internal_missing_hours),
        "internal_missing_percent": (
            round(
                internal_missing_percent,
                4,
            )
        ),
        "coverage_percent": round(
            coverage_percent,
            4,
        ),
        "coverage_status": (coverage_status),
        "negative_count": (negative_count),
        "zero_run_count_over_48h": (zero_run_count),
        "zero_run_day_count": (zero_run_day_count),
        "max_zero_run_hours": (max_zero_run_hours),
        "extreme_value_count": (extreme_value_count),
        "mean_kwh": (mean_consumption),
        "max_kwh": (max_consumption),
        "quality_status": (quality_status),
    }


def analyze_workbook(
    input_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meter_reports = []
    timeline_reports = []

    with pd.ExcelFile(
        input_path,
        engine="openpyxl",
    ) as excel_file:
        for sheet_name in excel_file.sheet_names:
            print(f"Analyzing sheet: {sheet_name}")

            base_df, meter_df, metadata_df = load_sheet(
                excel_file,
                sheet_name,
            )

            timeline_reports.append(
                analyze_timeline(
                    base_df,
                    sheet_name,
                )
            )

            dates = pd.to_datetime(
                base_df["Date"],
                errors="coerce",
            )

            hours = pd.to_numeric(
                base_df["Hour"],
                errors="coerce",
            )

            sort_order = (
                pd.DataFrame(
                    {
                        "Date": dates,
                        "Hour": hours,
                    }
                )
                .sort_values(
                    ["Date", "Hour"],
                    na_position="last",
                )
                .index
            )

            sorted_dates = dates.loc[sort_order].reset_index(drop=True)

            sorted_hours = hours.loc[sort_order].reset_index(drop=True)

            for metadata in metadata_df.to_dict(orient="records"):
                source_column = metadata["source_column"]

                sorted_values = meter_df.loc[
                    sort_order,
                    source_column,
                ].reset_index(drop=True)

                meter_reports.append(
                    analyze_meter(
                        sorted_values,
                        sorted_dates,
                        metadata,
                        hours=sorted_hours,
                    )
                )

    meter_quality_df = pd.DataFrame(meter_reports)

    timeline_quality_df = pd.DataFrame(timeline_reports)

    return (
        meter_quality_df,
        timeline_quality_df,
    )


def save_quality_report(
    meter_quality_df: pd.DataFrame,
    timeline_quality_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        meter_quality_df.to_excel(
            writer,
            sheet_name="Meter Quality",
            index=False,
        )

        timeline_quality_df.to_excel(
            writer,
            sheet_name="Timeline Quality",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/raw/"
            "Enerco_June_2025-June_2026_"
            "Hourly_Interval_Meters_ANONYMIZED.xlsx"
        ),
    )

    parser.add_argument(
        "--output",
        default=("reports/" "hapi_1_data_quality.xlsx"),
    )

    args = parser.parse_args()

    (
        meter_quality_df,
        timeline_quality_df,
    ) = analyze_workbook(args.input)

    save_quality_report(
        meter_quality_df,
        timeline_quality_df,
        args.output,
    )

    print()
    print("=" * 60)
    print("DATA QUALITY")
    print("=" * 60)

    print()
    print("Quality status:")
    print(meter_quality_df["quality_status"].value_counts().to_string())

    print()
    print("Coverage status:")
    print(meter_quality_df["coverage_status"].value_counts().to_string())

    print()
    print(f"Meters/series: " f"{len(meter_quality_df)}")

    print(f"Report saved: " f"{args.output}")


if __name__ == "__main__":
    main()
