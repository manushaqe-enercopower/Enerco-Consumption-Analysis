import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.config import BUSINESS_SECTORS
from src.loader import load_sheet

OUTPUT_COLUMNS = [
    "period_year",
    "period_month",
    "week_day",
    "date",
    "hour",
    "timestamp",
    "tariff",
    "company_code",
    "business_sector",
    "company_size",
    "voltage_level",
    "district",
    "ts_code",
    "metadata_status",
    "meter_id",
    "flow_type",
    "energy_kwh",
    "source_sheet",
    "source_column",
]


def load_company_metadata(
    metadata_path: str | Path | None,
) -> pd.DataFrame | None:
    if metadata_path is None:
        return None

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        return None

    metadata_df = pd.read_csv(metadata_path)

    required_columns = {
        "company_code",
        "business_sector",
        "company_size",
        "voltage_level",
        "district",
        "ts_code",
    }

    missing_columns = required_columns - set(metadata_df.columns)

    if missing_columns:
        raise ValueError(
            "Company metadata is missing columns: " f"{sorted(missing_columns)}"
        )

    metadata_df["company_code"] = metadata_df["company_code"].astype(str).str.strip()

    invalid_sectors = metadata_df.loc[
        metadata_df["business_sector"].notna()
        & ~metadata_df["business_sector"].isin(BUSINESS_SECTORS),
        "business_sector",
    ].unique()

    if len(invalid_sectors) > 0:
        raise ValueError("Unknown business sectors: " f"{invalid_sectors.tolist()}")

    if metadata_df["company_code"].duplicated().any():
        duplicates = (
            metadata_df.loc[
                metadata_df["company_code"].duplicated(keep=False),
                "company_code",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError("Duplicate company codes in metadata: " f"{duplicates}")

    return metadata_df


def enrich_company_metadata(
    long_df: pd.DataFrame,
    company_metadata_df: pd.DataFrame | None,
) -> pd.DataFrame:
    metadata_columns = [
        "business_sector",
        "company_size",
        "voltage_level",
        "district",
        "ts_code",
    ]

    if company_metadata_df is None:
        for column in metadata_columns:
            long_df[column] = pd.NA

        long_df["metadata_status"] = "pending_mapping"

        return long_df

    long_df = long_df.merge(
        company_metadata_df,
        on="company_code",
        how="left",
        validate="many_to_one",
    )

    has_metadata = long_df[metadata_columns].notna().any(axis=1)

    long_df["metadata_status"] = has_metadata.map(
        {
            True: "enriched",
            False: "pending_mapping",
        }
    )

    return long_df


def reshape_sheet(
    base_df: pd.DataFrame,
    meter_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    sheet_name: str,
    company_metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base_df = base_df.copy()
    meter_df = meter_df.copy()
    metadata_df = metadata_df.copy()

    base_df["Date"] = pd.to_datetime(
        base_df["Date"],
        errors="coerce",
    )

    base_df["Hour"] = pd.to_numeric(
        base_df["Hour"],
        errors="coerce",
    )

    base_df["PeriodYear"] = pd.to_numeric(
        base_df["PeriodYear"],
        errors="coerce",
    )

    base_df["PeriodMonth"] = pd.to_numeric(
        base_df["PeriodMonth"],
        errors="coerce",
    )

    valid_timeline = base_df["Date"].notna() & base_df["Hour"].between(1, 24)

    if not valid_timeline.all():
        invalid_count = int((~valid_timeline).sum())

        raise ValueError(
            f"{sheet_name}: " f"{invalid_count} rows have invalid " "Date/Hour values."
        )

    base_df["timestamp"] = base_df["Date"] + pd.to_timedelta(
        base_df["Hour"] - 1,
        unit="h",
    )

    combined_df = pd.concat(
        [
            base_df.reset_index(drop=True),
            meter_df.reset_index(drop=True),
        ],
        axis=1,
    )

    long_df = combined_df.melt(
        id_vars=[
            "PeriodYear",
            "PeriodMonth",
            "WeekDay",
            "Date",
            "Hour",
            "timestamp",
            "Tariff",
        ],
        value_vars=list(meter_df.columns),
        var_name="source_column",
        value_name="energy_kwh",
    )

    long_df["source_column"] = long_df["source_column"].astype(str).str.strip()

    metadata_df["source_column"] = metadata_df["source_column"].astype(str).str.strip()

    long_df = long_df.merge(
        metadata_df[
            [
                "source_column",
                "company_code",
                "meter_id",
                "flow_type",
            ]
        ],
        on="source_column",
        how="left",
        validate="many_to_one",
    )

    if (
        long_df[
            [
                "company_code",
                "meter_id",
                "flow_type",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        missing_metadata = (
            long_df.loc[
                long_df["meter_id"].isna(),
                "source_column",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"{sheet_name}: metadata missing for " f"columns: {missing_metadata}"
        )

    long_df["energy_kwh"] = pd.to_numeric(
        long_df["energy_kwh"],
        errors="coerce",
    )

    long_df["source_sheet"] = sheet_name

    long_df = long_df.rename(
        columns={
            "PeriodYear": "period_year",
            "PeriodMonth": "period_month",
            "WeekDay": "week_day",
            "Date": "date",
            "Hour": "hour",
            "Tariff": "tariff",
        }
    )

    long_df = enrich_company_metadata(
        long_df,
        company_metadata_df,
    )

    long_df = long_df[OUTPUT_COLUMNS]

    long_df = long_df.sort_values(
        [
            "date",
            "hour",
            "company_code",
            "meter_id",
            "flow_type",
        ]
    ).reset_index(drop=True)

    return long_df


def reshape_workbook(
    input_path: str | Path,
    output_dir: str | Path,
    company_metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    company_metadata_df = load_company_metadata(company_metadata_path)

    if company_metadata_df is None:
        print(
            "Company metadata mapping not provided. "
            "Metadata fields will be marked pending_mapping."
        )
    else:
        print(f"Loaded metadata for " f"{len(company_metadata_df)} companies.")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []
    all_metadata = []

    with pd.ExcelFile(
        input_path,
        engine="openpyxl",
    ) as excel_file:
        for index, sheet_name in enumerate(
            excel_file.sheet_names,
            start=1,
        ):
            print(f"Reshaping sheet: {sheet_name}")

            (
                base_df,
                meter_df,
                metadata_df,
            ) = load_sheet(
                excel_file,
                sheet_name,
            )

            long_df = reshape_sheet(
                base_df,
                meter_df,
                metadata_df,
                sheet_name,
                company_metadata_df=company_metadata_df,
            )

            output_file = output_dir / f"part_{index:02d}.parquet"

            long_df.to_parquet(
                output_file,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )

            metadata_copy = metadata_df.copy()
            all_metadata.append(metadata_copy)

            expected_rows = len(base_df) * len(meter_df.columns)

            actual_rows = len(long_df)

            if actual_rows != expected_rows:
                raise ValueError(
                    f"{sheet_name}: expected "
                    f"{expected_rows:,} long rows, "
                    f"got {actual_rows:,}."
                )

            manifest_rows.append(
                {
                    "source_sheet": sheet_name,
                    "hourly_rows": len(base_df),
                    "measurement_series": len(meter_df.columns),
                    "expected_long_rows": expected_rows,
                    "actual_long_rows": actual_rows,
                    "missing_energy_values": int(long_df["energy_kwh"].isna().sum()),
                    "pending_metadata_rows": int(
                        (long_df["metadata_status"] == "pending_mapping").sum()
                    ),
                    "output_file": str(output_file),
                }
            )

            print(f"  Series: " f"{len(meter_df.columns)}")

            print(f"  Long rows: " f"{actual_rows:,}")

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_df.to_csv(
        output_dir / "manifest.csv",
        index=False,
    )

    meter_metadata_df = (
        pd.concat(
            all_metadata,
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    meter_metadata_df.to_parquet(
        output_dir / "meter_metadata.parquet",
        engine="pyarrow",
        index=False,
    )

    total_rows = int(manifest_df["actual_long_rows"].sum())

    total_series = int(manifest_df["measurement_series"].sum())

    print()
    print("=" * 60)
    print("HAPI 2 - WIDE TO LONG")
    print("=" * 60)

    print(f"Measurement series: " f"{total_series}")

    print(f"Long-format rows: " f"{total_rows:,}")

    if company_metadata_df is None:
        print("Company metadata: " "pending mapping")
    else:
        print("Company metadata: enriched")

    print(f"Dataset saved to: " f"{output_dir}")

    return manifest_df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/raw/"
            "Enerco_June_2025-June_2026_"
            "Hourly_Interval_Meters_"
            "ANONYMIZED.xlsx"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=("data/processed/hourly_long"),
    )

    parser.add_argument(
        "--company-metadata",
        default=None,
    )

    args = parser.parse_args()

    reshape_workbook(
        args.input,
        args.output_dir,
        company_metadata_path=(args.company_metadata),
    )


if __name__ == "__main__":
    main()
