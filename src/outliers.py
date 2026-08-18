import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_START = pd.Timestamp("2025-07-01")
PROFILE_END = pd.Timestamp("2026-07-01")
Z_SCORE_THRESHOLD = 3.0
HAPI_4_1_REPORT_PATH = Path("reports/hapi_4_1_hourly_outliers.xlsx")
HAPI_4_3_REPORT_PATH = Path("reports/hapi_4_3_final_outliers.xlsx")


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


def prepare_company_hourly(
    input_dir: str | Path,
    quality_path: str | Path | None = None,
) -> pd.DataFrame:
    input_dir = Path(input_dir)

    parquet_files = sorted(input_dir.glob("part_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Hapi 2 parquet files found in {input_dir}")

    quality_df = load_quality_report(quality_path)

    hourly_parts = []

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

    for parquet_file in parquet_files:
        print(f"Loading: {parquet_file.name}")

        data = pd.read_parquet(
            parquet_file,
            columns=required_columns,
        )

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce",
        )

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
        )

        data["energy_kwh"] = pd.to_numeric(
            data["energy_kwh"],
            errors="coerce",
        )

        data = data[data["flow_type"] == "consumption"].copy()

        data = data[
            data["date"].notna()
            & (data["date"] >= PROFILE_START)
            & (data["date"] < PROFILE_END)
        ].copy()

        if quality_df is not None:
            data = data.merge(
                quality_df,
                on=[
                    "source_sheet",
                    "source_column",
                ],
                how="left",
                validate="many_to_one",
            )

            data = data[data["quality_status"] != "unusable"].copy()

        hourly_parts.append(
            data[
                [
                    "company_code",
                    "date",
                    "hour",
                    "timestamp",
                    "energy_kwh",
                ]
            ]
        )

    combined = pd.concat(
        hourly_parts,
        ignore_index=True,
    )

    company_hourly = combined.groupby(
        [
            "company_code",
            "date",
            "hour",
            "timestamp",
        ],
        as_index=False,
    )["energy_kwh"].sum(min_count=1)

    return company_hourly


def detect_company_hourly_outliers(
    company_hourly: pd.DataFrame,
    threshold: float = Z_SCORE_THRESHOLD,
) -> pd.DataFrame:
    data = company_hourly.copy()

    data["energy_kwh"] = pd.to_numeric(
        data["energy_kwh"],
        errors="coerce",
    )

    stats = (
        data.groupby("company_code")["energy_kwh"]
        .agg(
            company_mean_kwh="mean",
            company_std_kwh=lambda x: x.std(ddof=0),
        )
        .reset_index()
    )

    data = data.merge(
        stats,
        on="company_code",
        how="left",
        validate="many_to_one",
    )

    valid_std = data["company_std_kwh"].notna() & (data["company_std_kwh"] > 0)

    data["z_score"] = np.nan

    data.loc[
        valid_std,
        "z_score",
    ] = (
        data.loc[
            valid_std,
            "energy_kwh",
        ]
        - data.loc[
            valid_std,
            "company_mean_kwh",
        ]
    ) / data.loc[
        valid_std,
        "company_std_kwh",
    ]

    data["absolute_z_score"] = data["z_score"].abs()

    outliers = data[data["absolute_z_score"] > threshold].copy()

    outliers["outlier_direction"] = np.where(
        outliers["z_score"] > 0,
        "high",
        "low",
    )

    outliers["z_threshold"] = threshold

    outliers = outliers[
        [
            "company_code",
            "date",
            "hour",
            "timestamp",
            "energy_kwh",
            "company_mean_kwh",
            "company_std_kwh",
            "z_score",
            "absolute_z_score",
            "outlier_direction",
            "z_threshold",
        ]
    ]

    outliers = outliers.sort_values(
        [
            "absolute_z_score",
            "company_code",
            "timestamp",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    return outliers


def create_company_outlier_summary(
    company_hourly: pd.DataFrame,
    outliers: pd.DataFrame,
) -> pd.DataFrame:
    total_hours = company_hourly.groupby("company_code").size().rename("total_hours")

    if outliers.empty:
        summary = total_hours.reset_index()
        summary["outlier_hours"] = 0
        summary["outlier_percent"] = 0.0
        summary["max_abs_z_score"] = np.nan

        return summary

    outlier_summary = outliers.groupby("company_code").agg(
        outlier_hours=(
            "timestamp",
            "count",
        ),
        max_abs_z_score=(
            "absolute_z_score",
            "max",
        ),
    )

    summary = (
        total_hours.to_frame()
        .join(
            outlier_summary,
            how="left",
        )
        .reset_index()
    )

    summary["outlier_hours"] = summary["outlier_hours"].fillna(0).astype(int)

    summary["outlier_percent"] = summary["outlier_hours"] / summary["total_hours"] * 100

    return summary.sort_values(
        "outlier_hours",
        ascending=False,
    ).reset_index(drop=True)


def build_final_outlier_report(
    hourly_outliers: pd.DataFrame,
    company_summary: pd.DataFrame,
) -> pd.DataFrame:
    final_report = hourly_outliers.merge(
        company_summary[
            [
                "company_code",
                "total_hours",
                "outlier_hours",
                "max_abs_z_score",
                "outlier_percent",
            ]
        ],
        on="company_code",
        how="left",
        validate="many_to_one",
    )

    if final_report.empty:
        final_report["severity"] = pd.Series(dtype="object")
        final_report["reason"] = pd.Series(dtype="object")
        final_report["recommendation"] = pd.Series(dtype="object")

        return final_report

    final_report["severity"] = np.select(
        [
            final_report["absolute_z_score"] >= 6.0,
            final_report["absolute_z_score"] >= 4.0,
        ],
        [
            "critical",
            "high",
        ],
        default="moderate",
    )

    def create_reason(row: pd.Series) -> str:
        direction = (
            "mbi mesataren" if row["outlier_direction"] == "high" else "nen mesataren"
        )

        return (
            f"Konsum orar {direction} te kompanise; "
            f"|Z|={row['absolute_z_score']:.2f}."
        )

    def create_recommendation(row: pd.Series) -> str:
        if row["absolute_z_score"] >= 6.0 or row["outlier_percent"] >= 2.0:
            return "Per shqyrtim teknik"

        if row["absolute_z_score"] >= 4.0 or row["outlier_percent"] >= 1.0:
            return "Per verifikim operacional"

        return "Monitorim; mund te jete sjellje biznesi legjitime"

    final_report["reason"] = final_report.apply(
        create_reason,
        axis=1,
    )

    final_report["recommendation"] = final_report.apply(
        create_recommendation,
        axis=1,
    )

    return final_report.sort_values(
        [
            "absolute_z_score",
            "company_code",
            "timestamp",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)


def create_hapi_4_3_report(
    hourly_outliers: pd.DataFrame,
    company_summary: pd.DataFrame,
    output_path: str | Path = HAPI_4_3_REPORT_PATH,
) -> pd.DataFrame:
    output_path = Path(output_path)

    final_report = build_final_outlier_report(
        hourly_outliers,
        company_summary,
    )

    if final_report.empty:
        recommendation_summary = pd.DataFrame(
            columns=[
                "recommendation",
                "outlier_hours",
                "companies",
                "max_abs_z_score",
            ]
        )
    else:
        recommendation_summary = (
            final_report.groupby(
                "recommendation",
                observed=True,
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
        )

    methodology_status = pd.DataFrame(
        [
            {
                "step": "4.1",
                "status": "COMPLETED",
                "description": ("Hourly company outliers detected " "using |Z| > 3."),
            },
            {
                "step": "4.2",
                "status": "SKIPPED",
                "description": (
                    "Sector-level comparison cannot be "
                    "performed because business-sector "
                    "metadata is not mapped."
                ),
            },
            {
                "step": "4.3",
                "status": "COMPLETED",
                "description": (
                    "Final report generated from valid "
                    "Hapi 4.1 outliers with reason, "
                    "severity and recommendation."
                ),
            },
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        final_report.to_excel(
            writer,
            sheet_name="Final Outlier Report",
            index=False,
        )

        company_summary.to_excel(
            writer,
            sheet_name="Company Summary",
            index=False,
        )

        recommendation_summary.to_excel(
            writer,
            sheet_name="Recommendation Summary",
            index=False,
        )

        methodology_status.to_excel(
            writer,
            sheet_name="Methodology Status",
            index=False,
        )

    print()
    print("=" * 60)
    print("HAPI 4.3 - FINAL OUTLIER REPORT")
    print("=" * 60)

    print(f"Outlier observations: " f"{len(final_report):,}")

    print(f"Companies represented: " f"{final_report['company_code'].nunique():,}")

    if not recommendation_summary.empty:
        print()
        print("Recommendations:")

        for _, row in recommendation_summary.iterrows():
            print(
                f"  {row['recommendation']}: "
                f"{int(row['outlier_hours']):,} hours / "
                f"{int(row['companies'])} companies"
            )

    print()
    print("Hapi 4.2: SKIPPED - " "business-sector metadata unavailable.")

    print(f"Report saved: " f"{output_path}")

    return final_report


def run_outlier_analysis(
    input_dir: str | Path,
    quality_path: str | Path | None,
    output_dir: str | Path,
    report_path: str | Path,
    final_report_path: str | Path = HAPI_4_3_REPORT_PATH,
) -> dict:
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    final_report_path = Path(final_report_path)

    company_hourly = prepare_company_hourly(
        input_dir=input_dir,
        quality_path=quality_path,
    )

    outliers = detect_company_hourly_outliers(company_hourly)

    summary = create_company_outlier_summary(
        company_hourly,
        outliers,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    company_hourly.to_parquet(
        output_dir / "company_hourly.parquet",
        index=False,
    )

    outliers.to_parquet(
        output_dir / "company_hourly_outliers.parquet",
        index=False,
    )

    summary.to_parquet(
        output_dir / "company_outlier_summary.parquet",
        index=False,
    )

    with pd.ExcelWriter(
        report_path,
        engine="openpyxl",
    ) as writer:
        outliers.to_excel(
            writer,
            sheet_name="Hourly Outliers",
            index=False,
        )

        summary.to_excel(
            writer,
            sheet_name="Company Summary",
            index=False,
        )

    print()
    print("=" * 60)
    print("HAPI 4.1 - COMPANY HOURLY OUTLIERS")
    print("=" * 60)

    print(f"Companies analyzed: " f"{company_hourly['company_code'].nunique()}")

    print(f"Company-hour observations: " f"{len(company_hourly):,}")

    print(f"Outlier hours " f"(|Z| > {Z_SCORE_THRESHOLD}): " f"{len(outliers):,}")

    if not outliers.empty:
        print(f"Companies with outliers: " f"{outliers['company_code'].nunique()}")

        print(f"Maximum |Z|: " f"{outliers['absolute_z_score'].max():.4f}")

    print(f"Report saved: " f"{report_path}")

    final_report = create_hapi_4_3_report(
        hourly_outliers=outliers,
        company_summary=summary,
        output_path=final_report_path,
    )

    return {
        "company_hourly": company_hourly,
        "outliers": outliers,
        "summary": summary,
        "final_report": final_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="data/processed/hourly_long",
    )

    parser.add_argument(
        "--quality-report",
        default="reports/hapi_1_data_quality.xlsx",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/outliers",
    )

    parser.add_argument(
        "--report",
        default=HAPI_4_1_REPORT_PATH,
    )

    parser.add_argument(
        "--final-report",
        default=HAPI_4_3_REPORT_PATH,
    )

    args = parser.parse_args()

    run_outlier_analysis(
        input_dir=args.input_dir,
        quality_path=args.quality_report,
        output_dir=args.output_dir,
        report_path=args.report,
        final_report_path=args.final_report,
    )


if __name__ == "__main__":
    main()
