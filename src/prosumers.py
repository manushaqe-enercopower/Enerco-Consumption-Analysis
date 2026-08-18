import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROFILE_START = pd.Timestamp("2025-07-01")
PROFILE_END = pd.Timestamp("2026-07-01")

DEFAULT_INPUT_DIR = Path("data/processed/hourly_long")
DEFAULT_QUALITY_REPORT = Path("reports/hapi_1_data_quality.xlsx")
DEFAULT_OUTPUT_DIR = Path("data/processed/prosumers")
DEFAULT_REPORT_PATH = Path("reports/hapi_prosumers.xlsx")
DEFAULT_FIGURE_DIR = Path("reports/figures/prosumers")

PROSUMER_FLOWS = {
    "consumption",
    "solar_injection",
}


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

    missing = required_columns - set(quality.columns)

    if missing:
        raise ValueError("Quality report is missing columns: " f"{sorted(missing)}")

    return quality[
        [
            "source_sheet",
            "source_column",
            "quality_status",
        ]
    ].copy()


def prepare_prosumer_data(
    data: pd.DataFrame,
    quality_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_columns = {
        "date",
        "hour",
        "timestamp",
        "company_code",
        "meter_id",
        "flow_type",
        "energy_kwh",
        "source_sheet",
        "source_column",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError("Prosumer input is missing columns: " f"{sorted(missing)}")

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
        & result["flow_type"].isin(PROSUMER_FLOWS)
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

    meter_keys = [
        "source_sheet",
        "company_code",
        "meter_id",
    ]

    flow_counts = (
        result.groupby(meter_keys)["flow_type"]
        .nunique()
        .rename("flow_count")
        .reset_index()
    )

    paired_meters = flow_counts[flow_counts["flow_count"] == 2][meter_keys]

    result = result.merge(
        paired_meters,
        on=meter_keys,
        how="inner",
        validate="many_to_one",
    )

    return result.reset_index(drop=True)


def create_prosumer_hourly(
    prosumer_data: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "source_sheet",
        "company_code",
        "meter_id",
        "date",
        "hour",
        "timestamp",
    ]

    aggregated = prosumer_data.groupby(
        group_columns + ["flow_type"],
        as_index=False,
    )[
        "energy_kwh"
    ].sum(min_count=1)

    hourly = aggregated.pivot(
        index=group_columns,
        columns="flow_type",
        values="energy_kwh",
    ).reset_index()

    hourly.columns.name = None

    if "consumption" not in hourly.columns:
        hourly["consumption"] = np.nan

    if "solar_injection" not in hourly.columns:
        hourly["solar_injection"] = np.nan

    hourly = hourly.rename(
        columns={
            "consumption": "consumption_kwh",
            "solar_injection": "injection_kwh",
        }
    )

    hourly["net_grid_kwh"] = hourly["consumption_kwh"] - hourly["injection_kwh"]

    hourly["has_injection"] = hourly["injection_kwh"].fillna(0) > 0

    hourly["is_net_export"] = hourly["net_grid_kwh"] < 0

    hourly["complete_pair"] = (
        hourly["consumption_kwh"].notna() & hourly["injection_kwh"].notna()
    )

    return hourly.sort_values(
        [
            "company_code",
            "meter_id",
            "timestamp",
        ]
    ).reset_index(drop=True)


def create_prosumer_summary(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "source_sheet",
        "company_code",
        "meter_id",
    ]

    for keys, group in hourly.groupby(
        group_columns,
        sort=False,
    ):
        (
            source_sheet,
            company_code,
            meter_id,
        ) = keys

        total_consumption = group["consumption_kwh"].sum(min_count=1)

        total_injection = group["injection_kwh"].sum(min_count=1)

        net_grid = total_consumption - total_injection

        injection_import_ratio = (
            total_injection / total_consumption * 100
            if (pd.notna(total_consumption) and total_consumption > 0)
            else np.nan
        )

        total_exchange = total_consumption + total_injection

        injection_exchange_share = (
            total_injection / total_exchange * 100
            if (pd.notna(total_exchange) and total_exchange > 0)
            else np.nan
        )

        injection_values = group["injection_kwh"].dropna()

        positive_injection = injection_values[injection_values > 0]

        rows.append(
            {
                "source_sheet": source_sheet,
                "company_code": company_code,
                "meter_id": meter_id,
                "observed_hours": len(group),
                "complete_pair_hours": int(group["complete_pair"].sum()),
                "total_consumption_kwh": (total_consumption),
                "total_injection_kwh": (total_injection),
                "net_grid_kwh": (net_grid),
                "injection_import_ratio_percent": (injection_import_ratio),
                "injection_exchange_share_percent": (injection_exchange_share),
                "hours_with_injection": int(group["has_injection"].sum()),
                "net_export_hours": int(group["is_net_export"].sum()),
                "mean_consumption_kwh": (group["consumption_kwh"].mean()),
                "mean_injection_kwh": (group["injection_kwh"].mean()),
                "mean_active_injection_kwh": (
                    positive_injection.mean() if not positive_injection.empty else 0.0
                ),
                "max_consumption_kwh": (group["consumption_kwh"].max()),
                "max_injection_kwh": (group["injection_kwh"].max()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "total_injection_kwh",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def create_prosumer_monthly(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    data = hourly.copy()

    data["month"] = data["date"].dt.to_period("M").astype(str)

    monthly = (
        data.groupby(
            [
                "company_code",
                "meter_id",
                "month",
            ]
        )
        .agg(
            consumption_kwh=(
                "consumption_kwh",
                "sum",
            ),
            injection_kwh=(
                "injection_kwh",
                "sum",
            ),
            hours_with_injection=(
                "has_injection",
                "sum",
            ),
            net_export_hours=(
                "is_net_export",
                "sum",
            ),
        )
        .reset_index()
    )

    monthly["net_grid_kwh"] = monthly["consumption_kwh"] - monthly["injection_kwh"]

    monthly["injection_import_ratio_percent"] = np.where(
        monthly["consumption_kwh"] > 0,
        monthly["injection_kwh"] / monthly["consumption_kwh"] * 100,
        np.nan,
    )

    return monthly.sort_values(
        [
            "company_code",
            "meter_id",
            "month",
        ]
    ).reset_index(drop=True)


def create_prosumer_hourly_profile(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    return (
        hourly.groupby(
            [
                "company_code",
                "meter_id",
                "hour",
            ]
        )
        .agg(
            observations=(
                "timestamp",
                "count",
            ),
            mean_consumption_kwh=(
                "consumption_kwh",
                "mean",
            ),
            median_consumption_kwh=(
                "consumption_kwh",
                "median",
            ),
            p10_consumption_kwh=(
                "consumption_kwh",
                lambda x: x.quantile(0.10),
            ),
            p90_consumption_kwh=(
                "consumption_kwh",
                lambda x: x.quantile(0.90),
            ),
            mean_injection_kwh=(
                "injection_kwh",
                "mean",
            ),
            median_injection_kwh=(
                "injection_kwh",
                "median",
            ),
            p10_injection_kwh=(
                "injection_kwh",
                lambda x: x.quantile(0.10),
            ),
            p90_injection_kwh=(
                "injection_kwh",
                lambda x: x.quantile(0.90),
            ),
            mean_net_grid_kwh=(
                "net_grid_kwh",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "company_code",
                "meter_id",
                "hour",
            ]
        )
        .reset_index(drop=True)
    )


def create_portfolio_monthly(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    data = hourly.copy()

    data["month"] = data["date"].dt.to_period("M").astype(str)

    portfolio = (
        data.groupby("month")
        .agg(
            consumption_kwh=(
                "consumption_kwh",
                "sum",
            ),
            injection_kwh=(
                "injection_kwh",
                "sum",
            ),
        )
        .reset_index()
    )

    portfolio["net_grid_kwh"] = (
        portfolio["consumption_kwh"] - portfolio["injection_kwh"]
    )

    portfolio["injection_import_ratio_percent"] = np.where(
        portfolio["consumption_kwh"] > 0,
        portfolio["injection_kwh"] / portfolio["consumption_kwh"] * 100,
        np.nan,
    )

    return portfolio.sort_values("month").reset_index(drop=True)


def create_portfolio_hourly_profile(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    portfolio_hourly = hourly.groupby(
        [
            "date",
            "hour",
            "timestamp",
        ],
        as_index=False,
    ).agg(
        consumption_kwh=(
            "consumption_kwh",
            "sum",
        ),
        injection_kwh=(
            "injection_kwh",
            "sum",
        ),
    )

    portfolio_hourly["net_grid_kwh"] = (
        portfolio_hourly["consumption_kwh"] - portfolio_hourly["injection_kwh"]
    )

    return (
        portfolio_hourly.groupby("hour")
        .agg(
            observations=(
                "timestamp",
                "count",
            ),
            mean_consumption_kwh=(
                "consumption_kwh",
                "mean",
            ),
            p10_consumption_kwh=(
                "consumption_kwh",
                lambda x: x.quantile(0.10),
            ),
            p90_consumption_kwh=(
                "consumption_kwh",
                lambda x: x.quantile(0.90),
            ),
            mean_injection_kwh=(
                "injection_kwh",
                "mean",
            ),
            p10_injection_kwh=(
                "injection_kwh",
                lambda x: x.quantile(0.10),
            ),
            p90_injection_kwh=(
                "injection_kwh",
                lambda x: x.quantile(0.90),
            ),
            mean_net_grid_kwh=(
                "net_grid_kwh",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("hour")
        .reset_index(drop=True)
    )


def plot_prosumer_findings(
    summary: pd.DataFrame,
    portfolio_monthly: pd.DataFrame,
    portfolio_hourly: pd.DataFrame,
    figure_dir: str | Path,
) -> None:
    figure_dir = Path(figure_dir)

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Monthly import / injection / net exchange.
    monthly = portfolio_monthly.sort_values("month")

    positions = np.arange(len(monthly))

    width = 0.34

    plt.figure(figsize=(12, 6))

    plt.bar(
        positions - width / 2,
        monthly["consumption_kwh"] / 1_000,
        width=width,
        label="A+ Konsum",
    )

    plt.bar(
        positions + width / 2,
        monthly["injection_kwh"] / 1_000,
        width=width,
        label="A− Injektim",
    )

    plt.plot(
        positions,
        monthly["net_grid_kwh"] / 1_000,
        marker="o",
        linewidth=2,
        label="Bilanci neto me rrjetin",
    )

    plt.xticks(
        positions,
        monthly["month"],
        rotation=45,
        ha="right",
    )

    plt.ylabel("Energji (MWh)")

    plt.title("Konsumi, injektimi dhe bilanci neto mujor i Prosumer-ve")

    plt.grid(
        axis="y",
        alpha=0.2,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figure_dir / "prosumer_monthly_exchange.png",
        dpi=180,
    )

    plt.close()

    # 2. Portfolio hourly profile.
    profile = portfolio_hourly.sort_values("hour")

    plt.figure(figsize=(11, 6))

    plt.fill_between(
        profile["hour"],
        profile["p10_consumption_kwh"],
        profile["p90_consumption_kwh"],
        alpha=0.15,
        label="A+ P10–P90",
    )

    plt.plot(
        profile["hour"],
        profile["mean_consumption_kwh"],
        marker="o",
        linewidth=2,
        label="A+ mesatar",
    )

    plt.fill_between(
        profile["hour"],
        profile["p10_injection_kwh"],
        profile["p90_injection_kwh"],
        alpha=0.15,
        label="A− P10–P90",
    )

    plt.plot(
        profile["hour"],
        profile["mean_injection_kwh"],
        marker="o",
        linewidth=2,
        label="A− mesatar",
    )

    plt.xlabel("Ora")

    plt.ylabel("Energji mesatare (kWh)")

    plt.title("Profili orar i konsumit dhe injektimit të Prosumer-ve")

    plt.xticks(range(1, 25))

    plt.grid(
        alpha=0.2,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figure_dir / "prosumer_hourly_profile.png",
        dpi=180,
    )

    plt.close()

    # 3. Consumption vs injection positioning.
    plt.figure(figsize=(9, 7))

    x = summary["total_consumption_kwh"] / 1_000

    y = summary["total_injection_kwh"] / 1_000

    plt.scatter(
        x,
        y,
        s=60,
        alpha=0.75,
    )

    if not x.empty and not y.empty:
        limit = max(
            x.max(),
            y.max(),
        )

        plt.plot(
            [
                0,
                limit,
            ],
            [
                0,
                limit,
            ],
            linestyle="--",
            linewidth=1.5,
            label="A− = A+",
        )

    for _, row in summary.iterrows():
        plt.annotate(
            str(row["company_code"]),
            (
                row["total_consumption_kwh"] / 1_000,
                row["total_injection_kwh"] / 1_000,
            ),
            fontsize=7,
            alpha=0.75,
        )

    plt.xlabel("Konsum A+ (MWh)")

    plt.ylabel("Injektim A− (MWh)")

    plt.title("Pozicionimi i Prosumer-ve: konsum kundrejt injektimit")

    plt.grid(
        alpha=0.2,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figure_dir / "prosumer_import_export_positioning.png",
        dpi=180,
    )

    plt.close()

    # 4. Net grid balance.
    balance = summary.sort_values("net_grid_kwh")

    labels = (
        balance["company_code"].astype(str) + " · " + balance["meter_id"].astype(str)
    )

    plt.figure(figsize=(11, 8))

    plt.barh(
        labels,
        balance["net_grid_kwh"] / 1_000,
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel("Bilanci neto me rrjetin (MWh)")

    plt.ylabel("Prosumer")

    plt.title("Bilanci vjetor neto i energjisë për çdo Prosumer")

    plt.grid(
        axis="x",
        alpha=0.2,
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir / "prosumer_net_grid_balance.png",
        dpi=180,
    )

    plt.close()


def run_prosumer_analysis(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    quality_path: str | Path | None = DEFAULT_QUALITY_REPORT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> dict[str, pd.DataFrame]:
    input_dir = Path(input_dir)

    output_dir = Path(output_dir)

    report_path = Path(report_path)

    parquet_files = sorted(input_dir.glob("part_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Hapi 2 parquet files found in " f"{input_dir}")

    quality_df = load_quality_report(quality_path)

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

    parts = []

    for parquet_file in parquet_files:
        print(f"Loading: " f"{parquet_file.name}")

        data = pd.read_parquet(
            parquet_file,
            columns=required_columns,
        )

        prepared = prepare_prosumer_data(
            data,
            quality_df=quality_df,
        )

        if not prepared.empty:
            parts.append(prepared)

    if not parts:
        raise ValueError("No valid A+/A- Prosumer pairs found.")

    prosumer_data = pd.concat(
        parts,
        ignore_index=True,
    )

    hourly = create_prosumer_hourly(prosumer_data)

    summary = create_prosumer_summary(hourly)

    monthly = create_prosumer_monthly(hourly)

    hourly_profile = create_prosumer_hourly_profile(hourly)

    portfolio_monthly = create_portfolio_monthly(hourly)

    portfolio_hourly = create_portfolio_hourly_profile(hourly)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "prosumer_hourly": hourly,
        "prosumer_summary": summary,
        "prosumer_monthly": monthly,
        "prosumer_hourly_profile": (hourly_profile),
        "portfolio_monthly": (portfolio_monthly),
        "portfolio_hourly_profile": (portfolio_hourly),
    }

    filenames = {
        "prosumer_hourly": ("prosumer_hourly.parquet"),
        "prosumer_summary": ("prosumer_summary.parquet"),
        "prosumer_monthly": ("prosumer_monthly.parquet"),
        "prosumer_hourly_profile": ("prosumer_hourly_profile.parquet"),
        "portfolio_monthly": ("prosumer_portfolio_monthly.parquet"),
        "portfolio_hourly_profile": ("prosumer_portfolio_hourly_profile.parquet"),
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
        summary.to_excel(
            writer,
            sheet_name="Prosumer Summary",
            index=False,
        )

        monthly.to_excel(
            writer,
            sheet_name="Monthly",
            index=False,
        )

        hourly_profile.to_excel(
            writer,
            sheet_name="Hourly Profile",
            index=False,
        )

        portfolio_monthly.to_excel(
            writer,
            sheet_name="Portfolio Monthly",
            index=False,
        )

        portfolio_hourly.to_excel(
            writer,
            sheet_name="Portfolio Hourly",
            index=False,
        )

    plot_prosumer_findings(
        summary=summary,
        portfolio_monthly=portfolio_monthly,
        portfolio_hourly=portfolio_hourly,
        figure_dir=figure_dir,
    )

    total_consumption = summary["total_consumption_kwh"].sum()

    total_injection = summary["total_injection_kwh"].sum()

    total_net = total_consumption - total_injection

    injection_ratio = (
        total_injection / total_consumption * 100 if total_consumption > 0 else np.nan
    )

    print()
    print("=" * 60)
    print("PROSUMER A+ / A- ANALYSIS")
    print("=" * 60)

    print(f"Prosumer meters analyzed: " f"{len(summary)}")

    print(f"Companies represented: " f"{summary['company_code'].nunique()}")

    print(f"Total A+ consumption: " f"{total_consumption:,.2f} kWh")

    print(f"Total A- injection: " f"{total_injection:,.2f} kWh")

    print(f"Net grid exchange: " f"{total_net:,.2f} kWh")

    print(f"Injection / import ratio: " f"{injection_ratio:.2f}%")

    print(f"Net-export hours: " f"{summary['net_export_hours'].sum():,}")

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

    args = parser.parse_args()

    run_prosumer_analysis(
        input_dir=args.input_dir,
        quality_path=args.quality_report,
        output_dir=args.output_dir,
        report_path=args.report,
        figure_dir=args.figure_dir,
    )


if __name__ == "__main__":
    main()
