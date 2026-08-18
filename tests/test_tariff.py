import pandas as pd
import pytest

from src.tariff import (
    create_company_tariff_summary,
    create_hourly_tariff_profile,
    create_monthly_tariff_summary,
    create_tariff_portfolio_summary,
    create_tariff_schedule,
    prepare_tariff_data,
)


def make_tariff_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-07-01",
                    "2025-07-01",
                    "2025-07-01",
                    "2025-07-01",
                    "2025-07-01",
                    "2025-07-01",
                ]
            ),
            "hour": [
                1,
                1,
                2,
                2,
                3,
                3,
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-07-01 00:00",
                    "2025-07-01 00:00",
                    "2025-07-01 01:00",
                    "2025-07-01 01:00",
                    "2025-07-01 02:00",
                    "2025-07-01 02:00",
                ]
            ),
            "tariff": [
                "T2",
                "T2",
                "T2",
                "T2",
                "T1",
                "T1",
            ],
            "company_code": [
                "Kompania 1",
                "Kompania 2",
                "Kompania 1",
                "Kompania 2",
                "Kompania 1",
                "Kompania 2",
            ],
            "flow_type": ["consumption"] * 6,
            "energy_kwh": [
                10.0,
                20.0,
                15.0,
                25.0,
                30.0,
                10.0,
            ],
            "source_sheet": ["Sheet1"] * 6,
            "source_column": [
                "M1",
                "M2",
                "M1",
                "M2",
                "M1",
                "M2",
            ],
        }
    )


def test_prepare_tariff_data_filters_injection_and_unusable():
    data = make_tariff_data()

    injection = data.iloc[[0]].copy()

    injection["flow_type"] = "solar_injection"

    injection["energy_kwh"] = 500.0

    injection["source_column"] = "INJECTION"

    data = pd.concat(
        [
            data,
            injection,
        ],
        ignore_index=True,
    )

    quality = pd.DataFrame(
        {
            "source_sheet": [
                "Sheet1",
                "Sheet1",
                "Sheet1",
            ],
            "source_column": [
                "M1",
                "M2",
                "INJECTION",
            ],
            "quality_status": [
                "clean",
                "unusable",
                "clean",
            ],
        }
    )

    result = prepare_tariff_data(
        data,
        quality_df=quality,
    )

    assert set(result["source_column"]) == {"M1"}

    assert set(result["flow_type"]) == {"consumption"}


def test_prepare_tariff_data_uses_profile_period():
    data = make_tariff_data()

    outside = data.iloc[[0]].copy()

    outside["date"] = pd.Timestamp("2025-06-30")

    outside["timestamp"] = pd.Timestamp("2025-06-30 00:00")

    data = pd.concat(
        [
            data,
            outside,
        ],
        ignore_index=True,
    )

    result = prepare_tariff_data(data)

    assert len(result) == 6

    assert result["date"].min() == pd.Timestamp("2025-07-01")


def test_portfolio_tariff_shares_sum_to_100():
    data = prepare_tariff_data(make_tariff_data())

    result = create_tariff_portfolio_summary(data)

    assert result["total_kwh"].sum() == pytest.approx(110.0)

    assert result["energy_share_percent"].sum() == pytest.approx(100.0)


def test_company_tariff_shares_sum_to_100_per_company():
    data = prepare_tariff_data(make_tariff_data())

    result = create_company_tariff_summary(data)

    shares = result.groupby("company_code")["tariff_share_percent"].sum()

    assert (shares.round(8) == 100.0).all()


def test_monthly_tariff_shares_sum_to_100():
    data = prepare_tariff_data(make_tariff_data())

    result = create_monthly_tariff_summary(data)

    shares = result.groupby("month")["tariff_share_percent"].sum()

    assert (shares.round(8) == 100.0).all()


def test_hourly_profile_aggregates_portfolio_before_mean():
    data = prepare_tariff_data(make_tariff_data())

    result = create_hourly_tariff_profile(data)

    t2_hour_1 = result[(result["tariff"] == "T2") & (result["hour"] == 1)].iloc[0]

    assert t2_hour_1["mean_portfolio_kwh"] == pytest.approx(30.0)


def test_tariff_schedule_does_not_count_meter_duplicates():
    data = prepare_tariff_data(make_tariff_data())

    result = create_tariff_schedule(data)

    july_hour_1 = result[
        (result["month_number"] == 7)
        & (result["hour"] == 1)
        & (result["tariff"] == "T2")
    ].iloc[0]

    assert july_hour_1["occurrence_count"] == 1

    assert july_hour_1["occurrence_percent"] == pytest.approx(100.0)

    assert bool(july_hour_1["is_dominant_tariff"])
