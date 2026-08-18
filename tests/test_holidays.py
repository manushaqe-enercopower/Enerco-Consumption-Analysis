import pandas as pd
import pytest

from src.holidays import (
    attach_holiday_flags,
    calculate_holiday_impacts,
    create_company_daily,
    create_holiday_outlier_detail,
    create_portfolio_hourly_impact,
    get_kosovo_holiday_calendar,
    prepare_consumption_data,
)


def test_calendar_uses_observed_days_off():
    calendar = get_kosovo_holiday_calendar()

    easter = calendar[calendar["holiday_name"] == "Pashket Katolike"].iloc[0]

    europe = calendar[calendar["holiday_name"] == "Dita e Evropes"].iloc[0]

    assert easter["holiday_date"] == pd.Timestamp("2026-04-05")

    assert easter["observed_date"] == pd.Timestamp("2026-04-06")

    assert bool(easter["is_shifted"])

    assert europe["holiday_date"] == pd.Timestamp("2026-05-09")

    assert europe["observed_date"] == pd.Timestamp("2026-05-11")


def test_prepare_consumption_filters_injection_and_unusable():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                ]
            ),
            "hour": [1, 1, 1],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                ]
            ),
            "company_code": [
                "Kompania 1",
                "Kompania 1",
                "Kompania 2",
            ],
            "flow_type": [
                "consumption",
                "solar_injection",
                "consumption",
            ],
            "energy_kwh": [
                10.0,
                100.0,
                20.0,
            ],
            "source_sheet": [
                "Sheet1",
                "Sheet1",
                "Sheet1",
            ],
            "source_column": [
                "M1",
                "M2",
                "M3",
            ],
        }
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
                "M3",
            ],
            "quality_status": [
                "clean",
                "clean",
                "unusable",
            ],
        }
    )

    result = prepare_consumption_data(
        data,
        quality_df=quality,
    )

    assert len(result) == 1
    assert result.iloc[0]["source_column"] == "M1"


def test_attach_holiday_flags_uses_observed_date():
    calendar = get_kosovo_holiday_calendar()

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-09",
                    "2026-05-11",
                ]
            ),
            "value": [
                10.0,
                20.0,
            ],
        }
    )

    result = attach_holiday_flags(
        data,
        calendar,
    )

    may_9 = result[result["date"] == pd.Timestamp("2026-05-09")].iloc[0]

    may_11 = result[result["date"] == pd.Timestamp("2026-05-11")].iloc[0]

    assert not bool(may_9["is_holiday"])

    assert bool(may_11["is_holiday"])

    assert may_11["holiday_name"] == "Dita e Evropes"


def test_company_daily_aggregates_multiple_meters():
    calendar = get_kosovo_holiday_calendar()

    data = pd.DataFrame(
        {
            "company_code": [
                "Kompania 1",
                "Kompania 1",
            ],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                ]
            ),
            "energy_kwh": [
                10.0,
                20.0,
            ],
        }
    )

    result = create_company_daily(
        data,
        calendar,
    )

    assert len(result) == 1
    assert result.iloc[0]["daily_kwh"] == pytest.approx(30.0)

    assert bool(result.iloc[0]["is_holiday"])


def test_holiday_impact_uses_same_month_and_weekday():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-08",
                    "2026-01-15",
                    "2026-02-05",
                ]
            ),
            "portfolio_kwh": [
                60.0,
                100.0,
                120.0,
                1000.0,
            ],
            "is_holiday": [
                True,
                False,
                False,
                False,
            ],
            "holiday_name": [
                "Viti i Ri",
                None,
                None,
                None,
            ],
            "month_number": [
                1,
                1,
                1,
                2,
            ],
            "weekday_number": [
                3,
                3,
                3,
                3,
            ],
        }
    )

    result = calculate_holiday_impacts(
        data,
        value_column="portfolio_kwh",
    )

    row = result.iloc[0]

    assert row["baseline_mean"] == pytest.approx(110.0)

    assert row["impact_kwh"] == pytest.approx(-50.0)

    assert row["impact_percent"] == pytest.approx(
        -45.454545,
        rel=1e-5,
    )


def test_company_holiday_baseline_is_company_specific():
    data = pd.DataFrame(
        {
            "company_code": [
                "Kompania 1",
                "Kompania 1",
                "Kompania 1",
                "Kompania 2",
                "Kompania 2",
                "Kompania 2",
            ],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-08",
                    "2026-01-15",
                    "2026-01-01",
                    "2026-01-08",
                    "2026-01-15",
                ]
            ),
            "daily_kwh": [
                50.0,
                100.0,
                100.0,
                300.0,
                200.0,
                200.0,
            ],
            "is_holiday": [
                True,
                False,
                False,
                True,
                False,
                False,
            ],
            "holiday_name": [
                "Viti i Ri",
                None,
                None,
                "Viti i Ri",
                None,
                None,
            ],
            "month_number": [
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "weekday_number": [
                3,
                3,
                3,
                3,
                3,
                3,
            ],
        }
    )

    result = calculate_holiday_impacts(
        data,
        value_column="daily_kwh",
        group_columns=["company_code"],
    )

    company_1 = result[result["company_code"] == "Kompania 1"].iloc[0]

    company_2 = result[result["company_code"] == "Kompania 2"].iloc[0]

    assert company_1["baseline_mean"] == pytest.approx(100.0)

    assert company_1["impact_percent"] == pytest.approx(-50.0)

    assert company_2["baseline_mean"] == pytest.approx(200.0)

    assert company_2["impact_percent"] == pytest.approx(50.0)


def test_hourly_impact_matches_same_hour_baseline():
    calendar = pd.DataFrame(
        {
            "holiday_name": [
                "Test Holiday",
            ],
            "holiday_date": pd.to_datetime(
                [
                    "2026-01-01",
                ]
            ),
            "observed_date": pd.to_datetime(
                [
                    "2026-01-01",
                ]
            ),
            "is_shifted": [
                False,
            ],
        }
    )

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-08",
                    "2026-01-15",
                ]
            ),
            "hour": [
                1,
                1,
                1,
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-08 00:00",
                    "2026-01-15 00:00",
                ]
            ),
            "energy_kwh": [
                50.0,
                100.0,
                120.0,
            ],
        }
    )

    result = create_portfolio_hourly_impact(
        data,
        calendar,
    )

    assert len(result) == 1

    assert result.iloc[0]["baseline_mean_kwh"] == pytest.approx(110.0)

    assert result.iloc[0]["impact_kwh"] == pytest.approx(-60.0)


def test_holiday_outliers_only_include_observed_holidays():
    calendar = get_kosovo_holiday_calendar()

    outliers = pd.DataFrame(
        {
            "company_code": [
                "Kompania 1",
                "Kompania 2",
            ],
            "date": pd.to_datetime(
                [
                    "2026-05-11",
                    "2026-05-12",
                ]
            ),
            "absolute_z_score": [
                4.5,
                5.0,
            ],
        }
    )

    result = create_holiday_outlier_detail(
        outliers,
        calendar,
    )

    assert len(result) == 1

    assert result.iloc[0]["company_code"] == "Kompania 1"

    assert result.iloc[0]["holiday_name"] == "Dita e Evropes"
