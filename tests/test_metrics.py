import numpy as np
import pandas as pd

from src.metrics import (
    _calculate_company_hourly_profiles,
    _calculate_meter_hourly_profiles,
    calculate_profile_metrics,
)


def _hourly_day(
    date: str,
    peak_value: float,
    off_peak_value: float,
) -> pd.DataFrame:
    rows = []

    for hour in range(1, 25):
        value = peak_value if 7 <= hour <= 18 else off_peak_value

        rows.append(
            {
                "date": date,
                "hour": hour,
                "timestamp": (pd.Timestamp(date) + pd.Timedelta(hours=hour - 1)),
                "energy_kwh": value,
            }
        )

    return pd.DataFrame(rows)


def test_peak_ratio():
    data = _hourly_day(
        "2025-07-01",
        peak_value=20.0,
        off_peak_value=10.0,
    )

    result = calculate_profile_metrics(data)

    assert result["peak_ratio"] == 2.0


def test_constant_load_has_zero_cv_and_load_factor_one():
    data = _hourly_day(
        "2025-07-01",
        peak_value=10.0,
        off_peak_value=10.0,
    )

    result = calculate_profile_metrics(data)

    assert result["cv"] == 0.0
    assert result["load_factor"] == 1.0


def test_weekday_weekend_ratio():
    weekday = _hourly_day(
        "2025-07-04",
        peak_value=20.0,
        off_peak_value=20.0,
    )

    weekend = _hourly_day(
        "2025-07-05",
        peak_value=10.0,
        off_peak_value=10.0,
    )

    data = pd.concat(
        [
            weekday,
            weekend,
        ],
        ignore_index=True,
    )

    result = calculate_profile_metrics(data)

    assert result["weekday_weekend_ratio"] == 2.0


def test_trend_uses_first_three_active_months():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-07-01",
                    "2025-08-01",
                    "2025-09-01",
                    "2026-06-01",
                ]
            ),
            "hour": [12, 12, 12, 12],
            "timestamp": pd.to_datetime(
                [
                    "2025-07-01 11:00",
                    "2025-08-01 11:00",
                    "2025-09-01 11:00",
                    "2026-06-01 11:00",
                ]
            ),
            "energy_kwh": [
                100.0,
                100.0,
                100.0,
                120.0,
            ],
        }
    )

    result = calculate_profile_metrics(data)

    assert result["trend_percent"] == 20.0


def test_summer_seasonality():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-07-01",
                    "2025-08-01",
                    "2025-12-01",
                    "2026-01-01",
                    "2026-02-01",
                    "2026-06-01",
                ]
            ),
            "hour": [12] * 6,
            "timestamp": pd.to_datetime(
                [
                    "2025-07-01 11:00",
                    "2025-08-01 11:00",
                    "2025-12-01 11:00",
                    "2026-01-01 11:00",
                    "2026-02-01 11:00",
                    "2026-06-01 11:00",
                ]
            ),
            "energy_kwh": [
                200.0,
                200.0,
                100.0,
                100.0,
                100.0,
                200.0,
            ],
        }
    )

    result = calculate_profile_metrics(data)

    assert result["seasonality"] == "summer"

    assert result["summer_index"] > result["winter_index"]


def test_metrics_ignore_rows_outside_profile_period():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-06-01",
                    "2025-07-01",
                ]
            ),
            "hour": [12, 12],
            "timestamp": pd.to_datetime(
                [
                    "2025-06-01 11:00",
                    "2025-07-01 11:00",
                ]
            ),
            "energy_kwh": [
                1000.0,
                10.0,
            ],
        }
    )

    result = calculate_profile_metrics(data)

    assert result["observation_count"] == 1
    assert result["mean_kwh"] == 10.0


def test_zero_mean_does_not_create_infinite_ratios():
    data = _hourly_day(
        "2025-07-01",
        peak_value=0.0,
        off_peak_value=0.0,
    )

    result = calculate_profile_metrics(data)

    assert np.isnan(result["peak_ratio"])

    assert np.isnan(result["cv"])

    assert np.isnan(result["load_factor"])


def test_company_hourly_profiles_create_all_day_types():
    weekday = _hourly_day(
        "2025-07-04",
        peak_value=20.0,
        off_peak_value=10.0,
    )

    weekend = _hourly_day(
        "2025-07-05",
        peak_value=10.0,
        off_peak_value=5.0,
    )

    data = pd.concat(
        [
            weekday,
            weekend,
        ],
        ignore_index=True,
    )

    data["company_code"] = "Kompania 1"

    result = _calculate_company_hourly_profiles(data)

    assert set(result["profile_type"].unique()) == {
        "all_days",
        "weekday",
        "weekend",
    }

    assert len(result[result["profile_type"] == "all_days"]) == 24

    assert {
        "mean_kwh",
        "median_kwh",
        "p10_kwh",
        "p90_kwh",
    }.issubset(result.columns)


def test_meter_hourly_profiles_keep_meters_separate():
    meter_1 = _hourly_day(
        "2025-07-01",
        peak_value=20.0,
        off_peak_value=10.0,
    )

    meter_1["company_code"] = "Kompania 1"

    meter_1["meter_id"] = "Meter 1"

    meter_1["source_sheet"] = "Test"

    meter_1["source_column"] = "Meter 1 Column"

    meter_2 = _hourly_day(
        "2025-07-01",
        peak_value=40.0,
        off_peak_value=20.0,
    )

    meter_2["company_code"] = "Kompania 1"

    meter_2["meter_id"] = "Meter 2"

    meter_2["source_sheet"] = "Test"

    meter_2["source_column"] = "Meter 2 Column"

    data = pd.concat(
        [
            meter_1,
            meter_2,
        ],
        ignore_index=True,
    )

    result = _calculate_meter_hourly_profiles(data)

    assert result["meter_id"].nunique() == 2

    assert len(result) == 48

    meter_1_hour_12 = result[
        (result["meter_id"] == "Meter 1") & (result["hour"] == 12)
    ].iloc[0]

    meter_2_hour_12 = result[
        (result["meter_id"] == "Meter 2") & (result["hour"] == 12)
    ].iloc[0]

    assert meter_1_hour_12["mean_kwh"] == 20.0

    assert meter_2_hour_12["mean_kwh"] == 40.0
