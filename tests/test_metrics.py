import numpy as np
import pandas as pd

from src.metrics import (
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
