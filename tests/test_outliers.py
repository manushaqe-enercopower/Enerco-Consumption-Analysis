import numpy as np
import pandas as pd

from src.outliers import (
    create_company_outlier_summary,
    detect_company_hourly_outliers,
)


def test_detects_high_z_score_outlier():
    normal_values = [10.0] * 100

    data = pd.DataFrame(
        {
            "company_code": (["Kompania 1"] * 101),
            "date": pd.date_range(
                "2026-01-01",
                periods=101,
                freq="h",
            ).normalize(),
            "hour": [(i % 24) + 1 for i in range(101)],
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=101,
                freq="h",
            ),
            "energy_kwh": (normal_values + [1000.0]),
        }
    )

    result = detect_company_hourly_outliers(data)

    assert len(result) == 1
    assert result.iloc[0]["outlier_direction"] == "high"
    assert result.iloc[0]["absolute_z_score"] > 3


def test_constant_company_has_no_outliers():
    data = pd.DataFrame(
        {
            "company_code": (["Kompania 1"] * 100),
            "date": pd.date_range(
                "2026-01-01",
                periods=100,
                freq="h",
            ).normalize(),
            "hour": [(i % 24) + 1 for i in range(100)],
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=100,
                freq="h",
            ),
            "energy_kwh": [10.0] * 100,
        }
    )

    result = detect_company_hourly_outliers(data)

    assert result.empty


def test_threshold_is_strictly_greater_than_three():
    data = pd.DataFrame(
        {
            "company_code": ["Kompania 1"] * 4,
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                ]
            ),
            "hour": [1, 2, 3, 4],
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=4,
                freq="h",
            ),
            "energy_kwh": [
                10.0,
                10.0,
                10.0,
                10.0,
            ],
        }
    )

    result = detect_company_hourly_outliers(
        data,
        threshold=3.0,
    )

    assert result.empty


def test_summary_counts_company_outliers():
    company_hourly = pd.DataFrame(
        {
            "company_code": [
                "Kompania 1",
                "Kompania 1",
                "Kompania 2",
            ],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                ]
            ),
            "hour": [1, 2, 1],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 00:00",
                ]
            ),
            "energy_kwh": [
                10.0,
                100.0,
                20.0,
            ],
        }
    )

    outliers = pd.DataFrame(
        {
            "company_code": ["Kompania 1"],
            "timestamp": pd.to_datetime(["2026-01-01 01:00"]),
            "absolute_z_score": [4.5],
        }
    )

    result = create_company_outlier_summary(
        company_hourly,
        outliers,
    )

    company_1 = result[result["company_code"] == "Kompania 1"].iloc[0]

    company_2 = result[result["company_code"] == "Kompania 2"].iloc[0]

    assert company_1["outlier_hours"] == 1
    assert company_1["total_hours"] == 2

    assert company_2["outlier_hours"] == 0


def test_no_infinite_z_scores():
    data = pd.DataFrame(
        {
            "company_code": ["Kompania 1"] * 100,
            "date": pd.date_range(
                "2026-01-01",
                periods=100,
                freq="h",
            ).normalize(),
            "hour": [(i % 24) + 1 for i in range(100)],
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=100,
                freq="h",
            ),
            "energy_kwh": [0.0] * 100,
        }
    )

    result = detect_company_hourly_outliers(data)

    assert not np.isinf(
        result.get(
            "z_score",
            pd.Series(dtype=float),
        )
    ).any()
