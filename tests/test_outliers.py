import numpy as np
import pandas as pd

from src.outliers import (
    build_final_outlier_report,
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
            "hour": [
                1,
                2,
                1,
            ],
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
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00",
                ]
            ),
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


def test_final_report_adds_required_fields():
    hourly = pd.DataFrame(
        {
            "company_code": ["Kompania 1"],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                ]
            ),
            "hour": [10],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 09:00",
                ]
            ),
            "energy_kwh": [100.0],
            "company_mean_kwh": [20.0],
            "company_std_kwh": [10.0],
            "z_score": [8.0],
            "absolute_z_score": [8.0],
            "outlier_direction": ["high"],
            "z_threshold": [3.0],
        }
    )

    summary = pd.DataFrame(
        {
            "company_code": ["Kompania 1"],
            "total_hours": [8760],
            "outlier_hours": [20],
            "max_abs_z_score": [8.0],
            "outlier_percent": [0.23],
        }
    )

    result = build_final_outlier_report(
        hourly,
        summary,
    )

    assert "severity" in result.columns
    assert "reason" in result.columns
    assert "recommendation" in result.columns

    assert result.iloc[0]["severity"] == "critical"

    assert result.iloc[0]["recommendation"] == "Per shqyrtim teknik"


def test_repeated_outliers_trigger_technical_review():
    hourly = pd.DataFrame(
        {
            "company_code": ["Kompania 1"],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                ]
            ),
            "hour": [10],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 09:00",
                ]
            ),
            "energy_kwh": [50.0],
            "company_mean_kwh": [20.0],
            "company_std_kwh": [10.0],
            "z_score": [3.2],
            "absolute_z_score": [3.2],
            "outlier_direction": ["high"],
            "z_threshold": [3.0],
        }
    )

    summary = pd.DataFrame(
        {
            "company_code": ["Kompania 1"],
            "total_hours": [8760],
            "outlier_hours": [200],
            "max_abs_z_score": [3.2],
            "outlier_percent": [2.28],
        }
    )

    result = build_final_outlier_report(
        hourly,
        summary,
    )

    assert result.iloc[0]["recommendation"] == "Per shqyrtim teknik"


def test_moderate_outlier_not_automatically_error():
    hourly = pd.DataFrame(
        {
            "company_code": ["Kompania 1"],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                ]
            ),
            "hour": [10],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 09:00",
                ]
            ),
            "energy_kwh": [50.0],
            "company_mean_kwh": [20.0],
            "company_std_kwh": [10.0],
            "z_score": [3.2],
            "absolute_z_score": [3.2],
            "outlier_direction": ["high"],
            "z_threshold": [3.0],
        }
    )

    summary = pd.DataFrame(
        {
            "company_code": ["Kompania 1"],
            "total_hours": [8760],
            "outlier_hours": [5],
            "max_abs_z_score": [3.2],
            "outlier_percent": [0.06],
        }
    )

    result = build_final_outlier_report(
        hourly,
        summary,
    )

    assert result.iloc[0]["severity"] == "moderate"

    assert "sjellje biznesi legjitime" in result.iloc[0]["recommendation"]


def test_high_outlier_requires_operational_verification():
    hourly = pd.DataFrame(
        {
            "company_code": ["Kompania 1"],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                ]
            ),
            "hour": [10],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 09:00",
                ]
            ),
            "energy_kwh": [50.0],
            "company_mean_kwh": [20.0],
            "company_std_kwh": [10.0],
            "z_score": [4.5],
            "absolute_z_score": [4.5],
            "outlier_direction": ["high"],
            "z_threshold": [3.0],
        }
    )

    summary = pd.DataFrame(
        {
            "company_code": ["Kompania 1"],
            "total_hours": [8760],
            "outlier_hours": [5],
            "max_abs_z_score": [4.5],
            "outlier_percent": [0.06],
        }
    )

    result = build_final_outlier_report(
        hourly,
        summary,
    )

    assert result.iloc[0]["severity"] == "high"

    assert result.iloc[0]["recommendation"] == "Per verifikim operacional"
