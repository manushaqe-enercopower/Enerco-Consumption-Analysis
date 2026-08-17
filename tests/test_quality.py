import pandas as pd

from src.quality import analyze_meter


def test_clean_meter():
    values = pd.Series([10.0] * 100)
    dates = pd.date_range(
        "2026-01-01",
        periods=100,
        freq="h",
    )

    result = analyze_meter(
        values,
        pd.Series(dates),
        {
            "source_sheet": "Test",
            "company_code": "Kompania 1",
            "source_column": "METER1",
            "meter_id": "METER1",
            "flow_type": "consumption",
        },
    )

    assert result["quality_status"] == "clean"
    assert result["negative_count"] == 0


def test_more_than_10_percent_internal_missing_is_unusable():
    values = pd.Series([10.0] * 10 + [None] * 11 + [10.0] * 79)

    dates = pd.date_range(
        "2026-01-01",
        periods=100,
        freq="h",
    )

    result = analyze_meter(
        values,
        pd.Series(dates),
        {
            "source_sheet": "Test",
            "company_code": "Kompania 1",
            "source_column": "METER1",
            "meter_id": "METER1",
            "flow_type": "consumption",
        },
    )

    assert result["quality_status"] == "unusable"
    assert result["coverage_status"] == "full_period"
    assert result["internal_missing_hours"] == 11
    assert result["internal_missing_percent"] == 11.0


def test_negative_value_requires_review():
    values = pd.Series([10.0] * 99 + [-1.0])

    dates = pd.date_range(
        "2026-01-01",
        periods=100,
        freq="h",
    )

    result = analyze_meter(
        values,
        pd.Series(dates),
        {
            "source_sheet": "Test",
            "company_code": "Kompania 1",
            "source_column": "METER1",
            "meter_id": "METER1",
            "flow_type": "consumption",
        },
    )

    assert result["quality_status"] == "review"
    assert result["negative_count"] == 1


def test_zero_run_over_48_hours_requires_review():
    values = pd.Series([10.0] * 10 + [0.0] * 49 + [10.0] * 41)

    dates = pd.date_range(
        "2026-01-01",
        periods=100,
        freq="h",
    )

    result = analyze_meter(
        values,
        pd.Series(dates),
        {
            "source_sheet": "Test",
            "company_code": "Kompania 1",
            "source_column": "METER1",
            "meter_id": "METER1",
            "flow_type": "consumption",
        },
    )

    assert result["quality_status"] == "review"
    assert result["zero_run_count_over_48h"] == 1
    assert result["max_zero_run_hours"] == 49


def test_leading_missing_values_are_partial_coverage():
    values = pd.Series([None] * 20 + [10.0] * 80)

    dates = pd.date_range(
        "2026-01-01",
        periods=100,
        freq="h",
    )

    result = analyze_meter(
        values,
        pd.Series(dates),
        {
            "source_sheet": "Test",
            "company_code": "Kompania 1",
            "source_column": "METER1",
            "meter_id": "METER1",
            "flow_type": "consumption",
        },
    )

    assert result["quality_status"] == "clean"
    assert result["coverage_status"] == "partial_period"
    assert result["leading_missing_hours"] == 20
    assert result["internal_missing_hours"] == 0
    assert result["internal_missing_percent"] == 0.0


def test_internal_missing_over_10_percent_is_unusable():
    values = pd.Series([10.0] * 10 + [None] * 11 + [10.0] * 79)

    dates = pd.date_range(
        "2026-01-01",
        periods=100,
        freq="h",
    )

    result = analyze_meter(
        values,
        pd.Series(dates),
        {
            "source_sheet": "Test",
            "company_code": "Kompania 1",
            "source_column": "METER1",
            "meter_id": "METER1",
            "flow_type": "consumption",
        },
    )

    assert result["quality_status"] == "unusable"
    assert result["coverage_status"] == "full_period"
    assert result["internal_missing_hours"] == 11
    assert result["internal_missing_percent"] == 11.0
