import pandas as pd
import pytest

from src.prosumers import (
    create_portfolio_hourly_profile,
    create_portfolio_monthly,
    create_prosumer_hourly,
    create_prosumer_hourly_profile,
    create_prosumer_monthly,
    create_prosumer_summary,
    prepare_prosumer_data,
)


def make_prosumer_data() -> pd.DataFrame:
    rows = []

    timestamps = pd.date_range(
        "2026-01-01",
        periods=4,
        freq="h",
    )

    consumption = [
        10.0,
        12.0,
        8.0,
        5.0,
    ]

    injection = [
        0.0,
        2.0,
        10.0,
        7.0,
    ]

    for index, timestamp in enumerate(timestamps):
        rows.append(
            {
                "date": timestamp.normalize(),
                "hour": index + 1,
                "timestamp": timestamp,
                "company_code": "Kompania 1",
                "meter_id": "Meter 1",
                "flow_type": "consumption",
                "energy_kwh": consumption[index],
                "source_sheet": "Prosumer",
                "source_column": "Meter 1 - A+",
            }
        )

        rows.append(
            {
                "date": timestamp.normalize(),
                "hour": index + 1,
                "timestamp": timestamp,
                "company_code": "Kompania 1",
                "meter_id": "Meter 1",
                "flow_type": "solar_injection",
                "energy_kwh": injection[index],
                "source_sheet": "Prosumer",
                "source_column": "Meter 1 - A-",
            }
        )

    return pd.DataFrame(rows)


def test_prepare_prosumer_data_requires_both_flows():
    data = make_prosumer_data()

    normal_meter = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "hour": 1,
                "timestamp": pd.Timestamp("2026-01-01 00:00"),
                "company_code": "Kompania 2",
                "meter_id": "Normal Meter",
                "flow_type": "consumption",
                "energy_kwh": 50.0,
                "source_sheet": "Sheet1",
                "source_column": "Normal Meter",
            }
        ]
    )

    data = pd.concat(
        [
            data,
            normal_meter,
        ],
        ignore_index=True,
    )

    result = prepare_prosumer_data(data)

    assert result["company_code"].unique().tolist() == ["Kompania 1"]

    assert set(result["flow_type"]) == {
        "consumption",
        "solar_injection",
    }


def test_prepare_prosumer_data_excludes_unusable_series():
    data = make_prosumer_data()

    quality = pd.DataFrame(
        {
            "source_sheet": [
                "Prosumer",
                "Prosumer",
            ],
            "source_column": [
                "Meter 1 - A+",
                "Meter 1 - A-",
            ],
            "quality_status": [
                "clean",
                "unusable",
            ],
        }
    )

    result = prepare_prosumer_data(
        data,
        quality_df=quality,
    )

    assert result.empty


def test_create_prosumer_hourly_pairs_a_plus_and_a_minus():
    data = prepare_prosumer_data(make_prosumer_data())

    result = create_prosumer_hourly(data)

    assert len(result) == 4

    assert result.iloc[0]["consumption_kwh"] == pytest.approx(10.0)

    assert result.iloc[0]["injection_kwh"] == pytest.approx(0.0)

    assert result.iloc[0]["net_grid_kwh"] == pytest.approx(10.0)


def test_net_export_is_detected():
    data = prepare_prosumer_data(make_prosumer_data())

    result = create_prosumer_hourly(data)

    export_hours = result[result["is_net_export"]]

    assert len(export_hours) == 2


def test_prosumer_summary_calculates_energy_balance():
    data = prepare_prosumer_data(make_prosumer_data())

    hourly = create_prosumer_hourly(data)

    result = create_prosumer_summary(hourly)

    row = result.iloc[0]

    assert row["total_consumption_kwh"] == pytest.approx(35.0)

    assert row["total_injection_kwh"] == pytest.approx(19.0)

    assert row["net_grid_kwh"] == pytest.approx(16.0)

    assert row["injection_import_ratio_percent"] == pytest.approx(
        54.285714,
        rel=1e-5,
    )

    assert row["net_export_hours"] == 2


def test_monthly_summary_preserves_energy_balance():
    data = prepare_prosumer_data(make_prosumer_data())

    hourly = create_prosumer_hourly(data)

    result = create_prosumer_monthly(hourly)

    row = result.iloc[0]

    assert row["consumption_kwh"] == pytest.approx(35.0)

    assert row["injection_kwh"] == pytest.approx(19.0)

    assert row["net_grid_kwh"] == pytest.approx(16.0)


def test_hourly_profile_contains_variability_fields():
    data = prepare_prosumer_data(make_prosumer_data())

    hourly = create_prosumer_hourly(data)

    result = create_prosumer_hourly_profile(hourly)

    assert "p10_consumption_kwh" in result.columns

    assert "p90_consumption_kwh" in result.columns

    assert "p10_injection_kwh" in result.columns

    assert "p90_injection_kwh" in result.columns


def test_portfolio_aggregations_preserve_totals():
    data = prepare_prosumer_data(make_prosumer_data())

    hourly = create_prosumer_hourly(data)

    monthly = create_portfolio_monthly(hourly)

    hourly_profile = create_portfolio_hourly_profile(hourly)

    assert monthly["consumption_kwh"].sum() == pytest.approx(35.0)

    assert monthly["injection_kwh"].sum() == pytest.approx(19.0)

    assert len(hourly_profile) == 4
