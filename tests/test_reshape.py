import pandas as pd

from src.reshape import reshape_sheet


def test_reshape_normal_meter():
    base_df = pd.DataFrame(
        {
            "PeriodYear": [2026, 2026],
            "PeriodMonth": [6, 6],
            "WeekDay": ["Monday", "Monday"],
            "Date": [
                "2026-06-01",
                "2026-06-01",
            ],
            "Hour": [1, 2],
            "Tariff": ["T2", "T2"],
        }
    )

    meter_df = pd.DataFrame(
        {
            "METER1": [
                10.5,
                11.5,
            ]
        }
    )

    metadata_df = pd.DataFrame(
        [
            {
                "source_sheet": "Test",
                "company_code": "Kompania 1",
                "source_column": "METER1",
                "meter_id": "METER1",
                "flow_type": "consumption",
            }
        ]
    )

    result = reshape_sheet(
        base_df,
        meter_df,
        metadata_df,
        "Test",
    )

    assert len(result) == 2

    assert result.iloc[0]["company_code"] == "Kompania 1"

    assert result.iloc[0]["meter_id"] == "METER1"

    assert result.iloc[0]["flow_type"] == "consumption"

    assert result.iloc[0]["energy_kwh"] == 10.5


def test_timestamp_converts_hour_1_to_midnight():
    base_df = pd.DataFrame(
        {
            "PeriodYear": [2026],
            "PeriodMonth": [6],
            "WeekDay": ["Monday"],
            "Date": ["2026-06-01"],
            "Hour": [1],
            "Tariff": ["T2"],
        }
    )

    meter_df = pd.DataFrame({"METER1": [10.0]})

    metadata_df = pd.DataFrame(
        [
            {
                "source_sheet": "Test",
                "company_code": "Kompania 1",
                "source_column": "METER1",
                "meter_id": "METER1",
                "flow_type": "consumption",
            }
        ]
    )

    result = reshape_sheet(
        base_df,
        meter_df,
        metadata_df,
        "Test",
    )

    assert result.iloc[0]["timestamp"] == pd.Timestamp("2026-06-01 00:00:00")


def test_prosumer_consumption_and_solar_injection_remain_separate():
    base_df = pd.DataFrame(
        {
            "PeriodYear": [2026],
            "PeriodMonth": [6],
            "WeekDay": ["Monday"],
            "Date": ["2026-06-01"],
            "Hour": [1],
            "Tariff": ["T2"],
        }
    )

    meter_df = pd.DataFrame(
        {
            "DFE9013260 - A+": [20.0],
            "DFE9013260 - A-": [5.0],
        }
    )

    metadata_df = pd.DataFrame(
        [
            {
                "source_sheet": "Prosumer",
                "company_code": "Kompania 6",
                "source_column": "DFE9013260 - A+",
                "meter_id": "DFE9013260",
                "flow_type": "consumption",
            },
            {
                "source_sheet": "Prosumer",
                "company_code": "Kompania 6",
                "source_column": "DFE9013260 - A-",
                "meter_id": "DFE9013260",
                "flow_type": "solar_injection",
            },
        ]
    )

    result = reshape_sheet(
        base_df,
        meter_df,
        metadata_df,
        "Prosumer",
    )

    assert len(result) == 2

    consumption = result[result["flow_type"] == "consumption"].iloc[0]

    solar_injection = result[result["flow_type"] == "solar_injection"].iloc[0]

    assert consumption["energy_kwh"] == 20.0
    assert solar_injection["energy_kwh"] == 5.0

    assert consumption["meter_id"] == solar_injection["meter_id"]


def test_multiple_meters_create_multiple_long_rows():
    base_df = pd.DataFrame(
        {
            "PeriodYear": [2026, 2026],
            "PeriodMonth": [6, 6],
            "WeekDay": ["Monday", "Monday"],
            "Date": [
                "2026-06-01",
                "2026-06-01",
            ],
            "Hour": [1, 2],
            "Tariff": ["T2", "T2"],
        }
    )

    meter_df = pd.DataFrame(
        {
            "METER1": [10.0, 11.0],
            "METER2": [20.0, 21.0],
        }
    )

    metadata_df = pd.DataFrame(
        [
            {
                "source_sheet": "Test",
                "company_code": "Kompania 1",
                "source_column": "METER1",
                "meter_id": "METER1",
                "flow_type": "consumption",
            },
            {
                "source_sheet": "Test",
                "company_code": "Kompania 2",
                "source_column": "METER2",
                "meter_id": "METER2",
                "flow_type": "consumption",
            },
        ]
    )

    result = reshape_sheet(
        base_df,
        meter_df,
        metadata_df,
        "Test",
    )

    assert len(result) == 4
