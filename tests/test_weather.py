import numpy as np
import pandas as pd
import pytest

import src.weather as weather_module
from src.weather import (
    calculate_degree_days,
    create_company_weather_sensitivity,
    create_temperature_response,
    load_or_fetch_weather,
    merge_weather,
    parse_open_meteo_response,
    prepare_consumption_data,
)


def test_calculate_degree_days():
    temperatures = pd.Series(
        [
            10.0,
            18.0,
            25.0,
        ]
    )

    hdd, cdd = calculate_degree_days(temperatures)

    assert hdd.tolist() == [
        8.0,
        0.0,
        0.0,
    ]

    assert cdd.tolist() == [
        0.0,
        0.0,
        7.0,
    ]


def test_parse_open_meteo_response():
    payload = {
        "daily": {
            "time": [
                "2026-01-01",
                "2026-01-02",
            ],
            "temperature_2m_mean": [
                10.0,
                20.0,
            ],
            "temperature_2m_min": [
                5.0,
                15.0,
            ],
            "temperature_2m_max": [
                15.0,
                25.0,
            ],
        }
    }

    result = parse_open_meteo_response(payload)

    assert len(result) == 2

    assert result.iloc[0]["hdd"] == pytest.approx(8.0)

    assert result.iloc[1]["cdd"] == pytest.approx(2.0)

    assert result.iloc[0]["temperature_range_c"] == pytest.approx(10.0)


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


def test_merge_weather():
    consumption = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
            "portfolio_kwh": [
                100.0,
                120.0,
            ],
        }
    )

    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
            "temperature_mean_c": [
                10.0,
                20.0,
            ],
            "temperature_min_c": [
                5.0,
                15.0,
            ],
            "temperature_max_c": [
                15.0,
                25.0,
            ],
            "temperature_range_c": [
                10.0,
                10.0,
            ],
            "hdd": [
                8.0,
                0.0,
            ],
            "cdd": [
                0.0,
                2.0,
            ],
        }
    )

    result = merge_weather(
        consumption,
        weather,
    )

    assert len(result) == 2

    assert result["temperature_mean_c"].notna().all()


def test_company_weather_sensitivity_detects_heating_response():
    dates = pd.date_range(
        "2026-01-01",
        periods=30,
        freq="D",
    )

    hdd = np.linspace(
        1.0,
        15.0,
        30,
    )

    data = pd.DataFrame(
        {
            "company_code": ["Kompania 1"] * 30,
            "date": dates,
            "daily_kwh": (100.0 + hdd * 20.0),
            "temperature_mean_c": (18.0 - hdd),
            "temperature_min_c": (15.0 - hdd),
            "temperature_max_c": (21.0 - hdd),
            "temperature_range_c": [6.0] * 30,
            "hdd": hdd,
            "cdd": [0.0] * 30,
            "month": ["2026-01"] * 30,
        }
    )

    result = create_company_weather_sensitivity(data)

    row = result.iloc[0]

    assert row["hdd_correlation"] > 0.99

    assert row["heating_kwh_per_hdd"] == pytest.approx(
        20.0,
        rel=1e-3,
    )

    assert row["dominant_weather_response"] == "heating"


def test_temperature_response_contains_variability_band():
    data = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=10,
                freq="D",
            ),
            "portfolio_kwh": [
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
                180,
                190,
            ],
            "temperature_mean_c": [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
            ],
        }
    )

    result = create_temperature_response(
        data,
        bin_width=5.0,
    )

    assert not result.empty

    assert "p10_portfolio_kwh" in result.columns

    assert "p90_portfolio_kwh" in result.columns


def test_load_or_fetch_weather_uses_valid_cache(
    tmp_path,
):
    cache = tmp_path / "weather.csv"

    data = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ),
            "temperature_mean_c": [
                10.0,
                11.0,
                12.0,
            ],
            "temperature_min_c": [
                5.0,
                6.0,
                7.0,
            ],
            "temperature_max_c": [
                15.0,
                16.0,
                17.0,
            ],
            "temperature_range_c": [
                10.0,
                10.0,
                10.0,
            ],
            "hdd": [
                8.0,
                7.0,
                6.0,
            ],
            "cdd": [
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    data.to_csv(
        cache,
        index=False,
    )

    result, source = load_or_fetch_weather(
        cache_path=cache,
        start=pd.Timestamp("2026-01-01"),
        end=pd.Timestamp("2026-01-03"),
    )

    assert len(result) == 3

    assert source == "local_cache"


def test_api_failure_falls_back_to_cache(
    tmp_path,
    monkeypatch,
):
    cache = tmp_path / "weather.csv"

    data = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ),
            "temperature_mean_c": [
                10.0,
                11.0,
                12.0,
            ],
            "temperature_min_c": [
                5.0,
                6.0,
                7.0,
            ],
            "temperature_max_c": [
                15.0,
                16.0,
                17.0,
            ],
            "temperature_range_c": [
                10.0,
                10.0,
                10.0,
            ],
            "hdd": [
                8.0,
                7.0,
                6.0,
            ],
            "cdd": [
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    data.to_csv(
        cache,
        index=False,
    )

    def fail_fetch(
        *args,
        **kwargs,
    ):
        raise RuntimeError("API failure")

    monkeypatch.setattr(
        weather_module,
        "fetch_from_open_meteo",
        fail_fetch,
    )

    result, source = load_or_fetch_weather(
        cache_path=cache,
        start=pd.Timestamp("2026-01-01"),
        end=pd.Timestamp("2026-01-03"),
        refresh=True,
    )

    assert len(result) == 3

    assert source == "local_cache_fallback"
