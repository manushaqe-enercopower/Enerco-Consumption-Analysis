# EnerCo Consumption Analysis

Energy consumption analysis pipeline and interactive dashboard for anonymized EnerCo electricity-meter data.

The project analyzes hourly electricity consumption from June 2025 to June 2026, with the main annual profiling window covering July 2025 to June 2026.

## Project Objectives

The analysis covers:

- data quality assessment
- reshaping source data from wide to long format
- company-level consumption profiling
- meter-level analysis
- hourly outlier detection
- weather, holiday, and tariff analysis
- company clustering
- prosumer consumption and injection analysis
- interactive visualization through Streamlit

## Analysis Period

Source data:

- June 2025 – June 2026

Main annual profile window:

- July 2025 – June 2026

Energy values are analyzed primarily in kWh, with MWh and GWh used in the UI where appropriate.

---

## Project Structure

```text
Enerco-Consumption-Analysis/
├── data/
│   ├── processed/
│   │   ├── hourly_long/
│   │   ├── profile_metrics/
│   │   ├── outliers/
│   │   ├── factors/
│   │   └── prosumers/
│   └── reference/
│       └── weather/
│
├── reports/
│   ├── hapi_1_data_quality.xlsx
│   ├── hapi_3_profile_metrics.xlsx
│   ├── hapi_4_1_hourly_outliers.xlsx
│   ├── hapi_4_3_final_outliers.xlsx
│   ├── hapi_5_weather.xlsx
│   ├── hapi_5_holidays.xlsx
│   ├── hapi_5_tariff.xlsx
│   ├── hapi_6_clustering.xlsx
│   └── hapi_prosumers.xlsx
│
├── src/
│   ├── config.py
│   ├── loader.py
│   ├── quality.py
│   ├── reshape.py
│   ├── metrics.py
│   ├── outliers.py
│   ├── weather.py
│   ├── holidays.py
│   ├── tariff.py
│   ├── clustering.py
│   └── prosumers.py
│
├── tests/
│
├── ui/
│   ├── app.py
│   ├── components/
│   └── pages/
│       ├── data_quality.py
│       ├── company_profiles.py
│       ├── meter_analysis.py
│       ├── outliers.py
│       ├── external_factors.py
│       ├── clustering.py
│       └── prosumers.py
│
├── README.md
└── requirements.txt