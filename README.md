# EnerCo Consumption Analysis

Python-based electricity consumption analytics pipeline and interactive Streamlit dashboard for anonymized EnerCo meter data.

The project performs data-quality validation, reshaping, consumption profiling, meter-level analysis, outlier detection, external-factor analysis, clustering, and prosumer analysis.

---

## 1. Project Overview

The objective of this project is to analyze hourly electricity-consumption data and identify meaningful consumption patterns at portfolio, company, and meter level.

The solution includes:

* automated data-quality checks;
* wide-to-long data transformation;
* company-level consumption profiles;
* meter-level consumption analysis;
* peak and off-peak behavior;
* weekday and weekend behavior;
* seasonality and trend analysis;
* hourly outlier detection;
* weather, HDD, and CDD analysis;
* official-holiday analysis;
* tariff-period analysis;
* K-Means clustering;
* prosumer consumption and injection analysis;
* interactive Streamlit dashboard.

The analytical pipeline is implemented in Python, while the dashboard is built with Streamlit.

---

## 2. Analysis Period

### Source data

The source dataset covers:

```text
June 2025 – June 2026
```

### Main annual profiling window

The primary annual analysis uses:

```text
July 2025 – June 2026
```

This provides a complete 12-month period for seasonality, trends, load profiles, outliers, weather relationships, and clustering.

Energy values are stored primarily in **kWh**.

The dashboard also presents larger totals in **MWh** and **GWh** where appropriate.

---

## 3. Main Technologies

The project uses:

* Python 3
* pandas
* NumPy
* PyArrow
* OpenPyXL
* Matplotlib
* Plotly
* scikit-learn
* Streamlit
* pytest

---

## 4. Project Structure

```text
Enerco-Consumption-Analysis/
│
├── data/
│   ├── processed/
│   │   ├── hourly_long/
│   │   ├── profile_metrics/
│   │   ├── outliers/
│   │   ├── factors/
│   │   │   └── tariff/
│   │   └── prosumers/
│   │
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
│   ├── hapi_6_elbow.png
│   ├── hapi_6_silhouette.png
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
│   ├── test_loader.py
│   ├── test_quality.py
│   ├── test_reshape.py
│   ├── test_metrics.py
│   ├── test_outliers.py
│   ├── test_weather.py
│   ├── test_holidays.py
│   ├── test_tariff.py
│   ├── test_clustering.py
│   └── test_prosumers.py
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
```

---

# 5. Installation

## 5.1 Clone the repository

```bash
git clone <repository-url>
cd Enerco-Consumption-Analysis
```

If the repository has already been cloned, simply open a terminal in the project root.

Example:

```text
C:\Users\<user>\Desktop\Enerco-Consumption-Analysis
```

---

## 5.2 Create a virtual environment

This only needs to be done once.

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

When activated, the terminal should show something similar to:

```text
(.venv)
```

Do not recreate `.venv` if it already exists.

---

## 5.3 Install dependencies

```bash
pip install -r requirements.txt
```

The current dependency versions are defined in `requirements.txt`.

---

# 6. How to Run the Project

There are two ways to run the project:

1. run the full analytical pipeline;
2. run only the Streamlit dashboard using already-generated outputs.

---

# 7. Run the Full Analytical Pipeline

Run all commands from the **repository root**.

Recommended execution order:

```bash
python -m src.quality
python -m src.reshape
python -m src.metrics
python -m src.outliers
python -m src.weather
python -m src.holidays
python -m src.tariff
python -m src.clustering
python -m src.prosumers
```

After the analytical outputs have been generated, start the dashboard:

```bash
streamlit run ui/app.py
```

Streamlit will normally open:

```text
http://localhost:8501
```

If it does not open automatically, copy the local URL displayed in the terminal into a browser.

---

# 8. Run Only the Dashboard

If all processed datasets and reports already exist, the analytical pipeline does **not** need to be rerun.

Start the application directly with:

```bash
streamlit run ui/app.py
```

The main Streamlit entry file is:

```text
ui/app.py
```

The dashboard reads the generated Parquet and Excel outputs from:

```text
data/processed/
reports/
```

Therefore, those outputs must exist before the relevant dashboard pages can be used.

---

# 9. Pipeline Execution Details

## 9.1 Data Quality

Run:

```bash
python -m src.quality
```

This stage validates the source meter series.

Checks include:

* missing observations;
* coverage percentage;
* incomplete periods;
* invalid values;
* negative consumption;
* long zero-value runs;
* extreme values;
* first and last valid timestamps;
* internal gaps;
* annual-profile coverage.

Meters are classified into quality groups such as:

```text
clean
review
unusable
```

Primary output:

```text
reports/hapi_1_data_quality.xlsx
```

The report contains:

* `Meter Quality`
* `Timeline Quality`

Current source dataset:

```text
517 meter/series records
271 clean
42 review
204 unusable
```

---

## 9.2 Wide-to-Long Transformation

Run:

```bash
python -m src.reshape
```

The source Excel structure contains hourly series stored across multiple sheets.

This stage converts those source series into normalized long-format hourly data.

The resulting structure includes information such as:

```text
company_code
meter_id
timestamp
energy_kwh
flow_type
source_sheet
```

Output directory:

```text
data/processed/hourly_long/
```

The transformed dataset is split into multiple Parquet files to make processing more manageable.

---

## 9.3 Company and Meter Consumption Profiles

Run:

```bash
python -m src.metrics
```

This stage calculates annual consumption-profile metrics at company and meter level.

Main metrics include:

* observation count;
* total consumption;
* average consumption;
* maximum consumption;
* peak-period average;
* off-peak average;
* peak/off-peak ratio;
* weekday average;
* weekend average;
* weekday/weekend ratio;
* coefficient of variation;
* load factor;
* summer consumption;
* winter consumption;
* summer index;
* winter index;
* seasonality;
* seasonality index;
* trend percentage.

Outputs include:

```text
data/processed/profile_metrics/company_profiles.parquet
data/processed/profile_metrics/company_hourly_profile.parquet
data/processed/profile_metrics/company_monthly.parquet
data/processed/profile_metrics/meter_profiles.parquet
data/processed/profile_metrics/meter_hourly_profile.parquet
data/processed/profile_metrics/meter_monthly.parquet
```

Excel report:

```text
reports/hapi_3_profile_metrics.xlsx
```

Current profile results include:

```text
66 companies
300 usable consumption meters
```

Current seasonality classification:

```text
42 winter profiles
4 summer profiles
20 without strong seasonality
```

---

# 10. Meter-Level Analysis

Meter-level analysis allows individual meters belonging to the same company to be compared.

The dashboard provides:

* meter selection;
* annual metrics;
* 24-hour profile;
* monthly profile;
* meter-to-meter comparison;
* seasonality;
* trend;
* similarity analysis;
* technical identification fields;
* detailed meter table.

This analysis is available from:

```text
Analiza e njehsorëve
```

in the Streamlit sidebar.

---

# 11. Outlier Detection

Run:

```bash
python -m src.outliers
```

Consumption is aggregated at:

```text
company + hour
```

A Z-score is calculated independently for each company.

The basic formula is:

```text
Z = (hourly consumption - company mean) / company standard deviation
```

An hourly observation is classified as an outlier when:

```text
|Z| > 3
```

Current results:

```text
66 companies analyzed
578,160 company-hour observations
3,625 outlier hours
49 companies with outliers
approximately 0.63% outlier rate
maximum |Z| approximately 40.76
```

Outputs:

```text
data/processed/outliers/company_hourly.parquet
data/processed/outliers/company_hourly_outliers.parquet
data/processed/outliers/company_outlier_summary.parquet
```

Reports:

```text
reports/hapi_4_1_hourly_outliers.xlsx
reports/hapi_4_3_final_outliers.xlsx
```

The Streamlit interface includes:

* overview;
* company-level analysis;
* temporal distribution;
* severity classification;
* detailed outlier table;
* methodology.

### Outlier severity

The dashboard groups outliers approximately as:

```text
3 ≤ |Z| < 4     Moderate
4 ≤ |Z| < 6     High
|Z| ≥ 6         Very high
```

An outlier should not automatically be interpreted as a measurement error.

It indicates statistically unusual behavior requiring technical or operational interpretation.

---

# 12. Weather Analysis

Run:

```bash
python -m src.weather
```

Weather data is analyzed for Prishtina.

The analysis includes:

* temperature;
* Heating Degree Days;
* Cooling Degree Days;
* portfolio consumption;
* correlations with consumption.

The base temperature used for HDD/CDD calculations is:

```text
18°C
```

Weather data is retrieved from a public weather API.

The implementation includes:

* multiple API attempts;
* retry delays;
* cached local fallback.

This prevents temporary API failures from blocking the analytical pipeline.

Cached weather data is stored under:

```text
data/reference/weather/
```

Report:

```text
reports/hapi_5_weather.xlsx
```

---

# 13. Official Holiday Analysis

Run:

```bash
python -m src.holidays
```

This stage analyzes consumption behavior during official holidays.

The analysis includes:

* holiday dates found in the dataset;
* holiday consumption;
* comparable non-holiday consumption;
* percentage impact;
* interaction between holidays and outliers.

Report:

```text
reports/hapi_5_holidays.xlsx
```

---

# 14. Tariff Analysis

Run:

```bash
python -m src.tariff
```

Consumption is analyzed according to tariff periods.

The processed analysis includes:

```text
T1
T2
```

The dashboard provides:

* tariff summary;
* monthly tariff profile;
* hourly tariff profile;
* company-level tariff analysis.

Processed outputs are stored under:

```text
data/processed/factors/tariff/
```

Report:

```text
reports/hapi_5_tariff.xlsx
```

---

# 15. Clustering

Run:

```bash
python -m src.clustering
```

K-Means clustering groups companies with similar annual consumption behavior.

The model uses six profile features:

```text
peak_ratio
weekday_weekend_ratio
cv
load_factor
seasonality_index
trend_percent
```

Before clustering, features are standardized using:

```text
StandardScaler
```

Candidate values of `k` are evaluated using:

* inertia;
* Silhouette Score.

Current result:

```text
65 companies clustered
1 company excluded
optimal k = 2
Silhouette Score ≈ 0.458
```

The clustering configuration uses:

```python
KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10,
)
```

Outputs:

```text
reports/hapi_6_clustering.xlsx
reports/hapi_6_elbow.png
reports/hapi_6_silhouette.png
```

The Streamlit page includes:

* cluster overview;
* evaluation of `k`;
* cluster profiles;
* company assignments;
* excluded companies;
* methodology.

Cluster IDs are technical model labels.

For example:

```text
Cluster 0
Cluster 1
```

does not mean that one cluster is better or worse than another.

---

# 16. Prosumer Analysis

Run:

```bash
python -m src.prosumers
```

The prosumer source contains paired energy-flow measurements.

The project interprets:

```text
A+ = electricity consumed from the grid
A- = electricity injected into the grid
```

The prosumer pipeline calculates:

* total consumption;
* total injection;
* net-grid balance;
* injection/consumption ratio;
* hours with injection;
* net-export hours;
* monthly profile;
* average 24-hour profile.

Net-grid energy is calculated as:

```text
net_grid_kwh = consumption_kwh - injection_kwh
```

A negative value indicates net export to the grid for the corresponding period.

Processed files:

```text
data/processed/prosumers/prosumer_hourly.parquet
data/processed/prosumers/prosumer_summary.parquet
data/processed/prosumers/prosumer_monthly.parquet
data/processed/prosumers/prosumer_hourly_profile.parquet
data/processed/prosumers/prosumer_portfolio_monthly.parquet
data/processed/prosumers/prosumer_portfolio_hourly_profile.parquet
```

Report:

```text
reports/hapi_prosumers.xlsx
```

Current portfolio results include approximately:

```text
13 prosumer meters
12.62 GWh consumption
1.25 GWh injection
9.88% injection-to-consumption ratio
11.38 GWh net-grid balance
```

---

# 17. Streamlit Dashboard

Start the dashboard with:

```bash
streamlit run ui/app.py
```

The application contains the following pages:

```text
Përmbledhje
Cilësia e të dhënave
Profilet e kompanive
Analiza e njehsorëve
Outlier-at
Klasterizimi
Prosumerët
Faktorët shtesë
```

The dashboard is designed primarily in Albanian because it is intended for internal analytical use.

Source-code identifiers, modules, paths, and technical field names remain in English.

---

# 18. Running Tests

Run the complete automated test suite:

```bash
python -m pytest -q
```

Current validated result:

```text
68 passed
```

Run a specific test file with:

```bash
python -m pytest tests/test_metrics.py -q
```

Example:

```bash
python -m pytest tests/test_outliers.py -q
```

---

# 19. Python Compilation Check

To verify that the Python source files compile successfully:

```bash
python -m compileall src ui tests
```

This checks:

```text
src/
ui/
tests/
```

for syntax and import-level compilation issues.

---

# 20. Recommended Final Validation

Before committing or delivering the project, run:

```bash
python -m pytest -q
python -m compileall src ui tests
```

Then verify Git state:

```bash
git status
```

A fully committed project should return:

```text
nothing to commit, working tree clean
```

---

# 21. Known Limitations

## Business metadata

The anonymized source dataset does not include a verified mapping for:

```text
business_sector
company_size
voltage_level
TS / network grouping
```

These values are intentionally not inferred.

This means that analyses requiring verified company metadata cannot be performed reliably.

In particular:

* sector-level outlier comparison is skipped;
* cluster-to-sector comparison is skipped;
* voltage-level comparison is unavailable;
* TS/network grouping is unavailable.

The pipeline retains the corresponding metadata fields so that verified mappings can be incorporated later.

---

## Sector comparison

The clustering report explicitly marks sector comparison as unavailable because business-sector metadata is not mapped.

No assumptions are made about company sectors.

---

## Weather data

Weather information is obtained from an external public API.

Temporary API failures are handled through retries and a local fallback cache.

If both the API and local fallback are unavailable, weather-dependent analysis cannot be regenerated.

---

## Statistical interpretation

Outliers, correlations, clusters, and profile classifications are analytical indicators.

They should not automatically be interpreted as:

* equipment failure;
* billing error;
* fraud;
* causal relationships;
* operational faults.

Business and technical interpretation is still required.

---

# 22. Data Privacy

The analysis uses anonymized identifiers such as:

```text
Kompania 1
Kompania 2
Kompania 3
...
```

No mapping between anonymized company codes and real customer identities is included in the repository.

Any confidential mapping should remain outside the repository unless specifically authorized.

---

# 23. Deployment

The dashboard is fully functional locally with:

```bash
streamlit run ui/app.py
```

The Streamlit application entry point is:

```text
ui/app.py
```

The project can technically be deployed to Streamlit-compatible hosting.

However, cloud deployment should only be performed when authorization exists to host the repository and the associated processed company data on an external service.

Local execution does not require cloud deployment.

---

# 24. Current Validation Status

The current implementation has been validated with:

```text
68 automated tests passed
```

and:

```text
python -m compileall src ui tests
```

completes successfully.

The analytical pipeline, reports, tests, and Streamlit dashboard cover the implemented scope of the EnerCo consumption-analysis assignment.
