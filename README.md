**# EnerCo Consumption Analysis**
> Python-based electricity-consumption analytics pipeline and Streamlit dashboard for anonymized EnerCo hourly meter data.
The project transforms the supplied Excel workbook into validated analytical datasets, reports, figures, and an internal dashboard covering data quality, consumption profiles, outliers, external factors, clustering, and prosumer behavior.
**## Quick Start**
| Task | Command / Location |
|---|---|
| Install dependencies | `pip install -r requirements.txt` |
| Add source workbook | `data/raw/` |
| Run full pipeline | `python -m src.pipeline` |
| Start dashboard | `streamlit run ui/app.py` |
| Run tests | `python -m pytest -q` |
> The source workbook is not included in the repository and must be added manually before running the pipeline.
**## 1. Project Overview**
The project covers:
- data-quality validation
- wide-to-long transformation
- company-level consumption profiling
- meter-level profiling and comparison
- peak and off-peak behavior
- weekday and weekend behavior
- seasonality and trend analysis
- hourly outlier detection
- weather, HDD, and CDD analysis
- official-holiday impact analysis
- tariff-period analysis
- K-Means clustering
- prosumer A+ / A- analysis
- combined portfolio views across all analyzed companies
- Excel export of company lists by clustering group
- interactive Streamlit visualization.
The analytical pipeline is implemented in Python and the dashboard is implemented with Streamlit.
**## 2. Analysis Period**
Source data:
```text
June 2025 – June 2026
```
Main annual profiling period:
```text
July 2025 – June 2026
```
The July 2025 – June 2026 window provides a complete 12-month period for annual profiles, seasonality, trends, weather relationships, outlier analysis, and clustering.
Main energy unit:
```text
kWh
```
The dashboard may use MWh or GWh for larger totals.
**## 3. Technologies**
Main dependencies:
- Python 3
- pandas
- NumPy
- PyArrow
- OpenPyXL
- Matplotlib
- Plotly
- scikit-learn
- Streamlit
- pytest
Exact package versions are defined in `requirements.txt`.
**## 4. Project Structure**
```text
Enerco-Consumption-Analysis/
├── data/
│   ├── raw/
│   │   └── <source-meter-file>.xlsx
│   ├── processed/
│   │   ├── hourly_long/
│   │   ├── profile_metrics/
│   │   ├── outliers/
│   │   ├── factors/
│   │   │   ├── weather/
│   │   │   ├── holidays/
│   │   │   └── tariff/
│   │   └── prosumers/
│   └── reference/
│       └── weather/
├── reports/
│   ├── figures/
│   ├── hapi_1_data_quality.xlsx
│   ├── hapi_3_profile_metrics.xlsx
│   ├── hapi_4_1_hourly_outliers.xlsx
│   ├── hapi_4_3_final_outliers.xlsx
│   ├── hapi_5_weather.xlsx
│   ├── hapi_5_holidays.xlsx
│   ├── hapi_5_tariff.xlsx
│   ├── hapi_6_clustering.xlsx
│   ├── hapi_6_elbow.png
│   ├── hapi_6_silhouette.png
│   └── hapi_prosumers.xlsx
├── src/
│   ├── config.py
│   ├── loader.py
│   ├── pipeline.py
│   ├── quality.py
│   ├── reshape.py
│   ├── metrics.py
│   ├── outliers.py
│   ├── weather.py
│   ├── holidays.py
│   ├── tariff.py
│   ├── clustering.py
│   └── prosumers.py
├── tests/
├── ui/
│   ├── app.py
│   ├── components/
│   └── pages/
│       ├── data_quality.py
│       ├── company_profiles.py
│       ├── meter_analysis.py
│       ├── outliers.py
│       ├── external_factors.py
│       ├── clustering.py
│       └── prosumers.py
├── README.md
└── requirements.txt
```
**## 5. Installation**
Clone the repository:
```bash
git clone <repository-url>
cd Enerco-Consumption-Analysis
```
Create a virtual environment.
**### Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
**### Git Bash**
```bash
python -m venv .venv
source .venv/Scripts/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run all commands from the repository root unless stated otherwise.
**## 6. Source Data Setup**
The anonymized source Excel workbook is not included in the repository and must be added manually before the pipeline is executed.
Place the workbook inside:
```text
data/raw/
```
Example:
```text
data/
└── raw/
    └── Enerco_June_2025-June_2026_Hourly_Interval_Meters_ANONYMIZED.xlsx
```
The configured file path must match the source path defined in:
```text
src/config.py
```
Typical workbook sheets are:
```text
Meters 1-100
101-200
201-300
301-400
401-481
Prosumer
```
Do not commit confidential or non-anonymized customer data to the repository.
**## 7. Run the Full Pipeline**
Run the complete analytical workflow with:
```bash
python -m src.pipeline
```
The pipeline executes the stages in this order:
```text
quality
reshape
metrics
outliers
weather
holidays
tariff
clustering
prosumers
```
The orchestrator executes the existing modules sequentially. If one module fails, execution stops immediately and returns the failing exit code.
This prevents later stages from running against incomplete intermediate outputs.
Generated processed datasets are written to:
```text
data/processed/
```
Generated reports and figures are written to:
```text
reports/
```
**### Run a single stage**
For development or troubleshooting, modules can still be executed individually:
```bash
python -m src.quality
python -m src.metrics
python -m src.outliers
```
For a normal complete run, use `python -m src.pipeline`.
**## 8. Pipeline Stages**
**### 8.1 Data Quality**
**\*\*Module:\*\*** `src/quality.py`
This stage checks all meter and prosumer series before downstream analysis.
Checks include:
- missing observations
- leading and trailing missing periods
- internal gaps
- first and last valid timestamps
- incomplete coverage
- invalid values
- negative values
- long zero-value runs
- extreme values
- annual-profile coverage.
Series are classified into quality statuses such as:
```text
clean
review
unusable
```
Current validated result:
```text
Meters/series: 517
Clean: 271
Review: 42
Unusable: 204
Full-period coverage: 190
Partial-period coverage: 327
```
**\*\*Report:\*\*** `reports/hapi_1_data_quality.xlsx`
**### 8.2 Wide-to-Long Transformation**
**\*\*Module:\*\*** `src/reshape.py`
The source workbook contains hourly measurements across several sheets and columns. This stage converts those measurements into normalized long-format records.
Typical fields include:
```text
company_code
meter_id
timestamp
energy_kwh
flow_type
source_sheet
```
Current validated result:
```text
Measurement series: 517
Long-format rows: 4,901,160
```
**\*\*Output:\*\*** `data/processed/hourly_long/`
The data is stored as multiple Parquet files for more efficient processing.
If verified company metadata is not provided, metadata fields are marked as pending mapping rather than inferred.
**### 8.3 Company and Meter Profiles**
**\*\*Module:\*\*** `src/metrics.py`
This stage calculates annual profile metrics at company and meter level.
Main metrics include:
- total, mean, and maximum consumption
- peak and off-peak averages
- peak/off-peak ratio
- weekday and weekend averages
- weekday/weekend ratio
- coefficient of variation
- load factor
- summer and winter averages
- seasonality index
- seasonality classification
- trend percentage.
Current validated result:
```text
Companies analyzed: 81
Consumption meters analyzed: 483
```
Outputs:
```text
data/processed/profile_metrics/
reports/hapi_3_profile_metrics.xlsx
data/processed/profile_metrics/portfolio_profile.parquet
data/processed/profile_metrics/portfolio_monthly.parquet
data/processed/profile_metrics/portfolio_hourly_profile.parquet
```
These datasets are also used by the company-profile, meter-analysis, and clustering dashboard pages.
Supported analyses include a **Të gjitha kompanitë** option. When selected, the underlying hourly company data is aggregated first and visualized as one combined portfolio graph rather than as separate company lines.
**### 8.4 Outlier Detection**
**\*\*Module:\*\*** `src/outliers.py`
Consumption is analyzed at company-hour level.
A Z-score is calculated using the company consumption distribution:
```text
Z = (hourly consumption - company mean) / company standard deviation
```
An observation is considered an outlier when:
```text
|Z| > 3
```
Current validated result:
```text
Companies analyzed: 66
Company-hour observations: 578,160
Outlier hours: 3,625
Companies with outliers: 49
Maximum |Z|: 40.7580
```
The final report groups unusual observations into operational review categories.
Outputs:
```text
data/processed/outliers/
reports/hapi_4_1_hourly_outliers.xlsx
reports/hapi_4_3_final_outliers.xlsx
```
Sector-level comparison is skipped when verified sector metadata is unavailable.
An outlier represents statistically unusual behavior; it does not automatically indicate a measurement error or technical fault.
**### 8.5 Weather, HDD, and CDD**
**\*\*Module:\*\*** `src/weather.py`
Weather analysis uses Prishtina as the reference location.
The stage analyzes:
- temperature
- Heating Degree Days
- Cooling Degree Days
- portfolio consumption
- correlations between consumption and weather variables.
HDD/CDD base temperature:
```text
18°C
```
Current validated result:
```text
Weather days: 365
Reference location: Prishtina
Mean temperature: 11.93°C
Temperature range: -9.60°C to 39.80°C
Temperature correlation: -0.3096
HDD correlation: 0.3704
CDD correlation: -0.0704
```
Weather retrieval includes retries and a local cache fallback so temporary API failures do not automatically stop the full pipeline.
Outputs:
```text
data/reference/weather/
data/processed/factors/weather/
reports/hapi_5_weather.xlsx
reports/figures/weather/
```
**### 8.6 Official Holiday Analysis**
**\*\*Module:\*\*** `src/holidays.py`
This stage compares portfolio consumption on official observed holidays with comparable non-holiday periods. Holiday and hourly holiday charts already use the combined portfolio, so no additional company selector is required for the portfolio view.
The analysis includes:
- holiday dates present in the dataset
- holiday consumption
- comparison consumption
- percentage impact
- outlier activity during holidays.
Current validated result:
```text
Official observed holiday dates: 12
Holiday dates found in data: 12
Average portfolio holiday impact: -23.32%
Median portfolio holiday impact: -23.25%
Outlier hours on holidays: 33
Companies with holiday outliers: 11
```
Outputs:
```text
data/processed/factors/holidays/
reports/hapi_5_holidays.xlsx
```
**### 8.7 Tariff Analysis**
**\*\*Module:\*\*** `src/tariff.py`
Consumption is separated into:
```text
T1
T2
```
The analysis includes tariff totals, tariff shares, hourly behavior, monthly behavior, and company-level tariff profiles.
Current validated result:
```text
Companies analyzed: 66
Consumption observations used: 2,584,016
T1: 38,499,844.09 kWh (67.35%)
T2: 18,660,126.54 kWh (32.65%)
```
Outputs:
```text
data/processed/factors/tariff/
reports/hapi_5_tariff.xlsx
```
**### 8.8 Company Clustering**
**\*\*Module:\*\*** `src/clustering.py`
K-Means clustering groups companies according to similarities in annual consumption behavior.
Features used:
```text
peak_ratio
weekday_weekend_ratio
cv
load_factor
seasonality_index
trend_percent
```
Features are standardized before clustering.
Candidate values of `k` are evaluated using inertia and Silhouette Score.
Current validated result:
```text
Companies available: 81
Companies clustered: 81
Companies excluded: 0
Evaluated k: 2-10
Best k: 2
Cluster 0: 65 companies
Cluster 1: 16 companies
```
Outputs:
```text
reports/hapi_6_clustering.xlsx
reports/hapi_6_elbow.png
reports/hapi_6_silhouette.png
```
Cluster IDs are model labels only and do not indicate better or worse performance.
From the **Kompanitë** tab in the dashboard, the company lists can be exported to Excel. The export creates separate sheets for each clustering group, for example `Grupi_1` and `Grupi_2`.
Sector comparison is skipped when verified sector metadata is unavailable.
**### 8.9 Prosumer Analysis**
**\*\*Module:\*\*** `src/prosumers.py`
The prosumer source contains paired flow measurements:
```text
A+ = electricity consumed from the grid
A- = electricity injected into the grid
```
Net-grid exchange is calculated as:
```text
net_grid_kwh = consumption_kwh - injection_kwh
```
The analysis includes:
- total A+ consumption
- total A- injection
- net-grid balance
- injection/import ratio
- injection hours
- net-export hours
- monthly profiles
- average hourly profiles.
Current validated result:
```text
Prosumer meters analyzed: 13
Companies represented: 11
Total A+ consumption: 12,624,472.92 kWh
Total A- injection: 1,247,502.71 kWh
Net grid exchange: 11,376,970.21 kWh
Injection/import ratio: 9.88%
Net-export hours: 20,320
```
Outputs:
```text
data/processed/prosumers/
reports/hapi_prosumers.xlsx
reports/figures/prosumers/
```
**## 9. Streamlit Dashboard**
After the pipeline has generated the required outputs, start the application with:
```bash
streamlit run ui/app.py
```
**\*\*Main entry point:\*\*** `ui/app.py`
Streamlit normally opens a local URL similar to:
```text
http\://localhost:8501
```
Dashboard pages:
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
The dashboard is primarily in Albanian because it is intended for internal analytical use.
Supported company-level analyses include a **Të gjitha kompanitë** option. This combines the available data from all analyzed companies into a single portfolio view for aggregate charts such as hourly profiles and monthly trends. Clustering remains company-level by design because K-Means requires companies to remain separate observations.
The clustering page also supports Excel export of company lists by group.
Source-code identifiers, technical field names, and file paths remain in English.
**### Start only the dashboard**
If the processed datasets and reports already exist, the pipeline does not have to be rerun.
Start directly with:
```bash
streamlit run ui/app.py
```
Rerun the analytical pipeline when the source workbook or processing logic changes, or when generated outputs need to be refreshed.
**## 10. Main Outputs**
| Area | Main output |
|---|---|
| Data quality | `reports/hapi_1_data_quality.xlsx` |
| Profiles | `reports/hapi_3_profile_metrics.xlsx` |
| Outliers | `reports/hapi_4_1_hourly_outliers.xlsx` |
| Final outliers | `reports/hapi_4_3_final_outliers.xlsx` |
| Weather | `reports/hapi_5_weather.xlsx` |
| Holidays | `reports/hapi_5_holidays.xlsx` |
| Tariff | `reports/hapi_5_tariff.xlsx` |
| Clustering | `reports/hapi_6_clustering.xlsx` + dashboard Excel export by group |
| Prosumers | `reports/hapi_prosumers.xlsx` |
Processed Parquet datasets are stored under `data/processed/`.
Figures are stored under `reports/` and `reports/figures/`.
**## 11. Tests**
Run the complete automated test suite:
```bash
python -m pytest -q
```
Run the suite after changes and confirm all tests pass.
Run a specific test file when needed:
```bash
python -m pytest tests/test_metrics.py -q
```
Example:
```bash
python -m pytest tests/test_outliers.py -q
```
**## 12. Compilation Check**
Check Python source files for syntax and import-level compilation issues:
```bash
python -m compileall src ui tests
```
This checks:
```text
src/
ui/
tests/
```
**## 13. Recommended Final Validation**
Before committing or delivering the project, run:
```bash
python -m pytest -q
python -m compileall src ui tests
python -m src.pipeline
```
Then verify Git state:
```bash
git status
```
A fully committed repository should return:
```text
nothing to commit, working tree clean
```
**## 14. Known Limitations**
**### Business metadata**
The anonymized source data does not contain verified mappings for:
```text
business_sector
company_size
voltage_level
TS / network grouping
```
These values are intentionally not inferred.
Therefore:
- sector-level outlier comparison is skipped
- cluster-to-sector comparison is skipped
- voltage-level comparison is unavailable
- TS/network grouping is unavailable.
The relevant metadata fields remain in the processing structure so verified mappings can be added later.
**### Weather dependency**
Weather data is obtained from an external public source.
The implementation uses retries and a local cached fallback. If both the API and local cache are unavailable, weather-dependent analysis cannot be regenerated.
**### Statistical interpretation**
Outliers, correlations, clusters, seasonality classes, and trends are analytical indicators.
They should not automatically be interpreted as:
- equipment failure
- billing error
- fraud
- causal relationships
- operational faults.
Technical and business interpretation is still required.
**## 15. Data Privacy**
The project uses anonymized company and meter identifiers.
No mapping between anonymized identifiers and real customer identities is included in the repository.
Confidential mappings, credentials, and non-anonymized customer data should remain outside version control unless explicitly authorized.
**## 16. Current Validation Status**
Validate the current implementation with:
```bash
python -m pytest -q
```
Compilation:
```bash
python -m compileall src ui tests
```
Full analytical execution:
```bash
python -m src.pipeline
```
The latest full run completed successfully through:
```text
quality
reshape
metrics
outliers
weather
holidays
tariff
clustering
prosumers
```
The project currently includes the complete implemented analytical pipeline, generated reports, processed datasets, automated tests, and Streamlit dashboard.
