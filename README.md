# Data Analysis with Docker
### Danish Wind & Solar Energy — Time Series and Causality Analysis

A full-stack data science project analysing whether renewable energy production
(wind and solar) Granger-causes electricity spot prices in Denmark (DK1 price area).

Built as a portfolio piece demonstrating applied statistics, Python data engineering,
and containerised service architecture.

---

```
data-analysis_with_docker/
│
├── notebooks/                        
│   ├── 01_data_exploration.ipynb     # Data ingestion and visual inspection
│   ├── 02_timeseries_analysis.ipynb  # Stationarity testing and decomposition
│   ├── 03_correlation_analysis.ipynb # Static, rolling, and cross-correlation
│   └── 04_causality_testing.ipynb    # Granger causality and VAR modelling
│
├── services/
│   ├── ingestion/                    # Fetches data from Energinet API
│   ├── analysis/                     # Runs full statistical pipeline
│   └── api/                          # FastAPI service serving results
│
├── data/                             # Shared volume — CSVs and results.json
├── docker-compose.yml
└── README.md
```
---

## Quickstart

### Prerequisites
- Docker Desktop
- Docker Compose v2+

### Run

```bash
# 1. Fetch data from Energinet
docker compose run ingestion

# 2. Run the statistical analysis pipeline
docker compose run analysis

# 3. Start the API
docker compose up api
```

API is now live at `http://localhost:8000`

Interactive docs at `http://localhost:8000/docs`

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `GET /results` | Full results payload |
| `GET /stationarity` | Stationarity test results |
| `GET /correlation` | Correlation summary |
| `GET /granger` | Granger causality summary |
| `GET /var` | VAR optimal lag order |

---

## Data Source

All data is sourced from the **Energi Data Service** public API provided by Energinet,
the Danish Transmission System Operator. No API key required.

Datasets used:
- `Elspotprices` — hourly electricity spot prices (EUR/MWh), DK1 price area
- `ProductionConsumptionSettlement` — hourly wind and solar production (MWh)

Period: 90-day rolling window ending at last available data point.

---

## Statistical Methodology

### 1. Stationarity Testing
Each series is tested using both the **ADF test** (null: non-stationary) and the
**KPSS test** (null: stationary) before any modelling. Non-stationary series are
first-differenced to ensure valid inference.

### 2. Correlation Analysis
Three levels of correlation analysis are applied:
- **Static Pearson correlation** on raw and differenced series
- **Rolling 7-day correlation** to detect structural changes over time
- **Cross-correlation** across ±48 hour lags to identify lead/lag relationships

### 3. Granger Causality
Granger causality tests whether past values of X improve the prediction of Y
beyond Y's own past values. Tests are run in both directions for each variable pair
to identify bidirectionality.

> **Important limitation:** Granger causality is *predictive*, not mechanistic.
> A significant result means X is a useful predictor of Y — not that X physically
> causes Y. Shared confounders (e.g. weather) can produce significant results
> with no true causal link.

### 4. Vector Autoregression (VAR)
A VAR model is fit jointly across all series. Lag order is selected by AIC.
Impulse Response Functions are used to estimate how a shock to wind production
propagates through spot prices over a 48-hour horizon.

---

## Key Findings

| Variable | Correlation (raw) | Correlation (differenced) | Wind→Price | Price→Wind |
|---|---|---|---|---|
| Offshore Wind | -0.13 | +0.07 | sig. from lag 1 | sig. from lag 4 |
| Onshore Wind | -0.48 | -0.17 | sig. from lag 2 | sig. from lag 1 |
| Solar | -0.40 | -0.40 | sig. from lag 1 | sig. from lag 2 |

**All three relationships are bidirectional.** Wind and solar Granger-cause spot
prices, but spot prices also Granger-cause wind and solar — consistent with
producers responding to market signals through curtailment and scheduling.

The gap between raw and differenced correlations for offshore wind (+0.07 vs -0.13)
suggests its raw correlation was almost entirely trend-driven, with no genuine
short-term predictive relationship with price. This would be missed by a naive
correlation analysis.

Solar's correlation is stable across both raw and differenced series (-0.40),
indicating a structural relationship independent of trends — consistent with
midday solar production systematically suppressing prices.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | Python, requests, pandas |
| Statistical analysis | statsmodels, numpy, scipy |
| Visualisation | matplotlib |
| API | FastAPI, uvicorn |
| Containerisation | Docker, Docker Compose |
| Data source | Energinet Energi Data Service API |

---

## Potential Extensions
- Extend to DK2 price area for cross-region comparison
- Add real-time ingestion on a schedule using a Kubernetes CronJob
- Incorporate cross-border exchange flows as additional causal variables
- Apply VECM (Vector Error Correction Model) if cointegration is found
