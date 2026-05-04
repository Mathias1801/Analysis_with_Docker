"""
API Service
-----------
FastAPI service that serves the analysis results from results.json.

Endpoints:
    GET /              → health check
    GET /results       → full results JSON
    GET /stationarity  → stationarity test results
    GET /correlation   → correlation summary
    GET /granger       → granger causality summary
    GET /var           → VAR optimal lag
"""

import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "data")
RESULTS_PATH = os.path.join(DATA_DIR, "results.json")

app = FastAPI(
    title="Energy Causality API",
    description="Serves statistical analysis results for Danish wind/solar vs. spot price.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_results() -> dict:
    if not os.path.exists(RESULTS_PATH):
        raise HTTPException(
            status_code=503,
            detail="Results not available yet — run the analysis service first."
        )
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


@app.get("/")
def health():
    return {"status": "ok", "service": "energy-causality-api"}


@app.get("/results")
def get_results():
    """Full results payload."""
    return load_results()


@app.get("/stationarity")
def get_stationarity():
    """Stationarity test results for all series."""
    return load_results()["stationarity"]


@app.get("/correlation")
def get_correlation():
    """Raw and differenced correlation with spot price."""
    return load_results()["correlation"]


@app.get("/granger")
def get_granger():
    """Granger causality summary including bidirectional flags."""
    return load_results()["granger"]


@app.get("/var")
def get_var():
    """VAR model optimal lag order."""
    data = load_results()
    return {"var_optimal_lag": data["var_optimal_lag"]}