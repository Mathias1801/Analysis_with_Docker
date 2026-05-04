"""
Ingestion Service
-----------------
Fetches wind/solar production and spot price data from the
Energi Data Service API (api.energidataservice.dk) and saves
it as CSV files to the /data directory.

Datasets used:
  - Elspotprices                    : hourly electricity spot prices (EUR/MWh)
  - ProductionConsumptionSettlement : hourly wind + solar production (MWh)

No API key required. Rate limit: 1 request per dataset per minute.
"""

import requests
import pandas as pd
import os
import logging
import json
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL      = "https://api.energidataservice.dk/dataset"
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PRICE_AREA    = "DK1"
LOOKBACK_DAYS = 90
END_DATE_OVERRIDE = "2025-09-30"  # Elspotprices last available date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_dataset(dataset: str, start: str, end: str, filters: dict = None) -> pd.DataFrame:
    params = {"start": start, "end": end, "limit": 10_000}
    if filters:
        params["filter"] = json.dumps(filters)

    url = f"{BASE_URL}/{dataset}"
    log.info(f"Fetching {dataset} from {start} to {end}")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        records = response.json().get("records", [])
        log.info(f"  → {len(records)} records received")
        return pd.DataFrame(records)
    except requests.RequestException as e:
        log.error(f"Failed to fetch {dataset}: {e}")
        return pd.DataFrame()


def save(df: pd.DataFrame, filename: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    log.info(f"Saved {len(df)} rows → {path}")


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_spot_prices(start: str, end: str) -> pd.DataFrame:
    df = fetch_dataset(
        dataset="Elspotprices",
        start=start,
        end=end,
        filters={"PriceArea": [PRICE_AREA]}
    )
    if df.empty:
        return df
    df["HourDK"] = pd.to_datetime(df["HourDK"])
    df = df.sort_values("HourDK").reset_index(drop=True)
    return df


def fetch_production(start: str, end: str) -> pd.DataFrame:
    df = fetch_dataset(
        dataset="ProductionConsumptionSettlement",
        start=start,
        end=end,
        filters={"PriceArea": [PRICE_AREA]}
    )
    if df.empty:
        return df
    df["HourDK"] = pd.to_datetime(df["HourDK"])
    df = df.sort_values("HourDK").reset_index(drop=True)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    end       = datetime.fromisoformat(END_DATE_OVERRIDE)
    start     = end - timedelta(days=LOOKBACK_DAYS)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    prices     = fetch_spot_prices(start_str, end_str)
    production = fetch_production(start_str, end_str)

    if not prices.empty:
        save(prices, "spot_prices.csv")

    if not production.empty:
        save(production, "production.csv")

    log.info("Ingestion complete.")


if __name__ == "__main__":
    run()