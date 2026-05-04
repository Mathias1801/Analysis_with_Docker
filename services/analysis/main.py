"""
Analysis Service
-----------------
Orchestrates the full analysis pipeline:
  1. Load combined data from /data
  2. Run stationarity tests and differencing
  3. Run correlation analysis
  4. Run Granger causality tests and VAR
  5. Save results as JSON to /data/results.json

This is the cleaned-up version of notebooks 02-04.
"""

import pandas as pd
import os
import json
import logging

from timeseries  import test_stationarity, difference_series
from correlation import correlation_summary
from causality   import granger_test, granger_summary, fit_var

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DIFF_COLS = ["SpotPriceEUR", "OffshoreWind", "OnshoreWind", "Solar"]


def load_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "combined.csv")
    df   = pd.read_csv(path, parse_dates=["HourDK"])
    log.info(f"Loaded {len(df)} rows from {path}")
    return df


def run():
    df = load_data()

    # ── 1. Stationarity ───────────────────────────────────────────────────────
    log.info("Running stationarity tests...")
    stationarity = [test_stationarity(df[col], col) for col in DIFF_COLS]

    # ── 2. Differencing ───────────────────────────────────────────────────────
    log.info("Differencing series...")
    df = difference_series(df, DIFF_COLS)

    # ── 3. Correlation ────────────────────────────────────────────────────────
    log.info("Running correlation analysis...")
    corr = correlation_summary(
        df,
        target     = "SpotPriceEUR",
        predictors = ["OffshoreWind", "OnshoreWind", "Solar"]
    )

    # ── 4. Granger causality ──────────────────────────────────────────────────
    log.info("Running Granger causality tests...")
    granger_results = {
        "OffshoreWind→Price": granger_test(df, "OffshoreWind_diff", "SpotPriceEUR_diff"),
        "OnshoreWind→Price":  granger_test(df, "OnshoreWind_diff",  "SpotPriceEUR_diff"),
        "Solar→Price":        granger_test(df, "Solar_diff",        "SpotPriceEUR_diff"),
        "Price→OffshoreWind": granger_test(df, "SpotPriceEUR_diff", "OffshoreWind_diff"),
        "Price→OnshoreWind":  granger_test(df, "SpotPriceEUR_diff", "OnshoreWind_diff"),
        "Price→Solar":        granger_test(df, "SpotPriceEUR_diff", "Solar_diff"),
    }
    summary = granger_summary(granger_results)

    # Mark bidirectional relationships
    forward  = {"OffshoreWind", "OnshoreWind", "Solar"}
    for i, row in summary.iterrows():
        pair = row["pair"]
        if "→Price" in pair:
            variable  = pair.replace("→Price", "")
            reverse   = f"Price→{variable}"
            rev_row   = summary[summary["pair"] == reverse]
            if not rev_row.empty:
                is_bidi = bool(rev_row.iloc[0]["first_sig_lag"] is not None)
                summary.at[i, "bidirectional"] = is_bidi
        elif "Price→" in pair:
            variable  = pair.replace("Price→", "")
            forward_p = f"{variable}→Price"
            fwd_row   = summary[summary["pair"] == forward_p]
            if not fwd_row.empty:
                is_bidi = bool(fwd_row.iloc[0]["first_sig_lag"] is not None)
                summary.at[i, "bidirectional"] = is_bidi

    # ── 5. VAR ────────────────────────────────────────────────────────────────
    log.info("Fitting VAR model...")
    var_cols   = ["SpotPriceEUR_diff", "OffshoreWind_diff", "OnshoreWind_diff", "Solar_diff"]
    _, opt_lag = fit_var(df, var_cols)

    # ── 6. Save results ───────────────────────────────────────────────────────
    results = {
        "stationarity": stationarity,
        "correlation":  corr.to_dict(orient="records"),
        "granger":      summary.to_dict(orient="records"),
        "var_optimal_lag": int(opt_lag),
    }

    out_path = os.path.join(DATA_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    run()