"""
Time Series Analysis Module
----------------------------
Stationarity testing and decomposition logic from notebook 02.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose


def test_stationarity(series: pd.Series, name: str) -> dict:
    """
    Run ADF and KPSS stationarity tests on a series.

    Returns:
        dict with test statistics, p-values, and verdict.
    """
    series = series.dropna()

    adf_stat, adf_p, _, _, _, _  = adfuller(series, autolag="AIC")
    kpss_stat, kpss_p, _, _      = kpss(series, regression="c", nlags="auto")

    adf_result  = "stationary"     if adf_p  < 0.05 else "non-stationary"
    kpss_result = "non-stationary" if kpss_p < 0.05 else "stationary"

    if adf_result == kpss_result:
        verdict = adf_result
    else:
        verdict = "inconclusive"

    return {
        "series":    name,
        "adf_stat":  round(adf_stat,  4),
        "adf_p":     round(adf_p,     4),
        "kpss_stat": round(kpss_stat, 4),
        "kpss_p":    round(kpss_p,    4),
        "verdict":   verdict,
    }


def difference_series(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Apply first differencing to the specified columns.
    Adds new columns with _diff suffix and drops the single NaN row.

    Args:
        df   : input DataFrame
        cols : list of column names to difference

    Returns:
        DataFrame with added _diff columns, NaN row dropped.
    """
    df = df.copy()
    for col in cols:
        df[f"{col}_diff"] = df[col].diff()
    return df.dropna().reset_index(drop=True)


def decompose(series: pd.Series, period: int = 24) -> dict:
    """
    Seasonal decomposition of a time series.

    Args:
        series : time-indexed pandas Series
        period : seasonality period in observations (default 24 = daily)

    Returns:
        dict with trend, seasonal, and residual Series.
    """
    result = seasonal_decompose(series, model="additive", period=period)
    return {
        "trend":    result.trend,
        "seasonal": result.seasonal,
        "resid":    result.resid,
    }