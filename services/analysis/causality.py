"""
Causality Analysis Module
--------------------------
Granger causality and VAR modelling logic from notebook 04.
"""

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR


def granger_test(
    df: pd.DataFrame,
    cause: str,
    effect: str,
    max_lags: int = 24
) -> pd.DataFrame:
    """
    Test whether `cause` Granger-causes `effect`.

    Args:
        df       : DataFrame containing both columns
        cause    : name of the causing variable
        effect   : name of the effect variable
        max_lags : maximum lag to test

    Returns:
        DataFrame with columns [lag, f_stat, p_value, significant].
    """
    test_data = df[[effect, cause]].dropna()
    results   = grangercausalitytests(test_data, maxlag=max_lags, verbose=False)

    rows = []
    for lag, result in results.items():
        f_stat = result[0]["ssr_ftest"][0]
        p_val  = result[0]["ssr_ftest"][1]
        rows.append({
            "lag":         lag,
            "f_stat":      round(f_stat, 4),
            "p_value":     round(p_val,  6),
            "significant": p_val < 0.05,
        })

    return pd.DataFrame(rows)


def granger_summary(results: dict) -> pd.DataFrame:
    """
    Summarise Granger results across multiple variable pairs.

    Args:
        results : dict of {label: DataFrame from granger_test()}

    Returns:
        Summary DataFrame with first significant lag and minimum p-value.
    """
    rows = []
    for label, df in results.items():
        sig = df[df["significant"]]
        rows.append({
            "pair":             label,
            "first_sig_lag":    int(sig["lag"].min()) if not sig.empty else None,
            "min_p_value":      round(df["p_value"].min(), 6),
            "n_sig_lags":       len(sig),
            "bidirectional":    None,  # filled in by caller after testing both directions
        })
    return pd.DataFrame(rows)


def fit_var(df: pd.DataFrame, cols: list, max_lags: int = 24) -> tuple:
    """
    Fit a Vector Autoregression model on the specified columns.

    Args:
        df       : input DataFrame
        cols     : columns to include in the VAR
        max_lags : maximum lags to evaluate for order selection

    Returns:
        Tuple of (fitted VARResults, optimal lag order).
    """
    data      = df[cols].dropna()
    model     = VAR(data)
    lag_order = model.select_order(maxlags=max_lags)
    optimal   = lag_order.aic
    fitted    = model.fit(optimal)
    return fitted, optimal


def impulse_response(var_result, impulse: str, response: str, periods: int = 48) -> object:
    """
    Compute impulse response function.

    Args:
        var_result : fitted VARResults object
        impulse    : name of the shock variable
        response   : name of the response variable
        periods    : number of periods to simulate

    Returns:
        IRAnalysis object from statsmodels.
    """
    return var_result.irf(periods=periods)