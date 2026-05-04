"""
Correlation Analysis Module
----------------------------
Static, rolling, and cross-correlation logic from notebook 03.
"""

import pandas as pd
import numpy as np


def correlation_matrix(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for specified columns.

    Args:
        df   : input DataFrame
        cols : columns to include

    Returns:
        Correlation matrix as DataFrame.
    """
    return df[cols].corr()


def rolling_correlation(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 168  # 7 days * 24 hours
) -> pd.Series:
    """
    Compute rolling Pearson correlation between two series.

    Args:
        s1     : first series
        s2     : second series
        window : rolling window size in observations

    Returns:
        Series of rolling correlation values.
    """
    return s1.rolling(window).corr(s2)


def cross_correlations(
    cause: pd.Series,
    effect: pd.Series,
    max_lag: int = 48
) -> pd.DataFrame:
    """
    Compute cross-correlation between cause and effect across lags.

    A significant correlation at lag k means cause at time T
    correlates with effect at time T+k.

    Args:
        cause   : the leading variable
        effect  : the lagged variable
        max_lag : maximum lag in hours to test

    Returns:
        DataFrame with columns [lag, correlation].
    """
    lags  = range(-max_lag, max_lag + 1)
    corrs = [cause.corr(effect.shift(lag)) for lag in lags]
    ci    = 1.96 / np.sqrt(len(cause))

    return pd.DataFrame({
        "lag":         list(lags),
        "correlation": corrs,
        "ci_upper":    ci,
        "ci_lower":    -ci,
    })


def correlation_summary(df: pd.DataFrame, target: str, predictors: list) -> pd.DataFrame:
    """
    Summarise raw and differenced correlations with a target variable.

    Args:
        df         : combined DataFrame containing raw and _diff columns
        target     : target column name e.g. "SpotPriceEUR"
        predictors : list of predictor column names e.g. ["OffshoreWind"]

    Returns:
        Summary DataFrame.
    """
    rows = []
    for col in predictors:
        rows.append({
            "variable":       col,
            "raw_r":          round(df[target].corr(df[col]), 4),
            "differenced_r":  round(df[f"{target}_diff"].corr(df[f"{col}_diff"]), 4),
        })
    return pd.DataFrame(rows)