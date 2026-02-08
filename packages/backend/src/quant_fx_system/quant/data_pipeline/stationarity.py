"""Stationarity diagnostics for FX time series."""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def run_adf(
    series: pd.Series, *, maxlag: int | None = None, regression: str = "c"
) -> dict[str, Any]:
    """Run the Augmented Dickey-Fuller test and return a structured dict."""

    if not isinstance(series, pd.Series):
        raise ValueError("series must be a pandas Series.")

    data = series.dropna()
    if data.empty:
        raise ValueError("series is empty after dropping NaNs.")

    result = adfuller(data, maxlag=maxlag, regression=regression, autolag="AIC")
    statistic, pvalue, usedlag, nobs, critical_values, icbest = result
    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "usedlag": usedlag,
        "nobs": nobs,
        "critical_values": critical_values,
        "icbest": icbest,
    }


def run_kpss(
    series: pd.Series, *, regression: str = "c", nlags: str | int = "auto"
) -> dict[str, Any]:
    """Run the KPSS test and return a structured dict."""

    if not isinstance(series, pd.Series):
        raise ValueError("series must be a pandas Series.")

    data = series.dropna()
    if data.empty:
        raise ValueError("series is empty after dropping NaNs.")

    statistic, pvalue, lags, critical_values = kpss(data, regression=regression, nlags=nlags)
    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "lags": lags,
        "critical_values": critical_values,
    }


def stationarity_report(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Generate a tidy stationarity report for multiple series."""

    rows: list[dict[str, Any]] = []
    for name, series in series_map.items():
        cleaned = series.dropna()
        if cleaned.empty:
            raise ValueError(f"Series '{name}' is empty after dropping NaNs.")

        adf_result = run_adf(cleaned)
        kpss_result = run_kpss(cleaned)

        rows.append(
            {
                "series": name,
                "adf_stat": adf_result["statistic"],
                "adf_pvalue": adf_result["pvalue"],
                "adf_usedlag": adf_result["usedlag"],
                "adf_nobs": adf_result["nobs"],
                "adf_crit_1pct": adf_result["critical_values"].get("1%"),
                "adf_crit_5pct": adf_result["critical_values"].get("5%"),
                "adf_crit_10pct": adf_result["critical_values"].get("10%"),
                "kpss_stat": kpss_result["statistic"],
                "kpss_pvalue": kpss_result["pvalue"],
                "kpss_lags": kpss_result["lags"],
                "kpss_crit_10pct": kpss_result["critical_values"].get("10%"),
                "kpss_crit_5pct": kpss_result["critical_values"].get("5%"),
                "kpss_crit_2_5pct": kpss_result["critical_values"].get("2.5%"),
                "kpss_crit_1pct": kpss_result["critical_values"].get("1%"),
            }
        )

    return pd.DataFrame(rows)
