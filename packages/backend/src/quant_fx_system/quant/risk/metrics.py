from __future__ import annotations

import math

import pandas as pd


def compute_returns_from_price(price: pd.Series) -> pd.Series:
    returns = price.pct_change()
    returns.name = "returns"
    return returns


def rolling_vol(
    returns: pd.Series,
    window: int,
    ddof: int = 0,
    epsilon: float = 1e-12,
) -> pd.Series:
    vol = returns.rolling(window=window).std(ddof=ddof)
    vol = vol.clip(lower=epsilon)
    vol.name = "vol_rolling"
    return vol


def infer_periods_per_year(index: pd.DatetimeIndex) -> int | None:
    if index is None or len(index) < 2:
        return None
    freq = pd.infer_freq(index)
    if freq is None:
        return None
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except ValueError:
        return None

    name = offset.name
    if name in {"B", "C"}:
        return 252
    if name == "D":
        return 365
    if name == "W":
        return 52
    if name == "M":
        return 12
    if name == "Q":
        return 4
    if name in {"A", "Y"}:
        return 1
    if name == "H":
        return 24 * 365
    if name in {"T", "min"}:
        return 60 * 24 * 365
    if name == "S":
        return 60 * 60 * 24 * 365
    return None


def annualize_vol(vol: pd.Series, periods_per_year: int | None) -> pd.Series:
    if periods_per_year is None:
        return vol
    return vol * math.sqrt(periods_per_year)
