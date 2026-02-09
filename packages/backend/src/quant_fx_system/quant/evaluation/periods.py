from __future__ import annotations

import pandas as pd


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
