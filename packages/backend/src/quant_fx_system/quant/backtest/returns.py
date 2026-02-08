from __future__ import annotations

import pandas as pd


def compute_simple_returns(price: pd.Series) -> pd.Series:
    returns = price.pct_change()
    return returns.dropna()
