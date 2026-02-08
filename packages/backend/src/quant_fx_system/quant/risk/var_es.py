from __future__ import annotations

import pandas as pd


def historical_var(returns: pd.Series, window: int, alpha: float) -> pd.Series:
    var = returns.rolling(window=window).quantile(alpha)
    var.name = "var"
    return var


def historical_es(returns: pd.Series, window: int, alpha: float) -> pd.Series:
    def _es(x: pd.Series) -> float:
        cutoff = x.quantile(alpha)
        tail = x[x <= cutoff]
        if len(tail) == 0:
            return 0.0
        return tail.mean()

    es = returns.rolling(window=window).apply(_es, raw=False)
    es.name = "es"
    return es
