from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd


CorrelationMethod = Literal["pearson", "spearman"]


def _corr(a: pd.Series, b: pd.Series, method: CorrelationMethod) -> float:
    if method == "spearman":
        return float(a.rank().corr(b.rank()))
    return float(a.corr(b))


def compute_ic_series(
    signal: pd.Series,
    target: pd.Series,
    window: int | None = None,
    method: CorrelationMethod = "spearman",
) -> pd.Series:
    aligned = pd.concat([signal, target], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    if window is None:
        return pd.Series([_corr(aligned.iloc[:, 0], aligned.iloc[:, 1], method)])
    if method == "pearson":
        return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1]).dropna()
    return aligned.rolling(window).apply(
        lambda frame: _corr(frame.iloc[:, 0], frame.iloc[:, 1], method),
        raw=False,
    ).iloc[:, 0].dropna()


def _nw_tstat(values: np.ndarray, lags: int | None = None) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    mean = np.mean(values)
    if lags is None:
        lags = int(np.sqrt(n))
    lags = max(0, min(lags, n - 1))
    demeaned = values - mean
    gamma0 = np.mean(demeaned * demeaned)
    var = gamma0
    for lag in range(1, lags + 1):
        cov = np.mean(demeaned[lag:] * demeaned[:-lag])
        weight = 1.0 - lag / (lags + 1)
        var += 2.0 * weight * cov
    if var <= 0:
        return float("nan")
    return float(mean / np.sqrt(var / n))


def ic_stats(
    signal: pd.Series,
    target: pd.Series,
    window: int | None = None,
    method: CorrelationMethod = "spearman",
    nw_lags: int | None = None,
) -> dict[str, Any]:
    aligned = pd.concat([signal, target], axis=1).dropna()
    if aligned.empty:
        return {
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "tstat": float("nan"),
            "pvalue": float("nan"),
            "n_obs": 0,
        }

    ic_series = compute_ic_series(signal, target, window=window, method=method)
    ic_mean = float(ic_series.mean())
    if window is None or len(ic_series) <= 1:
        return {
            "ic_mean": ic_mean,
            "ic_std": float("nan"),
            "tstat": float("nan"),
            "pvalue": float("nan"),
            "n_obs": len(aligned),
        }

    ic_std = float(ic_series.std(ddof=1))
    n_obs = len(ic_series)
    tstat = _nw_tstat(ic_series.to_numpy(), lags=nw_lags)
    if math.isnan(tstat) and ic_std != 0:
        tstat = ic_mean / (ic_std / math.sqrt(n_obs))
    if math.isnan(tstat):
        pvalue = float("nan")
    else:
        pvalue = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(tstat) / math.sqrt(2.0))))
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "tstat": float(tstat),
        "pvalue": float(pvalue),
        "n_obs": n_obs,
    }
