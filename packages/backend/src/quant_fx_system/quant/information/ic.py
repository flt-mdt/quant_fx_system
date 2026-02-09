from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _prepare(signal: pd.Series, target: pd.Series, method: str) -> tuple[np.ndarray, np.ndarray]:
    aligned = pd.concat([signal, target], axis=1).dropna()
    x = aligned.iloc[:, 0].to_numpy(dtype=float)
    y = aligned.iloc[:, 1].to_numpy(dtype=float)
    if method == "spearman":
        x = pd.Series(x).rank().to_numpy(dtype=float)
        y = pd.Series(y).rank().to_numpy(dtype=float)
    return x, y


def compute_ic(signal: pd.Series, target: pd.Series, method: str = "spearman") -> float:
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    x, y = _prepare(signal, target, method)
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_ic_series(
    signal: pd.Series,
    target: pd.Series,
    window: int | None = None,
    method: str = "spearman",
) -> pd.Series:
    aligned = pd.concat([signal, target], axis=1)
    aligned.columns = ["signal", "target"]
    if window is None:
        return pd.Series(
            [compute_ic(aligned["signal"], aligned["target"], method=method)],
            index=[aligned.index.max()],
        )

    def _rolling_ic(frame: pd.DataFrame) -> float:
        return compute_ic(frame["signal"], frame["target"], method=method)

    return aligned.rolling(window=window, min_periods=window).apply(_rolling_ic, raw=False)[
        "signal"
    ]


def ic_by_feature(
    features: pd.DataFrame,
    target: pd.Series,
    method: str = "spearman",
    window: int | None = None,
    nw_lags: int | None = None,
) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        ic_series = compute_ic_series(features[col], target, window=window, method=method)
        stats = ic_stats(ic_series, nw_lags=nw_lags)
        stats["feature"] = col
        rows.append(stats)
    return pd.DataFrame(rows).set_index("feature")


def _nw_variance(series: np.ndarray, lags: int) -> float:
    n = len(series)
    mean = series.mean()
    gamma0 = np.sum((series - mean) ** 2) / n
    variance = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cov = np.sum((series[lag:] - mean) * (series[:-lag] - mean)) / n
        variance += 2.0 * weight * cov
    return variance


def ic_stats(ic_series: pd.Series, nw_lags: int | None) -> dict:
    ic_values = ic_series.dropna().to_numpy(dtype=float)
    n_obs = len(ic_values)
    if n_obs == 0:
        return {
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
            "tstat": float("nan"),
            "pvalue": float("nan"),
            "n_obs": 0,
            "nw_lags": nw_lags or 0,
        }

    mean = float(np.mean(ic_values))
    std = float(np.std(ic_values, ddof=1)) if n_obs > 1 else float("nan")
    ic_ir = mean / std if std and not math.isnan(std) else float("nan")
    lags = nw_lags or 0
    variance = _nw_variance(ic_values, lags) if lags > 0 else float(np.var(ic_values))
    se = math.sqrt(variance / n_obs) if variance > 0 else float("nan")
    tstat = mean / se if se and not math.isnan(se) else float("nan")
    pvalue = 2.0 * (1.0 - _normal_cdf(abs(tstat))) if not math.isnan(tstat) else float("nan")
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": ic_ir,
        "tstat": tstat,
        "pvalue": pvalue,
        "n_obs": n_obs,
        "nw_lags": lags,
    }


def ic_decay(signal: pd.Series, returns: pd.Series, horizons: list[int]) -> pd.DataFrame:
    rows = []
    from .targets import forward_returns

    for horizon in horizons:
        target = forward_returns(returns, horizon)
        ic = compute_ic(signal, target, method="spearman")
        rows.append({"horizon": horizon, "ic": ic})
    return pd.DataFrame(rows).set_index("horizon")


def ic_conditional(
    signal: pd.Series,
    target: pd.Series,
    regimes: pd.Series,
    method: str = "spearman",
) -> pd.DataFrame:
    aligned = pd.concat([signal, target, regimes], axis=1).dropna()
    aligned.columns = ["signal", "target", "regime"]
    rows = []
    for regime, frame in aligned.groupby("regime"):
        ic = compute_ic(frame["signal"], frame["target"], method=method)
        rows.append({"regime": regime, "ic": ic, "n_obs": len(frame)})
    return pd.DataFrame(rows).set_index("regime")
