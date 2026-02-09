from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .types import InformationConfig


_DEF_EPS = 1e-12


def _bin_series(series: pd.Series, bins: int, binning: str) -> pd.Series:
    if binning == "quantile":
        return pd.qcut(series, q=bins, labels=False, duplicates="drop")
    if binning == "equal_width":
        return pd.cut(series, bins=bins, labels=False)
    raise ValueError("binning must be 'quantile' or 'equal_width'")


def _contingency(x: pd.Series, y: pd.Series) -> np.ndarray:
    xy = pd.concat([x, y], axis=1).dropna()
    x_vals = xy.iloc[:, 0].astype(int).to_numpy()
    y_vals = xy.iloc[:, 1].astype(int).to_numpy()
    x_max = x_vals.max() + 1
    y_max = y_vals.max() + 1
    hist = np.zeros((x_max, y_max), dtype=float)
    for xi, yi in zip(x_vals, y_vals, strict=False):
        hist[xi, yi] += 1.0
    return hist


def _hist_mi(x: pd.Series, y: pd.Series, bins: int, binning: str) -> float:
    x_disc = _bin_series(x, bins, binning)
    y_disc = _bin_series(y, bins, binning)
    hist = _contingency(x_disc, y_disc)
    hist = hist + _DEF_EPS
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mi = np.sum(pxy * np.log(pxy / (px * py)))
    return float(mi)


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    # Acklam approximation
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    u = np.clip(u, 1e-10, 1 - 1e-10)
    plow = 0.02425
    phigh = 1 - plow
    z = np.zeros_like(u, dtype=float)

    lower = u < plow
    upper = u > phigh
    mid = (~lower) & (~upper)

    if np.any(lower):
        q = np.sqrt(-2 * np.log(u[lower]))
        z[lower] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if np.any(upper):
        q = np.sqrt(-2 * np.log(1 - u[upper]))
        z[upper] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if np.any(mid):
        q = u[mid] - 0.5
        r = q * q
        z[mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
        ) / (
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    return z


def _gaussian_copula_mi(x: pd.Series, y: pd.Series) -> float:
    xy = pd.concat([x, y], axis=1).dropna()
    if xy.shape[0] < 3:
        return float("nan")
    ranks = xy.rank(method="average")
    u = (ranks - 0.5) / len(xy)
    z = _norm_ppf(u.to_numpy())
    rho = np.corrcoef(z[:, 0], z[:, 1])[0, 1]
    rho = float(np.clip(rho, -0.999999, 0.999999))
    return float(-0.5 * math.log(1 - rho**2))


def mutual_information(x: pd.Series, y: pd.Series, cfg: InformationConfig) -> float:
    if cfg.mi_estimator == "hist":
        return _hist_mi(x, y, cfg.bins, cfg.binning)
    if cfg.mi_estimator == "gaussian_copula":
        return _gaussian_copula_mi(x, y)
    raise ValueError("Unsupported MI estimator")


def conditional_mutual_information(
    x: pd.Series,
    y: pd.Series,
    regimes: pd.Series,
    cfg: InformationConfig,
) -> float:
    aligned = pd.concat([x, y, regimes], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    total = len(aligned)
    mi_total = 0.0
    for _, frame in aligned.groupby(aligned.columns[-1]):
        weight = len(frame) / total
        mi_total += weight * mutual_information(frame.iloc[:, 0], frame.iloc[:, 1], cfg)
    return float(mi_total)


def mi_by_feature(
    features: pd.DataFrame,
    target: pd.Series,
    cfg: InformationConfig,
    regimes: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        mi_val = mutual_information(features[col], target, cfg)
        entry = {"feature": col, "mi": mi_val}
        if regimes is not None:
            entry["conditional_mi"] = conditional_mutual_information(
                features[col], target, regimes, cfg
            )
        rows.append(entry)
    return pd.DataFrame(rows).set_index("feature")


def pairwise_mi_matrix(
    features: pd.DataFrame,
    cfg: InformationConfig,
    max_features: int = 200,
) -> pd.DataFrame:
    cols = list(features.columns[:max_features])
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if j < i:
                matrix.loc[col_i, col_j] = matrix.loc[col_j, col_i]
                continue
            if col_i == col_j:
                matrix.loc[col_i, col_j] = 0.0
                continue
            mi_val = mutual_information(features[col_i], features[col_j], cfg)
            matrix.loc[col_i, col_j] = mi_val
    return matrix
